"""Deterministic shotlist builder -- the Director's stand-in.

This produces a musically-sensible shotlist from the audio analysis alone, with
no LLM involved. It exists for two reasons: Stage B needs something to render
long before the Director is written, and every LLM-authored shotlist should be
diffable against a known-good baseline.

The LLM Director will replace `build()` by choosing looks and prompts; the
beat-snapping and shot-splitting logic below stays useful either way.
"""

from __future__ import annotations

import bisect

from ..audio.analysis import Timeline
from .schema import (
    Anchor, Camera, ClipSpec, Grade, ReactiveBinding, Shot, Shotlist, Transition,
)

# Looks indexed by energy rank (0 = highest-energy cluster, usually the chorus).
LOOKS = [
    {  # rank 0 -- chorus: bright, busy, driving
        "camera": ("push_in", 0.35),
        "fx": ["bloom", "god_rays", "chroma", "grain"],
        "grade": dict(exposure=0.12, contrast=1.12, saturation=1.18, vignette=0.22, grain=0.025),
        "reactive": [("band_low", "zoom_pulse", 0.9), ("band_high", "bloom", 0.6),
                     ("onset", "shake", 0.25)],
        "transition": "cut",
    },
    {  # rank 1 -- verse: steadier, cooler
        "camera": ("drift", 0.20),
        "fx": ["grain", "fog_drift", "halation"],
        "grade": dict(exposure=0.0, contrast=1.05, saturation=0.98, vignette=0.32, grain=0.03),
        "reactive": [("band_low", "vignette_pulse", 0.5), ("energy", "brightness", 0.25)],
        "transition": "crossfade",
    },
    {  # rank 2 -- intro/outro/breakdown: slow, soft, wide
        "camera": ("pull_out", 0.15),
        "fx": ["grain", "fog_drift"],
        "grade": dict(exposure=-0.06, contrast=0.98, saturation=0.85, vignette=0.42, grain=0.035),
        "reactive": [("energy", "brightness", 0.2)],
        "transition": "dip_to_black",
    },
]

MAX_SHOT = 22.0  # split longer sections so the camera move never overstays
MIN_SHOT = 5.0


def _looks_by_energy(sections) -> dict[int, int]:
    """Assign a look to each section by *relative* energy.

    Not by raw ``energy_rank``: that counts distinct clusters, so it grows with
    the section count, and clamping it to len(LOOKS) collapses everything above
    rank 2 into the softest look. On a 9-cluster track that made 10 of 19
    sections render identically, including mid-energy passages that should read
    as verses.

    Ranking by energy percentile instead keeps the looks proportional to the
    song's own dynamics, whatever the cluster count.
    """
    order = sorted(sections, key=lambda s: s.energy)
    n = len(order)
    out: dict[int, int] = {}
    for i, s in enumerate(order):
        pct = i / max(n - 1, 1)  # 0 = quietest section, 1 = loudest
        # Top third gets the chorus look, middle third verse, bottom third soft.
        out[s.id] = 0 if pct >= 2 / 3 else (1 if pct >= 1 / 3 else 2)
    return out


def _snap(t: float, grid: list[float], tol: float = 1.2) -> float:
    """Snap a time to the nearest grid point (downbeat) within tolerance."""
    if not grid:
        return t
    i = bisect.bisect_left(grid, t)
    cands = [grid[j] for j in (i - 1, i) if 0 <= j < len(grid)]
    if not cands:
        return t
    best = min(cands, key=lambda g: abs(g - t))
    return best if abs(best - t) <= tol else t


def _split(start: float, end: float, grid: list[float]) -> list[tuple[float, float]]:
    """Break a long section into shots at downbeats."""
    if end - start <= MAX_SHOT:
        return [(start, end)]

    n = max(2, round((end - start) / MAX_SHOT))
    step = (end - start) / n
    cuts = [start]
    for i in range(1, n):
        cuts.append(_snap(start + i * step, grid, tol=step / 3))
    cuts.append(end)

    out = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a >= MIN_SHOT:
            out.append((a, b))
        elif out:  # too short -- absorb into the previous shot
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _assign_images(timeline: Timeline, look_of: dict[int, int],
                   images: dict[int, list[str]]) -> dict[str, str]:
    """Map each section *label* to an image id, tiered by the label's own energy.

    Keyed by label rather than section id so a repeated section (the same chorus
    returning) brings back its imagery -- that repetition is what makes the
    video feel composed rather than randomly assembled.

    The tier must come from the label's **duration-weighted mean energy across
    every occurrence**, not from where it first appears. A chorus often enters
    quietly the first time and only later carries the climax; ranking on first
    appearance sent the loudest passage of a song to mid-tier imagery and left
    the most dramatic images entirely unused.
    """
    labels: dict[str, list[tuple[float, float]]] = {}
    for sec in timeline.sections:
        labels.setdefault(sec.label, []).append((sec.energy, sec.end - sec.start))

    energy_of = {
        lbl: sum(e * d for e, d in v) / max(sum(d for _, d in v), 1e-6)
        for lbl, v in labels.items()
    }
    # Loudest label first, so tier 0 imagery lands on the song's peak material.
    order = sorted(energy_of, key=lambda l: -energy_of[l])

    n = len(order)
    out: dict[str, str] = {}
    used: dict[int, int] = {}
    for rank, lbl in enumerate(order):
        pct = rank / max(n - 1, 1)  # 0 = loudest label
        tier = 0 if pct < 1 / 3 else (1 if pct < 2 / 3 else 2)
        pool = images.get(tier)
        if not pool:
            raise ValueError(f"no images configured for tier {tier}")
        i = used.get(tier, 0)
        out[lbl] = pool[i % len(pool)]
        used[tier] = i + 1
    return out


def build(timeline: Timeline, theme: str, prompt: str = "",
          generative: bool = False, width: int = 1920, height: int = 1080,
          images: dict[int, list[str]] | None = None) -> Shotlist:
    """Build a shotlist from audio analysis.

    Args:
        timeline: output of ``audio.analysis.analyze``.
        theme: path to the theme image.
        prompt: user's animation prompt (recorded, and used for clip prompts).
        generative: if True, emit ClipSpecs chained through anchors (Stage A
            work). If False, emit still-based shots that render immediately.
    """
    # Every look binds curves by name; a missing curve would silently disable
    # that reactive channel, so check up front instead.
    needed = {c for look in LOOKS for c, _t, _a in look["reactive"]}
    missing = needed - set(timeline.curves)
    if missing:
        raise KeyError(
            f"timeline is missing curve(s) {sorted(missing)} required by the "
            f"look presets; have {sorted(timeline.curves)}"
        )

    grid = timeline.downbeats or timeline.beats
    if not grid:
        raise ValueError("timeline has no beats or downbeats to snap cuts to")

    anchors: list[Anchor] = []
    shots: list[Shot] = []

    # One anchor per distinct section label, chained so each derives from the
    # previous -- that chain is what gives the video continuity.
    labels: list[str] = []
    for s in timeline.sections:
        if s.label not in labels:
            labels.append(s.label)

    if generative:
        prev = "theme"
        for i, label in enumerate(labels):
            aid = f"a_{label.lower()}"
            anchors.append(Anchor(id=aid, source=prev, prompt=prompt, seed=1000 + i * 37,
                                  strength=0.3 if i == 0 else 0.42))
            prev = aid
    anchor_of = {label: f"a_{label.lower()}" for label in labels}

    look_of = _looks_by_energy(timeline.sections)

    # Multi-image mode: each section label gets its own still, so the video cuts
    # between real imagery instead of re-grading one picture for four minutes.
    image_of: dict[str, str] = {}
    if images:
        image_of = _assign_images(timeline, look_of, images)
        seen: list[str] = []
        for img in image_of.values():
            if img not in seen:
                seen.append(img)
        anchors.extend(Anchor(id=i, source="theme") for i in seen)

    first = True
    for sec in timeline.sections:
        look = LOOKS[look_of[sec.id]]
        cam_path, cam_amp = look["camera"]

        for (a, b) in _split(sec.start, sec.end, grid):
            a_snap = 0.0 if first else _snap(a, grid)
            b_snap = _snap(b, grid) if b < timeline.duration - 0.5 else timeline.duration
            if b_snap - a_snap < 1.0:
                raise ValueError(
                    f"beat snapping collapsed a shot to {b_snap - a_snap:.3f}s at "
                    f"{a_snap:.2f}s; dropping it would leave a hole in the timeline"
                )

            trans = Transition(type="cut") if first else Transition(
                type=look["transition"],
                duration=0.0 if look["transition"] == "cut" else 0.6,
                on_beat=bisect.bisect_left(timeline.beats, a_snap),
            )

            # Zoom is a crop factor, so it must never drop below 1.0 or the
            # camera would sample past the edge of the source. "Pulling out"
            # therefore means starting zoomed in and relaxing back to full frame.
            if cam_path == "push_in":
                zoom_start, zoom_end = 1.0, 1.0 + cam_amp * 0.5
            elif cam_path == "pull_out":
                zoom_start, zoom_end = 1.0 + cam_amp * 0.5, 1.0
            else:
                zoom_start, zoom_end = 1.0 + cam_amp * 0.08, 1.0 + cam_amp * 0.22

            shot = Shot(
                start=round(a_snap, 3),
                end=round(b_snap, 3),
                section=sec.label,
                clip=ClipSpec(
                    first=anchor_of[sec.label],
                    last=anchor_of[sec.label],  # loopable: same anchor both ends
                    duration=min(8.0, b_snap - a_snap),
                    seed=2000 + sec.id * 17,
                    prompt=prompt,
                ) if generative else None,
                still=None if generative else image_of.get(sec.label, "theme"),
                retime="loop_pingpong",
                camera=Camera(path=cam_path, amplitude=cam_amp,
                              zoom_start=zoom_start, zoom_end=zoom_end),
                grade=Grade(**look["grade"]),
                fx=list(look["fx"]),
                reactive=[ReactiveBinding(curve=c, target=t, amount=amt)
                          for c, t, amt in look["reactive"]],
                transition_in=trans,
            )
            shots.append(shot)
            first = False

    return Shotlist(
        version=1,
        audio=timeline.path,
        theme=theme,
        duration=timeline.duration,
        fps=timeline.fps,
        width=width,
        height=height,
        prompt=prompt,
        anchors=anchors,
        shots=shots,
    )
