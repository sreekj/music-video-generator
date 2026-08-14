"""The shotlist: the contract between the Director, the web UI, and Stage B.

This is the single editable artefact in the system. The LLM Director writes it,
the UI mutates it, and the renderer consumes it. Nothing else crosses that
boundary -- which is what makes preview fast (mutate + re-render Stage B only)
and generation cacheable (a clip's identity is a hash of its ClipSpec).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CameraPath = Literal[
    "static", "push_in", "pull_out", "pan_left", "pan_right", "orbit", "drift", "sway",
]

# Effects the uber-shader knows how to apply. Keep in sync with render.FX_FLAGS.
Effect = Literal[
    "bloom", "god_rays", "chroma", "grain", "fog_drift", "vignette_pulse",
    "kaleidoscope", "glitch", "halation", "ripple",
]

# What a reactive curve is allowed to drive.
ReactiveTarget = Literal[
    "zoom_pulse", "vignette_pulse", "bloom", "shake", "chroma", "rotate",
    "brightness", "saturation", "ripple",
]

RetimeMode = Literal["stretch", "loop", "loop_pingpong", "speed_ramp_to_beat", "hold_last"]

TransitionType = Literal["cut", "crossfade", "dip_to_black", "dip_to_white", "whip"]


class Anchor(BaseModel):
    """A still keyframe derived from the theme image.

    Anchors are what keep a 4-minute video from drifting: every generated clip
    is pinned to one at each end, and every anchor traces back to the theme.
    """

    id: str
    source: str = "theme"  # "theme" or another anchor's id
    prompt: str = ""
    seed: int = 0
    strength: float = Field(0.35, ge=0.0, le=1.0)  # img2img denoise strength

    def cache_key(self, theme_hash: str) -> str:
        payload = f"{theme_hash}|{self.source}|{self.prompt}|{self.seed}|{self.strength}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ClipSpec(BaseModel):
    """A generated video segment, conditioned on anchors at both ends.

    ``last=None`` means free-running I2V (motion is unconstrained at the tail);
    supplying both is what makes a clip loopable and chainable.
    """

    first: str
    last: str | None = None
    duration: float = Field(6.0, gt=0.0, le=20.0)
    seed: int = 0
    prompt: str = ""

    def cache_key(self, anchor_keys: dict[str, str]) -> str:
        payload = json.dumps(
            {
                "first": anchor_keys.get(self.first, self.first),
                "last": anchor_keys.get(self.last, self.last) if self.last else None,
                "duration": round(self.duration, 3),
                "seed": self.seed,
                "prompt": self.prompt,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Camera(BaseModel):
    path: CameraPath = "static"
    amplitude: float = Field(0.15, ge=0.0, le=1.0)
    # Extra push applied across the shot regardless of path, for slow drift.
    zoom_start: float = Field(1.0, gt=0.0)
    zoom_end: float = Field(1.0, gt=0.0)


class Grade(BaseModel):
    exposure: float = Field(0.0, ge=-2.0, le=2.0)
    contrast: float = Field(1.0, ge=0.0, le=3.0)
    saturation: float = Field(1.0, ge=0.0, le=3.0)
    vignette: float = Field(0.25, ge=0.0, le=1.0)
    grain: float = Field(0.02, ge=0.0, le=0.5)
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)


class ReactiveBinding(BaseModel):
    """Bind an audio curve to a visual parameter.

    ``curve`` must name a curve present in the timeline (energy, onset,
    centroid, band_low, band_mid, band_high).
    """

    curve: str
    target: ReactiveTarget
    amount: float = Field(0.5, ge=0.0, le=2.0)


class Transition(BaseModel):
    type: TransitionType = "cut"
    duration: float = Field(0.0, ge=0.0, le=5.0)
    on_beat: int | None = None  # informational: which beat index this landed on

    @model_validator(mode="after")
    def _cut_has_no_duration(self) -> Transition:
        if self.type == "cut":
            self.duration = 0.0
        elif self.duration <= 0.0:
            self.duration = 0.5
        return self


class Shot(BaseModel):
    start: float = Field(ge=0.0)
    end: float
    section: str = ""

    # Exactly one source: a generated clip, or a still anchor (Tier-1 look).
    clip: ClipSpec | None = None
    still: str | None = None

    retime: RetimeMode = "loop_pingpong"
    camera: Camera = Field(default_factory=Camera)
    grade: Grade = Field(default_factory=Grade)
    fx: list[Effect] = Field(default_factory=list)
    reactive: list[ReactiveBinding] = Field(default_factory=list)
    transition_in: Transition = Field(default_factory=Transition)

    @model_validator(mode="after")
    def _check(self) -> Shot:
        if self.end <= self.start:
            raise ValueError(f"shot end ({self.end}) must exceed start ({self.start})")
        if self.clip is None and self.still is None:
            self.still = "theme"
        if self.clip is not None and self.still is not None:
            raise ValueError("shot must have either a clip or a still, not both")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class Shotlist(BaseModel):
    version: int = 1
    audio: str
    theme: str
    duration: float
    fps: int = 30
    width: int = 1920
    height: int = 1080
    prompt: str = ""  # the user's original animation prompt, for provenance
    anchors: list[Anchor] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)

    @field_validator("shots")
    @classmethod
    def _ordered(cls, shots: list[Shot]) -> list[Shot]:
        for a, b in zip(shots, shots[1:]):
            if b.start < a.end - 1e-3:
                raise ValueError(f"shots overlap at {b.start:.3f}s")
            if b.start > a.end + 1e-3:
                raise ValueError(
                    f"gap in shot coverage: {a.end:.3f}s -> {b.start:.3f}s. "
                    f"Shots must tile the timeline with no holes."
                )
        return shots

    @model_validator(mode="after")
    def _covers_timeline(self) -> Shotlist:
        """Shots must span the whole song.

        A short or gapped shotlist would otherwise render a truncated video
        that still looks superficially correct.
        """
        if not self.shots:
            return self
        if self.shots[0].start > 1e-3:
            raise ValueError(f"shots start at {self.shots[0].start:.3f}s, not 0")
        if self.shots[-1].end < self.duration - 1e-3:
            raise ValueError(
                f"shots end at {self.shots[-1].end:.3f}s but the song is "
                f"{self.duration:.3f}s long"
            )
        return self

    @model_validator(mode="after")
    def _anchors_resolve(self) -> Shotlist:
        known = {a.id for a in self.anchors} | {"theme"}
        for a in self.anchors:
            if a.source not in known:
                raise ValueError(f"anchor {a.id!r} has unknown source {a.source!r}")
        for i, s in enumerate(self.shots):
            for ref in filter(None, [s.still,
                                     s.clip.first if s.clip else None,
                                     s.clip.last if s.clip else None]):
                if ref not in known:
                    raise ValueError(f"shot {i} references unknown anchor {ref!r}")
        return self

    def shot_index_at(self, t: float) -> int:
        """Index of the shot covering ``t``, or raise.

        Coverage is validated on construction, so a miss here means the
        timeline and shotlist have diverged -- worth failing on rather than
        guessing at.
        """
        for i, s in enumerate(self.shots):
            if s.start <= t < s.end:
                return i
        # The final frame lands exactly on the last shot's end.
        if self.shots and abs(t - self.shots[-1].end) < 1e-6:
            return len(self.shots) - 1
        raise IndexError(
            f"no shot covers t={t:.4f}s (shots span "
            f"{self.shots[0].start:.3f}-{self.shots[-1].end:.3f}s)"
            if self.shots else f"no shot covers t={t:.4f}s (shotlist is empty)"
        )

    def shot_at(self, t: float) -> Shot:
        return self.shots[self.shot_index_at(t)]

    def anchor_keys(self, theme_hash: str) -> dict[str, str]:
        """Resolve every anchor to a content hash, following ``source`` chains."""
        keys: dict[str, str] = {"theme": theme_hash}
        by_id = {a.id: a for a in self.anchors}
        for a in self.anchors:
            chain, cur = [], a
            seen = set()
            while cur.source != "theme" and cur.source in by_id:
                if cur.id in seen:
                    raise ValueError(f"anchor cycle at {cur.id!r}")
                seen.add(cur.id)
                chain.append(cur)
                cur = by_id[cur.source]
            chain.append(cur)
            h = theme_hash
            for node in reversed(chain):
                h = node.cache_key(h)
            keys[a.id] = h
        return keys

    def clips(self) -> list[ClipSpec]:
        return [s.clip for s in self.shots if s.clip is not None]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Shotlist:
        return cls.model_validate_json(Path(path).read_text())
