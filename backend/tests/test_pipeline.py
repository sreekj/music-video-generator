"""Tests for the logic that is easy to get subtly wrong.

Deliberately not testing "does it render" -- that is covered by actually
rendering. These cover invariants that would silently corrupt output: retiming
that walks off the end of a clip, camera moves that sample past the image edge,
and cache keys that collide or fail to invalidate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.assemble.render import _camera, _source_frame_index  # noqa: E402
from app.audio.analysis import _merge_short  # noqa: E402
from app.director.schema import (  # noqa: E402
    Anchor, Camera, ClipSpec, Shot, Shotlist,
)


# --- retiming -------------------------------------------------------------

def _shot(**kw) -> Shot:
    base = dict(start=10.0, end=30.0, still="theme")
    base.update(kw)
    return Shot(**base)


@pytest.mark.parametrize("mode", ["stretch", "loop", "loop_pingpong", "hold_last",
                                  "speed_ramp_to_beat"])
def test_retime_stays_in_bounds(mode: str) -> None:
    """No retime mode may ever index outside the decoded clip."""
    n = 97
    shot = _shot(retime=mode)
    for i in range(0, 20 * 30):  # every frame of a 20s shot at 30fps
        t = shot.start + i / 30
        idx = _source_frame_index(shot, t, n, 30, beat_phase=(i % 30) / 30)
        assert 0 <= idx < n, f"{mode} produced {idx} at t={t:.2f}"


def test_pingpong_has_no_seam() -> None:
    """Ping-pong must reverse without repeating or skipping the end frames."""
    n = 10
    shot = _shot(retime="loop_pingpong")
    seq = [_source_frame_index(shot, shot.start + i / 30, n, 30, 0.0)
           for i in range(2 * n - 2)]
    assert seq == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    # And it repeats cleanly from there.
    nxt = _source_frame_index(shot, shot.start + (2 * n - 2) / 30, n, 30, 0.0)
    assert nxt == 0


def test_stretch_spans_whole_clip() -> None:
    shot = _shot(retime="stretch")
    n = 50
    assert _source_frame_index(shot, shot.start, n, 30, 0.0) == 0
    assert _source_frame_index(shot, shot.end, n, 30, 0.0) == n - 1


def test_single_frame_clip_is_safe() -> None:
    assert _source_frame_index(_shot(), 15.0, 1, 30, 0.0) == 0


# --- camera ---------------------------------------------------------------

@pytest.mark.parametrize("path", ["static", "push_in", "pull_out", "pan_left",
                                  "pan_right", "orbit", "drift", "sway"])
def test_camera_never_samples_past_edge(path: str) -> None:
    """Zoom must always leave enough crop headroom to cover the offset.

    Otherwise the shader clamps and the frame shows a smeared edge.
    """
    shot = _shot(camera=Camera(path=path, amplitude=1.0, zoom_start=1.0, zoom_end=1.0))
    for i in range(101):
        zoom, off, rot = _camera(shot, i / 100)
        assert zoom >= 1.0
        # Mirrors the shader: sample point is rot(c)/zoom + off, so the frame
        # corner reaches this far on each axis. It must stay within [-0.5, 0.5].
        half = 0.5 * (abs(math.cos(rot)) + abs(math.sin(rot))) / zoom
        assert abs(off[0]) + half <= 0.5 + 1e-6, f"{path}: x overflow at p={i / 100}"
        assert abs(off[1]) + half <= 0.5 + 1e-6, f"{path}: y overflow at p={i / 100}"


def test_push_in_actually_zooms() -> None:
    shot = _shot(camera=Camera(path="push_in", amplitude=0.4, zoom_start=1.0, zoom_end=1.2))
    assert _camera(shot, 1.0)[0] > _camera(shot, 0.0)[0]


# --- schema ---------------------------------------------------------------

def test_shot_defaults_to_theme_still() -> None:
    assert Shot(start=0, end=5).still == "theme"


def test_shot_rejects_clip_and_still() -> None:
    with pytest.raises(ValueError, match="not both"):
        Shot(start=0, end=5, still="theme", clip=ClipSpec(first="a"))


def test_shot_rejects_inverted_times() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        Shot(start=10, end=5)


def test_shotlist_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Shotlist(audio="a.mp3", theme="t.png", duration=30,
                 shots=[Shot(start=0, end=20), Shot(start=10, end=30)])


def test_shotlist_rejects_gap() -> None:
    """A hole would render frames with no shot -- must not be constructible."""
    with pytest.raises(ValueError, match="gap in shot coverage"):
        Shotlist(audio="a.mp3", theme="t.png", duration=30,
                 shots=[Shot(start=0, end=10), Shot(start=15, end=30)])


def test_shotlist_rejects_short_coverage() -> None:
    with pytest.raises(ValueError, match="song is"):
        Shotlist(audio="a.mp3", theme="t.png", duration=30,
                 shots=[Shot(start=0, end=20)])


def test_shotlist_rejects_late_start() -> None:
    with pytest.raises(ValueError, match="not 0"):
        Shotlist(audio="a.mp3", theme="t.png", duration=30,
                 shots=[Shot(start=5, end=30)])


def test_shotlist_rejects_unknown_anchor() -> None:
    with pytest.raises(ValueError, match="unknown anchor"):
        Shotlist(audio="a.mp3", theme="t.png", duration=10,
                 shots=[Shot(start=0, end=10, still="nope")])


def test_cut_transition_has_zero_duration() -> None:
    s = Shot(start=0, end=5, transition_in={"type": "cut", "duration": 2.0})
    assert s.transition_in.duration == 0.0


def test_shot_at_finds_shot() -> None:
    sl = Shotlist(audio="a", theme="t", duration=30,
                  shots=[Shot(start=0, end=10), Shot(start=10, end=30)])
    assert sl.shot_at(5).end == 10
    assert sl.shot_at(15).start == 10
    assert sl.shot_index_at(30.0) == 1  # final frame lands on the last end


def test_shot_at_raises_past_end() -> None:
    """Out-of-range time is a bug upstream, not something to clamp away."""
    sl = Shotlist(audio="a", theme="t", duration=30,
                  shots=[Shot(start=0, end=30)])
    with pytest.raises(IndexError, match="no shot covers"):
        sl.shot_at(999.0)


def test_fx_mask_rejects_unknown_effect() -> None:
    from app.assemble.render import _fx_mask

    assert _fx_mask(["bloom", "grain"]) > 0
    with pytest.raises(ValueError, match="unknown effect"):
        _fx_mask(["bloom", "nonexistent_fx"])


def test_require_codec_rejects_missing(monkeypatch) -> None:
    """A missing encoder must fail, not silently downgrade to libx264."""
    from app.assemble import encode

    monkeypatch.setattr(encode.config, "VIDEO_CODEC", "definitely_not_a_codec")
    with pytest.raises(RuntimeError, match="not available"):
        encode.require_codec()


def test_verify_gl_renderer_rejects_software() -> None:
    from app import config

    config.verify_gl_renderer("D3D12 (NVIDIA GeForce RTX 4090)")
    with pytest.raises(RuntimeError, match="does not match required"):
        config.verify_gl_renderer("llvmpipe (LLVM 20.1.2, 256 bits)")


# --- cache keys -----------------------------------------------------------

def test_anchor_chain_keys_depend_on_ancestry() -> None:
    """Changing an early anchor must invalidate everything downstream."""
    def make(p0: str) -> dict:
        sl = Shotlist(audio="a", theme="t", duration=10,
                      anchors=[Anchor(id="a0", source="theme", prompt=p0),
                               Anchor(id="a1", source="a0", prompt="second")],
                      shots=[Shot(start=0, end=10, still="a1")])
        return sl.anchor_keys("THEMEHASH")

    base, changed = make("first"), make("first-edited")
    assert base["a0"] != changed["a0"]
    assert base["a1"] != changed["a1"], "downstream anchor must invalidate too"


def test_anchor_cycle_is_detected() -> None:
    sl = Shotlist(audio="a", theme="t", duration=10,
                  anchors=[Anchor(id="a0", source="a1"), Anchor(id="a1", source="a0")])
    with pytest.raises(ValueError, match="cycle"):
        sl.anchor_keys("H")


def test_clip_key_ignores_irrelevant_changes() -> None:
    anchors = {"a0": "hash0"}
    c1 = ClipSpec(first="a0", duration=6.0, seed=1)
    c2 = ClipSpec(first="a0", duration=6.0, seed=1)
    assert c1.cache_key(anchors) == c2.cache_key(anchors)
    assert c1.cache_key(anchors) != ClipSpec(first="a0", duration=6.0,
                                             seed=2).cache_key(anchors)


# --- aspect fit -----------------------------------------------------------

class _FakeTex:
    def __init__(self, w, h): self.size = (w, h)


@pytest.mark.parametrize("sw,sh", [(1536, 1024), (2048, 1152), (1024, 1536),
                                   (1000, 1000), (3840, 1080), (800, 2000)])
def test_aspect_fit_never_samples_outside_image(sw: int, sh: int) -> None:
    """Sampled extent must stay within the image on both axes.

    Regression: the cover-fit ratio was inverted, so a 3:2 still in a 16:9
    frame sampled 1.19x the image height -- the clamp then smeared the top and
    bottom edges into vertical streaks across every frame.
    """
    from app.assemble.render import Renderer

    r = Renderer.__new__(Renderer)  # no GL context needed for pure maths
    r.width, r.height = 1920, 1080
    ax, ay = Renderer._aspect_fit(r, _FakeTex(sw, sh))
    assert 0 < ax <= 1.0 + 1e-9, f"{sw}x{sh}: x extent {ax} exceeds image"
    assert 0 < ay <= 1.0 + 1e-9, f"{sw}x{sh}: y extent {ay} exceeds image"


@pytest.mark.parametrize("sw,sh", [(1536, 1024), (2048, 1152), (1024, 1536),
                                   (1000, 1000), (3840, 1080)])
def test_aspect_fit_preserves_target_aspect(sw: int, sh: int) -> None:
    """The sampled region must have the frame's aspect ratio -- no distortion."""
    from app.assemble.render import Renderer

    r = Renderer.__new__(Renderer)
    r.width, r.height = 1920, 1080
    ax, ay = Renderer._aspect_fit(r, _FakeTex(sw, sh))
    sampled = (ax * sw) / (ay * sh)
    assert sampled == pytest.approx(1920 / 1080, rel=1e-6), f"{sw}x{sh} distorts"


def test_aspect_fit_is_identity_for_matching_aspect() -> None:
    from app.assemble.render import Renderer

    r = Renderer.__new__(Renderer)
    r.width, r.height = 1920, 1080
    assert Renderer._aspect_fit(r, _FakeTex(2048, 1152)) == pytest.approx((1.0, 1.0))


# --- image assignment -----------------------------------------------------

def _tl_with_late_peak():
    """A song whose loudest label first appears quietly, then carries the climax.

    A very common real-world shape: label A enters at middling energy, then
    returns repeatedly as the loudest material in the song.
    """
    from app.audio.analysis import Section, Timeline

    secs = [
        Section(id=0, start=0, end=20, label="E", energy=0.06, energy_rank=3),
        Section(id=1, start=20, end=40, label="C", energy=0.14, energy_rank=2),
        Section(id=2, start=40, end=50, label="A", energy=0.15, energy_rank=1),
        Section(id=3, start=50, end=80, label="A", energy=0.19, energy_rank=1),
        Section(id=4, start=80, end=100, label="A", energy=0.18, energy_rank=1),
        Section(id=5, start=100, end=120, label="B", energy=0.16, energy_rank=0),
    ]
    return Timeline(path="x", duration=120, fps=30, n_frames=3600, tempo=120,
                    beats=[], downbeats=[], sections=secs, curves={},
                    waveform_peaks=[])


def test_images_tier_by_overall_energy_not_first_appearance() -> None:
    """The song's loudest label must get peak imagery.

    Regression: tiering on first appearance sent a 110-second climax to
    mid-tier imagery and left every dramatic image unused.
    """
    from app.director.builder import _assign_images, _looks_by_energy

    tl = _tl_with_late_peak()
    images = {0: ["peak1", "peak2"], 1: ["mid1", "mid2"], 2: ["calm1", "calm2"]}
    amap = _assign_images(tl, _looks_by_energy(tl.sections), images)

    # A is the loudest label by duration-weighted energy -> must be peak tier.
    assert amap["A"] in images[0], f"loudest label got {amap['A']!r}"
    # The quietest label must not consume peak imagery.
    assert amap["E"] in images[2], f"quietest label got {amap['E']!r}"
    # A repeated label always maps to one image, so its return is recognisable.
    assert len({amap[s.label] for s in tl.sections if s.label == "A"}) == 1


def test_images_cover_multiple_distinct_stills() -> None:
    from app.director.builder import _assign_images, _looks_by_energy

    tl = _tl_with_late_peak()
    images = {0: ["p1", "p2"], 1: ["m1", "m2"], 2: ["c1", "c2"]}
    amap = _assign_images(tl, _looks_by_energy(tl.sections), images)
    assert len(set(amap.values())) >= 3, "video would be visually monotonous"


def test_assign_images_raises_on_empty_tier() -> None:
    from app.director.builder import _assign_images, _looks_by_energy

    tl = _tl_with_late_peak()
    with pytest.raises(ValueError, match="no images configured"):
        _assign_images(tl, _looks_by_energy(tl.sections), {0: ["a"], 1: ["b"], 2: []})


# --- LTX frame/dim constraints -------------------------------------------

def test_snap_frames_is_valid_and_nearest() -> None:
    """LTX needs 8k+1 frames; snapping must go to the *nearest*, not floor.

    Flooring would silently shorten every generated clip.
    """
    from app.generate.ltx import snap_frames

    for n in range(9, 400):
        s = snap_frames(n)
        assert (s - 1) % 8 == 0, f"{n} -> {s} is not 8k+1"
        assert abs(s - n) <= 4, f"{n} -> {s} is not the nearest valid value"

    assert snap_frames(96) == 97   # was 89 when this floored
    assert snap_frames(144) == 145
    assert snap_frames(97) == 97   # already valid, unchanged


def test_snap_dim_is_multiple_of_32() -> None:
    from app.generate.ltx import snap_dim

    for d in range(32, 2048, 7):
        assert snap_dim(d) % 32 == 0
        assert abs(snap_dim(d) - d) <= 16


def test_cover_resize_matches_target_exactly() -> None:
    """Cover-fit must fill the frame with no letterboxing, at any aspect."""
    from PIL import Image

    from app.generate.ltx import cover_resize

    for src in [(2048, 1152), (1000, 1000), (500, 2000)]:
        out = cover_resize(Image.new("RGB", src), 768, 448)
        assert out.size == (768, 448), f"{src} -> {out.size}"


# --- section merging ------------------------------------------------------

def test_merge_short_removes_fragments() -> None:
    raw = [(0, 0.0, 8.0), (1, 8.0, 11.0), (0, 11.0, 25.0)]
    merged = _merge_short(raw, min_dur=8.0)
    assert all(e - s >= 8.0 for _, s, e in merged)
    assert merged[0][1] == 0.0 and merged[-1][2] == 25.0, "must preserve total span"


def test_merge_short_leaves_valid_input_alone() -> None:
    raw = [(0, 0.0, 16.0), (1, 16.0, 32.0)]
    assert _merge_short(raw, min_dur=8.0) == raw


def test_merge_short_handles_all_fragments() -> None:
    """Even if everything is too short, it must terminate with one section."""
    raw = [(i, float(i), float(i + 1)) for i in range(5)]
    merged = _merge_short(raw, min_dur=10.0)
    assert len(merged) == 1
    assert merged[0][1] == 0.0 and merged[0][2] == 5.0
