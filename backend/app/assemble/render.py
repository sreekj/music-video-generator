"""Stage B: shotlist + sources -> video.

Fast, deterministic, and re-runnable. Everything expensive (generation) happened
in Stage A and is cached on disk; this stage only composites, grades and encodes,
which is what makes the preview loop feel live.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from .. import config
from ..audio.analysis import Timeline
from ..director.schema import Shot, Shotlist
from . import shaders
from .encode import FrameWriter, decode_to_memmap

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]

_DIP_COLOURS = {"dip_to_black": (0.0, 0.0, 0.0), "dip_to_white": (1.0, 1.0, 1.0)}
_TRANSITION_MODE = {"crossfade": 0, "dip_to_black": 1, "dip_to_white": 1, "whip": 2}


def _set(prog: moderngl.Program, name: str, value) -> None:
    """Set a uniform, failing loudly if the shader has no such uniform.

    A missing uniform means GLSL optimised it away because nothing reads it --
    i.e. the Python side thinks it is driving something the shader ignores.
    Silently skipping would turn a dead parameter into an invisible no-op, so
    unused uniforms must be deleted from both sides rather than tolerated.
    """
    try:
        prog[name].value = value
    except KeyError as exc:
        raise KeyError(
            f"shader has no uniform {name!r} (it is declared but unused, so GLSL "
            f"removed it). Either read it in the shader or stop setting it here."
        ) from exc


def _fx_mask(fx: list[str]) -> int:
    """Bitmask for the named effects. Unknown names are a bug, not a no-op."""
    unknown = [f for f in fx if f not in shaders.FX_FLAGS]
    if unknown:
        raise ValueError(
            f"unknown effect(s) {unknown}; shader knows {sorted(shaders.FX_FLAGS)}"
        )
    return sum(shaders.FX_FLAGS[f] for f in fx)


def _load_image(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _camera(shot: Shot, p: float) -> tuple[float, tuple[float, float], float]:
    """Camera transform for progress ``p`` in [0,1]: (zoom, offset, rotation).

    Panning and orbiting need crop headroom or they would sample past the edge
    of the source, so the zoom floor rises with the offset magnitude.
    """
    cam = shot.camera
    amp = cam.amplitude
    zoom = cam.zoom_start + (cam.zoom_end - cam.zoom_start) * p
    off = [0.0, 0.0]
    rot = 0.0

    path = cam.path
    if path == "pan_left":
        off[0] = amp * 0.12 * (0.5 - p) * 2.0
    elif path == "pan_right":
        off[0] = amp * 0.12 * (p - 0.5) * 2.0
    elif path == "orbit":
        off[0] = amp * 0.07 * math.cos(2 * math.pi * p)
        off[1] = amp * 0.07 * math.sin(2 * math.pi * p)
        rot = amp * 0.05 * math.sin(2 * math.pi * p)
    elif path == "drift":
        off[0] = amp * 0.05 * math.sin(2 * math.pi * p * 0.7)
        off[1] = amp * 0.04 * math.cos(2 * math.pi * p * 0.5)
    elif path == "sway":
        rot = amp * 0.06 * math.sin(2 * math.pi * p)
        off[0] = amp * 0.03 * math.sin(2 * math.pi * p)

    # Crop headroom. The shader samples at rot(c)/zoom + off, so the corner
    # (0.5, 0.5) reaches 0.5*(|cos|+|sin|)/zoom + |off| on each axis. Keeping
    # that within 0.5 is what stops the camera sampling past the image edge
    # (which the shader would clamp into a smeared border).
    m = max(abs(off[0]), abs(off[1]))
    spread = abs(math.cos(rot)) + abs(math.sin(rot))
    headroom = 0.5 * spread / max(0.5 - m, 1e-3)
    return max(zoom, headroom, 1.0), (off[0], off[1]), rot


def _source_frame_index(shot: Shot, t: float, n_frames: int, fps: int,
                        beat_phase: float) -> int:
    """Map timeline time to a frame within the shot's generated clip."""
    if n_frames <= 1:
        return 0

    local = max(0.0, t - shot.start)
    mode = shot.retime

    if mode == "stretch":
        p = local / max(shot.duration, 1e-6)
        return int(np.clip(round(p * (n_frames - 1)), 0, n_frames - 1))

    # Round rather than truncate: local*fps lands on values like 0.9999999
    # for exact frame times, and int() would floor those to the previous
    # frame, duplicating frames and stuttering the motion.
    f = int(round(local * fps))

    if mode == "hold_last":
        return min(f, n_frames - 1)
    if mode == "loop":
        return f % n_frames
    if mode == "speed_ramp_to_beat":
        # Ease within each beat so motion lands on the downbeat.
        return int(round(f * (0.75 + 0.5 * beat_phase))) % n_frames

    # loop_pingpong (default): forward then back, no seam and no repeated
    # end frames -- period 2n-2 visits 0..n-1 then n-2..1.
    period = 2 * n_frames - 2
    k = f % period
    return k if k < n_frames else period - k


class Renderer:
    """Holds the GL context, programs and framebuffers for one render."""

    def __init__(self, width: int, height: int):
        config.setup_gl_env()
        self.ctx = moderngl.create_standalone_context()
        config.verify_gl_renderer(self.ctx.info["GL_RENDERER"])
        self.width, self.height = width, height

        self.shot_prog = self.ctx.program(vertex_shader=shaders.VERT,
                                          fragment_shader=shaders.SHOT_FRAG)
        self.comp_prog = self.ctx.program(vertex_shader=shaders.VERT,
                                          fragment_shader=shaders.COMPOSITE_FRAG)

        quad = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        self.shot_vao = self.ctx.vertex_array(self.shot_prog, [(quad, "2f", "in_pos")])
        self.comp_vao = self.ctx.vertex_array(self.comp_prog, [(quad, "2f", "in_pos")])

        # The two shot buffers are sampled by the composite pass, so they must
        # be texture-backed; only fbo_out is read straight back to the CPU.
        self.tex_a = self.ctx.texture((width, height), 3)
        self.tex_b = self.ctx.texture((width, height), 3)
        for t in (self.tex_a, self.tex_b):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        self.fbo_a = self.ctx.framebuffer(color_attachments=[self.tex_a])
        self.fbo_b = self.ctx.framebuffer(color_attachments=[self.tex_b])
        self.fbo_out = self.ctx.simple_framebuffer((width, height))

        self._textures: dict[str, moderngl.Texture] = {}
        self._clips: dict[str, np.ndarray] = {}
        self._clip_tex: moderngl.Texture | None = None

    @property
    def renderer_name(self) -> str:
        return self.ctx.info["GL_RENDERER"]

    def still_texture(self, key: str, path: str | Path) -> moderngl.Texture:
        if key not in self._textures:
            arr = _load_image(path)
            tex = self.ctx.texture((arr.shape[1], arr.shape[0]), 3, arr.tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            self._textures[key] = tex
        return self._textures[key]

    def clip_frames(self, key: str, path: str | Path) -> np.ndarray:
        if key not in self._clips:
            self._clips[key] = decode_to_memmap(path, config.GEN_WIDTH, config.GEN_HEIGHT)
        return self._clips[key]

    def _clip_texture(self, frame: np.ndarray) -> moderngl.Texture:
        h, w = frame.shape[:2]
        if self._clip_tex is None or self._clip_tex.size != (w, h):
            if self._clip_tex is not None:
                self._clip_tex.release()
            self._clip_tex = self.ctx.texture((w, h), 3, frame.tobytes())
            self._clip_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._clip_tex.repeat_x = self._clip_tex.repeat_y = False
        else:
            self._clip_tex.write(frame.tobytes())
        return self._clip_tex

    def _aspect_fit(self, tex: moderngl.Texture) -> tuple[float, float]:
        """Cover-fit: the fraction of the source sampled on each axis.

        The shader samples ``(p - 0.5) * aspect + 0.5``, so each component is
        the sampled *extent* and must never exceed 1.0 -- beyond that it reads
        past the image edge and the clamp smears the border into streaks.

        Solving ``(Wx/Wy) * src_aspect == dst_aspect`` for the largest region
        that still fits inside the image gives the crop below. A 3:2 still in a
        16:9 frame therefore keeps its full width and crops top and bottom.
        """
        src = tex.size[0] / tex.size[1]
        dst = self.width / self.height
        return (dst / src, 1.0) if src > dst else (1.0, src / dst)

    def render_shot(self, shot: Shot, t: float, fbo: moderngl.Framebuffer,
                    sources: dict[str, Path], timeline: Timeline,
                    frame_idx: int) -> None:
        p = float(np.clip((t - shot.start) / max(shot.duration, 1e-6), 0.0, 1.0))

        # --- source ---
        if shot.clip is not None:
            key = shot.clip.first + "|" + (shot.clip.last or "") + f"|{shot.clip.seed}"
            path = sources.get(key)
            if path is None or not Path(path).exists():
                # Substituting the theme here would render a video that looks
                # plausible while silently containing none of the generated
                # motion the shotlist asked for. Stage A must run first.
                raise FileNotFoundError(
                    f"shot at {shot.start:.2f}s needs generated clip {key!r}, "
                    f"which is missing ({path}). Run Stage A generation before "
                    f"rendering, or set the shot to use a still."
                )
            frames = self.clip_frames(key, path)
            beats = timeline.beats
            bp = 0.0
            if beats:
                i = int(np.searchsorted(beats, t)) - 1
                if 0 <= i < len(beats) - 1:
                    span = beats[i + 1] - beats[i]
                    bp = (t - beats[i]) / span if span > 0 else 0.0
            idx = _source_frame_index(shot, t, len(frames), timeline.fps, bp)
            tex = self._clip_texture(np.asarray(frames[idx]))
        else:
            key = shot.still or "theme"
            if key not in sources:
                raise KeyError(
                    f"shot at {shot.start:.2f}s references still {key!r}, which "
                    f"is not in sources (have: {sorted(sources)})"
                )
            tex = self.still_texture(key, sources[key])

        tex.use(0)
        prog = self.shot_prog
        _set(prog, "u_tex", 0)
        _set(prog, "u_tex_aspect", self._aspect_fit(tex))

        # --- reactive drives ---
        drives = dict(pulse=0.0, shake=0.0, bloom=0.0, chroma=0.0,
                      vig_pulse=0.0, bright=0.0, sat_drive=0.0, ripple=0.0)
        target_map = {
            "zoom_pulse": "pulse", "vignette_pulse": "vig_pulse", "bloom": "bloom",
            "shake": "shake", "chroma": "chroma", "rotate": "shake",
            "brightness": "bright", "saturation": "sat_drive", "ripple": "ripple",
        }
        for b in shot.reactive:
            if b.curve not in timeline.curves:
                raise KeyError(
                    f"shot at {shot.start:.2f}s binds curve {b.curve!r}, which the "
                    f"timeline does not have (has: {sorted(timeline.curves)})"
                )
            if b.target not in target_map:
                raise KeyError(
                    f"shot at {shot.start:.2f}s binds unknown target {b.target!r} "
                    f"(known: {sorted(target_map)})"
                )
            curve = timeline.curves[b.curve]
            v = curve[min(frame_idx, len(curve) - 1)]
            slot = target_map[b.target]
            drives[slot] += v * b.amount * (0.08 if slot == "pulse" else 1.0)

        # Effects named in fx get a floor value so they show without a binding.
        if "bloom" in shot.fx:
            drives["bloom"] = max(drives["bloom"], 0.35)
        if "chroma" in shot.fx:
            drives["chroma"] = max(drives["chroma"], 0.3)
        if "ripple" in shot.fx:
            drives["ripple"] = max(drives["ripple"], 0.4)

        zoom, off, rot = _camera(shot, p)
        _set(prog, "u_zoom", zoom)
        _set(prog, "u_offset", off)
        _set(prog, "u_rot", rot)
        for k, v in drives.items():
            _set(prog, f"u_{k}", float(v))

        g = shot.grade
        _set(prog, "u_exposure", g.exposure)
        _set(prog, "u_contrast", g.contrast)
        _set(prog, "u_saturation", g.saturation)
        _set(prog, "u_vignette", g.vignette)
        _set(prog, "u_grain", g.grain)
        _set(prog, "u_tint", tuple(g.tint))

        _set(prog, "u_fx", _fx_mask(shot.fx))
        _set(prog, "u_time", t)

        fbo.use()
        self.shot_vao.render()

    def composite(self, mix: float, mode: int, dip: tuple[float, float, float]) -> bytes:
        self.tex_a.use(0)
        self.tex_b.use(1)
        _set(self.comp_prog, "u_a", 0)
        _set(self.comp_prog, "u_b", 1)
        _set(self.comp_prog, "u_mix", mix)
        _set(self.comp_prog, "u_mode", mode)
        _set(self.comp_prog, "u_dip", dip)

        self.fbo_out.use()
        self.comp_vao.render()
        return self.fbo_out.read(components=3)

    def release(self) -> None:
        for t in self._textures.values():
            t.release()
        if self._clip_tex is not None:
            self._clip_tex.release()
        for f in (self.fbo_a, self.fbo_b, self.fbo_out):
            f.release()
        self.tex_a.release()
        self.tex_b.release()
        self.ctx.release()


def render(shotlist: Shotlist, timeline: Timeline, out: str | Path,
           sources: dict[str, Path] | None = None,
           preview: bool = False, audio: str | Path | None = None,
           progress: ProgressFn | None = None,
           start: float = 0.0, end: float | None = None) -> Path:
    """Render a shotlist to a video file.

    Args:
        sources: maps "theme"/anchor ids/clip keys to files on disk. Missing
            clip entries fall back to the theme image, so a shotlist can be
            previewed before Stage A has generated anything.
        preview: render at preview resolution/fps.
        start, end: render a sub-range (used by the UI to scrub a section).
    """
    width = config.PREVIEW_WIDTH if preview else shotlist.width
    height = config.PREVIEW_HEIGHT if preview else shotlist.height
    fps = config.PREVIEW_FPS if preview else shotlist.fps

    sources = dict(sources or {})
    sources.setdefault("theme", Path(shotlist.theme))

    end = shotlist.duration if end is None else min(end, shotlist.duration)
    n_frames = max(1, int(round((end - start) * fps)))

    r = Renderer(width, height)
    log.info("rendering %dx%d @%dfps on %s (%d frames)",
             width, height, fps, r.renderer_name, n_frames)

    writer = FrameWriter(out, width, height, fps, audio=audio,
                         quality=26 if preview else 20, loudnorm=not preview)

    t0 = time.perf_counter()
    try:
        for i in range(n_frames):
            t = start + i / fps
            # Curves are sampled at the timeline's fps, not the render fps.
            curve_idx = int(t * timeline.fps)

            # Raises if the shotlist has a hole: skipping the frame would
            # silently emit a video shorter than the song.
            si = shotlist.shot_index_at(t)
            shot = shotlist.shots[si]

            trans = shot.transition_in
            in_trans = (si > 0 and trans.type != "cut"
                        and t < shot.start + trans.duration)

            if in_trans:
                prev = shotlist.shots[si - 1]
                r.render_shot(prev, t, r.fbo_a, sources, timeline, curve_idx)
                r.render_shot(shot, t, r.fbo_b, sources, timeline, curve_idx)
                mix = (t - shot.start) / max(trans.duration, 1e-6)
                if trans.type not in _TRANSITION_MODE:
                    # Defaulting to crossfade would quietly render the wrong
                    # transition whenever a new type is added to the schema
                    # without a matching composite mode.
                    raise KeyError(
                        f"transition type {trans.type!r} has no composite mode "
                        f"(known: {sorted(_TRANSITION_MODE)})"
                    )
                mode = _TRANSITION_MODE[trans.type]
                dip = _DIP_COLOURS.get(trans.type, (0.0, 0.0, 0.0))  # unused unless mode 1
            else:
                r.render_shot(shot, t, r.fbo_b, sources, timeline, curve_idx)
                mix, mode, dip = 1.0, 0, (0.0, 0.0, 0.0)

            writer.write(r.composite(mix, mode, dip))

            if progress and (i % 60 == 0 or i == n_frames - 1):
                progress((i + 1) / n_frames, f"frame {i + 1}/{n_frames}")

        path = writer.close()
    finally:
        r.release()

    dt = time.perf_counter() - t0
    log.info("rendered %d frames in %.1fs (%.1f fps, %.1fx realtime) -> %s",
             n_frames, dt, n_frames / dt, (n_frames / dt) / fps, path)
    return path


def main() -> None:
    import argparse
    import json

    from ..audio.analysis import analyze
    from ..director.builder import build

    ap = argparse.ArgumentParser(description="Render a music video (Stage B)")
    ap.add_argument("audio")
    ap.add_argument("theme")
    ap.add_argument("-o", "--out", default=str(config.OUT / "out.mp4"))
    ap.add_argument("--prompt", default="")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--shotlist", help="use an existing shotlist.json")
    ap.add_argument("--save-shotlist", default=str(config.CACHE / "shotlist.json"))
    ap.add_argument("--seconds", type=float, default=None, help="render only the first N seconds")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tl = analyze(args.audio, fps=config.FPS)
    print(f"analysed: {tl.duration:.1f}s, {tl.tempo:.0f} BPM, {len(tl.sections)} sections")

    if args.shotlist:
        sl = Shotlist.load(args.shotlist)
    else:
        sl = build(tl, theme=args.theme, prompt=args.prompt, generative=False)
        sl.save(args.save_shotlist)
        print(f"shotlist: {len(sl.shots)} shots -> {args.save_shotlist}")

    render(sl, tl, args.out, preview=args.preview, audio=args.audio,
           end=args.seconds,
           progress=lambda p, m: print(f"  {p * 100:5.1f}%  {m}", end="\r"))
    print()


if __name__ == "__main__":
    main()
