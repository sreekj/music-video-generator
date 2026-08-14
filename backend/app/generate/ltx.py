"""Stage A: LTX image-to-video with first+last frame conditioning.

The anchor chain is the whole point of this module. Rather than generating
independent clips that drift apart, each clip is pinned to a still frame at both
ends:

    theme -> [clip 0] -> a1 -> [clip 1] -> a2 -> [clip 2] -> a3 ...

A clip conditioned with the *same* anchor at both ends is seamlessly loopable,
which is what lets ~60s of generated motion cover a 4-minute song.

Anchors are harvested from the chain itself (the last frame of clip N becomes
the first frame of clip N+1), so no second image model is needed.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]

# LTX constraints: spatial dims must be divisible by 32, and the frame count
# must satisfy (n - 1) % 8 == 0 because of the causal temporal VAE.
SPATIAL_MULTIPLE = 32
TEMPORAL_MULTIPLE = 8


def snap_frames(n: int) -> int:
    """Round a frame count to the nearest valid (8k + 1).

    Nearest, not floor: flooring 96 to 89 would quietly drop 7 frames from every
    clip, shortening generated motion against what the shotlist asked for.
    """
    n = max(TEMPORAL_MULTIPLE + 1, int(n))
    return round((n - 1) / TEMPORAL_MULTIPLE) * TEMPORAL_MULTIPLE + 1


def snap_dim(x: int) -> int:
    return max(SPATIAL_MULTIPLE, int(round(x / SPATIAL_MULTIPLE)) * SPATIAL_MULTIPLE)


def cover_resize(img: Image.Image, width: int, height: int) -> Image.Image:
    """Centre-crop to the target aspect, then resize. No letterboxing."""
    src = img.width / img.height
    dst = width / height
    if src > dst:
        w = int(img.height * dst)
        img = img.crop(((img.width - w) // 2, 0, (img.width + w) // 2, img.height))
    else:
        h = int(img.width / dst)
        img = img.crop((0, (img.height - h) // 2, img.width, (img.height + h) // 2))
    return img.resize((width, height), Image.LANCZOS)


@dataclass
class ClipResult:
    frames: list[Image.Image]
    path: Path | None = None
    seconds: float = 0.0

    @property
    def first(self) -> Image.Image:
        return self.frames[0]

    @property
    def last(self) -> Image.Image:
        return self.frames[-1]


class LTXGenerator:
    """Lazy-loading wrapper around the LTX condition pipeline."""

    def __init__(self, model: str | None = None, dtype: str | None = None,
                 width: int | None = None, height: int | None = None,
                 offload: bool = True):
        self.model = model or config.LTX_MODEL
        self.dtype_name = dtype or config.LTX_DTYPE
        self.width = snap_dim(width or config.GEN_WIDTH)
        self.height = snap_dim(height or config.GEN_HEIGHT)
        self.offload = offload
        self._pipe = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = self._load()
        return self._pipe

    def _load(self):
        import torch
        from diffusers import LTXConditionPipeline

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available; Stage A generation requires the GPU. "
                "Install the CUDA build of torch."
            )

        dtype = getattr(torch, self.dtype_name)
        log.info("loading %s (%s)", self.model, self.dtype_name)
        pipe = LTXConditionPipeline.from_pretrained(self.model, torch_dtype=dtype)
        self._check_scheduler(pipe)

        # 24GB is enough for inference but not for holding every component
        # resident alongside the VAE decode of a long clip. Tiling and slicing
        # are what keep decode inside the budget, so a pipeline without them is
        # a configuration we have not validated -- say so rather than OOM later.
        if self.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        for fn in ("enable_tiling", "enable_slicing"):
            if not hasattr(pipe.vae, fn):
                raise RuntimeError(
                    f"{type(pipe.vae).__name__} has no {fn}(); VAE decode will not "
                    f"fit in 24GB for long clips"
                )
            getattr(pipe.vae, fn)()
        return pipe

    @staticmethod
    def _check_scheduler(pipe) -> None:
        """Refuse a checkpoint whose scheduler does not match the pipeline.

        This pipeline builds its own linear-quadratic sigma schedule and calls
        ``retrieve_timesteps()`` without ``mu``, so a checkpoint shipping
        ``use_dynamic_shifting=True`` raises "`mu` must be passed".

        An earlier version of this code "fixed" that by disabling dynamic
        shifting. It ran -- and produced pure noise, because the sigma schedule
        no longer matched what the weights were trained against. That is the
        exact failure mode this project refuses: output that looks like a
        successful run but is silently garbage. So: raise, and name the
        checkpoint the installed pipeline was actually written for.
        """
        if not getattr(pipe.scheduler.config, "use_dynamic_shifting", False):
            return
        raise RuntimeError(
            f"{type(pipe.scheduler).__name__} has use_dynamic_shifting=True, but "
            f"{type(pipe).__name__} never passes `mu`. This checkpoint and this "
            f"diffusers version are incompatible -- do NOT work around it by "
            f"disabling dynamic shifting, which silently generates noise. Use a "
            f"checkpoint matching the installed pipeline (see its "
            f"EXAMPLE_DOC_STRING; currently {config.LTX_MODEL!r} is configured)."
        )

    def unload(self) -> None:
        import torch

        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- generation --------------------------------------------------------

    def generate(self, prompt: str, first: Image.Image,
                 last: Image.Image | None = None,
                 duration: float = 6.0, fps: int = 24, seed: int = 0,
                 steps: int = 40, guidance: float = 3.0,
                 negative: str = "worst quality, blurry, jittery, distorted, watermark",
                 progress: ProgressFn | None = None) -> ClipResult:
        """Generate one clip conditioned on ``first`` (and optionally ``last``).

        Passing ``last=first`` produces a loopable clip; passing a different
        image morphs between them; passing None lets motion run free, which is
        how new anchors are discovered.
        """
        import time

        import torch
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        n_frames = snap_frames(duration * fps)
        first_img = cover_resize(first, self.width, self.height)

        conditions = [LTXVideoCondition(image=first_img, frame_index=0)]
        if last is not None:
            conditions.append(
                LTXVideoCondition(image=cover_resize(last, self.width, self.height),
                                  frame_index=n_frames - 1)
            )

        gen = torch.Generator(device="cuda").manual_seed(seed)
        cb = None
        if progress is not None:
            def cb(_pipe, step: int, _t, kwargs):  # noqa: ANN001
                progress(min(1.0, (step + 1) / steps), f"denoising {step + 1}/{steps}")
                return kwargs

        t0 = time.perf_counter()
        out = self.pipe(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=negative,
            width=self.width,
            height=self.height,
            num_frames=n_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            output_type="pil",
            callback_on_step_end=cb,
        )
        dt = time.perf_counter() - t0

        frames = out.frames[0]
        log.info("generated %d frames (%.1fs of video) in %.1fs -- %.2f s/frame",
                 len(frames), len(frames) / fps, dt, dt / max(len(frames), 1))
        return ClipResult(frames=frames, seconds=dt)

    # -- chaining ----------------------------------------------------------

    def build_chain(self, theme: Image.Image, prompts: list[str], seeds: list[int],
                    duration: float = 5.0, fps: int = 24, steps: int = 40,
                    progress: ProgressFn | None = None) -> tuple[list[Image.Image], list[ClipResult]]:
        """Walk the anchor chain, harvesting each clip's last frame as the next anchor.

        Returns (anchors, clips) where ``anchors[0]`` is the theme itself.
        """
        anchors = [theme]
        clips: list[ClipResult] = []

        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            def sub(frac: float, msg: str, i=i) -> None:
                if progress:
                    progress((i + frac) / len(prompts), f"clip {i + 1}/{len(prompts)}: {msg}")

            clip = self.generate(prompt, first=anchors[-1], last=None,
                                 duration=duration, fps=fps, seed=seed,
                                 steps=steps, progress=sub)
            clips.append(clip)
            anchors.append(clip.last)

        return anchors, clips


def save_clip(frames: list[Image.Image], path: str | Path, fps: int = 24) -> Path:
    """Write frames to an mp4 via the shared ffmpeg encoder."""
    from ..assemble.encode import FrameWriter

    path = Path(path)
    w, h = frames[0].size
    with FrameWriter(path, w, h, fps, audio=None, quality=18, loudnorm=False) as fw:
        for f in frames:
            fw.write(np.asarray(f.convert("RGB"), dtype=np.uint8))
    return path


def structure_score(img: Image.Image) -> float:
    """Large-scale structural contrast, in 0-255 units.

    Heavily downsampling averages away high-frequency detail. A real frame keeps
    its composition (bright sky over dark sea) and so retains a high spread;
    noise averages toward flat mid-grey and collapses toward zero.
    """
    small = np.asarray(img.convert("RGB").resize((32, 18), Image.BOX), dtype=np.float32)
    return float(np.mean(np.std(small.reshape(-1, 3), axis=0)))


def assert_not_noise(frames: list[Image.Image], reference: Image.Image,
                     min_ratio: float = 0.3) -> None:
    """Fail if generated frames are noise rather than a scene.

    Without this, every downstream metric silently measures noise against noise
    -- which is how a broken scheduler produced a "stable drift, PASS" verdict.
    Validate that the output is real before measuring anything about it.
    """
    ref = structure_score(reference)
    scores = [structure_score(f) for f in (frames[0], frames[len(frames) // 2], frames[-1])]
    worst = min(scores)
    if worst < ref * min_ratio:
        raise RuntimeError(
            f"generated frames look like noise, not video: structure "
            f"{[round(s, 1) for s in scores]} vs reference {ref:.1f} "
            f"(need >= {ref * min_ratio:.1f}). Metrics on this output would be "
            f"meaningless. Check the checkpoint/scheduler/pipeline match."
        )


def loop_seam_error(frames: list[Image.Image]) -> float:
    """Mean absolute difference between first and last frame, in 0-255 units.

    The headline metric for the anchor-chain spike: a genuinely loopable clip
    should score close to the frame-to-frame difference in the middle of the
    clip, not dramatically worse.
    """
    a = np.asarray(frames[0].convert("RGB"), dtype=np.float32)
    b = np.asarray(frames[-1].convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def mean_adjacent_error(frames: list[Image.Image]) -> float:
    """Baseline: average adjacent-frame difference, for comparison."""
    diffs = []
    for a, b in zip(frames, frames[1:]):
        fa = np.asarray(a.convert("RGB"), dtype=np.float32)
        fb = np.asarray(b.convert("RGB"), dtype=np.float32)
        diffs.append(np.mean(np.abs(fa - fb)))
    return float(np.mean(diffs)) if diffs else 0.0
