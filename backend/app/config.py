"""Paths and render defaults.

The Stage A / Stage B split described in the README lives here as two distinct
cache roots: ANCHORS and CLIPS are expensive, content-addressed, and survive
across renders; OUT is cheap and disposable.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

UPLOADS = DATA / "uploads"
CACHE = DATA / "cache"
OUT = DATA / "out"

# Stage A artefacts, keyed by content hash so changing one shot does not
# invalidate the rest of the bank.
ANCHORS = CACHE / "anchors"
CLIPS = CACHE / "clips"

for _d in (UPLOADS, CACHE, OUT, ANCHORS, CLIPS):
    _d.mkdir(parents=True, exist_ok=True)

# --- render defaults -------------------------------------------------------

FPS = 30
WIDTH = 1920
HEIGHT = 1080

# Headless GL on WSL2. Mesa's d3d12 gallium driver reaches the real GPU through
# /dev/dxg, but defaults to the *integrated* adapter -- naming NVIDIA explicitly
# is what gets us the 4090 (473 fps @1080p vs 84 on llvmpipe).
GL_ENV = {
    "GALLIUM_DRIVER": "d3d12",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
}

# GL_RENDERER must contain this. Silently landing on llvmpipe or the integrated
# GPU is a 5.6x slowdown that looks like "the renderer is just slow" -- always a
# broken environment, never something to quietly accept.
REQUIRED_GL_RENDERER = os.environ.get("VC_REQUIRE_GL", "NVIDIA")


def setup_gl_env() -> None:
    """Point Mesa at the discrete GPU. Call before creating a GL context."""
    for k, v in GL_ENV.items():
        os.environ.setdefault(k, v)


def verify_gl_renderer(renderer: str) -> None:
    """Fail if GL did not land on the required device."""
    if REQUIRED_GL_RENDERER not in renderer:
        raise RuntimeError(
            f"GL context is on {renderer!r}, which does not match required "
            f"{REQUIRED_GL_RENDERER!r}. Check /dev/dxg exists and Mesa's d3d12 "
            f"driver is installed. Set VC_REQUIRE_GL to override the requirement."
        )

# Preview is deliberately small: Stage B must round-trip in a couple of
# seconds for the UI to feel live.
PREVIEW_WIDTH = 854
PREVIEW_HEIGHT = 480
PREVIEW_FPS = 24

# Required video codec. If it is missing, that is an environment fault to fix,
# not a reason to silently encode 10x slower on libx264. Override with
# VC_VIDEO_CODEC if you genuinely want a different encoder.
VIDEO_CODEC = os.environ.get("VC_VIDEO_CODEC", "h264_nvenc")
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
# YouTube's recommended rate. Must be set explicitly: loudnorm would otherwise
# leave the stream at its 192kHz internal analysis rate.
AUDIO_SAMPLE_RATE = 48000

# YouTube's normalisation target; keeps loudness sane across uploads.
LOUDNESS_LUFS = -14.0

# --- Stage A (generation) --------------------------------------------------

# Must match the checkpoint the installed diffusers LTXConditionPipeline was
# written against -- see its EXAMPLE_DOC_STRING. The base "Lightricks/LTX-Video"
# repo ships a scheduler wanting dynamic shifting, which this pipeline cannot
# drive; forcing it produces noise rather than an error.
LTX_MODEL = os.environ.get("VC_LTX_MODEL", "Lightricks/LTX-Video-0.9.5")
LTX_DTYPE = os.environ.get("VC_LTX_DTYPE", "bfloat16")
GEN_WIDTH = 1216
GEN_HEIGHT = 704
