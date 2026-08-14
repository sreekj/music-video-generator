"""Theme image generation via the OpenAI images API.

Stage B renders camera moves and effects over stills. With a single still, a
4-minute video is inescapably monotonous -- grading alone cannot carry it. So a
song gets a *set* of images, assigned per structural section, and the renderer
cuts and dissolves between them on downbeats.

Images are cached by a hash of (model, prompt, size), so re-running a song costs
nothing and editing one prompt regenerates only that image.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

API_URL = "https://api.openai.com/v1/images/generations"
MODEL = os.environ.get("VC_IMAGE_MODEL", "gpt-image-2")
# 3:2 landscape; the renderer cover-crops to 16:9 and needs margin for camera
# moves, so generating wider than the target frame is deliberate.
SIZE = os.environ.get("VC_IMAGE_SIZE", "1536x1024")
QUALITY = os.environ.get("VC_IMAGE_QUALITY", "high")

IMAGES = config.CACHE / "images"


def api_key() -> str:
    """Read the key from the environment, or from a key file."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    for p in (Path("/tmp/oai.key"), config.ROOT / ".openai.key"):
        if p.exists():
            key = p.read_text().strip()
            if key:
                return key
    raise RuntimeError(
        "no OpenAI API key: set OPENAI_API_KEY or write the key to /tmp/oai.key"
    )


@dataclass
class ThemeImage:
    id: str
    prompt: str
    path: Path


def _cache_path(prompt: str) -> Path:
    h = hashlib.sha256(f"{MODEL}|{SIZE}|{QUALITY}|{prompt}".encode()).hexdigest()[:16]
    return IMAGES / f"{h}.png"


def generate(prompt: str, retries: int = 4) -> Path:
    """Generate one image, or return the cached file for this exact prompt."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = _cache_path(prompt)
    if out.exists() and out.stat().st_size > 0:
        # Verify it decodes: a cached file that is present but corrupt would
        # otherwise fail deep inside the renderer, far from the cause.
        try:
            from PIL import Image

            with Image.open(out) as im:
                im.verify()
            log.info("cached: %s", out.name)
            return out
        except Exception:  # noqa: BLE001 - corrupt cache entry, regenerate
            log.warning("cached image %s is corrupt; regenerating", out.name)
            out.unlink(missing_ok=True)

    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "size": SIZE,
        "quality": QUALITY,
        "n": 1,
    }).encode()

    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL, data=body,
            headers={"Authorization": f"Bearer {api_key()}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = json.load(r)
            item = payload["data"][0]
            # gpt-image-* returns base64; older models may return a URL.
            if "b64_json" in item:
                data = base64.b64decode(item["b64_json"])
            else:
                with urllib.request.urlopen(item["url"], timeout=300) as im:
                    data = im.read()
            if not data:
                raise RuntimeError("API returned an empty image")

            # Write via a temp file and rename. A plain write that is
            # interrupted leaves a truncated PNG which the cache would then
            # trust forever, silently poisoning every later render.
            tmp = out.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(out)
            log.info("generated %s (%.1f KB)", out.name, out.stat().st_size / 1024)
            return out
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            last = RuntimeError(f"HTTP {e.code}: {detail}")
            # Retry rate limits and server faults; fail immediately on a bad
            # request, which will not become valid by trying again.
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last from e
        except Exception as e:  # noqa: BLE001 - network flakiness
            last = e
        wait = 2 ** attempt * 5
        log.warning("image attempt %d/%d failed (%s); retrying in %ds",
                    attempt + 1, retries, last, wait)
        time.sleep(wait)

    raise RuntimeError(f"image generation failed after {retries} attempts: {last}")


def generate_set(specs: list[tuple[str, str]],
                 progress=None) -> list[ThemeImage]:
    """Generate a named set of images. ``specs`` is [(id, prompt), ...]."""
    out: list[ThemeImage] = []
    for i, (img_id, prompt) in enumerate(specs):
        if progress:
            progress(i / len(specs), f"image {i + 1}/{len(specs)}: {img_id}")
        out.append(ThemeImage(id=img_id, prompt=prompt, path=generate(prompt)))
    if progress:
        progress(1.0, f"{len(out)} images ready")
    return out
