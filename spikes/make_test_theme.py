"""Generate a synthetic theme image for pipeline testing.

Deliberately full of high-frequency detail, a strong horizon and a bright
light source, so camera moves, bloom and god rays are all visibly exercised.

    python spikes/make_test_theme.py -o data/uploads/theme.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 2048, 1152


def build() -> Image.Image:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H

    img = np.zeros((H, W, 3), np.float32)

    # Sky gradient: deep indigo to warm horizon.
    horizon = 0.58
    sky = np.clip(v / horizon, 0, 1)
    img[..., 0] = 0.05 + 0.85 * sky ** 2.2
    img[..., 1] = 0.07 + 0.45 * sky ** 1.8
    img[..., 2] = 0.22 + 0.25 * sky ** 1.2

    # Sun near the horizon -- the god-ray / bloom source.
    sx, sy = 0.63, horizon - 0.03
    d = np.sqrt(((u - sx) * (W / H)) ** 2 + (v - sy) ** 2)
    img += np.exp(-d * 26.0)[..., None] * np.array([1.0, 0.82, 0.55], np.float32) * 2.2
    img += np.exp(-d * 5.0)[..., None] * np.array([0.9, 0.5, 0.25], np.float32) * 0.35

    # Layered cloud bands (value noise, cheap fbm).
    def noise(scale: float, seed: int) -> np.ndarray:
        h, w = int(H / scale), int(W / scale)
        n = rng.random((max(h, 2), max(w, 2))).astype(np.float32)
        return np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC),
                          np.float32) / 255.0

    fbm = sum(noise(s, i) * a for i, (s, a) in enumerate([(180, .5), (90, .25), (45, .15), (22, .1)]))
    cloud = np.clip((fbm - 0.45) * 3.0, 0, 1) * np.clip(1.0 - abs(v - 0.34) * 4.5, 0, 1)
    img = img * (1 - cloud[..., None] * 0.65) + cloud[..., None] * np.array([0.95, 0.72, 0.62], np.float32)

    # Water below the horizon: mirrored sky, banded ripples, sun glitter path.
    water = v > horizon
    depth = np.clip((v - horizon) / (1 - horizon), 0, 1)
    mirror = np.clip(1.0 - depth * 1.6, 0, 1)
    ripple = (np.sin(v * 420 + np.sin(u * 30) * 2.5) * 0.5 + 0.5) * (0.05 + 0.16 * (1 - depth))
    glitter = np.exp(-((u - sx) ** 2) / (0.006 + depth * 0.09)) * (1 - depth) * ripple * 7.0

    wcol = np.array([0.06, 0.12, 0.24], np.float32) * (1 - depth[..., None] * 0.55)
    wcol += mirror[..., None] * np.array([0.35, 0.22, 0.16], np.float32) * 0.55
    wcol += glitter[..., None] * np.array([1.0, 0.85, 0.6], np.float32)
    wcol += ripple[..., None] * 0.35
    img = np.where(water[..., None], wcol, img)

    # Foreground silhouette: rocky headland, keeps parallax legible.
    ridge = (0.80 + 0.045 * np.sin(u * 11.0) + 0.03 * np.sin(u * 27.0 + 1.3)
             + 0.018 * np.sin(u * 63.0))
    img = np.where((v > ridge)[..., None], img * 0.06 + 0.012, img)

    img += rng.normal(0, 0.006, img.shape).astype(np.float32)  # break up banding
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="data/uploads/theme.png")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build().save(out)
    print(f"wrote {out} ({W}x{H})")


if __name__ == "__main__":
    main()
