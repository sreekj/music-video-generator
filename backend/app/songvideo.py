"""End-to-end: a song file -> a finished music video.

    python -m app.songvideo song.wav --themes ../themes/example.json -o out.mp4

Generates a set of theme images from a themes file, analyses the audio, assigns
imagery per structural section by energy, and renders 1080p with the audio muxed
and loudness-normalised.

Themes are data, not code: see ``themes/example.json`` for the format. Images are
grouped into three energy tiers -- tier 0 plays under the loudest sections of the
song, tier 2 under the quietest -- so the visuals track the music rather than
cycling arbitrarily.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from . import config
from .assemble.render import render
from .audio.analysis import analyze
from .director.builder import build
from .generate.images import generate_set

log = logging.getLogger(__name__)

N_TIERS = 3


def load_themes(path: str | Path) -> tuple[dict[int, list[tuple[str, str]]], str]:
    """Load a themes file into {tier: [(id, prompt), ...]} plus the cover id.

    A shared ``style`` string is appended to every prompt so a theme set holds
    together visually without repeating the same clause nine times.
    """
    path = Path(path)
    data = json.loads(path.read_text())

    style = data.get("style", "").strip()
    tiers_raw = data.get("tiers")
    if not isinstance(tiers_raw, dict):
        raise ValueError(f"{path}: 'tiers' must be an object keyed by tier number")

    tiers: dict[int, list[tuple[str, str]]] = {}
    seen: set[str] = set()
    for t in range(N_TIERS):
        entries = tiers_raw.get(str(t)) or tiers_raw.get(t)
        if not entries:
            raise ValueError(
                f"{path}: tier {t} is empty. All {N_TIERS} tiers need at least "
                f"one image, or sections at that energy have nothing to show."
            )
        pairs = []
        for e in entries:
            img_id, prompt = e["id"], e["prompt"].strip()
            if img_id in seen:
                raise ValueError(f"{path}: duplicate image id {img_id!r}")
            seen.add(img_id)
            pairs.append((img_id, f"{prompt}, {style}" if style else prompt))
        tiers[t] = pairs

    cover = data.get("cover") or tiers[N_TIERS - 1][0][0]
    if cover not in seen:
        raise ValueError(f"{path}: cover {cover!r} is not one of the image ids")
    return tiers, cover


def make_video(audio: str | Path, out: str | Path, themes: str | Path,
               prompt: str = "", preview: bool = False,
               seconds: float | None = None) -> Path:
    audio = Path(audio)
    t0 = time.perf_counter()
    log.info("=== %s ===", audio.name)

    tiers, cover = load_themes(themes)
    specs = [(i, p) for tier in tiers.values() for i, p in tier]
    log.info("generating/loading %d theme images", len(specs))
    imgs = generate_set(specs, progress=lambda f, m: log.info("  %s", m))
    by_id = {im.id: im.path for im in imgs}
    tier_ids = {t: [i for i, _ in v] for t, v in tiers.items()}

    log.info("analysing audio")
    tl = analyze(audio, fps=config.FPS)
    log.info("  %.1fs, %.0f BPM, %d beats, %d sections",
             tl.duration, tl.tempo, len(tl.beats), len(tl.sections))

    theme_img = by_id[cover]
    sl = build(tl, theme=str(theme_img), prompt=prompt, generative=False,
               images=tier_ids)
    log.info("  %d shots across %d distinct images",
             len(sl.shots), len({s.still for s in sl.shots}))

    sources = {"theme": theme_img, **by_id}
    path = render(sl, tl, out, sources=sources, preview=preview, audio=audio,
                  end=seconds, progress=lambda f, m: None)
    log.info("done in %.0fs -> %s", time.perf_counter() - t0, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--themes", default=str(config.ROOT / "themes" / "example.json"),
                    help="themes JSON (see themes/example.json)")
    ap.add_argument("--prompt", default="", help="recorded in the shotlist for provenance")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    make_video(args.audio, args.out, themes=args.themes, prompt=args.prompt,
               preview=args.preview, seconds=args.seconds)


if __name__ == "__main__":
    main()
