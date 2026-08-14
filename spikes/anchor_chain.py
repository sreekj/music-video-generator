"""SPIKE: does first+last frame conditioning actually hold the anchor chain up?

This is the risk concentrator for the whole generative-first design. Three
claims are load-bearing, and all three are cheap to falsify here before any UI
is built on top of them:

  A. LOOP     -- conditioning first==last yields a seamlessly loopable clip.
                 Measured: seam error vs. the clip's own adjacent-frame error.
                 A real loop scores near 1.0x; a bad one scores many times worse.
  B. MORPH    -- conditioning on two *different* anchors produces a controlled
                 transition that actually arrives at the target.
                 Measured: how closely the final frame matches the requested one.
  C. CHAIN    -- harvesting each clip's last frame as the next anchor keeps the
                 look stable instead of drifting away from the theme.
                 Measured: drift of each anchor from the theme over the chain.

Run:
    cd backend && ../.venv/bin/python ../spikes/anchor_chain.py \
        --theme ../data/uploads/theme.png --prompt "stormy sea at dusk"

Outputs land in data/out/spike/ as mp4s plus a contact sheet per test.
Nothing here is imported by the app; it exists to produce a verdict.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config  # noqa: E402
from app.generate.ltx import (  # noqa: E402
    LTXGenerator, assert_not_noise, loop_seam_error, mean_adjacent_error,
    save_clip, snap_frames, structure_score,
)

log = logging.getLogger("spike")

OUT = config.OUT / "spike"


def frame_diff(a: Image.Image, b: Image.Image) -> float:
    fa = np.asarray(a.convert("RGB").resize((256, 144)), dtype=np.float32)
    fb = np.asarray(b.convert("RGB").resize((256, 144)), dtype=np.float32)
    return float(np.mean(np.abs(fa - fb)))


def contact_sheet(frames: list[Image.Image], path: Path, n: int = 6) -> Path:
    """Evenly-spaced frames in a row, for eyeballing motion and drift."""
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    picks = [frames[i].convert("RGB") for i in idx]
    w, h = picks[0].size
    scale = 320 / w
    tw, th = int(w * scale), int(h * scale)
    sheet = Image.new("RGB", (tw * n, th))
    for j, im in enumerate(picks):
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (j * tw, 0))
    sheet.save(path)
    return path


def report(title: str, lines: list[str]) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
    for ln in lines:
        print(f"  {ln}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theme", default="../data/uploads/theme.png")
    ap.add_argument("--prompt", default="stormy sea at dusk, drifting clouds, gentle swell")
    ap.add_argument("--model", default=None, help="HF model id (defaults to config.LTX_MODEL)")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chain", type=int, default=3, help="clips in the chain test")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--skip", default="", help="comma-separated: loop,morph,chain")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    theme = Image.open(args.theme).convert("RGB")
    n_frames = snap_frames(args.duration * args.fps)
    print(f"theme      : {args.theme} ({theme.width}x{theme.height})")
    print(f"prompt     : {args.prompt!r}")
    print(f"clip       : {n_frames} frames @ {args.fps}fps "
          f"({n_frames / args.fps:.2f}s), {args.width}x{args.height}, {args.steps} steps")

    gen = LTXGenerator(model=args.model, width=args.width, height=args.height)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this spike measures GPU generation")

    verdicts: dict[str, str] = {}
    t_start = time.perf_counter()

    # ---- A. LOOP -------------------------------------------------------
    if "loop" not in skip:
        clip = gen.generate(args.prompt, first=theme, last=theme,
                            duration=args.duration, fps=args.fps, seed=args.seed,
                            steps=args.steps,
                            progress=lambda f, m: print(f"  loop  {f * 100:5.1f}% {m}", end="\r"))
        print()
        # Gate every metric below on the output being real video. A broken
        # scheduler once produced noise that scored a confident PASS on drift.
        assert_not_noise(clip.frames, theme)
        print(f"  sanity: structure {structure_score(clip.frames[-1]):.1f} "
              f"vs theme {structure_score(theme):.1f} -- real output")
        seam = loop_seam_error(clip.frames)
        adj = mean_adjacent_error(clip.frames)
        ratio = seam / adj if adj > 1e-6 else float("inf")
        save_clip(clip.frames, OUT / "a_loop.mp4", fps=args.fps)
        contact_sheet(clip.frames, OUT / "a_loop.png")

        ok = ratio < 3.0
        verdicts["A loop"] = "PASS" if ok else "FAIL"
        report("A. LOOP -- first==last conditioning", [
            f"seam error (first vs last)   : {seam:6.2f} / 255",
            f"mean adjacent-frame error    : {adj:6.2f} / 255",
            f"ratio (want < 3.0)           : {ratio:6.2f}x  -> {verdicts['A loop']}",
            f"generation time              : {clip.seconds:6.1f}s",
            f"artefacts                    : {OUT / 'a_loop.mp4'}",
        ])

    # ---- B. MORPH ------------------------------------------------------
    if "morph" not in skip:
        # A visibly different target: darker, higher contrast version of the theme.
        arr = np.asarray(theme, dtype=np.float32) / 255.0
        target = Image.fromarray(
            (np.clip((arr - 0.5) * 1.35 + 0.5, 0, 1) * np.array([0.55, 0.62, 1.15])
             .clip(0, 1) * 255).astype(np.uint8).clip(0, 255))

        clip = gen.generate(args.prompt + ", turning to night, storm building",
                            first=theme, last=target,
                            duration=args.duration, fps=args.fps, seed=args.seed + 1,
                            steps=args.steps,
                            progress=lambda f, m: print(f"  morph {f * 100:5.1f}% {m}", end="\r"))
        print()
        arrive = frame_diff(clip.frames[-1], target)
        start_gap = frame_diff(theme, target)
        save_clip(clip.frames, OUT / "b_morph.mp4", fps=args.fps)
        contact_sheet(clip.frames, OUT / "b_morph.png")
        target.save(OUT / "b_target.png")

        ok = arrive < start_gap * 0.5
        verdicts["B morph"] = "PASS" if ok else "FAIL"
        report("B. MORPH -- first != last conditioning", [
            f"theme vs target (baseline)   : {start_gap:6.2f} / 255",
            f"final frame vs target        : {arrive:6.2f} / 255",
            f"arrival (want < 50% of base) : {arrive / start_gap * 100:5.1f}% -> {verdicts['B morph']}",
            f"generation time              : {clip.seconds:6.1f}s",
            f"artefacts                    : {OUT / 'b_morph.mp4'}",
        ])

    # ---- C. CHAIN ------------------------------------------------------
    if "chain" not in skip:
        prompts = [args.prompt] * args.chain
        seeds = [args.seed + 10 + i * 7 for i in range(args.chain)]
        anchors, clips = gen.build_chain(
            theme, prompts, seeds, duration=args.duration, fps=args.fps,
            steps=args.steps,
            progress=lambda f, m: print(f"  chain {f * 100:5.1f}% {m}", end="\r"))
        print()

        drift = [frame_diff(a, theme) for a in anchors]
        for i, c in enumerate(clips):
            save_clip(c.frames, OUT / f"c_chain_{i}.mp4", fps=args.fps)
        contact_sheet(anchors, OUT / "c_anchors.png", n=len(anchors))

        # Drift should grow sub-linearly, not run away.
        growth = [drift[i + 1] - drift[i] for i in range(len(drift) - 1)]
        runaway = len(growth) > 1 and growth[-1] > growth[0] * 2.0
        ok = drift[-1] < 90.0 and not runaway
        verdicts["C chain"] = "PASS" if ok else "FAIL"
        report(f"C. CHAIN -- {args.chain} clips, anchors harvested from last frames", [
            "drift from theme per anchor  : "
            + ", ".join(f"a{i}={d:.1f}" for i, d in enumerate(drift)),
            f"per-step growth              : "
            + ", ".join(f"{g:+.1f}" for g in growth),
            f"final drift (want < 90)      : {drift[-1]:6.2f} / 255 -> {verdicts['C chain']}",
            f"runaway growth               : {'YES' if runaway else 'no'}",
            f"artefacts                    : {OUT / 'c_anchors.png'}",
        ])

    # ---- verdict -------------------------------------------------------
    vram = torch.cuda.max_memory_allocated() / 1e9

    report("VERDICT", [
        *[f"{k:10s} {v}" for k, v in verdicts.items()],
        f"peak VRAM  {vram:.1f} GB (budget 24 GB)",
        f"wall clock {time.perf_counter() - t_start:.0f}s",
        "",
        "If A fails: loops are not seamless -- fall back to ping-pong retiming",
        "            in Stage B (already implemented) and drop the loop claim.",
        "If B fails: the model ignores the last-frame condition -- the anchor",
        "            chain degrades to free-running I2V and drift control is lost.",
        "If C fails: reduce clips per chain and re-pin to the theme more often,",
        "            or generate anchors with a separate image model instead.",
    ])
    return 0 if all(v == "PASS" for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
