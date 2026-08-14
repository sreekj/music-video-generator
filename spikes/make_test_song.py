"""Generate a synthetic test song with known structure.

Used as a fixture for the analysis pipeline: the section layout below is the
ground truth that segmentation should recover (the two choruses must land in
the same cluster, and outrank the verses on energy).

    python spikes/make_test_song.py -o data/uploads/test_song.mp3
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM

# (name, n_beats, drums, bass, lead) -- the ground truth structure
ARRANGEMENT = [
    ("intro", 16, False, False, False),
    ("verse", 32, True, True, False),
    ("chorus", 32, True, True, True),
    ("verse", 32, True, True, False),
    ("chorus", 32, True, True, True),
    ("outro", 16, False, False, False),
]

CHORD = [220.0, 277.18, 329.63]  # A major-ish pad
BASS_NOTES = [55.0, 55.0, 73.42, 65.41]  # one per bar
LEAD_NOTES = [440.0, 554.37, 659.25, 554.37]


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    a = max(1, int(attack * SR))
    e = np.ones(n)
    e[:a] = np.linspace(0, 1, a)
    e *= np.exp(-np.linspace(0, decay, n))
    return e


def _kick(dur: float = 0.18) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    # Pitch sweep 110 -> 45 Hz gives it a recognisable transient.
    freq = 45 + 65 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * _env(n, 0.001, 9) * 0.9


def _hat(dur: float = 0.05) -> np.ndarray:
    n = int(dur * SR)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n)
    # Crude high-pass: difference the noise.
    return np.diff(noise, prepend=0.0) * _env(n, 0.0005, 30) * 0.25


def _tone(freq: float, dur: float, harmonics: int = 1, amp: float = 0.2,
          decay: float = 2.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for h in range(1, harmonics + 1):
        sig += np.sin(2 * np.pi * freq * h * t) / h
    return sig * _env(n, 0.01, decay) * amp


def _add(buf: np.ndarray, sig: np.ndarray, at: float) -> None:
    i = int(at * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


def build() -> np.ndarray:
    total_beats = sum(s[1] for s in ARRANGEMENT)
    buf = np.zeros(int((total_beats * BEAT + 2.0) * SR))

    kick, hat = _kick(), _hat()
    beat_i = 0

    for _name, n_beats, drums, bass, lead in ARRANGEMENT:
        for b in range(n_beats):
            t = (beat_i + b) * BEAT
            bar_pos = b % 4

            if drums:
                if bar_pos in (0, 2):
                    _add(buf, kick, t)
                if lead:  # choruses get a busier kit
                    _add(buf, kick, t)
                    _add(buf, hat, t + BEAT / 2)
                _add(buf, hat, t)

            if bass and bar_pos == 0:
                note = BASS_NOTES[(b // 4) % len(BASS_NOTES)]
                _add(buf, _tone(note, BEAT * 4, harmonics=3, amp=0.30, decay=1.2), t)

            if lead and bar_pos in (0, 2):
                note = LEAD_NOTES[(b // 2) % len(LEAD_NOTES)]
                _add(buf, _tone(note, BEAT * 2, harmonics=6, amp=0.16, decay=3.0), t)

            # Pad chord on every bar, everywhere -- the harmonic bed.
            if bar_pos == 0:
                for f in CHORD:
                    _add(buf, _tone(f, BEAT * 4, harmonics=2, amp=0.07, decay=1.0), t)

        beat_i += n_beats

    peak = float(np.max(np.abs(buf))) or 1.0
    return (buf / peak * 0.89).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="data/uploads/test_song.mp3")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audio = build()

    wav = out.with_suffix(".wav")
    sf.write(wav, audio, SR)

    if out.suffix == ".mp3":
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k", str(out)],
            check=True,
        )
        wav.unlink()

    beats = sum(s[1] for s in ARRANGEMENT)
    print(f"wrote {out}  ({beats * BEAT:.1f}s, {BPM:.0f} BPM, {len(ARRANGEMENT)} sections)")
    print("ground truth:")
    at = 0.0
    for name, n, *_ in ARRANGEMENT:
        print(f"  {name:7s} {at:6.2f} - {at + n * BEAT:6.2f}")
        at += n * BEAT


if __name__ == "__main__":
    main()
