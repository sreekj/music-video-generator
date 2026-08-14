"""Audio analysis: MP3 -> timeline.

Produces the structural and reactive data that both the LLM Director and the
Stage B renderer consume:

  * beat grid + estimated downbeats (where cuts are allowed to land)
  * structural sections via Laplacian spectral clustering (verse/chorus/bridge
    boundaries, discovered rather than guessed)
  * per-video-frame feature curves (energy, onset, frequency bands, centroid)

Every curve in ``curves`` is resampled to the render fps, so Stage B can index
features by frame number with no interpolation at render time. That keeps the
hot render loop free of any librosa/scipy work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.sparse.csgraph
import librosa
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)

ANALYSIS_SR = 22050
HOP = 512
BINS_PER_OCTAVE = 12 * 3
N_OCTAVES = 7

# Frequency band edges (Hz) used for the reactive curves. These map onto the
# bindings the Director emits: kick -> low, vocals/body -> mid, hats -> high.
BANDS: dict[str, tuple[float, float]] = {
    "low": (20.0, 250.0),
    "mid": (250.0, 4000.0),
    "high": (4000.0, 16000.0),
}


@dataclass
class Section:
    """One structural region of the song."""

    id: int
    start: float
    end: float
    label: str  # cluster letter: A, B, C ...
    energy: float  # mean normalised RMS across the section
    energy_rank: int  # 0 = highest-energy cluster (usually the chorus)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Timeline:
    path: str
    duration: float
    fps: int
    n_frames: int
    tempo: float
    beats: list[float]
    downbeats: list[float]
    sections: list[Section]
    # name -> per-frame values in [0, 1], length == n_frames
    curves: dict[str, list[float]] = field(default_factory=dict)
    # coarse peak envelope for drawing the waveform in the web UI
    waveform_peaks: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sections"] = [asdict(s) for s in self.sections]
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _norm(name: str, x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A flat curve means the feature is dead, so raise.

    Returning zeros would leave every reactive binding on that curve silently
    doing nothing, which is far harder to diagnose than a failed analysis.
    """
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        raise ValueError(
            f"audio feature {name!r} is constant ({lo:.6g}) across the whole "
            f"track -- the input is probably silent or corrupt"
        )
    return (x - lo) / (hi - lo)


def _resample_to_fps(values: np.ndarray, sr: int, hop: int, n_frames: int, fps: int) -> np.ndarray:
    """Map a hop-length feature curve onto video frame times."""
    src_times = librosa.frames_to_time(np.arange(len(values)), sr=sr, hop_length=hop)
    dst_times = np.arange(n_frames) / fps
    return np.interp(dst_times, src_times, values)


def _estimate_downbeats(beat_times: np.ndarray, onset_env: np.ndarray, beats: np.ndarray,
                        meter: int = 4) -> np.ndarray:
    """Pick the beat phase whose onset strength is consistently strongest.

    A real downbeat tracker would model harmony too; for cut placement the
    strongest-phase heuristic is accurate enough and never off by more than one
    beat, which a beat-locked cut tolerates.
    """
    if len(beats) < meter:
        raise ValueError(
            f"only {len(beats)} beats detected -- need at least {meter} to "
            f"establish a downbeat phase"
        )

    strengths = onset_env[np.clip(beats, 0, len(onset_env) - 1)]
    best_phase, best_score = 0, -np.inf
    for phase in range(meter):
        score = float(np.mean(strengths[phase::meter]))
        if score > best_score:
            best_phase, best_score = phase, score
    return beat_times[best_phase::meter]


def _segment(y: np.ndarray, sr: int, beats: np.ndarray, k: int) -> np.ndarray:
    """Laplacian structural segmentation. Returns a cluster id per beat.

    Combines a recurrence (repetition) affinity with a local path affinity,
    then clusters the normalised graph Laplacian's leading eigenvectors. This
    is what finds "this chorus is the same as that chorus" without any labels.
    """
    C = librosa.amplitude_to_db(
        np.abs(librosa.cqt(y=y, sr=sr, hop_length=HOP,
                           bins_per_octave=BINS_PER_OCTAVE,
                           n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
        ref=np.max,
    )
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # Repetition affinity: which beats sound like which other beats.
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode="affinity", sym=True)
    median_filter = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = median_filter(R, size=(1, 7))

    # Local affinity: consecutive beats with similar timbre stay together.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP)
    Msync = librosa.util.sync(mfcc, beats)
    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = float(np.median(path_distance))
    if sigma <= 0.0:
        raise ValueError(
            "median beat-to-beat timbre distance is zero -- every beat sounds "
            "identical, so the track cannot be segmented"
        )
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # Balance the two graphs so neither dominates (McFee & Ellis).
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    denom = float(np.sum((deg_path + deg_rec) ** 2))
    if denom <= 0.0:
        raise ValueError("both affinity graphs are empty -- cannot segment")
    mu = float(deg_path.dot(deg_path + deg_rec) / denom)
    A = mu * Rf + (1.0 - mu) * R_path

    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    _evals, evecs = scipy.linalg.eigh(L)
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

    if not 2 <= k <= evecs.shape[1]:
        raise ValueError(
            f"requested {k} sections but only {evecs.shape[1]} eigenvectors are "
            f"available (track has {len(beats)} beats); ask for fewer sections"
        )
    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5
    X = evecs[:, :k] / np.maximum(Cnorm[:, k - 1: k], 1e-9)

    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


def _merge_short(raw: list[tuple[int, float, float]], min_dur: float) -> list[tuple[int, float, float]]:
    """Merge sub-``min_dur`` runs into a neighbour.

    Spectral clustering happily emits 3-second fragments when a song has
    internal call-and-response structure. Those are too short to hang a
    generative shot on, so absorb them into whichever neighbour is shorter
    (which keeps long sections from swallowing everything).
    """
    if len(raw) <= 1:
        return raw

    segs = [list(s) for s in raw]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (_cid, start, end) in enumerate(segs):
            if end - start >= min_dur:
                continue
            prev_d = segs[i - 1][2] - segs[i - 1][1] if i > 0 else np.inf
            next_d = segs[i + 1][2] - segs[i + 1][1] if i < len(segs) - 1 else np.inf
            if prev_d <= next_d:
                segs[i - 1][2] = end  # extend previous over this one
            else:
                segs[i + 1][1] = start  # pull next back over this one
            segs.pop(i)
            changed = True
            break

    return [(int(c), float(s), float(e)) for c, s, e in segs]


def _sections_from_labels(seg_ids: np.ndarray, beat_times: np.ndarray, duration: float,
                          rms_frames: np.ndarray, sr: int,
                          min_section: float = 8.0) -> list[Section]:
    """Collapse per-beat cluster ids into contiguous, labelled sections."""
    bounds = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])
    bounds = librosa.util.fix_frames(bounds, x_min=0, x_max=len(seg_ids))

    raw: list[tuple[int, float, float]] = []
    for i in range(len(bounds) - 1):
        b0, b1 = int(bounds[i]), int(bounds[i + 1])
        start = float(beat_times[b0]) if b0 < len(beat_times) else duration
        end = float(beat_times[b1]) if b1 < len(beat_times) else duration
        if end - start < 1e-3:
            continue
        raw.append((int(seg_ids[b0]), start, end))

    if not raw:
        raise ValueError(
            "segmentation produced no usable sections -- every cluster run was "
            "shorter than a single beat"
        )

    raw[0] = (raw[0][0], 0.0, raw[0][2])
    raw[-1] = (raw[-1][0], raw[-1][1], duration)
    raw = _merge_short(raw, min_section)

    # Mean energy per cluster, so the Director can tell chorus from verse.
    rms_times = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=HOP)
    cluster_energy: dict[int, list[float]] = {}
    seg_energy: list[float] = []
    for cid, start, end in raw:
        mask = (rms_times >= start) & (rms_times < end)
        e = float(np.mean(rms_frames[mask])) if np.any(mask) else 0.0
        seg_energy.append(e)
        cluster_energy.setdefault(cid, []).append(e)

    ranked = sorted(cluster_energy, key=lambda c: -float(np.mean(cluster_energy[c])))
    rank_of = {cid: i for i, cid in enumerate(ranked)}

    return [
        Section(
            id=i,
            start=round(start, 3),
            end=round(end, 3),
            label=chr(ord("A") + (cid % 26)),
            energy=round(seg_energy[i], 4),
            energy_rank=rank_of[cid],
        )
        for i, (cid, start, end) in enumerate(raw)
    ]


def analyze(path: str | Path, fps: int = 30, n_sections: int | None = None,
            min_section: float = 8.0) -> Timeline:
    """Analyse an audio file into a Timeline.

    Args:
        path: audio file (anything ffmpeg/soundfile can read).
        fps: render frame rate; all curves are resampled to this.
        n_sections: cluster count for segmentation. Defaults to a
            duration-derived heuristic (~one section per 25s, clamped 3..10).
        min_section: shorter runs are merged into a neighbour.
    """
    path = Path(path)
    y, sr = librosa.load(str(path), sr=ANALYSIS_SR, mono=True)
    duration = float(len(y) / sr)
    n_frames = max(1, int(round(duration * fps)))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo_raw, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, trim=False
    )
    tempo = float(np.atleast_1d(tempo_raw)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
    downbeat_times = _estimate_downbeats(beat_times, onset_env, beats)

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]

    # Per-band energy for the reactive bindings.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_curves: dict[str, np.ndarray] = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_curves[name] = S[mask].mean(axis=0) if np.any(mask) else np.zeros(S.shape[1])

    # Structural segmentation. Uniform blocks would be a lie -- the Director
    # picks looks from section labels and energy ranks, so fabricated sections
    # produce a video whose cuts have nothing to do with the music.
    if n_sections is None:
        n_sections = int(np.clip(round(duration / 25.0), 3, 10))
    if len(beats) < max(8, n_sections * 2):
        raise ValueError(
            f"only {len(beats)} beats detected in {duration:.1f}s -- too few to "
            f"segment into {n_sections} sections. The audio may be silent, "
            f"arrhythmic, or too short."
        )
    seg_ids = _segment(y, sr, beats, n_sections)
    sections = _sections_from_labels(seg_ids, beat_times, duration, rms, sr, min_section)

    curves_src = {
        "energy": rms,
        "onset": onset_env,
        "centroid": centroid,
        **{f"band_{k}": v for k, v in band_curves.items()},
    }
    curves = {
        name: [round(float(v), 4)
               for v in _resample_to_fps(_norm(name, vals), sr, HOP, n_frames, fps)]
        for name, vals in curves_src.items()
    }

    # Coarse peak envelope for the UI waveform.
    n_peaks = 800
    chunk = max(1, len(y) // n_peaks)
    peaks = [float(np.max(np.abs(y[i:i + chunk]))) for i in range(0, len(y), chunk)][:n_peaks]

    return Timeline(
        path=str(path),
        duration=round(duration, 3),
        fps=fps,
        n_frames=n_frames,
        tempo=round(tempo, 2),
        beats=[round(float(t), 4) for t in beat_times],
        downbeats=[round(float(t), 4) for t in downbeat_times],
        sections=sections,
        curves=curves,
        waveform_peaks=[round(p, 4) for p in peaks],
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Analyse audio into a timeline.json")
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="timeline.json")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sections", type=int, default=None)
    ap.add_argument("--min-section", type=float, default=8.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tl = analyze(args.audio, fps=args.fps, n_sections=args.sections,
                 min_section=args.min_section)
    out = tl.save(args.out)

    print(f"{tl.duration:.1f}s @ {tl.tempo:.1f} BPM, {len(tl.beats)} beats, "
          f"{len(tl.sections)} sections -> {out}")
    for s in tl.sections:
        print(f"  [{s.label}] {s.start:7.2f} - {s.end:7.2f}  "
              f"({s.duration:5.1f}s)  energy={s.energy:.3f} rank={s.energy_rank}")


if __name__ == "__main__":
    main()
