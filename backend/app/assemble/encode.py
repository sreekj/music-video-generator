"""ffmpeg frame sink and clip decoding.

Frames go out over a pipe as raw RGB rather than through intermediate PNGs --
at 473 fps the encoder is the only thing that should be touching the disk.
"""

from __future__ import annotations

import functools
import logging
import subprocess
from pathlib import Path

import numpy as np

from .. import config

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    """Video encoders ffmpeg reports. Raises if ffmpeg is missing or broken."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True, check=True).stdout
    return {line.split()[1] for line in out.splitlines()
            if line.startswith(" V") and len(line.split()) > 1}


def require_codec() -> str:
    """Return the configured codec, or raise if ffmpeg cannot provide it."""
    enc = available_encoders()
    if config.VIDEO_CODEC not in enc:
        raise RuntimeError(
            f"required video codec {config.VIDEO_CODEC!r} is not available in "
            f"this ffmpeg build. Install an nvenc-enabled ffmpeg, or set "
            f"VC_VIDEO_CODEC to one of: {sorted(enc)}"
        )
    return config.VIDEO_CODEC


class FrameWriter:
    """Writes raw RGB frames to ffmpeg, muxing audio in the same pass."""

    def __init__(self, out: str | Path, width: int, height: int, fps: int,
                 audio: str | Path | None = None, quality: int = 20,
                 loudnorm: bool = True, codec: str | None = None):
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height, self.fps = width, height, fps
        self.frames_written = 0

        codec = codec or require_codec()
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        ]
        if audio:
            cmd += ["-i", str(audio)]

        cmd += ["-map", "0:v"]
        if audio:
            cmd += ["-map", "1:a"]

        if codec.endswith("nvenc"):
            # p5 balances speed and quality; -cq is nvenc's CRF equivalent.
            cmd += ["-c:v", codec, "-preset", "p5", "-rc", "vbr", "-cq", str(quality),
                    "-b:v", "0"]
        else:
            cmd += ["-c:v", codec, "-preset", "medium", "-crf", str(quality)]

        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]

        if audio:
            cmd += ["-c:a", config.AUDIO_CODEC, "-b:a", config.AUDIO_BITRATE]
            if loudnorm:
                # loudnorm runs its internal analysis at 192kHz and leaves the
                # output resampled, so an explicit -ar is required or the muxed
                # audio lands at 96/192kHz instead of the source rate.
                cmd += ["-af", f"loudnorm=I={config.LOUDNESS_LUFS}:TP=-1.5:LRA=11"]
            cmd += ["-ar", str(config.AUDIO_SAMPLE_RATE), "-shortest"]

        cmd.append(str(self.out))
        self.cmd = cmd
        log.debug("ffmpeg: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: bytes | np.ndarray) -> None:
        if isinstance(frame, np.ndarray):
            frame = frame.tobytes()
        try:
            self.proc.stdin.write(frame)
        except BrokenPipeError as exc:
            err = self.proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg died after {self.frames_written} frames:\n{err}") from exc
        self.frames_written += 1

    def close(self) -> Path:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        code = self.proc.wait()
        err = self.proc.stderr.read().decode(errors="replace")
        self.proc.stderr.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited {code}:\n{err}")
        return self.out

    def __enter__(self) -> FrameWriter:
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.close()
        else:  # let the original exception surface, don't mask it
            self.proc.kill()
            self.proc.wait()


def decode_to_memmap(clip: str | Path, width: int, height: int,
                     cache_dir: Path | None = None) -> np.ndarray:
    """Decode a clip to a raw RGB memmap for random-access retiming.

    Retiming modes like ping-pong need arbitrary frame access; holding a full
    clip in RAM at render resolution is too expensive, so it goes to disk once
    and is memory-mapped thereafter.
    """
    clip = Path(clip)
    cache_dir = cache_dir or config.CLIPS
    raw = cache_dir / f"{clip.stem}_{width}x{height}.raw"

    if not raw.exists() or raw.stat().st_size == 0:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(clip),
             "-vf", f"scale={width}:{height}:flags=lanczos", "-f", "rawvideo",
             "-pix_fmt", "rgb24", str(raw)],
            check=True,
        )

    frame_bytes = width * height * 3
    n = raw.stat().st_size // frame_bytes
    if n == 0:
        raise RuntimeError(f"decoded no frames from {clip}")
    return np.memmap(raw, dtype=np.uint8, mode="r", shape=(n, height, width, 3))
