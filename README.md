# VideoCreator

Turn an **MP3 + a theme image + a prompt** into a full-length music video.

## The core problem

A 4-minute song is ~7,200 frames. Generative video models produce 2–5 second
clips. The whole architecture is an answer to *how do you get four minutes of
coherent motion without burning GPU-hours or drifting away from the theme.*

The answer has two halves:

**1. Anchor chains.** Every generated clip is conditioned on a still keyframe at
*both* ends (LTX supports first+last frame conditioning). That single mechanism
solves three problems at once:

- **Seamless loops** — first frame == last frame, no crossfade smear.
- **Drift control** — both ends are pinned to a frame derived from the theme.
- **Continuity** — shot N's end anchor is shot N+1's start anchor, so clips
  chain into a continuous film instead of unrelated bursts.

**2. A hard Stage A / Stage B split.** Generation is slow and cached;
assembly is fast and re-runnable. Live preview only ever re-runs Stage B.

```
STAGE A -- Clip Bank (minutes, cached by content hash)
  theme.png -> anchor frames (one per section, chained)
                    |
                    +-> LTX I2V, first+last frame conditioned
                        a0->a1, a1->a2, a2->a3 ...
                                |
                        data/cache/clips/

STAGE B -- Assembly (seconds; this is what preview drives)
  shotlist.json + clips -> moderngl: camera, grade, fx, transitions, beat-sync
                                |
                        ffmpeg h264_nvenc -> out.mp4
```

Get that boundary right and the UI feels responsive despite a generative core.
Get it wrong and every slider drag costs three minutes of GPU.

## The shotlist is the contract

`shotlist.json` is the only artefact crossing module boundaries. The Director
writes it, the UI edits it, the renderer consumes it. It is inspectable,
diffable, hand-editable, and deterministic — same shotlist + same seed produces
the same video.

```json
{
  "anchors": [
    { "id": "a_a", "source": "theme", "prompt": "stormy sea at dusk", "seed": 1000 },
    { "id": "a_b", "source": "a_a",   "prompt": "lightning on the horizon", "seed": 1037 }
  ],
  "shots": [
    { "start": 24.03, "end": 40.03, "section": "B",
      "clip": { "first": "a_b", "last": "a_b", "duration": 8.0, "seed": 2034 },
      "retime": "loop_pingpong",
      "camera": { "path": "push_in", "amplitude": 0.35 },
      "fx": ["bloom", "god_rays", "chroma"],
      "reactive": [{ "curve": "band_low", "target": "zoom_pulse", "amount": 0.9 }],
      "transition_in": { "type": "cut" } }
  ]
}
```

A shot's `duration: 8.0` covering a 16-second slot is normal — `retime` does the
stretching. ~60s of generated motion covers a 4-minute song.

## Multi-image mode (what works today)

The generative layer is still unproven (below), but the renderer does not need
it. Give a song a *set* of stills and it assigns them to structural sections by
energy — so the video cuts between real imagery instead of re-grading one
picture for four minutes, which is what makes a single-image render monotonous.

Images live in a themes file (`themes/example.json`) in three energy tiers:

| Tier | Plays under |
|---|---|
| 0 | the loudest sections |
| 1 | mid-energy sections |
| 2 | intros, outros, quiet passages |

A section *label* keeps its image, so a returning refrain brings back its own
visual rather than being re-dressed at random. Tiers are chosen from each
label's duration-weighted mean energy across every occurrence — not its first
appearance, which would send a chorus that enters quietly to the wrong tier.

```bash
cd backend && ../.venv/bin/python -m app.songvideo \
    song.wav --themes ../themes/example.json -o out.mp4
```

Images are generated once via the OpenAI images API and cached by prompt hash,
so re-rendering is free and editing one prompt regenerates only that image.
Set `OPENAI_API_KEY`, or write the key to `.openai.key` (gitignored).

## Status

| Stage | State |
|---|---|
| Audio analysis (beats, sections, curves) | **working, validated** |
| Shotlist schema + deterministic builder | **working** |
| Stage B renderer (GLSL, transitions, reactive) | **working, 3.5× realtime @1080p** |
| Multi-image assignment by section energy | **working** |
| Theme image generation + caching | **working** |
| FastAPI + WebSocket progress | **working** |
| Web UI (waveform, shot inspector, live preview) | **working** — untested in a browser |
| Stage A (LTX anchor chain) | **unproven** — see `spikes/anchor_chain.py` |
| LLM Director | not started |
| Lyrics (whisper karaoke) | not started |

### On the LTX spike

Its verdicts are currently **void**. An earlier run pointed at a checkpoint
whose scheduler the installed pipeline cannot drive, and a workaround that
disabled dynamic shifting let it run while producing pure noise — which every
downstream metric then measured, reporting a confident PASS on drift. The
workaround is gone (`_check_scheduler` now refuses the mismatch outright) and
`assert_not_noise()` gates every metric, but the spike has not been re-run
against a matching checkpoint. Treat first+last-frame conditioning as an open
question.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Generation additionally needs `requirements-gpu.txt` (torch + diffusers).

## Usage

```bash
# fixtures
.venv/bin/python spikes/make_test_song.py  -o data/uploads/test_song.mp3
.venv/bin/python spikes/make_test_theme.py -o data/uploads/theme.png

# analyse only
cd backend && ../.venv/bin/python -m app.audio.analysis ../data/uploads/test_song.mp3

# full render (Stage B)
cd backend && ../.venv/bin/python -m app.assemble.render \
    ../data/uploads/test_song.mp3 ../data/uploads/theme.png \
    -o ../data/out/test.mp4 --prompt "stormy sea at dusk"

# API server
cd backend && ../.venv/bin/python -m app.main       # :8000
```

## Failure policy: fail fast, no fallbacks

There are no degraded modes anywhere in this codebase. Every fallback that a
pipeline like this normally accumulates has been deliberately removed, because
each one produces a video that *looks* fine while being silently wrong — the
worst possible failure for a rendering tool, since nothing alerts you and the
output is only wrong in ways you have to notice by eye.

Specifically, these all raise rather than degrade:

| Condition | Would have silently... | Now |
|---|---|---|
| GL lands on llvmpipe / iGPU | rendered 5.6× slower | `RuntimeError` |
| `h264_nvenc` unavailable | encoded on libx264 | `RuntimeError` |
| Generated clip missing | rendered the theme still | `FileNotFoundError` |
| Shotlist has a gap | emitted a short video | rejected at construction |
| Reactive binding names a bad curve/target | done nothing | `KeyError` |
| Unknown effect name | been ignored | `ValueError` |
| Shader uniform unused | been a dead parameter | `KeyError` |
| Segmentation fails | invented uniform sections | `ValueError` |
| Audio feature is constant | zeroed the curve | `ValueError` |

Two things that look like exceptions to this but are not: the job runner
catches exceptions only to stop them vanishing into a detached asyncio task —
the job still fails and the full traceback reaches the client; and the progress
queue drops its *oldest* update under backpressure, which loses no error state.

`/api/health` asserts rather than reports: it 500s if the environment cannot
render, so green means green. Stage A has its own `/api/health/generation`.

## Environment notes (WSL2)

Headless GL runs through Mesa's **d3d12** driver over `/dev/dxg`. It defaults to
the *integrated* GPU, so `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` is required to
reach the 4090 — `config.setup_gl_env()` sets this, and
`config.verify_gl_renderer()` refuses to continue if it did not take effect.
Measured with `spikes/bench_gl.py`:

| Backend | 1080p (render+readback) | 4-min render |
|---|---|---|
| d3d12 / RTX 4090 | 473 fps (15.8× realtime) | 15s |
| llvmpipe (rejected) | 84 fps (2.8× realtime) | 85s |

Readback dominates (0.03ms shading vs 2.11ms readback), so shader complexity is
effectively free — there is no reason to specialise the uber-shader.

Override the device requirement only with a reason: `VC_REQUIRE_GL=llvmpipe`.

### cuDNN: pip/system library mixing

Every cuDNN convolution failed with
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` (conv2d as well as conv3d). Cause:
`nvidia-cudnn-cu13==9.20.0.48` — the version torch 2.13 pins — does **not** ship
`libcudnn_engines_tensor_ir`, so the loader picked up the *system*
`/usr/lib/x86_64-linux-gnu/libcudnn_engines_tensor_ir.so.9.23.2` while every
other sublibrary came from pip's 9.20. A 9.23 sublibrary inside a 9.20 runtime.

Fix — align pip's cuDNN to a build that ships the missing sublibrary:

```bash
.venv/bin/pip install -U "nvidia-cudnn-cu13>=9.23"
```

pip will warn that torch 2.13.0 pins 9.20.0.48. That warning is expected and
safe to ignore: cuDNN 9.x is ABI-compatible within the major version, and this
is the only configuration in which convolutions actually run.

Diagnose recurrences by listing what is loaded, not what is installed:

```python
# after a failed conv, grep the process map
[l.split()[-1] for l in open('/proc/self/maps') if 'cudnn' in l]
```

### LTX scheduler / diffusers version skew

The Lightricks checkpoint ships a diffusers 0.32-era scheduler config with
`use_dynamic_shifting=True`, but diffusers 0.39's `LTXConditionPipeline` builds
its own linear-quadratic sigma schedule and calls `retrieve_timesteps()` without
`mu`, so the scheduler raises ``` `mu` must be passed ```.
`LTXGenerator._reconcile_scheduler()` disables dynamic shifting at load time —
it inspects the pipeline source and leaves the config alone if a future
diffusers does pass `mu`, and raises if the override fails to take.

## Layout

```
backend/app/
  audio/analysis.py     mp3 -> beats, sections, per-frame curves
  director/schema.py    the shotlist contract (pydantic)
  director/builder.py   deterministic shotlist builder (Director stand-in)
  assemble/shaders.py   GLSL: uber-shader + compositor
  assemble/render.py    Stage B renderer
  assemble/encode.py    ffmpeg frame sink, clip decoding
  jobs.py               job runner + progress broadcast
  main.py               FastAPI
spikes/                 fixtures + benchmarks + the LTX spike
```
