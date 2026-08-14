"""FastAPI backend.

Endpoint layout mirrors the Stage A / Stage B split:

    POST /api/projects           upload mp3 + theme, analyse audio
    GET  /api/projects/{id}      timeline + shotlist
    PUT  /api/projects/{id}/shotlist   replace the shotlist (UI edits)
    POST /api/projects/{id}/render     Stage B  (seconds -- previewable)
    POST /api/projects/{id}/generate   Stage A  (minutes -- cached)
    WS   /api/jobs/{job_id}      live progress
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .audio.analysis import Timeline, analyze
from .director.builder import build
from .director.schema import Shotlist
from .jobs import manager, progress_reporter

log = logging.getLogger(__name__)

app = FastAPI(title="VideoCreator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local desktop tool
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECTS = config.DATA / "projects"
PROJECTS.mkdir(parents=True, exist_ok=True)


def _project_dir(pid: str) -> Path:
    d = PROJECTS / pid
    if not d.exists():
        raise HTTPException(404, f"no such project: {pid}")
    return d


def _load_timeline(pid: str) -> Timeline:
    d = _project_dir(pid)
    data = json.loads((d / "timeline.json").read_text())
    from .audio.analysis import Section

    data["sections"] = [Section(**s) for s in data["sections"]]
    return Timeline(**data)


class RenderRequest(BaseModel):
    preview: bool = True
    start: float = 0.0
    end: float | None = None


class BuildRequest(BaseModel):
    prompt: str = ""
    generative: bool = False


@app.get("/api/health")
async def health() -> dict:
    """Assert the environment can actually render. 500s if it cannot.

    Every check raises rather than reporting a degraded state, so a green
    health check is a real guarantee instead of a summary of what is broken.
    """
    import moderngl

    from .assemble.encode import require_codec

    codec = require_codec()

    config.setup_gl_env()
    ctx = moderngl.create_standalone_context()
    renderer = ctx.info["GL_RENDERER"]
    ctx.release()
    config.verify_gl_renderer(renderer)

    return {"status": "ok", "codec": codec, "gl_renderer": renderer}


@app.get("/api/health/generation")
async def health_generation() -> dict:
    """Separate check for Stage A, which needs torch + CUDA. 500s if absent."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch is installed but CUDA is not available")
    return {
        "status": "ok",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    }


@app.post("/api/projects")
async def create_project(
    audio: UploadFile = File(...),
    theme: UploadFile = File(...),
    prompt: str = Form(""),
) -> dict:
    """Upload audio + theme, analyse, and build a starting shotlist."""
    pid = uuid.uuid4().hex[:12]
    d = PROJECTS / pid
    d.mkdir(parents=True)

    audio_path = d / f"audio{Path(audio.filename or 'a.mp3').suffix}"
    theme_path = d / f"theme{Path(theme.filename or 't.png').suffix}"
    for upload, dest in ((audio, audio_path), (theme, theme_path)):
        with dest.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)

    # Analysis is CPU-bound; keep it off the event loop.
    tl = await asyncio.to_thread(analyze, audio_path, config.FPS)
    tl.save(d / "timeline.json")

    sl = build(tl, theme=str(theme_path), prompt=prompt, generative=False)
    sl.save(d / "shotlist.json")

    return {"id": pid, "timeline": tl.to_dict(), "shotlist": sl.model_dump()}


@app.get("/api/projects")
async def list_projects() -> list[dict]:
    out = []
    for d in sorted(PROJECTS.iterdir(), reverse=True):
        if (d / "timeline.json").exists():
            tl = json.loads((d / "timeline.json").read_text())
            out.append({"id": d.name, "duration": tl["duration"], "tempo": tl["tempo"]})
    return out


@app.get("/api/projects/{pid}")
async def get_project(pid: str) -> dict:
    d = _project_dir(pid)
    return {
        "id": pid,
        "timeline": json.loads((d / "timeline.json").read_text()),
        "shotlist": json.loads((d / "shotlist.json").read_text()),
    }


@app.put("/api/projects/{pid}/shotlist")
async def put_shotlist(pid: str, shotlist: Shotlist) -> dict:
    d = _project_dir(pid)
    shotlist.save(d / "shotlist.json")
    return {"ok": True, "shots": len(shotlist.shots)}


@app.post("/api/projects/{pid}/build")
async def rebuild_shotlist(pid: str, req: BuildRequest) -> dict:
    """Regenerate the shotlist from the analysis (the Director's slot)."""
    d = _project_dir(pid)
    tl = _load_timeline(pid)
    theme = next(d.glob("theme.*"))
    sl = build(tl, theme=str(theme), prompt=req.prompt, generative=req.generative)
    sl.save(d / "shotlist.json")
    return sl.model_dump()


@app.post("/api/projects/{pid}/render")
async def start_render(pid: str, req: RenderRequest) -> dict:
    d = _project_dir(pid)
    tl = _load_timeline(pid)
    sl = Shotlist.load(d / "shotlist.json")
    audio_path = next(d.glob("audio.*"))
    out = d / ("preview.mp4" if req.preview else "final.mp4")

    async def run(job) -> dict:
        from .assemble.render import render

        loop = asyncio.get_running_loop()
        report = progress_reporter(job, loop)
        path = await asyncio.to_thread(
            render, sl, tl, out,
            None, req.preview, audio_path, report, req.start, req.end,
        )
        return {"video": f"/api/projects/{pid}/video/{path.name}"}

    job = manager.submit("render", run)
    return job.snapshot()


@app.get("/api/projects/{pid}/video/{name}")
async def get_video(pid: str, name: str) -> FileResponse:
    path = _project_dir(pid) / name
    if not path.exists():
        raise HTTPException(404, "not rendered yet")
    # no-store: preview.mp4 is overwritten on every render, and a cached copy
    # would show the previous edit.
    return FileResponse(path, media_type="video/mp4",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/projects/{pid}/theme")
async def get_theme(pid: str) -> FileResponse:
    theme = next(_project_dir(pid).glob("theme.*"), None)
    if theme is None:
        raise HTTPException(404, "no theme image")
    return FileResponse(theme)


@app.get("/api/projects/{pid}/audio")
async def get_audio(pid: str) -> FileResponse:
    audio = next(_project_dir(pid).glob("audio.*"), None)
    if audio is None:
        raise HTTPException(404, "no audio")
    return FileResponse(audio)


@app.get("/api/jobs")
async def list_jobs() -> list[dict]:
    return manager.list()


@app.websocket("/api/jobs/{job_id}")
async def job_ws(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    job = manager.get(job_id)
    if job is None:
        await ws.send_json({"error": "no such job"})
        await ws.close()
        return

    q = job.subscribe()
    try:
        while True:
            snap = await q.get()
            await ws.send_json(snap)
            if snap["state"] in ("done", "failed", "cancelled"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        job.unsubscribe(q)


# Serve the frontend from source -- it is dependency-free by design, so there
# is no build step to run and no node_modules to install. Mounted last so the
# /api routes above take precedence.
_frontend = config.ROOT / "frontend"
if (_frontend / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
