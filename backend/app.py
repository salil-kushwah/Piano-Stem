import os
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import BASE_DIR, UPLOADS_DIR, STEMS_DIR, MIDI_DIR
from backend.pipeline import ProcessingPipeline

app = FastAPI(title="Piano Stem Studio", description="Stem Isolation + Transcription + Rousseau Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task state tracking
tasks_state: Dict[str, Dict[str, Any]] = {}
active_connections: Dict[str, list[WebSocket]] = {}

FRONTEND_DIR = BASE_DIR / "frontend"


async def broadcast_progress(task_id: str, progress: int, message: str, result: Dict[str, Any] = None):
    state = tasks_state.get(task_id, {})
    state["progress"] = progress
    state["message"] = message
    if progress < 0 or state.get("status") == "failed":
        state["status"] = "failed"
    elif result or progress == 100:
        state["result"] = result
        state["status"] = "completed"
    else:
        state["status"] = "processing"
    tasks_state[task_id] = state

    payload = {
        "task_id": task_id,
        "progress": progress,
        "message": message,
        "status": state["status"],
        "result": result
    }

    if task_id in active_connections:
        for ws in active_connections[task_id]:
            try:
                await ws.send_json(payload)
            except Exception:
                pass


def run_pipeline_task(task_id: str, file_path: Path, is_mixed_audio: bool, precision_mode: str, stem_mode: str, loop: asyncio.AbstractEventLoop):
    def progress_callback(progress: int, message: str):
        asyncio.run_coroutine_threadsafe(
            broadcast_progress(task_id, progress, message),
            loop
        )

    pipeline = ProcessingPipeline(task_id=task_id, progress_callback=progress_callback)
    try:
        result = pipeline.run(
            input_audio_path=file_path,
            is_mixed_audio=is_mixed_audio,
            precision_mode=precision_mode,
            stem_mode=stem_mode
        )
        asyncio.run_coroutine_threadsafe(
            broadcast_progress(task_id, 100, "Processing complete!", result),
            loop
        )
    except Exception as e:
        print(f"Pipeline error for task {task_id}: {e}")
        state = tasks_state.get(task_id, {})
        state["status"] = "failed"
        state["error"] = str(e)
        tasks_state[task_id] = state
        asyncio.run_coroutine_threadsafe(
            broadcast_progress(task_id, -1, f"Error: {e}"),
            loop
        )


@app.post("/api/process")
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    is_mixed: bool = Form(True),
    precision_mode: str = Form("clean"),
    stem_mode: str = Form("auto")
):
    task_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp3"
    input_path = UPLOADS_DIR / f"{task_id}_original{ext}"

    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    tasks_state[task_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Upload received. Initializing pipeline...",
        "filename": file.filename,
        "original_audio_file": input_path.name,
        "precision_mode": precision_mode,
        "stem_mode": stem_mode
    }

    loop = asyncio.get_running_loop()
    background_tasks.add_task(run_pipeline_task, task_id, input_path, is_mixed, precision_mode, stem_mode, loop)

    return {"task_id": task_id, "original_audio_file": input_path.name}


@app.get("/api/notes/{task_id}")
async def get_task_notes(task_id: str):
    """Returns canonical notes with full acoustic evidence and provenance for inspection."""
    if task_id not in tasks_state:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks_state[task_id]
    result = task.get("result", {})
    vis_data = result.get("visualizer_data", {})
    notes = vis_data.get("notes", [])
    stats = result.get("stats", {})
    return {
        "task_id": task_id,
        "notes_count": len(notes),
        "notes": notes,
        "sieve_stats": stats.get("spectral_sieve", {}),
        "voicing_pruned": stats.get("voicing_pruned", 0)
    }


@app.post("/api/notes/{task_id}/delete")
async def delete_task_note(task_id: str, note_id: str = Form(...)):
    """User correction: deletes a false note by ID from the studio session."""
    if task_id not in tasks_state:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks_state[task_id]
    result = task.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="Task not yet finished")

    vis_data = result.get("visualizer_data", {})
    notes = vis_data.get("notes", [])
    before_count = len(notes)
    vis_data["notes"] = [n for n in notes if n.get("id") != note_id]
    after_count = len(vis_data["notes"])

    return {
        "task_id": task_id,
        "deleted": before_count > after_count,
        "remaining_notes": after_count
    }


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_state:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_state[task_id]


@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    if task_id not in active_connections:
        active_connections[task_id] = []
    active_connections[task_id].append(websocket)

    # Immediately send current state if task already exists
    if task_id in tasks_state:
        await websocket.send_json(tasks_state[task_id])

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if task_id in active_connections and websocket in active_connections[task_id]:
            active_connections[task_id].remove(websocket)


@app.get("/api/audio/{filename:path}")
async def get_audio_file(filename: str):
    name = Path(filename).name
    for folder in [STEMS_DIR, UPLOADS_DIR, MIDI_DIR]:
        path = folder / name
        if path.exists():
            return FileResponse(path, media_type="audio/mpeg" if path.suffix == ".mp3" else "audio/wav")
    direct = Path(filename)
    if direct.exists() and direct.is_file():
        return FileResponse(direct, media_type="audio/mpeg" if direct.suffix == ".mp3" else "audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")


@app.get("/api/midi/{filename:path}")
async def get_midi_file(filename: str):
    name = Path(filename).name
    for folder in [MIDI_DIR, STEMS_DIR]:
        path = folder / name
        if path.exists():
            return FileResponse(path, filename=name, media_type="audio/midi")
    direct = Path(filename)
    if direct.exists() and direct.is_file():
        return FileResponse(direct, filename=direct.name, media_type="audio/midi")
    raise HTTPException(status_code=404, detail="MIDI file not found")


@app.get("/api/musicxml/{filename:path}")
async def get_musicxml_file(filename: str):
    name = Path(filename).name
    for folder in [MIDI_DIR, STEMS_DIR]:
        path = folder / name
        if path.exists():
            return FileResponse(path, filename=name, media_type="application/vnd.recordare.musicxml+xml")
    direct = Path(filename)
    if direct.exists() and direct.is_file():
        return FileResponse(direct, filename=direct.name, media_type="application/vnd.recordare.musicxml+xml")
    raise HTTPException(status_code=404, detail="MusicXML file not found")


# Mount static assets for frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
