"""
server.py  –  Flask backend for Agentic Document Extraction UI
--------------------------------------------------------------
Endpoints
  POST /api/extract         – upload file, run full pipeline, return JSON
  GET  /api/stream/<job_id> – SSE stream of live stage updates
  GET  /                    – serve the UI

Run:
  python server.py
  open http://localhost:5000
"""

import json
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Make sure the extraction agent is importable from the same directory ──────
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "docextract_uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
MAX_UPLOAD_MB = 20

# In-memory job store  {job_id: {"queue": Queue, "result": dict|None, "error": str|None}}
_jobs: dict = {}


#  Helper to broadcast stage updates into a job's SSE queue

def _make_event(event_type: str, payload: dict) -> str:
    """Format a Server-Sent Event string."""
    data = json.dumps({"type": event_type, **payload})
    return f"data: {data}\n\n"


#  Background extraction thread

def _run_extraction(job_id: str, file_path: str, doc_type_hint: str):
    """
    Runs inside a daemon thread.
    Patches the agent's logging so each node fires an SSE event,
    then stores the final result in _jobs[job_id].
    """
    q = _jobs[job_id]["queue"]

    def emit(event_type: str, **kwargs):
        q.put(_make_event(event_type, kwargs))

    try:
        import document_extraction_agent as agent

        # Monkey-patch _log so the UI gets live node updates 
        original_log = agent._log

        stage_map = {
            "PREPROCESS": "preprocess",
            "LAYOUT":     "layout",
            "CLASSIFY":   "classify",
            "EXTRACT":    "extract",
            "RELATIONS":  "relationships",
            "RECONSTRUCT":"reconstruct",
        }
        stage_start: dict = {}

        def patched_log(state, agent_name: str, message: str):
            original_log(state, agent_name, message)
            stage_id = stage_map.get(agent_name.upper())
            if not stage_id:
                return
            if "Ingesting" in message or "Analysing" in message or \
               "Assigning" in message or "Extracting" in message or \
               "Mining" in message or "Building" in message:
                stage_start[stage_id] = time.time()
                emit("stage_start", stage=stage_id, message=message)
            elif "Done" in message or "Identified" in message or \
                 "Classified" in message or "Populated" in message or \
                 "Found" in message or "assembled" in message:
                elapsed = round(time.time() - stage_start.get(stage_id, time.time()), 2)
                emit("stage_done", stage=stage_id, message=message, elapsed=elapsed)

        agent._log = patched_log

        # Run the pipeline
        emit("pipeline_start", message="Pipeline started")
        result = agent.extract_document(
            file_path=file_path,
            doc_type_hint=doc_type_hint or "invoice",
            output_path=str(UPLOAD_FOLDER / f"{job_id}_output.json"),
        )
        agent._log = original_log   # restore

        _jobs[job_id]["result"] = result
        emit("pipeline_done", result=result)

    except Exception as exc:
        tb = traceback.format_exc()
        _jobs[job_id]["error"] = str(exc)
        q.put(_make_event("pipeline_error", message=str(exc), traceback=tb))
    finally:
        q.put(None)   # sentinel – tells the SSE generator to close


#  Routes

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/extract", methods=["POST"])
def start_extraction():
    """
    Accept a multipart upload, save the file, spin up a background thread,
    and return a job_id immediately.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    # Size guard
    f.seek(0, 2)
    size_mb = f.tell() / (1024 * 1024)
    f.seek(0)
    if size_mb > MAX_UPLOAD_MB:
        return jsonify({"error": f"File too large ({size_mb:.1f} MB, max {MAX_UPLOAD_MB} MB)"}), 400

    job_id = str(uuid.uuid4())
    save_path = UPLOAD_FOLDER / f"{job_id}{ext}"
    f.save(str(save_path))

    doc_type_hint = request.form.get("doc_type", "invoice")

    _jobs[job_id] = {
        "queue":  queue.Queue(),
        "result": None,
        "error":  None,
        "file":   str(save_path),
    }

    thread = threading.Thread(
        target=_run_extraction,
        args=(job_id, str(save_path), doc_type_hint),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream(job_id: str):
    """
    Server-Sent Events endpoint.  The UI connects here after receiving a job_id.
    Each pipeline stage fires events; the final event carries the full result.
    """
    if job_id not in _jobs:
        return jsonify({"error": "Unknown job"}), 404

    def generate():
        q = _jobs[job_id]["queue"]
        while True:
            try:
                msg = q.get(timeout=120)   # 2-minute hard cap
            except queue.Empty:
                yield _make_event("pipeline_error", message="Extraction timed out")
                break
            if msg is None:
                break
            yield msg
        # Clean up after a delay to let the client collect the last event
        threading.Timer(30, lambda: _jobs.pop(job_id, None)).start()

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/result/<job_id>")
def get_result(job_id: str):
    """Fallback poll endpoint in case SSE isn't supported."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    if job["error"]:
        return jsonify({"error": job["error"]}), 500
    if job["result"] is None:
        return jsonify({"status": "running"}), 202
    return jsonify(job["result"])


# 

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  DocExtract UI  –  Flask server")
    print("  http://localhost:5000")
    print("═" * 60 + "\n")
    app.run(debug=False, port=5000, threaded=True)