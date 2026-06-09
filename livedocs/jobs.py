"""Background build job runner.

start_build(corpus, resume) -> job dict
get_status()               -> current job dict | None
log_stream()               -> generator yielding log lines (for SSE)
"""
import queue
import threading
import time
import uuid
from collections import deque

_lock = threading.Lock()
_current_job: dict | None = None
_log_queue: queue.Queue | None = None
_log_history: deque = deque(maxlen=2000)


def start_build(corpus: str = "both", resume: bool = False) -> dict:
    global _current_job, _log_queue, _log_history
    with _lock:
        if _current_job and _current_job.get("status") == "running":
            raise RuntimeError("Build already in progress")
        job_id = str(uuid.uuid4())[:8]
        _log_queue = queue.Queue()
        _log_history = deque(maxlen=2000)
        _current_job = {
            "id": job_id,
            "status": "running",
            "phase": "starting",
            "corpus": corpus,
            "started_at": time.time(),
            "finished_at": None,
            "error": None,
            "counts": {},
        }

    thread = threading.Thread(
        target=_run, args=(job_id, corpus, resume), daemon=True
    )
    thread.start()
    return dict(_current_job)


def get_status() -> dict | None:
    return dict(_current_job) if _current_job else None


def get_log_history() -> list[str]:
    return list(_log_history)


def log_stream():
    """Generator yielding log line strings. Ends when job finishes."""
    global _log_queue
    if _log_queue is None:
        return
    q = _log_queue
    while True:
        try:
            item = q.get(timeout=1.0)
            if item is None:
                break
            yield item
        except queue.Empty:
            job = _current_job
            if job and job.get("status") != "running":
                break
            yield ""  # keepalive heartbeat


def _emit(line: str) -> None:
    print(line)
    if _log_queue:
        _log_queue.put(line)
    _log_history.append(line)


def _set_phase(phase: str) -> None:
    global _current_job
    if _current_job:
        _current_job["phase"] = phase


def _run(job_id: str, corpus: str, resume: bool) -> None:
    global _current_job
    try:
        from livedocs.sources import clone_source
        from livedocs.settings import get_settings, save_settings
        from livedocs.build import run_build_code, run_build_docs

        s = get_settings()

        # 1. Clone / pull all GitHub sources
        github_sources = [src for src in s.sources if src["kind"] == "github"]
        if github_sources:
            _set_phase("cloning")
            _emit(f"\n=== Cloning {len(github_sources)} GitHub source(s) ===")
            for src in github_sources:
                clone_source(src, log=_emit)

        counts = {}

        # 2. Build code index
        if corpus in ("both", "code"):
            _set_phase("indexing_code")
            _emit("\n=== Building CODE index ===")
            try:
                c = run_build_code(log=_emit, resume=resume)
                counts["code"] = c
            except Exception as e:
                _emit(f"\nERROR in code build: {type(e).__name__}: {e}")
                raise

        # 3. Build docs index
        if corpus in ("both", "docs"):
            _set_phase("indexing_docs")
            _emit("\n=== Building DOCS index ===")
            try:
                c = run_build_docs(log=_emit, resume=resume)
                counts["docs"] = c
            except Exception as e:
                _emit(f"\nERROR in docs build: {type(e).__name__}: {e}")
                raise

        # 4. Success
        _set_phase("done")
        _emit("\n=== Build complete. ===")

        with _lock:
            _current_job["status"] = "done"
            _current_job["finished_at"] = time.time()
            _current_job["counts"] = counts

        save_settings({
            "deployed": True,
            "last_build": {
                "id": job_id,
                "status": "done",
                "corpus": corpus,
                "counts": counts,
                "finished_at": _current_job["finished_at"],
            },
        })

        # Invalidate pipeline cache so next query uses fresh index
        try:
            from livedocs.query import code_pipeline
            code_pipeline.reset()
        except Exception:
            pass

    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        _emit(f"\nBUILD FAILED: {err_msg}")
        with _lock:
            _current_job["status"] = "error"
            _current_job["error"] = err_msg
            _current_job["finished_at"] = time.time()
        from livedocs.settings import save_settings
        save_settings({
            "last_build": {
                "id": job_id,
                "status": "error",
                "error": err_msg,
                "finished_at": _current_job["finished_at"],
            },
        })
    finally:
        if _log_queue:
            _log_queue.put(None)  # sentinel
