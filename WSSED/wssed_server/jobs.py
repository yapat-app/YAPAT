"""Job directories and status.json persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from wssed_server.settings import JOBS_DIR


def get_job_dir(job_id: int) -> Path:
    job_dir = JOBS_DIR / str(job_id)
    job_dir.mkdir(exist_ok=True, parents=True)
    return job_dir


def save_status(job_id: int, status_data: Dict[str, Any]) -> None:
    status_file = get_job_dir(job_id) / "status.json"
    with open(status_file, "w") as f:
        json.dump(status_data, f, indent=2)


def load_status(job_id: int) -> Dict[str, Any]:
    status_file = get_job_dir(job_id) / "status.json"
    if not status_file.exists():
        return {"status": "PENDING"}
    with open(status_file) as f:
        return json.load(f)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_status_progress(job_id: int, **updates: Any) -> Dict[str, Any]:
    status_data = load_status(job_id)
    progress = status_data.get("progress") or {}
    progress.update(updates)
    status_data["progress"] = progress
    save_status(job_id, status_data)
    return status_data


def tail_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-max_chars:]
