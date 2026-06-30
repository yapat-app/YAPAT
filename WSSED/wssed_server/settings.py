"""Environment, paths, and logging for the WSSED GPU server."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

WSSED_ROOT = _REPO_ROOT
JOBS_DIR = Path(os.environ.get("WSSED_JOBS_DIR", "/wssed_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(os.environ.get("WSSED_DATA_ROOT", "/ds-iml/Bioacoustics"))


def subprocess_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build env for training/detection child processes."""
    env: Dict[str, str] = {**os.environ}
    if extra:
        env.update(extra)
    pinned = os.environ.get("WSSED_CUDA_VISIBLE_DEVICES")
    if pinned is not None and str(pinned).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(pinned).strip()
    return env


def resolve_dataset_path(dataset_path: str) -> Path:
    if not dataset_path.startswith("/"):
        return (DATA_ROOT / dataset_path).resolve()
    return Path(dataset_path).resolve()
