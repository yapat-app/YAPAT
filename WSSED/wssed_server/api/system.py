"""Health and root endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter

from wssed_server.settings import DATA_ROOT, JOBS_DIR, WSSED_ROOT

router = APIRouter(tags=["system"])


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "wssed_root": str(WSSED_ROOT),
        "jobs_dir": str(JOBS_DIR),
        "data_root": str(DATA_ROOT),
        "jobs_count": len(list(JOBS_DIR.iterdir())),
        "training_cuda_visible_devices": (
            os.environ.get("WSSED_CUDA_VISIBLE_DEVICES") or None
        ),
    }


@router.get("/")
def root():
    return {
        "message": "WSSED GPU Server",
        "version": "1.0.0",
        "endpoints": {
            "training": "/wssed/train",
            "status": "/wssed/train/{job_id}/status",
            "detection": "/wssed/detect",
            "artifacts": "/wssed/artifacts?dataset_path=...",
            "health": "/health",
        },
    }
