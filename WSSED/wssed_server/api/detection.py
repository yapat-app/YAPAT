"""Detection endpoint."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from wssed_server import jobs
from wssed_server.schemas import DetectionRequest, DetectionResponse
from wssed_server.settings import logger
from wssed_server.tasks import run_detection_task

router = APIRouter(prefix="/wssed", tags=["detection"])


@router.post("/detect", response_model=DetectionResponse)
async def run_detection(request: DetectionRequest, background_tasks: BackgroundTasks):
    job_id = request.job_id
    logger.info("Received detection request for job %s", job_id)

    try:
        jobs.get_job_dir(job_id)
        background_tasks.add_task(
            run_detection_task,
            job_id,
            request.model_path,
            request.dataset_path,
            request.threshold,
        )
        return DetectionResponse(job_id=job_id, message="Detection started")
    except Exception as exc:
        logger.error("Failed to start detection for job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
