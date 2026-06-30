"""Training start and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from wssed_server import birdnet, jobs
from wssed_server.config_generator import create_strong_labels_csv, generate_config_file
from wssed_server.schemas import TrainingRequest, TrainingStatusResponse
from wssed_server.settings import logger
from wssed_server.tasks import run_birdnet_pipeline_task, run_training_task

router = APIRouter(prefix="/wssed", tags=["training"])


@router.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    job_id = request.job_id
    logger.info("Received training request for job %s", job_id)

    try:
        job_dir = jobs.get_job_dir(job_id)
        config_path = generate_config_file(
            request.hyperparameters,
            request.dataset_path,
            job_dir,
        )
        if request.feedback_labels:
            create_strong_labels_csv(request.feedback_labels, job_dir)

        model_name_key = str(
            request.hyperparameters.get("model_name", "CDur")
        ).strip().lower()
        default_epochs = (
            birdnet.training_epochs(request.hyperparameters)
            if model_name_key == "birdnet"
            else request.hyperparameters.get("epochs", 100)
        )

        jobs.save_status(job_id, {
            "status": "TRAINING",
            "message": "Training started",
            "progress": {
                "phase": "queued",
                "current_epoch": 0,
                "total_epochs": default_epochs,
                "model_name": request.hyperparameters.get("model_name", "CDur"),
                "dataset_path": request.dataset_path,
                "bag_seconds": request.hyperparameters.get("bag_seconds", "full"),
                "hop_seconds": request.hyperparameters.get("hop_seconds", 1),
                "learning_rate": request.hyperparameters.get("learning_rate", 0.001),
                "threshold": request.hyperparameters.get("threshold", 0.5),
                "config_path": str(config_path),
                "job_dir": str(job_dir),
                "updated_at": jobs.utc_now_iso(),
            },
        })

        if model_name_key == "birdnet":
            background_tasks.add_task(
                run_birdnet_pipeline_task,
                job_id,
                request.dataset_path,
                job_dir,
                request.hyperparameters,
            )
        else:
            background_tasks.add_task(run_training_task, job_id, config_path, job_dir)

        return {
            "job_id": job_id,
            "status": "TRAINING",
            "message": "Training started",
        }

    except Exception as exc:
        logger.error("Failed to start training for job %s: %s", job_id, exc, exc_info=True)
        jobs.save_status(job_id, {"status": "FAILED", "error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/train/{job_id}/status", response_model=TrainingStatusResponse)
async def get_training_status(job_id: int):
    return TrainingStatusResponse(**jobs.load_status(job_id))
