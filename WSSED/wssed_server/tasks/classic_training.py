"""Classic WSSED training via main.py."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from wssed_server import artifacts, jobs
from wssed_server.settings import WSSED_ROOT, logger, subprocess_env
from wssed_server.subprocess_runner import EPOCH_LINE_RE, stream_process


def run_training_task(job_id: int, config_path: Path, job_dir: Path) -> None:
    logger.info("Starting classic training task for job %s", job_id)
    training_log_path = job_dir / "training.log"

    try:
        current_status = jobs.load_status(job_id)
        progress = current_status.get("progress") or {}
        progress.update({
            "phase": "training",
            "training_log": str(training_log_path),
            "started_at": jobs.utc_now_iso(),
            "updated_at": jobs.utc_now_iso(),
        })
        jobs.save_status(job_id, {
            **current_status,
            "status": "TRAINING",
            "message": "Training in progress",
            "progress": progress,
        })

        config_dest = WSSED_ROOT / f"config_{job_id}.py"
        shutil.copy(config_path, config_dest)
        train_env = {
            "PYTHONPATH": str(WSSED_ROOT),
            "WSSED_CONFIG": f"config_{job_id}",
        }
        logger.info(
            "Training subprocess CUDA_VISIBLE_DEVICES=%s",
            subprocess_env(train_env).get("CUDA_VISIBLE_DEVICES", "(inherit)"),
        )

        training_started_mtime = datetime.now(timezone.utc).timestamp()

        def on_epoch(epoch: int, line: str) -> None:
            jobs.update_status_progress(
                job_id,
                phase="epoch_completed",
                current_epoch=epoch,
                last_log_line=line,
                updated_at=jobs.utc_now_iso(),
            )

        with training_log_path.open("w", buffering=1) as log_file:
            returncode = stream_process(
                [sys.executable, "main.py"],
                cwd=WSSED_ROOT,
                log_file=log_file,
                env_extra=train_env,
                epoch_pattern=EPOCH_LINE_RE,
                on_epoch=on_epoch,
            )

        config_dest.unlink(missing_ok=True)

        if returncode == 0:
            model_paths = artifacts.collect_model_paths(
                job_dir, started_after=training_started_mtime
            )
            model_path = model_paths.get("preferred")
            final_progress = jobs.load_status(job_id).get("progress") or {}
            final_progress.update({
                "phase": "completed",
                "current_epoch": final_progress.get("total_epochs"),
                "completed_at": jobs.utc_now_iso(),
                "updated_at": jobs.utc_now_iso(),
                "training_log": str(training_log_path),
            })
            jobs.save_status(job_id, {
                "status": "COMPLETED",
                "model_path": model_path,
                "model_paths": model_paths,
                "metrics": {
                    "training_completed": True,
                    "model_saved": model_path is not None,
                    "registered_artifact": (
                        "best_micro_model_segment"
                        if model_paths.get("best_micro_model_segment")
                        else None
                    ),
                },
                "message": "Training completed successfully",
                "progress": final_progress,
            })
            logger.info("Training completed for job %s", job_id)
            return

        failure_progress = jobs.load_status(job_id).get("progress") or {}
        failure_progress.update({
            "phase": "failed",
            "failed_at": jobs.utc_now_iso(),
            "updated_at": jobs.utc_now_iso(),
            "training_log": str(training_log_path),
        })
        jobs.save_status(job_id, {
            "status": "FAILED",
            "error": jobs.tail_text(training_log_path),
            "message": "Training failed",
            "progress": failure_progress,
        })
        logger.error("Training failed for job %s", job_id)

    except Exception as exc:
        jobs.save_status(job_id, {
            "status": "FAILED",
            "error": str(exc),
            "message": f"Training error: {exc}",
        })
        logger.error("Training error for job %s: %s", job_id, exc, exc_info=True)
