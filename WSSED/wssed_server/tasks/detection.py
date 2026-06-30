"""Detection background task (placeholder)."""

from __future__ import annotations

from wssed_server.settings import logger


def run_detection_task(
    job_id: int,
    model_path: str,
    dataset_path: str,
    threshold: float,
) -> None:
    logger.info("Starting detection task for job %s", job_id)
    try:
        # TODO: load model, run inference, return predictions to YAPAT
        predictions = []
        logger.info(
            "Detection completed for job %s, predictions: %s",
            job_id,
            len(predictions),
        )
    except Exception as exc:
        logger.error("Detection error for job %s: %s", job_id, exc, exc_info=True)
