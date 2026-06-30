"""Two-step BirdNET pipeline: extract embeddings → train MIL model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from wssed_server import artifacts, birdnet, jobs
from wssed_server.settings import WSSED_ROOT, logger, resolve_dataset_path
from wssed_server.subprocess_runner import EPOCH_LINE_RE, stream_process


def _mark_completed(
    job_id: int,
    *,
    training_log_path: Path,
    output_dir: Path,
    model_paths: Dict[str, str],
    model_path: Optional[str],
    message: str,
    metrics: Dict[str, Any],
    progress_extra: Optional[Dict[str, Any]] = None,
) -> None:
    final_progress = jobs.load_status(job_id).get("progress") or {}
    final_progress.update({
        "phase": "completed",
        "completed_at": jobs.utc_now_iso(),
        "updated_at": jobs.utc_now_iso(),
        "training_log": str(training_log_path),
        "output_dir": str(output_dir),
    })
    if progress_extra:
        final_progress.update(progress_extra)
    jobs.save_status(job_id, {
        "status": "COMPLETED",
        "model_path": model_path,
        "model_paths": model_paths,
        "metrics": metrics,
        "message": message,
        "progress": final_progress,
    })


def run_birdnet_pipeline_task(
    job_id: int,
    dataset_path: str,
    job_dir: Path,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> None:
    logger.info("Starting BirdNET pipeline for job %s", job_id)
    training_log_path = job_dir / "training.log"
    hyperparameters = hyperparameters or {}
    force_reextract = bool(hyperparameters.get("force_reextract", False))
    force_retrain = bool(hyperparameters.get("force_retrain", False))

    try:
        abs_dataset_path = resolve_dataset_path(dataset_path)
        embeddings_path = birdnet.embeddings_path_for_dataset(abs_dataset_path)
        focal_data_dir = WSSED_ROOT / "focal-data"
        output_dir = birdnet.output_dir_for_dataset(abs_dataset_path, focal_data_dir)
        checkpoint_path = birdnet.checkpoint_path(output_dir)

        embeddings_complete, embeddings_reason = birdnet.embeddings_status(
            abs_dataset_path, embeddings_path
        )
        skip_extraction = embeddings_complete and not force_reextract
        skip_training = checkpoint_path.exists() and not force_retrain
        total_epochs = birdnet.training_epochs(hyperparameters)
        hp_env = birdnet.hyperparameters_env(hyperparameters)
        training_started_mtime = datetime.now(timezone.utc).timestamp()

        jobs.save_status(job_id, {
            "status": "TRAINING",
            "message": "BirdNET pipeline started",
            "progress": {
                "phase": "queued",
                "dataset_path": str(abs_dataset_path),
                "embeddings_path": str(embeddings_path),
                "output_dir": str(output_dir),
                "training_log": str(training_log_path),
                "embeddings_complete": embeddings_complete,
                "embeddings_status": embeddings_reason,
                "skip_extraction": skip_extraction,
                "skip_training": skip_training,
                "force_reextract": force_reextract,
                "force_retrain": force_retrain,
                "total_epochs": total_epochs,
                "learning_rate": hyperparameters.get("learning_rate"),
                "model_name": hyperparameters.get("model_name", "BirdNET"),
                "started_at": jobs.utc_now_iso(),
                "updated_at": jobs.utc_now_iso(),
            },
        })

        extract_rc = 0
        with training_log_path.open("w", buffering=1) as log_file:
            _write_pipeline_header(
                log_file,
                job_id=job_id,
                abs_dataset_path=abs_dataset_path,
                embeddings_path=embeddings_path,
                output_dir=output_dir,
                embeddings_reason=embeddings_reason,
                skip_extraction=skip_extraction,
                skip_training=skip_training,
                hyperparameters=hyperparameters,
            )

            if skip_extraction:
                log_file.write(
                    f"[Step 1/2] SKIPPED — {embeddings_reason}\n"
                    f"  (set hyperparameters.force_reextract=true to re-run)\n"
                    f"{'=' * 60}\n"
                )
                log_file.flush()
                jobs.update_status_progress(
                    job_id,
                    phase="skipped_extraction",
                    embeddings_status=embeddings_reason,
                    updated_at=jobs.utc_now_iso(),
                )
            else:
                jobs.update_status_progress(
                    job_id, phase="extracting_embeddings", updated_at=jobs.utc_now_iso()
                )
                log_file.write(
                    f"[Step 1/2] Extracting BirdNET embeddings\n"
                    f"  dataset  : {abs_dataset_path}\n"
                    f"  output   : {embeddings_path}\n"
                    f"  note     : existing .npz files are skipped by the extractor\n"
                    f"{'=' * 60}\n"
                )
                log_file.flush()
                extract_rc = stream_process(
                    [sys.executable, "extract_birdnet_embeddings.py"],
                    cwd=focal_data_dir,
                    log_file=log_file,
                    env_extra={
                        "WSSED_EXTRACT_DATASET_PATH": str(abs_dataset_path),
                        **hp_env,
                    },
                )

        if extract_rc != 0:
            jobs.save_status(job_id, {
                "status": "FAILED",
                "error": jobs.tail_text(training_log_path),
                "message": "BirdNET embedding extraction failed",
                "progress": {
                    "phase": "failed_extraction",
                    "training_log": str(training_log_path),
                    "failed_at": jobs.utc_now_iso(),
                },
            })
            logger.error("Embedding extraction failed for job %s", job_id)
            return

        if skip_training:
            with training_log_path.open("a", buffering=1) as log_file:
                log_file.write(
                    f"\n[Step 2/2] SKIPPED — checkpoint already exists\n"
                    f"  checkpoint : {checkpoint_path}\n"
                    f"  (set hyperparameters.force_retrain=true to re-run)\n"
                    f"{'=' * 60}\n"
                )
            model_paths = artifacts.collect_model_paths(job_dir, extra_roots=[output_dir])
            model_path = model_paths.get("preferred") or str(checkpoint_path)
            _mark_completed(
                job_id,
                training_log_path=training_log_path,
                output_dir=output_dir,
                model_paths=model_paths,
                model_path=model_path,
                message="Reused existing BirdNET checkpoint (training skipped)",
                metrics={
                    "training_completed": True,
                    "model_saved": True,
                    "reused_existing_checkpoint": True,
                },
                progress_extra={"reused_checkpoint": True},
            )
            logger.info("BirdNET job %s reused checkpoint: %s", job_id, model_path)
            return

        jobs.update_status_progress(
            job_id,
            phase="training_from_embeddings",
            embeddings_path=str(embeddings_path),
            total_epochs=total_epochs,
            updated_at=jobs.utc_now_iso(),
        )

        def on_epoch(epoch: int, line: str) -> None:
            jobs.update_status_progress(
                job_id,
                phase="epoch_completed",
                current_epoch=epoch,
                last_log_line=line,
                updated_at=jobs.utc_now_iso(),
            )

        with training_log_path.open("a", buffering=1) as log_file:
            log_file.write(
                f"\n[Step 2/2] Training from pre-extracted embeddings\n"
                f"  embeddings : {embeddings_path}\n"
                f"  output     : {output_dir}\n"
                f"{'=' * 60}\n"
            )
            log_file.flush()
            train_rc = stream_process(
                [sys.executable, "train_from_birdnet_embeddings.py"],
                cwd=focal_data_dir,
                log_file=log_file,
                env_extra={
                    "WSSED_TRAIN_EMBEDDINGS_PATH": str(embeddings_path),
                    **hp_env,
                },
                epoch_pattern=EPOCH_LINE_RE,
                on_epoch=on_epoch,
            )

        if train_rc == 0:
            model_paths = artifacts.collect_model_paths(
                job_dir,
                started_after=training_started_mtime,
                extra_roots=[output_dir],
            )
            model_path = model_paths.get("preferred")
            _mark_completed(
                job_id,
                training_log_path=training_log_path,
                output_dir=output_dir,
                model_paths=model_paths,
                model_path=model_path,
                message="Training completed successfully",
                metrics={
                    "training_completed": True,
                    "model_saved": model_path is not None,
                },
            )
            logger.info("BirdNET pipeline completed for job %s, model: %s", job_id, model_path)
            return

        jobs.save_status(job_id, {
            "status": "FAILED",
            "error": jobs.tail_text(training_log_path),
            "message": "BirdNET training from embeddings failed",
            "progress": {
                "phase": "failed_training",
                "training_log": str(training_log_path),
                "failed_at": jobs.utc_now_iso(),
            },
        })
        logger.error("BirdNET training failed for job %s", job_id)

    except Exception as exc:
        jobs.save_status(job_id, {
            "status": "FAILED",
            "error": str(exc),
            "message": f"BirdNET pipeline error: {exc}",
        })
        logger.error("BirdNET pipeline error for job %s: %s", job_id, exc, exc_info=True)


def _write_pipeline_header(
    log_file,
    *,
    job_id: int,
    abs_dataset_path: Path,
    embeddings_path: Path,
    output_dir: Path,
    embeddings_reason: str,
    skip_extraction: bool,
    skip_training: bool,
    hyperparameters: Dict[str, Any],
) -> None:
    log_file.write(
        f"BirdNET pipeline job {job_id}\n"
        f"  dataset           : {abs_dataset_path}\n"
        f"  embeddings        : {embeddings_path}\n"
        f"  training output   : {output_dir}\n"
        f"  embeddings status : {embeddings_reason}\n"
        f"  skip extraction   : {skip_extraction}\n"
        f"  skip training     : {skip_training}\n"
        f"  hyperparameters   : {json.dumps(hyperparameters, indent=2)}\n"
        f"{'=' * 60}\n"
    )
    log_file.flush()
