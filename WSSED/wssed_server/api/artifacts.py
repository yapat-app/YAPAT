"""Dataset artifact discovery (embeddings, checkpoints)."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from wssed_server import birdnet
from wssed_server.settings import WSSED_ROOT, resolve_dataset_path

router = APIRouter(prefix="/wssed", tags=["artifacts"])


@router.get("/artifacts")
def get_dataset_artifacts(
    dataset_path: str = Query(..., description="Relative or absolute dataset path"),
) -> Dict[str, Any]:
    """Report whether BirdNET embeddings and a trained checkpoint already exist."""
    try:
        abs_dataset_path = resolve_dataset_path(dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not abs_dataset_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset directory not found: {abs_dataset_path}",
        )

    embeddings_path = birdnet.embeddings_path_for_dataset(abs_dataset_path)
    focal_data_dir = WSSED_ROOT / "focal-data"
    output_dir = birdnet.output_dir_for_dataset(abs_dataset_path, focal_data_dir)
    checkpoint_path = birdnet.checkpoint_path(output_dir)
    embeddings_complete, embeddings_status = birdnet.embeddings_status(
        abs_dataset_path, embeddings_path
    )
    checkpoint_exists = checkpoint_path.is_file()

    return {
        "dataset_path": str(abs_dataset_path),
        "embeddings_path": str(embeddings_path),
        "embeddings_complete": embeddings_complete,
        "embeddings_status": embeddings_status,
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_path": str(checkpoint_path) if checkpoint_exists else None,
        "output_dir": str(output_dir),
        "audio_count": birdnet.count_audio_files(abs_dataset_path),
        "npz_count": birdnet.count_embedding_npz(embeddings_path),
    }
