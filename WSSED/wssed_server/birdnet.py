"""BirdNET pipeline helpers (embeddings layout, hyperparameters, paths)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

BIRDNET_AUDIO_SUFFIXES = {
    ".wav", ".WAV", ".flac", ".FLAC", ".mp3", ".MP3", ".aac", ".AAC", ".m4a", ".M4A",
}


def count_audio_files(dataset_root: Path) -> int:
    if not dataset_root.is_dir():
        return 0
    return sum(
        1
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix in BIRDNET_AUDIO_SUFFIXES
    )


def count_embedding_npz(embeddings_root: Path) -> int:
    if not embeddings_root.is_dir():
        return 0
    return sum(1 for path in embeddings_root.rglob("*.npz") if path.is_file())


def embeddings_status(dataset_root: Path, embeddings_root: Path) -> Tuple[bool, str]:
    audio_count = count_audio_files(dataset_root)
    npz_count = count_embedding_npz(embeddings_root)
    if audio_count == 0:
        return False, "no audio files found in dataset"
    if npz_count >= audio_count:
        return True, f"embeddings complete ({npz_count}/{audio_count} files)"
    return False, f"embeddings incomplete ({npz_count}/{audio_count} files)"


def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "best_micro_model_segment.pt"


def embeddings_path_for_dataset(abs_dataset_path: Path) -> Path:
    name = abs_dataset_path.name
    return abs_dataset_path.parent / f"{name}_birdnet_embeddings"


def output_dir_for_dataset(abs_dataset_path: Path, focal_data_dir: Path) -> Path:
    return focal_data_dir / f"outputs_of_{abs_dataset_path.name}"


def hyperparameters_env(hyperparameters: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not hyperparameters:
        return {}
    return {"WSSED_JOB_HYPERPARAMETERS": json.dumps(hyperparameters)}


def training_epochs(hyperparameters: Dict[str, Any]) -> int:
    return int(hyperparameters.get("epochs", 50))
