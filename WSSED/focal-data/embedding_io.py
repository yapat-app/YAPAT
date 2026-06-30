"""Embedding IO helpers for BirdNET and legacy precomputed text formats."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

import config


def embedding_cache_path(audio_path: str, cache_dir: str) -> Path:
    source = Path(audio_path)
    safe_name = source.stem + ".birdnet_segments.npz"
    return Path(cache_dir) / safe_name


def save_segment_embeddings(npz_path: Path, embeddings: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, embeddings=embeddings.astype(np.float32), starts=starts.astype(np.float32), ends=ends.astype(np.float32))


def load_segment_embeddings(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    starts = data["starts"]
    ends = data["ends"]
    return embeddings, starts, ends


def load_precomputed_txt_embeddings(txt_path: Path, hop_seconds: float, window_seconds: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with txt_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.replace(",", " ").split()]
            rows.append(vals)

    if not rows:
        raise ValueError(f"No numeric rows found in embedding text file: {txt_path}")

    matrix = np.asarray(rows, dtype=np.float32)
    target_dim = int(getattr(config, "EXPECTED_EMBED_DIM", 1024))
    if matrix.shape[1] < target_dim:
        raise ValueError(
            f"Embedding text file has fewer than {target_dim} features: {txt_path} shape={matrix.shape}"
        )
    if matrix.shape[1] > target_dim:
        matrix = matrix[:, -target_dim:]

    count = matrix.shape[0]
    starts = np.arange(count, dtype=np.float32) * float(hop_seconds)
    ends = starts + float(window_seconds)
    return matrix, starts, ends
