"""Train WSSED model directly from pre-extracted BirdNET .npz embeddings.

Usage:
    python train_from_birdnet_embeddings.py /path/to/embeddings
    # or: WSSED_TRAIN_EMBEDDINGS_PATH=/path/to/embeddings python train_from_birdnet_embeddings.py

Expected input structure:

    Serra_birdnet_embeddings
    ├── class_1
    │   ├── audio_001.npz
    │   └── audio_002.npz
    ├── class_2
    │   ├── audio_003.npz
    │   └── audio_004.npz
    └── class_3
        ├── audio_005.npz
        └── audio_006.npz

Important:
    - The first-level subfolders are treated as class/species names.
    - Class names are sorted alphabetically before assigning label indices.
    - .npz files must be directly inside each class folder.
    - Nested subfolders inside class folders are not used and will raise an error.
    - No BirdNET extraction happens in this script.
"""

from __future__ import annotations

import contextlib
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, TextIO

import numpy as np
import pandas as pd
import torch

import config
import main as core
from job_hyperparameters import apply_training_hyperparameters, load_job_hyperparameters


def _resolve_embeddings_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    path = os.environ.get("WSSED_TRAIN_EMBEDDINGS_PATH")
    if path:
        return path
    raise SystemExit(
        "Embeddings path required.\n"
        "  python train_from_birdnet_embeddings.py /path/to/embeddings\n"
        "  WSSED_TRAIN_EMBEDDINGS_PATH=/path/to/embeddings python train_from_birdnet_embeddings.py"
    )


EMBEDDINGS_PATH = _resolve_embeddings_path()


@dataclass
class EmbeddingRow:
    embedding_path: Path
    label_code: str
    species_name: str


class Tee:
    """Write printed output to both console and a log file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def pushd(path: Path):
    """Temporarily change working directory.

    main.py saves checkpoints and epoch artifacts relative to the current
    working directory, so we run training inside the output folder.
    """
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(previous)


def dataset_name_from_embeddings_path(embeddings_root: Path) -> str:
    """Infer dataset name from embedding folder name.

    Example:
        Serra_birdnet_embeddings -> Serra
    """
    name = embeddings_root.name
    suffix = "_birdnet_embeddings"

    if name.endswith(suffix):
        return name[: -len(suffix)]

    return name


def output_root_from_embeddings_path(embeddings_root: Path) -> Path:
    """Create output folder path next to this Python file.

    Example:
        dataset name: Serra
        -> <this_file_folder>/outputs_of_Serra
    """
    dataset_name = dataset_name_from_embeddings_path(embeddings_root)
    code_dir = Path(__file__).resolve().parent
    return code_dir / f"outputs_of_{dataset_name}"


def safe_code(name: str) -> str:
    """Normalize class name for label usage.

    Folder names are preserved except path separators are made safe.
    """
    return name.strip().replace("/", "_").replace("\\", "_")


def find_class_dirs(embeddings_root: Path) -> List[Path]:
    """Return first-level class/species folders sorted alphabetically."""
    class_dirs = [p for p in embeddings_root.iterdir() if p.is_dir()]
    return sorted(class_dirs, key=lambda p: p.name.lower())


def validate_no_nested_subfolders(class_dirs: List[Path]) -> None:
    """Training expects .npz files directly inside each class folder."""
    nested_dirs: List[Path] = []

    for class_dir in class_dirs:
        nested_dirs.extend([p for p in class_dir.iterdir() if p.is_dir()])

    if not nested_dirs:
        return

    lines = [
        "Nested subfolders were found inside class folders.",
        "This training script expects .npz files directly inside each class folder.",
        "",
        "Invalid nested folders:",
    ]

    for path in nested_dirs:
        lines.append(f"  - {path}")

    raise RuntimeError("\n".join(lines))


def collect_embedding_rows(embeddings_root: Path) -> tuple[List[EmbeddingRow], List[str]]:
    """Collect .npz files and class names.

    Class order is alphabetical and stable.
    """
    if not embeddings_root.exists():
        raise FileNotFoundError(f"Embeddings path does not exist: {embeddings_root}")

    if not embeddings_root.is_dir():
        raise NotADirectoryError(f"Embeddings path is not a directory: {embeddings_root}")

    class_dirs = find_class_dirs(embeddings_root)

    if not class_dirs:
        raise RuntimeError(f"No class folders found under: {embeddings_root}")

    validate_no_nested_subfolders(class_dirs)

    species_codes = [safe_code(class_dir.name) for class_dir in class_dirs]

    rows: List[EmbeddingRow] = []

    for class_dir in class_dirs:
        class_code = safe_code(class_dir.name)
        npz_files = sorted(class_dir.glob("*.npz"), key=lambda p: p.name.lower())

        if not npz_files:
            print(f"[warning] no .npz files found for class: {class_dir.name}")
            continue

        for npz_path in npz_files:
            rows.append(
                EmbeddingRow(
                    embedding_path=npz_path,
                    label_code=class_code,
                    species_name=class_dir.name,
                )
            )

    if not rows:
        raise RuntimeError(f"No .npz embedding files found under: {embeddings_root}")

    return rows, species_codes


def split_train_val(
    rows: List[EmbeddingRow],
    species_codes: List[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each class into train/validation.

    Uses the same split rule as the previous new_dataset_main.py code.
    """
    rng = random.Random(seed)

    grouped: Dict[str, List[EmbeddingRow]] = {
        label: [] for label in species_codes
    }

    for row in rows:
        grouped.setdefault(row.label_code, []).append(row)

    train_records: List[dict] = []
    val_records: List[dict] = []

    for label in species_codes:
        items = grouped.get(label, [])
        rng.shuffle(items)

        n = len(items)
        if n == 0:
            print(f"[split] {label}: total=0, val=0, train=0")
            continue

        n_val = max(1, ((n - 1) // 10) + 1)
        n_val = min(n_val, n - 1) if n > 1 else 1

        val_items = items[:n_val]
        train_items = items[n_val:]

        for item in train_items:
            train_records.append(
                {
                    "embedding_path": str(item.embedding_path),
                    "Code": item.label_code,
                    "label_code": item.label_code,
                    "species_name": item.species_name,
                    "strong_events": [],
                    "is_negative": False,
                }
            )

        for item in val_items:
            val_records.append(
                {
                    "embedding_path": str(item.embedding_path),
                    "Code": item.label_code,
                    "label_code": item.label_code,
                    "species_name": item.species_name,
                    "strong_events": [],
                    "is_negative": False,
                }
            )

        print(
            f"[split] {label}: total={n}, "
            f"val={n_val}, train={max(0, n - n_val)}"
        )

    return pd.DataFrame(train_records), pd.DataFrame(val_records)


def windows_from_segment_times(starts: np.ndarray, ends: np.ndarray) -> list:
    """Create lightweight window objects compatible with main.py visual/export code.

    main.py visualization and export functions only need:
        window.start_sec
        window.end_sec
    """
    windows = []

    for start, end in zip(starts, ends):
        windows.append(
            SimpleNamespace(
                start_sec=float(start),
                end_sec=float(end),
            )
        )

    return windows


def build_bags_from_embedding_df(df: pd.DataFrame) -> List[core.BagData]:
    """Build BagData objects directly from .npz embeddings.

    This replaces core.build_bags(), which normally loads audio and then loads
    corresponding embeddings. Here we already have embeddings, so audio loading
    is intentionally skipped.
    """
    bags: List[core.BagData] = []

    for row in df.to_dict(orient="records"):
        embedding_path = Path(row["embedding_path"])

        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

        embeddings_np, starts, ends = core.load_segment_embeddings(embedding_path)

        if embeddings_np.ndim != 2:
            raise RuntimeError(
                f"Invalid embedding shape {embeddings_np.shape} for {embedding_path}"
            )

        if embeddings_np.shape[0] == 0:
            raise RuntimeError(f"No embedding segments found in: {embedding_path}")

        if len(starts) != embeddings_np.shape[0] or len(ends) != embeddings_np.shape[0]:
            raise RuntimeError(
                f"Segment time count mismatch for {embedding_path}: "
                f"embeddings={embeddings_np.shape[0]}, "
                f"starts={len(starts)}, ends={len(ends)}"
            )

        windows = windows_from_segment_times(starts, ends)

        bag_id = embedding_path.stem
        label_code = str(row["label_code"])
        code = str(row.get("Code", label_code))

        bags.append(
            core.BagData(
                bag_id=bag_id,
                audio_path=str(embedding_path),
                code=code,
                label_code=label_code,
                embeddings=torch.from_numpy(embeddings_np).float(),
                instance_windows=windows,
                strong_events=row.get("strong_events", []),
                is_negative=bool(row.get("is_negative", False)),
            )
        )

    return bags


def check_embedding_dimensions(bags: List[core.BagData]) -> int:
    if not bags:
        raise RuntimeError("No bags were built.")

    in_dim = int(bags[0].embeddings.shape[1])

    for bag in bags:
        current_dim = int(bag.embeddings.shape[1])
        if current_dim != in_dim:
            raise RuntimeError(
                f"Inconsistent embedding dimension: expected {in_dim}, "
                f"got {current_dim} for {bag.audio_path}"
            )

    return in_dim


def print_class_order(species_codes: List[str]) -> None:
    print("[classes] alphabetical order used for label indices:")

    for idx, class_name in enumerate(species_codes):
        print(f"[classes] {idx}: {class_name}")


def train_from_embeddings() -> None:
    embeddings_root = Path(EMBEDDINGS_PATH).resolve()
    output_root = output_root_from_embeddings_path(embeddings_root)
    log_path = output_root / "run_info.txt"

    output_root.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = Tee(sys.__stdout__, log_file)
        tee_stderr = Tee(sys.__stderr__, log_file)

        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print("=" * 80)
            print("Training from pre-extracted BirdNET embeddings")
            print("=" * 80)
            print(f"Embeddings path : {embeddings_root}")
            print(f"Output path     : {output_root}")
            print(f"Run log         : {log_path}")
            print("=" * 80)

            job_hparams = load_job_hyperparameters()
            applied = apply_training_hyperparameters(job_hparams)
            if applied:
                print("[config] training hyperparameters from job request:")
                for entry in applied:
                    print(f"  {entry}")
            print(
                f"[config] EPOCHS={config.EPOCHS} "
                f"LEARNING_RATE={config.LEARNING_RATE} "
                f"MIL_POOLING={config.MIL_POOLING} "
                f"LOSS_NAME={config.LOSS_NAME} "
                f"EARLY_STOP={config.ENABLE_EARLY_STOPPING}"
            )
            print("=" * 80)

            core.seed_everything(int(config.SEED))

            rows, species_codes = collect_embedding_rows(embeddings_root)
            print_class_order(species_codes)

            train_df, val_df = split_train_val(
                rows=rows,
                species_codes=species_codes,
                seed=int(config.SEED),
            )

            print(
                f"[bundle] train={len(train_df)} "
                f"val={len(val_df)} "
                f"species={len(species_codes)}"
            )

            if len(train_df) == 0:
                raise RuntimeError(
                    "No training examples were created. "
                    "Check whether each class has enough .npz files."
                )

            train_bags = build_bags_from_embedding_df(train_df)
            val_bags = build_bags_from_embedding_df(val_df) if len(val_df) else []

            if not train_bags:
                raise RuntimeError("No train bags were built from embeddings.")

            in_dim = check_embedding_dimensions(train_bags)

            if val_bags:
                check_embedding_dimensions(val_bags)

            print(f"[birdnet] pre-extracted embeddings used, in_dim={in_dim}")

            if int(in_dim) != 1024:
                print(
                    f"[warning] Expected BirdNET embedding dimension is usually 1024, "
                    f"but got in_dim={in_dim}."
                )

            with pushd(output_root):
                model, _history = core.train_joint_weak_model(
                    train_bags=train_bags,
                    species_codes=species_codes,
                    in_dim=in_dim,
                    val_bags=val_bags,
                )

                core._save_split_visualizations(
                    split_name="training",
                    bags=train_bags,
                    model=model,
                    class_names=species_codes,
                    output_dir=Path("training"),
                )

                core._save_split_visualizations(
                    split_name="validation",
                    bags=val_bags,
                    model=model,
                    class_names=species_codes,
                    output_dir=Path("validation"),
                )

                dataset_name = dataset_name_from_embeddings_path(embeddings_root)

                core._export_eval_predictions(
                    model=model,
                    val_bags=val_bags,
                    test_bags=[],
                    species_codes=species_codes,
                    run_name=dataset_name,
                    output_dir=Path("."),
                )

            print("=" * 80)
            print("Training completed.")
            print(f"Outputs saved to: {output_root}")
            print(f"Run log saved to: {log_path}")
            print("=" * 80)


def main() -> None:
    train_from_embeddings()


if __name__ == "__main__":
    main()