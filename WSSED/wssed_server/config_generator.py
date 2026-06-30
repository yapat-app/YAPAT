"""Generate per-job config.py and feedback CSV for classic WSSED training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from wssed_server.settings import DATA_ROOT, WSSED_ROOT, logger, resolve_dataset_path


def generate_config_file(
    hyperparameters: Dict[str, Any],
    dataset_path: str,
    job_dir: Path,
) -> Path:
    abs_dataset_path = resolve_dataset_path(dataset_path)

    anuraset_root = (DATA_ROOT / "AnuraSet" / "raw_data").resolve()
    strong_458 = (
        DATA_ROOT / "AnuraSet" / "strong_labels" / "FNJV_458_species_labels.csv"
    ).resolve()
    strong_578 = (
        DATA_ROOT / "AnuraSet" / "strong_labels" / "FNJV_578_species_labels.csv"
    ).resolve()

    _bag_raw = hyperparameters.get("bag_seconds", "full")
    bag_seconds_repr = '"full"' if str(_bag_raw).strip() == "full" else int(_bag_raw)

    default_species_458 = [
        "DENMIN", "LEPLAT", "PHYCUV", "SPHSUR", "SCIPER",
        "BOABIS", "BOAFAB", "LEPPOD", "PHYALB",
    ]
    default_species_578 = ["DENMIN", "BOARAN", "DENNAN", "LEPFUS"]
    _species = hyperparameters.get("target_species", [])
    if not isinstance(_species, list) or not _species:
        _species = (
            default_species_458 if "458" in str(abs_dataset_path) else default_species_578
        )
    target_species_repr = repr(_species)
    model_name = str(hyperparameters.get("model_name", "CDur")).strip()
    is_birdnet = model_name.lower() == "birdnet"
    pipeline = "birdnet" if is_birdnet else "classic"
    normalized_model_name = "BirdNET" if is_birdnet else model_name
    birdnet_cache_dir = (WSSED_ROOT / "birdnet_cache").resolve()
    strong_458_subset = (
        DATA_ROOT / "AnuraSet" / "strong_labels" / "FNJV_458_species_labels_with_subset.csv"
    ).resolve()
    strong_578_subset = (
        DATA_ROOT / "AnuraSet" / "strong_labels" / "FNJV_578_species_labels_with_subset.csv"
    ).resolve()
    anuraset_strong_root = (DATA_ROOT / "AnuraSet" / "strong_labels").resolve()
    fnjv_458_root = (DATA_ROOT / "FNJV" / "458").resolve()
    fnjv_578_root = (DATA_ROOT / "FNJV" / "578").resolve()
    precomputed_txt = (WSSED_ROOT / "birdnet_txt_embeddings").resolve()

    config_content = f'''"""
Auto-generated WSSED configuration for job {hyperparameters.get('job_id', 'unknown')}.
Do not edit – recreated on every training run by wssed_server.
"""

PIPELINE = "{pipeline}"

SEED = 42

ANURASET_ROOT = "{anuraset_root}"
FNJV_ROOT = "{abs_dataset_path}"
FNJV_458_ROOT = "{fnjv_458_root}"
FNJV_578_ROOT = "{fnjv_578_root}"
ANURASET_STRONG_ROOT = "{anuraset_strong_root}"

DATASET_TRAIN = "FNJV"
DATASET_VAL = "FNJV"
DATASET_TEST = "FNJV"

POOLING = "{hyperparameters.get('pooling', 'mean')}"
BAG_SECONDS = {bag_seconds_repr}
FULL_BAG_METHOD = "batch"
PAD_MODE = "repeat"

MODEL_NAME = "{normalized_model_name}"
EPOCHS = {hyperparameters.get('epochs', 100)}
NUM_EPOCHS = EPOCHS
BATCH_SIZE = {hyperparameters.get('batch_size', 8)}
NUM_WORKERS = {hyperparameters.get('num_workers', 4)}
LEARNING_RATE = {hyperparameters.get('learning_rate', 0.001)}

sample_rate = {hyperparameters.get('sample_rate', 22000)}
n_mels = {hyperparameters.get('n_mels', 64)}
n_fft = {hyperparameters.get('n_fft', 1100)}
hop_length = {hyperparameters.get('hop_length', 550)}
SAMPLE_RATE = sample_rate
N_MELS = n_mels
N_FFT = n_fft
threshold = {hyperparameters.get('threshold', 0.5)}

VALIDATION_SPLIT = {hyperparameters.get('validation_split', 0.1)}
TEST_SPLIT = {hyperparameters.get('test_split', 0.1)}
TRAIN_SPLIT = {hyperparameters.get('train_split', 0.1)}
APPLY_TRAIN_SPLIT = {bool(hyperparameters.get('apply_train_split', False))}
APPLY_VALIDATION_SPLIT = True
APPLY_TEST_SPLIT = True
INCLUDE_NEGATIVE_SPLITS = True

USE_CLASS_SPECIFIC_THRESHOLD_TUNING = False

TARGET_SPECIES = {target_species_repr}

OVERLAP_BAGS = False
HOP_SECONDS = {hyperparameters.get('hop_seconds', 1)}

STRONG_LABELS_458 = "{strong_458}"
STRONG_LABELS_578 = "{strong_578}"
STRONG_LABELS_458_WITH_SUBSET = "{strong_458_subset}"
STRONG_LABELS_578_WITH_SUBSET = "{strong_578_subset}"
STRONG_LABEL_LEVELS = ["High", "Medium", "Low"]

ANURASET_EVAL = {{
    "tag_threshold": 0.5,
    "loc_threshold_high": 0.3,
    "loc_threshold_low": 0.1,
    "smooth": 10,
}}

LOCALIZATION_MODE = "frame"
BLOCK_SECONDS = 1.0

SEEDS = [SEED]
WINDOW_SECONDS = {hyperparameters.get('window_seconds', 3)}
MIL_POOLING = "{hyperparameters.get('mil_pooling', 'lin')}"
EMBEDDING_SOURCE = "birdnet_analyzer"
ALLOW_AUTO_EXTRACT = True
USE_FALLBACK_FEATURES = False
BIRDNET_CACHE_DIR = "{birdnet_cache_dir}"
PRECOMPUTED_TXT_DIR = "{precomputed_txt}"
BIRDNET_MODEL_PATH = None
BIRDNET_SR = 48000
EXPECTED_EMBED_DIM = 1024
LOSS_NAME = "{hyperparameters.get('loss_name', 'bce')}"
WEIGHTED_BCE_POS_WEIGHT_CLAMP = 20.0
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
ASL_GAMMA_POS = 0.0
ASL_GAMMA_NEG = 4.0
ASL_CLIP = 0.05
ENABLE_EARLY_STOPPING = {bool(hyperparameters.get('enable_early_stopping', True))}
EARLY_STOPPING_PATIENCE = {hyperparameters.get('early_stopping_patience', 20)}
EARLY_STOPPING_MIN_EPOCHS = {hyperparameters.get('early_stopping_min_epochs', 30)}
SMOKE_MODE = False
SMOKE_MAX_BAGS = 2
SMOKE_EPOCHS = 1
TARGET_SPECIES_458 = {repr(hyperparameters.get('target_species_458', default_species_458))}
TARGET_SPECIES_578 = {target_species_repr}
USE_NEGATIVE_SAMPLES = False
NEGATIVE_SAMPLE_RATIO = 0.05
GUI = False
MULTI_LABEL = False
RETRAIN_EVERY_N = 5
PRINT_EVERY_EPOCHS = 10
COLORED_GT = False
'''

    config_path = job_dir / "config.py"
    config_path.write_text(config_content)
    logger.info("Generated config file: %s", config_path)
    return config_path


def create_strong_labels_csv(
    feedback_labels: List[Dict[str, Any]],
    job_dir: Path,
) -> Optional[Path]:
    if not feedback_labels:
        return None

    import pandas as pd

    csv_path = job_dir / "feedback_strong_labels.csv"
    pd.DataFrame(feedback_labels).to_csv(csv_path, index=False)
    logger.info("Created strong labels CSV: %s", csv_path)
    return csv_path
