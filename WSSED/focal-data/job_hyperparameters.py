"""Apply WSSED job hyperparameters from the GPU server to focal-data config."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import config

_ENV_KEY = "WSSED_JOB_HYPERPARAMETERS"

# YAPAT UI uses pooling names that differ from focal-data MIL_POOLING values.
_POOLING_MAP = {
    "mean": "avg",
    "average": "avg",
    "avg": "avg",
    "max": "max",
    "min": "min",
    "lin": "lin",
    "linear": "lin",
    "exp": "exp",
    "exponential": "exp",
}

# Keys sent by YAPAT (see wssed_server.config_generator / training UI).
_TRAINING_KEYS: Dict[str, Tuple[str, ...]] = {
    "epochs": ("EPOCHS", "NUM_EPOCHS"),
    "learning_rate": ("LEARNING_RATE",),
    "mil_pooling": ("MIL_POOLING",),
    "loss_name": ("LOSS_NAME",),
    "enable_early_stopping": ("ENABLE_EARLY_STOPPING",),
    "early_stopping_patience": ("EARLY_STOPPING_PATIENCE",),
    "early_stopping_min_epochs": ("EARLY_STOPPING_MIN_EPOCHS",),
    "seed": ("SEED",),
    "batch_size": ("BATCH_SIZE",),
}

_EXTRACT_KEYS: Dict[str, Tuple[str, ...]] = {
    "hop_seconds": ("HOP_SECONDS",),
    "window_seconds": ("WINDOW_SECONDS",),
    "birdnet_sr": ("BIRDNET_SR",),
}


def normalize_job_hyperparameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map YAPAT API field names to focal-data config keys."""
    out = dict(data)
    if "pooling" in out and "mil_pooling" not in out:
        raw = str(out.pop("pooling")).strip().lower()
        out["mil_pooling"] = _POOLING_MAP.get(raw, raw)
    return out


def load_job_hyperparameters() -> Dict[str, Any]:
    """Load hyperparameters JSON from WSSED_JOB_HYPERPARAMETERS."""
    raw = os.environ.get(_ENV_KEY, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid {_ENV_KEY} JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{_ENV_KEY} must be a JSON object")
    return normalize_job_hyperparameters(data)


def apply_config_overrides(
    overrides: Dict[str, Any],
    key_map: Dict[str, Tuple[str, ...]],
) -> List[str]:
    applied: List[str] = []
    for api_key, config_names in key_map.items():
        if api_key not in overrides:
            continue
        value = overrides[api_key]
        for name in config_names:
            setattr(config, name, value)
        applied.append(f"{api_key}={value!r} -> {','.join(config_names)}")
    return applied


def apply_training_hyperparameters(
    overrides: Dict[str, Any] | None = None,
) -> List[str]:
    data = load_job_hyperparameters() if overrides is None else overrides
    return apply_config_overrides(data, _TRAINING_KEYS)


def apply_extraction_hyperparameters(
    overrides: Dict[str, Any] | None = None,
) -> List[str]:
    data = load_job_hyperparameters() if overrides is None else overrides
    return apply_config_overrides(data, _EXTRACT_KEYS)
