"""Direct BirdNET integration using the installed `birdnet-analyzer` package."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Tuple

import numpy as np

import config


def import_birdnet_analyzer_strict() -> object:
    try:
        module = importlib.import_module("birdnet_analyzer")
    except Exception as exc:
        raise RuntimeError(
            "FAIL: Could not import 'birdnet_analyzer'. "
            "Install it with: pip install birdnet-analyzer"
        ) from exc

    module_file = str(Path(getattr(module, "__file__", "")))
    print(f"[birdnet] birdnet_analyzer imported from={module_file}")
    return module


def _get_embeddings_callable():
    model_mod = importlib.import_module("birdnet_analyzer.model")
    fn = getattr(model_mod, "embeddings", None)
    if not callable(fn):
        raise RuntimeError("FAIL: birdnet_analyzer.model.embeddings callable not found")

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    required = [
        p for p in params
        if p.default is inspect._empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    print(f"[birdnet] birdnet_analyzer.model.embeddings signature={sig}")
    if len(required) != 1:
        raise RuntimeError(
            "FAIL: birdnet_analyzer.model.embeddings must accept exactly one required argument (sample_batch)"
        )
    return fn


def _configure_model_path_and_load_model() -> None:
    pkg = importlib.import_module("birdnet_analyzer")
    bcfg = importlib.import_module("birdnet_analyzer.config")
    butils = importlib.import_module("birdnet_analyzer.utils")
    configured = getattr(bcfg, "MODEL_PATH", None)
    override_path = getattr(config, "BIRDNET_MODEL_PATH", None)

    if override_path is not None and str(override_path).strip():
        configured = str(override_path).strip()
        setattr(bcfg, "MODEL_PATH", configured)
        print(f"[birdnet] MODEL_PATH overridden from config: {configured}")
    else:
        if configured is None or not str(configured).strip():
            # For pip-installed birdnet_analyzer, model files may not be pre-bundled.
            # Ask the library to download/prepare checkpoints when missing.
            butils.ensure_model_exists()
            pkg_dir = Path(getattr(pkg, "__file__", "")).resolve().parent
            candidates = [
                pkg_dir / "checkpoints" / "V2.4" / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite",
                pkg_dir / "checkpoints" / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite",
            ]
            resolved = next((str(p) for p in candidates if p.exists()), None)
            if resolved is None:
                checkpoint_roots = [
                    pkg_dir / "checkpoints",
                    pkg_dir / "models",
                ]
                for root in checkpoint_roots:
                    if not root.exists():
                        continue
                    tflites = sorted(root.rglob("*.tflite"))
                    if tflites:
                        resolved = str(tflites[0])
                        print(f"[birdnet] MODEL_PATH discovered by recursive scan: {resolved}")
                        break
            if resolved is None:
                raise RuntimeError(
                    "FAIL: birdnet_analyzer MODEL_PATH is not set and default packaged checkpoint was not found. "
                    "Set config.BIRDNET_MODEL_PATH to your .tflite model path."
                )
            configured = resolved
            setattr(bcfg, "MODEL_PATH", configured)
            print(f"[birdnet] MODEL_PATH auto-resolved from package files: {configured}")
        else:
            print(f"[birdnet] MODEL_PATH from package config: {configured}")

    if "mdata" in str(configured).lower():
        raise RuntimeError(f"FAIL: MData model is not allowed for embedding extraction: MODEL_PATH={configured}")

    bmodel = importlib.import_module("birdnet_analyzer.model")
    load_model_fn = getattr(bmodel, "load_model", None)
    if not callable(load_model_fn):
        raise RuntimeError("FAIL: birdnet_analyzer.model.load_model callable not found")

    sig = inspect.signature(load_model_fn)
    print(f"[birdnet] birdnet_analyzer.model.load_model signature={sig}")
    if "class_output" not in sig.parameters:
        raise RuntimeError("FAIL: load_model does not support class_output argument")

    load_model_fn(class_output=False)
    print(f"[birdnet] model loaded OK, MODEL_PATH={configured}")


class BirdNetDirectExtractor:
    """Batch extractor for BirdNET embeddings with sample batch shape [n_segments, 144000]."""

    def __init__(self) -> None:
        self.module = import_birdnet_analyzer_strict()
        _configure_model_path_and_load_model()
        self.embeddings_fn = _get_embeddings_callable()

    def embed_batch(self, sample_batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(sample_batch, dtype=np.float32)
        if batch.ndim != 2:
            raise RuntimeError(f"FAIL: BirdNET sample batch must be 2D [n_segments, n_samples], got {batch.shape}")

        expected_samples = int(config.WINDOW_SECONDS * config.BIRDNET_SR)
        if batch.shape[1] != expected_samples:
            raise RuntimeError(
                f"FAIL: BirdNET sample width must be {expected_samples} for {config.WINDOW_SECONDS}s@{config.BIRDNET_SR}Hz, got {batch.shape[1]}"
            )

        output = self.embeddings_fn(batch)
        output = np.asarray(output)
        print(f"[birdnet] embeddings output shape={output.shape}")

        if output.ndim != 2:
            raise RuntimeError(f"FAIL: BirdNET embeddings must be 2D, got {output.shape}")
        if output.shape[0] != batch.shape[0]:
            raise RuntimeError(
                f"FAIL: n_segments mismatch. batch={batch.shape[0]} embeddings={output.shape[0]}"
            )

        return output.astype(np.float32)


def smoke_check_single_wav(
    wav_path: str,
    cache_dir: str,
    builder_fn,
) -> None:
    embeddings, starts, ends = builder_fn(wav_path, cache_dir)
    out_path = Path(cache_dir) / (Path(wav_path).stem + ".birdnet_segments.npz")
    if not out_path.exists():
        raise RuntimeError(f"FAIL: Smoke check cache file missing: {out_path}")
    if embeddings.ndim != 2:
        raise RuntimeError(f"FAIL: Smoke check invalid embedding shape {embeddings.shape}")
    if len(starts) != len(ends) or len(starts) != embeddings.shape[0]:
        raise RuntimeError("FAIL: Smoke check starts/ends mismatch")
    if len(starts) > 1:
        diffs = np.diff(starts)
        if not np.allclose(diffs, 1.0, atol=1e-4):
            raise RuntimeError(f"FAIL: Smoke check start increments are not 1s: {diffs[:5]}")
    if not np.allclose(ends - starts, float(config.WINDOW_SECONDS), atol=1e-4):
        raise RuntimeError("FAIL: Smoke check end-start is not WINDOW_SECONDS")
    print(f"[birdnet] smoke_check PASS | cache={out_path} shape={embeddings.shape}")
