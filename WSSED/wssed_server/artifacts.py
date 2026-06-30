"""Locate training checkpoints and related artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from wssed_server.settings import WSSED_ROOT


def latest_existing_path(
    candidates: List[Path],
    started_after: Optional[float] = None,
) -> Optional[str]:
    existing = [p for p in candidates if p.exists()]
    if started_after is not None:
        existing = [p for p in existing if p.stat().st_mtime >= started_after]
    if not existing:
        return None
    return str(max(existing, key=lambda path: path.stat().st_mtime))


def collect_model_paths(
    job_dir: Path,
    started_after: Optional[float] = None,
    extra_roots: Optional[List[Path]] = None,
) -> Dict[str, str]:
    """Collect WSSED artifacts from classic and BirdNET training."""
    output_runs = [
        path
        for path in (WSSED_ROOT / "TALNet" / "outputs").glob("*")
        if started_after is None or path.stat().st_mtime >= started_after
    ]
    output_runs = sorted(output_runs, key=lambda path: path.stat().st_mtime, reverse=True)
    latest_output = output_runs[0] if output_runs else None

    search_roots = [job_dir, WSSED_ROOT, WSSED_ROOT / "focal-data"]
    focal_data_dir = WSSED_ROOT / "focal-data"
    for outputs_dir in focal_data_dir.glob("outputs_of_*/"):
        if started_after is None or outputs_dir.stat().st_mtime >= started_after:
            search_roots.append(outputs_dir)
    if extra_roots:
        search_roots.extend(extra_roots)

    artifacts: Dict[str, str] = {}
    for key, filename in {
        "best_micro_model_segment": "best_micro_model_segment.pt",
        "best_micro_model": "best_micro_model.pt",
        "best_macro_model_segment": "best_macro_model_segment.pt",
        "best_macro_model": "best_macro_model.pt",
    }.items():
        resolved = latest_existing_path(
            [root / filename for root in search_roots],
            started_after=started_after,
        )
        if resolved:
            artifacts[key] = resolved

    if latest_output is not None:
        for key, filename in {
            "classic_micro": "best_model_micro.pt",
            "classic_macro": "best_model_macro.pt",
            "training_metrics_plot": "training_metrics.png",
        }.items():
            path = latest_output / filename
            if path.exists():
                artifacts[key] = str(path)
        artifacts["output_dir"] = str(latest_output)

    preferred = (
        artifacts.get("best_micro_model_segment")
        or artifacts.get("best_micro_model")
        or artifacts.get("classic_micro")
        or artifacts.get("classic_macro")
        or artifacts.get("best_macro_model_segment")
        or artifacts.get("best_macro_model")
    )
    if preferred:
        artifacts["preferred"] = preferred
    return artifacts
