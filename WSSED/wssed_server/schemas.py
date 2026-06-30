"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    job_id: int
    dataset_id: int
    dataset_path: str = Field(
        ..., description="Path to dataset (relative to data root or absolute)"
    )
    hyperparameters: Dict[str, Any]
    feedback_labels: Optional[List[Dict[str, Any]]] = None


class TrainingStatusResponse(BaseModel):
    status: str
    model_path: Optional[str] = None
    model_paths: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


class DetectionRequest(BaseModel):
    job_id: int
    model_path: str
    dataset_path: str
    threshold: float = 0.5


class DetectionResponse(BaseModel):
    job_id: int
    message: str
    predictions_count: Optional[int] = None
