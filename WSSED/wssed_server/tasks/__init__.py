"""Background training and detection tasks."""

from wssed_server.tasks.birdnet_pipeline import run_birdnet_pipeline_task
from wssed_server.tasks.classic_training import run_training_task
from wssed_server.tasks.detection import run_detection_task

__all__ = [
    "run_birdnet_pipeline_task",
    "run_training_task",
    "run_detection_task",
]
