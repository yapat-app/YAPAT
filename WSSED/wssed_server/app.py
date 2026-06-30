"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from wssed_server.api import artifacts, detection, system, training


def create_app() -> FastAPI:
    application = FastAPI(
        title="WSSED GPU Server",
        description="GPU server for WSSED training and detection",
        version="1.0.0",
    )
    application.include_router(training.router)
    application.include_router(artifacts.router)
    application.include_router(detection.router)
    application.include_router(system.router)
    return application


app = create_app()
