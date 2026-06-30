"""
WSSED GPU Server — entry point.

Configuration: copy `.env.example` to `.env` in the repo root, or export variables.

  WSSED_DATA_ROOT       — root for relative dataset_path (default: /ds-iml/Bioacoustics)
  WSSED_JOBS_DIR        — writable job state directory (default: /wssed_jobs)
  WSSED_HOST            — bind host for uvicorn __main__ (default: 0.0.0.0)
  WSSED_PORT            — bind port (default: 8003)
  WSSED_CUDA_VISIBLE_DEVICES — optional GPU index for training subprocesses
"""

from wssed_server.app import app

__all__ = ["app"]

if __name__ == "__main__":
    import os

    import uvicorn

    host = os.environ.get("WSSED_HOST", "0.0.0.0")
    port = int(os.environ.get("WSSED_PORT", "8003"))
    uvicorn.run(app, host=host, port=port, log_level="info")
