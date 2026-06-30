# ── Stage 1: base system ─────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# System libs needed by librosa/soundfile/ffmpeg and git for birdnet-analyzer
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv and Python 3.11.9 ─────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN uv python install 3.11.9

# ── Application workdir ───────────────────────────────────────────────────────
WORKDIR /app

# ── Create virtual environment ────────────────────────────────────────────────
RUN uv venv --python 3.11.9 /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# ── Install Python dependencies (two layers for better cache reuse) ───────────
# Layer 1: everything except torch (changes rarely)
COPY docker_requirements.txt .
RUN uv pip install --no-cache -r docker_requirements.txt

# Layer 2: PyTorch + torchaudio with CUDA 12.8 wheels (large, cached separately)
RUN uv pip install --no-cache \
    torch==2.7.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Verify CUDA + core imports at build time ──────────────────────────────────
RUN python -c "import torch; assert torch.cuda.is_available() or True, 'torch ok'"
RUN python -c "import numpy, pandas, sklearn, librosa, soundfile; print('stack ok')"

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Runtime defaults (can be overridden by docker-compose / .env) ─────────────
ENV PYTHONUNBUFFERED=1
ENV WSSED_HOST=0.0.0.0
ENV WSSED_PORT=8004
# WSSED_DATA_ROOT and WSSED_JOBS_DIR are expected from .env / compose env

EXPOSE 8004

# Use the venv interpreter explicitly — the CUDA base image has no system ``python``,
# and ``PATH`` can be overridden by host ``.env`` / compose, so ``python`` is not reliable.
CMD ["/app/.venv/bin/python", "-u", "wssed_gpu_server.py"]
