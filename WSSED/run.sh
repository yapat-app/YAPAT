#!/bin/bash
# ===== SBATCH SETTINGS =====
#SBATCH --job-name=HiPool_FNJV
#SBATCH --partition=A100-IML,H100,H100-PCI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nmammadli/WSSED/HiPool/slurm-%x-%j.out
#SBATCH --chdir=/home/nmammadli/WSSED

srun -K \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-workdir=/home/nmammadli/WSSED \
  --container-mounts=/home/nmammadli:/home/nmammadli,/ds-iml:/ds-iml:ro \
  --gpus=1 \
  --task-prolog=/home/nmammadli/WSSED/HiPool/install.sh \
  python /home/nmammadli/WSSED/HiPool/main.py
