#!/bin/bash
set -euo pipefail
trap 'rc=$?; echo "[install.sh] ERROR: line=$LINENO rc=$rc cmd=$BASH_COMMAND" >&2; exit $rc' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
  set +a
fi

# Locale'i UTF-8'e zorla (heredoc/çikti kaynakli encoding hatalarini engeller)
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Override with WSSED_INSTALL_LOGDIR in `.env` or env (e.g. $SCRATCH/logs on a cluster)
LOGDIR="${WSSED_INSTALL_LOGDIR:-${HOME}/wssed-install-logs}"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/install-${SLURM_JOB_ID:-nojob}.log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo ">>> [install] Host: $(hostname)"
echo ">>> [install] User: $(whoami)"
echo ">>> [install] PWD : $(pwd)"
echo ">>> [install] HOME: ${HOME}"
echo ">>> [install] SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_LOCALID=${SLURM_LOCALID:-NA}"
echo ">>> [install] Logging to: ${LOGFILE}"

DONEFILE="/tmp/sed_torch_pip_install_done_${SLURM_JOB_ID:-$$}"
if [[ "${SLURM_LOCALID:-0}" != "0" ]]; then
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
  exit 0
fi

echo ">>> [install] Python:"
which python || true
python -V

echo ">>> [install] Upgrading pip..."
python -m pip install --upgrade pip --break-system-packages || python -m pip install --upgrade pip

echo ">>> [install] Installing validators (standalone)..."
python -m pip install --no-cache-dir "validators==0.18.2" --break-system-packages || \
python -m pip install --no-cache-dir "validators==0.18.2"

echo ">>> [install] Installing PyYAML (Py3.10 compatible)..."
python -m pip install --no-cache-dir "pyyaml>=6.0" --break-system-packages || \
python -m pip install --no-cache-dir "pyyaml>=6.0"

echo ">>> [install] Ensuring numba/llvmlite (Py3.10 compatible)..."
python -m pip install --no-cache-dir \
  "llvmlite==0.41.1" "numba==0.58.1" \
  --break-system-packages || true

echo ">>> [install] Installing remaining deps (avoid old pins that force source builds)..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  appdirs audioread cffi charset-normalizer cycler dcase-util decorator future idna jedi joblib kiwisolver \
  packaging parso pooch pudb pycparser pydot-ng pygments pyparsing python-dateutil python-magic pytz \
  requests resampy sed-eval six soundfile threadpoolctl tqdm typing-extensions urllib3 urwid \
  --break-system-packages || true

python -m pip install --no-cache-dir "librosa>=0.10.0" --break-system-packages || true

# ---------
# Sanity check: ASCII-only, fail etse bile prolog'u fail ettirme
# ---------
echo ">>> [install] Final sanity check (non-fatal):"
python - <<'PY' || true
import importlib

mods = ["validators", "yaml", "numpy", "scipy", "pandas", "librosa", "torch"]
for m in mods:
    try:
        x = importlib.import_module(m)
        v = getattr(x, "__version__", "NA")
        print(m + ": OK " + str(v))
    except Exception as e:
        print(m + ": FAILED " + repr(e))

try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
except Exception as e:
    print("torch cuda check failed:", repr(e))
PY

touch "${DONEFILE}"
echo ">>> [install] Done."
