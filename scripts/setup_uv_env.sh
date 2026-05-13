#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python}"
UV_BIN="${UV_BIN:-uv}"
KEEP_VENV="${KEEP_VENV:-0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required. Install it first:" >&2
  echo "  python -m pip install uv" >&2
  exit 1
fi

if [[ "${KEEP_VENV}" != "1" ]]; then
ROOT_DIR_ENV="${ROOT_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import shutil
path = Path(os.environ["ROOT_DIR_ENV"]) / ".venv"
if path.exists():
    shutil.rmtree(path)
PY
fi

"${UV_BIN}" venv --python "${PYTHON_BIN}" "${VENV_DIR}"

"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade pip setuptools wheel
DS_BUILD_OPS=0 "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" -r requirements.txt

if [[ "${INSTALL_FLASH_ATTN}" == "0" ]]; then
  echo "INSTALL_FLASH_ATTN=0 was set; skipping flash-attn reinstall."
else
  "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --no-build-isolation flash-attn
fi

"${VENV_DIR}/bin/python" - <<'PY'
import torch, transformers, datasets, wandb, accelerate, deepspeed, flash_attn, peft
print({
    "python": "ok",
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "peft": peft.__version__,
    "deepspeed": deepspeed.__version__,
    "flash_attn": flash_attn.__file__,
})
PY
