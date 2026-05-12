#!/bin/bash

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${ROOT_DIR}/.venv"

ROOT_DIR_ENV="${ROOT_DIR}" python - <<'PY'
import os
from pathlib import Path
import shutil
path = Path(os.environ["ROOT_DIR_ENV"]) / ".venv"
if path.exists():
    shutil.rmtree(path)
PY

uv venv --system-site-packages "${VENV_DIR}"

uv pip install --python "${VENV_DIR}/bin/python" \
  hjson \
  ninja \
  pydantic \
  psutil \
  py-cpuinfo \
  msgpack \
  packaging \
  tqdm \
  numpy \
  einops

DS_BUILD_OPS=0 uv pip install --python "${VENV_DIR}/bin/python" --no-deps deepspeed
uv pip install --python "${VENV_DIR}/bin/python" --no-deps peft

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
