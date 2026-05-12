#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODE="${MODE:-delta}"
MODEL_PATH="${MODEL_PATH:-/root/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
ADAPTER_DIR="${ADAPTER_DIR:-/root/models/qwen3_4b_instruct_delta_mem_qasper_6k_seed42_rank8_qo_TSW_write8192_consistent70/trainer/checkpoint-70}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" -m deltamem.demo.chat_demo
  --mode "${MODE}"
  --model-path "${MODEL_PATH}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ "${MODE}" != "base" ]]; then
  CMD+=(--adapter-dir "${ADAPTER_DIR}")
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${CMD[@]}" "$@"

