#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: set PYTHON_BIN or run ./scripts/setup_uv_env.sh first" >&2
  exit 1
fi

MODEL_KINDS="${MODEL_KINDS:-base}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-4096 16384 32768}"
DECODE_LENGTHS="${DECODE_LENGTHS:-64 256}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

read -r -a MODEL_KIND_ARRAY <<< "${MODEL_KINDS}"
read -r -a PROMPT_LENGTH_ARRAY <<< "${PROMPT_LENGTHS}"
read -r -a DECODE_LENGTH_ARRAY <<< "${DECODE_LENGTHS}"

CMD=(
  "${PYTHON_BIN}" -m deltamem.tools.generation_tps_benchmark
  --model-kinds "${MODEL_KIND_ARRAY[@]}"
  --model-path "${MODEL_PATH}"
  --device cuda:0
  --dtype bfloat16
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --prompt-lengths "${PROMPT_LENGTH_ARRAY[@]}"
  --decode-lengths "${DECODE_LENGTH_ARRAY[@]}"
  --batch-size 1
  --warmup-runs 1
  --measure-runs 3
  --measurement-mode full_generate
  --seed 42
)

if [[ -n "${ADAPTER_DIR}" ]]; then
  CMD+=(--adapter-dir "${ADAPTER_DIR}")
fi
if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD+=(--output-json "${OUTPUT_JSON}")
fi

printf '+'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
