#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
EVAL_TASKS_STRING="${EVAL_TASKS_STRING:-locomo hotpotqa gpqa_diamond ifeval memory_agent_bench}"

cd "${ROOT_DIR}"

echo "=== qwen3_4b_instruct: TSW train + benchmark ==="
TRAIN_VARIANTS_STRING="TSW_rank8_qasper_write8192" \
BENCHMARK_VARIANTS_STRING="TSW_rank8_qasper_write8192" \
EVAL_TASKS_STRING="${EVAL_TASKS_STRING}" \
bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite.sh

echo "=== qwen3_8b: TSW train + benchmark ==="
TRAIN_VARIANTS_STRING="TSW_rank8_qasper_write8192" \
BENCHMARK_VARIANTS_STRING="TSW_rank8_qasper_write8192" \
EVAL_TASKS_STRING="${EVAL_TASKS_STRING}" \
bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite_qwen3_8b.sh

echo "=== smollm3_3b: TSW train + benchmark ==="
TRAIN_VARIANTS_STRING="TSW_rank8_qasper_write8192" \
BENCHMARK_VARIANTS_STRING="TSW_rank8_qasper_write8192" \
EVAL_TASKS_STRING="${EVAL_TASKS_STRING}" \
bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite_smollm3_3b.sh

echo "Done: TSW three-model train + benchmark suite"
