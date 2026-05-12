#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
RUN_NAME="qwen3_4b_instruct_delta_mem_qasper_6k_seed42_rank8_qo_SSW_sentence_write8192_consistent70_temp"
MODEL_ROOT="/root/models/${RUN_NAME}"
PIPELINE_ROOT="/root/outputs/${RUN_NAME}_train_and_benchmark_temp"
SUITE_ROOT="/root/outputs/${RUN_NAME}_benchmark_temp"
EVAL_TASKS_STRING="${EVAL_TASKS_STRING:-locomo hotpotqa gpqa_diamond ifeval memory_agent_bench}"

cd "${ROOT_DIR}"

echo "=== qwen3_4b_instruct: sentence SSW train + benchmark temp ==="
echo "RUN_NAME=${RUN_NAME}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "SUITE_ROOT=${SUITE_ROOT}"
echo "EVAL_TASKS_STRING=${EVAL_TASKS_STRING}"

PIPELINE_ROOT="${PIPELINE_ROOT}" \
LOG_ROOT="${PIPELINE_ROOT}/logs" \
MANIFEST_PATH="${PIPELINE_ROOT}/run_manifest.txt" \
SUITE_ROOT="${SUITE_ROOT}" \
SSW_RUN_NAME="${RUN_NAME}" \
SSW_OUTPUT_DIR="${MODEL_ROOT}" \
SSW_WRITE_GRANULARITY="sentence_mean" \
SSW_MASTER_PORT="30282" \
SSW_WANDB_GROUP="qasper-sentence-ssw-temp" \
SSW_WANDB_TAGS="qasper,delta,SSW,sentence_mean,write8192,consistent70,temp" \
SSW_ADAPTER_DIR="${MODEL_ROOT}/trainer/checkpoint-70" \
TRAIN_VARIANTS_STRING="SSW_rank8_qasper_write8192" \
BENCHMARK_VARIANTS_STRING="SSW_rank8_qasper_write8192" \
EVAL_TASKS_STRING="${EVAL_TASKS_STRING}" \
bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite.sh

echo "Done: qwen3_4b sentence SSW train + benchmark temp"
