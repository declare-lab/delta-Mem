#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SUITE_ROOT="/root/outputs/qasper_multimodel_write8192_benchmark_suite_qwen3_8b_consistent70"
LOG_ROOT="${SUITE_ROOT}/logs"
MANIFEST_PATH="${SUITE_ROOT}/run_manifest.txt"
HF_HOME="/root/huggingface"
HF_HUB_CACHE="/root/huggingface/hub"
HF_DATASETS_CACHE="/root/datasets_cache"
MEMORY_AGENT_BENCH_ROOT="/root/external/MemoryAgentBench"
LOCOMO_DATA_FILE="${ROOT_DIR}/data/locomo10.json"
BASE_MODEL_PATH="/root/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
SSW_ADAPTER_DIR="${SSW_ADAPTER_DIR:-/root/models/qwen3_8b_delta_mem_qasper_6k_seed42_rank8_qo_SSW_write8192_consistent70/trainer/checkpoint-70}"
TSW_ADAPTER_DIR="${TSW_ADAPTER_DIR:-/root/models/qwen3_8b_delta_mem_qasper_6k_seed42_rank8_qo_TSW_write8192_consistent70/trainer/checkpoint-70}"
MSW_ADAPTER_DIR="${MSW_ADAPTER_DIR:-/root/models/qwen3_8b_delta_mem_qasper_6k_seed42_MSW_write8192_consistent70/trainer/checkpoint-70}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEFAULT_EVAL_TASKS=(locomo hotpotqa gpqa_diamond ifeval memory_agent_bench)
EVAL_TASKS_STRING="${EVAL_TASKS_STRING:-${DEFAULT_EVAL_TASKS[*]}}"
read -r -a EVAL_TASKS <<< "${EVAL_TASKS_STRING}"
if [[ ${#EVAL_TASKS[@]} -eq 0 ]]; then
  echo "EVAL_TASKS_STRING resolved to no tasks" >&2
  exit 1
fi
DEFAULT_BENCHMARK_VARIANTS=(base_model SSW_rank8_qasper_write8192 TSW_rank8_qasper_write8192 MSW_qasper_write8192)
BENCHMARK_VARIANTS_STRING="${BENCHMARK_VARIANTS_STRING:-${DEFAULT_BENCHMARK_VARIANTS[*]}}"
read -r -a BENCHMARK_VARIANTS <<< "${BENCHMARK_VARIANTS_STRING}"
if [[ ${#BENCHMARK_VARIANTS[@]} -eq 0 ]]; then
  echo "BENCHMARK_VARIANTS_STRING resolved to no variants" >&2
  exit 1
fi
for variant_slug in "${BENCHMARK_VARIANTS[@]}"; do
  case "${variant_slug}" in
    base_model|SSW_rank8_qasper_write8192|TSW_rank8_qasper_write8192|MSW_qasper_write8192) ;;
    *)
      echo "Unsupported BENCHMARK_VARIANTS_STRING entry: ${variant_slug}" >&2
      exit 1
      ;;
  esac
done
MEMORY_AGENT_BENCH_EVAL_BATCH_SIZE="${MEMORY_AGENT_BENCH_EVAL_BATCH_SIZE:-16}"
HOTPOTQA_OFFICIAL_DECODING="${HOTPOTQA_OFFICIAL_DECODING:-1}"
GPQA_OFFICIAL_DECODING="${GPQA_OFFICIAL_DECODING:-1}"
HOTPOTQA_OFFICIAL_DECODING_FLAG=()
if [[ "${HOTPOTQA_OFFICIAL_DECODING}" == "1" ]]; then
  HOTPOTQA_OFFICIAL_DECODING_FLAG+=(--hotpotqa-official-decoding)
fi
GPQA_OFFICIAL_DECODING_FLAG=()
if [[ "${GPQA_OFFICIAL_DECODING}" == "1" ]]; then
  GPQA_OFFICIAL_DECODING_FLAG+=(--gpqa-official-decoding)
fi
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
BASE_INFERENCE_BACKEND="${BASE_INFERENCE_BACKEND:-transformers}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "${SUITE_ROOT}" "${LOG_ROOT}"

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname -- "${log_file}")"
  print_cmd "$@"
  "$@" > >(tee "${log_file}") 2> >(tee -a "${log_file}" >&2)
}

locomo_is_complete() {
  local output_json="$1"
  [[ -f "${output_json}" ]]
}

benchmark_is_complete() {
  local output_json="$1"
  [[ -f "${output_json}" ]]
}

variant_enabled() {
  local requested="$1"
  local variant_slug
  for variant_slug in "${BENCHMARK_VARIANTS[@]}"; do
    [[ "${variant_slug}" == "${requested}" ]] && return 0
  done
  return 1
}

run_locomo_base_like() {
  local variant_slug="$1"
  local master_port="$2"
  local output_json="${SUITE_ROOT}/${variant_slug}/locomo.json"
  local log_file="${LOG_ROOT}/${variant_slug}_locomo.log"
  mkdir -p "$(dirname -- "${output_json}")"
  if locomo_is_complete "${output_json}"; then
    echo "Skip completed LoCoMo: ${variant_slug}"
    return 0
  fi
  run_with_log "${log_file}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port "${master_port}" \
    -m deltamem.eval.locomo_delta \
    --model-path "${BASE_MODEL_PATH}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --max-new-tokens 50 \
    --seed 42 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --answer-reserve-tokens 50 \
    --full-history-mode official_prompt \
    --categories 1 2 3 4 \
    --output-json "${output_json}" \
    --data-file "${LOCOMO_DATA_FILE}"
}

run_locomo_delta_only() {
  local variant_slug="$1"
  local adapter_dir="$2"
  local master_port="$3"
  local output_json="${SUITE_ROOT}/${variant_slug}/locomo.json"
  local log_file="${LOG_ROOT}/${variant_slug}_locomo.log"
  mkdir -p "$(dirname -- "${output_json}")"
  if locomo_is_complete "${output_json}"; then
    echo "Skip completed LoCoMo: ${variant_slug}"
    return 0
  fi
  run_with_log "${log_file}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port "${master_port}" \
    -m deltamem.eval.locomo_delta \
    --model-path "${BASE_MODEL_PATH}" \
    --adapter-dir "${adapter_dir}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --max-new-tokens 50 \
    --seed 42 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --answer-reserve-tokens 50 \
    --skip-base \
    --delta-conditions full_history_replay \
    --full-history-mode official_prompt \
    --categories 1 2 3 4 \
    --output-json "${output_json}" \
    --data-file "${LOCOMO_DATA_FILE}"
}

run_benchmark_task_base_like() {
  local variant_slug="$1"
  local task_name="$2"
  local master_port="$3"
  local output_json="${SUITE_ROOT}/${variant_slug}/${task_name}.json"
  local log_file="${LOG_ROOT}/${variant_slug}_${task_name}.log"
  mkdir -p "$(dirname -- "${output_json}")"
  if benchmark_is_complete "${output_json}"; then
    echo "Skip completed ${task_name}: ${variant_slug}"
    return 0
  fi
  # --memory-agent-bench-max-context-chars is a fast evaluation preclip for very long
  # MemoryAgentBench inputs before token-budget truncation.
  run_with_log "${log_file}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port "${master_port}" \
    -m deltamem.eval.benchmark_compare \
    --model-path "${BASE_MODEL_PATH}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --datasets-cache-dir "${HF_DATASETS_CACHE}" \
    --hub-cache-dir "${HF_HUB_CACHE}" \
    --external-memory-agent-bench-root "${MEMORY_AGENT_BENCH_ROOT}" \
    --tasks "${task_name}" \
    --memory-agent-bench-splits Accurate_Retrieval Test_Time_Learning Long_Range_Understanding Conflict_Resolution \
    --seed 42 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --base-inference-backend "${BASE_INFERENCE_BACKEND}" \
    --hotpotqa-max-new-tokens 32 \
    "${HOTPOTQA_OFFICIAL_DECODING_FLAG[@]}" \
    --gpqa-max-new-tokens 8192 \
    "${GPQA_OFFICIAL_DECODING_FLAG[@]}" \
    --ifeval-max-new-tokens 1500 \
    --memory-agent-bench-max-new-tokens 4096 \
    --memory-agent-bench-eval-batch-size "${MEMORY_AGENT_BENCH_EVAL_BATCH_SIZE}" \
    --memory-agent-bench-max-context-chars 120000 \
    --no-memory-agent-bench-use-official-prompt \
    --eval-do-sample \
    --eval-temperature 0.4 \
    --eval-top-p 0.9 \
    --eval-top-k 10 \
    --local-files-only \
    --skip-delta \
    --skip-lora \
    --output-json "${output_json}"
}

run_benchmark_task_delta_only() {
  local variant_slug="$1"
  local adapter_dir="$2"
  local task_name="$3"
  local master_port="$4"
  local output_json="${SUITE_ROOT}/${variant_slug}/${task_name}.json"
  local log_file="${LOG_ROOT}/${variant_slug}_${task_name}.log"
  mkdir -p "$(dirname -- "${output_json}")"
  if benchmark_is_complete "${output_json}"; then
    echo "Skip completed ${task_name}: ${variant_slug}"
    return 0
  fi
  # --memory-agent-bench-max-context-chars is a fast evaluation preclip for very long
  # MemoryAgentBench inputs before token-budget truncation.
  run_with_log "${log_file}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port "${master_port}" \
    -m deltamem.eval.benchmark_compare \
    --model-path "${BASE_MODEL_PATH}" \
    --delta-adapter-dir "${adapter_dir}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --datasets-cache-dir "${HF_DATASETS_CACHE}" \
    --hub-cache-dir "${HF_HUB_CACHE}" \
    --external-memory-agent-bench-root "${MEMORY_AGENT_BENCH_ROOT}" \
    --tasks "${task_name}" \
    --memory-agent-bench-splits Accurate_Retrieval Test_Time_Learning Long_Range_Understanding Conflict_Resolution \
    --seed 42 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --base-inference-backend "${BASE_INFERENCE_BACKEND}" \
    --hotpotqa-max-new-tokens 32 \
    "${HOTPOTQA_OFFICIAL_DECODING_FLAG[@]}" \
    --gpqa-max-new-tokens 8192 \
    "${GPQA_OFFICIAL_DECODING_FLAG[@]}" \
    --ifeval-max-new-tokens 1500 \
    --memory-agent-bench-max-new-tokens 4096 \
    --memory-agent-bench-eval-batch-size "${MEMORY_AGENT_BENCH_EVAL_BATCH_SIZE}" \
    --memory-agent-bench-max-context-chars 120000 \
    --no-memory-agent-bench-use-official-prompt \
    --eval-do-sample \
    --eval-temperature 0.4 \
    --eval-top-p 0.9 \
    --eval-top-k 10 \
    --local-files-only \
    --skip-base \
    --skip-lora \
    --output-json "${output_json}"
}

cat > "${MANIFEST_PATH}" <<EOF
SUITE_ROOT=${SUITE_ROOT}
HF_HOME=${HF_HOME}
HF_HUB_CACHE=${HF_HUB_CACHE}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
MEMORY_AGENT_BENCH_ROOT=${MEMORY_AGENT_BENCH_ROOT}
BASE_MODEL_PATH=${BASE_MODEL_PATH}
ACTIVE_BENCHMARK_VARIANTS=${BENCHMARK_VARIANTS[*]}
BENCHMARK_VARIANTS_STRING=${BENCHMARK_VARIANTS_STRING}
SSW_ADAPTER_DIR=${SSW_ADAPTER_DIR}
TSW_ADAPTER_DIR=${TSW_ADAPTER_DIR}
MSW_ADAPTER_DIR=${MSW_ADAPTER_DIR}
LOCOMO_DATA_FILE=${LOCOMO_DATA_FILE}
LOCOMO_FULL_HISTORY_MODE=official_prompt
EVAL_TASKS=${EVAL_TASKS[*]}
EVAL_TASKS_STRING=${EVAL_TASKS_STRING}
BENCHMARK_TEMPERATURE=0.4
BENCHMARK_TOP_P=0.9
BENCHMARK_TOP_K=10
HOTPOTQA_OFFICIAL_DECODING=${HOTPOTQA_OFFICIAL_DECODING}
GPQA_OFFICIAL_DECODING=${GPQA_OFFICIAL_DECODING}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}
BASE_INFERENCE_BACKEND=${BASE_INFERENCE_BACKEND}
BENCHMARK_LOCAL_FILES_ONLY_ENV=1
EOF

echo "Wrote manifest: ${MANIFEST_PATH}"

run_eval_task_for_all_variants() {
  local task_name="$1"
  case "${task_name}" in
    locomo)
      variant_enabled "base_model" && run_locomo_base_like "base_model" "30071"
      variant_enabled "SSW_rank8_qasper_write8192" && run_locomo_delta_only "SSW_rank8_qasper_write8192" "${SSW_ADAPTER_DIR}" "30072"
      variant_enabled "TSW_rank8_qasper_write8192" && run_locomo_delta_only "TSW_rank8_qasper_write8192" "${TSW_ADAPTER_DIR}" "30073"
      variant_enabled "MSW_qasper_write8192" && run_locomo_delta_only "MSW_qasper_write8192" "${MSW_ADAPTER_DIR}" "30074"
      ;;
    hotpotqa|gpqa_diamond|ifeval|memory_agent_bench)
      variant_enabled "base_model" && run_benchmark_task_base_like "base_model" "${task_name}" "30171"
      variant_enabled "SSW_rank8_qasper_write8192" && run_benchmark_task_delta_only "SSW_rank8_qasper_write8192" "${SSW_ADAPTER_DIR}" "${task_name}" "30172"
      variant_enabled "TSW_rank8_qasper_write8192" && run_benchmark_task_delta_only "TSW_rank8_qasper_write8192" "${TSW_ADAPTER_DIR}" "${task_name}" "30173"
      variant_enabled "MSW_qasper_write8192" && run_benchmark_task_delta_only "MSW_qasper_write8192" "${MSW_ADAPTER_DIR}" "${task_name}" "30174"
      ;;
    *)
      echo "Unsupported EVAL_TASK: ${task_name}" >&2
      exit 1
      ;;
  esac
  return 0
}

for task_name in "${EVAL_TASKS[@]}"; do
  run_eval_task_for_all_variants "${task_name}"
done

echo "Done: ${SUITE_ROOT}"
