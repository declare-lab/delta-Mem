#!/bin/bash

set -euo pipefail

# Training view note:
# This script sets --episode-recent-messages 1 and --memory-dropout-state-only-prob 0.2.
# With episode_recent_messages=1, the "both" view and the sampled "state_only"
# view use the same context split in practice: write all earlier turns into Delta
# state and keep only the latest non-system turn visible while reading the target.
# The 0.8/0.2 behavior is therefore stochastic resampling of an almost equivalent
# input view, not two separately weighted losses in one step.

# Paths
ROOT_DIR="/root/code/Delta-Mem"
PYTHON_BIN="/root/code/Delta-Mem/.venv/bin/python"
PIPELINE_ROOT="/vePFS-Mindverse/share/models/leijingdi/results/qasper_multimodel_write8192_train_and_benchmark_suite_qwen3_8b_consistent70"
LOG_ROOT="/vePFS-Mindverse/share/models/leijingdi/results/qasper_multimodel_write8192_train_and_benchmark_suite_qwen3_8b_consistent70/logs"
MANIFEST_PATH="/vePFS-Mindverse/share/models/leijingdi/results/qasper_multimodel_write8192_train_and_benchmark_suite_qwen3_8b_consistent70/run_manifest.txt"
EVAL_SCRIPT="/root/code/Delta-Mem/scripts/run_qasper_multimodel_write8192_benchmark_suite_qwen3_8b.sh"
WANDB_DIR="/vePFS-Mindverse/share/models/leijingdi/wandb"
HF_HOME="/vePFS-Mindverse/share/huggingface"
HF_HUB_CACHE="/vePFS-Mindverse/share/huggingface/hub"
HF_DATASETS_CACHE="/vePFS-Mindverse/share/datasets_cache"
TOKENIZED_DATASET_ROOT="/vePFS-Mindverse/share/dataset/deltamem_tokenized"
DEEPSPEED_CONFIG="/root/code/Delta-Mem/deepspeed_zero2.json"
BASE_MODEL_PATH="/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
TRAIN_FILE="/vePFS-Mindverse/share/models/leijingdi/data/agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl"

# Runtime
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
WANDB_PROJECT="delta-mem-qwen3"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"

DELTA_WRITE_SPARSITY_WEIGHT="0.01"
DELTA_WRITE_SPARSITY_TARGET="0.05"

# Variant-specific delta hyperparameters
SSW_RUN_NAME="qwen3_8b_delta_mem_qasper_6k_seed42_rank8_qo_SSW_write8192_consistent70"
SSW_OUTPUT_DIR="/vePFS-Mindverse/share/models/leijingdi/models/qwen3_8b_delta_mem_qasper_6k_seed42_rank8_qo_SSW_write8192_consistent70"
SSW_NUM_STATE_HEADS="1"
SSW_WRITE_GRANULARITY="message_mean"
SSW_MASTER_PORT="29882"
SSW_WANDB_GROUP="qasper-multimodel-write8192-delta"
SSW_WANDB_TAGS="qasper,delta,SSW,write8192,consistent70"

TSW_RUN_NAME="qwen3_8b_delta_mem_qasper_6k_seed42_rank8_qo_TSW_write8192_consistent70"
TSW_OUTPUT_DIR="/vePFS-Mindverse/share/models/leijingdi/models/qwen3_8b_delta_mem_qasper_6k_seed42_rank8_qo_TSW_write8192_consistent70"
TSW_NUM_STATE_HEADS="1"
TSW_WRITE_GRANULARITY="token"
TSW_MASTER_PORT="29883"
TSW_WANDB_GROUP="qasper-multimodel-write8192-delta"
TSW_WANDB_TAGS="qasper,delta,TSW,write8192,consistent70"

MSW_RUN_NAME="qwen3_8b_delta_mem_qasper_6k_seed42_MSW_write8192_consistent70"
MSW_OUTPUT_DIR="/vePFS-Mindverse/share/models/leijingdi/models/qwen3_8b_delta_mem_qasper_6k_seed42_MSW_write8192_consistent70"
MSW_NUM_STATE_HEADS="4"
MSW_WRITE_GRANULARITY="token"
MSW_MASTER_PORT="29884"
MSW_WANDB_GROUP="qasper-multimodel-write8192-delta"
MSW_WANDB_TAGS="qasper,delta,MSW,write8192,consistent70"

DEFAULT_TRAIN_VARIANTS=(SSW_rank8_qasper_write8192 TSW_rank8_qasper_write8192 MSW_qasper_write8192)
TRAIN_VARIANTS_STRING="${TRAIN_VARIANTS_STRING:-${DEFAULT_TRAIN_VARIANTS[*]}}"
read -r -a TRAIN_VARIANTS <<< "${TRAIN_VARIANTS_STRING}"
if [[ ${#TRAIN_VARIANTS[@]} -eq 0 ]]; then
  echo "TRAIN_VARIANTS_STRING resolved to no variants" >&2
  exit 1
fi
for variant_slug in "${TRAIN_VARIANTS[@]}"; do
  case "${variant_slug}" in
    SSW_rank8_qasper_write8192|TSW_rank8_qasper_write8192|MSW_qasper_write8192) ;;
    *)
      echo "Unsupported TRAIN_VARIANTS_STRING entry: ${variant_slug}" >&2
      exit 1
      ;;
  esac
done

COMMON_TRAIN_ARGS=(
  --model-path "${BASE_MODEL_PATH}"
  --train-file "${TRAIN_FILE}"
  --hf-cache-dir "${HF_HOME}"
  --tokenized-dataset-root "${TOKENIZED_DATASET_ROOT}"
  --tokenized-cache
  --dtype bfloat16
  --bf16
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --training-mode episode
  --assistant-loss-mode final_assistant_only
  --episode-recent-messages 1
  --max-length 512
  --max-write-length 8192
  --per-device-train-batch-size 1
  --gradient-accumulation-steps 4
  --learning-rate 2e-4
  --seed 42
  --data-seed 42
  --lr-scheduler-type cosine
  --warmup-ratio 0.10
  --weight-decay 0.0
  --optim adamw_torch_fused
  --num-train-epochs 1.0
  --max-steps -1
  --logging-steps 1
  --save-steps 200
  --dataset-num-proc 16
  --dataloader-num-workers 8
  --tf32
  --ddp-backend nccl
  --deepspeed-config "${DEEPSPEED_CONFIG}"
)

COMMON_WANDB_ARGS=(
  --wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-dir "${WANDB_DIR}"
  --wandb-mode online
)

cd "${ROOT_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export WANDB_DIR="${WANDB_DIR}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}"

mkdir -p "${PIPELINE_ROOT}" "${LOG_ROOT}" "${WANDB_DIR}"

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

training_is_complete() {
  local output_dir="$1"
  [[ -d "${output_dir}/trainer/checkpoint-70" || -f "${output_dir}/training_summary.json" ]]
}

run_delta_train() {
  local log_file="$1"
  local output_dir="$2"
  local run_name="$3"
  local master_port="$4"
  local num_state_heads="$5"
  local write_granularity="$6"
  local wandb_group="$7"
  local wandb_tags="$8"

  if training_is_complete "${output_dir}"; then
    echo "Skip completed training: ${run_name}"
    return 0
  fi

  run_with_log "${log_file}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port "${master_port}" \
    -m deltamem.train.delta_sft_experimental \
    "${COMMON_TRAIN_ARGS[@]}" \
    --output-dir "${output_dir}" \
    --rank 8 \
    --alpha 16.0 \
    --num-state-heads "${num_state_heads}" \
    --beta-bias-init -1.5 \
    --couple-lambda \
    --state-update-mode standard \
    --output-init base_slice_fixed \
    --base-slice-ref-width 8 \
    --delta-heads q,o \
    --online-gain 0.05 \
    --rankwise-gates \
    --target-layers off \
    --memory-readout-mode delta \
    --memory-write-source learned_hidden \
    --memory-write-granularity "${write_granularity}" \
    --memory-contrast-weight 1.0 \
    --memory-kl-weight 0.02 \
    --memory-margin 0.01 \
    --memory-causal-weight 1.0 \
    --memory-anchor-weight 1.0 \
    --memory-anchor-margin 0.005 \
    --memory-recover-weight 0.25 \
    --memory-need-floor 0.15 \
    --memory-dropout-state-only-prob 0.2 \
    --write-sparsity-weight "${DELTA_WRITE_SPARSITY_WEIGHT}" \
    --write-sparsity-target "${DELTA_WRITE_SPARSITY_TARGET}" \
    "${COMMON_WANDB_ARGS[@]}" \
    --wandb-run-name "${run_name}" \
    --wandb-group "${wandb_group}" \
    --wandb-tags "${wandb_tags}"
}

train_variant_enabled() {
  local target="$1"
  local variant_slug
  for variant_slug in "${TRAIN_VARIANTS[@]}"; do
    [[ "${variant_slug}" == "${target}" ]] && return 0
  done
  return 1
}

cat > "${MANIFEST_PATH}" <<EOF
PIPELINE_ROOT=${PIPELINE_ROOT}
HF_HOME=${HF_HOME}
HF_HUB_CACHE=${HF_HUB_CACHE}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
TRAIN_FILE=${TRAIN_FILE}
BASE_MODEL_PATH=${BASE_MODEL_PATH}
EVAL_SCRIPT=${EVAL_SCRIPT}
ACTIVE_TRAIN_VARIANTS=${TRAIN_VARIANTS_STRING}
TRAINING_MODE=episode
ASSISTANT_LOSS_MODE=final_assistant_only
EPISODE_RECENT_MESSAGES=1
MAX_LENGTH=512
MAX_WRITE_LENGTH=8192
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-4
LR_SCHEDULER_TYPE=cosine
WARMUP_RATIO=0.10
WEIGHT_DECAY=0.0
OPTIM=adamw_torch_fused
NUM_TRAIN_EPOCHS=1.0
MAX_STEPS=-1
LOGGING_STEPS=1
SAVE_STEPS=200
DATASET_NUM_PROC=16
DATALOADER_NUM_WORKERS=8
SSW_OUTPUT_DIR=${SSW_OUTPUT_DIR}
TSW_OUTPUT_DIR=${TSW_OUTPUT_DIR}
MSW_OUTPUT_DIR=${MSW_OUTPUT_DIR}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}
EOF

echo "Wrote manifest: ${MANIFEST_PATH}"

train_variant_enabled "SSW_rank8_qasper_write8192" && run_delta_train "${LOG_ROOT}/SSW_rank8_qasper_write8192.log" "${SSW_OUTPUT_DIR}" "${SSW_RUN_NAME}" "${SSW_MASTER_PORT}" "${SSW_NUM_STATE_HEADS}" "${SSW_WRITE_GRANULARITY}" "${SSW_WANDB_GROUP}" "${SSW_WANDB_TAGS}"
train_variant_enabled "TSW_rank8_qasper_write8192" && run_delta_train "${LOG_ROOT}/TSW_rank8_qasper_write8192.log" "${TSW_OUTPUT_DIR}" "${TSW_RUN_NAME}" "${TSW_MASTER_PORT}" "${TSW_NUM_STATE_HEADS}" "${TSW_WRITE_GRANULARITY}" "${TSW_WANDB_GROUP}" "${TSW_WANDB_TAGS}"
train_variant_enabled "MSW_qasper_write8192" && run_delta_train "${LOG_ROOT}/MSW_qasper_write8192.log" "${MSW_OUTPUT_DIR}" "${MSW_RUN_NAME}" "${MSW_MASTER_PORT}" "${MSW_NUM_STATE_HEADS}" "${MSW_WRITE_GRANULARITY}" "${MSW_WANDB_GROUP}" "${MSW_WANDB_TAGS}"
run_with_log "${LOG_ROOT}/eval_tasks.log" bash "${EVAL_SCRIPT}"

echo "Done: ${PIPELINE_ROOT}"
