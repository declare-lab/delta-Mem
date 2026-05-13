# Delta-Mem: Online Test-Time Memory for Long-Context LLM Agents

[![GitHub](https://img.shields.io/badge/GitHub-declare--lab%2Fdelta--Mem-blue)](https://github.com/declare-lab/delta-Mem)
[![Model](https://img.shields.io/badge/HuggingFace-delta--mem__qwen3__4b--instruct-yellow)](https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct)

Delta-Mem is a lightweight memory-augmented adaptation framework for long-context language-model agents. It adds a small trainable online memory module to selected attention layers, allowing a frozen base model to write information from earlier context and read it back during generation.

This repository contains the main Delta-Mem implementation, training scripts, evaluation scripts, and an interactive chat demo. The current public release focuses on Qwen3-4B/8B and SmolLM3-3B experiments with three write strategies: TSW, SSW, and MSW.

## Why Delta-Mem?

Long-context agents often need to reuse information from long histories, documents, or multi-session conversations. Standard approaches either keep all context in the prompt, which is expensive, or fine-tune model weights, which does not provide an explicit online memory state.

Delta-Mem is designed for a narrower goal:

- Keep the base LLM frozen.
- Add a compact adapter-style memory module.
- Write context into an online state during prefill.
- Read from that state during answer generation.
- Evaluate the same mechanism on long-context QA, instruction following, and memory-agent benchmarks.

## Core Idea

Delta-Mem wraps selected self-attention modules with a trainable memory path. During a write phase, hidden states update a low-rank memory state. During a read phase, the model uses that state to produce a delta over attention outputs.

The released mainline supports:

| Name | Meaning | Write granularity | Typical use |
| --- | --- | --- | --- |
| TSW | Token State Write | `token` | Dense token-level writes |
| SSW | Segment State Write | `message_mean` or `sentence_mean` | Coarser message/sentence writes |
| MSW | Multi-head State Write | `token` with multiple state heads | Multi-state token writes |

The best public Qwen3-4B checkpoint in this release is the TSW adapter.

## Released Model

| Model | Base model | Adapter | Hugging Face |
| --- | --- | --- | --- |
| Delta-Mem Qwen3-4B Instruct TSW | `Qwen/Qwen3-4B-Instruct-2507` | rank-8 Q/O TSW, write length 8192 | [`declare-lab/delta-mem_qwen3_4b-instruct`](https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct) |

The Hugging Face repository is an adapter repository, not a fully merged base model. To use it, load the base model first, attach Delta-Mem modules, then load the adapter weights.

Expected adapter files:

```text
delta_mem_config.json
delta_mem_adapter.pt
```

## What Is In This Repository?

```text
Delta-Mem/
├── data/
│   └── locomo10.json                     # local LoCoMo sample file used by scripts
├── deltamem/
│   ├── core/                             # Delta-Mem modules, config, adapter save/load
│   ├── demo/                             # interactive chat demo
│   ├── eval/                             # LoCoMo, HotpotQA, IFEval, GPQA, MemoryAgentBench
│   ├── kernels/                          # affine scan kernel wrapper
│   ├── runtime/                          # chat/session runtime
│   ├── tests/                            # regression tests
│   ├── tools/                            # TPS and inspection tools
│   └── train/                            # SFT training code
├── scripts/
│   ├── setup_uv_env.sh
│   ├── run_qasper_multimodel_write8192_train_and_benchmark_suite.sh
│   ├── run_qasper_multimodel_write8192_benchmark_suite.sh
│   ├── run_qasper_multimodel_write8192_*_qwen3_8b.sh
│   ├── run_qasper_multimodel_write8192_*_smollm3_3b.sh
│   └── run_generation_tps_benchmark.sh
└── deepspeed_zero2.json
```

## Environment Setup

### System Requirements

Recommended setup:

| Component | Recommendation |
| --- | --- |
| Python | 3.10 or newer |
| GPU | NVIDIA GPU for training/evaluation |
| CUDA/PyTorch | A CUDA-enabled PyTorch build matching your driver |
| Package manager | `uv` |

The training scripts are designed for bf16 GPU runs and use FlashAttention and DeepSpeed. CPU-only usage is not the target path for this release.

### One-Command Setup

Clone the repository and run the setup script:

```bash
git clone https://github.com/declare-lab/delta-Mem.git
cd delta-Mem
bash scripts/setup_uv_env.sh
```

The script creates a fresh `.venv/`, installs `requirements.txt`, installs FlashAttention with `--no-build-isolation`, and prints a short import/CUDA diagnostic at the end.

If `uv` is not installed:

```bash
python -m pip install uv
```

Activate the environment:

```bash
source .venv/bin/activate
```

### Setup Options

Use a specific Python executable:

```bash
PYTHON_BIN=python3.11 bash scripts/setup_uv_env.sh
```

Keep an existing `.venv/` instead of recreating it:

```bash
KEEP_VENV=1 bash scripts/setup_uv_env.sh
```

Skip FlashAttention reinstall if your cluster already provides a working build:

```bash
INSTALL_FLASH_ATTN=0 bash scripts/setup_uv_env.sh
```

### Manual Setup

If you prefer to manage the environment yourself:

```bash
python -m pip install uv
uv venv --python python3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -r requirements.txt
uv pip install --no-build-isolation flash-attn
```

If PyTorch needs to be installed from a specific CUDA index, install it before the requirements, for example:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

### Verify The Environment

Run:

```bash
python - <<'PY'
import torch, transformers, datasets, accelerate, deepspeed, flash_attn, peft
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("deepspeed:", deepspeed.__version__)
print("flash_attn:", flash_attn.__file__)
print("peft:", peft.__version__)
PY
```

Then run the local checks:

```bash
PYTHONPATH=. python -m compileall -q deltamem
PYTHONPATH=. python -m pytest -q deltamem/tests
```

## Path Configuration

The experiment scripts intentionally use placeholder paths under `/root/...`:

```text
/root/huggingface
/root/models
/root/data
/root/outputs
/root/external/MemoryAgentBench
```

Before running training or evaluation, either edit the script variables or override them from the shell:

```bash
BASE_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
TSW_ADAPTER_DIR=/path/to/delta-mem-adapter \
SUITE_ROOT=/path/to/results \
bash scripts/run_qasper_multimodel_write8192_benchmark_suite.sh
```

## Use The Released Adapter

Download the adapter from Hugging Face:

```bash
huggingface-cli download declare-lab/delta-mem_qwen3_4b-instruct \
  --local-dir ./delta-mem_qwen3_4b-instruct
```

Minimal loading example:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.core import HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter

base_model = "Qwen/Qwen3-4B-Instruct-2507"
adapter_dir = "./delta-mem_qwen3_4b-instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

config = HFDeltaMemConfig.from_pretrained(adapter_dir)
attach_delta_mem(model, config)
load_delta_mem_adapter(model, adapter_dir)
model.eval()
```

Delta-Mem adapters are not standard PEFT LoRA adapters and are not merged into the base model with `merge_and_unload()`. The runtime memory read/write path is part of the model execution.

## Chat Demo

Run the default shell wrapper:

```bash
bash deltamem/demo/run_chat_demo.sh
```

Typical override:

```bash
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
ADAPTER_DIR=/path/to/delta-mem_qwen3_4b-instruct \
bash deltamem/demo/run_chat_demo.sh
```

Run the base model without Delta-Mem:

```bash
MODE=base MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
bash deltamem/demo/run_chat_demo.sh
```

The Python entry point is:

```bash
PYTHONPATH=. .venv/bin/python -m deltamem.demo.chat_demo \
  --mode delta \
  --model-path /path/to/Qwen3-4B-Instruct-2507 \
  --adapter-dir /path/to/delta-mem_qwen3_4b-instruct \
  --device cuda:0 \
  --dtype bfloat16
```

## Training

The main Qwen3-4B training script trains SSW, TSW, and MSW variants by default:

```bash
bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite.sh
```

Run only TSW:

```bash
TRAIN_VARIANTS_STRING="TSW_rank8_qasper_write8192" \
BENCHMARK_VARIANTS_STRING="TSW_rank8_qasper_write8192" \
bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite.sh
```

Model-specific scripts:

```bash
bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite_qwen3_8b.sh
bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite_smollm3_3b.sh
```

Important default training settings in the public scripts:

| Setting | Default |
| --- | --- |
| Training mode | `episode` |
| Assistant loss | final assistant response only |
| Recent visible messages | `--episode-recent-messages 1` |
| Write length | `--max-write-length 8192` |
| Read sequence length | `--max-length 512` |
| Optimizer | `adamw_torch_fused` |
| Precision | bf16 |
| DeepSpeed | ZeRO-2 config in `deepspeed_zero2.json` |

The scripts include a note about the `both` and `state_only` views: with `episode_recent_messages=1`, both views use the same practical context split in the current setup, so the 0.8/0.2 sampling behaves like stochastic resampling of nearly equivalent views rather than two separately weighted losses in one step.

## Evaluation

The main benchmark suite covers:

| Benchmark | Entry |
| --- | --- |
| LoCoMo | `deltamem.eval.locomo_delta` |
| HotpotQA | `deltamem.eval.benchmark_compare --tasks hotpotqa` |
| IFEval | `deltamem.eval.benchmark_compare --tasks ifeval` |
| GPQA Diamond | `deltamem.eval.benchmark_compare --tasks gpqa_diamond` |
| MemoryAgentBench | `deltamem.eval.benchmark_compare --tasks memory_agent_bench` |

Run the bundled Qwen3-4B benchmark suite:

```bash
bash scripts/run_qasper_multimodel_write8192_benchmark_suite.sh
```

Run only the TSW adapter and skip base-model evaluation:

```bash
BENCHMARK_VARIANTS_STRING="TSW_rank8_qasper_write8192" \
EVAL_TASKS_STRING="locomo hotpotqa gpqa_diamond ifeval memory_agent_bench" \
bash scripts/run_qasper_multimodel_write8192_benchmark_suite.sh
```

Direct benchmark entry point:

```bash
PYTHONPATH=. .venv/bin/python -m deltamem.eval.benchmark_compare \
  --model-path /path/to/Qwen3-4B-Instruct-2507 \
  --delta-adapter-dir /path/to/delta-mem_qwen3_4b-instruct \
  --skip-base \
  --skip-lora \
  --tasks hotpotqa ifeval gpqa_diamond memory_agent_bench \
  --output-json outputs/qwen3_4b_tsw_benchmarks.json
```

Direct LoCoMo entry point:

```bash
PYTHONPATH=. .venv/bin/python -m deltamem.eval.locomo_delta \
  --model-path /path/to/Qwen3-4B-Instruct-2507 \
  --adapter-dir /path/to/delta-mem_qwen3_4b-instruct \
  --data-file data/locomo10.json \
  --skip-base \
  --output-json outputs/qwen3_4b_tsw_locomo.json
```

MemoryAgentBench evaluation expects a local checkout of the official MemoryAgentBench repository when official prompts/metrics are used. Set:

```bash
MEMORY_AGENT_BENCH_ROOT=/path/to/MemoryAgentBench
```

## Generation And Prompt Defaults

The benchmark code records generation settings in the output JSON. Important defaults:

| Task | Default max new tokens | Default decoding |
| --- | ---: | --- |
| LoCoMo answer generation | official LoCoMo max token setting | greedy unless overridden by LoCoMo-specific flags |
| HotpotQA | 32 | shared deterministic generation, or official greedy with `--hotpotqa-official-decoding` |
| IFEval | 1500 | shared deterministic generation |
| GPQA Diamond | 8192 | shared deterministic generation, or official greedy with `--gpqa-official-decoding` |
| MemoryAgentBench | 4096, or official per-source lengths | official prompt path by default |

For MemoryAgentBench, the code supports `--memory-agent-bench-use-official-prompt` and `--no-memory-agent-bench-use-official-prompt`. The released scripts use the configured benchmark path in `scripts/`.

## Speed Benchmark

Token generation throughput can be measured with:

```bash
bash scripts/run_generation_tps_benchmark.sh
```

This script supports base and Delta-Mem modes through `MODEL_KINDS`, `MODEL_PATH`, and `ADAPTER_DIR`.

## Tests

Run the local regression tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q deltamem/tests
```

Compile check:

```bash
PYTHONPATH=. .venv/bin/python -m compileall -q deltamem
```

## Notes On Adapter Files

Older internal checkpoints may use legacy filenames:

```text
delta_lora_adapter.pt
delta_lora_config.json
```

Delta-Mem mainline expects:

```text
delta_mem_adapter.pt
delta_mem_config.json
```

For public release, rename or upload the legacy files under the `delta_mem_*` names. The released Hugging Face adapter is expected to use the `delta_mem_*` filenames.

## License

TODO: add the final project license.

## Citation

TODO: add citation once the paper/preprint is public.
