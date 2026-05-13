<h1 align="center">
    δ-mem: Efficient Online Memory for Large Language Models
</h1>

<p align="center">
<a href="https://creativecommons.org/licenses/by/4.0/">
    <img alt="License: CC-BY-4.0" src="https://img.shields.io/badge/License-CC_BY_4.0-brightgreen.svg">
</a>
<a href="https://arxiv.org/abs/2605.12357">
     <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white">
</a>
<a href="https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct">
    <img alt="Huggingface" src="https://img.shields.io/badge/🤗_Huggingface-Model-ff9800.svg">
</a>
</p>

δ-mem introduces a compact Online State of Associative Memory alongside a frozen full-attention backbone. When a new token or interaction segment arrives, the model projects the current information into a low-dimensional memory space and writes it into the state through delta-rule learning.

This repository contains the main δ-mem implementation, training scripts, evaluation scripts, and an interactive chat demo. The current public release focuses on Qwen3-4B/8B and SmolLM3-3B experiments with three write strategies: TSW, SSW, and MSW.

## Why δ-mem?

In long-term agent scenarios, what is truly needed is a more efficient memory mechanism. Such a mechanism should not endlessly increase the context burden like full-text retrieval, nor should it behave like static parametric memory that becomes fixed after training. Instead, it should be able to update dynamically during interaction and directly influence the model’s internal computation during inference. Motivated by this, we propose **δ-mem**, a lightweight online memory mechanism for large language models.


## Released Model

| Model | Base model | Adapter | Hugging Face |
| --- | --- | --- | --- |
| δ-mem Qwen3-4B Instruct TSW | `Qwen/Qwen3-4B-Instruct-2507` | rank-8 Q/O TSW, write length 8192 | [`declare-lab/delta-mem_qwen3_4b-instruct`](https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct) |

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

δ-mem adapters are not standard PEFT LoRA adapters and are not merged into the base model with `merge_and_unload()`. The runtime memory read/write path is part of the model execution.

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

Run the base model without δ-mem:

```bash
MODE=base MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
bash deltamem/demo/run_chat_demo.sh
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

## Citation
If you find our work is usefule, please kindly cite:

```bibtex
@misc{lei2026deltamemefficientonlinememory,
      title={$\delta$-mem: Efficient Online Memory for Large Language Models}, 
      author={Jingdi Lei and Di Zhang and Junxian Li and Weida Wang and Kaixuan Fan and Xiang Liu and Qihan Liu and Xiaoteng Ma and Baian Chen and Soujanya Poria},
      year={2026},
      eprint={2605.12357},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.12357}, 
}
```
