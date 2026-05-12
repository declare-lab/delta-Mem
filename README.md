# Delta-Mem

Minimal code release for Delta-Mem training, evaluation, and chat demo.

## Setup

```bash
cd /path/to/Delta-Mem
bash scripts/setup_uv_env.sh
```

The scripts use placeholder paths under `/root/...` for models, datasets,
caches, outputs, and external benchmark code. Before running experiments,
edit the paths in the corresponding `.sh` script or override them with
environment variables.

## Chat Demo

```bash
bash deltamem/demo/run_chat_demo.sh
```

Useful overrides:

```bash
MODEL_PATH=/root/huggingface/hub/model-snapshot \
ADAPTER_DIR=/root/models/delta-mem-adapter/trainer/checkpoint-70 \
bash deltamem/demo/run_chat_demo.sh
```

Run without an adapter:

```bash
MODE=base MODEL_PATH=/root/huggingface/hub/model-snapshot \
bash deltamem/demo/run_chat_demo.sh
```

## Training And Evaluation

Qwen3-4B train + benchmark:

```bash
bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite.sh
```

Benchmark only:

```bash
bash scripts/run_qasper_multimodel_write8192_benchmark_suite.sh
```

Model-specific scripts are also provided for Qwen3-8B and SmolLM3-3B.

## Quick Checks

```bash
PYTHONPATH=. .venv/bin/python -m compileall -q deltamem
PYTHONPATH=. .venv/bin/python -m pytest -q deltamem/tests
```
