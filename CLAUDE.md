# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**slime** is an LLM post-training framework for RL scaling that connects [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (distributed training) with [SGLang](https://github.com/sgl-project/sglang) (fast inference/rollout). It supports models like GLM-4/4.5/4.6/4.7/5, Qwen, DeepSeek V3/R1, and Llama 3 series.

## Commands

### Linting and Formatting

```bash
# Run pre-commit checks on all files
pre-commit run --all-files

# Run individual formatters
black slime/ slime_plugins/ --line-length 119
isort slime/ slime_plugins/ --profile black
ruff check slime/ slime_plugins/ --fix
autoflake --remove-all-unused-imports --in-place -r slime/ slime_plugins/
```

### Testing

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_<name>.py

# Run by marker
pytest -m unit
pytest -m "not skipduringci"

# Run a single test function
pytest tests/test_<name>.py::test_function_name
```

Test markers: `unit`, `integration`, `system`, `acceptance`, `docs`, `skipduringci`, `pleasefixme`.

### Training Entry Points

```bash
# Synchronous training (blocking ray.get() between steps)
python train.py <megatron + sglang + slime args>

# Asynchronous training (generation and training are pipelined)
python train_async.py <megatron + sglang + slime args>
```

Training arguments are defined in `slime/utils/arguments.py` (~79KB). This file integrates Megatron, SGLang, and slime-specific argument groups.

## Architecture

The framework is organized around three cooperating modules managed via Ray:

```
┌──────────────────────────────────────────────────┐
│  Training Module (Megatron)                      │
│  slime/backends/megatron_utils/                  │
│  - Reads batches from Data Buffer                │
│  - Syncs updated weights back to Rollout         │
└──────────────────┬───────────────────────────────┘
                   │ Ray RPC
┌──────────────────▼───────────────────────────────┐
│  Data Buffer (RolloutManager)                    │
│  slime/ray/rollout.py                            │
│  - Orchestrates prompt sampling and generation   │
│  - Central coordinator between train and rollout │
└──────────────────┬───────────────────────────────┘
                   │ Ray RPC
┌──────────────────▼───────────────────────────────┐
│  Rollout Module (SGLang + Router)                │
│  slime/rollout/sglang_rollout.py                 │
│  - Generates completions and computes rewards    │
│  - Stores results back into Data Buffer          │
└──────────────────────────────────────────────────┘
```

### Key Directories

- **`slime/ray/`** — Ray actor orchestration. `rollout.py` (RolloutManager, ~55KB) is the central coordinator; `train_actor.py` manages actor/critic models; `placement_group.py` handles GPU allocation.
- **`slime/rollout/`** — Data generation logic. `sglang_rollout.py` wraps SGLang inference. Sub-hubs: `generate_hub/` (pluggable generation functions), `rm_hub/` (reward models), `filter_hub/` (post-generation filtering).
- **`slime/backends/megatron_utils/`** — Megatron integration: `actor.py` (training step), `loss.py`, `ckpt.py` (checkpointing), kernel optimizations (FP8, INT4 QAT).
- **`slime/backends/megatron_to_hf/`** — Checkpoint converters from Megatron format to HuggingFace for each supported model family.
- **`slime/utils/`** — Shared utilities: `arguments.py` (arg parsing), `ppo_utils.py` (advantage computation), `data.py`, `eval_config.py`, `distributed_utils.py`.
- **`slime_plugins/`** — Plugin system for extending model bridges (`mbridge/`), Megatron bridges (`megatron_bridge/`), and rollout buffers.
- **`examples/`** — Reference training configurations for different scenarios (multi-agent, VLM, on-policy distillation, tool use, search).
- **`tests/`** — pytest test suite; `plugin_contracts/` validates plugin interface compliance.

### Extension Points

The framework is designed to be extended via:
- **Custom rollout functions** — `--rollout-function-path` (see skill: `add-rollout-function`)
- **Custom reward functions** — `--custom-rm-path` (see skill: `add-reward-function`)
- **Dynamic filters** — buffer filtering/masking hooks (see skill: `add-dynamic-filter`)
- **Eval dataset configs** — `--eval-config` / `--eval-prompt-data` (see skill: `add-eval-dataset-config`)

## Code Style

- Line length: **119** characters (black + isort)
- Python ≥ 3.10; targets 3.10/3.11/3.12
- Pre-commit enforces: ruff (linting), black (formatting), isort (imports), autoflake (unused imports)

## Contributing Guidelines

Per `CONTRIBUTING.md`, in-scope contributions are: bug fixes, general-purpose RL optimizations with clear benchmarks. Out of scope: large refactoring, design/abstraction proposals, unverifiable features, major Megatron modifications.
