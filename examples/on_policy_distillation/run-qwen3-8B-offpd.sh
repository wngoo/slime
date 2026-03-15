#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-offpd.sh
#
# Off-policy distillation: train Qwen3-8B (student) using rollout data (PT files)
# that were previously dumped from an on-policy run via --dump-details.
#
# Workflow:
#   Step 1 (on-policy dump) — run run-qwen3-8B-opd.sh with --dump-details added:
#     --dump-details /root/opd_dump
#   This writes per-rollout PT files to /root/opd_dump/rollout_data/{rollout_id}.pt
#   Each Sample in the PT file already contains teacher_log_probs collected from
#   the teacher server during the on-policy run.
#
#   Step 2 (off-policy training, this script) — load those PT files instead of
#   running rollout/teacher server:
#     --load-debug-rollout-data /root/opd_dump/rollout_data/{rollout_id}.pt
#
# Key differences from run-qwen3-8B-opd.sh:
#   - No teacher SGLang server is launched (teacher_log_probs come from PT files).
#   - Setting --load-debug-rollout-data automatically sets debug_train_only=True,
#     which skips SGLang engine initialization entirely (skip_sglang=True in parse_args).
#   - No --custom-rm-path / --custom-reward-post-process-path / --rm-url needed.
#   - No rollout-related args needed (prompt-data, temperature, n-samples, etc.).
#   - --rollout-num-gpus 0: no GPUs wasted on idle SGLang engines; all GPUs go to
#     Megatron training. Increase --actor-num-gpus-per-node accordingly.
#   - --num-rollout must equal the number of PT files available (rollout_ids 0..N-1).
#     Set it to match however many steps were dumped in Step 1.

set -ex

export PYTHONBUFFERED=16

source "/root/slime/scripts/models/qwen3-8B.sh"


CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load /root/Qwen3-8B_torch_dist
   --load /root/Qwen3-8B_slime_offpd/
   --save /root/Qwen3-8B_slime_offpd/
   --save-interval 20
)

OFFPD_ARGS=(
   # Path template for the PT dump files from the on-policy run.
   # {rollout_id} is substituted with 0, 1, 2, ... at runtime.
   --load-debug-rollout-data /root/opd_dump/rollout_data/{rollout_id}.pt

   # Must match the number of PT files available (rollout_ids 0 .. num_rollout-1).
   --num-rollout 300

   --global-batch-size 64
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-8B-offpd
   # --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)


# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 6 \
   --rollout-num-gpus 0 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${OFFPD_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}



#### clear after training
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
