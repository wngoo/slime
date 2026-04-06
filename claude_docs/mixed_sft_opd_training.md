# Mixed SFT + On-Policy Distillation (OPD) Training

하나의 배치 내에서 SFT 샘플과 OPD 샘플을 동시에 학습하는 방법.
`examples/on_policy_distillation/run-qwen3-8B-opd.sh` 기반.

---

## 목표

- **OPD 샘플**: teacher 모델의 분포를 모방 (policy loss + teacher KL)
- **SFT 샘플**: 정답 response를 완벽하게 모방 (NLL loss = `-log_prob`)
- 하나의 배치에 두 타입이 섞여 있고, custom loss function에서 분기 처리

---

## 데이터 포맷

```jsonl
// OPD 샘플: rollout으로 response 생성 후 teacher 지도
{"prompt": "수학 문제 풀어줘", "type": "opd"}

// SFT 샘플: data의 response를 그대로 학습 (generation 없음)
{"prompt": "간단히 인사해줘", "response": "안녕하세요!", "type": "sft"}
```

---

## 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 데이터 로딩                                                  │
│    {"type": "sft", "response": "..."} → SFT 샘플               │
│    {"type": "opd"}                    → OPD 샘플               │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Rollout (SGLang)                                             │
│    SFT: data["response"] 그대로 사용 (generation 없음)          │
│    OPD: 모델이 response 생성                                    │
│    → custom rollout function 필요 (--rollout-function-path)     │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Reward (Teacher 서버)                                        │
│    SFT: teacher 호출 X                                          │
│         sample.teacher_log_probs = None                         │
│         sample.train_metadata = {"type": "sft"}                 │
│    OPD: teacher 호출 O                                          │
│         sample.teacher_log_probs = [t1, t2, ...]                │
│         sample.train_metadata = {"type": "opd"}                 │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. rollout.py: train_data 조립 (rollout.py:718-719)             │
│    batch["metadata"]         = [{"type": "sft"}, ...]           │
│    batch["teacher_log_probs"] = [None, tensor, ...]             │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Advantage 계산 (GRPO)                                        │
│    SFT: GRPO advantage 계산됨 → loss에서 무시됨                 │
│    OPD: GRPO advantage 계산 (reward=0이므로 0에 수렴)           │
│    ※ --use-opd 제거 → apply_opd_kl_to_advantages 실행 안 됨    │
│       OPD KL은 custom loss 내부에서 직접 계산                   │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Custom Loss: mixed_sft_opd_loss                              │
│                                                                 │
│    전체 배치 log_probs 계산 (get_log_probs_and_entropy 1회 호출) │
│                                                                 │
│    type == "sft"                                                │
│      → sft_loss = -mean(log_prob)        ← 완벽한 NLL          │
│                                                                 │
│    type == "opd"                                                │
│      → opd_loss = policy_loss(grpo_adv)                         │
│                 + opd_kl_coef × KL(student || teacher)          │
│                                                                 │
│    total_loss = sft_loss + opd_loss → gradient update           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 구현 파일 목록

| 파일 | 역할 |
|---|---|
| `slime/rollout/mixed_reward.py` | SFT/OPD 분기 reward 함수 |
| `slime/rollout/mixed_loss.py` | custom loss 함수 |
| custom rollout function | SFT 샘플 generation 스킵 |

---

## 구현 코드

### 1. `slime/rollout/mixed_reward.py`

```python
import aiohttp
import torch
from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    sample_type = sample.metadata.get("type", "opd")

    if sample_type == "sft":
        return None  # teacher 호출 안 함

    # OPD: teacher log prob 요청
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    for i, sample in enumerate(samples):
        sample_type = sample.metadata.get("type", "opd")
        sample.train_metadata = {"type": sample_type}

        if sample_type == "sft":
            sample.teacher_log_probs = None
        else:
            reward = raw_rewards[i]
            t_log_probs = torch.tensor(
                [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
                dtype=torch.float32,
            )
            sample.teacher_log_probs = t_log_probs[-sample.response_length:]

    return [0.0] * len(samples), [0.0] * len(samples)
```

### 2. `slime/rollout/mixed_loss.py`

```python
import torch
from argparse import Namespace

from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.utils.ppo_utils import compute_policy_loss
from slime.utils.types import RolloutBatch
from collections.abc import Callable


def mixed_sft_opd_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SFT + OPD 혼합 배치를 처리하는 custom loss 함수.

    batch["metadata"][i]["type"] == "sft" → NLL loss (완벽한 SFT)
    batch["metadata"][i]["type"] == "opd" → policy loss + teacher KL (OPD)
    """
    metadata = batch.get("metadata", [{"type": "opd"}] * len(batch["response_lengths"]))

    # 1. 전체 배치 log_probs 한 번만 계산
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    log_probs = log_probs_and_entropy["log_probs"]  # list[Tensor], 샘플별 response log probs

    # 2. 타입별 인덱스 분류
    sft_idx = [i for i, m in enumerate(metadata) if m.get("type") == "sft"]
    opd_idx = [i for i, m in enumerate(metadata) if m.get("type") != "sft"]

    total_loss = torch.tensor(0.0, device=logits.device)
    log = {}

    # 3. SFT Loss: 완벽한 NLL (-log_prob)
    if sft_idx:
        max_seq_lens = batch.get("max_seq_lens")
        sft_sum_fn = get_sum_of_sample_mean(
            [batch["total_lengths"][i] for i in sft_idx],
            [batch["response_lengths"][i] for i in sft_idx],
            [batch["loss_masks"][i] for i in sft_idx],
            args.calculate_per_token_loss,
            args.qkv_format,
            [max_seq_lens[i] for i in sft_idx] if max_seq_lens else None,
        )
        sft_log_probs = torch.cat([log_probs[i] for i in sft_idx], dim=0)
        sft_loss = -sft_sum_fn(sft_log_probs)

        # gradient backprop 보장
        if sft_log_probs.numel() == 0:
            sft_loss = sft_loss + 0 * logits.sum()

        total_loss = total_loss + sft_loss
        log["sft_loss"] = sft_loss.clone().detach()

    # 4. OPD Loss: policy loss + teacher KL
    if opd_idx:
        max_seq_lens = batch.get("max_seq_lens")
        opd_sum_fn = get_sum_of_sample_mean(
            [batch["total_lengths"][i] for i in opd_idx],
            [batch["response_lengths"][i] for i in opd_idx],
            [batch["loss_masks"][i] for i in opd_idx],
            args.calculate_per_token_loss,
            args.qkv_format,
            [max_seq_lens[i] for i in opd_idx] if max_seq_lens else None,
        )

        old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

        opd_new_lp = torch.cat([log_probs[i] for i in opd_idx], dim=0)
        opd_old_lp = torch.cat([old_log_probs[i] for i in opd_idx], dim=0)
        opd_advantages = torch.cat([batch["advantages"][i] for i in opd_idx], dim=0)

        # PPO-style policy gradient loss
        ppo_kl = opd_old_lp - opd_new_lp
        pg_loss, _ = compute_policy_loss(ppo_kl, opd_advantages, args.eps_clip, args.eps_clip_high)

        # Teacher KL: reverse KL(student || teacher) per token
        teacher_log_probs = batch.get("teacher_log_probs", [])
        opd_teacher_lp = torch.cat([teacher_log_probs[i] for i in opd_idx], dim=0)
        kl_loss = opd_new_lp - opd_teacher_lp  # student - teacher log prob

        opd_loss = opd_sum_fn(pg_loss + args.opd_kl_coef * kl_loss)
        total_loss = total_loss + opd_loss
        log["opd_loss"] = opd_loss.clone().detach()

    return total_loss, log
```

---

## 실행 스크립트 수정 (`run-qwen3-8B-opd.sh`)

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-opd 제거: OPD KL을 custom loss에서 직접 처리하므로 불필요
   # --opd-type sglang       # 제거
   # --opd-kl-coef 1.0       # args.opd_kl_coef로 여전히 접근 가능하지만
   --opd-kl-coef 1.0         # custom loss에서 직접 사용하려면 유지
   --use-kl-loss
   --kl-loss-coef 0.00
   --loss-type custom_loss
   --custom-loss-function-path slime.rollout.mixed_loss.mixed_sft_opd_loss
)

RM_ARGS=(
   --custom-rm-path slime.rollout.mixed_reward.reward_func
   --custom-reward-post-process-path slime.rollout.mixed_reward.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)
```

---

## 핵심 설계 결정 사항

### `--use-opd` 제거 이유

`--use-opd`를 유지하면 `apply_opd_kl_to_advantages` (`loss.py:359`)가 실행되어
모든 샘플에 대해 `teacher_log_probs`를 요구한다. SFT 샘플은 teacher를 호출하지 않으므로
`teacher_log_probs = None`이 되어 에러 발생.

→ `--use-opd` 제거 후 OPD KL을 custom loss 내부에서 직접 계산하면
core 코드 수정 없이 해결 가능.

### SFT loss가 완벽한 NLL인 이유

```python
sft_loss = -sft_sum_fn(log_probs)
         = -mean(log P(token | context))  per sample
```

policy_loss의 advantage=1 근사와 달리, ratio(=new/old)를 전혀 사용하지 않고
**현재 모델의 log_prob를 직접 최대화**하므로 완벽한 SFT NLL과 동일.

### SFT 샘플의 response 처리

SFT 샘플은 모델이 response를 생성하지 않고 `data["response"]`를 토큰화해서 사용.
이를 위해 `--rollout-function-path`로 custom rollout function이 필요하며,
`type == "sft"` 샘플은 SGLang generation을 스킵하고 고정 response를 사용.
→ `add-rollout-function` 스킬 참고.

---

## 관련 코드 위치

| 내용 | 파일 | 라인 |
|---|---|---|
| `apply_opd_kl_to_advantages` | `slime/backends/megatron_utils/loss.py` | 359 |
| `sft_loss_function` | `slime/backends/megatron_utils/loss.py` | 892 |
| `policy_loss_function` | `slime/backends/megatron_utils/loss.py` | 613 |
| `loss_function` 디스패처 | `slime/backends/megatron_utils/loss.py` | 943 |
| `compute_policy_loss` | `slime/utils/ppo_utils.py` | 125 |
| `get_sum_of_sample_mean` | `slime/backends/megatron_utils/cp_utils.py` | 53 |
| `train_metadata → batch["metadata"]` | `slime/ray/rollout.py` | 718 |
| `teacher_log_probs → batch` | `slime/ray/rollout.py` | 724 |
| `Sample.train_metadata` | `slime/utils/types.py` | 46 |
