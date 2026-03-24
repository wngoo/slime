# TensorBoard 손실 지표 설명

on-policy distillation 학습 (`examples/on_policy_distillation/run-qwen3-8B-opd.sh`) 시 TensorBoard에 기록되는 주요 손실 지표들의 계산 위치, 공식, 의미를 정리한 문서입니다.

## 지표 요약

| 지표 | 계산 위치 | 스크립트 역할 |
|------|-----------|--------------|
| `ppo_kl` | `loss.py:691-704, 766` | 학습 안정성 모니터링 |
| `pg_loss` | `loss.py:706`, `ppo_utils.py:125-148` | 실제 학습 손실 (핵심) |
| `kl_loss` | `loss.py:775-789`, `ppo_utils.py:12-51` | 모니터링만 (`coef=0.00`) |
| `entropy_loss` | `loss.py:768-771` | 모니터링만 (`coef=0.00`) |

---

## 1. `ppo_kl` — 현재 정책 vs 이전 정책 KL

### 계산 위치

`slime/backends/megatron_utils/loss.py:691-704, 766`

### 공식

```python
# 일반 경우 (GSPO 제외)
ppo_kl = old_log_probs - log_probs     # = log(π_old / π_new)
ppo_kl = sum_of_sample_mean(ppo_kl)   # 샘플별 평균 후 합산
```

GSPO advantage estimator 사용 시에는 `ppo_utils.py`의 `compute_gspo_kl()`을 통해 sequence-level KL을 계산한 뒤 per-token으로 확장합니다.

### 의미

- 현재 정책 `π_new`와 rollout 시점의 이전 정책 `π_old` 사이의 **per-token KL divergence 근사값**
- `ppo_kl > 0`: 현재 정책이 같은 토큰을 더 낮은 확률로 선택 → 정책이 크게 변화했음을 의미
- 값이 너무 크면 policy가 off-policy 영역으로 벗어난 것이므로 학습 안정성의 핵심 모니터링 지표

---

## 2. `pg_loss` — Policy Gradient (클리핑) 손실

### 계산 위치

- `slime/backends/megatron_utils/loss.py:706` — 호출부
- `slime/utils/ppo_utils.py:125-148` — `compute_policy_loss()` 구현

### 공식

```python
ratio = (-ppo_kl).exp()                                            # = π_new / π_old
pg_losses1 = -ratio * advantages                                   # 기본 PG 손실
pg_losses2 = -ratio.clamp(1-eps_clip, 1+eps_clip_high) * advantages  # 클리핑된 PG 손실
pg_loss = torch.maximum(pg_losses1, pg_losses2)                    # 보수적인 쪽 선택
```

수식: `-min(ratio · A, clip(ratio, 1±ε) · A)`

### 의미

- PPO의 **clipped surrogate objective** 손실
- `advantages > 0`(좋은 행동)일 때 `ratio`를 `1+ε` 이하로 제한해 과도한 업데이트 방지
- 값이 음수에 가까울수록 정책이 좋은 방향으로 개선되고 있음을 의미
- `run-qwen3-8B-opd.sh`는 `--advantage-estimator grpo`를 사용하므로 GRPO 방식으로 advantage를 계산한 뒤 이 손실에 적용

---

## 3. `kl_loss` — 참조 모델 대비 KL 손실

### 계산 위치

- `slime/backends/megatron_utils/loss.py:775-789` — 호출부
- `slime/utils/ppo_utils.py:12-51` — `compute_approx_kl()` 구현

### 공식

```python
log_ratio = log_probs.float() - ref_log_probs.float()  # log(π_new / π_ref)

# kl_loss_type에 따라 다른 공식 사용
# "low_var_kl" (스크립트 기본값, k3 근사):
log_ratio_neg = -log_ratio
kl = log_ratio_neg.exp() - 1 - log_ratio_neg           # exp(-r) - 1 + r
kl = torch.clamp(kl, min=-10, max=10)                  # 수치 안정성

kl_loss = sum_of_sample_mean(kl)
loss = loss + kl_loss_coef * kl_loss                   # 최종 손실에 가산
```

지원하는 `kl_loss_type`:
- `k1`: `log(π_new / π_ref)` (단순 log ratio)
- `k2`: `log(π_new / π_ref)² / 2`
- `k3` / `low_var_kl`: `exp(-r) - 1 + r` — 항상 ≥ 0, 분산이 낮은 단방향 KL

### 의미

- 현재 정책 `π_new`와 **참조(reference) 모델** `π_ref` 사이의 KL divergence 페널티
- 모델이 SFT 체크포인트에서 너무 멀리 벗어나지 않도록 정규화하는 역할
- `run-qwen3-8B-opd.sh`에서는 `--kl-loss-coef 0.00`이므로 계산 및 로깅은 하지만 **실제 학습에는 영향 없음** (모니터링 목적)

---

## 4. `entropy_loss` — 정책 엔트로피

### 계산 위치

`slime/backends/megatron_utils/loss.py:768-771`

### 공식

```python
entropy = log_probs_and_entropy["entropy"]   # 각 토큰의 엔트로피 H(π)
entropy = torch.cat(entropy, dim=0)
entropy_loss = sum_of_sample_mean(entropy)

# 최종 손실에서 차감 (엔트로피 보너스)
loss = loss - entropy_coef * entropy_loss
```

### 의미

- 정책 출력 분포의 **평균 엔트로피** (`H(π) = -Σ π·log π`)
- 엔트로피가 높을수록 모델이 다양한 토큰을 고루 고려함을 의미
- 손실에서 **빼주므로** 엔트로피를 높이는 방향으로 학습 유도 (탐색 장려)
- `run-qwen3-8B-opd.sh`에서는 `--entropy-coef 0.00`이므로 역시 모니터링만
- **주의:** 학습 중 엔트로피가 급격히 감소하면 특정 출력에 수렴(mode collapse)하고 있다는 경고 신호

---

## TensorBoard 기록 흐름

```
loss.py: policy_loss_function()
    └── reported_loss = {"pg_loss": ..., "kl_loss": ..., "entropy_loss": ..., "ppo_kl": ...}
         ↓
model.py: log_dict = {"train/pg_loss": ..., "train/kl_loss": ..., ...}
         ↓
logging_utils.py: log(args, log_dict, step_key="train/step")
         ↓
tensorboard_utils.py: writer.add_scalar(key, value, step)
```

TensorBoard 디렉토리는 환경변수 `TENSORBOARD_DIR` 또는 `tensorboard_log/{tb_project_name}/{tb_experiment_name}`으로 결정됩니다.

---

## OPD 스크립트 특이사항 (`run-qwen3-8B-opd.sh`)

`--use-opd --opd-kl-coef 1.0` 설정으로, **advantage에 teacher 모델과의 reverse KL 페널티를 추가**합니다.

```python
# loss.py:359-397, apply_opd_kl_to_advantages()
reverse_kl = student_log_probs[i] - teacher_log_probs[i]   # log(π_student / π_teacher)
advantages[i] = adv - opd_kl_coef * reverse_kl            # advantage에서 차감
```

이로 인해 `opd_reverse_kl`이라는 추가 지표도 TensorBoard에 기록됩니다.

### 스크립트 관련 전체 인수

```bash
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
```
