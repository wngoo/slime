# OPD 실험에서 pg_loss, loss, opd_reverse_kl 분석

## 현상

`examples/on_policy_distillation/run-qwen3-8B-opd.sh`로 on-policy distillation 실험 시
TensorBoard에서 `pg_loss`, `loss`, `opd_reverse_kl` 세 값이 동일하게 표시된다.

---

## 원인 1: `loss == pg_loss` (계수가 0)

**`slime/backends/megatron_utils/loss.py:773-789`**

```python
loss = pg_loss - args.entropy_coef * entropy_loss   # entropy_coef = 0.00
                                                      # → loss = pg_loss

if args.use_kl_loss:
    kl_loss = sum_of_sample_mean(kl)
    loss = loss + args.kl_loss_coef * kl_loss        # kl_loss_coef = 0.00
                                                      # → loss += 0
```

`run-qwen3-8B-opd.sh`의 설정:
```bash
--entropy-coef 0.00
--kl-loss-coef 0.00
```

두 항이 0으로 소멸하므로 `loss = pg_loss`가 수학적으로 **정확히** 동일하다.

---

## 원인 2: `pg_loss ≈ opd_reverse_kl` (수식 전개)

### Step 1: reward = 0 (pure distillation)

**`slime/rollout/on_policy_distillation.py:59`**

```python
scalar_rewards = [0.0] * len(samples)
```

task reward 없이 순수 distillation만 수행하므로 모든 샘플의 reward = 0.

### Step 2: GRPO advantage = 0

**`slime/utils/ppo_utils.py:201-208`**

```python
def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])  # rewards[i] = 0.0
    return returns
```

reward = 0이므로 모든 토큰의 GRPO advantage = 0.

### Step 3: OPD KL penalty가 advantage를 대체

**`slime/backends/megatron_utils/loss.py:391-393`**

```python
reverse_kl = student_log_probs[i] - teacher_log_probs[i]   # = log π_θ - log π_teacher
advantages[i] = adv - args.opd_kl_coef * reverse_kl        # = 0 - 1.0 * reverse_kl
                                                             # = -reverse_kl
```

`--opd-kl-coef 1.0`이고 `adv = 0`이므로 결국 **advantage = `-reverse_kl`**.

### Step 4: policy loss 전개

**`slime/utils/ppo_utils.py:132-134`**

```python
ratio = (-ppo_kl).exp()          # = exp(log_probs - old_log_probs)
pg_losses1 = -ratio * advantages # = -ratio * (-reverse_kl) = ratio * reverse_kl
```

학습 **초기**에는 현재 policy ≈ old policy이므로 `ratio ≈ 1`.
클리핑도 거의 발동하지 않으므로:

```
pg_loss = sum_of_sample_mean(ratio * reverse_kl)
        ≈ sum_of_sample_mean(reverse_kl)
        = opd_reverse_kl  ← 로깅 값과 동일
```

---

## 요약 (학습 초기 값이 같은 이유)

| 조건 | 결과 |
|---|---|
| `--entropy-coef 0.00` + `--kl-loss-coef 0.00` | `loss ≡ pg_loss` (수학적 항등식) |
| `reward = 0.0` (pure distillation) | GRPO advantage = 0 |
| `--opd-kl-coef 1.0` | advantage = `-reverse_kl` |
| 학습 초기 `ratio ≈ 1` | `pg_loss ≈ sum_of_sample_mean(reverse_kl)` = `opd_reverse_kl` |

세 값이 같은 것은 버그가 아니라 설계상 당연한 결과이다.
pure distillation에서 task reward가 없으면 OPD KL penalty가 학습의 유일한 신호이며,
그 값이 그대로 `pg_loss`이자 `loss`로 나타난다.

---

## `opd_reverse_kl` 로깅 버그

### 문제

학습이 진행되어 `ratio ≠ 1`이 되면 `pg_loss ≠ opd_reverse_kl`로 달라지는 것은 정상이다.
그러나 **`opd_reverse_kl` 메트릭 자체가 stale한 값을 찍고 있다는 별개의 문제**가 있다.

### 실행 순서 (`slime/backends/megatron_utils/actor.py:432-460`)

```
1. model을 "old_actor"로 스위치
2. compute_log_prob() → rollout_data["log_probs"] 저장  ← 학습 전 forward pass
3. compute_advantages_and_returns() 호출
   └─ apply_opd_kl_to_advantages(student_log_probs = log_probs)
      └─ reverse_kl = log_probs_before_train - teacher_log_probs
      └─ rollout_data["opd_reverse_kl"] = reverse_kl  ← 여기서 한 번만 저장 (고정)
4. train() 시작 → 여러 gradient step
   └─ 매 step마다 ratio = exp(old_log_probs - current_log_probs) 변화
   └─ pg_loss = sum(ratio * reverse_kl)  ← ratio 반영되어 변화
   └─ opd_reverse_kl 로깅 = sum(reverse_kl)  ← ratio 미반영, gradient step 내내 고정
```

### 결론

`opd_reverse_kl`이 나타내야 하는 것:
- **현재 모델**이 teacher로부터 얼마나 멀어졌나 (token-level reverse KL)

실제로 저장되는 것:
- **학습 직전 old_actor** 기준 KL (gradient step 내내 고정된 stale 값)

따라서 TensorBoard의 `opd_reverse_kl`로 "현재 모델이 teacher와 얼마나 가까운가"를 판단하면 **잘못된 해석**이 된다.

> **학습 자체는 문제없다.** 어드밴티지에 bake-in된 `reverse_kl`에 `ratio`로 importance sampling 보정이 되므로 gradient는 올바르게 흐른다. 잘못된 건 **측정값(metric)** 이지 **학습 신호**가 아니다.

### 정확한 로깅을 위한 수정 방향

`slime/backends/megatron_utils/loss.py:827-829`에서 현재 step의 `log_probs`로 재계산해야 한다:

```python
# 현재 (stale): 학습 전 old_actor 기준
if "opd_reverse_kl" in batch:
    opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
    reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

# 수정: 현재 step log_probs로 재계산
if "teacher_log_probs" in batch:
    teacher_lp = torch.cat(batch["teacher_log_probs"], dim=0)
    current_reverse_kl = log_probs - teacher_lp  # log_probs = 현재 step에서 재계산된 값
    reported_loss["opd_reverse_kl"] = sum_of_sample_mean(current_reverse_kl).clone().detach()
```

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `slime/backends/megatron_utils/actor.py:432-460` | 학습 전 log_probs 계산 및 advantage 준비 순서 |
| `slime/backends/megatron_utils/loss.py:359-397` | OPD KL penalty를 advantage에 적용, opd_reverse_kl 저장 |
| `slime/backends/megatron_utils/loss.py:448-501` | GRPO advantage 계산 및 OPD 적용 |
| `slime/backends/megatron_utils/loss.py:706-831` | pg_loss, loss 계산 및 metrics 로깅 |
| `slime/utils/ppo_utils.py:124-148` | `compute_policy_loss` 구현 |
| `slime/utils/ppo_utils.py:201-208` | `get_grpo_returns` 구현 |
| `slime/rollout/on_policy_distillation.py:26-61` | reward = 0 반환, teacher log-probs 추출 |
| `examples/on_policy_distillation/run-qwen3-8B-opd.sh:104-113` | 실험 하이퍼파라미터 설정 |
