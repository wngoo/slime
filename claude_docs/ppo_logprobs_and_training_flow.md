# PPO 학습 흐름과 log_probs 이해

## 전체 학습 루프 순서 (`train.py:69-93`)

```python
actor_model.update_weights()  # 루프 전 최초 1회: Megatron → SGLang 가중치 동기화

for rollout_id in range(...):
    rollout_manager.generate()    # ① SGLang(W_N)으로 새 배치 생성 + rollout_log_probs
    actor_model.train()           # ② old_log_probs 계산 + gradient steps (W_N → W_{N+1})
    actor_model.update_weights()  # ③ SGLang을 W_{N+1}으로 갱신
```

---

## old_log_probs vs current log_probs

| | old_log_probs | current log_probs |
|---|---|---|
| 계산 주체 | SGLang 또는 Megatron (설정에 따라) | Megatron (항상) |
| 계산 시점 | training 시작 전 1회 | 매 gradient step의 forward pass 중 |
| iteration 내 변화 | 고정 | gradient step마다 변화 |
| 역할 | PPO clip의 기준점 | 실제 업데이트 중인 policy |

### `use_rollout_logprobs=False` (기본값)
- Megatron이 training 시작 전 전체 rollout 데이터에 대해 **1회 forward pass** → `rollout_data["log_probs"]` 저장
- 추가 비용 있음, train-inference mismatch 없음

### `use_rollout_logprobs=True`
- SGLang이 generation 시 자동 산출한 `rollout_log_probs`를 그대로 재사용
- 추가 forward pass 없음, train-inference mismatch 존재

**어느 경우든 current log_probs는 항상 Megatron이 gradient step 중 계산한다.**

---

## old_log_probs 갱신 주기 (`actor.py:432-460`)

```python
# 매 iteration마다 train() 호출 시 실행
self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
    rollout_data.update(
        self.compute_log_prob(...)   # W_N으로 새로 계산 → rollout_data["log_probs"] 덮어씀
    )
compute_advantages_and_returns(self.args, rollout_data)
train(...)  # 모든 gradient step이 위에서 계산한 old_log_probs를 고정값으로 사용
```

- **iteration 내부**: `old_log_probs` 고정 (모든 gradient step에서 동일값 사용)
- **새 iteration 시작**: `compute_log_prob()`이 다시 호출되어 W_N으로 갱신

```
Iteration 0: old = W₀ 고정 → gradient steps → W₁
Iteration 1: old = W₁ 고정 → gradient steps → W₂
Iteration N: old = W_N 고정 → gradient steps → W_{N+1}
```

---

## old_log_probs를 매 iteration 갱신하는 이유

만약 W₀을 계속 쓴다면:
```
Iteration  0: ppo_kl = W₀ - W₀+ε  → 작음 (clip 정상 동작)
Iteration 10: ppo_kl = W₀ - W₁₀   → 매우 큼 (clip 항상 발동 → gradient 죽음)
```

매 iteration 갱신하면:
```
Iteration  0: ppo_kl = W₀  - W₀+ε   → 이번 iteration의 변화량만 측정
Iteration 10: ppo_kl = W₁₀ - W₁₀+ε  → 이번 iteration의 변화량만 측정
```

eps_clip이 **"이번 iteration에서 얼마나 멀리 움직였는가"** 를 제어하기 위해 기준점(old_log_probs)을 항상 최신으로 유지한다.

---

## current log_probs가 Megatron에서 계산되는 이유

current log_probs는 **별도 forward pass가 아니라** training의 기존 forward pass 도중에 함께 계산된다:

```
Megatron gradient step:
  forward pass → logits 계산
                 └─ get_log_probs_and_entropy(logits) → log_probs (current)
  loss = pg_loss(old_log_probs, log_probs, advantages)
  backward pass → gradient 계산
  optimizer.step()
```

SGLang은 inference 전용으로 gradient 계산이 불가능하므로, loss에 들어가서 backward가 흘러야 하는 current log_probs는 반드시 Megatron에서 계산해야 한다.

---

## gradient step의 의미

`--global-batch-size 64`일 때 Megatron의 gradient accumulation 구조:

```
[1 iteration = global batch 64개 샘플]

micro batch 1 (일부 샘플): forward/backward → gradient 누적
micro batch 2 (일부 샘플): forward/backward → gradient 누적
...
micro batch K (나머지  ): forward/backward → gradient 누적
→ optimizer.step() 1회

old_log_probs는 모든 micro batch에 걸쳐 고정
```

---

## 관련 파일

| 파일 | 내용 |
|---|---|
| `train.py:69-93` | 전체 학습 루프 (generate → train → update_weights) |
| `slime/backends/megatron_utils/actor.py:432-460` | old_log_probs 계산 및 train() 호출 순서 |
| `slime/backends/megatron_utils/loss.py:644` | `old_log_probs` 선택 (`use_rollout_logprobs` 분기) |
| `slime/backends/megatron_utils/loss.py:660` | current log_probs 계산 (forward pass 중) |
| `slime/backends/megatron_utils/loss.py:704` | `ppo_kl = old_log_probs - log_probs` |
