# On-Policy Knowledge Distillation 분석

> `examples/on_policy_distillation/run-qwen3-8B-opd.sh` 실행 기준

---

## 1. 전체 파이프라인 개요

```
[Student: Qwen3-8B on SGLang]  →  응답 생성 (on-policy)
        ↓
[Teacher: Qwen3-32B on SGLang]  →  생성된 응답에 대해 token-level logprobs 계산
        ↓
[Megatron Training]  →  reverse KL을 advantage에 per-token penalty로 합산 → policy gradient
```

---

## 2. 각 포지션별 logprobs를 어떻게, 몇 개 사용하는가

### Phase 1 – 롤아웃: 학생이 on-policy로 응답 생성

`--rollout-max-response-len 16384`이므로 응답 최대 16,384 토큰. student는 자신의 현재 policy로 응답을 생성합니다.

### Phase 2 – Teacher logprob 수집 (`slime/rollout/on_policy_distillation.py:7`)

```python
payload = {
    "input_ids": sample.tokens,          # prompt + response 전체 토큰
    "sampling_params": {
        "max_new_tokens": 0,             # 생성 없음, scoring만
        "temperature": 0,
    },
    "return_logprob": True,
    "logprob_start_len": 0,              # 전체 시퀀스에 대해 logprob 반환
}
```

teacher SGLang 서버에 **full sequence(prompt+response)**를 넘기고 `max_new_tokens=0`으로 생성 없이 각 입력 토큰의 `log π_teacher(t_i | t_<i)`를 받습니다.

### Phase 3 – Response 구간만 트리밍 (`slime/rollout/on_policy_distillation.py:43`)

```python
teacher_log_probs = [
    torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]])
    for reward in raw_rewards
]
teacher_log_probs = [
    t_log_prob[-response_length:]        # ← response 구간만 슬라이싱
    for t_log_prob, response_length in zip(teacher_log_probs, response_lengths)
]
```

- `input_token_logprobs[1:]`: 첫 토큰(BOS)을 제외한 나머지 전체
- `[-response_length:]`: prompt 부분을 버리고 **응답 토큰만 남김**
- 결과: `teacher_log_probs[i].shape == [response_length_i]`
- **사용 개수**: 샘플당 response 길이만큼 (최대 16,384개), 포지션별 1개씩

### Phase 4 – 학생 logprobs

Megatron training forward pass에서 `get_log_probs_and_entropy`(`slime/backends/megatron_utils/loss.py:225`)가 student 현재 policy의 `log π_student(t_i | t_<i)`를 response 구간에 대해 동일하게 계산합니다.

- `student_log_probs[i].shape == [response_length_i]` — teacher와 **포지션 완전 일치**

### Phase 5 – Reverse KL을 Advantage에 per-token 패널티로 적용 (`slime/backends/megatron_utils/loss.py:391`)

```python
reverse_kl = student_log_probs[i] - teacher_log_probs[i]   # [response_length]
advantages[i] = adv - args.opd_kl_coef * reverse_kl        # opd_kl_coef=1.0
```

각 포지션 `t`에서:

```
A_modified[t] = A_base[t] − 1.0 × (log π_student(t_t|t_<t) − log π_teacher(t_t|t_<t))
```

최종 policy gradient loss (per-token):

```
L[t] = −A_modified[t] × log π_student(t_t|t_<t)
     = −A_base[t] × log π_student[t]
       + coef × (log π_student[t] − log π_teacher[t]) × log π_student[t]
```

즉 **"reward를 최대화하면서 동시에 teacher와의 reverse KL을 minimize"**하는 objective입니다.

---

## 3. 스크립트 설정 요약

| 항목                      | 값                    | 의미                                        |
| ------------------------- | --------------------- | ------------------------------------------- |
| `--opd-kl-coef`           | 1.0                   | reverse KL penalty 가중치                   |
| `--kl-loss-coef`          | 0.00                  | reference model과의 KL loss는 **비활성**    |
| `--advantage-estimator`   | grpo                  | base advantage는 GRPO                       |
| reward                    | 0.0 (고정)            | task reward 없음, 학습 신호 전부 OPD KL에서 |
| logprobs 수               | response 길이 × batch | 포지션별 1개, 최대 16,384개/샘플            |
| teacher logprob 계산 시점 | rollout 단계          | SGLang 서버에서 비동기로 수집               |
| student logprob 계산 시점 | training forward      | Megatron에서 현재 policy로 계산             |

---

## 4. 어떤 논문의 어떤 부분을 구현했는가

README의 References:

| #   | 링크                                             | 정체                                                                                                                    |
| --- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| 1   | thinkingmachines.ai/blog/on-policy-distillation/ | Thinking Machines Lab 블로그 + 구현 코드                                                                                |
| 2   | arxiv.org/abs/2306.13649                         | **GKD** (Generalized Knowledge Distillation for Auto-regressive Sequence Models, Agarwal et al., Google DeepMind, 2023) |
| 3   | arxiv.org/abs/2306.08543                         | **DistiLLM** (Ko et al., 2023)                                                                                          |

`apply_opd_kl_to_advantages`의 docstring References에도 아래가 명시되어 있습니다:

```
https://github.com/thinking-machines-lab/tinker-cookbook/.../train_on_policy.py
```

### 핵심 논문: GKD (arxiv 2306.13649)

GKD 논문이 제안하는 핵심:

1. **On-policy generation**: KD 학습에 쓸 시퀀스를 teacher나 gold 데이터에서 가져오지 않고, **학생 자신의 현재 policy로 샘플링**
2. **Generalized KL objective**: forward KL `KL(p_T || p_S)`뿐 아니라 **reverse KL `KL(p_S || p_T)`**도 지원
3. **RL과의 결합**: KD 항을 RL reward shaping으로 해석할 수 있음을 보임

slime 구현과 GKD 논문의 대응:

```
GKD 논문 Eq. (3) / Section 3:
  L_GKD = E_{y ~ π_student} [ Σ_t KL(p_T(·|x,y_{<t}) || p_S(·|x,y_{<t})) ]
        = E_{y ~ π_student} [ Σ_t (log p_S[t] - log p_T[t]) ]   ← reverse KL
```

```python
# slime 코드 (loss.py:391)
reverse_kl[t] = log π_student[t] - log π_teacher[t]    # ← 논문 reverse KL 그대로
advantages[i] = adv - coef * reverse_kl                # ← RL advantage에 패널티로 추가
```

이것이 GKD 논문 **Section 4 "Connection to Reinforcement Learning"** 에서 설명하는 방식입니다: KD 항을 per-step reward shaping으로 표현하면 RL objective와 결합할 수 있으며, reward에 `-KL(p_S||p_T)` 패널티를 더하는 것과 수학적으로 동치입니다.

### DistiLLM (arxiv 2306.08543) 의 역할

DistiLLM은 skew-KL divergence와 on-policy 샘플링의 효율적 결합을 제안하는 논문으로, GKD의 on-policy 아이디어를 보완하는 참고 문헌으로 포함되어 있습니다. slime 구현에서 skew-KL을 쓰지 않고 순수 reverse KL을 쓰므로 **주 구현 근거는 GKD 논문**입니다.

### 구현상의 slime만의 특징

GKD 원논문 대비 slime이 추가한 설계 결정:

- **Advantage에 additive penalty로 통합**: KD를 별도 loss term이 아닌 advantage에 녹여서, **어떤 advantage estimator(GRPO, PPO, REINFORCE++ 등)와도 조합 가능**하게 만든 것이 핵심
- **Task reward = 0.0**: 순수 distillation 목적이면 task reward를 주지 않고 KL signal만 사용
- **Teacher를 외부 SGLang 서버로 분리**: 아키텍처가 달라도(Qwen3-32B → Qwen3-8B) 적용 가능

---

## 5. 왜 GRPO를 Advantage Estimator로 사용하는가

README에 명시된 대로 OPD는 advantage estimator와 **직교(orthogonal)**합니다:

> "OPD works as an additive KL penalty on top of any advantage estimator (GRPO, PPO, REINFORCE++, etc.), not as a separate estimator."

즉 OPD는 advantage estimator를 대체하는 게 아니라 그 위에 더해지는 구조입니다. GRPO를 쓰든, PPO를 쓰든 상관없이 동작합니다.

reward=0.0(고정)이므로 GRPO base advantage는 사실상 0이 되고:

```
A_base[t]    ≈ 0  (모든 샘플 reward가 동일하게 0이므로 GRPO 정규화 후 0)
A_modified[t] = 0 − coef × (log π_S[t] − log π_T[t]) = −reverse_KL[t]
```

결과적으로 학습 신호 100%가 OPD KL에서 나옵니다.

GRPO를 선택한 이유:

1. **확장성**: task reward를 나중에 추가하면 `scalar_rewards = [0.0]` 부분만 실제 reward 함수로 교체하면 됩니다. GRPO+OPD 조합이 그대로 유지됩니다.
2. **아키텍처 요구사항**: slime training loop은 advantage estimator가 반드시 필요합니다. OPD 자체는 별도 estimator가 아니므로 어떤 estimator든 dummy로 하나를 지정해야 합니다.
3. **동등성**: 어떤 estimator를 골라도 reward=0.0 조건에서 동일하게 동작합니다.

---

## 6. Off-Policy 구현 검증

`--load-debug-rollout-data`로 PT 파일을 로드하는 off-policy KD 실행 시 student logit 계산이 올바르게 이루어지는지 코드 흐름을 단계별로 추적합니다.

### ① PT 로드 → rollout_data 구성 (`rollout.py:549-555`, `rollout.py:724-725`)

```python
rollout_data["tokens"]            = [prompt_ids + student_response_ids]  # PT에서 복원
rollout_data["response_lengths"]  = [response lengths]
rollout_data["teacher_log_probs"] = [tensor per sample]   # PT에서 복원 ✓
rollout_data["rollout_log_probs"] = [tensor per sample]   # 구 rollout 시점 student logprobs
rollout_data["rewards"]           = [0.0, ...]
```

> PT 파일에는 on-policy rollout 당시 student가 생성한 응답이 담기고, teacher는 그 student 응답에 대해 scoring → `teacher_log_probs`가 저장됨.

### ② student 현재 log probs 계산 (`actor.py:432-445`)

```python
self._switch_model("actor")
if not self.args.use_rollout_logprobs or ...:
    rollout_data.update(
        self.compute_log_prob(data_iterator, ..., store_prefix="")
        # forward_only(get_log_probs_and_entropy, ...) 호출
        # → rollout_data["log_probs"] = 현재 student weight로 계산한 log probs
    )
```

`forward_only` (`model.py:295`): `rollout_data[f"{store_prefix}{key}"] = values` — `store_prefix=""` 이므로 `rollout_data["log_probs"]`에 저장됩니다. **이 단계가 student logit 계산의 핵심입니다.** ✓

### ③ advantage 계산 (`loss.py:421`, `loss.py:496-500`)

```python
# compute_advantages_and_returns
log_probs = rollout_data.get("log_probs")   # ← ②에서 방금 계산한 student log probs

apply_opd_kl_to_advantages(
    ...
    student_log_probs=log_probs,            # student 현재 log probs
    # rollout_data["teacher_log_probs"]도 여기서 사용
)
# → reverse_kl    = student_log_probs[i] - teacher_log_probs[i]
# → advantages[i] = 0 - 1.0 * reverse_kl  = -reverse_kl
```

✓

### ④ policy gradient backward (`loss.py:644-706`)

```python
old_log_probs = batch["log_probs"]           # ②에서 계산한 student log probs (detached)
log_probs     = get_log_probs_and_entropy()  # backward pass용 student log probs (with grad)
ppo_kl        = old_log_probs - log_probs    # ≈ 0 at step 시작, PPO clip용 ratio
ratio         = exp(-ppo_kl)
pg_loss       = -ratio.clamp(...) * advantages   # clipped policy gradient
```

✓

### 주의: `rollout_log_probs`의 의미

PT 파일에는 rollout 당시 student의 log probs (`sample.rollout_log_probs`)도 포함되어 있어 `rollout_data["rollout_log_probs"]`에 로드됩니다. 스크립트에서 `--use-rollout-logprobs`가 없으므로 이것은 policy gradient에 **사용되지 않고**, `train_rollout_logprob_abs_diff` 메트릭 계산(`loss.py:796-798`)에만 쓰입니다.

현재 student weights(②)와 rollout 당시 old policy(`rollout_log_probs`) 간의 drift 지표로만 기록되며 학습에는 영향 없습니다. ✓

### 최종 결론

student logit 계산(`compute_log_prob` → `forward_only` → `get_log_probs_and_entropy`)은 `compute_advantages_and_returns` 호출 **전**에 매 training step마다 정확히 실행되고 있습니다. 코드에 문제 없습니다.

---

## 7. GPU 배치 구조

> `examples/on_policy_distillation/run-qwen3-8B-opd.sh` 기준 (8 GPU 머신)

```
GPU 0-1  : Actor model (Megatron, TP=2)           ← training
           + Reference model (동일 GPU 공유, CPU pinned memory로 스위칭)
GPU 2-5  : Student rollout model (SGLang, 4 engines)  ← --rollout-num-gpus 4
GPU 7    : Teacher model (Qwen3-32B, standalone SGLang server, 별도 프로세스)
```

### Reference model이 actor GPU에 같이 올라가는 방식

`load_other_checkpoint("ref", args.ref_load)`가 actor와 **동일한 Megatron model 객체**에 ref 체크포인트를 로드한 뒤, `weights_backuper.backup("ref")`로 해당 weights를 **CPU pinned memory**에 복사합니다 (`tensor_backper.py:59`).

```python
# _TensorBackuperNormal.backup()
backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
backup_dict[name].copy_(param.detach(), non_blocking=True)
```

학습 중에는 `_switch_model("ref")` → `restore("ref")`로 CPU pinned memory에서 GPU로 복원하고, 끝나면 `_switch_model("actor")`로 다시 actor weights를 복원합니다. **GPU 메모리는 하나인데 actor/ref가 교대로 점유**하는 구조입니다.

```
_switch_model("ref")   → CPU pinned memory → GPU  (ref forward pass: ref log probs 계산)
_switch_model("actor") → CPU pinned memory → GPU  (actor training step)
```

이 스크립트에서 `with_ref` 조건은:
```python
with_ref = args.kl_coef != 0 or args.use_kl_loss   # --use-kl-loss 플래그가 있으므로 True
```
`--kl-loss-coef 0.00`이지만 `--use-kl-loss` 플래그가 설정되어 있어 `use_kl_loss=True` → `with_ref=True`가 됩니다.

### `--rollout-num-gpus 4`의 SGLang은 teacher가 아니라 student

| 항목 | GPU | 설명 |
|------|-----|------|
| `--rollout-num-gpus 4` | GPU 2-5 | **Student** Qwen3-8B, rollout 생성용 SGLang engines |
| `CUDA_VISIBLE_DEVICES=7` SGLang server | GPU 7 | **Teacher** Qwen3-32B, 별도 standalone 프로세스로 미리 기동 |
| `--rm-url http://...13141/generate` | — | Teacher server로 custom reward 함수가 HTTP 요청하는 엔드포인트 |

OPD(`--opd-type sglang`)에서 rollout server는 **student가 응답을 생성**하는 곳이고, teacher는 `--custom-rm-path`의 reward function이 log prob을 계산하기 위해 HTTP로 호출하는 **별도 서버**입니다. `opd-type megatron`이었다면 teacher를 Megatron으로 actor GPU에 올리지만, `sglang` 타입이므로 외부 서버로 분리됩니다.

```python
# placement_group.py
with_opd_teacher = args.use_opd and args.opd_type == "megatron"  # sglang이므로 False
```
