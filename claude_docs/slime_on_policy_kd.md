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

| 항목 | 값 | 의미 |
|------|-----|------|
| `--opd-kl-coef` | 1.0 | reverse KL penalty 가중치 |
| `--kl-loss-coef` | 0.00 | reference model과의 KL loss는 **비활성** |
| `--advantage-estimator` | grpo | base advantage는 GRPO |
| reward | 0.0 (고정) | task reward 없음, 학습 신호 전부 OPD KL에서 |
| logprobs 수 | response 길이 × batch | 포지션별 1개, 최대 16,384개/샘플 |
| teacher logprob 계산 시점 | rollout 단계 | SGLang 서버에서 비동기로 수집 |
| student logprob 계산 시점 | training forward | Megatron에서 현재 policy로 계산 |

---

## 4. 어떤 논문의 어떤 부분을 구현했는가

README의 References:

| # | 링크 | 정체 |
|---|------|------|
| 1 | thinkingmachines.ai/blog/on-policy-distillation/ | Thinking Machines Lab 블로그 + 구현 코드 |
| 2 | arxiv.org/abs/2306.13649 | **GKD** (Generalized Knowledge Distillation for Auto-regressive Sequence Models, Agarwal et al., Google DeepMind, 2023) |
| 3 | arxiv.org/abs/2306.08543 | **DistiLLM** (Ko et al., 2023) |

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
