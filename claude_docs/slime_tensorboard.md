# Slime TensorBoard 지표 설명

> 기준 스크립트: `examples/on_policy_distillation/run-qwen3-8B-opd.sh`
> 학습 방식: On-Policy Distillation (OPD) — 학생 모델 Qwen3-8B가 교사 모델 Qwen3-32B를 모방

---

## TensorBoard 활성화 방법

스크립트에 `--use-tensorboard` 플래그가 없으면 TensorBoard에 아무것도 저장되지 않습니다.

```bash
# 학습 인자에 추가 필요
--use-tensorboard
--tensorboard-dir ./tensorboard_log  # 또는 환경변수 TENSORBOARD_DIR 설정
```

---

## 1. `train/` — 학습 단계 지표

train step마다 기록됩니다 (`step_key = "train/step"`).

### `train/loss` — 전체 학습 손실

전체 손실의 합 (`pg_loss + kl_loss + entropy_loss`).

| 상태 | 해석 |
|---|---|
| 초반에 크고 점점 감소 | 정상. 모델이 교사를 따라가며 학습 중 |
| 갑자기 급등 | 학습 불안정. `grad_norm` 폭발과 함께 나타나면 학습률을 낮춰야 함 |
| 0에 수렴하거나 음수 | OPD에서 reward가 모두 0.0이므로 loss도 0 근처에서 수렴하는 것은 정상일 수 있으나, 음수가 된다면 클리핑 설정 점검 필요 |
| 전혀 감소하지 않음 | 학습이 이루어지지 않는 중. 학습률, 데이터, 모델 설정 점검 필요 |

---

### `train/pg_loss` — Policy Gradient 손실

모델이 advantage가 높은 토큰을 더 자주, 낮은 토큰을 덜 생성하도록 유도하는 핵심 손실.
OPD에서 advantage는 reward(0.0)가 아닌 **OPD KL 페널티**에서 나옵니다(`opd_kl_coef * reverse_kl`).

| 상태 | 해석 |
|---|---|
| 음수이고 점점 감소(더 음수) | 정상. 모델이 교사를 잘 따라가는 방향으로 정책을 업데이트 중 |
| 0 근처에서 진동 | 교사와의 차이가 크지 않아 advantage가 작은 상태. 학습 초반 또는 이미 잘 모방 중 |
| 갑자기 양수로 커짐 | 이상. 모델이 나쁜 방향으로 업데이트되고 있음. `pg_clipfrac`과 같이 확인 |

---

### `train/entropy_loss` — 엔트로피

모델이 각 토큰을 얼마나 확신 없이 다양하게 생성하는지를 나타냅니다.
본 스크립트는 `entropy_coef=0`이라 손실에 직접 반영되지는 않지만, 모델의 다양성을 모니터링하는 지표로 활용합니다.

| 상태 | 해석 |
|---|---|
| 적당히 높음 (초반) | 모델이 다양한 토큰을 고려 중. 탐색 능력 있음 |
| 학습이 진행되며 완만하게 감소 | 정상. 교사 분포를 따라가며 확신이 높아지는 것 |
| 급격히 0으로 수렴 | **나쁨.** 모델이 특정 패턴에 굳어지는 collapse 징후. 다양성 손실 |
| 너무 높고 감소하지 않음 | 모델이 수렴하지 않고 방황 중. 학습이 제대로 진행되지 않음 |

---

### `train/pg_clipfrac` — PPO 클리핑 발동 비율

새 정책(현재 학습 중인 모델)이 이전 정책(롤아웃 시점 모델)에서 너무 많이 벗어났을 때 클리핑이 발동됩니다.
클리핑이 발동된 토큰의 비율입니다.

| 상태 | 해석 |
|---|---|
| 0.1 ~ 0.2 | 정상 범위. 어느 정도 업데이트가 일어나지만 안정적 |
| 0.3 이상 | **주의.** 업데이트가 너무 커서 불안정. 학습률 낮추거나 `eps_clip` 값 확인 |
| 0에 가까움 | 업데이트가 거의 없음. 학습이 사실상 정체됨 |
| 0.5 이상 | **위험.** 학습이 매우 불안정. 즉시 조치 필요 |

---

### `train/ppo_kl` — 정책 변화량 (PPO KL)

이번 학습 step 후 정책이 롤아웃 당시 정책에서 얼마나 벗어났는지를 나타냅니다.
수식: `old_logp - new_logp` (토큰 평균).

| 상태 | 해석 |
|---|---|
| 0.01 ~ 0.1 | 정상. 각 step마다 적절한 크기로 업데이트 중 |
| 0에 가까움 | 업데이트가 거의 없음. 학습이 정체 |
| 0.2 이상 | **주의.** 업데이트가 너무 커서 off-policy 문제 발생 가능 |
| 급등 후 발산 | **위험.** 학습 불안정. `pg_clipfrac`이 함께 높다면 학습률 즉시 조절 필요 |

---

### `train/kl_loss` — Reference 모델과의 KL 손실

원본 Qwen3-8B(고정)와 현재 학습 모델 사이의 KL divergence 손실.
본 스크립트는 `kl_loss_coef=0.00`이라 **실제 손실에 반영되지 않지만 값은 기록됩니다.**

| 상태 | 해석 |
|---|---|
| 0에 가까움 (학습 초반) | 정상. 초기에는 ref 모델과 같음 |
| 점점 증가 | 학습이 진행되며 원본 모델에서 멀어지는 중. OPD에서는 교사를 따라가므로 어느 정도 증가는 자연스러움 |
| 매우 크게 증가 | 모델이 ref에서 너무 멀어짐. `kl_loss_coef`를 0이 아닌 값으로 올려서 제약을 줄 필요가 있을 수 있음 |

---

### `train/opd_reverse_kl` — 학생↔교사 Reverse KL ⭐ OPD 핵심 지표

**이 실험에서 가장 중요한 지표입니다.**
수식: `student_logp - teacher_logp` (토큰별 평균).
0보다 크면 학생이 교사보다 해당 토큰에 더 높은 확률을 부여 중, 0보다 작으면 반대입니다.

| 상태 | 해석 |
|---|---|
| 초반에 양수/음수이고 점점 0으로 수렴 | **이상적.** 학생이 교사 분포를 성공적으로 모방하는 중 |
| 음수이고 계속 감소 | 학생이 교사보다 낮은 확률을 부여하는 방향으로 수렴. `opd_kl_coef` 조정 고려 |
| 0 근처에서 수렴 완료 | 학생이 교사 분포를 잘 모방한 상태. 학습 목표 달성 |
| 발산하거나 진동이 심함 | 학습이 불안정. `opd_kl_coef`나 학습률 조정 필요 |

> **advantage 계산과의 관계:** advantage에서 `opd_kl_coef * reverse_kl`을 빼서, reverse KL이 클수록 해당 토큰의 advantage가 낮아져 모델이 자연스럽게 교사 분포를 따라가도록 유도합니다.

---

### `train/grad_norm` — 그래디언트 노름

역전파 시 그래디언트의 크기. 학습 안정성을 나타내는 대표 지표.

| 상태 | 해석 |
|---|---|
| 1 ~ 10 사이에서 안정적 | 정상. 학습이 안정적으로 진행 중 |
| 점진적으로 감소 | 수렴 중. 정상 |
| 갑자기 100 이상으로 급등 | **위험(그래디언트 폭발).** 직전 배치의 이상 데이터 또는 학습률이 너무 큰 것. 연속으로 발생하면 즉시 학습률 낮추기 |
| 0에 수렴 | 그래디언트 소실. 모델이 더 이상 학습하지 않음 |

---

### `train/lr-pg_0` — 학습률

현재 옵티마이저의 학습률. 본 스크립트는 `constant` 스케줄이라 `1e-6`으로 고정됩니다.

| 상태 | 해석 |
|---|---|
| `1e-6`으로 고정 | 이 스크립트의 정상 동작 |
| 예상보다 낮거나 높음 | 스케줄러 설정 오류. `--lr-decay-style` 확인 필요 |

---

## 2. `rollout/` — 롤아웃 단계 지표

rollout마다 기록됩니다 (`step_key = "rollout/step"`).

### `rollout/response_len/mean` (median, max, min) — 응답 길이 통계

생성된 응답의 토큰 길이 통계. `rollout_max_response_len=16384`가 상한선.

| 상태 | 해석 |
|---|---|
| mean이 점진적으로 증가 | 모델이 점점 더 길고 자세한 응답을 생성. OPD에서 교사(32B)가 긴 chain-of-thought를 하면 자연스럽게 따라가며 증가 |
| mean이 max에 가까움 | 많은 응답이 최대 길이에서 잘리는 중. `truncated_ratio` 함께 확인 |
| mean이 갑자기 급감 | 모델이 매우 짧은 답변만 생성하기 시작. 학습 붕괴 가능성 |
| max와 min의 격차가 매우 큼 | 응답 길이 편차가 큼. 데이터의 난이도 다양성 반영 |

---

### `rollout/truncated_ratio` — 최대 길이 초과 응답 비율

응답이 `rollout_max_response_len`(16384)에서 잘린 비율.

| 상태 | 해석 |
|---|---|
| 0.1 미만 | 정상. 대부분의 응답이 자연스럽게 완결됨 |
| 0.3 이상 | **주의.** 많은 응답이 잘림. `rollout_max_response_len` 증가 고려 |
| 1.0에 가까움 | **위험.** 거의 모든 응답이 잘림. 교사 모델이 매우 긴 응답을 유도하거나 모델이 반복 루프에 빠진 것. `repetition_frac`과 함께 확인 |

---

### `rollout/repetition_frac` — 반복 응답 비율

응답에서 압축률 기반으로 반복 패턴이 탐지된 비율(마지막 10000 토큰 기준 압축률 > 10이면 반복으로 간주).

| 상태 | 해석 |
|---|---|
| 0에 가까움 | 정상. 응답이 다양하고 의미 있음 |
| 0.1 이상으로 증가 | **주의.** 모델이 반복적인 패턴을 생성하기 시작. 학습 불안정 또는 교사 분포를 잘못 학습하는 중 |
| 지속적으로 증가 | **위험.** 모델 collapse 징후. 엔트로피 감소와 함께 나타나면 학습 중단 후 점검 필요 |

---

### `rollout/zero_std/count_0.0` — reward가 0.0인 그룹 수

한 프롬프트에서 생성한 여러 응답의 reward가 모두 동일(표준편차=0)한 그룹 수.
**OPD에서는 reward가 항상 0.0이므로 이 값은 항상 전체 그룹 수(`rollout_batch_size / n_samples_per_prompt = 16/4 = 4`)와 같습니다.** 이 값을 해석하는 대신, OPD에서는 `rollout/opd_reverse_kl`을 주목해야 합니다.

---

### `rollout/log_probs` — 학생 모델 log-probability

학생 모델이 자신이 생성한 토큰 시퀀스에 부여한 log-probability의 평균.
음수값이며, 0에 가까울수록 확신이 높음.

| 상태 | 해석 |
|---|---|
| 학습 중 점점 0에 가까워짐 | 정상. 모델이 생성한 토큰에 점점 더 확신을 가짐 |
| `ref_log_probs`와 동일 | 학습 초기 상태 또는 학습이 이루어지지 않고 있음 |
| `teacher_log_probs`에 수렴 | **이상적.** 학생이 교사 분포를 완전히 모방한 상태 |
| 갑자기 매우 작은 음수로 급락 | **위험.** 모델이 생성한 토큰에 극도로 낮은 확률을 부여. 학습 불안정 |

---

### `rollout/ref_log_probs` — Reference 모델 log-probability

원본 Qwen3-8B(고정, 학습 전 상태)의 log-probability. 학습 내내 변하지 않으므로 **기준선(baseline)** 역할을 합니다.

| 상태 | 해석 |
|---|---|
| 일정하게 유지 | 정상. ref 모델은 고정되어 있음 |
| 변동이 있음 | 이상. ref 모델이 업데이트되고 있거나 배치 구성이 크게 달라진 것 |

---

### `rollout/teacher_log_probs` — 교사 모델 log-probability ⭐ OPD 핵심 지표

교사 모델(Qwen3-32B)이 학생의 응답 토큰에 부여한 log-probability 평균.
교사가 "이 토큰은 내가 생성했을 것"이라고 평가하는 정도입니다.

| 상태 | 해석 |
|---|---|
| `log_probs`보다 높음 (초반) | 정상. 교사가 학생보다 해당 시퀀스에 더 높은 확률을 부여. 학생이 아직 교사를 못 따라감 |
| `log_probs`와 차이가 줄어듦 | **학습 진행 중.** 학생이 교사 분포를 따라가는 중 |
| `log_probs`와 거의 같아짐 | 이상적. 학생이 교사의 분포를 잘 모방한 상태 |
| `log_probs`보다 낮아짐 | 학생이 교사보다 높은 확률을 부여하는 방향으로 과도하게 수렴. `opd_kl_coef` 조정 고려 |

---

### `rollout/opd_reverse_kl` — 롤아웃 단계의 학생↔교사 Reverse KL

`student_logp - teacher_logp`의 평균. `train/opd_reverse_kl`과 같은 값이지만, **롤아웃 시점에서 계산된 값**으로 학습 전 상태를 반영합니다.

| 상태 | 해석 |
|---|---|
| 양수이고 점차 0으로 감소 | **이상적.** 학생이 교사 분포를 따라가며 격차가 줄어드는 중 |
| 0으로 수렴 완료 | 학생이 교사를 잘 모방한 상태 |
| 음수로 발산 | 학생이 교사보다 과도하게 높은 확률을 부여하는 방향으로 학습됨. `opd_kl_coef` 조정 필요 |

---

### `rollout/advantages` — GRPO Advantage 평균

OPD에서의 advantage = reward(0.0) - baseline에 OPD KL 페널티를 적용한 값.
수식: `advantage = (reward - group_mean_reward) - opd_kl_coef * reverse_kl`

| 상태 | 해석 |
|---|---|
| 0 근처에서 진동 | OPD에서는 reward가 모두 0.0이므로 그룹 내 분산이 없어 advantage의 reward 항목이 0. KL 페널티 항만 남음. 정상 |
| 음수 쪽으로 치우침 | KL 페널티가 advantage를 낮추는 중. 학생이 교사와 많이 달라서 페널티가 큰 상태 |
| 점차 0에 수렴 | 학생이 교사를 잘 따라가며 KL 페널티가 줄어드는 것. 학습 목표 달성 |

---

## 3. `perf/` — 성능/속도 지표

### `perf/rollout_time` — 롤아웃 소요 시간 (초)

한 번의 SGLang 응답 생성 전체 시간.

| 상태 | 해석 |
|---|---|
| 안정적으로 유지 | 정상 |
| 점점 증가 | 응답 길이가 늘어나거나 GPU 메모리 단편화 발생. `response_len`과 함께 확인 |
| 학습 진행 중 갑자기 증가 | SGLang 서버 메모리 부족 또는 KV cache 만료 증가 |

---

### `perf/tokens_per_gpu_per_sec` — GPU당 토큰 생성 속도

SGLang의 GPU 처리량. 높을수록 좋음.

| 상태 | 해석 |
|---|---|
| 높고 안정적 | 정상. GPU를 효율적으로 활용 중 |
| 낮음 | batch가 너무 작거나, GPU 메모리 부족으로 prefill이 느림. `sglang-mem-fraction-static` 조정 고려 |
| 학습 중 점점 감소 | 응답 길이 증가나 KV cache 압박. 장기 실험 시 모니터링 필요 |

---

### `perf/actor_train_tflops` — 학습 단계 연산 효율

학습(forward + backward) 단계에서의 실제 계산량(TFLOP/s). GPU 이론 성능 대비 효율 지표.
H100(BF16 기준 이론 989 TFLOP/s)과 비교하여 MFU(Model FLOP Utilization)를 유추할 수 있습니다.

| 상태 | 해석 |
|---|---|
| 높음 | GPU를 효율적으로 사용 중. 이상적 |
| 낮음 | micro-batch가 너무 작거나 sequence padding 낭비가 큼. `use_dynamic_batch_size` 활성화로 개선 가능 |

---

### `perf/wait_time_ratio` — 학습 대기 비율 ⭐ 병목 진단 핵심 지표

전체 step 시간 중 롤아웃(데이터 생성) 완료를 기다리는 시간의 비율.

| 상태 | 해석 |
|---|---|
| 0.3 미만 | 정상. 학습이 롤아웃보다 느려 GPU 활용 양호 |
| 0.5 이상 | **주의.** 롤아웃이 병목. 학습 GPU가 롤아웃 대기 중 낭비 중. SGLang GPU 수(`--rollout-num-gpus`) 증가 고려 |
| 0에 가까움 | 학습이 병목. 롤아웃이 학습보다 훨씬 빠름. 학습 GPU 추가 또는 `global_batch_size` 증가 고려 |

---

### `perf/step_time` — 전체 step 소요 시간

`rollout_time + actor_train_time`의 합. 전체 학습 속도를 나타냅니다.

| 상태 | 해석 |
|---|---|
| 일정하게 유지 | 정상 |
| 점진적으로 증가 | 응답 길이 증가에 따른 자연스러운 증가이거나, 메모리 단편화로 인한 성능 저하 |
| 갑자기 급증 | OOM에 가까운 상황이거나 SGLang/Megatron에서 재시도 발생 |

---

## 코드 위치 요약

| 카테고리 | 로깅 위치 |
|---|---|
| `train/` | `slime/backends/megatron_utils/model.py:651` |
| `rollout/` (생성 측) | `slime/ray/rollout.py:1178` (`_log_rollout_data`) |
| `rollout/` (학습 측) | `slime/backends/megatron_utils/data.py:217` (`gather_log_data`) |
| `perf/` (롤아웃) | `slime/ray/rollout.py:1174` (`compute_perf_metrics_from_samples`) |
| `perf/` (학습) | `slime/utils/train_metric_utils.py:48` (`log_perf_data_raw`) |
| TensorBoard 어댑터 | `slime/utils/tensorboard_utils.py` |
| 통합 로깅 함수 | `slime/utils/logging_utils.py:45` (`log()`) |

---

## OPD 학습 상태 진단 요약

### 학습이 잘 되고 있는 신호

- `train/opd_reverse_kl` 감소 → 0 수렴
- `rollout/log_probs`가 `rollout/teacher_log_probs`에 수렴
- `train/grad_norm` 이 1~10 사이에서 안정적
- `rollout/truncated_ratio` 0.1 미만
- `rollout/repetition_frac` 0에 가까움

### 학습이 불안정한 신호

- `train/grad_norm` 갑자기 100 이상으로 급등
- `train/pg_clipfrac` 0.3 이상
- `rollout/repetition_frac` 지속 증가
- `train/entropy_loss` 급격히 0으로 수렴
- `train/opd_reverse_kl` 발산하거나 진동이 심함

### 하드웨어 병목 진단

- `perf/wait_time_ratio` 높음 → 롤아웃 GPU 부족 (SGLang GPU 추가)
- `perf/wait_time_ratio` 낮음 → 학습 GPU 부족 (학습 GPU 추가 또는 batch size 증가)
- `perf/tokens_per_gpu_per_sec` 낮음 → SGLang 설정 점검 (`sglang-mem-fraction-static` 조정)
