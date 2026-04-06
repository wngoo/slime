# MIS (Train-Infer Mismatch Importance Sampling) + OPD 적용 가이드

## 개요

MIS는 훈련 모델과 롤아웃 모델 간의 분포 차이(train-rollout mismatch)를 importance sampling으로 보정하는 기법이다.  
관련 코드: `examples/train_infer_mismatch_helper/mis.py`, `mis.yaml`, `run-qwen3-4b-mis.sh`

OPD(`--use-opd`)는 teacher KL loss로 teacher-student 분포 차이를 보정하고, MIS는 student 자체의 train-rollout mismatch를 독립적으로 보정한다. 두 메커니즘은 함께 사용 가능하다.

## 동작 원리

```
롤아웃 시:  student 모델이 토큰 생성 (log_probs 기록)
훈련 시:    학습 중인 student 모델의 log_probs 재계산
MIS:        비율 w = exp(train_log_prob - rollout_log_prob) 를 계산하여
            policy gradient loss에 곱해 분포 드리프트 보정
```

## 적용 방법 (run-qwen3-8B-opd.sh 기준)

### 1. YAML 설정 파일 준비

`examples/train_infer_mismatch_helper/mis.yaml`을 그대로 사용하거나 복사해서 조정한다:

```bash
cp examples/train_infer_mismatch_helper/mis.yaml examples/on_policy_distillation/mis.yaml
```

### 2. 스크립트 수정

`GRPO_ARGS`에 `--use-tis` 플래그 추가:

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
   --use-tis                          # TIS 활성화
)
```

`CUSTOM_ARGS` 블록 추가:

```bash
CUSTOM_ARGS=(
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

`ray job submit` 명령 끝에 `${CUSTOM_ARGS[@]}` 추가:

```bash
   -- python3 train.py \
   ...
   ${RM_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
```

## YAML 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `use_tis` | `true` | Token-level Importance Sampling 활성화 |
| `use_rs` | `true` | Rejection Sampling 활성화 |
| `tis_level` | `"token"` | IS 가중치 집계 수준: `token` / `sequence` / `geometric` |
| `tis_mode` | `"truncate"` | IS 처리 방식: `truncate` (TIS) / `mask` (MIS) / `clip` (CIS) |
| `tis_upper_bound` | `2.0` | IS 가중치 상한 |
| `tis_lower_bound` | `0.5` | IS 가중치 하한 (clip/mask 모드에서 사용) |
| `rs_veto_threshold` | `1.0e-4` | 이 값보다 낮은 토큰 비율이 있으면 시퀀스 전체 veto |
| `tis_batch_normalize` | `true` | 배치 내 IS 가중치를 mean=1.0으로 정규화 |

## OPD 특성에 맞는 YAML 조정 포인트

| 상황 | 조정 항목 |
|---|---|
| teacher-student 분포 차이가 큰 경우 | `tis_upper_bound`를 더 넓게 (예: `3.0`) |
| 초기 학습 불안정 시 | `tis_mode: "mask"`로 변경해 범위 밖 시퀀스 제외 |
| teacher logit이 0에 가까운 토큰 많을 때 | `rs_veto_threshold` 완화 (예: `1.0e-6`) |

## IS 모드 비교

- **truncate (TIS)**: 가중치를 `[0, upper_bound]`로 자름. 하한 없음.
- **clip (CIS)**: 가중치를 `[lower_bound, upper_bound]`로 클리핑.
- **mask (MIS)**: 범위 밖 시퀀스의 loss_mask를 0으로 설정해 gradient 차단.

## OPD에서 TIS/MIS가 유효한 이유

"On-policy"라는 이름 때문에 IS 보정이 불필요하다고 느낄 수 있으나, TIS/MIS는 OPD에서도 유효하다.

### OPD의 loss 흐름

OPD는 advantage에 teacher KL 페널티를 녹인다 (`loss.py:393`):
```python
advantages[i] = adv - opd_kl_coef * (student_log_prob - teacher_log_prob)
```

이후 `pg_loss`는 이 수정된 advantage로 계산되고, TIS는 바로 이 `pg_loss`에 IS 가중치를 곱한다:
```python
pg_loss = compute_policy_loss(ppo_kl, advantages, ...)  # OPD KL이 녹아있는 advantage
pg_loss = pg_loss * is_weights                          # TIS: w = exp(train_log_prob - rollout_log_prob)
```

KL loss, entropy loss는 TIS 적용 대상이 아니다. 별도로 최종 loss에 합산된다:
```python
loss = pg_loss - entropy_coef * entropy_loss
loss = loss + kl_loss  # use_kl_loss가 활성화된 경우
```

### rollout 시점과 training 시점의 파라미터 차이

"On-policy"는 **student가 직접 rollout을 생성**한다는 의미이지, rollout과 training의 모델 파라미터가 동일하다는 의미가 아니다:

```
rollout 시: θ_t 로 생성 (log_probs 기록)
training 시: θ_t → θ_{t+1} 로 업데이트 중 (log_probs 재계산)
```

특히 아래 상황에서 mismatch가 커진다:
- **async training** (`train_async.py`): rollout과 training이 파이프라인되어 파라미터 드리프트 증가
- **여러 gradient step**: 동일 rollout 데이터로 반복 학습 시 드리프트 누적

따라서 TIS/MIS는 **student 자체의 train-rollout mismatch**를 보정하고, OPD의 teacher KL 보정(`--opd-kl-coef`)은 **teacher-student 분포 차이**를 보정하는 독립적인 메커니즘이다.

## 관련 파일

- `examples/train_infer_mismatch_helper/mis.py` — MIS 핵심 로직 (`compute_mis_weights`, `compute_mis_weights_with_cp`)
- `examples/train_infer_mismatch_helper/mis.yaml` — 기본 설정 파일
- `examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh` — MIS 단독 적용 예시 스크립트
- `examples/on_policy_distillation/run-qwen3-8B-opd.sh` — OPD 기본 스크립트
