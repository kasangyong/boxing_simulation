# CLAUDE.md — Boxing RL 프로젝트 AI 작업 지침

이 파일은 Claude Code가 이 프로젝트를 작업할 때 항상 준수해야 하는 규칙과 컨텍스트입니다.

---

## 자율 작업 권한

**이 프로젝트 폴더(`c:\Users\SSAFY\Desktop\boxing\`) 내의 모든 파일 수정, 생성, 삭제는 사용자 승인 없이 자율적으로 진행한다.**
파일 편집, bash 명령 실행, 체크포인트 관리 등 모든 작업에서 확인을 구하지 않는다.

---

## 프로젝트 개요

3D 휴머노이드 복싱 시뮬레이션. 두 AI 에이전트가 강화학습 + 자기대전(self-play)으로 복싱을 학습.

**핵심 논문**
- RoboStriker (arXiv 2601.22517): 계층적 복싱, LS-NFSP self-play
- DeepMimic (Peng 2018): RSI + Early Termination
- OpenAI Competitive Self-Play (2017): 리그 기반 과거 상대
- OpenAI Emergent Complexity (arXiv 1710.03748): MuJoCo 다중 에이전트

**기술 스택**
- MuJoCo >= 3.0 (Python bindings, mujoco-py 아님)
- Gymnasium >= 1.0
- Stable-Baselines3 >= 2.0
- Python 3.11

---

## 디렉터리 구조 & 파일 역할

```
envs/assets/boxer.xml     MJCF 래그돌 정의 (두 복서, 42 액추에이터)
envs/boxing_env.py        Gymnasium MultiAgentEnv
training/reward.py        모듈형 보상 함수 (Phase 3+ 전체 복싱)
training/curriculum.py    4단계 커리큘럼 + CurriculumEnv 래퍼 (Phase 1~3 보상 내장)
train.py                  SB3 PPO 학습 진입점
watch.py                  MuJoCo 실시간 뷰어 (체크포인트 hot-reload)
checkpoints/              학습 체크포인트 (.zip)
checkpoints/old/          이전 실험 체크포인트 보관 (삭제 금지)
PROGRESS.md               실험 기록 및 수치 추적
```

---

## 코드 수정 규칙

### boxer.xml 수정 시
- 관절 수(21개/복서)와 액추에이터 수(42개 총합)를 절대 변경하지 말 것.
  `boxing_env.py`의 `N_JOINTS=21`, `action_space`, `qpos/qvel` 인덱싱이 이에 의존.
- stiffness, damping 변경 시 → 기존 체크포인트와 물리가 달라짐 → 반드시 처음부터 재학습.
- geom 이름 규칙: `{b0|b1}_{부위}_geom`, fist는 반드시 `*_fist_geom` 포함.

### boxing_env.py 수정 시
- `OBS_DIM=75` 변경 시 → 기존 체크포인트 호환 불가. 반드시 재학습 필요.
- `TORSO_KO_Z=0.45` 변경 시 → curriculum.py의 같은 상수도 동기화 필요.
- `HP_FORCE_SCALE`, `HP_DAMAGE_MAX` 조정은 Phase 3+ 에서만 효과 있음.
- `_process_contacts()` 수정 시 reward.py의 contact 로그 구조도 확인.

### training/reward.py 수정 시
- 이 파일은 **Phase 3+ (Self-Play)** 전체 복싱 보상 담당.
- Phase 1~3 보상은 `curriculum.py`의 `_phase_reward()` 에서 별도 관리.
- 보상 가중치 변경 시 PROGRESS.md 실험 기록에 변경 이유와 결과를 추가.

### training/curriculum.py 수정 시
- `PHASE_CONFIG` 의 `step_budget`, `reward_threshold` 변경은 신중히.
  너무 낮으면 덜 배운 채로 다음 Phase로 넘어감.
- `_phase_reward()` 는 Phase 1~3 각각의 보상을 조건부로 활성화.
  Phase가 올라갈수록 이전 Phase 보상 위에 **추가**되는 구조.
- upright 보상은 계단형(quadratic):
  `torso_z >= 0.90` → full bonus / `0.70~0.90` → quadratic / `< 0.70` → 페널티

### train.py 수정 시
- PPO 하이퍼파라미터는 논문 스펙 기준. 변경 시 이유 명시.
- `CheckpointCallback(save_freq=50_000)` — 50K 스텝마다 저장 (watch.py 갱신 주기).
- `--eval-freq` 인자는 현재 미사용 (VisualEvalCallback 제거됨). 뷰어는 watch.py 사용.

### watch.py 수정 시
- `CurriculumEnv(phase=1)` — 뷰어가 사용하는 Phase는 현재 학습 Phase와 일치시킬 것.
- `viewer.cam` 설정 (distance, elevation, azimuth)은 자유롭게 조정 가능.

---

## 학습 실행 명령

```bash
# 새로 시작
py -3.11 train.py --phase 1 --total-steps 1000000

# 체크포인트에서 이어 학습
py -3.11 train.py --phase 1 --load checkpoints/boxer_p1_50000_steps.zip --total-steps 1000000

# 뷰어 (별도 터미널에서 실행)
py -3.11 watch.py

# WandB 로깅 포함
py -3.11 train.py --phase 1 --total-steps 1000000 --wandb
```

> **중요**: watch.py는 반드시 사용자 본인 터미널에서 직접 실행.
> Claude의 백그라운드 bash에서 실행하면 MuJoCo 창이 나타나지 않음.

---

## 현재 알려진 이슈 / 해결된 이슈

| 상태 | 이슈 | 해결책 |
|------|------|--------|
| 해결 | 에이전트가 쭈그리고 앉음 (보상 해킹) | upright 보상 계단형 + XML stiffness=3 |
| 해결 | 컨트롤이 0에 고정 | energy 페널티 0.005→0.001 |
| 해결 | 파랑이 안 움직임 | Phase 1은 single-agent, 의도된 동작 |
| 해결 | watch.py 창 안 뜸 | 백그라운드→포그라운드 분리 |
| 진행중 | R이 30에서 정체 | 1M 스텝까지 모니터링 중 |
| 미구현 | Phase 4 Self-Play | `training/self_play.py` 작성 필요 |
| 미구현 | ELO 트래킹 | self_play.py 구현 시 함께 추가 |
| 미구현 | 영상 녹화 | imageio[ffmpeg] 사용 예정 |
| 미구현 | WandB 연동 | `--wandb` 플래그는 있으나 미검증 |

---

## 새 기능 추가 시 체크리스트

- [ ] `boxer.xml` 물리 변경 → 체크포인트 무효화 여부 확인
- [ ] observation 차원 변경 → `OBS_DIM` 상수 및 공간 정의 동기화
- [ ] 보상 변경 → PROGRESS.md 실험 섹션에 기록
- [ ] 새 Phase 추가 → `PHASE_CONFIG` 딕셔너리에 등록
- [ ] 새 파일 추가 → 이 CLAUDE.md의 디렉터리 구조 업데이트

---

## 다음 구현 대기 항목 (Phase 4)

```python
# training/self_play.py 에 구현 예정

class SelfPlayManager:
    """
    - 500K 스텝마다 현재 정책을 checkpoints/league/ 에 저장
    - 상대 샘플링: 70% 현재 정책, 30% 과거 체크포인트 랜덤
    - ELO 트래킹 (K=32, 초기 1000)
    """

class SelfPlayEnv(gym.Env):
    """
    - BoxingEnv를 단일 에이전트 인터페이스로 래핑
    - 상대는 SelfPlayManager가 공급하는 frozen policy
    - observation: fighter_0 시점 75-dim
    """
```
