# Boxing RL — 진행 기록

> 마지막 업데이트: 2026-03-20
> 현재 상태: **Self-Play 학습 중 — 9,950,000 / 100,000,000 스텝 (10%)**

---

## 전체 구조 요약

```
boxing/
├── envs/
│   ├── boxing_env.py       # MuJoCo Gymnasium 환경 (MultiAgent)
│   └── assets/
│       └── boxer.xml       # MJCF 휴머노이드 래그돌 (2인) ← 물리 수정됨
├── training/
│   ├── curriculum.py       # 4단계 커리큘럼 + CurriculumEnv 래퍼
│   ├── reward.py           # 모듈형 보상 함수 (Phase 3+ 복싱 보상 포함)
│   └── self_play.py        # Self-Play 환경 + 콜백
├── train.py                # 메인 학습 진입점 (SB3 PPO)
├── watch.py                # 실시간 MuJoCo 뷰어 (self-play 자동 감지)
├── checkpoints/            # 50K 스텝마다 자동 저장
│   └── old/                # 이전 실험 체크포인트 보관
├── CLAUDE.md               # AI 작업 지침
└── PROGRESS.md             # 이 파일
```

---

## 현재 물리 설정 (boxer.xml) — 2026-03-19 수정

| 항목 | 이전 | 현재 |
|------|------|------|
| default stiffness | 5 | **0** |
| default damping | 2.0 | **1.5** |
| 다리 hip/knee kp | 350 | **150** |
| 발목 kp | 150 | **100** |
| 팔/어깨 kp | 80~100 | 유지 |

**변경 이유**: 이전 설정(stiffness=5, kp=350)에서 ctrl=0(정책 없음)이 186 스텝 생존 가능 → 훈련된 정책(160 스텝)보다 오히려 나음 → 정책이 능동 균형을 배울 필요가 없었음.

---

## PPO 하이퍼파라미터

```python
PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.001,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    target_kl     = 0.05,
    policy_kwargs = {"net_arch": [256, 256, 128]},
)
```

---

## 학습 실행 명령

```bash
# 현재 실행 중 (이어서)
py -3.11 train.py --selfplay --load checkpoints/boxer_selfplay_4050000_steps.zip --total-steps 100000000

# 뷰어 (별도 터미널에서 직접 실행)
py -3.11 watch.py
```

---

## 실험 기록 & 트러블슈팅

### 문제 1 — MuJoCo 뷰어 창이 안 보임
- **원인**: 백그라운드 프로세스에서 GUI 창 표시 안 됨
- **해결**: watch.py는 사용자 본인 터미널에서 직접 실행

### 문제 2 — 에이전트가 쭈그리고 앉음 (보상 해킹)
- **원인**: upright 보상이 선형 → 쭈그리기가 이득
- **해결**: 계단형(quadratic) upright 보상 + XML stiffness 추가

### 문제 3 — 컨트롤이 0에 고정
- **원인**: energy 페널티 너무 강함
- **해결**: `-0.005 → -0.001`

### 문제 4 — std 폭발 (3.8 → 4.4)
- **원인**: ent_coef=0.01 과다
- **해결**: `ent_coef=0.001` + `target_kl=0.05`

### 문제 5 — Phase 전환 시 KL 폭발
- **원인**: 가치함수 불일치
- **해결**: lr=3e-5 + target_kl=0.05 + 가치함수 리셋

### 문제 6 — 훈련된 정책이 ctrl=0보다 나쁨 (핵심 발견)
- ctrl=0: 평균 **186 스텝** 생존
- 훈련된 정책: 평균 **160 스텝** 생존
- **원인**: position actuator kp=350 + stiffness=5가 수동으로 직립 지지
- **해결**: stiffness=0, kp 축소 (350→150)

### 문제 7 — 커리큘럼이 10K 스텝 만에 전 Phase 통과
- **원인**: 새 물리에서 초기 보상이 R=170+으로 높아져서 임계값(80/120/180) 즉시 초과
- **해결**: 커리큘럼 건너뛰고 바로 Self-Play 모드로 전환

### 문제 8 — watch.py에서 에이전트가 완전히 멈춰 보임
- **원인**: 새 물리로 학습한 체크포인트가 아직 없어서 OLD 체크포인트 로드
- **해결**: 50K 스텝 후 새 체크포인트 생성되면 자동 갱신됨

---

## 현재 학습 수치 (2026-03-20 기준)

```
모드: Self-Play (두 에이전트 자기대전)
현재 스텝: ~10,000,000 / 100,000,000 (10%)
속도: ~50K 스텝/분 (n_envs=4 SubprocVecEnv)
예상 완료: 약 25시간 후

보상 추이:
  시작 (0K):    R = 45
  500K:         R = 68
  1M:           R = 70
  2M:           R = 75
  4M:           R = 75
  10M:          R = 96~105 (proximity+approach 보상 추가 후 상승)

에피소드 길이: ~297~299 스텝
KO 발생: YES
HP 데미지: 아직 없음 (주먹질 미발생)

최근 변경사항 (2026-03-20):
  - proximity 보상 0.3→0.8 상향
  - approach 보상 0.5/m 신규 추가
  - ent_coef 0.001→0.0001 (std 억제)
  - n_envs=4 SubprocVecEnv 병렬화
  - 텔레그램 봇 연동 (telegram_bot.py)
```

---

## 다음 단계 / 미구현

- [ ] 장기 모니터링 — R이 100+ 되는지, ep_len이 길어지는지 확인
- [ ] 주먹질 발생 여부 (HP 데미지) 확인
- [ ] ELO 트래킹 (현재 미구현)
- [ ] 과거 체크포인트 리그 (70% 현재 / 30% 과거)
- [ ] WandB 로깅 연동
- [ ] 영상 녹화 (imageio[ffmpeg])
