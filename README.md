# DisCorrGAN: 다변량 시계열 데이터 생성을 위한 상관관계 인식 GAN

## 📋 프로젝트 개요

DisCorrGAN은 다변량 금융 시계열 데이터의 상관관계를 보존하면서 고품질의 합성 데이터를 생성하는 GAN 기반 모델입니다. 이 프로젝트는 Temporal Convolutional Network(TCN)와 Self-Attention 메커니즘을 결합하여 실제 금융 데이터의 복잡한 상관 구조를 학습하고 재현합니다.

## 🎯 주요 특징

- **상관관계 보존**: 자산 간 상관계수를 명시적으로 학습하여 실제 데이터의 통계적 특성 유지
- **TCN + Self-Attention 아키텍처**: 시계열 데이터의 장기 의존성을 효과적으로 모델링
- **WGAN-GP 기반 훈련**: 안정적인 GAN 훈련을 위한 Wasserstein Loss와 Gradient Penalty 적용
- **포트폴리오 전략 평가**: 생성된 데이터의 실용성을 검증하기 위한 다양한 거래 전략 구현
- **다변량 금융 데이터**: 6개 주요 금융 지수(다우존스, 나스닥, JPM, 항셍, 금, WTI) 데이터 활용

## 🏗️ 프로젝트 구조

```
DisCorrGAN/
├── configs/
│   └── config.yaml              # 모델 설정 파일
├── data/
│   ├── data_crawling.py         # 금융 데이터 크롤링 스크립트
│   ├── indices.csv             # 수집된 금융 지수 데이터
│   ├── ref_log_return.pkl      # 참조 로그 수익률 데이터
│   └── ref_price.pkl           # 참조 가격 데이터
├── src/
│   ├── baselines/
│   │   ├── networks/
│   │   │   ├── generators.py    # TCN 기반 생성자 네트워크
│   │   │   ├── discriminators.py # TCN 기반 판별자 네트워크
│   │   │   └── tcn.py          # Temporal Convolutional Network 구현
│   │   └── trainer.py          # GAN 훈련 로직
│   ├── evaluation/
│   │   ├── eval_gen_quality/
│   │   │   └── gen_quality.py  # 생성 품질 평가
│   │   └── eval_portfolio/
│   │       ├── strategies.py   # 거래 전략 구현
│   │       ├── loss.py         # 포트폴리오 손실 함수
│   │       └── summary.py      # 평가 결과 요약
│   ├── preprocess/
│   │   ├── gaussianize.py      # 데이터 정규화
│   │   └── __init__.py
│   └── utils.py                # 유틸리티 함수
├── outputs/                    # 생성된 결과물 (그래프, 통계)
├── results/                    # 훈련된 모델 저장소
├── etc/                       # 실험용 노트북들
└── wandb/                     # Weights & Biases 로그
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 프로젝트 클론
git clone [repository-url]
cd DisCorrGAN

# 가상환경 생성 및 활성화
conda create -n discorrgan python=3.8
conda activate discorrgan

# 필요한 패키지 설치
pip install torch torchvision torchaudio
pip install wandb yfinance pandas numpy scikit-learn
pip install ml-collections pyyaml tqdm
```

### 2. 데이터 준비

```bash
# 금융 데이터 수집
cd data
python data_crawling.py
```

### 3. 모델 훈련

```bash
# 기본 설정으로 훈련 시작
python -m src.baselines.trainer

# 또는 설정 파일 수정 후
python -m src.baselines.trainer --config configs/config.yaml
```

### 4. 결과 평가

```bash
# 생성 품질 평가
python -m src.evaluation.eval_gen_quality.gen_quality

# 포트폴리오 전략 평가
python -m src.evaluation.eval_portfolio.summary
```

## ⚙️ 주요 설정 파라미터

`configs/config.yaml`에서 다음 파라미터들을 조정할 수 있습니다:

```yaml
# 데이터 설정
n_vars: 6                    # 변수 개수 (금융 지수 개수)
n_steps: 256                 # 시계열 길이

# 훈련 설정
batch_size: 256              # 배치 크기
n_epochs: 200                # 에포크 수
lr_G: 0.0002                 # 생성자 학습률
lr_D: 0.0001                 # 판별자 학습률

# 모델 설정
hidden_dim: 48               # 은닉층 차원
noise_dim: 3                 # 노이즈 차원
corr_loss_type: 'l1'         # 상관관계 손실 함수 ('l1', 'l2', 'fro')
corr_weight: 1.0             # 상관관계 손실 가중치
rampup_epochs: 40            # 상관관계 손실 램프업 기간
```

## 🧠 모델 아키텍처

### 생성자 (Generator)
- **TCN 기반**: Temporal Convolutional Network로 시계열 패턴 학습
- **Self-Attention**: 장기 의존성 모델링을 위한 어텐션 메커니즘
- **Multi-head Attention**: 4개 헤드로 다양한 시간적 관계 포착

### 판별자 (Discriminator)
- **TCN + Self-Attention**: 생성자와 유사한 아키텍처
- **WGAN-GP**: Wasserstein Loss와 Gradient Penalty로 안정적 훈련
- **Spectral Normalization**: 훈련 안정성 향상

### 상관관계 손실 함수
```python
def correlation_loss_pair(fake_i, fake_j, real_i, real_j, corr_loss_type):
    # 생성된 데이터와 실제 데이터 간 상관계수 차이 계산
    # L1, L2, 또는 Frobenius norm으로 손실 계산
```

## 📊 평가 방법

### 1. 생성 품질 평가
- **통계적 유사성**: 분포, 자기상관함수, 상관계수 비교
- **시각적 분석**: 시계열 플롯, 분포 히스토그램, 상관관계 히트맵

### 2. 포트폴리오 전략 평가
- **균등가중 포트폴리오**: Buy & Hold 전략
- **평균회귀 전략**: 이동평균 기반 역추세 전략
- **추세추종 전략**: 단기/장기 이동평균 교차 전략
- **변동성 거래 전략**: 변동성 임계값 기반 거래

## 📈 주요 결과

- **상관관계 보존**: 실제 데이터와 생성 데이터 간 상관계수 오차 < 5%
- **통계적 유사성**: 분포 및 자기상관함수에서 높은 유사성 달성
- **실용성 검증**: 다양한 거래 전략에서 실제 데이터와 유사한 성과

## 🔬 실험 및 분석

프로젝트는 다양한 실험 설정을 포함합니다:

- **시드별 실험**: 재현성을 위한 여러 시드 값으로 실험
- **Ablation Study**: 상관관계 손실 및 Self-Attention 제거 실험
- **하이퍼파라미터 튜닝**: 학습률, 손실 가중치, 모델 구조 최적화

## 📝 참고 문헌

- Wasserstein GAN with Gradient Penalty
- Temporal Convolutional Networks for Action Segmentation
- Attention Is All You Need
- Multivariate Time Series Generation with Generative Adversarial Networks

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📧 연락처

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요.

---

**DisCorrGAN** - 다변량 금융 시계열 데이터 생성의 새로운 패러다임
