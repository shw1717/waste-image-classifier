# 폐기물 이미지 분류 CNN — Keras & PyTorch 구현

인공지능개론 수업 프로젝트로, 동일한 CNN 구조를 **Keras**와 **PyTorch** 두 프레임워크로 각각 구현하여 생활 폐기물 이미지를 5개 클래스로 분류한 실험입니다. 단순히 두 프레임워크의 성능을 비교하는 것이 아니라, 구현 과정에서 마주친 문제(클래스 불균형, 프레임워크 간 정확도 격차, 학습 속도 저하)를 어떻게 발견하고 원인을 추정하고 해결했는지를 중심으로 정리했습니다.

> ⚠️ 본 저장소의 결과는 **1회(및 일부 다중 시드) 실험**을 기반으로 하며, 모든 수치는 재현 환경(GPU, 라이브러리 버전, 시드)에 따라 달라질 수 있습니다. 아래 서술은 확정적 결론이 아니라 관찰된 경향으로 이해해 주세요.

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [문제 정의](#문제-정의)
- [데이터셋](#데이터셋)
- [CNN 구조](#cnn-구조)
- [실험 과정 (1차~4차)](#실험-과정-1차4차)
- [최종 결과](#최종-결과)
- [오류 분석](#오류-분석)
- [기술 스택](#기술-스택)
- [저장소 구조](#저장소-구조)
- [실행 방법](#실행-방법)
- [한계 및 향후 개선 방향](#한계-및-향후-개선-방향)

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 과제 | 생활 폐기물 이미지 5-클래스 분류 (종이/유리/금속/플라스틱/일반쓰레기) |
| 구현 | 동일한 CNN 구조를 Keras(TensorFlow 백엔드)와 PyTorch로 각각 구현 |
| 데이터셋 | TrashNet (Hugging Face `datasets`, garythung/trashnet) |
| 핵심 비교축 | 정확도·추론 속도·학습 속도·파라미터 수 |
| 산출물 | Colab 노트북, 발표 PPT, 발표 대본 |

## 문제 정의

프로젝트를 진행하며 아래 두 가지 문제를 중심으로 다뤘습니다.

- **문제 1 (클래스 불균형)**: 학습 데이터의 클래스별 개수가 최대 7:1까지 차이가 나는 상황에서도 소수 클래스(일반쓰레기 등)를 과도하게 희생시키지 않고 안정적으로 분류해야 했습니다.
- **문제 2 (프레임워크 간 일관성)**: 구현 방식(Keras·PyTorch)이 달라도 비슷한 정확도를 내는지 확인이 필요했고, 실제로는 초기 구현에서 두 프레임워크 간 정확도 격차(약 7.24%p)가 나타나 그 원인을 찾아야 했습니다.

## 데이터셋

- **TrashNet**: 종이(paper), 유리(glass), 금속(metal), 플라스틱(plastic), 골판지(cardboard), 쓰레기(trash) — 총 6개 원본 클래스
- 과제 목표에 맞춰 `cardboard`를 `paper`로 통합하여 **5개 클래스**로 재구성
- 전체 5,054장 → **종이 1,994 / 유리 1,002 / 금속 820 / 플라스틱 964 / 일반쓰레기 274**
- Train/Val/Test = 70/15/15 (stratified) → 3,537 / 758 / 759장
- 입력 크기: 128×128×3

## CNN 구조

Keras와 PyTorch 모두 **동일한 구조**로 구현했습니다 (파라미터 수 3,305,285개로 동일 확인).

```
Conv(32, 3x3) → ReLU → MaxPool(2x2)
Conv(64, 3x3) → ReLU → MaxPool(2x2)
Conv(128, 3x3) → ReLU → MaxPool(2x2)
Flatten(25,088)
Dense(128) → ReLU → Dropout(0.5)
Dense(5) → Softmax
```

- Optimizer: Adam
- Loss: Cross-Entropy
- Batch size: 32, 최대 30 epoch (Early Stopping patience=5)
- 클래스 가중치: `total / (n_classes × class_count)`를 제곱근으로 완화하여 적용

## 실험 과정 (1차~4차)

문제를 발견하고 → 구현하고 → 판단하고 → 결과를 내는 과정을 4단계로 정리했습니다.

| 단계 | 내용 | 관련 문제 | 결과 |
|---|---|---|---|
| 1차 | Baseline (클래스 가중치 미적용) | 문제 2 | Keras 68.12% · PyTorch 75.36% (격차 7.24%p) |
| 2차 | 원본 비율(약 3.68배) 그대로 클래스 가중치 도입 | 문제 1 | PyTorch 75.36% → 67.72% (다수 클래스 성능 저하) |
| 3차 | 가중치를 제곱근으로 완화(3.68배 → 1.92배) | 문제 1 | 격차 7.24%p → 0.79%p로 수렴 (Keras 77.08% · PyTorch 77.87%) |
| 4차 | PyTorch 데이터 증강을 CPU(`torchvision.transforms`) → GPU 배치 연산(`affine_grid`+`grid_sample`)으로 전환 | 속도 최적화 (구현 중 추가로 발견) | 학습 시간 166.98초 → 45~75초 (최대 약 3.7배 단축), 결과 일관성 확인 |

제곱근을 택한 이유는, 값이 클수록 더 크게 줄어드는 특성 때문에 극단적인 보정은 완화하면서도 소수 클래스를 배려하는 효과는 유지할 수 있다고 판단했기 때문입니다.

4차 단계는 원래 두 문제(클래스 불균형·프레임워크 일관성)와는 별개로, 3차 구현 과정에서 PyTorch 학습이 유독 느리다는 점을 추가로 발견해 해결한 것입니다. 원인은 `torchvision.transforms`가 CPU에서 샘플 단위로 증강을 수행해 GPU 연산 중간에 병목이 생기는 것으로 추정했고, 이를 배치 전체를 GPU 위에서 한 번에 변형하는 방식으로 바꿔 개선했습니다.

## 최종 결과

| 지표 | Keras | PyTorch |
|---|---|---|
| 테스트 정확도 | 77.08% | 77.87% |
| 파라미터 수 | 3,305,285 | 3,305,285 |
| 이미지당 추론 시간 | 1~2ms 수준 | 1~2ms 수준 |
| 학습 시간(4차 적용 후) | — | 45~75초 (seed별 변동) |

- 정확도·파라미터 수는 구현 방식이 달라도 비슷하게 나왔습니다.
- 추론 속도는 두 구현 모두 단일 이미지 기준 빠른 편이었습니다.
- 다중 시드(42, 7, 123) 검증 결과, PyTorch 쪽이 Keras보다 시드 간 정확도 변동(표준편차 약 3.56%p vs 0.56%p)이 다소 더 크게 나타났습니다 — 다만 시드 3개로 내린 결론이라 추가 검증이 필요합니다.
- 추론 속도 차이의 원인은 단정하기 어렵지만, Keras `predict()`의 `tf.function` 재추적 오버헤드(TensorFlow 공식 이슈 트래커에 보고된 사례)와 PyTorch의 `eval()`/`no_grad()` 구조 차이 등이 영향을 줬을 가능성이 있다고 판단했습니다.

결과 이미지는 [`results/`](./results) 폴더에 있습니다 (학습 곡선, Confusion Matrix, 클래스별 Precision/Recall/F1, 오분류 샘플 등).

## 오류 분석

| 문제 | 원인 추정 | 관찰 | 개선 방법(검토) |
|---|---|---|---|
| 일반쓰레기 등 소수 클래스의 Recall이 상대적으로 낮음 | 학습 데이터 수 부족(274장, 최대 클래스의 약 1/7) | 혼돈 행렬에서 일반쓰레기가 종이로 오분류되는 비중이 높음 (`results/13_misclassified_sample.png` 참고) | Class Weight 추가 조정, Data Augmentation 강화, Transfer Learning, 추가 데이터 확보 |

## 기술 스택

- Python, Google Colab (GPU)
- TensorFlow/Keras (`tf.keras`)
- PyTorch, `torch.nn.functional` (GPU 배치 증강)
- Hugging Face `datasets` (TrashNet 로딩, `datasets==2.19.1`로 고정 — 이후 버전에서 로딩 오류 확인됨)
- scikit-learn (metrics), matplotlib/seaborn (시각화)

## 저장소 구조

```
waste-image-classifier/
├── README.md
├── notebook/
│   └── waste_classifier_keras_pytorch.ipynb   # 전체 실험 코드 (Colab)
├── presentation/
│   ├── waste_classifier_presentation.pptx     # 발표 슬라이드
│   ├── presentation_script_10min.docx         # 10분 발표 대본
│   └── presentation_script_full.docx          # 상세 대본 + 배경지식 + Q&A 대비
└── results/
    ├── 01_class_distribution.png
    ├── 02_sample_images.png
    ├── 03_keras_learning_curves.png
    ├── 04_keras_confusion_matrix.png
    ├── 05_pytorch_learning_curves.png
    ├── 06_accuracy_comparison.png
    ├── 07_loss_comparison.png
    ├── 08_pytorch_confusion_matrix.png
    ├── 09_overall_comparison_table.png
    ├── 10_overall_comparison_chart.png
    ├── 11_per_class_metrics_table.png
    ├── 12_per_class_metrics_chart.png
    └── 13_misclassified_sample.png
```

## 실행 방법

1. `notebook/waste_classifier_keras_pytorch.ipynb`를 Google Colab에서 엽니다 (GPU 런타임 권장).
2. 셀을 순서대로 실행합니다. 첫 셀에서 `datasets==2.19.1`을 설치하는데, 이는 최신 버전에서 TrashNet 로딩 시 `Invalid string class label` 오류가 발생하는 것을 피하기 위함입니다.
3. 데이터셋 다운로드 셀 실행 후 오류가 나면 런타임을 재시작하고 다시 시도하거나, `force_redownload` 옵션을 사용해 보세요.
4. 전체 실행에는 GPU 기준 약 10~15분 정도가 소요됩니다(환경에 따라 다를 수 있음).

## 한계 및 향후 개선 방향

- **성능 개선**: Transfer Learning(MobileNet, ResNet 등 사전학습 모델 활용) 적용 검토
- **신뢰성 개선**: 다중 시드 실험을 통해 성능의 평균과 표준편차를 더 폭넓게 검증할 예정
- **해석 가능성**: Grad-CAM 등으로 모델이 어떤 영역을 보고 판단하는지 시각화 검토

---

*본 프로젝트는 인공지능개론 수업의 일환으로 진행되었으며, 실험 결과는 특정 환경(Colab GPU, 특정 라이브러리 버전, 제한된 시드 수)에서의 1회성 관찰에 가깝습니다. 실제 서비스 적용 가능성에 대한 결론이 아니라, 학습 목적의 비교 실험으로 봐 주시면 좋겠습니다.*
