# Pelvic X-ray Data Augmentation and Sanity Check Using PGGAN

> **Improving Quality of Pelvic X-ray Images with Sanity Check**

## Table of Contents
- [About the Project](#about-the-project)
- [Motivation](#motivation)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## About the Project

본 프로젝트는 PGGAN을 활용해 생성한 **Pelvic X-ray** 영상의 정확성을 검사하는 **sanity check** 기법을 도입함으로써, 생성 의료 이미지의 정합성 향상을 목적으로 합니다. PGGAN(Progressive Growing GAN)을 활용하여 정상 및 비정상 상태의 데이터를 생성한 후, 특정 골반 관절 지표(acetabulum 외측점, subchondral sclerosis 내측점, teardrop 하단점, 대퇴골 중심점)와 각도(Sharp’s angle, center-edge (CE) angle, Tonnis angle)를 기준으로 정확성을 평가했습니다.

## Motivation

Pelvic X-ray 이미지는 골반 질환의 진단 및 평가에 필수적인 자료로 사용됩니다. 하지만 의료 데이터의 부족과 개인정보 보호로 인해 AI 기반 분석을 위한 데이터 확보가 제한적입니다. 이를 해결하기 위해 **GAN(Generative Adversarial Networks)**을 사용하여 고해상도의 데이터를 생성할 수 있으며, PGGAN은 이 중 고해상도 이미지 생성에 안정성을 제공하는 모델로 알려져 있습니다. 본 연구에서는 기존 연구에서 사용되지 않았던 **sanity check**를 생성된 X-ray 이미지에 적용하여, 생성된 데이터의 신뢰성과 정확성을 검증하고자 합니다.

## Methodology

### 데이터 생성 및 Sanity Check
- **PGGAN**: Progressive Growing GAN을 사용하여 단계적으로 해상도를 높이며 고품질 이미지를 생성합니다.
- **Sanity Check 지표**: 생성된 이미지의 정확성 평가를 위해 골반 이미지에서 특정 점(OPAC, IPSS, IPTE, FHCE)과 각도(Sharp's angle, CE angle, Tonnis angle)를 분석합니다.
- **모델 학습**: `fastai` 및 `U-Net` 모델을 사용하여 골반의 특정 점을 예측하는 모델을 학습합니다. ResNet34를 U-Net의 인코더로 사용하여 높은 학습 효율성을 확보하고자 하였습니다.

### 데이터 처리 및 모델 평가
1. **데이터 생성**: `PyTorch GAN Zoo`를 사용해 PGGAN 기반 고해상도 Pelvic X-ray 이미지 생성
2. **이미지 라벨링**: COCO Annotator 도구를 활용하여 실제 X-ray 이미지에서 8개의 주요 포인트 라벨링 수행
3. **데이터 분할 및 모델 학습**: 7:2:1 비율로 데이터를 훈련, 테스트, 검증용으로 분할하여 모델 학습 및 평가

### 모델 아키텍처
- **U-Net**: 의료 영상 분석에 적합한 U형 네트워크 구조로, 픽셀 단위의 정확한 예측을 수행할 수 있습니다.
- **ResNet34**: U-Net의 인코더로 사용되어 이미지의 복잡한 패턴을 학습합니다.
- **평가 지표**: 좌표 추정 모델의 학습과 검증에는 **MSE(Mean Squared Error)**가 사용되었습니다.

## Installation

1. **Python 및 패키지 설치**:  
   본 프로젝트는 `Python 3.*`에서 실행되며, `requirements.txt` 파일에 명시된 모든 패키지를 설치합니다.

   ```bash
   conda env create -f requirements.yml
   pip install fastai==1.0.61
   ```
2. Git LFS 활성화:
   큰 파일을 관리하기 위해 Git LFS를 사용합니다.

   ```bash
   git lfs install
   git lfs pull
   ```

 ## Results

1. 모델 성능 지표:
	MSE 손실: 남성 0.000302, 여성 0.000455 (500 epochs)
	정상 각도 기준: 남성 및 여성 각각의 Sharp’s angle, CE angle, Tonnis angle의 Q1 및 Q3 범위를 기준으로 정의
2. 산출된 데이터 분석:
   	남성 5000개, 여성 5000개에서 각각 215개, 174개의 이미지가 정의된 정상 각도 범위에 부합
   	검증된 이미지를 통해 고품질 데이터셋을 구축하여 향후 분류, 예측 및 세분화 작업에 사용할 수 있음
