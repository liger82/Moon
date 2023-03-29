---
layout: post
title: "deepspeed 소개"
date: 2023-03-29
excerpt: ""
categories: [Engineering/Python]
tags : [python, deepspeed, model pruning, pruning, Model Parallelism, Zero Redundancy Optimizer, ZeRO,Distributed Data Parallelism,DDP ]
comments: true
---

DeepSpeed는 딥러닝 모델의 학습과 추론을 가속화하는 PyTorch용 오픈 소스 최적화 라이브러리입니다. 메모리 제약, 느린 학습 시간 등 대규모 모델을 활용하려는 기업과 개발자가 직면한 문제를 해결하고 딥 러닝 워크플로의 전반적인 성능과 효율성을 개선하기 위해 Microsoft에서 설계했습니다. 

이 글에서는 딥 러닝 모델을 최대한 활용하는 데 사용할 수 있는 다양한 기술에 대해 설명합니다.

> <subtitle> Model Pruning </subtitle>

정확도를 유지하면서 학습된 모델의 크기와 복잡성을 줄일 수 있습니다.
딥러닝 모델을 최적화하는 가장 간단하고 효과적인 기술 중 하나는 model pruning 입니다. 몇 가지 방법이 있지만, 대부분의 방법은 모델의 전체 출력에 크게 기여하지 않는 가중치나 연결을 제거하는 것입니다. prining 는 모델에서 중요하지 않은 가중치를 제거하여 모델의 크기를 줄이고 학습 및 추론 속도를 높이는 것입니다. 학습 과정에서 모델이 이러한 가중치나 연결을 제거하도록 유도하는 제약 조건을 적용하여 실제로 pruning 을 통합할 수 있습니다.

딥스피드는 가중치 가지치기, 구조 가지치기, 레이어 가지치기를 포함한 여러 가지 가지치기 기법을 지원하며, 모든 모델 아키텍처에 적용할 수 있습니다.

<br>


> <subtitle> Model Parallelism </subtitle>

여러 GPU 또는 머신에 모델 분산.

모델 병렬화를 사용하면 모델을 여러 머신에 분할하여 더 큰 데이터 세트에 대해 학습할 수 있습니다. 이를 통해 학습 시간을 크게 단축하고 대규모 모델의 성능을 향상시킬 수 있습니다. 이 기법을 사용하면 모델의 여러 부분이 병렬로 처리되므로 실제로 대규모 모델을 단일 머신에서 학습할 때보다 더 빠르게 학습할 수 있습니다. 모델의 여러 부분이 통신 계층(일반적으로 고속 네트워크)을 통해 서로 통신하여 계산을 조정합니다. 이는 모델 크기가 단일 머신의 메모리 용량을 초과할 때 중요한 기술입니다.

딥스피드는 모델 병렬 처리를 지원하므로 여러 GPU 또는 머신에 모델을 쉽게 분산할 수 있으며, 모델의 메모리 사용량도 줄일 수 있습니다.

<br>

> <subtitle> Zero Redundancy Optimizer (ZeRO) </subtitle>

ZeRO는 서로 다른 GPU 간의 통신을 최적화하는 DeepSpeed에서 도입한 새로운 최적화 알고리즘입니다.
이 알고리즘은 중복 매개변수 저장을 제거하여 통신 오버헤드를 줄이고 성능을 개선합니다. 모델의 파라미터를 여러 머신에 분할한 다음 이를 통합하여 메모리 소비를 줄입니다.

최적화 상태 파티셔닝, 최적화 상태 메모리 관리, gradient accumulation 이 세 가지 구성 요소를 사용하여 구현됩니다. 즉, 이러한 구성 요소는 모델의 파라미터를 여러 장치에 분할하고 중복 파라미터 스토리지를 통합한 다음 통신 전에 로컬로 gradient를 누적합니다.

ZeRO는 DeepSpeed의 마이크로 배치 및 모델 병렬 처리 기능과 함께 작동합니다.

<br>

> <subtitle> Distributed Data Parallelism </subtitle>

DDP는 여러 머신에서 대규모 모델 학습의 계산 단계를 병렬화합니다.

딥스피드는 여러 대의 GPU 또는 머신에 걸쳐 계산을 분할하는 분산 데이터 병렬 처리(DDP)도 지원하므로 대규모 데이터 세트에서 모델을 학습할 수 있습니다. DDP를 구현하면 입력 데이터를 여러 머신에 분산하여 병렬로 처리하므로 학습 시간이 단축됩니다. 각 기기에서 계산된 기울기는 평균이 되어 모델의 가중치를 업데이트하는 데 사용됩니다. 이 과정은 모델이 전체 데이터 세트에 대해 학습될 때까지 각 부분에 대해 반복됩니다. DDP는 구현이 더 간단하고 많은 수의 머신으로 확장할 수 있기 때문에 모델 병렬 처리와 구별되며, 점점 더 많이 사용되고 있는 초대형 모델에 더 적합합니다.

간단한 구성 변경으로 DeepSpeed 내에서 DDP를 활성화할 수 있습니다.

<br>

> <subtitle> Hybrid Parallelism </subtitle>

모델 병렬 처리와 DDP의 조합.
딥스피드는 모델 병렬처리와 데이터 병렬처리를 결합한 하이브리드 병렬처리도 지원하므로 대규모 데이터 세트에서 모델을 학습시킬 수 있습니다. 다른 모델 병렬화 기법과 마찬가지로 모델의 일부를 여러 장치에 분할하여 병렬로 처리합니다. 이러한 서로 다른 기술을 결합하면 두 기술의 장점을 함께 활용하여 확장성, 효율성 및 속도를 개선할 수 있습니다.

간단한 구성 변경만으로 DDP 내에서 하이브리드 병렬 처리를 활성화할 수 있습니다.

<br>

> <subtitle> Automatic Mixed Precision Training </subtitle>


<br>

> <subtitle> Wrap up </subtitle>

딥스피드는 딥러닝 모델을 최대한 활용할 수 있도록 도와주는 강력한 최적화 라이브러리입니다. 하지만 이러한 기술을 도입하면 학습 프로세스가 복잡해지고 작업에 추가적인 오버헤드가 발생할 수 있습니다.


<br>

---

> <subtitle> References </subtitle>

* [https://lightning.ai/pages/community/article/using-deepspeed-to-optimize-models/](https://lightning.ai/pages/community/article/using-deepspeed-to-optimize-models/){:target="_blank"}
