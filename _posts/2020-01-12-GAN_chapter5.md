---
layout: post
title: "Generative Deep Learning Chapter 5 : Paint"
date: 2020-01-12
excerpt: "Paint"
tags : [GAN, Generative Deep Learning, Paint, David Foster]
comments: true
---

## 개괄

이전 챕터까지 우리가 따라해보려는 데이터셋 하나를 가지고 새로운 샘플을 만들어낼 수 있는 모델을 훈련하는 여러 방법들을 알아보았다.(VAE, GAN 등)
이번 챕터에서는 여러 데이터셋에 적용해보고, VAEs와 GANs가 내재된 잠재공간과 original pixel space 사이의 매핑을 어떻게 학습하는지 알아볼 것이다.

Generative model의 또 다른 응용분야에는 style transfer 분야가 있다.
여기서 우리의 목적은 입력 기반 이미지를 변형시킬 수 있는 모델을 만들어서 주어진 스타일 이미지 셋 같이 동일한 컬렌션으로부터 온 인상을 주는 것이다.

Figure 5-1. Style transfer examples image

style transfer의 목적은 스타일 이미지의 내재된 분포를 모델링하려는 것이 아니라 **이 이미지들로부터 stylistic component를 추출하고 이 컴포넌트를 베이스 이미지에 내장**하는 것이다.
베이스 이미지에 스타일 이미지를 덧붙이면 스타일 이미지의 컨텐트가 속이 들여다보이고 색깔이 탁하고 희미해져서 이 방식으로는 완전히 합칠 수 없다. 
그리고 한 종류의 스타일 이미지로는 화가의 스타일을 캡쳐할 수 없다. 그래서 그 화가의 전체 이미지에 걸쳐서 스타일(화풍)을 학습할 모델을 만드는 방법을 찾아야 한다.
그 화가가 베이스 이미지를 작품의 원작을 만들고 그 콜렉션 내의 다른 작품에서 동일한 화풍이 반영되도록 안내하는데 사용한다는 인상을 주고 싶다.

이번 챕터에서는 두 가지 종류의 style transfer model(CycleGAN, Neural Style Transfer)을 어떻게 빌드하는지 배우고, 이 기술들을 사진과 작품에 적용해 볼 것이다.
청과물 가게 예시로 시작을 하도록 한다.

## Apples and Oranges


