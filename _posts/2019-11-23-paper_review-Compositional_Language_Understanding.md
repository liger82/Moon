---
layout: post
title: "[paper review] Compositional Language Understanding with Text-based Relational Reasoning"
date: 2019-11-23
excerpt: "paper review : Compositional Language Understanding with Text-based Relational Reasoning"
tags : [paper review, Compositional_Language_Understanding,reasoning, NLP]
comments: true
---


# <center>Compositional Language Understanding with Text-based Relational Reasoning</center>

<p align="center">Koustuv Sinha, Shagun Sodhani, William L. Hamilton, Joelle Pineau</p>

McGill Univ(Canada), Univ de Montreal(Canada) & Facebook AI Research(FAIR)


# Abstract
>자연어 추론(reasoning)을 위한 신경망은 추출 위주의 사실 기반 QA와 상식 추론(inference)에 초점 맞춰져 왔다. 그러나 신경망이 자연어로부터 관계형 추론과 조합 일반화를 수행할 수 있는 정도를 이해하는 것도 중요하다. 이는 표준 QA 벤치 마크에서 주석 artifacts 및 언어 모델링의 우세로 인해 종종 가려지는 능력이다. 
관계형 추론에서 성능(performance : 앞서 말한 뛰어난 성능을 지닌 주석 artifacts와 언어모델링)을 떼어놓고, 언어 이해를 위한 새로운 벤치마크 데이터셋을 제공한다. 또한 neural message-passing baseline도 제공한다. 관계형 귀납적인 편향(bias)을 결합한 이 모델은 전통적인 RNN 접근법보다 조합 일반화에서 우수하였다.

# 1. Introduction

* NLU 시스템은 QA와 같은 정보 추출 과제에서 굉장히 좋은 성과를 내고 있다. 
    * 기존 데이터셋 배열을 이용할 수 있다는 것은 사실적인 대답 텍스트를 추출하는 시스템 능력과 단순하고 상식적인 추리를 강조하는 데이터셋을 테스트할 수 있다는 의미다.
* 하지만! 기존 데이터셋 사용을 배제한 채로 모델의 추론 능력을 평가하는 것은 어렵다
* 대부분 데이터셋은 몇몇 도전적인 언어처리 문제를 하나로 결합한 것이다. 
* 게다가 기존 벤치마크(대부분 데이터셋)에 대해 최신기술은 reasoning이 아니라, 거대하고 사전 학습된 언어 모델에 상당히 의존하고 이 데이터셋의 주요 난제를 자연어 통계치를 결합하는 것으로 강조한다.


* 본 연구자들은 QA 시스템의 구성적 추론 능력을 직접 평가하고 혁신하는 것을 보았다.
* CLEVR(Compositional Language and Elementary Visual Reasoning)에 영감을 받아서 CLUTRR(Compositional Language Understanding with Text-based Relational Reasoning)를 위한 텍스트 기반 데이터셋을 제안하였다.
    * CLEVR은 관계형 추론의 난제들을 배제한 합성 컴퓨터 비전 데이터셋
    * 최초버전인 CLUTRR v0.1은 친족 관계에 대한 추론과 일반화를 필요로 한다
    * 미래에 과제 셋을 확장하기 위해 제안한 데이터 생성 파이프라인을 사용하도록 계획하였다.
    * LSTM 모델과 message-passing GNN(Graph neural network)을 포함한 강력한 베이스라인을 만들어 초기버전에서 평가하였다.
    * 결과는 강력한 관계형 귀납적 편향을 결합한 GNN이 조합 일반화를 요구하는 과제에서 LSTM을 능가하는 것을 보여주었다.
