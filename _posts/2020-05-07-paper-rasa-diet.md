---
layout: post
title: "[paper review] DIET : Lightweight Language Understanding for Dialogue Systems"
date: 2020-05-07
excerpt: ""
tags : [virtual assistant, chatbot, lightweight dialogue system, Rasa, 라사, paper review]
comments: true
---

>Authors : Tanja Bunk, Daksh Varshneya, Vladimir Vlasov, Alan Nichol  
>Institution : Rasa  
>Publication Date : April 22, 2020  
>Paper link : [https://arxiv.org/abs/2004.09936](https://arxiv.org/abs/2004.09936){:target="_blank"}  
>Github : [https://github.com/RasaHQ/DIET-paper](https://github.com/RasaHQ/DIET-paper){:target="_blank"}  


# Abstract

큰 규모의 사전학습된 언어 모델들이 GLUE 와 SuperGLUE 와 같은 언어 이해 벤치마크에서 인상적인 결과들을 내고 있다.
이 논문에서 Dual Intent and Entity Transformer(DIET) architecture 를 소개하면서, 
인텐트와 엔티티 예측에서 그 효과성을 보이려고 한다. DIET 는 복잡한 멀티 도메인 NLU dataset 에서 SOTA 를 이뤄냈고
다른 더 간단한 데이터셋에서 비슷하게 높은 성능을 보였다.  
놀랍게도 이 과제에서 큰 규모의 사전학습 모델을 사용하는 것이 확실히 이득이 아님을 보였고, 
DIET 는 어떠한 사전학습 임베딩을 사용하지 않고 순수하게 지도학습 환경에서 SOTA 를 이뤄내는 성과를 만들었다.
DIET 는 fine-tuning BERT 의 성능을 뛰어넘었고, 학습은 6배나 빨랐다.


---

# 1. Introduction

data-drive dialogue modeling 접근법에는 보통 end-to-end 와 modular system 이 있다.
Modular 접근법은 NLU 와 NLG 시스템을 분리하여 사용한다. 예시로는 POMDP 기반 대화 정책과 Hybrid Code Networks 이 있다.

