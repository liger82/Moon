---
layout: post
title: "[paper review] Transformer-XL : Attentive  Language Models Beyond A Fixed-Length Context"
date: 2019-11-23
excerpt: "paper review : Transformer-XL"
tags : [paper review, TransformerXL, NLP]
comments: true
---

기존에 논문 리뷰를 스터디를 통해 해왔었고 이를 저만 보는 것이 아니라 공개하고 싶어서 github page로 남깁니다. 
psygrammer라는 그룹에서 '바벨봇' 스터디를 통해 자연어 처리 논문 리뷰를 격주로 진행하고 있습니다. 
이 논문은 19년도 초에 리뷰했던 것으로 기억합니다.


# <center>Transformer-XL : Attentive  Language Models Beyond A Fixed-Length Context</center>

Zihang Dai,    Zhilin Yang,   Yiming Yang,   William W. Cohen, 
Jaime Carbonell,   Quoc V. Le ,    Ruslan Salakhutdinov
Carnegie Mellon University,    Google Brain,    Google AI
 

## Abstract
* Transformer networks는 장기의존성을 학습할 잠재성을 지녔지만 language modeling에서 고정된 길이의 context에 의해 제한된다는 단점이 있다. 
* 해결책으로 새로운 neural architecture인 Transformer-XL를 제안
    * 시간적 일관성을 망치지 않고 transformer가 고정된 길이의 한계를 넘어서 의존성을 학습하도록 만듦
    * segment-level recurrence mechanism과 새로운 위치 인코딩 체계로 구성
    * 장기 의존성 잡아내고, context fragmentation 문제도 해결
* [성능] Transformer-XL은 
    * RNNs보다 약 80 % 더 긴 의존성을 학습
    * vanilla Transformers보다 450% 더 긴 의존성을 학습
    * 평가에서도 vanilla Transformer보다 짧고, 긴 시퀀스 모두에서 좋은 성능을 보였고 평가속도도 1,800배 이상 빠름
    * 최첨단의 결과를 향상시킴(bpc/perplexity)
        * enwiki8 : 1.06 -> 0.99
        * text8 : 1.13 -> 1.08
        * WikiText-103 : 20.5 -> 18.3
        * One Billion Word : 23.7 -> 21.8
        * Penn Treebank : 55.3 -> 54.5
* Pretrained Model
* hyperparameters are available in both Tensorflow and PyTorch
* github : [https://github.com/kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl)
* paper : [https://arxiv.org/abs/1901.02860](https://arxiv.org/abs/1901.02860)



