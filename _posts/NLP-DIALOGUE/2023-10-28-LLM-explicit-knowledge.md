---
layout: post
title: "[paper review] How Do Large Language Models Capture the Ever-changing World Knowledge? A Review of Recent Advances"
date: 2023-10-28
excerpt: ""
categories: [NLP/Dialogue]
tags : [virtual assistant, dialogue system, paper review, LLMs, knowledge]
comments: true
---

>Authors : Zihan Zhang, Meng Fang, Ling Chen, Mohammad-Reza Namazi-Rad, Jun Wang  

>Institution : University of Technology Sydney, University of Liverpool, 
University of Wollongong, University College London  

>Publication Date : Oct 11, 2023   

>Paper link : [https://arxiv.org/abs/2310.07343v1](https://arxiv.org/abs/2310.07343v1){:target="_blank"}  

---

<br>

> <subtitle> 소개 </subtitle>

이 논문은 계속 변화하는 세상의 지식을 어떻게 처리할 지에 대한 최근 연구들을 소개합니다.
지식을 처리하는 방식을 fig2 처럼 크게 2가지 방법으로 분리합니다.

<center><img src= "https://liger82.github.io/assets/img/post/20231028-LLM-explicit-knowledge/fig2.png" width="90%"></center><br>

1. Implicit(내재적 방식): 모델 학습 방식 
2. Explicit(외재적 방식): 외부 지식 활용 방식

요즘은 1, 2번 둘 다 활용하는 추세고 둘 다 필요하다는 입장이지만 
제가 하는 일이 2번이고 현재 2번에서 성능 개선을 바라고 있기 때문에 2번에 대해서만 각 방식의 특징과 장단점에 대해 다루도록 하겠습니다.

<br>

외재적 방식의 공통점은 다음 table1에서 확인할 수 있습니다.

<center><img src= "https://liger82.github.io/assets/img/post/20231028-LLM-explicit-knowledge/table1.png" width="90%"></center><br>

* LM params 고정: 변화하는 지식에 대한 대처를 외부 지식에 의존하기 때문에 Original LM 혹은 Base LM 학습을 하지 않습니다.  
* 추가 학습 안함: (SERAC를 제외하고는) 추가 학습을 하지 않는다.
* Black-box: 공개되지 않은 모델에 적합한지 여부(예: 모델 아키텍처, 매개변수, 활성화 또는 그라데이션을 사용할 수 없는 경우) -> 모두 함께 활용 가능

이런 특징들을 고려할 때 외재적 방식은 학습에 돈을 아낄 수 있다는 점에서 값싼 방식입니다.

기존 RAG 모델은 retrieval 모델을 LM과 함께 학습하였지만 이는 공개된 LLM(e.g. ChatGPT)에 적용하기 어렵게 만들었습니다.

(이것은 무슨 의미일까요??)
잠깐 찾아봤을 땐 retrieval 모델과 LM이 학습 데이터가 달라서? 학습 목표가 달라서? 두 개를 합치해서 사용하는 것이 어렵다 라는 의견을 봤는데 그럼 외재적 방식이 다 해당되는게 아닌가 싶네요.


외재적 방식의 3가지 방법은 LLM은 고정하고 

1. 외부 메모리를 사용하거나(Memory-enhanced) 
2. 기존에 존재하는 retrival을 사용하거나(Retrieval-enhanced) 
3. 인터넷을 사용하는 방식으로(Internet-enhanced)

변화하는 지식에 대처합니다.

<br>

> <subtitle> Memory-enhanced 방식 </subtitle>

정적인 LLM과 증가하는 비모수 메모리를 함께 사용하면 추론 중에 암기된 지식 이상의 정보를 잡아낼 수 있습니다. 외부 메모리는 모델 생성에 도움이 되는 새로운 정보가 포함된 최근 말뭉치 또는 피드백을 저장할 수 있습니다.

<br>

### kNN-LM: 코퍼스/문서 저장

* 메모리에 key-value 형태로 모든 <context, token> 을 저장
* 추론할 때, 메모리에 있는 k개의 가까운 토큰에서 추출한 분포로 고정된 LLM을 보간(interpolate)하여 다음 토큰의 확률을 계산
    - 보간하다는 말은 두 개의 값을 이용하여 사이에 존재할 법한 값을 추정하는 것을 의미
    - 고정된 LLM에서 "오늘 아침에" 다음 토큰을 예측한다고 했을 때 "일어났다"가 확률이 높다고 가정하자. 메모리에서는 "아침에 늦잠을 잤다" 라는 내용이 있다면 "일어났다" 보다 잠에 대한 내용이 반영되게 한다는 의미




During inference, it calculates the probability of the next token by interpolating a fixed LM with a distribution retrieved from the k nearest tokens in the memory. 

> <subtitle> Retrieval-enhanced 방식 </subtitle>


<br>

> <subtitle> Internet-enhanced 방식 </subtitle>


<br>


<br>

---

> <subtitle> References </subtitle>

* [](){:target="_blank"}










