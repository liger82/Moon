---
layout: post
title: "[paper review] Non-autoregressive dialog state tracking"
date: 2020-04-16
excerpt: ""
tags : [virtual assistant, chatbot, Rasa, 챗봇, tutorial, 설치]
comments: true
---

Author : Hung Le, Richard Socher†, Steven C.H. Hoi
Institution : Salesforce Research, Singapore Management University
Publication Date : Feb 19, 2020
Conference Paper at ICLR 2020
---

# Abstract

태스크 중심 대화(task oriented dialogues)를 위한 대화 상태 추적(Dialogue State Tracking, DST) 영역에서 최근 노력들은 오픈 어휘사전, 생성 기반 접근법(모델이 대화 이력 자체에서 slot value 후보를 
만들어낼 수 있음)을 향하고 있다. 이러한 접근법은 특히, 역동적인 slot value를 가지고 있는 복잡한 대화 도메인에서 좋은 성과를 보였다. 
하지만 두 가지 측면에서 부족한데 다음과 같다.
* 모델이 (domain, slot) pair 간 잠재적인 의존성을 감지해내기 위해 domain과 slot에서 신호를 명백하게 배우도록 하지 못한다. 
* 기존에 모델은 대화가 여러 도메인과 여러 회차(대화를 주고 받는 횟수)로 변화하면 많은 시간을 잡아먹는 auto-regressive 접근법을 따른다.

이 논문에서는 auto-regressive 하지 않은 DST framework를 제안한다.

**NADST의 장점**    
    * domain과 slot 간의 의존성 factor를 최적화하여 대화 상태를 더 잘 예측하게 한다.(분리된 slot 형태로 예측하는 것보다)
    * non-autoregressive 특성
        - 실시간 대화 응답을 생성할 때, DST의 대기시간(latency)를 줄이기 위해 병렬로 디코딩을 할 수 있게 했다.
        - 토큰 레벨 뿐만 아니라 slot, domain 레벨에서도 slot 사이의 의존성을 감지

* 결과
    * MultiWOZ 2.1 corpus 모든 도메인에서 SOTA를 달성
    * 시간이 지남에 따라 대화 기록이 늘어날수록 지연되는 시간(latency)이 이전 SOTA 모델보다 줄어들었다.
    

# 1. Introduction

task oriented dialogues 에서 에이전트는 식당 찾기나 호텔 예약 같은 일을 한다. 
사용자의 발화에는 한 대화 도메인에서 slot으로 인식될 수 있는 중요한 정보가 있다. 대화 상태를 (slot, value) 쌍으로 표현한다.
DST는 이 대화 상태를 추적하여 사용자의 목적을 알아내는 것을 목표로 한다. task oriented dialogues에서 중요한 부분이다.
"주변 극장 알려줘"라는 발화가 있다. slot은 장소 타입이고, value는 극장이다. (장소, 극장)으로 표현할 수 있다.

DST 모델은 2가지로 나뉜다.
1. fixed-vocabulary
    - 구축된 slot ontology를 이용해서 각 대화 상태 후보를 생성한다
2. open-vocabulary
    - 대화 기록에서 엔티티 이름이나 시간 같은 slot으로 구성된 대화 상태 후보를 생성한다.
    - 최근에 대두

기존 open-vocabulary DST model은 autoregressive encoder와 decoder를 사용해왔다.
    - autoregressive : 대화기록을 순서대로 인코딩하고 특정 토큰 $$ t_{i} $$ 를 생성하기 위해 $$ t_{i} $$ 이전 발생한 토큰과 조건부 확률을 계산한다.
autoregressive하면 시간이 오래 걸린다는 단점이 있다.
* 시간은 다음 조건에서 늘어난다.
    1. 대화 기록 길이가 길수록(대화 턴 횟수가 많을수록)
    2. slot values가 길수록
    3. 여러 도메인일수록

   
앞선 문제가 NMT(Neural Machine Translation) 영역에서도 발생하여 참고하였다.
convolution과 attention 같은 신경망 네트워크를 적용하여 latency 문제를 개선하였고 여러 non-autoregressive, semi-autoregressive
접근법이 대상 언어의 토큰으로 독립적으로 생성하는 것을 목표로 했다. 
이러한 연구에 영감을 받아 모델 성능을 떨어뜨리지 않고 DST model의 time cost를 최소화하기 위해 non-autoregressive 접근법을 사용하였다.

Non-autoregressive model을 만들기 위해 fertility 개념을 적용한다.
fertility는 non-autoregressive decoding 동안 각 입력 토큰이 디코더의 입력값인 시퀀스 형태로 복사되는 횟수를 의미한다.
첫번째로 concatenated slot value 의 sequence 로 dialogue state 를 재구성하고, 
이 결과 sequence 는 Fertility concept 을 적용할 수 있는 고유한 구조적 representation 을 가지고 있다. 
그리고 이 구조는 개별 slot values 의 boundary 에 의해 정의된다.

이 모델은 2 단계의 decoding process 가 존재한다. 
1) 첫 번째 디코더가 input dialogue history 에서 관련된 signal 학습하고, 각 입력 slot value 의 representation 에 대한 fertility 를 생성한다. 
2) 이 예측된 fertility 는 구조화된 sequence 를 만드는 데 사용되고 구조화된 sequence 는 multiple sub-sequence 로 구성되어 있다.
sub-sequence 는 (slot token x slot fertility) 로 표현된다. 한 번에 target dialogue state 의 모든 토큰을 생성하기 위해 
결과 sequence(구조화된 sequence) 는 두 번째 디코더에 입력값으로 사용된다. 

또한 이 모델은 slot level 과 token level 모두에서 의존성을 고려한다. 
기존의 대부분 DST model 들은 여러 slot 에 걸쳐 나타나는 잠재적인 시그널을 고려하지 않고 slot 간의 독립성을 가정했다. 
하지만 많은 경우에 slot 간에는 같은 도메인이건 다른 도메인이건 의존성이 나타날 수 있다. 예를 들어, 기차 출발 장소와 도착 장소는 
같은 value가 아니라는 점을 고려해야 한다. 
NADST model은 dialog state를 생성하기 위해 모든 도메인에서 가능한 모든 시그널과 slot 을 고려한다.
이것이 DST 평가 척도로 사용하는 joint accuracy 를 직접적으로 높인다.
    - joint accuracy : slot level 이 아니라 state (set of slots) level 에서 정확도를 측정
      
이 논문의 Contribution
1. Non-Autoregressive Dialogue State Tracking(NADST) 제안
    - Dialogue State 의 완벽한 set 을 디코딩하기 위해 slot 들 간에 inter-dependency 를 학습한다.
2. 실시간을 위해 latency 를 줄였고 와 token level 뿐 아니라 slot level 의 Dependency 를 capture 한다.
3. Multi-domain Task-oriented MultiWOZ 2.1 에서 SOTA 를 기록했고, Inference latency 를 줄였다.
4. Ablation study 를 통해서 이 논문에서 제안하는 모델이 slot 간에 잠재적인 Signal 을 학습하고, 또 효과가 있다는 것을 보여준다. 
또한 slot 의 sets 을 더 정확하게 dialogue domain 에서 생성한다는 것을 보여준다.

# 2. Related Work

## Dialog State Tracking(DST)

DST는 task-oriented dialogues, 특히 관련 slot 들의 세세한 추적이 필요한 복잡한 도메인의 대화에서 중요한 요소이다.
전통적으로 DST 는 NLU 와 함께 사용했었다. dialog state 를 업데이트하기 위해 사용자 발화에 태깅한 NLU의 출력값을 DST model 의 입력값으로 사용했다.  
최근에는 NLU 와 DST 를 합쳐서 credit assignment problem(CAP3)을 해결하고  NLU 의 필요성을 제거했다. 
DST 모델은 2가지로 나뉜다.
1. fixed-vocabulary
    - 구축된 slot ontology를 이용해서 각 대화 상태 후보를 생성한다
    - 추출 기반 메서드
2. open-vocabulary
    - 대화 기록에서 대화 상태 후보를 생성한다.
    - 최근 방법
    - NADST 모델도 이 방식
    - NADST는 현재의 다른 모델과도 달리, slots 과 domains 간 의존성도 고려한다. 

## Non-Autoregressive Decoding 
