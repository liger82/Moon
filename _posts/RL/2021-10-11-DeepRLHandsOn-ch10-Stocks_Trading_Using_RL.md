---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 10 : Stocks Trading Using RL"
date: 2021-10-11
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 10, Stocks Trading Using RL, Stocks]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

지금까지는 가벼운 예제를 대상으로 강화학습 방법을 적용해보았다면 이번 챕터에서는 "금융 거래"와 같이 더 현실적인 문제를 대상으로 DQN 을 적용해보는 시간을 가지도록 하겠습니다. 

챕터 목표   
* 커스텀 OpenAI Gym 환경을 구현하여 주식 시장을 시뮬레이션해보기
* 이윤 최대화를 위해 주식 거래를 위한 에이전트 학습
    - 기본 DQN
    - DQN Extensions(8장)

<br>

> <subtitle> Trading </subtitle>

매일 시장에서 거래되는 금융 수단은 여럿이 있습니다. 심지어 기상 예보도 기상 파생상품을 통해 사고 팔 수 있습니다. 작물 재배와 같이 수입이 미래의 기상 조건에 의존하는 사업을 하고 있으면,  기상 파생상품을 구매하여 리스크를 회피하려고 할 수 있습니다. 이런 다른 모든 종류의 아이템들은 시간에 따라 변화하는 가격을 가지고 있습니다. 거래는 이윤을 만들어내거나(투자) 미래의 가격 변동으로부터 보호하거나(hedging) 아니면 단순히 원하는 것을 얻으려는 등의 다양한 목표를 가지고 금융 수단을 사고 파는 활동을 뜻합니다. 

금융 시장이 만들어진 이래로, 사람들은 미래 가격 변동을 예측하기 위해 노력해왔습니다. 왜냐하면 미래 가격 변동을 잘 예측하는 것이 큰 이익을 보장하기 때문입니다.  

이 문제는 시장을 예측하고 최고의 순간을 찾아 이윤을 극대화하려는 재무 컨설턴트, 투자 펀드, 은행, 개인 트레이터가 얽혀 있어 복잡합니다.  

이 챕터를 관통하는 질문은 이것 입니다. "강화학습의 관점에서 이 문제를 풀어낼 수 있는가" 시장으로부터 관찰값이 있고, 매수, 매도, 대기 중 하나를 선택하고자 할 때, 가능한 한 큰 이윤을 내고자 합니다. 

<br>

> <subtitle> Data </subtitle>

이번 장에서 사용할 데이터 예제는 15~16년 러시아의 Yandex 주식 가격입니다. 파일은 *Chapter10/data/ch08-small-quotes.tgz* 입니다. (파일명의 챕터가 업데이트가 안된 듯..) data 디렉토리 안에 unpack 하는 실행파일도 있으니 학습 전에 실행해서 압축해주시기 바랍니다.

파일 내부는 다음과 같습니다.

```
<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
20160104,100100,1148.9000000,1148.9000000,1148.9000000,1148.9000000,0
20160104,100200,1148.9000000,1148.9000000,1148.9000000,1148.9000000,50
20160104,100300,1149.0000000,1149.0000000,1149.0000000,1149.0000000,33
20160104,100400,1149.0000000,1149.0000000,1149.0000000,1149.0000000,4
20160104,100500,1153.0000000,1153.0000000,1153.0000000,1153.0000000,0
20160104,100600,1156.9000000,1157.9000000,1153.0000000,1153.0000000,43
...
```

<br>

캔들스틱 차트(Candlestick chart)는 주식을 비롯해서 유가증권, 파생상품, 환율의 가격 움직임을 보여주는 금융 차트로 가격 변동을 보여주는 가장 전형적인 차트입니다. 

이 데이터는 캔들스틱 차트를 숫자를 표현했기 때문에 하루를 일정 간격으로 잘라서 하나의 행으로 표현하고 있습니다. 하나의 행 데이터는 봉(bar)을 뜻합니다.

DATE는 날짜, TIME 은 시간(분)을 의미하고, OPEN은 그 간격의 시작 가격, HIGH는 그 간격 기간 내 최고 가격, LOW는 그 간격 기간 내 최저 가격, CLOSE는 간격의 마지막 가격, VOL은 거래량을 의미합니다. 

다음은 Yandex의 2016년 2월 하루를 표현한 캔들스틱 차트입니다.

<center><img src= "https://liger82.github.io/assets/img/post/20211011-DeepRLHandsOn-ch10-Stocks_Trading_Using_RL/fig10.1.png" width="90%"></center><br>

16년 데이터는 학습에 15년은 검증에 사용할 예정입니다.

<br>

> <subtitle> Problem statements and key decisions </subtitle>

금융 도메인은 크고 복잡해서 매일 새로운 것을 배워도 몇 년은 걸릴 것입니다. 그래서 이 주식 거래 예제에서 관찰값을 가격으로만 한정하여 문제를 단순화하였습니다. (더 복잡한 현실을 반영하려고 주식 거래를 예제로 잡았는데 너무 단순화한게 아닌가 싶네요.)

이 예제의 목적은 RL 모델이 어떻게 유연해질 수 있는지와 보통 RL을 실제 use case에 적용하기 위해 해야하는 첫 번째 단계가 무엇인지를 보여주는 것입니다. 

강화학습 문제를 정의하기 위한 3가지  
* 환경의 관찰값
* 가능한 행동
* 보상 체계

이전 챕터에서는 이 세 가지는 항상 주어졌습니다. 이제는 우리의 에이전트가 볼 수 있는 것과 할 수 있는 행동을 정해줘야 합니다. 보상 체계도 엄격하게 정해져 있지 않고 도메인에 대한 우리의 감정, 지식에 기반하여 가이드할 수 있습니다. 즉, 우리는 유연성을 가지고 있습니다.

유연성은 장단을 가집니다.
* 장점 : 본인이 중요하다고 생각하는 정보를 에이전트에게 전달할 자유가 있음
* 단점 : 훌륭한 에이전트를 만들기 위해서는 다양한 데이터를 고려할 필요가 있음

기본 거래 에이전트를 가장 단순한 형태로 구현했습니다. 관찰값은 다음과 같은 정보를 포함하고 있습니다.  
* N개의 봉(bar)
    - 각각의 봉은 시작, 고점, 저점, 끝 가격을 가짐
* 주식이 몇 타임 전에 팔렸다는 정보
* 보유 중인 주식의 현재 위치에서의 손익

모든 스텝, 매 시간마다 에이전트는 다음과 같은 행동을 취할 수 있습니다.
* 대기
* 매수 : 해당 주식을 이미 보유하고 있으면 매수하지 않음. 그렇지 않을 경우 소정의 수수료를 지불하고 매수
* 매도 : 이전에 매수하지 않았을 경우 아무 행위 없음. 그렇지 않을 경우 거래에 수수료 있음.

<br>

> <subtitle> The trading environment </subtitle>

<br>

> <subtitle> Models </subtitle>

<br>

> <subtitle> Training code </subtitle>

<br>

> <subtitle> Results </subtitle>

<br>

> <subtitle> Things to try </subtitle>


<br>

> <subtitle> Summary </subtitle>


<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 10 : Stocks Trading Using RL
* [](){:target="_blank"}


<br>
