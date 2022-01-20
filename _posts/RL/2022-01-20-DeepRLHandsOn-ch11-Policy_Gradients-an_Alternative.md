---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 11 : Policy Gradients - an Alternative"
date: 2022-01-20
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 11, Policy Gradients, Policy Gradients python code, Policy Gradients pytorch code]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

책의 3번째 파트 시작 장에 해당하는 이 장에서는 MDP 문제를 다루는 또 다른 방법인 **policy gradient** 에 대해 알아보겠습니다.

챕터 목표   
* pg 개괄, 동기, q-learning과 비교한 강,약점을 다룬다.
* 간단한 policy gradient 방법인 **REINFORCE** 로 CartPole 예제를 다루면서 DQN 과 어떻게 다른지 비교해본다.

<br>

> <subtitle> Values and policy </subtitle>

제가 오랜만에 이 책을 볼 건지 예상이라도 한 건지 앞선 파트 리뷰를 해줍니다.

두 번째 파트에서 다룬 *value iteration* 과 *Q-learning* 의 중심 토픽은 상태의 value 혹은 value, action 이었습니다. 
value 는 지금 상태(or 상태 & 행동)로부터 얻을 수 있는 총 보상의 할인된 값으로 정의됩니다. value 를 알면 매 단계에서의 에이전트의 결정은 간단하고 명백합니다. 가치의 관점에서 **greedy** 행동하면 됩니다. 이는 에피소드 끝에서 높은 총 보상을 보장합니다. 이러한 가치를 얻기 위해 벨만 방정식을 사용합니다.  

첫 번째 챕터에서 존재를 매 상태에서 무엇을 할지(정책) 말해주는 것이라고 정의하였습니다. 큐러닝에서와 같이 가치가 어떻게 행동해야 할지 지시할 때, 가치는 정책을 실제로 정의한다고 말할 수 있습니다. 이를 공식으로 쓰면 다음과 같습니다.  

$$ \pi(s) = argmax_{a} Q(s,a) $$

정책과 가치의 관계는 명백해서 별도의 주체로서 정책을 강조하지 않고 대부분의 시간을 가치와 가치에 대한 정확한 근사 방법에 대해 이야기했습니다. 이제는 이러한 연결과 정책 자체에 초점을 맞춰야 할 때입니다. 

<br>

## Why the policy

정책이 살펴보기 흥미로운 주제인 이유엔 여러 가지가 있습니다. 첫 번째로는 정책은 우리가 강화학습 문제를 풀 때를 위해 찾고자 하는 것입니다. 에이전트가 관찰값을 가지고 있고 다음에 무엇을 할지 결정을 내려야 할 때, 가치나 행동이 아니라 정책이 필요합니다. 우리는 총 보상에 대해 신경 쓰지, 매 단계에서의 정확한 상태의 가치에는 관심이 없습니다. 

큐러닝은 간접적으로 상태의 가치를 근사하고 최고의 대안을 찾으려고 하면서 정책 질문에 답하고자 하지만, 가치에 관심이 없다면 굳이 추가적인 일을 해야 할까요?

가치보다 정책이 더 매력적인 다른 이유는 많은 행동이 있는 환경 혹은 연속적인 행동 공간 때문입니다. 우리는 $$ Q(s,a) $$ 를 최대화하는 행동 a 를 찾는 작은 최적화 문제를 풀어야 합니다. 몇 개의 행동을 가진 아타리 게임에서는 이것이 문제가 되지 않습니다. 모든 행동에 대해 가치를 근사하여 최고의 Q값을 갖는 행동을 선택하면 됩니다. 하지만 행동이 엄청 많거나 연속적이라면 최적화 문제는 풀기 어려워질 것입니다. Q는 보통 **비선형**인 뉴럴넷으로 표현되는데 그러면 가치를 최대화하는 arguments 를 찾는 것이 까다로워질 수 있기 때문입니다. 이 경우에 가치 보다는 정책을 직접 다루는 것이 현실 가능성이 높습니다. (value-based vs policy-based RL)

정책 기반 학습의 또 다른 이점은 stochastic policy 학습 가능하다는 점입니다. 챕터 8에서 categorical DQN 의 에이전트는 기댓값이 대신에 Q-value의 분포를 사용해서 많은 이점을 누렸습니다. 네트워크가 내재된 확률 분포를 더 정확하게 잡아낼 수 있기 때문입니다. 다음 섹션에서 정책은 본질적으로 행동의 확률로 표현된다는 것을 알 수 있습니다. 

<br>

## Policy representation

정책을 어떻게 표현할까요?  
Q 값의 경우 행동 값을 스칼라로 반환하는 뉴럴넷에 의해 매개 변수화되었습니다. 네트워크에서 행동을 매개 변수화하려면 몇 가지 방법이 있습니다. 가장 간단한 첫 번째 방법은 (이산 행동 집합의 경우) 행동의 식별자를 반환하는 것일 수 있습니다. 그러나 이 방법이 이산 집합을 처리하는 가장 좋은 방법은 아닙니다. 분류 작업에서 많이 사용되는 훨씬 일반적인 해결책은 행동의 확률 분포를 반환하는 것입니다. 즉, 상호 배타적인 N 개의 행동의 경우, 주어진 상태(네트워크에 입력으로 전달)에서 각 행동을 취할 확률을 나타내는 숫자 N 개를 반환합니다. 이 표현은 다음 다이어그램에 나와 있습니다.

<center><img src= "https://liger82.github.io/assets/img/post/20220120-DeepRLHandsOn-ch11-Policy-gradients/fig11.1.png" width="80%"></center><br>

확률로 행동을 표현한 것은 **smooth representation** 이라는 이점을 가집니다. 뉴럴넷의 weights 를 조금 바꾸면 그 출력값도 바뀝니다. 많지 않은 행동을 가진 경우도 조금의 가중치 변화는 행동 선택에 있어서 큰 차이를 가져올 수 있습니다. 하지만 출력값이 확률 분포라면 가중치의 조그만 변화는 보통 출력값 분포의 작은 변화로 이어질 것입니다. 이는 gradient 최적화 방법들이 모델 개선을 위해 아주 조금씩 패러미터를 조정한다는 점에서 매우 좋은 속성입니다. 정책은 $$ \pi(s) $$ 로 표현합니다.

<br>

## Policy gradients

policy gradients(pg) 를 다음과 같이 정의했습니다.

$$ \triangledown J \approx \mathbb{E}[Q(s,a)\triangledown \log \pi(a|s)] $$

이 식에 대한 증명은 있지만 여기선 식의 의미만 짚고 가겠습니다.

pg 는 축적된 총 보상의 관점에서 정책을 개선시키기 위해 네트워크의 패러미터를 바꾸는 방향을 정의합니다. gradient의 scale 은 공식에서 Q(s,a) (즉, 행동의 가치)에 비례하고, gradient 자체는 취한 행동의 로그 확률의 gradient 와 같습니다. 이는 **좋은 총 보상을 주는 행동의 확률을 높이고 안 좋은 결과를 낳는 행동의 확률은 줄인다**는 의미입니다. 
기댓값(공식에서 $$\mathbb{E} $$) 은 여러 스텝에서의 gradient 를 평균내겠다는 의미입니다. 

실용적인 관점에서 pg 는 다음 loss function 의 최적화를 수행하여 구현될 수 있습니다.  
$$L = -Q(s,a) \log \pi(a|s) $$

마이너스 표시인 이유는 stochastic gradient descent(SGD) 할 때, 손실 함수는 최소화해야 하기 때문입니다. 하지만 여기선 pg 를 최대화하고자 합니다. 

<br>

> <subtitle> The REINFORCE method </subtitle>

위에 나온 pg 공식은 policy-based methods 에서 흔히 볼 수 있을 것지만 세부적인 내용은 다를 것입니다. 중요한 포인트 한 가지는 얼마나 정확히 gradient scale, Q(s,a)를 계산할 것인가 입니다. 

cross-entropy 를 다루는 4장에서 몇개 에피소드를 돌리면서 총 보상을 계산해보고 *better-than-average* 보상과 함께 학습했습니다. 이 학습 절차를 pg에도 적용하여 좋은 에피소드에는 Q(s,a)=1 을, 안 좋은 에피소드에는 Q(s,a)=0 을 부여하였습니다. 

cross-entropy 와 다른 점은 단순히 0과 1을 쓰는 것 대신에 학습에 Q(s,a) 를 사용하는 것입니다. 무엇이 더 좋은가 보면, 일단 대답이 에피소드를 더 잘 정제해서 구분해줍니다. 예를 들어, 총 보상 10을 가진 에피소드의 transitions 은 보상 1을 가진 에피소드로부터 나온 transitions 보다 더 gradient 에 기여해야 합니다. 두 번째 이유는 에피소드 초반에 좋은 행동들의 확률을 높이고 에피소드 끝에 가까운 행동을 줄이기 위함입니다. 왜냐하면 Q(s,a)는 discounting factor 를 포함하고, 더 긴 행동 시퀀스에 대한 불확실성이 자동적으로 고려되기 때문입니다. 이 아이디어가 **REINFORCE** 방법입니다. REINFORCE 는 다음과 같이 진행됩니다.

1. 임의의 가중치로 네트워크를 초기화
1. N 개의 전체 에피소드를 수행하여 transition (s, a, r, s') 을 저장
1. 모든 에피소드(k), 매 단계(t)마다, 다음 스텝의 할인된 총 보상을 계산: $$ Q_{k,t} = \sum_{i=0} \gamma^{i} r_i $$ 
1. 모든 transitions 에 대해 손실 함수 계산: $$ L = - \sum_{k,t} \log(\pi(s_{k,t}, a_{k,t})) $$
1. loss를 최소하는 방향으로 가중치 업데이트(SGD 수행)
1. 수렴할 때까지 step 2부터 반복

<br>

REINFORCE 는 큐러닝과 몇몇 중요한 양상에서 다른 면을 보입니다.

* 명백한 탐험이 없어도 된다. 
    - 큐러닝에서는 greedy 행동을 하면서도 탐험을 하기 위해 epsilon-greedy 전략을 사용했다. REINFORCE 에서는 확률론적이기 때문에 탐험이 자동적으로 수행된다. 시작할 때, 네트워크는 임의의 가중치로 초기화되고 그것은 uniform probability distribution 을 반환한다. 이 분포는 임의의 행동과 부합한다.  
* replay buffer 를 사용하지 않는다. 
    - policy gradient 는 on-policy 에 해당한다. 이는 이전 정책에서 나온 데이터에선 학습할 수 없다는 의미다. 여기에는 장단점이 있다. 좋은 점은 이러한 방법들은 보통 수렴이 빠르다는 점이다. 단점은 off-policy 보다 환경과의 상호작용을 더 요구한다는 점이다.
* target network 를 필요로 하지 않는다.
    - Q값을 사용하지만 Q값은 환경에서의 경험으로부터 얻어진다. DQN 에서는 Q 값 근사에서 그 상관관계를 깨기 위해 target network 를 사용했지만 REINFORCE 에서는 근사를 하지 않는다. (다만 다음 챕터에서 pg 에서 target network 트릭을 사용하면 유용하다는 점을 볼 수 있긴 하다.)

<br>

## The CartPole example

REINFORCE 구현 코드를 익숙한 CartPole 환경에서 먼저 다뤄보도록 하겠습니다. 전체 코드는 *Chapter11/02_cartpole_reinforce.py* 입니다. 

```python
# code
```

<br>

## Results

<br>

## Policy-based versus value-based methods

<br>

> <subtitle> REINFORCE issues </subtitle>

<br>

## Full episodes are required

<br>

## High gradients variance

<br>

## Exploration

<br>

## Correlation between samples

<br>

> <subtitle> Policy gradient methods on CartPole </subtitle>

<br>

## Implemenation

<br>

## Results

<br>

> <subtitle> Policy gradient methods on Pong </subtitle>

<br>

## Implemenation

<br>

## Results

<br>

> <subtitle> Summary </subtitle>

이 챕터에서는  

<br>


<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 11 : Policy Gradients - an Alternative


<br>
