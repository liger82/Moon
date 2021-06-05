---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 04 : The Cross-Entropy Method"
date: 2021-06-05
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 4, The cross-entropy method]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

이번 챕터에서는 강화학습 방법 중 하나인 cross-entropy 에 대해 알아보겠습니다. 다른 방법들에 비해 덜 유명하지만, cross-entropy도 그만의 장점을 지닙니다. 
1. 매우 간단하다 --> 파이토치 코드가 100줄이 안 된다.
2. 훌륭한 수렴성 --> 간단한 환경에서는 cross-entropy도 매우 잘 작동한다.

이번 챕터에서는 다음과 같은 내용을 다룰 예정입니다.
* cross-entropy의 실용적인 면을 다룬다.
* CartPole과 FrozenLake, 두 개의 Gym 환경에서 cross-entropy가 어떻게 작동하는지 알아본다.
* cross-entropy의 이론적 배경에 대해 다룬다.(Optional)

<br>

> <subtitle> The taxonomy of RL methods  </subtitle>

<br><center><img src= "https://dreamgonfly.github.io/images/rl-taxonomy/rl_taxonomy.png" width="60%"></center><br>

cross-entropy는 model-free, policy-based 방법입니다. 이 의미에 대해 알아보기 위해 강화학습을 나누는 여러 측면에 대해 먼저 얘기해보겠습니다. 
* Model-free vs Model-based
* Value-based vs Policy-based
* On-policy vs Off-policy

<br>

## Model-free vs Model-based

<br><center><img src= "https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg" width="80%"></center><br>

강화학습 알고리즘을 구분하는 주요 지점 중 하나는 에이전트가 **환경의 모델에 접근(혹은 학습)**을 할 수 있느냐입니다. 환경의 모델은 상태 전이와 보상을 예측하는 함수를 의미합니다. 

Model-based 알고리즘은 **계획(planning)**이 가능하다는 장점을 지닙니다. 자신의 행동에 따라 환경이 어떻게 변할지 예측해서 최적의 행동을 계획하여 실행할 수 있다는 것입니다. 모델을 지닌다면 에이전트는 효율적으로 행동할 수 있습니다.

문제는 이 환경이 복잡할 경우에 환경의 모델을 알아내기 어렵거나 불가능하다는 점입니다. 모델이 환경을 제대로 반영하지 않는다면 이는 바로 에이전트의 오류로 이어지게 될 것입니다. 

Model-based 알고리즘은 모델이 주어져 있는지 학습해야 하는지에 따라 또 구분이 가능합니다.

Model-free 알고리즘은 환경의 모델을 사용하지 않는 경우를 말하며, 관찰값과 행동을 바로 연결시킵니다. 현재 관찰값을 받아서 이를 기반으로 계산하고 그 결과로 행동을 하는 것입니다.

강화학습에서는 주로 model-free 알고리즘이 각광받고 있으며 우리가 아는 대부분의 유명한 알고리즘들(DQN, policy gradient, A2C, A3C 등)은 모두 model-free 입니다.

최근에는 두 알고리즘을 융합하는 시도도 하고 있습니다. 예를 들어, 딥마인드에서 내놓은 *[Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/abs/1707.06203)* 도 이러한 부류입니다. (22장에서 다룰 예정)


<br>

> <subtitle> The cross-entropy method on CartPole  </subtitle>

<br>

> <subtitle> The cross-entropy method on FrozenLake </subtitle>

<br>

> <subtitle> The theoretical background of the cross-entropy method </subtitle>



<br>

> <subtitle> Summary </subtitle>



<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 04 : The Cross-Entropy Method
* [https://dreamgonfly.github.io/blog/rl-taxonomy/](https://dreamgonfly.github.io/blog/rl-taxonomy/){:target="_blank"}
* [https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html){:target="_blank"}

<br>
