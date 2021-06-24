---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 04 : The Cross-Entropy Method"
date: 2021-06-23
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 4, The cross-entropy method]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

[이전 챕터 4장](https://liger82.github.io/rl/rl/2021/06/05/DeepRLHandsOn-ch04-The_Cross-Entropy_Method.html){:target="_blank"}에서는 첫 번째 RL 알고리즘인 cross-entropy 방법과 그 장단점에 대해 알아보았습니다. 

이번 챕터에서는 더 유연하고 강력한 방법 중 한 그룹인 tabular method(특히 Q-learning)에 대해 다룰 예정입니다. tabular method가 기반으로 하는 배경지식과 함께 FrozenLake 환경에 적용해보고 어떤 결과가 나오지 보며 tabular method의 특성에 대해 조명하겠습니다.

이번 챕터에서는 다음과 같은 내용을 다룰 예정입니다. 

* 상태의 가치와 행동의 가치의 개념을 정리하고 간단한 예시로 이 두 개를 계산하는 것을 학습하는지 알아본다.
* Bellman equation과 우리가 가치를 알고 있을 때 Bellman equation은 어떻게 최적 정책을 찾아내는지 다룬다.
* value iteration method에 대해 논의하고 FrozenLake 환경에 적용해본다
* Q-learning에 대해 알아보고 FrozenLake 환경에 적용해본다.

<br>

> <subtitle> Value, state, and optimality </subtitle>

이 책의 모든 파트는 가치에 대해 말하면서 가치를 근사하는 방법에 대해 고민하고 있습니다. 가치는 상태로부터 얻을 수 있는 (할인된) 총 보상 기댓값으로 정의됩니다. 

<center> $$ V(s) = \mathbb{E}[\sum_{t=0}^{\infty}r_t \gamma^t ] $$ </center>

$$r_t$$는 t 단계에서 얻은 보상값이고, $$\gamma$$는 할인율입니다. (할인되지 않는 경우는 $$\gamma$$=1)
가치는 항상 에이전트의 특정 정책의 관점에서 계산됩니다. 3개의 상태를 가진 간단한 환경으로 설명해보겠습니다.

* State 1 : 시작 상태
* State 2 : 시작 상태에서 오른쪽으로 움직였을 경우(행동) 도착하는 최종 상태. 여기에 도착하면 보상 1을 받음.
* State 3 : 시작 상태에서 아래로 움직였을 경우 도착하는 최종 상태. 여기에 도착하면 보상 2를 받음.

<br><center><img src= "https://liger82.github.io/assets/img/post/20210623-DeepRLHandsOn-ch05-Tabular-learning-and-the-Bellman-equation/fig5.1.png" width="60%"></center><br>

여기서 환경은 항상 결정론적입니다. 모든 행동은 성공이고(100% 확률로 의도한 행동이 실행된다), 항상 state 1부터 시작합니다. state 2이나 state 3에 도착하면 에피소드는 종료됩니다. 
그렇다면 state 1의 가치는 무엇일까요? 이 질문은 에이전트의 행동, 달리 말하면 정책에 대한 정보 없이는 무의미한 것입니다. 
정책은 어떤 것이 있을까
* 매번 오른쪽으로
* 매번 아래로
* 50퍼센트 확률로 오른쪽, 50퍼센트 확률로 아래쪽
* 10퍼센트 오른쪽, 90퍼센트 아래

간단한 환경에서도 이런 식으로 하면 다양한 수가 나옵니다. 이 경우 그 가치를 계산해보면 
* 매번 오른쪽으로 : 1.0 * 1.0 = 1.0
* 매번 아래로 : 2.0 * 1.0 = 2.0
* 50퍼센트 확률로 오른쪽, 50퍼센트 확률로 아래쪽 : 1.0 * 0.5 + 2.0 * 0.5 = 1.5
* 10퍼센트 오른쪽, 90퍼센트 아래 : 1.0 * 0.1 + 2.0 * 0.9 = 1.9

최적 정책은 총 보상을 가장 크게 얻는 정책입니다. one-step 환경에서는 2번째 정책이 가장 총 보상이 크니 항상 아래로가 최적 정책입니다. 간단한 환경도 이런 식인데 실제와 같은 복잡한 환경에서는
계산이 더 많아질까 또 최적인지 증명하는게 더 어려워지는 우려스러운 상황입니다. 그래서 뒤에서는 에이전트가 스스로 최적의 행동을 학습할 수 있도록 하는 방법에 대해 다룹니다.
 
앞선 예제로 돌아와서, 이번에는 State 3 다음에 State 4가 있고 State 4에 도착하면 보상 -20을 받습니다. 

<br><center><img src= "https://liger82.github.io/assets/img/post/20210623-DeepRLHandsOn-ch05-Tabular-learning-and-the-Bellman-equation/fig5.2.png" width="60%"></center><br>

항상 아래로 간다는 2번째 정책이 이 경우엔 오히려 악재로 작용합니다. 이번에는 오히려 1번 정책(항상 오른쪽)이 최적 정책입니다. 

이 문제는 최적화 문제의 복잡성을 깨닫고 벨만의 결과를 더 잘 이해할 수 있도록 논의해본 것입니다. 벨만 방정식은 앞선 예제를 다루기 좋은 식입니다.

<br>

> <subtitle> The Bellman equation of optimality  </subtitle>


<br>

> <subtitle> The value of the action </subtitle>


<br>

> <subtitle> The value iteration method  </subtitle>


<br>

> <subtitle> Value iteration in practice </subtitle>


<br>

> <subtitle> Q-learning for FrozenLake </subtitle>



<br>


<br>

> <subtitle> Summary </subtitle>


<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 05 : Tabular Learning and the Bellman Equation
* [](){:target="_blank"}

<br>
