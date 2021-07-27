---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 06 : Deep Q-Networks"
date: 2021-07-02
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 6, Deep Q-Networks, DQN, Q-leanring]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

**[5장](https://liger82.github.io/rl/rl/2021/06/23/DeepRLHandsOn-ch05-Tabular-Learning-and-the-Bellman-Equation.html){:target="_blank"}**에서는 벨만 방정식과 그 응용 방법인 *value iteration*에 대해 다뤘습니다. 이런 접근 방법은 FrozenLake 환경에서 수렴 속도가 굉장히 빨랐습니다. 이번 챕터에서는 더 복잡한 환경(아타리 게임)에서 value iteration을 적용해볼 예정입니다.

이번 챕터에서는 다음과 같은 내용을 다룰 예정입니다. 

* value iteration의 문제에 대해 다루고 Q-learning 일 때는 어떤지 확인해본다.
* grid world 환경에서 Q-learning을 적용(tabular Q-learning)한다.
* Q-learning과 뉴럴넷의 결합인 deep Q-networks(**DQN**)에 대해 알아본다.

마지막으로 *Playing Atari with Deep Reinforcement Learning* 논문에서 나온 DQN 알고리즘을 구현해보도록 하겠습니다.

<br>

> <subtitle> Real-life value iteration </subtitle>

FrozenLake와 같은 심플한 환경에서 value iteration은 cross-entropy보다 훨씬 좋은 성능을 보여주었습니다. 이번엔 더 복잡한 환경에 적용해보려 합니다.

value iteration에 대해 빠르게 복기해보면, value iteration은 매 스텝마다 모든 상태에 대해 벨만 근사를 통해 가치를 업데이트 합니다. 
Q value(행동 가치)를 위한 버전도 거의 동일합니다. 차이점은 모든 (상태, 행동) 쌍(pair)에 대해 가치를 추정하고 저장한다는 점입니다.

Value Iteration에는 **명백한 문제점**이 있습니다.

### 문제 1. 환경의 모든 상태에 대해 알고 있어야 한다.
### 문제 2. 모든 상태들에 대해 여러 번 반복할 수 있고, 근사값을 저장도 할 수 있어야 한다.

<br>

현실적인 문제들에서 첫번째 전제부터 실현 불가능한 경우가 많고 또 알더라도 그 모든 상태에 대해 반복 작업을 하는 것은 비효율적입니다.
 
예를들어 Atari 2600 게임이 있습니다. 아타리 게임은 1980년대 유행했고, 아케이드 스타일의 게임입니다. 이 게임은 RL 연구에서 가장 인기 있는 벤치마크 플랫폼입니다.

**[아타리 게임의 상태 공간]**

* 화면 해상도 : 210 X 160 pixels, 126개의 색깔(channels)
    * 화면당 프레임 : 210 X 160 = 33,600 pixels, 색이 126개이므로 총 가능한 화면 수는 $$128^{33600}$$

아타리 게임의 가능한 상태를 한번 돌리는 것만 해도 수퍼컴퓨터로도 수십억년 소요되고 99% 이상의 반복 작업은 낭비에 가까운 작업입니다. 
    
<br>

### 문제 3. 또 다른 문제는 value iteration 접근법은 이산 행동 공간으로 그 대상을 한정한다는 점입니다. 

실제로 Q(s,a)와 V(s) 근사는 행동이 서로 배제적이며 이산적인 셋일 때 가능합니다. 휠 조종 각도나 히터기의 온도 조절 같이 연속적인 컨트롤 문제에는 적용할 수 없습니다. 

연속 행동 공간을 다루는 것은 꽤나 도전적인 과제이기 때문에 책의 뒷 부분에서 다룰 예정입니다. 

<br>

일단 아래 내용에서는 행동 공간이 이산적이고 행동 개수도 그렇게 크지 않다고 가정했을 때, 이를 Q-learning으로 해결해보는 과정에 대해 다루겠습니다.

<br>

> <subtitle> Tabular Q-learning </subtitle>

상태 공간의 모든 상태에 대해 반복을 해야 할까요? 만약 상태 공간의 일부 상태가 보이지 않는다면, 그 가치를 우리가 왜 신경써야 하나요? 우리가 얻은 상태만 이용해서 업데이트할 수 있다면 효율성이 올라가지 않을까요?

이런 생각에서 나온 value iteration의 변형이 **Q-learnig**이라 불리는 방법입니다. 명시적으로 상태-가치 매핑이 있는 경우에 다음과 같은 단계를 따릅니다.

1. 초기화(empty table, mapping states to values of actions)
2. 환경과 상호작용함으로써, *s, a, r, s'* (state, action, reward, and the new state) 를 튜플로 얻는다. 
    - 이 단계에서 어떤 행동을 취할지 정해야 하고 항상 옳은 정책이 있다기 보다는 상황에 따라 다를 수 있다.
3. 벨만 근사를 이용해서 Q(s,a) 를 업데이트
    - <center> $$ Q(s,a) \leftarrow r + \gamma \max_{a'\in A} Q(s', a') $$ <center> 
4. step 2부터 반복



<br>

> <subtitle> Deep Q-learning </subtitle>


<br>

> <subtitle> DQN on Pong </subtitle>


<br>

> <subtitle> Things to try </subtitle>


<br>




<br>

> <subtitle> Summary </subtitle>


<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 06 : Deep Q-Networks
* [](){:target="_blank"}

<br>
