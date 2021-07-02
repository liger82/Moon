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

지난 챕터에서 심플한 환경에 value iteration은 cross-entropy보다 훨씬 좋은 성능을 보여주었습니다. 그래서 더 복잡한 환경에 적용해보려 합니다.

value iteration에 대해 빠르게 복기해보면, value iteration은 매 스텝마다 모든 상태에 대해 벨만 근사를 통해 가치를 업데이트 합니다. 
Q value(행동 가치)를 위한 버전도 거의 동일합니다. 차이점은 모든 상태와 행동에 대해 가치를 추정하고 저장한다는 점입니다.

Value Iteration에는 명백한 문제점이 있습니다.

### - 환경의 모든 상태에 대해 알고 있어야 한다.
### - 모든 상태들에 대해 여러 번 반복할 수 있고, 근사값을 저장도 할 수 있어야 한다.

현실적인 문제들에서 첫번째 전제부터 실현 불가능한 경우가 많고 또 알더라도 그 모든 상태에 대해 반복 작업을 하는 것은 비효율적입니다.
 
예를들어 Atari 2600 게임이 있습니다. 아타리 게임은 1980년대 유행했고, 아케이드 스타일의 게임입니다. 이 게임은 RL 연구에서 가장 인기 있는 벤치마크 플랫폼입니다.

** [아타리 게임의 상태 공간] **

* 화면 해상도 : 210 X 160 pixels, 126개의 색깔(channels)
    * 화면당 프레임 : 210 X 160 = 33,600 pixels, 색이 126개이므로 총 가능한 화면 수는 $$128^{33600}$$

아타리 게임의 가능한 상태를 한번 돌리는 것만 해도 수퍼컴퓨터로도 수십억년 소요되고 99% 이상의 반복 작업은 낭비에 가까운 작업입니다. 
    
<br>

### 또 다른 문제는 value iteration 접근법은 이산 행동 공간으로 그 대상을 한정한다는 점입니다. 

실제로 Q(s,a)와 V(s) 근사는 행동이 서로 배제적이며 이산적인 셋일 때 가능합니다. 휠 조종 각도나 히터기의 온도 조절 같이 연속적인 컨트롤 문제에는 적용할 수 없습니다. 

<br>

연속 행동 공간을 다루는 것은 꽤나 도전적인 과제이기 때문에 책의 뒷 부분에서 다룰 예정입니다. 일단 행동 공간이 이산적이고 행동 개수도 그렇게 크지 않다고 가정했을 때, 이를 Q-learning으로 해결해보는 과정에 대해 다루겠습니다.

<br>

> <subtitle> Tabular Q-learning </subtitle>

# 작성 중 ...

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
