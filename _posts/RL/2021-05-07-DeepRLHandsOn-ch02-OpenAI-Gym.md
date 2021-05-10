---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 02 : OpenAI Gym"
date: 2021-05-07
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, OpenAI Gym, Gym, OpenAI]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

Deep Reinforcement Learning Hands-On 2/E 책을 바탕으로 강화학습 기초부터 비교적 최신 알고리즘까지 코드 중심으로 다뤄보도록 하겠습니다.

위에는 교보문고 교재 링크와 깃헙 링크도 올려두었습니다.

2판 코드에서는 텐서플로우와 파이토치, gym 버전도 업데이트되었습니다.  
* gym==0.17.3
* torch==1.7.0
* tensorflow==2.3.1  

2장에서는 OpenAI의 Gym의 기초에 대해서 다룹니다. OpenAI는 일론 머스크와 샘 알트만이 공동 설립한 인공지능 회사입니다. 인류에게 이익을 주는 것을 목표로 하는 인공지능 연구소입니다. **Gym** 은 OpenAI에서 만든 라이브러리로 RL agent 와 여러 RL 환경을 제공합니다. 

<br>

> <subtitle> The anatomy of the agent </subtitle>

에이전트와 환경을 파이썬으로 간단하게 구현한 코드를 보면서 감을 익히도록 하겠습니다.

에이전트의 행동과 무관하게 랜덤한 보상을 주는 환경을 만들었습니다. 지금 아래 코드는 사실 환경과 에이전트의 구조가 어떤 식인지를 보여주는 것이 목적이어서 진정한 상호작용 측면은 배제되어 있습니다.

```python
import random
from typing import List


class Environment:
    def __init__(self):
        # 제한된 스텝 수를 가정
        self.steps_left = 10

    # 에이전트에게 전달할 현재 환경의 관찰값 반환
    # 에이전트 행동에 따라서 상태가 변경되지 않음.
    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]
    
    # 가능한 행동 목록
    def get_actions(self) -> List[int]:
        return [0, 1]

    # 남은 스텝이 있는지 아닌지
    def is_done(self) -> bool:
        return self.steps_left == 0

    # 어떤 행동이 입력으로 들어오든 랜덤한 보상 지급
    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()
```

<br>

다음은 에이전트입니다. 총 보상을 변수로 가지고, 각 스텝별로 환경의 관찰값을 바탕으로 가능한 행동을 고르고 그에 따른 보상을 누적합니다.

```python
class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: Environment):
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward

```

<br>

위 코드의 심플함은 RL 모델의 기본 개념의 중요성을 그리고 있습니다. 
환경은 복잡한 물리적 모델이 될 수도 있고, 에이전트는 최신 RL 알고리즘을 구현한 뉴럴넷이 될 수도 있습니다. 하지만 이 기존 패턴은 동일하게 유지됩니다. "매 단계별로 에이전트는 환경으로부터 관찰값을 받고, 계산하고, 행동을 선택한다. 행동의 결과로 보상과 새로운 관찰값을 받는다."

<br>

> <subtitle> The OpenAI Gym API </subtitle>

**Gym** 의 주요 목적은 통일된 인터페이스로 다양하고 풍부한 RL 실험 환경을 제공하는 것입니다. 라이브러리의 중심 클래스 이름이 Env 인 것도 그 증거입니다. Env 클래스의 인스턴스가 제공하는 일부 메서드와 필드를 알아보겠습니다. 

* 환경에서 실행할 수 있는 행동 집합
* 환경이 에이전트에게 제공하는 관찰값의 형태와 경계
* step : 행동을 실행하는 메서드로, 현재 관찰값과 보상, 에피소드가 끝났는지를 반환
* reset : 환경 초기 상태와 첫 번째 관찰값을 반환한다.

환경의 구성요소에 대해 자세히 알아보도록 하겠습니다.

## The action space

에이전트의 행동은 이산적(discrete), 연속적, 혹은 둘 다 일 수 있습니다. 또한 행동을 하나로 제한할 필요도 없습니다. 이를 대비해서 Gym에서는 여러 행동을 단일한 행동으로 중첩할 수 있는 container class를 제공합니다.

## The Observation space

관찰값은 매 스텝마다 환경이 에이전트에게 부여하는 정보의 일부입니다. 관찰값은 단순한 숫자 뭉치일 수도 있고 복잡한 2차원이상의 텐서일 수도 있습니다.  
다음은 Gym에서 이 공간을 다이어그램으로 표현한 것입니다.

<br><center><img src= "https://liger82.github.io/assets/img/post/20210507-DeepRLHandsOn-ch02-OpenAI-Gym/fig_2_1.png" width="60%"></center><br>

가장 기본인 abstract class인 *Space*는 두 가지 메서드를 포함하고 있습니다.  
* sample() : 해당 공간으로부터 랜덤하게 뽑은 표본 반환
* contains(x) : x가 해당 공간에 속하는지 확인함





<br>

---

> <subtitle> References </subtitle>

<br>
