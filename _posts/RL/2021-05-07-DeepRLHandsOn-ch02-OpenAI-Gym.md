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

```python
import random
from typing import List


class Environment:
    def __init__(self):
        self.steps_left = 10

    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]

    def get_actions(self) -> List[int]:
        return [0, 1]

    def is_done(self) -> bool:
        return self.steps_left == 0

    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()
```









<br>

---

> <subtitle> References </subtitle>

<br>
