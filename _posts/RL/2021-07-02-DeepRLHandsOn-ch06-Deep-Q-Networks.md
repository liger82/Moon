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

상태 공간의 모든 상태에 대해 반복을 해야 할까요? 만약 상태 공간의 일부 상태가 보이지 않는다면, 그 가치를 우리가 왜 신경써야 하나요? 우리가 환경으로부터 얻은 상태만 이용해서 업데이트할 수 있다면 효율성이 올라가지 않을까요?

이런 생각에서 나온 value iteration의 변형이 **Q-learnig**이라 불리는 방법입니다. 명시적으로 상태-가치 매핑이 있는 경우에 다음과 같은 단계를 따릅니다.

1. 초기화(empty table, mapping states to values of actions)
2. 환경과 상호작용함으로써, *s, a, r, s'* (state, action, reward, and the new state) 를 튜플로 얻는다. 
    - 이 단계에서 어떤 행동을 취할지 정해야 하고 항상 옳은 정책이 있다기 보다는 상황에 따라 다를 수 있다.
3. 다음과 같은 "blending" 기술을 이용한 근사를 사용해서 Q(s,a) 를 업데이트
    - <center>$$ Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha(r + \gamma \max_{a'\in A} Q(s', a')) $$</center>
    - 이는 학습율($$\alpha$$)을 사용해서 Q의 이전 가치와 새로운 가치의 weighted sum한 것
    - 환경에 노이즈가 많더라도 Q value를 부드럽게 수렴하게 해줌
4. 수렴 조건을 검사하고 만족하지 않으면 step 2부터 반복
    - 업데이트 정도나 테스트 성능이 특정 역치에 다다르면 멈추는게 일반적인 종료 조건

<br>

## Q-learning 보충

현 교재가 코드 중심이어서 Q-learning에 대한 설명이 부족한 듯 하여 Sutton 교수님의 책에서 나온 내용으로 일부 보강하였습니다.

- Q-learning은 **Off-policy TD learning**(Watkins, '89)이다.
- 이 수식은 위에서 나온 식과 동일하고 배치를 바꾼 것이다.
<center>$$ Q(s,a) \leftarrow Q(s,a) + \alpha[R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)] $$</center>

- backup diagram

<center><img src= "https://liger82.github.io/assets/img/post/20210702-DeepRLHandsOn-ch06-Deep-Q-Networks/fig6.1-qlearning-backup-diagram.png" width="50%"></center><br>

- Q-learning은 기존의 off-policy와 다르다.
    - 기존 off-policy는 behavior policy와 target policy가 다르기 때문에 behavior policy에서 학습한 내용을 target policy에서도 보장하기 위해 sampling data의 분포가 같다는 것을 증명해야 한다. 여기서 사용되는 것이 **importance sampling**이다. importance sampling은 다른 분포에서 sampling한 데이터를 알아야할 분포에 맞게끔 보정해서 estimate하는 기법을 말한다.
        - [importance sampling 자세한 설명은 여기에서 확인](https://talkingaboutme.tistory.com/entry/RL-Off-policy-Learning-for-Prediction){:target="_blank"}
    - 위 수식에서도 확인할 수 있듯이, Q-learning은 importance sampling ratio가 없다.
        - 1 step 진행하고 바로 업데이트하기 때문에 target, behavior policy가 달라질 수 없어서 importance sampling ratio를 제거
    
<br>

이제 다시 원래 교재로 돌아와서 코드를 살펴보도록 하겠습니다. 첫 번째 코드는 *Chapter06/01_frozenlake_q_learning.py* 파일입니다. 파일 제목에서도 알 수 있듯이 frozenlake 환경에서 q learning을 돌린 것입니다.

```python

#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9 # discounting factor
ALPHA = 0.2 # learning rate in the value update
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        '''
        간단한 환경이라 보상 이력이나 전이에 대한 기록을 하지 않고 value table만 유지. 
        복잡한 환경에서는 모두 필요
        '''
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        '''
        (old state, action, reward, new state) 튜플을 환경으로부터 얻기 위한 전처리 함수
        '''
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        '''
        value table 참조하여 인력값 state에 따른 최고 action과 value 반환
        두 군데에서 사용
        1. test method에서 정책의 퀄리티를 평가하기 위해 현재 value table을 이용하여 하나의 에피소드를 돌릴 때
        2. 다음 state의 가치를 얻기 위해 value update를 수행할 때
        '''
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        '''
        Q-learning의 value update
        value table에 Q value 유지
        '''
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)]
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA

    def play_episode(self, env):
        '''
        하나의 에피소드 전체를 진행
        '''
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

```

<br>

실제 코드를 돌리면 다음과 같은 결과가 나왔습니다.

```
.../Chapter06$ CUDA_VISIBLE_DEVICES=1 python 01_frozenlake_q_learning.py

[Best reward updated 0.000 -> 0.350
Best reward updated 0.350 -> 0.450
Best reward updated 0.450 -> 0.550
Best reward updated 0.550 -> 0.600
Best reward updated 0.600 -> 0.650
Best reward updated 0.650 -> 0.800
Best reward updated 0.800 -> 0.850
Solved in 13117 iterations!
```

이전 챕터에서 20~30 iteration만에 역치를 넘긴 것에 비하면 엄청나게 차이가 나는 수치입니다. 이런 차이가 나는 것은 위 코드에서는 test 시에 value update를 하지 않았고 5장에서의 코드에서는 했기 때문입니다.
전반적으로 환경에 필요한 총 샘플 수도 거의 같고 TensorBoard의 reward 차트도 value iteration 방법과 유사하게 우수한 학습 결과를 보여줍니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210702-DeepRLHandsOn-ch06-Deep-Q-Networks/fig6.2-reward-dynamics-frozenlake.png" width="70%"></center><br>

> <subtitle> Deep Q-learning </subtitle>

방금 다룬 Q-learning 방법은 전체 상태 집합에서 반복되는 문제를 해결하긴 하지만 관찰 가능한 상태 집합의 수가 매우 많은 상황에서는 여전히 어려움을 겪을 수 있습니다.

atari 게임 중 pong은 $$10^{70802}$$ 개의 가능한 상황이 있습니다. 이렇게 많은 상황에서 에이전트는 다르게 행동해야 합니다.

이 문제에 대한 해결책으로 state와 action을 모두 value에 매핑하는 nonlinear representation을 사용할 수 있습니다. 머신러닝에서는 이를 "회귀 문제"라고 합니다. 이를 구현하는 방법은 다양하지만 이미지로 표현되는 관찰을 처리할 때 deep neural net을 사용하는 것이 가장 인기가 많습니다. 이 점을 염두에 두고 Q-learning 알고리즘을 수정하면 다음과 같습니다.

1. Q(s,a) 초기 근사값으로 초기화
2. 환경과 상호작용하면서 (s, a, r, s') 튜플을 얻는다.
3. Loss 계산
    1. if 에피소드 종료: $$L = (Q(s,a) - r)^2 $$
    2. else : $$L=(Q(s,a) - (r+\gamma \max_{a' \in A} Q_{s', a'}))^2$$
4. 각 모델 패러미터에 대해 loss 값을 줄이면서, stochastic gradient descent(SGD) 알고리즘으로 Q(s,a) 업데이트
5. 수렴할 때까지 스텝2부터 반복

이 알고리즘은 간단해보이지만 작동도 잘 안되다고 하네요.... 왜 그러는지 살펴보겠습니다.

<br>

## Interaction with the environment

랜덤하게 행동하면 효율적이지 못하고 현재 경험 바탕으로 근사만 하면 local minima에 머무를 수 있습니다. exploration과 exploitation은 tradeoff 관계면서도 이를 적절히 해결하기 위한 방법이 **epsilon-greedy**($$\epsilon - greedy$$) 알고리즘입니다. 엡실론은 랜덤 행동의 비율입니다. 처음에 100%의 엡실론으로 시작하여 점차 그 값을 줄여나가서 2~5% 정도까지 줄이면 적절하게 랜덤 행동하면서 최적 행동을 찾고 기존 경험에서 최적 행동을 하면서 학습하는 정책입니다.

<br>

## SGD optimization

Q-learning 절차의 핵심은 지도학습에서 빌려온 것입니다. 뉴럴넷과 함께 복잡하고 비선형 함수 Q(s,a)를 근사하고자 했습니다.  

<br>

## Correlation between steps

<br>

## The Markov property

<br>

## The final form of DQN training

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
* [https://talkingaboutme.tistory.com/entry/RL-Off-policy-Learning-for-Prediction](https://talkingaboutme.tistory.com/entry/RL-Off-policy-Learning-for-Prediction){:target="_blank"}
* [https://mangkyu.tistory.com/61](https://mangkyu.tistory.com/61){:target="_blank"}
* [](){:target="_blank"}

<br>
