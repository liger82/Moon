---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 05 : Tabular Learning and the Bellman Equation"
date: 2021-06-23
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 5, tabular method, bellman equation, q-learning]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

[이전 챕터 4장](https://liger82.github.io/rl/rl/2021/06/05/DeepRLHandsOn-ch04-The_Cross-Entropy_Method.html){:target="_blank"}에서는 첫 번째 RL 알고리즘인 cross-entropy 방법과 그 장단점에 대해 알아보았습니다. 

이번 챕터에서는 더 유연하고 강력한 방법 중 한 그룹인 tabular method(특히 Q-learning)에 대해 다룰 예정입니다. tabular method의 배경지식과 함께 FrozenLake 환경에 적용했을 때 어떤 결과가 나오는지 보며 tabular method의 특성에 대해 조명하겠습니다.

이번 챕터에서는 다음과 같은 내용을 다룰 예정입니다. 

* 상태의 가치와 행동의 가치의 개념을 정리하고 간단한 예시로 이 두 개를 계산하는 방법에 대해 알아본다.
* 벨만 방정식의 개념에 대해 알아보고, 가치를 알고 있을 때 벨만 방정식은 어떻게 최적 정책을 찾아내는지 다룬다.
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
정책은 예를 들면 다음과 같습니다.
* 매번 오른쪽으로
* 매번 아래로
* 50퍼센트 확률로 오른쪽, 50퍼센트 확률로 아래쪽
* 10퍼센트 오른쪽, 90퍼센트 아래

간단한 환경에서도 이런 식으로 하면 다양한 경우가 나옵니다. 이 경우들의 가치를 계산해보면 
* 매번 오른쪽으로 : 1.0 * 1.0 = 1.0
* 매번 아래로 : 2.0 * 1.0 = 2.0
* 50퍼센트 확률로 오른쪽, 50퍼센트 확률로 아래쪽 : 1.0 * 0.5 + 2.0 * 0.5 = 1.5
* 10퍼센트 오른쪽, 90퍼센트 아래 : 1.0 * 0.1 + 2.0 * 0.9 = 1.9

최적 정책은 총 보상을 가장 크게 얻는 정책입니다. one-step 환경에서는 2번째 정책이 가장 총 보상이 크니 항상 아래로가 최적 정책입니다. 이러한 방식으로는 실제와 같은 복잡한 환경에서는 계산이 더 많아지고 또 최적인지 증명하는게 더 어려워지는 우려스러운 상황입니다. 그래서 뒤에서는 에이전트가 스스로 최적의 행동을 학습할 수 있도록 하는 방법에 대해 다룹니다.
 
앞선 예제로 돌아와서, 이번에는 State 3 다음에 State 4가 있고 State 4에 도착하면 보상 -20을 받는다고 하면,

<br><center><img src= "https://liger82.github.io/assets/img/post/20210623-DeepRLHandsOn-ch05-Tabular-learning-and-the-Bellman-equation/fig5.2.png" width="60%"></center><br>

항상 아래로 간다는 2번째 정책이 이 경우엔 악재로 작용합니다. 이번에는 오히려 1번 정책(항상 오른쪽)이 최적 정책입니다. 

이 문제는 최적화 문제의 복잡성을 깨닫고 벨만의 결과를 더 잘 이해할 수 있도록 논의해본 것입니다. 벨만 방정식은 앞선 예제를 다루기 좋은 식입니다.

<br>

> <subtitle> The Bellman equation of optimality  </subtitle>

벨만 방정식을 설명하기 위해 약간의 추상화를 통해 해보겠습니다. 

예) 결정론적인 경우, 모든 행동이 100% 결과를 보장한다고 할 때, 에이전트는 상태 $$ s_0 $$를 관찰값으로 가지고 N개의 가능한 행동들을 할 수 있습니다. 모든 행동에 따라 상태와 행동이 다릅니다. 그리고 상태 $$ s_0 $$와 연결된 모든 상태의 가치를 알고 있습니다. 

<br><center><img src= "https://liger82.github.io/assets/img/post/20210623-DeepRLHandsOn-ch05-Tabular-learning-and-the-Bellman-equation/fig5.3.png" width="60%"></center><br>

위 조건에서 최적 행동을 찾는 것이 목표라고 할 때, 에이전트는 어떻게 해야 할까요? 에이전트는 행동에 따른 보상값을 알고 있기 때문에 단순히 계산해보면 됩니다. 각 행동을 취했을 때의 가치를요. 할인율까지 고려하면 다음과 같은 식이 나옵니다. 

<center> $$ V_0 = max_{a\in 1...N}(r_a + \gamma V_a) $$ </center>

벨만 방정식입니다. 일견 그리디 알고리즘과 유사해보입니다. 차이는 즉각적인 보상만을 고려하는 것이 아니라 먼 미래 상태의 가치도 고려한다는 점입니다. 그렇기 때문에 벨만 방정식을 찾으면 최적 행동을 찾을 수 있습니다.

하나의 행동이 각기 다른 확률을 가지고 다른 상태로 이끈다면 어떨까요? 이 경우는 그 행동의 기대값을 계산하면 됩니다. 식으로 표현하면 다음과 같습니다. 

<br><center><img src= "https://liger82.github.io/assets/img/post/20210623-DeepRLHandsOn-ch05-Tabular-learning-and-the-Bellman-equation/fig5.4.png" width="60%"></center><br>


<center> $$ V_0(a) = \mathbb{E}_{s\sim S}[r_{s,a} + \gamma V_s]=\sum _{s\in S} p_{a,0 \rightarrow s} (r_{s,a} + \gamma V_s) $$ </center>

위 두 가지 케이스를 모두 커버할 수 있게 결합하면 더 일반적인 벨만 방정식을 얻을 수 있습니다.

<center> $$ V_0 = max_{a \in A}\mathbb{E}_{s\sim S}[r_{s,a} + \gamma V_s]=max_{a \in A} \sum _{s\in S} p_{a,0 \rightarrow s} (r_{s,a} + \gamma V_s) $$ </center>

<br>

벨만 방정식으로 계산한 가치는 최고의 보상일 뿐만 아니라 그 가치를 얻을 수 있는 최적 정책을 주는 것이라 볼 수 있습니다. 에이전트가 모든 상태의 가치를 알고 있다면, 자동적으로 어떻게 이 보상들을 얻을 수 있는 지도 알 것입니다. 벨만의 최적성 입증 덕분에, 에이전트는 모든 상태에서 (즉각적인 보상과 한 단계 할인된 장기적 보상의 합계인) 보상의 최대 기댓값을 기반으로 행동을 선택할 수 있습니다. 

이 내용을 바로 코드에 적용하기 앞서, 몇 가지 수학적 notation에 대해 알아보겠습니다.

<br>

> <subtitle> The value of the action </subtitle>

* 상태의 가치 : $$ V(s) $$
* 행동의 가치 : $$ Q(s, a) $$

우리의 주된 목적은 모든 상태, 행동 쌍에 대해 Q값을 얻는 것입니다.

<center> $$ Q(s,a) = \mathbb{E}_{s' \sim S}[r(s,a) + \gamma V(s')] = \sum _{s' \in S} p_{a,s \rightarrow s'} (r(s,a) + \gamma V(s')) $$ </center>
<br>

$$V(s)$$는 $$Q(s,a)$$를 통해 정의될 수도 있습니다. 아래 식을 풀어서 생각해보면, "특정 상태의 가치는 특정 상태에서 실행할 수 있는, 최대 가치를 갖는 행동의 가치와 같다."라는 것입니다.

<center> $$ V(s) = max_{a' \in A}Q(s,a) $$ </center>
<br>

마지막으로 $$ Q(s,a) $$를 재귀적 형태를 지닌 모습으로 표현할 수 있습니다.

<center> $$ Q(s,a) = r(s,a) + \gamma max_{a' \in A}Q(s',a') $$ </center>

<br>

> <subtitle> The value iteration method  </subtitle>

이제 V와 Q를 계산할 수 있는 일반적인 방법에 대해 알아보겠습니다. 여기서도 예를 들어보겠습니다.

<br><center><img src= "https://liger82.github.io/assets/img/post/20210623-DeepRLHandsOn-ch05-Tabular-learning-and-the-Bellman-equation/fig5.7.png" width="60%"></center><br>

상태 s1에서는 상태 s2로만 갈 수 있고 이 때 보상은 1 입니다. s2에서 s1로만 갈 수 있고 보상은 2 입니다. 이 에이전트의 상태 시퀀스는 s1, s2의 반복일 것입니다.
이 무한 루프를 다루기 위해 할인율 $$ \gamma $$ 0.9를 적용합니다.

그럼 이 두 상태의 가치는 몇 일까요? 

답은 복잡하지 않습니다. 보상의 시퀀스는 1,2 반복입니다. 모든 상태에서 가능한 행동이 단 하나여서 선택권이 없으니 최대값 연산은 생략할 수 있습니다.
이 두 상태의 가치는 다음처럼 무한 합과 같습니다.

<center> $$ V(s_1) = 1+\gamma(2+ \gamma (1+ \gamma(2+ ...))) = \sum _{i=0}^{\infty } 1 \gamma^{2i} + 2 \gamma^{2i+1} $$ </center>

<center> $$ V(s_2) = 2+\gamma(1+\gamma (2+ \gamma(1+ ...))) = \sum _{i=0}^{\infty } 2 \gamma^{2i} + 1 \gamma^{2i+1} $$ </center>

엄격하게 말하면 이 값들을 구할 수 없지만, 할인율을 적용하여 전이에 따른 값을 줄여나갈 수 있습니다.  
10 스텝을 지났다고 하면 $$\gamma^{10} = {0.9}^{10} = 0.349 $$이고, 100 스텝 지나면 0.0000266 입니다. 이런 원리로 50번 반복하고 나서 멈췄을 경우, 꽤 정확한 추정값을 얻을 수 있습니다.

```
>>> sum([0.9**(2*i) + 2*(0.9**(2*i+1)) for i in range(50)])
14.736450674121663

>>> sum([2*(0.9**(2*i)) + 0.9**(2*i+1) for i in range(50)])
15.262752483911719
```

<br>

앞의 예는 **value iteration algorithm**이라고 하는 일반적인 절차의 핵심이라 할 수 있습니다. 이를 통해 알려진 전이 확률과 보상을 통해 마르코프 의사결정 프로세스(MDP)의 상태와 행동의 가치를 수치적으로 계산할 수 있습니다. (상태의 가치를 계산하는) 절차는 다음과 같은 단계를 따릅니다.

1. 모든 상태의 가치를 특정 값(보통 0)으로 초기화
2. 모든 상태 s에 대해 MDP에서는 벨만 업데이트를 진행한다.
    <center> $$ V_s \leftarrow  max_{a} \sum _{s'} p_{a,s \rightarrow s'} (r_{s,a} + \gamma V_{s'}) $$ </center>
3. 일정 수 이상의 2단계를 실행하거나 변화한 정도가 정말 작을 때까지 2단계를 반복한다.

<br>

행동 가치의 경우 앞선 절차와 비교했을 때 일부 차이가 있습니다.

1. 모든 행동 가치, $$Q_{s,a}$$ 값을 초기화
2. 모든 상태 s와 행동 a에 대해 업데이트
    <center> $$ Q_{s,a} \leftarrow  \sum _{s'} p_{a,s \rightarrow s'} (r_{s,a} + \gamma max_{a'} Q_{s', a'}) $$ </center>
3. 2단계 반복

<br>

실용적인 부분에서 이 방법은 몇 가지 명백한 **제한점**을 지닙니다.

1. 상태 공간은 이산적(discrete)이어야 하고, 모든 상태에 대해 여러번 반복을 돌 정도로 충분히 작아야 한다. (이번 예시에서 사용할 FrozenLake와 같은 환경에서는 문제가 되지 않음)
2. 실제에선 행동과 보상 행렬을 위한 전이 확률을 알 경우가 거의 없다. 

<br>

이제 FrozenLake 환경에서 value iteration method가 어떻게 작동하는지 살펴보겠습니다.

<br>

> <subtitle> Value iteration in practice </subtitle>

*Chapter05/01_frozenlake_v_iteration.py* 에 전체 코드 있습니다. 

이 예제에서 사용하는 주요 데이터 구조는 다음과 같습니다.

* Reward table : 딕셔너리 / (현재 상태, 행동, 다음 상태) : 보상
* Transition table : 딕셔너리(Counter 딕셔너리) / (현재 상태, 행동)를 키 값으로 하는 딕셔너리가 있고 그 안에는 다음 행동 값에 따른 등장 횟수를 등록해두었다. 
ex) self.transits[(s2, a3)][s3] = 4 는 상태 s2에서 a3 행동을 했을 때 다음 상태 s3이 4번 나왔다는 의미입니다. 만약 총 10번 반복했다면 전이 확률은 0.4입니다.
* Value table : 딕셔너리 / 상태 : 상태의 (계산된) 가치

<br>

전체적인 코드 로직은 간단합니다.
1. 루프를 돌며, FrozenLake 환경에서 100 번의 랜덤 실행을 하며, reward table과 transition table을 채웁니다.
2. 100번 스텝 이후에 모든 상태에 대해 value iteration을 수행하며, value table을 업데이트 합니다. 
3. 몇 개의 full episode를 돌면서 업데이트된 것을 확인해본다.
4. 만약 테스트 에피소드의 평균 보상값이 0.8 이상이면 학습을 멈춘다. 테스트 에피소드 과정에서도 보상과 전이 테이블은 업데이트 할 수 있다.

<br>

```python
#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        # 3개의 주요 데이터
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(
            collections.Counter)
        self.values = collections.defaultdict(float)

    # 2단계에서 count번의 랜덤 실행을 통해 reward, transition table을 채운다. --> 경험을 쌓는다.
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() \
                if is_done else new_state

```

<br>

cross-entropy와 다른 점은 N 스텝이 에피소드의 끝이 아닐 수 있다는 점입니다. cross entropy는 에피소드가 끝날 때까지 기다렸다가 업데이트를 하지만 value iteration은 N 스텝 이후에 해당 저장 내용을 기반으로 바로 학습하면 됩니다.

<br>

calc_action_value 은 Q값을 구하는 메서드입니다. 벨만 방정식을 이용하고 있습니다.

```python
    def calc_action_value(self, state, action):
        # 현재 상태, 행동이 주어졌을 때의 다음 행동의 등장 횟수 딕셔너리
        target_counts = self.transits[(state, action)]
        # 현재 상태, 행동이 주어졌을 때의 다음 행동의 등장 횟수 총합
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            # 현재 상태, 행동, 다음 상태에 따른 보상
            reward = self.rewards[(state, action, tgt_state)]
            # 가치 = 보상 + 할인율 * 다음 상태의 가치
            val = reward + GAMMA * self.values[tgt_state]
            # 행동 가치 += 전이확률 * 가치
            action_value += (count / total) * val
        return action_value

    # 주어진 상태에서 가장 q값이 큰 행동을 선택
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    # 테스트 에피소드를 돌려본다.
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            # 테스트 에피소드 와중에도 업데이트는 된다.
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [
                # 현재 상태, 행동이 주어졌을 때의 Q값
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            # 상태별 최대 가치값 저장
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        # 평균 보상을 기준으로 한다.
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        # 보상이 0.8보다 크면 종료
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
```

<br>

코드를 돌려보면 다음과 같은 결과가 나옵니다.

```
$ python 01_frozenlake_v_iteration.py

Best reward updated 0.000 -> 0.050
Best reward updated 0.050 -> 0.400
Best reward updated 0.400 -> 0.450
Best reward updated 0.450 -> 0.550
Best reward updated 0.550 -> 0.650
Best reward updated 0.650 -> 0.750
Best reward updated 0.750 -> 0.800
Best reward updated 0.800 -> 0.950
Solved in 29 iterations!
```

<br>

cross-entropy 방법과 비교했을 때 엄청난 진전입니다. 동일한 환경에 대해 cross-entropy는 60% 성공률을 넘기 위해 굉장히 오랜 시간을 필요로 했는데 value iteration은 29번만에 해냈습니다. 여기에는 두 가지 이유가 있습니다.

1. FrozenLake의 확률론적 결과값과 에피소드 길이(6~10 스텝)로는 cross-entropy가 어떤 에피소드가 잘 된 것인지 판단하기 어렵습니다. value iteration은 상태(또는 행동)의 개별 가치와 함께 작동하며 확률을 추정하고 예상 가치를 계산함으로써 자연스럽게 행동의 확률론적 결과를 통합했습니다. 따라서 가치 반복이 훨씬 간편하고 환경의 데이터 요구량이 훨씬 적습니다(RL에서는 **샘플 효율성**이라고 함).
2. value iteration은 학습 시작을 위해 full episode가 필요하지 않습니다. 일부만으로도 학습 가능합니다. 

<br>

> <subtitle> Q-learning for FrozenLake </subtitle>

두 번째 코드는 *Chapter05/02_frozenlake_q_iteration.py* 파일에 있고 첫 번째 파일과는 작은 차이가 있습니다. 가장 큰 차이는 **value table** 입니다. 이전 예제 코드에서는 상태의 가치를 저장했다면 이번에는 q값을 저장합니다. 즉, 2개의 패러미터(상태, 행동)를 사용합니다. 

두 번째 차이는 **calc_action_value() method가 필요없다**는 점입니다. 행동 가치는 value table에 저장하기 때문입니다. 

마지막 차이는 value_iteration() 에 있습니다. 이전 코드에서 value_iteration()는 calc_action_value()의 wrapper와 다름 없었습니다. 이번에는 value table로 대체되었으니 value iteration()에서 벨만 근사를 해야 합니다.

```python
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    key = (state, action, tgt_state)
                    reward = self.rewards[key]
                    best_action = self.select_action(tgt_state)
                    val = reward + GAMMA * \
                          self.values[(tgt_state, best_action)]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value

```

이 코드는 전 예제에서 calc_action_value()와 상당히 유사합니다. V와 Q 값의 관계를 보면 그럴 수 밖에 없습니다. (위 수식에서도 확인 가능)

<br>

```python
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action
```

v iteration에서 select_action은 행동 가치를 계산했겠지만, 여기서는 value table에서 가져오기만 하면 됩니다. 이 부분은 사실 작은 개선이지만 calc_action_value에서 사용한 데이터를 생각해 보면 RL에서 V-function 학습보다 Q-function 학습이 훨씬 더 인기 있는 이유가 분명해보입니다. 

이번 calc_action_value 함수는 보상과 확률에 대한 정보를 모두 사용합니다. 이는 학습 중에 이러한 정보에 의존하는 value iteration method에는 큰 문제가 되지 않습니다. 그러나 다음 장에서는 확률 근사치가 필요 없고 환경 샘플에서 추출하는 value iteration 확장판에 대해 알아볼 예정인데 이러한 방법의 경우, 확률에 대한 의존성은 에이전트에 추가적인 부담을 가중시킵니다. **Q-learning의 경우, 에이전트가 결정을 내리는 데 필요한 것은 Q값 뿐입니다.**

V function이 완전히 쓸모없다는 게 아닙니다. (actor-critic에서 또 사용됩니다.) 하지만 value learning 영역에서는 Q function이 선호도가 높습니다.

이 예제에서는 결과 차이가 거의 없습니다.

```
$ python 02_frozenlake_q_iteration.py

Best reward updated 0.000 -> 0.100
Best reward updated 0.100 -> 0.200
Best reward updated 0.200 -> 0.400
Best reward updated 0.400 -> 0.700
Best reward updated 0.700 -> 0.750
Best reward updated 0.750 -> 0.800
Best reward updated 0.800 -> 0.900
Solved in 21 iterations!

```

<br>

> <subtitle> Summary </subtitle>

이번 챕터에서는 Deep RL에서 널리 쓰이고 있는 중요한 개념들(상태 가치, 행동 가치, 벨만 방정식)을 배웠습니다.

value iteration 방법에 대해 다뤘고 FrozenLake 환경에서 실험해보았습니다. 

다음 챕터에서는 deep Q-networks에 대해 알아보겠습니다. DQN은 아타리 2600개의 게임들 중 많은 게임에서 인간을 이겨 2013년 deep RL 혁명을 시작한 장본인입니다.

<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 05 : Tabular Learning and the Bellman Equation
* [벨만 방정식 위키피디아](https://en.wikipedia.org/wiki/Bellman_equation){:target="_blank"}

<br>
