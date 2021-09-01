---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 08 : DQN Extensions"
date: 2021-09-01
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 8, DQN, DQN 변형 버전, n-step DQN, Double DQN, Noisy Network, prioritized replay buffer, dueling DQN, categorical dqn, rainbow]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

2015년 처음 DQN이 나오고 나서부터 개선사항을 반영한 다양한 버전들의 DQN이 나왔습니다. DeepMind의 기본 DQN보다 수렴도, 안정성, 표본 효율성 면에서 성능이 올라갔습니다. 이번 8장에서는 DQN의 여러 버전에 대해 자세히 보려고 합니다.

편리하게도 17년 10월에 DeepMind에서 *"Rainbow: Combining Improvements in Deep Reinforcement Learning"*라는 이름의 논문으로 가장 유명한 7개의 DQN 버전을 정리해놨습니다. 이 논문에서는 7개 방법을 결합하여 아타리 게임의 새로운 SOTA를 달성하였습니다. 이번 챕터는 DQN 확장판들과 이것들을 결합한 버전을 알아볼 것입니다.

DQN 확장판은 다음과 같습니다.  
* N-step DQN: 벨만 방정식을 간단히 풀어헤치는 것으로 수렴 속도와 안정성을 높임. (다만 궁극적 해결책은 아님)
* Double DQN: action의 가치를 과대평가하는 DQN에 대처함
* Noisy Network: 노이즈를 네트워크 가중치에 추가함으로써 더 효율적으로 탐험
* Prioritized replay buffer: 경험 데이터를 uniform sampling 하는 것이 학습하는데 최고의 방법이 아닌 이유를 보여줌
* Dueling DQN: 네트워크 아키텍쳐를 풀려고 하는 문제를 더 가깝게 표현하게 하여 수렴 속도를 높임
* Categorical DQN: 단일 행동 가치 기댓값을 넘어서서 full distribution에서 작동하게 함.

<br>

> <subtitle> Basic DQN </subtitle>

기본적인 DQN에 대한 것은 [6장](https://liger82.github.io/rl/rl/2021/07/02/DeepRLHandsOn-ch06-Deep-Q-Networks.html){:target="_blank"} 에서 다뤘기 때문에 내용적인 측면은 생략합니다. 다만 7장의 고수준의 라이브러리인 PTAN 과 ignite 을 도입하여 코드를 더 간결하게 바꿨습니다. 

* *Chapter08/lib/dqn_model.py*: DQN 뉴럴넷 코드이고 6장의 내용과 동일. (이 부분은 생략함)
* *Chapter08/lib/common.py*: 이 챕터에서 공용으로 사용하는 함수들과 변수들
* *Chapter08/01_dqn_basic.py*: PTAN과 ignite를 사용하여 구현한 기본 DQN

<br>

## Common library

### hyperparameter

*lib/common.py* 파일부터 보면, 이전 챕터에서 Pong의 환경을 위한 hyperparameter가 있습니다. *SimpleNamespace* object에 key:value 형태로 담아놨습니다.  

```python
HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    }),
```

SimpleNamespace 인스턴스는 다음처럼 간단하게 사용할 수 있습니다. 나머지 게임도 이런 방식으로 hyperparameter가 저장되어 있습니다.

```python
params = common.HYPERPARAMS['pong']
print("Env %s, gamma %.2f" % (params.env_name, params.gamma))
```

<br>

### unpack_batch

unpack_batch 함수는 transition 정보를 담은 배치를 입력으로 받아서 학습에 알맞은 NumpyArray로 변환시켜주는 역할을 합니다. 모든 transition은 *ptan.experience.ExperienceFirstLast* type으로 저장되어 있고 이를 입력으로 받습니다. 이는 다음과 같은 필드를 갖습니다.

* state : 환경으로부터 받은 관찰값
* action : 에이전트가 취한 행동
* reward : 보상값
* last state : 이번 스텝이 마지막이면 None이고, 아니면 가장 최근의 관찰값이 있음

```python
def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)
```

<br>

### calc_loss_dqn

이제 배치 내에서 최종 transition을 어떻게 처리하는지 알아봅시다. 

```python
def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = \
        unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)
```

<br>

> <subtitle> N-step DQN </subtitle>

<br>

> <subtitle> Double DQN </subtitle>

<br>

> <subtitle> Noisy networks </subtitle>

<br>

> <subtitle> Prioritized replay buffer </subtitle>

<br>

> <subtitle> Dueling DQN </subtitle>

<br>

> <subtitle> Categorical DQN </subtitle>

<br>

> <subtitle> Combining everything </subtitle>


<br>

> <subtitle> Summary </subtitle>

* 

<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 08 : DQN Extensions
* [](){:target="_blank"}

<br>
