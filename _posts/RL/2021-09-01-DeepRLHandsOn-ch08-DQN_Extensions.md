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

DQN의 loss function을 계산하는 함수입니다. 코드는 6장의 것과 거의 유사하고 다른 점은 *torch.no_grad()* 를 추가했다는 점입니다.

```python
def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = \
        unpack_batch(batch)

    # torch.tensor로 변환, device 할당
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # (N) -> (N, 1)
    actions_v = actions_v.unsqueeze(-1)
    # state 관찰값을 네트워크의 입력값으로 넣은 후 행동에 따른 상태행동가치(Q값)을 추출한다
    # .gather의 첫 번째 argument는 모으고 싶은 차원 인덱스를 뜻한다. 1이 행동
    # .gather의 두 번째 argument는 선택한 요소의 인덱스에 해당하는 텐서.
    state_action_vals = net(states_v).gather(1, actions_v)
    # (N, 1) -> (N)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        # target network의 입력으로 다음 상태값을 주고 행동 가치가 최대인 다음 상태 값을 추출
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    # .detach() gradient 전파가 안되는 텐서 복사
    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    # behavior policy 의 값과 target policy의 값을 MSE 를 통해 Loss 계산
    return nn.MSELoss()(state_action_vals, bellman_vals)
```

<br>

### utilities

핵심적인 DQN 함수들 이외에도 *common.py* 는 학습 루프, 데이터 생성, 텐서보드 추적 관련한 유용한 도구들을 제공합니다. 

#### EpsilonTracker

EpsilonTracker 는 학습 과정에서의 epsilon 감소를 구현한 간단한 클래스입니다. epsilon 은 에이전트가 랜덤 행동할 확률을 정의합니다. 1.0 에서 0.02 혹은 0.01 로 줄어들게 해놨습니다. 코드 자체는 엄청 사소하지만 거의 모든 DQN 에서 필요한 부분이라 이렇게 공용 유틸리티로 등록해놨습니다.

```python
class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - \
              frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)
```

<br>

#### batch_generator

batch_generator 는 *ExperienceReplayBuffer* (ptan class) 를 입력으로 받아서 끝없이 버퍼로부터 학습 배치를 표집합니다.

```python
def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)
```

<br>

#### setup_ignite

setup_ignite 는 학습 과정과 텐서보드에 중간 출력물을 위한 ignite를 적용시키는 함수입니다. 두 개의 주요 ignite handler 가 있습니다.  
* *EndOfEpisodeHandler* : 에피소드가 끝나는 매 시점마다의 ignite event를 관리
* *EpisodeFPSHandler* : 에피소드가 가지는 시간이나 환경과의 상호작용의 양을 추적하는 클래스로 frames per second(FPS)로 계산합니다.

```python
def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, run_name: str,
                 extra_metrics: Iterable[str] = ()):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)
    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    # 에피소드가 끝날 때마다 다음 항목들을 출력한다.
    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
              "speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    # 평균 보상값이 보상 기준치에 다다르면 게임을 종료하기 전에 다음 항목들을 출력한다.
    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    # 아래는 Tensorboard 관련 내용
    now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics,
        output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)
```

<br>

## Implementation

이제 *01_dqn_basic.py* 에 대해 알아보겠습니다. 

```python
#!/usr/bin/env python3
import gym
import ptan
import argparse
import random

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, common

NAME = "01_baseline"


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    # Pong game이 환경
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # 환경 설정
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)

    # 학습 네트워크 인스턴스 생성
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)

    # 타겟 네트워크 인스턴스 생성
    tgt_net = ptan.agent.TargetNet(net)

    # epsilon-greedy action selector를 가진 에이전트 생성
    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # 에이전트와 환경을 입력으로 받아서 게임 에피소드에 따른 transition을 제공한다.
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma)
    # 이 transitions들을 exprience replay buffer에 저장한다.
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    
    # Optimizer : Adam
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate)
    # 배치 단위 모델 학습을 위한 코드
    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(
            batch, net, tgt_net.target_model,
            gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)

        # 특정 iteration 마다 학습 네트워크와 타겟 네트워크 동기화
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)
    engine.run(common.batch_generator(buffer, params.replay_initial,
                                      params.batch_size))

```

<br>

## Results

```
/Chapter08$ ./01_dqn_basic.py --cuda
Episode 2: reward=-21, steps=809, speed=0.0 f/s, elapsed=0:00:15
Episode 3: reward=-21, steps=936, speed=0.0 f/s, elapsed=0:00:15
Episode 4: reward=-21, steps=817, speed=0.0 f/s, elapsed=0:00:15
Episode 5: reward=-20, steps=927, speed=0.0 f/s, elapsed=0:00:15
Episode 6: reward=-19, steps=1164, speed=0.0 f/s, elapsed=0:00:15
Episode 7: reward=-20, steps=955, speed=0.0 f/s, elapsed=0:00:15
Episode 8: reward=-21, steps=783, speed=0.0 f/s, elapsed=0:00:15
Episode 9: reward=-21, steps=785, speed=0.0 f/s, elapsed=0:00:15
Episode 10: reward=-19, steps=1030, speed=0.0 f/s, elapsed=0:00:15
Episode 11: reward=-21, steps=761, speed=0.0 f/s, elapsed=0:00:15
Episode 12: reward=-21, steps=968, speed=162.7 f/s, elapsed=0:00:19
Episode 13: reward=-19, steps=1081, speed=162.7 f/s, elapsed=0:00:26
Episode 14: reward=-19, steps=1184, speed=162.7 f/s, elapsed=0:00:33
Episode 15: reward=-20, steps=894, speed=162.7 f/s, elapsed=0:00:39
Episode 16: reward=-21, steps=880, speed=162.6 f/s, elapsed=0:00:44
...
```

기본 DQN 버전에서는 평균 보상이 18점(보상 목표값)에 도달하려면 보통 1백만 프레임 정도 걸립니다. 
학습 도중에 학습 과정의 흐름을 Tensorboard로 엿볼 수 있습니다.

<br>
<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.1_2.png" width="90%"></center><br>

> <subtitle> N-step(Multi-step) DQN </subtitle>

<br>
<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.3.png" width="80%"></center><br>
(그림 출처 : 김경환씨 rainbow 발표)

기존 DQN은 단일 step 의 보상을 축적하고 다음 스텝에서의 최대 가치를 갖는 행동을 사용하여 부트스트랩하였습니다. multi-step DQN은 주어진 상태 $$S_t$$로 부터 **truncated** n-step return 을 정의하여 사용합니다.

$$ R_t^{(n)} \equiv \sum_{k=0}^{n-1} \gamma_t^{(k)} R_{t+k+1} $$

이는 Sutton의 n-step TD 방식과도 다릅니다. Sutton(1988)은 forward-view multi-step target 을 사용했습니다.

multi-step DQN에 대한 알고리즘을 살펴보려면 식을 다시 볼 필요가 있습니다. 아래는 Q-learning의 벨만 업데이트 식입니다.

$$ Q(s_t, a_t) = r_t + \gamma \max_a Q(s_{t+1}, a_{t+1}) $$

이 식은 재귀적으로 펼쳐질 수 있습니다. 위 식의 $$ Q(s_{t+1}, a{t+1}) $$ 부분을 또 위 식으로 대입하면 다음과 같이 됩니다.

$$ Q(s_t, a_t) = r_t + \gamma \max_a [r_{a,t+1} + \gamma \max_{a'} Q(s_{t+2}, a')] $$

$$ r_{a,t+1} $$ 는 행동 a 후에 * t+1 * 타임 스텝에서 local reward 를 의미합니다. *t+1* 에서 행동 a 를 최적값으로 선택하거나 최적에 가깝게 선택한다고 가정하면 $$ \max_a $$ operation 생략할 수 있어서 다음과 같이 표현할 수 있습니다. 

$$ Q(s_t, a_t) = r_t + \gamma r_{t+1} + \gamma^2 \max_{a'} Q(s_{t+2}, a') $$

이 식은 계속해서 펼쳐질 수 있습니다. 이러한 속성을 DQN 업데이트에도 적용시킨 것입니다. 

1998년 Sutton과 Barto가 multi-step 이 학습 속도를 빠르게 한다고 했습니다. 실제 돌려봤을 때도 n이 1,2,3 늘어날 때 학습이 빨라지는 것을 확인할 수 있었습니다. 그러나 이는 어느 정도 숫자까지만 적용되었습니다. 숫자가 터무늬 없이 커질 때 예를 들어 100이 되면 오히려 성능이 떨어질 수 있습니다.

그 원인 첫 번째는 multi-step 과정에서 중간 단계에서 최대값을 갖는 행동을 선택할 것이라는 가정 아래 max operation 을 생략했는데 그러지 않을 수 있다는 점입니다. 랜덤하게 행동을 취했을 경우에 계산된 Q값은 감소될 것입니다. 그래서 이 방식으로 할 경우 벨만 방정식을 더 풀어헤칠수록(n이 커질수록) 업데이트가 부정확해질 수 있는 것입니다. 

원인 두 번째는 experience replay buffer 의 큰 사이즈가 상황을 악화시킬 수 있다는 점입니다. "n-stepness" 가 off-policy 였던 DQN을 on-policy로 만든다는 점이 문제입니다. off-policy 는 데이터의 신선함에 의존하지 않습니다. off-policy는 학습과 타겟 정책이 다르기 때문에 몇 백만 스텝 전에 썼던 데이터를 샘플링해서 써도 문제가 없기 때문에 buffer 사이즈를 크게 만들어두었습니다. 반면 on-policy는 정책이 하나여서 학습 데이터의 신선함에 엄청 의존합니다. multi-step learning 은 replay buffer가 크다보니 이곳에서 받은 데이터에서 예전의 안좋은 정책의 경험으로부터 학습할 수 있다는 게 상황을 악화시키는 원인입니다. 

이런 단점들에도 불구하고 실전에서 multi-step learning은 종종 쓰입니다. 왜냐하면 실전은 흑백으로 나뉘는 상황이 아니기 때문에 DQN의 학습 속도를 높이는 적절한 N을 설정하면 효과가 있어서 입니다. 보통 2, 3과 같이 작은 값을 썼을 때 잘 작동합니다. 

<br>

## Implementation

*ExperienceSourceFirstLast* class 가 multi-step Bellman unroll을 지원해서 아~~주 간단하게 multi-step DQN을 구현할 수 있습니다. 

앞선 코드와 차이는 두 군데입니다.

* *ExperienceSourceFirstLast* 인스턴스 생성시 *steps_count* parameter에 N을 주면 됩니다.  
```python
exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=args.n)
```

* *calc_loss_dqn* 함수에 감마 값 n 제곱해주는 것입니다. 이는 수렴에 안 좋은 영향이 있을 수 있습니다. 체인의 마지막 상태를 위한 discount coefficient가 감마가 아니라 감마의 n 제곱을 곱한 값이기 때문입니다.  
```python
    loss_v = common.calc_loss_dqn(
        batch, net, tgt_net.target_model,
        gamma=params.gamma**args.n, device=device)
```

<br>

## Results

베이스라인(step=1)과 비교해보면 n이 2,3 일 때 수렴 속도가 훨씬 빨라지는 것을 볼 수 있습니다. 

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.4.png" width="80%"></center><br>

> <subtitle> Double DQN </subtitle>

Q-learning 은 최대화하는 부분이 반복되기 때문에 과대평가하는 문제(Maximization Bias)가 발생하고 이는 학습 효율을 떨어뜨립니다. 이 문제를 해결하고자 나온 것이 Double Q-learning 입니다.

Double Q-learning 은 두 개의 독립적인 Q function을 사용합니다.  
* 최댓값을 갖는 행동을 선택하는 용도
* Q function 평가하는 용도

$$ A^* = argmax _a Q_1 (a) $$

$$ Q_2 (A^*) = Q_2(argmax _a Q_1 (a)) $$

$$ Q_2 $$ 가 최종인데 두 Q function의 역할을 고정하면 두 Q function 간 간극이 벌어지므로 역할을 교대로 번갈아가면서 수행합니다.

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
* [김경환씨 rainbow 발표](https://www.slideshare.net/KyunghwanKim27/rainbow-2nd-dlcat-in-daejeon){:target="_blank"}
* [https://wonseokjung.github.io/RL-Totherb7/](https://wonseokjung.github.io/RL-Totherb7/){:target="_blank"}
* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf){:target="_blank"}
* [](){:target="_blank"}

<br>
