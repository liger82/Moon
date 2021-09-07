---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 08 : DQN Extensions"
date: 2021-09-01
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 8, DQN, DQN 변형 버전, n-step DQN, multi-step DQN, Double DQN, Noisy Network, prioritized replay buffer, dueling DQN, categorical dqn, rainbow]
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

종합해서 target Q value에 대한 표현은 다음과 같습니다.

$$ Q(s_t, a_t) = r_{t} + \gamma \max_a Q'(s_{t+1}, argmax_{a} Q(s_{t+1},a)) $$

Double Q-learning에 DQN에도 적용한 것이 Double DQN 입니다. 저자들에 의하면 이러한 작은 수정이 과대평가하는 것을 완전히 고친다고 합니다.

<br>

## Implementation

이번에도 기존 DQN 코드에서 바꿀 내용이 많지는 않습니다. loss function 을 수정하는 것이 대부분입니다. 

완성된 코드는 *Chapter08/03_dqn_double.py* 에서 확인할 수 있습니다.

loss function 은 다음과 같습니다.

```python
# loss function for double DQN
def calc_loss_double_dqn(batch, net, tgt_net, gamma,
                         device="cpu", double=True):
    # double : double dqn 방식 사용여부
    states, actions, rewards, dones, next_states = \
        common.unpack_batch(batch)

    # torch.tensor & assign device
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    # 학습 네트워크에 상태값을 입력으로 주어 상태 행동 가치를 얻는다.
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        if double:
            # 학습 네트워크에 다음 상태값을 입력으로 넣었을 때 best action 계산
            next_state_acts = net(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            # 하지만 타켓 네트워크로부터 이 best action 에 부합하는 상태값을 받아옴
            next_state_vals = tgt_net(next_states_v).gather(
                1, next_state_acts).squeeze(-1)
        else:
            next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0
        # approximated Q-values
        exp_sa_vals = next_state_vals.detach() * gamma + rewards_v
    # MSE loss between Q-values predicted by the network and approximated Q-values.
    return nn.MSELoss()(state_action_vals, exp_sa_vals)
```

그 다음 *calc_values_of_states()* 는 보류 상태의 값을 계산합니다. 

```python
@torch.no_grad()
def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    # split held-out states array into equal chunks
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        # pass every chunk to the network to obtain action values
        action_values_v = net(states_v)
        # From action values, choose the action with the largest value (for every state)
        best_action_values_v = action_values_v.max(1)[0]
        # calculate the mean of such values and store it
        # 총 1,000개의 states를 저장하기 때문에 충분히 커서 mean value의 변화를 볼 수 있다.
        mean_vals.append(best_action_values_v.mean().item())
    # calculate the mean of mean_vals
    return np.mean(mean_vals)
```

이 코드 파일의 나머지는 basic DQN 코드와 동일합니다. 이 두 차이 부분이 서로 꼬여있는 loss function을 사용하게 하고, 주기적인 평가를 위한 1,000개의 상태를 랜덤하게 샘플링하는 것을 유지시켜줍니다. 

<br>

## Results

**--double** 을 argument를 추가/삭제하여 두번 돌려서 Double DQN과 Basic DQN을 비교하였습니다.

GTX 1080 Ti 에서 했을 때 백만 프레임 학습은 2시간 정도 걸렸습니다. 

기본 버전에 비해 double DQN의 수렴 빈도가 낮다는 것을 알게 되었습니다. 기본 DQN에서는 10회 시도 중 약 1회가 수렴에 실패하지만 double은 3회 중 약 1회입니다. 하이퍼 파라미터 조정이 필요할 가능성이 높지만, 하이퍼 파라미터에 손대지 않고 확장 버전의 효과를 확인할 수 있도록 비교하려고 동일하게 했습니다.

두 모델을 비교해본 결과, double DQN의 평균 보상이 더 빠르게 올라가는 것을 볼 수 있습니다. 다만 문제를 푸는 최종 시간은 이 코드로는 비슷했습니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.6.png" width="80%"></center><br>

평균 보상 메트릭 말고도 보류 상태들에 대한 평균 값의 변화를 보여준 차트도 있습니다. 기본 DQN 은 값을 과대평가해서, 값이 어느 수준 이상 가면 떨어지는 경향이 있습니다. double DQN 은 꾸준히 오르고요. 그런데 지금 실험 상황이 Pong 이라는 너무 간단한 환경이라서 이것이 정확히 표현되지 않았습니다. 더 복잡한 게임에서는 double DQN 이 더 나은 결과를 가져올 것이라고 저자들은 말합니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.7.png" width="80%"></center><br>

> <subtitle> Noisy networks </subtitle>

이번 네트워크는 RL 의 또 다른 문제인 환경 탐색에 주목합니다. 기존 DQN 은 epsilon-greedy 방식으로 탐색의 정도를 줄여나갔습니다. 짧은 에피소드의 단순한 환경일 때는 이 기법이 잘 작동했지만 큰 규모와 복잡한 환경에서는 비효율적이었습니다. 

이 새로운 네트워크 저자의 해결책은 간단했습니다.

<center>"Network 에 noise 를 추가하여 exploration 을 하자!"</center>

구체적으로는, 네트워크의 fully connected layer의 가중치에 noise를 추가하고 역전파를 이용해서 학습하는 동안 noise 의 패러미터에 적응해나가도록 한 것입니다. 
물론 이 방법도 어디를 더 탐험할지 결정하는 방법은 아닙니다. 21장에서 더 진보한 탐험 기법에 대해 다룰 것입니다. 

이 방법의 기안자는 두 가지 noise 추가 방법을 제안하였습니다. 이 둘은 다른 계산 오버헤드를 지닙니다.

1. Independent Gaussian noise 
    - fully connected layer의 모든 가중치에 정규분포에서 뽑은 랜덤 값을 가집니다. 노이즈의 패러미터 $\mu$, $\sigma$ 는 레이어 안에 저장되고, 일반적인 선형 레이어의 가중치를 학습하는 동일한 방식으로 역전파를 이용해 학습됩니다. noisy layer의 출력값은 선형 레이어와 동일한 방식으로 계산됩니다.
2. Factorized Gaussian noise
    - 표집하는 랜덤 값의 개수를 최소화하기 위해 두 개의 랜덤 벡터를 유지합니다. 입력 크기의 벡터, 출력 크기의 벡터입니다. 그 다음, 벡터들 간 외적(텐서곱)을 통해 레이어의 랜덤 행렬이 만들어집니다. 

<br>

## Implementation

이 노이즈 네트워크는 nn.Linear 를 상속 받아 커스텀하였고, *Chapter08/lib/dqn_extra.py* 에서 *NoisyLinear* 는 independe Gaussian noise, *NoisyFactorizedLinear* 는 factorized noise 버전을 위한 클래스입니다. 

```python
# for independent Gaussian noise
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        # create a matrix of sigma
        # (mu의 값은 nn.Linear 를 상속받은 행렬에 저장될 것이다.)
        w = torch.full((out_features, in_features), sigma_init)
        # sigma를 학습 가능하도록 하기 위해 nn.Parameter로 래핑함
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        # register_buffer 는 네트워크 내의, 역전파하는 동안 업데이트되지 않고 nn.Module 에 의해 통제되는 텐서를 만드는 메서드이다.
        self.register_buffer("epsilon_weight", z)
        if bias:
            # sigma_init==0.017 은 Noisy Network 논문에 나온 수치
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    # nn.Linear에서 override 한 메서드로, 내용은 논문의 추천을 따랐음.
    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        '''
        weight와 bias buffer 둘 다에서 랜덤 노이즈를 표집하고 nn.Linear가 하는 방식으로 input data의 linear transformation 을 수행
        '''
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * \
                   self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        return F.linear(input, v, bias)
```

<br>

다음은 Factorized Guassian noise 를 적용한 코드입니다. 위 코드와 비슷한 점이 많고,(이 코드로는) 결과가 그리 다르지 않습니다. 

```python
# for factorized Gaussian noise
class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features,
                 sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(
            in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z1 = torch.zeros(1, in_features)
        self.register_buffer("epsilon_input", z1)
        z2 = torch.zeros(out_features, 1)
        self.register_buffer("epsilon_output", z2)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * \
                         torch.sqrt(torch.abs(x))
        # input size의 벡터
        eps_in = func(self.epsilon_input.data)
        # output size의 벡터
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        # 텐서곱으로 노이즈를 만든다. 
        noise_v = torch.mul(eps_in, eps_out)
        v = self.weight + self.sigma_weight * noise_v
        return F.linear(input, v, bias)
```

학습 과정에서 내부 노이즈 레벨을 확인하기 위해 노이즈 레이어의 signal-to-noise ratio(SNR) 을 모니터링할 것입니다.  
* SNR = RMS($$\mu$$) / RMS($$\sigma$$)
    - RMS : root mean square (제곱 평균 제곱근)
* SNR 은 노이즈 레이어의 정적인 구성요소가 주입된 노이즈보다 몇 배 더 큰지 보여줍니다.

<br>

# Results

모델은 60만(600k) 프레임만에 18점(목표치)에 도달합니다. 베이스라인(basic DQN)보다도 빠른 학습 속도를 보이는 것이 확연히 보입니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.8.png" width="90%"></center><br>

figure 8.9 를 보면 두 레이어 모두에서 노이즈 레벨이 빠르게 감소하는 것을 볼 수 있습니다. 첫 번째 레이어(independent guassian noise)는 1 에서 거의 1/2.5 까지 갔고 두 번째 레이어(factorized guassian noise)는 1/3 에서 1/15 까지 감소했습니다. 흥미로운 점은 250k 프레임 이후에는 두 번째 레이어의 노이즈 수준이 다시 증가하기 시작하면서 에이전트가 환경을 더 많이 탐색하게 되었습니다. 이는 높은 점수 수준에 도달한 후 에이전트가 기본적으로 좋은 수준에서 플레이할 줄 알면서도 결과를 더욱 개선하기 위해 행동을 "연마"해야 하기 때문에 의미가 있습니다. 

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.9.png" width="90%"></center><br>

> <subtitle> Prioritized replay buffer </subtitle>

### Prioritizing with TD-error

이번 세션의 아이디어는 *Prioritized Experience Replay* 라는 논문에서 나온 것으로 training loss에 따라 replay buffer 의 샘플의 우선순위를 달리함으로써 샘플 효율성을 높이고자 했습니다.

간단히 말하면, 
* PER은 replay buffer의 성능 개선으로 DQN을 바꾸고자 함
* replay buffer 의  저장된 경험을 우선순위를 정해 우선순위가 높은 것을 표집하도록 한다.
* 우선순위는 TD error에 기반한다. TD error 가 클수록 우선순위를 높인다.

저자가 유의한 점은 양질의 경험을 중요시 여기면서도 샘플의 새로움도 놓치면 안되도록 밸런스를 맞추고자 했다는 점입니다. 버퍼의 작은 섭셋에만 집중하면 i.i.d 속성을 잃고 해당 섭셋에 과적합할 것이기 때문입니다.

그러나 위 내용만 보면 *greedy TD-error prioritization* 알고리즘과 동일합니다. 
이 알고리즘의 원리는 다음과 같습니다

1. 매 transition을 따라 TD error를 계산해 replay memory에 저장한다.
2. TD error 의 크기가 가장 큰 transition은 memory로부터 replay 된다.
3. 각 transition 에는 q-learning update 가 진행되며, TD error에 비례하도록 업데이트된다.
4. 새로운 transition 은 가장 높은 우선순위를 부여하여 momory 에 저장한다. 이는 한 번 이상 replay 되는 것을 보장하기 위함이다.

PER 제안 논문에서는 이 greedy TD-error 우선순위 방법에 드러나 문제점을 극복하면서 새로운 방법을 제안합니다.

greedy TD error 우선순위 알고리즘의 문제점은 다음과 같습니다.

1. TD error 가 replay된 transition 에 대해서만 업데이트된다. (모든 메모리에 대해 계산하는 것은 계산 부담이 크다)
    - 이로 인해 처음에 TD-error 가 낮게 평가된 transition에 대해 방문할 기회가 사라진다.
    - 일부 데이터에 집중하여 다앙햔 경험을 충분히 전달하지 못한다. -> overfitting 위험 올라감.
2. (보상이 확률적인 경우) noise에 취약하다.

<br>

### Stochastic Prioritization

PER 논문 저자들은 이러한 문제점을 해결하기 위해 greedy 우선순위 방법과 uniform random sampling 방법을 섞은 **stochastic sampling method**를 제안합니다.

이는 replay memory 내의 transition의 우선순위는 유지하되, 모든 transition 에 대해서 non-zero 확률의 방문이 가능하도록 보장할 수 있습니다. (처음 진입 이외에도)

샘플의 우선순위를 수학적으로 표현하면 다음과 같습니다.

$$ P(i) = \frac{p_i^{\alpha}}{\sum _k p_k^{a'}} $$

* P(i): k개의 transition 중 i번째 transition의 sampling 확률
* $$\alpha$$ : 얼마나 우선순위에 의한 샘플링을 많이 할 것인가를 결정([0,1])
    - 0 이면 기본 DQN에서처럼 uniform sample
    - 1 이면 greedy prioritization
    - 튜닝이 필요한 hyperparameter로, 논문은 0.6을 시작점으로 하라고 권고
* $$p_i$$ 는 버퍼 내 i 번째 샘플의 우선순위

우선순위를 정의하는 방법은 두 가지가 있으며, 직접적인 방법으로는 **proportional prioritization** 방식으로 TD error 에 비례하지만 작은 constant 값을 포함시켜줌으로써 모든 transition의 방문 확률을 0이 아니도록 만들어줍니다.

$$p_i = |\delta| + \epsilon $$
$$TD-error : \delta = (R + \gamma \max_a Q(S', a)) - Q(S,A)$$

간접적인 방법인 rank-based prioritization 은 replay memory 내의 transition에 TD error 에 따라 rank 를 매기는 것입니다.

$$ p_i = \frac{1}{rank(i)} $$

rainbow 논문에서도 그렇고 **proportional prioritization** 방식이 더 인기있는 방법이라고 합니다. 

<br>

### Annealing the Bias

prioritized replay 는 보통 편향치를 가져오는데, 주로 expectation 에 대한 distribution이 정형화되지 않은 상태인데 업데이트 때마다 또 바뀌기 때문입니다.

이에 대해서는 **importance-sampling weights** 를 이용해 bias 를 잡으려고 했습니다.

$$ w_i = (\frac{1}{N} \cdot \frac{1}{P(i)})^{\beta}$$

Q-learning 부분에서 TD-error 대신 weighted IS 를 곱한 것을 이용해 업데이트합니다. 

일반적인 RL 시나리오에서 unbiased updates 는 학습 막바지에 수렴하도록 하는 가장 중요한 역할을 합니다. 이 논문에서는 importance sampling correction 정도를 점진적으로 상승시켜서 학습 막바지에 최대로 correction 이 되도록 합니다. 

$$\beta$$ 가 correction의 정도를 조절하는 패러미터이고 학습시 선형적으로 상승하여 학습 마지막에는 1이 되도록 합니다. 특히 우선순위에 대한 조절계수인 $$\alpha$$ 와 함께 올려주면 더욱 확실하게 correction 이 이루어진다고 합니다.

뉴럴넷과 같은 비선형 근사함수와 prioritized replay를 함께 했을 때, importance sampling의 또 다른 이점이 있습니다.

gradient의 first-order approximation의 경우 일반적으로 local하게만 신뢰할 수 있고 큰 step으로 학습할 때는 성능이 좋지 않습니다.
prioritization 과정에서 transition의 높은 에러가 learning step을 넓게 만들어주기도 하는데, 이 과정에서 IS의 correction이 gradient의 크기를 줄여줘서 효과적으로 step의 크기를 줄여준다는 것입니다. 
(마지막 부분은 저도 이해가 안되는데 일단 그렇다고 합니다.)

<br>

## Implementation

이번 구현에서는 몇 가지 변화가 있습니다.

1. 새로운 replay buffer
    - 우선순위 추적
    - 우선순위에 따라 배치 샘플링
    - 가중치 계산
    - loss 계산 후 우선순위 업데이트 
2. loss function
    - 가중치를 모든 샘플에 결합시키기
    - 표집한 transition의 우선순위를 조정하기 위해 loss 값을 replay buffer로 돌려보내기

코드는 *Chapter08/05_dqn_prio_replay.py* 에서 확인할 수 있습니다. 단순성을 위해 새로운 우선순위 replay buffer 클래스는 이전 replay buffer 와 매우 유사한 저장 체계를 사용합니다. 하지만 우선순위 지정을 위한 새로운 요구 사항으로 인해 버퍼 크기에 O(1) 시간 내에 샘플링을 구현하는 것이 불가능합니다. 

단순 리스트를 사용하는 경우, 새로운 배치를 샘플링할 때마다 모든 우선순위를 처리해야 합니다. 따라서 샘플링은 버퍼 크기에 비례하여 O(N) 시간의 복잡성을 가집니다. 10만 개 샘플처럼 버퍼가 작으면 큰 문제가 되지 않지만 수백만 개의 transition이 있는 실제 대용량 버퍼는 문제가 될 수 있습니다. segment tree 데이터 구조를 사용하는 것과 같이 O(log N) 시간 내에 효율적인 샘플링을 지원하는 다른 스토리지 체계도 있습니다. 

1. [OpenAI Baselines project](https://github. com/openai/baselines){:target="_blank"} 
2. *ptan.experience.PrioritizedReplayBuffer* class 내에 효율적인 우선순위 replay buffer를 제공

일단 내부 구조를 살펴보기 위해 리스트를 사용하여 버퍼를 쓰는 단순한 버전의 우선순위 replay buffer를 살펴보겠습니다.
(이 코드는 *dqn_extra.py*에 있습니다.)

```python
# replay buffer params
# beta는 correction 정도를 조절하는 패러미터
# 최초 100k frame 동안 0.4에서 시작하여 1.0까지 상승시킨다.
BETA_START = 0.4
BETA_FRAMES = 100000


class PrioReplayBuffer:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = [] # 단순하게 리스트로 구성
        # 우선순위 저장
        self.priorities = np.zeros(
            (buf_size, ), dtype=np.float32)
        self.beta = BETA_START

    # 최초값 0.4에서 1.0까지 조금씩 업데이트한다.
    # 주기적으로 호출 필요
    def update_beta(self, idx):
        v = BETA_START + idx * (1.0 - BETA_START) / \
            BETA_FRAMES
        self.beta = min(1.0, v)
        return self.beta

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        # 버퍼가 빈값이 아니면 최대 우선순위 반환
        max_prio = self.priorities.max() if \
            self.buffer else 1.0
        for _ in range(count):
            # count 만큼 ExperienceSource object에서 transition을 추출하여 버퍼에 저장한다.
            sample = next(self.exp_source_iter)
            # 버퍼에 여유가 있으면 추가
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else: # 버퍼가 꽉 찼으면 교체한다.
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            # buffer는 circular buffer 이다. 
            self.pos = (self.pos + 1) % self.capacity

    # 우선순위들을 알파값을 이용해서 확률로 변환
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha

        probs /= probs.sum()

        # 버퍼 내의 인덱스를 배치사이즈만큼 랜덤하게 고른다
        # p에 확률값을 입력으로 주면, 표본이 추출된 확률을 반영해서 샘플링한다
        indices = np.random.choice(len(self.buffer),
                                   batch_size, p=probs)
        # 인덱스에서 값 추출하여 리스트에 담음
        samples = [self.buffer[idx] for idx in indices]
        # 가중치 계산
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, \
               np.array(weights, dtype=np.float32)

    # 새로운 우선순위를 업데이트한다
    def update_priorities(self, batch_indices,
                          batch_priorities):
        for idx, prio in zip(batch_indices,
                             batch_priorities):
            self.priorities[idx] = prio
```

<br>

다음은 loss 계산 함수입니다. PyTorch 의 *MSELoss* class 는 weights 를 지원하지 않습니다. MSE 는 회귀 문제의 loss 계산에서 사용되고 샘플의 가중치 계산은 보통 분류 문제의 loss 계산에서 활용되기 때문입니다. 그래서 MSE 를 계산하고 그 결과에 가중치를 곱하는 custom function을 만들었습니다.

```python
def calc_loss(batch, batch_weights, net, tgt_net,
              gamma, device="cpu"):
    states, actions, rewards, dones, next_states = \
        common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    # 학습 네트워크의 상태 행동 가치 Q값 계산(prediction)
    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        # 타켓 네트워크의 다음 상태에서의 best action의 가치 계산
        next_states_v = torch.tensor(next_states).to(device)
        next_s_vals = tgt_net(next_states_v).max(1)[0]
        next_s_vals[done_mask] = 0.0
        # target
        exp_sa_vals = next_s_vals.detach() * gamma + rewards_v
    # MSE = (prediction - target)^2
    l = (state_action_vals - exp_sa_vals) ** 2
    # loss = weights * MSE
    losses_v = batch_weights_v * l
    # 1e-5 는 loss가 0인 상황에 대비하는 상수
    return losses_v.mean(), \
           (losses_v + 1e-5).data.cpu().numpy()
```

<br>

그 다음 main 함수에서는 위에서 설명한 우선순위 replay buffer 생성하는 부분과 process_batch() 에서 처리 과정을 수정하였습니다.

```python
if __name__ == "__main__":
    
    ...(생략)...
    
    # 우선순위 replay buffer 인스턴스 생성
    buffer = dqn_extra.PrioReplayBuffer(
        exp_source, params.replay_size, PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch_data):
        batch, batch_indices, batch_weights = batch_data
        optimizer.zero_grad()
        loss_v, sample_prios = calc_loss(
            batch, batch_weights, net, tgt_net.target_model,
            gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        # 버퍼로 다시 우선순위를 돌려보내서 버퍼 내 샘플의 우선순위를 업데이트한다
        buffer.update_priorities(batch_indices, sample_prios)
        epsilon_tracker.frame(engine.state.iteration)
        # 동기화
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
            # 베타 상승
            "beta": buffer.update_beta(engine.state.iteration),
        }
    
    ...(생략)...

```

<br>

## Results

Prioritized replay buffer 는 문제를 푸는데 베이스라인과 거의 유사하게 2시간 걸렸습니다. (왼쪽이 기본 DQN, 오른쪽이 우선순위 DQN)

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.10.png" width="90%"></center><br>

그렇지만 더 적은 학습 이터레이션과 더 적은 에피소드로 문제를 해결했습니다. 
물론 이는 비효율적인 버퍼를 사용해서 그렇습니다. 

Figure 8.11에서는 베이스라인보다 Prioritized replay buffer 가 더 낮은 loss 를 가짐을 알 수 있습니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.11.png" width="90%"></center><br>

> <subtitle> Dueling DQN </subtitle>

Dueling DQN 은 정확한 값보다 차이를 배우는 것이 더 쉽다는 데에서 시작합니다. 그래서 Q 를 두 가지로 나눕니다. 상태의 가치 V(s) 와 그 상태에서 행동의 advantage A(s,a) 입니다. 

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.12-1.png" width="60%"></center><br>

Dueling DQN 은 네트워크 아키텍쳐에서 value와 advantage 를 명백하게 분리하여 (아타리 게임에서) 더 나은 학습 안정성, 빠른 수렴 속도, 더 나은 성능을 가져왔습니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.12.png" width="80%"></center><br>

Dueling DQN에서 Q 를 구하는 방식이 3가지가 있습니다.

* Sum : 단순 sum으로는 Q 에 대한 V와 A 값이 unique하지 않음. Q가 4일 때 V, A가 (1,3), (2,2), (3,1) 처럼 여러 경우가 존재
$$ Q(s, a; \theta, \alpha, \beta) = V(s;\theta, \beta) + A(s, a ; \theta, \alpha) $$  
* Max : 유일한 V와 A를 보장  
$$ Q(s, a; \theta, \alpha, \beta) = V(s;\theta, \beta) + (A(s, a ; \theta, \alpha) - \max_{a' \in |A|}A(s, a' ; \theta, \alpha))$$  
* **Average** : 유일한 V와 A를 보장하지는 않지만, max와 유사한 성능을 보이며, 최적화의 안정성이 증가하는 효과 있어서 이 방식을 사용
$$ Q(s, a; \theta, \alpha, \beta) = V(s;\theta, \beta) + ((A(s, a ; \theta, \alpha) - \frac{1}{|A|}\sum_{a'}A(s, a'; \theta, \alpha))) $$

<br>

## Implementation

코드는 *Chapter08/06_dqn_dueling.py* 에서 학습 프로세스 진행을 할 수 있고 *lib/dqn_extra.py* 에서 DuelingDQN class 를 확인할 수 있습니다.

학습 과정 자체는 바뀐 것이 거의 없어서 DuelingDQN 만 살펴보겠습니다. 

```python
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        
        # convolution layers 부분은 기본 DQN과 동일
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # inner dim 512 -> 256
        # fully connected layer for advantage
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        # fully connected layer for value prediction
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        # average 방법
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)
```

<br>

## Results

Figure 8.13을 보면 기본 DQN보다 수렴하는 속도가 빠른 것을 볼 수 있습니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.13.png" width="80%"></center><br>

V와 A를 분리해서도 보면, advantage는 0과 그리 다르지 않지만 시간이 지남에 따라 조금씩 상승하는 것을 확인할 수 있습니다. value는 Double DQN 과 닮았습니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210901-DeepRLHandsOn-ch08-DQN_Extensions/fig8.14.png" width="80%"></center><br>


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
* [PER 참고 Lunabot87 블로그](https://ropiens.tistory.com/86){:target="_blank"}
* [](){:target="_blank"}

<br>
