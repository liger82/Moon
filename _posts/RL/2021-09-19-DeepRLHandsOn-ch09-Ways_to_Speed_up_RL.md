---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 09 : Ways to Speed up RL"
date: 2021-09-19
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 9, Ways to Speed up RL]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

[8장 DQN extensions](https://liger82.github.io/rl/rl/2021/09/01/DeepRLHandsOn-ch08-DQN_Extensions.html){:target="_blank"} 에서 DQN 을 더 안정적이고 수렴을 빠르게 하는 실용적인 트릭들을 배웠습니다. 이 트릭들은 더 적은 학습 시간과 함께 더 좋은 정책을 얻기 위해 DQN 방법 자체를 수정하는 형태라면 9장에서는 구현의 디테일한 부분을 수정함으로써 학습 속도를 개선하고자 합니다. 순수한 엔지니어링 관점입니다.

이 챕터에서는 다음 내용을 다룰 예정입니다.  
* 8장의 Pong 환경을 그대로 사용하고 이 게임을 가능한 한 빨리 풀고자 한다.
* 단계적인 방식으로 동일한 하드웨어 장비를 사용하여 Pong을 3.5배 빠르게 해결한다.
* 미래에 일반적인 방식이 될만한 fancy한 RL 학습 속도를 늘릴만한 방법에 대해 논의한다.

<br>

> <subtitle> Why speed matters </subtitle>

이 세션에서는 왜 속도가 중요한지 왜 속도를 최적화해야 하는지 알아보겠습니다. 

지난 1~20년 간 하드웨어의 성능이 엄청나게 개선되었습니다. 2005년에 CFD(computational fluid dynamics, 전산 유체 역학) 시뮬레이션을 위한 수퍼컴퓨터를 만들 때, 시스템은 64개의 서버(42인치 랙 3개와 쿨링 및 파워 시스템 필요)로 구성되어 있었습니다. 
쿨링 시스템을 제외한 하드웨어만으로 거의 백만(1M) 달러가 들었습니다.  

이 수퍼컴퓨터의 이론적인 성능은 922 GFLOPS(초당 10억개의 부동소수점 연산)이었지만, 16년에 나온 GTX 1080 Ti에 비하면 수퍼컴퓨터의 큰 부피나 기능이 보잘 것 없어 보입니다.
GTX 1080 Ti 하나로 11,340 GFLOPS 성능을 낼 수 있습니다. 가격도 카드당 700 달러로 엄청나게 줄었습니다. 

머신러닝을 비롯한 AI 분야의 성장은 데이터 활용성과 컴퓨팅 파워가 증가해서라고 말합니다. 
한 기계에서 한 달이 걸리는 연산을 상상해 보십시오(CFD 및 기타 물리학 시뮬레이션에서 매우 일반적인 상황). 만약 우리가 속도를 5배 높일 수 있다면, 한달이라는 시간은 6일로 줄어들 것입니다. 속도를 100배 높이면 8시간이 걸릴거고 하루 만에 세 번이나 돌릴 수 있습니다. 같은 돈으로 2만 배나 더 많은 전력을 아낄 수 있습니다. 

이런 현상은 "big iron"(고성능 컴퓨팅) 세계에서만 일어난 것이 아닙니다. 현대의 micro controller에서도 나타납니다. 작은 스마트폰에 이젠 4~8개의 코어, GPU, 몇기가의 RAM 의 스펙을 갖추고 있습니다. 물론 이런 변화에는 코드의 개선도 한 몫 했습니다. 
현재 하드웨어의 모든 능력을 쓰려면 코드를 병렬화해야 됩니다. 이는 분산 시스템, 데이터 위치, 통신, 하드웨어와 라이브러리의 내부적 특징을 고려한다는 것입니다. 고급 라이브러리들은 이런 복잡성을 숨기려는 경향이 있지만 효율적으로 쓰려면 이 모든 것을 무시할 수 없습니다. 한 달이라는 시간이 3분으로 바뀔 수도 있기 때문에 가치가 있습니다.

<br>

> <subtitle> The baseline </subtitle>

아타리의 Pong 게임을 대상으로 수렴 속도를 빠르게 만드는 방법에 대해 다뤄볼 예정입니다. 베이스라인은 8장에서 다룬 기존 DQN으로 합니다. 하이퍼패러미터도 동일합니다. 비교를 위해 두 가지 지표를 사용하겠습니다.

* 프레임 개수(FPS)
    - 학습을 위해 매 초마다 소비하는 프레임 개수
    - 학습할 때 얼마나 빨리 통신할 수 있는지 보여주는 지표
    - 강화학습 논문에서 흔하게 사용회는 지표로 보통 25M ~ 50M 개의 프레임이 나옴
    - 프레임 생략을 설정해둔 경우(거의 대부분 함)는 skip factor로 나눠야 함
        - skip factor 는 보통 4인데 그러면 환경의 FPS는 4배 더 크다고 봐야함.
* The wall clock time
    - 게임이 해결될 때까지 걸린 시간
    - 마지막 100개의 에피소드에 대한 보상 평균이 17 점에 도달하면 학습을 중단 (Pong의 최고 점수는 21점) 
    - 이 경계는 증가할 수 있지만 일반적으로 17은 에이전트가 게임을 거의 마스터했음을 나타내는 좋은 표시이며 정책을 완벽하게 다듬는 것은 학습 시간의 문제

이 챕터에서 비교하는 벤치마크들은 모두 동일한 기기에서 수행합니다.  
* i5-6600K(CPU)
* GTX 1080 Ti(GPU)
    - CUDA 10.0
    - Nvidia driver version : 410.79

베이스라인을 돌릴 수 있는 코드는 *Chapter09/01_baseline.py* 입니다. 다만 8장에서 다룬 내용과 완전히 동일하기 때문에 다루지는 않습니다.

<br>

학습하는 동안 TensorBoard 에는 다음과 같은 메트릭을 사용합니다.

* reward
    - discount를 하지 않은 보상값, x 축이 에피소드 번호
* avg_reward
    - reward에 alpha=0.98로 평균 낸 값
* steps
    - 에피소드 스텝 넘버
* loss
    - 100개 iteration마다 loss
* avg_loss
    - a smoothed version of the loss
* epsilon
    - epsilon 현재값
* avg_fps
    - 환경과 통신하는 에이전트의 속도. FPS 평균값

<br>

Figure 9.1, Figure 9.2 는 베이스라인을 여러 번 돌려서 평균 낸 보상, 에피소드 스텝, Loss, FPS의 시간에 따른 수치를 보여주고 있습니다. 2개의 x 축을 지니고 있는데 아래 것은 wall clock time, 위는 step number 입니다. 

<center><img src= "https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/fig9.1_2.png" width="70%"></center><br>

> <subtitle> The computation graph in PyTorch </subtitle>

먼저, 베이스라인의 속도를 높이는 것이 아니라 성능 저하가 발생할 수 있는 일반적인 상황을 살펴보겠습니다. PyTorch는 gradient 계산할 때, 텐서에 대해 수행하는 모든 연산의 그래프를 빌드하고 최종 손실의 backward() 를 호출하면 모델 파라미터의 모든 gradient는 자동으로 계산됩니다. 

이 과정은 off-policy RL의 경우 더 복잡합니다. 2개의 policy를 위해 각각의 네트워크를 운용해야 하기 때문입니다. DQN에서 뉴럴넷은 세 가지 다른 상황에서 사용됩니다.

1. 벨만 방정식에 의해 근사된 기준 Q-value 에 대한 loss 를 얻기 위해 네트워크에 의해 예측된 Q-value를 계산하고자 할 때
2. 타켓 네트워크를 적용하여 Bellman 근사치를 계산하기 위한 다음 상태의 Q 값을 얻고자 할 때
3. 작동하는 액션에 대한 결정을 에이전트가 할 때

첫 번째 상황에서만 그래디언트 계산을 하게 하는 것이 매우 중요합니다. target network에서는 그래디언트 계산을 막기 위해 *detach()* 를 사용합니다(두 번째 상황). 세 번째 상황에서는, 네트워크 결과를 NumPy array 로 변환함으로써 그래디언트 계산을 중지시킵니다. 

또한 그래디언트 계산을 하지 않더라도 파이토치는 computation graph 를 만드는데 이를 막기 위해 *torch.no_grad()* 를 사용하면 됩니다. 이는 쓸데없는 메모리 소비를 줄여주는 역할을 합니다. 

torch.no_grad() 의 효과를 알기 위해 no_grad()를 제외하고 모두 같은 *Chapter09/00_slow_grads.py* 와 비교해보았습니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/fig9.3.png" width="70%"></center><br>

figure 9.3에서 엄청나게 큰 차이는 없어보이지만 더 복잡한 구조의 대규모 네트워크에서는 차이가 더 클 것입니다. 

<br>

> <subtitle> Several environments </subtitle>

deep learning 학습 속도를 높이는 첫 번째 방법은 **batch size 를 크게** 하는 것입니다. 일반적인 지도학습의 경우 큰 배치 사이즈는 더 좋다는 규칙이 보통 맞습니다. 

RL의 경우는 조금 다릅니다. RL의 수렴은 보통 학습과 탐험 사이의 깨지기 쉬운 균형에 놓여 있는데 다른 조치 없이 batch size 만 키우면, 현재 데이터에 쉽게 과적합될 수 있기 때문입니다. 

*Chapter09/02_n_envs.py* 에서 에이전트는 학습 데이터를 모으기 위해 동일한 환경의 복제본을 사용합니다. 매 학습 iteration에서 모든 환경에서 replay buffer 를 샘플로 채운 다음, 기존 코드에서보다 큰 배치사이즈로 샘플링을 합니다. 이렇게 하면 inference time 이 약간 빨라지긴 합니다.

<br>

구현 측면에서 8장의 로직에서 몇 가지를 변경하였습니다.

* PTAN은 여러 환경을 지원하므로 N개의 Gym 환경을 ExperienceSource 인스턴스로 전달하기만 하면 된다.
* agent code(DQNAgent)는 배치에 맞게 이미 최적화되어 있음.

baseline과 차이나는 부분만 설명한 코드입니다.

```python
# 말그대로 배치를 생성하는 함수
def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int, steps: int):
    buffer.populate(initial)
    while True:
        buffer.populate(steps)
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        yield buffer.sample(batch_size)

(생략)

    # 여러 개의 환경을 생성해서 목록에 넣는다.
    envs = []
    for _ in range(args.envs):
        env = gym.make(params.env_name)
        env = ptan.common.wrappers.wrap_dqn(env)
        env.seed(common.SEED)
        envs.append(env)

(생략)

    # 단일 환경이 아니라 환경 목록을 입력으로 받음
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=params.gamma)
```

<br>

환경의 개수가 새로운 hyperparameter 가 됐기 때문에 환경의 개수 설정을 위한 실험을 하였습니다. 1개(베이스라인)와 2~6개의 환경으로 했을 때의 결과입니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/fig9.4.png" width="70%"></center><br>

<center><img src= "https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/fig9.5.png" width="70%"></center><br>

figure 9.4를 보면 환경이 1개일 때(베이스라인)보다 2, 3개로 늘어날수록 수렴 속도도 빨라지고 FPS도 증가했습니다. 
하지만 그 이상 환경을 늘리면 FPS는 증가하지만 수렴 속도가 느려지는 부정적인 효과가 있습니다. N=3 인 것이 최적값으로 보입니다.

<br>

> <subtitle> Play and train in separate processes </subtitle>

학습 과정은 다음과 같은 단계를 반복합니다.

1. 현재 네트워크에 선택할 행동들을 물어보고, 환경 목록들에서 행동들을 실행한다.
2. 관찰값들을 replay buffer에 넣는다
3. replay buffer로부터 학습 배치를 랜덤 샘플링한다.
4. 그 배치에서 학습

1,2 단계는 환경으로부터 얻은 샘플을 replay buffer에 축적하는 데 목표가 있습니다. 3,4 단계는 네트워크 학습을 위한 것이고요.

다음 figure 9.6 은 잠재적인 병렬화(potential parallelism) 적용되기 전 단계의 예입니다.  
* 왼쪽에는 학습 흐름이 표시
* 학습 단계에서는 환경, replay buffer, 학습 뉴럴넷을 사용
* 실선은 데이터와 코드 흐름을 나타냄
* 점선은 학습과 추론을 위한 뉴럴넷의 흐름을 나타냄

<center><img src= "https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/fig9.6.png" width="70%"></center><br>

figure 9.6 에서 위 2개의 단계는 replay buffer와 NN 을 통해서 아래 두 단계와 연결됩니다. 이는 이 과정을 별도의 과정으로 분리할 수 있다는 것을 뜻하며 그 과정이 아래 그림입니다.

<center><img src= "https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/fig9.7.png" width="70%"></center><br>

Pong 환경에서 이런 복잡한 내용을 불필요해보일 수 있지만, 이 분리는 다른 몇몇 경우에 굉장히 유용합니다. 엄청 느리고, 무거운 환경이 있다고 할 때, 모든 스텝이 연산을 위해 몇 초가 걸릴 것입니다. 이는 부자연스러운 예제가 아니라 현실에서 충분히 가능한 일입니다. 그래서 학습 과정으로부터 경험을 모으는 것을 분리하는 것이 필요합니다. 이렇게 하면 학습 프로세스에 경험을 제공하는 여러 가지 동시 환경을 가질 수 있습니다. 

코드를 병렬화한 것이 *Chapter09/03_parallel.py* 입니다. 주요 차이점만 보겠습니다.

```python
# python multiprocessing을 거의 코드변환 없이 대체한 torch 내의 multiprocessing module
# 프로세스 간 공유를 torch tensor로 함
# 이 공유 시스템은 단일 컴퓨터에서 통신이 이루어질 때 나타나는 병목현상을 제거한다.
import torch.multiprocessing as mp

BATCH_MUL = 4

EpisodeEnded = collections.namedtuple(
    'EpisodeEnded', field_names=('reward', 'steps', 'epsilon'))


def play_func(params, net, cuda, exp_queue):
    '''
    play process 구현한 함수
    train process에 의해 시작되는 별도의 child process에서 실행됨.

    * 환경으로부터 경험을 얻어 공유 대기열(shared queue)에 넣는다
    * 에피소드 종료에 대한 정보를 namedtuple로 싸서 동일한 대기열에 밀어넣어 에피소드 보상과 단계 수에 대해 train process에 계속 알려준다.
    '''
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)
    device = torch.device("cuda" if cuda else "cpu")

    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma)

    for frame_idx, exp in enumerate(exp_source):
        epsilon_tracker.frame(frame_idx/BATCH_MUL)
        exp_queue.put(exp)
        for reward, steps in exp_source.pop_rewards_steps():
            exp_queue.put(EpisodeEnded(reward, steps, selector.epsilon))
```

<br>

앞 세션에서 소개한 *batch_generator* function은 *BatchGenerator* class로 대체하였습니다. 

*BatchGenerator* class는 배치에 대해 iterator를 제공하고 추가적으로 pop_reward_steps() 메서드를 사용하여 ExperienceSource 인터페이스를 모방하였습니다. 

이 클래스의 오직은 간단합니다.
* queue(play process 에 의해 채워짐)를 소비하고, 에피소드 보상 및 단계에 대한 정보인 경우 저장한다. 
* 그렇지 않은 경우 object는 replay buffer에 추가해야 하는 경험의 일부로 간주
* 큐에서 현재 사용 가능한 모든 object를 소비한 다음 버퍼에서 학습 배치가 샘플링되어 산출됩니다.

```python
class BatchGenerator:
    def __init__(self, buffer: ptan.experience.ExperienceReplayBuffer,
                 exp_queue: mp.Queue,
                 fps_handler: ptan_ignite.EpisodeFPSHandler,
                 initial: int, batch_size: int):
        self.buffer = buffer
        self.exp_queue = exp_queue
        self.fps_handler = fps_handler
        self.initial = initial
        self.batch_size = batch_size
        self._rewards_steps = []
        self.epsilon = None

    def pop_rewards_steps(self) -> List[Tuple[float, int]]:
        res = list(self._rewards_steps)
        self._rewards_steps.clear()
        return res

    def __iter__(self):
        while True:
            while exp_queue.qsize() > 0:
                exp = exp_queue.get()
                if isinstance(exp, EpisodeEnded):
                    self._rewards_steps.append((exp.reward, exp.steps))
                    self.epsilon = exp.epsilon
                else:
                    self.buffer._add(exp)
                    self.fps_handler.step()
            if len(self.buffer) < self.initial:
                continue
            yield self.buffer.sample(self.batch_size * BATCH_MUL)
```

<br>

```python
if __name__ == "__main__":
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    # 새로운 프로세스를 시작하는 방법 지정
    # spawn로 하면, child process가 최소한의 자원만 승계받음 --> 가장 유연함
    mp.set_start_method('spawn')
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    buffer = ptan.experience.ExperienceReplayBuffer(
        experience_source=None, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    # start subprocess and experience queue
    # play process의 산출물을 담을 queue 생성
    exp_queue = mp.Queue(maxsize=BATCH_MUL*2)
    # play process를 위한 subprocess 생성
    play_proc = mp.Process(target=play_func, args=(params, net, args.cuda,
                                                   exp_queue))
    # subprocess 시작                                               
    play_proc.start()
    fps_handler = ptan_ignite.EpisodeFPSHandler()
    batch_generator = BatchGenerator(buffer, exp_queue, fps_handler, params.replay_initial, params.batch_size)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model,
                                      gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": batch_generator.epsilon,
        }

    engine = Engine(process_batch)
    ptan_ignite.EndOfEpisodeHandler(batch_generator, bound_avg_reward=17.0).attach(engine)
    fps_handler.attach(engine, manual_step=True)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        print("Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps, trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=trainer.state.metrics.get('time_passed', 0))))

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=trainer.state.metrics['time_passed']),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    logdir = f"runs/{datetime.now().isoformat(timespec='minutes')}-{params.run_name}-{NAME}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")

    episode_handler = tb_logger.OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
    tb.attach(engine, log_handler=episode_handler, event_name=ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    handler = tb_logger.OutputHandler(tag="train", metric_names=['avg_loss', 'avg_fps'],
                                      output_transform=lambda a: a)
    tb.attach(engine, log_handler=handler, event_name=ptan_ignite.PeriodEvents.ITERS_100_COMPLETED)

    engine.run(batch_generator)
    # 강제 종료
    play_proc.kill()
    # 큐의 모든 항목을 꺼내서 처리할 때까지 block
    play_proc.join()
```

<br>

다음 차트의 오른쪽은 병렬 버전의 FPS(402)이며, 베이스라인(159)과 비교하면 152% 이상 증가하였습니다.
왼쪽은 베이스라인과 병렬화 버전을 비교한 것으로 병렬화 버전의 수렴 속도가 빨라졌습니다. 하지만 앞서 N_envs=3 으로 한 것은 45분 걸렸는데 병렬화 버전은 1시간으로 더 느려졌습니다. 이 문제는 replay buffer에 더 많은 데이터를 공급하기 때문에 발생했으며 이로 인해 학습 시간이 길어졌습니다. (병렬 버전의 일부 하이퍼 파라미터 고치면 수렴 속도를 개선할 수 있을 것으로 보이긴 합니다.) 

<center><img src= "https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/fig9.8.png" width="70%"></center><br>


> <subtitle> Tweaking wrappers </subtitle>

이번에는 wrapper를 수정하여 환경에 적용시키는 방법입니다.

wrapper 를 수정하겠다는 생각은 보통 간과하기 쉽습니다. 왜냐하면 보통 wrapper는 딱 한 번 쓰여지거나 다른 코드에서 빌려오고 환경에 적용하면 바꾸지 않기 때문입니다. 

하지만 속도와 수렴의 관점에서 wrapper의 중요성은 꼭 알고 있어야 합니다. 예를 들어, 평범한 DeepMind 스타일의 wrapper stack을 아타리 게임에 적용하면 다음과 같습니다.

1. NoopResetEnv: NOOP(No Operation; 아무 일도 하지 않음)을 게임 리셋시 랜덤하게 적용시킴. 일부 아타리 게임에서 이는 이상한 초기값을 만들어서 지워야 한다.
2. MaxAndSkipEnv: N개의 관찰값(default: 4)을 모아서 최대값을 step의 관찰값으로 반환한다. 이러면 짝수 프레임과 홀수 프레임에 다른 부분을 그릴 경우 발생하는 "깜빡임" 문제를 해결할 수 있다.
3. EpisodicLifeEnv: 게임에서 죽은 것을 탐지해내서 에피소드르 종료시킨다. 에피소드가 더 짧아지기 때문에 수렴도가 상당히 올라간다. 이는 아타리 일부 게임에만 해당한다.
4. FireResetEnv: 게임 리셋할 때 FIRE action을 실행한다. 
5. WarpFrame(ProcessFrame84) : image 를 그레이스케일로 바꾸고 84*84 사이즈로 변환해준다.
6. ClipRewardEnv: 보상을 -1~1 사이로 자른다. 이 방식이 최고의 방법은 아니지만 여러 아타리게임에서 다양한 점수를 얻을 수 있는 효과적인 솔루션이다.
7. FrameStack: N개의 연속적인 관찰값을 쌓는다(default: 4). 

<br>

이 wrapper 들의 코드는 여러 사람들에 의해 만들어지고 최적화되어 버전도 여럿 있습니다. 그 중 OpenAI 의 것은 좋은 옵션이 될 수 있습니다. ([https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py){:target="_blank"}) 물론 구체적으로 바꿀 요소가 있다면 이를 수정해서 사용해도 됩니다. 

이 세션에서는 다음과 같은 환경 wrapper를 사용하여 성능을 올렸습니다.

* *cv2* 라이브러리를 *pillow-simd* 로 대체
    - pillow-simd 설치 방법
        - pip install pillow-simd
* NoopResetEnv 비활성화
* MaxAndSkipEnv 를 max pooling 없이 4개 프레임을 생략하는 환경으로 대체
* FrameStack 은 2개의 프레임 사용

파일은 3개가 있습니다.  
* *Chapter09/04_new_wrappers_n_env.py*
* *Chapter09/04_new_wrappers_parallel.py*
* *Chapter09/lib/atari_wrappers.py* 

<br>

OpenAI Baselines repository에서 가져온 atari_wrappers.py는 텐서플로우에 맞는 tensor shape 으로 구현되어 있어서 pytorch에 맞는 tensor shape으로 변환한 것이 *Chapter09/lib/atari_wrappers.py* 입니다.  
(width, height, channel) --> (channel, width, height)

이 형식 변경을 반영하기 위해 FrameStack과 LazyFrames class에서 일부 변경이 있습니다.

```python
class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        # for tensorflow
        old_shape = self.observation_space.shape
        # for pytorch
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)
```

<br>

WarpFrame 에서는 cv2 대신에 pillow-simd 를 사용하는 부분을 수정했습니다.

```python
USE_PIL = True
if USE_PIL:
    # you should use pillow-simd, as it is faster than stardand Pillow
    from PIL import Image
else:
    import cv2
    cv2.ocl.setUseOpenCL(False)


class WarpFrame(gym.ObservationWrapper):
    ...(생략)...

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]
        if USE_PIL:
            frame = Image.fromarray(frame)
            if self._grayscale:
                frame = frame.convert("L")
            frame = frame.resize((self._width, self._height))
            frame = np.array(frame)
        else:
            if self._grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(
                frame, (self._width, self._height),
                interpolation=cv2.INTER_AREA
            )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs
```

<br>

앞서 말한대로 MaxAndSkipEnv 가 아니라 SkipEnv 로 사용합니다. maxpool이 없습니다.

```python
class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

```

<br>

*make_atari* 함수와  *wrap_deepmind* 함수는 둘 다 환경을 커스텀하는 역할을 합니다. (한 번에 구성할 수 있을 것 같은데 분리를 했네요)

```python
def make_atari(env_id, max_episode_steps=None,
               skip_noop=False, skip_maxskip=False):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    if not skip_noop:
        env = NoopResetEnv(env, noop_max=30)
    if not skip_maxskip:
        env = MaxAndSkipEnv(env, skip=4)
    else:
        env = SkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True,
                  frame_stack=False, scale=False, pytorch_img=False,
                  frame_stack_count=4, skip_firereset=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        if not skip_firereset:
            env = FireResetEnv(env)
    env = WarpFrame(env)
    if pytorch_img:
        env = ImageToPyTorch(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, frame_stack_count)
    return env
```

환경 구성시에 위 함수 호출은 다음과 같이 사용합니다.

```python
env = atari_wrappers.make_atari(params.env_name,
                                skip_noop=True,
                                skip_maxskip=True)
env = atari_wrappers.wrap_deepmind(env, pytorch_img=True,
                                   frame_stack=True,
                                   frame_stack_count=2)
```

이 내용을 n_env(N개의 환경 사용), parallel 버전에 적용하여 결과를 본 것이 아래 그림입니다. (코드상 차이는 환경 호출 부분이라 생략)

<center><img src="https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/fig9.9.png" width="70%"></center><br>

wrapper를 수정하고 3개 환경을 쓴 경우가 baseline보다 성능이 좋았고, wrapper를 사용하지 않은 경우와 비교하면 수렴속도는 거의 비슷했습니다. FPS는 wrapper 수정 없을 때가 더 작았습니다. 

<center><img src="https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/fig9.10.png" width="70%"></center><br>

parallel 버전은 wrapper 수정했을 때 수렴 시간을 30분이나 단축할 수 있었습니다.(1시간 -> 30분) 

<br>

> <subtitle> Benchmark summary </subtitle>

지금까지 한 실험들의 결과를 다음 테이블로 정리했습니다.

<center><img src="https://liger82.github.io/assets/img/post/20210919-DeepRLHandsOn-ch09-Ways_to_Speed_up_RL/comparison_table.png" width="80%"></center><br>

괄호 안의 퍼센티지는 베이스라인과 비교하여 변화한 정도를 뜻합니다. wrapper 를 수정하고 병렬화한 버전이 가장 많은 성능 개선을 보였습니다.

<br>

> <subtitle> Going hardcore: CuLE </subtitle>

NVIDIA 연구자들은 GPU에서 Atari emulator를 돌리는 내용의 논문과 코드를 공개했습니다.(Steven Dalton, Iuri Frosio, GPU-Accelerated Atari Emulation for Reinforcement Learning, 2019) 이를 CuLE(CUDA Learning Environment)라고 부르고 코드는 다음 깃헙에 공개했습니다.  
* [https://github.com/NVlabs/cule](https://github.com/NVlabs/cule){:target="_blank"}

이 논문에 의하면 A2C로 Pong 게임을 2분 만에 풀고 FPS는 50k까지 도달했습니다. 성능 개선의 핵심 포인트는 CPU와 GPU 간 상호작용을 없앰으로써 속도를 올린 것입니다.

<br>

또 다른 하드코어한 방법은 환경 구현을 위해 field-programmable gate array(FPGA) 를 사용한 것입니다. 이런 방식의 프로젝트 중 하나는 Verilog로 Game Boy를 구현한 것입니다.([https://github.com/krocki/gb](https://github.com/krocki/gb){:target="_blank"})

<br>

> <subtitle> Summary </subtitle>

8장이 DQN을 알고리즘적인 측면에서 개선한 것이라면 9장은 엔지니어링적인 측면에서 개선한 내용을 소개하였습니다.

주요 방법은 다음과 같았습니다.

1. N 개의 환경
2. 병렬화 버전
3. wrapper 수정

다음 장에서는 주식 시장에 DQN을 적용해보는 시간을 갖겠습니다.

<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 09 : Ways to Speed up RL

<br>
