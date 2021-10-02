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



<br>

> <subtitle> Tweaking wrappers </subtitle>

<br>

> <subtitle> Benchmark summary </subtitle>

<br>

> <subtitle> Going hardcore: CuLE </subtitle>

<br>


<br>


<br>

> <subtitle> Summary </subtitle>


<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 09 : Ways to Speed up RL
* [](){:target="_blank"}

<br>
