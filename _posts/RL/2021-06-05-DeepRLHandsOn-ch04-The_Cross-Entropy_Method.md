---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 04 : The Cross-Entropy Method"
date: 2021-06-05
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 4, The cross-entropy method]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

이번 챕터에서는 강화학습 방법 중 하나인 cross-entropy 에 대해 알아보겠습니다. 다른 방법들에 비해 덜 유명하지만, cross-entropy도 그만의 장점을 지닙니다. 
1. 매우 간단하다 --> 파이토치 코드가 100줄이 안 된다.
2. 훌륭한 수렴성 --> 간단한 환경에서는 cross-entropy도 매우 잘 작동한다.

이번 챕터에서는 다음과 같은 내용을 다룰 예정입니다.
* cross-entropy의 실용적인 면을 다룬다.
* CartPole과 FrozenLake, 두 개의 Gym 환경에서 cross-entropy가 어떻게 작동하는지 알아본다.
* cross-entropy의 이론적 배경에 대해 다룬다.(Optional)

<br>

> <subtitle> The taxonomy of RL methods  </subtitle>

<br><center><img src= "https://dreamgonfly.github.io/images/rl-taxonomy/rl_taxonomy.png" width="60%"></center><br>

cross-entropy는 model-free, policy-based 방법입니다. 이 의미에 대해 알아보기 위해 강화학습을 나누는 여러 측면에 대해 먼저 얘기해보겠습니다. 
* Model-free vs Model-based
* Value-based vs Policy-based
* On-policy vs Off-policy

<br>

## Model-free vs Model-based

<br><center><img src= "https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg" width="80%"></center><br>

강화학습 알고리즘을 구분하는 주요 지점 중 하나는 에이전트가 **환경의 모델에 접근(혹은 학습)**을 할 수 있느냐입니다. 환경의 모델은 상태 전이와 보상을 예측하는 함수를 의미합니다. 

Model-based 알고리즘은 환경의 모델을 지닌 알고리즘을 뜻합니다.  Model-based 알고리즘은 **계획(planning)**이 가능하다는 장점을 지닙니다. 자신의 행동에 따라 환경이 어떻게 변할지 예측해서 최적의 행동을 계획하여 실행할 수 있다는 것입니다. 모델을 지닌다면 에이전트는 효율적으로 행동할 수 있습니다.

문제는 이 환경이 복잡할 경우에 환경의 모델을 알아내기 어렵거나 불가능하다는 점입니다. 모델이 환경을 제대로 반영하지 않는다면 이는 바로 에이전트의 오류로 이어지게 될 것입니다. 

Model-based 알고리즘은 모델이 주어져 있는지 학습해야 하는지에 따라 또 구분이 가능합니다.

Model-free 알고리즘은 환경의 모델을 사용하지 않는 경우를 말하며, 관찰값과 행동을 바로 연결시킵니다. 현재 관찰값을 받아서 이를 기반으로 계산하고 그 결과로 행동을 하는 것입니다.

강화학습에서는 주로 model-free 알고리즘이 각광받고 있으며 우리가 아는 대부분의 유명한 알고리즘들(DQN, policy gradient, A2C, A3C 등)은 모두 model-free 입니다.

최근에는 두 알고리즘을 융합하는 시도도 하고 있습니다. 예를 들어, 딥마인드에서 내놓은 *[Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/abs/1707.06203)* 도 이러한 부류입니다. (22장에서 다룰 예정)

<br>

## Policy-based vs Value-based

Policy-based 방법은 에이전트의 정책, 즉 에이전트가 모든 단계에서 수행해야 하는 작업을 직접 근사화합니다. 여기서 정책은 일반적으로 사용 가능한 작업에 대한 확률 분포로 표시됩니다.

반면, Value-based 방법의 에이전트는 액션의 확률 대신 가능한 모든 액션의 값을 계산하고 최상의 값을 가진 액션을 선택합니다. 그렇기 때문에 policy-based 방법과 달리 value-based 방법은 value function을 먼저 구하고 정책을 추정합니다. 

<br>

## On-policy vs Off-policy

off-policy는 action을 취하는 policy(behavior policy)와 improve하는 policy(target policy)를 다르게 취하는 것이고, on-policy는 두 policy가 동일한 경우를 뜻합니다.

cross-entropy는 on-policy에 해당합니다.

<br>

> <subtitle> The cross-entropy method in practice  </subtitle>

cross-entropy method의 핵심은 안좋은 에피소드는 버리고 좋은 에피소드에서 학습하는 것입니다. 학습 단계는 다음과 같습니다.

1. 현재 모델과 환경으로 N번의 에피소드를 돌린다.
2. 매 에피소드마다 총 보상을 계산하고 보상의 경계를 정한다. 보통 모든 보상의 백분위수를 사용한다.
3. 경계 밖의 에피소드는 모두 버린다.
4. 남은 에피소드를 기반으로 학습하고 결과를 비교한다.
5. 만족스러운 결과가 나올 때까지 1~4의 스텝을 반복한다.

<br>

> <subtitle> The cross-entropy method on CartPole  </subtitle>

이 세션에서는 앞서 다룬 학습 단계를 기반으로 CartPole 환경에서 코드를 돌려보도록 하겠습니다. 전체 코드는 **Chapter04/01_cartpole.py** 에서 확인할 수 있습니다.

뉴럴넷은 간단한 구조를 사용하며 다음과 같은 하이퍼패러미터를 쓰고 있습니다.

* HIDDEN_SIZE = 128
* BATCH_SIZE = 16
* PERCENTILE = 70

PERCENTILE은 에피소드 필터링을 위한 백분위수입니다. 

```python
HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
```

구조는 정말 간단합니다. 다만 마지막에 행동들에 대한 softmax를 계산하는게 보통인데 여기선 없습니다. 대신에, softmax와 cross-entropy를 하나의 수치적으로 더 안정적인 표현으로 결합하는 **nn.CrossEntropyLoss** 를 사용합니다. **nn.CrossEntropyLoss** 는 뉴럴넷에서 나온 정규화되지 않은 원시 값(logit)이 필요합니다. 단점은 뉴럴넷의 출력으로부터 확률을 얻기 위해서는 필요할 때마다 softmax를 적용해야 한다는 점입니다. 

<br>

```python
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
```

collections.namedtuple로 2개의 helper class를 구성하였습니다.

* EpisodeStep : 에이전트가 에피소드에서 수행한 하나의 단계를 나타내며, 해당 환경에서 관찰한 내용과 에이전트가 완료한 액션을 저장합니다. 엘리트 에피소드의 episodestep를 학습데이터로 사용합니다.
* Episode : 할인되지 않은 총 보상과 EpisodeStep의 모음으로 구성된 단일 에피소드입니다.

<br>

우선 iterate_batches()를 살펴보면,

```python
def iterate_batches(env, net, batch_size):
    batch = [] # 배치사이즈만큼 Episode instance를 담을 리스트
    episode_reward = 0.0 # 현재 에피소드의 reward counter
    episode_steps = [] # episode_step을 담을 리스트
    obs = env.reset() # 관찰값 초기화
    sm = nn.Softmax(dim=1) # 뉴럴넷의 출력값을 action의 확률분포로 변환
```

이 함수는 gym의 환경, 뉴럴넷, 배치사이즈를 입력으로 받고 초기세팅을 합니다. 

```python
    while True:
        # PyTorch tensor로 변환
        obs_v = torch.FloatTensor([obs])
        # 뉴럴넷 -> softmax -> action 확률값
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        # 랜덤하게 action 선택
        action = np.random.choice(len(act_probs), p=act_probs)
        # action에 대한 환경의 response 받음.
        next_obs, reward, is_done, _ = env.step(action)
        
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        # 하나의 에피소드가 끝나면
        if is_done:
            # EpisodeSteps 모은 것을 하나의 에피소드 인스턴스로 저장한다.
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            # 배치사이즈만큼 모이면 반환하고(generator) 초기화
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs
```

<br>




<br>

> <subtitle> The cross-entropy method on FrozenLake </subtitle>

<br>

> <subtitle> The theoretical background of the cross-entropy method </subtitle>

cross-entropy method는 importance sampling theorem에 기초를 두고 있습니다. (다음은 importance sampling 수식입니다.)

<center>$$ \mathbb{E}_{x \sim p(x)}[H(x)]=\int_{x}p(x)H(x)dx = \int_{x}q(x)\frac{p(x)}{q(x)}H(x)dx = \mathbb{E}_{x \sim q(x)}[\frac{p(x)}{q(x)}H(x)] $$ </center><br>

RL에서 $$H(X)$$ 는 policy $$x$$에 의해 얻어진 보상값이고, $$p(x)$$ 는 모든 가능한 정책의 분포입니다. 모든 정책들을 뒤져가며 보상을 최대화하는 것이 아니라 두 확률 분포의 거리를 줄이기를 반복하면서 $$ \frac{p(x) H(x)}{q(x)} $$ 을 근사하는 방법을 찾고자 합니다. 두 확률 분포 간 거리는 Kullback-Leibler(KL) divergence로 계산합니다.

<center> $$ KL(p_1(x) \parallel p_2(x)) = \mathbb{E}_{x \sim p_1(x)}log\frac{p_1(x)}{p_2(x)} = {\color{Blue} \mathbb{E}_{x \sim p_1(x)}[log p_1(x)]} - {\color{Red}\mathbb{E}_{x \sim p_1(x)}[log p_2(x)]} $$ </center><br>

<span style="color:blue">첫번째 텀이 엔트로피</span>이고 $$p_2(x)$$ 에 의존하지 않기 때문에 최소화할 때 생략가능합니다. <span style="color:red">두번째 텀이 cross-entropy</span>이고, 이는 딥러닝에서 가장 일반적인 최적화 목적함수입니다.

두 수식을 결합하여, $$q_0(x)=p(x)$$ 에서 시작하여 매 단계마다 개선되는 반복 알고리즘을 얻을 수 있습니다. 다음과 같은 업데이트식을 통해 $$p(x)H(x)$$ 를 근사할 수 있습니다.

<center> $$ q_{i+1}(x) = \underset{q_{i+1}(x)}{argmin} - \mathbb{E}_{x \sim q_i(x)}log\frac{p(x)}{q_i(x)}H(x)log q_{i+1}(x) $$ </center><br>

아래 식은 이 장에서 다룬 RL 예제에서 크게 단순화할 수 있는 일반적인 교차 엔트로피 방법입니다. 먼저, $$H(x)$$ 를 지시 함수로 대체하는데, 이는 에피소드에 대한 보상이 역치를 초과할 때 1이고 보상이 역치보다 작으면 0 이 됩니다. 정책 업데이트는 다음과 같습니다.

<center> $$ \pi_{i+1}(a|s) = \underset{\pi_{i+1}}{argmin} - \mathbb{E}_{z \sim \pi_i(a|s)}[R(z) \ge \Psi_i] log \pi_{i+1}(a|s) $$ </center><br>

엄격하게 봤을 때 위 식은 정규화 조건을 빠뜨렸으나 우리 예제에서는 잘 작동합니다. 


<br>

> <subtitle> Summary </subtitle>



<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 04 : The Cross-Entropy Method
* [https://dreamgonfly.github.io/blog/rl-taxonomy/](https://dreamgonfly.github.io/blog/rl-taxonomy/){:target="_blank"}
* [https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html){:target="_blank"}

<br>
