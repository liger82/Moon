---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 10 : Stocks Trading Using RL"
date: 2021-10-11
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, chapter 10, Stocks Trading Using RL, Stocks]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

지금까지는 가벼운 예제를 대상으로 강화학습 방법을 적용해보았다면 이번 챕터에서는 "금융 거래"와 같이 더 현실적인 문제를 대상으로 DQN 을 적용해보는 시간을 가지도록 하겠습니다. 

챕터 목표   
* 커스텀 OpenAI Gym 환경을 구현하여 주식 시장을 시뮬레이션해보기
* 이윤 최대화를 위해 주식 거래를 위한 에이전트 학습
    - 기본 DQN
    - DQN Extensions(8장)

<br>

> <subtitle> Trading </subtitle>

매일 시장에서 거래되는 금융 수단은 여럿이 있습니다. 심지어 기상 예보도 기상 파생상품을 통해 사고 팔 수 있습니다. 작물 재배와 같이 수입이 미래의 기상 조건에 의존하는 사업을 하고 있으면,  기상 파생상품을 구매하여 리스크를 회피하려고 할 수 있습니다. 이런 다른 모든 종류의 아이템들은 시간에 따라 변화하는 가격을 가지고 있습니다. 거래는 이윤을 만들어내거나(투자) 미래의 가격 변동으로부터 보호하거나(hedging) 아니면 단순히 원하는 것을 얻으려는 등의 다양한 목표를 가지고 금융 수단을 사고 파는 활동을 뜻합니다. 

금융 시장이 만들어진 이래로, 사람들은 미래 가격 변동을 예측하기 위해 노력해왔습니다. 왜냐하면 미래 가격 변동을 잘 예측하는 것이 큰 이익을 보장하기 때문입니다.  

이 문제는 시장을 예측하고 최고의 순간을 찾아 이윤을 극대화하려는 재무 컨설턴트, 투자 펀드, 은행, 개인 트레이터가 얽혀 있어 복잡합니다.  

이 챕터를 관통하는 질문은 이것 입니다. "강화학습의 관점에서 이 문제를 풀어낼 수 있는가" 시장으로부터 관찰값이 있고, 매수, 매도, 대기 중 하나를 선택하고자 할 때, 가능한 한 큰 이윤을 내고자 합니다. 

<br>

> <subtitle> Data </subtitle>

이번 장에서 사용할 데이터 예제는 15~16년 러시아의 Yandex 주식 가격입니다. 파일은 *Chapter10/data/ch08-small-quotes.tgz* 입니다. (파일명의 챕터가 업데이트가 안된 듯..) data 디렉토리 안에 unpack 하는 실행파일도 있으니 학습 전에 실행해서 압축해주시기 바랍니다.

파일 내부는 다음과 같습니다.

```
<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
20160104,100100,1148.9000000,1148.9000000,1148.9000000,1148.9000000,0
20160104,100200,1148.9000000,1148.9000000,1148.9000000,1148.9000000,50
20160104,100300,1149.0000000,1149.0000000,1149.0000000,1149.0000000,33
20160104,100400,1149.0000000,1149.0000000,1149.0000000,1149.0000000,4
20160104,100500,1153.0000000,1153.0000000,1153.0000000,1153.0000000,0
20160104,100600,1156.9000000,1157.9000000,1153.0000000,1153.0000000,43
...
```

<br>

캔들스틱 차트(Candlestick chart)는 주식을 비롯해서 유가증권, 파생상품, 환율의 가격 움직임을 보여주는 금융 차트로 가격 변동을 보여주는 가장 전형적인 차트입니다. 

이 데이터는 캔들스틱 차트를 숫자를 표현했기 때문에 하루를 일정 간격으로 잘라서 하나의 행으로 표현하고 있습니다. 하나의 행 데이터는 봉(bar)을 뜻합니다.

DATE는 날짜, TIME 은 시간(분)을 의미하고, OPEN은 그 간격의 시작 가격, HIGH는 그 간격 기간 내 최고 가격, LOW는 그 간격 기간 내 최저 가격, CLOSE는 간격의 마지막 가격, VOL은 거래량을 의미합니다. 

다음은 Yandex의 2016년 2월 하루를 표현한 캔들스틱 차트입니다.

<center><img src= "https://liger82.github.io/assets/img/post/20211011-DeepRLHandsOn-ch10-Stocks_Trading_Using_RL/fig10.1.png" width="90%"></center><br>

16년 데이터는 학습에 15년은 검증에 사용할 예정입니다.

<br>

> <subtitle> Problem statements and key decisions </subtitle>

금융 도메인은 크고 복잡해서 매일 새로운 것을 배워도 몇 년은 걸릴 것입니다. 그래서 이 주식 거래 예제에서 관찰값을 가격으로만 한정하여 문제를 단순화하였습니다. (더 복잡한 현실을 반영하려고 주식 거래를 예제로 잡았는데 너무 단순화한게 아닌가 싶네요.)

이 예제의 목적은 RL 모델이 어떻게 유연해질 수 있는지와 보통 RL을 실제 use case에 적용하기 위해 해야하는 첫 번째 단계가 무엇인지를 보여주는 것입니다. 

강화학습 문제를 정의하기 위한 3가지  
* 환경의 관찰값
* 가능한 행동
* 보상 체계

이전 챕터에서 이 세 가지는 항상 주어졌습니다. 이제는 에이전트가 볼 수 있는 것, 할 수 있는 행동을 정해줘야 합니다. 보상 체계도 엄격하게 정해져 있지 않고 도메인에 대한 우리의 감정, 지식에 기반하여 가이드할 수 있습니다. 즉, 우리는 유연성을 가지고 있습니다.

유연성은 장단점을 가집니다.
* 장점 : 본인이 중요하다고 생각하는 정보를 에이전트에게 전달할 자유가 있음
* 단점 : 훌륭한 에이전트를 만들기 위해서 다양한 데이터를 고려할 필요가 있음

기본적인 거래 에이전트를 가장 단순한 형태로 구현하려 했고 관찰값은 다음과 같은 정보를 포함하고 있습니다.  
* N개의 봉(bar)
    - 각각의 봉은 시작, 고점, 저점, 끝 가격을 가짐
* 주식이 몇 타임 전에 팔렸다는 정보
* 보유 중인 주식의 현재 위치에서의 손익

모든 스텝, 매 시간마다 에이전트는 다음과 같은 행동을 취할 수 있습니다.
* 대기
* 매수 : 해당 주식을 이미 보유하고 있으면 매수하지 않음. 그렇지 않을 경우 소정의 수수료를 지불하고 매수
* 매도 : 이전에 매수하지 않았을 경우 아무 행위 없음. 그렇지 않을 경우 거래에 수수료 있음.

<br>

> <subtitle> The trading environment </subtitle>

환경에 대해 유연성을 지닌다고 했지만 기본적으로는 익숙한 OpenAI Gym을 기반으로 사용할 것입니다. 환경은 *Chapter10/lib/environ.py*에 *StocksEnv* class 입니다. 

```python
import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
# 거래 수수료
DEFAULT_COMMISSION_PERC = 0.1

# 행동들을 enumerator의 필드로 인코딩해놓음
class Actions(enum.Enum):
    Skip = 0  # 대기 0 
    Buy = 1   # 매수 1
    Close = 2 # 매도 2
```

<br>

*StocksEnv* class 에서 metadata와 spec 변수는 gym.Env 를 상속하여 호환성을 충족시키기 위해 필요한 것입니다.

```python
class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("StocksEnv-v0")
    ...
```

<br>

*StocksEnv* class 의 instance 를 생성하는 방법은 두 가지입니다. 첫 번째 방법은 데이터 디렉토리를 argument로 주어, class method *from_dir* 을 호출하는 것입니다. 그러면 디렉토리의 csv file을 로드하여 환경을 구성할 것입니다. 

```python
@classmethod
def from_dir(cls, data_dir, **kwargs):
    prices = {
        file: data.load_relative(file)
        for file in data.price_files(data_dir)
    }
    return StocksEnv(prices, **kwargs)
```

데이터를 읽고 특정 형태로 구성하는 코드는 *Chapter10/lib/data.py* 에 있습니다. 

두 번째 방법은 class instance 를 직접적으로 구성하는 것입니다. 
이 방법은 from_dir에서 나온 prices dictionary를 직접 전달해야 합니다. prices 는 data.py에 **Prices** namedtuple 로 만들어져야 합니다. (**Prices** 는 open, high, low, close, volume 총 5가지의 필드를 가지고 각 필드는 NumPy array 이다.)

<br>

```python
class StocksEnv(gym.Env):

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC,
                 reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False,
                 volumes=False):
        '''
        StockEnv class의 기능 대부분은 State와 State1D 두개의 클래스로 구현되어 있음.
        Gym에서 요구하는 state, action, observation을 생성
        '''
        assert isinstance(prices, dict)
        self._prices = prices
        # state
        if state_1d:
            self._state = State1D(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes)
        else:
            self._state = State(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes)
        # action space
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        # observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()
```

StockEnv의 constructor 를 보면, 환경이나 관찰값 표현을 바꾸는 용도로 여러 arguments를 사용하고 있습니다.

* prices : 1개 이상의 주식 가격을 지님. data.Prices 포맷
* bars_count : 관찰값에서 나타나는 봉의 개수(default=10)
* commission : 주식 거래시 발생하는 수수료로 주식 가격의 일부를 뜻하며 defalut는 0.1%
* reset_on_close : True(defalut)면, 에이전트가 주식을 팔 것인지 물어볼 때마다 에피소드를 중지한다. 그렇지 않으면, 그 해의 데이턱 끝날 때까지 에피소드는 계속된다.
* state_1d : True 면 1D convolution 을 위한 *State1D* class를 사용한다.(데이터 인코딩 matrix 형태) False(defalut)면 *State* class를 사용한다. (데이터 인코딩 vector 형태)

<center><img src= "https://liger82.github.io/assets/img/post/20211011-DeepRLHandsOn-ch10-Stocks_Trading_Using_RL/fig10.2.png" width="90%"></center><br>

* random_ofs_on_reset : True(default)면 환경 리셋할 때, offset을 랜덤하게 선정. 아니면 데이터의 앞부분으로 함
* reward_on_close : True면 에이전트는 매도 행위에 대해서만 보상을 받지만 False면 모든 바에 대해 작은 보상도 부여한다.
* volumes

<br>

```python
    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        # 랜덤하게 가격을 뽑는다.
        self._instrument = self.np_random.choice(
            list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(
                prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "instrument": self._instrument,
            "offset": self._state._offset
        }
        return obs, reward, done, info
```

<br>

environ.State class는 환경의 대부분 기능을 구현하고 있습니다.

```python
class State:
    # 환경의 대부분 기능을 구현
    def __init__(self, bars_count, commission_perc,
                 reset_on_close, reward_on_close=True,
                 volumes=True):
        # 검사와 저장 역할이 주
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    # numpy array의 state representation의 shape을 반환
    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit
        if self.volumes:
            return 4 * self.bars_count + 1 + 1,
        else:
            return 3*self.bars_count + 1 + 1,

    def encode(self):
        """
        Convert current state into numpy array.
        State class를 단일 벡터로 인코딩
        
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            # 고점, 저점, 끝지점 + 볼륨(옵션) 순
            ofs = self._offset + bar_idx
            res[shift] = self._prices.high[ofs]
            shift += 1
            res[shift] = self._prices.low[ofs]
            shift += 1
            res[shift] = self._prices.close[ofs]
            shift += 1
            # 선택사항임
            if self.volumes:
                res[shift] = self._prices.volume[ofs]
                shift += 1
        # 주식 보유여부
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = self._cur_close() / self.open_price - 1.0
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        현재 봉의 마지막 가격을 계산해줌

        State class에 전달된 가격은 오픈 가격과 관련하여 상대적인 형태를 가짐
        고점, 저점, 종가는 오픈 가격에 대한 상대적인 비율. 
        
        이 방식은 실제 가격 값과 무관한 가격 패턴을 학습하는 데 도움이 될 것이다
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        '''
        현재 봉의 종가로 즉시 주문 실행을 가정하고 있음.(단순화한 결과) 
        일반적으로 주문이 다른 가격으로 실행될 수 있으며, 이를 가격 미끄러짐(price slippage)이라고 한다.
        '''
        # 매수 행동이고, 주식을 가지고 있지 않을 때
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            # 커미션 떼기
            reward -= self.commission_perc
        # 매도 행동이고, 주식을 가지고 있을 때
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            # reward_on_close True면 매도 시에만 보상을 받음
            if self.reward_on_close:
                reward += 100.0 * (close / self.open_price - 1.0)
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1
        
        # 주식 보유하고 reward_on_close가 False 일 때, 지난 봉 움직임에 보상을 부여
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close / prev_close - 1.0)

        return reward, done
```

<br>

*State1D* class는 State class 상속 받아서 일부 메서드만 override 한 클래스입니다. 

```python
class State1D(State):
    """
    State with shape suitable for 1D convolution
    """

    # 1D convolution operator에 적합한 2D matrix
    # return shape
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        start = self._offset-(self.bars_count-1)
        stop = self._offset+1
        res[0] = self._prices.high[start:stop]
        res[1] = self._prices.low[start:stop]
        res[2] = self._prices.close[start:stop]
        if self.volumes:
            res[3] = self._prices.volume[start:stop]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = self._cur_close() / self.open_price - 1.0
        return res
```

<br>

여기까지가 거래 환경 관련 코드입니다. 

<br>

> <subtitle> Models </subtitle>

2개의 DQN 아키텍쳐를 사용할 예정입니다.  

1. Simple feed-forward network with three layers
2. 1D convolution as a feature extractor + two fully connected layers to output Q-values

둘 다 8장(DQN Extensions)에 나온 Dueling, Double DQN, 2-step bellman unrolling 을 적용했습니다. 나머지는 기본 DQN과 동일합니다.

두 모델 모두 *Chapter10/lib/models.py* 에서 확인할 수 있습니다.

```python
class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        # Dueling DQN 방식 - average 
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class DQNConv1D(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1D, self).__init__()

        # feature extraction layer with the 1D convolution
        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )
        
    # 중간 output 알아내기 위한 메서드
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        # Dueling DQN 방식 - average
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))
```

<br>

> <subtitle> Training code </subtitle>

모델이 2개인 만큼 학습 코드도 2개입니다. feed-forward model은 *Chapter10/train_model.py* 이고, 1D conv model 은 *Chapter10/train_model_conv.py* 입니다.

8장의 DQN Extensions 코드에서 새로운 것은 없습니다.

* 탐험을 하기 위해 epsilon-greedy action selection 사용. epsilon 은 1.0으로 시작하여 백만 스텝까지 0.1까지 선형적으로 줄임
* 100k 사이즈의 간단한 experience replay buffer
* 학습하는 동안, 매 1000 스텝마다 Q-value의 변화하는 모습을 확인하기 위해 일정 개수 state의 평균값을 계산
* 매 100k 스텝마다 검증 수행.

<br>

두 파일을 돌릴 때 학습과 검증으로 사용할 데이터를 지정하지 않으면 알아서 지정이 되어있으므로 명령어는 거의 유사합니다.

```python
# -r 은 Tensorboard run name
$ python train_model.py --cuda -r ff-211011
$ python train_model_conv.py --cuda -r 1dconv-211011
```

<br>

> <subtitle> Results </subtitle>

## The feed-forward model

1년짜리 1개 기업(Yandex) 데이터의 수렴은 10M 학습스텝이 필요했고(GTX 1080 ti 기준) 학습하는 동안, 테스트 동안의 시간에 따른 평균 보상값의 변화는 다음과 같습니다.



<br>

## The convolution model

<br>

> <subtitle> Things to try </subtitle>


<br>

> <subtitle> Summary </subtitle>


<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 10 : Stocks Trading Using RL
* [](){:target="_blank"}


<br>
