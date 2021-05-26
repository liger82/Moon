---
layout: post
title: "[Deep Reinforcement Learning Hands On 2/E] Chapter 03 : Deep Learning with PyTorch"
date: 2021-05-21
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Deep Reinforcement Learning Hands On, OpenAI Gym, PyTorch, torch]
comments: true
---

* 교재 : [Deep Reinforcement Learning Hands-On 2/E](http://www.kyobobook.co.kr/product/detailViewEng.laf?mallGb=ENG&ejkGb=ENG&barcode=9781838826994){:target="_blank"}
* github : [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition){:target="_blank"}

<br>

> <subtitle> Intro </subtitle>

이번 챕터에서는 다음과 같은 내용을 다룰 예정입니다.
* PyTorch 라이브러리 세부사항 및 구현 세부 정보
* DL 문제를 단순화하기 위한 목적으로 PyTorch 상의 고차원 라이브러리들
* PyTorch ignite

<br>

> <subtitle> Tensors </subtitle>

tensor는 모든 딥러닝 toolkit에서 기본적인 빌딩 블럭입니다. 텐서의 근본적인 개념은 다차원 배열이라는 것입니다.

딥러닝에서 사용하는 텐서에 대해 주의해야 할 점은 딥러닝의 텐서가 수학에서 텐서 미적분(tensor calculus), 텐서 대수(tensor algebra)와 부분적으로만 관련 있다는 것입니다. 즉, 완전히 같은 텐서가 아닙니다. DL에서 텐서는 임의의 다차원 배열이지만 수학에서 텐서는 벡터 공간 사이의 매핑으로, 경우에 따라서 다차원 배열이 될 수도 있지만 그 뒤에 더 많은 의미를 가지고 있습니다. 

<br>

## The creation of tensors

pytorch는 8개의 텐서 데이터 타입을 가지고 있습니다. 
* 3개의 float type
    - 16-bit
    - 32-bit (torch.FloatTensor)
    - 64-bit
* 5개의 integer type
    - 8-bit signed
    - 8-bit unsigned (torch.ByteTensor)
    - 16-bit
    - 32-bit
    - 64-bit (torch.LongTensor)

위에 이름 달아둔 저 3개를 주로 많이 사용합니다.

<br>

파이토치에서 텐서를 생성하는 방법은 3가지가 있습니다.
* 해당 데이터 타입의 contructor를 호출
* numpy array나 python list를 tensor로 변환
* 특정 데이터로 텐서 만들기 
    - ex)torch.zeros()

위 방법 순서대로 생성해보겠습니다.

```python
import torch
import numpy as np

a = torch.FloatTensor(3, 2)
b = torch.FloatTensor([[1,2,3], [3,2,1]])
c = torch.zeros((3,2))

```

numpy array를 활용할 때 유의할 점이 있습니다. numpy.zeros() 로 만든 배열을 tensor로 만들 경우에 numpy array는 디폴트로 64-bit float으로 만들어집니다. 이 배열을 입력으로 받은 텐서도 DoubleTensor(64-bit float) type으로 만들어지게 되고요. 근데, DL에서 보통 이 정도 정밀하게 필요없고 또 메모리가 더 필요하거나 성능 오버헤드가 일어날 수 있습니다. 일반적으로는 32-bit float이나 심지어 16-bit float으로 해도 충분합니다. 그래서 이 경우에 다음과 같이 데이터 타입을 어느 한 쪽에서라도 특정해주는 것이 좋습니다.  
```
>>> n = np.zeros(shape=(3, 2), dtype=np.float32)
>>> torch.tensor(n)
tensor([[ 0., 0.],
        [ 0., 0.],
        [ 0., 0.]])

>>> n = np.zeros(shape=(3,2))
>>> torch.tensor(n, dtype=torch.float32)
tensor([[ 0., 0.],
        [ 0., 0.],
        [ 0., 0.]])
```

<br>

## Scalar tensors

PyTorch 0.4.0 이후로 0차원 텐서, 즉 scalar tensor 값도 지원합니다. 1차원 벡터의 연산 결과일 수도 있고 바로 생성한 결과일 수도 있습니다. item()이라는 메서드로 python value에 접근도 할 수 있습니다.

```
>>> a = torch.tensor([1,2,3])
>>> a
tensor([ 1, 2, 3])
>>> s = a.sum()
>>> s
tensor(6)
>>> s.item()
6
>>> torch.tensor(1)
tensor(1)
```

<br>

## Tensor operations

텐서로 할 수 있는 operation은 엄청 많습니다. [PyTorch 공식 문서](http://pytorch.org/docs/){:target="_blank"} 에서 확인할 수 있습니다. 대부분의 연산 메서드는 numpy 연산에 대응하고 있습니다.  
tensor operation에는 두가지 타입이 있습니다. 
* inplace operation
    - 이름에 _(underscore) 가 있음
    - tensor의 content를 가지고 작업 수행
    - object 그 자체가 반환됨.
    - 성능과 메모리 측면에서 보통 더 효과적
    - ex) tensor.abs_()
* functional operation
    - 원래 tensor를 건드리지 않고 카피 떠서 작업 수행
    - ex) abs(tensor)

<br>

## GPU tensors

PyTorch는 GPU와 CPU 모두에서 지원합니다. 모든 연산은 두 개의 버전을 가지고 있고 이는 자동으로 선택됩니다.
차이점은 GPU tensor는 torch 대신에 torch.cuda 패키지에 있다는 점입니다.  
예를 들어, FloatTensor
* CPU : torch.FloatTensor
* GPU : torch.cuda.FloatTensor

CPU에서 GPU로 변환시킬 때는 **to(device)** 를 사용하면 됩니다. 이는 텐서 자체를 바꾸는 것이 아니고 카피 떠서 특정 디바이스에 올라간 텐서를 만드는 것입니다. device에 "cpu"라고 하면 CPU memory에 올라가고 "cuda"라고 하면 GPU에 올라갑니다. 특정 GPU 번호에 할당할 때는 "cuda:번호"로 지정합니다.

```
>>> a = torch.FloatTensor([2,3])
>>> a
tensor([ 2., 3.])
>>> ca = a.to('cuda')
>>> ca
tensor([ 2.,3.], device='cuda:0')
>>> ca.device
device(type='cuda', index=0)
```

<br>

> <subtitle> Gradients </subtitle>

gradients의 자동계산은 Caffe toolkit에서 처음 구현되었고 DL 라이브러리에서 표준으로 정착되었습니다.  
최근 라이브러리들에서는 input에서 output까지 레이어의 순서만 잘 정의하면 모든 gradient 계산, 역전파가 정확하게 계산될 것입니다. 

gradient 계산에는 두 가지 접근법이 있습니다.

* Static graph : 사전에 계산법을 정의해야 하며 나중에는 변경할 수 없음.
* Dynamic graph : 
    - 실행되기 전까지 미리 그래프를 정의할 필요가 없다. 실제 데이터의 데이터 변환에 사용할 작업만 실행하면 된다.
    - 이 기간에 라이브러리는 수행된 연산의 순서를 기록하고, 그래디언트 계산을 요청하면 연산 기록을 제거하여 네트워크 패러미터의 그래디언트를 누적시킨다. 
    - 이 방법은 notebook gradients라고도 불리며 PyTorch, Chainer 등에서 구현되어 있다.

<br>

## Tensors and gradients

pytorch tensor는 gradient 계산 기능을 내장하고 있습니다. gradient와 관련한 tensor의 속성은 다음과 같습니다.  
* grad : 계산된 그래디언트 값을 포함하여 동일한 모양의 텐서를 유지시키는 속성
* is_leaf : 해당 텐서가 유저에 의해 만들어졌으면 True, 아니면 False
* requires_grad : 텐서의 그래디언트를 계산하길 원하면 True, 아니면 False. 이 속성은 리프 텐서부터 상속받을 수 있다.

```python
>>> v1 = torch.tensor([1.0, 1.0], requires_grad=True)
>>> v2 = torch.tensor([2.0, 2.0])

>>> v_sum = v1 + v2
>>> v_res = (v_sum*2).sum()
>>> v_res
tensor(12., grad_fn=<SumBackward0>)

>>> v_sum = v1 + v2
>>> v_res = (v_sum*2).sum()
>>> v_res
tensor(12., grad_fn=<SumBackward0>)

>>> v1.is_leaf, v2.is_leaf
(True, True)
>>> v_sum.is_leaf, v_res.is_leaf
(False, False)
>>> v1.requires_grad
True
>>> v2.requires_grad
False
>>> v_sum.requires_grad
True
>>> v_res.requires_grad
True

>>> v_res.backward()
>>> v1.grad
tensor([ 2., 2.])
>>> v2.grad
```

v2는 requires_grad=True를 주지 않았기 때문에 grad 계산이 되지 않습니다.

위와 같은 그래디언트 계산 능력으로 neuralnet optimizer를 만들 수 있습니다. 댜음 세션에서는 뉴럴넷 빌딩 블럭, 인기 있는 최적화 알고리즘, loss function을 간편하게 사용할 수 있는 함수들을 다룰 예정이지만 그것들이 아니더라도 torch tensor를 이용해서 자기 입맛에 맞게 바꿀 수 있습니다. 이 점이 pytorch가 DL 연구자들에게 인기 있는 이유 중 하나 입니다.

<br>

> <subtitle> NN building blocks </subtitle>

**torch.nn** package는 기본적인 기능 블럭들과 함께 사전 정의한 클래스들을 제공합니다. 모두 실제 상황을 고려하여 설계되었습니다. 예를들어, minibatch를 지원하고 기본값이 정상이며 가중치가 적절히 초기화되어 있습니다. 모든 모듈들은 callable하게 만들었습니다. 모든 클래스의 인스턴스는 인자가 적용되면 함수 역할을 할 수 있다는 의미입니다. 아래 예시에서 nn.Linear는 인스턴스 생성 후에는 함수로 역할을 수행합니다.

```python
>>> import torch.nn as nn
>>> l = nn.Linear(2, 5)
>>> v = torch.FloatTensor([1, 2])
>>> l(v)
tensor([ 1.0532,  0.6573, -0.3134,  1.1104, -0.4065], grad_
fn=<AddBackward0>)
```

<br>

torch.nn의 모든 클래스는 nn.Module 클래스를 상속하는데 이 nn.Module 클래스의 유용한 메서드를 살펴봅시다.

* parameters(): 그래디언트 계산에 필요한 모든 변수들의iterator를 반환 (즉, module weigts)
* zero_grad(): 모든 패러미터의 그래디언트를 0으로 초기화
* to(device): 모든 모듈 패러미터를 주어진 디바이스로 옮김.
* state_dict(): 모든 모듈 패러미터를 딕셔너리 형태로 반환하고 이는 model serialization에 유용함.
* load_state_dict(): state dictionary를 지닌 모듈을 초기화.

모든 클래스 목록은 [다음 공식 문서](http://pytorch.org/docs){:target="_blank"}에서 찾아볼 수 있습니다.

<br>

## Sequential

Sequential은 파이프라인 구조로 레이어를 쌓을 때 쓸 수 있는 아주 유용한 클래스입니다. 다음처럼 사용할 수 있습니다.

```python
>>> s = nn.Sequential(
... nn.Linear(2, 5),
... nn.ReLU(),
... nn.Linear(5, 20),
... nn.ReLU(),
... nn.Linear(20, 10),
... nn.Dropout(p=0.3),
... nn.Softmax(dim=1))
>>> s
Sequential(
    (0): Linear(in_features=2, out_features=5, bias=True)
    (1): ReLU()
    (2): Linear(in_features=5, out_features=20, bias=True)
    (3): ReLU()
    (4): Linear(in_features=20, out_features=10, bias=True)
    (5): Dropout(p=0.3)
    (6): Softmax()
)

>>> s(torch.FloatTensor([[1,2]]))
tensor([[0.1115, 0.0702, 0.1115, 0.0870, 0.1115, 0.1115, 0.0908,
0.0974, 0.0974, 0.1115]], grad_fn=<SoftmaxBackward>)
```

<br>

> <subtitle> Custom layers </subtitle>


<br>

> <subtitle> The final glue - loss functions and optimizers </subtitle>


<br>

> <subtitle> Monitoring with TensorBoard </subtitle>


<br>

> <subtitle> Example - GAN on Atari images </subtitle>


<br>

> <subtitle> PyTorch Ignite </subtitle>







<br><center><img src= "https://liger82.github.io/assets/img/post/20210507-DeepRLHandsOn-ch02-OpenAI-Gym/fig_record.png" width="70%"></center><br>

> <subtitle> Summary </subtitle>


<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 03 : Deep Learning with PyTorch
<br>
