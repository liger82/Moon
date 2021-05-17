---
layout: post
title: "Natural Gradient 인기 없는 이유?"
date: 2021-05-17
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Natural Gradient]
comments: true
---

> <subtitle> Natural Gradient </subtitle>

Natural Policy Gradient를 배우면서 이 알고리즘의 핵심인 natural gradient에 대해 알게 되었습니다. 

[natural policy gradient slideshare](https://www.slideshare.net/SooyoungMoon3/natural-policy-gradient){:target="_blank"}

솔직히 이 분이 정말 설명을 잘 해놓으셔서 natural gradient에 대한 설명은 링크만 올려두겠습니다.

> <subtitle> Natural Gradient 인기가 왜 없을까? </subtitle>

한데 생긴 의문이 이거였습니다. 보기에는 엄청 좋아보이는데 왜? 왜? 다른 딥러닝 모델에서는 많이 안보일까였습니다. 역시 누군가 의문을 던진 분이 있었고 또 누가 답해주셨더라고요. 저는 기록 차원에서 번역해두려고 합니다. 아래 원문 링크는 달아두었습니다.

<br>

딥러닝 모델의 경우, 사람들은 Hessian Free optimization을 사용하고 있었고 근사치로 사용하는 Hesian Free는 Natural gradient와 같은 것으로 밝혀졌습니다. 그러나 SGD가 일반적으로 더 효율적이어서 Hessian Free나 Natural Gradient를 사용하지 않게 된 것입니다. 최적화에 비용이 많이 드는 방법을 사용하는 어려운 모델보다는 SGD를 잘 작동하도록 모델을 변경하는 것이 훨씬 쉽습니다.(예: 시그모이드 대신 rectified linear units/maxout 사용 또는 기존 RNN 대신 LSTM 사용). Natural gradient는 종종 SGD보다 업데이트 횟수 면에서 훨씬 더 빠르게 수렴하지만 업데이트당 wall time의 비용은 그 효과를 취소할 만큼 충분히 높습니다.

<br>

> For deep learning models, people were using Hessian Free optimization for a while. With the approximations people were using, Hessian Free turns out to be equivalent to natural gradient. But eventually people quit using Hessian Free / Natural Gradient because Stochastic Gradient Descent was usually more efficient. It's a lot easier to change your model family (e.g., use rectified linear units/maxout rather than sigmoids, or use an LSTM instead of a traditional RNN) to make SGD work well than to use a difficult model family with an expensive optimization method. Natural gradient often converges much faster than SGD in terms of the number of updates it makes. But the cost in wall time per update is high enough to cancel out that effect.


---

> <subtitle> References </subtitle>

* [https://www.reddit.com/r/MachineLearning/comments/2qpf9x/why_is_the_natural_gradient_not_used_more_in/](https://www.reddit.com/r/MachineLearning/comments/2qpf9x/why_is_the_natural_gradient_not_used_more_in/){:target="_blank"}