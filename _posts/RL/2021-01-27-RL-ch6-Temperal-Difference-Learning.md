---
layout: post
title: "[RL] ch06 Temperal Difference Learning [1]"
date: 2021-01-27
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, Sutton, TD, Q-learning, SNU CML]
comments: true
---

Sutton 교수의 "introduction to reinforcement learning" 교재를 기반으로 공부하다가 참고용으로 듣던 서울대학교 CML 연구실에서 하는 강의가 더 공부하기 좋아서 기준을 바꾸었습니다. 유튜브 주소는 다음과 같습니다.

* [SNU CML 강의 목록 in youtube](https://www.youtube.com/playlist?list=PLKs7xpqpX1beJ5-EOFDXTVckBQFFyTxUH){:target="_blank"}

이 강의가 책을 기반으로 하지만 필요에 따라 책의 챕터를 왔다리 갔다리 해서 TD learning에서 다루는 챕터는 책 기준으로 사실상 6장, 7장, 12장에 걸쳐있습니다. 

<br>

> <subtitle> Intro </subtitle>

* TD는 RL에서 중요하고 비교적 새로운 아이디어
* Monte Carlo와 Dynamic Programming의 장점들을 모은 조합
    - MC의 장단점
        - 장점 : 경험으로부터 학습 가능(model-free method)
        - 단점 : 에피소드가 끝나야 학습 가능. 실시간 학습 불가
    - DP의 장단점
        - 장점 : 최종 결과를 기다리지 않고 추정 가능
        - 단점 : 환경의 동역학이 필요
    - TD : 환경의 동역학 없이 경험으로부터 학습이 가능하면서, 최종 결과를 기다리지 않고 추정치를 업데이트할 수 있다.

<br>

> <subtitle> TD Prediction </subtitle>

* constant $$\alpha$$ Monte Carlo Algorithm
    - $$G_t$$는 total return 이기 때문에 하나의 에피소드가 끝나야 얻을 수 있다. 즉, 하나의 에피소드가 끝나야 가치를 업데이트할 수 있다.(offline method)

<center> $$ V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t)) $$ </center>

<br>

* TD에서는 즉각적인 업데이트를 위해 $$G_t$$를 $$R_{t+1}+\gamma V(S_{t+1})$$로 대체한다.
    - 한 스텝만 고려하고 있어서 **one-step TD, TD(0)**로 표현

<center> $$ V(S_t) \leftarrow V(S_t) + \alpha( R_{t+1}+\gamma V(S_{t+1}) - V(S_t)) $$ </center>

* TD(0)의 프로세스

<br><center><img src= "https://liger82.github.io/assets/img/post/20210127-RL-ch6-Temperal-Difference-Learning/algo-td.png" width="60%"></center><br>

> <subtitle> n-Step TD Prediction </subtitle>

* n-step으로 확장한 TD method
* MC와 TD(0)를 일반화한 것
    - N = 종단 상태면 MC와 같음

<br><center><img src= "https://liger82.github.io/assets/img/post/20210127-RL-ch6-Temperal-Difference-Learning/fig7.1.png" width="80%"></center><br>

<br>

* n-step TD의 업데이트 식은 다음과 같다.

    - For n = 1, one-step return

<center> $$ G_{t:t+1} = R_{t+1} + \gamma V_t(S_{t+1}) $$ </center>

    - For n = 2, two-step return

<center> $$ G_{t:t+2} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V_{t+1}(S_{t+2}) $$ </center>

    - For general n, n-step return

<center> $$ G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^{n} V_{t+n-1}(S_{t+n}) $$ </center>

    - n-step TD prediction algorithm

<center> $$ V_{t+n}(S_{t}) = V_{t+n-1}(S_{t}) + \alpha(G_{t:t+n} - V_{t+n-1}(S_{t})) $$ </center>
<center> while $$ V_{t+n}(s) \neq V_{t+n-1}(s), \forall s = S_t $$ </center>

        - while 조건의 의미는 변화가 없을 때까지 계속 업데이트 한다는 것입니다.
<br>

* Error reduction property

<center> $$ max_s |E_{\pi}[G_{t:t+n}|S_t=s] - v_{\pi}(s)| \leq \gamma^n max_s \left | V_{t+n-1}(S_t) - v_{\pi}(s) \right | $$ </center>


> <subtitle> $\lambda$-return </subtitle>

<br>

> <subtitle> Model-Free Control </subtitle>

<br>

> <subtitle>  </subtitle>

<br>

> <subtitle>  </subtitle>

<br>

> <subtitle>  </subtitle>




<br>

---

> <subtitle> References </subtitle>
* Deep Reinforcement Learning Hands On 2/E Chapter 06 : Deep Q-Networks
* [](){:target="_blank"}

<br>
