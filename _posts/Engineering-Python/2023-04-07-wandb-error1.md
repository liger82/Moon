---
layout: post
title: "wandb error | Selected runs are not logging media for the key"
date: 2023-04-07
excerpt: ""
categories: [Engineering/Python]
tags : [wandb, logging error, not logging media for the key ]
comments: true
---

WandB(Weights and Biases)는 하이퍼파라미터, 시스템 메트릭 및 예측을 추적하는 중앙 대시보드로, 실시간으로 모델을 비교하고 결과를 공유할 수 있습니다.

(완디비로 읽었었는데 이게 Weights 의 W 와 Biases의 B 를 and 로 엮은 것을 말하는 것이었네요..ㅎㅎ)

이 글에서는 wandb를 사용할 때 저를 당황스럽게 했던 에러와 그 해결책을 다루겠습니다.

> <subtitle> Issue </subtitle>

전 다음과 같이 accelerate 로 학습을 진행했었고 로그 툴로 wandb를 사용했습니다.

```python
from accelerate import Accelerator

# Tell the Accelerator object to log with wandb
accelerator = Accelerator(log_with="wandb")

# Initialise your wandb run, passing wandb parameters and any config information
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team",
    "name":"nlp-task-230407"}}
    )

...

# Log to wandb by calling `accelerator.log`, `step` is optional
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# Make sure that the wandb tracker finishes correctly
accelerator.end_training()
```

<br>

문제는 log 를 찍을 때 제가 리스트로 주어서 에러가 났습니다.

그 이후에 그 부분을 단일값으로 변경했는데도 다음과 같은 에러 메세지가 wandb 차트에 등장하였습니다.

```bash
wandb Selected runs are not logging media for the key train_loss, but instead are logging values of type number.
```

<br>

> <subtitle> Solution </subtitle>

솔직히 이걸로 삽질 좀 많이 했습니다. 별별 타입으로 다 변환했는데고 안 되고 깃헙 이슈에 있는 해결책들도 여럿 해봤는데 안되었습니다. 그러다가 아래 레퍼에 있는 깃헙 이슈 마지막 글에 달린 가장 간단한 해결책을 시도했더니 바로 올라가더군요!!

<br>

해결책은 **프로젝트를 지우고 새로 만들기**였습니다.

<br><center><img src= "https://liger82.github.io/assets/img/post/20230407-wandb-error1\fig1.PNG" width="80%"></center><br>


<br>

허무했지만 다른 분들도 이런 일이 없도록 바로 결론 나오는 글을 남겨봅니다.


<br>

---

> <subtitle> References </subtitle>

* [https://docs.wandb.ai/guides/integrations/accelerate#start-logging-with-accelerate](https://docs.wandb.ai/guides/integrations/accelerate#start-logging-with-accelerate){:target="_blank"}
* [https://github.com/wandb/wandb/issues/1206](https://github.com/wandb/wandb/issues/1206){:target="_blank"}
