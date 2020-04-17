---
layout: post
title: "Introduction to Rasa"
date: 2020-04-16
excerpt: ""
tags : [virtual assistant, chatbot, Rasa, 챗봇, tutorial, 설치]
comments: true
---

Rasa는 빌더 형태가 아닌 대화형 에이전트(챗봇)를 만들 수 있는 오픈소스입니다. 독일 스타트업인데 오픈소스 중에서
꽤 커뮤니티 활동에도 참여하고 있어서 쉽게 찾아볼 수 있습니다.

저도 라사에서 제공하는 마스터클래스 영상을 보고 라사에 대해 알아보겠습니다.

# Rasa 설치하기

## 설치 환경

* OS :  MacOS Mojave
* anaconda 사용

## 설치

1. conda environment 만들기(환경 이름은 rasa로 했습니다.)

```shell script
# env 만들기
conda create --name rasa python==3.7

# 새로 만든 가상환경에 접속하기
conda activate rasa
```

2. rasa 설치하고 rasa project 만들기

```shell script
# rasa 설치
pip install rasa

# rasa project 설치하기
rasa init
```

아래 캡쳐본처럼 rasa init을 치면 일단 설치 위치를 물어봅니다.
이미 설치하려는 프로젝트라면 그냥 enter 치시면 되고 아니라면 저처럼 디렉토리 이름을 지정해주시면 됩니다.
저는 튜토리얼대로 rasabot으로 했습니다.

그다음에는 만드냐는 질문에 Y,
최초 모델을 학습시키냐는 질문에 Y

![rasa_init](../assets/img/post/20200416-rasa/rasa_init.png)

학습이 마무리되면 아래와 같은 질문이 나옵니다.
> Do you want to speak to the trained assistant on the command line?
> (커맨드 라인으로 학습된 어시스턴트에 말 걸어볼래?)

Y 를 누르면 대화를 할 수 있습니다.

슬프다고 하면 새끼 호랑이 사진을 올려줍니다. 
/stop 을 입력하면 대화는 종료합니다.

여기까지가 [첫 번째 동영상](https://youtu.be/-F6h43DRpcU){:target="_blank"} 내용입니다.

---

# Rasa 살펴보기

최초 모델이 어떻게 구성되어 있는지 살펴보겠습니다.

여기까지 했으면 아래와 같은 파일이 해당 디렉토리에 있을 겁니다.

![setup_files](../assets/img/post/20200416-rasa/setupfiles.png)

여기서 인텐트와 엔티를 담고 있는 것이 *data/nlu.md* 파일입니다.
내용을 보면 아래 형식처럼 되어 있습니다.

![nlu.md](../assets/img/post/20200416-rasa/intent.png)

## training data format

rasa에서는 두 종류의 training data format를 지원하고 있습니다.

1. markdown
    * 사람이 쓰고 읽기 편한 포맷
    * '*', '-', '+'를 사용한 unordered list로 예시를 작성
    * intent와 entity로 그룹지을 수 있음 
    * 예시에서도 entity를 명명할 수 있음, e.g. \[entity\](entity_name)
    * 아래와 같이 다양하게 사용 가능
    ![markdown](../assets/img/post/20200416-rasa/dataformat_markdown.png)
    
2. json
    * 최상위에 'rasa_nlu_data'가 있고 그 아래가 다음과 같습니다.
    ```json
   {
    "rasa_nlu_data": {
        "common_examples": [],
        "regex_features" : [],
        "lookup_tables"  : [],
        "entity_synonyms": []
        }
    }
    ```
   * common_examples가 train data
   * regex_features는 정규식으로 intent와 entity를 잡고 싶을 때 사용할 수 있습니다.


config.yml 파일을 보면 현재 파이프라인이 어떻게 구성되어 있는지 볼 수 있습니다.
rasa init 직후에 보면 다음과 같습니다.

![config](../assets/img/post/20200416-rasa/config.png)


## story

가상 어시스턴트의 대부분은 대화 흐름을 가지고 있습니다. 다양한 용어로 표현되는데 
Rasa에서는 이를 story 라고 부릅니다.

어시스턴트에게 유저의 질문에 어떻게 답변해야 할지 알려주는 것입니다.
Rasa 의 core model 이 story 에서 대화를 학습합니다.

story는 기본적으로 다음과 같이 구성됩니다.

```markdown
## story1
* greet
   - utter_greet
```

실제 내용이 아니라 인텐트나 엔티티 이름이 들어갑니다.
* '##' 스토리 이름
* '*' user intent 혹은 user가 말한 entity
* '-' 어시스턴트의 action

어시스턴트의 action은 API를 콜하는 것을 포함해서 정의하는 바에 따라 모든 행동을 할 수 있습니다.

story 에 대해 더 자세히 알고 싶으면 [여기](https://rasa.com/docs/rasa/core/stories/#stories){:target="_blank"}를 눌러주세요


## domain

domain은 어시스턴트가 있는 세상이라고 표현합니다. 도메인은 인텐트, 액션, 템플릿으로 분리됩니다.
* 인텐트 : 유저가 말하는 내용의 의도
* 액션 : 어시스턴트가 행동하거나 말할 수 있는 것
* 템플릿 : 액션의 구체적인 내용

템플릿에서 이미지도 url로 올릴 수 있고 custom action도 endpoints.yml에 url 등록해놓으면 api도 콜할 수 있습니다.

....작성 중....

# Reference

* [Rasa Masterclass EP01](https://youtu.be/-F6h43DRpcU){:target="_blank"}
* [https://rasa.com/docs/getting-started/](https://rasa.com/docs/getting-started/){:target="_blank"}
* [https://rasa.com/docs/rasa/user-guide/rasa-tutorial/#create-a-new-project](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/#create-a-new-project){:target="_blank"}
