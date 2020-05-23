---
layout: post
title: "Rasa Masterclass : Choosing a Pipeline"
date: 2020-05-19
excerpt: ""
tags : [virtual assistant, chatbot, Rasa, 챗봇, tutorial, pipeline, 파이프라인 구성]
comments: true
---

에피소드 3과 4을 정리하면서도 파이프라인 구성이 어려운데 바로 에피소드 5로 넘어가는 것은 아니다 싶어서 
공식 docs 를 살펴보니 "choosing a pipeline" 부분이 있어 정리하고자 한다.

크게 두 가지로 구분한다: 짧은 대답과 긴 대답이다.


## 짧은 대답

### 1. 영어

```yaml
language: "en"

pipeline:
  - name: ConveRTTokenizer
  - name: ConveRTFeaturizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
```

라사에서 제안하는 영어 버전은 짧은 대답이나 긴 대답이나 동일하다.
DIET 아키텍처의 최적 조건을 보여주고 있다.
DIET 아키텍처는 Dual Intent and Entity Transformer 의 줄임말로,
인텐트 분류와 엔티티 인식을 같이 할 수 있으면서 input feature 를 dense 한 것 뿐 아니라
sparse 한 것도 함께 사용할 수 있다. 이는 두 종류의 feature 의 차원을 동일하게 만들어서
concatenate 한 후에 입력으로 Transformer 의 입력으로 제공한다.
또한 인텐트는 Transformer 를 통해서 분류하고, 엔티티는 CRF 와 Transformer 의 출력값을 함께 사용하여 최종값을 예측한다.
자세한 방법론은 논문 보는 것을 추천한다. 길지도 않다. 

DIET 아키텍처는 dense feature 를 다양하게 받아서 사용할 수 있다. 다만 논문에 의하면
ConveRT 를 사용했을 때 성능과 속도면에서 가장 훌륭했다.
그래서 여기서도 dense feature 는 ConveRT 모델의 임베딩이다. ConveRT 는 PolyAI 라는 회사에서
BERT 보다 실용적인 목적으로 만들었다. 성능은 유지하면서 속도는 빠르다.  

1. ConveRTTokenizer : 그래서 ConveRT 의 토크나이저를 사용한다. 
2. ConveRTFeaturizer : ConveRT 로 dense feature 를 추출한다.
3. RegexFeaturizer : 정규식으로 추출할 수 있는 feature 를 뽑아낸다
4. LexicalSyntacticFeaturizer : 어휘, 통사적으로 알아낼 수 있는 feature 를 추출한다
5. CountVectorsFeaturizer : 단어 수준으로 BOW feature 추출한다.
6. CountVectorsFeaturizer(char_wb) : 문자 단위로 추출한다.
7. DIETClassifier
8. EntitySynonymMapper : entity 중에 유사어로 등록된 value 를 매핑한다. 의문은 왜 이것을 DIET 뒤에 하는지다.
9. ResponseSelector : response 후보군 중에 최종 response 를 예측하는 작업을 한다. 


### 2. 비영어

이 부분이 사실 초점이다. 영어와 비영어를 나눈 이유는 사실 ConveRT 를 사용할 수 있느냐 아니다 이다.
왜냐면 라사에서 ConveRT 모델을 영어로만 사전 학습 시켜놓아서 성능이 좋지만 비영어권 언어에서는 사용할 수 없다.

```yaml

language: "fr"  # your two-letter language code

pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
```

비영어권 언어의 짧은 대답용은 sparse feature 만 입력으로 사용하는 것을 추천다.

1. WhitespaceTokenizer : 그 언어 속성을 모를 경우 가장 기본으로 사용하면 좋다. 공백 기준
2. RegexFeaturizer
3. LexicalSyntacticFeaturizer
4. CountVectorsFeaturizer
5. CountVectorsFeaturizer
6. DIETClassifier
7. EntitySynonymMapper
8. ResponseSelector

DIETClassifier 빛을 보는 순간입니다. dense feature 를 사용하지 않은 경우에도 
본인들이 자랑하는 아키텍처를 사용할 수 있다는 점이다. 


## 긴 대답

### 1. 영어 : 짧은 대답과 동일

### 2. 비영어

비영어라도 pre-trained word embeddings(dense feature)를 사용하고 싶을 때는 
SpaCy 를 사용하면 된다고 라사 공식 문서에는 나와 있습니다만...!

spacy 에서 지원하는 언어가 별도로 존재하기 때문에 확인이 필요합니다.

2020년 5월 23일 기준으로 봤을 때(나중에 업데이트 될 수도 있으니 확인 바랍니다.)  
[https://spacy.io/models](https://spacy.io/models){:target="_blank"}  

다음 언어들을 지원합니다.
* 영어, 독일어, 프랑스어, 이태리어, 스페인어, 포르투갈어, 네덜란드어, 그리스어, norwegian bokmal, 리투아니아어

한글의 경우 mecab 을 따로 설치하면 토크나이저를 활용하여 기본적인 것을 활용할 수 있으나
사전학습된 모델은 없어서 dense feature 는 뽑을 수 없습니다.

spacy 가 지원하는 언어일 경우는 다음과 같이 사용할 수 있습니다.

```yaml
language: "fr"  # your two-letter language code

pipeline:
  - name: SpacyNLP
  - name: SpacyTokenizer
  - name: SpacyFeaturizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
```

한글을 언어로 하고 dense feature 도 사용할 수 있게 하려면
HFTransformersNLP 를 사용하면 된다.

HFTransformers 의 경우는 정해진 임베딩을 사용하는 것은 아니고 BERT, GloVe 등 다양하게 활용할 수 있다고 한다.
기본으로 설정하면 BERT 를 사용하여 다언어 모델을 활용한다. 이는 학습 과정에서 로그를 통해 확인하였다.
HFTransformersNLP 도 tokenizer 와 featurizer 에 디펜던시가 있어서 지정한 항목으로만 파이프라인을 구성해야 한다.
추가로 따로 아래 명령어를 통해 HFTransformersNLP 를 설치해주어야 한다
>pip install rasa[transformers] 

예시로 짜보면 다음과 같다.

```yaml
language: kr
pipeline:
  - name: HFTransformersNLP
  - name: LanguageModelTokenizer
  - name: LanguageModelFeaturizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100

```

# Choosing the Right Components

파이프라인은 주로 다음의 파트로 구성되어 있다. 
1. Tokenization
2. Featurization
3. Entity recognition/ Intent classification/ Response Selectors
  
적절한 component 선택을 위해 각 component 에 대해 알고 있어야 한다.
[component 공식 문서](https://rasa.com/docs/rasa/nlu/components/){:target="_blank"}를 참조하길 바란다.
custom component 는 따로 다룰 예정이다.

# Multi-Intent Classification

rasa 에서는 인텐트를 여러 개의 레이블로 분리할 수 있다. 수평적 구조와 수직적(위계적) 구조로 나뉜다.
예를 들어, *thank+goodbye* 는 고맙다는 인텐트와 인사 인텐트가 수평적으로 구분될 수 있고,
*feedback+positive* 는 위계적으로 구분할 수 있다.
이 기능은 **DIET classifier 를 사용할 때만 가능하다.** 
또한 아래 flag 들을 파이프라인에 명시해주어야 한다.

* intent_tokenization_flag: True 면 인텐트를 토크나이징 시도한다.
* intent_split_symbol: intent 를 구분할 delimeter 이며 기본은 "_" 이다. 위의 예시에서는 "+"이다.

```yaml
language: "en"

pipeline:
- name: "WhitespaceTokenizer"
  intent_tokenization_flag: True
  intent_split_symbol: "+"
- name: "CountVectorsFeaturizer"
- name: "DIETClassifier"
```

사실 이것을 직접 테스트 해본 결과가 약간 이상하다. 구분자를 설정했다는 의미는 파이프라인에서뿐만 아니라
학습 데이터 상에서도 모두 바꿔 주었다는 의미이다.
* WhitespaceTokenizer 에 구분자를 "+"로 설정했을 때 : 영어, 한글 둘 다 인텐트 분리 안됨.
* WhitespaceTokenizer 에 구분자를 "/"로 설정했을 때 : 영어, 한글 둘 다 인텐트 분리 됨.
* WhitespaceTokenizer 에 파이프라인 설정은 안 하고 데이터에서만 인텐트 이름에 "/"를 설정했을 때 : 영어, 한글 둘 다 인텐트 분리 됨.
* custom Tokenizer 에 구분자를 "+"로 설정했을 때 : 영어, 한글 둘 다 인텐트 분리 안됨.
* custom Tokenizer 에 구분자를 "/"로 설정했을 때 : 영어, 한글 둘 다 인텐트 분리 됨.
* custom Tokenizer 에 파이프라인 설정은 안 하고 데이터에서만 인텐트 이름에 "/"를 설정했을 때 : 영어, 한글 둘 다 인텐트 분리 됨.
 
즉, 구분자를 / 으로 하면 되었다. 파이프라인에서 설정하지 않아도.

인텐트가 분리된 예시이다. chitchat 단일의 인텐트를 입력하지 않았고, chitchat/ask_age 로 입력했었다. 

```shell script
NLU model loaded. Type a message and press enter to parse it.
Next message:
몇 살이니?
{
  "intent": {
    "name": "chitchat+ask_age",
    "confidence": 0.5868811011314392
  },
  "entities": [],
  "intent_ranking": [
    {
      "name": "chitchat+ask_age",
      "confidence": 0.5868811011314392
    },
    {
      "name": "chitchat",
      "confidence": 0.3009713590145111
    },
    {
      "name": "bot_challenge",
      "confidence": 0.09405096620321274
    },
    {
      "name": "greet",
      "confidence": 0.0055800131522119045
    },
    {
      "name": "칫챗",
      "confidence": 0.0037472655531018972
    },
    {
      "name": "goodbye",
      "confidence": 0.0036815390922129154
    },
    {
      "name": "칫챗+이름묻기",
      "confidence": 0.003399777924641967
    },
    {
      "name": "thank",
      "confidence": 0.0016880047041922808
    }
  ],
  "response_selector": {
    "default": {
      "response": {
        "name": "utter_age",
        "confidence": 0.999998927116394
      },
      "ranking": [
        {
          "name": "utter_age",
          "confidence": 0.999998927116394
        },
        {
          "name": "utter_이름_알려주기",
          "confidence": 1.113874986913288e-06
        }
      ],
      "full_retrieval_intent": "chitchat/ask_age"
    }
  },
  "text": "몇 살이니?"
}
Next message:

```

커스텀 토크나이저에서 안 작동이 안 되는 것은 그 부분을 구현 안한 것일 수도 있다는 의심이라도 들지만,
WhitespaceTokenizer 에서 안 되는 것은 에러가 있는 듯 하다. 

# Handling Class Imbalance

데이터 양에 편향이 있을 수 있다. 라사에서는 이를 교정하기 위해 balanced batch strategy 를 기본으로 실시하고 있다.
즉, 파이프라인에서 특별히 설정하지 않아도 작동한다. balanced batch strategy 는 배치 단위로 학습을 하여 
데이터가 적은 클래스도 배치에 고루 분포되게 한다. 다만 데이터의 비율 자체도 의미가 있을 수 있기 때문에 이는 유지한다.
이 전략을 사용하고 싶지 않을 때는 다음과 같이 config.yml 에서 설정하면 된다.

```yaml
language: "en"

pipeline:
# - ... other components
- name: "DIETClassifier"
  batch_strategy: sequence
```

# Comparing Pipelines

앞선 여러 컴포넌트를 구성하는 과정에서도 테스트가 필요하지만 완성 후에도 성능 테스트가 필요하다.
라사에서는 파이프라인 비교 툴을 제공한다. 

```shell script
$ rasa test nlu --config pretrained_embeddings_spacy.yml supervised_embeddings.yml
  --nlu data/nlu.md --runs 3 --percentages 0 25 50 70 90
```

nlu.md(학습 데이터 파일)의 데이터를 학습/테스트 셋으로 나누어서 진행한다. percentages 뒤에 수치들이 테스트셋의 비율이다.
metric 은 f1 score 를 사용한다.

이렇게 비교해볼 때는 데이터가 충분해야 의미가 있으니 데이터부터 준비하길 바란다.


# Pipeline Templates (deprecated)

파이프라인 shortcut 기능이다.


# 마무리 소견

라사에서 사용할 수 있는 것은 사용하되, 커스텀 component 는 구현이 필요하다. 이는 나중에 다룰 예정이다.
파이프라인 구성에 테스트는 필수다!! 특히 한국어를 사용언어로 지정하려면 많은 테스트가 필요해보인다.
또한 파이프라인 비교할 수 있도록 구성되어 있다는 점에서 배려심이 돋보인다.


# References

* [https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/){:target="_blank"}
* [https://rasa.com/docs/rasa/user-guide/testing-your-assistant/#comparing-nlu-pipelines](https://rasa.com/docs/rasa/user-guide/testing-your-assistant/#comparing-nlu-pipelines){:target="_blank"}
