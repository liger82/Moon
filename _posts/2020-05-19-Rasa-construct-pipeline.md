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

```markdown
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

```markdown

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
SpaCy 를 사용한다. 


# 여기서부터 시작할 것.

의문점...
spaCy 가 한글을 지원하고 있지 않다.
다만 mecab 을 따로 설치하면 분석이 되긴 한다. 모델이 있지는 않다.
feature 만 spaCy 로 뽑고 모델은 DIET 에서 구성하는 것이니 괜찮을까??

# References

* [https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/){:target="_blank"}
