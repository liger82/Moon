---
layout: post
title: "Rasa Masterclass Episode 4 : Pipeline components"
date: 2020-05-11
excerpt: ""
tags : [virtual assistant, chatbot, Rasa, 챗봇, tutorial, pipeline, training, test, pre-configured pipelines]
comments: true
---

#  Training Pipeline Overview

파이프라인에서는 어떤 구성요소가 있어야 하는지를 정의하는 것뿐만 아니라, 
어떻게 구성요소가 배치되어야 하는지 순서까지 규정한다.

1. 필요하다면 사전학습 언어 모델을 로드한다(Optional)
2. 데이터를 단어 혹은 토큰으로 토크나이즈한다.
3. Named Entity Recognition(NER). 모델에게 메시지 안의 어떤 단어가 엔티티이고
어떤 엔티티 유형인지 인식하도록 학습시킨다.
4. Featurization. 토큰을 벡터나 dense numeric representation 으로 변환한다.
이 단계는 ner 전이나 후에 할 수 있다. 하지만 토크나이징 후에, 인텐트 분류 전에 실시한다.
5. Intent Classification. 사용자 메시지의 의미를 예측할 수 있도록 모델을 학습시킨다.


# Training Pipeline Components


