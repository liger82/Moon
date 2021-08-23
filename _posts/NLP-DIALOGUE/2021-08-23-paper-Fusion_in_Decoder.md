---
layout: post
title: "[paper review] Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering ; Fusion in Decoder"
date: 2021-08-23
excerpt: ""
categories: [NLP/Dialogue]
tags : [virtual assistant, Open-domain Question Answering, QA, BlenderBot 2.0, FiD, Fusion in Decoder, paper review]
comments: true
---

>Authors : Gautier Izacard, Edouard Grave 
>Institution : Facebook AI Research, ENS, PSL University, Inria 
>Publication Date : Feb 3, 2021   
>Paper link : [https://arxiv.org/pdf/2007.01282.pdf](https://arxiv.org/pdf/2007.01282.pdf){:target="_blank"}  
>Github : [https://github.com/facebookresearch/FiD](https://github.com/facebookresearch/FiD){:target="_blank"}  

<br>

> <subtitle> Abstract & Introduction </subtitle>

BlenderBot 2.0 을 이해하기 위해서 꼭 알아야 하는 아키텍쳐 중 하나인 FiD가 처음 소개된 논문입니다.  
open domain 대화 시스템에서 외부 지식을 활용하는 부분은 일부 지식을 학습시킨다는 점에서 유한한 학습 데이터의 한계를 해결할 수 있어서 중요합니다.
FiD의 목적은 open domain QA를 위한 생성 모델이 "외부 지식"을 입력으로 받았을 때 이를 효과적으로 활용하는 것입니다.

FiD는 검색 기반 생성 모델입니다. 검색을 통해 나온 구절과 질문을 입력으로 받아 최종 응답을 생성해내는 방식입니다.
개념적으로 굉장히 간단하지만 TriviaQA와 NaturalQuestions(NQ)에서 기존 방법들을 제치고 SOTA를 찍었습니다.

검색한 구절의 개수가 늘어날 때, 성능 개선도 크게 있었다는 점을 고려하면 생성 모델은 추출 모델에 비해 여러 구절로부터 나오는 
힌트를 결합하는 것을 더 잘하는 것으로 보입니다.

<br>

> <subtitle> Related work </subtitle>

## Open domain question answering

이 과제는 특정 도메인이 아닌 범용 목적의 QA 과제입니다. 
* open domain QA는 오랜 기간 어려운 문제였지만 [Chan et al. (2017)](https://arxiv.org/abs/1704.00051){:target="_blank"} 연구에서 처음 이렇다 할 성과가 나왔습니다. 위키피디아에서 관련 문서를 검색하고, 해당 문서로부터 대답을 추출하는 방식입니다. 
* 그 후에 대답에 대응하는 범위 전체에 대해 global normalization 하는 것도 나왔습니다. 
* [Wang et al. (2018)](https://arxiv.org/abs/1711.05116){:target="_blank"}에서는 신뢰도와 커버리지 점수를 사용하여 여러 단락의 답변을 종합하는 기법을 제시하였습니다.

<br>

## Passage retrieval

open domain QA에서 중요한 단계로, QA 시스템을 개선하는데 핵심적인 영역입니다.

* (2017) [Reading Wikipedia to answer open-domain questions.](https://arxiv.org/abs/1704.00051){:target="_blank"}
    - 처음에는 TF/IDF 기반의 sparse representation 을 기반으로 관련 문서를 검색
* (2018) [Ranking paragraphs for improving answer recall in open-domain question answering.](https://aclanthology.org/D18-1053/){:target="_blank"}
    - 지도학습 방법 도입. BiLSTM 기반으로 단락을 rerank하는 방식
* (2018) [Reinforced ranker-reader for open-domain question answering.](https://arxiv.org/abs/1711.05116){:target="_blank"}
    - ranking system에 강화학습 사용
* (2019) [Knowledge guided text retrieval and reading for open domain question answering.](https://arxiv.org/abs/1911.03868){:target="_blank"}
    - 추가적인 정보로 위키피디아 사용
* (2020) [Dense passage retrieval for open-domain question answering.](https://arxiv.org/abs/2004.04906){:target="_blank"}
    - dense representation 기반으로 가장 가까운 것을 찾음. 질문-대답 쌍 형태로 weak supervision으로 학습
* (2020) [Realm: Retrieval-augmented language model pre-training.](https://arxiv.org/abs/2002.08909){:target="_blank"}
    - cloze(빈칸 채우기) task로 사전학습하고 end-to-end로 finetuning

<br>

## Generative question answering

질문에 대해 응답을 생성해서 답변해야 하는 과제입니다. 응답이 관련 문서 내에 일치하는 것이 없어야 합니다. 

* (2019) [Exploring the limits of transfer learning with a unified text-to-text transformer.](https://arxiv.org/abs/1910.10683){:target="_blank"}
    - SQuAD와 같은 독해 과제에서 생성모델이 경쟁력 있음을 보임
* (2020) [How much knowledge can you pack into the parameters of a language model?](https://arxiv.org/abs/2002.08910){:target="_blank"}
    - 추가적이 지식을 사용하지 않는 거대한 사전학습 생성 모델
* (2020) [Retrieval-augmented generation for knowledge-intensive nlp tasks.](https://arxiv.org/abs/2005.11401){:target="_blank"}
    - Retrieval Augmented Generative model 제안. 현 연구와 유사점 있음
    - 검색된 구절을 어떻게 처리하는지가 RAG와 다른 점.

<br>

> <subtitle> Method </subtitle>

open-domain qa를 처리하기 위해 두 가지의 단계를 거칩니다.

1. support passages를 검색한다.
2. passages 를 seq2seq model로 처리하여 응답을 생성한다.

## Retrieval

관련 구절 검색에 두 가지 방법을 비교하였습니다.

1. BM25
    - 구절을 bag of words로 표현
    - ranking function은 TF/IDF에 기반
    - Apache Lucene 사용
    - SpaCy Tokenizer 사용

2. DPR(Dense Passages Retrieval)
    - 구절과 질문을 dense representation로 나타내면서 각각의 BERT network 사용(Bi-encoder)
    - ranking function은 질문과 구절 간 내적(dot-product) 값을 기준으로 함
    - FAISS 라이브러리로 최적 근사값 계산

## Reading

생성 모델은 비지도 학습 데이터로 사전학습한 seq2seq network에 기반합니다.


<br>

> <subtitle> Experiments </subtitle>

<br>

> <subtitle> Conclusion </subtitle>


<br>

---

> <subtitle> References </subtitle>

* [](){:target="_blank"}









