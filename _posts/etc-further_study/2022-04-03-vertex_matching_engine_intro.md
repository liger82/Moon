---
layout: post
title: "Vertex Matching Engine: Blazing fast and massively scalable nearest neighbor search"
date: 2022-04-03
excerpt: " "
categories: [etc/further_study]
tags : [Vertex Matching Engine, embedding search, matching engine, google, gcs, search engine]
comments: true
---

> <subtitle> Intro </subtitle>

이 글은 구글 클라우드 블로그 포스트에서 Vertex Matching Engine(VME)에 대한 간략한 소개 글을 번역, 정리한 글입니다.

<br>

> <subtitle> Background </subtitle>

vector embedding 이 굉장히 유용한 방법임에도 현 시점의 데이터베이스가 벡터 임베딩과 함께 쓰기 좋게 만들어지지 않아 문제가 있습니다.

특히, 데이터베이스는 특정 벡터의 최근접 이웃(nearest neighbors)을 찾기 위해 고안되어 있지 않습니다. 
대량의 데이터셋에서 계산하기 어렵고, 신속하고 규모에 맞는 정교한 근사 알고리즘이 필요한 상황입니다.

VME 는 앞선 문제를 해결하고자 등장했습니다. 두둥탁! 

빠르고 확장 가능하며, 벡터 유사도 검색에서 완전히 통제할 수 있는 솔루션입니다.

검색은 물론이고 추천 처럼 다양하게 사용될 수 있습니다.


<br>

> <subtitle> Vertex Matching Engine </subtitle>

VME 의 장점은 다음과 같습니다.

1. Scale
    * 초당 고수준 쿼리, 낮은 지연 속도로, 수십억 임베딩 벡터에서 검색을 가능하도록 한다.
    * 반면, 전형적인 nearest neighbor service는 수백만 벡터에서 검색 가능하다.
2. Lower TCO
    * 수백만 벡터 규모에서 작동하는 서비스를 설계하려면, 자원을 효율적으로 쓰도록 만들어야 한다.
    * 구글의 자원 효율성 덕분에 실제 VME 는 다른 비교군들보다 최대 40% 저렴하다.
3. Low latency with high recall
    * 근거 논문: [https://arxiv.org/abs/1908.10396](https://arxiv.org/abs/1908.10396){:target="_blank"}
    * 실험 결과, VME 는 recall 95-98% 상황에서 90 백분위수 지연속도가 10ms 미만이었다.
4. Fully managed
    * VME 는 오토스케일링까지 해주는 완전히 통제된 솔루션이다. 인프라를 관리하는데 신경 쓸 필요가 없다.
5. Built-in filtering
    * VME 는 간단한 것부터 복잡한 필터링 기능을 내장하여 고객이 boolean logic으로 쉽게 필터링을 사용할 수 있도록 제공한다.


<br>

> <subtitle> How to use Vertex Matching Engine </subtitle>

VME를 사용하는 방법은 어렵지 않습니다.

1. 사용자가 미리 계산된 임베딩을 파일로 GCS(Google Cloud System)에 제공한다.
    - gcs에서 제공하는 임베딩 계산 방식을 써도 되고 사용자가 직접 만든 임베딩을 올려도 된다.
    - 각 임베딩은 ID, optional tags(tokens or labels) 를 가지고 있어야 한다.
1. VME는 임베딩을 받아들여 인덱스를 생성한다
    - 이 인덱스는 벡터 유사도 매칭을 위한 온라인 쿼리를 수용할 준비가 된 지점에 클러스터에 디폴로이된다.
1. 클라이언트는 쿼리 벡터와 함께 해당 인덱스를 요청하고 돌려 받을 nearest neighbors 개수를 지정한다
1. 클라이언트는 매칭된 벡터의 ID와 유사도 점수를 돌려 받는다.

<br>

실제 어플리케이션에서 주기적으로 임베딩을 업데이트하거나 새로운 임베딩을 생성하는 것은 흔한 일입니다.
* 사용자는 인덱스 업데이트를 수행하여 임베딩 배치를 업데이트 할 수 있다.
* 새로운 임베딩으로부터 업데이트된 인덱스가 생성될 것이고, 새로운 인덱스가 기존 인덱스를 지연 시간 없이 대체한다.

인덱스를 생성할 때, 지연 시간과 recall 사이의 밸런스를 조정하기 위해 인덱스를 튜닝하는 것은 중요합니다.
- 주요 튜닝 패러미터는 [여기](https://cloud.google.com/vertex-ai/docs/matching-engine/using-matching-engine#tuning_the_index){:target="_blank"} 에 정리되어 있다.
- Matching engine은 미세한 조정을 하기 위한 brute-force search 할 수 있는 기능도 제공한다. 
    - 이 기능은 느리기 때문에 production 단계에서는 사용하지 않는 게 좋다.

<br>

> <subtitle> Opinion </subtitle>

임베딩 벡터 검색이 정확도 면에서 좋을 지 몰라도 실제 서비스에서는 속도가 엄청 중요해서 (빨리빨리 민족이라고요!) 상용화가 어려웠는데 VME는 이를 해결하고자 한 솔루션이면서 사용하기 편리하다고 강조하네요.

다음엔 실제로 써보고 글을 써보도록 하겠습니다!

<br>

> <subtitle> References </subtitle>

* [google cloud vertex matching engine 소개 블로그 포스트](https://cloud.google.com/blog/products/ai-machine-learning/vertex-matching-engine-blazing-fast-and-massively-scalable-nearest-neighbor-search){:target="_blank"}

<br>