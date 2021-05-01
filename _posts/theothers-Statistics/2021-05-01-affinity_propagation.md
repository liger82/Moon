---
layout: post
title: "[Clustering] Affinity Propagation"
date: 2021-05-01
excerpt: " "
categories: [The Others/Statistics]
tags : [AP, Affinity Propagation, Statistics]
comments: true
---

<br><center><img src="https://uploads.toptal.io/blog/image/92524/toptal-blog-image-1463639242851-65077729f48e9e7f8e0d0ca68cb4a19f.jpg" width=70%></center><br>

> <subtitle> 개괄 </subtitle>

<br>

Affinity Propagtion(AP) 은 데이터 포인트 간 "message passing" 개념을 기반으로 하는 클러스터링 알고리즘이다.  
k-means와 k-medoids 같은 클러스터링 알고리즘과 달리, AP는 알고리즘 실행 전에 클러스터의 개수를 요구하지 않는다. 
k-medoids 와 유사하게, AP는 입력 셋에서 클러스터를 대표하는 데이터(exemplars)를 찾는다.

<br>

K-means와 그 유사한 알고리즘의 주요 단점은 클러스터의 개수를 정하고 최초 대표 점들을 선정해야 한다는 점이다.  
AP는 데이터 포인트 쌍 간의 유사도를 측정하여 입력값으로 사용하고, 동시에 모든 데이터 포인트를 잠재적인 대표 데이터로 여긴다.
좋은 품질의 대표점들과 해당 클러스터들이 나타날 때까지 데이터 포인트 간에 실제 값 메시지가 교환된다.

<br>

AP는 입력 값으로 두 개의 데이터 셋을 요구한다.  

1. 데이터 포인트들 간의 유사도 : 한 개의 데이터 포인트가 다른 점의 대표 점이 되는 데 얼마나 적합한 지를 나타내는 지표.  
    * 두 개의 점 간 유사도가 없으면 같은 클러스터에 속해있을 수 없으므로, 유사도는 생략되거나 
    $$ -\infty $$ 로 표현될 수 있다. (자기가 구현을 어떻게 하느냐에 따라 다름)  
2. 선호도(Preferences) : 각 데이터 포인트의 대표 점이 되기 위한 적합성을 나타내는 지표.  
    * 특정 데이터를 선호하게 할 수 있는 우선 정보가 있을 수 있다. 선호도를 통해 표현할 수 있다.

<br>

유사도와 선호도 모두 단일 대각 행렬(value가 대각선에 있고 나머지는 0)로 표현할 수 있다. 행렬 표현은 dense dataset에 사용하기 좋다. 점 간 연결이 드물 경우, 전체 n x n 행렬을 메모리에 저장하지 않고 연결된 점 간의 유사도 리스트를 유지하는 것이 더 실용적이다. '점 간 메시지 교환'은 행렬을 다루는 것과 동일하다. 이는 관점과 구현의 문제일 뿐이다.


<br>



<br><br>

> <subtitle> References </subtitle>

<br>

* [https://www.toptal.com/machine-learning/clustering-algorithms](https://www.toptal.com/machine-learning/clustering-algorithms){:target="_blank"}
* [https://en.wikipedia.org/wiki/Affinity_propagation](https://en.wikipedia.org/wiki/Affinity_propagation){:target="_blank"}
* [https://datascienceschool.net/03%20machine%20learning/16.05%20Affinity%20Propagation.html](https://datascienceschool.net/03%20machine%20learning/16.05%20Affinity%20Propagation.html){:target="_blank"}
