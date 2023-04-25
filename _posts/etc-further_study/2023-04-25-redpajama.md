---
layout: post
title: "Red Pajama: A project to create leading open-source models"
date: 2023-04-25
excerpt: " "
categories: [etc/further_study]
tags : [deep learning, ML, DL, redpajama, llama, LLM, alpaca, vicuna, koala, gpt-4]
comments: true
---


>blog : [https://www.together.xyz/blog/redpajama](https://www.together.xyz/blog/redpajama){:target="_blank"}  
>github : [https://github.com/togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data){:target="_blank"}  

---

<br>

> <subtitle> 정리 </subtitle>

<b>블로그 글을 번역 및 정리한 글입니다.</b>

<br>

GPT-4와 같은 foundation models은 인공지능 분야에서 빠른 발전을 이루고 있으나, 대부분의 강력한 모델들은 상용화된 모델로 비공개되어 있거나 일부만 공개되어 있습니다. RedPajama 프로젝트는 선도적으로 완전한 오픈 소스 모델을 만들기 위한 프로젝트입니다. 이번에 LLaMA 학습 데이터셋의 1.2조 토큰 복제 작업이 완료되어 기쁩니다.

오늘날 가장 능력있는 foundation models은 상용 API 뒤에 숨겨져 있어 연구, 맞춤화, 민감한 데이터와의 사용에 제한이 있습니다. 오픈 소스 모델은 이러한 제한을 없앨 수 있지만, 오픈과 비공개 모델 사이의 품질 차이가 있었습니다. 그 차이를 줄이는데 최근에 큰 진전이 있었습니다. Stable Diffusion은 DALL-E와 같은 상용 제품의 품질을 견줄 수 있을 뿐 아니라 세계 각지의 커뮤니티로부터 넓은 참여로 인한 놀라운 창의성을 낼 수 있다는 것을 보여 주었습니다. 이와 유사하게 최근 LLaMA, Alpaca, Vicuna, Koala와 같은 준오픈 모델 및 Pythia, OpenChatKit, Open Assistant 및 Dolly와 같은 완전히 오픈 소스 모델을 포함한 대규모 언어 모델을 중심으로 비슷한 움직임이 시작되었습니다.

RedPajama는 재현 가능하고 완전히 오픈 소스인 선도적인 언어 모델을 만드는 노력의 과정입니다. RedPajama는 Together, Ontocord.ai, ETH DS3Lab, Stanford CRFM, Hazy Research 및 MILA Québec AI Institute의 협력으로 이루어졌습니다. RedPajama는 세 가지 주요 구성 요소가 있습니다.  
* 높은 품질과 넓은 커버리지를 갖춘 **사전 학습 데이터**  
* 이 데이터에 대해 규모가 큰 기초 모델을 훈련하는 **베이스 모델**  
* 베이스 모델을 사용 가능하고 안전하게 만드는 **인스트럭션 튜닝 데이터 및 모델**  

**오늘, RedPajama의 첫 번째 구성 요소인 사전 훈련 데이터를 공개합니다.**

RedPajama의 시작점은 LLaMA입니다. LLaMA는 두 가지 이유로 최고의 오픈 베이스 모델 스위트입니다. 

첫째, LLaMA는 매우 큰(1,200억개의 토큰) 데이터셋으로 훈련되었으며, 품질을 신중하게 필터링했습니다.  
둘째, 70억 개의 매개변수를 가진 LLaMA 모델은 Chincilla-optimal 지점을 훨씬 넘어서 훈련되어 이 모델 크기에서 최고의 품질을 보장합니다. 70억 개의 매개변수 모델은 일반 소비자급 GPU를 비롯한 다양한 GPU에서 실행할 수 있어 오픈 커뮤니티에게 특히 유용합니다. 그러나 LLaMA 및 그 파생 모델(Alpaca, Vicuna, Koala 등)은 비상업적인 연구 목적으로만 사용할 수 있습니다. 우리는 LLaMA의 완전한 오픈 소스 재현을 만들어 상용 응용 프로그램에서도 사용 가능하게 하고 연구에 대한 더 투명한 파이프 라인을 제공하기 위한 목적으로 RedPajama를 만들기 위해 노력합니다.

RedPajama의 베이스 데이터셋은 1.2조 토큰으로 이루어진 완전히 오픈 소스 데이터셋이며, LLaMA 논문에서 제시한 레시피를 따라 생성되었습니다. RedPajama-Data-1T는 다음과 같이 7개의 데이터 슬라이스로 구성됩니다.

* CommonCrawl: CCNet 파이프라인을 사용하여 처리된 CommonCrawl의 다섯 개의 덤프로, 위키피디아와 비슷한 페이지를 선택하는 선형 분류기를 포함한 여러 품질 필터를 거친 것
* C4: 표준 C4 데이터셋
* GitHub: 라이선스 및 품질로 필터링 된 GitHub 데이터
* arXiv: 보일러 플레이트를 제거한 과학 기사
* Books: 콘텐츠 유사성에 따라 중복을 제거한 공개 도서 말뭉치
* Wikipedia: 보일러 플레이트를 제거한 위키피디아의 하위 집합
* StackExchange: 보일러 플레이트를 제거한 인기 웹사이트의 하위 집합

<table>
    <tr>
        <th></th>
        <th>RedPajama</th>
        <th>LLaMA</th>
    </tr>
    <tr>
        <td>CommonCrawl</td>
        <td>878 billion</td>
        <td>852 billion</td>
    </tr>
    <tr>
        <td>C4</td>
        <td>175 billion</td>
        <td>190 billion</td>
    </tr>
    <tr>
        <td>Github</td>
        <td>59 billion</td>
        <td>100 billion</td>
    </tr>
    <tr>
        <td>Books</td>
        <td>26 billion</td>
        <td>25 billion</td>
    </tr>
    <tr>
        <td>ArXiv</td>
        <td>28 billion</td>
        <td>33 billion</td>
    </tr>
    <tr>
        <td>Wikipedia</td>
        <td>24 billion</td>
        <td>25 billion</td>
    </tr>
    <tr>
        <td>StackExchange</td>
        <td>20 billion</td>
        <td>27 billion</td>
    </tr>
    <tr>
        <td>Total</td>
        <td>1.2 trillion</td>
        <td>1.25 trillion</td>
    </tr>
</table>


각 데이터 슬라이스마다 신중한 데이터 전처리와 필터링을 수행하고, Meta AI의 LLaMA 논문에서 보고한 토큰 수와 거의 일치하도록 품질 필터를 조정합니다. 이를 위해 우리는 모든 데이터 전처리 및 품질 필터를 Github에서 공개적으로 제공하고 있습니다. 누구든지 이 데이터 준비 레시피를 따라하고 RedPajama-Data-1T를 재현할 수 있습니다.

RedPajama의 전체 1.2조 토큰 데이터셋과 더 작은, 더 쉽게 소비 가능한 무작위 샘플은 Hugging Face를 통해 다운로드할 수 있습니다. 전체 데이터셋은 압축 해제하면 디스크에 약 5TB이며, 압축하여 다운로드하면 약 3TB입니다.

RedPajama-Data-1T는 RedPajama 프로젝트의 첫 번째 구성 요소입니다. 이제 우리는 베이스 모델과 인스트럭션 튜닝 데이터와 모델을 개발하고 향후 공개할 것입니다. RedPajama 프로젝트는 전체적으로 AI의 투명성과 신뢰성을 향상시키고, AI 기술이 모든 사람들의 생활에 영향을 미치는 방식을 개선하는 데 기여할 것입니다.

<br>

> <subtitle> 다음 행보: 모델, 지침 및 OpenChatKit </subtitle>

사전 훈련 데이터를 재현한 다음 단계는 강력한 기본 모델을 학습시키는 것입니다. Oak Ridge Leadership Computing Facility(OLCF)의 지원을 받아 INCITE 프로그램의 일환으로 전체 모델을 훈련하고 있으며, 몇 주 내에 첫 번째 모델을 사용할 수 있게 될 것입니다.

강력한 기본 모델을 손에 넣었으니 이제 모델에 대한 인스트럭션 튜닝을 진행할 수 있게 되어 기대가 큽니다. 알파카는 인스트럭션 튜닝의 힘을 보여줬는데, 단 5만 개의 고품질의 다양한 인스트럭션만으로 극적으로 향상된 기능을 구현할 수 있었습니다. 오픈챗킷을 통해 수십만 개의 고품질의 자연스러운 사용자 인스트럭션을 받았으며, 이 인스트럭션은 인스트럭션 튜닝 버전의 레드파자마 모델을 출시하는 데 사용될 예정입니다.

<br>

> <subtitle> References </subtitle>

* [https://www.together.xyz/blog/redpajama](https://www.together.xyz/blog/redpajama){:target="_blank"}  
* [https://github.com/togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data){:target="_blank"}  

<br>