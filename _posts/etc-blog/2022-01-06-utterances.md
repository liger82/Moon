---
layout: post
title: "github.io 에 댓글 기능 달기; utterances"
date: 2022-01-06
excerpt: ""
categories: [etc/Blog]
tags : [utterances, disqus, 댓글 기능, comments]
comments: true
---

> <subtitle> utterances 로 댓글 사용하기! </subtitle>

[이전 포스트](https://liger82.github.io/the%20others/blog/2022/01/05/disqus.html){:target="_blank"}에서는 disqus를 활용해서 github.io 에 댓글 기능을 추가하는 방법에 대해 다뤄봤는데요. 무겁기도 할 뿐더러 basic plan은 광고도 막 붙어서 보기 싫더라고요. 

그래서 오늘은 utterances 를 활용해서 댓글다는 기능에 대해 알아보도록 하겠습니다.

### utterances 특징

* github 로그인 해야 댓글을 달 수 있다. -> 이 블로그는 개발자가 보지 않을까 싶어서 허용가능한 수준이라고 본다.
* github repo에 issue 로 댓글을 관리할 수 있다.
* 댓글이 달릴 시에 슬랙이나 메일이 오도록 설정을 할 수 있다.
* 가벼운 편이다.
* 댓글에 Markdown 을 사용할 수 있다!!

<br>

> <subtitle> utterances 설치해보자! </subtitle>

## 1. github app 에서 utterances 설치

* [https://github.com/apps/utterances](https://github.com/apps/utterances){:target="_blank"}

#### 1. "install" 버튼 클릭

<br><center><img src= "https://liger82.github.io/assets/img/post/20220106-utterances/fig1.png" width="100%"></center><br>

#### 2. Only select repositories > 댓글을 이슈로 관리할 저장소를 선택 > "install" 버튼 클릭 > 비밀번호 입력

<br><center><img src= "https://liger82.github.io/assets/img/post/20220106-utterances/fig2.png" width="100%"></center><br>

#### 3. 아래와 같은 페이지가 나오면 configuration > Repository > repo: 빈칸에 **github 계정명/저장소 이름** 을 입력한다.

<br><center><img src= "https://liger82.github.io/assets/img/post/20220106-utterances/fig3.png" width="100%"></center><br>

#### 4. 다음과 같이 선택했습니다.

<br><center><img src= "https://liger82.github.io/assets/img/post/20220106-utterances/fig4.png" width="90%"></center><br>

#### 5. copy 합니다.

<br><center><img src= "https://liger82.github.io/assets/img/post/20220106-utterances/fig5.png" width="90%"></center><br>


## 2. github io에 적용

이 부분은 각자의 웹 환경에 따라 다르겠지만 기본적으로 _layout/post.html 맨 아래에 넣는 경우가 많습니다. 결국 post 마다 댓글을 달려고 하는 거니깐요.

음 제 블로그는 기존에 disqus 를 사용하는 테마여서 disqus를 주석처리하고 disqus 가 적용된 동일한 형태를 사용하였습니다.

_includes/utterances.html 을 만들고 그 안에 아래처럼 집어넣었습니다.

```html
<div id="utterances_thread"></div>

<script src="https://utteranc.es/client.js"
        repo="liger82/liger82.github.io"
        issue-term="pathname"
        label="comments"
        theme="github-light"
        crossorigin="anonymous"
        async>
</script>
```

그 다음 _layouts/post.html 에서 아래처럼 추가했어요.

page.comments 는 실제 포스트 글에서 comments 라는 플래그를 true로 놓으면 댓글창이 나오도록 한 것입니다.

<br><center><img src= "https://liger82.github.io/assets/img/post/20220106-utterances/fig9.png" width="80%"></center><br>


이제 이렇게 등록하면 아래처럼 나옵니다!

<br><center><img src= "https://liger82.github.io/assets/img/post/20220106-utterances/fig6.png" width="80%"></center><br>

## 3. (유의사항) issue 활성화

utterances가 issue 기반으로 하고 있다 보니 issue 가 비활성화되어 있으면 댓글 등록이 안됩니다.

아래처럼 Issues 탭이 있으면 활성화된 상태인거고 없다면 비활성화되어 있는 것입니다.

<br><center><img src= "https://liger82.github.io/assets/img/post/20220106-utterances/fig7.png" width="90%"></center><br>

#### Issues 비활성화되어 있을 경우

Settings > Options > Features > Issues 체크하기

<br><center><img src= "https://liger82.github.io/assets/img/post/20220106-utterances/fig8.png" width="90%"></center><br>

이상으로 "깃헙 블로그 댓글 기능 추가하기" 였습니다. 감사합니다~


<br>

---

> <subtitle> References </subtitle>

* [https://outstanding1301.github.io/dev/2021/01/07/utterances/](https://outstanding1301.github.io/dev/2021/01/07/utterances/){:target="_blank"}
* [https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/disabling-issues](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/disabling-issues){:target="_blank"}