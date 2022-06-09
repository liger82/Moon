---
layout: post
title: "Install minikube in silicon mac/m1"
date: 2022-06-09
excerpt: ""
categories: [Engineering/Environment]
tags : [kubernetes, install, m1, mac, minikube, silicon mac, arm]
comments: true
---

> <subtitle> 설치 환경 </subtitle>

* device : macbook pro (16형, 2021)
* processor: m1
* os : macOS Monterey (12.4)

<br>

> <subtitle> 설치 방법 </subtitle>



### 최신 릴리즈 버전을 다운로드
```
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-darwin-arm64
```

### minikube 설치
```
sudo install minikube-darwin-arm64 /usr/local/bin/minikube
```

### minikube 실행

* 도커 기반이라 도커를 키고 다음 명령어를 실행해주세요

```
minikube start --driver=docker --alsologtostderr
```

결과는 엄청 많이 뜨다가 최종적으로 아래처럼 뜹니다.

<center><img src= "https://liger82.github.io/assets/img/post/20220609-install-minikube-in-mac/minikube1.png" width="100%"></center><br>

현재 떠 있는 docker container를 살펴보면 minikube가 떠 있는 것을 확인할 수 있습니다.

```
docker ps
CONTAINER ID   IMAGE                                 COMMAND                  CREATED          STATUS          PORTS                                                                                                                                  NAMES
debb44186fe3   gcr.io/k8s-minikube/kicbase:v0.0.30   "/usr/local/bin/entr…"   45 seconds ago   Up 44 seconds   127.0.0.1:50292->22/tcp, 127.0.0.1:50293->2376/tcp, 127.0.0.1:50295->5000/tcp, 127.0.0.1:50296->8443/tcp, 127.0.0.1:50294->32443/tcp   minikube
```

<br>

---

> <subtitle> References </subtitle>

* [https://kubernetes.io/ko/docs/tasks/tools/install-kubectl-macos/](https://kubernetes.io/ko/docs/tasks/tools/install-kubectl-macos/){:target="_blank"}