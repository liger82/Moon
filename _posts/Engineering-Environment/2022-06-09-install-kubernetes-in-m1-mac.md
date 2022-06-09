---
layout: post
title: "Install kubernetes in silicon mac/m1"
date: 2022-06-09
excerpt: ""
categories: [Engineering/Environment]
tags : [kubernetes, install, m1, mac]
comments: true
---

> <subtitle> 설치 환경 </subtitle>

* device : macbook pro (16형, 2021)
* processor: m1
* os : macOS Monterey (12.4)

<br>

> <subtitle> 설치 방법 </subtitle>

설치 방법은 크게 세 가지로 나뉩니다. 

1. curl 사용해서 kubectl 바이너리 설치
1. Homebrew 를 통한 설치
1. Macports 를 통한 설치

<br>

## 1. curl 사용해서 kubectl 바이너리 설치

<br>

### 1.1 최신 릴리즈 버전을 설치
```
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/arm64/kubectl"
```

* 특정 버전을 다운로드하려면 다음 명령어를 사용. 다음 예제는 v1.24.0 기준입니다.

```
curl -LO "https://dl.k8s.io/release/v1.24.0/bin/darwin/arm64/kubectl"
```

<br>

### 1.2 바이너리 검증(optional)

* kubectl 체크섬 파일을 다운로드한다.
```
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/arm64/kubectl.sha256"
```

* 검증
```
echo "$(<kubectl.sha256)  kubectl" | shasum -a 256 --check
```

검증이 성공하면 다음과 같이 나옵니다.

```
kubectl: OK
```

실패하면 이렇게 나온다고 하네요 전 성공해서 모르겠습니다~

```
kubectl: FAILED
shasum: WARNING: 1 computed checksum did NOT match
```

<br>

### 1.3 kubectl 바이너리를 실행 가능하게 한다.

```
chmod +x ./kubectl
```

<br>

### 1.4 kubectl 바이너리를 시스템 PATH 파일 위치로 옮긴다.

```
sudo mv ./kubectl /usr/local/bin/kubectl
sudo chown root: /usr/local/bin/kubectl
```

<br>

### 1.5 설치한 버전 확인

```
kubectl version --client
```

<br>

## 2. Homebrew 를 사용하여 설치

이거 한 줄이에요!

```
brew install kubectl
```

<br>

## 3. Macports 를 사용하여 설치

macports 를 사용하는 것도 꽤 간단하네요

```
sudo port selfupdate
sudo port install kubectl
```

<br>

> <subtitle> kubectl 구성 확인 </subtitle>

클러스터 상태를 가져와서 kubectl이 올바르게 구성되어 있는지 확인하는 명령어

```
kubectl cluster-info
```

아래처럼 URL이 표시되면 kubectl이 클러스터에 접근하도록 올바르게 구성된 것입니다.

```
Kubernetes control plane is running at https://127.0.0.1:50296
CoreDNS is running at https://127.0.0.1:50296/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
```

<br>

혹시 아래와 같은 출력값이 나온다면 kubectl이 올바르게 구성되지 않았거나 쿠버네티스 클러스터가 실행되지 않은 것입니다.

```
The connection to the server <server-name:port> was refused - did you specify the right host or port?
```

저의 경우 Minikube를 설치하고 실행하고 다시 해보니 URL이 나왔습니다.

Minikube는 로컬 머신에 VM을 만들고 하나의 노드로 구성된 간단한 클러스터를 배포하는 가벼운 쿠버네티스 구현체입니다. Minikube 는 리눅스, 맥, 윈도우 시스템에서 구동이 가능합니다.

minikube 설치는 [다음 페이지](https://liger82.github.io/engineering/environment/2022/06/09/install-minikube-in-m1-mac.html){:target="_blank"}에 설명해두었습니다.

<br>

---

> <subtitle> References </subtitle>

* [https://kubernetes.io/ko/docs/tasks/tools/install-kubectl-macos/](https://kubernetes.io/ko/docs/tasks/tools/install-kubectl-macos/){:target="_blank"}