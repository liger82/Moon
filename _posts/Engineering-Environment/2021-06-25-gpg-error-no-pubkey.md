---
layout: post
title: "GPG error : NO_PUBKEY"
date: 2021-06-25
excerpt: ""
categories: [Engineering/Environment]
tags : [apt-get update, gpg error, no_pubkey, googlecloudsdk]
comments: true
---

> <subtitle> 1. 현재 환경 </subtitle>

* os : Ubuntu 18.04

<br>

> <subtitle> 2. 문제 상황 </subtitle>

```
ubuntu@super:~$ sudo apt-get update

Hit:1 https://download.docker.com/linux/ubuntu bionic InRelease
Hit:2 https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64  InRelease
Ign:3 http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
Hit:4 https://nvidia.github.io/nvidia-container-runtime/stable/ubuntu18.04/amd64  InRelease
Hit:5 http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release
Hit:6 https://nvidia.github.io/nvidia-docker/ubuntu18.04/amd64  InRelease
Ign:7 http://dl.google.com/linux/chrome-remote-desktop/deb stable InRelease
Hit:8 http://dl.google.com/linux/chrome/deb stable InRelease
Hit:9 http://dl.google.com/linux/chrome-remote-desktop/deb stable Release
Hit:10 http://kr.archive.ubuntu.com/ubuntu bionic InRelease
Hit:11 http://security.ubuntu.com/ubuntu bionic-security InRelease
Hit:12 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
Hit:14 http://kr.archive.ubuntu.com/ubuntu bionic-updates InRelease
Get:15 https://packages.cloud.google.com/apt cloud-sdk InRelease [6,739 B]
Hit:16 http://kr.archive.ubuntu.com/ubuntu bionic-backports InRelease
Err:15 https://packages.cloud.google.com/apt cloud-sdk InRelease
  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY FEEA9169307EA071 NO_PUBKEY 8B57C5C2836F4BEB
Fetched 6,739 B in 2s (2,856 B/s)
Reading package lists... Done
W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error: https://packages.cloud.google.com/apt cloud-sdk InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY FEEA9169307EA071 NO_PUBKEY 8B57C5C2836F4BEB
W: Failed to fetch https://packages.cloud.google.com/apt/dists/cloud-sdk/InRelease  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY FEEA9169307EA071 NO_PUBKEY 8B57C5C2836F4BEB
W: Some index files failed to download. They have been ignored, or old ones used instead.
```

<br>

처음에 누군가(구글 서포터 포함)는 아래 명령어로 해결이 된다고 하는데 저는 작동하지 않았습니다.

```
ubuntu@super:~$ sudo curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
```

<br>

> <subtitle> 3. 해결책 </subtitle>

google cloud 공개 키를 가져오는 것이 답인데 위와 미묘하게 다릅니다. 저는 아래 명령어로 공개키 등록 후 apt-get update가 작동하는 것을 확인하였습니다.

```
ubuntu@super:~$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  2537  100  2537    0     0   6054      0 --:--:-- --:--:-- --:--:--  6040
OK
```

<br>

저는 위 명령어로 공개키 등록 후 apt-get update가 작동하는 것을 확인하였습니다.

```
ubuntu@super:~$ sudo apt-get update

Hit:1 https://download.docker.com/linux/ubuntu bionic InRelease
Ign:2 http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
Hit:3 http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release
Hit:4 https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64  InRelease
Ign:5 http://dl.google.com/linux/chrome-remote-desktop/deb stable InRelease
Hit:6 https://nvidia.github.io/nvidia-container-runtime/stable/ubuntu18.04/amd64  InRelease
Hit:7 https://nvidia.github.io/nvidia-docker/ubuntu18.04/amd64  InRelease
Hit:9 http://dl.google.com/linux/chrome/deb stable InRelease
Hit:10 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
Hit:11 http://security.ubuntu.com/ubuntu bionic-security InRelease
Hit:12 http://kr.archive.ubuntu.com/ubuntu bionic InRelease
Get:13 https://packages.cloud.google.com/apt cloud-sdk InRelease [6,739 B]
Hit:14 http://dl.google.com/linux/chrome-remote-desktop/deb stable Release
Hit:15 http://kr.archive.ubuntu.com/ubuntu bionic-updates InRelease
Hit:16 http://kr.archive.ubuntu.com/ubuntu bionic-backports InRelease
Get:18 https://packages.cloud.google.com/apt cloud-sdk/main i386 Packages [142 kB]
Get:19 https://packages.cloud.google.com/apt cloud-sdk/main amd64 Packages [171 kB]
Fetched 320 kB in 2s (135 kB/s)
Reading package lists... Done
```

<br>

별거 아닌데 저는 이 작은 차이로 계속 헤매고 다녀서 혹시나 다른 분들도 그럴까봐 남깁니다. 감사합니다.

<br>

---

> <subtitle> References </subtitle>

* [https://groups.google.com/g/gce-discussion/c/zeGb4gdK2Iw?pli=1](https://groups.google.com/g/gce-discussion/c/zeGb4gdK2Iw?pli=1){:target="_blank"}
* [https://cloud.google.com/sdk/docs/install#deb](https://cloud.google.com/sdk/docs/install#deb){:target="_blank"}