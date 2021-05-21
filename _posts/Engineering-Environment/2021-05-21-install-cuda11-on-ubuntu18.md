---
layout: post
title: "Install Nvidia driver, CUDA 11.0 and cuDNN 8.0.5 on Ubuntu 18.04(GPU=RTX 3090)"
date: 2021-05-21
excerpt: ""
categories: [Engineering/Environment]
tags : [install, cuda, cuda 11.0, cuDNN, cuDNN 8.0.5, ubuntu 18.04, rtx 3090, 3090ti]
comments: true
---

> <subtitle> 1. 현재 환경과 설치 대상 </subtitle>

* gpu : RTX 3090
* os : Ubuntu 18.04

3090의 경우 cuda 11.x 부터 가능하고 torch나 tensorflow도 버전에 제한이 있습니다. 
torch는 1.7.0 이상부터 가능하고, tensorflow의 경우 실행 자체는 2.x 대도 가능하나 제대로 gpu를 쓰지 못합니다.
금일 기준으로는 그렇습니다. 아래 블로그에서 정리한 것이 있는데 안정적인 지원은 아직인가봅니다.

[https://koos808.tistory.com/41](https://koos808.tistory.com/41){:target="_blank"}

암튼 여기서는 nvidia-driver, CUDA 11.0, cuDNN 8.0.5 설치에 대해 다루도록 하겠습니다.

<br>

> <subtitle> 2. nouveau 제거 </subtitle>

nouveau 대신에 nvidia-driver를 사용할 것이기 때문에 nouveau는 제거합니다.

A. 설치 확인 -> 없으면 skip, 3 세션으로 넘어감  
```
$ lsmod | grep nouveau
```

B. 편집기로 아래 파일을 연다.  
```
$ sudo vim /etc/modprobe.d/blacklist-nouveau.conf  
```

C. 아래 두 줄을 입력하고 저장한다.
```
blacklist nouveau
options nouveau modset=0
```

D. kernel initramfs 업데이트
```
$ sudo update-initramfs -u
$ sudo service gdm stop
```

E. 제거 확인. 아무 것도 안 나와야 함. 혹시 나오면 reboot도 해볼 것.
```
$ lsmod | grep nouveau
```

<br>

> <subtitle> nvidia driver 설치 </subtitle>

A. install gcc and make  
```
$ sudo apt-get install gcc
$ sudo apt-get install make
```

B. update kernel initramfs  
```
$ sudo update-initramfs -u
```

C. install driver
```
$ sudo apt update
$ sudo ubuntu-drivers autoinstall
```

* 제 서버에서는 nvidia-driver-465 가 설치되었습니다.

* autoinstall이 안 먹을때는 아래 명령어로 설치가능한 목록 보고 직접 설치
```
$ ubuntu-drivers devices
$ sudo apt-get install nvidia-driver-[버전]
```

D. reboot
```
$ sudo reboot
````

E. 검증
```
$ nvidia-smi
```

<br><center><img src= "https://liger82.github.io/assets/img/post/20210521-install-cuda11-on-ubuntu18/nvidia-smi.png" width="70%"></center><br>


<br>

> <subtitle> CUDA 11.0 설치 </subtitle>

A. Install CUDA 11.0

환경과 원하는 설치 방식에 맞게 명령어 혹은 파일을 다음 웹사이트에서 제공합니다.  
[https://developer.nvidia.com](https://developer.nvidia.com/cuda-11.0-update1-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork){:target="_blank"}

아래 명령어는 다음 선택지의 결과로 나온 것입니다.  
* operating system : Linux
* Architecture : x86_64
* Distribution : Ubuntu
* Version : 18.04
* Installer Type : deb(network) 

```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
$ sudo apt-get update
$ sudo apt-get -y install cuda-11-0
```

B. path 등록

* bash_profile 이거나 zshrc 인 경우 알아서 바꿔서 열기
```
sudo vim ~/.bashrc
```

* 맨 아래에 다음 두 줄을 입력하고 저장
    - /usr/local/에 cuda-[version number]가 cuda라는 디렉토리에 링크가 있을 때에만 가능하고 링크가 없을 경우 cuda 대신에 cuda-[version number]를 바꿔준다.
```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```  

아래 명령어 실행(bashrc가 아닌 경우 알아서 바꿔서 실행)
source ~/.bashrc

C. 검증  
```
$ nvcc -V

nvcc: NVIDIA (R) Cuda compiler driver     
Copyright (c) 2005-2020 NVIDIA Corporation 
Built on Wed_Jul_22_19:09:09_PDT_2020                           
Cuda compilation tools, release 11.0, V11.0.221
Build cuda_11.0_bu.TC445_37.28845127_0
```

* nvidia-smi로 나온 cuda version은 실제와 다를 수 있습니다.


<br>

> <subtitle> cuDNN 8.0.5 설치 </subtitle>

A. cuDNN 파일 다운로드  
* 아래 사이트에서 로그인  
[https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download){:target="_blank"}

* CUDA 11.0을 위한 cuDNN 8.0.5 파일 중 다음 파일을 다운로드
    > cuDNN Library for Linux(x86)


B. 다운로드된 파일 압축 풀기(다른 이름일 경우 바꿀 것.)  
```
$ sudo tar -xvf cudnn-11.0-linux-x64-v8.0.5.39.tar
```

C. 파일을 CUDA toolkit 디렉토리로 옮기고 권한 부여
```
$ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

D. 설치 검증
```
$ cat /usr/local/cuda-11.0/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

#define CUDNN_MAJOR 8
#define CUDNN_MINOR 0
#define CUDNN_PATCHLEVEL 5
--
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#endif /* CUDNN_VERSION_H */
````

<br>

> <subtitle> 요약 </subtitle>

다음 환경에서 nvidia-driver, cuda, cudnn을 설치하고 검증해보았습니다.  
* RTX 3090
* ubuntu 18.04

<br>

---

> <subtitle> References </subtitle>

* [https://kyumdoctor.co.kr/30](https://kyumdoctor.co.kr/30){:target="_blank"}
* [https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download){:target="_blank"}