---
layout: post
title: "(Error solution) Pyenv won't install python 3 on Mac os 11 & Unable to install tkinter with pyenv Pythons on MacOS"
date: 2020-12-23
excerpt: ""
tags : [pyenv, mac os, Big Sur, mac os 11, tkinter]
comments: true
---

본 글은 두 종류의 연결된 문제를 다루는 글이다. 일단 글쓴이는 Mac OS 업데이트를 나오자마자 해버렸다.(Catalina -> Big Sur)  
이렇게 했더니 pyenv로 새로운 버전의 python을 설치할 때 에러가 났다.  
이게 관련이 있는지 별개인지(별개일 확률이 높다) 모르겠지만 tkinter를 사용하려는데 설치가 쉽지 않다.

# How to install python 3 using pyenv on Mac os 11

일단 본 시기에는 이 문제뿐만 아니라 다른 어플리케이션에서도 mac os update로 인한 문제가 있어서 업데이트를 유예하는 것을 권장한다.  
다만 이미 해서 롤백하기가 싫거나 할 수 없는 상황에서 다음과 같은 조금 귀찮은 방법이 있음을 알린다.

다음과 같은 에러메시지가 나온다. 
 
```
$pyenv install 3.9.0

python-build: use readline from homebrew
Downloading Python-3.9.0.tar.xz...
-> https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
Installing Python-3.9.0...
python-build: use readline from homebrew
python-build: use zlib from xcode sdk

BUILD FAILED (OS X 11.0.1 using python-build 20180424)

Inspect or clean up the working tree at /var/folders/n6/q2b78971589bltfczw539flh0000gn/T/python-build.20201114175722.7103
Results logged to /var/folders/n6/q2b78971589bltfczw539flh0000gn/T/python-build.20201114175722.7103.log

Last 10 log lines:
checking for python3... python3
checking for --enable-universalsdk... no
checking for --with-universal-archs... no
checking MACHDEP... "darwin"
checking for gcc... clang
checking whether the C compiler works... no
configure: error: in '/var/folders/n6/q2b78971589bltfczw539flh0000gn/T/python-build.20201114175722.7103/Python-3.9.0':
configure: error: C compiler cannot create executables
See 'config.log' for more details
make: *** No targets specified and no makefile found.  Stop.
```


## Trials (failure examples)

위 문제를 해결하려고 여러 시도를 해보았다. 누군가에게는 통했는지 모르겠지만 일단 나한테는 적용이 안된다.

* software 업데이트  
```
$softwareupdate --all --install --force
```

* xcode 재설치  
```
$sudo rm -rf /Library/Developer/CommandLineTools
$sudo xcode-select --install
```

## My Solution

아래 명령어로 pyenv 내에 설치가 되었다.    

```  
# python 3.6.9 설치
$CFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix bzip2)/include -I$(brew --prefix readline)/include -I$(xcrun --show-sdk-path)/usr/include" LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix bzip2)/lib" pyenv install --patch 3.6.9 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)
```

# How to install tkinter with pyenv on Mac os

pyenv로 python 설치에 문제가 없는 분은 위 내용은 생략해도 무방하다.
tkinter는 python 표준 GUI 툴킷이다.
맥에서 tkinter 설치 문제가 되는 것은 파이썬 공홈에서도 확인할 수 있다.

>If you are using a Python from any current python.org Python installer for macOS (3.9.0+, 3.8.0+, or 3.7.2+), no further action is needed to use IDLE or tkinter. A built-in version of Tcl/Tk 8.6 will be used.  
>If you are using macOS 10.6 or later, the Apple-supplied Tcl/Tk 8.5 has serious bugs that can cause application crashes. If you wish to use IDLE or Tkinter, do not use the Apple-supplied Pythons. Instead, install and use a newer version of Python from python.org or a third-party distributor that supplies or links with a newer version of Tcl/Tk.

암튼 우리는 해결책이 필요하고 찾았으니 공유한다.

1. install tcl-tk

```
# 설치
$brew install tcl-tk
# 확인
$brew info tcl-tk
```

2-1. reinstall python (mac os version <= 10)

```
# 기존에 3.6.9가 있으면 없애기
$pyenv uninstall 3.6.9
$env \
  PATH="$(brew --prefix tcl-tk)/bin:$PATH" \
  LDFLAGS="-L$(brew --prefix tcl-tk)/lib" \
  CPPFLAGS="-I$(brew --prefix tcl-tk)/include" \
  PKG_CONFIG_PATH="$(brew --prefix tcl-tk)/lib/pkgconfig" \
  CFLAGS="-I$(brew --prefix tcl-tk)/include" \
  PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I$(brew --prefix tcl-tk)/include' --with-tcltk-libs='-L$(brew --prefix tcl-tk)/lib -ltcl8.6 -ltk8.6'" \
  pyenv install 3.6.9
```

2-2. reinstall python (mac os version == 11)

이 경우는 pyenv install 이 방식이 작동 안 하므로 위에서 pyenv로 파이썬 설치하는 명령어와 병합하여 사용한다.
사실 이렇게 사용하는게 맞는지 모르겠지만 실행이 되니 세부적인 명령어를 알아보지 않았다.

```
# 기존에 3.6.9가 있으면 없애기
$pyenv uninstall 3.6.9
$env \
  PATH="$(brew --prefix tcl-tk)/bin:$PATH" \
  LDFLAGS="-L$(brew --prefix tcl-tk)/lib" \
  CPPFLAGS="-I$(brew --prefix tcl-tk)/include" \
  PKG_CONFIG_PATH="$(brew --prefix tcl-tk)/lib/pkgconfig" \
  CFLAGS="-I$(brew --prefix tcl-tk)/include" \
  PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I$(brew --prefix tcl-tk)/include' --with-tcltk-libs='-L$(brew --prefix tcl-tk)/lib -ltcl8.6 -ltk8.6'" \
  CFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix bzip2)/include -I$(brew --prefix readline)/include -I$(xcrun --show-sdk-path)/usr/include" LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix bzip2)/lib" pyenv install --patch 3.6.9 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)
```

3. test tkinter

```
$ import tkinter
$ tkinter.TclVersion, tkinter.TkVersion
(8.6, 8.6)
# 아래 명령어 입력하면 간단한 GUI 창이 뜬다.
$ tkinter._test()
```

아래 레퍼런스 블로그를 작성하신 분들에게 감사의 말씀 드리며 끝!!

# References

* [https://stackoverflow.com/questions/60469202/unable-to-install-tkinter-with-pyenv-pythons-on-macos/60469203#60469203](https://stackoverflow.com/questions/60469202/unable-to-install-tkinter-with-pyenv-pythons-on-macos/60469203#60469203){:target="_blank"}
* [https://medium.com/@xogk39/install-tkinter-on-mac-pyenv-f112bd3f4663](https://medium.com/@xogk39/install-tkinter-on-mac-pyenv-f112bd3f4663){:target="_blank"}
* [https://stackoverflow.com/questions/63972113/big-sur-clang-invalid-version-error-due-to-macosx-deployment-target/63972598#63972598](https://stackoverflow.com/questions/63972113/big-sur-clang-invalid-version-error-due-to-macosx-deployment-target/63972598#63972598){:target="_blank"}
* [https://dev.to/kojikanao/install-python-3-8-0-via-pyenv-on-bigsur-4oee](https://dev.to/kojikanao/install-python-3-8-0-via-pyenv-on-bigsur-4oee){:target="_blank"}
