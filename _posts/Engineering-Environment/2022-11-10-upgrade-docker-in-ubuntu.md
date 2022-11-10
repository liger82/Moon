---
layout: post
title: "Upgrade docker version in ubuntu"
date: 2022-11-10
excerpt: ""
categories: [Engineering/Environment]
tags : [ubuntu, docker version, upgrade, update]
comments: true
---

Ubuntu os에서 docker version 을 업그레이드하는 방법에 대해 포스트하도록 하겠습니다.

> <subtitle> 설치 환경 </subtitle>

* device : tesla p40 * 4
* os : ubuntu 18.04

<br>

> <subtitle> 업그레이드 방법 </subtitle>

```bash
$ sudo apt-get update

...(생략)...

$ curl -sSL https://get.docker.com/ | sudo bash

Executing docker install script, commit: 4f282167c425347a931ccfd95cc91fab041d414f

Warning: the "docker" command appears to already exist on this system.

If you already have Docker installed, this script can cause trouble, which is
why we're displaying this warning and provide the opportunity to cancel the
installation.

If you installed the current Docker package using this script and are using it
again to update Docker, you can safely ignore this message.

You may press Ctrl+C now to abort this script.
+ sleep 20
+ sh -c 'apt-get update -qq >/dev/null'
+ sh -c 'DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apt-transport-https ca-certificates curl >/dev/null'
+ sh -c 'mkdir -p /etc/apt/keyrings && chmod -R 0755 /etc/apt/keyrings'
+ sh -c 'curl -fsSL "https://download.docker.com/linux/ubuntu/gpg" | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg'
gpg: WARNING: unsafe ownership on homedir '/home/seokkkim/.gnupg'
+ sh -c 'chmod a+r /etc/apt/keyrings/docker.gpg'
+ sh -c 'echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu bionic stable" > /etc/apt/sources.list.d/docker.list'
+ sh -c 'apt-get update -qq >/dev/null'
W: Target Packages (stable/binary-amd64/Packages) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target Packages (stable/binary-all/Packages) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target Translations (stable/i18n/Translation-en_US) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target Translations (stable/i18n/Translation-en) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target CNF (stable/cnf/Commands-amd64) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target CNF (stable/cnf/Commands-all) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target Packages (stable/binary-amd64/Packages) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target Packages (stable/binary-all/Packages) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target Translations (stable/i18n/Translation-en_US) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target Translations (stable/i18n/Translation-en) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target CNF (stable/cnf/Commands-amd64) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
W: Target CNF (stable/cnf/Commands-all) is configured multiple times in /etc/apt/sources.list:55 and /etc/apt/sources.list.d/docker.list:1
+ sh -c 'DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-scan-plugin >/dev/null'
+ version_gte 20.10
+ '[' -z '' ']'
+ return 0
+ sh -c 'DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-ce-rootless-extras >/dev/null'
+ sh -c 'docker version'
Client: Docker Engine - Community
Version: 20.10.21
API version: 1.41
Go version: go1.18.7
Git commit: baeda1f
Built: Tue Oct 25 18:02:00 2022
OS/Arch: linux/amd64
Context: default
Experimental: true

Server: Docker Engine - Community
Engine:
Version: 20.10.21
API version: 1.41 (minimum version 1.12)
Go version: go1.18.7
Git commit: 3056208
Built: Tue Oct 25 17:59:53 2022
OS/Arch: linux/amd64
Experimental: false
containerd:
Version: 1.6.9
GitCommit: 1c90a442489720eec95342e1789ee8a5e1b9536f
runc:
Version: 1.1.4
GitCommit: v1.1.4-0-g5fd4c4d
docker-init:
Version: 0.19.0
GitCommit: de40ad0

================================================================================

To run Docker as a non-privileged user, consider setting up the
Docker daemon in rootless mode for your user:

dockerd-rootless-setuptool.sh install

Visit https://docs.docker.com/go/rootless/ to learn about rootless mode.


To run the Docker daemon as a fully privileged service, but granting non-root
users access, refer to https://docs.docker.com/go/daemon-access/

WARNING: Access to the remote API on a privileged Docker daemon is equivalent
to root access on the host. Refer to the 'Docker daemon attack surface'
documentation for details: https://docs.docker.com/go/attack-surface/

================================================================================
```

<br>

---

> <subtitle> References </subtitle>

* [https://askubuntu.com/questions/472412/how-do-i-upgrade-docker](https://askubuntu.com/questions/472412/how-do-i-upgrade-docker){:target="_blank"}