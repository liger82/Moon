---
layout: post
title: "[Import Error] gym render error; Error occurred while running `from pyglet.gl import *`"
date: 2021-05-12
excerpt: ""
categories: [RL/RL]
tags : [Reinforcement Learning, RL, OpenAI Gym, Gym, render, pyglet, OpenGL, GL, error]
comments: true
---

> <subtitle> Error </subtitle>

맥에서 gym 설치만 하면 모든 기능을 쓸 수 있을 줄 알았는데 다음과 같은 에러가 났습니다.

```python

Traceback (most recent call last):
  File "/Users/stuartkim/.pyenv/versions/3.7.7/envs/deepRL/lib/python3.7/site-packages/gym/envs/classic_control/rendering.py", line 25, in <module>
    from pyglet.gl import *
  File "/Users/stuartkim/.pyenv/versions/3.7.7/envs/deepRL/lib/python3.7/site-packages/pyglet/gl/__init__.py", line 101, in <module>
    from pyglet.gl.lib import GLException
  File "/Users/stuartkim/.pyenv/versions/3.7.7/envs/deepRL/lib/python3.7/site-packages/pyglet/gl/lib.py", line 142, in <module>
    from pyglet.gl.lib_agl import link_GL, link_GLU, link_AGL
  File "/Users/stuartkim/.pyenv/versions/3.7.7/envs/deepRL/lib/python3.7/site-packages/pyglet/gl/lib_agl.py", line 50, in <module>
    framework='/System/Library/Frameworks/OpenGL.framework')
  File "/Users/stuartkim/.pyenv/versions/3.7.7/envs/deepRL/lib/python3.7/site-packages/pyglet/lib.py", line 129, in load_library
    return self.load_framework(kwargs['framework'])
  File "/Users/stuartkim/.pyenv/versions/3.7.7/envs/deepRL/lib/python3.7/site-packages/pyglet/lib.py", line 284, in load_framework
    raise ImportError("Can't find framework %s." % path)
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/stuartkim/projects/study/reinforcementLearning/Deep-Reinforcement-Learning-Hands-On-Second-Edition/Chapter02/02_cartpole_random.py", line 13, in <module>
    env.render()
  File "/Users/stuartkim/.pyenv/versions/3.7.7/envs/deepRL/lib/python3.7/site-packages/gym/core.py", line 240, in render
    return self.env.render(mode, **kwargs)
  File "/Users/stuartkim/.pyenv/versions/3.7.7/envs/deepRL/lib/python3.7/site-packages/gym/envs/classic_control/cartpole.py", line 174, in render
    from gym.envs.classic_control import rendering
  File "/Users/stuartkim/.pyenv/versions/3.7.7/envs/deepRL/lib/python3.7/site-packages/gym/envs/classic_control/rendering.py", line 32, in <module>
    ''')
ImportError: 
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s "-screen 0 1400x900x24" python <your_script.py>'
    
```

GL, python-opengl 두 개를 pip install로 설치해보았지만 동일한 문제가 발생했습니다.

동일한 에러가 나는 문제를 다룬 깃헙 이슈에서 해결책을 찾았습니다. 

```
$ pip install pyglet==1.5.11
```

pyglet을 1.5.11 버전으로 설치하니 다음과 같은 충돌이 발생했습니다.

```
$ pip install pyglet==1.5.11
Collecting pyglet==1.5.11
  Using cached pyglet-1.5.11-py3-none-any.whl (1.1 MB)
Installing collected packages: pyglet
  Attempting uninstall: pyglet
    Found existing installation: pyglet 1.4.11
    Uninstalling pyglet-1.4.11:
      Successfully uninstalled pyglet-1.4.11
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you have pyglet 1.5.11 which is incompatible.
Successfully installed pyglet-1.5.11
```

제가 사용하는 gym이 pyglet을 1.5.0보다 아래 버전을 써야 한다는 것입니다. 

해결책?? 없습니다.

그냥 저대로 쓰면 됩니다. 성공적으로 나옵니다. 아래 캡쳐본이 성공적으로 출력된 내용입니다.


<br><center><img src= "https://liger82.github.io/assets/img/post/20210512-error-pyglet/cartpole.png" width="60%"></center><br>



---

> <subtitle> References </subtitle>

* [https://github.com/openai/gym/issues/2101](https://github.com/openai/gym/issues/2101){:target="_blank"}