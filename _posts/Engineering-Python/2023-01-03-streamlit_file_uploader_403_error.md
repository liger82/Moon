---
layout: post
title: "Streamlit file_uploader: Error: Request failded with status code 403"
date: 2023-01-03
excerpt: ""
categories: [Engineering/Python]
tags : [python, streamlit, file_uploader, 403, error, request]
comments: true
---

> <subtitle> Issue </subtitle>


streamlit 으로 간단히 데모 서비스를 만들던 중에 파일 업로더가 말썽이었습니다. 에러는 다음과 같았습니다.

<br><center><img src= "https://liger82.github.io/assets/img/post/20230103-streamlit_file_uploader_403_error/error_capture.png" width="70%"></center><br>

문제는 로컬에서 테스트할 때는 잘 되던게 쿠버네티스에 올려서 테스트하면 안되는 것이었습니다.

<br>

> <subtitle> Solution </subtitle>

해결법은 레퍼런스 사이트에서 나온대로 "--server.enableCORS=false --server.enableXsrfProtection=false" 를 streamlit 구동 시 같이 해주는 것입니다.

```bash
streamlit run main.py --server.enableCORS=false --server.enableXsrfProtection=false
```

<br>

> <subtitle> Why </subtitle>

단순히 로컬이냐 아니냐의 문제보다는 방화벽 이슈였습니다. 제가 올리던 곳이 회사의 사내망이라서 방화벽 이슈가 있었습니다. 실제로 공개망에서 올렸을 때는 문제없이 구동되었습니다.

추가로 streamlit으로 실제 서비스를 하려고 할 때는 위와 같은 방식이 아니라 보안팀과 이를 논의하는 것이 어떨까 싶습니다.

<br>

---

> <subtitle> References </subtitle>

* [https://discuss.streamlit.io/t/file-upload-fails-with-error-request-failed-with-status-code-403/27143](https://discuss.streamlit.io/t/file-upload-fails-with-error-request-failed-with-status-code-403/27143){:target="_blank"}
