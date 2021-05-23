---
layout: post
title: "Building Machine Learning Pipelines Review"
date: 2021-05-22
excerpt: "Book Review."
tags: [MLOps]
comments: False
use_math: true
---

![image](https://user-images.githubusercontent.com/49096513/119250865-eccdaa00-bbdd-11eb-92ae-52121a43ead2.png){: width="50%" height="50%"}

# Building Machine Learning Pipelines


이번 구글 IO에서 발표됐듯이 Tensorflow Extended ver.1 release가 된 기념으로 열심히 읽었던 책인 “Building Machine Learning Pipelines”을 리뷰합니다.


제목만 봐서는 굉장히 다양한 파이프라인을 다룰 것 같지만, 사실은 TFX에 초점이 맞춰져서 쓰였습니다. 저자가 GDE이기 때문인가 싶기도 하고.. 유튜브에 저자를 검색하면 아주 많은 발표에서 TFX에 대한 사랑이 느껴지기도 합니다(TFX 조아! 이런느낌?). 


구성은 TFX 개별 컴포넌트들에 대한 아주 구체적인 설명이 절반을 훌쩍 넘어가고, Apache Airflow나 Kubeflow의 오케스트레이터를 이용하여 파이프라인을 동작하는 것에 대한 설명이 후반부에 있습니다. 그런데 솔직히 이러한 툴들을 사용해본 적이 없으신 분들께는 많이 어려울 수는 있을 것 같습니다(저도 계속 봤던 ㅠㅠ).


따라서 책에서도 kubeflow에 대해 구체적인 설명이 적힌 서적을 추천하고 있는데, 사실 이것보다는 dudaji의 이명환님께서 낸 책인 “쿠버네티스에서 머신러닝이 처음이라면”이라는 책이 더 좋은 것 같습니다(사실 이것도 쿠버네티스 모르면 힘들다는…ㅠㅠ).


물론 TFX가 좋긴 하지만, 하나의 프레임워크에만 의존하는 것이 그다지 좋은 선택은 아니기에 다른 툴을 함께 적용할 수 있는 파이프라인 구축을 하는 것도 좋을 듯 합니다(그게 KFP!).


그럼에도 불구하고 가장 추천드리고 싶은 컴포넌트는 TFDV(SchemaGen, StatisticsGen, ExampleValidator) 입니다. Andrew ng 교수님께서 말씀해주셨듯이 Data Centric AI가 중요해지는 만큼 데이터 드리프트를 감지할 수 있다는 점에서 효과적이고, 독립적으로 실행 가능한 컴포넌트이기에 더욱 좋은 것 같습니다.


요즘도 종종 찾아보고 있는데, 한국어로 번역판은 안나오더라고요. TFX 1.0이 배포된 만큼 곧 능력자 분들께서 번역해주시지 않을까 기대해 봅니다. ㅎㅎㅎㅎ


다음 책으로는 Data Science on AWS를 읽어보려 합니다~! :)