---
layout: post
title: "Kubernetes의 job을 이용한 ML학습"
date: 2021-03-31
excerpt: "k8s에서 ML training을 job을 이용하여 진행하는 것을 배워봅니다."
tags: [MLOps]
comments: False
use_math: true
---

## 1. ML training 효율화의 중요성
많은 사람들이 최근에 Machine Learning에 관심을 갖고 열정적으로 공부하고 있음을 부쩍 느끼고 있습니다. 그러나, 대부분의 경우에는 ML Code에 막대한 시간을 투자하여 노력을 기울이죠.

그러나, Andrew Ng 교수님께서 지난 [MLOps Seminar](https://www.youtube.com/watch?v=06-AZXmwHjo&t=980s)에서 말하셨듯, 대개의 머신러닝 task에서 성능향상은 모델의 발전 보다도 잘 정제된 많은 양의 데이터로부터 손쉽게 얻을 수 있다고 믿어집니다.

그럼에도 불구하고 잘 만들어진 데이터는 우리가 쉽게 구할 수 있는 것이 아닙니다. 결국에는 AI 비즈니스에서 막대한 금액이 투자되어야 하는 부분은 이러한 부분들이고, 따라서 다른방향의 성능향상을 노려볼 필요가 있습니다.

바로 HPO(HyperParameter Optimization)입니다. Machine Learning 모델에서는 다양한 parameter와 hyperparameter가 존재하는데, parameter 같은 경우에는 모델 내부의 학습(의 과정에서 알아서 조정되는)이 되지만, hyperparameter는 opimizer, loss function, hidden layer 등 수동으로 조정해야 하는 요소들을 의미합니다. 물론 grid search를 비롯한 다양한 방법론들이 존재하지만, 결국엔 많은 학습을 통해 귀납적으로 밝혀내야만 합니다.

실제로 수많은 기업들은 컴퓨팅 리소스를 활용하여 일주일에 많게는 수만개의 학습을 진행한다고 합니다. 단순히 컴퓨팅 자원뿐만 아니라, 이를 가능하게 하기 위해서는 효율적인 학습 시스템이 만들어져야 합니다.

특히 머신러닝의 경우에는 자원이 많이 들어가는 부분이 딱 훈련 or 배포에 한정적으로 정해져 있죠. V100, A100과 같은 GPU가 한 대에 수천만원을 호가하는 것을 본다면, 유휴 자원은 곧 비즈니스에서 손해로 직결될 수밖에 없습니다.

더욱이 성능 강화를 위해서 노력하는 중이라면, 단위 시간당 훈련의 숫자를 최대한으로 늘릴 필요가 있죠.
실제로도 대부분의 AI 비즈니스를 진행하는 기업들의 개발팀 능력은 실험 횟수에서 비롯된다고 하기도 합니다.

이를 위해서 최근에는 ML Researcher 뿐만 아니라, ML Engineer, MLOps 등 다양한 직군에서의 소프트웨어 성향의 엔지니어 채용 공고 또한 지속적으로 나오고 있는 상황입니다.

따라서 이번 포스트에서는 Kubernetes를 이용하여 효과적인 병렬학습을 하는 방법을 배워볼 예정입니다.

## 2. 환경 세팅
사실 쿠버네티스가 다수의 서버를 효율적으로 운용하기 위해서 등장하였고, 이러한 목적을 달성하기 위해서 관념적으로 지향하는 것은 맞습니다!! 그러나, 이번 포스트에서는 사실 다수의 서버를 운용할 환경이나, 여건을 갖추기 쉽지 않아서 minikube를 이용하여 단일 서버에서 실습을 진행해 볼 예정입니다.

그래도 항상 scale-up에 대한 고려는 해두어야 합니다!!

#### GCP에서 인스턴스 만들기
해당 실습은 GCP의 무료 크레딧으로 진행하실 수 있습니다!! 우선 GCP에 가입을 하여 300달러 무료 크레딧을 받은 후에 아래 그림과 같이 Compute - Compute Engine - VM 인스턴스를 선택해 줍니다.
![image](https://user-images.githubusercontent.com/49096513/113117391-574f1300-9249-11eb-9397-d8041e9ab8da.png){: width="40%" height="40%"}


그리고 새 인스턴스 만들기에서 이름을 편한대로 변경해주시고, 다음의 3가지 항목을 변경해주시면 됩니다.

1. 머신 구성 변경
![image](https://user-images.githubusercontent.com/49096513/113117417-5fa74e00-9249-11eb-92c6-d3b5674dc759.png){: width="40%" height="40%"}

2. OS 및 디스크 용량 변경
![image](https://user-images.githubusercontent.com/49096513/113117435-63d36b80-9249-11eb-8e32-af69690ee84e.png){: width="40%" height="40%"}

3. 방화벽 변경
![image](https://user-images.githubusercontent.com/49096513/113117455-6766f280-9249-11eb-9a5c-ee0d73fb446b.png){: width="40%" height="40%"}

모두 변경을 완료하였으면, 인스턴스를 생성하고 SSH 쉘을 클릭하여 터미널을 열어주시면 됩니다.

#### Minikube 설치하기
이제 Docker, kubectl, minikube를 설치할 차례입니다. Docker는 컨테이너를 다루기 위해 필요한 오픈소스로 활용되며, kubectl은 kubernetes를 컨트롤하기 위한 cli 툴이라고 생각하시면 됩니다. 마지막으로 minikube는 단일노드에서 운용하기 위해 만들어진 것입니다.

아래 코드를 이용하여 Docker, kubectl, minikube를 차례대로 설치해주시면 됩니다(복붙!!).

1. docker

~~~sh
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get updates
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
~~~

2. kubectl

~~~sh
sudo apt-get update && sudo apt-get install -y apt-transport-https gnupg2 curl

curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl
~~~

3. minikube

~~~sh
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
sudo usermod -aG docker $USER && newgrp docker
minikube start --cpus 6 --memory 12288 --disk-size=120g --extra-config=apiserver.service-account-issuer=api --extra-config=apiserver.service-account-signing-key-file=/var/lib/minikube/certs/apiserver.key --extra-config=apiserver.service-account-api-audiences=api
~~~

아래 명령어를 통해서 제대로 작동을 하는지 확인해보세요!!

~~~sh
minikube status
~~~

## 3. Job 생성하기
이제 모든 환경 세팅이 완료되었으면, 실제 훈련을 쿠버네티스에 생성해보는 작업을 진행해봅시다!!

실질적인 실습 부분에 해당하고, job을 생성하기 위해서는 다음과 같은 프로세스를 거치게 됩니다. 

1. python train script 작성 
2. Dockerfile 작성 
3. Image build 
4. k8s job yaml 파일 작성 
5. job 생성

그럼 이제 하나하나 진행해보도록 합시다.

#### python train script 작성
파이썬 훈련 스크립트는 아래와 같이 작성하였습니다. Tensorflow에서 기본적으로 제공하는 튜토리얼에 있는 코드를 가져다 쓰되, epochs, optimizer, dropout같은 경우에는 환경변수를 지정하여 추후에 바꾸면서 실행할 수 있도록 하였습니다.

~~~python
import tensorflow as tf
import os, sys, json

epochs = int(sys.argv[1])
optimizer = sys.argv[2]
dropout = float(sys.argv[3])
print(sys.argv)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(dropout),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs)

score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
~~~

#### Dockerfile 작성
Dockerfile은 간단하게 아래와 같이 작성합니다. tensorflow의 기본 최신 이미지를 베이스 이미지로 하고, 앞서 만든 train script를 컨테이너 생성시 내부로 복사할 수 있도록 합니다.

~~~Dockerfile
FROM tensorflow/tensorflow

COPY train.py .
~~~

#### Image build
아래 명령어를 통해 이미지를 빌드합니다.

여기서 $DOCKER_HUB_REPO는 본인의 도커허브 repository를 넣어주시면 됩니다!
~~~sh
docker build -t $DOCKER_HUB_REPO ./
~~~

#### k8s job yaml 파일 작성
이젠 job을 생성하기 위한 yaml파일을 작성하도록 합시다. 쿠버네티스엔 추상화된 오브젝트들이 정말 다양하게 존재하는데 이들은 모두 yaml파일을 작성하여 손쉽게 생성, 삭제할 수 있습니다. 

yaml 파일은 아래와 같이 작성하면 되는데, 구체적인 k8s 설명은 별도의 포스트에서 소개해드리도록 하겠습니다. cpu는 1개 memory는 5g로 설정하였습니다.

작성은 vim, vi같은 에디터로 진행하면 되고, $DOCKER_HUB_REPO에는 본인의 docker hub repository를 넣으시면 됩니다. 작성이 끝나면, 해당 파일을 example.yaml로 저장해주시면 됩니다.

~~~yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: example
spec:
  template:
    spec:
      containers:
      - name: job-example
        image: $DOCKER_HUB_REPO
        command: ["python", "train.py", "5", "adam", "0.5"]
        resources:
          limits: 
            cpu: "1"
            memory: "5Gi"
      restartPolicy: Never
  backoffLimit: 0
~~~

#### job 생성
최종적으로 아래 명령어를 통해 job을 생성하시면 됩니다.

~~~sh
kubectl apply -f example.yaml
~~~

로그(결과)를 조회하기 위해서는 아래 명령어를 통해 간단히 조회하실 수 있습니다.
$POD_NAME에는 본인의 pod 이름을 입력하시면 됩니다.

~~~sh
kubectl logs -f $POD_NAME
~~~

이렇게 kubernetes에서 하나의 훈련 잡을 성공적으로 생성할 수 있었습니다. 추후에는 여러개의 job을 생성하는 방법을 다뤄볼 예정입니다.

## Reference
* [커피고래님의 MLOps post](https://coffeewhale.com/kubernetes/ml/k8s/docker/machine-learning/2019/03/18/k8s-ml-02/)
* [TF Example](https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko)
