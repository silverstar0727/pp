---
layout: post
title: "ML with Graphs(3): Embedding"
date: 2021-07-12
excerpt: "Stanford CS244W 정리"
tags: [GNN]
category: GNN
comments: False
use_math: true
---

# Lec. 3-1
## 1. Node Embedding Intro
이전 ML with Graphs(2)에서는 Traditional ML의 기법에 대해서 살펴보았다. 가장 주목해야할 점은 Node, Edge, Graph를  수학적으로 표현하기 위해서 이들을 나타내는 feature를 추출하는 작업을 진행하였다.

그러나, 이번 lecture3에서는 각각의 feature를 수동으로 추출하기 보단 **Representation Learning을 이용하여 이들을 하나의 벡터로 가정하고 자동으로 그 벡터를 구하자는(학습하자는) 것이다.**

이것을 식으로 표현하자면, $f = u \to \R^d$이다. 이때 f를 학습하는 것이 목적이라고 보면 된다(이때 u는 노드이며, $\R^d$는 d차원의 벡터를 의미한다.).

## 2. Node Embedding 목표
"그렇다면 이것을 어떻게 학습해야 하는가?"는 당연한 물음일 것이다. 우선적으로 우리는 유사도 함수 $similarity(u, v)$를 가정하게 된다. 그리고 u, v노드 각각의 임베딩 벡터를 $z_u, z_v$라고 하자. 

그러면 우리가 목적하는 것은 다음의 수식이 된다.

$$similarity(u, v) \approx z_v^T z_u$$

여기서의 좌변은 original network에 해당하고 우변은 embedding space에 해당한다.

**즉!!! Orginal Network의 노드 사이의 유사도가 임베딩한 벡터의 유사도와 같도록 해야한다는 것이다.**

조금 어려울 수 있으나, u, v를 벡터화 했다면, 벡터화 이전의 유사도와 이후의 유사도가 같아야 하는 것은 Trivial하다..!

## 3. 추상화(Encoder & Decoder)
여기서 우리는 새로운 개념을 도입하여 이것을 좀 더 추상화 하게 되는데, Encoder와 Decoder가 그것이다.

Encoder는 말 그대로 노드를 벡터로 인코딩하는 것이고, 디코더는 벡터를 노드로 변환하는 것이다.

이렇게 개념을 도입하게 되면 우리는 수식으로 다음과 같이 두 가지 수식을 나타낼 수 있다.

* Encoder: $Enc(v), Enc(u) = z_v, z_u$
* Decoder: $Dec(z_v^T z_u) =$ similarity score

### 3.1. Encoder
그렇다면 노드의 벡터화 즉 Encoder는 어떻게 할 것인가?

간단하게 shallow encoding기법을 소개하면 다음과 같다.
$Enc(v) = z_v = \Z \cdot V$ 즉 matrix $\Z$를 look-up 하겠다는 의미와 동일하다.($\Z \in \R^{d \times \mid V \mid}$) 그리고 우리는 여기서 $\Z$를 학습하면 된다.

이를 그림으로 보면 다음과 같다.
![](https://images.velog.io/images/djm0727/post/ea28a3d0-4aa8-4b9f-9fc5-8f1b8307ca17/image.png)


그러나 이러한 방법에 단점이 존재하는데 **노드의 수가 늘어남에 따라 복잡성이 증가**한다는 것이다. 따라서 추후에 이를 보완할 Deepwalk와 Node2Vec 방법을 3-3에서 소개하겠다.

이제 벡터화하는 것을 마쳤으니 우리에게 남은 과제는 **similarity를 어떻게 정의하느냐**이다. 이것은 다음의 3-2에서 random walk를 설명할 예정이다.

> 우리가 현재 배우고 있는 임베딩은 Node의 label이나 feature가 필요 없는 self or unsupervised learning에 해당한다. 따라서 task-independent하다.

# Lec. 3-2
## 1. Random walk
### 1.1. Random walk 개요
랜덤워크의 목적은 '현재 위치에서 무작위로 선택하여 이동할 때 u에서 출발해서 v를 들릴 확률'을 similarity로 정하여 학습하는 것이다.

이것은 $p(v \mid z_u)$로 표현된다. 이러한 방법으로 임베딩하는 것을 Random walk embedding 이라고 하고 $z_u^T z_v$로 표현하면 u, v가 동시에 랜덤워크에서 일어날 확률과 같다.

그렇다면 우리가 이것을 어떻게 최적화 시킬 수 있을까? 다음의 두 과정으로 이를 간단하게 표현이 가능하다.

1. Random walk 전략 R에 대해서 u에서 시작하여 v로 가는 확률 $p_R(v \mid u)$를 구하자.
2. 이때 $z_i$와 $z_j$가 이루는 각도 $\theta$가 $p_R(v \mid u)$와 비례하도록 최적화를 하면 된다.

### 1.2. Why Random walks?
* **expressivity**: 표현력(정보의 압축)이 좋다. 즉, 지역과 고차원 이웃정보를 효율적으로 통합할 수 있다.
* **efficiency**: 모든 노드쌍이 아니라 동시에 발생하는 것만 고려하면 되기 때문에 효율성이 좋다.

### 1.3. Random walk process
1.1의 과정은 정말 '간단히' 표현한 것이고, 우리가 실제로 실행한다고 했을 때는 아래의 프로세스를 거쳐야 한다.

> Notation) $N_R(u) =$ R전략에서 u의 이웃들

그 전에 우리가 하고자 하는 목적을 나타내면, **그래프가 주어졌을 때 $f: u \to \R^d$인 f 즉, $f(u) = z_u$인 f를 찾아야 한다는 것을 알 수 있고**, 이때 목적함수는 다음과 같다.

$max_f \sum_{u \in V} log(p(N_R(u) \mid z_u))$

이제 본격적인 프로세스는

1. 길이를 고정하고 u에서 전략 R인 Random walk를 실행한다.
2. 각 노드 u에서 생성되는 이웃 $N_R(u)$를 모은다.
3. u가 주어질 때 $N_R(u)$를 예측하며 최적화를 진행한다.($max_f \sum_{u \in V} log(p(N_R(u) \mid z_u))$)

이것을 조금 더 수학적으로 의미를 뽑아내어 도출하면 다음과 같다.

위의 프로세스와 동일한 식으로 $L = \sum_{u \in V} \sum_{v \in N_R(u)} (- log(p(v \mid z_u)))$이다. 

여기서 $P(v \mid z_u) = {exp(z_u^Tz_v) \over \sum_{n \in V} exp(z_u^Tz_n)}$은 자명하고

이를 이용하여 L을 다시 쓰면, $L = \sum_{u \in V} \sum_{v \in N_R(u)} (- log({exp(z_u^Tz_v) \over \sum_{n \in V} exp(z_u^Tz_n)}))$

즉 optimize random walk embedding이라는 것은 최소 L일때 $z_u$를 찾는 것이다.

> 정말 어려워졌다. 위 한 줄의 식에서 각 항을 설명하면 아래와 같다. 이해가 가지 않는 다면 꼭 주석을 보자.

> * $\sum_{u \in V}$ : 모든 노드에 대해 반복
> * $\sum_{v \in N_R(u)}$ : u에서 출발할 때 전략 R에서 v가 나온 모든 경우
> * ${exp(z_u^Tz_v) \over \sum_{n \in V} exp(z_u^Tz_n)}$: u, v가 동시에 일어날 것을 예측할 확률 (softmax)

그러나 여기서 $\sum_{u \in V}$의 이중합이 된다. 즉 $O(V^2)$의 복잡도를 갖게 되므로 계산이 아주 힘들어지게 된다. 