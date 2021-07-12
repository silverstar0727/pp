---
layout: post
title: "ML with Graphs(2): Traditional ML"
date: 2021-06-30
excerpt: "Stanford CS244W 정리"
tags: [GNN]
category: GNN
comments: False
use_math: true
---


이번 ML with Graph(2)에서는 제목인 "Traditional method"와 같이 전통적인 방법론들에 대해서 살펴볼 예정이다.

그렇담 전통적인 방법론은 무엇인가?

우리가 Lec1에서 Course에 대한 overview를 진행했을 때, GNN은 전통적인(수동의) feature extraction 대신에 자동으로 이러한 feature를 representation learning으로 학습하여 추출한다고 배웠다.

따라서 **Node, Edge, Graph 수준에서의 feature를 과거에는 어떻게 추출했는지**에 대해서 배워보자.

> 그런데 사실 '전통적인'이라고 하기 민망할 정도로 십수년 내에 개발된 방법론들이다.

> 한 가지 선험적인 가정은 이번 Lec2에서는 논의의 편의성을 위해서 Undirected graph만을 다룬다.

# Lec. 2-1
Lec. 2-1에서는 전통적인 방법론 중 Node 수준에서의 feature extraction을 다룬다.

## 1. Node Classification
Lec. 2-1에서는 전통적인 방법론 중 Node 수준에서의 feature extraction을 다룬다.

주된 쟁점은 Semi-supervised Learning처럼 노드에 대한 label이 일부만 있을 때, 나머지 노드를 분류 하도록 하는 것이다. 그 방법론은 다시 여러 갈래로 나뉘게 되는데 아래의 4개가 바로 그것이다.

* Degree of Node
* Node Centrality
* Clustering Coefficient
* Graphlets

이제 위의 방법론을 차례로 설명할 예정인데 Degree of Node는 Lec1의 그래프 표현에서 간단하게 보았다. 크게 특별한 내용이 없으므로 간단하게만 예를 들어보자.

직관적으로 우리는 Node와 연결된 Edge의 숫자를 통해서 노드를 분류할 수 있다고 생각할 수 있다. 가령 30개 가량의 노드가 3정도의 degree(엣지의 수)를 갖고 있다면, 이들을 공통된 class로 볼 수 있다는 것이다.

> degree of node에 대한 특징은 Lec. 1-3을 참고하자.

## 2. Node Centrality
노드의 중요성은 말 그대로 노드가 얼마나 그래프에서 중요한 위상을 차지하고 있는가에 대한 기준을 갖고 노드들을 분류하는 것이다. 

그 방법에 대해서는 Eigenvector Centrality, Betweness Centrality, Closeness Centrality의 세 가지를 소개할 예정이다.

### 2.1. Eigenvector Centrality
고유값 중요도는 Node Centrality 중에서도 핵심적인 방법론이라 할 수 있기에 잘 알아두어야 한다. 추후에 비슷한 내용으로 다시 등장하기 때문이다.

우선 논의를 시작하기에 앞서 다음의 가정 및 Notation을 정해야 한다.
* 노드는 centrality vector $c_v$를 갖는다고 생각하자.(이것을 예측하는 것이 목표!)
* $N(v)$는 v의 노드 집합이다.
* $\lambda$는 정규화 인자이다.

이때 다음의 식이 성립하게 된다. $c_v = {1\over \lambda} \sum_{u \in N(v)} c_u$
이를 말로 풀어 설명하면 **노드 v의 중요도 벡터 $c_v$는 주변 노드의 중요도 벡터들의 합을 정규화한 것이다.**
의미를 좀 더 뽑아내자면, 주변 노드가 중요하면 당연하게도 해당 노드도 중요해질 것이다.

그런데 이것은 **고유값 방정식** $\lambda c = Ac$으로 나타낼 수 있는데, 여기서 A는 인접행렬이고, c는 고유벡터가 되는 중요도 벡터에 해당한다. 즉, 인접행렬을 look-up하는 방식으로 앞에서 본 식을 선형대수적으로 바꿀 수 있다는 것이다.

한편 이러한 고유방정식은 다음의 정리들을 만족한다.
* $\lambda_{max}$는 항상 양수이면서 유일하다(by Perron-Frobenious정리)
* $c_{max}$는 $c_v$의 중요도 점수에 해당한다.

### 2.2. Betwenness Centrality
**다른 노드의 최단경로에 해당 노드가 포함되어 있는가**에 대한 여부를 기준으로 그 중요도를 판단한다.

다음의 식이 아주 직관적이기에 구체적인 설명은 생략한다.
$$c_v = \sum_{s \neq v \neq t} {(v를 포함하는 s와 t사이의 최단경로) \over (s와 t 사이의 최단경로)}$$

### 2.3. Closeness Centrality
**다른 노드와의 거리가 가까울수록 중요하다는 것**을 기준으로 노드의 중요도를 판단하는데, 이것 역시 아주 직관적이다.

$$c_v = \sum_{u \in V} {1 \over u, v사이의 최단경로}$$
## 3. Clustering Coefficient
**얼마나 많은 이웃 노드들이 연결되어 있는가**를 기준으로 판단하게 되는데, 이웃 노드들의 개별 중요도 역시 고려해야할 대상이다. 따라서 식은 다음과 같다.

$e_v = {이웃 노드들의 Edges  수 \over {k_v \choose 2}}$

여기서 $e_v$는 0부터 1까지의 수만을 가질 수 있다는 특징이 존재한다.

## 4. Graphlets
그리 어려운 논의는 아니다. 직관적으로 이해하자. 추후 kernel을 논의할 때 다시 등장하므로 잘 이해해보도록 하자. 노드의 수를 k라고 할때, k별로 가능한 그래프의 종류는 당연하게도 정해져 있다. 따라서 그 수를 graphlet이라 한다.

아래 그림을 보면 조금 더 이해가 잘 간다.

![](https://images.velog.io/images/djm0727/post/3095fa88-7da6-465e-b416-9301c55a5b71/image.png)

### 4.1. Graphlet Degree Vector(GDV)
이것 역시 어렵지 않은 내용이다. 그래프와 root 노드가 주어질 때, root로부터 가능한 sub-graph의 개형 별로 숫자를 세어서 vector로 만든 것이다. 아래 그림을 참고하자.

![](https://images.velog.io/images/djm0727/post/6ea19aec-f8de-4eeb-80d4-26f6b90e67a1/image.png)

# Lec. 2-2
## 1. Link Prediction
링크를 예측하는 작업은 기존에 존재했던 링크를 바탕으로 새로운 링크를 예측하는 것이다.

이러한 예측의 핵심은 노드쌍에 대한 feature를 어떻게 디자인 하는가이므로 상당히 prediction의 두 가지 공식과 공식을 이용한 방법론에 대해서 살펴보고, 링크 수준에서의 feature는 어떠한 것들이 있는지에 대해서 알아보자.

### 1.1. 두 가지 공식
* 랜덤으로 링크를 지우고 예측하는 방법(Links missing at random)
* 시간에따라 진화하는 그래프를 예측하는 방법(Links over time)
  * 이를 notation으로 자세히 설명하자면, $G(t_0, t'_0)$에서 $G(t_1, t'_1)$로 진화할 때에 새롭게 생겨난 링크를 아웃풋으로 하는 Ranked List L을 아웃풋으로 반환해야 한다.

### 1.2. 방법론
그렇다면 위 공식을 어떻게 적용해야 하는가?

다음의 프로세스를 보자.

1. 각 노드쌍 $(x, y)$에 대해서 노드쌍의 점수 $c(x, y)$를 계산한다.
2. 이들을 $c(x, y)$에 따라서 내림차순으로 정렬한다.
3. 상위 n개의 노드쌍를 예측한다.
4. 실제로 $G(t_1, t'_0)$에서 예측한 링크들이 나타나는지 확인한다.

아니 그런데 조금 이상한 것이 있다. $c(x, y)$ 도대체 무어란 말인가?
이것이 바로 link수준에서의 feature에 해당하며, 아래에서 다루는 것들을 이용하여 그 점수를 매긴다고 보면 되겠다.

## 2. Link-level features
### 2.1. Distance-based feature
거리에 기반한 feature는 두 노드 사이의 최단경로를 기준으로 feature를 정하는 방식이다.

아주 간단한 방법이지만, 치명적인 단점이 존재하는데, 이웃이 겹치는 정도나 연결 강도를 측정하지 못한다는 것이 바로 그것이다. 이것은 다시 Local Neighborhood overlap에서 해결이 가능하다.

### 2.2. Local Neighborhood overlap
이웃이 겹치는 정도 및 연결 강도를 측정하기 위한 다음의 3가지 방법을 소개하겠다.

* Common neighbors: 단순히 공통된 이웃의 개수를 세는 것이다.
  * Notation은 ${\mid N(v_1) \cap N(v_2)\mid}$과 같다.
* Jaccard's Coefficient: common neighbors의 정규화 버전이라고 보면 된다.
  * Notation은 ${\mid N(v_1) \cap N(v_2)\mid \over \mid N(v_1) \cup N(v_2)\mid}$과 같다.
* Adamic-Adar Index: 공유하고 있는 노드(두 노드의 공통된 이웃) 집합에 대해서 각 노드의 이웃의 개수를 기준으로 판단하는 것이다.
  * Notation은 $\sum_{u \in N(v_1) \cap N(v_2)} {1 \over log(k_u)}$이다.

그러나 이러한 방법론들도 문제가 존재하는데, 연결되지 않은 두 노드에 대해서는 항상 $\mid N(v_1) \cap N(v_2) \mid = 0$이라는 것이다.

얼핏보면 무슨 문제가 있나 싶지만, 현재에는 두 노드가 연결되어 있지 않더라도, 미래의 어느시점 t에서는 두 노드가 연결될 수 있다는 가능성 자체를 배제하기 때문이다. 

### 2.3. Global Neighborhood overlap
Global Neighborhood overlap에서는 위에서 지적한 오류를 바로잡아준다.

이때 kartz index를 활용하게 되는데 핵심적인 아이디어는, 두 노드 사이의 모든 경로의 개수를 count하는 것이다.

그렇다면 어떻게 카운트를 하는 것이 좋을까? 

여기서는 인접 행렬의 거듭제곱을 이용하여 간단하게 셀 수 있다.
인접 행렬의 거듭제곱이 경로의 개수를 세는 것에 사용이 가능한 이유의 증명은 다음과 같다.

1. $A_{uv} = 1$은 u, v가 이웃노드라는 것을 의미한다.
2. $P_{uv}^{(k)}$는 길이가 k인 u, v 사이의 경로라고 가정하자.
3. 이때 $P^{(k)} = A^k$임을 보이자.
4. $P_{uv}^{(1)} = A_{uv}$가 성립하는 것은 자명하다.
5. 이때 다음이 성립한다. $P_{uv}^{2} = \sum_i A_{ui} \times P_{iv}^{(1)} = \sum_i A_{ui} \times A_{iv} = A_{uv}^2$ 여기서 i는 u, v의 경로에 존재하는 공통된 이웃노드이다.

따라서 karts index는 인접행렬로 간단하게 계산이 가능한데, 이를 식으로 표현하면 다음과 같다.

$S_{v_1, v_2} = \sum_{l = 1}^{\infty} \beta^l A_{uv}^l$

여기서 등장한 $\beta$는 discount factor로 0과 1 사이의 값을 갖는다.

이러한 수식을 행렬로 변환하여 선형대수적으로 표현하면 다음과 같다.

$S = \sum_{i = 1}^{\infty}\beta^i A^i = (I - BA)^{-1} - I$

# Lec. 2-3
## 1. Kernel method
이번 Lec. 2-3은 그래프 수준에서의 전통적인 feature를 다룬다. 그러나 조금 다른 것은 feature vector 대신에 커널을 디자인 한다는 점이다.

이러한 커널은 다음의 특징을 갖기 때문에 이러한 방법이 성공적으로 사용되고 있는 것이므로 잘 보고 무엇을 목적으로 해야하는지 알아보자.

* 커널 $K(G, G')$ 은 두 그래프 G, G' 사이의 유사성을 나타내는 실수이다.
* 커널 행렬 $\Kappa = (K(G, G'))_{(G, G')}$은 항상 양의 값을 가지면서 대칭인 행렬이다.
* 이때 $\phi(G)^{T}\phi(G') = K(G, G')$을 만족하는 feature의 표현 $\phi(G)$가 존재한다.
* 커널이 한번 정의되면 추후 예측에 계속적으로 사용될 수 있다.

그렇다면 이젠 커널이 그래프에서 어떻게 생성되는지에 대해서 알아보자.

## 2. Graph kernel
### 2.1. Graphlet kernel
결국에 목표해야 하는 것은 graph kernel을 구하는 것임을 기억하면서 시작하자.

여기서 핵심적인 아이디어는 Bag of Word처럼 Node를 세어서 넣는 방식인 Bag of Node 방식을 사용한다. 그러나 노드의 개수가 다른 구조일 수 있는데 이것을 Degree of Node 형태로 저장하게 된다. 그 예시는 아래와 같다.

![](https://images.velog.io/images/djm0727/post/8804f1cb-85f7-40bb-a760-e9ce808bc26a/image.png)

이때 Graphlet kernel은 graphlet의 개수를 기준으로 그 점수를 책정한다.

우리는 앞서 graphlet에 대해서 알아보았는데, 여기서 개수를 셀 때의 graphlet은 앞선 것과 다음의 두 가지 차이가 존재한다.
* 모두 이어져 있지 않아도 됨
* root 노드가 없음

따라서 k = 3일때 다음의 4개가 가능하여 graphlet은 4가된다.

![](https://images.velog.io/images/djm0727/post/685f3991-c945-428b-9eda-4aac49471037/image.png)

즉, 주어진 그래프 G에 대해서 graphlet list $g_k = (g_1, ..., g_{n_k})$일 때, graphlet count vector $f_G \in \R^{n_k}$에서
1 ... $n_k$에 대해 다음이 성립한다. $(f_G)_i = (g_i \in G)의 개수$

예시를 보면 더 이해가 잘 가는데 아래의 예에서 주어진 그래프 G에 대한 graphlet count vector $f_G = (1,3,6,0)^T$임을 알 수 있다.

![](https://images.velog.io/images/djm0727/post/e7b1f33f-0e39-4251-82b7-4a7b2492dafd/image.png)

이쯤에서 다시 우리의 목표를 상기하자. 궁극적으로 우리는 graph kernel를 구해야 한다.
그렇다면 우리가 도출한 graphlet count vector로부터 graphlet kernel을 도출하면 다음과 같다.
$K(G, G') = f_G^T f_{G'}$

초기에 목표한 바를 달성하였으나, 한 가지 문제가 생기는데 G, G'의 크기가 다를 때에 skew가 생기는 것이 바로 그것이다. 즉, 각 그래프의 크기를 고려하지 못한다.

따라서 실제로는 이것을 정규화 한 다음의 커널을 사용한다.
우선, $h_G = {f_G \over sum(f_G)}$을 가정하여 정규화를 한 뒤, 다시 커널을 다음과 같이 계산하면 된다.

$K(G, G') = h_G^T h_G$

모두 끝난 것 같으나, 언제나 그렇듯 다시 문제가 발생한다. graphlet의 계산 비용이 너무나도 크다는 것이다. k의 graph에서 graphlet을 계산하는 비용은 $n^k$에 해당하기 때문이다. 따라서 이를 보완한 방법을 알아보자.


### 2.2. Weisfeiler-Lehman kernel
해당 방법론은 graphlet을 계산하는 비용을 줄이기 위해 vocab을 풍부하게 하는 방식을 사용하였다.

갑자기 vocab이 뭔데 싶겠지만, Natural Language Processing의 Bag of Word 방식을 차용하였기 때문에 용어가 그렇다!

즉, 노드를 분류할 때 더욱 다양화 하자는 것인데 이를 위해서 주변 노드와의 연결성을 바탕으로 진행하게 된다. 여기서부터는 직관적으로 이해하기 위해서 노력을 해야하므로 집중해서 읽어보자.

여기서는 핵심적인 아이디어를 두 단어로 "color refinement"로 정의하고 있으며, 그 과정은 다음과 같다.

1. 모든 노드의 가중치를 1로 초기화한다.
2. 주변 노드의 가중치를 바탕으로 현재 노드의 가중치를 업데이트 한다.
3. 적절한 hash 함수를 이용하여 현재 노드의 가중치를 hashing한 후, 그 값들을 다시 넣어주게 된다.
4. 2, 3을 반복한다. 이때 반복횟수를 k라고 할때 k-hop neighborhood의 정보를 요약한 결과가 나오게 된다.

이를 식으로 나타내면 다음과 같다.
$c^{(k + 1)}(v) = HASH(c^{(k)}(v), \{ c^{(k)}(u)\}_{u \in N(v)})$

그런데 너무 식이 어렵다는 생각이 든다. 예시로 이것을 다시 살펴보자.

다음의 구조만 살짝 다른 두 그래프가 있다고 가정하고, 이들의 kernel을 계산해보자. 그리고 위에서 살펴본 과정을 그대로 가져와서 직관적으로 이것이 가능함을 느껴보자.

1. 모든 노드의 가중치를 1로 초기화 한다.
![](https://images.velog.io/images/djm0727/post/9023726b-c35e-43b9-bd67-1302bf088f75/image.png)

2. 주변 노드의 가중치를 바탕으로 현재 노드의 가중치를 업데이트 한다.
![](https://images.velog.io/images/djm0727/post/722d3c08-b3b4-4f56-bfa7-62052a7a2803/image.png)

3. 적절한 hash 함수를 이용하여 현재 노드의 가중치를 hashing한 후, 그 값들을 다시 넣어주게 된다.
![](https://images.velog.io/images/djm0727/post/50d5d2b3-d8b8-4cd3-8b01-b40a7c2c032b/image.png)

4. 주변 노드의 가중치를 바탕으로 현재 노드의 가중치를 업데이트 한다.
![](https://images.velog.io/images/djm0727/post/9e9277f7-d085-4675-9df3-5512f7dfc791/image.png)

5. 적절한 hash 함수를 이용하여 현재 노드의 가중치를 hashing한 후, 그 값들을 다시 넣어주게 된다.
![](https://images.velog.io/images/djm0727/post/1c285ea8-579b-4d03-8218-5b1cc475cc71/image.png)

6. 이제 $\phi$를 구할 수 있다.
![](https://images.velog.io/images/djm0727/post/0dd7e79c-9312-4814-b7a5-36f91b81a078/image.png)

7. kernel을 계산하자.
$K(G, G') = \phi(G)^T \phi(G') = 49$

이제 모든 작업이 끝났다.

한편, 해당 방법은 시간복잡도 면에서 각 스텝에 대해 linear한 시간복잡도를 가지므로 효율적이라고 할 수 있다.

#### Reference
* http://web.stanford.edu/class/cs224w/