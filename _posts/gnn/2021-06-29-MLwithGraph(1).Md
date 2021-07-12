---
layout: post
title: "ML with Graphs(1): Intro"
date: 2021-06-29
excerpt: "Stanford CS244W 정리"
tags: [GNN]
category: GNN
comments: False
use_math: true
---

ML with Graph는 Stanford CS224W의 course입니다.

해당 시리즈는 CS224W의 강의를 들으며 한국어로 정리하는 글입니다. 글의 구성 역시 본 강의와 대부분 동일합니다.


# Lec. 1-1
## 1. Graph 종류
종류는 크게 Network(=Natural Graphs)와 Graphs(=Representation)으로 나누어 진다.

* Network(Natural Graphs) examples
  * Social network
  * Communication & Transections
  * Biomedicine
  * Brain connection
* Graphs(Representation) examples
  * Information or Knowledge graphs
  * Sortware
  * Similarity networks
  * Relation structures
    
이러한 복잡한 도메인은 '상호 관계가 있는 그래프'로 표현할 수 있는 일종의 **관계성**을 지닌다는 특징이 있다.
 
현대의 많은 딥러닝은 이완 달리 **sequence** 또는 **grid** 형태의 데이터이다. 
> cf) sequence data는 NLP와 같은 분야에 해당하고 grid data는 CV같은 분야에 해당한다.

따라서 이들과 비교했을 때에 그래프는 다음의 특징을 가진다.
* Arbitrary size & 복잡한 위상 구조를 가짐(grid처럼 공간적 지역성(spatial locality)이 없음)
* 고정된 기준점 및 순서가 없음
* dynamic & multimodal한 특성을 지님

## 2. GNN Overview
### 2.1. GNN 구조
GNN(Graph Neural Network)는 다음과 같은 구조를 가진다.

1. Input으로 Graph를 받는다.
2. Graph Conv 등의 layer를 통과한다.
3. Activation function 등의 layer를 통과한다.
4. Dropout과 같은 Regularization layer를 통과한다.
5. prediction을 진행한다.(nodes, links, graphs 수준의 에측)

이것을 그림으로 표현하면 아래와 같다.
![](https://images.velog.io/images/djm0727/post/e0f2702d-9d5d-49b6-9512-737280d15aec/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-06-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.38.10.png)

### 2.2. Lifecycle
이러한 ML with graphs의 라이프 사이클은 아래 그림과 같다.

![](https://images.velog.io/images/djm0727/post/0d4d30c7-c910-4e0e-9d49-201209b7f1d2/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-06-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.51.51.png)

이때 raw data를 graph data로 변환하는 과정에서 기존의 ML과 다른 지점을 알 수 있는데 기존 ML의 경우에는 feature engineering을 수행했다면, graph에서는 Representation Learning을 통해서 feature의 자동 학습을 수행한다는 차이가 존재한다.

#### Representation Learning
위에서 살펴본 representation learning은 **node(or graph, link etc.)를 d차원의 벡터에 임베딩** 하는 것을 의미한다 즉 아래의 식과 같다.

$f:u \to \R^d$

따라서 목표는 이러한 함수 f를 학습시키는 것이 된다.

## 3. 앞으로 배울 것들
* **전통적인 방법**: Graphlets, Graph kernels
* **노드임베딩 방법**: Deepwalks, Node2Vec
* **GNN**: GCN, GraphSAGE, Graph Attention
* **Knowledge graph & Resoning**: TransE, BetaE
* **깊은 생성모델**
* **Application**

# Lec. 1-2
## 1. 활용 개요
어느 수준에서 문제(or goal)을 정의할 것인가에 따라서 나뉠 수 있다.
**대체로 Nodes, Edges를 각각 Objects, Interaction에 대응**하여 과학 등의 분야에 적용하는 것을 상기하면서 보자.

* Node-level: Node classification(노드의 속성 분류)
ex) user의 특성 예측
* Edge-level: Link prediction(두 노드 사이의 누락된 링크 예측)
ex) knowledge graph 완성
* Community-level: Clustring(노드들이 클러스터를 형성하는 가에 대한 예측)
ex) social circle 탐지
* Graph-level: Graph classification(그래프 속성 분류)
ex) 분자의 속성 예측
  * other task: Graph generation(ex. 신약 개발), Graph evolution(ex. physical simulation)

## 2. Node-level
노드 수준에서의 Graph 활용의 대표적인 예는 단백질 폴딩(Protein-folding)이다. 

아미노산이 가진 염기서열 즉, sequence data를 통해서 단백질을 이루는 구조를 3D로 예측하는 것이다.

현재는 Google DeepMind의 Alpha-fold2가 SOTA(최고 성능의 모델, State-Of-The-Art)를 달성하고 있는데, 여기서의 핵심 아이디어는 **Spatial Graph**를 구성하는 Nodes와 Edges를 다음과 같이 정하고 있다는 것이다.

* Nodes: 단백질 sequence에서의 아미노산
* Edges: 아미노산(Nodes) 사이의 근접성

이를 통해 새로운 아미노산의 위치를 예측하도록 훈련하는 방식으로 SOTA를 달성하였다.

## 3. Edge-level
Edge level에서의 예는 추천 시스템이나 약물 조합의 부작용 예측 등이 있다. 추천 시스템 같은 경우에는 유저와 아이템의 상호작용을 예측하는 것이라고 볼 수 있고, 약물 조합의 부작용 예측의 경우에는 약물간 상호작용을 예측하는 것으로 간주할 수 있다. 따라서, edge level은 상호작용에 집중하는 것을 알 수 있다.

> 사실 직관적으로 edge는 대부분 interation과 대응되기 때문에 당연히 상호작용을 예측하지 않을까...?

### 3.1. 추천 시스템
추천 시스템의 Nodes, Edges는 다음과 같이 정의한다.
* Nodes: Users & Items
* Edges: Interation(영화보기, 구매 등)

이때 유저가 좋아하는 Item을 예측하도록 훈련을 진행한다.

### 3.2. 약물 조합 부작용 예측
약물 조합 부작용을 예측하는 task에서는 Nodes, Edges를 다음과 같이 정의한다.

* Nodes: Drugs & Proteins(약물이 적용되는 타겟 단백질)
* Edges: Interaction

이때, drugs와 proteins의 interaction을 예측하도록 훈련을 진행한다.


## 4. Community-level
community(sub-graph)에서는 교통예측(최단 시간 경로)을 예로 들 수 있다. 길을 부분부분으로 나누어 각각을 segments로 간주하면 Nodes, Edges는 다음과 같이 정의한다.

* Nodes: Road Segments
* Edges: 연결 여부

이때 최단 시간을 예측하도록 훈련을 진행한다.


## 5. Graph-level
graph-level에서는 신약개발, 물리 시뮬레이션 등을 예로 들 수 있다.

### 5.1. 신약개발
신약개발에서는 Nodes, Edges를 다음과 같이 정의한다.

* Nodes: 원자들
* Edges: 화학적 결합

이때 새로운 분자를 생성하도록 훈련을 진행한다.

### 5.2. 물리 시뮬레이션
물리 시뮬레이션에서는 Nodes, Edges를 다음과 같이 정의한다.
* Nodes: 입자들
* Edges: 입자간 상호작용(ex. Central Force)

이때 그래프의 진화(graph evolution)를 예측하도록 훈련한다. 즉, 재료를 일련의 입자로 표현하고 근접성에 따라서 interaction을 정의한다. 이후에 위치, 속도 pass message(update)를 진행한다. 이후에 동역학적 정보를 다시 추출한 다음에 입자로 표현하는 과정을 반복하게 된다.

# Lec. 1-3
## 1. 그래프에 대해 알아보기
### 1.1. 그래프의 구성요소
Lec 1-2에서 계속 보았듯, 우리는 Node, Edge를 각각 어떻게 대응하여 문제를 해결할 지 고민을 해야한다. 이것을 수학적 기호로 표현하면 아래와 같다.

* Objects = Nodes => $N$
* Interations = Edges => $E$
* System = Graph => $G(N, E)$

### 1.2. 상황에 따른 그래프 디자인
그래프는 노드간 상호작용이 단방향으로 이루어질 수도 있고, 양방향으로 이루어질 수도 있다. 따라서 상황에 따라 적절하게 그래프를 설정해야 한다.

* Undirected Graph: 대칭적인 상호작용 시
* Directed Graph: 비대칭적인 상호작용 시

### 1.3. Degree of Node
노드에 연결된 Edge의 수를 Degree of Node라고 한다. 

아직 왜 이것을 계산하는 지 모르겠지만, 평균 degree를 계산하면 다음과 같다.
* Degree of Node: $k_{i} = i$번째 노드와 이어진 Edges 수
* Undirected Graph에서의 평균 : $\bar{k} = {1 \over N} \sum_{i=1}^N k_i = {2E \over N}$
* Directed Graph에서의 평균: $\bar{k} = {E \over N}$

## 2. 그래프의 표현 방식
### 2.1. Adjacency Matrix
인접 행렬을 이용한 방법은 $A_{ij}$임을 의미하는데, **이것은 i번째 노드가 j번째 노드와 이어져 있는 것**과 동일한 의미이다.

한편 **Undirected Graph에서는 인접 행렬이 무조건 대칭**(symmetry)임을 상기하자. 아래 그림은 이러한 인접 행렬의 예시를 보여준다. 

![](https://images.velog.io/images/djm0727/post/97217154-bf7f-4a21-86b8-7ffe4b6fbbc5/image.png)

그러나 대부분의 실용적인 그래프에서 노드는 대부분의 다른 노드들과 이어져있지 않은데, 이러한 현상은 **인접행렬의 대부분이 0으로 되는 결과를 낳는다(super sparse matrix, 희소행렬**).
즉, 이는 메모리에서의 효율성을 가져오지 못할 뿐만 아니라, 계산량을 증가시키기도 한다. 따라서 이러한 단점을 보완해야 할 필요가 있다.

### 2.2. Edge List
엣지 리스트는 노드쌍을 다음과 같이 표현한다. $(i, j)$ 이러한 **노드 쌍은 노드가 i번째 노드와 j번째 노드가 이어져 있음**을 의미한다. 
따라서 이어져 있는 모든 노드 쌍에 대한 것을 list형태로 나열한 것이 Edge list이며, 이는 인접 행렬에서의 sparse한 단점을 보완한 형태라고 할 수 있다.

아래 그림은 이러한 Edge List의 예를 보여준다.

![](https://images.velog.io/images/djm0727/post/fa193037-cdfa-47a2-b591-3769f149522f/image.png)

그러나 Edge List도 단점을 갖는데 **너무 큰 그래프에 대해서는 많은 엣지**가 형성되기 때문에 이것을 List 형태로 나타내면 접근하는 것이 쉽지 않다는 것이다.

### 2.3. Adjacency List
인접 리스트는 위에서 본 두 가지의 방식을 혼합했다 할 수 있다.** 각 노드 인덱스를 나열한 후 각각이 이어진 모든 노드들을 다시 리스트로 나열한 것**을 의미한다. 앞서 말한 두 가지 방식의 그래프 표현에서 단점을 모두 보완하였으며 그 예시는 아래 그림과 같다.

![](https://images.velog.io/images/djm0727/post/221e9e60-67b9-46fa-b155-233261e550c7/image.png)


#### Reference
* http://web.stanford.edu/class/cs224w/