---
layout: post
title: "ML with Graph(4): PageRank"
date: 2021-07-12
excerpt: "Stanford CS244W 정리"
tags: [GNN]
category: GNN
comments: False
use_math: true
---
# Lec. 4-1
## 1. Web as a Graph
고전적 웹페이지는 자세히 들여다 보면 특정 사이트를 연결하는 hyperlink를 가지고 있다. 따라서, 특정 웹페이지간의 관계성이 충족되는데 가령 이런식이다.

현재의 velog page에서 참조 항목으로 다른 velog를 하이퍼링크로 걸었다고 해보자. 그렇다면, 현재 페이지에서 다음 페이지로 향하는 digraph로 표현된다. 이때 노드는 2개이며 하나는 현재 페이지 나머지 하나는 연결된 페이지일 것이다.

아직도 무슨말인지 이해가 안갈 수 있다. 

따라서 그림으로 표현하면 아래와 같은데, 이는 Network를 클릭하면 다음페이지로, CS224W의 Gates를 클릭하면 CS홈페이지로, Stanford를 클릭하면 Stanford홈페이지로 연결되는 그런식이다.

이렇게 웹 페이지를 그래프로 표현할 때 그들을 잇는 edge를 link라고 하면 link analysis algorithm으로 다음의 것들이 존재한다.

* PageRank
* Personalized PageRank
* Random Walk with Restarts

언제나 그렇듯 핵심 아이디어는 PageRank가 되나, 단점을 지니고 있어 보완하기 위한 방법론으로 두 가지를 더 설명할 예정이다.

> 이번 lecture에서 핵심적으로 다루게 되는 PageRank는 스탠포드에서 개발된 알고리즘으로 구글의 검색 알고리즘으로 널리 알려져 있다. 따라서, 많은 예시들이 page간의 interaction으로 표현될 예정이다.

다음으로 넘어가서 Page Rank를 설명하기 전에 간단히 직관을 길러보자. 웹 페이지를 그래프로 표현할 때, 우리는 당연하게도 **많은 페이지로부터 링크가 걸려있는 페이지가 중요하다는** 것을 알 수 있다.

가령, 스탠포드 홈페이지에서 각 학과는 한번씩 링크가 걸려있겠지만, 각 학과로부터 홈페이지가 다시 한번씩 걸려있기 때문에 학교 자체의 홈페이지가 학과의 홈페이지보다 중요하다는 것이 와닿는가?

조금 직관을 길렀으면 이제 다음으로 넘어가자.

## 2. PageRank
PageRank는 앞선 직관에서 사용한 예시와 같이 "많은 페이지로부터 접속링크를 받을수록 중요도가 올라간다"를 이용하게 된다.

이것을 vote로 표현하고 있는데, 말로 설명하면 아래와 같다.

* 각 페이지 i는 중요도 벡터 $r_i$를 갖고 있다.
* $r_i$ = 들어오는 중요도의 합 = 나가는 중요도의 합
* 즉 개당 중요도는 $r_i / d_i$이다.

예로써 다음의 그림에서 j의 중요도 벡터는 들어오는 중요도의 합으로 다음과 같이 결정된다.
![](https://images.velog.io/images/djm0727/post/8359041d-d620-4ac9-ab83-e8dfe232e41d/image.png)

$r_j = r_i/3 + r_k/4$

이제 이것을 좀 더 수학적으로 표현하면 $r_j = \sum_{i \rightarrow j} {r_i \over d_i}$이 된다. 이를 두고 **"Flow equation"** 이라 부른다.

다음의 예시를 통해 solution을 구해보도록 하자.

여기서 세가지 식이 아래와 같이 도출될 수 있는데, 다른 노드 중요도로 현재의 노드 중요도가 정해지며, 현재 노드 중요도로 다른 노드 중요도가 정해지므로 **'recursive'** 한 형태를 갖고 있다.

![](https://images.velog.io/images/djm0727/post/0ea32997-cd76-4af5-90ae-ee2623a42870/image.png)

* $r_y = r_y/2 + r_a/2$
* $r_a = r_y/2 + r_m$
* $r_m = r_a/2$

여기서 **미지수3개 식3개이므로 당연히! 해를 구할 수 있다**. 심지어 어렵지도 않은 1차 연립방정식에 해당한다. CS적 관점에서 볼때, 간단히 가우스소거법을 쓰면 자동화까지 할 수 있을 것 같다.

> 해치웠나? 
> No...

여기서 단점이 발생하는데, **scalable하지 못하다는** 것이다. 즉, 노드가 더 들어오게 되면, 더 많은 방정식을 풀어야한다. 따라서 해당 방법을 사용하지 않고, stochastic adjacency matrix를 이용하게 된다.

## 3. Matrix Formulation
### 3.1. stochastic adjacency matrix
stochastic adjacency matrix가 뭔데!??

당연히 궁금할 수 있다. 그냥 진짜 간단하게 **adjacency matrix에서의 값들이 확률로 표현**되어 있는 것이다.

그래도 아직도 모를 수 있다! 그래 그럴수 있다!
그래도 걱정하지 말라. 앞으로 예제를 보면 이해가 쏙쏙 될 것이니까!

### 3.2. Notation
본격적인 설명에 앞서 notation을 조금 정해보자.

* stochastic adjacency matrix: M
* page j가 갖는 나가는 링크(즉, j페이지에서 다른 페이지를 가리키는 하이퍼 링크): $d_j$
* 만약 j가 i를 가리킨다면(i도 페이지): $M_{ij} = {1 \over d_j}$
  * 따름정리: **M의 칼럼의 합은 1**이다. (당연히 확률합은 1이지! 자명!!)
* page i가 갖는 중요도 벡터: $r_i$

자 Notation이 끝났으니 핵심적인 논의를 이끌어내보자.

> 벌써 이끌어 냈다면 천재~!

### 3.3. Flow equation
조금 수학적인 센스를 발휘해보자.

앞서 우리는 flow equation을 $r_i = \sum_{j \rightarrow i} {r_j \over d_j}$라고 했다(i, j를 편의를 위해 바꾸었다.). 위에서 notation을 잠깐 보면, $M_{ij} = {1 \over d_j}$이고, flow equation에 이것이 들어있지 않은가...?

오... 신기하다! 따라서 **flow equation은 간단하게 다음과 같이 표현이 가능하다 $r = M \cdot r$**

다시! 어떤 의미인가를 한번 더 살펴보자. $M_{ij} = {1 \over d_j}$는 page j로부터 나가는 i의 중요도이다. **즉 j가 받는 모든 중요도의 합은 벡터의 내적을 통해 표현이 가능**하게 되어 위의 식이 성립하게 된다는 것이다.

## 4. Connection to Random Walk
잠깐 지난 ML with Graph(2)에서 나온 Random Walk를 생각해보자. 그리고 누군가가 인터넷 사이트를 보는 중이라고 생각해보자.

* 시간 t에서 page i에 있다고 가정하면,
* 시간 t+1에서는 page i로부터 나가는 특정한 페이지에 있을 것이다.(그것도 랜덤한 확률로!! 왜냐하면 Random walk니까!)

그렇다. **특정한 페이지에 머무르는 것을 우리는 확률로 정의할 수 있다!!!** 따라서 그 확률을 간단히 p(t)라고 하자. 이때 벡터로 모든 페이지를 확률로 표기하여 일종의 **확률 분포**를 만들어버리고 이를 $\mathbf{p}(t)$라고 하면, 앞선 matrix M과 결합하여 아주 훌륭한 식이 만들어진다.

t+1에 있을 확률분포 $\mathbf{p}(t+1) = M \cdot \mathbf{p}(t)$이다.

> "와 진짜 진짜 엄청나다"라는 생각이 들어야 한다. 정말 놀랍다... 
> 간단한 가정만으로도 논리의 비약이 없이 효율적인 논의를 이끌어가고 있다
> 더 나아가보자.

이러한 **process(확률 과정)이 언젠간 안정화 될 것이라고 가정**하자. 즉, $\mathbf{p}(t+1) = \mathbf{p}(t)$이라고 가정하자.

그렇다면 다음의 식이 성립한다. $M \cdot \mathbf{p}(t) = \mathbf{p}(t)$

이거 어디서 많이 본 식이다. 그렇다 $r = M \cdot r$의 flow equation과 정확히 동일하다.

따라서 **r은 random walk에서의 stationary distribution**이다.

## 5. Eigenvalue equation
더 더 혁신적인 아이디어는 $r = M \cdot r$을 eigenvalue equation으로 볼 수 있다는 것이다. 

우리가 알고 있는 일반적인 고유값 방정식 꼴은 $\lambda c = Ac$인데 꼴이 아주 비슷하다. 즉 $\lambda=1$인 flow equation은 고유값 방정식이 되어버린다는 것이다. 

이것은 앞선 논의와 일치하는 아주 재미있는 논리의 전개를 보여주는데, **초기에 어떤 확률분포 u**를 갖는다고 가정하자. 이때 **long-term distribution은 M(M(M... (Mu)))** 와 같이 M을 아주 많이 취한 형태가 되고, 결국에는 flow equation $1 \cdot r = M \cdot r$을 만족하게 된다는 것이다.

그리고 이러한 flow equation을 lecture 4-2에서 아주 놀라운 방식으로 간단하게 풀 수 있음을 보여주겠다.

# Lec. 4-2
## 1. Power Iteration
**최종적인 목표는 $r = M \cdot r$의 r을 구하는 것**이다. 즉, eigenvalue equation을 해를 구하는 것과 같다. 이때 우리는 앞서 본 stationary state를 이용하여 power iteration이라는 개념을 통해 풀 수 있다.

아직 이해가 안된다면 하나하나씩 뜯어보자.

시간에 따라 r이 변한다고 생각하여 다음과 같이 나타낼 수 있다. $\mathbf{r}^{(t)} = M \cdot \mathbf{r}^{(t+1)}$ 그리고 stationary state에서 분명 $\mathbf{r}^{(t)} =  \mathbf{r}^{(t+1)}$이라고 했다.

>여기까지 잘 이해가 안된다면 윗 문단을 한 번 더 읽어보자.

그런데 둘이 완벽히 같아질 수는 없을 것이다. 따라서 다음과 같이 error보다 작은 경우를 가정하여 그 때를 stationary state라고 **근사적으로 인정하자.
$\mid \mathbf{r}^{(t)} - \mathbf{r}^{(t)} \mid _1 < \epsilon$ **

> 여기서 $|x|_1$ 는 L1-norm을 의미한다.

그렇다면 우리는 근사할 때까지 r을 시간에 따라 업데이트하면서 계속 실행할 수 있을 것이다.

그런데 한 가지 빼먹은 것이 있다. "그렇다면 초기조건은? 맨 처음에는 기존 상태가 없잖아!"라는 물음에 답을 해야한다는 것이다.
**따라서 초기화는 다음과 같이 진행하면 된다. $\mathbf{r}^0 = [{1 \over N}, ..., {1 \over N}]^T$ 즉, 모든 값을 1/N으로 초기화 한 것이다.**

이제 모든 것을 정의했으니, 이를 알고리즘화 할 수 있다. 그 프로세스는 다음과 같다.

1. Initialize: $\mathbf{r}^0 = [{1 \over N}, ..., {1 \over N}]^T$
2. Iterate:  $\mathbf{r}^{(t)} = M \cdot \mathbf{r}^{(t+1)}$
3. stop when $\mid \mathbf{r}^{(t)} - \mathbf{r}^{(t)} \mid _1 < \epsilon$ 

이때 iteration은 대개 50step 정도면 충분하다.

그러나, 모두 좋은데, 다음의 Dead ends, spider traps의 문제가 발생하게 된다. 둘이 비슷한 문제를 야기하는데, **특정 노드집합에 갇혀버려서 iteration을 반복할 수록 이들의 중요도가 급증**한다는 것이다. 이것은 차차 살펴보자.

## 2. Dead Ends
Dead Ends는 특정 웹사이트에서 다른 사이트로 나가는 노드가 없을 경우 발생하는 문제이다.

즉, 인터넷을 서치하고 있는데, 다른 사이트로 갈 수 없음을 의미한다. 
잘 이해가 가지 않는다면 아래 그림을 보자.

![](https://images.velog.io/images/djm0727/post/4fdcdcef-18ae-42e6-9211-686ad1f78d20/image.png)

m노드로 **들어오는 edge는 있으나, 나가는 edge는 없게** 된다. 이것이 dead ends라고 보면 될 것이다.

이를 해결하기 위해서 Teleports라는 개념을 도입하는데, **다른 노드로 이동할 수 없으니, 강제적으로 모든 노드**들로 옮기는 것이다. 그 **확률을 모든 노드에 동일하게 나누면 된다.**

> 마찬가지로 잘 이해가 가지 않는다면 위 그림의 오른쪽 행렬을 보자.

## 3. Spider Traps
spider traps도 사실 dead ends와 맥락상 큰 차이가 없다. dead ends가 나가는 edge가 없어서 고립되었다면, spider traps은 나가는 노드는 있으나, **특정 노드 집합에 recursive한 형태로 고립**되어 버리는 것이다.

가령 a, b, c라는 노드가 a->b, b->c, c->b로 그래프를 구성한다고 가정해보자. 이때, a에서 시작하면 b와 c만을 맴돌게 된다. 따라서 그들이 모든 중요도를 다 가져갈 것이다. (_a가 중요할 수 있음에도.._)

따라서 이를 방지하기 위해서 역시 teleport를 이용하게 되는데, 이번엔 균등하게 분할 하는 것이 아니라. parameter $\beta$를 도입하여 $\beta$의 확률로 random으로 선택하고, **$1-\beta$의 확률로 teleport**를 하게 된다.

보통 **$\beta$는 0.8 ~ 0.9**의 값이 적절하다.

## 4. Google Matrix
이러한 위의 문제를 해결하기 위해서 각각의 솔루션을 **짬뽕**하여 M(stochastic adjacency matrix)을 개선하게 되는데, 이렇게 탄생한 matrix는 google matrix(G)라고 불리며 아래와 같이 정의된다.

$G = \beta M + (1- \beta)[{1\over N}]_{N \times N}$


# Lec. 4-3
위 lecture 4-1, 4-2에서 우리는 PageRank를 모두 배웠다. 그러나 아직도 개선점이 남아있는데, teleport로 dead ends랑 spider traps를 개선하는 것은 좋으나, "teleport시에 **모든 노드가 '균일'하게 확률을 갖는 것은 현실과 부합한가?**"의 물음에 답을 하지 못하기 때문이다.

즉, teleport를 할 때에 노드(pages)간의 유사도에 따라서, 더 유사한 곳으로 teleport를 시켜야 하는 것 아닌가? 라는 것이다.

따라서 이를 해결하기 위해 Personalized PageRank, Random Walk with Restarts를 배워보자.

## 1. Personalized PageRank
말한대로 node간 유사도를 알아야 한다. 따라서, 이제 page는 조금 잊어버리고, recommendation을 잠깐 가정해보자.

주된 태스크는 user와 item이 아래 그림과 같이 상호작용을 하고 있는 bipartite graph에서 "item Q를 구매한 사람에게 어떤 아이템을 추천해줄것인가?"이다. 

![](https://images.velog.io/images/djm0727/post/d5eb55b8-358c-434c-8429-93a903a3b530/image.png) 

여기서 우리가 바로 알 수 있는 핵심적인 직관은 **Q와 P를 동시에 구매한 사용자와 비슷한 특성을 가진 사용자가 Q를 구매했다면 P를 추천하는 것이 적절할 것**이라는 거다.

> 이해가 안된다면 위 문단을 두 번만 더 읽자...!

즉 users, items가 아래와 같은 관계를 갖고 있을 때 우리는 직관적으로 어떤 item이 더 추천에 적합한지(유사도가 높은지)에 대해서 알 수 있다.

![](https://images.velog.io/images/djm0727/post/1d36b00e-8821-4b38-a78d-d5dd5634d347/image.png)

> 당연히 노드간 유사도는 C-C' > A-A' > B-B' 이다.

이제 모든 상황의 설정이 완료되었다. 최종적으로 우리는 **"bipartite graph에서 특정 아이템과 유사한 다른 아이템을 찾는 것"** 을 알아내어야 한다.

따라서 PageRank의 아이디어를 차용할 수 있을 것이다. **특정 아이템에서 시작하여 반복하면서 다른 item들을 Random walk한다면, 그 방문의 횟수를 카운트 하여 유사도를 판단할 수 있을 것이다.**

하지만 앞에서도 말했듯이, 모든 노드가 **teleport에서 균등한 확률을 갖는다면, 현실과 부합하지 않아** 효율적인 결과가 나오지 않을 것이다.

따라서 새롭게 도입한 Personalized PageRank의 핵심적인 아이디어는, **teleport할 때에 유사한 노드집합 S**로 해야한다는 것이다.

따라서 아래와 같은 알고리즘이 완성된다.

0. 노드집합 S는 주어진다. 시작점은 Q이다.
1. Random walk를 한 단계 실행한다. 이때, $1-\beta$의 확률로 노드집합 S로 돌아가게 된다.
2. 1번의 작업을 n번 반복하고 나온 결과에서 각 노드의 방문 횟수를 벡터화 하면, Q와 다른 노드의 유사도 벡터가 된다.

## 2. Random Walk with Restarts
너어어어ㅓ어어어ㅓ 무 간단하다 그냥 Personalized PageRank에서 노드 집합에 Q만 넣으면 된다. 나머지는 모두 같다.


## 3. Benefits
이것이 유효한 이유는 아래와 같은 아주 많은 요소들을 고려하기 때문이다.

* Multiple connection
* Multiple path
* Direct & indirect connections
* Degree of the node

## 4. Summary
이번 lecture 4-3에서 소개한 두 가지 방법과 PageRank는 transport의 관점에서만 차이가 존재한다. 그러한 노드집합 S를 확률로 표시하면 아래와 같다.

* PageRank: S = [0.2, 0.2, 0.2, 0.2, 0.2]
* Personalized PageRank: S = [0.5, 0.4, 0.1, 0, 0]
* Random Walk with Restarts: S = [1, 0, 0, 0, 0]

> Lec. 4-4는 통합적 얘기가 많아 생략하겠다. 주된 논의는 아래에 주석으로 정리 함.

> * node embedding에서 $A \approx Z^TZ$와 같이 matrix factorization으로 나타낼 수 있다.
> * deepwalk도 matrix factorization으로 나타낼 수 있으나 복잡하다(node2vec도 마찬가지). 
>   * $log(vol(G)({1 \over T} \sum_{r=1}^T(D^{-1}A)^r)D^{-1}) - logb$
> * Limitation이 존재한다.
>   * training data에 없는 노드는 임베딩할 수 없다.
>   * 구조적 유사성을 알 수 없다(궁금하면 최하단 그림 참조). 
>   * 노드에 포함된 feature vector를 고려하지 못한다.
> * 암튼 이런 노드를 GNN, Deep Representation Learning에서 해결할 예정이다! 



> ![](https://images.velog.io/images/djm0727/post/197652fe-99e9-4242-8906-b682770a9bcb/image.png)


#### Reference
* http://web.stanford.edu/class/cs224w/