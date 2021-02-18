---
layout: post
title: Regression Test, Are you sure?
author: wontak ryu
categories: [test]
image: assets/images/2020-02-10-Performance-Test/13_.gif
---

## 들어가며

안녕하세요. 마키나락스의 류원탁입니다.

마키나락스는 AI Project를 넘어 AI Product로 나아가고 있습니다. Product로 나아가는 여정 속에서 재미있는 엔지니어링 이슈들이 생겼습니다.
특히, 마키나락스가 제공하는 Machine Learning Software의 성능 대해서 신속하고 정확한 검증에 대한 필요가 있었습니다. 마키나락스에서는 Unittest는 기본적으로 적용하고 있지만, Unittest만으로는 성능을 보장 할 수 없습니다.

이번 포스트에서는 Machine Learning Software에 대한 성능검증을 어떤 방식으로 진행하고 있는지 공유드리도록 하겠습니다.


## Problem: Can't find the cause of the lower performance!

우선, 내부에서 겪었던 문제에 대해서 공유드리도록 하겠습니다. 마키나락스에서는 협업기반의 개발문화를 가지고 있습니다. 개발을 하면서, Branch들이 Pull-Request를 날리고 Review하고 Merge되는 과정을 반복하게 됩니다.


<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/1.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림1] - Gitflow Workflow [16]</figcaption>
</p>
</figure>

내부에 Unittest가 구현되어 있어서, 어느정도의 안전성이 검증된 상태였습니다. 하지만, 마키나락스는 Machine Learning Software를 다루는 조직이기 때문에 더 높은 수준의 Test System이 필요했습니다. 


만약 Unittest는 통과하지만, 성능저하를 일으키는 Commit같은 경우에는 추적을 할 수 없다는 문제가 있습니다. 아래의 [그림2]는 실제로 마키나락스에서 겪었던 문제입니다. 여러가지 Branch와 Commit들이 혼재된 상황속에서 성능이 저하되었다는 것을 발견하였습니다. 

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/2.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림2] - Problem in Makinarocks</figcaption>
</p>
</figure>

이런 문제가 발생했을 때, Debugging 해야할 Search Space는 [그림3]과 같습니다. 즉, 이전까지의 변화를 모두 살펴봐야 했습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/3.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림3] - Search Space</figcaption>
</p>
</figure>

이번 문제의 경우에는 성능저하를 일으키는 원인이 두 개의 Commit이였습니다. 성능저하가 복합적인 원인들에 의해서 발생하면, 해결하는데 더 큰 어려움을 겪습니다. 두 개의 Commit을 모두 고치지 않으면, 성능저하 이슈는 계속해서 발생하기 때문입니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/4.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림4] - Causes</figcaption>
</p>
</figure>


결국, 수많은 Debugging 끝에 원인을 찾을 수 있었습니다. 그리고 원인들은 생각보다 사소한 변화였습니다. 일반적으로 생각해봤을 때 큰 문제를 야기할 것이라고 생각하기 어려운 부분이였습니다. 아마도 그렇기 때문에, 뒤늦게 발견이 되었을 것이라고 생각합니다. 더욱이 이런 Commit들이 모두 Unittest를 통과했기 때문에, 개발자 입장에서 무엇이 원인인지 파악하기 힘듭니다.

이런 경험을 한 후에, **Are you Sure?** (이 코드 문제가 없을까요?)라는 질문에 답하기 위해서는 사실상 작은 변화더라도 **Regression Test**를 진행해야 하는 것을 깨달았습니다. 

여기서 Regression Test라고 정의한 것은 Machine Learning Software의 전체적인 실험 및 테스트를 진행하고 성능을 확인하는 작업을 의미합니다.

변화량에 대해서 Debugging 비용을 그래프로 그려보면 [그래프1]이 나옵니다. 위에서 언급했듯이, 복합적인 원인에 의해서 성능저하 이슈가 발생했을 경우, 모든 변화의 조합에 대해서 실험 및 테스트를 진행해야합니다. (이 때, 모든 변화는 독립적이라는 가정을 하고 계산하였습니다.)

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/5.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림5] - 경우의 수</figcaption>
</p>
</figure>
<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/graph1.png" alt="Gitflow Workflow" width="40%">
  <figcaption style="text-align: center;">[그래프1] - Cost of Debugging</figcaption>
</p>
</figure>



그럼 "언제 Regression Test를 진행해야하는가?" 라는 의문이 들 수 있습니다. [그래프1]을 근거로 내부에서는 "As Soon As Possible"(가능한 빨리)라는 결론을 내리게 되었고, 이를 위해서 Regression Test를 위한 Pipeline을 구성하였습니다.


## Trial and Errors

Regression Test Pipeline을 만들기 위해서, 여러가지 시행착오를 겪었습니다. 겪었던 시행착오를 통해서, 필요했던 **추상화 과정**에 대해서 설명드리겠습니다.

우선, 자동화 도구로 Jenkins를 활용하였습니다. [[1]](#ref-1) Jenkins는 소프트웨어 개발 시 지속적으로 통합 서비스를 제공하는 툴입니다. 비교적 높은 자유도가 있었고, 자동화 도구로 접근성이 좋다고 판단했습니다. 

### Pipeline #1: Dependent on Repository

첫 번째로 구현한 Pipeline은 아래 [그림6]에서 볼 수 있습니다. Jenkins Container가 Regression Test 대상이 되는 Repository의 Requiremnts를 미리 가지고 있습니다. 학습에 필요한 데이터의 경우 NAS에 저장해두고 요청 시 접근하여 사용합니다. GitHub에서 Test요청을 보내면, Regression Test를 진행하게 됩니다. 이런 구조는 한 Repository에 의존성을 가지게 된다는 문제를 가지고 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/6.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림6] - Pipeline #1</figcaption>
</p>
</figure>

### Pipeline #2: Independent on Repository, But Inefficient

두 번째로 구현한 Pipeline은 아래 [그림7]에서 볼 수 있습니다. Pipeline #1과 다르게 Jenkins Container가 Repository에 정의된 Dockerfile을 기반으로 Regression Test Container를 만듭니다. 이를 통해서 Repository에 의존성을 가지던 문제를 해결할 수 있었습니다. 하지만, Docker Image를 Build하는 작업은 상당히 오랜시간이 걸리기 때문에, 비효율적이라는 문제가 있었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/7.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림7] - Pipeline #2</figcaption>
</p>
</figure>


### Pipeline #3: InDependent on Repository, But!

첫 번째로 구현한 Pipeline은 아래 [그림7]에서 볼 수 있습니다. Docker Image는 Requirements가 변경되었을 때만 Update가 필요했습니다. 따라서, 미리 DockerImage를 만들어두고, Jenkins Container가 이를 받아서 사용하도록 변경하였습니다. Pipeline #3과 비교해봤을 때, 효율적이었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/8.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림8] - Pipeline #3</figcaption>
</p>
</figure>

### Device Dependency

하지만, Pipeline #1 ~ #3이 모두 공통적으로 한 컴퓨팅 자원에 의존적이라는 문제가 있었습니다. 예를 들어서, Regression Test에 사용하는 컴퓨팅자원에 만약 다른 작업이 돌아가고 있었다면, Regression Test가 아예 작동하지 못하거나, 다른 작업을 망칠 수도 있습니다. [그림9]를 보면, 3개의 노트북이 MRX-Desktop1에 접속하여 사용하고 있는 모습을 볼 수 있습니다. 만약 Jenkins Container가 MRX-Desktop1에서 작동하고 있다면, Regression Test가 정상적으로 작동하지 않을 것입니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/9.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림9] - Problem of Device Dependency</figcaption>
</p>
</figure>

또한, 다른 MRX-Desktop2, 3를 보면 컴퓨팅 자원이 여유있다는 것을 알 수 있습니다. 이런 자원을 효율적으로 사용하기 위해서, Device Dependency를 제거하는 작업이 필요했습니다.

### Pipeline #4: InDependent on Device

Device Dendency를 해결하기 위해서, Kubernetes를 사용하였습니다. [[2]](#ref-2) 대략적으로 Kubernetes에 대해서 알고 싶으신 분들은 다음 Reference를 참고하시는 것을 추천드립니다. [[6]](#ref-2)

Kubernetes를 사용한 목적은 내부의 컴퓨팅 자원을 추상화하기 위함입니다. 쉽게 풀어쓰면, **Kubernetes에 특정 Device를 요청하는 것이 아니라, 필요한 컴퓨팅 자원에 대해서 요청만 하면, 그에 맞는 자원할당을 받기 위해서입니다.** [그림10]을 보면, 여러가지 컴퓨팅 자원이 하나의 클러스터로 묶여있습니다. 이제 원하는 자원의 스펙을 적으면, 그에 맞는 자원이 할당될 것입니다.



<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/10.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림10] - Kubernetes in Makinarocks </figcaption>
</p>
</figure>

Jenkins Container의 역할은 특정 Device내에서 Container로 Regression Test를 진행하는 것이 아닙니다. Jenkins Container는 미리 정의된 컴퓨팅 자원 스펙에 해당하는 Ray Cluster를 만드는 것입니다. [[3]](#ref-2) 여기서 Ray Cluster의 역할은 Regression Test를 병렬적으로 진행하기 위한 목적으로 사용되고, 작업이 끝나게 되면 Ray Cluster는 사라지게 됩니다. 참고로 [그림10]에서 구성한 Cluster와 Ray Cluster는 다른 역할을 합니다. [그림10]은 자원자체를 묶는 작업을 의미한다면, Ray Cluster는 이미 묶인 자원을 활용하는 것입니다. 

**이번 포스팅에서 Kubernetes와 Ray Cluster에 대해서 자세히 다루지는 않겠지만, 수요가 있다면 마키나락스에서 Kubernetes를 활용하는 방법을 다룰 예정입니다. 혹시 관심있으신 분은 댓글 남겨주시면 감사하겠습니다.**

이제 Kubernetes 그리고 Ray Cluster를 활용하여, [그림11]과 같은 Pipeline을 구축하였습니다. Repository에 의존성을 제거하였으며, Docker Image도 미리 만들어둔 Image를 활용하였습니다. 또한, Device에 대한 의존성을 제거하여, 내부의 컴퓨팅 자원을 더욱 효율적으로 사용할 수 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/11.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림11] - Pipeline #4 </figcaption>
</p>
</figure>


## (Selected) Method: Self-Hosted Runner in GitHub Action

위에서, Pipeline #1 ~ #4까지 살펴볼 수 있었습니다. 하지만, Jenkins에 익숙하지 않다보니 기술적인 이슈가 발생할때 대처하는데 쉽지 않았습니다. 특히 Kubernetes환경에서 jenkins를 활용하기 위해서는 조금 더 많은 지식이 필요했습니다. 아쉽게도 이에 대한 문서를 쉽게 찾을 수 없어서, 유지보수 측면에서 아쉬움이 있었습니다.

그러던 중, GitHub Action에서 Self-Hosted Runner라는 서비스를 제공하는 것을 발견했습니다.[[4]](#ref-2) Self-Hosted-Runner는 가지고 있는 자원을 통해서 Github Action 진행할 수 있었습니다. 상대적으로 GitHub에서 관련내용에 대해서 문서를 제공하였고, 문법도 직관적이라고 생각이 들었습니다. 이런 특징들은 유지보수 관점에서 높은 점수를 줄 수 있었고, 기존의 Jenkins의 역할을 GitHub Action으로 대체하기로 하였습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/12.png" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림12] - Pipeline #5 </figcaption>
</p>
</figure>


## Are You Sure? Yes!

GitHub Action에서 Trigger Event Type에 대해서 정할 수 있습니다. 여러 논의 끝에, Workflow Dispatch라는 Type을 선택하였습니다. 이제 마우스 클릭으로 GitHub Web에서 Regression Test를 실행할 수 있습니다. [[5]](#ref-2)

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/13.gif" alt="Gitflow Workflow" width="60%">
  <figcaption style="text-align: center;">[그림13] - Click for Regression Test </figcaption>
</p>
</figure>

Regression Test Pipeline의 모습을 [그림13]으로 도식화해봤습니다. GitHub에서 미리 설정한 Event Type에 해당하는 Event가 발생하면, MRX-Hosted-Runner에게 Regression Test를 요청합니다. MRX-Hosted-Runner는 Ray Cluster를 구성합니다. 학습 및 실험을 진행할 때는 MLflow에 실험정보를 로깅하고, 학습이 끝나면 이에 대한 정보를 GitHub에 전달합니다. 현재는 해당 PR에 Comment로 MLflow 실험링크를 달아주는 방식으로 사용중입니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/14.png" alt="Gitflow Workflow" width="80%">
  <figcaption style="text-align: center;">[그림14] - Pipeline Overview </figcaption>
</p>
</figure>


앞서, Regression Test가 없는 상황에서 Search Space는 [그림3]으로 표현될 수 있습니다. 그렇다면, Regression Test를 진행한다면, Search Space는 어떻게 변할까요? 각각의 Feature Branch마다 Regression Test가 진행된다는 것을 가정해보면, [그림14]처럼 Search Space가 줄어듭니다. [그래프1]에서 봤듯이, 탐색할 변화량과 디버깅 비용이 지수함수 관계라는 것을 고려해보면, 상당히 많은 비용이 절약될 수 있음을 알 수 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-02-10-Performance-Test/15.png" alt="Gitflow Workflow" width="80%">
  <figcaption style="text-align: center;">[그림15] - Reduced Search Space </figcaption>
</p>
</figure>

또한 Regression Test에서 정상작동한 Branch에 대해서, **Are You Sure?** 라고 누가 묻는다면 이제는 자신있게 **Yes!**라고 할 수 있습니다.


## 마치며

이번 포스팅에서는 Machine Learning Software의 Regression Test에 대해서 다뤘습니다. 

개발을 하다보면, 의도치 않은 부작용들이 발생하게 되며 이런 부작용은 발견하기 어렵습니다. 이런 부채들이 쌓인 후 디버깅하게 되면, 생각보다 훨씬 큰 비용을 치뤄야 합니다. 이런 문제를 보다 효과적으로 대처하기 위해서 Regression Test를 제안하게 되었습니다.

Regression Test Pipeline을 구성하기 위해서, 여러가지 추상화과정이 필요했습니다. Repository에 독립적으로 작동할 수 있어야 했습니다. 또한, Machine Learning Software(AI)는 많은 컴퓨팅 자원을 요구하기 때문에, 효율적인 자원사용이 필요했습니다. 이를 위해서 Kubernetes를 활용하여 컴퓨팅 자원을 가상화하였습니다. 

Regression Test를 통해서 Search Space를 줄일 수 있었고, $2^\text{ReducedSearchSpace}$ 만큼의 Debugging Cost를 줄일 수 있었습니다. 그리고 클릭 한 번으로 실험을 진행할 수 있다는 것도 굉장히 매력적인 일이였습니다.

이번 포스트를 통해서 비슷한 문제를 고민하는 분들께 작은 도움이 되었으면 좋겠습니다.


## Reference

<a name="ref-1">[1]</a>  [jenkins[websites], (2020, Feb, 10)](https://www.jenkins.io/)

<a name="ref-2">[2]</a>  [Kubernetes[websites], (2020, Feb, 10)](https://kubernetes.io/)

<a name="ref-3">[3]</a>  [Ray cluster[websites], (2020, Feb, 10)](https://docs.ray.io/en/master/cluster/index.html)

<a name="ref-4">[4]</a>  [About Self Hosted Runners[websites], (2020, Feb, 10)](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners)

<a name="ref-5">[5]</a>  [Workflow Dispatch[websites], (2020, Feb, 10)](https://docs.github.com/en/actions/reference/events-that-trigger-workflows#workflow_dispatch)

<a name="ref-6">[6]</a>  [What is Kubernetes[websites], (2020, Feb, 10)](https://www.google.com/search?q=what+is+kubernetes&oq=what+is+kubernetes&aqs=chrome..69i57j0l5j69i60l2.7878j0j4&sourceid=chrome&ie=UTF-8)
