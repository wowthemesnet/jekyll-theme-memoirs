---
layout: post
title: Integrated Gradient
author: wontak ryu
categories: [ deeplearning, XAI ]
image: assets/images/2020-05-05-Integrated-gradient-정리글/integrated_gradient.png
---


XAI을 위한 다양한 방법론들이 있습니다. 하지만, 정량적으로 이들을 비교하고 성능을 평가하는 것은 쉽지 않습니다. 일반적인 모델들은 accuracy를 가지고 판단할 수 있지만, attribution을 생성하는 모델의 경우에는 정량적으로 평가하기 쉽지 않습니다.

이번글에서는 Attribution 방법론에서 중요한 요소들에 대해서 정리해보고, 이러한 요구조건에 부합하는 integrated gradient에 대해서 설명드리겠습니다.

## Motivation

Attribution을 파악한다는 것은 모델의 input과 output간의 관계를 파악하는 것입니다. 즉, model이 예측을 할 때, 어떤 input feature가 해당 예측에 큰 영향을 주었는지 파악하는 것이 주요한 목적입니다.

Attribution은 아래와 같이 정의될 수 있습니다.

- deep network: $F: R^n \rightarrow [0, 1]$
- input: $x = (x_1, \cdots, x_n) \in R^n$
- baseline input: $\acute{x}$

$$
A_F(x, \acute{x}) = (a_1, \cdots, a_n) \in R^n
$$

$a_1, \cdots, a_n$은 feature importance와 같은 개념입니다.

baseline이란, 일종의 비교대상입니다. 예를 들어, object recognition에 경우에 input image의 어느 pixel이 특정 class라고 판단하게 하는지 구할 수 있습니다.  일반적인 경우에는 baseline 이미지는 zero pixel로 두어 구하기도 합니다.

![](/assets/images/2020-05-05-Integrated-gradient-정리글/ex1.png "Example")

## Two Fundamental Axioms

해당글에서는 크게 두 가지의 조건에 대해서 알아볼것입니다. 

- Sensitivity
- Implementation Invariance

### Sensitivity

baseline과 input과의 차이가 오직 하나의 feature이고 baseline의 예측과 input의 예측이 다른경우를 가정해보겠습니다.  이런 상황에서는 차이나는 feature가 모델의 예측에 영향을 끼쳤다고 생각할 수 있습니다.  이렇게 차이나는 feature는 non-zero의 attribution의 값을 가져야합니다. 그리고 이러한 조건이 만족된다면, sensitivity 조건을 만족하게 됩니다.



#### Gradients

gradient는 model coefficient를 쉽게 알 수 있는 방법입니다. backpropagation을 통해서 작업하게 되면, 쉽게 input의 gradient값을 구할 수 있습니다. 하지만, sensitivity 하지 않다는 단점을 가집니다.

![](/assets/images/2020-05-05-Integrated-gradient-정리글/gradient.png)

위와 같은 함수가 있다고 가정해보겠습니다.  baseline $x=0$이 주어졌을때, input $x=2$의 gradient를 구해보겠습니다.  $x=2 $에서 함수는 평평하므로 gradient값은 0이 됩니다. 하지만, baseline을 고려해보면, 함수값의 차이는 1이 되므로, attribution은 non-zero여야 합니다.



### Impementation Invariance

서로 다른 network이지만 같은 input ~ output 관계를 가진다면, 두 network는 동일한 attribution을 가져야 합니다. 이러한 특성을 implementation invariance라고 합니다.

![](/assets/images/2020-05-05-Integrated-gradient-정리글/implementation_invariance.png)

gradient는 sensitivity하지는 않지만, chain rule을 가지기 때문에, implementation invariant하다는 장점을 가지고 있습니다.  수식으로 살펴보겠습니다.

- model output: $f$
- model input: $g$
- network: $h$

$$
\frac{\partial f}{\partial g} = \frac{\partial f}{\partial h} \cdot \frac{\partial h}{\partial g}
$$



하지만, LRP 혹은 DeepLIFT와 같은 방법론은 chain rule이 성립하지 않습니다.


$$
\frac{f(x_1) - f(x_0)}{g(x_1) - g(x_0)} \ne \frac{f(x_1) - f(x_0)}{h(x_1) - h(x_0)} \cdot \frac{h(x_1) - h(x_0)}{g(x_1) - g(x_0)} \text{   for all  } x_1, x_0
$$


아래의 이미지는 이를 실험한 결과입니다.

![](assets/images/2020-05-05-Integrated-gradient-정리글/figure7.png)



만약, implementation에 따라서 attribution이 달라진다면, input-output관계가 아닌 것에 영향을 받는 것이고 이는 중요하지 않는 feature에서 attribution이 높게 나올수 있음을 의미합니다.



## Integrated Gradients

위에서 gradient가 implementation invariant하지만 sensitivity의 속성은 충족시키지 못하는 것을 살펴봤습니다.  integrated gradient는 이러한 한계를 극복하면서 implementation invariance를 유지하는 방법입니다.

![](/assets/images/2020-05-05-Integrated-gradient-정리글/integrated_gradient.png)

직관적으로 살펴보면, integrated gradient는 baseline에서 input까지의 모든 gradient를 고려하는 방법입니다. 결과적으로 각 path의 gradient를 모두 고려하므로 특정 지점에서 gradient값이 0이 되는 이슈를 해결할 수 있으면서, gradient를 활용하므로 implementation invariance합니다.


$$
IntegratedGrads_i(x) = (x_i - \acute{x}_i) \times \int_{\alpha=0}^1 \frac{\partial F(\acute{x} + \alpha(x-\acute{x}) )}{\partial x_i} d\alpha
$$


### Completeness

integrated gradient는 completeness라는 재밌는 특성을 가집니다.  integrated gradient로 나오는 attribution들을 모두 더하면, 결국은 두 모델의 예측값의 차이가 됩니다.
$$
\sum_{i=1}^n IntegratedGrads_i(x) = F(x) - F(\acute{x})
$$


참고로, Completness는 sensitivity를 내포하고 있습니다. Completeness는 함수값의 차이는 attribution의 합이어야 합니다. sensitivity는 한 feature가 예측값의 차이를 발생시켰다면 해당 attribution은 non-zero어야합니다. 



### Uniqueness of Integrated Gradients

이미지 인식에서 attribution method를 평가하는 방법으로 다음과 같은 방법이 있습니다.

- Attribution score가 높은 pixel을 제거해나가면서, 성능의 저하를 확인합니다.
- 좋은 attribution 방법이라면, model score가 급격히 떨어질 것으로 기대합니다.



하지만, 이런 방법은 문제가 있습니다. pixel을 제거한 이미지의 성능저하가 attribution이 좋아서 그런 것인지 아니면 처음 본 이미지 형식이라서 그런지 알 수 없다는 것입니다. 이런 이유로 해당논문은 수치적으로 attribution 방법을 증명하기 보다는 sensitivity와 implementation invariance의 특성을 만족하는 것으로 integrated gradient의 정당성을 설득합니다.



이런 문제의식을 바탕으로 해당 연구에서는 두 가지 단계의 논리를 전개합니다.

1. path method 소개
2. path method중에서 integrated gradient가 선택된 이유





### Path Methods

모든 path methods는**implementation invariance** 성질을 만족합니다.  또한 path method만이 sensitivity와 implementation invariance를 모두 만족할 수 있다고 주장합니다.

> Theorem 1 (Friedman, 2004))  
>
> Path methods are the only attribution methods that always satisfy
> Implementation Invariance, Sensitivity, Linearity, and Completeness.  

integrated gradient도 path method중 하나이며,  위의 이미지에서 $P2$ linear combination의 path에 해당합니다.

![](/assets/images/2020-05-05-Integrated-gradient-정리글/figure1.png)

- $\gamma = (\gamma_1, \cdots, \gamma_n) : [0, 1] \rightarrow R^n$
- $\gamma(0) = \acute{x}$
- $\gamma(1) = x$

$$
PathIntegratedGrads_i^\gamma(x) = \int_{\alpha=0}^1 \frac{\partial F(\gamma(\alpha))}{\partial \gamma_i(\alpha)} \frac{\partial \gamma_i(\alpha)}{\partial \alpha} d\alpha
$$



#### Linearity

$f_1, f_2$ 의 두 모델이 있다고 가정해보겠습니다. 그리고 이를 바탕으로 $f_3 = a \times f_1 + b \times f_2$를 만들었습니다. $f_3$에 대해서 attribution을 구하면, f1과 $f_2$의 attribution에서 각각 $a, b$만큼의 가중치를 부여해서 구할 수 있는 속성입니다.



### Integrated Gradients is Symmetry-Preserving

$x, y$가 $F$에 대해서 대칭이라면, 다음과 같이 나타낼 수 있습니다.
$$
F(x, y) = F(y, x)
$$
attribution method는 동일한 symmetry value를 가지고 있고 baseline의 symmetric variable이 동일한 attribution을 가진다면, symmetry preserving하다고 합니다.

예시) 
$$
Sigmoid(x1 + x2, \cdots)
$$
$x_1, x_2$는 symmetric variable이고 input에서는 $x_1=x_2=1$ 이며, basline에서는 $x_1=x_2=0$입니다. symmetry preserving하다면, $x_1, x_2$에 모두 동일한 attribution 값이 나와야합니다.



그리고, integrated gradient는 이러한 조건을 만족합니다. 아래를 간략히 정리하면, non-straightline은 symmetry preserving하지 않다는 것입니다.

![](/assets/images/2020-05-05-Integrated-gradient-정리글/proof1.png)



## Computing Integrated Gradients

$$
IntegratedGrads_i^{approx}(x) = (x_i - \acute{x_i}) \times \sum_{k=1}^m \frac{\partial F(\acute{x} + \frac{k}{m} \times (x-\acute{x}) )}{\partial x_i} \times \frac{1}{m}
$$

- $m$: the number of steps in the Riemman approximation

실제 어플리케이션에서는 위와 같은 과정을 통해서 근사합니다.






## Reference

- [1]  Axiomatic Attribution for Deep Networks  2017