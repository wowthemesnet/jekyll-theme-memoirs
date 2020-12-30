---
layout: post
title: Noisy Gradient 다루기
author: wontak ryu
categories: [ deeplearning]
image: assets/images/2020-12-29-Gradient-Accumulation/noisy_gradient.jpeg
---

## 들어가며

안녕하세요. 마키나락스의 류원탁입니다.

딥러닝 문제를 다루다보면, gradient의 분산이 커져 학습이 불안정적으로 진행되는 경우가 발생합니다.

본 블로그에서는 noisy gradient에 대해서 살펴보고, 해결할 수 있는 방법에 대해서 다루겠습니다.


## Noisy Gradient Problem

일반적으로 수렴하는 네트워크의 경우 아래의 이미지와 같이 학습이 진행됩니다.



<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-12-29-Gradient-Accumulation/gradient.jpeg" alt="normal gradient" width="40%">
  <figcaption style="text-align: center;">normal gradient</figcaption>
</p>
</figure>



반면에, gradient의 분산이 커진다면 아래의 이미지처럼 수렴이 제대로 진행되지 않을 것입니다. 이런 문제를 본 글에서는 'noisy gradient'라고 부르겠습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img style="width: 40%" src="/assets/images/2020-12-29-Gradient-Accumulation/noisy_gradient.jpeg" alt="noisy gradient">
  <figcaption style="text-align: center;">noisy gradient</figcaption>
</p>
</figure>



실제로 NLP영역에서 많이 활용되고 있는 transformer 구조도 학습시키는게 쉽지 않다고 합니다. 이는 초기 학습에서 발생하는 noisy gradient가 원인입니다. [3]


## 사례: Residual AutoEncoder with FC layer

마키나락스에서도 유사한 문제를 겪었습니다. 내부에서 autoencoder 기반의 anomaly detection task를 수행하고 있습니다. 더 깊은 모델을 사용하고자 기존에 사용하던 autoencoder 모델들에 residual connection을 추가해봤습니다. 하지만, 예상치 못한 문제가 발생했습니다. 레이어가 깊어질수록 학습이 매우 불안정적으로 진행되었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img style="width: 70%" src="/assets/images/2020-12-29-Gradient-Accumulation/residual_ae.jpeg" alt="rae">
  <figcaption style="text-align: center;">Residual AE</figcaption>
</p>
</figure>
  
결론적으로 학습에 발생하는 gradient가 noisy하다는 문제가 있었습니다. 그렇다면, 기존의 모델과 대비해서 왜 이런 문제가 발생했는지 살펴보겠습니다.

아래와 같이 4개의 레이어로 이루어진 간단한 모델이 있다고 가정해보겠습니다. 이 모델은 아래와 같은 forward과정을 가질 것입니다.

- 레이어: $h_0, h_1, h_2, h_3$
- 각 레이어의 입력: $x_0, x_1, x_2, x_3$

$$
x_0 \rightarrow h_0(x_0) = x_1 \rightarrow h_1(x_1) = x_2 \rightarrow h_2(x_2) = x_3 \rightarrow h_3(x_3) = x_4 \rightarrow loss
$$

이 모델에 residual connection을 적용하면 아래와 같은 forward 과정을 가집니다.

- 레이어: $h_0, h_1, h_2, h_3$
- 각 레이어의 입력: $x_0, x_1, x_2, x_3$

$$
x_0 \rightarrow h_0(x_0) + x_0 = x_1 \rightarrow h_1(x_1) + x_1 = x_2 \rightarrow h_2(x_2) + x_2 = x_3 \rightarrow h_3(x_3) + x_3 = x_4 \rightarrow loss
$$


이 모델의 backpropagation과정을 살펴보면 아래와 같습니다.

$$
\frac{d loss}{d x_0} =\frac{d loss}{d x_4} \frac{d x_4}{d x_0} = 

\frac{d loss}{d x_4} \frac{d(x_0 + h_1(x_0) + h_2(x_1) + h_3(x_2) + h_4(x_3))}{d x_0}
$$


각 layer마다 발생하는 gradient vector를 random variable로 보면, 위의 gradient vector의 분산은 아래의 식으로 구할 수 있습니다.

우선, 두 random variable의 합의 분산은 아래와 같이 전개됩니다.

$$
Var(aX + bY) = a^2 Var(X) + b^2 Var(Y) + 2abCov(XY)
$$

이를 위의 backpropagation의 식에 대입해보면, 기존 모델 대비 더 큰 분산을 가지는 것을 알 수 있습니다.

$\frac{dh_1(x_0)}{dx_0}, \frac{dh_2(x_1)}{dx_0}, \frac{dh_3(x_2)}{dx_0}, \frac{dh_3(x_3)}{dx_0}$ 각 요소들이 모두 유사한 스케일을 가진다고 가정해보면, 최소 레이어 수 배 만큼 큰 분산을 가진다고 할 수 있습니다.


## Method: Gradient Accumulation

이런 문제를 해결하기 위해서 크게 3가지 정도의 시도를 했으나, 가장 효과적이고 보편적으로 사용할만한 것은 batch size를 키우는 것이였습니다.

- 모멘텀을 사용하지 않는 옵티마이저를 사용하여 안정적인 학습을 진행했으나, 수렴정도가 만족스럽지 않았습니다.
- warm up을 사용하여 안정적인 학습과 수렴정도도 만족스러웠으나, 하이퍼파라미터에 민감하다는 단점이 있었습니다.


batch size를 키우게 되면, 통계학적으로 표준편차가 주는 효과가 있습니다.

$$
std = \frac{\sigma}{\sqrt{n}}
$$

따라서, batch size를 키우게되면, 학습이 진행되는 중에 발생하는 nosisy gradient가 경감되는 것을 알 수 있습니다. 다른 연구에서도 batch size가 커지면 학습이 불안정하던 학습이 안정적으로 진행되는 것을 보였습니다. [1]


batch size를 키우는 것은 좋지만, gpu의 memory는 한정적입니다. 따라서, 한정된 gpu memory내에서 batch size를 키우는 효과를 내기 위해서, gradient accumulation이라는 방법을 사용했습니다. 

gradient accumulation은 매 step마다 파라미터를 업데이트 하지않고, gradient를 모으다가 일정한 수의 graidient vector들이 모이면 파라미터를 업데이트합니다.

구체적인 알고리즘을 알고싶다면, 아래의 예시코드를 참고하시기 바랍니다.

```python

model.zero_grad()                                   
for i, (inputs, labels) in enumerate(training_set):
    predictions = model(inputs)                     
    loss = loss_function(predictions, labels)      
    loss = loss / accumulation_steps                
    loss.backward()                                 
    if (i+1) % accumulation_steps == 0:             
        optimizer.step()                            
        model.zero_grad()                            
```


## Result

gradient accumulation을 통해서 불안정적이던 학습을 안정적으로 진행할 수 있었습니다. 또한, 위에서 언급한 Residual VAE를 안정적으로 학습하여 기존의 VAE보다 우수한 성능을 보일 수 있었습니다.

### Gradient Accumulation

아래의 학습 그래프는 residual VAE를 기존의 방법대로 학습시킨 것입니다. 보이는 것처럼 학습이 매우 불안정적으로 진행됩니다.

<figure class="image" style="align: center;">
<p align="center">
  <img style="width: 70%" src="/assets/images/2020-12-29-Gradient-Accumulation/vanilla.png" alt="rae">
  <figcaption style="text-align: center;">vanilla training</figcaption>
</p>
</figure>



반면에, graidient accumulation을 적용하게 되면, 아래의 그래프처럼 안정적으로 학습이 진행됩니다.

<figure class="image" style="align: center;">
<p align="center">
  <img style="width: 70%" src="/assets/images/2020-12-29-Gradient-Accumulation/gradient_accumulation.png" alt="rae">
  <figcaption style="text-align: center;">gradient accumulation</figcaption>
</p>
</figure>


### Residual VAE vs VAE

Residual VAE와 VAE의 실험을 비교해봤습니다. 

- Dataset은 MNIST를 사용했습니다.
- 실험셋팅은 class 0, 1을 target class를 두고 학습하였으며, 아래의 값은 그것의 평균값입니다.

<figure class="image" style="align: center;">
<p align="center">
  <img style="width: 70%" src="/assets/images/2020-12-29-Gradient-Accumulation/table.png" alt="rae">
  <figcaption style="text-align: center;">Residual VAE vs VAE</figcaption>
</p>
</figure>



## 끝으로

이번 글에서 학습과정에서 발생하는 noisy gradient에 대해서 다뤘습니다. 이를 해결하기 위해서 batch size를 키우기 위한 노력을 했습니다. 그 과정에서 발생하는 gpu memory 문제를 해결하기 위해서 gradient accumulation을 활용했습니다.

혹시 모델의 학습이 불안정하다면, 위와 같은 방법을 고려해보는 것을 추천드립니다.

## Reference

[1] [RAdam](https://github.com/LiyuanLucasLiu/RAdam)

[2] [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)

[3] [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)

[4] [Gradient Accumulation](https://towardsdatascience.com/gradient-accumulation-overcoming-memory-constraints-in-deep-learning-36d411252d01)


