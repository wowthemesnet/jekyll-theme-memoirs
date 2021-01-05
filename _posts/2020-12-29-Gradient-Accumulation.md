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

본 포스트에서는 noisy gradient에 대해서 살펴보고, 해결할 수 있는 방법에 대해서 다루겠습니다.


## Noisy Gradient Problem

 Stochastic Gradient Descent(SGD)를 활용한 경우, 일반적으로 미니배치의 gradient는 전체 데이터셋으로부터 구한 gradient에 비해서 오차(variance)가 존재할 수 있습니다. 이때 이 variance가 큰 경우 gradient가 noisy하다고 볼 수 있고, 이는 최적화를 수행할 때 어려움으로 작용할 수 있습니다. 이 포스트에서는 이와 같은 문제를 **noisy gradient problem**이라고 부르도록 하겠습니다.


아래의 이미지들은 loss surface에서 noisy gradient problem이 발생하냐에 따른 수렴하는 경향성을 표현한 것들입니다.

SGD를 진행하는 동안, noisy gradient problem이 없다면, 아래 이미지와 같이 정상적으로 수렴할 수 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2020-12-29-Gradient-Accumulation/gradient.jpeg" alt="normal gradient" width="40%">
  <figcaption style="text-align: center;">normal gradient</figcaption>
</p>
</figure>



반면에, noisy gradient problem을 가지고 있는 모델은 아래 이미지와 같이 수렴하는데 어려움을 겪습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img style="width: 40%" src="/assets/images/2020-12-29-Gradient-Accumulation/noisy_gradient.jpeg" alt="noisy gradient">
  <figcaption style="text-align: center;">noisy gradient</figcaption>
</p>
</figure>



실제로 NLP영역에서 많이 활용되고 있는 transformer 구조도 이와 같은 문제를 겪었습니다. 실제로 'Attention Is All You Need' 논문을 살펴보면, 학습을 위해서 warmup을 사용했습니다.
warmup을 사용한 이유는 학습 초기에 발생하는 noisy gradient문제를 해결하기 위해서입니다. [3, 6]


이처럼 noisy gradient problem은 원활한 학습을 저해하는 주요한 요인입니다. 다음으로 마키나락스에서 새로운 모델을 개발하면서, 겪은 사례에 대해서 설명드리겠습니다.

## 사례: residual AutoEncoder with FC layer

마키나락스에서도 유사한 문제를 겪었습니다. 내부에서 autoencoder 기반의 anomaly detection task를 수행하고 있습니다. 더 깊은 모델을 사용하고자 기존에 사용하던 autoencoder 모델들에 residual connection을 추가해봤습니다. 하지만, 예상치 못한 문제가 발생했습니다. 레이어가 깊어질수록 학습이 매우 불안정적으로 진행되었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img style="width: 70%" src="/assets/images/2020-12-29-Gradient-Accumulation/residual_ae.jpeg" alt="rae">
  <figcaption style="text-align: center;">Residual AE</figcaption>
</p>
</figure>
  

결론적으로 학습에 발생하는 gradient가 noisy하다는 문제가 있었습니다. 그렇다면, 기존의 모델과 대비해서 왜 이런 문제가 발생했는지 살펴보겠습니다.


아래와 같이 $\ell$개의 레이어로 이루어진 간단한 모델(no residual connection)이 있다고 가정해보겠습니다. 이 모델은 아래와 같은 forward과정을 가질 것입니다.

- $h_i$: i번째 hidden layer의 output
- $f_i$: i번째 hidden layer

$$
h_0=x, h_\ell=\hat{y}, loss=||x-\hat{y}||_2^2 
$$

backpropagation은 아래와 같이 전개됩니다.

$$
\frac{d loss}{d x_0} =\frac{d loss}{d h_{\ell}} \frac{d h_{\ell}}{d h_{\ell - 1}} \cdots \frac{d h_1}{d x_0}
$$


이제 residual connection이 추가된 모델은 어떤 식으로 전개되는지 살펴보겠습니다.

**Residual connection**에 대해서 수식으로 표현하면, 아래와 같습니다. [13]

- $h_i$: i번째 hidden layer의 input
- $f_i$: i번째 hidden layer
  
$$
h_{i} = f_i(h_{i-1}) + h_{i-1}
$$

Forward 과정을 전개해보면, 아래와 같습니다.


- $h_i$: i번째 hidden layer의 output
- $f_i$: i번째 hidden layer
- $h_0=x$

$$
h_\ell=\hat{y} = h_0 + h_1 + \cdots + f_{\ell}(h_{\ell - 1})
$$


이 모델의 backpropagation과정을 살펴보면 아래와 같습니다.

$$
\frac{d loss}{d x_0} =\frac{d loss}{d h_{\ell}} \frac{d h_{\ell}}{d x_0} = 

\frac{d loss}{d h_{\ell}} \frac{d(h_0 + h_1 + \cdots + f_{\ell}(h_{\ell - 1}))}{d x_0}
$$


각 layer마다 발생하는 gradient vector를 random variable로 보면, 위의 gradient vector의 분산은 아래의 식으로 구할 수 있습니다.

우선, 두 random variable의 합의 분산은 아래와 같이 전개됩니다.

$$
Var(aX + bY) = a^2 Var(X) + b^2 Var(Y) + 2abCov(XY)
$$

이를 위의 backpropagation의 식에 대입해보면, 기존 모델 대비 더 큰 분산을 가지는 것을 알 수 있습니다.

$\frac{d h_1}{dx_0}, \frac{dh_2}{dx_0}, \cdots, \frac{df_{\ell}(h_{\ell - 1})}{dx_0}$ 각 요소들이 모두 유사한 스케일을 가진다고 가정해보면, 최소 레이어 수 배 만큼 큰 분산을 가진다고 할 수 있습니다.


## Method: Gradient Accumulation

이런 문제를 해결하기 위해서 아래의 두 가지 방법을 시도를 하였으나, 각각 방법은 명확한 한계를 가지고 있었습니다.

- 모멘텀을 사용하지 않는 옵티마이저(eg. AdaGrad, AdaDelta, RMSProp)를 사용하여 안정적인 학습을 진행했으나, 수렴정도가 만족스럽지 않았습니다. [7, 8, 9]
- warm up을 사용하여 안정적인 학습과 수렴정도도 만족스러웠으나, 하이퍼파라미터에 민감하다는 단점이 있었습니다. [10]

모멘텀을 사용하는 옵티마이저를 활용하면서, 하이퍼파라미터에 상대적으로 덜 민감한 방법에 대해서 고민하기 시작했고, **large batch size**에서 답을 찾을 수 있었습니다.


Batch size를 키우게 되면, 통계학적으로 표준편차가 주는 효과가 있습니다. **Central Limit Theorem**에 따르면 아래와 같은 수식이 전개됩니다. [5]

$$
std = \frac{\sigma}{\sqrt{n}}
$$

따라서, Batch size를 키우게되면, 학습이 진행되는 중에 발생하는 nosisy gradient가 경감되는 것을 알 수 있습니다. 다른 연구에서도 batch size가 커지면 학습이 불안정하던 학습이 안정적으로 진행되는 것을 보였습니다. [1, 10]


Batch size를 키우는 것은 좋지만, gpu의 memory는 한정적입니다. 따라서, 한정된 gpu memory내에서 batch size를 키우는 효과를 내기 위해서, **gradient accumulation**이라는 방법을 사용했습니다. [4, 12]

Gradient accumulation은 매 step마다 파라미터를 업데이트 하지않고, gradient를 모으다가 일정한 수의 graidient vector들이 모이면 파라미터를 업데이트합니다.

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

모멘텀을 사용하는 옵티마이저의 경우(eg. AdaGrad, AdaDelta, RMSProp), 모멘텀을 사용하지 않는 옵티마이저와 비교했을 때 noisy gradient에 더 취약합니다. 이는 모멘텀의 구성을 생각해보면, 쉽게 알 수 있습니다. 모멘텀이라는 것은 결국 이전 모멘텀의 정보를 활용하여 현재 모멘텀을 결정하게 됩니다. 이와 같은 작업이 연쇄적으로 동작하게 되고 이는 아래의 수식처럼 표현할 수 있습니다. 결국 모멘텀을 사용하는 옵티마이저의 경우 학습할 때 발생한 noisy gradient가 모멘텀 백터에 남게 되고, 이는 수렴을 더욱 방해하는 요인이 됩니다. [14, 15]

- $p_t$: t시점의 모멘텀
- $\beta \in (0, 1)$: 모멘텀 팩터
- $w_t$: t시점의 weight
- $f(w_t)$: $w_t$를 바탕으로 구한 loss 


$$
p_t = \beta p_{t-1} + \nabla_{w_t} f(w_t) = \sum_{i=0}^{t-1} \beta^{i}\nabla_{w_{t-i}} f(w_{t-i}) + \beta^t p_0
$$

이와 더불어, 옵티마이저를 선택할 때도 noisy gradient problem을 고려하여, RAdam을 선택하였습니다. RAdam은 Adam 옵티마이저를 사용시 발생하는 **large variance of the adtheaptive learning rates** 문제를 해결하기 위해 나온 옵티마이저입니다.[1]

내부실험을 통해서 Adam보다 RAdam이 더 안정적인 학습을 한다는 것을 알 수 있었습니다. 본 포스트에서는 noisy gradient와 gradient accumulation에 대한 글이므로, 자세히 다루지는 않겠습니다.


## Result

Gradient accumulation을 통해서 불안정적이던 학습을 안정적으로 진행할 수 있었습니다. 또한, 위에서 언급한 Residual VAE를 안정적으로 학습하여 기존의 VAE보다 우수한 성능을 보일 수 있었습니다.

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


### Evaluation: Residual VAE vs VAE

Residual VAE와 VAE의 실험을 비교해봤습니다. 

- anomaly detection task를 수행하였습니다.
- Dataset은 MNIST를 사용했습니다.
- 실험셋팅은 class 0, 1을 target class를 두고 학습하였으며, 아래의 값은 그것의 평균값입니다.
- target class 0이라는 것은 0은 비정상 데이터, 나머지 클래스는 모두 정상데이터로 두고 실험하는 셋팅을 의미합니다. 따라서 train 및 valid 데이터로 $1, 2, \cdots, 9$ 클래스의 데이터를 활용했으며, test 데이터로 정상데이터와 비정상데이터를 합쳐서 실험했습니다. 참고로 비정상데이터의 비율은 0.35로 두고 실험하였습니다.
- VAE는 batch size 256, Residual VAE는 batch size 4000으로 두고 실험했습니다.
- 두 모델모두 200 epochs를 학습하였습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img style="width: 70%" src="/assets/images/2020-12-29-Gradient-Accumulation/table.png" alt="rae">
  <figcaption style="text-align: center;">Residual VAE vs VAE</figcaption>
</p>
</figure>

일반적으로 레이어의 수가 80개 정도되면, gradient vanishing의 영향으로 학습이 제대로 진행되지 않습니다. 하지만, residual connection을 추가해주면, gradient vanishing문제가 해결되며 상대적으로 더 낮은 train loss와 valid loss를 가지는 것을 확인할 수 있으며, anomaly detection task에서도 더 우수한 성능을 보여줍니다.


## 끝으로

이번 글에서 학습과정에서 발생하는 noisy gradient에 대해서 다뤘습니다. 이를 해결하기 위해서 batch size를 키우기 위한 노력을 했고, 그 과정에서 발생하는 gpu memory 문제를 해결하기 위해서 gradient accumulation을 활용했습니다.
Gradient accumulation을 진행하게 되면, batch size가 커지는 효과를 가지게 됩니다. [4, 12] 
Batch size가 커지게되면, Central Limit Theorem을 통해서 gradient의 분산이 줄어드는 효과를 기대할 수 있으며, 이는 안정적인 학습으로 이어질 수 있습니다. [5, 10] 

이번 포스트에서는 noisy gradient에 대해서 다뤘습니다. 혹시 비슷한 문제를 겪고 있으시다면, 도움이 되었으면 좋겠습니다.

## References

<a name="ref-1">[1]</a>  [Liyuan Liu , Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han (2020). On the Variance of the Adaptive Learning Rate and Beyond. the Eighth International Conference on Learning Representations.](https://github.com/LiyuanLucasLiu/RAdam)

<a name="ref-2">[2]</a>  [Popel, Martin, and Ondřej Bojar. "Training tips for the transformer model." The Prague Bulletin of Mathematical Linguistics 110.1 (2018): 43-70.](https://arxiv.org/abs/1804.00247)


<a name="ref-3">[3]</a>  [Xiong, Ruibin, et al. "On layer normalization in the transformer architecture." arXiv preprint arXiv:2002.04745 (2020).](https://arxiv.org/abs/2002.04745)

<a name="ref-4">[4]</a>  [Gradient Accumulation: Overcoming Memory Constraints in Deep Learning](https://towardsdatascience.com/gradient-accumulation-overcoming-memory-constraints-in-deep-learning-36d411252d01)

<a name="ref-5">[5]</a>  [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)


<a name="ref-6">[6]</a>  [Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.](https://arxiv.org/pdf/1706.03762.pdf)

<a name="ref-7">[7]</a>  [RMSprop](http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)

<a name="ref-8">[8]</a>  [Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online learning and stochastic optimization." Journal of machine learning research 12.7 (2011)](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf?source=post_page---------------------------)

<a name="ref-9">[9]</a>  [Zeiler, Matthew D. "Adadelta: an adaptive learning rate method." arXiv preprint arXiv:1212.5701 (2012).](https://arxiv.org/pdf/1212.5701.pdf)

<a name="ref-10">[10]</a>  [Gotmare, Akhilesh, et al. "A closer look at deep learning heuristics: Learning rate restarts, warmup and distillation." arXiv preprint arXiv:1810.13243 (2018).](https://arxiv.org/pdf/1810.13243.pdf)

<a name="ref-11">[11]</a>  [Z. Huo, B. Gu, and H. Huang, “Large batch training does not need warmup,” arXiv preprint arXiv:2002.01576, 2020.](https://arxiv.org/pdf/2002.01576.pdf)

<a name="ref-12">[12]</a>  [What is Gradient Accumulation in Deep Learning?](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)

<a name="ref-13">[13]</a>  [K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016](https://arxiv.org/pdf/1512.03385.pdf)
http://proceedings.mlr.press/v28/sutskever13.pdf

<a name="ref-14">[14]</a>  [Sutskever, Ilya, et al. "On the importance of initialization and momentum in deep learning." International conference on machine learning. 2013.](http://proceedings.mlr.press/v28/sutskever13.pdf)

<a name="ref-15">[15]</a>  [DIVE INTO DEEP LEARNING](https://d2l.ai/chapter_optimization/momentum.html)

 