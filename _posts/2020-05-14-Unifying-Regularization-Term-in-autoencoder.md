---
title: Unifying Regularization Term in Auto-encoder
key: 20200514
tags: DeepLearning GenerativeLearning Auto-encoder Adversarial
category: blog
---

## Generative Adversarial Networks (GAN)

$$
\underset{G}{argmin}\text{ }\underset{D}{argmax}\mathcal{L}(G, D)=\mathbb{E}_{Z\sim P(Z)}[\log{D(Z)}] + \mathbb{E}_{X\sim P(X)}{[\log{(1-D(G(X)))}]}
$$

## Variational Auto-Encoder (VAE)

Minimize loss.

$$
\mathcal{L}(\phi, \theta)=-\mathbb{E}_{X\sim P(X)}[\mathbb{E}_{Z\sim q_\phi(Z|X)}[\log{p_\theta(X|Z)}] - KL(q_\phi(Z|X)||p(Z))]
$$

## Adversarial Auto-Encoder (AAE)

Minimize both losses.

$$
\begin{aligned}
\mathcal{L}_G&=-\mathbb{E}_{X\sim P(X)}[\mathbb{E}_{Z\sim q_\phi(Z|X)}[\log{p_\theta(X|Z) - D_\psi(Z)}]] \\
\mathcal{L}_D&=-\mathbb{E}_{X\sim P(X)}[\mathbb{E}_{Z\sim q_\phi(Z|X)}[D_\psi(Z)] + \mathbb{E}_{Z\sim P(Z)}[\log{(1-D_\psi(Z))}]]
\end{aligned}
$$

## Adversarial Variational Bayes (AVB)

Minimize both losses.

$$
\begin{aligned}
\mathcal{L}_G&=-\mathbb{E}_{X\sim P(X)}[\mathbb{E}_{Z\sim q_\phi(Z|X)}[\log{p_\theta(X|Z) - D_\psi(X, Z)}]] \\
\mathcal{L}_D&=-\mathbb{E}_{X\sim P(X)}[\mathbb{E}_{Z\sim q_\phi(Z|X)}[D_\psi(X,Z)] + \mathbb{E}_{Z\sim P(Z)}[\log{(1-D_\psi(X,Z))}]]
\end{aligned}
$$
