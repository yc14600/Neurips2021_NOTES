# Neurips2021 NOTES


## Invited talks
1. How Duolingo Uses AI to Assess, Engage and Teach Better: 
   
   An interesting talk about Duolingo during which the website of Duolingo was down for a while :p. AI is used to aid human detecting cheating during tests and seems very effective. There are also a lot of efforts for improving users' engagement and it is usually hard to get informative feedback from dropped users. Japanese are the most commitment users. Duolingo also adaptively adjusts the curriculum on the basis of at least 6 months. They consider to make some data public.
   (I started to learn Japanese 10 minutes per day on Duolingo for free.) 

2. Do We Know How to Estimate the Mean?

   Takeaways: sub-Gaussian bound (Chebyshev inequality) is the best error bound we can get for the expectation estimators when the variance is finite; there are median-mean, trim-mean estimators; for multivariate random variables, there are different ways to define median hence leading to different bounds; computation of the estimators for multivariate variables is non-trivial.   


## Interesting papers

1. [MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers](https://openreview.net/forum?id=Tqx7nJp7PR): proposed a measurement for text generation, which is the area under a precision-recall like curve. The curve is drawn by varying $\lambda$ to compute $KL(P||R)$ and $KL(Q||R)$, where $R = \lambda P + (1-\lambda)Q$, $P$ is the distribution of human-generated text, $Q$ is the distribution of model-generated text. This metric is shown to be highly correlated to human assessment. [Code](https://github.com/krishnap25/mauve) is available.
2. [Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization](https://openreview.net/forum?id=jlchsFOLfeF): gave some theoretical results of learning invariant representations from the perspective of causality, proposed a term to help with learning invariant representations which is simply minimizing the variance of representations. 
3. [Reliable and Trustworthy Machine Learning for Health Using Dataset Shift Detection](https://openreview.net/forum?id=hNMOSUxE8o6): an empirical study on detecting out-of-distribution test samples in health care applications; applied  Mahalanobis distance [1],
Gram matrices [2], and energy-based [3]  out-of-distribution detection methods because they can work on existing models, energy-based methods show worse performance than others; proposed a **confidence score** by scaling raw out-of-distribution scores from out-of-distribution detectors [1,2] to 0–100, scaling is done in a piecewise manner that most of the in-distribution examples have confidence scores of 90 or above;  higher confidence score led to increasing trustworthiness which is also related to interpretability of input data.

4. [Does Knowledge Distillation Really Work?](https://openreview.net/forum?id=7J-fKoXiReA): an empirical study about how the loss of knowledge distillation works, which shows: 1). good student accuracy does not imply good distillation fidelity, 2). student fidelity is correlated with calibration when distilling ensembles, 3). optimization of the distillation loss usually leads to sub-optimum when the initialization is not close to the teacher's optimum.  

5. [Self-Diagnosing GAN: Diagnosing Underrepresented Samples in Generative Adversarial Networks](https://openreview.net/forum?id=SGZn06ZXcG): proposed two metrics for identifying minor samples and features during training: the mean and variance of Log-Density-Ratio (LDRM & LDRV) for each sample across training steps. 

6. [Amortized Variational Inference for Simple Hierarchical Models](https://openreview.net/forum?id=Rw_fo_Z2vV): proposed an amortized way to learn parameters of local variables in a hierarchical model without assumptions of conjugacy or mean-field in variational inference, it is similarly accurate as using a given joint distribution such as a full-rank Gaussian.

[1] Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting outof-distribution samples and adversarial attacks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

[2] Chandramouli Shama Sastry and Sageev Oore. Detecting out-of-distribution examples with gram matrices. In 37th International Conference on Machine Learning, volume 1, pages 8491–8501, 2020.

[3] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 21464–21475. Curran Associates, Inc., 2020.



## Continual learning

1. [Towards a robust experimental framework and benchmark for lifelong language learning](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/b3e3e393c77e35a4a3f3cbd1e429b5dc-Abstract-round1.html): provided a framework and multiple benchmarks for language learning in continual learning, including multi-domain, multilingual and different levels of linguistic hierarchy. [Code](https://github.com/AmanDaVinci/lifelong-learning) is available.
2. [Formalizing the Generalization-Forgetting Trade-off in Continual Learning](https://openreview.net/forum?id=u1XV9BPAB9): formulates the continual learning problem via dynamic programming and model the trade-off between catastrophic forgetting and generalization as a two-player sequential game; the results look much better than ER on CIFAR100, but memory is 16000 samples that is much larger than those in related work. Not fully understand the formulation but the algorithm seems a meta-learning approach with a different cost function on outter loop; [code](https://github.com/krm9c/Balanced-Continual-Learning) is available, need to check how $x^{PN}$ is updated in the inner loop.
3. [Class-Incremental Learning via Dual Augmentation](https://openreview.net/forum?id=8dqEeFuhgMG): introduces 2 augmentation procedures in CL, one is augmentation of classes for a new task by generating pseudo classes (mixing up real classes), the other is generating pseudo samples of old classes by statistical approximation of old classes (class mean and variance). It seems  multi-head that needs keeping linear layer of each task 
, check [code](
https://github.com/Impression2805/IL2A.).

4. [Gradient-based Editing of Memory Examples for
Online Task-free Continual Learning](https://openreview.net/forum?id=gL8btosnTj): GMED-edited examples remain similar to their unedited forms, but can yield increased loss in the upcoming model updates, thereby making the future replays more effective in overcoming
catastrophic forgetting. [Code](https://github.com/INK-USC/GMED) is available.

## Unsupervised & Semisupervised learning
1. [Detecting Errors and Estimating Accuracy on Unlabeled Data with Self-training Ensembles](https://github.com/INK-USC/GMED)

## Representation learning

1. [Learning Compact Representations of Neural Networks using DiscriminAtive Masking (DAM)](https://openreview.net/forum?id=jE5UVpKhkUG): a new single-stage structured pruning
method that learns compact representations while training and does not require fine-tuning; it applies a gate function that gradually masks more and more neurons out during training: $g_{ij} = RELU(tanh(\alpha_i (\mu_{ij}+\beta_i)))$ is the gate function for neuron $j$ in layer $i$, where $\beta_i$ is learnable, $\mu_{ij}=kj/n_i$ is decided at the intialization ($n_i$ is number of neurons and $k$ is a hyperparameter), $\alpha_i$ is steepness parameter; minimizing L0 norm of $g_i$  for each layer is directly proportional to minimizing $\beta_i$ with a scaling factor of $n_i/k$, hence, the learning objective is directly minimizing the sum of $\beta_i$ across all layers.
