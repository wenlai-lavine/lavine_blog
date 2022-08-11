---
layout: post
title: Contrastive Learning in NLP
categories: notes
tags: [NLP, Contrastive-Learning]
---

In this post, I would like to introduce a tutorial in NAACL 2022 named [Contrastive Data and Learning for Natural Language Processing](https://aclanthology.org/2022.naacl-tutorials.6.pdf). The tutorial introduce some recent works in NLP using contrastive learning techniques. Fore more details, I'd recommend you refer to the tutorials [website](https://contrastive-nlp-tutorial.github.io/) and the [paper list](https://github.com/ryanzhumich/Contrastive-Learning-NLP-Papers) of contrastive learning. In addition, I also recommend the readers to read this [survey paper](https://arxiv.org/pdf/2102.12982.pdf).

## What is Contrastive Learning?

Contrastive learning is one such technique to learn an embedding space such that similar data sample pairs have close representations while dissimilar samples stay far apart from each other. While it has originally enabled the success for vision tasks, recent years have seen a growing number of publications in contrastive NLP.

The first NLP paper ([Smith and Eisner, 2005](https://aclanthology.org/P05-1044.pdf)) introducing 'contrastive estimation' as an unsupervised training objective for log-linear models. And the most sucessful example of contrastive learning for NLP is word2vec ([Mikolov et al., 2013](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)) for word embeddings.

## Foundations of Contrastive Learning

Basiclly, there are two elements of contrastive learning: ```Contrastive Learning = Contrastive Data Creation + Contrastive Objective Optimization```

#### 1. Contrastive Learning Objectives

There are different contrastive learning objectives:

+ Contrastive Loss ([Chopra et al., 2005](https://ieeexplore.ieee.org/document/1467314))
  + minimizes the embedding distance when they are from the same class and maximizes the embedding distance when they are from the different class;
+ Triplet Loss ([Schroff et al., 2015](https://arxiv.org/abs/1503.03832))
  + push the distance between ```positive and anchor + margin``` to be smaller than the distance between ```negative and anchor```;
+ Lifted Structured Loss ([Oh song et al., 2016](https://arxiv.org/abs/1511.06452?context=cs.LG))
  + Take into account all pairwise edges within the batch 
+ N-pair Loss ([Sohn, 2016](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective))
  + compare with Triplet loss, N-pair loss extended to ```N-1``` negative examples; it is similar to multi-class classification; the ```total loss = inner product similarity + softmax loss```
+ Noise Contrastive Estimation (NCE) ([Guttmann and Hyvärinen, 2010](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf))
  + use logistic regression with cross-entropy loss to differentiate positive samples (i.e., target distribution) and negative samples (i.e., noise distribution)
+ InfoNCE ([van den Oord et al., 2018](https://arxiv.org/abs/1807.03748))
  + use softmax loss to differentiate a positive sample from a set of noise examples
+ Soft-Nearest Neighbors Loss ([Salakhutdinov and Hinton, 2007](https://proceedings.mlr.press/v2/salakhutdinov07a.html) and [Frosst et al., 2019](https://arxiv.org/abs/1902.01889))
  +  Extend to different numbers of positive (M) and negative examples (N)

The summaries of differnet contrastive learning objectives is as follows:

![image-20220808140735565](https://github.com/lavine-lmu/lavine_blog/raw/main/assets/paper-notes/2022-08-08-Contrastive-Learning-in-NLP.assets/image-20220808140735565.png)

#### 2. Data Sampling and Augmentation Strategies

+ Self-Supervised Contrastive Learning
  + Data Augmentation
    + Text Space
      + Lexical Editing (token-level)
      + Back-Translation (sentence-level)
    + Embedding Space
      + Dropout
      + Cutoff
      + Mixup
  + Sampling Bias 
    + Debased Contrastive Learning
      + Assume a prior probability between positive and negative, then approximate the distribution of negative examples to debias the loss.
  + Hard Negative Mining
    + importance sampling
      + if this negative sample is close to the anchor sample, then up-weight its probability of being selected
    + Adversarial Examles
      + create adversarial examples that are positive but conduses the model
  + Large Batch Size
+ Supervised Contrastive Learning
  + SimCSE ([Gao et al., 2021](https://arxiv.org/abs/2104.08821))
  + CLIP ([Radford et al., 2021](https://arxiv.org/abs/2103.00020))

#### 3. Analysis of Contrastive Learning

+ Geometric Interpretation
  + when the class label is used, then supervised contrastive learning will converge to class collapse to a regular simplex.
+ Connection to Mutual Information
+ Theoretical Analasis
+ Robutness and Security

## Contrastive Learning for NLP

Contrastive learning has shown success in many NLP tasks.

![image-20220808143149679](https://github.com/lavine-lmu/lavine_blog/raw/main/assets/paper-notes/2022-08-08-Contrastive-Learning-in-NLP.assets/image-20220808143149679.png)

For the paper in different fileds, please refer to the paper link in the [original tutorials paper](https://aclanthology.org/2022.naacl-tutorials.6.pdf).



__BibTeX Reference__

```bibtex
@inproceedings{zhang-etal-2022-contrastive-data,
    title = "Contrastive Data and Learning for Natural Language Processing",
    author = "Zhang, Rui  and
      Ji, Yangfeng  and
      Zhang, Yue  and
      Passonneau, Rebecca J.",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Tutorial Abstracts",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-tutorials.6",
    doi = "10.18653/v1/2022.naacl-tutorials.6",
    pages = "39--47",
}
```