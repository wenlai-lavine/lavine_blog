---
layout: post
title: Improving Both Domain Robustness and Domain Adaptability in Machine Translation
categories: paper-notes
tags: [MT, NLP, Meta-Learning]
---

In [previous post](https://lavine-lmu.github.io/lavine_blog/notes/2022/08/20/Meta-Learning-for-Low-Resource-Neural-Machine-Translation.html), we introduced the meta-learning technology used in machine translation. In this post, we prepare to introduce our paper [Improving Both Domain Robustness and Domain Adaptability in Machine Translation](https://arxiv.org/pdf/2112.08288.pdf) published in COLING 2022 more details.

### Background and Motivation

The success of Neural Machine Translation (NMT) heavily relies on large-scale high-quality parallel data, which is difficult to obtain in some domains. [Chu et al., 2018](https://aclanthology.org/C18-1111/) summarized current NMT domain adaptation methods into two types: data-centric and model-centric. We highly recommend the reader to read this survey paper to get an overview of NMT domain adaptation. Also, we recommend the reader to read this [survey paper](https://arxiv.org/abs/2007.09604) to understand the meta-learning technologies using in NLP community.

There are two major problems in NMT domain adaptation: domain robustness and domain adaptability.

+ **Domain Robustness**: models should work well on both seen domains (the domains in the training data) and unseen domains (domains which do not occur in the training data).
+ **Domain Adaptability**: with just hundreds of in-domain sentences, we want to be able to quickly adapt to a new domain. 

Current methods improving both two properties:

+ Domain Adaptability
  + Traditional finetuning: an out-of-domain model is continually trained on in-domain data
  + Meta-Learning: trains models which can be later rapidly adapted to new scenarios using only a small amount of data.
+ Domain Robustness
  + [Müller et al., 2020](https://aclanthology.org/2020.amta-research.14.pdf) first defined the concept of domain robustness and propose to improving the domain robustness by subword regularization, definsive distilation, reconstruction and neural noisy channel reranking.
  + [Jiang et al., 2020](https://doi.org/10.18653/v1/2020.acl-main.165) proposed using individual modules for each domain with a word-level domain mixing strategy, which they showed has domain robustness.

Although effectively, current methods has usually focused on only one aspect of domain adaptation at the expense of the other one, and **our motivation is to consider both of the two properties**.

### Method

In this paper, we propose a novel approach, ***RMLNMT***, which combines meta-learning with a word-level domain-mixing system (for improving domain robustness) in a single model. RMLNMT consists of three parts: Word-Level Domain Mixing, Domain Classification, and Online Meta-Learning. The following fingure illustrates our proposed methods.

![](https://github.com/lavine-lmu/lavine_blog/raw/main/assets/paper-notes/2022-08-22-RMLNMT/lai-2022.png)

+ **Word-level Domain Mixing**
  + The domain of a word in the sentence is not necessarily consistent with the sentence domain. Therefore, we assume that every word in the vocabulary has a domain proportion, which indicates its domain preference.
  + Each domain has its own multi-head attention modules. Therefore, we can integrate the domain proportion of each word into its multi-head attention module.
  + Apply the domain mixing scheme in the same way for all attention layers and the fully-connected layers.
  + The model can be efficiently trained by minimizing the composite loss: $$L^{*}=L_{\mathrm{gen}}(\theta)+L_{\mathrm{mix}}(\theta)$$

+ **Domain Classification**

  + [Rieß et al. (2021)](https://aclanthology.org/2021.mtsummit-research.15.pdf) show that using scores from simple domain classifier are more effective than scores from language models for NMT domain adaptation.
  + We compute domain similarity using a sentence-level classifier, but in contrast with previous work, we based our classifier on a pre-trained language model (BERT).

+ **Oneline Meta-Learning**

  + We use domain classification scores as the curriculum to split the corpus into small tasks, so that the sentences more similar to the general domain sentences are selected in early tasks.

  + Previous meta-learning approaches are based on token-size based sampling, which proved be not balanced since some tasks did not contain all seen domains, especially in the early tasks. To address these issues, we sample the data uniformly from the domains to compensate for imbalanced domain distributions based on domain classifier scores.

  + Following the balanced sampling, the process of meta-training is to update the current model parameter from $\theta$ to $\theta^{\prime}$ using a MAML (Finn et al., 2017) objective with the traditional sentence-level meta-learning loss $\mathcal{L}_{\mathcal{T}}\left(f_{\theta}\right)$ and the word-level loss $\Gamma_{\mathcal{T}}\left(f_{\theta}\right)$ ($L^{*}$ of $\mathcal{T}$).
    $$
    L_{\mathcal{T}}\left(f_{\theta}\right) = \mathcal{L}_{\mathcal{T}}\left(f_{\theta}\right) +  \Gamma_{\mathcal{T}}\left(f_{\theta}\right)
    $$

### Results and Analysis

+ **Main Results**
  + **Domain Robustness**
    + RMLNMT shows the best domain robustness compared with other models both in seen and unseen domains.
    + In addition, the traditional meta-learning approach (Meta-MT, Meta-Curriculum) without fine-tuning is even worse than the standard transformer model in seen domains. In other words, we cannot be sure whether the improvement of the meta-based method is due to the domain adaptability of meta-learning or the robustness of the teacher model. **This phenomenon is our motivation for improving the robustness of traditional meta-learning based approach.** 
    + Note this setup differs from the previous work because we included the $\mathcal{D}_{\text {meta-train }}$ data to the vanilla system to insure all systems in the table use the same training data.
  + **Domain Adaptability**
    + We observe that the traditional meta-learning approach shows high adaptability to unseen domains but fails on seen domains due to limited domain robustness. In contrast, RMLNMT shows its domain adaptability both in seen and unseen domains, and maintains the domain robustness simultaneously.
    + Compared with RMLNMT, traditional meta-learning approach show much improvement between *w/o FT* model and *FT* model. This phenomenon meets our expectations since *RMLNMT* without finetuning is already strong enough due to the domain robustness of word-level domain mixing. In other words, the improvement of traditional meta-learning approach is to some extent due to the unrobustness of the model.
  + **Cross-Domain Robustness**
    + To better show the cross-domain robustness of RMLNMT, we use the fine-tuned model of one specific domain to generate the translation for other domains. More formally, given $k$ domains, we use the fine-tuned model $M_{J}$ with the domain label of $J$ to generate the translation of $k$ domains.
    + We reports the average difference of $k \times k$ BLEU scores; a larger positive value means a more robust model. 
    + We observed that the plain meta-learning based methods have a negative value, which means the performance gains in the specific domains come at the cost of performance decreases in other domains. In other words, the model is not domain robust enough. In contrast, RMLNMT has a positive difference with the vanilla system, showing that the model is robust.
+ **Ablation Study**
  + **Different Clssifiers**
    + We observed that the performance of RMLNMT is not directly proportional to the accuracy of the classifier. In other words, slightly higher classification accuracy does not lead to better BLEU scores. This is because the accuracy of the classifier is close between BERT-based models and the primary role of the classifier is to construct the curriculum for splitting the tasks.
    + When we use a significantly worse classifier, i.e., the CNN in our experiments, the overall performance ofRMLNMT is worse than the BERT-based classifier.
  + **Different Smapling Strategy**
    + Plain meta-learning uses a token-based sampling strategy to split sentences into small tasks. However, the token-based strategy could cause unbalanced domain distribution in some tasks, especially in the early stage of training due to domain mismatches. To address this issue, we proposed to balance the domain distribution after splitting the task.
    + Results shows that our methods can result in small improvements in performance.
  + **Different Fine-tuning Strategy**
    + The model for each domain has its own multi-head and feed-forward layers. During fine-tuning stage of RMLNMT, we devise four strategies;
    + Finetuning on specific domain (*FT-specific*) obtains robust results among all the strategies.
    + Although other strategies outperform *FT-specific* in some domains, *FT-specific* is robust across all domains. Furthermore, *FT-specific* is the fairest comparison because it uses only a specific domain corpus to fine-tune, which is the same as the baseline systems.

### Conclusion

+ We presented RMLNMT, a robust meta-learning framework for low-resource NMT domain adaptation reaching both high domain adaptability and domain robustness (both in the seen domains and unseen domains).
+ We found that domain robustness dominates the results compared to domain adaptability in meta-learning based approaches.
+ The results show that RMLNMT works best in setups that require high robustness in low-resource scenarios. 



__BibTeX Reference__

```bibtext
@article{lai2021improving,
  title={Improving both domain robustness and domain adaptability in machine translation},
  author={Lai, Wen and Libovick{\`y}, Jind{\v{r}}ich and Fraser, Alexander},
  journal={arXiv preprint arXiv:2112.08288},
  year={2021}
}
```

