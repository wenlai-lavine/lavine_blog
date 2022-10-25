---
layout: post
title: Multilingual Multi Domain Adaptation for Machine Translation
categories: paper-notes
tags: [MT, NLP, Meta-Learning, Multilingual]
---

In our [previous paper](https://aclanthology.org/2022.coling-1.461/) published in COLING 2022, we investigate the domain robustness and domain adaptability in machine translation using meta-learning. As an extension of our COLING 2022 paper, we investigate the methods in multilingual scenarios, which adapting the multilingual neural machien translation (MNMT) model to both a new domain and to a new language pair at the same time. Finally, [this paper](https://arxiv.org/abs/2210.11912) was accepted to Findings of EMNLP 2022. In this post, I will introduce our paper.

### Background and Motivation

Adapting MNMT models to multiple domains is still a challenge task, particularly when domains are distant to the domain of the training data. Common practice is ***fine-tuning*** ([Dakwale et al., 2017](https://staff.science.uva.nl/c.monz/ltl/publications/mtsummit2017.pdf)) and ***adapters*** ([Bapna et al., 2019](https://doi.org/10.18653/v1/D19-1165)). Similarly, there is research work on adapting MNMT models to a new language pair using fine-tuning and adapters.

Although effective, the above approaches treat domain adaptation and language adaptation in MNMT models separately and also have some limitations:

+ Fine-tuning methods require updating the parameters of the whole model for each new domain, which is costly;
+ when finetuning on a new domain, catastrophic forgetting reduces the performance on all other domains, and proves to be a significant issue when data resources are limited;
+ adapter-based approaches require training domain adapters for each domain and language adapters for all languages, which also becomes parameterinefficient when adapting to a new domain and a new language because the parameters scale linearly with the number of domains and languages.

To this end, we proposed $m^4Adapter$, which facilitates the transfer between different domains and languages using meta-learning with adapters. Our hypothesis is that we can formulate the task, which is to adapt to new languages and domains, as a multi-task learning problem (and denote it as  $D_i$-$L_1$-$L_2$, which stands for translating from a language $L_1$ to a language $L_2$ *in a specific domain* $D_i$). Our approach is two-step: initially, we perform meta-learning with adapters to efficiently learn parameters in a shared representation space across multiple tasks using a small amount of training data (5000 samples); we refer to this as the meta-training step. Then, we fine-tune the trained model to a new domain and language pair simultaneously using an even smaller dataset (500 samples); we refer to this as the meta-adaptation step.

### Method

#### Meta-Training

+ **Task Definition**
  + A translation task in a specific textual domain corresponds to a Domain-Language-Pair (**DLP**). For example, an English-Serbian translation task in the *Ubuntu* domain is denoted as a DLP ```Ubuntu-en-sr```. 
+ **Task Sampling**
  + We follow a temperature-based heuristic sampling strategy, which defines the probability of any dataset as a function of its size
+ **Meta-Learning Algorithm**
  + We follow Reptile ([Nichol et al., 2018](https://arxiv.org/abs/1803.02999)), an alternative first-order meta-learning algorithm.
+ **Meta-Adapter**
  + We inserts adapter layers into the meta-learning training process. Different from the traditional adapter training process, we only need to train a single meta-adapter to adapt to all new language pairs and domains. The architecture of the Meta-Adapter is as shown in Figure 1.

!(https://github.com/lavine-lmu/lavine_blog/raw/main/assets/paper-notes/2022-10-25-M4Adapter/m4adapter.png)

### Meta-Adaptation

+ After the meta-training phase, the parameters of the adapter are fine-tuned to adapt to new tasks (as both the domain and language pair of interest are not seen during the meta-training stage) using a small amount of data to simulate a low-resource scenario.

### Results and Analysis

+ **Main Results**
  + We obtains a performance that is on par or better than *agnostic-adapter*, which is a robust model.
  + Our methods performs well when adapting to a new domain and a new language pair
  + Our methods has a more stable performance when adapting to new domains and laguage pairs
  + Our methods has the ability to improve the performance of some DLPs on which baseline models obtain extremely low BLEU scores, especially in some distant domains.
+ **Analysis**
  + **Efficiency**
    + Compare with baseline systems, our methods is more efficient than traditional meta-learning methods and a little more time consuming in training process than adapter based methods, however, we got a better performance.
  + **Domain Transfer via Languages**
    + Our methods outperforms the original m2m model, indicating that the model encodes language knowledge and can transfer this knowledge to a new language pairs.
    + We also discover that domain transfer through languages is desirable in some distant domains.
  + **Language Transfer via Domains**
    + We show that our methods permits cross-lingual transfer across domains.
    + Similarly, our methods has demonstrated significant language transfer ability in distant domains.

### Conclusion

+ We present $m^4Adapter$, a novel multilingual multi-domain NMT adaptation framework which combines meta-learning and parameter-efficient fine-tuning with adapters.
+ $m^4Adapter$ is effective on adapting to new languages and domains simultaneously in low-resource settings.
+ We find that $m^4Adapter$ also transfers language knowledge across domains and transfers domain information across languages. 
+ In addition, $m^4Adapter$ is efficient in training and adaptation, which is practical for online adaptation ([Etchegoyhen et al., 2021](https://aclanthology.org/2021.ranlp-1.47)) to complex scenarios (new languages and new domains) in the real world.



__BibTeX Reference__

```bibtext
@article{lai2022m4adapter,
  title={m^4Adapter: Multilingual Multi-Domain Adaptation for Machine Translation with a Meta-Adapter},
  author={Lai, Wen and Chronopoulou, Alexandra and Fraser, Alexander},
  journal={arXiv preprint arXiv:2210.11912},
  year={2022}
}
```

