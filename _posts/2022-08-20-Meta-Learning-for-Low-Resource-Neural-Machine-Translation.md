---
layout: post
title: Meta-Learning for Neural Machine Translation
categories: notes
tags: [MT, NLP, Meta-Learning]
---

Meta-learning, also known as "learning to learn", has been shown to allow faster finetuning, converge to better performance, and achieve outstanding results for few-shot learning in many applications. 

It is believed that meta-learning has excellent potential to be applied in NLP, and some works has been proposed with notable achievements in several relevant problems, e.g., relation extraction, machine translation, and dialogue generation and state tracking. However, it does not catch the same level of attention as in the Computer Vision (CV) community. For meta-learning methods in NLP, we recommend the readers to read the survey paper writen by ([Yin et al., 2020](https://arxiv.org/abs/2007.09604))and the tutorial in [ACL 2021](https://aclanthology.org/2021.acl-tutorials.3/).

In machine translation area, ([Gu et al., 2018](https://aclanthology.org/D18-1398/)) first proposed using model-agnostic meta-learning (MAML) to NMT. They show that MAML effectively improves low-resource NMT and since then, more and more researchers have applied meta-learning to the filed of machine translation. ([Li et al., 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6339)) and ([Sharaf et al., 2020](https://aclanthology.org/2020.ngt-1.5/)) proposed to formulate the problem of low-resource domain adaptation in NMT as a meta-learning problem. ([Zhan et al., 2021](https://www.aaai.org/AAAI21Papers/AAAI-4465.ZhanR.pdf)) proposed to using curriculum learning to imrpove the performance of previous meta-learning methods. ([Lai et al., 2021](https://arxiv.org/abs/2112.08288)) explored the problems of using meta-learning in NMT domain adaptation and proposed a novel methods to improve both domain adaptability and domain robustness simultaneously. We will discuss more details for each papers above.

### Meta-Learning for Low-Resource Neural Machine Translation

In this paper, they proposed to using meta-learning algorithm for low-resource neural machine translation. More specifically, they extend the idea of model-agnostic meta-learning (MAML, [Finn et al., 2017]()) in multilingual scenarios. Figure 1 illustrate the overall framework of this paper.



<center><b>Fig.1</b> Framework of Gu et al., 2018</center>

The underlying idea of MAML is to use a set of source tasks $\left\{ \mathcal{T}^1, \ldots, \mathcal{T}^K \right\}$ to find the initialization of parameters $\theta^0$ from which learning a target task $\mathcal{T}^0$ would require only a small number of training examples. In the context of machine translation, this amounts to using many high-resource language pairs to find good initial parameters and training a new translation model on a low-resource language starting from the found initial parameters. This process can be understood as
$$
\begin{align*}
\theta^* = \text{Learn}(\mathcal{T}^0; \text{MetaLearn}(\mathcal{T}^1, \ldots, \mathcal{T}^K)).
\end{align*}
$$
The idea for this paper is, we ***meta-learn*** the initialization from auxiliary tasks and continue to ***learn*** the target task.

In their expeiments, they using 18 high-resource source tasks and 5 low-resource target tasks, has shown that the proposed MetaNMT significantly outperforms the existing approach of multilingual, transfer learning in low-resource neural machine translation across all the language pairs considered.



### Meta-Learning for Few-Shot NMT Adaptation

In this paper, they present a meta-learning approach (**Meta-MT**) that learns to adapt neural machne translation systems to new domains given only a small amount of training data in that domain. Figure 2 illustrate the difference between traditional fine-tuning and meta-learning.

![](../assets/paper-notes/2022-08-20-Meta-Learning-for-Low-Resource-Neural-Machine-Translation.assets/meta_learning_sharf.pdf)

<p align='left'><b>Figure.2</b> <b>[Top-A]</b> a training step of META_MT. <b>[Bottom-B]</b> Differences between  meta-learning and Traditional fine-tuning. Wide lines represent high resource domains (Medical, News), while thin lines represent low-resource domains (TED, Books). Traditional fine-tuning may favor high-resource domains over low-resource ones while meta-learning aims at learning a good initialization that can be adapted to any domain with minimal training samples. </p>

The goal in this paper is to learn how to adapt a neural machine translation system from experience.  The training procedure for META-MT uses offline simulated adaptation problems to learn model parameters $\theta$ which can adaptf aster to previously unseen domains.  The whole framework of META_MT contains two parts: meta-train and meta-test.

**Meta-Test Stage**

At test time, META_MT adapts a pre-trained NMT model to a new given domain. The adaptation is done using a small in-domain data that we call the *support set* and then tested on the new domain using a *query set*. The meta-learned model $\theta$ interacts with the world as follows (see more details in Figure 2):

- **Step1**: The world draws an adaptation task $\Tau$ from a distribution $P$, $\Tau\sim P(\Tau)$;
- **Step2**: The model adapts from $\theta$ to $\theta'$ by fine-tuning on the task's support set $\Tau_{\textrm{support}}$; 
- **Step3:** The fine-tuned model $\theta^{\prime}$ is evaluated on the query set $\Tau_{query}$.

Intuitively, meta-learning should optimize for a representation $\theta$ that can quickly adapt to new tasks rather than a single individual task.

**Meta-Train Stage**

At training time, META-MT will treat one of the simulated domains $\Tau$ as if it were a domain adaptation dataset. At each time step, it will update the current model representation from $\theta$ to $\theta'$ by fine-tuning on $\Tau_{\textrm{support}}$ and then ask: what is  the meta-learning loss estimate given $\theta$, $\theta'$, and $\Tau_{query}$? The model representation $\theta$ is then updated to minimize this meta-learning loss. Algorithm 1 illustrates the details of the whole meta-train procedure.

![](../assets/paper-notes/2022-08-20-Meta-Learning-for-Low-Resource-Neural-Machine-Translation.assets/sharf_alg-20220822144313475.png)

Over learning rounds, META_MT selects a random batch of training tasks from the meta-training dataset and simulates the test-time behavior on these tasks (Line 2). The core functionality is to observe how the current model representation $\theta$ is adapted for each task in the batch, and to use this information to improve $\theta$ by optimizing the meta-learning loss (Line 7). META-MT achieves this by simulating a domain adaptation setting by fine-tuning on the task specific support set (Line 4). This yields, for each task $\Tau_i$, a new adapted set of parameters $\theta'_i$ (Line 5). These parameters are evaluated on the query sets for each task $\Tau_{i,\textrm{query}}$, and a meta-gradient w.r.t the original model representation $\theta$ is used to improve $\theta$ (Line 7).

In experiments, they evaluate the proposed meta-learning strategy on ten domains with general large scale NMT systems and show that META-MT significantly outperforms classical domain adaptation when very few in-domain  examples are available. The experiments shows that META-MT can outperform classical fine-tuning by up to  2.5 BLEU points after seeing only $4,000$ translated words ($300$ parallel sentences).

### MetaMT, a Meta Learning Method Leveraging Multiple Domain Data for Low Resource Machine Translation

In this paper, they present a novel NMT model with a new word embedding transition technique for fast domain adaptation. They propose to split parameters in the model into two groups: model parameters and meta parameters. The former are used to model the translation while the latter are used to adjust the representational space to generalize the model to different domains. Figure 3 illustrate the archtectures of the proposed model.

<img src="https://github.com/lavine-lmu/lavine_blog/blob/main/assets/paper-notes/2022-08-20/meta_Li.png">



### Meta-Curriclum Learning for Neural Machine Translation

Traditional meta-learning for NMT domain adaptation often fails to improve the translation performance of the domain unseen at the meta-training stage. In their paper, they aim to alleviate this issue by proposing a novel meta-curriculum learning for NMT domain adaptation. The whole framework is as follows: 1) during meta-training, the NMT first learns the similar curricula from each domain to avoid falling into a bad local optimum early, and finally learns the curricula of individualities to improve the model robustness for learning domain-specific knowledge.

![zhan.pdf](https://github.com/lavine-lmu/lavine_blog/raw/main/assets/paper-notes/2022-08-20/zhan.pdf)

### Improving Both Domain Robustness and Domain Adaptability in Machine Translation

In this paper, they study two major problems in NMT domain adaptation. First, models should work well on both seen domains (the domains in the training data) and unseen domains (domains which do not occur in the training data). We call this property \textit{domain robustness}. Second, with just hundreds of in-domain sentences, we want to be able to quickly adapt to a new domain. We call this property \textit{domain adaptability}. Previous work on NMT domain adaptation has usually focused on only one aspect of domain adaptation at the expense of the other one, and our motivation is to consider both of the two properties.

Motivated by [Jiang et al., 2018](https://aclanthology.org/2020.acl-main.165.pdf) and [Zhan et al., 2021](https://arxiv.org/abs/2103.02262), to address both domain adaptability and domain robustness at the same time, they propose RMLNMT (robust meta-learning NMT), a more robust meta-learning-based NMT domain adaptation framework.  First, train a word-level domain mixing model to improve the robustness on seen domains, and show that, surprisingly, this improves robustness on unseen domains as well. Then, we train a domain classifier based on BERT to score training sentences; the score measures similarity between out-of-domain and general-domain sentences.
This score is used to determine a curriculum to improve the meta-learning process. Finally, we improve domain adaptability by integrating the domain-mixing model into a meta-learning framework with the domain classifier using a balanced sampling strategy.

The whole framework of this paper is shown as follows:

![lai et al., 2022](https://github.com/lavine-lmu/lavine_blog/blob/raw/assets/paper-notes/2022-08-20/lai.pdf)



**Reference**

[1] Yin, Wenpeng. "Meta-learning for few-shot natural language processing: A survey." *arXiv preprint arXiv:2007.09604* (2020).

[2] Lee, Hung-yi, Ngoc Thang Vu, and Shang-Wen Li. "Meta learning and its applications to natural language processing." *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Tutorial Abstracts*. 2021.

[3] Gu, Jiatao, et al. "Meta-learning for low-resource neural machine translation." *2018 Conference on Empirical Methods in Natural Language Processing, EMNLP 2018*. Association for Computational Linguistics, 2020.

[4] Li, Rumeng, Xun Wang, and Hong Yu. "MetaMT, a Meta Learning Method Leveraging Multiple Domain Data for Low Resource Machine Translation." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 34. No. 05. 2020.

[5] Sharaf, Amr, Hany Hassan, and Hal Daumé III. "Meta-learning for few-shot NMT adaptation." *arXiv preprint arXiv:2004.02745* (2020).

[6] Zhan, Runzhe, et al. "Meta-Curriculum Learning for Domain Adaptation in Neural Machine Translation." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 35. No. 16. 2021.

[7] Lai, Wen, Jindřich Libovický, and Alexander Fraser. "Improving both domain robustness and domain adaptability in machine translation." *arXiv preprint arXiv:2112.08288* (2021).

[8] Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." *International Conference on Machine Learning*. PMLR, 2017.