---
layout: post
title: Joint Localization and Activation Editing for Low-Resource Fine-Tuning
categories: paper-notes
tags: [Activation Editing, LLMs, Low-Resource, Fine-Tuning, ICML 2025]
---



Adapting LLMs with very limited data is challenging. Common *parameter-efficient fine-tuning* (PEFT) methods – for example, injecting low-rank weight updates as in LoRA – can struggle when only a few hundred labeled examples are available. Recent work has explored **activation editing**: instead of tuning weights, one directly adjusts intermediate activations of the model using lightweight modifications. For instance, one can fine-tune only model bias vectors (BitFit) or apply small scaling and bias adjustments to MLP outputs (RED, ReFT) or attention heads (LoFIT). Activation editing methods involve *very* few parameters (often well under 0.01% of the model). For example, LoRA updating an 8B-parameter LLaMA model uses ≈0.826% of parameters, whereas LOFIT (which edits attention heads) updates only ≈0.002% to standard PEFT methods,hundred training examples are available). This dramatic reduction makes activation editing especially appealing for low-resource tasks.

However, existing activation-editing approaches tend to be brittle. They typically **predefine** where to intervene (e.g. always adjust all biases or fixed layers) and use a fixed form of intervention (usually only additive or only multiplicative). In practice, this can lead to unstable results when switching tasks or data distributions. For example, as shown in our paper, we found that intervening in all biases or multiple components often hurts performance, whereas focusing on the right subset can help. In particular, we find that *attention-head outputs* are the most impactful targets – editing a few heads yields greater gains than editing biases or hidden MLP outputs. This suggests two key questions: **(1)** which parts of the Transformer should be edited, and **(2)** whether to use additive or multiplicative edits (or both). Prior methods do not answer these jointly; they fix these choices by hand or in a two-step process, which can limit effectiveness and robustness.

We propose JoLA, (**Joint Localization and Activation Editing**), which addresses these challenges by *learning* all three aspects simultaneously: (i) which attention heads to edit, (ii) whether to apply additive bias, multiplicative scaling, or both, and (iii) the actual edit vectors for those heads. The core idea is to attach small *gating mechanisms* to each attention head and to equip each head with both an additive offset and a multiplicative scale. Concretely, for $i$-th head and layer $l$ in the Transformer, JoLA introduces two scalar gates $g_a^{(l,i)}$ and $g_m^{(l,i)}$ (one for the additive part and one for the multiplicative part) and two corresponding vectors (one additive bias vector  $a^{(l,i)}$ and one multiplicative scaling vector $m^{(l,i)}$). The head’s output $z_t^{(l,i)^{\prime}}$ is then transformed as:
$$
z_t^{(l,i)^{\prime}} =  (\mathbf{1} + g_{m}^{(l,i)} \cdot m^{(l,i)}) \odot z_{t}^{(l,i)} + g_{a}^{(l,i)} \cdot a^{(l,i)}
$$
where $\mathbf{1} \in \mathbb{R}^{d_l} $ is a vector of ones, $\odot$ denotes elementwise multiplication. Equivalently, one can think of separately adding $a^{(l,i)}$ and scaling by $m^{(l,i)}$; when $a^{(l,i)}=m^{(l,i)}=0$, the head is unchanged.) As shown in Figure 1, when both gates are zero the head’s output is left *identical* to before, so only a subset of heads ever get modified. Importantly, both gates are learnable and applied during training: the model effectively decides **which heads** to “turn on” for editing and whether to apply an additive or multiplicative (or hybrid) adjustment. This stands in contrast to previous work like LoFIT, which required manually picking heads in advance. LoFIT first uses multiplicative probes to identify a fixed set of heads and then restarts training with additive edits on those heads – a cumbersome two-phase process. JoLA integrates these choices end-to-end, making adaptation fully data-driven.

![m4Adapter](https://github.com/lavine-lmu/lavine_blog/raw/main/assets/paper-notes/JoLA/jola.png)

Key components of JoLA’s method are:

+ **Dynamic head selection:** Each attention head has two learnable gates that control whether it will be edited. During training, the model learns to close (set to zero) the gates of irrelevant heads. Heads with gates remaining “open” are the ones actually used for adaptation. This joint learning of *which* heads to use is novel compared to prior PEFT or editing methods.
+ **Hybrid interventions:** Each selected head can be modified by both an additive offset and a multiplicative scaling. In practice, the authors’ experiments found that additive (bias) edits tend to be more effective than purely multiplicative ones. By providing both capabilities (and separate gates), JoLA allows the model to employ whichever form is needed per head. When both gates are open, the head’s activation is first scaled and then an offset is added.
+ **Sparse training via Hard-Concrete gating:** To encourage only a few heads to be edited, JoLA imposes an $L_0$-like penalty on the gates. Specifically, each gate variable is drawn from a *Hard-Concrete* (stochastic binary) distribution during training (Louizos et al. 2018). The probability that a gate is non-zero can be computed in closed form, allowing the loss to include a sparsity regularizer of the form $\lambda \sum_l P\left(a_l \neq 0 \vee m_l \neq 0\right) $.  Here $\lambda$  trades off between task accuracy and sparsity. In effect, the training pushes most gates to zero unless the head is genuinely useful. After training, any head with an extremely low expected gate value can be pruned away with negligible impact.



__BibTeX Reference__

```bibtext
@InProceedings{lai2025-jola,
  title = 	 {Joint Localization and Activation Editing for Low-Resource Fine-Tuning},
  author =       {Wen Lai, Alexander Fraser and Ivan Titov},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  year = 	 {2025},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
}

```

