---
layout: post
title: Domain Adaptation and Multi-Domain Adaptation for Neural Machine Translation
categories: paper-notes
tags: [MT, Domain Adaptation]
---

In this post, I would like to introduce a survey paper titled [Domain Adaptation and Multi-Domain Adaptation for Neural Machine Translation: A Survey](https://arxiv.org/pdf/2104.06951.pdf). Here, I also recommend the readers to read another survey paper of domain adaptation in NMT [A Survey of Domain Adaptation for Neural Machine Translation](https://aclanthology.org/C18-1111) (Chu & Wang, COLING 2018). Also, you can find more details in her [phd theis](https://dcsaunders.github.io/thesis.pdf).

## 1. What Constitutes a Domain Adaptation Problem?

#### 1.1 Exploring the Definition of Domain

+ [Van der Wess 2017](https://irlab.science.uva.nl/wp-content/papercite-data/pdf/van-der-wees-phd-thesis-2017.pdf) has demonstrated three promary elements of a *domain* in the context of machine translation research in her PhD thesis.
  + ***Provenance*** is the source of the text, usually a single discrete label;
  + ***Topic*** is the subject of text, for example news, software, biomedical;
  + ***Genre*** may be interpreted as a concept that is orthogonal to topic, consisting of func- tion, register, syntax and style.

#### 1.2 Domain Adaptation and Multi-Domain Adaptation

+ Domain Adaptation Problem
  + We wish to *improve translation performance* on some set of sentences with identifiable characteristics. The characteristic may be a distinct vocabulary distribution, a stylo-metric feature such as sentence length distribution, some other feature of language, or otherwise meta-information such as provenance.
  + We wish to *avoid retraining* the system from scratch. Retraining is generally not desir- able since it can take a great deal of time and computation, with correspondingly high financial and energy costs. Retraining may also be impossible, practically speaking, if we do not have access to the original training data or computational resources.
+ Multi-Domain Adaptation Problem
  + We wish to achieve good translation performance on text from *more than one domain* using the same system.

## 2. Fine-Tuning as a Domain Adaptation Baseline, and its Difficulties

Given an in-domain dataset and a pre-trained neural model, domain adaptation can often be achieved by continuing training – ‘fine-tuning’ – the model on that dataset.

#### 2.1 Difficulty: Not Enough In-Domain Data

+ In some shared tasks, we only have the small amount of in-domain data (test set) and it is not always possible to assume the existence of a sufficiently large and high- quality in-domain parallel dataset.

#### 2.2 Difficulty: Forgetting Previous Domains

+ If a neural model with strong performance on domain A is fine-tuned on domain B, it often gains strong translation performance on domain B at the expense of extreme performance degradation on domain A. We called this '*catostrophic forgetting*' problem.
+ a common practice to alleivate the 'catastrophic forgetting' problem is to simply tune for fewer steps, although this introduces an inherent trade-off between better performance on the new domain and worse performance on the old domain ([Xu et al., 2019](https://aclanthology.org/2019.iwslt-1.27/)).
+ other methods is to introduce additional domain-specific parameters or subnetworks for the new domains.

#### 2.3 Difficulty: Overfitting to New Domains

+ *Overfitting* or ‘exposure bias’ is common when the fine-tuning dataset is very small or repetitive. The model may be capable of achieving excellent performance on the precise adaptation domain B, but any small variations from it – say, B′ – cause difficulties.

## 3. Data Centric Adaptation Methods

#### 3.1 Selecting Additional Natural Data for Adaptation

Given a test domain of interest and a large pool of general-domain sentences, there are many ways to extract relevant data. We divide these into methods using discrete **token- level measures** for retrieval, those using continuous **sentence representations** for retrieval, or those that use some **external model** for scoring.

#### 3.2 Filtering Existing Neural Data

In the case where we do take a full corpus as in-domain, *data filterin*g can be applied to ensure that the selected data is actually representative of a domain. A special case of data filtering is targeted to remove ‘noisy’ training examples, as for example in the WMT parallel corpus filtering task [Koehn et al., 2018](https://aclanthology.org/W18-6453).

#### 3.3 Generating Synthetic Bilingual Adaptation Data from Monolingual Data

Bilingual training data that is relevant to the domain of interest may not be available. However, source or target language monolingual data in the domain of interest is often much easier to acquire. In-domain monolingual data can be used to construct partially synthetic bilingual training corpora by forward- or back translation. This is a case of bilingual data generation for adaptation rather than data selection.

+ **Back-Translation**: Back translation uses natural monolingual data as target sentences, and requires a target-to- source NMT model to generate synthetic source sentences. Back translations are commonly used to augment general domain translation corpora, with strong improvements over models not trained on back-translated data [Sennrich et al., 2016](https://aclanthology.org/P16-1009).
+ **Forward-Translation**: Forward translation generates synthetic target sentences with an existing source-to-target NMT model. A sub-type, self-learning, trains a model with its own synthetic translations. Forward translation is less common than back translation, perhaps because the synthetic data is on the target side and so any forward translation errors may be reinforced and produced at inference. However, self-learning can mean more efficient domain adaptation than back translation, as it requires no additional NMT model.

#### 3.4 Artificially Noising and Simplifying Natural Data

In some cases only a small in-domain dataset is available for adaptation, with neither bilingual nor monolingual large in-domain corpora from which to extract additional relevant sentences. In this scenario we can generate new data from the available in-domain set by changing its source or target sentences in some way. Including these variations in tuning can reduce the likelihood of overfitting or over-exposure to a small set of one-to-one adaptation examples.

#### 3.5 Synthetic Bilingual Adaptation Data

A final type of data used for adaptation is purely synthetic. Purely synthetic data may be beneficial when even monolingual in-domain natural data is unavailable, or when it is not practical to back- or forward-translate such data, perhaps due to lack of a sufficiently strong NMT model. Synthetic data may be obtained from an **external or induced lexicon**, or constructed from a **template**.

## 4. Architecture-Centric Adaptation

Architecture-centric approaches to domain adaptation typically add trainable parameters to the NMT model itself. This may be a single new layer, a domain discriminator, or a new subnetwork. If domains of interest are known ahead of model training, domain- specific parameters may be learned jointly with NMT model training. However, in a domain adaptation scenario where we wish to avoid retraining, domain-specific parameters can be added after pre-training but before any fine-tuning on in-domain data.

#### 4.1 Domain Labels and Control

Where data is domain-labelled, the labels themselves can be used to signal domain for a multi-domain system. Domain labels may come from a human-curated source or from a model trained purely to infer domain.

+ **using tags**: please refer to [Pham et al., 2021](https://aclanthology.org/2021.tacl-1.2), which comparing several previously proposed tagging approaches, note that introducing new domains by specifying new labels at fine-tuning time is straightforward.

#### 4.2 Domain-Specific Subnetworks

Another architecture-based approach introduces new subnetworks for domains of interest. This could be a domain-specific element of the original NMT architecture (e.g. vocabulary embedding, encoder, decoder), or a domain embedding determined by a subnetwork not typically found in the Transformer.

+ **Vocabulary embedding** can be made wholly or partially domain-specific;
+ learn a **domain classifier**;
+ duplicate encoders or decoders for each domain of interest, although this quickly becomes expensive in terms of added model size and therefore com- putational and energy cost for training and inference.

#### 4.3 Lightweight Added Parameters

A lightweight architectural approach to domain adaptation for NMT adds only a limited number of parameters after pre-training. The added parameters are usually adapted on in- domain data while pre-trained parameters are ‘frozen’ – held at their pre-trained values. We called this technologies 'Adapter'.

#### 4.4. Architecture-Centric Multi-Domain Adaptation

Schemes that leave original parameters unchanged and only adapt a small added set of parameters can avoid any performance degradation from forgetting or overfitting by simply using the original parameters. Adapter-like architectural approaches may therefore have a natural application to continual learning, and can also be a lightweight approach for other multi-domain scenarios.

## 5. Training Schemes for Adaptation

Once data is selected or generated for adaptation and a neural architecture is determined and pre-trained, the model can be adapted to the in-domain data. One straightforward approach is fine-tuning the neural model with the same MLE objective function used in pre-training. However, simple fine-tuning approaches to domain adaptation can cause *catastrophic forgetting* of old domains and *overfitting* to new domains.

#### 5.1 Objective Function Regularization

A straightforward way to mitigate forgetting is to minimize changes to the model parameters. Intuitively, if parameters stay close to their pre-trained values they will give similar performance on the pre-training domain.

+ **Frezzing Parameters**: One way to ensure model parameters do not change is to simply not update them, effec- tively dropping them out of adaptation.
+ **Regularizing Parameters**: If adapting all parameters, regularizing the amount of adaptation can mitigate forgetting. Parameter regularisation methods involve a higher computational load than parameter freezing, since all parameters in the adapted model are likely to be different from the baseline, and the baseline parameter values are often needed for regularisation. However, regularisation may also function more predictably across domains, language pairs and models than selecting parameters to freeze by subnetwork.
+ **Knowledge Distillation**: Knowledge distillation and similar ‘teacher-student’ model compression schemes effectively use one teacher model to regularize training or tuning of a separate student model.

#### 5.2 Curriculum Learning

The ranking guides the order in which examples are presented to the model during training or fine-tuning. A typical curriculum orders training examples by difficulty, with the easiest examples shown first and the more complex examples introduced later.

In domain adaptation for NMT, a curriculum can be constructed from least-domain-relevant examples to most-domain-relevant. In fact, simple fine-tuning is effectively curriculum learning where the final part of the curriculum contains only in-domain data. However, curriculum-based approaches to domain adaptation generally involve a gradual transition to in-domain data.

#### 5.3 Instance Weighting

Instance weighting adjusts the loss function to weight training examples according to their target domain relevance. The weight may be determined in various ways. It may be the same for all sentences marked as from a given domain, or defined for each sentence using a domain measure like n-gram similarity or cross-entropy difference.

#### 5.4 Non-MLE Traning

MLE training is particularly susceptible to exposure bias, since it tunes for high likelihood only on the sentences available in the training or adaptation corpus. MLE also experiences loss-evaluation metric mismatch, since it optimizes the log likelihood of training data while machine translation is usually evaluated with translation- specific metrics. Tuning an NMT system with a loss function other than the simple MLE on pairs of training sentences may therefore improve domain adaptation.

+ **Minimum Risk Training**: MRT is of particular relevance to domain adaptation for two reasons. *Firstly*, in the NMT literature we find that MRT is exclusively applied to fine-tune a model that has already converged under a maximum likelihood objective. MRT therefore fits naturally into a discussion of improvements to pre-trained NMT models via parameter adaptation. *Secondly*, there is some indication that MRT may be effective at reducing the effects of exposure bias. Exposure bias can be a particular difficulty where there is a risk of overfitting a small dataset, which is often the case for domain adaptation, especially if there is a domain mismatch between adaptation and test data.
+ **Meta-Learning**: tuning a neural model such that it can easily learn new tasks in few additional steps: 'learning to learn’. The aim is to find model parameters that can be adapted to a range of domains easily, in few training steps and with few training examples. This is achieved by introducing a meta-learning phase after pre-training and before fine-tuning on any particular target domain.

## 6. Inference Schemes for Adaptation

#### 6.1 Multi-Domain Ensembling

+ **Domain Adaptive Ensembling**: Certain models in an ensemble may be more useful than others for certain inputs. For example, if ensembling a software-domain model with a science-domain model, we might expect the science model to be more useful for translating medical abstracts. This idea of varying utility across an ensemble is particularly relevant when the domain of a test sentence is unknown and therefore the best ensemble weighting must be determined at inference time.
+ **Retrieval-Based Ensembling**: please refer to [Khandelwal et al., 2021](https://arxiv.org/abs/2010.00710), [Zheng et al., 2021a](https://aclanthology.org/2021.acl-short.47.pdf) and [Zheng et al., 2021b](https://aclanthology.org/2021.findings-emnlp.358.pdf).

#### 6.2 Constrained Inference and Rescoring

Ensembling uses multiple models to produce a translation simultaneously. An alterna- tive is a multi-pass approach to inference, in which one model produces an initial translation which is adjusted or corrected using another model. While this involves multiple models performing their own separate inference passes, this can be more efficient than ensembling. Multi-pass approaches do not involve holding multiple models in memory at once, and the second translation pass is commonly held close to the initial translation in some way, reducing the number of translations to be scored.

#### 6.3 Domain Terminology Via Pre- and Post-Processing

+ **Terminology Tagging**: using taggings
+ **In-Domain Priming**: An extension to priming the model with individual terminology incorporates entire related sentences. For a given sentence, similar source sentences and their translations can be extracted from a parallel dataset using fuzzy matching as described in section 5.1. The similar sentence or its translation can be used as a domain-specific input prompt.



__BibTeX Reference__

```bibtex
@article{saunders2021domain,
  title={Domain adaptation and multi-domain adaptation for neural machine translation: A survey},
  author={Saunders, Danielle},
  journal={arXiv preprint arXiv:2104.06951},
  year={2021}
}
```