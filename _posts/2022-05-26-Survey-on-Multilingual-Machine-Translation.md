---
layout: post
title: Survey on Multilingual Machine Translation
categories: paper-notes
tags: [MT, Multilingual, en]
---

In this post, I would like to introduce a survey paper titled [A Survey of Multilingual Neural Machine Translation](https://dl.acm.org/doi/pdf/10.1145/3406095)  published on [ACM Computing Surveys](https://dl.acm.org/journal/csur), which is writen by the team of *NICT*. Here, I also recommend the readers to read another analysis paper of MNMT ([Kudugunta et al., 2019](https://doi.org/10.18653/v1/D19-1167)), which is also a nice paper for comparing representations across different languages, models, and layers.

## MNMT research category

In this paper, they first categorize various approaches based on their **central use-case** and then further categorize them based on **resource scenarios**, underlying *modeling principles*, *core- issues*, and *challenges*. Figure 1 illustrate the overall scenarioes in MNMT.

+ **Multiway Translation.** The goal is constructing a single NMT system for one-to-many, many-to-one, or many-to-many translation using parallel corpora for more than one language pair. In this scenarios, we need the parallel corpora for each language pair and the ultimate goal is to incorporate a number of languages into one single model.
+ **Low-resource Translation.** For thoes languages without / little parallel corpus, many studies have explored using assist language ti improve translation between low-resource language pairs. These multilingual NMT methods can be divided into two different scenarios: (a). a high-resource language pair is avaliable to assist a low-resource language pair (ie. Transfer learning etc.). (b). no direct parallel corpus for low-resource language pair, but languages share a parallel corpus with one or more pivot language(s).
+ **Multi-source Translation.** Documents that have been translated into more than one language might, in the future, be required to be translated into another language. In this scenario, existing multilingual complementary content on the source side can be exploited for multi-source translation. Multilingual complementary content can help in better disambiguation of content to be translated, leading to an improvement in translation quality.

![overall scenarioes](https://github.com/lavine-lmu/lavine_blog/raw/main/assets/paper-notes/MNMT-Survey/overview_scenarioes.png)

<center><b>Figure 1.</b> MNMT research categorized according to use-cases, core-issues, and the challenges involved.</center>

## Multiway NMT

In multiway NMT, parallel corpora are avaliable for all language pairs and the objective in this specific scenarios is to train **a** translation system between **all language pairs**. Particularly, one-to-many, many-to-one and many-to-many NMT models are specific instances of this general framework. The training objective for multiway NMT is maximization of the log-likelihood of all training data jointly for all language pairs (different weights may be assigned to the likelihoods of different pairs. Please refer to figure 2 for an overview of the multiway NMT paradigm.

#### Parameter Sharing

There are a wide range of architectural choices in the design of MNMT models. The choices are primarily defined by the degree of parameter sharing among various supported languages.

+ **Minimal Parameter Sharing.**
  + [Dabre et al., 2017]([An Empirical Study of Language Relatedness for Transfer Learning in Neural Machine Translation](https://aclanthology.org/Y17-1038.pdf)) proposed a model comprised of separate embeddings, encoders and decoders **for each language** that all **shared a single attention mechanism**. However, this model has a <u>large number of parameters</u>, usually around 270M or more. Furthermore, <u>the number of parameters only grows linearly with the number of languages</u>, while it grows quadratically for bilingual systems spanning all the language pairs in the multiway system. Another problem is that the shared attention mechanism has to <u>bear the burden of connecting different language pairs</u> and this can introduce a representational bottleneck where a model cannot learn the necessary repre- sentations for the best translation quality.
+ **Complete Parameter Sharing.**
  + [Johnson et al., 2017](https://aclanthology.org/Q17-1024/) proposed a highly compact model where **all languages share the same embeddings, encoder, decoder, and attention mechanism**. In this paper, a common vocabulary across all languages is first generated and then all corpora are concatenated and the input sentences are prefixed with a specific token (called *language tag*). This approach maybe helpful for thoes related languages because of sharing the same vocabulary.
  + [Ha et al., 2016](https://aclanthology.org/2016.iwslt-1.6/) proposed a similar model, but they maintained separate vocabularies for each language. While this might help in faster inference due to smaller softmax layers, the possibility of cognate sharing is lower, especially for linguistically close lan- guages sharing a common script. These approach can also useful for thoes unrelated languages.
  + Also, some papers leverage the lexical similarity is also useful:
    + representing all languages in a common script using script conversion ([Dabre et al., 2018](https://aclanthology.org/Y18-3003/), [Lee et al., 2017](https://aclanthology.org/Q17-1026.pdf))  or transliteration ([Nakow et al., 2019](https://aclanthology.org/D09-1141/)).
    + using a common subword-vocabulary across all languages, e.g., character ([Lee et al., 2017](https://aclanthology.org/Q17-1026.pdf)) and BPE ([Nguyen et al., 2017](https://aclanthology.org/I17-2050.pdf)).
    + representing words by both character encoding and a latent embedding space shared by all languages ([Wang et al., 2019](https://arxiv.org/abs/1902.03499)).
  + Massively multilingual NMT([Aharoni et al., 2019](https://aclanthology.org/N19-1388/); [Arivazhagan et al., 2019](https://arxiv.org/abs/1907.05019); [Bapna et al., 2019](https://aclanthology.org/D19-1165.pdf)) explore a wide range of model configurations focusing on data selection, corpora balancing, vocabulary, deep stacking, training, and decoding approaches. However, a massively multilingual system also runs into *representation bottlenecks* ([Aharoni et al., 2019](https://aclanthology.org/N19-1388/); [Siddhant et al., 2020](https://arxiv.org/abs/1909.00437)), where not all translation directions show improved performance despite a massive amount of data being fed to a model with a massive number of parameters.
+ **Controlled Parameter Sharing.**
  + Sharing encoders among multiple languages is very effective and is widely used ([Lee et al., 2017](https://aclanthology.org/Q17-1026.pdf); [Sachan et al., 2018](http://aclweb.org/anthology/W18-6327)).
  + [Blackwood et al., 2018](http://aclweb.org/anthology/C18-1263) explored target language, source language, and pair-specific attention parameters and showed that target language-specific atten- tion performs better than other attention-sharing configurations, thus highlighting that designing a strong decoder is extremely important.
  + Attention sharing strategies ([Sachan et al., 2018](http://aclweb.org/anthology/W18-6327)).
  + [Wang et al., 2019](https://aclanthology.org/P19-1117/) proposed a mechanism to generate a universal representation in- stead of separate encoders and decoders to maximize parameter sharing.
  + [Bapna et al., 2019](https://aclanthology.org/D19-1165.pdf) extend a fully shared model with language-pair-specific adaptor layers that are fine-tuned for those pairs.
  + [Zaremoodi et al., 2018](http://aclweb.org/anthology/P18-2104) proposed a routing network to dynamically control parameter sharing where the parts to be shared depend on the parallel corpora used for training.
  + [Platanios et al., 2018](http://aclweb.org/anthology/D18-1039) learned the degree of parameter sharing from the training data.

#### Addressing Language Divergence

A central task in MNMT is alignment of representations of words and sentences across languages so divergence between languages can be bridged, enabling the model to handle many languages. This involves the study and understanding of the representations learned by multilingual models and using this understanding to further improve modeling choices.

+ **Vocabulary.**
  + Temperature based vocabulary sampling ([Aharoni et al., 2019](https://www.aclweb.org/anthology/N19-1388)) 
+ **The Nature of Multilingual Representations.**
  + the encoder learns similar representations for similar sentences across languages ([Dabre et al., 2017](https://arxiv.org/abs/1702.06135); [Johnson et al., 2017](http://aclweb.org/anthology/Q17-1024)).
  + [Kudugunta et al., 2019](http://aclanthology.lst.uni-saarland.de/D19-1167.pdf)do a systematic study of representations generated from a massively mul- tilingual system using SVCCA.
+ **Encoder Representations.**
  + an attention bridge network generates a fixed number of contextual representations that are input to the attention network ([Lu et al., 2918](http://aclweb.org/anthology/W18-6309); [Vázquez et al., 2018](https://aclanthology.org/W19-4305/)).
  + [Hokamp et al., ](https://doi.org/10.18653/v1/W19-5319) opt for language-specific decoders.
  + [Murthy et al., ](https://www.aclweb.org/anthology/N19-1387) pointed out that the sentence representations generated by the encoder are dependent on the word order of the language and are, hence, language-specific.
+ **Decoder Representations.**
  + Language tag trick has been very effective in preventing vocabulary leakage ([Blackwood et al., 2018](http://aclweb.org/anthology/C18-1263); [Johnson et al., 2017](http://aclweb.org/anthology/Q17-1024)).
  + [Hokamap et al., 2019](https://doi.org/10.18653/v1/W19-5319) showed that using separate decoders and attention mechanisms gives better results as compared to a shared decoder and attention mechanism.
  + cluster languages into language families and train separate MNMT models per family ([Prasanna et al., 2018](http://hdl.handle.net/2433/232411); [Tan et al., 2019](https://doi.org/10.18653/v1/D19-1089)).
+ **Impact of Language Tag**
  + [Wang et al., 2018](https://arxiv.org/abs/1902.03499) explored three multiple methods for supporting multiple target languages.
  + [Hokamap et al., 2019](https://doi.org/10.18653/v1/W19-5319) showed that in a shared decoder setting, using a task- specific (language pair to be translated) embedding works better than using language tokens.

#### Training Protocols

There are two main types of training approaches: single stage or parallel or joint training and sequential or multi-stage training. Depending on the use-case, multi-stage training can be used for model compression (knowledge distillation) or fine- tuning, addition of data and/or languages (incremental training).

+ **Single Stage Parallel / Joint Training**
  + In this scenarios, we simply pre-process and concatenate the parallel corpora for all language pairs and then feed them to the model batch-wise. To avoid the imbalance training corpus in different language pair, the most common way is to upsampling the datasets or temperature-based methods.
+ **Knowledge Distillation**
  + [Tan et al., 2019](https://arxiv.org/abs/1902.10461) trained bilingual models for all language pairs involved and then these bilingual models are used as *teacher models* to train a single *student model* for all language pairs.
+ **Incremental Traning**
  + These approaches aim to decrease the cost of incorporating new languages or data in multilingual models by avoiding expensive retraining. A practical concern with training MNMT in an incremental fashion is dealing with vocabulary.
  + [Lakew et al., 2018](https://arxiv.org/abs/1811.01137) updated the vocabulary of the parent model with the low-resource language pair’s vocabulary before trans- ferring parameters. Embeddings of words that are common between the low- and high-resource languages are left untouched and randomly initialized embeddings may be used for as yet unseen words.
  + [Escolano et al., 2019](https://aclanthology.org/P19-2033/) focused on first training bilingual models and then gradually increasing their capacity to include more languages.
  + [Bapna et al., 2019](https://aclanthology.org/D19-1165.pdf) proposed expanding the capacities of pre-trained MNMT models (especially those trained on massive amounts of multilin- gual data) using tiny feed-forward components that they call **adaptors**. The adapters can also boost in massively multilingual models ([Aharoni et al., 2019](https://www.aclweb.org/anthology/N19-1388); [Arivazhagan et al., 2019](https://arxiv.org/abs/1907.05019)). Note that, this methods can be used to incorporate new data, but new language pairs cannot be added.
+ **Other methods**
  + [Jean et al., 2019](https://arxiv.org/abs/1909.06434) propose to focus on scaling learning rates or gradients differently for high-resource and low-resource language pairs.
  + [Kiperwasser et al., 2018](https://aclanthology.org/Q18-1017.pdf) proposed a multi- task learning model for learning syntax and translation, where they showed different effects of their model for high-resource and low-resource language pairs.

## MNMT for Low-resource Languages Pairs

#### Training

+ Most studies have explored **transfer learning** on the source-side: The high-resource and low- resource language pairs share the same target language. The simplest approach is **jointly training** both language pairs ([Johson et al., 2017](http://aclweb.org/anthology/Q17-1024)).
+  [Zoph et al., 2016](https://aclanthology.org/D16-1163/) proposed to **fine-tune** the parent model with data from the chlid language pair.
+ [Gu et al., 2018](http://aclweb.org/anthology/D18-1398) used the model-agnostic meta-learning (MAML) framework to learn appropriate parameter initialization from the parent pair(s) by taking the child pair into consideration. These methods called **meta-learning**.
+ [Dabre et al., 2019](https://aclanthology.org/D19-1146/) showed that a multi-stage fine-tuning process is beneficial when multiple target languages are involved.

#### Lexical Transfer

+ [Zoph et al., 2016](https://aclanthology.org/D16-1163/) **randomly initialized the word embeddings** of the child source language, because those could not be transferred from the parent. However, this approach does not map the embeddings of similar words across the source languages a priori.
+ [Gu et al., 2018](https://aclanthology.org/N18-1032.pdf) improved on the simple initialization by **mapping pre-trained monolingual word embeddings** of the parent and child sources to a common vector space.
+ the parent model is first trained and monolingual word-embeddings of the child source are mapped to the parent source’s embeddings prior to fine-tuning ([Kim et al., 2019](https://arxiv.org/abs/1905.05475)).

#### Syntactic Transfer

+ [Murthy et al., 2019](https://www.aclweb.org/anthology/N19-1387) showed that reducing the word order divergence between source languages by reordering the parent sentences to match child word or- der is beneficial in extremely low-resource scenarios.
+ [Kim et al., 2019](https://arxiv.org/abs/1905.05475) took a different approach to mit- igate syntactic divergence. They trained the parent encoder with noisy source data introduced via probabilistic insertion and deletion of words as well as permutation of word pairs.
+ [Gu et al., 2018](https://aclanthology.org/N18-1032.pdf) pro- posed to achieve better transfer of syntax-sensitive contextual representations from parents using a mixture of language experts network.

#### Language Relatedness

+ [Zoph et al., 2016](https://aclanthology.org/D16-1163/) and [Dabre et al., 2019](https://aclanthology.org/D19-1146/) empirically showed that a related parent language benefits the child language more than an unrelated parent.
+ [Maimaiti et al., 2019](https://dl.acm.org/doi/10.1145/3314945) further showed that using multiple highly related high-resource language pairs and applying fine-tuning in multi- ple rounds can improve translation performance more, compared to only using one high-resource language pair for transfer learning.
+ [Kocmi et al., 2018](http://www.aclweb.org/anthology/W18-6325) suggest that size of the parent is more important.
+ [Wang et al., 2019](https://arxiv.org/abs/1905.08212) proposed selection of sen- tence pairs from the parent task based on the similarity of the parent’s source sentences to the child’s source sentences.

## MNMT for unseen language pairs

#### Pivot Translation

+ The simplest approach to pivot translation is building independent source-pivot (S-P) and pivot- target (P-T) MT systems. This simple process has two limitations due to its pipeline characteristic: (a) translation errors compound in a pipeline; (b) decoding time is doubled, since inference has to be run twice.

#### Zero-shot Translation

+ **Challenges of Zero-shot Translation**
  + Spurious correlations between input and output language.
  + Language variant encoder representations
+ **Minimize divergence between encoder representation**
  + [Arivazhagan et al., 2019](https://arxiv.org/abs/1903.07091) suggested an unsupervised approach to align the source and pivot vector spaces by minimizing a domain adversarial loss.
  + [Ji et al., 2019](https://arxiv.org/abs/1912.01214) proposed to use pre-trained cross-lingual encoders trained using multilingual MLM, XLM, and BRLM objectives to obtain language-invariant encoder repre- sentations.
  + [Sen et al., 2019](https://aclanthology.org/P19-1297/) used denoising autoencoding and back-translation to obtain language- invariant encoder representations.
+ **Encourage output agreement**
  + [Al-Shedivat et al., 2019](https://www.aclweb.org/anthology/N19-1121) incorporated additional terms in the training objective to encourage source and pivot representations of parallel sentences to generate similar output sentences (synthetic) in an auxiliary language (possibly an unseen pair).
  + [Pham et al., 2019](https://www.statmt.org/wmt19/pdf/52/WMT02.pdf) incorporate additional loss terms that encourage the attention-context vec- tors as well as decoder output representations to agree while generating the same pivot output.
  + [Xu et al., 2019](https://doi.org/10.24963/ijcai.2019/739) considered different translation paths among multiple languages in unsupervised NMT by designing training objectives for these paths to achieve the same goal.
+ **Effect of corpus size and number of languages**
  + [Arivazhagan et al., 2019](https://arxiv.org/abs/1903.07091) showed that cosine distance-based alignment can be scaled to a small set of languages.
  + Some studies suggest that zero-shot translation works reasonably well only when the multilingual parallel corpora is large ([Lakew et al., 2018](https://arxiv.org/abs/1811.01389); [Mattoni et al., 2017](https://research.tilburguniversity.edu/en/publications/zero-shot-translation-for-indian-languages-with-sparse-data)).
+ **Addressing wrong language generation**
  + [Ha et al., 2017](https://arxiv.org/abs/1711.07893) proposed to filter the output of the softmax, forcing the model to translate into the desired language.

#### Zero-resource Translation

+ **Synthetic Corpus Generation**
  + Using back-translation and pivot-translation.  Some works have shown that this approach can outperform the pivot translation approach ([Firat et al., 2016](https://aclanthology.org/D16-1026/); [Gu et., 2019](http://aclanthology.lst.uni-saarland.de/P19-1121.pdf); [Lakew et al., 2018](https://arxiv.org/abs/1811.01389)).
  + Also can use monolingual data only ([Currey et al., 2019](https://aclanthology.org/D19-5610/)).
+ **Iterative Approaches**
  + The S-T and T-S systems can be trained iteratively such that the two directions reinforce each other ([Lakew et al., 2018](https://arxiv.org/abs/1811.01389)).
  + [Setorain et al., 2018](https://arxiv.org/abs/1805.10338) jointly trained both the models incorporating language modelling and reconstruction objectives via reinforcement learning.
+ **Teacher-student Training**
  + [Chen et al., 2017](https://arxiv.org/abs/1705.00753) assumed that the source and pivot sentences of the S-P parallel corpus will generate similar probability distributions for translating into a third lan- guage (target).
+ **Combining Pre-trained Encoders and Decoders**
  + [Kim et al., 2019](https://aclanthology.org/D19-1080.pdf) combined S-P encoder with P-T decoder to create the S-T model.

## Multi-Source NMT

If a source sentence has already been translated into multiple languages, then these sentences can be used together to improve the translation into the target language. This technique is known as multi-source MT. In this part, I think the table following illustrate detailly in the multi-source scenarios. For more details, please see in the paper.

![multi-source NMT](https://github.com/lavine-lmu/lavine_blog/raw/main/assets/paper-notes/MNMT-Survey/multi_source.png)

## Multilingual in Older Paradigms

In this part, they discuss some MNMT usage in traditional rule-based method and statistic machine translation. I recommend the readers to read more in the original paper if you are interested.

## Datasets and Resources

+ **Multiway**
  + TED corpus
  + UN corpus
  + European languages (JRC, DGT, ECDC,EAC)
  + WikiMatrix
  + JW300
  + Indic languages (CVIT-PIB, PMIndia, IndoWordNet), IndicNLP catalog
  + OPUS
+ **Low- or Zero-resource**
  + FLORES101
  + XNLI test set
  + CVIT-Mannki Baat
  + WMT
+ **Multi-Source**
  + Europarl
  + TED
  + UN
  + The Indian Language Corpora Initiative (ILCI) corpus
  + The Asian Language Treebank
  + MMCR4NLP project
  + Bible 1000 languages
+ Shared Tasks
  + IWSLT
  + WAT
  + WMT

## Future Research Directions

+ Exploring Pre-trained Models
+ Unseen language Pair Translation
+ Fast Multi-Source NMT
+ Related Languages, Language Registers, and Dialects
+ Code-mixed Language
+ Visualization and Model Inspection
+ Learning Effective Language Representations
+ Multiple Target Language MNMT
+ Representation Bottleneck
+ Joint Multilingual and Multi-domain NMT
+ Multilingual Speech-to-speech NMT



__BibTeX Reference__

```bibtex
@article{10.1145/3406095,
  author = {Dabre, Raj and Chu, Chenhui and Kunchukuttan, Anoop},
  title = {A Survey of Multilingual Neural Machine Translation},
  year = {2020},
  issue_date = {September 2021},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {53},
  number = {5},
  issn = {0360-0300},
  url = {https://doi.org/10.1145/3406095},
  doi = {10.1145/3406095},
  month = {sep},
  articleno = {99},
  numpages = {38},
  keywords = {low-resource, multilingualism, zero-shot, Neural machine translation, multi-source, survey}
}
```