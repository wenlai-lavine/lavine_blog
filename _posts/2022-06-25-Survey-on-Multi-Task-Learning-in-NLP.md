---
layout: post
title: Survey on Multi-Task Learning in NLP
categories: paper-notes
tags: [NLP, Multi-Task-Learning]
---

In this post, I would like to introduce a survey paper titled [Multi-Task Learning in Natural Language Processing: An Overview](https://arxiv.org/abs/2109.09138). Here, I also recommend the readers to read another survey paper on Multi-Task Learning in NLP ([Multi-task learning for natural language processing in the 2020s: where are we going?](https://arxiv.org/abs/2007.16008)) and a classical paper ([Multi-Task Deep Neural Networks for Natural Language Understanding](https://aclanthology.org/P19-1441.pdf)).

## Introduction

Current NLP models have some problems: i) require a large amount of labeled training samples; ii) immense computing power (huge time and storage budget); iii) data scarcity. Multi-Task Learning (MTL) is an effective methods to alleivate this problems.

MTL trains machine learning models from multiple related tasks simultaneously or enhances the model for a specific task using auxiliary tasks. The tasks in MTL can be categories into: 

+ tasks with assumed relatedness;
+ tasks with different styles of supervision;
+ tasks with different types of goals;
+ tasks with different levels of features;
+ Tasks in different modalities.

In this survey paper, the authors focus on the ways in which researchers apply MTL to NLP tasks, including model architecture, training process, and data source.

## MTL Architectures for NLP Tasks

MTL architectures can be categories into four classes: 1) **paralllel architecture**; 2) **hierarchical architecture**; 3) **modular architecture**; 4) **generative adversarial architecture**. Note that the boundaries between different categories are not always solid and hence a specific model may fit into multiple classes.

There are two definitions need to be clarified: hard and soft parameter sharing. Hard parameter sharing refers to sharing the same model parameters among multiple tasks. Soft parameter sharing, on the other hand, constrains a distance metric between the intended parameters.

### Parallel Architectures

As its name suggests, the model for each task run in parallel under the parallel architecture, which is implemented by sharing certain intermediate layers. In this case, there is no dependency other than layer sharing among tasks. Therefore, there is no constraint on the order of training samples from each task. During training, the shared parameters receive gradients from samples of each task, enabling knowledge sharing among tasks.

##### 1. Vanilla Tree-like Architectuers

In this architecture, the models for different tasks share a base feature extractor (i.e., the trunk) followed by task-specific encoders and output layers (i.e., the branches). A shallow trunk can be simply the word representation layer [108] while a deep trunk can be the entire model except output layers. The tree-like architecture is proposed by [Caruana et al., 1997](https://link.springer.com/article/10.1023/A:1007379606734). Such tree-like architecture is also known as hard sharing architecture or multi-head architecture, where each head corresponds to the combination of a task-specific encoder and the corresponding output layer or just a branch.

The tree-like architecture uses a single trunk to force all tasks to share the same low-level feature representation, which may limit the expressive power of the model for each task. A solution is to equip the shared trunk with task-specific encoders and another way is to make different groups of tasks share different parts of the trunk.

##### 2. Parallel Feature Fusion

In this architecture, models can use a globally shared encoder to produce shared representations that can be used as additional features for each task-specific model.

However, different parts of the shared features are not equally important to each task. To selectively choose features from the shared features and alleviate inter-task interference, several models have been devised to control information flow between tasks. Some models directly integrate features from different tasks. A straightforward solution is to compute a weighted average of feature representations. Task routing is another way to achieve feature fusion, where the paths that samples go through in the model differ by their task.

##### 3. Supervision at Different Feature Levels

Models using the parallel architecture handle multiple tasks in parallel. These tasks may concern features at different abstraction levels. For NLP tasks, such levels can be character-level, token-level, sentence-level, paragraph-level, and document-level. It is natural to give supervision signals at different depths of an MTL model for tasks at different levels. In some settings where MTL is used to improve the performance of a primary task, the introduction of auxiliary tasks at different levels could be helpful.

### Hierarchical Architectures

The hierarchical architecture considers hierarchical relationships among multiple tasks. The features and output of one task can be used by another task as an extra input or additional control signals. The design of hierarchical architectures depends on the tasks at hand and is usually more complicated than parallel architectures.

##### 1. Hierarchical Feature Fusion

Different from parallel feature fusion that combines features of different tasks at the same depth, hierarchical feature fusion can explicitly combine features at different depths and allow different processing for different features.

##### 2. Hierarchical Pipeline

Instead of aggregating features from different tasks as in feature fusion architectures, pipeline architectures treat the output of a task as an extra input of another task and form a hierarchical pipeline between tasks. The extra input can be used directly as input features or used indirectly as control signals to enhance the performance of other tasks. The pipeline can further divided  into *hierarchical feature pipeline* and *hierarchical signal pipeline*.

***In hierarchical feature pipeline***, the output of one task is used as extra features for another task. The tasks are assumed to be directly related so that outputs instead of hidden feature representations are helpful to other tasks. Hierarchical feature pipeline is especially useful for tasks with hierarchical relationships. ***In hierarchical signal pipeline***, the outputs of tasks are used indirectly as external signals to help improve the performance of other tasks.

##### 3. Hierarchical Interactive Architecture

Different from most machine learning models that give predictions in a single pass, hierarchical interactive MTL explicitly models the interactions between tasks via a multi-turn prediction mechanism which allows a model to refine its predictions over multiple steps with the help of the previous outputs from other tasks in a way similar to recurrent neural networks.

### Modular Architectures

The idea behind the modular MTL architecture is simple: breaking an MTL model into **shared modules** and **task-specific modules**. The shared modules learn shared features from multiple tasks. Since the shared modules can learn from many tasks, they can be sufficiently trained and can generalize better, which is particularly meaningful for low-resource scenarios. On the other hand, task-specific modules learn features that are specific to a certain task. Compared with shared modules, task-specific modules are usually much smaller and thus less likely to suffer from overfitting caused by insufficient training data.

### Generative Adversarial Architectures

This architecture introduce a discriminator 𝐺 that predicts which task a given training instance comes from, the shared feature extractor 𝐸 is forced to produce more generalized task-invariant features and therefore improve the performance and robustness of the entire model. An additional benefit of generative adversarial architectures is that unlabeled data can be fully utilized.

## Optimization for MTL Models

### 1. Loss Construction

The most common approach to train an MTL model is to linearly combine loss functions of different tasks into a single global loss function. An important question is how to assign a proper weight 𝜆𝑡 to each task. The simplest way is to set them equally.

In addition to combining loss functions from different tasks, researchers also use additional adaptive loss functions L𝑎𝑑𝑎𝑝𝑡 to enhance MTL models.

Aside from studying how to combine loss functions of different tasks, some studies optimize the training process by manipulating gradients.

### 2. Data Sampling

Machine learning models often suffer from imbalanced data distributions. MTL further complicates this issue in that training datasets of multiple tasks with potentially different sizes and data distributions are involved. To handle data imbalance, various data sampling techniques have been proposed to properly construct training datasets. the following sampling strategies are studied:

- proportional sampling;
- Task-oriented sampling;
- Anneled sampling;
- Etc;

### 3. Task Scheduling

Task scheduling determines the order of tasks in which an MTL model is trained. A naive way is to train all tasks together.

Alternatively, we can train an MTL model on different tasks at different steps. Similar to the data sampling, we can assign a task sampling weight 𝑟𝑡 for task 𝑡, which is also called mixing ratio, to control the frequency of data batches from task 𝑡 . The most common task scheduling technique is to shuffle between different tasks, either randomly or according to a pre-defined schedule.

Instead of using a fixed mixing ratio designed by hand, some researchers explore using a dynamic mixing ratio during the training process.

Besides probabilistic approaches, task scheduling could also use heuristics based on certain performance metrics.

For auxiliary MTL, some researchers adopt a *pre-train then fine-tune* methodology, which trains auxiliary tasks first before fine-tuning on the down-stream primary tasks.

## Application Problems

In this section, we summarize the application of multi-task learning in NLP tasks, including applying MTL to optimize certain primary tasks (corresponding to the auxiliary MTL setting), to jointly learn multiple tasks (corresponding to the joint MTL setting), and to improve the performance in multi-lingual multi-task and multimodal scenarios. Existing research works have also explored different ways to improve the performance and efficiency of MTL models, as well as using MTL to study the relatedness of different tasks.

+ **Auxiliary MTL**: Auxiliary MTL aims to improve the performance of certain primary tasks by introducing auxiliary tasks and is widely used in the NLP field for different types of primary task, including sequence tagging, classification, text generation, and representation learning.
+ **Joint MTL**: Different from auxiliary MTL, joint MTL models optimize its performance on several tasks simultaneously. Similar to auxiliary MTL, tasks in joint MTL are usually related to or complementary to each other.
+ **Multilingual MTL**: Since a single data source may be limited and biased, leveraging data from multiple languages through MTL can benefit multi-lingual machine learning models.
+ **Multimodal MTL**: Researchers have incorporated features from other modalities, including auditory, visual, and cognitive features, to perform text-related cross-modal tasks. MTL is a natural choice for implicitly injecting multimodal features into a single model.
+ **Task Relatedness in MTL**: A key issue that affects the performance of MTL is how to properly choose a set of tasks for joint training. Generally, tasks that are similar and complementary to each other are suitable for multi-task learning, and there are some works that studies this issue for NLP tasks.

## Data Source and Multi-Task Benchmark

### 1. Data Source

Given𝑀taskswithcorrespondingdatasetsD𝑡 ={X𝑡,Y𝑡},𝑡=1,...,𝑀,whereX𝑡 denotesthesetofdatainstances in task 𝑡 and Y𝑡 denotes the corresponding labels, we denote the entire dataset for the 𝑀 tasks by D = {X, Y}. We describe different forms of D in the following sections.

+ **Disjoint Dataset**: In most multi-task learning literature, the training sets of different tasks have distinct label spaces. The most popular way to train MTL models on such tasks is to alternate between different tasks.
+ **Multi-label Dataset**: Multi-label datasets can be created by giving extra manual annotations to existing data.

### 2. Multi-task Benchmark Dataset

+ **GLUE**: is a benchmark dataset for evaluating natural language understanding (NLU) models;

+ **SuperGLUE**: is a generalization of GLUE;

+ **Measuring Massive Multitask Language Understanding (MMMLU)** : is a multi-task few-shot learn- ing dataset for world knowledge and problem solving abilities of language processing models;

+ **Xtreme**: is a multi-task benchmark dataset for evaluating cross-lingual generalization capabilities of multi-lingual representations covering 9 tasks in 40 languages;

+ **XGLUE**: is a benchmark dataset that supports the development and evaluation of large cross-lingual pre-trained language models;

+ **LSPard**: is a multi-task semantic parsing dataset with 3 tasks, including question type classification, entity mention detection, and question semantic parsing;

+ **ECSA**: is a dataset for slot filling, named entity recognition, and segmentation to evaluate online shopping assistant systems in Chinese;

+ **ABC**: the Anti-reflexive Bias Challenge, is a multi-task benchmark dataset designed for evaluating gender assumptions in NLP models;

+ **CompGuessWhat?!**: is a dataset for grounded language learning with 65,700 collected dialogues;

+ **SCIERC**: is a multi-label dataset for identifying entities, relations, and cross-sentence coreference clus-

  ters from abstracts of research papers.

## some other resources about Multi-Task Learning

+ [Multi-Task Learning with Deep Neural Networks: A Survey](https://arxiv.org/abs/2009.09796)
+ [A Survey on Multi-Task Learning](https://arxiv.org/abs/1707.08114)
+ [**Blog**: An Overview of Multi-Task Learning in Deep Neural Networks](https://ruder.io/multi-task/)
+ [papers](https://github.com/Manchery/awesome-multi-task-learning)





__BibTeX Reference__

```bibtex
@article{chen2021multi,
  title={Multi-task learning in natural language processing: An overview},
  author={Chen, Shijie and Zhang, Yu and Yang, Qiang},
  journal={arXiv preprint arXiv:2109.09138},
  year={2021}
}
```