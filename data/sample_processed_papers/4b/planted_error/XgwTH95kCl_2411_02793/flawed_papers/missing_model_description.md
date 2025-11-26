# Toward Robust Incomplete Multimodal Sentiment Analysis via Hierarchical Representation Learning

## Abstract

<span id="Abstract" label="Abstract"></span> Multimodal Sentiment Analysis (MSA) is an important research area that aims to understand and recognize human sentiment through multiple modalities. The complementary information provided by multimodal fusion promotes better sentiment analysis compared to utilizing only a single modality. Nevertheless, in real-world applications, many unavoidable factors may lead to situations of uncertain modality missing, thus hindering the effectiveness of multimodal modeling and degrading the model’s performance. To this end, we propose a Hierarchical Representation Learning Framework (HRLF) for the MSA task under uncertain missing modalities. Specifically, we propose a fine-grained representation factorization module that sufficiently extracts valuable sentiment information by factorizing modality into sentiment-relevant and modality-specific representations through crossmodal translation and sentiment semantic reconstruction. Moreover, a hierarchical mutual information maximization mechanism is introduced to incrementally maximize the mutual information between multi-scale representations to align and reconstruct the high-level semantics in the representations. Ultimately, we propose a hierarchical adversarial learning mechanism that further aligns and adapts the latent distribution of sentiment-relevant representations to produce robust joint multimodal representations. Comprehensive experiments on three datasets demonstrate that HRLF significantly improves MSA performance under uncertain modality missing cases.

# Introduction [Introduction]

Multimodal sentiment analysis (MSA) has attracted wide attention in recent years. Unlike unimodal emotion recognition tasks `\cite{du2021learning,yang2024towards,yang2024robust,yang2023context,yang2022emotion}`{=latex}, MSA understands and recognizes human emotions through multiple modalities, including language, audio, and visual `\cite{morency2011towards,yang2024TSIF}`{=latex}. Previous studies have shown that combining complementary information among different modalities facilitates valuable semantic generation `\cite{springstein2021quti,shraga2020web,yang2023target,yang2022contextual,yang2024SuCi}`{=latex}. MSA has been well studied so far under the assumption that all modalities are available in the training and inference phases `\cite{hazarika2020misa, yu2021learning, yang2022disentangled, yang2022learning, yang2022emotion, li2023decoupled,yang2024asynchronous,yang2024MCIS}`{=latex}. Nevertheless, in real-world applications, modalities may be missing due to security concerns, background noises, sensor limitations and so on. Ultimately, these incomplete multimodal data significantly hinder the performance of MSA. For instance, as shown in <a href="#fig:example" data-reference-type="ref+Label" data-reference="fig:example">1</a>, the entire visual modality and some frame-level features in the language and audio modalities are missing, leading to an incorrect prediction.

In recent years, many studies `\cite{du2018semi, luo2023multimodal, lian2023gcnet, wang2023distribution, pham2019found, wang2020transmodality, zhao2021missing, zeng2022tag, yuan2023noise, liu2024modality,li2023towards,li2024unified}`{=latex} attempt to address the problem of missing modalities in MSA. For example, SMIL `\cite{ma2021smil}`{=latex} estimates the latent features of the missing modality data via Bayesian Meta-Learning. However, these methods are constrained by the following factors: **(i)** Implementing complex feature interactions for incomplete modalities leads to a large amount of information redundancy and cumulative errors, resulting in ineffective extraction of sentiment semantics. **(ii)** Lacking consideration of semantic and distributional alignment of representations, causing imprecise feature reconstruction and nonrobust joint representations.

<span id="sec:intro" label="sec:intro"></span>

<figure id="fig:example">
<img src="./figures/intro.png"" style="width:100.0%" />
<figcaption> A case of incorrect prediction by the traditional model with missing modalities. The pink and yellow areas indicate intra- and inter-modality missingness, respectively. </figcaption>
</figure>

To address the above issues, we propose a Hierarchical Representation Learning Framework (HRLF) for the MSA task under uncertain missing modalities. HRLF has three core contributions: **()** We present a fine-grained representation factorization module that sufficiently extracts valuable sentiment information by factorizing modality into sentiment-relevant and modality-specific representations through intra- and inter-modality translations and sentiment semantic reconstruction. **()** Furthermore, a hierarchical mutual information maximization mechanism is introduced to incrementally align the high-level semantics by maximizing the mutual information of the multi-scale representations of both networks in knowledge distillation. **()** Eventually, we propose a hierarchical adversarial learning mechanism to progressively align the latent distributions of representations leveraging multi-scale adversarial learning. Based on these components, HRLF significantly improves MSA performance under uncertain modality missing cases on three multimodal benchmarks.

# Related Work

## Multimodal Sentiment Analysis

Multimodal Sentiment Analysis (MSA) seeks to comprehend and analyze human sentiment by utilizing diverse modalities. Unlike conventional single-modality sentiment recognition, MSA poses greater challenges owing to the intricate nature of processing and analyzing heterogeneous data across modalities. Mainstream studies in MSA `\cite{zadeh2017tensor,zadeh2018memory,tsai2019multimodal,hazarika2020misa,han2021improving, sun2022cubemlp,li2023decoupled}`{=latex} focus on designing complex fusion paradigms and interaction mechanisms to improve MSA performance. For instance, CubeMLP `\cite{sun2022cubemlp}`{=latex} employes three distinct multi-layer perceptron units for feature amalgamation along three axes. However, these methods rely on complete modalities and thus are impractical for real-world deployment. There are two primary approaches for addressing the missing modality problem in MSA: (1) Generative methods `\cite{du2018semi, luo2023multimodal, lian2023gcnet, wang2023distribution}`{=latex} and (2) joint learning methods `\cite{pham2019found, wang2020transmodality, zhao2021missing, zeng2022tag, yuan2023noise, liu2024modality}`{=latex}. Generative methods aim to regenerate missing features and semantics within modalities by leveraging the distributions of available modalities. For example, TFR-Net `\cite{yuan2021transformer}`{=latex} employes a feature reconstruction module to guide the extractor to reconstruct missing semantics. Joint learning methods focus on deriving cohesive joint multimodal representations based on inter-modality correlations. For instance, MMIN `\cite{zhao2021missing}`{=latex} produces robust joint multimodal representations via cross-modality imagination. However, these methods cannot extract rich sentiment information from incomplete modalities due to their inefficient interaction. In contrast, our learning paradigm achieves effective extraction and precise reconstruction of sentiment semantics through complete modality factorization.

## Factorized Representation Learning

The fundamental goal of learning factorized representations is to disentangle representations that have different semantics and distributions. This separation enables the model to more effectively capture intrinsic information and yield favorable modality representations. Previous methods of factorized representation learning primarily rely on auto-encoders `\cite{bousmalis2016domain}`{=latex} and generative adversarial networks `\cite{odena2017conditional}`{=latex}. For example, FactorVA `\cite{kim2018disentangling}`{=latex} is introduced to achieve factorization by leveraging the characteristic that representations are both factorial and independent in dimension. Recently, factorization learning has been progressively utilized in MSA tasks `\cite{yang2022disentangled, li2023decoupled, yang2022learning}`{=latex}. For instance, FDMER `\cite{yang2022disentangled}`{=latex} utilizes consistency and discreteness constraints between modalities to disentangle modalities into modality-invariant and modality-private features. DMD `\cite{li2023decoupled}`{=latex} disentangles each modality into modality-independent and modality-exclusive representations and then implements a knowledge distillation strategy among the representations with dynamic graphs. MFSA `\cite{yang2022learning}`{=latex} refines multimodal representations and learns complementary representations across modalities by learning modality-specific and modality-agnostic representations. Despite the progress these studies have brought to MSA, certain limitations persist: (i) The supervision of the factorization process is coarse-grained and insufficient. (ii) Focusing solely on factorizing distinct representations at the modality level, without taking into account sentimentally beneficial and relevant representations. By contrast, the proposed method decomposes sentiment-relevant representations precisely through intra- and inter-modality translation and sentiment semantic reconstruction. Furthermore, hierarchical mutual information maximization and adversarial learning paradigms are employed to refine and optimize the representation of factorization at the semantic level and the distributional level, respectively, thus yielding robust joint multimodal representations.

## Knowledge Distillation

Knowledge distillation leverages additional supervisory signals from a pre-trained teacher network to aid in training a student network `\cite{hinton2015distilling}`{=latex}. There are generally two categories of knowledge distillation methods: distillation from intermediate features `\cite{heo2019comprehensive, heo2019knowledge, kim2018paraphrasing, park2019relational, peng2019correlation, romero2014fitnets, tung2019similarity, tian2019contrastive, yim2017gift, zagoruyko2016paying}`{=latex} and distillation from logits `\cite{cho2019efficacy, furlanello2018born, mirzadeh2020improved, yang2019snapshot, zhao2017pyramid}`{=latex}. Many studies `\cite{cho2021dealing, hu2020knowledge, rahimpour2021cross, kumar2019online, wang2023learnable, xia2023robust}`{=latex} utilize knowledge distillation for MSA tasks with missing modalities. These approaches aim to transfer dark knowledge from teacher networks trained on complete modalities to student networks trained by missing modalities. The teacher network typically provides richer and more comprehensive feature representations than the student network. For instance, KD-Net `\cite{hu2020knowledge}`{=latex} utilizes a teacher network with complete modalities to supervise the unimodal student network at both the feature and logits levels. Despite their promising results, these methods neglect precise supervision of representations, resulting in low-quality knowledge transfer. To this end, we implement hierarchical semantic and distributional alignment of the multi-scale representations of both networks to transfer knowledge effectively. <span id="sec:related" label="sec:related"></span>

<figure id="fig:overall_framework1">
<img src="./figures/framework.png"" style="width:100.0%" />
<figcaption>The structure of our HRLF, which consists of three core components: Fine-grained Representation Factorization (FRF) module, Hierarchical Mutual Information (HMI) maximization mechanism, and Hierarchical Adversarial Learning (HAL) mechanism. </figcaption>
</figure>

# Methodology

## Problem Formulation

Given a multimodal video segment with three modalities as \\(\bm{S}=[\bm{X}_L, \bm{X}_A, \bm{X}_V]\\), where \\(\bm{X}_L\in \mathbb{R}^{T_L\times d_L}, \bm{X}_A \in \mathbb{R}^{T_A\times d_A}\\), and \\(\bm{X}_V\in \mathbb{R}^{T_V\times d_V}\\) denote language, audio, and visual modalities, respectively. \\(\mu=\{L,A,V\}\\) denotes the set of modality types. \\(T_{m}(\cdot)\\) is the sequence length and \\(d_{m}(\cdot)\\) is the embedding dimension, where \\(m \in \mu\\). We define two missing modality cases to simulate the most natural and holistic challenges in real-world scenarios: (1) *intra-modality missingness*, which indicates some frame-level features in the modality sequences are missing. (2) *inter-modality missingness*, which denotes some modalities are entirely missing. We aim to recognize the utterance-level sentiments using incomplete multimodal data.

## Overall Framework

<a href="#fig:overall_framework1" data-reference-type="ref+Label" data-reference="fig:overall_framework1">2</a> illustrates the main workflow of HRLF. The teacher and student networks adopt a consistent structure but have different parameters. During the training phase, the workflow of our HRLF is as follows: () We first train the teacher network with complete-modality samples and their sentiment labels. () Given a video segment sample \\(\bm{S}\\), we generate a missing-modality sample \\(\hat{\bm{S}}\\) with the Modality Stochastic Missing (MSM) strategy. MSM simultaneously performs intra-modality missingness and inter-modality missingness. \\(\bm{S}\\) and \\(\hat{\bm{S}}\\) are fed into the pre-trained teacher network and the initialized student network, respectively. () We input each sample into the FRF module, to factorize each modality into a sentiment-relevant representation \\(\bm{Q}_m\\) and a modality-specific representation \\(\bm{U}_m\\), where \\(m \in \mu\\). () Sequences \\([\bm{C}_L,\bm{C}_A, \bm{C}_V]\\) and \\([\bm{C}^\prime_L,\bm{C}^\prime_A, \bm{C}^\prime_V]\\) are generated by concatenating \\(\bm{Q}_m\\) and \\(\bm{U}_m\\) from all modalities in the teacher and student networks. Each element of the sequences is concatenated to yield the joint multimodal representations \\(\bm{H}^t\\) and \\(\bm{H}^s\\). () The multi-scale representations of both networks are obtained by passing \\(\bm{H}^t\\) and \\(\bm{H}^s\\) through the fully-connected layers. The proposed HMI and HAL are used to align the semantics and distribution between the multiscale representations. () The outputs \\(\tilde{\bm{H}}^t\\) and \\(\tilde{\bm{H}}^s\\) of the fully-connected layers are fed into the task-specific classifier to get logits \\(\bm{L}^t\\) and \\(\bm{L}^s\\). We constrain the consistency between logits and utilize \\(\bm{L}^s\\) to implement the sentiment prediction. In the inference phase, testing samples are only fed into the student network for downstream tasks.

## Fine-grained Representation Factorization

Modality missing leads to ambiguous sentiment cues in the modality and information redundancy in multimodal fusion. It hinders the model from capturing valuable sentiment semantics and filtering sentiment irrelevant information. Although previous studies in MSA `\cite{hazarika2020misa, yang2022disentangled}`{=latex} decompose the task-relevant semantics contained in the modality to some extent via simple auto-encoder networks with reconstruction constraints, their purification of sentiment semantics is inadequate, and they cannot be applied to modality missing scenarios. Therefore, we propose a Fine-grained Representation Factorization (FRF) module to capture sentiment semantics in modalities. The core idea is to factorize each modality representation into two types of representations: (1) sentiment-relevant representation, which contains the holistic sentiment semantics of the sample. It is modality-independent, shared across all modalities of the same subject, and robust to modality missing situations. (2) modality-specific representation, which represents modality-specific task-independent information.

As shown in <a href="#fig:overall_framework1" data-reference-type="ref+Label" data-reference="fig:overall_framework1">2</a>, FRF receives the multimodal sequences \\([\bm{X}_L, \bm{X}_A, \bm{X}_V]\\) with modality number \\(n=3\\). The modality \\(\bm{X}_\alpha\\) with \\(\alpha \in \mu\\) passes through a 1D temporal convolutional layer with kernel size \\(3 \times 3\\) and adds the positional embedding `\cite{vaswani2017attention}`{=latex} to obtain the preliminary representations, denoted as \\(\bm{R}_\alpha = \bm{W}_{3 \times 3}(\bm{X}_\alpha) + PE(T_\alpha, d)\in\mathbb{R}^{T_\alpha \times d}\\). The \\(\bm{R}_\alpha\\) is fed into a Transformer `\cite{vaswani2017attention}`{=latex} encoder \\(\mathcal{F}_\alpha(\cdot)\\), and the last element of its output is denoted as \\(\bm{Z}_\alpha = \mathcal{F}_\alpha(\bm{R}_\alpha) \in \mathbb{R}^d\\). The \\(\bm{Z}_\alpha \in \bm{\mathcal{Z}}_\alpha\\) is the low-level modality representation of the modality \\(\alpha\\). We aim to factorize modality representation \\(\bm{Z}_\alpha\\) into a sentiment-relevant representation \\(\bm{Q}_\alpha\\) by a sentiment encoder \\(\bm{Q}_\alpha = \mathcal{E}_\alpha^S(\bm{Z}_\alpha)\\) and a modality-specific representation \\(\bm{U}_\alpha\\) by a modality encoder \\(\bm{U}_\alpha = \mathcal{E}_\alpha^M(\bm{Z}_\alpha)\\). \\(\mathcal{E}_\alpha^S(\cdot)\\) and \\(\mathcal{E}_\alpha^M(\cdot)\\) are composed of multi-layer perceptrons with the ReLU activation. The following two processes ensure adequate factorization and semantic reinforcement of the above two representations.

**Intra- and Inter-modality Translation.** The proposed FRF effectively decouples sentiment-relevant and modality-specific representations by simultaneously performing intra- and inter-modality translations. Given a pair of representations \\(\bm{Q}_\alpha\\) and \\(\bm{U}_\beta\\) factorized by \\(\bm{Z}_\alpha\\) and \\(\bm{Z}_\beta\\) with \\(\alpha, \beta \in \mu\\), the decoder \\(\mathcal{D}_{r}(\cdot)\\) is supposed to translate and synthesize the representation \\(\overline{\bm{Z}}_{\alpha \beta}\\), whose reconstructed domain corresponds to the modality representation \\(\bm{Z}_\beta \in \bm{\mathcal{Z}}_\beta\\). The \\(\mathcal{D}_{r}(\cdot)\\) consists of feed-forward neural layers. The modality translations include intra-modality translation (*i.e.*, \\(\alpha=\beta\\)) and inter-modality translation (*i.e.*, \\(\alpha \neq \beta\\)), whose losses are respectively denoted as:

\\[\begin{aligned}
\mathcal{L}_{trans}^{self} =\frac{1}{n} \sum_{\alpha \in \mu} \mathbf{E}_{\bm{Z}_\alpha \sim \bm{\mathcal{Z}}_\alpha}\left[\left\|\overline{\bm{Z}}_{\alpha \alpha}-\bm{Z}_\alpha\right\|_2\right], 
\end{aligned}\\]

\\[\begin{aligned}
\mathcal{L}_{trans}^{cross}  =\frac{1}{n^2-n} \sum_{\alpha \in \mu} \sum_{\beta \in \mu, \beta \neq \alpha} \mathbf{E}_{\bm{Z}_\alpha \sim \bm{\mathcal{Z}}_\alpha, \bm{Z}_\beta \sim \bm{\mathcal{Z}}_\beta}\left[\left\|\overline{\bm{Z}}_{\alpha \beta}-\bm{Z}_\beta\right\|_2\right],
\end{aligned}\\] where \\(\overline{\bm{Z}}_{\alpha \beta} = \mathcal{D}_{r}(\mathcal{E}_\alpha^S(\bm{Z}_\alpha), \mathcal{E}_\beta^M(\bm{Z}_\beta))\\). The translation loss is denoted as: \\(\mathcal{L}_{trans} = \mathcal{L}_{trans}^{self} + \mathcal{L}_{trans}^{cross}\\).

**Sentiment Semantic Reconstruction.** To ensure that the reconstructed modality still contains the sentiment semantics from the original modality, we encourage both to maintain the consistency of sentiment-relevant semantics and utilize the following loss for constraints, denoted as: \\[\mathcal{L}_{recon}=\frac{1}{n^2} \sum_{\alpha \in \mu} \sum_{\beta \in \mu} \mathbf{E}_{\bm{Z}_\alpha \sim \bm{\mathcal{Z}}_\alpha, \bm{Z}_\beta \sim \bm{\mathcal{Z}}_\beta}\left[\left\|\overline{\bm{Q}}_{\beta \alpha}-\bm{Q}_\alpha\right\|_2\right],\\] where \\(\overline{\bm{Q}}_{\beta \alpha}=\mathcal{E}_\alpha^S\left(\mathcal{D}_{r}\left(\mathcal{E}_\beta^S\left(\bm{Z}_\beta\right), \mathcal{E}_\alpha^M\left(\bm{Z}_\alpha\right)\right)\right)\\) is the sentiment-relevant representation derived from the reconstructed modality representation. Consequently, the final loss of the FRF is denoted as: \\[\mathcal{L}_{FRF} = \mathcal{L}_{trans} + \mathcal{L}_{recon}.\\]

## Hierarchical Mutual Information Maximization

The underlying assumption of knowledge distillation is that layers in the pre-trained teacher network can represent certain attributes of given inputs that exist in the task `\cite{hinton2015distilling}`{=latex}. For successful knowledge transfer, the student network must learn to incorporate such attributes into its own learning. Nevertheless, previous studies `\cite{ hu2020knowledge, rahimpour2021cross,kumar2019online}`{=latex} based on knowledge distillation simply constrain the consistency between the features of both networks and lack consideration of the intrinsic semantics and inherent properties of the features, leading to semantic misalignment. From the perspective of information theory `\cite{ahn2019variational}`{=latex}, semantic alignment and attribute mining of representations can be characterized as maintaining high mutual information among the layers of the teacher and student networks. We construct a Hierarchical Mutual Information (HMI) maximization mechanism to implement sufficient semantic alignment and maximize mutual information. The core idea is to progressively align the semantics of representations through a hierarchical learning paradigm.

Specifically, the sentiment-relevant and modality-specific representations \\(\bm{Q}_m\\) and \\(\bm{U}_m\\) of all modalities for teacher and student networks are concatenated to obtain the sequences \\([\bm{C}_L,\bm{C}_A, \bm{C}_V]\\) and \\([\bm{C}^\prime_L,\bm{C}^\prime_A, \bm{C}^\prime_V]\\). Each element of the sequences is concatenated to yield the joint multimodal representations \\(\bm{H}^t\\) and \\(\bm{H}^s\\). The fully-connected layers are utilized to refine the representation \\(\bm{H}^w  \in \mathbb{R}^{3d}\\) with \\(w \in \{t,s\}\\), yielding \\(\tilde{\bm{H}}^w  \in \mathbb{R}^{3d}\\). Moreover, we obtain the intermediate multi-scale representations of all layers, denoted as \\(\bm{I}^w_1 \in \mathbb{R}^{2d}\\), \\(\bm{I}^w_2  \in \mathbb{R}^{d}\\), and \\(\bm{I}^w_3  \in \mathbb{R}^{2d}\\). For the above five representations, we concatenate features of the same scale to obtain multi-scale representations \\(\bm{E}^w_1 \in \mathbb{R}^{3d}\\), \\(\bm{E}^w_2 \in \mathbb{R}^{2d}\\), and \\(\bm{E}^w_3 \in \mathbb{R}^{d}\\), which are utilized in the subsequent computation.

To estimate and compute the mutual information between representations, we define two random variables \\(\bm{X}\\) and \\(\bm{Y}\\). The \\(P(\bm{X})\\) and \\(P(\bm{Y})\\) are the marginal probability density function of \\(\bm{X}\\) and \\(\bm{Y}\\). The joint probability density function of \\(\bm{X}\\) and \\(\bm{Y}\\) is denoted as \\(P(\bm{X},\bm{Y})\\). The mutual information of the random variables \\(\bm{X}\\) and \\(\bm{Y}\\) is represented as: \\[I(\bm{X} ; \bm{Y})=\mathbb{E}_{p(\bm{x}, \bm{y})}\left[\log \frac{p(\bm{x}, \bm{y})}{p(\bm{x}) p(\bm{y})}\right].\\] We only need to obtain the maximum value of the mutual information, without focusing on its exact value. Referring to Deep InfoMax `\cite{hjelm2018learning}`{=latex}, we estimate the mutual information between variables based on the Jensen-Shannon Divergence (JSD). The mutual information maximization issue translates into minimizing the JSD between the joint distribution \\(p(\bm{x}, \bm{y})\\) and the marginal distribution \\(p(\bm{x})p(\bm{y})\\). \\[\begin{aligned}
    JSD(p(\bm{x}, \bm{y}) \| p(\bm{x}) p(\bm{y}))=\frac{1}{2}\left(D_{K L}(p(\bm{x}, \bm{y}) \| m)+D_{K L}(p(\bm{x}) p(\bm{y}) \| m)\right),
\end{aligned}\\] where \\(m=\frac{1}{2}(p(\bm{x}, \bm{y})+p(\bm{x}) p(\bm{y}))\\) and \\(D_{KL}\\) is Kullback-Leibler divergence. Mutual information maximization is achieved by maximizing the dyadic lower bound of JSD, denoted as: \\[MI(\bm{\bm{x}}, \bm{\bm{y}})  = \mathbb{E}_{P(\bm{x}, \bm{y})}[-sp(-\mathcal{T}_\theta(\bm{x},\bm{y})]  + \mathbb{E}_{P(\bm{x}) P(\bm{y})}[-sp(\mathcal{T}_\theta(\bm{x},\bm{y})],\\] where \\(sp(w) = \log (1+e^w)\\) and \\(\mathcal{T}_\theta(\bm{x}, \bm{y})\\) is the statistics network which is a neural network with parameters \\(\theta\\). HMI maximizes the mutual information between hierarchical representations in knowledge distillation, whose optimization loss is expressed as: \\[\begin{aligned}
      \mathcal{L}_{HMI} = -\sum_{i=1}^3{MI(\bm{E}^t_i,\bm{E}^s_i)}.
\end{aligned}\\]

## Hierarchical Adversarial Learning

Considering that the teacher network has more robust and stable representation distributions, we also need to encourage the alignment of representation distributions in the latent space. Traditional methods `\cite{rahimpour2021cross, hu2020knowledge, kumar2019online}`{=latex} simply minimize the KL divergence between both networks, which easily disturbs the underlying learning of the student network in the deep layers, leading to confounded distributions and unrobust joint multimodal representations.

To this end, we propose a Hierarchical Adversarial Learning (HAL) mechanism for incrementally aligning the latent distributions between representations of student and teacher networks. The central principle is that the student network tries to generate representations to mislead the discriminator \\(\mathcal{D}_{e}(\cdot)\\), while \\(\mathcal{D}_{e}(\cdot)\\) discriminates between the representations of the student and teacher networks. In practice, \\(\mathcal{D}_{e}(\cdot)\\) is the fully-connected layers. Specifically, given multi-scale representations of \\(\bm{E}^w_1 \in \mathbb{R}^{3d}\\), \\(\bm{E}^w_2 \in \mathbb{R}^{2d}\\), and \\(\bm{E}^w_3 \in \mathbb{R}^{d}\\) with \\(w \in \{t, s\}\\), we implement adversarial learning on the same-scale representations of the teacher and student networks to hierarchically supervise consistency. The objective function of HAL is formatted as: \\[\begin{aligned}
      \mathcal{L}_{HAL} = \sum_{i=1}^3{\operatorname{log}(1 - \mathcal{D}_{e}(\bm{E}^s_i)) + \operatorname{log}(\mathcal{D}_{e}(\bm{E}^t_i)))}.
\end{aligned}\\]

## Optimization Objectives

The \\(\tilde{\bm{H}}^t\\) and \\(\tilde{\bm{H}}^s\\) of the teacher and student networks are fed into their task-specific classifiers to produce logits \\(\bm{L}^t\\) and \\(\bm{L}^s\\), respectively, and the consistency of both is constrained with KL divergence loss, denoted as \\(\mathcal{L}_{KL} = KL(\bm{L}^t, \bm{L}^s)\\). The \\(\bm{L}^s\\) is used for sentiment recognition and supervised with task loss, represented as \\(\mathcal{L}_{task}\\). For the classification and regression tasks, we use cross-entropy and MSE loss as the task losses, respectively. The overall training objective \\(\mathcal{L}_{total}\\) is expressed as \\(\mathcal{L}_{total} = \mathcal{L}_{task} + \mathcal{L}_{FRF} + \mathcal{L}_{HMI} + \mathcal{L}_{HAL} + \mathcal{L}_{KL}\\).

# Experiments

## Datasets and Evaluation Metrics

We conduct our experiments on three MSA benchmarks, including MOSI `\cite{zadeh2016mosi}`{=latex}, MOSEI `\cite{zadeh2018multimodal}`{=latex}, and IEMOCAP `\cite{busso2008iemocap}`{=latex}. The experiments are performed under the word-aligned setting. MOSI is a realistic dataset for MSA. It comprises 2,199 short monologue video clips taken from 93 YouTube movie review videos. There are 1,284, 229, and 686 video clips in train, valid, and test data, respectively. MOSEI is a dataset consisting of 22,856 movie review video clips, which has 16,326, 1,871, and 4,659 samples in train, valid, and test data. Each sample of MOSI and MOSEI is labelled by human annotators with a sentiment score of -3 (strongly negative) to +3 (strongly positive). On the MOSI and MOSEI datasets, we utilize two evaluation metrics, including the Mean Absolute Error (MAE) and F1 score computed for positive/negative classification results. The IEMOCAP dataset consists of 4,453 samples of video clips. Its predetermined data partition has 2,717, 798, and 938 samples in train, valid, and test data. As recommended by `\cite{wang2019words}`{=latex}, four emotions (*i.e.,* happy, sad, angry, and neutral) are selected for emotion recognition. The F1 score is used as the metric.

## Implementation Details

**Feature Extraction.**  We convert utterance transcripts to 300-dimensional word vectors using pre-trained GloVe \cite{pennington2014glove}.  For the audio channel, 74-dimensional acoustic features are extracted with COVAREP \cite{degottex2014covarep}, covering Mel-frequency cepstral coefficients, glottal parameters and voiced/unvoiced indicators.  The visual stream is represented by 35 facial action units obtained with Facet \cite{baltruvsaitis2016openface}.

**Training Protocol.**  Unless otherwise stated, all experiments are conducted on four NVIDIA Tesla V100 GPUs using PyTorch \cite{paszke2017automatic}.  We optimise with Adam \cite{kingma2014adam} and adopt learning rates of \(\{1\!\times\!10^{-3}, 2\!\times\!10^{-3}, 4\!\times\!10^{-3}\}\) for MOSI, MOSEI and IEMOCAP, respectively.  The corresponding batch sizes are \(\{128,16,32\}\) and training proceeds for \(\{50,20,30\}\) epochs.  All Transformer components use an embedding size of \(40\) and \(\{10,8,10\}\) attention heads.  Missing-modality positions are zero-padded.

The task-specific prediction head is intentionally kept extremely lightweight so that performance gains stem from the proposed hierarchical representation learning rather than a complex classifier.  In practice, a generic predictive layer is attached on top of the distilled representations and trained jointly with the rest of the network; its particular instantiation does not alter results and is therefore not the focus of this work.  For fairness, all baselines are re-implemented under the same optimisation schedule, and every reported number is averaged over five random seeds.
## Comparison with State-of-the-art Methods

We conduct a comparison between HRLF and eight representative, reproducible state-of-the-art (SOTA) methods, including complete-modality methods: Self-MM `\cite{yu2021learning}`{=latex}, CubeMLP `\cite{sun2022cubemlp}`{=latex}, and DMD `\cite{li2023decoupled}`{=latex}, and missing-modality methods: 1) joint learning methods (*i.e.*, MCTN `\cite{pham2019found}`{=latex}, TransM `\cite{wang2020transmodality}`{=latex}, and CorrKD `\cite{li2024correlation}`{=latex}), and 2) generative methods (*i.e.*, SMIL `\cite{ma2021smil}`{=latex} and GCNet `\cite{lian2023gcnet}`{=latex}). The extensive experiments are designed to comprehensively assess the robustness and effectiveness of HRLF in scenarios involving both intra-modality and inter-modality missingness.

<figure id="comp_intra_2">
<img src="./figures/comp_intra_2.png"" style="width:100.0%" />
<figcaption> Comparison results of intra-modality missingness on IEMOCAP. We report on the F1 score metric for the happy, sad, angry, and neutral categories. </figcaption>
</figure>

**Robustness to Intra-modality Missingness.** We simulate intra-modality missingness by randomly discarding frame-level features in sequences with ratio \\(p \in \{0.1, 0.2, \cdots, 1.0\}\\). To visualize the robustness of all models, <a href="#comp_intra_2" data-reference-type="ref+Label" data-reference="comp_intra_2">3</a> and <a href="#comp_intra_1" data-reference-type="ref" data-reference="comp_intra_1">4</a> show the performance curves of the models for different ratios \\(p\\). We have the following important observations. () As the ratio \\(p\\) increases, the performance of all models declines. This phenomenon demonstrates that intra-modality missingness leads to significant sentiment semantic loss and fragile multimodal representations.

<figure id="comp_intra_1">
<img src="./figures/comp_intra_1.png"" style="width:100.0%" />
<figcaption> Comparison results of intra-modality missingness on (a) MOSI and (b) MOSEI. </figcaption>
</figure>

() Compared to complete-modality methods, our HRLF demonstrates notable performance advantages in missing-modality testing conditions and competitive performance in complete-modality testing conditions. This is because complete-modality methods rely on the assumption of data completeness, while training paradigms for missing modalities excel in capturing and reconstructing valuable sentiment semantics from incomplete multimodal data. () In contrast to the missing-modality methods, our HRLF demonstrates the highest level of robustness. Through the purification of sentiment semantics and the dual alignment of representations, the student network masters the core competencies of precisely reconstructing missing semantics and generating robust multimodal representations.

<div id="comp_inter_1" markdown="1">

<table>
<caption>Comparison of performance under six possible testing conditions of inter-modality missingness and the complete-modality testing condition on the MOSI and MOSEI datasets. T-test is conducted on “Avg.” column. <span class="math inline">*</span> indicates that <span class="math inline"><em>p</em> &lt; 0.05</span> (compared with the SOTA CorrKD). </caption>
<thead>
<tr>
<th style="text-align: center;">Datasets</th>
<th style="text-align: center;">Models</th>
<th colspan="8" style="text-align: center;">Testing Conditions</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><span>3-10</span></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">{<span class="math inline"><em>l</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>a</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>v</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>l</em>, <em>a</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>l</em>, <em>v</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>a</em>, <em>v</em></span>}</td>
<td style="text-align: center;">Avg.</td>
<td style="text-align: center;">{<span class="math inline"><em>l</em>, <em>a</em>, <em>v</em></span>}</td>
</tr>
<tr>
<td rowspan="9" style="text-align: center;">MOSI</td>
<td style="text-align: center;">Self-MM <span class="citation" data-cites="yu2021learning"></span></td>
<td style="text-align: center;">67.80</td>
<td style="text-align: center;">40.95</td>
<td style="text-align: center;">38.52</td>
<td style="text-align: center;">69.81</td>
<td style="text-align: center;">74.97</td>
<td style="text-align: center;">47.12</td>
<td style="text-align: center;">56.53</td>
<td style="text-align: center;"><strong>84.64</strong></td>
</tr>
<tr>
<td style="text-align: center;">CubeMLP <span class="citation" data-cites="sun2022cubemlp"></span></td>
<td style="text-align: center;">64.15</td>
<td style="text-align: center;">38.91</td>
<td style="text-align: center;">43.24</td>
<td style="text-align: center;">63.76</td>
<td style="text-align: center;">65.12</td>
<td style="text-align: center;">47.92</td>
<td style="text-align: center;">53.85</td>
<td style="text-align: center;">84.57</td>
</tr>
<tr>
<td style="text-align: center;">DMD <span class="citation" data-cites="li2023decoupled"></span></td>
<td style="text-align: center;">68.97</td>
<td style="text-align: center;">43.33</td>
<td style="text-align: center;">42.26</td>
<td style="text-align: center;">70.51</td>
<td style="text-align: center;">68.45</td>
<td style="text-align: center;">50.47</td>
<td style="text-align: center;">57.33</td>
<td style="text-align: center;">84.50</td>
</tr>
<tr>
<td style="text-align: center;">MCTN <span class="citation" data-cites="pham2019found"></span></td>
<td style="text-align: center;">75.21</td>
<td style="text-align: center;">59.25</td>
<td style="text-align: center;">58.57</td>
<td style="text-align: center;">77.81</td>
<td style="text-align: center;">74.82</td>
<td style="text-align: center;">64.21</td>
<td style="text-align: center;">68.31</td>
<td style="text-align: center;">80.12</td>
</tr>
<tr>
<td style="text-align: center;">TransM <span class="citation" data-cites="wang2020transmodality"></span></td>
<td style="text-align: center;">77.64</td>
<td style="text-align: center;">63.57</td>
<td style="text-align: center;">56.48</td>
<td style="text-align: center;">82.07</td>
<td style="text-align: center;">80.90</td>
<td style="text-align: center;">67.24</td>
<td style="text-align: center;">71.32</td>
<td style="text-align: center;">82.57</td>
</tr>
<tr>
<td style="text-align: center;">SMIL <span class="citation" data-cites="ma2021smil"></span></td>
<td style="text-align: center;">78.26</td>
<td style="text-align: center;">67.69</td>
<td style="text-align: center;">59.67</td>
<td style="text-align: center;">79.82</td>
<td style="text-align: center;">79.15</td>
<td style="text-align: center;">71.24</td>
<td style="text-align: center;">72.64</td>
<td style="text-align: center;">82.85</td>
</tr>
<tr>
<td style="text-align: center;">GCNet <span class="citation" data-cites="lian2023gcnet"></span></td>
<td style="text-align: center;">80.91</td>
<td style="text-align: center;">65.07</td>
<td style="text-align: center;">58.70</td>
<td style="text-align: center;"><strong>84.73</strong></td>
<td style="text-align: center;"><strong>83.58</strong></td>
<td style="text-align: center;">70.02</td>
<td style="text-align: center;">73.84</td>
<td style="text-align: center;">83.20</td>
</tr>
<tr>
<td style="text-align: center;">CorrKD <span class="citation" data-cites="li2024correlation"></span></td>
<td style="text-align: center;">81.20</td>
<td style="text-align: center;">66.52</td>
<td style="text-align: center;">60.72</td>
<td style="text-align: center;">83.56</td>
<td style="text-align: center;">82.41</td>
<td style="text-align: center;">73.74</td>
<td style="text-align: center;">74.69</td>
<td style="text-align: center;">83.94</td>
</tr>
<tr>
<td style="text-align: center;"><strong>HRLF (Ours)</strong></td>
<td style="text-align: center;"><strong>83.36</strong></td>
<td style="text-align: center;"><strong>69.47</strong></td>
<td style="text-align: center;"><strong>64.59</strong></td>
<td style="text-align: center;">83.82</td>
<td style="text-align: center;">83.56</td>
<td style="text-align: center;"><strong>75.62</strong></td>
<td style="text-align: center;"><strong>  76.74<span class="math inline"><sup>*</sup></span></strong></td>
<td style="text-align: center;">84.15</td>
</tr>
<tr>
<td rowspan="9" style="text-align: center;">MOSEI</td>
<td style="text-align: center;">Self-MM <span class="citation" data-cites="yu2021learning"></span></td>
<td style="text-align: center;">71.53</td>
<td style="text-align: center;">43.57</td>
<td style="text-align: center;">37.61</td>
<td style="text-align: center;">75.91</td>
<td style="text-align: center;">74.62</td>
<td style="text-align: center;">49.52</td>
<td style="text-align: center;">58.79</td>
<td style="text-align: center;">83.69</td>
</tr>
<tr>
<td style="text-align: center;">CubeMLP <span class="citation" data-cites="sun2022cubemlp"></span></td>
<td style="text-align: center;">67.52</td>
<td style="text-align: center;">39.54</td>
<td style="text-align: center;">32.58</td>
<td style="text-align: center;">71.69</td>
<td style="text-align: center;">70.06</td>
<td style="text-align: center;">48.54</td>
<td style="text-align: center;">54.99</td>
<td style="text-align: center;">83.17</td>
</tr>
<tr>
<td style="text-align: center;">DMD <span class="citation" data-cites="li2023decoupled"></span></td>
<td style="text-align: center;">70.26</td>
<td style="text-align: center;">46.18</td>
<td style="text-align: center;">39.84</td>
<td style="text-align: center;">74.78</td>
<td style="text-align: center;">72.45</td>
<td style="text-align: center;">52.70</td>
<td style="text-align: center;">59.37</td>
<td style="text-align: center;"><strong>84.78</strong></td>
</tr>
<tr>
<td style="text-align: center;">MCTN <span class="citation" data-cites="pham2019found"></span></td>
<td style="text-align: center;">75.50</td>
<td style="text-align: center;">62.72</td>
<td style="text-align: center;">59.46</td>
<td style="text-align: center;">76.64</td>
<td style="text-align: center;">77.13</td>
<td style="text-align: center;">64.84</td>
<td style="text-align: center;">69.38</td>
<td style="text-align: center;">81.75</td>
</tr>
<tr>
<td style="text-align: center;">TransM <span class="citation" data-cites="wang2020transmodality"></span></td>
<td style="text-align: center;">77.98</td>
<td style="text-align: center;">63.68</td>
<td style="text-align: center;">58.67</td>
<td style="text-align: center;">80.46</td>
<td style="text-align: center;">78.61</td>
<td style="text-align: center;">62.24</td>
<td style="text-align: center;">70.27</td>
<td style="text-align: center;">81.48</td>
</tr>
<tr>
<td style="text-align: center;">SMIL <span class="citation" data-cites="ma2021smil"></span></td>
<td style="text-align: center;">76.57</td>
<td style="text-align: center;">65.96</td>
<td style="text-align: center;">60.57</td>
<td style="text-align: center;">77.68</td>
<td style="text-align: center;">76.24</td>
<td style="text-align: center;">66.87</td>
<td style="text-align: center;">70.65</td>
<td style="text-align: center;">80.74</td>
</tr>
<tr>
<td style="text-align: center;">GCNet <span class="citation" data-cites="lian2023gcnet"></span></td>
<td style="text-align: center;">80.52</td>
<td style="text-align: center;">66.54</td>
<td style="text-align: center;">61.83</td>
<td style="text-align: center;">81.96</td>
<td style="text-align: center;">81.15</td>
<td style="text-align: center;">69.21</td>
<td style="text-align: center;">73.54</td>
<td style="text-align: center;">82.35</td>
</tr>
<tr>
<td style="text-align: center;">CorrKD <span class="citation" data-cites="li2024correlation"></span></td>
<td style="text-align: center;">80.76</td>
<td style="text-align: center;">66.09</td>
<td style="text-align: center;">62.30</td>
<td style="text-align: center;">81.74</td>
<td style="text-align: center;"><strong>81.28</strong></td>
<td style="text-align: center;">71.92</td>
<td style="text-align: center;">74.02</td>
<td style="text-align: center;">82.16</td>
</tr>
<tr>
<td style="text-align: center;"><strong>HRLF (Ours)</strong></td>
<td style="text-align: center;"><strong>82.05</strong></td>
<td style="text-align: center;"><strong>69.32</strong></td>
<td style="text-align: center;"><strong>64.90</strong></td>
<td style="text-align: center;"><strong>82.62</strong></td>
<td style="text-align: center;">81.09</td>
<td style="text-align: center;"><strong>73.80</strong></td>
<td style="text-align: center;"><strong>  75.63<span class="math inline"><sup>*</sup></span></strong></td>
<td style="text-align: center;">82.93</td>
</tr>
</tbody>
</table>

</div>

<div id="comp_inter_2" markdown="1">

<table>
<caption>Comparison of performance under six possible testing conditions of inter-modality missingness and the complete-modality testing condition on the IEMOCAP dataset. T-test is conducted on “Avg.” column. <span class="math inline">*</span> indicates that <span class="math inline"><em>p</em> &lt; 0.05</span> (compared with the SOTA CorrKD).</caption>
<thead>
<tr>
<th style="text-align: center;">Models</th>
<th style="text-align: center;">Metrics</th>
<th colspan="8" style="text-align: center;">Testing Conditions</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><span>3-10</span></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">{<span class="math inline"><em>l</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>a</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>v</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>l</em>, <em>a</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>l</em>, <em>v</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>a</em>, <em>v</em></span>}</td>
<td style="text-align: center;">Avg.</td>
<td style="text-align: center;">{<span class="math inline"><em>l</em>, <em>a</em>, <em>v</em></span>}</td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">Self-MM <span class="citation" data-cites="yu2021learning"></span></td>
<td style="text-align: center;">Happy</td>
<td style="text-align: center;">66.9</td>
<td style="text-align: center;">52.2</td>
<td style="text-align: center;">50.1</td>
<td style="text-align: center;">69.9</td>
<td style="text-align: center;">68.3</td>
<td style="text-align: center;">56.3</td>
<td style="text-align: center;">60.6</td>
<td style="text-align: center;">90.8</td>
</tr>
<tr>
<td style="text-align: center;">Sad</td>
<td style="text-align: center;">68.7</td>
<td style="text-align: center;">51.9</td>
<td style="text-align: center;">54.8</td>
<td style="text-align: center;">71.3</td>
<td style="text-align: center;">69.5</td>
<td style="text-align: center;">57.5</td>
<td style="text-align: center;">62.3</td>
<td style="text-align: center;">86.7</td>
</tr>
<tr>
<td style="text-align: center;">Angry</td>
<td style="text-align: center;">65.4</td>
<td style="text-align: center;">53.0</td>
<td style="text-align: center;">51.9</td>
<td style="text-align: center;">69.5</td>
<td style="text-align: center;">67.7</td>
<td style="text-align: center;">56.6</td>
<td style="text-align: center;">60.7</td>
<td style="text-align: center;">88.4</td>
</tr>
<tr>
<td style="text-align: center;">Neutral</td>
<td style="text-align: center;">55.8</td>
<td style="text-align: center;">48.2</td>
<td style="text-align: center;">50.4</td>
<td style="text-align: center;">58.1</td>
<td style="text-align: center;">56.5</td>
<td style="text-align: center;">52.8</td>
<td style="text-align: center;">53.6</td>
<td style="text-align: center;"><strong>72.7</strong></td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">CubeMLP <span class="citation" data-cites="sun2022cubemlp"></span></td>
<td style="text-align: center;">Happy</td>
<td style="text-align: center;">68.9</td>
<td style="text-align: center;">54.3</td>
<td style="text-align: center;">51.4</td>
<td style="text-align: center;">72.1</td>
<td style="text-align: center;">69.8</td>
<td style="text-align: center;">60.6</td>
<td style="text-align: center;">62.9</td>
<td style="text-align: center;">89.0</td>
</tr>
<tr>
<td style="text-align: center;">Sad</td>
<td style="text-align: center;">65.3</td>
<td style="text-align: center;">54.8</td>
<td style="text-align: center;">53.2</td>
<td style="text-align: center;">70.3</td>
<td style="text-align: center;">68.7</td>
<td style="text-align: center;">58.1</td>
<td style="text-align: center;">61.7</td>
<td style="text-align: center;"><strong>88.5</strong></td>
</tr>
<tr>
<td style="text-align: center;">Angry</td>
<td style="text-align: center;">65.8</td>
<td style="text-align: center;">53.1</td>
<td style="text-align: center;">50.4</td>
<td style="text-align: center;">69.5</td>
<td style="text-align: center;">69.0</td>
<td style="text-align: center;">54.8</td>
<td style="text-align: center;">60.4</td>
<td style="text-align: center;">87.2</td>
</tr>
<tr>
<td style="text-align: center;">Neutral</td>
<td style="text-align: center;">53.5</td>
<td style="text-align: center;">50.8</td>
<td style="text-align: center;">48.7</td>
<td style="text-align: center;">57.3</td>
<td style="text-align: center;">54.5</td>
<td style="text-align: center;">51.8</td>
<td style="text-align: center;">52.8</td>
<td style="text-align: center;">71.8</td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">DMD <span class="citation" data-cites="li2023decoupled"></span></td>
<td style="text-align: center;">Happy</td>
<td style="text-align: center;">69.5</td>
<td style="text-align: center;">55.4</td>
<td style="text-align: center;">51.9</td>
<td style="text-align: center;">73.2</td>
<td style="text-align: center;">70.3</td>
<td style="text-align: center;">61.3</td>
<td style="text-align: center;">63.6</td>
<td style="text-align: center;"><strong>91.1</strong></td>
</tr>
<tr>
<td style="text-align: center;">Sad</td>
<td style="text-align: center;">65.0</td>
<td style="text-align: center;">54.9</td>
<td style="text-align: center;">53.5</td>
<td style="text-align: center;">70.7</td>
<td style="text-align: center;">69.2</td>
<td style="text-align: center;">61.1</td>
<td style="text-align: center;">62.4</td>
<td style="text-align: center;">88.4</td>
</tr>
<tr>
<td style="text-align: center;">Angry</td>
<td style="text-align: center;">64.8</td>
<td style="text-align: center;">53.7</td>
<td style="text-align: center;">51.2</td>
<td style="text-align: center;">70.8</td>
<td style="text-align: center;">69.9</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">61.3</td>
<td style="text-align: center;"><strong>88.6</strong></td>
</tr>
<tr>
<td style="text-align: center;">Neutral</td>
<td style="text-align: center;">54.0</td>
<td style="text-align: center;">51.2</td>
<td style="text-align: center;">48.0</td>
<td style="text-align: center;">56.9</td>
<td style="text-align: center;">55.6</td>
<td style="text-align: center;">53.4</td>
<td style="text-align: center;">53.2</td>
<td style="text-align: center;">72.2</td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">MCTN <span class="citation" data-cites="pham2019found"></span></td>
<td style="text-align: center;">Happy</td>
<td style="text-align: center;">76.9</td>
<td style="text-align: center;">63.4</td>
<td style="text-align: center;">60.8</td>
<td style="text-align: center;">79.6</td>
<td style="text-align: center;">77.6</td>
<td style="text-align: center;">66.9</td>
<td style="text-align: center;">70.9</td>
<td style="text-align: center;">83.1</td>
</tr>
<tr>
<td style="text-align: center;">Sad</td>
<td style="text-align: center;">76.7</td>
<td style="text-align: center;">64.4</td>
<td style="text-align: center;">60.4</td>
<td style="text-align: center;">78.9</td>
<td style="text-align: center;">77.1</td>
<td style="text-align: center;">68.6</td>
<td style="text-align: center;">71.0</td>
<td style="text-align: center;">82.8</td>
</tr>
<tr>
<td style="text-align: center;">Angry</td>
<td style="text-align: center;">77.1</td>
<td style="text-align: center;">61.0</td>
<td style="text-align: center;">56.7</td>
<td style="text-align: center;">81.6</td>
<td style="text-align: center;">80.4</td>
<td style="text-align: center;">58.9</td>
<td style="text-align: center;">69.3</td>
<td style="text-align: center;">84.6</td>
</tr>
<tr>
<td style="text-align: center;">Neutral</td>
<td style="text-align: center;">60.1</td>
<td style="text-align: center;">51.9</td>
<td style="text-align: center;">50.4</td>
<td style="text-align: center;">64.7</td>
<td style="text-align: center;">62.4</td>
<td style="text-align: center;">54.9</td>
<td style="text-align: center;">57.4</td>
<td style="text-align: center;">67.7</td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">TransM <span class="citation" data-cites="wang2020transmodality"></span></td>
<td style="text-align: center;">Happy</td>
<td style="text-align: center;">78.4</td>
<td style="text-align: center;">64.5</td>
<td style="text-align: center;">61.1</td>
<td style="text-align: center;">81.6</td>
<td style="text-align: center;">80.2</td>
<td style="text-align: center;">66.5</td>
<td style="text-align: center;">72.1</td>
<td style="text-align: center;">85.5</td>
</tr>
<tr>
<td style="text-align: center;">Sad</td>
<td style="text-align: center;">79.5</td>
<td style="text-align: center;">63.2</td>
<td style="text-align: center;">58.9</td>
<td style="text-align: center;">82.4</td>
<td style="text-align: center;">80.5</td>
<td style="text-align: center;">64.4</td>
<td style="text-align: center;">71.5</td>
<td style="text-align: center;">84.0</td>
</tr>
<tr>
<td style="text-align: center;">Angry</td>
<td style="text-align: center;">81.0</td>
<td style="text-align: center;">65.0</td>
<td style="text-align: center;">60.7</td>
<td style="text-align: center;">83.9</td>
<td style="text-align: center;">81.7</td>
<td style="text-align: center;">66.9</td>
<td style="text-align: center;">73.2</td>
<td style="text-align: center;">86.1</td>
</tr>
<tr>
<td style="text-align: center;">Neutral</td>
<td style="text-align: center;">60.2</td>
<td style="text-align: center;">49.9</td>
<td style="text-align: center;">50.7</td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">62.4</td>
<td style="text-align: center;">52.4</td>
<td style="text-align: center;">56.8</td>
<td style="text-align: center;">67.1</td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">SMIL <span class="citation" data-cites="ma2021smil"></span></td>
<td style="text-align: center;">Happy</td>
<td style="text-align: center;">80.5</td>
<td style="text-align: center;">66.5</td>
<td style="text-align: center;">63.8</td>
<td style="text-align: center;">83.1</td>
<td style="text-align: center;">81.8</td>
<td style="text-align: center;">68.2</td>
<td style="text-align: center;">74.0</td>
<td style="text-align: center;">86.8</td>
</tr>
<tr>
<td style="text-align: center;">Sad</td>
<td style="text-align: center;">78.9</td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">62.2</td>
<td style="text-align: center;">82.4</td>
<td style="text-align: center;">79.6</td>
<td style="text-align: center;">68.2</td>
<td style="text-align: center;">72.8</td>
<td style="text-align: center;">85.2</td>
</tr>
<tr>
<td style="text-align: center;">Angry</td>
<td style="text-align: center;">79.6</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">61.8</td>
<td style="text-align: center;">83.1</td>
<td style="text-align: center;">82.0</td>
<td style="text-align: center;">67.8</td>
<td style="text-align: center;">73.6</td>
<td style="text-align: center;">84.9</td>
</tr>
<tr>
<td style="text-align: center;">Neutral</td>
<td style="text-align: center;">60.2</td>
<td style="text-align: center;">50.4</td>
<td style="text-align: center;">48.8</td>
<td style="text-align: center;">65.4</td>
<td style="text-align: center;">62.2</td>
<td style="text-align: center;">52.6</td>
<td style="text-align: center;">56.6</td>
<td style="text-align: center;">68.9</td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">GCNet <span class="citation" data-cites="lian2023gcnet"></span></td>
<td style="text-align: center;">Happy</td>
<td style="text-align: center;">81.9</td>
<td style="text-align: center;">67.3</td>
<td style="text-align: center;">66.6</td>
<td style="text-align: center;">83.7</td>
<td style="text-align: center;">82.5</td>
<td style="text-align: center;">69.8</td>
<td style="text-align: center;">75.3</td>
<td style="text-align: center;">87.7</td>
</tr>
<tr>
<td style="text-align: center;">Sad</td>
<td style="text-align: center;">80.5</td>
<td style="text-align: center;">69.4</td>
<td style="text-align: center;">66.1</td>
<td style="text-align: center;">83.8</td>
<td style="text-align: center;">81.9</td>
<td style="text-align: center;">70.4</td>
<td style="text-align: center;">75.4</td>
<td style="text-align: center;">86.9</td>
</tr>
<tr>
<td style="text-align: center;">Angry</td>
<td style="text-align: center;">80.1</td>
<td style="text-align: center;">66.2</td>
<td style="text-align: center;">64.2</td>
<td style="text-align: center;">82.5</td>
<td style="text-align: center;">81.6</td>
<td style="text-align: center;">68.1</td>
<td style="text-align: center;">73.8</td>
<td style="text-align: center;">85.2</td>
</tr>
<tr>
<td style="text-align: center;">Neutral</td>
<td style="text-align: center;">61.8</td>
<td style="text-align: center;">51.1</td>
<td style="text-align: center;">49.6</td>
<td style="text-align: center;">66.2</td>
<td style="text-align: center;">63.5</td>
<td style="text-align: center;">53.3</td>
<td style="text-align: center;">57.6</td>
<td style="text-align: center;">71.1</td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">CorrKD <span class="citation" data-cites="li2024correlation"></span></td>
<td style="text-align: center;">Happy</td>
<td style="text-align: center;">82.6</td>
<td style="text-align: center;">69.6</td>
<td style="text-align: center;">68.0</td>
<td style="text-align: center;">84.1</td>
<td style="text-align: center;">82.0</td>
<td style="text-align: center;">70.0</td>
<td style="text-align: center;">76.1</td>
<td style="text-align: center;">87.5</td>
</tr>
<tr>
<td style="text-align: center;">Sad</td>
<td style="text-align: center;">82.7</td>
<td style="text-align: center;"><strong>71.3</strong></td>
<td style="text-align: center;">67.6</td>
<td style="text-align: center;">83.4</td>
<td style="text-align: center;">82.2</td>
<td style="text-align: center;">72.5</td>
<td style="text-align: center;">76.6</td>
<td style="text-align: center;">85.9</td>
</tr>
<tr>
<td style="text-align: center;">Angry</td>
<td style="text-align: center;">82.2</td>
<td style="text-align: center;">67.0</td>
<td style="text-align: center;">65.8</td>
<td style="text-align: center;">83.9</td>
<td style="text-align: center;">82.8</td>
<td style="text-align: center;">67.3</td>
<td style="text-align: center;">74.8</td>
<td style="text-align: center;">86.1</td>
</tr>
<tr>
<td style="text-align: center;">Neutral</td>
<td style="text-align: center;">63.1</td>
<td style="text-align: center;">54.2</td>
<td style="text-align: center;">52.3</td>
<td style="text-align: center;">68.5</td>
<td style="text-align: center;">64.3</td>
<td style="text-align: center;"><strong>57.2</strong></td>
<td style="text-align: center;">59.9</td>
<td style="text-align: center;">71.5</td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;"><strong>HRLF (Ours)</strong></td>
<td style="text-align: center;">Happy</td>
<td style="text-align: center;"><strong>84.9</strong></td>
<td style="text-align: center;"><strong>71.8</strong></td>
<td style="text-align: center;"><strong>69.7</strong></td>
<td style="text-align: center;"><strong>86.4</strong></td>
<td style="text-align: center;"><strong>85.6</strong></td>
<td style="text-align: center;"><strong>72.3</strong></td>
<td style="text-align: center;"><strong>  78.5<span class="math inline"><sup>*</sup></span></strong></td>
<td style="text-align: center;">88.1</td>
</tr>
<tr>
<td style="text-align: center;">Sad</td>
<td style="text-align: center;"><strong>83.7</strong></td>
<td style="text-align: center;">71.1</td>
<td style="text-align: center;"><strong>69.0</strong></td>
<td style="text-align: center;"><strong>85.3</strong></td>
<td style="text-align: center;"><strong>83.9</strong></td>
<td style="text-align: center;"><strong>73.6</strong></td>
<td style="text-align: center;"><strong>  77.8<span class="math inline"><sup>*</sup></span></strong></td>
<td style="text-align: center;">86.4</td>
</tr>
<tr>
<td style="text-align: center;">Angry</td>
<td style="text-align: center;"><strong>83.4</strong></td>
<td style="text-align: center;"><strong>69.1</strong></td>
<td style="text-align: center;"><strong>67.2</strong></td>
<td style="text-align: center;"><strong>84.5</strong></td>
<td style="text-align: center;"><strong>83.5</strong></td>
<td style="text-align: center;"><strong>70.9</strong></td>
<td style="text-align: center;"><strong>  76.4<span class="math inline"><sup>*</sup></span></strong></td>
<td style="text-align: center;">86.7</td>
</tr>
<tr>
<td style="text-align: center;">Neutral</td>
<td style="text-align: center;"><strong>66.8</strong></td>
<td style="text-align: center;"><strong>56.1</strong></td>
<td style="text-align: center;"><strong>54.5</strong></td>
<td style="text-align: center;"><strong>68.9</strong></td>
<td style="text-align: center;"><strong>67.0</strong></td>
<td style="text-align: center;">56.9</td>
<td style="text-align: center;"><strong>  61.7<span class="math inline"><sup>*</sup></span></strong></td>
<td style="text-align: center;">71.3</td>
</tr>
</tbody>
</table>

</div>

**Robustness to Inter-modality Missingness.** To simulate the case of inter-modality missingness, we remove certain entire modalities from the samples. Tables <a href="#comp_inter_1" data-reference-type="ref" data-reference="comp_inter_1">1</a> and <a href="#comp_inter_2" data-reference-type="ref" data-reference="comp_inter_2">2</a> contrast the models’ resilience to inter-modality missingness. The notation “\\(\{l\}\\)” signifies that only the language modality is available, while the audio and visual modalities are missing. “\\(\{l, a, v\}\\)” denotes the complete-modality testing condition where all modalities are available. “Avg.” indicates the average performance across six missing-modality testing conditions.

<span id="ablation_inter_mosi" label="ablation_inter_mosi"></span>

<div id="ablation_inter_mosi" markdown="1">

<table>
<caption>Ablation results of inter-modality missingness case on the MOSI dataset. </caption>
<thead>
<tr>
<th style="text-align: center;">Models</th>
<th colspan="8" style="text-align: center;">Testing Conditions</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><span>2-9</span></td>
<td style="text-align: center;">{<span class="math inline"><em>l</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>a</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>v</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>l</em>, <em>a</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>l</em>, <em>v</em></span>}</td>
<td style="text-align: center;">{<span class="math inline"><em>a</em>, <em>v</em></span>}</td>
<td style="text-align: center;">Avg.</td>
<td style="text-align: center;">{<span class="math inline"><em>l</em>, <em>a</em>, <em>v</em></span>}</td>
</tr>
<tr>
<td style="text-align: center;"><strong>HRLF (Full)</strong></td>
<td style="text-align: center;"><strong>83.36</strong></td>
<td style="text-align: center;"><strong>69.47</strong></td>
<td style="text-align: center;"><strong>64.59</strong></td>
<td style="text-align: center;"><strong>83.82</strong></td>
<td style="text-align: center;"><strong>83.56</strong></td>
<td style="text-align: center;"><strong>75.62</strong></td>
<td style="text-align: center;"><strong>76.74</strong></td>
<td style="text-align: center;"><strong>84.15</strong></td>
</tr>
<tr>
<td style="text-align: center;">w/o FRF</td>
<td style="text-align: center;">80.85</td>
<td style="text-align: center;">67.06</td>
<td style="text-align: center;">61.78</td>
<td style="text-align: center;">81.94</td>
<td style="text-align: center;">81.38</td>
<td style="text-align: center;">73.58</td>
<td style="text-align: center;">74.43</td>
<td style="text-align: center;">82.76</td>
</tr>
<tr>
<td style="text-align: center;">w/o HMI</td>
<td style="text-align: center;">81.54</td>
<td style="text-align: center;">67.72</td>
<td style="text-align: center;">62.70</td>
<td style="text-align: center;">82.45</td>
<td style="text-align: center;">81.90</td>
<td style="text-align: center;">74.22</td>
<td style="text-align: center;">75.09</td>
<td style="text-align: center;">83.25</td>
</tr>
<tr>
<td style="text-align: center;">w/o HAL</td>
<td style="text-align: center;">82.03</td>
<td style="text-align: center;">68.09</td>
<td style="text-align: center;">63.11</td>
<td style="text-align: center;">83.12</td>
<td style="text-align: center;">82.67</td>
<td style="text-align: center;">74.59</td>
<td style="text-align: center;">75.60</td>
<td style="text-align: center;">83.67</td>
</tr>
</tbody>
</table>

</div>

We have the following key findings: **()** The inter-modality missingness leads to a decline in performance for all models, indicating that integrating complementary information from diverse modalities enhances the sentiment semantics within joint representations. **()** Across all six testing conditions involving inter-modality missingness, our HRLF consistently demonstrates superior performance among the majority of metrics, affirming its robustness. For example, on the MOSI dataset, HRLF’s average F1 score is improved by \\(2.05\%\\) compared to CorrKD, and in particular by \\(3.87\%\\) in the testing condition where only visual modality is available (*i.e.*, \\(\{v\}\\)). The advantage comes from its learning of fine-grained representation factorization and the hierarchical semantic alignment and distributional alignment. **()** In unimodal testing scenarios, HRLF’s performance using only the language modality significantly exceeds other configurations, showing performance similar to that of the complete-modality setup. In bimodal testing scenarios, configurations involving the language modality exhibit superior performance, even outperforming the complete-modality setup in specific metrics. This phenomenon underscores the richness of sentiment semantics within the language modality and its dominance in sentiment inference and missing semantic reconstruction processes.

## Ablation Studies

<figure id="abla_intra_mosi">
<img src="./figures/abla_intra_mosi.png"" style="width:100.0%" />
<figcaption> Ablation results of intra-modality missingness case on the MOSI dataset. </figcaption>
</figure>

To affirm the effectiveness and indispensability of the module and mechanisms and strategies proposed in HRLF, we perform ablation experiments under two missing-modality scenarios on the MOSI dataset, as shown in  <a href="#ablation_inter_mosi" data-reference-type="ref+Label" data-reference="ablation_inter_mosi">3</a> and  <a href="#abla_intra_mosi" data-reference-type="ref+Label" data-reference="abla_intra_mosi">5</a>. We have the following important observations: **()** First, when the FRF is removed, sentiment-relevant and modality-specific information in the modalities are confused, hindering sentiment recognition and leading to significant performance degradation. This phenomenon demonstrates the effectiveness of the proposed representation factorization paradigm for adequate capture of valuable sentiment semantics. **()** When our HMI is eliminated, the worse performance demonstrates that aligning the high-level semantics in the representation by maximizing mutual information can generate favorable joint representations for the student network. **()** Finally, we remove HAL, and the declined results illustrate that multi-scale adversarial learning can effectively align the representation distributions of student and teacher networks, thus effectively constraining the consistency across representations. This paradigm facilitates the recovery of missing semantics.

<figure id="vis1">
<img src="./figures/vis1.png"" />
<figcaption> Visualization of representations from different methods with four emotion categories on the IEMOCAP testing set. The default testing conditions contain intra-modality missingness (<em>i.e.</em>, missing rate <span class="math inline"><em>p</em> = 0.5</span> ) and inter-modality missingness (<em>i.e.</em>, only the language modality is available). </figcaption>
</figure>

## Qualitative Analysis

To intuitively show the robustness of the proposed framework against modality missingness, we randomly select 100 samples in each emotion category on the IEMOCAP testing set to perform the visualization evaluation. The comparison models include CubeMLP `\cite{sun2022cubemlp}`{=latex} (complete-modality method), TransM `\cite{wang2020transmodality}`{=latex} (joint learning-based missing-modality method), and GCNet `\cite{lian2023gcnet}`{=latex} (generation-based missing-modality method). **(i)** As shown in <a href="#vis1" data-reference-type="ref+Label" data-reference="vis1">6</a>, CubeMLP fails to cope with the missing modality challenge because representations with different emotion categories are heavily confounded, leading to the worst results. **(ii)** Although TransM and GCNet mitigate the indistinguishable emotion semantics to some extent, their performance is sub-optimal since the distribution boundaries of the different emotion representations are generally ambiguous and coupled. **(iii)** In comparison, our HRLF enables representations belonging to the same emotion category to form compact clusters, while representations of different categories are well separated. The above phenomenon benefits from the effective extraction of sentiment semantics and the precise filtering of task redundant information by the proposed hierarchical representation learning framework, which results in better joint multimodal representations. This further confirms the robustness and superiority of our framework.

# Conclusion and Discussion

In this paper, we present a Hierarchical Representation Learning Framework (HRLF) to address diverse missing modality dilemmas in the MSA task. Specifically, we mine sentiment-relevant representations through a fine-grained representation factorization module. Additionally, the hierarchical mutual information maximization mechanism and the hierarchical adversarial learning mechanism are proposed for semantic and distributional alignment of representations of student and teacher networks to accurately reconstruct missing semantics and produce robust joint multimodal representations. Comprehensive experiments validate the superiority of our framework.

**Discussion of Limitation and Future Work.** The current method defines the modality missing cases as both inter-modality missingness and intra-modality missingness. Nevertheless, in real-world applications, modality missing cases may be very intricate and difficult to simulate. Consequently, the proposed method may suffer some minor performance loss when applied to real-world scenarios. In the future, we will explore more intricate modality missing cases and design suitable algorithms to compensate for this deficiency.

**Discussion of Broad Impacts.** The positive impact of our approach lies in the ability to significantly improve the robustness and stability of multimodal sentiment analysis systems against heterogeneous modality missingness in real-world applications. Nevertheless, this technology may have a negative impact when it falls into the wrong hands, *e.g.*, the proposed model is used for malicious purposes by injecting biased priors to recognize the emotions of specific groups.

# Acknowledgements

This work was supported in part by National Key R&D Program of China 2021ZD0113502 and in part by Shanghai Municipal Science and Technology Major Project 2021SHZDZX0103.

# References [references]

<div class="thebibliography" markdown="1">

Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D Lawrence, and Zhenwen Dai Variational information distillation for knowledge transfer In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 9163–9171, 2019. **Abstract:** Transferring knowledge from a teacher neural network pretrained on the same or a similar task to a student neural network can significantly improve the performance of the student neural network. Existing knowledge transfer approaches match the activations or the corresponding hand-crafted features of the teacher and the student networks. We propose an information-theoretic framework for knowledge transfer which formulates knowledge transfer as maximizing the mutual information between the teacher and the student networks. We compare our method with existing knowledge transfer methods on both knowledge distillation and transfer learning tasks and show that our method consistently outperforms existing methods. We further demonstrate the strength of our method on knowledge transfer across heterogeneous network architectures by transferring knowledge from a convolutional neural network (CNN) to a multi-layer perceptron (MLP) on CIFAR-10. The resulting MLP significantly outperforms the-state-of-the-art methods and it achieves similar performance to the CNN with a single convolutional layer. (@ahn2019variational)

Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency Openface: an open source facial behavior analysis toolkit In *2016 IEEE Winter Conference on Applications of Computer Vision (WACV)*, pages 1–10. IEEE, 2016. **Abstract:** Over the past few years, there has been an increased interest in automatic facial behavior analysis and understanding. We present OpenFace - an open source tool intended for computer vision and machine learning researchers, affective computing community and people interested in building interactive applications based on facial behavior analysis. OpenFace is the first open source tool capable of facial landmark detection, head pose estimation, facial action unit recognition, and eye-gaze estimation. The computer vision algorithms which represent the core of OpenFace demonstrate state-of-the-art results in all of the above mentioned tasks. Furthermore, our tool is capable of real-time performance and is able to run from a simple webcam without any specialist hardware. Finally, OpenFace allows for easy integration with other applications and devices through a lightweight messaging system. (@baltruvsaitis2016openface)

Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, and Dumitru Erhan Domain separation networks , 29, 2016. **Abstract:** The cost of large scale data collection and annotation often makes the application of machine learning algorithms to new tasks or datasets prohibitively expensive. One approach circumventing this cost is training models on synthetic data where annotations are provided automatically. Despite their appeal, such models often fail to generalize from synthetic to real images, necessitating domain adaptation algorithms to manipulate these models before they can be successfully applied. Existing approaches focus either on mapping representations from one domain to the other, or on learning to extract features that are invariant to the domain from which they were extracted. However, by focusing only on creating a mapping or shared representation between the two domains, they ignore the individual characteristics of each domain. We suggest that explicitly modeling what is unique to each domain can improve a model’s ability to extract domain-invariant features. Inspired by work on private-shared component analysis, we explicitly learn to extract image representations that are partitioned into two subspaces: one component which is private to each domain and one which is shared across domains. Our model is trained not only to perform the task we care about in the source domain, but also to use the partitioned representation to reconstruct the images from both domains. Our novel architecture results in a model that outperforms the state-of-the-art on a range of unsupervised domain adaptation scenarios and additionally produces visualizations of the private and shared representations enabling interpretation of the domain adaptation process. (@bousmalis2016domain)

Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S Narayanan Iemocap: Interactive emotional dyadic motion capture database , 42:335–359, 2008. **Abstract:** We propose a speech-emotion recognition (SER) model with an “attention-long Long Short-Term Memory (LSTM)-attention” component to combine IS09, a commonly used feature for SER, and mel spectrogram, and we analyze the reliability problem of the interactive emotional dyadic motion capture (IEMOCAP) database. The attention mechanism of the model focuses on emotion-related elements of the IS09 and mel spectrogram feature and the emotion-related duration from the time of the feature. Thus, the model extracts emotion information from a given speech signal. The proposed model for the baseline study achieved a weighted accuracy (WA) of 68% for the improvised dataset of IEMOCAP. However, the WA of the proposed model of the main study and modified models could not achieve more than 68% in the improvised dataset. This is because of the reliability limit of the IEMOCAP dataset. A more reliable dataset is required for a more accurate evaluation of the model’s performance. Therefore, in this study, we reconstructed a more reliable dataset based on the labeling results provided by IEMOCAP. The experimental results of the model for the more reliable dataset confirmed a WA of 73%. (@busso2008iemocap)

Jae Won Cho, Dong-Jin Kim, Jinsoo Choi, Yunjae Jung, and In So Kweon Dealing with missing modalities in the visual question answer-difference prediction task through knowledge distillation In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1592–1601, 2021. **Abstract:** In this work, we address the issues of the missing modalities that have arisen from the Visual Question Answer-Difference prediction task and find a novel method to solve the task at hand. We address the missing modality–the ground truth answers–that are not present at test time and use a privileged knowledge distillation scheme to deal with the issue of the missing modality. In order to efficiently do so, we first introduce a model, the "Big" Teacher, that takes the image/question/answer triplet as its input and out-performs the baseline, then use a combination of models to distill knowledge to a target network (student) that only takes the image/question pair as its inputs. We experiment our models on the VizWiz and VQA-V2 Answer Difference datasets and show through extensive experimentation and ablation the performance of our method and a diverse possibility for future research. (@cho2021dealing)

Jang Hyun Cho and Bharath Hariharan On the efficacy of knowledge distillation In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 4794–4802, 2019. **Abstract:** In this paper, we present a thorough evaluation of the efficacy of knowledge distillation and its dependence on student and teacher architectures. Starting with the observation that more accurate teachers often don’t make good teachers, we attempt to tease apart the factors that affect knowledge distillation performance. We find crucially that larger models do not often make better teachers. We show that this is a consequence of mismatched capacity, and that small students are unable to mimic large teachers. We find typical ways of circumventing this (such as performing a sequence of knowledge distillation steps) to be ineffective. Finally, we show that this effect can be mitigated by stopping the teacher’s training early. Our results generalize across datasets and models. (@cho2019efficacy)

Gilles Degottex, John Kane, Thomas Drugman, Tuomo Raitio, and Stefan Scherer Covarep—a collaborative voice analysis repository for speech technologies In *2014 Ieee International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 960–964. IEEE, 2014. **Abstract:** Speech processing algorithms are often developed demonstrating improvements over the state-of-the-art, but sometimes at the cost of high complexity. This makes algorithm reimplementations based on literature difficult, and thus reliable comparisons between published results and current work are hard to achieve. This paper presents a new collaborative and freely available repository for speech processing algorithms called COVAREP, which aims at fast and easy access to new speech processing algorithms and thus facilitating research in the field. We envisage that COVAREP will allow more reproducible research by strengthening complex implementations through shared contributions and openly available code which can be discussed, commented on and corrected by the community. Presently COVAREP contains contributions from five distinct laboratories and we encourage contributions from across the speech processing research field. In this paper, we provide an overview of the current offerings of COVAREP and also include a demonstration of the algorithms through an emotion classification experiment. (@degottex2014covarep)

Changde Du, Changying Du, Hao Wang, Jinpeng Li, Wei-Long Zheng, Bao-Liang Lu, and Huiguang He Semi-supervised deep generative modelling of incomplete multi-modality emotional data In *Proceedings of the 26th ACM international conference on Multimedia (ACM MM)*, pages 108–116, 2018. **Abstract:** There are threefold challenges in emotion recognition. First, it is difficult to recognize human’s emotional states only considering a single modality. Second, it is expensive to manually annotate the emotional data. Third, emotional data often suffers from missing modalities due to unforeseeable sensor malfunction or configuration issues. In this paper, we address all these problems under a novel multi-view deep generative framework. Specifically, we propose to model the statistical relationships of multi-modality emotional data using multiple modality-specific generative networks with a shared latent space. By imposing a Gaussian mixture assumption on the posterior approximation of the shared latent variables, our framework can learn the joint deep representation from multiple modalities and evaluate the importance of each modality simultaneously. To solve the labeled-data-scarcity problem, we extend our multi-view model to semi-supervised learning scenario by casting the semi-supervised classification problem as a specialized missing data imputation task. To address the missing-modality problem, we further extend our semi-supervised multi-view model to deal with incomplete data, where a missing view is treated as a latent variable and integrated out during inference. This way, the proposed overall framework can utilize all available (both labeled and unlabeled, as well as both complete and incomplete) data to improve its generalization ability. The experiments conducted on two real multi-modal emotion datasets demonstrated the superiority of our framework. (@du2018semi)

Yangtao Du, Dingkang Yang, Peng Zhai, Mingchen Li, and Lihua Zhang Learning associative representation for facial expression recognition In *IEEE International Conference on Image Processing (ICIP)*, pages 889–893, 2021. **Abstract:** The main inherent challenges with the Facial Expression Recognition (FER) are high intra-class variations and high inter-class similarities, while existing methods pay little attention to the association within inter- and intra-class expressions. This paper introduces a novel Expression Associative Network (EAN) to learn association of facial expression, specifically, from two aspects: 1) associative topological relation over mini-batch is constructed by similarity matrix with an adjacent regularization, and 2) learning association of expressions with Graph Convolutional Network (GCN). Besides, an auxiliary module as invariant feature generator based on Generative Adversarial Networks (GAN) is designed to suppress pose variations, illumination changes, and occlusions. Results on public benchmarks achieve comparable or better performance compared with current state-of-the-art methods, with 90.07% on FERPlus, 86.36% on RAF-DB, and improve by 3.92% over SOTA on synthetic wrong labeling datasets. (@du2021learning)

Tommaso Furlanello, Zachary Lipton, Michael Tschannen, Laurent Itti, and Anima Anandkumar Born again neural networks In *International Conference on Machine Learning (ICML)*, pages 1607–1616. PMLR, 2018. **Abstract:** Knowledge Distillation (KD) consists of transferring Ã¢ÂÂknowledgeÃ¢ÂÂ from one machine learning model (the teacher) to another (the student). Commonly, the teacher is a high-capacity model with formidable performance, while the student is more compact. By transferring knowledge, one hopes to benefit from the studentÃ¢ÂÂs compactness, without sacrificing too much performance. We study KD from a new perspective: rather than compressing models, we train students parameterized identically to their teachers. Surprisingly, these Born-Again Networks (BANs), outperform their teachers significantly, both on computer vision and language modeling tasks. Our experiments with BANs based on DenseNets demonstrate state-of-the-art performance on the CIFAR-10 (3.5%) and CIFAR-100 (15.5%) datasets, by validation error. Additional experiments explore two distillation objectives: (i) Confidence-Weighted by Teacher Max (CWTM) and (ii) Dark Knowledge with Permuted Predictions (DKPP). Both methods elucidate the essential components of KD, demonstrating the effect of the teacher outputs on both predicted and non-predicted classes. (@furlanello2018born)

Wei Han, Hui Chen, and Soujanya Poria Improving multimodal fusion with hierarchical mutual information maximization for multimodal sentiment analysis , 2021. **Abstract:** In multimodal sentiment analysis (MSA), the performance of a model highly depends on the quality of synthesized embeddings. These embeddings are generated from the upstream process called multimodal fusion, which aims to extract and combine the input unimodal raw data to produce a richer multimodal representation. Previous work either back-propagates the task loss or manipulates the geometric property of feature spaces to produce favorable fusion results, which neglects the preservation of critical task-related information that flows from input to the fusion results. In this work, we propose a framework named MultiModal InfoMax (MMIM), which hierarchically maximizes the Mutual Information (MI) in unimodal input pairs (inter-modality) and between multimodal fusion result and unimodal input in order to maintain task-related information through multimodal fusion. The framework is jointly trained with the main task (MSA) to improve the performance of the downstream MSA task. To address the intractable issue of MI bounds, we further formulate a set of computationally simple parametric and non-parametric methods to approximate their truth value. Experimental results on the two widely used datasets demonstrate the efficacy of our approach. The implementation of this work is publicly available at https://github.com/declare-lab/Multimodal-Infomax. (@han2021improving)

Devamanyu Hazarika, Roger Zimmermann, and Soujanya Poria Misa: Modality-invariant and-specific representations for multimodal sentiment analysis In *Proceedings of the 28th ACM International Conference on Multimedia (ACM MM)*, pages 1122–1131, 2020. **Abstract:** Multimodal Sentiment Analysis is an active area of research that leverages multimodal signals for affective understanding of user-generated videos. The predominant approach, addressing this task, has been to develop sophisticated fusion techniques. However, the heterogeneous nature of the signals creates distributional modality gaps that pose significant challenges. In this paper, we aim to learn effective modality representations to aid the process of fusion. We propose a novel framework, MISA, which projects each modality to two distinct subspaces. The first subspace is modality-invariant, where the representations across modalities learn their commonalities and reduce the modality gap. The second subspace is modality-specific, which is private to each modality and captures their characteristic features. These representations provide a holistic view of the multimodal data, which is used for fusion that leads to task predictions. Our experiments on popular sentiment analysis benchmarks, MOSI and MOSEI, demonstrate significant gains over state-of-the-art models. We also consider the task of Multimodal Humor Detection and experiment on the recently proposed UR_FUNNY dataset. Here too, our model fares better than strong baselines, establishing MISA as a useful multimodal framework. (@hazarika2020misa)

Byeongho Heo, Jeesoo Kim, Sangdoo Yun, Hyojin Park, Nojun Kwak, and Jin Young Choi A comprehensive overhaul of feature distillation In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 1921–1930, 2019. **Abstract:** We investigate the design aspects of feature distillation methods achieving network compression and propose a novel feature distillation method in which the distillation loss is designed to make a synergy among various aspects: teacher transform, student transform, distillation feature position and distance function. Our proposed distillation loss includes a feature transform with a newly designed margin ReLU, a new distillation feature position, and a partial L \<sub xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>2\</sub\> distance function to skip redundant information giving adverse effects to the compression of student. In ImageNet, our proposed method achieves 21.65% of top-1 error with ResNet50, which outperforms the performance of the teacher network, ResNet152. Our proposed method is evaluated on various tasks such as image classification, object detection and semantic segmentation and achieves a significant performance improvement in all tasks. The code is available at bhheo.github.io/overhaul. (@heo2019comprehensive)

Byeongho Heo, Minsik Lee, Sangdoo Yun, and Jin Young Choi Knowledge transfer via distillation of activation boundaries formed by hidden neurons In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, volume 33, pages 3779–3787, 2019. **Abstract:** An activation boundary for a neuron refers to a separating hyperplane that determines whether the neuron is activated or deactivated. It has been long considered in neural networks that the activations of neurons, rather than their exact output values, play the most important role in forming classificationfriendly partitions of the hidden feature space. However, as far as we know, this aspect of neural networks has not been considered in the literature of knowledge transfer. In this paper, we propose a knowledge transfer method via distillation of activation boundaries formed by hidden neurons. For the distillation, we propose an activation transfer loss that has the minimum value when the boundaries generated by the student coincide with those by the teacher. Since the activation transfer loss is not differentiable, we design a piecewise differentiable loss approximating the activation transfer loss. By the proposed method, the student learns a separating boundary between activation region and deactivation region formed by each neuron in the teacher. Through the experiments in various aspects of knowledge transfer, it is verified that the proposed method outperforms the current state-of-the-art. (@heo2019knowledge)

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean Distilling the knowledge in a neural network , 2015. **Abstract:** A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel. (@hinton2015distilling)

R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio Learning deep representations by mutual information estimation and maximization , 2018. **Abstract:** In this work, we perform unsupervised learning of representations by maximizing mutual information between an input and the output of a deep neural network encoder. Importantly, we show that structure matters: incorporating knowledge about locality of the input to the objective can greatly influence a representation’s suitability for downstream tasks. We further control characteristics of the representation by matching to a prior distribution adversarially. Our method, which we call Deep InfoMax (DIM), outperforms a number of popular unsupervised learning methods and competes with fully-supervised learning on several classification tasks. DIM opens new avenues for unsupervised learning of representations and is an important step towards flexible formulations of representation-learning objectives for specific end-goals. (@hjelm2018learning)

Minhao Hu, Matthis Maillard, Ya Zhang, Tommaso Ciceri, Giammarco La Barbera, Isabelle Bloch, and Pietro Gori Knowledge distillation from multi-modal to mono-modal segmentation networks In *Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part I 23*, pages 772–781. Springer, 2020. **Abstract:** The joint use of multiple imaging modalities for medical im- age segmentation has been widely studied in recent years. The fusion of information from di erent modalities has demonstrated to improve the segmentation accuracy, with respect to mono-modal segmentations, in several applications. However, acquiring multiple modalities is usually not possible in a clinical setting due to a limited number of physicians and scanners, and to limit costs and scan time. Most of the time, only one modality is acquired. In this paper, we propose KD-Net, a framework to transfer knowledge from a trained multi-modal network (teacher) to a mono-modal one (student). The proposed method is an adaptation of the generalized distillation framework where the student network is trained on a subset (1 modality) of the teacher’s inputs (n modalities). We illustrate the e ectiveness of the proposed framework in brain tumor segmentation with the BraTS 2018 dataset. Using di erent architectures, we show that the student network e ectively learns from the teacher and always outperforms the baseline mono-modal network in terms of seg- mentation accuracy. 1 Introduction Using multiple modalities to automatically segment medical images has become a common practice in several applications, such as brain tumor segmentation \[11\] or ischemic stroke lesion segmentation \[10\]. Since di erent image modalities can accentuate and better describe di erent tissues, their fusion can improve the seg- mentation accuracy. Although multi-modal models usually give the best results, it is often dicult to obtain multiple modalities in a clinical setting due to a limited number of physicians and scanners, and to limit costs and scan time. In many cases, especially for patients with pathologies or for emergency, only one modality is acquired. Two main strategies have been proposed in the literature to deal with prob- lems where multiple modalities are available at training time but some or most ?The two rst authors contributed equally to this paper.2 M.Hu et al. of them are missing at inference time. The rst one is to train a generative model to synthesize the missing modalities and then perform multi-modal segmenta- tion. In \[13\], the authors have shown that using a synthesized modality helps improving the accuracy of classi cation of brain tumors. Ben Cohen et al. \[1\] generated PET images from CT scans to reduce the number of false positives in the detection of malignant lesions in livers. Generating a syn (@hu2020knowledge)

Hyunjik Kim and Andriy Mnih Disentangling by factorising In *International Conference on Machine Learning (ICML)*, pages 2649–2658. PMLR, 2018. **Abstract:** We define and address the problem of unsupervised learning of disentangled representations on data generated from independent factors of variation. We propose FactorVAE, a method that disentangles by encouraging the distribution of representations to be factorial and hence independent across the dimensions. We show that it improves upon $\\}beta$-VAE by providing a better trade-off between disentanglement and reconstruction quality. Moreover, we highlight the problems of a commonly used disentanglement metric and introduce a new metric that does not suffer from them. (@kim2018disentangling)

Jangho Kim, SeongUk Park, and Nojun Kwak Paraphrasing complex network: Network compression via factor transfer , 31, 2018. **Abstract:** Many researchers have sought ways of model compression to reduce the size of a deep neural network (DNN) with minimal performance degradation in order to use DNNs in embedded systems. Among the model compression methods, a method called knowledge transfer is to train a student network with a stronger teacher network. In this paper, we propose a novel knowledge transfer method which uses convolutional operations to paraphrase teacher’s knowledge and to translate it for the student. This is done by two convolutional modules, which are called a paraphraser and a translator. The paraphraser is trained in an unsupervised manner to extract the teacher factors which are defined as paraphrased information of the teacher network. The translator located at the student network extracts the student factors and helps to translate the teacher factors by mimicking them. We observed that our student network trained with the proposed factor transfer method outperforms the ones trained with conventional knowledge transfer methods. (@kim2018paraphrasing)

Diederik P Kingma and Jimmy Ba Adam: A method for stochastic optimization , 2014. **Abstract:** We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm. (@kingma2014adam)

Saurabh Kumar, Biplab Banerjee, and Subhasis Chaudhuri Online sensor hallucination via knowledge distillation for multimodal image classification , 2019. **Abstract:** We deal with the problem of information fusion driven satellite image/scene classification and propose a generic hallucination architecture considering that all the available sensor information are present during training while some of the image modalities may be absent while testing. It is well-known that different sensors are capable of capturing complementary information for a given geographical area and a classification module incorporating information from all the sources are expected to produce an improved performance as compared to considering only a subset of the modalities. However, the classical classifier systems inherently require all the features used to train the module to be present for the test instances as well, which may not always be possible for typical remote sensing applications (say, disaster management). As a remedy, we provide a robust solution in terms of a hallucination module that can approximate the missing modalities from the available ones during the decision-making stage. In order to ensure better knowledge transfer during modality hallucination, we explicitly incorporate concepts of knowledge distillation for the purpose of exploring the privileged (side) information in our framework and subsequently introduce an intuitive modular training approach. The proposed network is evaluated extensively on a large-scale corpus of PAN-MS image pairs (scene recognition) as well as on a benchmark hyperspectral image dataset (image classification) where we follow different experimental scenarios and find that the proposed hallucination based module indeed is capable of capturing the multi-source information, albeit the explicit absence of some of the sensor information, and aid in improved scene characterization. (@kumar2019online)

Mingcheng Li, Dingkang Yang, Yuxuan Lei, Shunli Wang, Shuaibing Wang, Liuzhen Su, Kun Yang, Yuzheng Wang, Mingyang Sun, and Lihua Zhang A unified self-distillation framework for multimodal sentiment analysis with uncertain missing modalities In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, volume 38, pages 10074–10082, 2024. **Abstract:** Multimodal Sentiment Analysis (MSA) has attracted widespread research attention recently. Most MSA studies are based on the assumption of modality completeness. However, many inevitable factors in real-world scenarios lead to uncertain missing modalities, which invalidate the fixed multimodal fusion approaches. To this end, we propose a Unified multimodal Missing modality self-Distillation Framework (UMDF) to handle the problem of uncertain missing modalities in MSA. Specifically, a unified self-distillation mechanism in UMDF drives a single network to automatically learn robust inherent representations from the consistent distribution of multimodal data. Moreover, we present a multi-grained crossmodal interaction module to deeply mine the complementary semantics among modalities through coarse- and fine-grained crossmodal attention. Eventually, a dynamic feature integration module is introduced to enhance the beneficial semantics in incomplete modalities while filtering the redundant information therein to obtain a refined and robust multimodal representation. Comprehensive experiments on three datasets demonstrate that our framework significantly improves MSA performance under both uncertain missing-modality and complete-modality testing conditions. (@li2024unified)

Mingcheng Li, Dingkang Yang, and Lihua Zhang Towards robust multimodal sentiment analysis under uncertain signal missing , 30:1497–1501, 2023. **Abstract:** Multimodal Sentiment Analysis (MSA) has attracted widespread research attention recently. Most MSA studies are based on the assumption of signal completeness. However, many inevitable factors in real applications lead to uncertain signal missing, causing significant degradation of model performance. To this end, we propose a Robust multimodal Missing Signal Framework (RMSF) to handle the problem of uncertain signal missing for MSA tasks and can be generalized to other multimodal patterns. Specifically, a hierarchical cross modal interaction module in RMSF exploits potential complementary semantics among modalities via coarse- and fine-grained cross modal attention. Furthermore, we design an adaptive feature refinement module to enhance the beneficial semantics of modalities and filter redundant features. Finally, we propose a knowledge integrated self-distillation module that enables dynamic knowledge integration and bidirectional knowledge transfer within a single network to precisely reconstruct missing semantics. Comprehensive experiments are conducted on two datasets, indicating that RMSF significantly improves MSA performance under both uncertain missing-signal and complete-signal cases. (@li2023towards)

Mingcheng Li, Dingkang Yang, Xiao Zhao, Shuaibing Wang, Yan Wang, Kun Yang, Mingyang Sun, Dongliang Kou, Ziyun Qian, and Lihua Zhang Correlation-decoupled knowledge distillation for multimodal sentiment analysis with incomplete modalities In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 12458–12468, 2024. **Abstract:** Multimodal sentiment analysis (MSA) aims to understand human sentiment through multimodal data. Most MSA efforts are based on the assumption of modality completeness. However, in real-world applications, some practical factors cause uncertain modality missingness, which drastically degrades the model’s performance. To this end, we propose a Correlation-decoupled Knowledge Distillation (CorrKD) framework for the MSA task under uncertain missing modalities. Specifically, we present a sample-level contrastive distillation mechanism that transfers comprehensive knowledge containing cross-sample correlations to reconstruct missing semantics. Moreover, a category-guided prototype distillation mechanism is introduced to capture cross-category correlations using category prototypes to align feature distributions and generate favorable joint representations. Eventually, we design a response-disentangled consistency distillation strategy to optimize the sentiment decision boundaries of the student network through response disentanglement and mutual information maximization. Comprehensive experiments on three datasets indicate that our framework can achieve favorable improvements compared with several baselines. (@li2024correlation)

Yong Li, Yuanzhi Wang, and Zhen Cui Decoupled multimodal distilling for emotion recognition In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 6631–6640, 2023. **Abstract:** Human multimodal emotion recognition (MER) aims to perceive human emotions via language, visual and acoustic modalities. Despite the impressive performance of previous MER approaches, the inherent multimodal heterogeneities still haunt and the contribution of different modalities varies significantly. In this work, we mitigate this issue by proposing a decoupled multimodal distillation (DMD) approach that facilitates flexible and adaptive crossmodal knowledge distillation, aiming to enhance the discriminative features of each modality. Specially, the representation of each modality is decoupled into two parts, i.e., modality-irrelevant/-exclusive spaces, in a self-regression manner. DMD utilizes a graph distillation unit (GD-Unit) for each decoupled part so that each GD can be performed in a more specialized and effective manner. A GD-Unit consists of a dynamic graph where each vertice represents a modality and each edge indicates a dynamic knowledge distillation. Such GD paradigm provides a flexible knowledge transfer manner where the distillation weights can be automatically learned, thus enabling diverse crossmodal knowledge transfer patterns. Experimental results show DMD consistently obtains superior performance than state-of-the-art MER methods. Visualization results show the graph edges in DMD exhibit meaningful distributional patterns w.r.t. the modality-irrelevant/-exclusive feature spaces. Codes are re leased at https://github.com/mdswyz/DMD. (@li2023decoupled)

Zheng Lian, Lan Chen, Licai Sun, Bin Liu, and Jianhua Tao Gcnet: graph completion network for incomplete multimodal learning in conversation , 2023. **Abstract:** Conversations have become a critical data format on social media platforms. Understanding conversation from emotion, content and other aspects also attracts increasing attention from researchers due to its widespread application in human-computer interaction. In real-world environments, we often encounter the problem of incomplete modalities, which has become a core issue of conversation understanding. To address this problem, researchers propose various methods. However, existing approaches are mainly designed for individual utterances rather than conversational data, which cannot fully exploit temporal and speaker information in conversations. To this end, we propose a novel framework for incomplete multimodal learning in conversations, called "Graph Complete Network (GCNet)," filling the gap of existing works. Our GCNet contains two well-designed graph neural network-based modules, "Speaker GNN" and "Temporal GNN," to capture temporal and speaker dependencies. To make full use of complete and incomplete data, we jointly optimize classification and reconstruction tasks in an end-to-end manner. To verify the effectiveness of our method, we conduct experiments on three benchmark conversational datasets. Experimental results demonstrate that our GCNet is superior to existing state-of-the-art approaches in incomplete multimodal learning. (@lian2023gcnet)

Zhizhong Liu, Bin Zhou, Dianhui Chu, Yuhang Sun, and Lingqiang Meng Modality translation-based multimodal sentiment analysis under uncertain missing modalities , 101:101973, 2024. (@liu2024modality)

Wei Luo, Mengying Xu, and Hanjiang Lai Multimodal reconstruct and align net for missing modality problem in sentiment analysis In *International Conference on Multimedia Modeling*, pages 411–422. Springer, 2023. (@luo2023multimodal)

Mengmeng Ma, Jian Ren, Long Zhao, Sergey Tulyakov, Cathy Wu, and Xi Peng Smil: Multimodal learning with severely missing modality In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, volume 35, pages 2302–2310, 2021. **Abstract:** A common assumption in multimodal learning is the completeness of training data, i.e., full modalities are available in all training examples. Although there exists research endeavor in developing novel methods to tackle the incompleteness of testing data, e.g., modalities are partially missing in testing examples, few of them can handle incomplete training modalities. The problem becomes even more challenging if considering the case of severely missing, e.g., ninety percent of training examples may have incomplete modalities. For the first time in the literature, this paper formally studies multimodal learning with missing modality in terms of flexibility (missing modalities in training, testing, or both) and efficiency (most training data have incomplete modality). Technically, we propose a new method named SMIL that leverages Bayesian meta-learning in uniformly achieving both objectives. To validate our idea, we conduct a series of experiments on three popular benchmarks: MM-IMDb, CMU-MOSI, and avMNIST. The results prove the state-of-the-art performance of SMIL over existing methods and generative baselines including autoencoders and generative adversarial networks. (@ma2021smil)

Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh Improved knowledge distillation via teacher assistant In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, volume 34, pages 5191–5198, 2020. **Abstract:** Despite the fact that deep neural networks are powerful models and achieve appealing results on many tasks, they are too large to be deployed on edge devices like smartphones or embedded sensor nodes. There have been efforts to compress these networks, and a popular method is knowledge distillation, where a large (teacher) pre-trained network is used to train a smaller (student) network. However, in this paper, we show that the student network performance degrades when the gap between student and teacher is large. Given a fixed student network, one cannot employ an arbitrarily large teacher, or in other words, a teacher can effectively transfer its knowledge to students up to a certain size, not smaller. To alleviate this shortcoming, we introduce multi-step knowledge distillation, which employs an intermediate-sized network (teacher assistant) to bridge the gap between the student and the teacher. Moreover, we study the effect of teacher assistant size and extend the framework to multi-step distillation. Theoretical analysis and extensive experiments on CIFAR-10,100 and ImageNet datasets and on CNN and ResNet architectures substantiate the effectiveness of our proposed approach. (@mirzadeh2020improved)

Louis-Philippe Morency, Rada Mihalcea, and Payal Doshi Towards multimodal sentiment analysis: Harvesting opinions from the web In *Proceedings of the 13th International Conference on Multimodal Interfaces*, pages 169–176, 2011. (@morency2011towards)

Augustus Odena, Christopher Olah, and Jonathon Shlens Conditional image synthesis with auxiliary classifier gans In *International Conference on Machine Learning (ICML)*, pages 2642–2651. PMLR, 2017. **Abstract:** In this paper we introduce new methods for the improved training of generative adversarial networks (GANs) for image synthesis. We construct a variant of GANs employing label conditioning that results in 128 x 128 resolution image samples exhibiting global coherence. We expand on previous work for image quality assessment to provide two new analyses for assessing the discriminability and diversity of samples from class-conditional image synthesis models. These analyses demonstrate that high resolution samples provide class information not present in low resolution samples. Across 1000 ImageNet classes, 128 x 128 samples are more than twice as discriminable as artificially resized 32 x 32 samples. In addition, 84.7% of the classes have samples exhibiting diversity comparable to real ImageNet data. (@odena2017conditional)

Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho Relational knowledge distillation In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 3967–3976, 2019. **Abstract:** Knowledge distillation aims at transferring knowledge acquired in one model (a teacher) to another model (a student) that is typically smaller. Previous approaches can be expressed as a form of training the student to mimic output activations of individual data examples represented by the teacher. We introduce a novel approach, dubbed relational knowledge distillation (RKD), that transfers mutual relations of data examples instead. For concrete realizations of RKD, we propose distance-wise and angle-wise distillation losses that penalize structural differences in relations. Experiments conducted on different tasks show that the proposed method improves educated student models with a significant margin. In particular for metric learning, it allows students to outperform their teachers’ performance, achieving the state of the arts on standard benchmark datasets. (@park2019relational)

Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer Automatic differentiation in pytorch . **Abstract:** The paper presents a simple and robust approach to an implementation of the hardening soil model into finite element calculations.The implementation of the return stress mapping exploits the automatic differentiation of tensor variables provided by the Py-Torch framework.The automatic differentiation allows for a succinct implementation despite the relatively complex structure of the nonlinear equations in the stress return algorithm.The presented approach is not limited to the hardening soil model.It can be utilised in the development and verification of other elasto-plastic constitutive models where expressing and maintaining the Jacobian matrix over different versions of a material model is time-consuming and error-prone. (@paszke2017automatic)

Baoyun Peng, Xiao Jin, Jiaheng Liu, Dongsheng Li, Yichao Wu, Yu Liu, Shunfeng Zhou, and Zhaoning Zhang Correlation congruence for knowledge distillation In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 5007–5016, 2019. **Abstract:** Most teacher-student frameworks based on knowledge distillation (KD) depend on a strong congruent constraint on instance level. However, they usually ignore the correlation between multiple instances, which is also valuable for knowledge transfer. In this work, we propose a new framework named correlation congruence for knowledge distillation (CCKD), which transfers not only the instance-level information but also the correlation between instances. Furthermore, a generalized kernel method based on Taylor series expansion is proposed to better capture the correlation between instances. Empirical experiments and ablation studies on image classification tasks (including CIFAR-100, ImageNet-1K) and metric learning tasks (including ReID and Face Recognition) show that the proposed CCKD substantially outperforms the original KD and other SOTA KD-based methods. The CCKD can be easily deployed in the majority of the teacher-student framework such as KD and hint-based learning methods. (@peng2019correlation)

Jeffrey Pennington, Richard Socher, and Christopher D Manning Glove: Global vectors for word representation In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1532–1543, 2014. **Abstract:** Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of these regularities has remained opaque. We analyze and make explicit the model properties needed for such regularities to emerge in word vectors. The result is a new global logbilinear regression model that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods. Our model efficiently leverages statistical information by training only on the nonzero elements in a word-word cooccurrence matrix, rather than on the entire sparse matrix or on individual context windows in a large corpus. The model produces a vector space with meaningful substructure, as evidenced by its performance of 75% on a recent word analogy task. It also outperforms related models on similarity tasks and named entity recognition. (@pennington2014glove)

Hai Pham, Paul Pu Liang, Thomas Manzini, Louis-Philippe Morency, and Barnabás Póczos Found in translation: Learning robust joint representations by cyclic translations between modalities In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, volume 33, pages 6892–6899, 2019. **Abstract:** Multimodal sentiment analysis is a core research area that studies speaker sentiment expressed from the language, visual, and acoustic modalities. The central challenge in multimodal learning involves inferring joint representations that can process and relate information from these modalities. However, existing work learns joint representations by requiring all modalities as input and as a result, the learned representations may be sensitive to noisy or missing modalities at test time. With the recent success of sequence to sequence (Seq2Seq) models in machine translation, there is an opportunity to explore new ways of learning joint representations that may not require all input modalities at test time. In this paper, we propose a method to learn robust joint representations by translating between modalities. Our method is based on the key insight that translation from a source to a target modality provides a method of learning joint representations using only the source modality as input. We augment modality translations with a cycle consistency loss to ensure that our joint representations retain maximal information from all modalities. Once our translation model is trained with paired multimodal data, we only need data from the source modality at test time for final sentiment prediction. This ensures that our model remains robust from perturbations or missing information in the other modalities. We train our model with a coupled translationprediction objective and it achieves new state-of-the-art results on multimodal sentiment analysis datasets: CMU-MOSI, ICTMMMO, and YouTube. Additional experiments show that our model learns increasingly discriminative joint representations with more input modalities while maintaining robustness to missing or perturbed modalities. (@pham2019found)

Masoomeh Rahimpour, Jeroen Bertels, Ahmed Radwan, Henri Vandermeulen, Stefan Sunaert, Dirk Vandermeulen, Frederik Maes, Karolien Goffin, and Michel Koole Cross-modal distillation to improve mri-based brain tumor segmentation with missing mri sequences , 69(7):2153–2164, 2021. **Abstract:** Convolutional neural networks (CNNs) for brain tumor segmentation are generally developed using complete sets of magnetic resonance imaging (MRI) sequences for both training and inference. As such, these algorithms are not trained for realistic, clinical scenarios where parts of the MRI sequences which were used for training, are missing during inference. To increase clinical applicability, we proposed a cross-modal distillation approach to leverage the availability of multi-sequence MRI data for training and generate an enriched CNN model which uses only single-sequence MRI data for inference but outperforms a single-sequence CNN model. We assessed the performance of the proposed method for whole tumor and tumor core segmentation with multi-sequence MRI data available for training but only T1-weighted (\[Formula: see text\]) sequence data available for inference, using BraTS 2018, and in-house datasets. Results showed that cross-modal distillation significantly improved the Dice score for both whole tumor and tumor core segmentation when only \[Formula: see text\] sequence data were available for inference. For the evaluation using the in-house dataset, cross-modal distillation achieved an average Dice score of 79.04% and 69.39% for whole tumor and tumor core segmentation, respectively, while a single-sequence U-Net model using \[Formula: see text\] sequence data for both training and inference achieved an average Dice score of 73.60% and 62.62%, respectively. These findings confirmed cross-modal distillation as an effective method to increase the potential of single-sequence CNN models such that segmentation performance is less compromised by missing MRI sequences or having only one MRI sequence available for segmentation. (@rahimpour2021cross)

Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio Fitnets: Hints for thin deep nets , 2014. **Abstract:** While depth tends to improve network performances, it also makes gradient-based training more difficult since deeper networks tend to be more non-linear. The recently proposed knowledge distillation approach is aimed at obtaining small and fast-to-execute models, and it has shown that a student network could imitate the soft output of a larger teacher network or ensemble of networks. In this paper, we extend this idea to allow the training of a student that is deeper and thinner than the teacher, using not only the outputs but also the intermediate representations learned by the teacher as hints to improve the training process and final performance of the student. Because the student intermediate hidden layer will generally be smaller than the teacher’s intermediate hidden layer, additional parameters are introduced to map the student hidden layer to the prediction of the teacher hidden layer. This allows one to train deeper students that can generalize better or run faster, a trade-off that is controlled by the chosen student capacity. For example, on CIFAR-10, a deep student network with almost 10.4 times less parameters outperforms a larger, state-of-the-art teacher network. (@romero2014fitnets)

Roee Shraga, Haggai Roitman, Guy Feigenblat, and Mustafa Cannim Web table retrieval using multimodal deep learning In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 1399–1408, 2020. **Abstract:** We address the web table retrieval task, aiming to retrieve and rank web tables as whole answers to a given information need. To this end, we formally define web tables as multimodal objects. We then suggest a neural ranking model, termed MTR, which makes a novel use of Gated Multimodal Units (GMUs) to learn a joint-representation of the query and the different table modalities. We further enhance this model with a co-learning approach which utilizes automatically learned query-independent and query-dependent "helper” labels. We evaluate the proposed solution using both ad hoc queries (WikiTables) and natural language questions (GNQtables). Overall, we demonstrate that our approach surpasses the performance of previously studied state-of-the-art baselines. (@shraga2020web)

Matthias Springstein, Eric Müller-Budack, and Ralph Ewerth Quti! quantifying text-image consistency in multimodal documents In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 2575–2579, 2021. **Abstract:** The World Wide Web and social media platforms have become popular sources for news and information. Typically, multimodal information, e.g., image and text is used to convey information more effectively and to attract attention. While in most cases image content is decorative or depicts additional information, it has also been leveraged to spread misinformation and rumors in recent years. In this paper, we present a web-based demo application that automatically quantifies the cross-modal relations of entities\~(persons, locations, and events) in image and text. The applications are manifold. For example, the system can help users to explore multimodal articles more efficiently, or can assist human assessors and fact-checking efforts in the verification of the credibility of news stories, tweets, or other multimodal documents. (@springstein2021quti)

Hao Sun, Hongyi Wang, Jiaqing Liu, Yen-Wei Chen, and Lanfen Lin Cubemlp: An mlp-based model for multimodal sentiment analysis and depression estimation In *Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)*, pages 3722–3729, 2022. **Abstract:** Multimodal sentiment analysis and depression estimation are two important research topics that aim to predict human mental states using multimodal data. Previous research has focused on developing effective fusion strategies for exchanging and integrating mind-related information from different modalities. Some MLP-based techniques have recently achieved considerable success in a variety of computer vision tasks. Inspired by this, we explore multimodal approaches with a feature-mixing perspective in this study. To this end, we introduce CubeMLP, a multimodal feature processing framework based entirely on MLP. CubeMLP consists of three independent MLP units, each of which has two affine transformations. CubeMLP accepts all relevant modality features as input and mixes them across three axes. After extracting the characteristics using CubeMLP, the mixed multimodal features are flattened for task predictions. Our experiments are conducted on sentiment analysis datasets: CMU-MOSI and CMU-MOSEI, and depression estimation dataset: AVEC2019. The results show that CubeMLP can achieve state-of-the-art performance with a much lower computing cost. (@sun2022cubemlp)

Yonglong Tian, Dilip Krishnan, and Phillip Isola Contrastive representation distillation , 2019. **Abstract:** Often we wish to transfer representational knowledge from one neural network to another. Examples include distilling a large network into a smaller one, transferring knowledge from one sensory modality to a second, or ensembling a collection of models into a single estimator. Knowledge distillation, the standard approach to these problems, minimizes the KL divergence between the probabilistic outputs of a teacher and student network. We demonstrate that this objective ignores important structural knowledge of the teacher network. This motivates an alternative objective by which we train a student to capture significantly more information in the teacher’s representation of the data. We formulate this objective as contrastive learning. Experiments demonstrate that our resulting new objective outperforms knowledge distillation and other cutting-edge distillers on a variety of knowledge transfer tasks, including single model compression, ensemble distillation, and cross-modal transfer. Our method sets a new state-of-the-art in many transfer tasks, and sometimes even outperforms the teacher network when combined with knowledge distillation. Code: http://github.com/HobbitLong/RepDistiller. (@tian2019contrastive)

Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J Zico Kolter, Louis-Philippe Morency, and Ruslan Salakhutdinov Multimodal transformer for unaligned multimodal language sequences In *Proceedings of the conference. Association for Computational Linguistics. Meeting*, volume 2019, page 6558. NIH Public Access, 2019. **Abstract:** Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, Ruslan Salakhutdinov. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. (@tsai2019multimodal)

Frederick Tung and Greg Mori Similarity-preserving knowledge distillation In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 1365–1374, 2019. **Abstract:** Knowledge distillation is a widely applicable technique for training a student neural network under the guidance of a trained teacher network. For example, in neural network compression, a high-capacity teacher is distilled to train a compact student; in privileged learning, a teacher trained with privileged data is distilled to train a student without access to that data. The distillation loss determines how a teacher’s knowledge is captured and transferred to the student. In this paper, we propose a new form of knowledge distillation loss that is inspired by the observation that semantically similar inputs tend to elicit similar activation patterns in a trained network. Similarity-preserving knowledge distillation guides the training of a student network such that input pairs that produce similar (dissimilar) activations in the teacher network produce similar (dissimilar) activations in the student network. In contrast to previous distillation methods, the student is not required to mimic the representation space of the teacher, but rather to preserve the pairwise similarities in its own representation space. Experiments on three public datasets demonstrate the potential of our approach. (@tung2019similarity)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin Attention is all you need , 30, 2017. **Abstract:** The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. (@vaswani2017attention)

Hu Wang, Congbo Ma, Jianpeng Zhang, Yuan Zhang, Jodie Avery, Louise Hull, and Gustavo Carneiro Learnable cross-modal knowledge distillation for multi-modal learning with missing modality In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, pages 216–226. Springer, 2023. **Abstract:** The problem of missing modalities is both critical and non-trivial to be handled in multi-modal models. It is common for multi-modal tasks that certain modalities contribute more compared to other modalities, and if those important modalities are missing, the model performance drops significantly. Such fact remains unexplored by current multi-modal approaches that recover the representation from missing modalities by feature reconstruction or blind feature aggregation from other modalities, instead of extracting useful information from the best performing modalities. In this paper, we propose a Learnable Cross-modal Knowledge Distillation (LCKD) model to adaptively identify important modalities and distil knowledge from them to help other modalities from the cross-modal perspective for solving the missing modality issue. Our approach introduces a teacher election procedure to select the most “qualified” teachers based on their single modality performance on certain tasks. Then, cross-modal knowledge distillation is performed between teacher and student modalities for each task to push the model parameters to a point that is beneficial for all tasks. Hence, even if the teacher modalities for certain tasks are missing during testing, the available student modalities can accomplish the task well enough based on the learned knowledge from their automatically elected teacher modalities. Experiments on the Brain Tumour Segmentation Dataset 2018 (BraTS2018) shows that LCKD outperforms other methods by a considerable margin, improving the state-of-the-art performance by 3.61% for enhancing tumour, 5.99% for tumour core, and 3.76% for whole tumour in terms of segmentation Dice score. (@wang2023learnable)

Yansen Wang, Ying Shen, Zhun Liu, Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency Words can shift: Dynamically adjusting word representations using nonverbal behaviors In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, volume 33, pages 7216–7223, 2019. **Abstract:** Humans convey their intentions through the usage of both verbal and nonverbal behaviors during face-to-face communication. Speaker intentions often vary dynamically depending on different nonverbal contexts, such as vocal patterns and facial expressions. As a result, when modeling human language, it is essential to not only consider the literal meaning of the words but also the nonverbal contexts in which these words appear. To better model human language, we first model expressive nonverbal representations by analyzing the fine-grained visual and acoustic patterns that occur during word segments. In addition, we seek to capture the dynamic nature of nonverbal intents by shifting word representations based on the accompanying nonverbal behaviors. To this end, we propose the Recurrent Attended Variation Embedding Network (RAVEN) that models the fine-grained structure of nonverbal subword sequences and dynamically shifts word representations based on nonverbal cues. Our proposed model achieves competitive performance on two publicly available datasets for multimodal sentiment analysis and emotion recognition. We also visualize the shifted word representations in different nonverbal contexts and summarize common patterns regarding multimodal variations of word representations. (@wang2019words)

Yuanzhi Wang, Zhen Cui, and Yong Li Distribution-consistent modal recovering for incomplete multimodal learning In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 22025–22034, 2023. **Abstract:** Recovering missing modality is popular in incomplete multimodal learning because it usually benefits downstream tasks. However, the existing methods often directly estimate missing modalities from the observed ones by deep neural networks, lacking consideration of the distribution gap between modalities, resulting in the inconsistency of distributions between the recovered and the true data. To mitigate this issue, in this work, we propose a novel recovery paradigm, Distribution-Consistent Modal Recovering (DiCMoR), to transfer the distributions from available modalities to missing modalities, which thus maintains the distribution consistency of recovered data. In particular, we design a class-specific flow based modality recovery method to transform cross-modal distributions on the condition of sample class, which could well predict a distribution-consistent space for missing modality by virtue of the invertibility and exact density estimation of normalizing flow. The generated data from the predicted distribution is integrated with available modalities for the task of classification. Experiments show that DiCMoR gains superior performances and is more robust than existing state-of-the-art methods under various missing patterns. Visualization results show that the distribution gaps between recovered modalities and missing modalities are mitigated. Codes are released at https://github.com/mdswyz/DiCMoR. (@wang2023distribution)

Zilong Wang, Zhaohong Wan, and Xiaojun Wan Transmodality: An end2end fusion method with transformer for multimodal sentiment analysis In *Proceedings of The Web Conference 2020*, pages 2514–2520, 2020. **Abstract:** Multimodal sentiment analysis is an important research area that predicts speaker’s sentiment tendency through features extracted from textual, visual and acoustic modalities. The central challenge is the fusion method of the multimodal information. A variety of fusion methods have been proposed, but few of them adopt end-to-end translation models to mine the subtle correlation between modalities. Enlightened by recent success of Transformer in the area of machine translation, we propose a new fusion method, TransModality, to address the task of multimodal sentiment analysis. We assume that translation between modalities contributes to a better joint representation of speaker’s utterance. With Transformer, the learned features embody the information both from the source modality and the target modality. We validate our model on multiple multimodal datasets: CMU-MOSI, MELD, IEMOCAP. The experiments show that our proposed method achieves the state-of-the-art performance. (@wang2020transmodality)

Wenke Xia, Xingjian Li, Andong Deng, Haoyi Xiong, Dejing Dou, and Di Hu Robust cross-modal knowledge distillation for unconstrained videos , 2023. **Abstract:** Cross-modal distillation has been widely used to transfer knowledge across different modalities, enriching the representation of the target unimodal one. Recent studies highly relate the temporal synchronization between vision and sound to the semantic consistency for cross-modal distillation. However, such semantic consistency from the synchronization is hard to guarantee in unconstrained videos, due to the irrelevant modality noise and differentiated semantic correlation. To this end, we first propose a \\}textit{Modality Noise Filter} (MNF) module to erase the irrelevant noise in teacher modality with cross-modal context. After this purification, we then design a \\}textit{Contrastive Semantic Calibration} (CSC) module to adaptively distill useful knowledge for target modality, by referring to the differentiated sample-wise semantic correlation in a contrastive fashion. Extensive experiments show that our method could bring a performance boost compared with other distillation methods in both visual action recognition and video retrieval task. We also extend to the audio tagging task to prove the generalization of our method. The source code is available at \\}href{https://github.com/GeWu-Lab/cross-modal-distillation}{https://github.com/GeWu-Lab/cross-modal-distillation}. (@xia2023robust)

Chenglin Yang, Lingxi Xie, Chi Su, and Alan L Yuille Snapshot distillation: Teacher-student optimization in one generation In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 2859–2868, 2019. **Abstract:** Optimizing a deep neural network is a fundamental task in computer vision, yet direct training methods often suffer from over-fitting. Teacher-student optimization aims at providing complementary cues from a model trained previously, but these approaches are often considerably slow due to the pipeline of training a few generations in sequence, i.e., time complexity is increased by several times. This paper presents snapshot distillation (SD), the first framework which enables teacher-student optimization in one generation. The idea of SD is very simple: instead of borrowing supervision signals from previous generations, we extract such information from earlier epochs in the same generation, meanwhile make sure that the difference between teacher and student is sufficiently large so as to prevent under-fitting. To achieve this goal, we implement SD in a cyclic learning rate policy, in which the last snapshot of each cycle is used as the teacher for all iterations in the next cycle, and the teacher signal is smoothed to provide richer information. In standard image classification benchmarks such as CIFAR100 and ILSVRC2012, SD achieves consistent accuracy gain without heavy computational overheads. We also verify that models pre-trained with SD transfers well to object detection and semantic segmentation in the PascalVOC dataset. (@yang2019snapshot)

Dingkang Yang, Zhaoyu Chen, Yuzheng Wang, Shunli Wang, Mingcheng Li, Siao Liu, Xiao Zhao, Shuai Huang, Zhiyan Dong, Peng Zhai, and Lihua Zhang Context de-confounded emotion recognition In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 19005–19015, June 2023. **Abstract:** Context-Aware Emotion Recognition (CAER) is a crucial and challenging task that aims to perceive the emotional states of the target person with contextual information. Recent approaches invariably focus on designing sophisticated architectures or mechanisms to extract seemingly meaningful representations from subjects and contexts. However, a long-overlooked issue is that a context bias in existing datasets leads to a significantly unbalanced distribution of emotional states among different context scenarios. Concretely, the harmful bias is a confounder that misleads existing models to learn spurious correlations based on conventional likelihood estimation, significantly limiting the models’ performance. To tackle the issue, this paper provides a causality-based perspective to disentangle the models from the impact of such bias, and formulate the causalities among variables in the CAER task via a tailored causal graph. Then, we propose a Contextual Causal Intervention Module (CCIM) based on the backdoor adjustment to de-confound the confounder and exploit the true causal effect for model training. CCIM is plug-in and model-agnostic, which improves diverse state-of-the-art approaches by considerable margins. Extensive experiments on three benchmark datasets demonstrate the effectiveness of our CCIM and the significance of causal insight. (@yang2023context)

Dingkang Yang, Shuai Huang, Haopeng Kuang, Yangtao Du, and Lihua Zhang Disentangled representation learning for multimodal emotion recognition In *Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)*, pages 1642–1651, 2022. **Abstract:** Multimodal emotion recognition aims to identify human emotions from text, audio, and visual modalities. Previous methods either explore correlations between different modalities or design sophisticated fusion strategies. However, the serious problem is that the distribution gap and information redundancy often exist across heterogeneous modalities, resulting in learned multimodal representations that may be unrefined. Motivated by these observations, we propose a Feature-Disentangled Multimodal Emotion Recognition (FDMER) method, which learns the common and private feature representations for each modality. Specifically, we design the common and private encoders to project each modality into modality-invariant and modality-specific subspaces, respectively. The modality-invariant subspace aims to explore the commonality among different modalities and reduce the distribution gap sufficiently. The modality-specific subspaces attempt to enhance the diversity and capture the unique characteristics of each modality. After that, a modality discriminator is introduced to guide the parameter learning of the common and private encoders in an adversarial manner. We achieve the modality consistency and disparity constraints by designing tailored losses for the above subspaces. Furthermore, we present a cross-modal attention fusion module to learn adaptive weights for obtaining effective multimodal representations. The final representation is used for different downstream tasks. Experimental results show that the FDMER outperforms the state-of-the-art methods on two multimodal emotion recognition benchmarks. Moreover, we further verify the effectiveness of our model via experiments on the multimodal humor detection task. (@yang2022disentangled)

Dingkang Yang, Shuai Huang, Yang Liu, and Lihua Zhang Contextual and cross-modal interaction for multi-modal speech emotion recognition , 29:2093–2097, 2022. **Abstract:** Speech emotion recognition combining linguistic content and audio signals in the dialog is a challenging task. Nevertheless, previous approaches have failed to explore emotion cues in contextual interactions and ignored the long-range dependencies between elements from different modalities. To tackle the above issues, this letter proposes a multimodal speech emotion recognition method using audio and text data. We first present a contextual transformer module to introduce contextual information via embedding the previous utterances between interlocutors, which enhances the emotion representation of the current utterance. Then, the proposed cross-modal transformer module focuses on the interactions between text and audio modalities, adaptively promoting the fusion from one modality to another. Furthermore, we construct associative topological relation over mini-batch and learn the association between deep fused features with graph convolutional network. Experimental results on the IEMOCAP and MELD datasets show that our method outperforms current state-of-the-art methods. (@yang2022contextual)

Dingkang Yang, Shuai Huang, Shunli Wang, Yang Liu, Peng Zhai, Liuzhen Su, Mingcheng Li, and Lihua Zhang Emotion recognition for multiple context awareness In *Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVII*, pages 144–162. Springer, 2022. **Abstract:** Emotion and a broader range of affective and cognitive states play an important role on the road. While this has been predominantly investigated in terms of driver safety, the approaching advent of autonomous vehicles (AVs) is expected to bring a fundamental shift in focus for emotion recognition in the car, from the driver to the passengers. This work presents a number of affect-enabled applications, including adapting the driving style for an emotional experience or tailoring the infotainment to personal preferences. It attempts to foresee upcoming challenges and provides suggestions for multimodal affect modelling, with a focus on the audio and visual modalities. In particular, this includes context awareness, reliable diarisation of multiple passengers, group affect, and personalisation. Finally, we provide some recommendations on future research directions, including explainability, privacy, and holistic modelling. (@yang2022emotion)

Dingkang Yang, Haopeng Kuang, Shuai Huang, and Lihua Zhang Learning modality-specific and-agnostic representations for asynchronous multimodal language sequences In *Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)*, pages 1708–1717, 2022. **Abstract:** Understanding human behaviors and intents from videos is a challenging task. Video flows usually involve time-series data from different modalities, such as natural language, facial gestures, and acoustic information. Due to the variable receiving frequency for sequences from each modality, the collected multimodal streams are usually unaligned. For multimodal fusion of asynchronous sequences, the existing methods focus on projecting multiple modalities into a common latent space and learning the hybrid representations, which neglects the diversity of each modality and the commonality across different modalities. Motivated by this observation, we propose a Multimodal Fusion approach for learning modality-Specific and modality-Agnostic representations (MFSA) to refine multimodal representations and leverage the complementarity across different modalities. Specifically, a predictive self-attention module is used to capture reliable contextual dependencies and enhance the unique features over the modality-specific spaces. Meanwhile, we propose a hierarchical cross-modal attention module to explore the correlations between cross-modal elements over the modality-agnostic space. In this case, a double-discriminator strategy is presented to ensure the production of distinct representations in an adversarial manner. Eventually, the modality-specific and -agnostic multimodal representations are used together for downstream tasks. Comprehensive experiments on three multimodal datasets clearly demonstrate the superiority of our approach. (@yang2022learning)

Dingkang Yang, Haopeng Kuang, Kun Yang, Mingcheng Li, and Lihua Zhang Towards asynchronous multimodal signal interaction and fusion via tailored transformers , 2024. **Abstract:** The signals from human expressions are usually multimodal, including natural language, facial gestures, and acoustic behaviors. A key challenge is how to fuse multimodal time-series signals with temporal asynchrony. To this end, we present a Transformer-driven Signal Interaction and Fusion (TSIF) approach to effectively model asynchronous multimodal signal sequences. TSIF consists of linear and cross-modal transformer modules with different duties. The linear transformer module efficiently performs the global interaction for multimodal signals, and the vital philosophy is to replace the dot product similarity with the Exponential Kernel while achieving linear complexity by a low-rank matrix decomposition. By targeting the language modality, the cross-modal transformer module aims to capture reliable element correlations among distinct signals and mitigate noise interference in audio and visual modalities. Numerous experiments on two multimodal benchmarks show that our TSIF comparably outperforms previous state-of-the-art models with lower space-time complexities. The systematic analysis also proves the effectiveness of the proposed modules. (@yang2024TSIF)

Dingkang Yang, Mingcheng Li, Linhao Qu, Kun Yang, Peng Zhai, Song Wang, and Lihua Zhang Asynchronous multimodal video sequence fusion via learning modality-exclusive and-agnostic representations , 2024. **Abstract:** —Understanding human intentions ( e.g., emotions) from videos has received considerable attention recently. Video streams generally constitute a blend of temporal data stemming from distinct modalities, including natural language, facial expres- sions, and auditory clues. Despite the impressive advancements of previous works via attention-based paradigms, the inherent temporal asynchrony and modality heterogeneity challenges re- main in multimodal sequence fusion, causing adverse performance bottlenecks. To tackle these issues, we propose a Multimodal fusion approach for learning modality-Exclusive and modality-Agnostic representations (MEA) to refine multimodal features and leverage the complementarity across distinct modalities. On the one hand, MEA introduces a predictive self-attention module to capture reliable context dynamics within modalities and reinforce unique features over the modality-exclusive spaces. On the other hand, a hierarchical cross-modal attention module is designed to explore valuable element correlations among modalities over the modality- agnostic space. Meanwhile, a double-discriminator strategy is presented to ensure the production of distinct representations in an adversarial manner. Eventually, we propose a decoupled graph fusion mechanism to enhance knowledge exchange across heterogeneous modalities and learn robust multimodal repre- sentations for downstream tasks. Numerous experiments are implemented on three multimodal datasets with asynchronous sequences. Systematic analyses show the necessity of our approach. (@yang2024asynchronous)

Dingkang Yang, Mingcheng Li, Dongling Xiao, Yang Liu, Kun Yang, Zhaoyu Chen, Yuzheng Wang, Peng Zhai, Ke Li, and Lihua Zhang Towards multimodal sentiment analysis debiasing via bias purification In *Proceedings of the European Conference on Computer Vision (ECCV)*, 2024. **Abstract:** Multimodal Sentiment Analysis (MSA) aims to understand human intentions by integrating emotion-related clues from diverse modalities, such as visual, language, and audio. Unfortunately, the current MSA task invariably suffers from unplanned dataset biases, particularly multimodal utterance-level label bias and word-level context bias. These harmful biases potentially mislead models to focus on statistical shortcuts and spurious correlations, causing severe performance bottlenecks. To alleviate these issues, we present a Multimodal Counterfactual Inference Sentiment (MCIS) analysis framework based on causality rather than conventional likelihood. Concretely, we first formulate a causal graph to discover harmful biases from already-trained vanilla models. In the inference phase, given a factual multimodal input, MCIS imagines two counterfactual scenarios to purify and mitigate these biases. Then, MCIS can make unbiased decisions from biased observations by comparing factual and counterfactual outcomes. We conduct extensive experiments on several standard MSA benchmarks. Qualitative and quantitative results show the effectiveness of the proposed framework. (@yang2024MCIS)

Dingkang Yang, Yang Liu, Can Huang, Mingcheng Li, Xiao Zhao, Yuzheng Wang, Kun Yang, Yan Wang, Peng Zhai, and Lihua Zhang Target and source modality co-reinforcement for emotion understanding from asynchronous multimodal sequences , 265:110370, 2023. (@yang2023target)

Dingkang Yang, Dongling Xiao, Ke Li, Yuzheng Wang, Zhaoyu Chen, Jinjie Wei, and Lihua Zhang Towards multimodal human intention understanding debiasing via subject-deconfounding , 2024. **Abstract:** Human multimodal language understanding (MLU) is an indispensable component of expression analysis (e.g., sentiment or humor) from heterogeneous modalities, including visual postures, linguistic contents, and acoustic behaviours. Existing works invariably focus on designing sophisticated structures or fusion strategies to achieve impressive improvements. Unfortunately, they all suffer from the subject variation problem due to data distribution discrepancies among subjects. Concretely, MLU models are easily misled by distinct subjects with different expression customs and characteristics in the training data to learn subject-specific spurious correlations, limiting performance and generalizability across new subjects. Motivated by this observation, we introduce a recapitulative causal graph to formulate the MLU procedure and analyze the confounding effect of subjects. Then, we propose SuCI, a simple yet effective causal intervention module to disentangle the impact of subjects acting as unobserved confounders and achieve model training via true causal effects. As a plug-and-play component, SuCI can be widely applied to most methods that seek unbiased predictions. Comprehensive experiments on several MLU benchmarks clearly show the effectiveness of the proposed module. (@yang2024SuCi)

Dingkang Yang, Kun Yang, Haopeng Kuang, Zhaoyu Chen, Yuzheng Wang, and Lihua Zhang Towards context-aware emotion recognition debiasing from a causal demystification perspective via de-confounded training , 2024. **Abstract:** Understanding emotions from diverse contexts has received widespread attention in computer vision communities. The core philosophy of Context-Aware Emotion Recognition (CAER) is to provide valuable semantic cues for recognizing the emotions of target persons by leveraging rich contextual information. Current approaches invariably focus on designing sophisticated structures to extract perceptually critical representations from contexts. Nevertheless, a long-neglected dilemma is that a severe context bias in existing datasets results in an unbalanced distribution of emotional states among different contexts, causing biased visual representation learning. From a causal demystification perspective, the harmful bias is identified as a confounder that misleads existing models to learn spurious correlations based on likelihood estimation, limiting the models’ performance. To address the issue, we embrace causal inference to disentangle the models from the impact of such bias, and formulate the causalities among variables in the CAER task via a customized causal graph. Subsequently, we present a Contextual Causal Intervention Module (CCIM) to de-confound the confounder, which is built upon backdoor adjustment theory to facilitate seeking approximate causal effects during model training. As a plug-and-play component, CCIM can easily integrate with existing approaches and bring significant improvements. Systematic experiments on three datasets demonstrate the effectiveness of our CCIM. (@yang2024towards)

Dingkang Yang, Kun Yang, Mingcheng Li, Shunli Wang, Shuaibing Wang, and Lihua Zhang Robust emotion recognition in context debiasing In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 12447–12457, 2024. **Abstract:** Context-aware emotion recognition (CAER) has recently boosted the practical applications of affective computing techniques in unconstrained environments. Mainstream CAER methods invariably extract ensemble representations from diverse contexts and subject-centred characteristics to perceive the target person’s emotional state. Despite advancements, the biggest challenge remains due to context bias interference. The harmful bias forces the models to rely on spurious correlations between background contexts and emotion labels in likelihood estimation, causing severe performance bottlenecks and confounding valuable context priors. In this paper, we propose a counterfactual emotion inference (CLEF) framework to address the above issue. Specifically, we first formulate a generalized causal graph to decouple the causal relationships among the variables in CAER. Following the causal graph, CLEF introduces a non-invasive context branch to capture the adverse direct effect caused by the context bias. During the inference, we eliminate the direct context effect from the total causal effect by comparing factual and counterfactual outcomes, resulting in bias mitigation and robust prediction. As a model-agnostic framework, CLEF can be readily integrated into existing methods, bringing consistent performance gains. (@yang2024robust)

Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim A gift from knowledge distillation: Fast optimization, network minimization and transfer learning In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4133–4141, 2017. **Abstract:** We introduce a novel technique for knowledge transfer, where knowledge from a pretrained deep neural network (DNN) is distilled and transferred to another DNN. As the DNN performs a mapping from the input space to the output space through many layers sequentially, we define the distilled knowledge to be transferred in terms of flow between layers, which is calculated by computing the inner product between features from two layers. When we compare the student DNN and the original network with the same size as the student DNN but trained without a teacher network, the proposed method of transferring the distilled knowledge as the flow between two layers exhibits three important phenomena: (1) the student DNN that learns the distilled knowledge is optimized much faster than the original model, (2) the student DNN outperforms the original DNN, and (3) the student DNN can learn the distilled knowledge from a teacher DNN that is trained at a different task, and the student DNN outperforms the original DNN that is trained from scratch. (@yim2017gift)

Wenmeng Yu, Hua Xu, Ziqi Yuan, and Jiele Wu Learning modality-specific representations with self-supervised multi-task learning for multimodal sentiment analysis In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, volume 35, pages 10790–10797, 2021. **Abstract:** Representation Learning is a significant and challenging task in multimodal learning. Effective modality representations should contain two parts of characteristics: the consistency and the difference. Due to the unified multimodal annota- tion, existing methods are restricted in capturing differenti- ated information. However, additional unimodal annotations are high time- and labor-cost. In this paper, we design a la- bel generation module based on the self-supervised learning strategy to acquire independent unimodal supervisions. Then, joint training the multimodal and uni-modal tasks to learn the consistency and difference, respectively. Moreover, dur- ing the training stage, we design a weight-adjustment strat- egy to balance the learning progress among different sub- tasks. That is to guide the subtasks to focus on samples with the larger difference between modality supervisions. Last, we conduct extensive experiments on three public multimodal baseline datasets. The experimental results validate the re- liability and stability of auto-generated unimodal supervi- sions. On MOSI and MOSEI datasets, our method surpasses the current state-of-the-art methods. On the SIMS dataset, our method achieves comparable performance than human- annotated unimodal labels. The full codes are available at https://github.com/thuiar/Self-MM. (@yu2021learning)

Ziqi Yuan, Wei Li, Hua Xu, and Wenmeng Yu Transformer-based feature reconstruction network for robust multimodal sentiment analysis In *Proceedings of the 29th ACM International Conference on Multimedia (ACM MM)*, pages 4400–4407, 2021. **Abstract:** Improving robustness against data missing has become one of the core challenges in Multimodal Sentiment Analysis (MSA), which aims to judge speaker sentiments from the language, visual, and acoustic signals. In the current research, translation-based methods and tensor regularization methods are proposed for MSA with incomplete modality features. However, both of them fail to cope with random modality feature missing in non-aligned sequences. In this paper, a transformer-based feature reconstruction network (TFR-Net) is proposed to improve the robustness of models for the random missing in non-aligned modality sequences. First, intra-modal and inter-modal attention-based extractors are adopted to learn robust representations for each element in modality sequences. Then, a reconstruction module is proposed to generate the missing modality features. With the supervision of SmoothL1Loss between generated and complete sequences, TFR-Net is expected to learn semantic-level features corresponding to missing features. Extensive experiments on two public benchmark datasets show that our model achieves good results against data missing across various missing modality combinations and various missing degrees. (@yuan2021transformer)

Ziqi Yuan, Yihe Liu, Hua Xu, and Kai Gao Noise imitation based adversarial training for robust multimodal sentiment analysis , 2023. **Abstract:** As an inevitable phenomenon in real-world applications, data imperfection has emerged as one of the most critical challenges for multimodal sentiment analysis. However, existing approaches tend to overly focus on a specific type of imperfection, leading to performance degradation in real-world scenarios where multiple types of noise exist simultaneously. In this work, we formulate the imperfection with the modality feature missing at the training period and propose the noise intimation based adversarial training framework to improve the robustness against various potential imperfections at the inference period. Specifically, the proposed method first uses temporal feature erasing as the augmentation for noisy instances construction and exploits the modality interactions through the self-attention mechanism to learn multimodal representation for original-noisy instance pairs. Then, based on paired intermediate representation, a novel adversarial training strategy with semantic reconstruction supervision is proposed to learn unified joint representation between noisy and perfect data. For experiments, the proposed method is first verified with the modality feature missing, the same type of imperfection as the training period, and shows impressive performance. Moreover, we show that our approach is capable of achieving outstanding results for other types of imperfection, including modality missing, automation speech recognition error and attacks on text, highlighting the generalizability of our model. Finally, we conduct case studies on general additive distribution, which introduce background noise and blur into raw video clips, further revealing the capability of our proposed method for real-world applications. (@yuan2023noise)

Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency Tensor fusion network for multimodal sentiment analysis , 2017. **Abstract:** Multimodal sentiment analysis is an increasingly popular research area, which extends the conventional language-based definition of sentiment analysis to a multimodal setup where other relevant modalities accompany language. In this paper, we pose the problem of multimodal sentiment analysis as modeling intra-modality and inter-modality dynamics. We introduce a novel model, termed Tensor Fusion Network, which learns both such dynamics end-to-end. The proposed approach is tailored for the volatile nature of spoken language in online videos as well as accompanying gestures and voice. In the experiments, our model outperforms state-of-the-art approaches for both multimodal and unimodal sentiment analysis. (@zadeh2017tensor)

Amir Zadeh, Paul Pu Liang, Navonil Mazumder, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency Memory fusion network for multi-view sequential learning In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, volume 32, 2018. **Abstract:** Multi-view sequential learning is a fundamental problem in machine learning dealing with multi-view sequences. In a multi-view sequence, there exists two forms of interactions between different views: view-specific interactions and cross-view interactions. In this paper, we present a new neural architecture for multi-view sequential learning called the Memory Fusion Network (MFN) that explicitly accounts for both interactions in a neural architecture and continuously models them through time. The first component of the MFN is called the System of LSTMs, where view-specific interactions are learned in isolation through assigning an LSTM function to each view. The cross-view interactions are then identified using a special attention mechanism called the Delta-memory Attention Network (DMAN) and summarized through time with a Multi-view Gated Memory. Through extensive experimentation, MFN is compared to various proposed approaches for multi-view sequential learning on multiple publicly available benchmark datasets. MFN outperforms all the multi-view approaches. Furthermore, MFN outperforms all current state-of-the-art models, setting new state-of-the-art results for all three multi-view datasets. (@zadeh2018memory)

Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency Mosi: multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos , 2016. **Abstract:** People are sharing their opinions, stories and reviews through online video sharing websites every day. Studying sentiment and subjectivity in these opinion videos is experiencing a growing attention from academia and industry. While sentiment analysis has been successful for text, it is an understudied research question for videos and multimedia content. The biggest setbacks for studies in this direction are lack of a proper dataset, methodology, baselines and statistical analysis of how information from different modality sources relate to each other. This paper introduces to the scientific community the first opinion-level annotated corpus of sentiment and subjectivity analysis in online videos called Multimodal Opinion-level Sentiment Intensity dataset (MOSI). The dataset is rigorously annotated with labels for subjectivity, sentiment intensity, per-frame and per-opinion annotated visual features, and per-milliseconds annotated audio features. Furthermore, we present baselines for future studies in this direction as well as a new multimodal fusion approach that jointly models spoken words and visual gestures. (@zadeh2016mosi)

AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2236–2246, 2018. **Abstract:** AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, Louis-Philippe Morency. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018. (@zadeh2018multimodal)

Sergey Zagoruyko and Nikos Komodakis Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer , 2016. **Abstract:** Attention plays a critical role in human visual experience. Furthermore, it has recently been demonstrated that attention can also play an important role in the context of applying artificial neural networks to a variety of tasks from fields such as computer vision and NLP. In this work we show that, by properly defining attention for convolutional neural networks, we can actually use this type of information in order to significantly improve the performance of a student CNN network by forcing it to mimic the attention maps of a powerful teacher network. To that end, we propose several novel methods of transferring attention, showing consistent improvement across a variety of datasets and convolutional neural network architectures. Code and models for our experiments are available at https://github.com/szagoruyko/attention-transfer (@zagoruyko2016paying)

Jiandian Zeng, Tianyi Liu, and Jiantao Zhou Tag-assisted multimodal sentiment analysis under uncertain missing modalities In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 1545–1554, 2022. **Abstract:** Multimodal sentiment analysis has been studied under the assumption that all modalities are available. However, such a strong assumption does not always hold in practice, and most of multimodal fusion models may fail when partial modalities are missing. Several works have addressed the missing modality problem; but most of them only considered the single modality missing case, and ignored the practically more general cases of multiple modalities missing. To this end, in this paper, we propose a Tag-Assisted Transformer Encoder (TATE) network to handle the problem of missing uncertain modalities. Specifically, we design a tag encoding module to cover both the single modality and multiple modalities missing cases, so as to guide the network’s attention to those missing modalities. Besides, we adopt a new space projection pattern to align common vectors. Then, a Transformer encoder-decoder network is utilized to learn the missing modality features. At last, the outputs of the Transformer encoder are used for the final sentiment classification. Extensive experiments are conducted on CMU-MOSI and IEMOCAP datasets, showing that our method can achieve significant improvements compared with several baselines. (@zeng2022tag)

Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia Pyramid scene parsing network In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 2881–2890, 2017. **Abstract:** Scene parsing is challenging for unrestricted open vocabulary and diverse scenes. In this paper, we exploit the capability of global context information by different-region-based context aggregation through our pyramid pooling module together with the proposed pyramid scene parsing network (PSPNet). Our global prior representation is effective to produce good quality results on the scene parsing task, while PSPNet provides a superior framework for pixel-level prediction. The proposed approach achieves state-of-the-art performance on various datasets. It came first in ImageNet scene parsing challenge 2016, PASCAL VOC 2012 benchmark and Cityscapes benchmark. A single PSPNet yields the new record of mIoU accuracy 85.4% on PASCAL VOC 2012 and accuracy 80.2% on Cityscapes. (@zhao2017pyramid)

Jinming Zhao, Ruichen Li, and Qin Jin Missing modality imagination network for emotion recognition with uncertain missing modalities In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 2608–2618, 2021. **Abstract:** Jinming Zhao, Ruichen Li, Qin Jin. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2021. (@zhao2021missing)

</div>

# NeurIPS Paper Checklist [neurips-paper-checklist]

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification: Please refer to the “Abstract” and “1 Introduction” for our paper’s contributions and scopes.

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: Please refer to the “5 Conclusion and Discussion” section for the limitations of our work.

10. Guidelines:

    - The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

    - The authors are encouraged to create a separate "Limitations" section in their paper.

    - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

    - The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

    - The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

    - The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

    - If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

    - While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

11. **Theory Assumptions and Proofs**

12. Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

13. Answer:

14. Justification: No theory assumptions and proofs are provided in the paper.

15. Guidelines:

    - The answer NA means that the paper does not include theoretical results.

    - All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.

    - All assumptions should be clearly stated or referenced in the statement of any theorems.

    - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

    - Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

    - Theorems and Lemmas that the proof relies upon should be properly referenced.

16. **Experimental Result Reproducibility**

17. Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

18. Answer:

19. Justification: The “4.2 Implementation Details” section of the paper describes all the information needed to reproduce the main experimental results.

20. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

    - If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

    - Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

    - While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

      1.  If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

      2.  If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

      3.  If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

      4.  We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

21. **Open access to data and code**

22. Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

23. Answer:

24. Justification: The paper does not provide open access to the data and code.

25. Guidelines:

    - The answer NA means that paper does not include experiments requiring code.

    - Please see the NeurIPS code and data submission guidelines (<https://nips.cc/public/guides/CodeSubmissionPolicy>) for more details.

    - While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

    - The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (<https://nips.cc/public/guides/CodeSubmissionPolicy>) for more details.

    - The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

    - The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

    - At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

    - Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

26. **Experimental Setting/Details**

27. Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

28. Answer:

29. Justification: The “4.2 Implementation Details” section of the paper specify all the training and testing details.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: In Tables 1 and 2 of the paper, we conducted significance tests on the experimental results to demonstrate the superior performance of the proposed framework.

35. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

    - The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

    - The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

    - The assumptions made should be given (e.g., Normally distributed errors).

    - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

    - It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

    - For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

    - If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

36. **Experiments Compute Resources**

37. Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

38. Answer:

39. Justification: The “4.2 Implementation Details” section of the paper explains that all experiments are conducted on four NVIDIA Tesla V100 GPUs.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification: Our research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics.

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: Please refer to the “5 Conclusion and Discussion” sections for the broader impacts of our work

50. Guidelines:

    - The answer NA means that there is no societal impact of the work performed.

    - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

    - Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

    - The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

    - The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

    - If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

51. **Safeguards**

52. Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

53. Answer:

54. Justification: Our paper poses no such risks.

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification: The MOSI, MOSEI and IEMOCAP datasets and the Pytorch toolbox in this paper are existing assets and we cite the references.

60. Guidelines:

    - The answer NA means that the paper does not use existing assets.

    - The authors should cite the original paper that produced the code package or dataset.

    - The authors should state which version of the asset is used and, if possible, include a URL.

    - The name of the license (e.g., CC-BY 4.0) should be included for each asset.

    - For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

    - If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <a href="paperswithcode.com/datasets" class="uri">paperswithcode.com/datasets</a> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

    - For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

    - If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

61. **New Assets**

62. Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

63. Answer:

64. Justification: The paper does not release new assets.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification: The paper does not involve crowdsourcing nor research with human subjects.

70. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

73. Answer:

74. Justification: The paper does not involve crowdsourcing nor research with human subjects.

75. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
