# Propensity Score Alignment of  
Unpaired Multimodal Data

## Abstract

Multimodal representation learning techniques typically require paired samples to learn shared representations, but collecting paired samples can be challenging in fields like biology, where measurement devices often destroy the samples.  We introduce an approach that aligns *unpaired* observations across disparate modalities by borrowing ideas from causal inference.  Viewing each modality as a potential outcome, we show that a propensity score—estimated independently in each modality—captures all information shared between a latent state and an experimental perturbation.  Matching samples on this score yields a lightweight, task-agnostic alignment procedure.  Only two widely satisfied technical conditions are needed—noise that is independent of the perturbation within a modality and injective measurement maps—both of which hold for standard imaging and sequencing pipelines.  Across synthetic benchmarks, the NeurIPS Multimodal Single-Cell Integration Challenge data, and a large-scale PerturbSeq–microscopy experiment, our optimal-transport implementation of propensity-score matching achieves state-of-the-art alignment and significantly boosts downstream cross-modality prediction, even outperforming models trained on ground-truth pairs.  The method requires nothing more than off-the-shelf classifiers, scales linearly with data, and seamlessly generalises to new perturbations and modalities, making it an attractive drop-in replacement for existing paired-data pipelines.
# Introduction

Large-scale multimodal representation learning techniques such as CLIP `\citep{radford2021learning}`{=latex} have lead to remarkable improvements in zero-shot classification performance and have enabled the recent success in conditional generative models. However, the effectiveness of multimodal methods hinges on the availability of *paired* samples—such as images and their associated captions—across data modalities. This reliance on paired samples is most obvious in the InfoNCE loss `\citep{gutmann10a,
oord2018representation}`{=latex} used in CLIP `\citep{radford2021learning}`{=latex} which explicitly learns representations to maximize the true matching between images and their captions.

<figure id="fig:matching">
<img src="./figures/Matching_Figure_1.png"" />
<figcaption> Visualization of propensity score matching for two modalities (e.g., Microscopy images and RNA expression data). We first train classifiers to estimate the propensity score for samples from each modalities; the propensity score reveals the shared information <span class="math inline"><em>p</em>(<em>t</em>|<em>z</em><sub><em>i</em></sub>)</span>, which allows us to re-pair the observed disconnected modalities. The matching procedure is then performed within each perturbation class based on the similarity bewteen the propensity scores. </figcaption>
</figure>

While paired image captioning data is abundant on the internet, paired multimodal data is often challenging to collect in scientific experiments. For instance, unpaired data are the norm in biology for technical reasons: RNA sequencing, protein expression assays, and the collection of microscopy images for cell painting assays are all destructive processes. As such, we cannot collect multiple different measurements from the same cell, and can only explicitly group cells by their experimental condition. If we could accurately match unpaired samples across modalities, we could use the aligned samples as proxies for paired samples and apply existing multimodal learning techniques.

In this paper, we formalize this setting by viewing each modality as a *potential* measurement, \\(X^{(1)}(Z) \in \mathcal{X}^{(1)}, X^{(2)}(Z) \in \mathcal{X}^{(2)}\\), of the same underlying latent state \\(Z\in \mathcal{Z}\\), where we are only able to make a single measurement for each sample unit (e.g. an individual cell). The task is to reconcile (*match*) unpaired observations \\(x^{(1)}\\) and \\(x^{(2)}\\) with the same (or maximally similar) \\(z\\). Estimating the latent, \\(Z\\), is hopelessly underspecified without making unverifiable assumptions on the system, and furthermore, \\(Z\\) may still be sparse and high-dimensional, leading to inefficient matching. This motivates the need for approaches that use only the observable data.

We identify two major challenges for this problem. First, measurements are often made in very different spaces \\(\mathcal{X}^{(1)}\\) and \\(\mathcal{X}^{(2)}\\) (e.g., pixel space and gene expression counts), which make defining a notion of similarity across modalities challenging. Second, the measurement process inevitably introduces modality-specific variation that can be impossible to disentangle from the relevant information (\\(Z\\)). For example in cell imaging, we would not want the matching to depend on irrelevant appearance features such as the orientation of the cell or the lighting of the plate.

In this paper, we address these challenges by appealing to classical ideas from causal inference `\citep{rubin1974estimating}`{=latex}, in the case where we additionally observe some label \\(t\\) for each unit, e.g., indexing an experiment. By making the assumption that \\(t\\) perturbs the observations via their shared latent state, we identify an observable link between modalities with the same underlying \\(z\\). Under conditions which we discuss in  
efsec:dgp, the propensity score, defined as \\(p(t|Z)\\), is a transformation of the latent \\(Z\\) that satisfies three remarkable properties (  
efprop::ps): (1) it provides a common space for matching, (2) it is fully identifiable via classification on individual modalities, and (3) it maximally reduces the dimension of \\(Z\\), retaining only the information revealed by the perturbations.

The practical implementation of the methodology (as illustrated in  
effig:matching) is then straightforward: we train two separate classifiers, one for each modality, to predict the treatment \\(t\\) applied to \\(X^{(i)}\\). We then match across modalities based on the similarity between the predicted probabilities (the propensity score) within each treatment group. This matching procedure is highly versatile and can be applied to match labeled observations between any modalities for which a classifier can be efficiently trained. However, since the same sample unit does not appear in both modalities, we cannot use naive bipartite matching. To address this, we use soft matching techniques to estimate the missing modality for each sample unit by allowing matching to multiple observations. We experiment with two recent matching approaches: shared nearest neighbours (SNN) matching `\citep{lance2022multimodal, cao2022multi}`{=latex} and optimal transport (OT) matching `\cite{villani2009}`{=latex}.

In our experiments, we find that OT matching with distances defined on the proposenity score leads to significant improvement on matching and a downstream cross-modality prediction task on both synthetic and real-world biological data. Notably, our prediction method, which leverages the soft matching to optimize an OT projected loss, outperforms supervised learning on the true pairs on CITE-seq data from the NeurIPS Multimodal Single-Cell Integration Challenge `\citep{lance2022multimodal}`{=latex}. Finally, we applied our method to match single-cell expression data (from a PeturbSeq assay `\citep{dixit2016perturb}`{=latex}) with single cell crops of image data `\citep{Fay2023}`{=latex}. We find improved generalization in predicting the distribution of gene expression from the cell imaging data in with unseen perturbations.

## Related Work

#### Unpaired and Multimodal Data

Learning from unpaired data has long been considered for image translation `\citep{liu2017unsupervised, zhu2017unpaired,
almahairi2018augmented}`{=latex}, and more recently for biological modality translation `\citep{amodio2018magan, yang2021multi}`{=latex}. In particular, `\cite{yang2021multi}`{=latex} also takes the perspective of a shared latent variable for biological modalities. This setting has been studied more generally for multi-view representation learning `\citep{gresele2020incomplete,
sturma2023unpaired}`{=latex} for its identifiability benefits.

**Perturbations and Heterogeneity**   Many methods in biology treat observation-level heterogeneity as a nuisance dimension to globally integrate, even when cluster labels are observed `\citep{butler2018integrating, korsunsky2019fast,
Foster2022}`{=latex}. This is sensible when clusters correspond to noise rather than the signal of interest. However, it is well known in causal representation learning that heterogeneity—particularly heterogeneity arising from perturbations—has theoretical benefits in constraining the solution set `\citep{Khem2020a, squires2023linear,
ahuja2023interventional, buchholz2023learning, von2023nonparametric}`{=latex}. There, the benefits (weakly) increase with the number of perturbations, which is also true of our setting (  
efprop:dimensionality). In the context of unpaired data, only `\citet{yang2021multi}`{=latex} explicitly leverage this heterogeneity in their method, while `\citet{ryu2024cross}`{=latex} treat it as a constraint in solving OT. Specifically, `\citet{yang2021multi}`{=latex} require their VAE representations to classify experimental labels in addition to reconstructing modalities, while our method is simpler, only requiring the classification objective. Notably, `\citet{yang2021multi}`{=latex} treat our objective as a regularizer, but our theory suggests that it is actually primarily responsible for the matching performance. Our experiment results coincide with the theoretical insights; requiring reconstruction, as in a VAE, led to worse matching performance with identical model architectures.

**Optimal Transport Matching**   OT is a common tool in single-cell biology. In cell trajectory inference, the unpaired samples are gene expression values measured at different time points in a shared (metric) space. OT matching minimizes this shared metric between time points `\citep{schiebinger2019optimal, tong2020trajectorynet}`{=latex}. Recent work `\citep{demetci2022scot}`{=latex} extends this to our setting where each modality is observed in separate metric spaces by using the Gromov-Wasserstein distance, which computes the difference between the metric evaluated within pairs of points from each modality `\citep{demetci2022scot}`{=latex}. In concurrent work, this approach was recently extended to ensure matching within experimental labels `\citep{ryu2024cross}`{=latex}. In addition to these “pure” OT approaches, `\citet{gossi2023matching}`{=latex} use OT on contrastive learning representations, though this approach requires matched pairs for training, while `\citet{cao2022unified}`{=latex} use OT in the latent space of a multi-modal VAE.

# Setting [sec:dgp]

We consider the setting where there exist two potential views, \\(X^{(e)} \in \mathcal{X}^{(e)}\\) from two different modalities indexed by \\(e \in \{1, 2\}\\), and experiment \\(t\\) that perturbs a shared latent state of these observations. This defines a jointly distributed random variable \\((X^{(1)}, X^{(2)}, e, t)\\), from which we observe only a single modality, its index, and label, \\(\{x_i^{(e_i)}, e_i, t_i\}_{i=1}^n\\).[^2] We aim to match or estimate the samples from the missing modality, which corresponds to the realization of the missing random variable. Since \\(t\\) is observed, in practice we match observations *within* the same label class \\(t\\).

Formally, we assume each modality is generated by a common latent random variable \\(Z\\) as follows: \\[\begin{aligned}
    \label{eqn::dgp}
    &t \sim P_T, \  Z^{(t)} \mid t \sim P_Z^{(t)}, \  U^{(e)} \sim P_{U}^{(e)}, \   U^{(e)} {\perp\!\!\!\perp}Z,  \  U^{(e)} {\perp\!\!\!\perp}U^{(e')}, \  X^{(e)} \mid t = f^{(e)}(Z^{(t)}, U^{(e)}),
\end{aligned}\\] where \\(t\\) indexes the experimental perturbations, and we take \\(t = 0\\) to represent a base environment. \\(U^{(e)}\\) represents the modality-specific measurement noise that is unperturbed by \\(t\\), and also independent across samples. The structural equations \\(f^e\\) are deterministic after accounting for the randomness in \\(Z\\) and \\(U\\): it represents the measurement process that captures the latent state. For example, in a microscopy image, this would be the microscope and camera that maps a cell to pixels.

#### Comparison to Multimodal Generative Models

Our setting is technically that of a multimodal generative model with latent perturbations. However, by focusing on matching rather than generation, we are able to make significantly weaker and more meaningful assumptions while still ensuring the theoretical validity of our method. Without the effects of the perturbation, our  
efeqn::dgp is essentially the same as `\citep[Equation 1]{yang2021multi}`{=latex} in an abstract sense. However, in order to fit the generative model, it is required to formulate explicit models over \\(f^{(e)}\\) and \\(P_{Z}^{(t)}\\), which requires specifying the function class (e.g., continuous) and the space of \\(Z\\) (e.g., \\(\mathbb{R}^d\\)) as assumptions, even in universal approximation settings. In contrast, since we will not directly fit the model  
efeqn::dgp, we do not make any technical assumptions about the generative model. Instead, we will make the following assumptions on the underlying data generating process itself.

**Key Assumptions**   Our theory makes the following assumptions about the data generating process.

- \\(t \not\!\!\!{\perp\!\!\!\perp}Z\\), and \\(t {\perp\!\!\!\perp}U^{(e)}\\). In words, \\(t\\) has a non-trivial effect on \\(Z\\), but does not affect \\(U^{(e)}\\), implying that interventions target the common underlying process without affecting modality-specific properties. For example, an intervention that affects the underlying cell state, but not the measurement noise of the individual modalities.

- Injectivity of \\(f^{(e)}\\): \\(f^{(e)}(z, u) = f^{(e)}(z', u') \implies (z, u) = (z', u')\\). In words, each modality captures enough information to distinguish changes in the underlying state.[^3]

(A1) ensures that the conditional distribution \\(t \mid X^{(e)}\\) is identical for \\(e = 1, 2\\). (A2) then ensures that \\(t \mid X^{(1)} %
  \mathrel{\mathop{=}\limits^{
    \vbox to0.25ex{\kern-2\ex@
    \hbox{$\scriptstyle\text{\rm\tiny d}$}\vss}}}\ t \mid X^{(2)} %
  \mathrel{\mathop{=}\limits^{
    \vbox to0.25ex{\kern-2\ex@
    \hbox{$\scriptstyle\text{\rm\tiny d}$}\vss}}}\ t \mid Z\\), which allows us to estimate the conditional distribution \\(t \mid Z\\) with observed data alone. Though sharp assumptions are required for the theory, versions replaced with approximate distributional equalities intuitively also allow for effective matchings when combined with our soft matching procedures in practice. A particular relaxation of (A1) when combined with OT matching is described in  
efsec:relaxing.

# Multimodal Propensity Scores

Under <a href="#eqn::dgp" data-reference-type="eqref" data-reference="eqn::dgp">[eqn::dgp]</a>, if \\(Z\\) were observable, an optimal matching can be constructed by simply matching the samples with the most similar \\(z_i\\). However, the prerequisite of inverting the model and disentangling \\(Z\\) is arguably more difficult than the matching problem itself. In particular, \\(Z\\) is unidentifiable without strong assumptions on  
efeqn::dgp `\citep{xi2023indeterminacy}`{=latex}, and even formulating the identifiability problem requires well-specification of the model as a prerequisite. We take an alternative approach that is robust to these problems, by using the perturbations \\(t\\) as an observable link to reveal information about \\(Z\\). Specifically, we show that the propensity score \\[\begin{aligned}
    \pi(z):=P(t | Z = z) \in [0,1]^{T+1},
\end{aligned}\\] is identifiable as a proxy for the latent \\(Z\\) under our assumptions of the data generating process. This is a consequence of the injectivity of \\(f^{(e)}\\), since it will be that \\(\pi(Z) = \pi(X^{(e)})\\), \\(e = 1, 2\\), indicating that we can compute it from either modality. Not only does the propensity score reveal shared information, classical causal inference theory `\citep{rubin1974estimating}`{=latex} states that it captures *all* information about \\(Z\\) that is contained in \\(t\\), and does so minimally, in terms of having minimum dimension and entropy. Since \\(t\\) contains the only observable information that is useful for matching, the propensity score is hence an optimal compression of the observed information. We collect these observations into the following proposition.

<div id="prop::ps" class="proposition" markdown="1">

**Proposition 1**. *In the model described by  
efeqn::dgp, further assume that \\(f^{(e)}\\) are injective for \\(e = 1, 2\\). Then, the propensity scores in either modality is equal to the propensity score given by \\(Z\\), i.e., \\(\pi(X^{(1)}) = \pi(X^{(2)}) = \pi(Z)\\) as random variables. This implies \\[\begin{aligned}
        I(t, Z \mid \pi(Z)) = I(t, Z \mid \pi(X^{(e)})) = 0,
    
\end{aligned}\\] for each \\(e = 1,2\\), where \\(I\\) is the mutual information. Furthermore, any other function \\(b(Z)\\) satisfying \\(I(t, Z \mid b(Z)) = 0\\) is such that \\(\pi(Z) = f(b(Z))\\).*

</div>

The proof can be found in  
efsec:proof. Practically,  
efprop::ps shows that computing the propensity score on either modality is equivalent to computing it on the unobserved shared latent, which means that it is identifiable, and thus estimable, from the observations alone. Furthermore, the estimation does not require modified objectives or architectures for joint multimodal processing, instead they are simple and separate classification problems for each modality. Finally, \\(t\\) does not affect \\(U^{(e)}\\) by assumption, and thus the propensity score, being a representation of the information in \\(t\\), discards the modality-specific information that may be counterproductive to matching. Therefore, even if \\(Z\\) were observed, it may be sensible to match on its propensity score instead.

**Number of Perturbations**   Note that point-wise equality of the propensity score \\(\pi(z_1) = \pi(z_2)\\) does not necessarily imply equality of the latents \\(z_1 = z_2\\), due to potential non-injectivity of \\(\pi\\). For example, consider \\(t \in \{0,1\}\\), then \\(\pi(z)\\) is a compression to a single dimension \\(z \to p(t = 1 \mid z)\\). Intuitively, collecting data from more perturbations improves the amount of information contained in the label \\(t\\). If the latent space is \\(\mathbb{R}^d\\), the propensity score necessarily compresses information about \\(Z^{(t)}\\) if the latent dimension exceeds the number of perturbations, echoing impossibility results from the causal representation learning literature `\citep{squires2023linear}`{=latex}.

<div id="prop:dimensionality" class="proposition" markdown="1">

**Proposition 2**. *Let \\(Z^{(t)} \in \mathbb{R}^{d}\\). Suppose that \\(P_Z^{(t)}\\) has a smooth density \\(p(z|t)\\) for each \\(t = 0, \dots, T\\). Then, if \\(T < d\\), the propensity score \\(\pi\\), restricted to its strictly positive part, is non-injective.*

</div>

The proof can be found in  
efsec:proof. Note the above only states an impossibility result when \\(T<d\\). More generally, it can be seen from the proof of  
efprop:dimensionality that the injectivity of the propensity score depends on the injectivity of the following expression in \\(z\\): \\[\begin{aligned}
    \label{eqn::injective}
    g(z) = \begin{bmatrix} 
    \log(p(z|t=1)) - \log(p(z|t=0)) \\
    \vdots \\
    \log(p(z|t=T)) - \log(p(z|t=0)) 
    \end{bmatrix},
\end{aligned}\\] which then depends on the latent process itself. If the above mapping is non-injective, this represents a fundamental indeterminacy that cannot be resolved without making strong assumptions on point-wise latent variable recovery. As we have already established in  
efprop::ps, the propensity score contains the maximal shared information across modalities. Nonetheless, collecting data form a larger number of perturbations is clearly beneficial for matching, since \\(g\\) in  
efeqn::injective is injective if any of the subset of its entries are.

# Estimation and Matching

For the remainder of the paper, we drop the notation \\(e\\) and use \\((x_i, t_i)\\) to denote observations from modality 1, and \\((x_j, t_j)\\) to denote observations from modality 2. Given a multimodal dataset with observations \\(\{(x_i, t_i)\}_{i=1}^{n_1}\\) and \\(\{(x_j, t_j)\}_{j=1}^{n_2}\\), we wish to compute a matching matrix (or coupling) between the two modalities. We define a \\(n_1 \times n_2\\) matching matrix \\(M\\) where \\(M_{ij}\\) represents the likelihood of \\(x_{i}\\) being matched to \\(x_{j}\\). Since \\(t\\) is observed, we always perform matching only within observations with the same value of \\(t\\), so that in practice we obtain a matrix \\(M_t\\) for each \\(t\\).

Our method approximates the propensity scores by training separate classifiers that predicts \\(t\\) given \\(x\\) for each modality. We denote the estimated propensity score by \\(\pi_i\\) and \\(\pi_j\\) respectively, where \\[\begin{aligned}
    \pi_i
    \approx \pi(x_i) = P(T = t \mid X_i^{(e)} = x_i).
\end{aligned}\\] This yields the transformed datasets \\(\{\pi_i\}_{i=1}^{n_1}\\) and \\(\{\pi_j\}_{j=1}^{n_2}\\), where \\(\pi_i\\), \\(\pi_j\\) are in the \\(T\\) dimensional simplex. We use this correspondence to compute a cross-modality distance function: \\[\begin{aligned}
d(x_i, x_j) := d'(\pi_i, \pi_j). 
\end{aligned}\\] In practice, we typically compute the Euclidean distance in \\(\mathbb{R}^{T}\\) of the logit-transformed classification scores, but any metric over a bijective transformation of the propensity scores are also theoretically valid. Given this distance function, we use existing matching techniques to constructing a matching matrix. In our experiments, we found that OT matching gave the best performance, but we also evaluated Shared Nearest Neighbour matching; details of the latter can be found in Appendix <a href="#sec:shared-nn" data-reference-type="ref" data-reference="sec:shared-nn">11</a>.

#### Optimal Transport Matching

The propensity score distance allows us to easily compute a cost function associated with transporting mass between modalities, \\(c(x_i, x_j) =
    d'(\pi_i, \pi_j)\\). Let \\(p_1, p_2\\) denote the uniform distribution over \\(\{\pi_i\}_{i=1}^{n_1}\\) and \\(\{\pi_j\}_{j=1}^{n_2}\\) respectively. Discrete OT aims to solve the problem of optimally redistributing mass from \\(p_1\\) to \\(p_2\\) in terms of incurring the lowest cost. Let \\(C_{ij} = c(x_i, x_j)\\) denote the \\(n_1 \times n_2\\) cost matrix. The Kantorovich formulation of optimal transport aims to solve the following constrained optimization problem: \\[\begin{aligned}
    \min_{M} \sum_{i}^{n_1}\sum_j^{n_2} C_{ij}M_{ij}, \quad M_{ij} \geq 0,  \quad M\mathbf{1} = p_1, \quad M^\top \mathbf{1} = p_2.
\end{aligned}\\] This is a linear program, and for \\(n_1 = n_2\\), it can be shown that the optimal solution is a bipartite matching between \\(\{\pi_i\}_{i=1}^{n_1}\\) and \\(\{\pi_j\}_{j=1}^{n_2}\\). We refer to this as exact OT; in practice we add an entropic regularization term, resulting in a soft matching, that ensures smoothness and uniqueness, and can be solved efficiently using Sinkhorn’s algorithm. Entropic OT takes the following form: \\[\begin{aligned}
    \min_{M} \sum_{i}^{n_1}\sum_j^{n_2} C_{ij}M_{ij} - \lambda H(M), \quad M_{ij} \geq 0, \quad M\mathbf{1} = p_1, \quad M^\top \mathbf{1} = p_2, 
\end{aligned}\\] where \\(H(M) = - \sum_{i,j} M_{ij} \log(M_{ij})\\), the entropy of the joint distribution implied by \\(M\\). This approach regularizes towards a higher entropy solution, which has been shown to have statistical benefits `\citep{genevay2018learning}`{=latex}, but nonetheless for small enough \\(\lambda\\) serves as a computationally appealing approximation to exact OT.

# Downstream Tasks

The matching matrix \\(M\\) can be seen as defining an empirical joint distribution over the samples in each modality. The OT approach in particular makes this explicit. Each row is proportional to the probability that each sample \\(i\\) from modality (1) is matched to sample \\(j\\) in modality (2), i.e., \\(M_{i, j} = P(x_j | x_i)\\). We can thus use \\(M\\) to obtain pseudosamples for any learning task that uses paired samples by \\((x_i, \hat{x}_j)\\), where \\(\hat{x}_j\\) is obtained by sampling from the conditional distribution defined by \\(M\\), or by a suitable conditional expectation, e.g., the barycentric projection (conditional mean) as \\(E_{M}\left[X_j \mid X_i = x_i \right] = \sum_{j} M_{i,j} x_j\\). In what follows, we describe a cross-modality prediction method based on both barycentric projection and stochastic gradients according to \\(M_{i,j}\\).

#### Cross-modality prediction [sec::unbiased]

We can use the matching matrix to design a method for cross-modality prediction/translation. The following MSE loss corresponds to constructing a prediction function \\(f_{\theta}\\) such that the barycentric projection \\(E_{M}\left[ f_{\theta}(X_j) \mid X_i = x_i\right]\\), under \\(M\\) minimizes the squared error for predicting \\(x_i\\): \\[\begin{aligned}
    \mathcal{L}(\theta) := \sum_{i}(x_i - \sum_{j} M_{i,j} f_\theta(x_j))^{2}. 
\label{eqn::loss}
\end{aligned}\\] However, this requires evaluating \\(f_\theta\\) for all \\(n_2\\) examples from modality (2) for each of the \\(n_1\\) examples in modality (1). In practice, we can avoid this cost with stochastic gradient descent by sampling from modality \\((2)\\) via \\(M_{i\cdot}\\) for each training example \\((1)\\). To obtain an unbiased estimate of \\(\nabla_\theta \mathcal{L}\\), we need two independent samples from modality (2) for each sample from modality (1), \\[\begin{aligned}
    \nabla \mathcal{L}(\theta) \approx & -2\left(x_i -  f_\theta(\dot{x}_j)\right)\nabla_\theta f_\theta(\ddot{x}_j)  \quad \dot{x}_j, \ddot{x}_j \sim P(x_j | x_i). \label{eqn::gradtheta}
\end{aligned}\\] By taking two samples as in  
efeqn::gradtheta, we get an unbiased estimator of \\(\nabla \mathcal{L}(\theta)\\), whereas a single sample would have resulted in optimizing an upper-bound on equation (<a href="#eqn::loss" data-reference-type="ref" data-reference="eqn::loss">[eqn::loss]</a>); for details, see `\citet{hartford2017deep}`{=latex} where a similar issue arises in the gradient of their causal effect estimator. We thus refer to prediction models trained via  
efeqn::gradtheta as *unbiased*.

# Experiments

We present a comprehensive evaluation of our proposed methodology on three distinct datasets: (1) synthetic paired images, (2) single-cell CITE-seq dataset (simultaneous measurement of single-cell RNA-seq and surface protein measurements) `\citep{stoeckius2017simultaneous}`{=latex}, and (3) Perturb-seq and single-cell image data. In the first two cases, there is a ground-truth matching that we use for evaluation, but samples are randomly permuted during training. This allows us to exactly compute the quality of the matching in comparison to the ground truth. The final dataset is a more realistic setting where ground truth paired samples do not exist, and matching becomes necessary in practice. In this case, we compute distributional metrics to compare our proposed methodology against other baselines.

#### Experimental Details

All models for the experiments are implemented using `Torch v2.2.2` `\citep{paszke2017automatic}`{=latex} and `Pytorch Lightning v2.2.4` `\citep{pytorch_lightning}`{=latex}. The classifier used to estimate the propensity score is always a linear head on top of an encoder \\(E_i\\), which is specific to each modality and dataset. All models are saved at the optimal validation loss to perform subsequent matching. Shared nearest neighbours (SNN) is implemented using `scikit-learn v1.4.0` `\citep{scikit-learn}`{=latex} using a single neighbour, and OT is implemented using the Sinkhorn algorithm as implemented in the `pot v0.9.3` package `\citep{pot}`{=latex}. Both SNN and OT use the Euclidean distance as the metric. Whenever random variation can affect the results of the experiments, we report quantiles corresponding to variation from different random seeds. Additional experimental details are provided in  
efsec:exptdetails.

#### Description of Baselines

Our main baseline, which we evaluate against on all three datasets, is matching using representations learned by the multimodal VAE of `\citet{yang2021multi}`{=latex}, which is the only published method that is able to leverage perturbation labels for unpaired multimodal data (they refer to the labels as “prior information”). The standard multimodal VAE loss is a reconstruction loss based on encoder and decoders \\(E_i\\), \\(D_i\\) for each modality, plus a latent invariance loss that aims to align the modalities in the latent space. In our setting, the multimodal VAE loss further includes an additional label classification loss from the latent space of each modality, i.e., encouraging the encoder to simultaneously learn \\(P(t \mid E_i(x_i))\\). This additional objective, which acts as a regularizer for the multimodal VAE, is exactly the loss for our proposed method. To ensure a fair comparison, we always use the same architecture in the encoders \\(E_i\\) of multimodal VAE and in our propensity score classifier. The performance differences between propensity score matching and multimodal VAE then represent the effects of the VAE reconstruction objective and latent invariance objectives. For additional baselines, we also compare against a random matching, where the samples are matched with equal weight within each perturbation as a sanity check. For datasets (1) and (2), we also compare against Gromov-Wasserstein OT (SCOT) `\citep{demetci2022scot}`{=latex} computed separately within each perturbation. SCOT uses OT directly by computing a cost function derived based on pairwise distances within each modality, thus learning a local description of the geometry which can be compared between modalities. For the CITE-seq dataset, we also compare against matching using a graph-linked VAE, scGLUE `\citep{cao2022multi}`{=latex}, where the graph is constructed from linking genes with the associated proteins.

#### Evaluation Metrics

We use the known ground truth matching to compute performance metrics on datasets (1) and (2). The trace and FOSCTTM `\citep{liu2019}`{=latex} measure how much weight \\(M\\) places on the true pairing. However, this is not necessarily indicative of downstream performance as similar, but not exact matches are penalized equally to wildly incorrect matches. For this reason, we also measure the latent MSE for dataset (1) and the performance of a CITE-seq gene–to–protein predictive model based on the learned matching for dataset (2). For more details, see  
efsec:eval.

## Experiment 1: Synthetic Interventional Images

**Data**   We followed the data generating process  
efeqn::dgp with a latent variable \\(Z\\) encoding the coordinates of two objects. Perturbations represent different do-interventions on the different dimensons of \\(Z\\). The difference between modalities corresponds to whether the objects are circular or square, and a fixed transformation of \\(Z\\), while the modality-specific noise \\(U\\) controls background distortions.

**Model and Evaluation**   We used a convolutional neural network adapted from `\citet{yang2021multi}`{=latex} as the encoder. We report two evaluation metrics: (1) the trace metric, and (2) the MSE between the matched and the true latents. The latent MSE metric does not penalize close neighbours of the true match (i.e. examples for which \\(\|z_i - z_i^*\|\\) is small) as heavily as the trace metric. These “near matches” will typically still be useful on downstream multimodal tasks.

**Results**   In Table <a href="#tab:alignment" data-reference-type="ref" data-reference="tab:alignment">1</a>, metrics are computed on a held out test set over 12 groups corresponding to interventions on the latent position, with approximately 1700 observations per group. A random matching, with weight \\(1/n\\), will hence have a trace metric of of \\(1/1700 \approx 0.588 \times 10^{-3}\\). This implies, for example, that the median performance of PS+OT is approximately 31 times that of random matching. On both metrics, we found that propensity scores matched with OT (PS + OT) consistently outperformed other matching methods on both metrics.

<div id="tab:alignment" markdown="1">

<table>
<caption>Alignment metrics results using synthetic interventional image dataset and CITE-seq data.</caption>
<thead>
<tr>
<th style="text-align: center;"></th>
<th colspan="2" style="text-align: center;"><strong>Synthetic Image Data</strong></th>
<th colspan="2" style="text-align: center;"><strong>CITE-seq Data</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span>2-3</span> (lr)<span>4-5</span></td>
<td style="text-align: center;"><strong>MSE (<span class="math inline">↓</span>)</strong></td>
<td style="text-align: center;"><strong>Trace (<span class="math inline">↑</span>)</strong></td>
<td style="text-align: center;"><strong>FOSCTTM (<span class="math inline">↓</span>)</strong></td>
<td style="text-align: center;"><strong>Trace (<span class="math inline">↑</span>)</strong></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"><strong>Med (Q1, Q3)</strong></td>
<td style="text-align: center;"><strong>Med (Q1, Q3) <span class="math inline">×10<sup>−3</sup></span></strong></td>
<td style="text-align: center;"><strong>Med (Q1, Q3)</strong></td>
<td style="text-align: center;"><strong>Med (Q1, Q3)</strong></td>
</tr>
<tr>
<td style="text-align: left;">PS+OT</td>
<td style="text-align: center;"><strong>0.0316</strong></td>
<td style="text-align: center;"><strong>18.329</strong></td>
<td style="text-align: center;"><strong>0.3049</strong></td>
<td style="text-align: center;"><strong>0.1163</strong></td>
</tr>
<tr>
<td style="text-align: right;"></td>
<td style="text-align: center;"><strong>(0.0300, 0.0330)</strong></td>
<td style="text-align: center;"><strong>(17.068, 18.987)</strong></td>
<td style="text-align: center;"><strong>(0.3008, 0.3078)</strong></td>
<td style="text-align: center;"><strong>(0.1093, 0.1250)</strong></td>
</tr>
<tr>
<td style="text-align: left;">VAE+OT</td>
<td style="text-align: center;">0.0324</td>
<td style="text-align: center;">7.733</td>
<td style="text-align: center;">0.3953</td>
<td style="text-align: center;">0.0814</td>
</tr>
<tr>
<td style="text-align: right;"></td>
<td style="text-align: center;">(0.0316, 0.0350)</td>
<td style="text-align: center;">(7.473, 7.794)</td>
<td style="text-align: center;">(0.3912, 0.4045)</td>
<td style="text-align: center;">(0.0777, 0.8895)</td>
</tr>
<tr>
<td style="text-align: left;">PS+SNN</td>
<td style="text-align: center;">0.0552</td>
<td style="text-align: center;">7.924</td>
<td style="text-align: center;">0.3126</td>
<td style="text-align: center;">0.0941</td>
</tr>
<tr>
<td style="text-align: right;"></td>
<td style="text-align: center;">(0.0530, 0.0558)</td>
<td style="text-align: center;">(7.569, 9.504)</td>
<td style="text-align: center;">(0.3121, 0.3160)</td>
<td style="text-align: center;">(0.0880, 0.0989)</td>
</tr>
<tr>
<td style="text-align: left;">VAE+SNN</td>
<td style="text-align: center;">0.0622</td>
<td style="text-align: center;">3.116</td>
<td style="text-align: center;">0.3816</td>
<td style="text-align: center;">0.0612</td>
</tr>
<tr>
<td style="text-align: right;"></td>
<td style="text-align: center;">(0.0571, 0.0676)</td>
<td style="text-align: center;">(2.818, 3.213)</td>
<td style="text-align: center;">(0.3760, 0.3822)</td>
<td style="text-align: center;">(0.0588, 0.0634)</td>
</tr>
<tr>
<td style="text-align: left;">SCOT</td>
<td style="text-align: center;">0.0354</td>
<td style="text-align: center;">0.5964</td>
<td style="text-align: center;">0.4596</td>
<td style="text-align: center;">0.0200</td>
</tr>
<tr>
<td style="text-align: left;">GLUE+SNN</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">0.4412</td>
<td style="text-align: center;">0.0362</td>
</tr>
<tr>
<td style="text-align: left;">GLUE+OT</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">0.5309</td>
<td style="text-align: center;">0.0323</td>
</tr>
<tr>
<td style="text-align: left;">Random</td>
<td style="text-align: center;">0.0709</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
</tr>
<tr>
<td style="text-align: right;"></td>
<td style="text-align: center;">(0.0707, 0.0714)</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

## Experiment 2: CITE-Seq Data

**Data**   We used the CITE-seq dataset from the NeurIPS 2021 Multimodal single-cell data integration competition `\citep{lance2022multimodal}`{=latex}, consisting of paired RNA-seq and surface level protein measurements over \\(45\\) cell types. In the absence of perturbations, we used the cell type as the observed label to classify and match within. Note the cell types are determined by consensus by pooling annotations from marker genes/proteins. In most cells, the annotations from each modality agreed, suggesting that the label is independent from the modality-specific noise. We used the first 200 principal components as the gene expression modality, and normalized (but otherwise raw) protein measurements as input.

**Model and Evaluation**   We used fully-connected MLPs as encoders. To assess matching, we report (1) the trace, and (2) the Fraction Of Samples Closer Than the True Match (FOSCTTM) (`\citep{demetci2022scot}`{=latex}, `\citep{liu2019}`{=latex}) (lower is better, 0.5 corresponds to random guessing). To evaluate against a downstream task, we also compared the performance of random and VAE matching procedures, as well as directly using the ground truth (\\(M_{ii} = 1\\)), on predicting protein levels from gene expression. We trained a 2-layer MLP (the same architecture for all matchings) with both MSE loss and the unbiased procedure as described in  
efsec::unbiased using pseudosamples sampled according to the matching matrix. We evaluated the predictive models against ground truth pairs by computing the prediction \\(R^2\\) (higher is better) on a held-out, unpermuted, test set.

**Results**   In Table <a href="#tab:alignment" data-reference-type="ref" data-reference="tab:alignment">1</a>, metrics are computed on a held-out test set averaged over 45 cell types with varying observation counts per group. While interpreting the average trace can be challenging due to group size variations, OT matching on PS consistently outperformed other methods both within and across groups. In these experiments, OT matching on PS was consistently the top performer, often followed by SNN matching on PS or OT matching on VAE embeddings.

We present downstream task performance in  
eftab:cross-modal-pred. Note that \\(R^2\\) is computed using the sample average across possibly multiple cell types, which explains why random matching within each cell type results in non-zero \\(R^2\\) (see  
efsec:eval). We found that PS + OT matching outperforms other methods on this task. Surprisingly, the PS + OT prediction model performed even better on average than training with the standard MSE loss on ground truth pairings (though confidence intervals overlap). This highlights the potential benefit of soft (OT) matching as a regularizer, beyond that of simply reconciling most likely pairs: the soft matching effectively averages over modality specific variation from samples with similar latent states in a manner analogous to data augmentation (with an unknown group action).

## Experiment 3: PerturbSeq and Single Cell Images

<div id="tab:cross-modal-pred" markdown="1">

<table>
<caption>Cross-modal prediction results using CITE-seq data and PerturbSeq/single cell image data including an out of distribution distance evaluation for PerturbSeq/single cell images.</caption>
<thead>
<tr>
<th style="text-align: center;"></th>
<th colspan="2" style="text-align: center;"><strong>CITE-seq Data</strong></th>
<th colspan="2" style="text-align: center;"><strong>PerturbSeq/Single Cell Image Data</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span>2-3</span> (lr)<span>4-5</span> <strong>Method</strong></td>
<td style="text-align: center;"><strong>MSE Loss</strong></td>
<td style="text-align: center;"><strong>Unbiased Loss</strong></td>
<td style="text-align: center;"><strong>In Distribution</strong></td>
<td style="text-align: center;"><strong>Out of Distribution</strong></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"><strong><span class="math inline"><em>R</em><sup>2</sup></span> Med (Q1, Q3) (<span class="math inline">↑</span>)</strong></td>
<td style="text-align: center;"><strong><span class="math inline"><em>R</em><sup>2</sup></span> Med (Q1, Q3) (<span class="math inline">↑</span>)</strong></td>
<td style="text-align: center;"><strong>KL Med (Q1, Q3) (<span class="math inline">↓</span>)</strong></td>
<td style="text-align: center;"><strong>KL (<span class="math inline">↓</span>)</strong></td>
</tr>
<tr>
<td style="text-align: left;">Random</td>
<td style="text-align: center;">0.138</td>
<td style="text-align: center;">0.173</td>
<td style="text-align: center;">58.806</td>
<td style="text-align: center;">51.310</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;">(0.137, 0.140)</td>
<td style="text-align: center;">(0.170, 0.173)</td>
<td style="text-align: center;">(58.771, 60.531)</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">VAE+OT</td>
<td style="text-align: center;">0.149</td>
<td style="text-align: center;">0.114</td>
<td style="text-align: center;">55.483</td>
<td style="text-align: center;">47.910</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;">(0.118, 0.172)</td>
<td style="text-align: center;">(0.079, 0.159)</td>
<td style="text-align: center;">(55.410, 56.994)</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>PS+OT</strong></td>
<td style="text-align: center;"><strong>0.217</strong></td>
<td style="text-align: center;"><strong>0.233</strong></td>
<td style="text-align: center;"><strong>50.967</strong></td>
<td style="text-align: center;"><strong>43.554</strong></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"><strong>(0.206, 0.223)</strong></td>
<td style="text-align: center;"><strong>(0.207, 0.250)</strong></td>
<td style="text-align: center;"><strong>(50.898, 52.457)</strong></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">True Pairs</td>
<td style="text-align: center;">0.224</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;">(0.223, 0.226)</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

**Data**   We collected PerturbSeq data (200 genes) and single-cell images of HUVEC cells with 24 gene perturbations and a control perturbation, resulting in 25 total labels across both modalities. As preprocessing, we embed the raw PerturbSeq counts into a 128-dimensional space using scVI `\citep{lopez2018deep}`{=latex} and the cell images into a 1024-dimensional space using a pre-trained Masked Autoencoder `\citep{he2022masked, kraus2023masked}`{=latex} to train our gene expression and image classifiers.

**Model and Evaluation**   We used a fully connected 2-layer MLP as the encoder for both PerturbSeq and cell image classifiers. Similarly to the CITE-seq dataset, we evaluated the matchings based on downstream prediction of gene expression from (embeddings of) images. We used the unbiased procedure to minimize the projected loss  
efeqn::loss and evaluated on two held-out sets, one consisting of in-distribution samples from the 25 perturbations the classifier was trained on, and an out-of-distribution set consisting of an extra perturbation not seen in training. In the absence of ground truth matching, we assessed three distributional metrics between the actual and predicted gene expression values within each perturbation: the L2 norm of the difference in means, the Kullback-Leibler (KL) divergence, and 1-Wasserstein distance (lower indicates better alignment). We report inverse cell-count weighted averages over each perturbation group. Each metric measures a slightly different aspect of fit—the L2 norm reports a first-order deviation, while the KL divergence is an empirical estimate of the deviation of the underlying predicted distribution, while the 1-Wasserstein distance measures deviations in terms of the empirical samples themselves.

Note that matching is performed using classifiers trained on scVI embeddings, but the cross-modal predictions are generated in the original log transformed gene expression space (i.e. we predicted actual observations, not embeddings). We also evaluated distance measures on an out-of-distribution gene perturbation that was not used in either the matching or training of the translation model.

**Results**   We present KL divergence values for in-distribution and out-of-distribution in  
eftab:cross-modal-pred.[^4] Additional metrics show similar patterns and can be found in  
efsec::suppresults. OT + PS matching consistently outperforms its VAE counterpart both on in-distribution and out-of-distribution metrics, supporting our findings on the CITE-seq data to the case where ground truth pairs are not available.

## Validation Monitor

As in our Perturb-seq and cell imaging example, the ground truth matching is typically unknown in real problems. It is hence desirable to have an observable proxy of the matching performance as a validation during hyperparameter tuning. Figure <a href="#fig:monitor" data-reference-type="ref" data-reference="fig:monitor">2</a> demonstrates that the propensity score validation loss (cross-entropy) empirically satisfies this role in our CITE-seq experiments, where lower validation loss corresponds to better matching performance, as if it were computed with the ground truth. By contrast, we found that the optimal VAE, in terms of matching, had higher validation loss. This empirically supports our intuition that the reconstruction loss minimization requires the VAE to capture modality specific information, i.e., the \\(U^{(e)}\\) variables, which hinders its matching performance.

<figure id="fig:monitor">
<img src="./figures/foscttm.png"" />
<figcaption>VAE and classifier validation metrics on the CITE-seq dataset. Notice that validation cross-entropy inversely tracks the ground truth matching metrics, and thus can be used as a proxy in practical settings where the ground truth is unknown. The same pattern does not hold for the VAE <span class="citation" data-cites="yang2021multi"></span>, which we suspect is because reconstruction is largely irrelevant for matching.</figcaption>
</figure>

# Limitations

While our framework is broadly applicable, two practical considerations remain.  First, accurate propensity-score estimation benefits from moderately sized labelled datasets; extremely small perturbation groups can lead to high-variance classifiers and, hence, less precise matchings.  Second, our current implementation explicitly handles two modalities; extending to three or more will primarily require additional engineering to manage memory during large-scale optimal-transport computations.  Addressing these points—perhaps through semi-supervised classifiers or block-sparse transport solvers—constitutes an exciting direction for future work.
# Conclusion

We have presented a simple yet powerful strategy for aligning unpaired multimodal datasets by matching propensity scores computed independently in each modality.  Because the required independence and injectivity conditions are typically met by modern measurement technologies, the method delivers rigorous guarantees without restricting real-world applicability.  Extensive experiments—from controlled synthetic scenes to cutting-edge single-cell perturbation assays—demonstrate that propensity-score optimal transport reliably surpasses competing alignment techniques and even outperforms models trained on true pairs when used for downstream prediction.  Given its minimal assumptions, negligible tuning overhead, and strong empirical performance, we expect this approach to become a standard component of future multimodal learning pipelines.
# Acknowledgements [sec:ack]

We are extremely grateful for the discussions with many external collaborators and colleagues at Recursion that lead to this work. The original ideas for this work stemed from conversations with Alex Tong with feedback from Yoshua Bengio at Mila. We received a lot of helpful feedback from all of our colleagues at Valence Labs, especially Berton Earnshaw and Ali Denton. The single cell image experiments are built on code originally written by Oren Kraus and his team.

# References [references]

<div class="thebibliography" markdown="1">

K. Ahuja, D. Mahajan, Y. Wang, and Y. Bengio Interventional causal representation learning In *ICML*, 2023. **Abstract:** Causal representation learning seeks to extract high-level latent factors from low-level sensory data. Most existing methods rely on observational data and structural assumptions (e.g., conditional independence) to identify the latent factors. However, interventional data is prevalent across applications. Can interventional data facilitate causal representation learning? We explore this question in this paper. The key observation is that interventional data often carries geometric signatures of the latent factors’ support (i.e. what values each latent can possibly take). For example, when the latent factors are causally connected, interventions can break the dependency between the intervened latents’ support and their ancestors’. Leveraging this fact, we prove that the latent causal factors can be identified up to permutation and scaling given data from perfect $do$ interventions. Moreover, we can achieve block affine identification, namely the estimated latent factors are only entangled with a few other latents if we have access to data from imperfect interventions. These results highlight the unique power of interventional data in causal representation learning; they can enable provable identification of latent factors without any assumptions about their distributions or dependency structure. (@ahuja2023interventional)

A. Almahairi, S. Rajeshwar, A. Sordoni, P. Bachman, and A. Courville Augmented cyclegan: Learning many-to-many mappings from unpaired data In *ICML*, 2018. **Abstract:** Learning inter-domain mappings from unpaired data can improve performance in structured prediction tasks, such as image segmentation, by reducing the need for paired data. CycleGAN was recently proposed for this problem, but critically assumes the underlying inter-domain mapping is approximately deterministic and one-to-one. This assumption renders the model ineffective for tasks requiring flexible, many-to-many mappings. We propose a new model, called Augmented CycleGAN, which learns many-to-many mappings between domains. We examine Augmented CycleGAN qualitatively and quantitatively on several image datasets. (@almahairi2018augmented)

M. Amodio and S. Krishnaswamy Magan: Aligning biological manifolds In *ICML*, 2018. **Abstract:** It is increasingly common in many types of natural and physical systems (especially biological systems) to have different types of measurements performed on the same underlying system. In such settings, it is important to align the manifolds arising from each measurement in order to integrate such data and gain an improved picture of the system. We tackle this problem using generative adversarial networks (GANs). Recently, GANs have been utilized to try to find correspondences between sets of samples. However, these GANs are not explicitly designed for proper alignment of manifolds. We present a new GAN called the Manifold-Aligning GAN (MAGAN) that aligns two manifolds such that related points in each measurement space are aligned together. We demonstrate applications of MAGAN in single-cell biology in integrating two different measurement types together. In our demonstrated examples, cells from the same tissue are measured with both genomic (single-cell RNA-sequencing) and proteomic (mass cytometry) technologies. We show that the MAGAN successfully aligns them such that known correlations between measured markers are improved compared to other recently proposed models. (@amodio2018magan)

S. Buchholz, G. Rajendran, E. Rosenfeld, B. Aragam, B. Schölkopf, and P. Ravikumar Learning linear causal representations from interventions under general nonlinear mixing In *NeurIPS*, 2023. **Abstract:** We study the problem of learning causal representations from unknown, latent interventions in a general setting, where the latent distribution is Gaussian but the mixing function is completely general. We prove strong identifiability results given unknown single-node interventions, i.e., without having access to the intervention targets. This generalizes prior works which have focused on weaker classes, such as linear maps or paired counterfactual data. This is also the first instance of causal identifiability from non-paired interventions for deep neural network embeddings. Our proof relies on carefully uncovering the high-dimensional geometric structure present in the data distribution after a non-linear density transformation, which we capture by analyzing quadratic forms of precision matrices of the latent distributions. Finally, we propose a contrastive algorithm to identify the latent variables in practice and evaluate its performance on various tasks. (@buchholz2023learning)

A. Butler, P. Hoffman, P. Smibert, E. Papalexi, and R. Satija Integrating single-cell transcriptomic data across different conditions, technologies, and species *Nature Biotechnology*, 36 (5): 411–420, 2018. **Abstract:** Single cell RNA-seq (scRNA-seq) has emerged as a transformative tool to discover and define cellular phenotypes. While computational scRNA-seq methods are currently well suited for experiments representing a single condition, technology, or species, analyzing multiple datasets simultaneously raises new challenges. In particular, traditional analytical workflows struggle to align subpopulations that are present across datasets, limiting the possibility for integrated or comparative analysis. Here, we introduce a new computational strategy for scRNA-seq alignment, utilizing common sources of variation to identify shared subpopulations between datasets as part of our R toolkit Seurat. We demonstrate our approach by aligning scRNA-seq datasets of PBMCs under resting and stimulated conditions, hematopoietic progenitors sequenced across two profiling technologies, and pancreatic cell ‘atlases’ generated from human and mouse islets. In each case, we learn distinct or transitional cell states jointly across datasets, and can identify subpopulations that could not be detected by analyzing datasets independently. We anticipate that these methods will serve not only to correct for batch or technology-dependent effects, but also to facilitate general comparisons of scRNA-seq datasets, potentially deepening our understanding of how distinct cell states respond to perturbation, disease, and evolution. Availability Installation instructions, documentation, and tutorials are available at http://www.satijalab.org/seurat (@butler2018integrating)

K. Cao, Q. Gong, Y. Hong, and L. Wan A unified computational framework for single-cell data integration with optimal transport *Nature Communications*, 13 (1): 7419, 2022. **Abstract:** Abstract Single-cell data integration can provide a comprehensive molecular view of cells. However, how to integrate heterogeneous single-cell multi-omics as well as spatially resolved transcriptomic data remains a major challenge. Here we introduce uniPort, a unified single-cell data integration framework that combines a coupled variational autoencoder (coupled-VAE) and minibatch unbalanced optimal transport (Minibatch-UOT). It leverages both highly variable common and dataset-specific genes for integration to handle the heterogeneity across datasets, and it is scalable to large-scale datasets. uniPort jointly embeds heterogeneous single-cell multi-omics datasets into a shared latent space. It can further construct a reference atlas for gene imputation across datasets. Meanwhile, uniPort provides a flexible label transfer framework to deconvolute heterogeneous spatial transcriptomic data using an optimal transport plan, instead of embedding latent space. We demonstrate the capability of uniPort by applying it to integrate a variety of datasets, including single-cell transcriptomics, chromatin accessibility, and spatially resolved transcriptomic data. (@cao2022unified)

Z.-J. Cao and G. Gao Multi-omics single-cell data integration and regulatory inference with graph-linked embedding *Nature Biotechnology*, 40 (10): 1458–1466, 2022. **Abstract:** Despite the emergence of experimental methods for simultaneous measurement of multiple omics modalities in single cells, most single-cell datasets include only one modality. A major obstacle in integrating omics data from multiple modalities is that different omics layers typically have distinct feature spaces. Here, we propose a computational framework called GLUE (graph-linked unified embedding), which bridges the gap by modeling regulatory interactions across omics layers explicitly. Systematic benchmarking demonstrated that GLUE is more accurate, robust and scalable than state-of-the-art tools for heterogeneous single-cell multi-omics data. We applied GLUE to various challenging tasks, including triple-omics integration, integrative regulatory inference and multi-omics human cell atlas construction over millions of cells, where GLUE was able to correct previous annotations. GLUE features a modular design that can be flexibly extended and enhanced for new analysis tasks. The full package is available online at https://github.com/gao-lab/GLUE . (@cao2022multi)

P. Demetci, R. Santorella, B. Sandstede, W. S. Noble, and R. Singh Scot: single-cell multi-omics alignment with optimal transport *Journal of Computational Biology*, 29 (1): 3–18, 2022. **Abstract:** Recent advances in sequencing technologies have allowed us to capture various aspects of the genome at single-cell resolution. However, with the exception of a few of co-assaying technologies, it is not possible to simultaneously apply different sequencing assays on the same single cell. In this scenario, computational integration of multi-omic measurements is crucial to enable joint analyses. This integration task is particularly challenging due to the lack of sample-wise or feature-wise correspondences. We present single-cell alignment with optimal transport (SCOT), an unsupervised algorithm that uses the Gromov-Wasserstein optimal transport to align single-cell multi-omics data sets. SCOT performs on par with the current state-of-the-art unsupervised alignment methods, is faster, and requires tuning of fewer hyperparameters. More importantly, SCOT uses a self-tuning heuristic to guide hyperparameter selection based on the Gromov-Wasserstein distance. Thus, in the fully unsupervised setting, SCOT aligns single-cell data sets better than the existing methods without requiring any orthogonal correspondence information. (@demetci2022scot)

A. Dixit, O. Parnas, B. Li, J. Chen, C. P. Fulco, L. Jerby-Arnon, N. D. Marjanovic, D. Dionne, T. Burks, R. Raychowdhury, et al Perturb-seq: dissecting molecular circuits with scalable single-cell rna profiling of pooled genetic screens *cell*, 167 (7): 1853–1866, 2016. **Abstract:** Keywords Single-cell RNA-seq; pooled screen; CRISPR; epistasis; genetic interactions (@dixit2016perturb)

W. Falcon and PyTorch Lightning Team Pytorch lightning 2023. URL <https://www.pytorchlightning.ai>. **Abstract:** NewsRecLib is an open-source library based on Pytorch-Lightning and Hydra developed for training and evaluating neural news recommendation models. The foremost goals of NewsRecLib are to promote reproducible research and rigorous experimental evaluation by (i) providing a unified and highly configurable framework for exhaustive experimental studies and (ii) enabling a thorough analysis of the performance contribution of different model architecture components and training regimes. NewsRecLib is highly modular, allows specifying experiments in a single configuration file, and includes extensive logging facilities. Moreover, NewsRecLib provides out-of-the-box implementations of several prominent neural models, training methods, standard evaluation benchmarks, and evaluation metrics for news recommendation. (@pytorch_lightning)

M. M. Fay, O. Kraus, M. Victors, L. Arumugam, K. Vuggumudi, J. Urbanik, K. Hansen, S. Celik, N. Cernek, G. Jagannathan, J. Christensen, B. A. Earnshaw, I. S. Haque, and B. Mabey Rxrx3: Phenomics map of biology *bioRxiv*, 2023. . URL <https://www.biorxiv.org/content/early/2023/02/08/2023.02.07.527350>. **Abstract:** Abstract The combination of modern genetic perturbation techniques with high content screening has enabled genome-scale cell microscopy experiments that can be leveraged to construct maps of biology . These are built by processing microscopy images to produce readouts in unified and relatable representation space to capture known biological relationships and discover new ones. To further enable the scientific community to develop methods and insights from map-scale data, here we release RxRx3 , the first ever public high-content screening dataset combining genome-scale CRISPR knockouts with multiple-concentration screening of small molecules (a set of FDA approved and commercially available bioactive compounds). The dataset contains 6-channel fluorescent microscopy images and associated deep learning embeddings from over 2.2 million wells that span 17,063 CRISPR knockouts and 1,674 compounds at 8 doses each. RxRx3 is one of the largest collections of cellular screening data, and as far as we know, the largest generated consistently via a common experimental protocol within a single laboratory. Our goal in releasing RxRx3 is to demonstrate the benefits of generating consistent data, enable the development of the machine learning methods on this scale of data and to foster research, methods development, and collaboration. For more information about RxRx3 please visit RxRx.ai/rxrx3 (@Fay2023)

R. Flamary, N. Courty, A. Gramfort, M. Z. Alaya, A. Boisbunon, S. Chambon, L. Chapel, A. Corenflos, K. Fatras, N. Fournier, L. Gautheron, N. T. Gayraud, H. Janati, A. Rakotomamonjy, I. Redko, A. Rolet, A. Schutz, V. Seguy, D. J. Sutherland, R. Tavenard, A. Tong, and T. Vayer Pot: Python optimal transport *Journal of Machine Learning Research*, 22 (78): 1–8, 2021. **Abstract:** Optimal transport has recently been reintroduced to the machine learning community thanks in part to novel eﬃcient optimization procedures allowing for medium to large scale applications. We propose a Python toolbox that implements several key optimal transport ideas for the machine learning community. The toolbox contains implementations of a number of founding works of OT for machine learning such as Sinkhorn algorithm and Wasserstein barycenters, but also provides generic solvers that can be used for conducting novel fundamental research. This toolbox, named POT for Python Optimal Transport, is open source with an MIT license. (@pot)

A. Foster, Á. Vezér, C. A. Glastonbury, P. Creed, S. Abujudeh, and A. Sim Contrastive mixture of posteriors for counterfactual inference, data integration and fairness In *ICML*, 2022. **Abstract:** Learning meaningful representations of data that can address challenges such as batch effect correction and counterfactual inference is a central problem in many domains including computational biology. Adopting a Conditional VAE framework, we show that marginal independence between the representation and a condition variable plays a key role in both of these challenges. We propose the Contrastive Mixture of Posteriors (CoMP) method that uses a novel misalignment penalty defined in terms of mixtures of the variational posteriors to enforce this independence in latent space. We show that CoMP has attractive theoretical properties compared to previous approaches, and we prove counterfactual identifiability of CoMP under additional assumptions. We demonstrate state-of-the-art performance on a set of challenging tasks including aligning human tumour samples with cancer cell-lines, predicting transcriptome-level perturbation responses, and batch correction on single-cell RNA sequencing data. We also find parallels to fair representation learning and demonstrate that CoMP is competitive on a common task in the field. (@Foster2022)

A. Genevay, G. Peyré, and M. Cuturi Learning generative models with sinkhorn divergences In *AISTATS*, 2018. **Abstract:** The ability to compare two degenerate probability distributions (i.e. two probability distributions supported on two distinct low-dimensional manifolds living in a much higher-dimensional space) is a crucial problem arising in the estimation of generative models for high-dimensional observations such as those arising in computer vision or natural language. It is known that optimal transport metrics can represent a cure for this problem, since they were specifically designed as an alternative to information divergences to handle such problematic scenarios. Unfortunately, training generative machines using OT raises formidable computational and statistical challenges, because of (i) the computational burden of evaluating OT losses, (ii) the instability and lack of smoothness of these losses, (iii) the difficulty to estimate robustly these losses and their gradients in high dimension. This paper presents the first tractable computational method to train large scale generative models using an optimal transport loss, and tackles these three issues by relying on two key ideas: (a) entropic smoothing, which turns the original OT loss into one that can be computed using Sinkhorn fixed point iterations; (b) algorithmic (automatic) differentiation of these iterations. These two approximations result in a robust and differentiable approximation of the OT loss with streamlined GPU execution. Entropic smoothing generates a family of losses interpolating between Wasserstein (OT) and Maximum Mean Discrepancy (MMD), thus allowing to find a sweet spot leveraging the geometry of OT and the favorable high-dimensional sample complexity of MMD which comes with unbiased gradient estimates. The resulting computational architecture complements nicely standard deep network generative models by a stack of extra layers implementing the loss function. (@genevay2018learning)

F. Gossi, P. Pati, P. Chouvardas, A. L. Martinelli, M. Kruithof-de Julio, and M. A. Rapsomaniki Matching single cells across modalities with contrastive learning and optimal transport *Briefings in Bioinformatics*, 24 (3), 2023. **Abstract:** Abstract Understanding the interactions between the biomolecules that govern cellular behaviors remains an emergent question in biology. Recent advances in single-cell technologies have enabled the simultaneous quantification of multiple biomolecules in the same cell, opening new avenues for understanding cellular complexity and heterogeneity. Still, the resulting multimodal single-cell datasets present unique challenges arising from the high dimensionality and multiple sources of acquisition noise. Computational methods able to match cells across different modalities offer an appealing alternative towards this goal. In this work, we propose MatchCLOT, a novel method for modality matching inspired by recent promising developments in contrastive learning and optimal transport. MatchCLOT uses contrastive learning to learn a common representation between two modalities and applies entropic optimal transport as an approximate maximum weight bipartite matching algorithm. Our model obtains state-of-the-art performance on two curated benchmarking datasets and an independent test dataset, improving the top scoring method by 26.1% while preserving the underlying biological structure of the multimodal data. Importantly, MatchCLOT offers high gains in computational time and memory that, in contrast to existing methods, allows it to scale well with the number of cells. As single-cell datasets become increasingly large, MatchCLOT offers an accurate and efficient solution to the problem of modality matching. (@gossi2023matching)

L. Gresele, P. K. Rubenstein, A. Mehrjou, F. Locatello, and B. Schölkopf The incomplete rosetta stone problem: Identifiability results for multi-view nonlinear ica In *UAI*, 2020. **Abstract:** We consider the problem of recovering a common latent source with independent components from multiple views. This applies to settings in which a variable is measured with multiple experimental modalities, and where the goal is to synthesize the disparate measurements into a single unified representation. We consider the case that the observed views are a nonlinear mixing of component-wise corruptions of the sources. When the views are considered separately, this reduces to nonlinear Independent Component Analysis (ICA) for which it is provably impossible to undo the mixing. We present novel identifiability proofs that this is possible when the multiple views are considered jointly, showing that the mixing can theoretically be undone using function approximators such as deep neural networks. In contrast to known identifiability results for nonlinear ICA, we prove that independent latent sources with arbitrary mixing can be recovered as long as multiple, sufficiently different noisy views are available. (@gresele2020incomplete)

M. Gutmann and A. Hyvärinen Noise-contrastive estimation: A new estimation principle for unnormalized statistical models In *AISTATS*, 2010. **Abstract:** We present a new estimation principle for parameterized statistical models. The idea is to perform nonlinear logistic regression to discriminate between the observed data and some artificially generated noise, using the model log-density function in the regression nonlinearity. We show that this leads to a consistent (convergent) estimator of the parameters, and analyze the asymptotic variance. In particular, the method is shown to directly work for unnormalized models, i.e. models where the density function does not integrate to one. The normalization constant can be estimated just like any other parameter. For a tractable ICA model, we compare the method with other estimation methods that can be used to learn unnormalized models, including score matching, contrastive divergence, and maximum-likelihood where the normalization constant is estimated with importance sampling. Simulations show that noise-contrastive estimation offers the best trade-off between computational and statistical efficiency. The method is then applied to the modeling of natural images: We show that the method can successfully estimate a large-scale two-layer model and a Markov random field. (@gutmann10a)

J. Hartford, G. Lewis, K. Leyton-Brown, and M. Taddy Deep iv: A flexible approach for counterfactual prediction In *ICML*, 2017. **Abstract:** Counterfactual prediction requires understanding causal relationships between so-called treatment and outcome variables. This paper provides a recipe for augmenting deep learning methods to accurately characterize such relationships in the presence of instrument variables (IVs)—sources of treatment randomization that are conditionally independent from the outcomes. Our IV specification resolves into two prediction tasks that can be solved with deep neural nets: a first-stage network for treatment prediction and a second-stage network whose loss function involves integration over the conditional treatment distribution. This Deep IV framework allows us to take advantage of off-the-shelf supervised learning techniques to estimate causal effects by adapting the loss function. Experiments show that it outperforms existing machine learning approaches. (@hartford2017deep)

K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick Masked autoencoders are scalable vision learners In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 16000–16009, 2022. **Abstract:** This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3× or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior. (@he2022masked)

I. Khemakhem, D. P. Kingma, R. P. Monti, and A. Hyvärinen Variational autoencoders and nonlinear ICA: A unifying framework In *AISTATS*, 2020. **Abstract:** The framework of variational autoencoders allows us to efficiently learn deep latent-variable models, such that the model’s marginal distribution over observed variables fits the data. Often, we’re interested in going a step further, and want to approximate the true joint distribution over observed and latent variables, including the true prior and posterior distributions over latent variables. This is known to be generally impossible due to unidentifiability of the model. We address this issue by showing that for a broad family of deep latent-variable models, identification of the true joint distribution over observed and latent variables is actually possible up to very simple transformations, thus achieving a principled and powerful form of disentanglement. Our result requires a factorized prior distribution over the latent variables that is conditioned on an additionally observed variable, such as a class label or almost any other observation. We build on recent developments in nonlinear ICA, which we extend to the case with noisy, undercomplete or discrete observations, integrated in a maximum likelihood framework. The result also trivially contains identifiable flow-based generative models as a special case. (@Khem2020a)

I. Korsunsky, N. Millard, J. Fan, K. Slowikowski, F. Zhang, K. Wei, Y. Baglaenko, M. Brenner, P.-r. Loh, and S. Raychaudhuri Fast, sensitive and accurate integration of single-cell data with harmony *Nature Methods*, 16 (12): 1289–1296, 2019. **Abstract:** Abstract The rapidly emerging diversity of single cell RNAseq datasets allows us to characterize the transcriptional behavior of cell types across a wide variety of biological and clinical conditions. With this comprehensive breadth comes a major analytical challenge. The same cell type across tissues, from different donors, or in different disease states, may appear to express different genes. A joint analysis of multiple datasets requires the integration of cells across diverse conditions. This is particularly challenging when datasets are assayed with different technologies in which real biological differences are interspersed with technical differences. We present Harmony, an algorithm that projects cells into a shared embedding in which cells group by cell type rather than dataset-specific conditions. Unlike available single-cell integration methods, Harmony can simultaneously account for multiple experimental and biological factors. We develop objective metrics to evaluate the quality of data integration. In four separate analyses, we demonstrate the superior performance of Harmony to four single-cell-specific integration algorithms. Moreover, we show that Harmony requires dramatically fewer computational resources. It is the only available algorithm that makes the integration of ∼ 10 6 cells feasible on a personal computer. We demonstrate that Harmony identifies both broad populations and fine-grained subpopulations of PBMCs from datasets with large experimental differences. In a meta-analysis of 14,746 cells from 5 studies of human pancreatic islet cells, Harmony accounts for variation among technologies and donors to successfully align several rare subpopulations. In the resulting integrated embedding, we identify a previously unidentified population of potentially dysfunctional alpha islet cells, enriched for genes active in the Endoplasmic Reticulum (ER) stress response. The abundance of these alpha cells correlates across donors with the proportion of dysfunctional beta cells also enriched in ER stress response genes. Harmony is a fast and flexible general purpose integration algorithm that enables the identification of shared fine-grained subpopulations across a variety of experimental and biological conditions. (@korsunsky2019fast)

O. Kraus, K. Kenyon-Dean, S. Saberian, M. Fallah, P. McLean, J. Leung, V. Sharma, A. Khan, J. Balakrishnan, S. Celik, et al Masked autoencoders are scalable learners of cellular morphology *arXiv preprint arXiv:2309.16064*, 2023. **Abstract:** Inferring biological relationships from cellular phenotypes in high-content microscopy screens provides significant opportunity and challenge in biological research. Prior results have shown that deep vision models can capture biological signal better than hand-crafted features. This work explores how self-supervised deep learning approaches scale when training larger models on larger microscopy datasets. Our results show that both CNN- and ViT-based masked autoencoders significantly outperform weakly supervised baselines. At the high-end of our scale, a ViT-L/8 trained on over 3.5-billion unique crops sampled from 93-million microscopy images achieves relative improvements as high as 28% over our best weakly supervised baseline at inferring known biological relationships curated from public databases. Relevant code and select models released with this work can be found at: https://github.com/recursionpharma/maes_microscopy. (@kraus2023masked)

C. Lance, M. D. Luecken, D. B. Burkhardt, R. Cannoodt, P. Rautenstrauch, A. Laddach, A. Ubingazhibov, Z.-J. Cao, K. Deng, S. Khan, et al Multimodal single cell data integration challenge: Results and lessons learned In *NeurIPS 2021 Competitions and Demonstrations Track*, pages 162–176, 2022. **Abstract:** Abstract Biology has become a data-intensive science. Recent technological advances in single-cell genomics have enabled the measurement of multiple facets of cellular state, producing datasets with millions of single-cell observations. While these data hold great promise for understanding molecular mechanisms in health and disease, analysis challenges arising from sparsity, technical and biological variability, and high dimensionality of the data hinder the derivation of such mechanistic insights. To promote the innovation of algorithms for analysis of multimodal single-cell data, we organized a competition at NeurIPS 2021 applying the Common Task Framework to multimodal single-cell data integration. For this competition we generated the first multimodal benchmarking dataset for single-cell biology and defined three tasks in this domain: prediction of missing modalities, aligning modalities, and learning a joint representation across modalities. We further specified evaluation metrics and developed a cloud-based algorithm evaluation pipeline. Using this setup, 280 competitors submitted over 2600 proposed solutions within a 3 month period, showcasing substantial innovation especially in the modality alignment task. Here, we present the results, describe trends of well performing approaches, and discuss challenges associated with running the competition. (@lance2022multimodal)

J. Liu, Y. Huang, R. Singh, J.-P. Vert, and W. S. Noble In *19th International Workshop on Algorithms in Bioinformatics (WABI 2019)*, 2019. **Abstract:** Abstract Many single-cell sequencing technologies are now available, but it is still difficult to apply multiple sequencing technologies to the same single cell. In this paper, we propose an unsupervised manifold alignment algorithm, MMD-MA, for integrating multiple measurements carried out on disjoint aliquots of a given population of cells. Effectively, MMD-MA performs an in silico co-assay by embedding cells measured in different ways into a learned latent space. In the MMD-MA algorithm, single-cell data points from multiple domains are aligned by optimizing an objective function with three components: (1) a maximum mean discrepancy (MMD) term to encourage the differently measured points to have similar distributions in the latent space, (2) a distortion term to preserve the structure of the data between the input space and the latent space, and (3) a penalty term to avoid collapse to a trivial solution. Notably, MMD-MA does not require any correspondence information across data modalities, either between the cells or between the features. Furthermore, MMD-MA’s weak distributional requirements for the domains to be aligned allow the algorithm to integrate heterogeneous types of single cell measures, such as gene expression, DNA accessibility, chromatin organization, methylation, and imaging data. We demonstrate the utility of MMD-MA in simulation experiments and using a real data set involving single-cell gene expression and methylation data. (@liu2019)

M.-Y. Liu, T. Breuel, and J. Kautz Unsupervised image-to-image translation networks *NeurIPS*, 2017. **Abstract:** Unsupervised image-to-image translation aims at learning a joint distribution of images in different domains by using images from the marginal distributions in individual domains. Since there exists an infinite set of joint distributions that can arrive the given marginal distributions, one could infer nothing about the joint distribution from the marginal distributions without additional assumptions. To address the problem, we make a shared-latent space assumption and propose an unsupervised image-to-image translation framework based on Coupled GANs. We compare the proposed framework with competing approaches and present high quality image translation results on various challenging unsupervised image translation tasks, including street scene image translation, animal image translation, and face image translation. We also apply the proposed framework to domain adaptation and achieve state-of-the-art performance on benchmark datasets. Code and additional results are available in https://github.com/mingyuliutw/unit . (@liu2017unsupervised)

R. Lopez, J. Regier, M. B. Cole, M. I. Jordan, and N. Yosef Deep generative modeling for single-cell transcriptomics *Nature methods*, 15 (12): 1053–1058, 2018. **Abstract:** As the number of single-cell transcriptomics datasets grows, the natural next step is to integrate the accumulating data to achieve a common ontology of cell types and states. However, it is not straightforward to compare gene expression levels across datasets and to automatically assign cell type labels in a new dataset based on existing annotations. In this manuscript, we demonstrate that our previously developed method, scVI, provides an effective and fully probabilistic approach for joint representation and analysis of scRNA-seq data, while accounting for uncertainty caused by biological and measurement noise. We also introduce single-cell ANnotation using Variational Inference (scANVI), a semi-supervised variant of scVI designed to leverage existing cell state annotations. We demonstrate that scVI and scANVI compare favorably to state-of-the-art methods for data integration and cell state annotation in terms of accuracy, scalability, and adaptability to challenging settings. In contrast to existing methods, scVI and scANVI integrate multiple datasets with a single generative model that can be directly used for downstream tasks, such as differential expression. Both methods are easily accessible through scvi-tools. (@lopez2018deep)

A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer Automatic differentiation in pytorch In *NIPS-W*, 2017. **Abstract:** The paper presents a simple and robust approach to an implementation of the hardening soil model into finite element calculations.The implementation of the return stress mapping exploits the automatic differentiation of tensor variables provided by the Py-Torch framework.The automatic differentiation allows for a succinct implementation despite the relatively complex structure of the nonlinear equations in the stress return algorithm.The presented approach is not limited to the hardening soil model.It can be utilised in the development and verification of other elasto-plastic constitutive models where expressing and maintaining the Jacobian matrix over different versions of a material model is time-consuming and error-prone. (@paszke2017automatic)

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay Scikit-learn: Machine learning in Python *Journal of Machine Learning Research*, 12: 2825–2830, 2011. **Abstract:** Scikit-learn is a Python module integrating a wide range of state-of-the-art machine learning algorithms for medium-scale supervised and unsupervised problems. This package focuses on bringing mach... (@scikit-learn)

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al Learning transferable visual models from natural language supervision In *ICML*, 2021. **Abstract:** State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at https://github.com/OpenAI/CLIP. (@radford2021learning)

D. B. Rubin Estimating causal effects of treatments in randomized and nonrandomized studies *Journal of Educational Psychology*, 66 (5): 688, 1974. **Abstract:** A discussion of matching, randomization, random sampling, and other methods of controlling extraneous variation is presented. The objective is to specify the benefits of randomization in estimating causal effects of treatments. The basic conclusion is that randomization should be employed whenever possible but that the use of carefully controlled nonrandomized data to estimate causal effects is a reasonable and necessary procedure in many cases. Recent psychological and educational literature has included extensive criticism of the use of nonrandomized studies to estimate causal effects of treatments (e.g., Campbell & Erlebacher, 1970). The implication in much of this literature is that only properly randomized experiments can lead to useful estimates of causal effects. If taken as applying to all fields of study, this position is untenable. Since the extensive use of randomized experiments is limited to the last half century,8 and in fact is not used in much scientific investigation today,4 one is led to the conclusion that most scientific truths have been established without using randomized experiments. In addition, most of us successfully determine the causal effects of many of our everyday actions, even interpersonal behaviors, without the benefit of randomization. Even if the position that causal effects of treatments can only be well established from randomized experiments is taken as applying only to the social sciences in which (@rubin1974estimating)

J. Ryu, R. Lopez, C. Bunne, and A. Regev Cross-modality matching and prediction of perturbation responses with labeled gromov-wasserstein optimal transport *arXiv preprint arXiv:2405.00838*, 2024. **Abstract:** It is now possible to conduct large scale perturbation screens with complex readout modalities, such as different molecular profiles or high content cell images. While these open the way for systematic dissection of causal cell circuits, integrated such data across screens to maximize our ability to predict circuits poses substantial computational challenges, which have not been addressed. Here, we extend two Gromov-Wasserstein Optimal Transport methods to incorporate the perturbation label for cross-modality alignment. The obtained alignment is then employed to train a predictive model that estimates cellular responses to perturbations observed with only one measurement modality. We validate our method for the tasks of cross-modality alignment and cross-modality prediction in a recent multi-modal single-cell perturbation dataset. Our approach opens the way to unified causal models of cell biology. (@ryu2024cross)

G. Schiebinger, J. Shu, M. Tabaka, B. Cleary, V. Subramanian, A. Solomon, J. Gould, S. Liu, S. Lin, P. Berube, et al Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming *Cell*, 176 (4): 928–943, 2019. **Abstract:** Schiebinger et al. Page 2 Cell. Author manuscript; available in PMC 2020 February 07. Author Manuscript Author Manuscript Author Manuscript Author ManuscriptIntroduction Waddington introduced two metaphors that shaped biological thinking about cellular differentiation: first, trains moving along branching railroad tracks and, later, marbles rolling through a developmental landscape ( Waddington, 1936 , 1957 ). Studying the actual landscapes, fates and trajectories associated with cellular differentiation and de- differentiation — in development, physiological responses, and reprogramming — requires us to answer questions such as: What classes of cells are present at each stage? What was their origin at earlier stages? What are their likely fates at later stages? What regulatory programs control their dynamics? Approaches based on bulk analysis of cell populations are not well suited to address these questions, because they do not provide general solutions to two challenges: discovering cell classes in a population and tracing the development of each class. The first challenge has been largely solved by the advent of single-cell RNA-Seq (scRNA- seq) ( Tanay and Regev, 2017 ). The second remains a work-in-progress. Because scRNA-seq destroys cells in the course of recording their profiles, one cannot follow expression the same cell and its direct descendants across time. While various approaches can record information about cell lineage, they currently provide only very limited information about a cell’s state at earlier time points ( Kester and van Oudenaarden, 2018 ). Comprehensive studies of cell trajectories thus rely heavily on computational approaches to connect discrete ‘snapshots’ into continuous ‘movies.’ Pioneering work to infer trajectories (Saelens et al., 2018 ) has shed light on various biological systems, including whole- organism development ( Farrell et al., 2018 ; Wagner et al., 2018 ), but many important challenges remain. First, with few exceptions, most methods do not explicitly leverage temporal information (Table S6). Historically, most were designed to extract information about stationary processes, such as adult stem cell differentiation, in which all stages exist simultaneously. However, time-courses are becoming commonplace. Second, many methods model trajectories in terms of graph theory, which imposes strong constraints on the model, such as one-dimensional trajectories (“edges”) and zero-dimensional branch po (@schiebinger2019optimal)

M. Spivak *Calculus on Manifolds: a Modern Approach to Classical Theorems of Advanced Calculus* CRC press, 2018. **Abstract:** Functions on Euclidean Space \* Norm and inner Product \* Subsets of Euclidean Space \* Functions and Continuity Differentiation \* Basic Definitions \* Basic Theorems \* Partial Derivatives \* Inverse Functions \* Implicit Functions \* Notation Integration \* Basic Definitions \* Measure Zero and Content Zero \* Integrable Functions \* Fubinis Theorem \* Partitions of Unity \* Change of Variable Integration on Chains \* Algebraic Preliminaries \* Fields and Forms \* Geometric Preliminaries \* The Fundamental Theorem of Calculus Integration on Manifolds \* Manifolds \* Fields and Forms on Manifolds \* Stokes Theorem on Manifolds \* The Volume Element \* The Classical Theorems (@spivak2018)

C. Squires, A. Seigal, S. S. Bhate, and C. Uhler Linear causal disentanglement via interventions In *ICML*, 2023. **Abstract:** Causal disentanglement seeks a representation of data involving latent variables that relate to one another via a causal model. A representation is identifiable if both the latent model and the transformation from latent to observed variables are unique. In this paper, we study observed variables that are a linear transformation of a linear latent causal model. Data from interventions are necessary for identifiability: if one latent variable is missing an intervention, we show that there exist distinct models that cannot be distinguished. Conversely, we show that a single intervention on each latent variable is sufficient for identifiability. Our proof uses a generalization of the RQ decomposition of a matrix that replaces the usual orthogonal and upper triangular conditions with analogues depending on a partial order on the rows of the matrix, with partial order determined by a latent causal model. We corroborate our theoretical results with a method for causal disentanglement that accurately recovers a latent causal model. (@squires2023linear)

M. Stoeckius, C. Hafemeister, W. Stephenson, B. Houck-Loomis, P. K. Chattopadhyay, H. Swerdlow, R. Satija, and P. Smibert Simultaneous epitope and transcriptome measurement in single cells *Nature methods*, 14 (9): 865–868, 2017. **Abstract:** Recent high-throughput single-cell sequencing approaches have been transformative for understanding complex cell populations, but are unable to provide additional phenotypic information, such as protein levels of cell-surface markers. Using oligonucleotide-labeled antibodies, we integrate measurements of cellular proteins and transcriptomes into an efficient, sequencing-based readout of single cells. This method is compatible with existing single-cell sequencing approaches and will readily scale as the throughput of these methods increase. The unbiased and extremely high-throughput nature of modern scRNA-seq approaches has proved invaluable for describing heterogeneous cell populations1–3. Prior to the use of single-cell genomics, detailed definitions of cellular states were routinely obtained via carefully curated panels of fluorescently labeled antibodies directed at cell surface proteins, which are often reliable indicators of cellular activity and function4. Recent studies5,6 have demonstrated the potential for coupling ‘index-sorting’ measurements with single-cell Users may view, print, copy, and download text and data-mine the content in such documents, for the purposes of academic research, subject always to the full Conditions of use: http://www.nature.com/authors/editorial_policies/license.html#terms \*Correspondence: mstoeckius@nygenome.org. ORCIDs: Marlon orcid.org/0000-0002-5658-029X Christoph orcid.org/0000-0001-6365-8254 William orcid.org/0000-0002-3779-417X Pratip orcid.org/0000-0002-5457-9666 Peter orcid.org/0000-0003-0772-1647 Rahul orcid.org/0000-0001-9448-8833 Harold orcid.org/0000-0002-9510-160X Brian orcid.org/0000-0002-1863-1199 Author contributions MS conceived and designed the study with input from BH-L, RS, HS and PS. MS performed all experiments. CH and RS designed and contributed the computational analyses. WS assisted with Drop-seq experiments. PC provided conceptual input on how to benchmark CITE-seq to flow cytometry and performed multiparameter flow cytometry analysis. MS, CH, RS and PS interpreted the data. MS and PS wrote the manuscript with input from all authors. Competing financial interests MS, BH-L and PS have filed a patent application based on this work. Data availability All raw data generated in this project has been deposited to the gene expression omnibus (GEO) with the accession code XXXXXX. HHS Public Access Author manuscript Nat Methods . Author manuscript; available in PMC 2018 January 31. Publi (@stoeckius2017simultaneous)

N. Sturma, C. Squires, M. Drton, and C. Uhler Unpaired multi-domain causal representation learning In *NeurIPS*, 2023. **Abstract:** The goal of causal representation learning is to find a representation of data that consists of causally related latent variables. We consider a setup where one has access to data from multiple domains that potentially share a causal representation. Crucially, observations in different domains are assumed to be unpaired, that is, we only observe the marginal distribution in each domain but not their joint distribution. In this paper, we give sufficient conditions for identifiability of the joint distribution and the shared causal graph in a linear setup. Identifiability holds if we can uniquely recover the joint distribution and the shared causal representation from the marginal distributions in each domain. We transform our identifiability results into a practical method to recover the shared latent causal graph. (@sturma2023unpaired)

A. Tong, J. Huang, G. Wolf, D. Van Dijk, and S. Krishnaswamy Trajectorynet: A dynamic optimal transport network for modeling cellular dynamics In *ICML*, 2020. **Abstract:** It is increasingly common to encounter data from dynamic processes captured by static cross-sectional measurements over time, particularly in biomedical settings. Recent attempts to model individual trajectories from this data use optimal transport to create pairwise matchings between time points. However, these methods cannot model continuous dynamics and non-linear paths that entities can take in these systems. To address this issue, we establish a link between continuous normalizing flows and dynamic optimal transport, that allows us to model the expected paths of points over time. Continuous normalizing flows are generally under constrained, as they are allowed to take an arbitrary path from the source to the target distribution. We present TrajectoryNet, which controls the continuous paths taken between distributions to produce dynamic optimal transport. We show how this is particularly applicable for studying cellular dynamics in data from single-cell RNA sequencing (scRNA-seq) technologies, and that TrajectoryNet improves upon recently proposed static optimal transport-based models that can be used for interpolating cellular distributions. (@tong2020trajectorynet)

A. van den Oord, Y. Li, and O. Vinyals Representation learning with contrastive predictive coding *arXiv preprint arXiv:1807.03748*, 2018. **Abstract:** While supervised learning has enabled great progress in many applications, unsupervised learning has not seen such widespread adoption, and remains an important and challenging endeavor for artificial intelligence. In this work, we propose a universal unsupervised learning approach to extract useful representations from high-dimensional data, which we call Contrastive Predictive Coding. The key insight of our model is to learn such representations by predicting the future in latent space by using powerful autoregressive models. We use a probabilistic contrastive loss which induces the latent space to capture information that is maximally useful to predict future samples. It also makes the model tractable by using negative sampling. While most prior work has focused on evaluating representations for a particular modality, we demonstrate that our approach is able to learn useful representations achieving strong performance on four distinct domains: speech, images, text and reinforcement learning in 3D environments. (@oord2018representation)

C. Villani *Optimal Transport: Old and New*, volume 338 Springer, 2009. (@villani2009)

J. von Kügelgen, M. Besserve, L. Wendong, L. Gresele, A. Kekić, E. Bareinboim, D. M. Blei, and B. Schölkopf Nonparametric identifiability of causal representations from unknown interventions In *NeurIPS*, 2023. **Abstract:** We study causal representation learning, the task of inferring latent causal variables and their causal relations from high-dimensional mixtures of the variables. Prior work relies on weak supervision, in the form of counterfactual pre- and post-intervention views or temporal structure; places restrictive assumptions, such as linearity, on the mixing function or latent causal model; or requires partial knowledge of the generative process, such as the causal graph or intervention targets. We instead consider the general setting in which both the causal model and the mixing function are nonparametric. The learning signal takes the form of multiple datasets, or environments, arising from unknown interventions in the underlying causal model. Our goal is to identify both the ground truth latents and their causal graph up to a set of ambiguities which we show to be irresolvable from interventional data. We study the fundamental setting of two causal variables and prove that the observational distribution and one perfect intervention per node suffice for identifiability, subject to a genericity condition. This condition rules out spurious solutions that involve fine-tuning of the intervened and observational distributions, mirroring similar conditions for nonlinear cause-effect inference. For an arbitrary number of variables, we show that at least one pair of distinct perfect interventional domains per node guarantees identifiability. Further, we demonstrate that the strengths of causal influences among the latent variables are preserved by all equivalent solutions, rendering the inferred representation appropriate for drawing causal conclusions from new data. Our study provides the first identifiability results for the general nonparametric setting with unknown interventions, and elucidates what is possible and impossible for causal representation learning without more direct supervision. (@von2023nonparametric)

Q. Xi and B. Bloem-Reddy Indeterminacy in generative models: Characterization and strong identifiability In *AISTATS*, 2023. **Abstract:** Most modern probabilistic generative models, such as the variational autoencoder (VAE), have certain indeterminacies that are unresolvable even with an infinite amount of data. Different tasks tolerate different indeterminacies, however recent applications have indicated the need for strongly identifiable models, in which an observation corresponds to a unique latent code. Progress has been made towards reducing model indeterminacies while maintaining flexibility, and recent work excludes many–but not all–indeterminacies. In this work, we motivate model-identifiability in terms of task-identifiability, then construct a theoretical framework for analyzing the indeterminacies of latent variable models, which enables their precise characterization in terms of the generator function and prior distribution spaces. We reveal that strong identifiability is possible even with highly flexible nonlinear generators, and give two such examples. One is a straightforward modification of iVAE (arXiv:1907.04809 \[stat.ML\]); the other uses triangular monotonic maps, leading to novel connections between optimal transport and identifiability. (@xi2023indeterminacy)

K. D. Yang, A. Belyaeva, S. Venkatachalapathy, K. Damodaran, A. Katcoff, A. Radhakrishnan, G. Shivashankar, and C. Uhler Multi-domain translation between single-cell imaging and sequencing data using autoencoders *Nature Communications*, 12 (1): 31, 2021. **Abstract:** Abstract The development of single-cell methods for capturing different data modalities including imaging and sequencing has revolutionized our ability to identify heterogeneous cell states. Different data modalities provide different perspectives on a population of cells, and their integration is critical for studying cellular heterogeneity and its function. While various methods have been proposed to integrate different sequencing data modalities, coupling imaging and sequencing has been an open challenge. We here present an approach for integrating vastly different modalities by learning a probabilistic coupling between the different data modalities using autoencoders to map to a shared latent space. We validate this approach by integrating single-cell RNA-seq and chromatin images to identify distinct subpopulations of human naive CD4+ T-cells that are poised for activation. Collectively, our approach provides a framework to integrate and translate between data modalities that cannot yet be measured within the same cell for diverse applications in biomedical discovery. (@yang2021multi)

J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros Unpaired image-to-image translation using cycle-consistent adversarial networks In *Proceedings of the IEEE international conference on computer vision*, pages 2223–2232, 2017. **Abstract:** Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G : X → Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F : Y → X and introduce a cycle consistency loss to push F(G(X)) ≈ X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach. (@zhu2017unpaired)

</div>

# Relaxing (A1)

#### Relaxing Assumption 1 [sec:relaxing]

Consider the propensity score \\[\begin{aligned}
    \pi(x^{(e, t)}) = P(t|X^{(e, t)} = x^{(e, t)}) 
\end{aligned}\\] where we do not necessarily require \\(U^{(e)} {\perp\!\!\!\perp}t \mid Z^{(t)}\\), and thus we obtain \\[\begin{aligned}
    \pi(x^{(1, t)}) = P(t| Z^{(t)} = z^{(t)}, U^{(1)} = u^{(1)} ) \neq  P(t| Z^{(t)} = z^{(t)}, U^{(2)} = u^{(2)} ) = \pi(x^{(2, t)}),
\end{aligned}\\] see the proof of  
efprop::ps for details.

Suppose that the two observed modalities are indeed generated by a shared \\(\{z_i\}_{i=1}^{n}\\), but where the indices of modality \\(2\\) are potentially permuted, and with values differing by modality specific information: \\[\begin{aligned}
    \{x^{(1,t)}_i = f^{(1)}(z_i, u_i^{(1)})\}_{i=1}^{n},  \{x^{(2,t)}_j = f^{(2)}(z_2, u_j^{(2)})\}_{j=1}^{n},
\end{aligned}\\] where \\(j = \pi(i)\\) denotes a permutation of the sample indices. Under (A1), we would be able to find some \\(j\\) such that \\(\pi(x^{(1,t)}_i) = \pi(x^{(2,t)}_j)\\) for each \\(i\\).

Matching via OT can allow us to relax (A1) in a very particular way. Consider the simple case where \\(t \in \{0,1\}\\), so that \\(\pi\\) can be written in a single dimension, e.g., \\(P(t = 1 | X^{(e, t)} = x^{(e, t)}) \in [0,1]\\). In this case, exact OT is equivalent to sorting \\(\pi(x^{(1,t)}_i)\\) and \\(\pi(x^{(2,t)}_j)\\), and matching the sorted versions 1-to-1. Under (A1), the sorted versions will be exactly equal. A relaxed version of (A1) that would still result in the correct ground truth matching is to assume that \\(t\\) affects \\(U^{(1)}\\) and \\(U^{(2)}\\) differently, but that the difference is order preserving, or monotone. Denote \\((\pi(x^{(1,t)}_i), \pi(x^{(2,t)}_i)) = (\pi_i^{(1)}, \pi_i^{(2)})\\) as the true pairing, noting that we use the same index \\(i\\). We require the following: \\[\begin{aligned}
    \label{eqn::monotonicity}
    (\pi_{i_1}^{(1)} - \pi_{i_2}^{(1)})(\pi_{i_1}^{(2)} - \pi_{i_2}^{(2)}) \geq 0, \quad \forall i_1, i_2 = 1, \dots, n.
\end{aligned}\\] This says that, even if \\(\pi_i^{(1)} \neq \pi_i^{(2)}\\), that their relative orderings will still coincide. Then, exact OT will still recover the ground truth matching. See  
effig::nocrossing for a visual example of this type of monotonicity. For example, suppose that \\(t\\) is a chemical perturbation of a cell, and thus \\(\pi_i^{(1)}\\), \\(\pi_j^{(2)}\\) can be seen as a measure of biological response to the perturbation, e.g., in a treated population, \\(\pi_{i_1} > \pi_{i_2}\\) indicates samples \\(i_1\\) had a stronger response than sample \\({i_2}\\), as perceived by the first modality indexed by \\(i\\). Then, this monotonocity states that we should see the same \\(\pi_{j_1} > \pi_{j_2}\\) in the other modality as well, if the samples \\(i_1\\) and \\(i_2\\) truly corresponded to \\(j_1\\) and \\(j_2\\).

## Cyclic Monotonicity [sec:cyclic]

We can see the monotonicity requirement  
efeqn::monotonicity as the monotonicity of the function with graph \\((\pi_i^{(1)}, \pi_i^{(2)}) \in
[0,1]^2\\). In higher dimensions, we require that the “graph” satisfies the following cyclic monotonicity property `\citep{villani2009}`{=latex}:

<div class="definition" markdown="1">

**Definition 3**. The collection \\(\{(\pi_i^{(1)}, \pi_{i}^{(2)})\}_{i=1}^{n}\\) is said to be \\(c\\)-cyclically monotone for some cost function \\(c\\), if for any \\(n = 1, \dots, N\\), and any subset of pairs \\((\pi_1^{(1)}, \pi_{1}^{(2)}), \dots, (\pi_n^{(1)}, \pi_{n}^{(2)})\\), we have \\[\begin{aligned}
        \sum_{n=1}^{N} c(\pi_n^{(1)}, \pi_{n}^{(2)}) \leq \sum_{n=1}^{N} c(\pi_n^{(1)}, \pi_{n+1}^{(2)}).
    
\end{aligned}\\] Importantly, we define \\(\pi_{n+1} = \pi_1\\), so that the sequence represents a cycle.

</div>

Note in our setting, the OT cost function is the Euclidean distance, \\(c(x, y) =
\|x - y\|_2\\). It is known that the OT solution must satisfy cyclic monotonicity. Thus, if the true pairing is uniquely cyclically monotone, we can recover it with OT. However, we are unaware of common violations of (A1) that would satisfy cyclic monotonicity.

<figure id="fig::nocrossing">
<img src="./figures/monotone.png"" />
<figcaption>OT matching allows for <span class="math inline"><em>t</em></span> to have different effects on the modality specific information, here <span class="math inline"><em>u</em><sub><em>i</em></sub><sup>(1)</sup></span> and <span class="math inline"><em>u</em><sub><em>i</em></sub><sup>(2)</sup></span>, as long as they can be written as transformations that preserve the relative order within modalities. Exact OT in 1-d always matches according to the relative ordering, and thus exhibits this type of “no crossing” behaviour shown in the figure on the left. The figure on the right shows a case where we would fail to correctly match across modalities because of the crossing shown in orange.</figcaption>
</figure>

# Shared Nearest Neighbours Matching [sec:shared-nn]

Using the propensity score distance, we can compute nearest neighbours both within and between the two modalities. We follow `\cite{cao2022multi}`{=latex} and compute the normalized shared nearest neighbours (SNN) between each pair of observations as the entry of the matching matrix. For each pair of observations \\((\pi_i^{(1)}, \pi_j^{(2)})\\), we define four sets:

- \\(\texttt{11}_{ij}\\): the k nearest neighbours of \\(\pi_i^{(1)}\\) amongst \\(\{\pi_i^{(1)}\}_{i=1}^{n_1}\\). \\(\pi_i^{(1)}\\) is considered a neighbour of itself.

- \\(\texttt{12}_{ij}\\): the k nearest neighbours of \\(\pi_j^{(2)}\\) amongst \\(\{\pi_i^{(1)}\}_{i=1}^{n_1}\\).

- \\(\texttt{21}_{ij}\\): the k nearest neighbours of \\(\pi_i^{(1)}\\) amongst \\(\{\pi_j^{(2)}\}_{j=1}^{n_2}\\).

- \\(\texttt{22}_{ij}\\): the k nearest neighbours of \\(\pi_j^{(2)}\\) amongst \\(\{\pi_j^{(2)}\}_{j=1}^{n_2}\\). \\(\pi_j^{(2)}\\) is considered a neighbour of itself.

Intuitively, if \\(\pi_i^{(1)}\\) and \\(\pi_j^{(2)}\\) correspond to the same underlying propensity score, their nearest neighbours amongst observations from each modality should be the same. This is measured as a set difference between \\(\texttt{11}_{ij}\\) and \\(\texttt{12}_{ij}\\), and likewise for \\(\texttt{21}_{ij}\\) and \\(\texttt{22}_{ij}\\). Then, a modified Jaccard index is computed as follows. Define \\[\begin{aligned}
    J_{ij} = |\texttt{11}_{ij} \cap \texttt{12}_{ij}| + |\texttt{21}_{ij} \cap \texttt{22}_{ij}|,
\end{aligned}\\] the sum of the number of shared neighbours measured in each modality. Then, we compute the following Jaccard distance to populate the unnormalized matching matrix: \\[\begin{aligned}
    \tilde{M}_{ij} = \frac{J_{ij}}{4k - J_{ij}},
\end{aligned}\\] where notice that \\(4k = |\texttt{11}_{ij}| + |\texttt{12}_{ij}| + |\texttt{21}_{ij}| + |\texttt{22}_{ij}|\\), since each set contains \\(k\\) distinct neighbours, and thus \\(0 \leq \tilde{M}_{ij} \leq 1\\), as with the standard Jaccard index. Then, we normalize each row to produce the final matching matrix: \\[\begin{aligned}
    M_{ij} = \frac{\tilde{M}_{ij}}{\sum_{i = 1}^{n_1} \tilde{M}_{ij}}.
\end{aligned}\\] Note \\(M_{ij}\\) is always well defined because \\(\pi_i^{(1)}\\) and \\(\pi_j^{(2)}\\) are always considered neighbours of themselves.

<div class="lemma" markdown="1">

**Lemma 4**. *\\(\tilde{M}_{ij}\\) has at least one non-zero entry in each of its rows and columns for any number of neighbours \\(k \geq 1\\).*

</div>

<div class="proof" markdown="1">

*Proof.* We prove that \\(J_{ij} > 0\\) for at least one \\(j\\) in each \\(i\\), which is equivalent to \\(\tilde{M}_{ij} > 0\\). Fix an arbitrary \\(i\\). \\(\texttt{21}_{ij}\\) by definition is the same set for every \\(j\\). By the assumption of \\(k \geq 1\\) it is non-empty, so there exists \\(\pi_{j^*}^{(2)} \in \texttt{21}_{ij}\\). Since \\(\pi_{j^*}^{(2)}\\) is a neighbour of itself, we have \\(\pi_{j^*}^{(2)} \in \texttt{22}_{ij^*}\\), showing that \\(J_{ij^*} > 0\\). The same reasoning applied to \\(\texttt{11}\\) and \\(\texttt{12}\\) also shows that \\(J_{ij}\\) for at least one \\(i\\) in each \\(j\\). ◻

</div>

# Proofs [sec:proof]

## Proof of efprop::ps

<div class="proof" markdown="1">

*Proof.* Let \\(x^{(e)}\\) denote the observed modality and \\(z, u^{(e)}\\) be the unique corresponding latent values. By injectivity, \\[\begin{aligned}
      \pi(x^{(e)}) &= P(t|X^{(e)} = x^{(e)})  \nonumber \\&=  P(t| Z = z, U^{(e)} = u^{(e)} ) \nonumber \\   &=  P(t|Z = z) = \pi(z),
\end{aligned}\\] for \\(e = 1, 2\\), since we assumed \\(U^{(e)} {\perp\!\!\!\perp}t \mid Z\\). Since this holds pointwise, it shows that \\(\pi(X^{(1}) = \pi(X^{(2)}) = \pi(Z)\\) as random variables. Now, a classical result of `\citet{rubin1974estimating}`{=latex} gives that \\(Z {\perp\!\!\!\perp}
t \mid \pi(Z)\\), and that for any other function \\(b\\) (a *balancing score*) such that \\(Z {\perp\!\!\!\perp}t \mid b(Z)\\), we have \\(\pi(Z) =
g(b(Z))\\). The first property written in information theoretic terms yields, \\[\begin{aligned}
    I(t, Z \mid \pi(Z)) = I(t, Z \mid \pi(X^{(e)})) = 0,
\end{aligned}\\] since \\(\pi(X^{(e)}) = \pi(Z^{(t)})\\) as random variables, as required. ◻

</div>

## Proof of efprop:dimensionality

<div class="proof" markdown="1">

*Proof.* In what follows, we write \\(\pi\\) to be the restriction to its domain where it is strictly positive. The \\(i\\)-th dimension of the propensity score can be written as \\[\begin{aligned}
    (\pi(z))_i = p(t = i|z) = \frac{p(z|t = i)p(t = i)}{\sum_{i=0}^T p(z|t=i) p(t = i)},
\end{aligned}\\] which, when restricted to be strictly positive, maps to the relative interior of the \\(T\\)-dimensional probability simplex. Consider the following transformation: \\[\begin{aligned}
    h(\pi(z))_i = \log\left( \frac{(\pi(z))_i}{(\pi(z))_0} \right) \\
    = \log(p(z|t=i)) - \log(p(z|t=0)) + C,
\end{aligned}\\] where \\(C = \log(p(t=i)) - \log(p(t=0))\\) is constant in \\(z\\), and that \\(h(\pi(z))_0 \equiv 0\\). Ignoring the constant first dimension, we can view \\(h\\) as an invertible map to \\(\mathbb{R}^{T}\\). Under this convention, the map \\(h
\circ \pi: \mathbb{R}^d \to \mathbb{R}^T\\) is smooth (\\(\log\\) is smooth, and the densities are smooth by assumption). Since it is smooth, it cannot be injective if \\(T < d\\) `\citep{spivak2018}`{=latex}. Finally, since \\(h\\) is bijective, this implies that \\(\pi\\) cannot be injective. ◻

</div>

# Experimental Details [sec:exptdetails]

## Evaluation Metrics [sec:eval]

### Known Ground Truth

In the synthetic image and CITE-seq datasets, a ground truth matching is known, and we can evaluate the quality of the synthetic matching directly against the truth. In these cases, the dataset sizes are necessarily balanced, so that \\(n = n_1 = n_2\\). In each case, we evaluate the quality of our \\(n \times n\\) matching matrix \\(M\\), which we compute within samples with the same \\(t\\). Our reported results are then averaged over each cluster. Note we randomize the order of the datasets before performing the matching to avoid pathologies.

**Trace Metric**   Assuming the sample indices correspond to the true matching, we can compute the average weight on correct matches, which is the normalized trace of \\(M\\): \\[\begin{aligned}
    \frac{1}{n}\text{Tr}(M) = \frac{1}{n}\sum_{i=1}^{n} M_{ii}.
\end{aligned}\\] As a baseline, notice that a uniformly random matching that assigns \\(M_{ij} =
1/n\\) for each cell yields \\(\text{Tr}(M) = 1\\) and hence will obtain a metric of \\(1/n\\). This metric however does not capture potential failure modes of matching. For example, exactly matching one sample, while adversarially matching dissimiliar samples for the remainder also yields a trace of \\(1/n\\), which is equal to that of a random matching.

**Latent MSE**   On the image dataset, we have access to the ground truth latent values that generated the images, \\(\mathbf{z} =
\{z_i\}_{i=1}^{n}\\). We compute matched latents as \\(M\mathbf{z}\\), the barycentric projection according to the matching matrix. Then, to evaluate the quality of the matching in terms of finding similar latents, we compute the MSE: \\[\begin{aligned}
    \text{MSE}(M) = \frac{1}{n}\| \mathbf{z} - M\mathbf{z} \|_{2}^2.
\end{aligned}\\]

**FOSCTTM**   We do not have access to ground truth latents in the CITE-seq dataset, so use the Fraction Of Samples Closer Than the True Match (FOSCTTM) `\citep{demetci2022scot, liu2019}`{=latex} as an alternative matching metric. First, we use \\(M\\) to project \\(\mathbf{x}^{(2)}
= \{x_j\}_{j=1}^{n}\\) to \\(\mathbf{x}^{(1)} = \{x_i\}_{i=1}^{n}\\) as \\(\hat{\mathbf{x}}^{(1)} = M \mathbf{x}^{(2)}\\). Then, we can compute a cross-modality distance as follows. For each point in \\(\hat{\mathbf{x}}^{(1)}\\), we compute the Euclidean distance to each point in \\(\mathbf{x}^{(1)}\\), and compute the fraction of samples in \\(\mathbf{x}^{(1)}\\) that are closer than the true match. We also repeat this for each point in \\(\mathbf{x}^{(1)}\\), computing the fraction of samples in \\(\hat{\mathbf{x}}^{(1)}\\) in this case. That is, assuming again that the given indices correspond to the true matching, we compute: \\[\begin{aligned}
    &\text{FOSCTTM}(M) = \nonumber \\  &\frac{1}{2n} \bigg[ \sum_{i=1}^{n} \bigg( \frac{1}{n} \sum_{j\neq i} \mathds{1}\{d(\hat{\mathbf{x}}_i^{(1)}, \mathbf{x}^{(1)}_j) < d(\hat{\mathbf{x}}_i^{(1)}, \mathbf{x}^{(1)}_i)\} \bigg)  \\  &+ \sum_{j=1}^{n} \bigg( \frac{1}{n} \sum_{i\neq j} \mathds{1}\{ d(\mathbf{x}_j^{(1)}, \hat{\mathbf{x}}_i^{(1)}) < d(\mathbf{x}^{(1)}_j, \hat{\mathbf{x}}_j^{(1)})\} \bigg) \bigg],
\end{aligned}\\] where notice that this evaluates \\(M\\) through the computation \\(\hat{\mathbf{x}}^{(1)} = M \mathbf{x}^{(2)}\\). As a baseline, we should expect a random matching, when distances between points are randomly distributed, to have an FOSCTTM of \\(0.5\\).

**Prediction Accuracy**   We also trained a cross-modality prediction (translation) model \\(f_{\theta, M}\\) to predict CITE-seq protein levels from gene expression based on matched pseudosamples. Let \\(\mathbf{x^{(1)}} = \{x_i\}\\), \\(\mathbf{x^{(2)}} = \{x_j\}\\) denote protein and gene expression, respectively. We trained a simple 2-layer MLP minimizing either the standard MSE, using pairs \\((x_i, \hat{x}_j)\\), \\(\hat{x}_j \sim M_{i\cdot}\\), or following the projected loss with unbiased estimates in  
efsec::unbiased. Each batch in general consists of samples from all \\(t\\), but the \\(\hat{x}_j\\) sampling step occurs within the perturbation. Let \\(\hat{\mathbf{x}}^{(1)}_{test} = \{f_{\theta, M}(x_{j})\}\\). We report the \\(R^2\\) on a randomly held-out test set of ground truth pairs (again, consisting of samples from all \\(t\\)), which is defined as the following: \\[\begin{aligned}
    R^2 (f_{\theta, M}) = \frac{MSE(\mathbf{x}^{(1)}_{test}, \hat{\mathbf{x}}^{(1)}_{test})}{MSE(\mathbf{x}^{(1)}_{test}, \bar{\mathbf{x}}^{(1)}_{test})},
\end{aligned}\\] where \\(\bar{\mathbf{x}}^{(1)}_{test}\\) is the naive mean (over all perturbations) estimator which acts as a baseline.

### Unknown Ground Truth

We train a cross-modality prediction model to predict gene expression from cell images based on matched pseudosamples in the same way as in CITE-seq, but only using the projected loss with unbiased estimates. Denote this model for a matching matrix \\(M\\) by \\(f_{\theta, M}\\).

Because we do not have access to ground truth pairs within each perturbation, we resort to distributional metrics. Let \\(\mathbf{x^{(1)}}_t = \{x_{i, t}\}_{i=1}^{n_{t1}}\\), \\(\mathbf{x^{(2)}}_t = \{x_{j, t}\}_{j=1}^{n_{t2}}\\) denote gene expression and cell images in a held out test set respectively in perturbation \\(t\\). Let \\(\hat{\mathbf{x}}^{(1)}_t = \{f_{\theta, M}(x_{j,t})\}_{j=1}^{n_{t2}}\\). We compute empirical versions of statistical divergences \\[\begin{aligned}
    D_t(f_{\theta, M}) := D(\mathbf{x}^{(1)}_t, \hat{\mathbf{x}}^{(1)}_t),
\end{aligned}\\] where \\(D\\) is either the L2 norm of the difference in empirical mean, empirical Kullback-Leibler divergence or 1-Wasserstein distance. We report these weighted averages of \\(D_t\\) over the perturbations \\(t\\) according to the number of samples in the modality of prediction interest.

## Models

In this section we describe experimental details pertaining to the propensity score and VAE `\citep{yang2021multi}`{=latex}. SCOT `\citep{demetci2022scot}`{=latex} and scGLUE `\citep{cao2022multi}`{=latex} are used according to tutorials and recommended default settings by the authors.

**Loss Functions**   The propensity score approach minimizes the standard cross-entropy loss for both modalities. The VAE includes, in addition to the standard ELBO loss (with parameter \\(\lambda\\) on the KL term), two cross-entropy losses based on classifiers from the latent space: one, weighted by a parameter \\(\alpha\\) to classify \\(t\\) as in the propensity score, and another, weighted by a parameter \\(\beta\\), that classifies which modality the latent point belongs to.

**Hyperparameters and Optimization**   We use the Adam optimizer with learning rate \\(0.0001\\) and one cycle learning rate scheduler. We follow `\citet{yang2021multi}`{=latex} and set \\(\alpha = 1\\), \\(\beta = 0.1\\), but found that \\(\lambda = 10^{-9}\\) (compared to \\(\lambda = 10^{-7}\\) in `\citet{yang2021multi}`{=latex}) resulted in better performance. We used batch size 256 in both instances and trained for either 100 epochs (image) or 250 epochs (CITE-seq).

For the VAE and classifiers of experiment 3, we use an Adam optimizer with learning 0.001 and weight decay 0.001 and max epoch of 100 (PerturbSeq) and 250 (single cell images) using batch sizes of 256 and 2048 correspondingly. We follow similar settings as `\citet{yang2021multi}`{=latex} and implement \\(\alpha=1\\) with \\(\lambda=10^{-9}\\), and since we do not have matched data, \\(\beta = 0\\). For the cross-modal prediction models in experiment 3, we use Stochastic Gradient Descent optimizer with learning rate 0.001 and weight decay 0.001 with max epochs 250 and batch size 256. We implement early stopping with delay of 50 epochs which we then checkpoint the last model to use for downstream tasks

**Architecture**   For the synthetic image dataset, we use an 5-layer convolutional network (channels \\(= 32, 54, 128, 256, 512\\)) with batch normalization and leaky ReLU activations, with linear heads for classification (propensity score and VAE) and posterior mean and variance estimation (VAE). For the VAE, the decoder consists of convolutional transpose layers that reverse those of the encoder.

For the CITE-seq dataset, we use a 5-layer MLP with constant hidden dimension \\(1024\\), with batch normalization and ReLU activations (adapted from the fully connected VAE in `\citet{yang2021multi}`{=latex}) as both the encoder and VAE decoder. We use the same architecture for both modalities, RNA-seq (as we process the top 200 PCs) and protein.

For the PerturbSeq classifier encoder, we use a 2-layer MLP architecture. Each layer consists of a linear layer with an output feature dimension of 64, followed by Rectified Linear Unit (ReLU) activation, Batch Normalization, and dropout (p=0.1). A final layer with Leaky ReLU activation that brings dimensionality to 128 before feeding into a linear classification head with an output feature dimension of 25.

For the single-cell image encoder classifier, we use a proprietary Masked Autoencoder `\citep{kraus2023masked}`{=latex} to generate 1024-dimensional embeddings. Subsequently, a 2-layer MLP is trained on these embeddings. Each MLP layer has a linear layer, Batch Normalization, and Leaky ReLU activation. The output feature dimensions of the linear layers are 512 and 256, respectively, and the latent dimension remains at 1024 before entering a linear classification head with an output feature dimension of 25.

**Optimal Transport**   We used POT `\citep{pot}`{=latex} to solve the entropic OT problem, using the log-sinkhorn solver, with regularization strength \\(\gamma =
0.05\\).

## Data

**Synthetic Data**   We follow the data generating process  
efeqn::dgp to generate coloured scenes of two simple objects (circles, or squares) in various orientations and with various backgrounds. The position of the objects are encoded in the latent variable \\(z\\), which is perturbed by a do-intervention (setting to a fixed value) randomly sampled for each \\(t\\). Each object has an \\(x\\) and \\(y\\) coordinate, leading to a \\(4\\)-dimensional \\(z\\), for which we consider \\(3\\) separate interventions each, leading to \\(12\\) different settings. The modality then corresponds to whether the objects are circular or square, and a fixed transformation of \\(z\\), while the modality-specific noise \\(U\\) controls background distortions. Scenes are generated using a rendering engine from PyGame as \\(f^{(e)}\\). Example images are given in  
effig::synimg.

<figure id="fig::synimg">
<img src="./figures/synthetic_img.png"" style="width:90.0%" />
<figcaption>Example pair of synthetic images with the same underlying <span class="math inline"><em>z</em></span>.</figcaption>
</figure>

**CITE-seq Data**   We also use the CITE-seq dataset from `\citet{lance2022multimodal}`{=latex} as a real-world benchmark (obtained from GEO accession GSE194122). These consist of paired RNA-seq and surface level protein measurements, and their cell type annotations over \\(45\\) different cell types. We used scanpy, a standard bioinformatics package, to perform PCA dimension reduction on RNA-seq by taking the first 200 principal components. The protein measurements (134-dimensional) was processed in raw form. For more details, see `\citet{lance2022multimodal}`{=latex}.

**PerturbSeq and Single Cell Image Data**   We collect single-cell PerturbSeq data (200 genes) and single-cell painting images in HUVEC cells with 24 gene perturbations and a control perturbation, resulting in 25 labels for matching across both modalities. The target gene perturbations are selected based on the 24 genes with the highest number of cells affected by the CRISPR guide RNAs targeting those genes. The PerturbSeq data is filtered to include the top 200 genes with the highest mean count, then normalized and log-transformed. The single-cell painting images are derived from multi-cell images, with each single-cell nucleus centered within a 32x32 pixel box. We use scVI `\cite{lopez2018deep}`{=latex} to embed the raw PerturbSeq counts into a 128-dimensional space before training the gene expression classifier. Similarly, we train our image classifier using 1024-dimensional embeddings obtained from a pre-trained Masked Autoencoder `\cite{kraus2023masked, he2022masked}`{=latex}. Following matching, we perform cross-modality translation from the single-cell embeddings to the transformed gene expression counts.

## Supplementary Results [sec::suppresults]

<div id="tab:exp-3-extra" markdown="1">

<table>
<caption>Wasserstein-1 and L2 norm distance values for PerturbSeq and single cell images experiments where distance is evaluated between cross-modal predictions and actual gene expression values.</caption>
<thead>
<tr>
<th style="text-align: center;"></th>
<th colspan="2" style="text-align: center;"><strong>In Distribution</strong></th>
<th colspan="2" style="text-align: center;"><strong>Out of Distribution</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span>2-3</span> (lr)<span>4-5</span></td>
<td style="text-align: center;"><strong>Wasserstein-1 (<span class="math inline">↓</span>)</strong></td>
<td style="text-align: center;"><strong>L2 Norm (<span class="math inline">↓</span>)</strong></td>
<td style="text-align: center;"><strong>Wasserstein-1 (<span class="math inline">↓</span>)</strong></td>
<td style="text-align: center;"><strong>L2 Norm (<span class="math inline">↓</span>)</strong></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"><strong>Med (Q1, Q3)</strong></td>
<td style="text-align: center;"><strong>Med (Q1, Q3)</strong></td>
<td style="text-align: center;"><strong></strong></td>
<td style="text-align: center;"><strong></strong></td>
</tr>
<tr>
<td style="text-align: left;">PS+OT</td>
<td style="text-align: center;"><strong>4.199</strong></td>
<td style="text-align: center;"><strong>3.280</strong></td>
<td style="text-align: center;"><strong>5.394</strong></td>
<td style="text-align: center;"><strong>7.219</strong></td>
</tr>
<tr>
<td style="text-align: right;"></td>
<td style="text-align: center;"><strong>(4.173, 4.226)</strong></td>
<td style="text-align: center;"><strong>(3.267, 3.284)</strong></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">VAE+OT</td>
<td style="text-align: center;">4.339</td>
<td style="text-align: center;">3.490</td>
<td style="text-align: center;">5.629</td>
<td style="text-align: center;">7.444</td>
</tr>
<tr>
<td style="text-align: right;"></td>
<td style="text-align: center;">(4.314, 4.348)</td>
<td style="text-align: center;">(3.486, 3.495)</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">Random</td>
<td style="text-align: center;">4.499</td>
<td style="text-align: center;">3.826</td>
<td style="text-align: center;">6.239</td>
<td style="text-align: center;">7.793</td>
</tr>
<tr>
<td style="text-align: right;"></td>
<td style="text-align: center;">(4.478, 4.525)</td>
<td style="text-align: center;">(3.823, 3.828)</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

[^1]: Work done during an internship at Valence Labs

[^2]: We will denote random variables by upper-case letters, and samples by their corresponding lower-case letter.

[^3]: Note that the injectivity is in the sense of \\(f\\) as a function of both \\(u\\) and \\(z\\), which allows observations that have a shared \\(z\\) but differ by their value in \\(u\\), and the function remains injective. For example, rotated images with the exact same content can have a shared \\(z\\), but remain injective due to the rotation being captured in \\(u\\).

[^4]: We computation of in-distribution metrics using random subsamples from the test set. The out-of-distribution metric was computed on a small dataset with a single perturbation and subsamples were not needed.
