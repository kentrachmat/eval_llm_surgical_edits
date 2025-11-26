# Bounds for the smallest eigenvalue of the NTK for arbitrary spherical data of arbitrary dimension

## Abstract

Bounds on the smallest eigenvalue of the neural tangent kernel (NTK) are a key ingredient in the analysis of neural network optimization and memorization. However, existing results require distributional assumptions on the data and are limited to a high-dimensional setting, where the input dimension \\(d_0\\) scales at least logarithmically in the number of samples \\(n\\). In this work we remove both of these requirements and instead provide bounds in terms of a measure of the collinearity of the data: notably these bounds hold with high probability even when \\(d_0\\) is held constant versus \\(n\\). We prove our results through a novel application of the hemisphere transform.

# Introduction

A popular approach for studying the optimization dynamics of neural networks is analyzing the neural tangent kernel (NTK), which corresponds to the Gram matrix obtained from the Jacobian of the network parametrization map `\citep{jacot2018neural}`{=latex}. When the network parameters are adjusted by gradient descent, the network function follows a kernel gradient descent in function space with respect to the NTK. By bounding the smallest eigenvalue of the NTK away from zero it is possible to obtain global convergence guarantees for gradient descent parameter optimization `\citep{du2018gradient,Oymak2019TowardMO}`{=latex} as well as results on generalization `\citep{pmlr-v97-arora19a,montanari2022interpolation}`{=latex} and data memorization capacity `\citep{montanari2022interpolation,nguyen2021tight,bombari2022memorization}`{=latex}. These key advances highlight the importance of deriving tight, quantitative bounds for the smallest eigenvalue of the NTK at initialization.

While initial breakthroughs on the convergence of gradient optimization in neural networks `\citep{NEURIPS2018_54fe976b, du2019gradient,
allenzhu2019convergence}`{=latex} required unrealistic conditions on the width of the layers, subsequent and substantive efforts have reduced the level of overparametrization required to ensure that the NTK is well conditioned at initialization `\citep{zou2019improved,
Oymak2019TowardMO}`{=latex}. In particular, `\citet{nguyenrelu,nguyen2021tight,banerjee2023neural}`{=latex} showed that layer width scaling linearly in the number of training samples \\(n\\) suffices to bound the smallest eigenvalue and `\cite{montanari2022interpolation, bombari2022memorization}`{=latex} obtained results for networks with sub-linear layer width and the minimum possible number of parameters \\(\tilde{\Omega}(n)\\) up to logarithmic factors. However, and as discussed in Section <a href="#sec:related-work" data-reference-type="ref" data-reference="sec:related-work">2</a>, the bounds provided in prior works require that the data is drawn from a distribution satisfying a Lipschitz concentration property, and only hold with high probability if the input dimension \\(d_0\\) scales as \\(\sqrt{n}\\) `\citep{bombari2022memorization}`{=latex} or \\(\text{polylog}(n)\\) `\citep{nguyen2021tight}`{=latex}. These existing results therefore require that the dimension of the data grows unbounded as the number of training samples \\(n\\) increases and as such there is a gap in our understanding of cases where the data is sampled from a fixed, or lower-dimensional space.

In this work we present new lower and upper bounds on the smallest eigenvalue of a randomly initialized, fully connected ReLU network: compared with prior work, our results hold for arbitrary data on a sphere of arbitrary dimension. Our techniques are novel and rely on the hemisphere transform as well as the addition formula for spherical harmonics.

We study neural networks denoted as functions \\(f: \mathbb{R}^{d_0} \times \mathcal{P} \rightarrow \mathbb{R}\\), where \\(\mathcal{P}\\) is an inner product space. To be clear, \\(f({\bm{x}}; {\bm{\theta}})\\) denotes the output of the network for a given input \\({\bm{x}}\in \mathbb{R}^{d_0}\\) and parameter choice \\({\bm{\theta}}\in \mathcal{P}\\). For brevity we occasionally write \\(f({\bm{x}})\\) in place of \\(f({\bm{x}}; {\bm{\theta}})\\) if the context is clear. We use \\(n\\) to denote the size of the training sample, \\(d_0\\) the dimension of the input features, \\(L\\) the network depth, \\(d_l\\) the width of the \\(l\\)th layer and \\(\sigma: \mathbb{R}\rightarrow \mathbb{R}\\) the ReLU activation function. Given \\(n\\) input data points \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{R}^{d_0}\\) we write \\({\bm{X}}= [{\bm{x}}_1, \cdots, {\bm{x}}_n] \in \mathbb{R}^{d \times n}\\) and define \\(F: \mathcal{P} \to \mathbb{R}^n\\) to be the evaluation of the network on these \\(n\\) data points as a function of the parameter \\(\bm{\theta}\\), \\[F(\bm{\theta}) = [f({\bm{x}}_1; {\bm{\theta}}), \cdots, f({\bm{x}}_n;{\bm{\theta}})]^T.\\] We define the neural tangent kernel (NTK) of \\(F\\) as \\[\label{eq:NTK-def}
{\bm{K}}({\bm{\theta}}) = (\nabla_{{\bm{\theta}}} F({\bm{\theta}}))^*(\nabla_{{\bm{\theta}}} F({\bm{\theta}})) \in \mathbb{R}^{n \times n} ,\\] where the gradient \\(\nabla\\) and adjoint \\(*\\) are taken with respect to the inner product on \\(\mathcal{P}\\) and the Euclidean inner product on \\(\mathbb{R}^n\\). More explicitly \\([{\bm{K}}({\bm{\theta}})]_{ik} = \langle \nabla_{{\bm{\theta}}} f({\bm{x}}_i; {\bm{\theta}}), \nabla_{{\bm{\theta}}}f({\bm{x}}_k; {\bm{\theta}}) \rangle\\). For convenience we write \\({\bm{K}}\\) in place of \\({\bm{K}}({\bm{\theta}})\\). We are concerned with the minimum eigenvalue \\(\lambda_{\min}({\bm{K}})\\), which depends both on the input data \\({\bm{X}}\\) and the parameter \\({\bm{\theta}}\\). We say the dataset \\({\bm{x}}_1, \cdots, {\bm{x}}_n\\) is *\\({\delta}\\)-separated* for \\(\delta \in (0,\sqrt{2}]\\) if \\(\min_{i\neq k} \min(\|{\bm{x}}_i - {\bm{x}}_k\|, \|{\bm{x}}_i + {\bm{x}}_k\|) \geq \delta\\), which is a measure of collinearity.

#### Main contributions.

Our results are for data that lies on a sphere and is \\(\delta\\)-separated for some \\(\delta \in (0,\sqrt{2}]\\). Unlike prior work we do not make any assumptions on the distribution from which the data is sampled, e.g., uniform on the sphere or Lipschitz concentrated, and we do not require the input dimension \\(d_0\\) to scale with the number of samples \\(n\\).

- In Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a> we consider shallow ReLU networks with input dimension \\(d_0\\) and hidden width \\(d_1\\) and prove that if \\(d_1 = \tilde{\Omega}(\|{\bm{X}}\|^2 d_0^3 \delta^{-2})\\) then with high probability \\(\lambda_{\min}({\bm{K}}) = \tilde{\Omega}(d_0^{-3}\delta^2)\\). Furthermore, defining \\(\delta' = \min_{i \neq k}\|{\bm{x}}_i - {\bm{x}}_k\|\\), we have \\(\lambda_{\min}({\bm{K}}) = O( \delta')\\).

- In Theorem <a href="#thm:deep-main" data-reference-type="ref" data-reference="thm:deep-main">[thm:deep-main]</a> we illustrate how our results for shallow networks can be extended to cover depth-\\(L\\) networks. In particular, if the layer widths satisfy a pyramidal condition, meaning \\(d_l \geq d_{l+1}\\) for \\(l \in     \{1,\cdots,L-1\}\\), \\(d_{L-1} \gtrsim 2^L \log(nL/ \epsilon)\\) and \\(d_1 = \tilde{\Omega}(nd_0^3 \delta^{-4} )\\), then \\(\lambda_{\min}({\bm{K}}) = \tilde{\Omega}(d_0^{-3}\delta^{4})\\) and \\(\lambda_{\min}({\bm{K}}) = O(L)\\) with high probability.

- Our results allow us to analyze the smallest eigenvalue of the NTK for data drawn from any distribution for which one can establish \\(\delta\\)-separation with high probability in terms of \\(d_0\\) and \\(n\\). For example, for shallow networks with data drawn uniformly from a sphere, in Corollary <a href="#corr:uniform" data-reference-type="ref" data-reference="corr:uniform">[corr:uniform]</a> we show that if \\(d_0d_1 = \tilde{\Omega}(n^{1 + 4/(d_0-1)})\\), then with high probability \\(\lambda_{\min}({\bm{K}}) = \tilde{O}\left(n^{-2/(d_0-1)} \right)\\) and \\(\lambda_{\min}({\bm{K}}) = \tilde{\Omega}\left(n^{-4/(d_0-1)} \right)\\). Moreover, this bound is tight up to logarithmic factors for \\(d_0=\Omega(\log(n))\\) matching prior findings for this regime.

The rest of this paper is structured as follows: in Section <a href="#sec:related-work" data-reference-type="ref" data-reference="sec:related-work">2</a> we provide a summary of related works and compare and contrast our results with the existing state of the art; in Section <a href="#section:shallow" data-reference-type="ref" data-reference="section:shallow">3</a> we present our results for shallow networks; finally in Section <a href="#sec:deep" data-reference-type="ref" data-reference="sec:deep">4</a> we extend our shallow results to the deep case.

#### Notations.

With regard to general points on notation we let \\([n] = \{1, 2, \cdots, n\}\\) denote the set of the first \\(n\\) positive integers. If \\({\bm{x}}\in \mathbb{R}^d\\) then we let \\([{\bm{x}}]_i\\) denote the \\(i\\)th entry of \\({\bm{x}}\\). If \\(f\\) and \\(g\\) are real-valued functions, we write \\(f \lesssim g\\) or \\(f = O(g)\\) when there exists an absolute constant \\(C\\) such that \\(f(x) \leq C g(x)\\) for all \\(x\\). Similarly, we write \\(f \gtrsim g\\) or \\(f = \Omega(g)\\) when there exists a constant \\(c\\) such that \\(f(x) \geq c g(x)\\) for all \\(x\\). We write \\(f \asymp g\\) when \\(f \lesssim g\\) and \\(f \gtrsim g\\) both hold. The notation \\(\tilde\Omega\\) hides logarithmic factors. Logarithms are generally considered to be in base \\(e\\), though in most settings the particular choice of base can be absorbed by a constant.

# Related work [sec:related-work]

#### Prior work on the NTK.

`\citet{jacot2018neural}`{=latex} highlight that the optimization dynamics of neural networks are controlled by the Gram matrix of the Jacobian of the network function, an object referred to as the NTK Gram matrix, or, as we refer to it here, simply the NTK. That work also shows that in the infinite-width limit the NTK converges in probability to a deterministic kernel. Of particular interest is the observation that in the infinite-width setting the network behaves like a linear model `\citep{Lee2019WideNN-SHORT}`{=latex}. Further, if a network is polynomially wide in the number of samples then the smallest eigenvalue of the NTK can be lower bounded in terms of the smallest eigenvalue of its infinite-width analog. As a result, assuming the latter is positive, global convergence guarantees for gradient descent can be obtained `\citep{du2019gradient,du2018gradient,allenzhu2019convergence, zou2019improved,Lee2019WideNN-SHORT,Oymak2019TowardMO,zou2020gradient,marco,nguyenrelu, banerjee2023neural}`{=latex}. The positive definiteness of the NTK is equivalent to the Jacobian having full rank, which can also be used to study the loss landscape `\citep{NEURIPS2020_b7ae8fec, LIU202285, karhadkar2023mildly}`{=latex}. Beyond the smallest eigenvalue, there is interest in characterizing the full spectrum of the NTK `\citep{uniform_sphere_data,geifman2020similarity,NEURIPS2020_572201a4,bietti2021deep, murray2023characterizing}`{=latex}, which has implications on the dynamics of the empirical risk `\citep{arora_exact_comp, velikanov2021explicit}`{=latex} as well as the generalization error `\citep{cao2019towards, basri2020frequency, cui2021generalization,jin2022learning, bowman2022spectral}`{=latex}. Finally, although a powerful and successful tool for analyzing neural networks it must be noted that the NTK has limitations, most notably perhaps that it struggles to explain the rich feature learning commonly observed in practice `\citep{10.5555/3495724.3496995, lazy_training,NEURIPS2020_b7ae8fec}`{=latex}.

#### Prior work on the smallest eigenvalue of the NTK.

Many of the prior works discussed so far assume or prove that \\(\lambda_{\min}({\bm{K}})\\) is positive, but do not provide a quantitative lower bound. Here we discuss works seeking to address this issue and to which we view our work as complementary. For shallow ReLU networks and data drawn uniformly from the sphere, `\citet[Theorem 3]{xie2017diverse}`{=latex} and `\citet[Theorem 3.2]{montanari2022interpolation}`{=latex} provide lower bounds on the smallest singular and eigenvalue value of the Jacobian and NTK respectively. In addition to requiring the data to be drawn uniform from the sphere both of these results are high dimensional in the sense that for `\citet[Theorem 3]{xie2017diverse}`{=latex} to be non-vacuous it is necessary that \\(d_0 = \Omega(d_1 n^2)\\), while `\citet[Theorem 3.2]{montanari2022interpolation}`{=latex} requires, as per their Assumption 3.1, that \\(d_0 = \tilde{\Omega}(\sqrt{n})\\).

`\citet[Theorem 4.1]{nguyen2021tight}`{=latex} derives lower and upper bounds for the smallest eigenvalue of the NTK for deep ReLU networks under standard initialization conditions assuming the data is drawn from a distribution satisfying a Lipschitz concentration property. They show that the NTK is well conditioned if the network has a layer of width of order equal to the number of data points \\(n\\) up to logarithmic factors. Concretely, if at least one layer has width linear in \\(n\\) (ignoring logarithmic factors) and the others are at least poly-logarithmic in \\(n\\), then \\(\lambda_{\min}({\bm{K}}) = \Omega(\mu_r^2(\sigma)d_0)\\) (or \\(\Omega(\mu_r^2(\sigma) )\\) with normalized data), where \\(\mu_r(\sigma)\\) denotes the \\(r\\)th Hermite coefficient of \\(\sigma\\) with any even integer \\(r \geq 2\\). However, in their result the bound holds with high probability only if \\(d_0\\) scales as \\(\log(n)\\).

`\citet[Theorem 1]{bombari2022memorization}`{=latex} derive lower and upper bounds for the smallest eigenvalue of the NTK under similar conditions as `\citet[Theorem 4.1]{nguyen2021tight}`{=latex} aside from the following: they consider smooth rather than ReLU activation functions, the widths follow a loose pyramidal topology, meaning \\(d_l = O(d_{l-1})\\) for all \\(l \in [L-1]\\), \\(d_{L-1}d_{L-2}\\) scales linearly in \\(n\\) (ignoring logarithmic factors), and there exists a \\(\gamma >0\\) such that \\(n^{\gamma} = O(d_{L-1})\\). Under these conditions they show that \\(\lambda_{\min}({\bm{K}}) = \Omega(d_{L-1} d_{L-2})\\) with high probability as both \\(d_{L-1}\\) and \\(n\\) grow. This result illustrates that for the NTK to be well conditioned it suffices that the number of neurons grows as \\(\tilde{\Omega}(\sqrt{n})\\). The loose pyramidal condition on the widths implies \\(d_{L-1} d_{L-2} = O(d_0^2)\\) and as they also assume that \\(n = o(d_{L-1}d_{L-2})\\) then \\(n = o(d_0^2)\\) which in turn implies \\(d_0 = \Omega(\sqrt{n})\\).

# Shallow networks [section:shallow]

Here we study the smallest eigenvalue of the NTK of a shallow neural network. The parameter space \\(\mathcal{P}\\) of this network is \\(\mathbb{R}^{d_1 \times d_0} \times \mathbb{R}^{d_1}\\) and it is equipped with the inner product \\[\begin{aligned}
    \langle ({\bm{W}}, {\bm{v}}), ({\bm{W}}', {\bm{v}}') \rangle = \text{Trace}({\bm{W}}^T {\bm{W}}') + {\bm{v}}^T {\bm{v}}'.
\end{aligned}\\] For convenience we sometimes write \\(d = d_0\\). The neural network \\(f:  \mathbb{R}^{d_0} \times \mathcal{P}  \rightarrow \mathbb{R}\\) is defined as \\[\label{eq:shallow-network-map}
f({\bm{x}}; {\bm{W}}, {\bm{v}}) = \frac{1}{\sqrt{d_1}} \sum_{j=1}^{d_1} v_j \sigma ({\bm{w}}_j^T {\bm{x}}) ,\\] where \\({\bm{W}}= [{\bm{w}}_1, \cdots, {\bm{w}}_{d_1}]^T \in \mathbb{R}^{d_1 \times d_0}\\) are the inner layer weights, \\({\bm{v}}= [v_1, \cdots, v_{d_1}]^T\in \mathbb{R}^{d_1}\\) the outer layer weights, and \\({\bm{\theta}}= ({\bm{W}}, {\bm{v}})\\). We consider the ReLU activation function applied entrywise with \\(\sigma(z) = \max \{0, z \}\\). The derivative \\(\dot{\sigma}\\) satisfies \\(\dot{\sigma}(z) = 1\\) for \\(z > 0\\) and \\(\dot{\sigma}(z) = 0\\) for \\(z < 0\\). Although \\(\sigma\\) is not differentiable at 0, we take \\(\dot{\sigma}(0) = 0\\) by convention. Unless otherwise stated we assume that the entries of \\({\bm{W}}\\) and \\({\bm{v}}\\) are drawn mutually iid from a standard Gaussian distribution \\(\mathcal{N}(0, 1)\\). Our main result for shallow networks is the following theorem.

<div class="restatable" markdown="1">

theoremThmShallowMain<span id="thm:shallow-main" label="thm:shallow-main"></span> Let \\(d \geq 3\\), \\(\epsilon \in (0,1)\\), and \\(\delta, \delta' \in (0, \sqrt{2})\\). Suppose that \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{S}^{d -1}\\) are \\(\delta\\)-separated and \\(\min_{i \neq k}\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta'\\). Define \\[\lambda = \left( 1 + \frac{d\log(1/\delta)}{\log(d)} \right)^{-3} \delta^2.\\] If \\(d_1 \gtrsim \frac{\|{\bm{X}}\|^2 }{\lambda}\log \frac{n}{\epsilon}\\), then with probability at least \\(1 - \epsilon\\), \\[\lambda \lesssim \lambda_{\min}({\bm{K}}) \lesssim \delta'.\\]

</div>

A proof of Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a> is provided in Appendix <a href="#app:shallow-main" data-reference-type="ref" data-reference="app:shallow-main">8.7</a>. Suppressing logarithmic factors, Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a> implies that \\(d_1 = \tilde{\Omega}\left(\|{\bm{X}}\|^2d_0^3\delta^{-2}\right)\\) suffices to ensure that \\(\lambda_{\min}({\bm{K}}) = \tilde{\Omega}(d_0^{-3}\delta^2)\\) and \\(\lambda_{\min}({\bm{K}}) = O(\delta')\\) with high probability (note the trivial bound \\(\|{\bm{X}}\|^2\leq\|{\bm{X}}\|_F^2\leq n\\)). We emphasize that unlike existing results i) we make no distributional assumptions on the data, instead only assuming a milder \\(\delta\\)-separated condition, and ii) our bounds hold with high probability even if \\(d_0\\) is held constant.

A few further remarks are in order. First, the condition \\(d_0 \geq 3\\) is necessary because our technique relies on the addition formula for spherical harmonics `\cite[Theorem 4.11]{efthimiou2014spherical}`{=latex}; the bound we derive based on this formula (Lemma <a href="#lemma:addition-formula-bound" data-reference-type="ref" data-reference="lemma:addition-formula-bound">4</a> in Appendix <a href="#sec:app-spherical" data-reference-type="ref" data-reference="sec:app-spherical">6.2</a>) becomes vacuous for \\(d_0<3\\). However, for \\(d_0=2\\) analogous bounds could be derived using more elementary tools while the case \\(d_0=1\\) is of little interest as only a trivial dataset is possible. Moreover, data in \\(\mathbb{S}^1\\) could be embedded in \\(\mathbb{S}^2\\) since we do not impose any distributional assumptions.

Second, one can use Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a> to bound the smallest eigenvalue of the NTK for data drawn from the uniform distribution on the sphere by bounding \\(\delta\\) with high probability in terms of \\(n\\) and \\(d\\). We use that \\(\delta = \Omega(n^{-2/d_0})\\) and \\(\delta' = O(n^{-2/d_0})\\) with high probability. We direct the interested reader to Appendix <a href="#app:subsec:uniform-sphere" data-reference-type="ref" data-reference="app:subsec:uniform-sphere">8.8</a> for further details.

<div class="restatable" markdown="1">

corollaryCorollaryUniform<span id="corr:uniform" label="corr:uniform"></span> Let \\(d\geq 3\\), \\(n \geq 2\\), \\(\epsilon \in (0,1)\\), \\({\bm{x}}_1, \cdots, {\bm{x}}_n \sim U(\mathbb{S}^{d -1})\\) be mutually iid. Define \\[\lambda = \left(1 + \frac{\log(n/\epsilon) }{\log(d)} \right)^{-3}\left(\frac{\epsilon^2}{n^4}\right)^{1/(d - 1)}.\\] If \\(d_1 \gtrsim \frac{1}{\lambda}\left(1 + \frac{n + \log(1/\epsilon) }{d} \right)\log \frac{n}{\epsilon}\\), then with probability at least \\(1-\epsilon\\) over the data and network parameters, \\[\lambda  \lesssim \lambda_{\min}({\bm{K}}) \lesssim \left(\frac{\log(1/\epsilon) }{n^2} \right)^{1/(d - 1) }.\\]

</div>

The above corollary implies that if \\(d_0d_1 = \tilde{\Omega}\left(n^{1+4/(d_0-1)} \right)\\), then with high probability \\(\lambda_{\min}({\bm{K}}) = \tilde{\Omega}(n^{-4/(d_0 - 1)})\\) and \\(\lambda_{\min}({\bm{K}}) = \tilde{O}(n^{-2/(d_0 - 1)})\\). In particular, for data sampled uniformly from a sphere, the scaling \\(d_0 = {\Omega}(\log n)\\) is both necessary and sufficient for \\(\lambda_{\min}({\bm{K}})\\) to be \\(\tilde{\Theta}(1)\\). In particular the bounds are sharp in this case.

## Proof outline for Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a> [subsec:shallow-proof-outline]

Recall the definitions of \\(F({\bm{\theta}})\\) and \\({\bm{K}}\\) in <a href="#eq:NTK-def" data-reference-type="eqref" data-reference="eq:NTK-def">[eq:NTK-def]</a>. For the choice of \\(f\\) given in <a href="#eq:shallow-network-map" data-reference-type="eqref" data-reference="eq:shallow-network-map">[eq:shallow-network-map]</a>, a straightforward decomposition of the NTK with respect to the inner and outer weights gives \\[\begin{aligned}
 \label{eq:ntk-shallow-decomp}
    {\bm{K}}= {\bm{K}}_1 + {\bm{K}}_2, 
\end{aligned}\\] where \\({\bm{K}}_1 = \nabla_{{\bm{W}}} F({\bm{\theta}})^* \nabla_{{\bm{W}}} F({\bm{\theta}})\\) and \\({\bm{K}}_2 = \nabla_{{\bm{v}}} F({\bm{\theta}})^* \nabla_{{\bm{v}}} F({\bm{\theta}}) = \frac{1}{d_1}\sigma({\bm{W}}{\bm{X}})^T \sigma({\bm{W}}{\bm{X}})\\). As both \\({\bm{K}}_1\\) and \\({\bm{K}}_2\\) are positive semi-definite, \\[\lambda_{\min}({\bm{K}}) \geq  \lambda_{\min}({\bm{K}}_1) +  \lambda_{\min}({\bm{K}}_2);\\] see, e.g., `\citet[Theorem 4.3.1]{Horn_Johnson_2012}`{=latex}. Our proof now follows the highlighted steps below.

#### 1) Bound the smallest eigenvalue in terms of the infinite-width limit.

We proceed to bound both \\(\lambda_{\min}({\bm{K}}_1)\\) and \\(\lambda_{\min}({\bm{K}}_2)\\) in terms of the smallest eigenvalues of their infinite-width counterparts, see Lemmas <a href="#lemma:K1-inf" data-reference-type="ref" data-reference="lemma:K1-inf">[lemma:K1-inf]</a> and <a href="#lemma:K2-inf" data-reference-type="ref" data-reference="lemma:K2-inf">[lemma:K2-inf]</a> below, which act as good approximations for sufficiently wide networks.

<div class="restatable" markdown="1">

lemmalemmaKOneinf<span id="lemma:K1-inf" label="lemma:K1-inf"></span> Suppose that \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{S}^{d - 1}\\). Let \\[\lambda_1 = \lambda_{\min}\left(\mathbb{E}_{{\bm{u}}\sim U(\mathbb{S}^{d - 1}) } \left[\dot{\sigma}\left({\bm{X}}^T{\bm{u}}\right)\dot{\sigma}\left({\bm{u}}^T{\bm{X}}\right) \right] \right).\\] If \\(\lambda_1 > 0\\) and \\(d_1 \gtrsim \lambda_1^{-1}\|{\bm{X}}\|^2 \log \frac{n}{\epsilon}\\), then with probability at least \\(1 - \epsilon\\) \\[\lambda_{\min}({\bm{K}}_1) \gtrsim \lambda_1.\\]

</div>

<div class="restatable" markdown="1">

lemmalemmaKTwoinf<span id="lemma:K2-inf" label="lemma:K2-inf"></span> Suppose that \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{S}^{d - 1 }\\). Let \\[\lambda_2 = d \lambda_{\min}\left(\mathbb{E}_{{\bm{u}}\sim U(\mathbb{S}^{d - 1}) }\left[\sigma({\bm{X}}^T {\bm{u}})\sigma({\bm{u}}^T {\bm{X}}) \right] \right).\\] If \\(\lambda_2 > 0\\) and \\(d_1 \gtrsim  \frac{n}{\lambda_2}  \log\left(\frac{n}{\lambda_2}\right)\log \left(\frac{n}{\epsilon}\right)\\), then with probability at least \\(1 - \epsilon\\) \\[\lambda_{\min}({\bm{K}}_2) \gtrsim 
\lambda_2.\\]

</div>

We prove Lemmas <a href="#lemma:K1-inf" data-reference-type="ref" data-reference="lemma:K1-inf">[lemma:K1-inf]</a> and <a href="#lemma:K2-inf" data-reference-type="ref" data-reference="lemma:K2-inf">[lemma:K2-inf]</a> in Appendices <a href="#app:lemma:K1-inf" data-reference-type="ref" data-reference="app:lemma:K1-inf">8.1</a> and <a href="#app:lemma:K2-inf" data-reference-type="ref" data-reference="app:lemma:K2-inf">8.2</a> respectively. Observe that while the parameters of the model are initialized as Gaussian, the expectations above are taken with respect to the uniform measure on the sphere. The motivation for using the uniform measure on the sphere is that it enables us to work with spherical harmonics, for which there is the highly useful *addition formula* `\citep[see, e.g.,][Theorem 4.11]{efthimiou2014spherical}`{=latex}. The exchange of measures is possible in the case of Lemma <a href="#lemma:K1-inf" data-reference-type="ref" data-reference="lemma:K1-inf">[lemma:K1-inf]</a> due to the scale invariance of \\(\dot{\sigma}\\), while for Lemma <a href="#lemma:K2-inf" data-reference-type="ref" data-reference="lemma:K2-inf">[lemma:K2-inf]</a> it is possible because \\(\sigma\\) is homogeneous.

#### 2) Interpret the infinite-width kernel in terms of a hemisphere transform.

Next, for a given \\({\bm{X}}\\) and \\(\psi \in 
\{\sqrt{d}\sigma, \dot{\sigma}\}\\) we define the limiting NTK \\({\bm{K}}^{\infty}_{\psi} \in \mathbb{R}^{n \times n}\\) as \\[\label{eq:uniform-limiting-NTK}
{\bm{K}}^{\infty}_{\psi} = \mathbb{E}_{{\bm{u}}\sim U(\mathbb{S}^{d - 1})}\left[\psi\left( {\bm{X}}^T{\bm{u}}\right) \psi\left({\bm{u}}^T{\bm{X}}\right) \right].\\] Consider a fixed vector \\({\bm{z}}\in \mathbb{S}^{n-1}\\) and interpret the Euclidean inner product \\(\langle \psi({\bm{X}}^T {\bm{u}}), {\bm{z}}\rangle\\) as a function of \\({\bm{u}}\in \mathbb{S}^{d - 1}\\). It will prove useful to think of this map as an integral transform. To this end let \\(\mathcal{M}(\mathbb{S}^{d - 1})\\) denote the vector space of signed Radon measures on \\(\mathbb{S}^{d - 1}\\) and fix \\(\psi \in \{\sqrt{d}\sigma, \dot{\sigma}\}\\). For a signed Radon measure \\(\mu \in \mathcal{M}(\mathbb{S}^{d - 1})\\) we introduce the integral transform \\(T_{\psi}\mu: \mathbb{S}^{d - 1} \to \mathbb{R}\\), defined as \\[\label{eq:hempisphere-transform}
(T_{\psi}\mu)({\bm{u}}) = \int_{\mathbb{S}^{d - 1}}\psi(\langle {\bm{u}}, {\bm{x}}\rangle) d\mu({\bm{x}}).\\] Note for \\(\psi \in \{\sqrt{d}\sigma, \dot{\sigma}\}\\) this is a *hemisphere transform* `\citep{rubin1999inversion}`{=latex} as the integrand \\(\psi(\langle {\bm{u}}, \cdot \rangle)\\) is supported on a hemisphere normal to \\({\bm{u}}\\). We provide background material on the hemisphere transform in Appendix <a href="#app:hemisphere" data-reference-type="ref" data-reference="app:hemisphere">7</a>. Let \\(\mathcal{M}_{{\bm{X}}} \subset \mathcal{M}\\) denote the space of signed Radon measures supported on the data set \\(\{{\bm{x}}_1, \cdots, {\bm{x}}_n\}\\). For each measure \\(\mu \in \mathcal{M}_{{\bm{X}}}\\) there exists a vector \\({\bm{z}}\in \mathbb{R}^{n}\\) such that \\(\mu = \sum_{i = 1}^n z_i \delta_{{\bm{x}}_i}\\), where \\(\delta_{{\bm{x}}}\\) is the Dirac measure supported on \\({\bm{x}}\\). We write \\(\mu = \mu_{{\bm{z}}}\\) to indicate this correspondence. The following lemma relates the smallest eigenvalue of \\({\bm{K}}_{\psi}^{\infty}\\) to the norm of the hemisphere transform of a measure supported on the data; a proof is provided in Appendix <a href="#app:gram-to-hemisphere-transform" data-reference-type="ref" data-reference="app:gram-to-hemisphere-transform">8.3</a>.

<div class="restatable" markdown="1">

lemmalemmaGradToHemi<span id="lem:gram-to-hemisphere-transform" label="lem:gram-to-hemisphere-transform"></span> Fix \\({\bm{X}}\in \mathbb{R}^{d \times n}\\) and \\(\psi \in \{\sqrt{d}\sigma, \dot{\sigma}\}\\). For all \\({\bm{z}}\in \mathbb{R}^n\\), \\(\langle {\bm{K}}_{\psi}^{\infty}{\bm{z}}, {\bm{z}}\rangle = \|T_{\psi}\mu_{{\bm{z}}}\|^2\\). Moreover, \\[\lambda_{\min}({\bm{K}}_{\psi}^{\infty}) = \inf_{\|{\bm{z}}\| = 1 }\|T_{\psi}\mu_{{\bm{z}}}\|^2.\\]

</div>

#### 3) Bound the hemisphere transform norm via spherical harmonics.

We proceed to lower bound \\(\|T_{\psi} \mu_{{\bm{z}}}\|^2\\) for all \\({\bm{z}}\in \mathbb{R}^d\\). Let \\(L^2(\mathbb{S}^{d - 1})\\) denote the Hilbert space of real-valued, square-integrable functions with respect to the uniform probability measure on \\(\mathbb{S}^{d - 1}\\), and let \\(\mathcal{C}(\mathbb{S}^{d - 1}) \subset L^2(\mathbb{S}^{d - 1})\\) denote the subspace of continuous functions. For \\(\mu \in \mathcal{M}(\mathbb{S}^{d -1 })\\) and \\(g \in \mathcal{C}(\mathbb{S}^{d - 1})\\) we define \\[\langle \mu, g \rangle := \int_{\mathbb{S}^{d-1}} {g({\bm{x}})}d\mu({\bm{x}}) .\\]

If \\(g_1, \cdots, g_N \in L^2(\mathbb{S}^{d -1 })\\) are orthonormal, in particular consider \\(g_r\\) as spherical harmonics, then via a Bessel inequality \\[\begin{aligned}
    \|T_{\psi}\mu_{{\bm{z}}}\|^2 \geq \sum_{a = 1}^N |\langle T_{\psi}\mu_{{\bm{z}}}, g_a \rangle|^2
    = \sum_{a = 1}^N |\langle \mu_{{\bm{z}}}, T_{\psi}g_a \rangle|^2
    = \sum_{a = 1}^N \left|\sum_{i = 1}^n  (T_{\psi} g_a)({\bm{x}}_i) z_i\right|^2.
\end{aligned}\\] Importantly, \\(T_{\psi}\\) is self-adjoint (see Lemma <a href="#lemma:T-self-adjoint" data-reference-type="ref" data-reference="lemma:T-self-adjoint">6</a> in Appendix <a href="#app:hemisphere" data-reference-type="ref" data-reference="app:hemisphere">7</a> for details) and the spherical harmonics are eigenfunctions of \\(T_{\psi}\\), i.e., \\(T_{\psi} g_a = \kappa_a g_a\\). A summary of the key properties of spherical harmonics needed for our results are provided in Appendix <a href="#sec:app-spherical" data-reference-type="ref" data-reference="sec:app-spherical">6.2</a>. Therefore \\[\begin{aligned}
    \|T_{\psi}\mu_{{\bm{z}}}\|^2 \geq \sum_{a = 1}^N \left|\sum_{i = 1}^n  (T_{\psi} g_a)({\bm{x}}_i) z_i\right|^2 = \sum_{a = 1}^N \kappa_a^2 \left|\sum_{i = 1}^n  g_a({\bm{x}}_i) z_i\right|^2 \geq \min_a \kappa_a^2 \| {\bm{D}}{\bm{z}}\|_2^2 , 
\end{aligned}\\] where \\({\bm{D}}\in \mathbb{R}^{N \times n}\\) is a matrix with entries \\([{\bm{D}}]_{ai} = g_a({\bm{x}}_i)\\). As a result \\[\lambda_{\min}({\bm{K}}_{\psi}^{\infty}) \geq \min_a \kappa_a^2 \sigma^2_{\min}({\bm{D}}).\\]

#### 4) Bound the hemisphere transform and spherical harmonics on the data.

The following result shows that if we let the functions \\((g_a)_{a \in [N]}\\) be spherical harmonics and allow \\(N\\) to be sufficiently large, then we can bound the minimum singular value of \\({\bm{D}}\\). In what follows let \\(\mathcal{H}_r^d\\) denote the vector space of degree-\\(r\\) harmonic homogeneous polynomials on \\(d\\) variables.

<div class="restatable" markdown="1">

lemmalemmaMatrixSphericalHarmonic <span id="lem:matrix-spherical-harmonic" label="lem:matrix-spherical-harmonic"></span> Suppose \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{S}^{d - 1}\\) are \\(\delta\\)-separated. Suppose that \\(\beta \in \{0, 1\}\\) and that \\(R \in \mathbb{Z}_{\geq 0}\\) are such that \\(N := \sum_{r = 0}^R \dim(\mathcal{H}_{2r + \beta}^d)\\) satisfies \\(N \geq  C\left(\frac{\delta^4}{2}\right)^{-(d - 2)/2}\\) where \\(C >0\\) is a universal constant. Let \\(g_1, \cdots, g_N\\) be spherical harmonics which form an orthonormal basis of \\(\bigoplus_{r = 0}^R \mathcal{H}_{2r + \beta}^d.\\) If \\({\bm{D}}\in \mathbb{R}^{N \times n}\\) is defined as \\({\bm{D}}_{ai} = g_a({\bm{x}}_i)\\) then \\(\sigma_{\min}({\bm{D}}) \geq \sqrt{\frac{N}{2}}.\\)

</div>

A proof of Lemma <a href="#lem:matrix-spherical-harmonic" data-reference-type="ref" data-reference="lem:matrix-spherical-harmonic">[lem:matrix-spherical-harmonic]</a> can be found in Appendix <a href="#app:matrix-spherical-harmonic" data-reference-type="ref" data-reference="app:matrix-spherical-harmonic">8.4</a>. By carefully choosing values for \\(R\\) and \\(N\\) in Lemma <a href="#lem:matrix-spherical-harmonic" data-reference-type="ref" data-reference="lem:matrix-spherical-harmonic">[lem:matrix-spherical-harmonic]</a> and performing some asymptotics on the resulting expressions, we arrive at the following bound on the hemisphere transform of a measure.

<div class="restatable" markdown="1">

lemmaCorrHemisphereTransformAsymptotics<span id="corr:hemisphere-transform-asymptotics" label="corr:hemisphere-transform-asymptotics"></span> Let \\(d \geq 3\\) and suppose that \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{S}^{d - 1}\\) are \\(\delta\\)-separated. For all \\({\bm{z}}\in \mathbb{R}^n\\) with \\(\|{\bm{z}}\| \leq 1\\) then \\[\|T_{\psi} \mu_{{\bm{z}}}\|^2 \gtrsim \begin{cases}
            \left(1 + \frac{d\log(1/\delta)}{\log d}\right)^{-3} \delta^2 & \text{ if $\psi = \dot{\sigma}$}\\
            \left(1+ \frac{d\log(1/\delta) }{\log d}\right)^{-3}\delta^4 & \text{ if $\psi = \sqrt{d}\sigma$}.
        \end{cases}\\]

</div>

A proof of Lemma <a href="#corr:hemisphere-transform-asymptotics" data-reference-type="ref" data-reference="corr:hemisphere-transform-asymptotics">[corr:hemisphere-transform-asymptotics]</a> is provided in Appendix <a href="#app:hemi-transform-asymp" data-reference-type="ref" data-reference="app:hemi-transform-asymp">8.5</a>. The lower bound of Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a> follows by bounding \\(\lambda_1\\), as defined in Lemma <a href="#lemma:K1-inf" data-reference-type="ref" data-reference="lemma:K1-inf">[lemma:K1-inf]</a>, using Lemma <a href="#corr:hemisphere-transform-asymptotics" data-reference-type="ref" data-reference="corr:hemisphere-transform-asymptotics">[corr:hemisphere-transform-asymptotics]</a>.

Before proceeding to the upper bound, we pause to remark on the generality of this argument for handling other activation functions. First, we use the positive homogeneity of the activation function in order to write \\(\lambda_{\min}({\bm{K}}_{\psi}^{\infty})\\) as the \\(L_2(\mathbb{S}^{d-1})\\) norm of a function on the sphere. This is beneficial as it allows us to work with the spherical harmonics and use the associated addition formula. The ReLU activation and its derivative are also convenient with regard to computing the eigenvalues of the hemisphere transform (or more generally the eigenvalues of the integral operator). In particular, this requires evaluating integrals against Gegenbauer polynomials for which analytic expressions are available. For polynomial or piecewise polynomial activations similar results could be obtained. However, for other activations, e.g., tanh or sigmoid, such quantities appear challenging to compute.

#### 5) Upper bound.

The upper bound of Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a> is simpler than the lower bound and hinges on the following calculation. Let \\({\bm{x}}_i, {\bm{x}}_k\\) be two data points. Then \\[\begin{aligned}
    \lambda_{\min}({\bm{K}}) &\leq \frac{1}{2} ({\bm{e}}_i - {\bm{e}}_k)^T {\bm{K}}({\bm{e}}_i - {\bm{e}}_k)  = \frac{1}{2}\|\nabla_{{\bm{\theta}}}f({\bm{x}}_i) - \nabla_{{\bm{\theta}}}f({\bm{x}}_k)\|^2.
\end{aligned}\\] Therefore it suffices to upper bound the norm of \\(\nabla_{{\bm{\theta}}} f({\bm{x}}_i) - \nabla_{{\bm{\theta}}} f({\bm{x}}_k)\\). We choose \\(i, k \in [n]\\) such that \\({\bm{x}}_i, {\bm{x}}_k\\) are the two closest points in the dataset. We then translate this into a statement about the gradients. If \\(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta\\), then with high probability over the network parameters, \\(\|\nabla_{{\bm{\theta}}} f({\bm{x}}_i) - \nabla_{{\bm{\theta}}} f({\bm{x}}_k)\|^2 \lesssim \delta\\) (see Lemma <a href="#lemma:separation-gradient-shallow" data-reference-type="ref" data-reference="lemma:separation-gradient-shallow">18</a>), and we arrive at the desired upper bound in Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a>.

# From shallow to deep neural networks [sec:deep]

Our goal here is to detail just one approach as how the results of Section <a href="#section:shallow" data-reference-type="ref" data-reference="section:shallow">3</a> can be extended to deep networks. To be clear, here we consider a fully connected network with input dimension \\(d_0\\) and \\(L\\) layers, where each layer has width \\(d_1, \cdots, d_L\\) respectively and \\(d_L = 1\\). The parameter space \\(\mathcal{P}\\) is a product space of matrices \\(\prod_{l= 1}^L \mathbb{R}^{d_l \times d_{l -1 } }\\), equipped with the inner product \\[\langle ({\bm{W}}_1, \cdots, {\bm{W}}_L), ({\bm{W}}_1', \cdots, {\bm{W}}_L') \rangle = \sum_{l = 1}^L \text{Trace}({\bm{W}}_l^T{\bm{W}}_l').\\] The feature maps \\(f_l: \mathbb{R}^{d_{0}} \times \mathcal{P} \to \mathbb{R}^{d_l}\\) of the neural network are given by \\[f_l({\bm{x}}; {\bm{\theta}}) = \begin{cases}
        {\bm{x}}& l = 0\\
         \sigma({\bm{W}}_l f_{l - 1}({\bm{x}}; {\bm{\theta}})) & l \in [L - 1]\\
        {\bm{W}}_l f_{l - 1}({\bm{x}};{\bm{\theta}}) & l = L,
    \end{cases}\\] where \\({\bm{W}}_l \in \mathbb{R}^{d_l \times d_{l-1}}\\) for all \\(l \in [L]\\), \\({\bm{\theta}}= ({\bm{W}}_1, \cdots, {\bm{W}}_L)\\) and \\(\sigma\\) is the ReLU function \\(x \mapsto \max(0, x)\\) applied elementwise. We define the network map \\(f\\) to be the final feature map multiplied by a normalizing constant: \\[\label{eq:normalization}
f = \left(\prod_{l = 1}^{L - 1}\sqrt{\frac{2}{d_l}  }  \right)f_L.\\] Given \\(n\\) data points \\({\bm{x}}_1, \cdots, {\bm{x}}_n\\), we bound the smallest eigenvalue of the NTK <a href="#eq:NTK-def" data-reference-type="eqref" data-reference="eq:NTK-def">[eq:NTK-def]</a> associated with this particular choice of \\(f\\).

<div class="restatable" markdown="1">

theoremThmDeepMain<span id="thm:deep-main" label="thm:deep-main"></span> Suppose \\(\epsilon \in (0,1/3)\\), \\(\delta \in (0,\sqrt{2}]\\), \\(d_0 \geq 3\\), the data \\({\bm{x}}_1, {\bm{x}}_2, \cdots, {\bm{x}}_n \in \mathbb{S}^{d_0 -1 }\\) is \\(\delta\\)-separated and define \\[\lambda = \left(1+ \frac{d_0\log(1/\delta) }{\log d_0}\right)^{-3}\delta^4.\\] With regard to the network architecture, let \\(L \geq 3\\), \\(d_l \geq d_{l+1}\\) for all \\(l \in [L - 1]\\), \\(d_{L-1} \gtrsim 2^L \log \left (\frac{nL}{\epsilon}\right)\\) and \\(d_1 \gtrsim \tfrac{n}{\lambda} \log \left( \tfrac{n}{\lambda}\right) \log \left( \tfrac{n}{\epsilon}\right)\\). Then with probability at least \\(1- \epsilon\\) over the network parameters \\[\lambda \lesssim \lambda_{\min}({\bm{K}}) \lesssim L .\\]

</div>

We emphasize that these bounds make no distributional assumptions on the data other than lying on the sphere and hold even for constant \\(d_0\\). Indeed, if we consider \\(d_0\\) as some constant then Theorem <a href="#thm:deep-main" data-reference-type="ref" data-reference="thm:deep-main">[thm:deep-main]</a> implies that if the first layer is sufficiently wide, \\(d_1 = \tilde{\Omega}(n \delta^{-4})\\), then with high probability over the parameters \\(\lambda_{\min}({\bm{K}}) = \tilde{\Omega}(\delta^4)\\) and \\(\lambda_{\min}({\bm{K}}) = O(1)\\).

A few remarks are in order. First, the pyramidal condition on the network widths could be relaxed by more directly borrowing techniques from `\cite{nguyen2021tight}`{=latex}. We adopt this condition as it has the advantage of making the dependence of our bounds on the network depth \\(L\\) clearer. Second, compared with Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a> and ignoring log factors, we observe the lower bound differs by a factor of \\(\delta^2\\). This arises as a result of the smallest eigenvalue of the feature Gram matrix \\({\bm{F}}_1^T {\bm{F}}_1\\) being equivalent to the Jacobian of a shallow network with respect to the second layer weights, not the inner layer weights, which has a different lower bound as per Lemma <a href="#corr:hemisphere-transform-asymptotics" data-reference-type="ref" data-reference="corr:hemisphere-transform-asymptotics">[corr:hemisphere-transform-asymptotics]</a>. For reasons apparent in the proof outline below the lower bound on \\(\lambda_{\min}({\bm{K}})\\) lacks a dependency on \\(L\\), however we hypothesize it should also grow linearly with \\(L\\) thereby matching the dependency of the upper bound. Finally, the upper bound itself follows a similar approach as used by `\cite{nguyen2021tight}`{=latex} and is weak in the sense that we cannot take advantage of the dataset separation for gradients deeper into the network. We remark that this is also a common problem in the prior work of `\cite{nguyen2021tight}`{=latex} and `\cite{bombari2022memorization}`{=latex}, we refer the reader to the proof outline below for further details.

## Proof outline for Theorem <a href="#thm:deep-main" data-reference-type="ref" data-reference="thm:deep-main">[thm:deep-main]</a> [subsec:deep-proof-sketch]

The proof of the deep case is structured around the decomposition of the NTK provided in Lemma <a href="#lemma:NTKdecomp" data-reference-type="ref" data-reference="lemma:NTKdecomp">[lemma:NTKdecomp]</a> below. To state this decomposition we introduce the following quantities. For \\(l \in [L - 1]\\) we define the feature matrices \\({\bm{F}}_l \in \mathbb{R}^{d_l \times n}\\) by \\[{\bm{F}}_l = [f_l({\bm{x}}_1), \cdots, f_l({\bm{x}}_n)].\\] For \\(l \in [L - 1]\\) and \\({\bm{x}}\in \mathbb{R}^d\\) we define the activation patterns \\({\bm{\Sigma}}_l({\bm{x}}) \in \{0, 1\}^{d_l \times d_l}\\) to be the diagonal matrices \\[{\bm{\Sigma}}_l({\bm{x}}) = \text{diag}(\dot{\sigma}({\bm{W}}_{l}f_{l - 1}({\bm{x}}))).\\] Finally, we let \\(\textbf{1}_{n}\\) denote the vector of all ones in \\(\mathbb{R}^n\\).

<div class="restatable" markdown="1">

lemmalemmaNTKdecomp<span id="lemma:NTKdecomp" label="lemma:NTKdecomp"></span> Let \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{R}^d\\) be nonzero. There exists an open set \\(\mathcal{U} \subset \mathcal{P}\\) of full Lebesgue measure such that \\(f({\bm{x}}_i;\cdot)\\) is continuously differentiable on \\(\mathcal{U}\\) for all \\(i \in [n]\\). Moreover, for all \\(\bm{\theta} \in \mathcal{U}\\) the NTK Gram matrix \\({\bm{K}}\\) defined in <a href="#eq:NTK-def" data-reference-type="eqref" data-reference="eq:NTK-def">[eq:NTK-def]</a> with network function <a href="#eq:normalization" data-reference-type="eqref" data-reference="eq:normalization">[eq:normalization]</a> satisfies \\[\left( \prod_{l = 1}^{L - 1} \frac{d_l}{2}\right){\bm{K}}
    {=} \sum_{l = 0}^{L-1} ({\bm{F}}_{l}^T {\bm{F}}_{l}) \odot ({\bm{B}}_{l+1} {\bm{B}}_{l+1}^T),\\] where the \\(i\\)th row of \\({\bm{B}}_l \in \mathbb{R}^{n \times n_l}\\) is defined as \\[\begin{aligned}
        [{\bm{B}}_l]_{i,:} = \begin{cases}
            {\bm{\Sigma}}_l({\bm{x}}_i) \left( \prod_{k = l+1}^{L-1} {\bm{W}}_k^T {\bm{\Sigma}}_k({\bm{x}}_i) \right){\bm{W}}_L^T, &l \in [L-1],\\
            \normalfont{\textbf{1}}_{n}, &l = L.
        \end{cases}
    
\end{aligned}\\]

</div>

For completeness we prove Lemma <a href="#lemma:NTKdecomp" data-reference-type="ref" data-reference="lemma:NTKdecomp">[lemma:NTKdecomp]</a> in Appendix <a href="#app:deep-setting" data-reference-type="ref" data-reference="app:deep-setting">9.1</a>. Observe each matrix summand in Lemma <a href="#lemma:NTKdecomp" data-reference-type="ref" data-reference="lemma:NTKdecomp">[lemma:NTKdecomp]</a> is positive semi-definite (PSD) and recall for any two PSD matrices \\({\bm{A}}\\) and \\({\bm{B}}\\) one has \\(\lambda_{\min}({\bm{A}}+ {\bm{B}}) \geq \lambda_{\min}({\bm{A}}) + \lambda_{\min}({\bm{B}})\\) `\citep[see e.g.][Theorem 4.3.1]{Horn_Johnson_2012}`{=latex} and \\(\lambda_{\min}({\bm{A}}\odot {\bm{B}}) \geq \lambda_{\min}({\bm{A}}) \min_{i \in[n]} [{\bm{B}}]_{ii}\\) `\citep{Schur1911}`{=latex}. Therefore \\[\begin{aligned}
 \label{eq:breakdown-deep-to-shallow}
     \left( \prod_{l = 1}^{L - 1} \frac{d_l}{2}\right)\lambda_{\min}({\bm{K}}) & \geq \sum_{l = 0}^{L-1} \lambda_{\min} \left( ({\bm{F}}_{l}^T {\bm{F}}_{l}) \odot ({\bm{B}}_{l+1} {\bm{B}}_{l+1}^T)\right)    \geq \lambda_{\min}\left( {\bm{F}}_{1}^T {\bm{F}}_{1} \right) \min_{i \in [n]} \left\|[{\bm{B}}_2]_{i,:} \right\|^2.
\end{aligned}\\]

In order to upper bound the smallest eigenvalue we follow `\cite{nguyen2021tight}`{=latex} and analyze the Raleigh quotient \\(R({\bm{u}}) = \tfrac{{\bm{u}}^T {\bm{K}}{\bm{u}}}{\| {\bm{u}}\|^2}\\). In particular, for any nonzero \\({\bm{u}}\in \mathbb{R}^n\\) we have \\(\lambda_{\min}({\bm{K}}) \leq R({\bm{u}})\\) and therefore \\(\lambda_{\min}({\bm{K}}) \leq R({\bm{e}}_i) = [{\bm{K}}]_{ii}\\) for all \\(i \in [n]\\). As a result \\[\begin{aligned}
     \left( \prod_{l = 1}^{L - 1} \frac{d_l}{2}\right)\lambda_{\min}({\bm{K}}) &\leq \left[ \sum_{l = 0}^{L-1} ({\bm{F}}_l^T {\bm{F}}_l) \odot ({\bm{B}}_{l+1} {\bm{B}}_{l+1}^T) \right]_{ii}    = \sum_{l=0}^{L-1} \| f_l({\bm{x}}_i) \|^2 \| [{\bm{B}}_{l+1}]_{i,:} \|^2.
\end{aligned}\\] Combining the upper and lower bounds we have \\[\label{eq:min-eig-NTK-bounds1}
    \lambda_{\min}\left( {\bm{F}}_{1}^T {\bm{F}}_{1}\right) \min_{i \in [n]} \|[{\bm{B}}_2]_{i,:} \|^2 \leq \lambda_{\min}({\bm{K}})  \left( \prod_{l = 1}^{L - 1} \frac{d_l}{2}\right) \leq \sum_{l=0}^{L-1} \| f_l({\bm{x}}_i) \|^2 \| [{\bm{B}}_{l+1}]_{i,:} \|^2 ,\\] where the right hand side holds for any \\(i \in [n]\\). Based on <a href="#eq:min-eig-NTK-bounds1" data-reference-type="eqref" data-reference="eq:min-eig-NTK-bounds1">[eq:min-eig-NTK-bounds1]</a>, we proceed first by bounding the norm of the network features. We achieve this via an inductive argument, bounding the norm of the features at one layer with high probability, and then conditioning on this event to bound the norm of the features at the next layer with high probability.

<div class="restatable" markdown="1">

lemmalemmaFeatureNorms<span id="lemma:FeatureNorms" label="lemma:FeatureNorms"></span> Let \\({\bm{x}}\in \mathbb{S}^{d_0-1}\\), \\(L \geq 2\\) and \\(l \in [L-1]\\). If \\(d_k \gtrsim l^2 \log(l/\epsilon)\\) for all \\(k\in [l]\\), then \\[e^{-1} \left( \prod_{h=1}^l \frac{d_h}{2}\right) \leq \| f_l({\bm{x}}) \|^2 \leq e \left( \prod_{h=1}^l \frac{d_h}{2}\right)\\] holds with probability at least \\(1- \epsilon\\) over the network parameters.

</div>

A proof of Lemma <a href="#lemma:FeatureNorms" data-reference-type="ref" data-reference="lemma:FeatureNorms">[lemma:FeatureNorms]</a> is provided in Appendix <a href="#app:FeatureNorms" data-reference-type="ref" data-reference="app:FeatureNorms">9.2</a>. Next we derive upper and lower bounds on the backpropagation terms \\([{\bm{B}}_l]_{i,:}\\). Our strategy for this is as follows: for \\(l\in [L-2]\\), let \\({\bm{S}}_{l}({\bm{x}}) = {\bm{\Sigma}}_l({\bm{x}}) \left( \prod_{k = l+1}^{L-1} {\bm{W}}_k^T {\bm{\Sigma}}_k({\bm{x}}) \right)\\) and observe \\[[{\bm{B}}_l]_{i,:} = {\bm{S}}_{l}({\bm{x}}_i){\bm{W}}_L^T.\\] Since \\({\bm{x}}_i \in \mathbb{S}^{d_0-1}\\), it is sufficient to lower bound \\(\| {\bm{S}}_{l}({\bm{x}}) {\bm{W}}_L^T\|_2^2\\) for an arbitrary \\({\bm{x}}\in \mathbb{S}^{d_0-1}\\). As the vector \\({\bm{W}}_L^T \in \mathbb{R}^{d_{L-1}}\\) is distributed as \\({\bm{W}}_L^T \sim \mathcal{N}(\textbf{0}_{d_{L-1}}, \textit{I}_{d_{L-1}})\\), following `\citet[Theorem 6.3.2]{vershynin2018high}`{=latex} we have that for any \\({\bm{A}}\in \mathbb{R}^{d_l \times d_{L-1}}\\) and \\(t \geq 0\\) \\[\begin{aligned}
    \mathbb{P}( | \|{\bm{A}}{\bm{W}}_L^T \| - \|{\bm{A}}\|_F | \geq t) \leq 2 \exp \left( -\frac{Ct^2}{ \|{\bm{A}}\|^2}\right)
\end{aligned}\\] for some constant \\(C>0\\). As a result, with \\(t = \tfrac{1}{2}\| {\bm{A}}\|_F^2\\) then \\[\mathbb{P}\left( \frac{1}{4} \| {\bm{A}}\|_F^2 \leq \| {\bm{A}}{\bm{W}}_L^T \|^2 \leq \frac{3}{4} \| {\bm{A}}\|_F^2 \right) \geq 1 - \exp\left( -C\frac{\| {\bm{A}}\|_F^2}{\| {\bm{A}}\|^2}\right).\\] In order to lower bound \\(\| {\bm{S}}_{l}({\bm{x}}) {\bm{W}}_L^T \|^2\\) with high probability over the parameters it therefore suffices to condition on appropriate bounds for \\(\| {\bm{S}}_{l}({\bm{x}}) \|_F^2\\) and \\(\| {\bm{S}}_{l}({\bm{x}}) \|_2^2\\). These bounds are provided in Lemmas <a href="#lemma:Frob-S" data-reference-type="ref" data-reference="lemma:Frob-S">[lemma:Frob-S]</a> and <a href="#lemma:Op-S" data-reference-type="ref" data-reference="lemma:Op-S">[lemma:Op-S]</a> in Appendices <a href="#app:Frob-S" data-reference-type="ref" data-reference="app:Frob-S">9.3</a> and <a href="#app:Op-S" data-reference-type="ref" data-reference="app:Op-S">9.4</a> respectively. With these two lemmas in place we can bound \\(\|{\bm{S}}_{l}({\bm{x}}_i){\bm{W}}_L^T\|^2\\).

<div class="restatable" markdown="1">

lemmaLemmaMinBTwo<span id="lemma:min-B2" label="lemma:min-B2"></span> Let \\({\bm{x}}\in \mathbb{S}^{d_0-1}\\), suppose \\(L \geq 3\\), \\(d_k \geq d_{k+1}\\) for all \\(k \in [L - 1]\\) and \\(d_{L-1} \gtrsim 2^L \log \left (\frac{L}{\epsilon}\right)\\). Then, for any \\(l \in [L-1]\\), with probability at least \\(1 - \epsilon\\) over the network parameters \\[\|{\bm{S}}_l({\bm{x}}) {\bm{W}}_L^T \|^2 \asymp 2^{-L+l+1}\prod_{k = l}^{L-1} d_k .\\]

</div>

By combining Lemma <a href="#lemma:min-B2" data-reference-type="ref" data-reference="lemma:min-B2">[lemma:min-B2]</a> with a union bound we arrive at the following corollary, relevant for the lower bound of <a href="#eq:min-eig-NTK-bounds1" data-reference-type="eqref" data-reference="eq:min-eig-NTK-bounds1">[eq:min-eig-NTK-bounds1]</a>.

<div id="corr:B2-lb" class="corollary" markdown="1">

**Corollary 1**. *Let \\({\bm{x}}_i \in \mathbb{S}^{d_0-1}\\) for all \\(i \in [n]\\), \\(L \geq 3\\), \\(d_l \geq d_{l+1}\\) for all \\(l \in [L - 1]\\) and \\(d_{L-1} \gtrsim 2^L \log \left (\frac{nL}{\epsilon}\right)\\). Then, for any \\(l \in [L-1]\\), with probability at least \\(1 - \epsilon\\) over the network parameters \\[\min_{i \in [n]} \|[{\bm{B}}_2]_{i,:} \|^2 \gtrsim 2^{-L} \prod_{k = 2}^{L-1} d_k .\\]*

</div>

The first-layer feature Gram matrix \\({\bm{F}}_1^T {\bm{F}}_1\\) in the deep case is identically distributed to \\({\bm{K}}_2\\) in the two-layer case; see <a href="#eq:ntk-shallow-decomp" data-reference-type="eqref" data-reference="eq:ntk-shallow-decomp">[eq:ntk-shallow-decomp]</a> and the related definitions. Therefore we can apply Lemma <a href="#lemma:K2-inf" data-reference-type="ref" data-reference="lemma:K2-inf">[lemma:K2-inf]</a> to lower bound the smallest eigenvalue of \\({\bm{F}}_1^T {\bm{F}}_1\\). This, in combination with Corollary <a href="#corr:B2-lb" data-reference-type="ref" data-reference="corr:B2-lb">1</a>, yields the lower bound of Theorem <a href="#thm:deep-main" data-reference-type="ref" data-reference="thm:deep-main">[thm:deep-main]</a>. The upper bound follows by combining the bound on the feature norms provided by Lemma <a href="#lemma:FeatureNorms" data-reference-type="ref" data-reference="lemma:FeatureNorms">[lemma:FeatureNorms]</a> with the bound on the backpropagation terms given in Lemma <a href="#lemma:min-B2" data-reference-type="ref" data-reference="lemma:min-B2">[lemma:min-B2]</a>. A detailed proof of Theorem <a href="#thm:deep-main" data-reference-type="ref" data-reference="thm:deep-main">[thm:deep-main]</a> is provided in Appendix <a href="#app:thm-deep-main" data-reference-type="ref" data-reference="app:thm-deep-main">9.6</a>.

# Conclusion

#### Summary and implications.

Quantitative bounds on the smallest eigenvalue of the NTK are a critical ingredient for many current analyses of network optimization. Prior works provide bounds which are only applicable for data drawn from particular distributions and for which the input dimension \\(d_0\\) scales appropriately with the number of data samples \\(n\\). This work plugs an important gap in the existing literature by providing bounds for arbitrary datasets on the sphere (including those drawn from any distribution on the sphere) in terms of a measure of their collinearity. Furthermore, these bounds are applicable for any \\(d_0\\), in particular even \\(d_0\\) held constant with respect to \\(n\\).

#### Limitations.

Our bounds currently only hold for the ReLU activation function. Another limitation, also present in prior work, is that our upper bound on the smallest eigenvalue of the NTK for deep networks in Theorem <a href="#thm:deep-main" data-reference-type="ref" data-reference="thm:deep-main">[thm:deep-main]</a> does not capture the data separation. Finally, a mild limitation of this work is that we require the data to be normalized so as to lie on the sphere.

#### Future work.

The proof techniques developed here could be applied to analyze the NTK in the context of other homogeneous activation functions. One could potentially relax the homogeneity condition on the activation function, or the condition of unit norm data, by considering an integral transform on the space \\(L^2(\mathbb{R}^d, \mu)\\) rather than \\(L^2(\mathbb{S}^{d - 1})\\), where \\(\mu\\) denotes the standard Gaussian measure (since the weights are drawn from a Gaussian distribution). Beyond fully connected networks, conducting comparable analyses in the context of other architectures, e.g., CNNs, GNNs, or transformers, would be valuable future work.

### Acknowledgments [acknowledgments]

GM and KK were partly supported by NSF CAREER DMS 2145630 and DFG SPP 2298 Theoretical Foundations of Deep Learning grant 464109215. GM was also partly supported by NSF grant CCF 2212520, ERC Starting Grant 757983 (DLT), and BMBF in DAAD project 57616814 (SECAI).

# References [references]

<div class="thebibliography" markdown="1">

Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song A convergence theory for deep learning via over-parameterization In *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of Machine Learning Research*, pp. 242–252. PMLR, 2019. URL <https://proceedings.mlr.press/v97/allen-zhu19a.html>. **Abstract:** Deep neural networks (DNNs) have demonstrated dominating performance in many fields; since AlexNet, networks used in practice are going wider and deeper. On the theoretical side, a long line of works has been focusing on training neural networks with one hidden layer. The theory of multi-layer networks remains largely unsettled. In this work, we prove why stochastic gradient descent (SGD) can find $\\}textit{global minima}$ on the training objective of DNNs in $\\}textit{polynomial time}$. We only make two assumptions: the inputs are non-degenerate and the network is over-parameterized. The latter means the network width is sufficiently large: $\\}textit{polynomial}$ in $L$, the number of layers and in $n$, the number of samples. Our key technique is to derive that, in a sufficiently large neighborhood of the random initialization, the optimization landscape is almost-convex and semi-smooth even with ReLU activations. This implies an equivalence between over-parameterized neural networks and neural tangent kernel (NTK) in the finite (and polynomial) width setting. As concrete examples, starting from randomly initialized weights, we prove that SGD can attain 100% training accuracy in classification tasks, or minimize regression loss in linear convergence speed, with running time polynomial in $n,L$. Our theory applies to the widely-used but non-smooth ReLU activation, and to any smooth and possibly non-convex loss functions. In terms of network architectures, our theory at least applies to fully-connected neural networks, convolutional neural networks (CNN), and residual neural networks (ResNet). (@allenzhu2019convergence)

Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks In *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of Machine Learning Research*, pp. 322–332. PMLR, 09–15 Jun 2019. URL <https://proceedings.mlr.press/v97/arora19a.html>. **Abstract:** Recent works have cast some light on the mystery of why deep nets fit any data and generalize despite being very overparametrized. This paper analyzes training and generalization for a simple 2-layer ReLU net with random initialization, and provides the following improvements over recent works: (i) Using a tighter characterization of training speed than recent papers, an explanation for why training a neural net with random labels leads to slower training, as originally observed in \[Zhang et al. ICLR’17\]. (ii) Generalization bound independent of network size, using a data-dependent complexity measure. Our measure distinguishes clearly between random labels and true labels on MNIST and CIFAR, as shown by experiments. Moreover, recent papers require sample complexity to increase (slowly) with the size, while our sample complexity is completely independent of the network size. (iii) Learnability of a broad class of smooth functions by 2-layer ReLU nets trained via gradient descent. The key idea is to track dynamics of training and generalization via properties of a related kernel. (@pmlr-v97-arora19a)

Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and Ruosong Wang On exact computation with an infinitely wide neural net In *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL <https://proceedings.neurips.cc/paper/2019/file/dbc4d84bfcfe2284ba11beffb853a8c4-Paper.pdf>. **Abstract:** How well does a classic deep net architecture like AlexNet or VGG19 classify on a standard dataset such as CIFAR-10 when its “width”— namely, number of channels in convolutional layers, and number of nodes in fully-connected internal layers — is allowed to increase to infinity? Such questions have come to the forefront in the quest to theoretically understand deep learning and its mysteries about optimization and generalization. They also connect deep learning to notions such as Gaussian processes and kernels. A recent paper \[Jacot et al., 2018\] introduced the Neural Tangent Kernel (NTK) which captures the behavior of fully-connected deep nets in the infinite width limit trained by gradient descent; this object was implicit in some other recent papers. An attraction of such ideas is that a pure kernel-based method is used to capture the power of a fully-trained deep net of infinite width. The current paper gives the first efficient exact algorithm for computing the extension of NTK to convolutional neural nets, which we call Convolutional NTK (CNTK), as well as an efficient GPU implementation of this algorithm. This results in a significant new benchmark for performance of a pure kernel-based method on CIFAR-10, being 10% higher than the methods reported in \[Novak et al., 2019\], and only 6% lower than the performance of the corresponding finite deep net architecture (once batch normalization etc. are turned off). Theoretically, we also give the first non-asymptotic proof showing that a fully-trained sufficiently wide net is indeed equivalent to the kernel regression predictor using NTK. (@arora_exact_comp)

Sheldon Axler, Paul Bourdon, and Ramey Wade *Harmonic function theory*, volume 137 Springer Science & Business Media, 2013. URL <https://doi.org/10.1007/978-1-4757-8137-3>. **Abstract:** This is a book about harmonic functions in Euclidean space. Readers with a background in real and complex analysis at the beginning graduate level will feel comfortable with the material presented here. The authors have taken unusual care to motivate concepts and simplify proofs. Topics include: basic properties of harmonic functions, Poisson integrals, the Kelvin transform, spherical harmonics, harmonic Hardy spaces, harmonic Bergman spaces, the decomposition theorem, Laurent expansions, isolated singularities, and the Dirichlet problem. The new edition contains a completely rewritten chapter on spherical harmonics, a new section on extensions of Bocher’s Theorem, new exercises and proofs, as well as revisions throughout to improve the text. A unique software package-designed by the authors and available by e-mail - supplements the text for readers who wish to explore harmonic function theory on a computer. (@axler2013harmonic)

Keith Ball An elementary introduction to modern convex geometry *Flavors of geometry*, 31: 1–58, 1997. **Abstract:** Preface 1 Lecture 1. Basic Notions 2 Lecture 2. Spherical Sections of the Cube 8 Lecture 3. Fritz John’s Theorem 13 Lecture 4. Volume Ratios and Spherical Sections of the Octahedron 19 Lecture 5. The Brunn–Minkowski Inequality and Its Extensions 25 Lecture 6. Convolutions and Volume Ratios: The Reverse Isoperimetric Problem 32 Lecture 7. The Central Limit Theorem and Large Deviation Inequalities 37 Lecture 8. Concentration of Measure in Geometry 41 Lecture 9. Dvoretzky’s Theorem 47 Acknowledgements 53 References 53 Index 55 (@ball1997elementary)

Arindam Banerjee, Pedro Cisneros-Velarde, Libin Zhu, and Mikhail Belkin Neural tangent kernel at initialization: Linear width suffices In *The 39th Conference on Uncertainty in Artificial Intelligence*, 2023. URL <https://openreview.net/forum?id=VJaoe7Rp9tZ>. **Abstract:** We study the eigenvalue distributions of the Conjugate Kernel and Neural Tangent Kernel associated to multi-layer feedforward neural networks. In an asymptotic regime where network width is increasing linearly in sample size, under random initialization of the weights, and for input samples satisfying a notion of approximate pairwise orthogonality, we show that the eigenvalue distributions of the CK and NTK converge to deterministic limits. The limit for the CK is described by iterating the Marcenko-Pastur map across the hidden layers. The limit for the NTK is equivalent to that of a linear combination of the CK matrices across layers, and may be described by recursive fixed-point equations that extend this Marcenko-Pastur map. We demonstrate the agreement of these asymptotic predictions with the observed spectra for both synthetic and CIFAR-10 training data, and we perform a small simulation to investigate the evolutions of these spectra over training. (@banerjee2023neural)

Ronen Basri, David W. Jacobs, Yoni Kasten, and Shira Kritchman The convergence rate of neural networks for learned functions of different frequencies In *Advances in Neural Information Processing Systems 32*, pp. 4763–4772, 2019. URL <https://proceedings.neurips.cc/paper/2019/hash/5ac8bb8a7d745102a978c5f8ccdb61b8-Abstract.html>. **Abstract:** We study the relationship between the frequency of a function and the speed at which a neural network learns it. We build on recent results that show that the dynamics of overparameterized neural networks trained with gradient descent can be well approximated by a linear system. When normalized training data is uniformly distributed on a hypersphere, the eigenfunctions of this linear system are spherical harmonic functions. We derive the corresponding eigenvalues for each frequency after introducing a bias term in the model. This bias term had been omitted from the linear network model without significantly affecting previous theoretical results. However, we show theoretically and experimentally that a shallow neural network without bias cannot represent or learn simple, low frequency functions with odd frequencies. Our results lead to specific predictions of the time it will take a network to learn functions of varying frequency. These predictions match the empirical behavior of both shallow and deep networks. (@uniform_sphere_data)

Ronen Basri, Meirav Galun, Amnon Geifman, David Jacobs, Yoni Kasten, and Shira Kritchman Frequency bias in neural networks for input of non-uniform density In *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 685–694. PMLR, 13–18 Jul 2020. URL <https://proceedings.mlr.press/v119/basri20a.html>. **Abstract:** Recent works have partly attributed the generalization ability of over-parameterized neural networks to frequency bias – networks trained with gradient descent on data drawn from a uniform distribution find a low frequency fit before high frequency ones. As realistic training sets are not drawn from a uniform distribution, we here use the Neural Tangent Kernel (NTK) model to explore the effect of variable density on training dynamics. Our results, which combine analytic and empirical observations, show that when learning a pure harmonic function of frequency $\\}kappa$, convergence at a point $\\}x \\}in \\}Sphere\^{d-1}$ occurs in time $O(\\}kappa\^d/p(\\}x))$ where $p(\\}x)$ denotes the local density at $\\}x$. Specifically, for data in $\\}Sphere\^1$ we analytically derive the eigenfunctions of the kernel associated with the NTK for two-layer networks. We further prove convergence results for deep, fully connected networks with respect to the spectral decomposition of the NTK. Our empirical study highlights similarities and differences between deep and shallow networks in this model. (@basri2020frequency)

Alberto Bietti and Francis Bach Deep equals shallow for ReLU networks in kernel regimes In *International Conference on Learning Representations*, 2021. URL <https://openreview.net/forum?id=aDjoksTpXOP>. **Abstract:** Deep networks are often considered to be more expressive than shallow ones in terms of approximation. Indeed, certain functions can be approximated by deep networks provably more efficiently than by shallow ones, however, no tractable algorithms are known for learning such deep models. Separately, a recent line of work has shown that deep networks trained with gradient descent may behave like (tractable) kernel methods in a certain over-parameterized regime, where the kernel is determined by the architecture and initialization, and this paper focuses on approximation for such kernels. We show that for ReLU activations, the kernels derived from deep fully-connected networks have essentially the same approximation properties as their shallow two-layer counterpart, namely the same eigenvalue decay for the corresponding integral operator. This highlights the limitations of the kernel framework for understanding the benefits of such deep architectures. Our main theoretical result relies on characterizing such eigenvalue decays through differentiability properties of the kernel function, which also easily applies to the study of other kernels defined on the sphere. (@bietti2021deep)

Simone Bombari, Mohammad Hossein Amani, and Marco Mondelli Memorization and optimization in deep neural networks with minimum over-parameterization In *Advances in Neural Information Processing Systems*, volume 35, pp. 7628–7640. Curran Associates, Inc., 2022. URL <https://proceedings.neurips.cc/paper_files/paper/2022/file/323746f0ae2fbd8b6f500dc2d5c5f898-Paper-Conference.pdf>. **Abstract:** The Neural Tangent Kernel (NTK) has emerged as a powerful tool to provide memorization, optimization and generalization guarantees in deep neural networks. A line of work has studied the NTK spectrum for two-layer and deep networks with at least a layer with $\\}Omega(N)$ neurons, $N$ being the number of training samples. Furthermore, there is increasing evidence suggesting that deep networks with sub-linear layer widths are powerful memorizers and optimizers, as long as the number of parameters exceeds the number of samples. Thus, a natural open question is whether the NTK is well conditioned in such a challenging sub-linear setup. In this paper, we answer this question in the affirmative. Our key technical contribution is a lower bound on the smallest NTK eigenvalue for deep networks with the minimum possible over-parameterization: the number of parameters is roughly $\\}Omega(N)$ and, hence, the number of neurons is as little as $\\}Omega(\\}sqrt{N})$. To showcase the applicability of our NTK bounds, we provide two results concerning memorization capacity and optimization guarantees for gradient descent training. (@bombari2022memorization)

Benjamin Bowman and Guido Montúfar Spectral bias outside the training set for deep networks in the kernel regime In *Advances in Neural Information Processing Systems*, 2022. URL <https://openreview.net/forum?id=a01PL2gb7W5>. **Abstract:** We provide quantitative bounds measuring the $L\^2$ difference in function space between the trajectory of a finite-width network trained on finitely many samples from the idealized kernel dynamics of infinite width and infinite data. An implication of the bounds is that the network is biased to learn the top eigenfunctions of the Neural Tangent Kernel not just on the training set but over the entire input space. This bias depends on the model architecture and input distribution alone and thus does not depend on the target function which does not need to be in the RKHS of the kernel. The result is valid for deep architectures with fully connected, convolutional, and residual layers. Furthermore the width does not need to grow polynomially with the number of samples in order to obtain high probability bounds up to a stopping time. The proof exploits the low-effective-rank property of the Fisher Information Matrix at initialization, which implies a low effective dimension of the model (far smaller than the number of parameters). We conclude that local capacity control from the low effective rank of the Fisher Information Matrix is still underexplored theoretically. (@bowman2022spectral)

Yuan Cao, Zhiying Fang, Yue Wu, Ding-Xuan Zhou, and Quanquan Gu Towards understanding the spectral bias of deep learning In *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21*, pp. 2205–2211, August 2021. URL <https://doi.org/10.24963/ijcai.2021/304>. **Abstract:** An intriguing phenomenon observed during training neural networks is the spectral bias, which states that neural networks are biased towards learning less complex functions. The priority of learning functions with low complexity might be at the core of explaining the generalization ability of neural networks, and certain efforts have been made to provide a theoretical explanation for spectral bias. However, there is still no satisfying theoretical result justifying the underlying mechanism of spectral bias. In this paper, we give a comprehensive and rigorous explanation for spectral bias and relate it with the neural tangent kernel function proposed in recent work. We prove that the training process of neural networks can be decomposed along different directions defined by the eigenfunctions of the neural tangent kernel, where each direction has its own convergence rate and the rate is determined by the corresponding eigenvalue. We then provide a case study when the input data is uniformly distributed over the unit sphere, and show that lower degree spherical harmonics are easier to be learned by over-parameterized neural networks. Finally, we provide numerical experiments to demonstrate the correctness of our theory. Our experimental results also show that our theory can tolerate certain model misspecification in terms of the input data distribution. (@cao2019towards)

Lénaı̈c Chizat, Edouard Oyallon, and Francis Bach On lazy training in differentiable programming In *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL <https://proceedings.neurips.cc/paper_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf>. **Abstract:** In a series of recent theoretical works, it was shown that strongly over-parameterized neural networks trained with gradient-based methods could converge exponentially fast to zero training loss, with their parameters hardly varying. In this work, we show that this "lazy training" phenomenon is not specific to over-parameterized neural networks, and is due to a choice of scaling, often implicit, that makes the model behave as its linearization around the initialization, thus yielding a model equivalent to learning with positive-definite kernels. Through a theoretical analysis, we exhibit various situations where this phenomenon arises in non-convex optimization and we provide bounds on the distance between the lazy and linearized optimization paths. Our numerical experiments bring a critical note, as we observe that the performance of commonly used non-linear deep convolutional neural networks in computer vision degrades when trained in the lazy regime. This makes it unlikely that "lazy training" is behind the many successes of neural networks in difficult high dimensional tasks. (@lazy_training)

Hugo Cui, Bruno Loureiro, Florent Krzakala, and Lenka Zdeborová Generalization error rates in kernel regression: The crossover from the noiseless to noisy regime In *Advances in Neural Information Processing Systems*, 2021. URL <https://openreview.net/forum?id=Da_EHrAcfwd>. **Abstract:** In this manuscript we consider Kernel Ridge Regression (KRR) under the Gaussian design. Exponents for the decay of the excess generalization error of KRR have been reported in various works under the assumption of power-law decay of eigenvalues of the features co-variance. These decays were, however, provided for sizeably different setups, namely in the noiseless case with constant regularization and in the noisy optimally regularized case. Intermediary settings have been left substantially uncharted. In this work, we unify and extend this line of work, providing characterization of all regimes and excess error decay rates that can be observed in terms of the interplay of noise and regularization. In particular, we show the existence of a transition in the noisy setting between the noiseless exponents to its noisy values as the sample complexity is increased. Finally, we illustrate how this crossover can also be observed on real data sets. (@cui2021generalization)

Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai Gradient descent finds global minima of deep neural networks In *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of Machine Learning Research*, pp. 1675–1685. PMLR, 2019. URL <https://proceedings.mlr.press/v97/du19c.html>. **Abstract:** Gradient descent finds a global minimum in training deep neural networks despite the objective function being non-convex. The current paper proves gradient descent achieves zero training loss in polynomial time for a deep over-parameterized neural network with residual connections (ResNet). Our analysis relies on the particular structure of the Gram matrix induced by the neural network architecture. This structure allows us to show the Gram matrix is stable throughout the training process and this stability implies the global optimality of the gradient descent algorithm. We further extend our analysis to deep residual convolutional neural networks and obtain a similar convergence result. (@du2019gradient)

Simon S. Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh Gradient descent provably optimizes over-parameterized neural networks In *International Conference on Learning Representations*, 2019. URL <https://openreview.net/forum?id=S1eK3i09YQ>. **Abstract:** One of the mysteries in the success of neural networks is randomly initialized first order methods like gradient descent can achieve zero training loss even though the objective function is non-convex and non-smooth. This paper demystifies this surprising phenomenon for two-layer fully connected ReLU activated neural networks. For an $m$ hidden node shallow neural network with ReLU activation and $n$ training data, we show as long as $m$ is large enough and no two inputs are parallel, randomly initialized gradient descent converges to a globally optimal solution at a linear convergence rate for the quadratic loss function. Our analysis relies on the following observation: over-parameterization and random initialization jointly restrict every weight vector to be close to its initialization for all iterations, which allows us to exploit a strong convexity-like property to show that gradient descent converges at a global linear rate to the global optimum. We believe these insights are also useful in analyzing deep models and other first order methods. (@du2018gradient)

Costas Efthimiou and Christopher Frye *Spherical harmonics in p dimensions* World Scientific, 2014. URL <https://doi.org/10.1142/9134>. **Abstract:** The current book makes several useful topics from the theory of special functions, in particular the theory of spherical harmonics and Legendre polynomials in arbitrary dimensions, available to undergraduates studying physics or mathematics. With this audience in mind, nearly all details of the calculations and proofs are written out, and extensive background material is covered before exploring the main subject matter. (@efthimiou2014spherical)

Zhou Fan and Zhichao Wang Spectra of the conjugate kernel and neural tangent kernel for linear-width neural networks In *Advances in Neural Information Processing Systems*, volume 33, pp. 7710–7721. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper/2020/file/572201a4497b0b9f02d4f279b09ec30d-Paper.pdf>. **Abstract:** We study the eigenvalue distributions of the Conjugate Kernel and Neural Tangent Kernel associated to multi-layer feedforward neural networks. In an asymptotic regime where network width is increasing linearly in sample size, under random initialization of the weights, and for input samples satisfying a notion of approximate pairwise orthogonality, we show that the eigenvalue distributions of the CK and NTK converge to deterministic limits. The limit for the CK is described by iterating the Marcenko-Pastur map across the hidden layers. The limit for the NTK is equivalent to that of a linear combination of the CK matrices across layers, and may be described by recursive fixed-point equations that extend this Marcenko-Pastur map. We demonstrate the agreement of these asymptotic predictions with the observed spectra for both synthetic and CIFAR-10 training data, and we perform a small simulation to investigate the evolutions of these spectra over training. (@NEURIPS2020_572201a4)

Amnon Geifman, Abhay Yadav, Yoni Kasten, Meirav Galun, David Jacobs, and Basri Ronen On the similarity between the Laplace and neural tangent kernels In *Advances in Neural Information Processing Systems*, volume 33, pp. 1451–1461. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper/2020/file/1006ff12c465532f8c574aeaa4461b16-Paper.pdf>. **Abstract:** Recent theoretical work has shown that massively overparameterized neural net- works are equivalent to kernel regressors that use Neural Tangent Kernels (NTKs). Experiments show that these kernel methods perform similarly to real neural net- works. Here we show that NTK for fully connected networks with ReLU activation is closely related to the standard Laplace kernel. We show theoretically that for normalized data on the hypersphere both kernels have the same eigenfunctions and their eigenvalues decay polynomially at the same rate, implying that their Repro- ducing Kernel Hilbert Spaces (RKHS) include the same sets of functions. This means that both kernels give rise to classes of functions with the same smoothness properties. The two kernels differ for data off the hypersphere, but experiments indicate that when data is properly normalized these differences are not signiﬁcant. Finally, we provide experiments on real data comparing NTK and the Laplace kernel, along with a larger class of -exponential kernels. We show that these perform almost identically. Our results suggest that much insight about neural networks can be obtained from analysis of the well-known Laplace kernel, which has a simple closed form. 1 Introduction Neural networks with signiﬁcantly more parameters than training examples have been successfully applied to a variety of tasks. Somewhat contrary to common wisdom, these models typically generalize well to unseen data. It has been shown that in the limit of inﬁnite model size, these neural networks are equivalent to kernel regression using a family of novel Neural Tangent Kernels (NTK) \[26,2\]. NTK methods can be analyzed to explain many properties of neural networks in this limit, including their convergence in training and ability to generalize \[ 8,9,13,32\]. Recent experimental work has shown that in practice, kernel methods using NTK perform similarly, and in some cases better, than neural networks \[ 4\], and that NTK can be used to accurately predict the dynamics of neural networks \[ 1,2,7\]. This suggests that a better understanding of NTK can lead to new ways to analyze neural networks. These results raise an important question: Is NTK signiﬁcantly different from standard kernels? For the case of fully connected (FC) networks, \[ 4\] provides experimental evidence that NTK is especially effective, showing that it outperforms the Gaussian kernel on a large suite of machine learning problems. Consequently, they argue that NTK should be a (@geifman2020similarity)

Izrail Solomonovich Gradshteyn and Iosif Moiseevich Ryzhik *Table of integrals, series, and products* Academic press, 2014. URL <https://doi.org/10.1016/C2010-0-64839-5>. **Abstract:** Introduction. Elementary Functions. Indefinite Integrals of Elementary Functions. Definite Integrals of Elementary Functions. Indefinite Integrals of Special Functions. Definite Integrals of Special Functions. Special Functions. Vector Field Theory. Algebraic Inequalities. Integral Inequalities. Matrices and Related Results. Determinants. Norms. Ordinary Differential Equations. Fourier, Laplace, and Mellin Transforms. Bibliographic References. Classified Supplementary References. (@gradshteyn2014table)

Roger A. Horn and Charles R. Johnson *Matrix Analysis* Cambridge University Press, 2 edition, 2012. URL <https://doi.org/10.1017/CBO9780511810817>. **Abstract:** Linear algebra and matrix theory are fundamental tools in mathematical and physical science, as well as fertile fields for research. This new edition of the acclaimed text presents results of both classic and recent matrix analyses using canonical forms as a unifying theme, and demonstrates their importance in a variety of applications. The authors have thoroughly revised, updated, and expanded on the first edition. The book opens with an extended summary of useful concepts and facts and includes numerous new topics and features, such as: - New sections on the singular value and CS decompositions - New applications of the Jordan canonical form - A new section on the Weyr canonical form - Expanded treatments of inverse problems and of block matrices - A central role for the Von Neumann trace theorem - A new appendix with a modern list of canonical forms for a pair of Hermitian matrices and for a symmetric-skew symmetric pair - Expanded index with more than 3,500 entries for easy reference - More than 1,100 problems and exercises, many with hints, to reinforce understanding and develop auxiliary themes such as finite-dimensional quantum systems, the compound and adjugate matrices, and the Loewner ellipsoid - A new appendix provides a collection of problem-solving hints. (@Horn_Johnson_2012)

Arthur Jacot, Franck Gabriel, and Clement Hongler Neural tangent kernel: Convergence and generalization in neural networks In *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL <https://proceedings.neurips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf>. **Abstract:** At initialization, artificial neural networks (ANNs) are equivalent to Gaussian processes in the infinite-width limit, thus connecting them to kernel methods. We prove that the evolution of an ANN during training can also be described by a kernel: during gradient descent on the parameters of an ANN, the network function $f\_\\}theta$ (which maps input vectors to output vectors) follows the kernel gradient of the functional cost (which is convex, in contrast to the parameter cost) w.r.t. a new kernel: the Neural Tangent Kernel (NTK). This kernel is central to describe the generalization features of ANNs. While the NTK is random at initialization and varies during training, in the infinite-width limit it converges to an explicit limiting kernel and it stays constant during training. This makes it possible to study the training of ANNs in function space instead of parameter space. Convergence of the training can then be related to the positive-definiteness of the limiting NTK. We prove the positive-definiteness of the limiting NTK when the data is supported on the sphere and the non-linearity is non-polynomial. We then focus on the setting of least-squares regression and show that in the infinite-width limit, the network function $f\_\\}theta$ follows a linear differential equation during training. The convergence is fastest along the largest kernel principal components of the input data with respect to the NTK, hence suggesting a theoretical motivation for early stopping. Finally we study the NTK numerically, observe its behavior for wide networks, and compare it to the infinite-width limit. (@jacot2018neural)

Hui Jin, Pradeep Kr. Banerjee, and Guido Montúfar Learning curves for Gaussian process regression with power-law priors and targets In *International Conference on Learning Representations*, 2022. URL <https://openreview.net/forum?id=KeI9E-gsoB>. **Abstract:** We characterize the power-law asymptotics of learning curves for Gaussian process regression (GPR) under the assumption that the eigenspectrum of the prior and the eigenexpansion coefficients of the target function follow a power law. Under similar assumptions, we leverage the equivalence between GPR and kernel ridge regression (KRR) to show the generalization error of KRR. Infinitely wide neural networks can be related to GPR with respect to the neural network GP kernel and the neural tangent kernel, which in several cases is known to have a power-law spectrum. Hence our methods can be applied to study the generalization error of infinitely wide neural networks. We present toy experiments demonstrating the theory. (@jin2022learning)

Kedar Karhadkar, Michael Murray, Hanna Tseran, and Guido Montúfar Mildly overparameterized ReLU networks have a favorable loss landscape *arXiv:2305.19510*, 2023. **Abstract:** We study the loss landscape of both shallow and deep, mildly overparameterized ReLU neural networks on a generic finite input dataset for the squared error loss. We show both by count and volume that most activation patterns correspond to parameter regions with no bad local minima. Furthermore, for one-dimensional input data, we show most activation regions realizable by the network contain a high dimensional set of global minima and no bad local minima. We experimentally confirm these results by finding a phase transition from most regions having full rank Jacobian to many regions having deficient rank depending on the amount of overparameterization. (@karhadkar2023mildly)

B. Laurent and P. Massart *The Annals of Statistics*, 28 (5): 1302–1338, 2000. URL <https://doi.org/10.1214/aos/1015957395>. **Abstract:** We consider the problem of estimating $\\}\|s\\}\|\^2$ when $s$ belongs to some separable Hilbert space and one observes the Gaussian process $Y(t) = \\}langles, t\\}rangle + \\}sigmaL(t)$, for all $t \\}epsilon \\}mathbb{H}$,where $L$ is some Gaussian isonormal process. This framework allows us in particular to consider the classical “Gaussian sequence model” for which $\\}mathbb{H} = l_2(\\}mathbb{N}\*)$ and $L(t) = \\}sum\_{\\}lambda\\}geq1}t\_{\\}lambda}\\}varepsilon\_{\\}lambda}$, where $(\\}varepsilon\_{\\}lambda})\_{\\}lambda\\}geq1}$ is a sequence of i.i.d. standard normal variables. Our approach consists in considering some at most countable families of finite-dimensional linear subspaces of $\\}mathbb{H}$ (the models) and then using model selection via some conveniently penalized least squares criterion to build new estimators of $\\}\|s\\}\|\^2$. We prove a general nonasymptotic risk bound which allows us to show that such penalized estimators are adaptive on a variety of collections of sets for the parameter $s$, depending on the family of models from which they are built.In particular, in the context of the Gaussian sequence model, a convenient choice of the family of models allows defining estimators which are adaptive over collections of hyperrectangles, ellipsoids, $l_p$-bodies or Besov bodies.We take special care to describe the conditions under which the penalized estimator is efficient when the level of noise $\\}sigma$ tends to zero. Our construction is an alternative to the one by Efroïmovich and Low for hyperrectangles and provides new results otherwise. (@10.1214/aos/1015957395)

Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington Wide neural networks of any depth evolve as linear models under gradient descent In *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL <https://proceedings.neurips.cc/paper/2019/file/0d1a9651497a38d8b1c3871c84528bd4-Paper.pdf>. **Abstract:** A longstanding goal in deep learning research has been to precisely characterize training and generalization. However, the often complex loss landscapes of neural networks have made a theory of learning dynamics elusive. In this work, we show that for wide neural networks the learning dynamics simplify considerably and that, in the infinite width limit, they are governed by a linear model obtained from the first-order Taylor expansion of the network around its initial parameters. Furthermore, mirroring the correspondence between wide Bayesian neural networks and Gaussian processes, gradient-based training of wide neural networks with a squared loss produces test set predictions drawn from a Gaussian process with a particular compositional kernel. While these theoretical results are only exact in the infinite width limit, we nevertheless find excellent empirical agreement between the predictions of the original network and those of the linearized version even for finite practically-sized networks. This agreement is robust across different architectures, optimization methods, and loss functions. (@Lee2019WideNN-SHORT)

Jaehoon Lee, Samuel Schoenholz, Jeffrey Pennington, Ben Adlam, Lechao Xiao, Roman Novak, and Jascha Sohl-Dickstein Finite versus infinite neural networks: an empirical study In *Advances in Neural Information Processing Systems*, volume 33, pp. 15156–15172. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper/2020/file/ad086f59924fffe0773f8d0ca22ea712-Paper.pdf>. **Abstract:** We perform a careful, thorough, and large scale empirical study of the correspondence between wide neural networks and kernel methods. By doing so, we resolve a variety of open questions related to the study of infinitely wide neural networks. Our experimental results include: kernel methods outperform fully-connected finite-width networks, but underperform convolutional finite width networks; neural network Gaussian process (NNGP) kernels frequently outperform neural tangent (NT) kernels; centered and ensembled finite networks have reduced posterior variance and behave more similarly to infinite networks; weight decay and the use of a large learning rate break the correspondence between finite and infinite networks; the NTK parameterization outperforms the standard parameterization for finite width networks; diagonal regularization of kernels acts similarly to early stopping; floating point precision limits kernel performance beyond a critical dataset size; regularized ZCA whitening improves accuracy; finite network performance depends non-monotonically on width in ways not captured by double descent phenomena; equivariance of CNNs is only beneficial for narrow networks far from the kernel regime. Our experiments additionally motivate an improved layer-wise scaling for weight decay which improves generalization in finite-width networks. Finally, we develop improved best practices for using NNGP and NT kernels for prediction, including a novel ensembling technique. Using these best practices we achieve state-of-the-art results on CIFAR-10 classification for kernels corresponding to each architecture class we consider. (@10.5555/3495724.3496995)

Wonyeol Lee, Hangyeol Yu, Xavier Rival, and Hongseok Yang On correctness of automatic differentiation for non-differentiable functions In *Advances in Neural Information Processing Systems*, volume 33, pp. 6719–6730. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper_files/paper/2020/file/4aaa76178f8567e05c8e8295c96171d8-Paper.pdf>. **Abstract:** Differentiation lies at the core of many machine-learning algorithms, and is well-supported by popular autodiff systems, such as TensorFlow and PyTorch. Originally, these systems have been developed to compute derivatives of differentiable functions, but in practice, they are commonly applied to functions with non-differentiabilities. For instance, neural networks using ReLU define non-differentiable functions in general, but the gradients of losses involving those functions are computed using autodiff systems in practice. This status quo raises a natural question: are autodiff systems correct in any formal sense when they are applied to such non-differentiable functions? In this paper, we provide a positive answer to this question. Using counterexamples, we first point out flaws in often-used informal arguments, such as: non-differentiabilities arising in deep learning do not cause any issues because they form a measure-zero set. We then investigate a class of functions, called PAP functions, that includes nearly all (possibly non-differentiable) functions in deep learning nowadays. For these PAP functions, we propose a new type of derivatives, called intensional derivatives, and prove that these derivatives always exist and coincide with standard derivatives for almost all inputs. We also show that these intensional derivatives are what most autodiff systems compute or try to compute essentially. In this way, we formally establish the correctness of autodiff systems applied to non-differentiable functions. (@10.5555/3495724.3496288)

Shengqiao Li Concise formulas for the area and volume of a hyperspherical cap *Asian Journal of Mathematics & Statistics*, 4 (1): 66–70, 2010. (@li2010concise)

Yuanzhi Li and Yingyu Liang Learning overparameterized neural networks via stochastic gradient descent on structured data In *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL <https://proceedings.neurips.cc/paper_files/paper/2018/file/54fe976ba170c19ebae453679b362263-Paper.pdf>. **Abstract:** Neural networks have many successful applications, while much less theoretical understanding has been gained. Towards bridging this gap, we study the problem of learning a two-layer overparameterized ReLU neural network for multi-class classification via stochastic gradient descent (SGD) from random initialization. In the overparameterized setting, when the data comes from mixtures of well-separated distributions, we prove that SGD learns a network with a small generalization error, albeit the network has enough capacity to fit arbitrary labels. Furthermore, the analysis provides interesting insights into several aspects of learning neural networks and can be verified based on empirical studies on synthetic data and on the MNIST dataset. (@NEURIPS2018_54fe976b)

Chaoyue Liu, Libin Zhu, and Mikhail Belkin On the linearity of large non-linear models: when and why the tangent kernel is constant In *Advances in Neural Information Processing Systems*, volume 33, pp. 15954–15964. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper_files/paper/2020/file/b7ae8fecf15b8b6c3c69eceae636d203-Paper.pdf>. **Abstract:** The goal of this work is to shed light on the remarkable phenomenon of transition to linearity of certain neural networks as their width approaches infinity. We show that the transition to linearity of the model and, equivalently, constancy of the (neural) tangent kernel (NTK) result from the scaling properties of the norm of the Hessian matrix of the network as a function of the network width. We present a general framework for understanding the constancy of the tangent kernel via Hessian scaling applicable to the standard classes of neural networks. Our analysis provides a new perspective on the phenomenon of constant tangent kernel, which is different from the widely accepted "lazy training". Furthermore, we show that the transition to linearity is not a general property of wide neural networks and does not hold when the last layer of the network is non-linear. It is also not necessary for successful optimization by gradient descent. (@NEURIPS2020_b7ae8fec)

Chaoyue Liu, Libin Zhu, and Mikhail Belkin Loss landscapes and optimization in over-parameterized non-linear systems and neural networks *Applied and Computational Harmonic Analysis*, 59: 85–116, 2022. URL <https://www.sciencedirect.com/science/article/pii/S106352032100110X>. Special Issue on Harmonic Analysis and Machine Learning. **Abstract:** The success of deep learning is due, to a large extent, to the remarkable effectiveness of gradient-based optimization methods applied to large neural networks. The purpose of this work is to propose a modern view and a general mathematical framework for loss landscapes and efficient optimization in over-parameterized machine learning models and systems of non-linear equations, a setting that includes over-parameterized deep neural networks. Our starting observation is that optimization problems corresponding to such systems are generally not convex, even locally. We argue that instead they satisfy PL$\^\*$, a variant of the Polyak-Lojasiewicz condition on most (but not all) of the parameter space, which guarantees both the existence of solutions and efficient optimization by (stochastic) gradient descent (SGD/GD). The PL$\^\*$ condition of these systems is closely related to the condition number of the tangent kernel associated to a non-linear system showing how a PL$\^\*$-based non-linear theory parallels classical analyses of over-parameterized linear equations. We show that wide neural networks satisfy the PL$\^\*$ condition, which explains the (S)GD convergence to a global minimum. Finally we propose a relaxation of the PL$\^\*$ condition applicable to "almost" over-parameterized systems. (@LIU202285)

Andrea Montanari and Yiqiao Zhong The interpolation phase transition in neural networks: Memorization and generalization under lazy training *The Annals of Statistics*, 50 (5): 2816–2847, 2022. URL <https://doi.org/10.1214/22-AOS2211>. **Abstract:** Modern neural networks are often operated in a strongly overparametrized regime: they comprise so many parameters that they can interpolate the training set, even if actual labels are replaced by purely random ones. Despite this, they achieve good prediction error on unseen data: interpolating the training set does not lead to a large generalization error. Further, overparametrization appears to be beneficial in that it simplifies the optimization landscape. Here, we study these phenomena in the context of two-layers neural networks in the neural tangent (NT) regime. We consider a simple data model, with isotropic covariates vectors in d dimensions, and N hidden neurons. We assume that both the sample size n and the dimension d are large, and they are polynomially related. Our first main result is a characterization of the eigenstructure of the empirical NT kernel in the overparametrized regime Nd≫n. This characterization implies as a corollary that the minimum eigenvalue of the empirical NT kernel is bounded away from zero as soon as Nd≫n and, therefore, the network can exactly interpolate arbitrary labels in the same regime. Our second main result is a characterization of the generalization error of NT ridge regression including, as a special case, min-ℓ2 norm interpolation. We prove that, as soon as Nd≫n, the test error is well approximated by the one of kernel ridge regression with respect to the infinite-width kernel. The latter is in turn well approximated by the error of polynomial ridge regression, whereby the regularization parameter is increased by a "self-induced" term related to the high-degree components of the activation function. The polynomial degree depends on the sample size and the dimension (in particular on logn/logd). (@montanari2022interpolation)

Michael Murray, Hui Jin, Benjamin Bowman, and Guido Montúfar Characterizing the spectrum of the NTK via a power series expansion In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=Tvms8xrZHyR>. **Abstract:** Under mild conditions on the network initialization we derive a power series expansion for the Neural Tangent Kernel (NTK) of arbitrarily deep feedforward networks in the infinite width limit. We provide expressions for the coefficients of this power series which depend on both the Hermite coefficients of the activation function as well as the depth of the network. We observe faster decay of the Hermite coefficients leads to faster decay in the NTK coefficients and explore the role of depth. Using this series, first we relate the effective rank of the NTK to the effective rank of the input-data Gram. Second, for data drawn uniformly on the sphere we study the eigenvalues of the NTK, analyzing the impact of the choice of activation function. Finally, for generic data and activation functions with sufficiently fast Hermite coefficient decay, we derive an asymptotic upper bound on the spectrum of the NTK. (@murray2023characterizing)

Paul Nevai, Tamás Erdélyi, and Alphonse P Magnus Generalized jacobi weights, christoffel functions, and jacobi polynomials *SIAM Journal on Mathematical Analysis*, 25 (2): 602–614, 1994. **Abstract:** The authors obtain upper bounds for Jacobi polynomials which are uniform in all the parameters involved and which contain explicit constants. This is done by a combination of some results on generalized Christoffel functions and some estimates of Jacobi polynomials in terms of Christoffel functions. (@nevai1994generalized)

Quynh Nguyen On the proof of global convergence of gradient descent for deep ReLU networks with linear widths In *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pp. 8056–8062. PMLR, 2021. URL <https://proceedings.mlr.press/v139/nguyen21a.html>. **Abstract:** We give a simple proof for the global convergence of gradient descent in training deep ReLU networks with the standard square loss, and show some of its improvements over the state-of-the-art. In particular, while prior works require all the hidden layers to be wide with width at least $\\}Omega(N\^8)$ ($N$ being the number of training samples), we require a single wide layer of linear, quadratic or cubic width depending on the type of initialization. Unlike many recent proofs based on the Neural Tangent Kernel (NTK), our proof need not track the evolution of the entire NTK matrix, or more generally, any quantities related to the changes of activation patterns during training. Instead, we only need to track the evolution of the output at the last hidden layer, which can be done much more easily thanks to the Lipschitz property of ReLU. Some highlights of our setting: (i) all the layers are trained with standard gradient descent, (ii) the network has standard parameterization as opposed to the NTK one, and (iii) the network has a single wide layer as opposed to having all wide hidden layers as in most of NTK-related results. (@nguyenrelu)

Quynh Nguyen and Marco Mondelli Global convergence of deep networks with one wide layer followed by pyramidal topology In *Advances in Neural Information Processing Systems*, volume 33, pp. 11961–11972. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper/2020/file/8abfe8ac9ec214d68541fcb888c0b4c3-Paper.pdf>. **Abstract:** Recent works have shown that gradient descent can find a global minimum for over-parameterized neural networks where the widths of all the hidden layers scale polynomially with $N$ ($N$ being the number of training samples). In this paper, we prove that, for deep networks, a single layer of width $N$ following the input layer suffices to ensure a similar guarantee. In particular, all the remaining layers are allowed to have constant widths, and form a pyramidal topology. We show an application of our result to the widely used LeCun’s initialization and obtain an over-parameterization requirement for the single wide layer of order $N\^2.$ (@marco)

Quynh Nguyen, Marco Mondelli, and Guido Montúfar Tight bounds on the smallest eigenvalue of the neural tangent kernel for deep ReLU networks In *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pp. 8119–8129. PMLR, 18–24 Jul 2021. URL <https://proceedings.mlr.press/v139/nguyen21g.html>. **Abstract:** A recent line of work has analyzed the theoretical properties of deep neural networks via the Neural Tangent Kernel (NTK). In particular, the smallest eigenvalue of the NTK has been related to the memorization capacity, the global convergence of gradient descent algorithms and the generalization of deep nets. However, existing results either provide bounds in the two-layer setting or assume that the spectrum of the NTK matrices is bounded away from 0 for multi-layer networks. In this paper, we provide tight bounds on the smallest eigenvalue of NTK matrices for deep ReLU nets, both in the limiting case of infinite widths and for finite widths. In the finite-width setting, the network architectures we consider are fairly general: we require the existence of a wide layer with roughly order of $N$ neurons, $N$ being the number of data samples; and the scaling of the remaining layer widths is arbitrary (up to logarithmic factors). To obtain our results, we analyze various quantities of independent interest: we give lower bounds on the smallest singular value of hidden feature matrices, and upper bounds on the Lipschitz constant of input-output feature maps. (@nguyen2021tight)

Samet Oymak and Mahdi Soltanolkotabi Toward moderate overparameterization: Global convergence guarantees for training shallow neural networks *IEEE Journal on Selected Areas in Information Theory*, 1 (1): 84–105, 2020. URL <https://doi.org/10.1109/JSAIT.2020.2991332>. **Abstract:** Many modern neural network architectures are trained in an overparameterized regime where the parameters of the model exceed the size of the training dataset. Sufficiently overparameterized neural network architectures in principle have the capacity to fit any set of labels including random noise. However, given the highly nonconvex nature of the training landscape it is not clear what level and kind of overparameterization is required for first order methods to converge to a global optima that perfectly interpolate any labels. A number of recent theoretical works have shown that for very wide neural networks where the number of hidden units is polynomially large in the size of the training data gradient descent starting from a random initialization does indeed converge to a global optima. However, in practice much more moderate levels of overparameterization seems to be sufficient and in many cases overparameterized models seem to perfectly interpolate the training data as soon as the number of parameters exceed the size of the training data by a constant factor. Thus there is a huge gap between the existing theoretical literature and practical experiments. In this paper we take a step towards closing this gap. Focusing on shallow neural nets and smooth activations, we show that (stochastic) gradient descent when initialized at random converges at a geometric rate to a nearby global optima as soon as the square-root of the number of network parameters exceeds the size of the training data. Our results also benefit from a fast convergence rate and continue to hold for non-differentiable activations such as Rectified Linear Units (ReLUs). (@Oymak2019TowardMO)

Boris Rubin Inversion and characterization of the hemispherical transform *Journal d’Analyse Mathématique*, 77: 105–128, 1999. URL <https://doi.org/10.1007/BF02791259>. **Abstract:** Seismic post-stack inversion is one of the best techniques for effective reservoir characterization. This studyintends to articulate the application of Model-Based Inversion (MBI) and Probabilistic Neural Networks (PNN) for theidentification of reservoir properties i.e. porosity estimation. MBI technique is applied to observe the low impedancezone at the porous reservoir formation. PNN is a geostatistical technique that transforms the impedance volume intoporosity volume. Inverted porosity is estimated to observe the spatial distribution of porosity in the Lower Goru sandreservoir beyond the well data control. The result of inverted porosity is compared with that of well-computed porosity.The estimated inverted porosity ranges from 13-13.5% which shows a correlation of 99.63% with the computed porosityof the Rehmat-02 well. The observed low impedance and high porosity cube at the targeted horizon suggest that it couldbe a probable potential sand channel. Furthermore, the results of seismic post-stack inversion and geostatistical analysisindicate a very good agreement with each other. Hence, the seismic post-stack inversion technique can effectively beapplied to estimate the reservoir properties for further prospective zones identification, volumetric estimation and futureexploration. (@rubin1999inversion)

J. Schur Bemerkungen zur Theorie der beschränkten Bilinearformen mit unendlich vielen Veränderlichen *Journal für die reine und angewandte Mathematik*, 140: 1–28, 1911. URL <http://eudml.org/doc/149352>. **Abstract:** Article Bemerkungen zur Theorie der beschränkten Bilinearformen mit unendlich vielen Veränderlichen. was published on January 1, 1911 in the journal Journal für die reine und angewandte Mathematik (volume 1911, issue 140). (@Schur1911)

Robert T Seeley Spherical harmonics *The American Mathematical Monthly*, 73 (4P2): 115–121, 1966. URL <https://doi.org/10.1080/00029890.1966.11970927>. **Abstract:** One of the challenges in 3D shape matching arises from the fact that in many applications, models should be considered to be the same if they differ by a rotation. Consequently, when comparing two models, a similarity metric implicitly provides the measure of similarity at the optimal alignment. Explicitly solving for the optimal alignment is usually impractical. So, two general methods have been proposed for addressing this issue: (1) Every model is represented using rotation invariant descriptors. (2) Every model is described by a rotation dependent descriptor that is aligned into a canonical coordinate system defined by the model. In this paper, we describe the limitations of canonical alignment and discuss an alternate method, based on spherical harmonics, for obtaining rotation invariant representations. We describe the properties of this tool and show how it can be applied to a number of existing, orientation dependent descriptors to improve their matching performance. The advantages of this tool are two-fold: First, it improves the matching performance of many descriptors. Second, it reduces the dimensionality of the descriptor, providing a more compact representation, which in turn makes comparing two models more efficient. (@seeley1966spherical)

Joel A. Tropp User-friendly tail bounds for sums of random matrices *Foundations of computational mathematics*, 12: 389–434, 2012. URL <https://doi.org/10.1007/s10208-011-9099-z>. **Abstract:** This paper presents new probability inequalities for sums of independent, random, self-adjoint matrices. These results place simple and easily verifiable hypotheses on the summands, and they deliver strong conclusions about the large-deviation behavior of the maximum eigenvalue of the sum. Tail bounds for the norm of a sum of random rectangular matrices follow as an immediate corollary. The proof techniques also yield some information about matrix-valued martingales. In other words, this paper provides noncommutative generalizations of the classical bounds associated with the names Azuma, Bennett, Bernstein, Chernoff, Hoeffding, and McDiarmid. The matrix inequalities promise the same diversity of application, ease of use, and strength of conclusion that have made the scalar inequalities so valuable. (@tropp2012user)

Maksim Velikanov and Dmitry Yarotsky Explicit loss asymptotics in the gradient descent training of neural networks In *Advances in Neural Information Processing Systems*, volume 34, pp. 2570–2582. Curran Associates, Inc., 2021. URL <https://proceedings.neurips.cc/paper_files/paper/2021/file/14faf969228fc18fcd4fcf59437b0c97-Paper.pdf>. **Abstract:** Current theoretical results on optimization trajectories of neural networks trained by gradient descent typically have the form of rigorous but potentially loose bounds on the loss values. In the present work we take a different approach and show that the learning trajectory can be characterized by an explicit asymptotic at large training times. Specifically, the leading term in the asymptotic expansion of the loss behaves as a power law $L(t) \\}sim t\^{-\\}xi}$ with exponent $\\}xi$ expressed only through the data dimension, the smoothness of the activation function, and the class of function being approximated. Our results are based on spectral analysis of the integral operator representing the linearized evolution of a large network trained on the expected loss. Importantly, the techniques we employ do not require specific form of a data distribution, for example Gaussian, thus making our findings sufficiently universal. (@velikanov2021explicit)

Roman Vershynin *High-dimensional probability: An introduction with applications in data science*, volume 47 Cambridge university press, 2018. URL <https://doi.org/10.1017/9781108231596>. **Abstract:** © 2018, Cambridge University Press Let us summarize our findings. A random projection of a set T in R n onto an m-dimensional subspace approximately preserves the geometry of T if m ⪆ d ( T ) . For... (@vershynin2018high)

Bo Xie, Yingyu Liang, and Le Song Diverse neural network learns true target functions In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, pp. 1216–1224. PMLR, 2017. URL <https://proceedings.mlr.press/v54/xie17a.html>. **Abstract:** Neural networks are a powerful class of functions that can be trained with simple gradient descent to achieve state-of-the-art performance on a variety of applications. Despite their practical success, there is a paucity of results that provide theoretical guarantees on why they are so effective. Lying in the center of the problem is the difficulty of analyzing the non-convex loss function with potentially numerous local minima and saddle points. Can neural networks corresponding to the stationary points of the loss function learn the true target function? If yes, what are the key factors contributing to such nice optimization properties? In this paper, we answer these questions by analyzing one-hidden-layer neural networks with ReLU activation, and show that despite the non-convexity, neural networks with diverse units have no spurious local minima. We bypass the non-convexity issue by directly analyzing the first order optimality condition, and show that the loss can be made arbitrarily small if the minimum singular value of the "extended feature matrix" is large enough. We make novel use of techniques from kernel methods and geometric discrepancy, and identify a new relation linking the smallest singular value to the spectrum of a kernel function associated with the activation function and to the diversity of the units. Our results also suggest a novel regularization function to promote unit diversity for potentially better generalization. (@xie2017diverse)

Ziqing Xie, Li-Lian Wang, and Xiaodan Zhao On exponential convergence of Gegenbauer interpolation and spectral differentiation *Mathematics of Computation*, 82 (282): 1017–1036, 2013. URL <https://doi.org/10.1090/S0025-5718-2012-02645-7>. **Abstract:** This paper is devoted to a rigorous analysis of exponential convergence of polynomial interpolation and spectral differentiation based on the Gegenbauer-Gauss and Gegenbauer-Gauss-Lobatto points, when the underlying function is analytic on and within an ellipse. Sharp error estimates in the maximum norm are derived. (@xie2013exponential)

Difan Zou and Quanquan Gu An improved analysis of training over-parameterized deep neural networks In *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL <https://proceedings.neurips.cc/paper/2019/file/6a61d423d02a1c56250dc23ae7ff12f3-Paper.pdf>. **Abstract:** A recent line of research has shown that gradient-based algorithms with random initialization can converge to the global minima of the training loss for over-parameterized (i.e., sufficiently wide) deep neural networks. However, the condition on the width of the neural network to ensure the global convergence is very stringent, which is often a high-degree polynomial in the training sample size $n$ (e.g., $O(n\^{24})$). In this paper, we provide an improved analysis of the global convergence of (stochastic) gradient descent for training deep neural networks, which only requires a milder over-parameterization condition than previous work in terms of the training sample size and other problem-dependent parameters. The main technical contributions of our analysis include (a) a tighter gradient lower bound that leads to a faster convergence of the algorithm, and (b) a sharper characterization of the trajectory length of the algorithm. By specializing our result to two-layer (i.e., one-hidden-layer) neural networks, it also provides a milder over-parameterization condition than the best-known result in prior work. (@zou2019improved)

Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu Gradient descent optimizes over-parameterized deep ReLU networks *Machine learning*, 109 (3): 467–492, 2020. URL <https://doi.org/10.1007/s10994-019-05839-6>. **Abstract:** We study the problem of training deep neural networks with Rectified Linear Unit (ReLU) activation function using gradient descent and stochastic gradient descent. In particular, we study the binary classification problem and show that for a broad family of loss functions, with proper random weight initialization, both gradient descent and stochastic gradient descent can find the global minima of the training loss for an over-parameterized deep ReLU network, under mild assumption on the training data. The key idea of our proof is that Gaussian random initialization followed by (stochastic) gradient descent produces a sequence of iterates that stay inside a small perturbation region centering around the initial weights, in which the empirical loss function of deep ReLU networks enjoys nice local curvature properties that ensure the global convergence of (stochastic) gradient descent. Our theoretical results shed light on understanding the optimization for deep learning, and pave the way for studying the optimization dynamics of training modern deep neural networks. (@zou2020gradient)

</div>

# Background material

## Concentration bounds

In order to bound the smallest eigenvalue of the finite-width NTK in terms of the expected, or infinite width NTK, we use the following matrix Chernoff bound variant.

<div id="lem:matrix-chernoff" class="lemma" markdown="1">

**Lemma 2**. *Let \\(R > 0\\), and let \\({\bm{Z}}_1, \cdots, {\bm{Z}}_m \in \mathbb{R}^{n \times n}\\) be iid symmetric random matrices such that \\(0 \preceq {\bm{Z}}_1 \preceq R{\bm{I}}\\) almost surely. Then \\[\mathbb{P}\left(\lambda_{\min}\left(\frac{1}{m}\sum_{j = 1}^m {\bm{Z}}_j \right) \leq \frac{1}{2}\lambda_{\min}\left(\mathbb{E}[{\bm{Z}}_1] \right) \right) \leq n \exp\left(-\frac{Cm\lambda_{\min}(\mathbb{E}[{\bm{Z}}_1]) }{R} \right).\\] Here \\(C > 0\\) is a universal constant.*

</div>

<div class="proof" markdown="1">

*Proof.* By Theorem 1.1 of `\cite{tropp2012user}`{=latex}, for all \\(\delta > 0\\) \\[\begin{aligned}
        &\mathbb{P}\left(\lambda_{\min}\left(\frac{1}{m} \sum_{j = 1}^m {\bm{Z}}_j\right) \leq (1 - \delta)\lambda_{\min}(\mathbb{E}[{\bm{Z}}_1])  \right)  \\&=\mathbb{P}\left(\lambda_{\min}\left( \sum_{j = 1}^m {\bm{Z}}_j \right) \leq (1 - \delta)\lambda_{\min}\left(\sum_{j = 1}^m \mathbb{E}[{\bm{Z}}_j] \right) \right)\\
        &\leq n \left(\frac{e^{-\delta} }{(1 - \delta)^{1 - \delta}} \right)^{\frac{1}{R}\lambda_{\min}\left(\sum_{j = 1}^m \mathbb{E}[{\bm{Z}}_j]\right) }\\
        &= n \left(\frac{e^{-\delta} }{(1 - \delta)^{1 - \delta}} \right)^{\frac{m}{R}\lambda_{\min}\left(\mathbb{E}[{\bm{Z}}_1]\right) }.
    
\end{aligned}\\] Let \\(\delta = \frac{1}{2}\\) and let \\(C = \frac{1}{2}\log\left(\frac{e}{2}\right) > 0\\). Substituting into the above bound, we obtain \\[\begin{aligned}
        \mathbb{P}\left(\lambda_{\min}\left(\frac{1}{m}\sum_{j = 1}^m {\bm{Z}}_j \right) \leq \frac{1}{2} \lambda_{\min}(\mathbb{E}[{\bm{Z}}_1])\right) &\leq n \left(\frac{2}{e} \right)^{\frac{m}{2R}\lambda_{\min}(\mathbb{E}[{\bm{Z}}_1]) }\\
        &= n \exp\left(-\frac{C m \lambda_{\min}(\mathbb{E}[{\bm{Z}}_1]) }{R} \right).
    
\end{aligned}\\] ◻

</div>

Some of our NTK bounds will depend on the operator norm of the input data matrix \\({\bm{X}}\\), so it will be helpful to upper bound \\(\|{\bm{X}}\|\\) with high probability.

<div id="lemma:input-data-conditioning" class="lemma" markdown="1">

**Lemma 3**. *Let \\(\epsilon > 0\\). Let \\({\bm{X}}= [{\bm{x}}_1, \cdots, {\bm{x}}_n] \in \mathbb{R}^{d \times n}\\) be a random matrix whose columns are independent and uniformly distributed on \\(\mathbb{S}^{d - 1}\\). Then with probability at least \\(1 - \epsilon\\), \\[\|{\bm{X}}\|^2 \lesssim 1 + \frac{n + \log \frac{1}{\epsilon} }{d}.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* We use a covering argument. Fix \\({\bm{u}}\in \mathbb{S}^{d-1}\\) and \\({\bm{v}}\in \mathbb{S}^{n - 1}\\). By Lemma 2.2 of `\cite{ball1997elementary}`{=latex}, for each \\(i \in [n]\\) and \\(t \geq 0\\), \\[\mathbb{P}(|\langle {\bm{u}}, {\bm{x}}_i \rangle| \geq t) \leq 2\exp\left(-\frac{dt^2}{2}\right).\\] In other words \\(\|\langle {\bm{u}}, {\bm{x}}_i \rangle\|_{\psi_2} \lesssim \frac{1}{\sqrt{d}}\\). Then by Hoeffding’s inequality, for all \\(t \geq 0\\) \\[\begin{aligned}
        \mathbb{P}(|{\bm{u}}^T{\bm{X}}{\bm{v}}|\geq t) &= \mathbb{P}\left(\left|\sum_{i = 1}^n [{\bm{v}}]_i \langle {\bm{u}}, {\bm{x}}_i \rangle\right| \geq t \right)\nonumber\\
        &\leq 2\exp\left(-C_1 dt^2 \right)\label{eqn:uXv-concentration},
    
\end{aligned}\\] where \\(C_1 > 0\\) is a constant.

Let \\({\bm{u}}_1, \cdots, {\bm{u}}_M\\) be a \\(\left(\frac{1}{4}\right)\\)-covering of \\(\mathbb{S}^{d - 1}\\). That is, \\({\bm{u}}_1, \cdots, {\bm{u}}_M\\) are a set of points in \\(\mathbb{S}^{d - 1}\\) such that for all \\({\bm{u}}\in \mathbb{S}^{d - 1}\\), there exists \\(j \in [M]\\) such that \\(\|{\bm{u}}- {\bm{u}}_j\| \leq \frac{1}{4}\\). Since the \\(\left(\frac{1}{4}\right)\\)-covering number of \\(\mathbb{S}^{d - 1}\\) is at most \\(12^d\\) `\citep[see][Corollary 4.2.13]{vershynin2018high}`{=latex}, we can take \\(M \leq 12^d\\). Similarly, let \\({\bm{u}}_1, \cdots, {\bm{u}}_N\\) be a \\(\left(\frac{1}{4}\right)\\)-covering of \\(\mathbb{S}^{n - 1}\\) with \\(N \leq 12^n\\). By applying a union bound to <a href="#eqn:uXv-concentration" data-reference-type="eqref" data-reference="eqn:uXv-concentration">[eqn:uXv-concentration]</a>, we obtain \\[\begin{aligned}
        \mathbb{P}(|{\bm{u}}_j^T {\bm{X}}{\bm{v}}_k| \geq t \text{ for some $j \in [M], k \in [N]$}) &\leq 2(12^{d + n})\exp\left(-C_1 d t^2\right).
    
\end{aligned}\\] Hence if \\[t = \sqrt{\frac{(d + n)\log 12 + \log \frac{2}{\epsilon}}{d} } ,\\] then \\[\mathbb{P}(|{\bm{u}}_j^T {\bm{X}}{\bm{v}}_k| \leq t \text{ for all $j \in [M], k \in [N]$}) \geq 1 -\epsilon.\\] Let us condition on this event for the rest of the proof. Now suppose that \\({\bm{u}}\in \mathbb{S}^{d - 1}\\) and \\({\bm{v}}\in \mathbb{S}^{n - 1}\\). By construction there exist \\(j \in [M]\\) and \\(k \in [N]\\) such that \\(\|{\bm{u}}- {\bm{u}}_j\| \leq \frac{1}{4}\\) and \\(\|{\bm{v}}- {\bm{v}}_k\| \leq \frac{1}{4}\\). Then \\[\begin{aligned}
        |{\bm{u}}^T {\bm{X}}{\bm{v}}| &\leq |{\bm{u}}_j^T {\bm{X}}{\bm{v}}_k| + |({\bm{u}}- {\bm{u}}_j)^T{\bm{X}}{\bm{v}}_k| + |{\bm{u}}^T {\bm{X}}({\bm{v}}- {\bm{v}}_k)|\\
        &\leq t + \|{\bm{u}}- {\bm{u}}_j\| \cdot \|{\bm{v}}_k\| \cdot \|{\bm{X}}\| + \|{\bm{u}}\| \cdot \|{\bm{X}}\| \cdot \|{\bm{v}}- {\bm{v}}_k\|\\
        &\leq t + \frac{1}{4}\|{\bm{X}}\| + \frac{1}{4}\|{\bm{X}}\|\\
        &= t + \frac{1}{2}\|{\bm{X}}\|.
    
\end{aligned}\\] Since this holds for all \\({\bm{u}}\in \mathbb{S}^{d - 1}\\) and \\({\bm{v}}\in \mathbb{S}^{n - 1}\\), we obtain \\[\|{\bm{X}}\| \leq t + \frac{1}{2}\|{\bm{X}}\|.\\] Rearranging yields \\[\begin{aligned}
        \|{\bm{X}}\|^2 &\leq 4t^2\\
        &\lesssim 1 + \frac{n + \log \frac{1}{\epsilon} }{d}.
    
\end{aligned}\\] ◻

</div>

## Spherical harmonics [sec:app-spherical]

Here we review some preliminaries on spherical harmonics necessary for our main results. For further details we refer the reader to `\citet{efthimiou2014spherical}`{=latex} and `\citet[Chapter 5]{axler2013harmonic}`{=latex}. Let \\(L^2(\mathbb{S}^{d - 1})\\) denote the Hilbert space of real-valued, square-integrable functions on the sphere \\(\mathbb{S}^{d - 1}\\), equipped with the inner product \\[\langle g, h \rangle = \int_{\mathbb{S}^{d - 1}} g({\bm{x}}) {h({\bm{x}})} \; dS({\bm{x}}),\\] where \\(dS\\) is the uniform probability measure on \\(\mathbb{S}^{d - 1}\\). We let \\(\mathcal{C}(\mathbb{S}^{d - 1}) \subset L^2(\mathbb{S}^{d - 1})\\) denote the subset of functions which are continuous. We say that a function \\(g: \mathbb{R}^d \to \mathbb{R}\\) is *harmonic* if it is twice continuously differentiable and \\[\sum_{r = 1}^d \frac{\partial^2 g}{\partial^2 x_r}({\bm{x}}) = 0\\] for all \\({\bm{x}}\in \mathbb{S}^{d - 1}\\). We say that a polynomial \\(g: \mathbb{R}^d \to \mathbb{R}\\) is *homogeneous* if there exists \\(r \in \mathbb{Z}_{\geq 0}\\) such that \\[g(\lambda {\bm{x}}) = \lambda^r g({\bm{x}})\\] for all \\(\lambda \in \mathbb{R}\\) and \\({\bm{x}}\in \mathbb{R}^d\\). Let \\(\mathcal{H}_r^d\\) denote the vector space of degree \\(r\\) harmonic homogeneous polynomials on \\(d\\) variables, viewed as functions \\(\mathbb{S}^{d - 1} \to \mathbb{R}\\). Each space \\(\mathcal{H}_r^d\\) is a finite-dimensional vector space, with \\[\begin{aligned}
    \dim(\mathcal{H}_r^d) &= \binom{r + d - 1}{d - 1} - \binom{r + d - 3}{d - 1}\\
    &= \frac{2r + d - 2}{r} \binom{r + d - 3}{d - 2}.
\end{aligned}\\] For \\(\nu \geq 0\\) and \\(r \in \mathbb{N}\\), we define the *Gegenbauer polynomials* \\(C_r^{\nu}\\) by \\[C_r^{\nu}(t) = \sum_{k = 0}^{\lfloor r/2 \rfloor }(-1)^k \frac{\Gamma(r - k + \nu) }{\Gamma(\nu)\Gamma(k + 1)\Gamma(r - 2k + 1) }(2t)^{r - 2k}.\\]

There exists an orthonormal basis of \\(\mathcal{H}_r^d\\) consisting of functions \\(Y_{r,s}^d\\), \\(1 \leq s \leq \dim(\mathcal{H}_r^d)\\), known as *spherical harmonics*. The spherical harmonics in \\(\mathcal{H}_r^d\\) satisfy the addition formula \\[\begin{aligned}
\label{eqn:addition-formula}
    \sum_{s = 1}^{\dim(\mathcal{H}_r^d)} Y_{r,s}^d({\bm{x}}) Y_{r, s}^d({\bm{x}}') &= \frac{\dim(\mathcal{H}_r^d)C_r^{(d-2)/2}(\langle {\bm{x}}, {\bm{x}}' \rangle)\Gamma(r + 1)\Gamma(d - 2) }{\Gamma(r + d - 2)}\nonumber\\
    &= \frac{(2r + d - 2)C_r^{(d - 2)/2}(\langle {\bm{x}}, {\bm{x}}' \rangle) }{d - 2}
\end{aligned}\\] for all \\({\bm{x}}, {\bm{x}}' \in \mathbb{S}^{d-1}\\). In particular, from the identity \\(C_r^{\nu}(1) = \frac{\Gamma(2\nu + r) }{\Gamma(2\nu)\Gamma(r + 1)}\\) it follows that \\[\begin{aligned}
    \sum_{s = 1}^{\dim(\mathcal{H}_r^d) }|Y_{r, s}^d({\bm{x}})|^2 &= \dim(\mathcal{H}_r^d).
\end{aligned}\\] We can orthogonally decompose \\(L^2(\mathbb{S}^{d - 1})\\) into a direct sum of the spaces of spherical harmonics: \\[L^2(\mathbb{S}^{d - 1}) = \bigoplus_{r = 1}^{\infty} \mathcal{H}_r^d.\\] That is, the spaces \\(\mathcal{H}_r^d\\) are orthogonal and their linear span is dense in \\(L^2(\mathbb{S}^{d - 1})\\).

<div id="lemma:addition-formula-bound" class="lemma" markdown="1">

**Lemma 4**. *Let \\(\delta > 0\\) and suppose that \\({\bm{x}}, {\bm{x}}' \in \mathbb{S}^{d - 1}\\) satisfy \\(\|{\bm{x}}- {\bm{x}}'\|, \|{\bm{x}}+ {\bm{x}}'\| \geq \delta\\). If \\(R \in \mathbb{Z}_{\geq 0}\\), and \\(\beta \in \{0, 1\}\\), then \\[\left|\sum_{r = 0}^R \sum_{s= 1}^{\dim(\mathcal{H}_{2r + \beta}^d) }Y_{2r + \beta, s}^d({\bm{x}}) Y_{2r + \beta, s}^d({\bm{x}}')\right| \lesssim \left(\frac{\|{\bm{x}}- {\bm{x}}'\|^2}{2} \right)^{-(d-2)/4 } \binom{2R + \beta + d - 1}{d - 1}^{1/2}.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Let us define \\[P({\bm{x}}, {\bm{x}}') := \sum_{r = 0}^R \sum_{s= 1}^{\dim(\mathcal{H}_{2r + \beta}^d) }Y_{2r + \beta, s}^d({\bm{x}}) Y_{2r + \beta, s}^d({\bm{x}}').\\] By the addition formula (<a href="#eqn:addition-formula" data-reference-type="ref" data-reference="eqn:addition-formula">[eqn:addition-formula]</a>), \\[\begin{aligned}
       |P({\bm{x}}, {\bm{x}}')| &= \left|\sum_{r = 0}^R \frac{(4r + 2\beta + d - 2)C_{2r + \beta}^{(d-2)/2}(\langle {\bm{x}}, {\bm{x}}' \rangle)  }{d - 2} \right|\nonumber\\
       &\lesssim \sum_{r = 0}^R \frac{(r + d)|C_{2r + \beta}^{(d - 2)/2}(\langle {\bm{x}}, {\bm{x}}' \rangle)|  }{d}.\label{eqn:sum-gegenbauer}
   
\end{aligned}\\] In order to bound the right hand side of the above equation, we will need a bound for the Gegenbauer polynomials \\(C_{2r + \beta}^{(d - 2)/2}\\). By Theorem 1 of `\cite{nevai1994generalized}`{=latex} (see also equation 2.8 of `\citealt{xie2013exponential}`{=latex}), for all \\(\nu \geq \frac{1}{2}\\), \\(r \geq 0\\), and \\(t \in [0, 1)\\), \\[\begin{aligned}
       (1 - t^2)^{\nu} C_r^{\nu}(t)^2 &\leq \frac{2e(2 + \sqrt{2}\nu) }{\pi}\frac{2^{1 - 2\nu}\pi }{\Gamma(\nu)^2 }\frac{\Gamma(r + 2\nu) }{\Gamma(r + 1)(r + \nu) }\\
       &\lesssim \frac{\nu\Gamma(r + 2\nu) }{2^{2\nu}(r + \nu)\Gamma(\nu)^2 \Gamma(r + 1) }.
   
\end{aligned}\\] Rearranging the above expression yields \\[\begin{aligned}
       |C_r^{\nu}(t)| &\lesssim \frac{\nu^{1/2}\Gamma(r + 2\nu)^{1/2}  }{2^{\nu}(r + \nu)^{1/2}\Gamma(\nu)\Gamma(r + 1)^{1/2}(1 - t^2)^{\nu/2} }.
   
\end{aligned}\\] We now substitute the above bound into (<a href="#eqn:sum-gegenbauer" data-reference-type="ref" data-reference="eqn:sum-gegenbauer">[eqn:sum-gegenbauer]</a>): \\[\begin{aligned}
       |P({\bm{x}}, {\bm{x}}')| &\lesssim \sum_{r = 0}^R \frac{(r + d)\left(\frac{d - 2}{2}\right)^{1/2}\Gamma\left(2r + \beta + d - 2\right)^{1/2}   }{d 2^{(d - 2)/2 }\left(2r + \beta + \frac{d - 2}{2}\right)^{1/2}\Gamma\left(\frac{d - 2}{2}\right)\Gamma(2r + \beta +  1)^{1/2}(1 - \langle {\bm{x}}, {\bm{x}}' \rangle^2)^{(d - 2)/4}  }\\
       &\lesssim \frac{1}{(1 - \langle {\bm{x}}, {\bm{x}}'\rangle^2)^{(d - 2)/4} }\sum_{r = 0}^R \left(\frac{r + d}{d}\right)^{1/2} \frac{\Gamma(2r + \beta + d - 2)^{1/2} }{2^{(d - 2)/2}\Gamma\left(\frac{d - 2}{2}\right)\Gamma(2r + \beta + 1)^{1/2}}.
   
\end{aligned}\\] The expression inside the sum is increasing as a function of \\(r\\), so the above expression is bounded above by \\[\begin{aligned}
       &\frac{1}{(1 - \langle {\bm{x}}, {\bm{x}}' \rangle^2)^{(d - 2)/4} }\left(\frac{R + d}{d}\right)^{1/2}\frac{\Gamma(2R + \beta + d - 2)^{1/2} }{2^{(d - 2)/2}\Gamma\left(\frac{d - 2}{2}\right)\Gamma(2R + \beta + 1)^{1/2} }\nonumber\\
       &\lesssim \frac{1}{d^{1/2}(1 - \langle {\bm{x}}, {\bm{x}}' \rangle^2)^{(d - 2)/4} } \frac{\Gamma(2R + \beta + d - 1)^{1/2} }{2^{(d - 2)/2}\Gamma\left(\frac{d - 2}{2}\right)\Gamma(2R + \beta + 1)^{1/2} }.\label{eqn:gegenbauer-sum-2}
   
\end{aligned}\\] By Stirling’s approximation, \\[\begin{aligned}
       2^{(d - 2)/2}\Gamma\left(\frac{d - 2}{2}\right) &\asymp 2^{(d - 2)/2}\left(\frac{d - 2}{2}\right)^{(d - 3)/2 } e^{-(d - 2)/2}\\
       &= (d - 2)^{(d -3 )/2}e^{-(d - 2)/2}\\
       &\asymp d^{-1/4}(d - 2)^{(d - 1.5)/2 }e^{-(d - 2)/2}\\
       &\asymp d^{-1/4}\Gamma(d - 1)^{1/2}.
   
\end{aligned}\\] Substituing this into (<a href="#eqn:gegenbauer-sum-2" data-reference-type="ref" data-reference="eqn:gegenbauer-sum-2">[eqn:gegenbauer-sum-2]</a>) yields \\[\begin{aligned}
       |P({\bm{x}}, {\bm{x}}')| &\leq \frac{1}{d^{1/4}(1 - \langle {\bm{x}}, {\bm{x}}' \rangle^2)^{(d-2)/4} }\frac{\Gamma(2R + \beta + d - 1)^{1/2} }{\Gamma(d - 1)^{1/2}\Gamma(2R + \beta + 1)^{1/2} }\\
       &\asymp \frac{d^{1/4}}{(R + d)^{1/2} (1 - \langle {\bm{x}}, {\bm{x}}' \rangle^2)^{(d-2)/4} } \frac{\Gamma(2R + \beta + d)^{1/2} }{\Gamma(d)\Gamma(2R + \beta + 1)^{1/2} }\\
       &= \frac{d^{1/4}}{(R + d)^{1/2} (1 - \langle {\bm{x}}, {\bm{x}}' \rangle^2)^{(d-2)/4} }\binom{2R + \beta + d - 1}{d - 1}^{1/2}\\
       &\lesssim \frac{1}{(1 - \langle {\bm{x}}, {\bm{x}}' \rangle^2)^{(d-2)/4} }\binom{2R + \beta + d - 1}{d - 1}^{1/2}
   
\end{aligned}\\] Since \\({\bm{x}}, {\bm{x}}' \in \mathbb{S}^{d - 1}\\), \\[\begin{aligned}
       1 - \langle {\bm{x}}, {\bm{x}}' \rangle^2 &= (1 + \langle {\bm{x}}, {\bm{x}}' \rangle)(1 - \langle {\bm{x}}, {\bm{x}}' \rangle)\\
       &= \frac{1}{4}\|{\bm{x}}+ {\bm{x}}'\|^2 \|{\bm{x}}- {\bm{x}}'\|^2\\
       &\gtrsim \frac{1}{4} \delta^4.
   
\end{aligned}\\] To conclude, we rewrite \\[\begin{aligned}
       |P({\bm{x}}, {\bm{x}}')| &\lesssim \left(\frac{\delta^4}{2} \right)^{-(d-2)/4 } \binom{2R + \beta + d - 1}{d - 1}^{1/2}.
   
\end{aligned}\\] ◻

</div>

# Preliminaries on hemisphere transforms [app:hemisphere]

Let \\(\mathcal{M}(\mathbb{S}^{d - 1})\\) denote the vector space of signed Radon measures on \\(\mathbb{S}^{d - 1}\\). We denote the total variation of \\(\mu\\) by \\(|\mu|\\). We have a natural inclusion \\(L^2(\mathbb{S}^{d - 1}) \subset \mathcal{M}(\mathbb{S}^{d -1})\\) by associating a function \\(g\\) to a signed measure \\(\mu\\) defined by \\[\mu(E) = \int_E g({\bm{x}}) dS({\bm{x}}).\\] If \\(\mu \in \mathcal{M}(\mathbb{S}^{d -1 })\\) and \\(g \in \mathcal{C}(\mathbb{S}^{d - 1})\\), we define the pairing \\(\langle \mu, g \rangle\\) by \\[\langle \mu, g \rangle = \int_{\mathbb{S}^{d-1}} {g({\bm{x}})}d\mu({\bm{x}}).\\] This agrees with the usual definition of the inner product on \\(L^2(\mathbb{S}^{d-1})\\) when \\(\mu \in L^2(\mathbb{S}^{d-1})\\).

Fix \\(\psi \in \{\sqrt{d}\sigma, \dot{\sigma}\}\\). If \\(\mu \in \mathcal{M}(\mathbb{S}^{d - 1})\\), we define its *hemisphere transform* `\citep{rubin1999inversion}`{=latex} \\(T_{\psi}\mu: \mathbb{S}^{d - 1} \to \mathbb{R}\\) by \\[(T_{\psi}\mu)(\bm{\xi}) = \int_{\mathbb{S}^{d - 1}}\psi(\langle \bm{\xi}, {\bm{x}}\rangle) d\mu({\bm{x}}).\\] As is the case with many integral transforms, a hemisphere transform increases the regularity of the functions it is applied to.

<div class="lemma" markdown="1">

**Lemma 5**. *If \\(\mu \in \mathcal{M}(\mathbb{S}^{d - 1})\\), then \\(T_{\psi}\mu \in L^2(\mathbb{S}^{d - 1})\\). If \\(g \in L^2(\mathbb{S}^{d - 1})\\), then \\(T_{\psi}g \in \mathcal{C}(\mathbb{S}^{d - 1})\\).*

</div>

<div class="proof" markdown="1">

*Proof.* Suppose that \\(\mu \in \mathcal{M}(\mathbb{S}^{d - 1})\\). Then \\[\begin{aligned}
        \int_{\mathbb{S}^{d-1}} (T_{\psi}\mu)(\bm{\xi})^2 dS(\bm{\xi}) &= \int_{\mathbb{S}^{d-1}}\left|\int_{\mathbb{S}^{d-1}}\psi(\langle \bm{\xi}, {\bm{x}}\rangle) d\mu({\bm{x}}) \right|^2 dS(\bm{\xi})\\
        &\leq \int_{\mathbb{S}^{d-1}}\left|\int_{\mathbb{S}^{d-1}} \psi(\langle \bm{\xi}, {\bm{x}}\rangle) d|\mu|({\bm{x}}) \right|^2dS(\bm{\xi})\\
        &= \int_{\mathbb{S}^{d-1}} \int_{\mathbb{S}^{d-1}}\int_{\mathbb{S}^{d-1}}\psi(\langle \bm{\xi}, {\bm{x}}\rangle)\psi(\langle \bm{\xi}, {\bm{x}}' \rangle)d|\mu|({\bm{x}})d|\mu|({\bm{x}}')dS(\bm{\xi})\\
        &\leq\int_{\mathbb{S}^{d-1}} \int_{\mathbb{S}^{d-1}}\int_{\mathbb{S}^{d-1}}d^2 d|\mu|({\bm{x}})d|\mu|({\bm{x}}')dS(\bm{\xi})\\
        &= |\mu|(\mathbb{S}^{d-1})^2 d^2\\
        &< \infty,
    
\end{aligned}\\] so \\(T \mu \in L^2(\mathbb{S}^{d-1})\\).

Now suppose that \\(g \in L^2(\mathbb{S}^{d -1})\\) and \\(\psi = \dot{\sigma}\\). Suppose that \\(\bm{\xi}, \bm{\xi'} \in \mathbb{S}^{d -1}\\), and observe that \\[\begin{aligned}
        dS(\{{\bm{x}}\in \mathbb{S}^{d - 1}: \langle {\bm{x}}, \bm{\xi} \rangle > 0, \langle {\bm{x}}, \bm{\xi}'\rangle \leq 0\}) &= \frac{1}{2 \pi}\arccos(\langle \bm{\xi}, \bm{\xi}' \rangle).
    
\end{aligned}\\] Similarly, \\[\begin{aligned}
        dS(\{{\bm{x}}\in \mathbb{S}^{d - 1}: \langle {\bm{x}}, \bm{\xi} \rangle \leq 0, \langle {\bm{x}}, \bm{\xi}'\rangle > 0\}) &= \frac{1}{2 \pi}\arccos(\langle \bm{\xi}, \bm{\xi}' \rangle),
    
\end{aligned}\\] so \\[\begin{aligned}
        dS(\{{\bm{x}}\in \mathbb{S}^{d - 1}: \dot{\sigma}(\langle {\bm{x}}, \bm{\xi} \rangle)  \neq \dot{\sigma}(\langle {\bm{x}}, \bm{\xi}' \rangle)   ) &= \frac{1}{ \pi}\arccos(\langle \bm{\xi}, \bm{\xi}' \rangle).
    
\end{aligned}\\] We apply this calculation to bound the distance between \\(T_{\psi}g(\bm{\xi})\\) and \\(T_{\psi}g(\bm{\xi'})\\): \\[\begin{aligned}
        |T_{\psi}g(\bm{\xi}) - T_{\psi}g(\bm{\xi'})|&= \left|\int_{\mathbb{S}^{d-1}} \dot{\sigma}(\langle {\bm{x}}, \bm{\xi} \rangle) g({\bm{x}})dS({\bm{x}}) - \int_{\mathbb{S}^{d-1}} \dot{\sigma}(\langle {\bm{x}}, \bm{\xi}' \rangle) g({\bm{x}})dS({\bm{x}}) \right|\\
        &\leq \int_{\mathbb{S}^{d-1}}|\dot{\sigma}(\langle {\bm{x}}, \bm{\xi} \rangle) - \dot{\sigma}(\langle {\bm{x}}, \bm{\xi}' \rangle)|g({\bm{x}}) dS({\bm{x}})\\
        &\leq \|g\|_{L^{2}}\left(\int_{\mathbb{S}^{d-1}}|\dot{\sigma}(\langle {\bm{x}}, \bm{\xi} \rangle) - \dot{\sigma}(\langle {\bm{x}}, \bm{\xi}' \rangle)|^2 dS({\bm{x}})\right)^{1/2}\\
        &= \|g\|_{L^2} \left(dS(\{{\bm{x}}\in \mathbb{S}^{d - 1}: \dot{\sigma}(\langle {\bm{x}}, \bm{\xi} \rangle) \neq \dot{\sigma}(\langle {\bm{x}}, \bm{\xi}' \rangle)) \right)^{1/2}\\
        &= \frac{1}{\pi}\|g\|_{L^2}\sqrt{\arccos(\langle \bm{\xi}, \bm{\xi'} \rangle)}.
    
\end{aligned}\\] Here the third line follows from Cauchy-Schwarz. As \\(\bm{\xi} \to \bm{\xi}'\\), \\(\arccos(\langle \bm{\xi}, \bm{\xi}' \rangle) \to 0\\) and so \\(|T_{\psi}g(\bm{\xi}) - T_{\psi}g(\bm{\xi'})| \to 0\\). Therefore, \\(T_{\psi}g \in \mathcal{C}(\mathbb{S}^{d-1})\\).

Finally suppose that \\(g \in L^2(\mathbb{S}^{d -1 })\\) and \\(\psi = \sqrt{d}\sigma\\). For all \\(\bm{\xi} \in \mathbb{S}^{d - 1}\\), \\[|d\sigma(\langle {\bm{x}}, \bm{\xi} \rangle) g({\bm{x}})| \leq \sqrt{d}|g({\bm{x}})| \in L^1(\mathbb{S}^{d -1 }).\\] So by the dominated convergence theorem, for all \\(\bm{\xi}' \in \mathbb{S}^{d - 1}\\), \\[\begin{aligned}
        \lim_{\bm{\xi} \to \bm{\xi}'} T_{\psi}g(\bm{\xi}) &= \lim_{\bm{\xi} \to \bm{\xi'} } \int_{\mathbb{S}^{d-1}}\sqrt{d}\sigma(\langle {\bm{x}}, \bm{\xi} \rangle) g({\bm{x}})dS({\bm{x}})\\
        &= \int_{\mathbb{S}^{d -1 }} \lim_{\bm{\xi} \to \bm{\xi}'} \sqrt{d}\sigma(\langle {\bm{x}}, \bm{\xi}\rangle) g({\bm{x}})dS({\bm{x}})\\
        &= \int_{\mathbb{S}^{d - 1}}\sqrt{d}\sigma(\langle {\bm{x}}, \bm{\xi}' \rangle)g({\bm{x}})dS({\bm{x}})\\
        &= T_{\psi}g(\bm{\xi}').
    
\end{aligned}\\] Therefore \\(T_{\psi}g \in \mathcal{C}(\mathbb{S}^{d - 1})\\). ◻

</div>

By the above lemma, for any \\(\mu \in \mathcal{M}(\mathbb{S}^{d -1})\\) and \\(g \in L^2(\mathbb{S}^{d - 1})\\), the expressions \\(\langle T_{\psi}\mu, g \rangle\\) and \\(\langle \mu, T_{\psi}g \rangle\\) are well-defined and finite. In fact, they are equal to each other.

<div id="lemma:T-self-adjoint" class="lemma" markdown="1">

**Lemma 6**. *Suppose that \\(\mu \in \mathcal{M}(\mathbb{S}^{d - 1})\\) and \\(g \in L^2(\mathbb{S}^{d - 1})\\). Then \\[\langle T_{\psi}\mu, g \rangle = \langle \mu, T_{\psi}g \rangle.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* We compute \\[\begin{aligned}
        \langle T_{\psi} \mu, g \rangle &= \int_{\mathbb{S}^{d-1}} (T_{\psi}\mu)(\bm{\xi}) {g(\bm{\xi})} dS(\xi)\\
        &= \int_{\mathbb{S}^{d-1}}\int_{\mathbb{S}^{d-1}} \psi(\langle {\bm{x}}, \bm{\xi} \rangle) {g(\bm{\xi})} d\mu({\bm{x}}) dS(\bm{\xi})\\
        &= \int_{\mathbb{S}^{d-1}}\int_{\mathbb{S}^{d-1}} \psi(\langle {\bm{x}}, \bm{\xi} \rangle) {g(\bm{\xi})} dS(\bm{\xi})d\mu({\bm{x}})\\
        &= \int_{\mathbb{S}^{d-1}} {T_{\psi}g({\bm{x}})} d\mu({\bm{x}})\\
        &= \langle \mu, T_{\psi}g \rangle.
    
\end{aligned}\\] It remains to justify the change in order of integration in the third line. This follows from Fubini’s theorem and the calculation \\[\begin{aligned}
        \int_{\mathbb{S}^{d-1}}\int_{\mathbb{S}^{d-1}} |\psi(\langle {\bm{x}}, \bm{\xi} \rangle)g(\bm{\xi})|dS(\bm{\xi}) d|\mu|({\bm{x}})
        &\leq \int_{\mathbb{S}^{d-1}} \int_{\mathbb{S}^{d-1}}\sqrt{d}|g(\bm{\xi})|dS(\bm{\xi})d|\mu|(\bm{x})\\
        &= \int_{\mathbb{S}^{d-1}}\sqrt{d}\|g\|_{L^1} d|\mu|({\bm{x}})
        \\&= \sqrt{d}\|g\|_{L^1} |\mu|(\mathbb{S}^{d -1 })\\
        &< \infty,
    
\end{aligned}\\] where the last line follows since \\(g \in L^2(\mathbb{S}^{d-1}) \subset L^1(\mathbb{S}^{d-1})\\). ◻

</div>

In order to characterize how a hemisphere transform acts on \\(L^2(\mathbb{S}^{d -1 })\\) and in particular on the spherical harmonics, we will use the *Funk-Hecke formula* `\citep[see][]{seeley1966spherical}`{=latex} which states that a certain class of integral operators on \\(\mathbb{S}^{d - 1}\\) has an eigendecomposition of spherical harmonics.

<div id="lem:funk-hecke" class="lemma" markdown="1">

**Lemma 7** (Funk-Hecke formula). *Let \\(\psi: [-1, 1] \to \mathbb{R}\\) be a measurable function such that \\[\int_{-1}^1 |\psi(t)|(1 - t^2)^{(d - 3)/2}dt < \infty.\\] Then for all \\(g \in \mathcal{H}_r^d\\) \\[\begin{aligned}
        \int_{\mathbb{S}^{d - 1}} \psi(\langle {\bm{x}}, \bm{\xi}\rangle)g({\bm{x}}) dS({\bm{x}}) &= c_{r,d}g(\bm{\xi}),
    
\end{aligned}\\] where \\[\begin{aligned}
        c_{r,d} &= \frac{\Gamma(r + 1)\Gamma(d - 2)\Gamma\left(\frac{d}{2}\right)  }{\sqrt{\pi}\Gamma(d - 2 + r)\Gamma\left(\frac{d - 1}{2}\right) } \int_{-1}^1 \psi(t)C_r^{(d - 2)/2}(t)(1 - t^2)^{(d - 3)/2}dt.
    
\end{aligned}\\]*

</div>

We will now use the Funk-Hecke formula to compute the coefficients \\(c_{r, d}\\) in the cases where \\(\psi = \sqrt{d}\sigma\\) and \\(\psi = \dot{\sigma}\\). In the following calculations we will use the *Legendre duplication formula* \\[\Gamma(z)\Gamma\left(z + \frac{1}{2} \right) = 2^{1 - 2z}\sqrt{\pi}\Gamma(2z)\\] and *Euler’s reflection formula* \\[\Gamma(1 - z)\Gamma(z) = \frac{\pi}{\sin \pi z}.\\]

<div id="lem:gradshteyn" class="lemma" markdown="1">

**Lemma 8**. *For all \\(d \geq 3\\) and \\(r \geq 0\\), \\[\int_0^1 C_r^{(d-2)/2}(t)(1 - t^2)^{(d-3)/2}dt = \frac{\sqrt{\pi}\Gamma(d + r - 2)\Gamma\left(\frac{d - 1}{2}\right)  }{2\Gamma(d - 2) \Gamma(r + 1)\Gamma\left(1 - \frac{r}{2}\right)\Gamma\left(\frac{d + r}{2} \right) }.\\] and \\[\int_0^1 t C_r^{(d - 2)/2}(t)(1 - t^2)^{(d - 3)/2}dt = \frac{\sqrt{\pi}\Gamma(d + r - 2)\Gamma\left(\frac{d - 1}{2}\right) }{4\Gamma(d - 2)\Gamma(r + 1)\Gamma\left(\frac{3 - r}{2}\right)\Gamma\left(\frac{d + r + 1}{2}\right) }.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* We apply the following identity `\citep[see][Equation 7.311.2]{gradshteyn2014table}`{=latex}: \\[\begin{aligned}
        \int_0^1 t^{r + 2\rho}C_r^{\nu}(t)(1 - t^2)^{\nu - 1/2}dt &= \frac{\Gamma(2 \nu+r) \Gamma(2 \rho+r+1) \Gamma\left(\nu+\frac{1}{2}\right) \Gamma\left(\rho+\frac{1}{2}\right)}{2^{r+1} \Gamma(2 \nu) \Gamma(2 \rho+1) r ! \Gamma(r+\nu+\rho+1)}.
    
\end{aligned}\\] By the Legendre duplication formula, we have \\[\Gamma\left(\rho + \frac{1}{2}\right)\Gamma(\rho + 1) = 2^{-2\rho}\sqrt{\pi}\Gamma(2\rho +1)\\] so we can rewrite the above equation as \\[\begin{aligned}
        \int_0^1 t^{r + 2\rho}C_r^{\nu}(t)(1 - t^2)^{\nu - 1/2}dt &= \frac{\sqrt{\pi} \Gamma(2 \nu + r)\Gamma(2\rho + r + 1)\Gamma\left(\nu + \frac{1}{2}\right) }{2^{2\rho + r + 1}\Gamma(2\nu)\Gamma(\rho + 1) \Gamma(r + 1)\Gamma(r + \nu + \rho + 1) }. \label{eqn:gradshteyn}
    
\end{aligned}\\] Substituting \\(\rho = -r/2\\) and \\(\nu = (d - 2)/2\\) into (<a href="#eqn:gradshteyn" data-reference-type="ref" data-reference="eqn:gradshteyn">[eqn:gradshteyn]</a>) yields \\[\begin{aligned}
        \int_0^1 C_r^{(d - 2)/2}(t)(1 - t^2)^{(d - 3)/2}dt &= \frac{\sqrt{\pi}\Gamma(d + r - 2)\Gamma\left(\frac{d - 1}{2}\right) }{2\Gamma(d - 2)\Gamma\left(1 - \frac{r}{2}\right)\Gamma(r + 1)\Gamma\left(\frac{d + r}{2}\right) } , 
    
\end{aligned}\\] which establishes the first identity of the claim.

Substituting \\(\rho = (1 - r)/2\\) and \\(\nu = (d - 2)/2\\) into (<a href="#eqn:gradshteyn" data-reference-type="ref" data-reference="eqn:gradshteyn">[eqn:gradshteyn]</a>) yields \\[\begin{aligned}
        \int_0^1 C_r^{(d - 2)/2}(t) (1 - t^2)^{(d - 3)/2}dt = \frac{\sqrt{\pi}\Gamma(d + r - 2)\Gamma\left(\frac{d - 1}{2}\right) }{4\Gamma(d - 2)\Gamma\left(\frac{3 - r}{2}\right)\Gamma(r + 1)\Gamma\left(\frac{d + r + 1}{2}\right) } , 
    
\end{aligned}\\] which establishes the second identity of the claim. ◻

</div>

<div id="lemma:eigendecomposition-sign" class="lemma" markdown="1">

**Lemma 9**. *Suppose that \\(g \in \mathcal{H}_r^d\\) and \\(d \geq 3\\). Then for all \\(r \geq 0\\), \\(T_{\dot{\sigma}}g = c_{r,d}g\\), where \\[\begin{aligned}
        c_{r,d} &= \frac{\Gamma\left(\frac{d}{2}\right)}{2\Gamma\left(1 - \frac{r}{2}\right)\Gamma\left(\frac{r}{2} + \frac{d}{2}\right)  }.
    
\end{aligned}\\] Moreover, if \\(0 \leq r \leq R\\), then \\[\begin{aligned}
        |c_{2r + 1, d}| &\geq  \frac{\Gamma\left(\frac{d}{2}\right)\Gamma\left(\frac{2R + 1}{2}\right) }{2\pi \Gamma\left(\frac{d + 2R + 1}{2}\right) }.
    
\end{aligned}\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Let \\(g \in \mathcal{H}_r^d\\). By Lemma <a href="#lem:funk-hecke" data-reference-type="ref" data-reference="lem:funk-hecke">7</a>, \\[T_{\dot{\sigma}} g = c_{r, d}g,\\] where \\[\begin{aligned}
        c_{r,d} &= \frac{\Gamma(r + 1)\Gamma(d - 2)\Gamma\left(\frac{d}{2}\right)  }{\sqrt{\pi}\Gamma(d - 2 + r)\Gamma\left(\frac{d - 1}{2}\right) } \int_{-1}^1 \dot{\sigma}(t) C_r^{(d - 2)/2}(t)(1 - t^2)^{(d - 3)/2}dt\\
        &= \frac{\Gamma(r + 1)\Gamma(d - 2)\Gamma\left(\frac{d}{2}\right)  }{\sqrt{\pi}\Gamma(d - 2 + r)\Gamma\left(\frac{d - 1}{2}\right) } \int_{0}^1 C_r^{(d - 2)/2}(t)(1 - t^2)^{(d - 3)/2}dt.
    
\end{aligned}\\] By Lemma <a href="#lem:gradshteyn" data-reference-type="ref" data-reference="lem:gradshteyn">8</a>, this is equal to \\[\begin{aligned}
        \frac{\Gamma(r + 1)\Gamma(d - 2)\Gamma\left(\frac{d}{2}\right)  }{\sqrt{\pi}\Gamma(d - 2 + r)\Gamma\left(\frac{d - 1}{2}\right) } \cdot \frac{\sqrt{\pi}\Gamma(d + r - 2)\Gamma\left(\frac{d - 1}{2}\right) }{2\Gamma(d - 2)\Gamma(r + 1)\Gamma\left(1 - \frac{r}{2}\right)\Gamma\left(\frac{d + r}{2}\right)  } &= \frac{\Gamma\left(\frac{d}{2}\right) }{2\Gamma\left(1 - \frac{r}{2}\right)\Gamma\left(\frac{d + r}{2}\right) }
    
\end{aligned}\\] as claimed.

Now we proceed with the second statement. We claim that whenever \\(0 \leq r \leq R\\), \\[|c_{2R + 1, d}| \leq |c_{2r + 1, d}|.\\] We prove this by induction on \\(R\\). For the base case \\(R = r\\), the claim trivially holds. Now suppose that the claim holds for some \\(R \geq r\\). Then \\[\begin{aligned}
        |c_{2(R + 1) + 1, d}| &= \left|\frac{\Gamma\left(\frac{d}{2}\right) }{2\Gamma\left(1 - \frac{2R + 3}{2}\right)\Gamma\left(\frac{2R + 3}{2} + \frac{d}{2}\right)  }\right|\\
        &= \left|\frac{\left(-\frac{2R + 1}{2}\right)
 \Gamma\left(\frac{d}{2}\right) }{2\Gamma\left(1 - \frac{2R + 1}{2}\right)\left(\frac{2R + 1}{2} + \frac{d}{2}\right)
 \Gamma\left(\frac{2R + 1}{2} + \frac{d}{2}\right)  }\right|\\
 &= |c_{2R + 1, d}| \frac{2R + 1}{2R + 1 + d}\\
 &\leq |c_{2R + 1, d}|\\
 &\leq |c_{2r + 1, d}| . 
    
\end{aligned}\\] Hence by induction \\(|c_{2R + 1, d}| \leq |c_{2r + 1, d}|\\) for all \\(0 \leq r \leq R\\). Now suppose that \\(0 \leq r \leq R\\). By Euler’s reflection formula, \\[\begin{aligned}
        c_{2R + 1, d} &= \frac{\Gamma\left(\frac{d}{2} \right) }{2\Gamma\left(1 - \frac{2R + 1}{2}\right) \Gamma\left(\frac{2R + 1}{2} + \frac{d}{2}\right) }\\
        &= \frac{\Gamma\left(\frac{d}{2}\right)\sin\left(\pi \frac{2R + 1}{2} \right)\Gamma\left(\frac{2R + 1}{2}\right) }{2\pi \Gamma\left(\frac{2R + 1}{2} + \frac{d}{2}\right) }\\
        &= \frac{\Gamma\left(\frac{d}{2}\right)(-1)^R \Gamma\left(\frac{2R + 1}{2}\right) }{2\pi \Gamma\left(\frac{2R + 1}{2} + \frac{d}{2}\right) }
    
\end{aligned}\\] so \\[\begin{aligned}
        |c_{2r +1 , d}| &\geq |c_{2R + 1, d}|\\
        &= \frac{\Gamma\left(\frac{d}{2}\right)\Gamma\left(\frac{2R + 1}{2}\right) }{2\pi \Gamma\left(\frac{d + 2R + 1}{2}\right) }.
    
\end{aligned}\\] ◻

</div>

<div id="lem:eigendecomposition-relu" class="lemma" markdown="1">

**Lemma 10**. *Suppose that \\(g \in \mathcal{H}_r^d\\) and \\(d \geq 3\\). Then \\(T_{\sqrt{d}\sigma}g = c_{r, d}g\\), where \\[c_{r, d} = \frac{\sqrt{d}\Gamma\left(\frac{d}{2}\right) }{4\Gamma\left(\frac{3 - r}{2}\right)\Gamma\left(\frac{d + r + 1}{2}\right)  }.\\] Moreover, if \\(0 \leq r \leq R\\), then \\[|c_{2r, d}| \geq \frac{\sqrt{d}\Gamma\left(\frac{d}{2}\right) \Gamma\left(\frac{2R - 1}{2}\right) }{4\pi \Gamma\left(\frac{d + 2R + 1}{2}\right) }.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* The proof is analogous to that of Lemma <a href="#lemma:eigendecomposition-sign" data-reference-type="ref" data-reference="lemma:eigendecomposition-sign">9</a>. Let \\(g \in \mathcal{H}_r^d\\). By Lemma <a href="#lem:funk-hecke" data-reference-type="ref" data-reference="lem:funk-hecke">7</a>, \\[T_{\sqrt{d}\sigma}g = c_{r, d} g,\\] where \\[\begin{aligned}
        c_{r, d} &= \frac{\Gamma(r + 1)\Gamma(d - 2)\Gamma\left(\frac{d}{2}\right)  }{\sqrt{\pi}\Gamma(d - 2 + r)\Gamma\left(\frac{d - 1}{2}\right) } \int_{-1}^1 \sqrt{d}\sigma(t) C_r^{(d - 2)/2}(t)(1 - t^2)^{(d - 3)/2}dt\\
        &= \frac{\sqrt{d}\Gamma(r + 1)\Gamma(d - 2)\Gamma\left(\frac{d}{2}\right)  }{\sqrt{\pi}\Gamma(d - 2 + r)\Gamma\left(\frac{d - 1}{2}\right) } \int_0^1 t C_r^{(d - 2)/2}(t)(1 - t^2)^{(d - 3)/2}dt.
    
\end{aligned}\\] By Lemma <a href="#lem:gradshteyn" data-reference-type="ref" data-reference="lem:gradshteyn">8</a>, this is equal to \\[\begin{aligned}
        \frac{\sqrt{d}\Gamma(r + 1)\Gamma(d - 2)\Gamma\left(\frac{d}{2}\right)  }{\sqrt{\pi}\Gamma(d - 2 + r)\Gamma\left(\frac{d - 1}{2}\right) } \cdot \frac{\sqrt{\pi}\Gamma(d + r - 2)\Gamma\left(\frac{d - 1}{2}\right) }{4\Gamma(d - 2)\Gamma(r + 1)\Gamma\left(\frac{3 - r}{2}\right)\Gamma\left(\frac{d + r + 1}{2}\right) } &= \frac{\sqrt{d}\Gamma\left(\frac{d}{2}\right) }{4\Gamma\left(\frac{3 - r}{2}\right)\Gamma\left(\frac{d + r + 1}{2}\right)  }
    
\end{aligned}\\] as claimed.

We claim that whenever \\(0 \leq r \leq R\\), \\[|c_{2R, d}| \leq |c_{2r, d}|.\\] We prove this by induction on \\(R\\). For the base case \\(R = r\\), the claim trivially holds. Now suppose that the claim holds for some \\(R \geq r\\). Then \\[\begin{aligned}
        |c_{2(R + 1)}| &= \left|\frac{\sqrt{d}\Gamma\left(\frac{d}{2} \right) }{4\Gamma\left(\frac{1 - 2R}{2} \right)\Gamma\left(\frac{d + 2R + 3}{2}\right)  }  \right|\\
        &= \left|\frac{\left(\frac{1 - 2R }{2}\right)\sqrt{d}\Gamma\left(\frac{d}{2}\right)}{4 \Gamma\left(\frac{1 - 2R}{2}\right)\left(\frac{d + 2R + 1}{2}\right)\Gamma\left(\frac{d + 2R + 1}{2}\right) }\right|\\
        &= c_{2R} \frac{|2R - 1|}{d + 2R + 1}\\
        &\leq c_{2R}.
    
\end{aligned}\\] Hence by induction \\(|c_{2R}| \leq |c_{2r}|\\) for all \\(0 \leq r \leq R\\). Now suppose that \\(0 \leq r \leq R\\). By Euler’s reflection formula, \\[\begin{aligned}
        c_{2R, d} &= \frac{\sqrt{d}\Gamma\left(\frac{d}{2}\right) }{4\Gamma\left(\frac{3 - 2R}{2}\right)\Gamma\left(\frac{d + 2R + 1}{2}\right) }\\
        &= \frac{\sqrt{d} \Gamma\left(\frac{d}{2}\right)\sin\left(\pi \frac{2R - 1}{2} \right)\Gamma\left(\frac{2R - 1}{2}\right) }{4\pi \Gamma\left(\frac{d + 2R + 1}{2} \right) }\\
        &= \frac{(-1)^{R + 1} \sqrt{d}\Gamma\left(\frac{d}{2}\right)\Gamma\left(\frac{2R - 1}{2}\right) }{4\pi \Gamma\left(\frac{d + 2R + 1}{2}\right) }
    
\end{aligned}\\] so \\[\begin{aligned}
        |c_{2r, d}| &\geq |c_{2R, d}|\\
        &= \frac{\sqrt{d}\Gamma\left(\frac{d}{2}\right) \Gamma\left(\frac{2R - 1}{2}\right) }{4\pi \Gamma\left(\frac{d + 2R + 1}{2}\right) }.
    
\end{aligned}\\] ◻

</div>

# Proofs for Section <a href="#section:shallow" data-reference-type="ref" data-reference="section:shallow">3</a> [proofs-for-section-sectionshallow]

First we observe the connection between the smallest eigenvalue of the expected NTK when the weights are drawn uniformly over the sphere versus as Gaussian.

<div id="lem:spherical-gaussian-equivalence" class="lemma" markdown="1">

**Lemma 11**. *If \\({\bm{X}}\in \mathbb{R}^{d_0 \times n}\\), then \\[\lambda_{\min}\left(\mathbb{E}_{{\bm{w}}\sim \mathcal{N}(\bm{0}_d, {\bm{I}}_d) }\left[ \sigma\left( {\bm{X}}^T {\bm{w}}\right) \sigma\left({\bm{w}}^T {\bm{X}}\right) \right] \right) = d_0\lambda_{\min}\left(\mathbb{E}_{{\bm{u}}\sim U(\mathbb{S}^{d_0 - 1}) }\left[\sigma\left( {\bm{X}}^T {\bm{u}}\right)\sigma\left({\bm{u}}^T {\bm{X}}\right) \right] \right).\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Since the distribution of \\({\bm{w}}\\) is rotationally invariant, we can decompose \\({\bm{w}}= \alpha {\bm{u}}\\), where \\(\alpha = \|{\bm{w}}\|\\), \\({\bm{u}}\\) is uniformly distributed on \\(\mathbb{S}^{d_0 -1 }\\), and \\(\alpha\\) and \\({\bm{u}}\\) are independent. Then \\[\begin{aligned}
        \lambda_{\min}\left(\mathbb{E}_{{\bm{w}}\sim \mathcal{N}(\bm{0}_d, {\bm{I}}_d) }\left[ \sigma\left( {\bm{X}}^T {\bm{w}}\right) \sigma\left({\bm{w}}^T {\bm{X}}\right) \right] \right) &= \lambda_{\min}\left(\mathbb{E}\left[ \sigma\left( {\bm{X}}^T {\bm{w}}\right) \sigma\left({\bm{w}}^T {\bm{X}}\right) \right] \right)\\
        &= \lambda_{\min}\left(\mathbb{E}\left[\alpha^2 \sigma\left( {\bm{X}}^T {\bm{u}}\right) \sigma\left({\bm{u}}^T{\bm{X}}\right) \right] \right)\\
        &= \lambda_{\min}\left(\mathbb{E}\left[\alpha^2\right]\mathbb{E}\left[ \sigma\left( {\bm{X}}^T {\bm{u}}\right) \sigma\left({\bm{u}}^T{\bm{X}}\right) \right] \right)\\
        &= d_0 \lambda_{\min}\left(\mathbb{E}\left[ \sigma\left( {\bm{X}}^T {\bm{u}}\right) \sigma\left({\bm{u}}^T{\bm{X}}\right) \right]\right). 
    
\end{aligned}\\] ◻

</div>

Lemma <a href="#lem:spherical-gaussian-equivalence" data-reference-type="ref" data-reference="lem:spherical-gaussian-equivalence">11</a> is useful in that studying the expected NTK in the shallow setting for uniform weights here will prove more convenient than working directly with Gaussian weights.

## Proof of Lemma <a href="#lemma:K1-inf" data-reference-type="ref" data-reference="lemma:K1-inf">[lemma:K1-inf]</a> [app:lemma:K1-inf]

<div class="proof" markdown="1">

*Proof.* By the scale-invariance of \\(\dot{\sigma}\\), \\[\begin{aligned}
        \lambda_1 = \lambda_{\min}\left(\mathbb{E}_{{\bm{u}}\sim  \mathcal{N}(\bm{0}_d, {\bm{I}}_d) } \left[\dot{\sigma}\left({\bm{X}}^T {\bm{u}}\right)\dot{\sigma}\left({\bm{u}}^T{\bm{X}}\right) \right] \right).
    
\end{aligned}\\] For each \\(i \in [n]\\) and \\(j \in [d_1]\\), \\[\nabla_{{\bm{w}}_j}f({\bm{x}}_i) = \frac{1}{\sqrt{d_1}} v_j \dot{\sigma}\left(\langle {\bm{w}}_j^T, {\bm{x}}_i \rangle\right) {\bm{x}}_i\\] and therefore \\[{\bm{K}}_1 = \frac{1}{d_1}\sum_{j = 1}^{d_1} {\bm{Z}}_j,\\] where \\[{\bm{Z}}_j = v_j^2 \left(\dot{\sigma}\left( {\bm{X}}^T {\bm{w}}_j \right) \dot{\sigma}\left({\bm{w}}_j^T {\bm{X}}\right)\right) \odot \left({\bm{X}}^T {\bm{X}}\right).\\] For each \\(j \in [d_1]\\), let \\(\xi_j \in \{0, 1\}\\) be a random variable taking value 1 if \\(|v_j| \leq 1\\) and taking value 0 otherwise. Since \\(v_j\\) is a standard Gaussian there exists a universal constant \\(C_1 > 0\\) with \\(\mathbb{E}[\xi_jv_j] = C_1\\) for all \\(j\\). We also define \\({\bm{Z}}_j' = \xi_j {\bm{Z}}_j\\). Note that \\({\bm{Z}}_j' \succeq \bm{0}\\), and by the inequality \\(\lambda_{\max}({\bm{A}}\odot {\bm{B}}) \leq \max_i [{\bm{A}}]_{ii} \lambda_{\max}({\bm{B}})\\), \\[\begin{aligned}
        \|{\bm{Z}}'_j\| &= \left\|\xi_jv_j^2 \left(\dot{\sigma}\left( {\bm{X}}^T {\bm{w}}_j \right) \dot{\sigma}\left({\bm{w}}_j {\bm{X}}\right)\right) \odot \left({\bm{X}}^T {\bm{X}}\right)  \right\|\\
        &\leq \max_{i \in [n]}\left|\left(\xi_jv_j^2 \left[\dot{\sigma}\left({\bm{X}}^T {\bm{w}}_j \right) \dot{\sigma}\left({\bm{w}}_j^T {\bm{X}}\right)\right)\right]_{ii} \right| \cdot \left\| {\bm{X}}^T {\bm{X}}\right\|\\
        &= \max_{i \in [n]} \left|\xi_j v_j^2 \dot{\sigma}\left({\bm{w}}_j^T {\bm{x}}_i \right)^2 \right| \cdot \|{\bm{X}}\|^2\\
        &\leq \|{\bm{X}}\|^2.
    
\end{aligned}\\]

Furthermore by the inequality \\(\lambda_{\min}({\bm{A}}\odot {\bm{B}}) \geq \min_i [{\bm{A}}]_{ii} \lambda_{\min}({\bm{B}})\\), \\[\begin{aligned}
    \lambda_{\min}\left(\mathbb{E}[{\bm{Z}}_j'] \right) &= \lambda_{\min}\left(\mathbb{E}\left[\xi_jv_j^2\left(\dot{\sigma}\left({\bm{X}}^T {\bm{w}}_j \right)\dot{\sigma}\left({\bm{w}}_j^T {\bm{X}}\right)\right)  \right]\odot \left({\bm{X}}^T {\bm{X}}\right) \right)\\
    &\geq \lambda_{\min}\left(\mathbb{E}\left[\xi_jv_j^2\left(\dot{\sigma}\left({\bm{X}}^T {\bm{w}}_j \right)\dot{\sigma}\left({\bm{w}}_j^T{\bm{X}}\right)\right)  \right] \right)\min_{i \in [n]}\left|\left({\bm{X}}^T {\bm{X}}\right)_{ii} \right|\\
    &= \lambda_{\min}\left(\mathbb{E}\left[\xi_jv_j^2\right] \mathbb{E}\left[\left(\dot{\sigma}\left({\bm{X}}^T {\bm{w}}_j \right)\dot{\sigma}\left({\bm{w}}_j^T{\bm{X}}\right)\right)  \right] \right)\min_{i \in [n]} \|{\bm{x}}_i\|^2\\
    &= C_1\lambda_{\min}\left( \mathbb{E}\left[\left(\dot{\sigma}\left({\bm{X}}^T {\bm{w}}_j \right)\dot{\sigma}\left({\bm{w}}_j^T{\bm{X}}\right)\right)  \right] \right)\\
    &= C_1 \lambda_1.
\end{aligned}\\] So by Lemma <a href="#lem:matrix-chernoff" data-reference-type="ref" data-reference="lem:matrix-chernoff">2</a>, for all \\(t \geq 0\\) \\[\begin{aligned}
    \mathbb{P}\left(\lambda_{\min}\left(\frac{1}{d_1}\sum_{j = 1}^{d_1}{\bm{Z}}_j' \right) \leq C_1 \lambda_1 \right) &\leq \mathbb{P}\left(\lambda_{\min}\left(\frac{1}{d_1}\sum_{j = 1}^{d_1}{\bm{Z}}_j' \right) \leq \mathbb{E}[{\bm{Z}}_1'] \right)\\
    &\leq n\exp\left(-\frac{C_2 d_1 \lambda_1 }{\|{\bm{X}}\|^2} \right)
\end{aligned}\\] where \\(C_2 > 0\\) is a constant. Since \\({\bm{Z}}_j \succeq {\bm{Z}}_j'\\) for all \\(j \in [d_1]\\), if \\(d_1 \geq \frac{1}{C_2\lambda_1}\|{\bm{X}}\|^2 \log\left(\frac{n}{\epsilon}\right)\\), then \\[\begin{aligned}
    \mathbb{P}\left(\lambda_{\min}\left(\frac{1}{d_1}\sum_{j  =1}^{d_1}{\bm{Z}}_j \right) \leq C_1 \lambda_1 \right) &\leq n \exp\left(-\frac{C_2d_1 \lambda_1}{\|{\bm{X}}\|^2}\right)\\
    &\leq \epsilon.
\end{aligned}\\] ◻

</div>

## Proof of Lemma <a href="#lemma:K2-inf" data-reference-type="ref" data-reference="lemma:K2-inf">[lemma:K2-inf]</a> [app:lemma:K2-inf]

<div class="lemma" markdown="1">

**Lemma 12**. *Suppose that \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{S}^{d_0 -1 }\\). Let \\[\lambda_2 = d_0\lambda_{\min}\left(\mathbb{E}_{{\bm{u}}\sim U(\mathbb{S}^{d_0 - 1}) }\left[\sigma( {\bm{X}}^T {\bm{u}})\sigma({\bm{u}}^T {\bm{X}}) \right] \right).\\] If \\(\lambda_2 > 0\\) and \\(d_1 \gtrsim  \frac{n}{\lambda_2}  \log\left(\frac{n}{\lambda_2}\right)\log \left(\frac{n}{\epsilon}\right)\\), then with probability at least \\(1 - \epsilon\\), \\(\lambda_{\min}({\bm{K}}_2) \geq 
 \frac{\lambda_2}{4}\\).*

</div>

<div class="proof" markdown="1">

*Proof.* Note that by Lemma <a href="#lem:spherical-gaussian-equivalence" data-reference-type="ref" data-reference="lem:spherical-gaussian-equivalence">11</a>, \\[\lambda_2 = \lambda_{\min}\left(\mathbb{E}_{{\bm{w}}\sim \mathcal{N}(\bm{0}_d, {\bm{I}}_d) }\left[ \sigma\left( {\bm{X}}^T {\bm{w}}\right) \sigma\left({\bm{w}}^T {\bm{X}}\right) \right] \right).\\] For each \\(i \in [n]\\) and \\(j \in [d_1]\\), \\[\nabla_{v_j} f({\bm{x}}_i) = \frac{1}{\sqrt{d_1}} \sigma({\bm{w}}_j^T {\bm{x}}_i)\\] and therefore \\[{\bm{K}}_2 = \frac{1}{d_1} \sum_{j = 1}^{d_1}  {\bm{Z}}_j,\\] where \\[{\bm{Z}}_j = \sigma\left( {\bm{X}}^T {\bm{w}}_j\right)\sigma\left({\bm{w}}_j^T {\bm{X}}\right).\\] By `\citet[Theorem 6.3.2]{vershynin2018high}`{=latex}, for each \\(j \in [d_1]\\) \\[\begin{aligned}
       \left\| \left\|{\bm{X}}^T {\bm{w}}_j \right\| \right\|_{\psi_2} &\lesssim  \left\| \left\|{\bm{X}}^T {\bm{w}}_j \right\| - \left\| {\bm{X}}^T\right\|_F  \right\|_{\psi_2} + \|{\bm{X}}^T\|_F\\
       &\lesssim \|{\bm{X}}^T\| + \|{\bm{X}}^T\|_F\\
       &\lesssim \|{\bm{X}}^T\|_F\\
       &= \|{\bm{X}}\|_F\\
       &= \sqrt{n}.
   
\end{aligned}\\] So by Hoeffding’s inequality, for all \\(t \geq 0\\) \\[\begin{aligned}
    \mathbb{P}\left(\left\|{\bm{X}}^T {\bm{w}}_j \right\|^2 \geq t \right) = \mathbb{P}\left(\left\|{\bm{X}}^T {\bm{w}}_j \right\| \geq \sqrt{t} \right) \leq 2\exp\left(-\frac{C_1t}{n } \right)\label{eq:activation-exp-tail}
   
\end{aligned}\\] for some constant \\(C_1 > 0\\). Let \\(s = \frac{n}{C_1}\log \frac{4n}{\lambda_2C_1}\\). For each \\(j \in [d_1]\\) let \\(\xi_j \in \{0, 1\}\\) be a random variable taking value 1 if \\(\|{\bm{X}}^T {\bm{w}}_j\|^2 \leq s\\) and taking value 0 otherwise. Let \\({\bm{Z}}_j' = \xi_j {\bm{Z}}_j\\). For each \\(j \in [m]\\), \\({\bm{Z}}_j' \succeq 0\\), and \\[\begin{aligned}
       \left\|{\bm{Z}}_j'\right\| &=  \left\|\xi_j \sigma\left( {\bm{X}}^T {\bm{w}}_j \right)\sigma\left({\bm{w}}_j^T {\bm{X}}\right) \right\|\\
       &= \left\|\xi_j \sigma\left({\bm{X}}^T {\bm{w}}_j\right) \right\|^2\\
       &\leq s.
   
\end{aligned}\\] Moreover, \\[\begin{aligned}
       \left\|\mathbb{E}[{\bm{Z}}_j] - \mathbb{E}[{\bm{Z}}_j'] \right\| &= \left\|\mathbb{E}\left[(1 - \xi_j) \sigma\left( {\bm{X}}^T {\bm{w}}_j \right)\sigma\left({\bm{w}}_j^T {\bm{X}}\right) \right] \right\|\\
       &\leq \mathbb{E}\left[(1 - \xi_j)\left\|\sigma\left( {\bm{X}}^T {\bm{w}}_j \right)\sigma\left({\bm{w}}_j^T {\bm{X}}\right) \right\|\right]\\
       &= \mathbb{E}\left[(1 - \xi_j)\left\|\sigma\left({\bm{X}}^T {\bm{w}}_j\right) \right\|^2 \right]\\
       &= \frac{1}{2} \mathbb{E}\left[(1 - \xi_j)\left\|{\bm{X}}^T {\bm{w}}_j \right\|^2 \right]\\
       &= \frac{1}{2}\int_{s}^{\infty}\mathbb{P}\left(\left\|{\bm{X}}^T {\bm{w}}_j\right\|^2 \geq t \right)dt\\
       &\leq 2 \int_s^{\infty}\exp\left(-\frac{C_1t}{n} \right)dt\\
       &= \frac{2n}{C_1}\exp\left(-\frac{C_1s}{n} \right)\\
       &= \frac{\lambda_2}{2}.
   
\end{aligned}\\] Here we used (<a href="#eq:activation-exp-tail" data-reference-type="ref" data-reference="eq:activation-exp-tail">[eq:activation-exp-tail]</a>) in line 6. By Weyl’s inequality, \\[\lambda_{\min}(\mathbb{E}[ {\bm{Z}}_j']) \geq \lambda_{\min}(\mathbb{E}[{\bm{Z}}_j]) - \left\|\mathbb{E}[{\bm{Z}}_j] - \mathbb{E}[{\bm{Z}}_j']\right\| = \lambda_2 - \frac{\lambda_2}{2} = \frac{\lambda_2}{2}.\\] By Lemma <a href="#lem:matrix-chernoff" data-reference-type="ref" data-reference="lem:matrix-chernoff">2</a>, \\[\begin{aligned}
       \mathbb{P}\left(\lambda_{\min}\left(\frac{1}{d_1}\sum_{j = 1}^{d_1} {\bm{Z}}_j' \right) \leq \frac{\lambda_2}{4}  \right) &\leq \mathbb{P}\left(\lambda_{\min}\left(\frac{1}{m}\sum_{j = 1}^m {\bm{Z}}_j'\right) \leq \frac{1}{2} \lambda_{\min}(\mathbb{E}[{\bm{Z}}_1']) \right)\\
       &\leq n \exp\left(-\frac{C_2d_1 \lambda_{\min}(\mathbb{E}[{\bm{Z}}_1']) }{s} \right)\\
       &\leq n \exp\left(\frac{-C_2 d_1 \lambda_2  }{2s} \right).
   
\end{aligned}\\] Since \\({\bm{Z}}_j' \preceq {\bm{Z}}_j\\) for all \\(j\\), for \\(d_1 \geq \frac{2s}{C_2\lambda_2} \log \frac{n}{\epsilon}\\) this implies \\[\begin{aligned}
       \mathbb{P}\left(\lambda_{\min}\left(\frac{1}{d_1}\sum_{j = 1}^{d_1}{\bm{Z}}_j \right) \leq \frac{\lambda_2}{4} \right) &\leq n \exp\left(-\frac{C_2d_1 \lambda_2}{2s}\right)\\
       &\leq \epsilon.
   
\end{aligned}\\] In other words, \\[\mathbb{P}\left(\lambda_{\min}({\bm{K}}_2) \geq \frac{\lambda_2}{4}\right) \geq 1 - \epsilon.\\] ◻

</div>

## Proof of Lemma <a href="#lem:gram-to-hemisphere-transform" data-reference-type="ref" data-reference="lem:gram-to-hemisphere-transform">[lem:gram-to-hemisphere-transform]</a> [app:gram-to-hemisphere-transform]

<div class="proof" markdown="1">

*Proof.* We compute \\[\begin{aligned}
        \langle {\bm{K}}_{\psi}^{\infty}{\bm{z}}, {\bm{z}}\rangle &= \mathbb{E}_{{\bm{w}}\sim U(\mathbb{S}^{d - 1})}\left[\left| \psi\left({\bm{w}}^T {\bm{X}}\right) {\bm{z}}\right|^2 \right]\\
        &= \int_{\mathbb{S}^{d - 1}} \left| \psi\left({\bm{w}}^T {\bm{X}}\right) {\bm{z}}\right|^2 dS({\bm{w}})\\
        &= \int_{\mathbb{S}^{d - 1}}\left|\sum_{i = 1}^n \psi(\langle {\bm{w}}, {\bm{x}}_i \rangle) z_i\right|^2 dS({\bm{w}})\\
        &= \int_{\mathbb{S}^{d - 1}}\left|\int_{\mathbb{S}^{d - 1}} \psi(\langle {\bm{w}}, {\bm{x}}\rangle) d\mu_{{\bm{z}}}({\bm{x}}) \right|^2dS({\bm{w}})\\
        &= \int_{\mathbb{S}^{d - 1}} \left| T_{\psi} \mu_{{\bm{z}}}({\bm{w}})\right|^2 dS({\bm{w}})\\
        &= \|T_{\psi}\mu_{{\bm{z}}}\|^2
    
\end{aligned}\\] which establishes the first part of the result. The second part of the result follows immediately by writing \\[\lambda_{\min}({\bm{K}}_{\psi}^{\infty}) = \inf_{\|{\bm{z}}\| = 1} \langle {\bm{K}}_{\psi}^{\infty}{\bm{z}}, {\bm{z}}\rangle = \inf_{\|{\bm{z}}\| = 1} \|T_{\psi}\mu_{{\bm{z}}}\|^2.\\] ◻

</div>

## Proof of Lemma <a href="#lem:matrix-spherical-harmonic" data-reference-type="ref" data-reference="lem:matrix-spherical-harmonic">[lem:matrix-spherical-harmonic]</a> [app:matrix-spherical-harmonic]

<div class="proof" markdown="1">

*Proof.* Note that \\[\begin{aligned}
        N = \sum_{r = 0}^R \left(\binom{2r + \beta + d - 1}{d - 1} - \binom{2r + \beta + d - 3}{d - 1}  \right)
        = \binom{2R + \beta + d - 1}{d - 1}.
    
\end{aligned}\\] Let us write \\({\bm{D}}= [{\bm{d}}_1, \cdots, {\bm{d}}_n]\\). Fix \\(i, k \in [n]\\) with \\(i \neq k\\). By the addition formula (<a href="#eqn:addition-formula" data-reference-type="ref" data-reference="eqn:addition-formula">[eqn:addition-formula]</a>), \\[\begin{aligned}
        \|{\bm{d}}_i\|^2 &= \sum_{a = 1}^N g_a({\bm{x}}_i)^2\\
        &= \sum_{r = 0}^R \sum_{s = 1}^{\dim(\mathcal{H}_{2r +\beta}^d) }Y_{r,s}^d({\bm{x}}_i)^2\\
        &= \sum_{r = 0}^R \dim(\mathcal{H}_{2r + \beta}^d)\\
        &= N.
    
\end{aligned}\\] By Lemma <a href="#lemma:addition-formula-bound" data-reference-type="ref" data-reference="lemma:addition-formula-bound">4</a> and \\(\delta\\)-separation, there exists a constant \\(C > 0\\) such that \\[\begin{aligned}
         |\langle {\bm{d}}_i, {\bm{d}}_k \rangle| &= \left|\sum_{a = 1}^N g_a({\bm{x}}_i)g_a({\bm{x}}_k)\right|\\
         &\leq C\left(\frac{\delta^4 }{2} \right)^{-(d-2)/4}\binom{2R + \beta + d - 1 }{d -1 }^{1/2}\\
         &= C N^{1/2}\left(\frac{\delta^4 }{2} \right)^{-(d-2)/4}.
     
\end{aligned}\\] Suppose that \\[N \geq 2C^2\left(\frac{\delta^4}{2}\right)^{-(d - 2)/2}.\\]

Observe that \\(\sigma_{\min}({\bm{D}})\\) is the square root of the minimum eigenvalue of \\({\bm{D}}^T {\bm{D}}\\). By the Gershgorin circle theorem, the minimum eigenvalue of \\({\bm{D}}^T {\bm{D}}\\) is at least \\[\begin{aligned}
        \min_{i \in [n]}\left(|({\bm{D}}^T{\bm{D}})_{ii}| - \sum_{k \neq i}|{\bm{D}}^T{\bm{D}}|_{ik} \right) &= \min_{i \in [n]}\left(\|{\bm{d}}_i\|^2 - \sum_{k \neq i}|\langle {\bm{d}}_i, {\bm{d}}_k \rangle|\right)\\
        &\geq \frac{N}{2}.
    
\end{aligned}\\] The result follows. ◻

</div>

## Proof of Lemma <a href="#corr:hemisphere-transform-asymptotics" data-reference-type="ref" data-reference="corr:hemisphere-transform-asymptotics">[corr:hemisphere-transform-asymptotics]</a> [app:hemi-transform-asymp]

<div id="lemma:hemisphere-transform-asymptotics-implicit" class="lemma" markdown="1">

**Lemma 13**. *Let \\(\epsilon \in (0, 1)\\) and let \\(\delta > 0\\). Suppose that \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{S}^{d - 1}\\) form a \\(\delta\\)-separated dataset. Let \\(R \in \mathbb{N}\\) be such that \\[\binom{2R + d - 1}{d - 1} \geq C\left(\frac{\delta^4}{2}\right)^{-(d - 2)/2}\\] where \\(C > 0\\) is a universal constant. Then \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 \gtrsim \begin{cases}
            (d + R)^{1/2}d^{-1/2} R^{-3/2} & \text{ if $\psi = \dot{\sigma}$}\\
            (d + R)^{-1/2}d^{1/2}R^{-3/2} & \text{if $\psi = \sqrt{d}\sigma$}
        \end{cases} 
    
\end{aligned}\\] for all \\({\bm{z}}\in \mathbb{R}^n\\) with \\(\|{\bm{z}}\| \leq 1\\).*

</div>

<div class="proof" markdown="1">

*Proof.* Let \\(C\\) be the same constant as in Lemma <a href="#lem:matrix-spherical-harmonic" data-reference-type="ref" data-reference="lem:matrix-spherical-harmonic">[lem:matrix-spherical-harmonic]</a> and suppose that \\[\binom{2R + d - 1}{d - 1} \geq C\left(\frac{\delta^4}{2}\right)^{-(d - 2)/2}.\\] Let \\(\beta \in \{0, 1\}\\) satisfy \\(\beta = 1\\) when \\(\psi = \dot{\sigma}\\) and \\(\beta = 0\\) when \\(\psi = d\sigma\\). Let \\(N = \sum_{r = 0}^R \dim(\mathcal{H}_{2r + \beta}^d)\\). Note that \\[\begin{aligned}
        N &= \sum_{r = 0}^R\left(\binom{2r + d + \beta - 1}{d - 1} - \binom{2r + d + \beta - 3}{d - 1} \right) \\&= \binom{2R + d + \beta - 1}{d - 1}\\
        &\geq \binom{2R + d - 1}{d - 1}\\
        &\geq C\left(\frac{\delta^4}{2}\right)^{-(d - 2)/2}.
    
\end{aligned}\\] Let \\(g_1, \cdots, g_N\\) be spherical harmonics forming an orthonormal basis of \\(\bigoplus_{r = 1}^R \mathcal{H}_{2r - 1}^d\\), and let \\({\bm{B}}\in \mathbb{R}^{N \times n}\\) be the matrix defined by \\({\bm{B}}_{ai} = g_a({\bm{x}}_i)\\). By Lemma <a href="#lem:matrix-spherical-harmonic" data-reference-type="ref" data-reference="lem:matrix-spherical-harmonic">[lem:matrix-spherical-harmonic]</a>, \\(\sigma_{\min}({\bm{B}}) \geq \sqrt{\frac{N}{2} }\\) with probability at least \\(1 - \epsilon\\). Since the functions \\(g_a\\) are orthonormal, \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\geq \sum_{a = 1}^N  |\langle T_{\psi}\mu_{{\bm{z}}}, g_a \rangle|^2.
    
\end{aligned}\\] By Lemma <a href="#lemma:T-self-adjoint" data-reference-type="ref" data-reference="lemma:T-self-adjoint">6</a> the above expression is equal to \\[\begin{aligned}
        \sum_{a = 1}^N |\langle \mu_{{\bm{z}}}, T_{\psi} g_a \rangle|^2 &= \sum_{r = 0}^R\sum_{s = 1}^{\dim\left(\mathcal{H}_{2r + \beta}^d \right) } \left|\langle \mu_{{\bm{z}}}, T_{\psi}Y_{2r + \beta, s} \rangle \right|^2.
    
\end{aligned}\\] By Lemmas <a href="#lemma:eigendecomposition-sign" data-reference-type="ref" data-reference="lemma:eigendecomposition-sign">9</a> and <a href="#lem:eigendecomposition-relu" data-reference-type="ref" data-reference="lem:eigendecomposition-relu">10</a>, \\(T_{\psi} Y_{2r + \beta, s} = c_{2r + \beta,d} Y_{2r + \beta, s}\\), where \\(c_{2r + \beta} \in \mathbb{R}\\) and \\[\begin{aligned}
\label{eq:coefficient-asymptotics}
        |c_{2r + \beta,d}| &\gtrsim \begin{cases}
           \frac{\Gamma\left(\frac{d}{2}\right)\Gamma\left(\frac{2R + 1}{2}\right) }{\Gamma\left(\frac{d + 2R + 1}{2}\right) } & \text{ if $\psi = \dot{\sigma}$}\\
           \frac{\sqrt{d}\Gamma\left(\frac{d}{2}\right)\Gamma\left(\frac{2R - 1}{2}\right) }{\Gamma\left(\frac{d + 2R + 1}{2}\right) } & \text{ if $\psi = \sqrt{d}\sigma$.}
        \end{cases}
    
\end{aligned}\\] Hence \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\geq \sum_{r = 0}^R \sum_{s = 1}^{\dim\left(\mathcal{H}_{2r + \beta}^d\right) }|c_{2r + \beta, d}|^2 |\langle \mu_{{\bm{z}}}, Y_{2r + \beta, s} \rangle|^2\\
        &\geq \min_{0 \leq r \leq R}\left(|c_{2r + \beta, d}|^2 \right)\sum_{r = 0}^R \sum_{s = 1}^{\dim\left(\mathcal{H}_{2r + \beta}^d\right)}|\langle \mu_{{\bm{z}}}, Y_{2r + \beta, s} \rangle|^2\\
        &= \min_{0 \leq r \leq R}\left(|c_{2r + \beta, d}|^2 \right)\sum_{a = 1}^N |\langle \mu_{{\bm{z}}}, g_a \rangle|^2\\
        &= \min_{0 \leq r \leq R}\left(|c_{2r + \beta, d}|^2 \right)\sum_{a = 1}^N \left|\sum_{i = 1}^n z_i g_a({\bm{x}}_i) \right|^2\\
        &= \min_{0 \leq r \leq R}\left(|c_{2r + \beta, d}|^2 \right) \|{\bm{B}}{\bm{z}}\|^2\\
        &\geq \min_{0 \leq r \leq R}\left(|c_{2r + \beta, d}|^2 \right)\sigma_{\min}({\bm{B}})^2\\
        &\geq \frac{N}{2}\min_{0 \leq r \leq R}\left(|c_{2r + \beta, d}|^2 \right).
    
\end{aligned}\\] So by (<a href="#eq:coefficient-asymptotics" data-reference-type="ref" data-reference="eq:coefficient-asymptotics">[eq:coefficient-asymptotics]</a>), \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\gtrsim \begin{cases}
           \frac{N\Gamma\left(\frac{d}{2}\right)^2\Gamma\left(\frac{2R + 1}{2}\right)^2 }{\Gamma\left(\frac{d + 2R + 1}{2}\right)^2 } & \text{ if $\psi = \dot{\sigma}$}\\
           \frac{Nd^2\Gamma\left(\frac{d}{2}\right)^2\Gamma\left(\frac{2R - 1}{2}\right)^2 }{\Gamma\left(\frac{d + 2R + 1}{2}\right)^2 } & \text{ if $\psi = d\sigma$.}
        \end{cases}
    
\end{aligned}\\] We now separately analyze the cases where \\(\psi = \dot{\sigma}\\) and \\(\psi = d\sigma\\).

**Case 1: \\(\psi = \dot{\sigma}\\)**. In this case \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\gtrsim N\frac{ \Gamma\left(\frac{d}{2}\right)^2 \Gamma\left(\frac{2R + 1}{2}\right)^2 }{\Gamma\left(\frac{d + 2R + 1}{2}\right)^2 }\\
        &= \binom{2R + d}{d - 1}\cdot \frac{ \Gamma\left(\frac{d}{2}\right)^2 \Gamma\left(\frac{2R + 1}{2}\right)^2 }{\Gamma\left(\frac{d + 2R + 1}{2}\right)^2 }\\
        &= \frac{\Gamma(2R + d + 1) }{\Gamma(d)\Gamma(2R + 2) } \cdot \frac{ \Gamma\left(\frac{d}{2}\right)^2 \Gamma\left(\frac{2R + 1}{2}\right)^2 }{\Gamma\left(\frac{d + 2R + 1}{2}\right)^2 }.
    
\end{aligned}\\] Then by Stirling’s approximation, \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\gtrsim \frac{(2R + d + 1)^{2R + d + 1/2}e^{-2R - d - 1 }  }{d^{d -1/2}e^{-d}(2R + 2)^{2R + 3/2}e^{-2R - 2}  } \cdot \frac{\left(\frac{d}{2}\right)^{d - 1}e^{-d}\left(\frac{2R + 1}{2}\right)^{2R} e^{-2R - 1}  }{\left(\frac{d + 2R + 1}{2}\right)^{d + 2R} e^{-d - 2R - 1}  }\\
        &\gtrsim (d + 2R + 1)^{1/2} d^{-1/2}\left(\frac{2R + 1}{2R + 2}\right)^{2R}(2R + 2)^{-3/2}\\
        &\gtrsim (d + 2R + 1)^{1/2}d^{-1/2} (2R + 2)^{-3/2}\\
        &\gtrsim (d + R)^{1/2}d^{-1/2} R^{-3/2}.
    
\end{aligned}\\] Here the third inequality follows from the observations \\[\begin{aligned}
        \left(\frac{2R + 1}{2R + 2}\right)^{2R} > 0
    
\end{aligned}\\] and \\[\begin{aligned}
        \lim_{R \to \infty}\left(\frac{2R + 1}{2R + 2}\right)^{2R} = \lim_{R \to \infty}\left(1 - \frac{1}{2R +2}\right)^{2R} = e^{-1}.
    
\end{aligned}\\] **Case 2: \\(\psi = \sqrt{d}\sigma\\)**. In this case \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\gtrsim N\frac{ d \Gamma\left(\frac{d}{2}\right)^2 \Gamma\left(\frac{2R - 1}{2}\right)^2 }{\Gamma\left(\frac{d + 2R + 1}{2}\right)^2 }\\
        &=\binom{2R + d - 1}{d  - 1}\cdot \frac{ d \Gamma\left(\frac{d}{2}\right)^2 \Gamma\left(\frac{2R - 1}{2}\right)^2 }{\Gamma\left(\frac{d + 2R + 1}{2}\right)^2 }\\
        &= \frac{\Gamma(2R + d) }{\Gamma(d)\Gamma(2R + 1) }\cdot \frac{ d \Gamma\left(\frac{d}{2}\right)^2 \Gamma\left(\frac{2R - 1}{2}\right)^2 }{\Gamma\left(\frac{d + 2R + 1}{2}\right)^2 }.
    
\end{aligned}\\] Then by Stirling’s approximation, \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\gtrsim \frac{(2R + d)^{2R + d - 1/2}e^{-2R - d} }{d^{d - 1/2}e^{-d}(2R + 1)^{2R + 1/2}e^{-2R - 1} } \cdot \frac{d \left(\frac{d}{2}\right)^{d -1 }e^{-d}\left(\frac{2R - 1}{2}\right)^{2R - 2}e^{-2R + 1} }{\left(\frac{d + 2R + 1}{2}\right)^{d + 2R } e^{-d - 2R - 1} }\\
        &\gtrsim (d + 2R)^{-1/2}\left(\frac{d + 2R}{d + 2R + 1}\right)^{d + 2R}d^{1/2}(2R - 1)^{-2}(2R + 1)^{1/2}\left(\frac{2R - 1}{2R + 1}\right)^{2R}\\
        &\gtrsim (d + 2R)^{-1/2}d^{1/2} R^{-3/2}\left(\frac{d + 2R}{d + 2R + 1}\right)^{d + 2R}\left(\frac{2R - 1}{2R + 1}\right)^{2R}\\
        &= (d + 2R)^{-1/2} d^{1/2}\left(1 - \frac{1}{d + 2R + 1}\right)^{d + 2R}\left(1 - \frac{2}{2R + 1}\right)^{2R} \\
        &\gtrsim (d + R)^{-1/2}d^{1/2}R^{-3/2}.
    
\end{aligned}\\] Hence we have established the desired bound on \\(\|T_{\psi}\mu_{{\bm{z}}}\|^2\\) in all cases. ◻

</div>

<div class="proof" markdown="1">

*Proof.* We will consider multiple cases depending on the relative scaling of \\(d\\) and \\(n\\). Let \\(C > 0\\) be the same constant as in Lemma <a href="#lemma:hemisphere-transform-asymptotics-implicit" data-reference-type="ref" data-reference="lemma:hemisphere-transform-asymptotics-implicit">13</a>. First suppose that \\(d \geq C\left(\frac{\delta^4}{2} \right)^{-(d - 2)/2}\\). Let \\(R = 1\\). Then \\[\binom{2R + d - 1}{d - 1} = d \geq C\left(\frac{\delta^4}{2}\right)^{(d - 2)/2}.\\] By Lemma <a href="#lemma:hemisphere-transform-asymptotics-implicit" data-reference-type="ref" data-reference="lemma:hemisphere-transform-asymptotics-implicit">13</a>, \\(\|T_{\psi}\mu_{{\bm{z}}}\|^2 \gtrsim 1\\) in this case.

Next suppose that \\(d \leq C\left(\frac{\delta^4}{2}\right)^{-(d - 2)/2}\\) and \\(\sqrt{d} \log d \geq (8\log(1 + C) + 16 d)\log \frac{2}{\delta}\\). Let \\[R = \left\lceil    \frac{\log(1 + C) + 2d \log(2/\delta) }{\log d} \right\rceil.\\] Note that since \\(d \leq \left(\frac{\delta^4}{2}\right)^{-(d - 2)/2}\\), we have \\[\begin{aligned}
         \frac{\log(1 + C) +2d \log(2/\delta)}{\log d} \geq \frac{2d\log(2/\delta) }{\frac{d - 2}{2}\log(2/\delta^4) } \geq 1
    
\end{aligned}\\] and therefore \\[\begin{aligned}
        R \leq  \frac{2\log(1 + C) +4d\log(2/\delta) }{\log d} \leq \frac{\sqrt{d}}{4}.
    
\end{aligned}\\] By definition, \\[R \geq \frac{\log(1 + C) + 2d\log(2/\delta) }{\log(d)}\\] so that \\[\begin{aligned}
        \binom{2R + d - 1}{d - 1} &\geq \left(\frac{2R + d - 1}{2R}\right)^{2R}\\
        &\geq \left(\frac{d}{2R}\right)^{2R}\\
        &= \exp\left(2R(\log (d) - \log(2R) \right)\\
        &\geq \exp\left(2R\left(\log(d) - \log\left(\sqrt{d}\right)\right) \right)\\
        &= \exp\left(R \log d \right)\\
        &\geq \exp(\log(1 + C) + 2d\log(2/\delta))\\
        &\geq C\left(\frac{2}{\delta}\right)^{2d}\\
        &\geq C\left(\frac{2}{\delta^4}\right)^{(d - 2)/2}.
    
\end{aligned}\\] Then by Lemma <a href="#lemma:hemisphere-transform-asymptotics-implicit" data-reference-type="ref" data-reference="lemma:hemisphere-transform-asymptotics-implicit">13</a>, the following bounds hold. If \\(\psi = \dot{\sigma}\\), then \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\gtrsim (d + R)^{1/2}d^{-1/2}R^{-3/2}\\
        &\gtrsim R^{-3/2}\\
        &\gtrsim \left(1  +\frac{d \log(1/\delta) }{\log d}\right)^{-3/2}\\
        &\gtrsim \left(1 + \frac{d \log(1/\delta)}{\log d}\right)^{-3} \delta^2.
    
\end{aligned}\\] If \\(\psi = \sqrt{d}\sigma\\), then \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\gtrsim (d + R)^{-1/2}d^{1/2}R^{-3/2}\\
        &\gtrsim (d + \sqrt{d})^{-1/2}d^{1/2}R^{-3/2}\\
        &\gtrsim R^{-3/2}\\
        &\gtrsim \left(1 + \frac{ d\log(2/\delta)}{\log d}\right)^{-3/2}\\
        &\gtrsim \left(1 + \frac{\log(n/\epsilon)}{\log d}\right)^{-3} \delta^4.
    
\end{aligned}\\] Finally suppose that \\(\sqrt{d} \log d \leq (8 \log(1 + C) + 16 d) \log \frac{2}{\delta}\\) and let \\(R = \left\lceil (1 + 2C)d\left(\frac{2}{\delta}\right)^{2(d - 2)/(d - 1)}\right\rceil\\). Then \\[\begin{aligned}
        R &\lesssim  1 + d\left(\frac{2}{\delta}\right)^{2(d - 2)/(d -1) }\\
        &\leq (1 + d)\left(\frac{2}{\delta}\right)^{2(d - 2)/(d - 1) }\\
        &\leq \left(1 + \sqrt{d}\right)^2 \left(\frac{2}{\delta}\right)^{2(d - 2)/(d - 1)}\\
        &\lesssim \left(1 + \frac{d \log(1/\delta)}{\log(d)}\right)^2\left(\frac{2}{\delta}\right)^{2(d - 2)/(d - 1)}\\
        &\lesssim \left(1 + \frac{d \log(1/\delta)}{\log(d)}\right)^2 \delta^{-2}
    
\end{aligned}\\] and \\[\begin{aligned}
        \binom{2R + d - 1}{d - 1} &\geq \left(\frac{2R + d - 1}{d - 1}\right)^{d -1 }\\
        &\geq \left(\frac{R}{d}\right)^{d -1 }\\
        &\geq \left(1 + \frac{2C}{d}\right)^{d - 1}\left(\frac{2}{\delta}\right)^{2/(d - 2)}.\\
        &\geq \frac{2C(d - 1)}{d}\left(\frac{2}{\delta}\right)^{2/(d - 2)}\\
        &\geq C\left(\frac{2}{\delta}\right)^{2/(d - 2)}.
    
\end{aligned}\\] So by Lemma <a href="#lemma:hemisphere-transform-asymptotics-implicit" data-reference-type="ref" data-reference="lemma:hemisphere-transform-asymptotics-implicit">13</a> the following bounds hold. If \\(\psi = \dot{\sigma}\\), then \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\gtrsim (d + R)^{1/2}d^{-1/2}R^{-3/2}\\
        &\gtrsim (1 + d)^{-1/2}R^{-1}\\
        &\gtrsim \left(1 + \frac{d\log(1/\delta) }{\log d} \right)^{-1}\left(1 + \frac{d\log(1/\delta) }{\log d}\right)^{-2} \delta^2\\
        &= \left(1 + \frac{d\log(1/\delta) }{\log d} \right)^{-3} \delta^{2}.
    
\end{aligned}\\] If \\(\psi = \sqrt{d}\sigma\\), then \\[\begin{aligned}
        \|T_{\psi}\mu_{{\bm{z}}}\|^2 &\gtrsim (d + R)^{-1/2}d^{1/2}R^{-3/2}\\
        &\gtrsim \left(d + d\left(\frac{2}{\delta} \right)^{2(d - 2)/(d - 1)}  \right)^{-1/2}d^{1/2}\left(d\left(\frac{2}{\delta}\right)^{2(d - 2)/(d - 1)}\right)^{-3/2}\\
        &\gtrsim d^{-3/2}\left(1 + \left(\frac{2}{\delta}\right)^{2(d -2 )/(d - 1)}\right)^{-1/2}\left(\frac{2}{\delta}\right)^{-3(d - 2)/(d - 1) }\\
        &\gtrsim (1 + d)^{-3/2}\left(\frac{2}{\delta}\right)^{-4(d-2)/(d -1 )}\\
        &\gtrsim \left(1 + \frac{d\log(1/\delta) }{\log d}\right) \delta^4.
    
\end{aligned}\\] Hence we have shown the desired bound on \\(\|T_{\psi}\mu_{{\bm{z}}}\|^2\\) in all cases. ◻

</div>

## Upper bound on the minimum eigenvalue of the NTK [app:upper-bound-min-eigenvalue]

Our strategy to upper bound \\(\lambda_{\min}({\bm{K}})\\) will be to prove that if two data points \\({\bm{x}}, {\bm{x}}'\\) are close, then the Jacobian of the network does not separate points too much. We will need to find upper bounds for both \\(\|\sigma({\bm{W}}{\bm{x}}) - \sigma({\bm{W}}{\bm{x}}')\|\\) and \\(\|\dot{\sigma}({\bm{W}}{\bm{x}}) - \dot{\sigma}({\bm{W}}{\bm{x}}')\|\\).

<div id="lemma:feature-map-regularity" class="lemma" markdown="1">

**Lemma 14**. *Let \\(\epsilon \in (0, 1)\\). Suppose that \\({\bm{x}}, {\bm{x}}' \in \mathbb{S}^{d - 1}\\) with \\(\|{\bm{x}}- {\bm{x}}'\| = \delta\\). If \\(d_1 = \Omega\left(\log \frac{1}{\epsilon}\right)\\), then with probability at least \\(1 - \epsilon\\), \\[\|\sigma({\bm{W}}{\bm{x}}) - \sigma({\bm{W}}{\bm{x}}')\| \lesssim \delta \sqrt{d_1}.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Note that \\(\|\sigma({\bm{W}}{\bm{x}}) - \sigma({\bm{W}}{\bm{x}}')\|^2\\) can be written a sum of iid subexponential random variables: \\[\|\sigma({\bm{W}}{\bm{x}}) - \sigma({\bm{W}}{\bm{x}}')\|^2 = \sum_{j = 1}^{d_{1}} (\sigma(\langle {\bm{w}}_j, {\bm{x}}\rangle) - \sigma(\langle {\bm{w}}_j, {\bm{x}}' \rangle )^2.\\] Since the entries of each \\({\bm{w}}_j\\) are iid standard Gaussian random variables and \\(\sigma\\) is \\(1\\)-Lipschitz, \\[\begin{aligned}
        \|(\sigma(\langle {\bm{w}}_j, {\bm{x}}\rangle) - \sigma(\langle {\bm{w}}_j, {\bm{x}}' \rangle) )^2\|_{\psi_1} &= \|\sigma(\langle {\bm{w}}_j, {\bm{x}}\rangle) - \sigma(\langle {\bm{w}}_j, {\bm{x}}' \rangle)\|_{\psi_2}^2\\
        &\leq \|\langle {\bm{w}}_j, {\bm{x}}- {\bm{x}}' \rangle\|_{\psi_2}^2\\
        &= \|{\bm{x}}- {\bm{x}}'\|^2\\
        &= \delta^2.
    
\end{aligned}\\] Moreover, \\[\begin{aligned}
        \mathbb{E}[(\sigma(\langle {\bm{w}}_j, {\bm{x}}\rangle) - \sigma(\langle {\bm{w}}_j, {\bm{x}}' \rangle) )^2 ] &\leq  \mathbb{E}[|\langle {\bm{w}}_j, {\bm{x}}- {\bm{x}}' \rangle|^2 ]\\
        &= \|{\bm{x}}- {\bm{x}}'\|^2\\
        &= \delta^2.
    
\end{aligned}\\] So by Bernstein’s inequality, for all \\(t \geq 0\\) \\[\begin{aligned}
        &\mathbb{P}\left(\|\sigma({\bm{W}}{\bm{x}}) - \sigma({\bm{W}}{\bm{x}}')\|^2 \geq \delta^2d_1 + t  \right) \\&\leq \mathbb{P}\left(\|\sigma({\bm{W}}{\bm{x}}) - \sigma({\bm{W}}{\bm{x}}')\|^2 \geq \mathbb{E}[\|\sigma({\bm{W}}{\bm{x}}) - \sigma({\bm{W}}{\bm{x}}')\|^2 ] + t  \right)\\
        &\leq 2 \exp\left(-C\min\left(\frac{t^2}{d_1 \delta^4}, \frac{t}{\delta^2} \right) \right)
    
\end{aligned}\\] where \\(C > 0\\) is a universal constant. Setting \\(t =  \delta^2 d_1\\) with \\(d_1 \geq \frac{1}{C} \log \frac{2}{\epsilon}\\) yields \\[\begin{aligned}
        \mathbb{P}(\|\sigma({\bm{W}}{\bm{x}}) - \sigma({\bm{W}}{\bm{x}}')\|^2 \geq 2\delta^2 d_1) \leq 2 \exp\left(-C d_1 \right) \leq \epsilon.
    
\end{aligned}\\] This establishes the result. ◻

</div>

<div id="lemma:cosine-probabilities" class="lemma" markdown="1">

**Lemma 15**. *Suppose that \\({\bm{x}}, {\bm{x}}' \in \mathbb{S}^{d-1}\\). If \\({\bm{w}}\sim \mathcal{N}(\bm{0}, {\bm{I}}_d)\\), then \\[\mathbb{P}(\dot{\sigma}(\langle {\bm{w}}, {\bm{x}}\rangle) \neq \dot{\sigma}(\langle {\bm{w}}, {\bm{x}}' \rangle)) \asymp  \|{\bm{x}}- {\bm{x}}'\|.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Recall that for \\({\bm{x}}, {\bm{x}}' \in \mathbb{S}^{d - 1}\\), \\[\mathbb{P}(\dot{\sigma}(\langle {\bm{w}}, {\bm{x}}\rangle) \neq \dot{\sigma}(\langle {\bm{w}}, {\bm{x}}' \rangle)) =  \frac{\theta}{\pi} ,\\] where \\(\theta\\) is the angle formed by \\({\bm{x}}\\) and \\({\bm{x}}'\\); that is, \\(\theta \in [0, \pi]\\) with \\[\cos(\theta) = \langle {\bm{x}}, {\bm{x}}' \rangle = 1 - \frac{1}{2}\|{\bm{x}}- {\bm{x}}'\|^2.\\] By Taylor’s theorem, \\(1 - \cos(\theta) = \frac{1}{2}\theta^2 + O(\theta^3)\\), so \\(1 - \cos(\theta) \asymp \theta^2\\) for \\(\theta \in [0, \pi]\\). This implies that \\(\theta^2 \asymp \|{\bm{x}}- {\bm{x}}'\|^2\\), so \\(\theta \asymp \|{\bm{x}}- {\bm{x}}'\|\\) and therefore \\[\mathbb{P}(\dot{\sigma}(\langle {\bm{w}}, {\bm{x}}\rangle) \neq \dot{\sigma}(\langle {\bm{w}}, {\bm{x}}' \rangle)) \asymp \|{\bm{x}}- {\bm{x}}'\|.\\] ◻

</div>

<div id="lemma:activation-pattern-lipschitz" class="lemma" markdown="1">

**Lemma 16**. *Let \\(\epsilon \in (0, 1)\\). Suppose that \\({\bm{x}}, {\bm{x}}' \in \mathbb{S}^{d - 1}\\) with \\(\|{\bm{x}}- {\bm{x}}'\| \leq \delta\\). If \\(d_1 = \Omega\left(\frac{1}{\delta}\log \frac{1}{\epsilon}\right)\\), then with probability at least \\(1 - \epsilon\\), \\[\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})\| \lesssim \sqrt{\delta d_1}.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Observe that \\[\|\dot{\sigma}({\bm{W}}{\bm{x}}) - \dot{\sigma}({\bm{W}}{\bm{x}}')\|^2 = 4\sum_{j = 1}^{d_1} Z_j = 4|\mathcal{S}|,\\] where \\(Z_j \in \{0, 1\}\\) is equal to 1 if \\[\dot{\sigma}(\langle {\bm{w}}_j, {\bm{x}}\rangle) \neq \dot{\sigma}(\langle {\bm{w}}_j, {\bm{x}}' \rangle)\\] and 0 otherwise, and \\(\mathcal{S}\\) consists of the \\(j \in [d_1]\\) such that \\(Z_j = 1\\). The \\(Z_j\\) are iid Bernoulli random variables with parameter \\(p\\), where \\(p \asymp \delta\\) by Lemma <a href="#lemma:cosine-probabilities" data-reference-type="ref" data-reference="lemma:cosine-probabilities">15</a>. By Chernoff’s inequality `\citep[see][Theorem 2.3.1]{vershynin2018high}`{=latex}, for all \\(t \geq d_1p\\) \\[\begin{aligned}
        \mathbb{P}\left(|\mathcal{S}| \geq t\right) &\leq e^{-d_1 p}\left(\frac{e d_1p}{t} \right)^t
    
\end{aligned}\\] Then setting \\(t = ed_1p\\) with \\(d_1 \geq \frac{1}{p}\log \frac{4}{\epsilon}\\) yields \\[\begin{aligned}
        \mathbb{P}\left(|\mathcal{S}| \geq ed_1 \delta \right) &\leq \mathbb{P}\left(|\mathcal{S}| \geq ed_1 p \right)\\
        &\leq e^{-d_1p}\\
        &\leq \frac{\epsilon}{4}.
    
\end{aligned}\\] By the lower bound of Chernoff’s inequality, for all \\(t \leq d_1p\\) \\[\begin{aligned}
        \mathbb{P}(|\mathcal{S}| \leq t) &\leq e^{-d_1p }\left(\frac{ed_1 p}{t}\right)^t.
    
\end{aligned}\\] Then setting \\(t = \frac{d_1p}{e}\\) with \\(d_1 \geq \frac{2}{e - 2} \frac{1}{p} \log \frac{4}{\epsilon}\\) yields \\[\begin{aligned}
        \mathbb{P}\left(|\mathcal{S}| \leq \frac{d_1p}{2} \right) &\leq \exp\left(-\frac{e - 2}{e}d_1p \right)\\
        &\leq \frac{\epsilon}{4}.
    
\end{aligned}\\] Therefore, with probability at least \\(1- \frac{\epsilon}{2}\\), \\[\frac{d_1\delta}{e}\leq |\mathcal{S}| \leq ed_1\delta.\\] Let us denote this event by \\(\omega\\). Observe that \\[\begin{aligned}
        \|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})  -  {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\|^2 &= 2\sum_{j \in \mathcal{S}}v_j^2
    
\end{aligned}\\] and recall that \\(v_j^2 \sim \mathcal{N}(0, 1)\\) for all \\(j \in [d_1]\\). By Bernstein’s inequality, for all \\(t \geq 0\\) \\[\begin{aligned}
        \mathbb{P}\left(\frac{1}{2}\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\|^2 \geq |\mathcal{S}| + t \;\; \middle| \;\; \mathcal{S} \right) &\leq 2\exp\left(-C_1 \min\left(\frac{t^2}{|\mathcal{S}| }, t \right) \right)
    
\end{aligned}\\] where \\(C_1 > 0\\) is a universal constant. Setting \\(t = |\mathcal{S}|\\) yields \\[\begin{aligned}
        \mathbb{P}\left(\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\| \geq 2\sqrt{|\mathcal{S}|} \;\; \middle| \;\; \mathcal{S} \right) &\leq 2\exp\left(-C_1|\mathcal{S}|\right).
    
\end{aligned}\\] Then \\[\begin{aligned}
        &\mathbb{P}\left(\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\| \leq 2\sqrt{ed_1 \delta}\right)
        \\&\geq\mathbb{P}\left(\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\| \leq 2\sqrt{ed_1 \delta}  ,\; \omega \right) \\&\geq \mathbb{P}\left(\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\| \leq 2\sqrt{|\mathcal{S}|}  ,\; \omega \right)
        \\&\geq \mathbb{E}\left[\mathbb{P}\left(\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\| \leq 2\sqrt{|\mathcal{S}|}   \;\; \middle|\;\; \mathcal{S}\right)1_{\omega}\right]\\
        &\geq \mathbb{E}\left[\left(1 - 2\exp\left(-C_1|\mathcal{S}|\right)\right) 
 1_{\omega}\right]\\
 &\geq \left(1 - 2\exp\left(-C_1\frac{d_1 \delta}{e} \right)\right)\mathbb{P}(\omega)\\
 &\geq \left(1 - 2\exp\left(-C_1\frac{d_1 \delta}{e} \right)\right)\left(1 - \frac{\epsilon}{2}\right),
    
\end{aligned}\\] where we used that \\(\omega\\) is measurable with respect to \\(\mathcal{S}\\) in the fourth line. So if \\(d_1 \geq \frac{e}{C_1 \delta} \log \frac{4}{\epsilon}\\), then \\[\begin{aligned}
        \mathbb{P}\left(\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\| \leq 2\sqrt{ed_1 \delta}\right) &\geq \left(1 - \frac{\epsilon}{2}\right)\left(1 - \frac{\epsilon}{2}\right)\\
        &\geq 1 - \epsilon.
    
\end{aligned}\\] ◻

</div>

<div id="lemma:frob-bounds-diagonal" class="lemma" markdown="1">

**Lemma 17**. *Suppose that \\({\bm{x}}\in \mathbb{S}^{d - 1}\\). If \\(d_1 = \Omega\left(\log \frac{1}{\epsilon}\right)\\), then with probability at least \\(1 - \epsilon\\), \\[\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})\| \lesssim \sqrt{d_1}.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Since \\(\dot{\sigma}(\langle {\bm{w}}_j, {\bm{x}}\rangle) \in \{0, 1\}\\) for all \\(j \in [d_1]\\), \\[\begin{aligned}
        \|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})\|^2 &= \sum_{j = 1}^{d_1} v_j^2 \dot{\sigma}(\langle {\bm{w}}_j, {\bm{x}}\rangle)\\
        &\leq \sum_{j = 1}^{d_1}v_j^2.
    
\end{aligned}\\] Since the entries \\(v_j\\) are iid standard Gaussian random variables, Bernstein’s inequality implies for all \\(t \geq 0\\) \\[\begin{aligned}
        \mathbb{P}(\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})\|^2 \geq d_1 + t) &\leq \mathbb{P}\left(\sum_{j = 1}^{d_1}v_j^2 \geq d_1 + t\right)\\
        &\leq 2\exp\left(-C\min\left(\frac{t^2}{d_1}, t \right) \right).
    
\end{aligned}\\] Setting \\(t = d_1\\) with \\(d_1 \geq \frac{1}{C} \log \frac{2}{\epsilon}\\) yields \\[\begin{aligned}
        \mathbb{P}(\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})\|^2 \geq 2d_1) \leq 2 \exp(-C d_1) \leq \epsilon.
    
\end{aligned}\\] ◻

</div>

Now we prove our main lemma which we will use to relate the separation between data points to the NTK.

<div id="lemma:separation-gradient-shallow" class="lemma" markdown="1">

**Lemma 18**. *Let \\({\bm{x}}, {\bm{x}}' \in \mathbb{S}^{d-1}\\) with \\(\|{\bm{x}}- {\bm{x}}'\| \leq \delta \leq 2\\). Let \\(\epsilon \in (0, 1)\\). If \\(d_1 = \Omega\left(\frac{1}{\delta} \log \frac{1}{\epsilon}\right)\\), then with probability at least \\(1 - \epsilon\\), \\[\begin{aligned}
        \|\nabla_{{\bm{\theta}}} f({\bm{x}}) - \nabla_{{\bm{\theta}}} f({\bm{x}}')\| \lesssim \sqrt{\delta}.
    
\end{aligned}\\]*

</div>

<div class="proof" markdown="1">

*Proof.* By Lemma <a href="#lemma:activation-pattern-lipschitz" data-reference-type="ref" data-reference="lemma:activation-pattern-lipschitz">16</a>, if \\(d_1 \gtrsim  \frac{1}{\delta}\log \frac{1}{\epsilon}\\), then with probability at least \\(1 - \frac{\epsilon}{4}\\), \\[\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\| \lesssim \sqrt{\delta d_1}.\\]

Let us denote this event by \\(\omega_1\\). By Lemma <a href="#lemma:frob-bounds-diagonal" data-reference-type="ref" data-reference="lemma:frob-bounds-diagonal">17</a>, if \\(d_1 \gtrsim \log \frac{1}{\epsilon}\\), then with probability at least \\(1 - \frac{\epsilon}{4}\\), \\[\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) \| \lesssim \sqrt{d_1}.\\] Let us denote this event by \\(\omega_2\\). If both \\(\omega_1\\) and \\(\omega_2\\) occur, then \\[\begin{aligned}
    &\|\nabla_{{\bm{W}}_1}f({\bm{x}}) - \nabla_{{\bm{W}}_1}f({\bm{x}}')\|_F\\&= \frac{1}{\sqrt{d_1}}\|({\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})) \otimes {\bm{x}}- ({\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')) \otimes {\bm{x}}'\|_F\\
    &\leq \frac{1}{\sqrt{d_1}}\|({\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})) \otimes {\bm{x}}- ({\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})) \otimes {\bm{x}}'\|_F \\&+ \frac{1}{\sqrt{d_1}}\|({\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})) \otimes {\bm{x}}' - ({\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')) \otimes {\bm{x}}'\|_F\\
    &\leq \frac{1}{\sqrt{d_1}} \|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}})\| \cdot \|{\bm{x}}- {\bm{x}}'\| + \frac{1}{\sqrt{d_1}}\|{\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}) - {\bm{v}}\odot \dot{\sigma}({\bm{W}}{\bm{x}}')\|\cdot \|{\bm{x}}'\|\\
    &\lesssim \frac{1}{\sqrt{d_1}}\sqrt{d_1} \delta + \frac{1}{\sqrt{d_1}}\sqrt{\delta d_1}\\
    &\lesssim \sqrt{\delta}.
\end{aligned}\\] By Lemma <a href="#lemma:feature-map-regularity" data-reference-type="ref" data-reference="lemma:feature-map-regularity">14</a>, if \\(d_l \gtrsim \log \frac{1}{\epsilon}\\), then with probability at least \\(1 - \frac{\epsilon}{2}\\), \\[\begin{aligned}
    \|\nabla_{{\bm{W}}_2}f({\bm{x}}) - \nabla_{{\bm{W}}_2}f({\bm{x}}')\| &= \frac{1}{\sqrt{d_1}}\|f_1({\bm{x}}) - f_1({\bm{x}}')\|\\
    &\lesssim \delta.
\end{aligned}\\] Let us denote this event by \\(\omega_3\\). If \\(\omega_1, \omega_2\\), and \\(\omega_3\\) all occur (which happens with probability at least \\(1 - \epsilon)\\), then \\[\begin{aligned}
    \|\nabla_{{\bm{\theta}}}f({\bm{x}}) - \nabla_{{\bm{\theta}}}f({\bm{x}}')\| &\lesssim \|\nabla_{{\bm{W}}_1}f({\bm{x}}) - \nabla_{{\bm{W}}_1}f({\bm{x}}')\|_F + \|\nabla_{{\bm{W}}_2}f({\bm{x}})  - \nabla_{{\bm{W}}_2}f({\bm{x}}')\|\\
    &\lesssim \sqrt{\delta} + \delta\\
    &\lesssim \sqrt{\delta}.
\end{aligned}\\] ◻

</div>

## Proof of Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a> [app:shallow-main]

<div class="proof" markdown="1">

*Proof.* First we prove the lower bound. Let \\(\lambda_1\\) be as it is defined in Lemma <a href="#lemma:K1-inf" data-reference-type="ref" data-reference="lemma:K1-inf">[lemma:K1-inf]</a>. By Lemma <a href="#lem:gram-to-hemisphere-transform" data-reference-type="ref" data-reference="lem:gram-to-hemisphere-transform">[lem:gram-to-hemisphere-transform]</a>, \\[\begin{aligned}
     &\lambda_1 = \inf_{\| {\bm{z}}\| = 1} \| T_{\dot{\sigma}} \mu_{{\bm{z}}} \|^2.
     
\end{aligned}\\] Let \\[\lambda = \left( 1 + \frac{d\log(1/\delta)}{\log(d)} \right)^{-3} \delta^2.\\] By Lemma <a href="#corr:hemisphere-transform-asymptotics" data-reference-type="ref" data-reference="corr:hemisphere-transform-asymptotics">[corr:hemisphere-transform-asymptotics]</a>, \\(\lambda_1 \geq C_1\lambda\\) for some constant \\(C_1 > 0\\). By Lemma <a href="#lemma:K1-inf" data-reference-type="ref" data-reference="lemma:K1-inf">[lemma:K1-inf]</a>, there exist constants \\(C_2, C_3 > 0\\) such that if \\(d_1 \geq \frac{C_2}{\lambda_1}\|{\bm{X}}\|^2 \log \frac{n}{\epsilon}\\) then \\[\begin{aligned}
        \mathbb{P}(\lambda_{\min}({\bm{K}}_1) < C_3 \lambda_1) \leq \frac{\epsilon}{2}.\label{eqn:C3lambda1-conditional}
    
\end{aligned}\\] Then for such \\(d_1\\), \\[\begin{aligned}
        \mathbb{P}(\lambda_{\min}({\bm{K}}_1) \geq C_3C_1 \lambda) \geq 1 - \frac{\epsilon}{2}.
    
\end{aligned}\\] This establishes the lower bound.

Next we prove the upper bound. Let \\(i, k \in [n]\\) be two indices with \\(i \neq k\\) such that \\(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta'\\). If \\(d_1 \gtrsim \frac{1}{\lambda}\log \frac{1}{\epsilon} \gtrsim \frac{1}{\delta'} \log \frac{1}{\epsilon}\\), then by Lemma <a href="#lemma:separation-gradient-shallow" data-reference-type="ref" data-reference="lemma:separation-gradient-shallow">18</a> there exists \\(C_4 > 0\\) such that \\[\begin{aligned}
        \mathbb{P}(\|\nabla_{{\bm{\theta}}}f({\bm{x}}_i)  - \nabla_{{\bm{\theta}}}f({\bm{x}}_k)\|^2 \geq C_4\delta') \geq 1 - \frac{\epsilon}{2}.
    
\end{aligned}\\] Let us denote this event by \\(\omega\\). If \\(\omega\\) occurs, then \\[\begin{aligned}
        \lambda_{\min}({\bm{K}}) &\lesssim ({\bm{e}}_i - {\bm{e}}_k)^T {\bm{K}}({\bm{e}}_i - {\bm{e}}_k)\\
        &= \| \nabla_{{\bm{\theta}}}f({\bm{x}}) - \nabla_{{\bm{\theta}}}f({\bm{x}}_k)\|^2\\
        &\lesssim \delta'.
    
\end{aligned}\\] Hence, with probability at least \\(1 - \frac{\epsilon}{2}\\), \\(\lambda_{\min}({\bm{K}}) \lesssim \delta'\\). This establishes the upper bound for the minimum eigenvalue. The two-sided bound then immediately follows from a union bound. ◻

</div>

## Uniform data on a sphere [app:subsec:uniform-sphere]

Our main bounds for the smallest eigenvalue of the NTK are stated in terms of the amount of separation between data points. To interpret our results in terms of probability distributions on the sphere, we will use a couple of lemmas which quantify the amount of separation for data which is uniformly distributed.

For \\(\delta \in (0, 1/2)\\) and \\({\bm{x}}\in \mathbb{S}^{d -1 }\\), we define the spherical cap \\[\text{Cap}({\bm{x}}, \delta) = \{{\bm{y}}\in \mathbb{S}^{d - 1}: \|{\bm{y}}- {\bm{x}}\| \leq \delta\}.\\] and the double spherical cap \\[\begin{aligned}
        \text{DoubleCap}({\bm{x}}, \delta) = \text{Cap}({\bm{x}}, \delta) \cup \text{Cap}(-{\bm{x}}, \delta).
    
\end{aligned}\\] By Lemma 2.3 of `\cite{ball1997elementary}`{=latex}, \\[\begin{aligned}
     dS(\text{Cap}({\bm{x}}, \delta)) \geq \frac{1}{2}\left(\frac{\delta}{2}\right)^{d - 1}.\label{eqn:spherical-cap-lower-bound}
    
\end{aligned}\\] We can also obtain a corresponding upper bound on the volume of a spherical cap.

<div id="lemma:spherical-cap-upper-bound" class="lemma" markdown="1">

**Lemma 19**. *For \\({\bm{x}}\in \mathbb{S}^{d - 1}\\) and \\(\delta \in (0, 1/2)\\), \\[dS(\mathop{\mathrm{Cap}}({\bm{x}}, \delta)) \leq \frac{4\sqrt{\pi}(C\delta)^{d - 1} }{d^2}.\\] Here \\(C > 0\\) is a universal constant.*

</div>

<div class="proof" markdown="1">

*Proof.* For \\(\phi \in [0, \pi]\\), let \\(\mathcal{S}_{\phi}\\) denote the set of all \\({\bm{x}}' \in \mathbb{S}^{d - 1}\\) such that the angle between \\({\bm{x}}\\) and \\({\bm{x}}'\\) is at most \\(\phi\\) (that is, \\(\langle {\bm{x}}, {\bm{x}}' \rangle \geq \cos(\phi)\\)). The measure of \\(\mathcal{S}_{\phi}\\) is given by \\[\begin{aligned}
        \frac{B(\sin^2(\phi); (d-1)/2, 1/2) }{B((d - 1)/2, 1/2)}
    
\end{aligned}\\] `\citep[see, e.g.][]{li2010concise}`{=latex}. Here the numerator refers to the incomplete beta function and the denominator refers to the beta function. We can bound \\[\begin{aligned}
        B\left(\sin^2(\phi); \frac{d - 1}{2}, \frac{1}{2}\right) &= \int_0^{\sin^2(\phi)} t^{(d-3)/2 }(1 - t)^{-1/2}dt\\
        &\leq \int_0^{\sin^2(\phi)}t^{(d - 3)/2}dt\\
        &= \frac{2}{d - 1} \sin(\phi)^{d - 1}.
    
\end{aligned}\\] and \\[\begin{aligned}
        B\left(\frac{d - 1}{2}, \frac{1}{2}\right) &= \frac{\Gamma\left(\frac{d -1}{2}\right)\Gamma\left(\frac{1}{2}\right) }{\Gamma\left(\frac{d}{2}\right) }\\
        &\geq \frac{\Gamma\left(\frac{d - 2}{2}\right)\sqrt{\pi} }{\Gamma\left(\frac{d}{2}\right) }\\
        &= \frac{2 \sqrt{\pi}}{d - 2}.
    
\end{aligned}\\] The above two bounds imply \\[\begin{aligned}
        dS(\mathcal{S}_{\phi}) \leq \frac{4\sqrt{\pi}\sin(\phi)^{d - 1} }{(d - 1)(d - 2) } \leq \frac{4\sqrt{\pi}\sin(\phi)^{d - 1} }{d^2} \leq \frac{4\sqrt{\pi}\phi^{d -1 } }{d^2}.\label{eqn:angular-cap-volume}
    
\end{aligned}\\] Now suppose that \\({\bm{x}}' \in \text{Cap}({\bm{x}}, \delta)\\). Then \\(\|{\bm{x}}- {\bm{x}}'\| \leq \delta\\), so \\(1 - \langle {\bm{x}}, {\bm{x}}' \rangle \leq 2\delta^2\\). Let \\(\phi = \arccos(\langle {\bm{x}}, {\bm{x}}' \rangle)\\) be the angle between \\({\bm{x}}\\) and \\({\bm{x}}'\\). By Taylor’s theorem, \\(\cos(\phi) = 1 - \frac{\phi^2}{2} + O(\phi^3)\\), so \\(1 - \cos(\phi) \asymp \phi^2\\) for \\(\phi \in [0, \pi]\\). Thus \\[2\delta^2 \geq 1 - \langle {\bm{x}}, {\bm{x}}' \rangle = 1 - \cos(\phi) \asymp \phi^2.\\] So the angle between \\({\bm{x}}\\) and \\({\bm{x}}'\\) is at most \\(C\delta\\) for some universal constant \\(C > 0\\). It follows that \\(\text{Cap}({\bm{x}}, \delta) \subseteq \mathcal{S}_{C\delta}\\). Finally by (<a href="#eqn:angular-cap-volume" data-reference-type="ref" data-reference="eqn:angular-cap-volume">[eqn:angular-cap-volume]</a>), \\[\begin{aligned}
        dS(\mathop{\mathrm{Cap}}({\bm{x}}, \delta)) \leq \frac{4\sqrt{\pi}(C\delta)^{d - 1} }{d^2}.
    
\end{aligned}\\] ◻

</div>

Since \\(\delta \leq \frac{1}{2}\\), the sets \\(\text{Cap}({\bm{x}}, \delta)\\) and \\(\text{Cap}(-{\bm{x}}, \delta)\\) are disjoint by the triangle inequality. Hence \\[\begin{aligned}
        dS(\text{DoubleCap}({\bm{x}}, \delta)) &= 2 \text{Cap}({\bm{x}}, \delta)
    
\end{aligned}\\] and in particular by Lemma <a href="#lemma:spherical-cap-upper-bound" data-reference-type="ref" data-reference="lemma:spherical-cap-upper-bound">19</a> \\[\begin{aligned}
         dS(\text{DoubleCap}({\bm{x}}, \delta)) \leq \frac{4\sqrt{\pi}(C\delta)^{d - 1} }{d^2}. \label{eqn:double-spherical-cap-bound}
    
\end{aligned}\\] for a constant \\(C > 0\\).

<div id="lemma:sphere-separation" class="lemma" markdown="1">

**Lemma 20**. *Suppose that \\(n \geq 2\\) and \\(\epsilon \in (0, 1)\\). If \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{S}^{d - 1}\\) are independent and uniformly distributed on \\(\mathbb{S}^{d - 1}\\), then with probability at least \\(1 - \epsilon\\), the dataset is \\(\delta\\)-separated with \\[\delta  \gtrsim \left(\frac{\epsilon}{n^2}\right)^{1/(d - 1)}.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Let \\({\bm{e}}= [1, 0, \cdots, 0]^T \in \mathbb{S}^{d - 1}\\). For each \\({\bm{x}}\in \mathbb{S}^{d - 1}\\), there exists an orthogonal matrix \\({\bm{O}}_x\\) such that \\({\bm{O}}_x {\bm{x}}= {\bm{e}}\\). Note that for all \\({\bm{x}}\in \mathbb{S}^{d - 1}\\) and \\(i \in [n]\\), \\({\bm{O}}_x {\bm{x}}_i \stackrel{d}{=} {\bm{x}}_i\\). Let \\(i, k \in [n]\\) with \\(i \neq k\\). Then for all \\(\delta \in (0, 1/2)\\), \\[\begin{aligned}
        \mathbb{P}(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta \text{ or } \|{\bm{x}}_i + {\bm{x}}_k\| \leq \delta) &= \mathbb{E}[\mathbb{P}(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta \text { or } \|{\bm{x}}_i + {\bm{x}}_k\| \leq \delta \mid {\bm{x}}_k) ]\\
        &= \mathbb{E}[\mathbb{P}(\|{\bm{O}}_{x_k}{\bm{x}}_i - {\bm{O}}_{x_k}{\bm{x}}_k\| \leq \delta \text { or } \|{\bm{O}}_{{\bm{x}}_k}{\bm{x}}_i + {\bm{O}}_{{\bm{x}}_k} {\bm{x}}_k\| \leq \delta \mid {\bm{x}}_k) ]\\
        &= \mathbb{E}[\mathbb{P}(\|{\bm{O}}_{x_k}{\bm{x}}_i - {\bm{e}}\| \leq \delta \text{ or } \|{\bm{O}}_{{\bm{x}}_k}{\bm{x}}_i + {\bm{e}}\| \leq \delta \mid {\bm{x}}_k) ]\\
        &= \mathbb{E}[\mathbb{P}(\|{\bm{x}}_i - {\bm{e}}\| \leq \delta \text{ or } \|{\bm{x}}_i + {\bm{e}}\| \leq \delta \mid {\bm{x}}_k) ]\\
        &= \mathbb{P}(\|{\bm{x}}_i - {\bm{e}}\| \leq \delta \text{ or } \|{\bm{x}}_i + {\bm{e}}\| \leq \delta).
    
\end{aligned}\\] The expression on the final line is the measure of \\(\text{DoubleCap}({\bm{e}}, \delta)\\), and by (<a href="#eqn:double-spherical-cap-bound" data-reference-type="ref" data-reference="eqn:double-spherical-cap-bound">[eqn:double-spherical-cap-bound]</a>) is bounded above by \\[\frac{4\sqrt{\pi}(C\delta)^{d - 1} }{d^2},\\] where \\(C > 0\\) is a constant. So \\[\begin{aligned}
        \mathbb{P}(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta \text{ or } \|{\bm{x}}_i + {\bm{x}}_k\| \leq \delta \;\; \text{ for some } i \neq k) &\leq \sum_{i \neq k}\mathbb{P}(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta \text{ or } \|{\bm{x}}_i + {\bm{x}}_k\| \leq \delta)\\
        &\leq \frac{4\sqrt{\pi}n^2(C\delta)^{d - 1} }{d^2}.
    
\end{aligned}\\] Setting \\(\delta = \min\left(\frac{1}{4}, \frac{1}{C}\left(\frac{\epsilon d^2}{4\sqrt{\pi}n^2} \right)^{1/(d - 1)}\right)\\), we obtain \\[\begin{aligned}
        \mathbb{P}(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta \text{ or } \|{\bm{x}}_i + {\bm{x}}_k\| \leq \delta \;\; \text{ for some } i \neq k) &\leq \epsilon.
    
\end{aligned}\\] Therefore, for this value of \\(\delta\\), the dataset is \\(\delta\\)-separated with probability at least \\(1 - \epsilon\\). To conclude, note that \\[\begin{aligned}
        \frac{1}{C}\left(\frac{\epsilon d^2}{4\sqrt{\pi}n^2} \right)^{1/(d - 1)} \gtrsim \left(\frac{\epsilon}{n^2}\right)^{1/(d - 1)}
    
\end{aligned}\\] since \\[\lim_{d \to \infty} \left(\frac{d^2}{4\sqrt{\pi}}\right)^{1/(d - 1)} = 1.\\] ◻

</div>

<div id="lemma:sphere-antiseparation" class="lemma" markdown="1">

**Lemma 21**. *Suppose that \\(n \geq 2\\) and \\(\epsilon \in (0, 1)\\). If \\({\bm{x}}_1, \cdots, {\bm{x}}_n \in \mathbb{S}^{d - 1}\\) are selected iid from \\(U(\mathbb{S}^{d - 1})\\), then with probability at least \\(1 - \epsilon\\), there exist \\(i, k \in [n]\\) with \\(i \neq k\\) such that \\[\|{\bm{x}}_i - {\bm{x}}_k\|  \lesssim \left(\frac{\log(1/\epsilon) }{n^2} \right)^{1/(d - 1) }.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Let \\({\bm{e}}= [1, 0, \cdots, 0]^T \in \mathbb{S}^{d - 1}\\). For each \\({\bm{x}}\in \mathbb{S}^{d - 1}\\), there exists an orthogonal matrix \\({\bm{O}}_x\\) such that \\({\bm{O}}_x {\bm{x}}= {\bm{e}}\\). Note that for all \\({\bm{x}}\in \mathbb{S}^{d - 1}\\) and \\(i \in [n]\\), \\({\bm{O}}_x {\bm{x}}_i \stackrel{d}{=} {\bm{x}}_i\\). Let \\(i, k \in [n]\\) with \\(i \neq k\\). Then for all \\(\delta \in (0, 1/2)\\), \\[\begin{aligned}
        \mathbb{P}(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta) &= \mathbb{E}[\mathbb{P}(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta \mid {\bm{x}}_k) ]\\
        &= \mathbb{E}[\mathbb{P}(\|{\bm{O}}_{x_k}{\bm{x}}_i - {\bm{O}}_{x_k}{\bm{x}}_k\| \leq \delta \mid {\bm{x}}_k) ]\\
        &= \mathbb{E}[\mathbb{P}(\|{\bm{O}}_{x_k}{\bm{x}}_i - {\bm{e}}\| \leq \delta \mid {\bm{x}}_k) ]\\
        &= \mathbb{E}[\mathbb{P}(\|{\bm{x}}_i - {\bm{e}}\| \leq \delta \mid {\bm{x}}_k) ]\\
        &= \mathbb{P}(\|{\bm{x}}_i - {\bm{e}}\| \leq \delta).
    
\end{aligned}\\] The expression on the final line is the measure of \\(\text{Cap}({\bm{e}}, \delta)\\), and by Lemma 2.3 of `\cite{ball1997elementary}`{=latex} it is bounded below by \\(\frac{1}{2}\left(\frac{\delta}{2}\right)^{d - 1}\\). For each \\(i \in [n]\\), let \\(\omega_i\\) denote the event that \\(\|{\bm{x}}_j - {\bm{x}}_k\| > \delta\\) for all \\(j, k \in [1, i]\\) with \\(j \neq k\\). Trivially \\(\mathbb{P}(\omega_1) = 1\\). If \\(\omega_i\\) occurs for some \\(i \in [1, n - 1]\\), then the sets \\(\text{Cap}({\bm{x}}_j, \delta/2)\\) for \\(j \in [i]\\) are disjoint. Indeed, if \\({\bm{x}}\in \text{Cap}({\bm{x}}_j, \delta/2) \cap \text{Cap}({\bm{x}}_k, \delta/2)\\), then by the triangle inequality \\[\|{\bm{x}}_j - {\bm{x}}_k\| \leq \|{\bm{x}}- {\bm{x}}_j\| + \|{\bm{x}}- {\bm{x}}_k\| \leq \frac{\delta}{2} + \frac{\delta}{2} = \delta\\] which contradicts \\(\omega_i\\). Now since these smaller spherical caps are disjoint, we can bound \\[\begin{aligned}
        dS\left(\cup_{j = 1}^i\{{\bm{x}}\in \mathbb{S}^{d - 1}: \|{\bm{x}}- {\bm{x}}_j\| \leq \delta \}\right) &\geq dS\left(\cup_{j = 1}^i\{{\bm{x}}\in \mathbb{S}^{d - 1}: \|{\bm{x}}- {\bm{x}}_j\| \leq \delta/2 \}\right)\\
        &= dS\left(\cup_{j = 1}^i \text{Cap}({\bm{x}}_j, \delta/2)\right)\\
        &= \sum_{j = 1}^i dS(\text{Cap}({\bm{x}}_j, \delta/2))\\
        &\geq \sum_{j = 1}^i \frac{1}{2}\left(\frac{\delta}{4}\right)^{d - 1}\\
        &= \frac{i}{2}\left(\frac{\delta}{4}\right)^{d - 1}.
    
\end{aligned}\\] Since \\({\bm{x}}_{i + 1}\\) is chosen independently from \\({\bm{x}}_1, \cdots, {\bm{x}}_i\\), this implies \\[\begin{aligned}
        \mathbb{P}(\omega_{i + 1} \mid \omega_i) &= \mathbb{P}(\|{\bm{x}}_{i + 1} - {\bm{x}}_j\| > \delta \;\; \forall j \in [i] \mid \omega_i)\\
        &\leq 1 - \frac{i}{2}\left(\frac{\delta}{4}\right)^{d - 1}.
    
\end{aligned}\\] By repeatedly conditioning we obtain \\[\begin{aligned}
        \mathbb{P}(\|{\bm{x}}_j - {\bm{x}}_k\| > \delta \;\; \forall j, k \in [n]) &= \mathbb{P}(\omega_n)\\
        &= \mathbb{P}(\omega_1)\prod_{i = 2}^n \mathbb{P}(\omega_i \mid \omega_1, \cdots, \omega_{i - 1})\\
        &= \prod_{i = 2}^n \mathbb{P}(\omega_i \mid \omega_{i - 1})\\
        &\leq \prod_{i = 2}^n \left(1 - \frac{i}{2}\left(\frac{\delta}{4}\right)^{d -1 }\right)\\
        &\leq \prod_{i = 2}^n \exp\left(-\frac{i}{2}\left(\frac{\delta}{4}\right)^{d - 1}\right)\\
        &\leq \exp\left(-\frac{n^2}{2}\left(\frac{\delta}{4}\right)^{d - 1}\right).
    
\end{aligned}\\] Let us set \\(\delta = \min\left(\frac{1}{4}, 4\left( \frac{2}{n^2}\log \frac{1}{\epsilon} \right)^{\frac{1}{d -1 }} 
 \right)\\). The above bounds imply that \\[\mathbb{P}(\|{\bm{x}}_j - {\bm{x}}_k\| > \delta \;\; \forall j, k \in [n]) \leq \epsilon\\] so with probability at least \\(1 - \epsilon\\), there exist \\(i, k \in [n]\\) such that \\(\|{\bm{x}}_i - {\bm{x}}_k\| \leq \delta\\) with \\[\delta \lesssim \left(n^{-2}\log \frac{1}{\epsilon}\right)^{1/(d - 1) }\\] which is what we needed to show. ◻

</div>

<div class="proof" markdown="1">

*Proof.* By Lemma <a href="#lemma:input-data-conditioning" data-reference-type="ref" data-reference="lemma:input-data-conditioning">3</a>, with probability at least \\(1 - \frac{\epsilon}{4}\\), \\[\|{\bm{X}}\|^2 \lesssim \left(1 + \frac{n + \log \frac{1}{\epsilon} }{d}\right).\\] Let us denote this event by \\(\omega_1\\). Let us define \\[\delta := \min_{i \neq k} \min(\|{\bm{x}}_i - {\bm{x}}_k\|, \|{\bm{x}}_i+ {\bm{x}}_k\|)\\] and \\[\delta' := \min_{i \neq k} \|{\bm{x}}_i - {\bm{x}}_k\|.\\] In particular, the dataset \\({\bm{x}}_1, \cdots, {\bm{x}}_n\\) is \\(\delta\\)-separated. By Lemma <a href="#lemma:sphere-separation" data-reference-type="ref" data-reference="lemma:sphere-separation">20</a>, with probability at least \\(1 - \frac{\epsilon}{4}\\), \\[\begin{aligned}
        \delta \gtrsim  \left(\frac{\epsilon}{n^2}\right)^{1/(d - 1)}.
    
\end{aligned}\\] Let us denote this event by \\(\omega_2\\). By Lemma <a href="#lemma:sphere-antiseparation" data-reference-type="ref" data-reference="lemma:sphere-antiseparation">21</a>, with probability at least \\(1 - \frac{\epsilon}{4}\\), \\[\delta' \lesssim  \left(\frac{\log(1/\epsilon) }{n^2} \right)^{1/(d - 1)}.\\] Let us denote this event by \\(\omega_3\\). We condition on \\(\omega_1, \omega_2,\\) and \\(\omega_3\\) for the remainder of the proof. Define \\[\lambda' = \left(1 + \frac{d\log(1/\delta) }{\log(d)}\right)^{-3} \delta^2\\] and \\[\lambda = \left(1 + \frac{\log(n/\epsilon)}{\log(d)}\right)^{-3}\left(\frac{\epsilon^2}{n^4}\right)^{1/(d - 1)};\\] note that \\[\begin{aligned}
        \lambda' &\gtrsim \left(1 + \frac{d \log\left( (n^2/\epsilon)^{1/(d-1)}\right) }{\log(d)} \right)^{-3}\left(\frac{\epsilon}{n^2}\right)^{2/(d - 1)}\\
        &\gtrsim \left(1 + \frac{\log(n/\epsilon)}{\log(d)}\right)^{-3}\left(\frac{\epsilon^2}{n^4}\right)^{1/(d - 1)}\\
        &= \lambda.
    
\end{aligned}\\]

By Theorem <a href="#thm:shallow-main" data-reference-type="ref" data-reference="thm:shallow-main">[thm:shallow-main]</a>, if \\[d_1 \gtrsim \frac{1}{\lambda}\left(1 + \frac{n + \log(1/\epsilon)}{d}\right) \log\left(\frac{n}{\epsilon}\right) \gtrsim \frac{1}{\lambda'}\|{\bm{X}}\|^2 \log\left(\frac{n}{\epsilon}\right),\\] then with probability at least \\(1 - \frac{\epsilon}{4}\\) over the network weights, \\[\begin{aligned}
        \lambda_{\min}({\bm{K}}) \gtrsim \lambda' \gtrsim \lambda
    
\end{aligned}\\] and \\[\begin{aligned}
        \lambda_{\min}({\bm{K}}) \lesssim \delta' \lesssim \left(\frac{\log(1/\epsilon)}{n^2}\right)^{1/(d - 1)}.
    
\end{aligned}\\] This is exactly the bound that we needed to show. By taking a union bound over all of the favorable events, it follows that this event happens with probability at least \\(1 - \epsilon\\). ◻

</div>

# Proof of Theorem <a href="#thm:deep-main" data-reference-type="ref" data-reference="thm:deep-main">[thm:deep-main]</a>

## Recap of the deep setting [app:deep-setting]

Recall for the deep case we consider fully connected networks with \\(L\\) layers and denote the layer widths with positive integers, \\(d_0, \cdots, d_L\\) where \\(d_0 = d\\) and \\(d_L = 1\\). For \\(l \in [L - 1]\\) we define the feature matrices \\({\bm{F}}_l \in \mathbb{R}^{d_l \times n}\\) as \\[{\bm{F}}_l = [f_l({\bm{x}}_1), \cdots, f_l({\bm{x}}_n)].\\] For \\(l \in [L - 1]\\) and \\({\bm{x}}\in \mathbb{R}^d\\) we define the activation patterns \\({\bm{\Sigma}}_l({\bm{x}}) \in \{0, 1\}^{d_l \times d_l}\\) to be the diagonal matrices \\[{\bm{\Sigma}}_l({\bm{x}}) = \text{diag}(\dot{\sigma}({\bm{W}}_{l}f_{l - 1}({\bm{x}}))).\\]

Lemma <a href="#lemma:NTKdecomp" data-reference-type="ref" data-reference="lemma:NTKdecomp">[lemma:NTKdecomp]</a> provides a useful decomposition of the NTK.

<div class="proof" markdown="1">

*Proof.* For any \\(i \in [n]\\), observe that \\(f({\bm{x}}_i, \cdot)\\) is a PAP function `\cite[Definition 5]{10.5555/3495724.3496288}`{=latex} and therefore \\(f({\bm{x}}_i, \cdot)\\) is differentiable almost everywhere `\cite[Proposition 4]{10.5555/3495724.3496288}`{=latex}. As the union of \\(n\\) null sets is also a null set, we conclude that there exists an open set \\(U\\) of full measure such that for all \\(i \in [n]\\) then \\(f({\bm{x}}_i, \theta)\\) is differentiable for any \\(\theta \in U\\).

Let \\(\tfrac{\partial f}{ \partial {\bm{\theta}}}\\) denote the true derivative of \\(f\\) with respect to \\({\bm{\theta}}\\) when it exists and be the minimum norm sub-gradient otherwise. Using `\cite[Corollary 13]{10.5555/3495724.3496288}`{=latex} then \\[\left( \prod_{l = 1}^{L - 1} \frac{d_l}{2}\right){\bm{K}}\stackrel{a.e.}{=} \frac{\partial F_L({\bm{\theta}})}{\partial {\bm{\theta}}}^T  \frac{\partial F_L({\bm{\theta}})}{\partial {\bm{\theta}}} = \sum_{l = 1}^{L} \frac{\partial F_L({\bm{\theta}})}{\partial {\bm{W}}_l }^T  \frac{\partial F_L({\bm{\theta}})}{\partial {\bm{W}}_l },\\] where \\(\tfrac{\partial F_L({\bm{\theta}})}{\partial {\bm{W}}_l} \in {\bm{R}}^{d_ld_{l-1} \times n}\\). By inspection, to prove the result claimed it therefore suffices to show for any \\(l \in  [L]\\), \\(\theta \in U\\) and \\(i,j \in [n]\\) that \\[\label{eq:decom-prf-element}
       \langle  \tfrac{\partial f_L({\bm{x}}_i; {\bm{\theta}})}{\partial {\bm{W}}_{l} }, \tfrac{\partial f_L({\bm{x}}_j; {\bm{\theta}})}{\partial {\bm{W}}_{l} } \rangle   = \left( f_{l-1}({\bm{x}}_i)^T f_{l-1}({\bm{x}}_j; {\bm{\theta}}) \right) \left( [{\bm{B}}_{l}]_{i,:}^T [{\bm{B}}_{l}]_{j,:} \right).\\] First observe \\[\begin{aligned}
        \langle  \tfrac{\partial f_L({\bm{x}}_i; {\bm{\theta}})}{\partial {\bm{W}}_L }, \tfrac{\partial f_L({\bm{x}}_j; {\bm{\theta}})}{\partial {\bm{W}}_L } \rangle &= f_{L-1}({\bm{x}}; {\bm{\theta}})^Tf_{L-1}({\bm{x}}; {\bm{\theta}}) \times 1
    
\end{aligned}\\] therefore establishing <a href="#eq:decom-prf-element" data-reference-type="eqref" data-reference="eq:decom-prf-element">[eq:decom-prf-element]</a> for \\(l = L\\). To establish <a href="#eq:decom-prf-element" data-reference-type="eqref" data-reference="eq:decom-prf-element">[eq:decom-prf-element]</a> for \\(l \in [L-1]\\), recall for \\(k \in [L-1]\\) that \\({\bm{\Sigma}}_k({\bm{x}}) = \text{diag}\left( \dot{\sigma}({\bm{W}}_k f_{k-1}({\bm{x}}))\right)\\) and define \\({\bm{\Sigma}}_L({\bm{x}}) = 1\\). Observe for \\(1 \leq l < k\\), \\(k \in [L]\\) that \\[\label{eq:decomp-prf-1}
    \frac{\partial f_{k}({\bm{x}}; {\bm{\theta}})}{ \partial {\bm{W}}_l } = {\bm{\Sigma}}_k({\bm{x}}) {\bm{W}}_k  \frac{\partial f_{k-1}({\bm{x}};{\bm{\theta}}) }{\partial {\bm{W}}_{l}}\\] while for \\(k=l\\) \\[\label{eq:decomp-prf-2}
         \frac{\partial f_{k}({\bm{x}}; {\bm{\theta}})}{ \partial {\bm{W}}_k } = {\bm{\Sigma}}_k({\bm{x}}) \otimes f_{k-1}({\bm{x}}; {\bm{\theta}})^T.\\] As a result, \\[\begin{aligned}
        \frac{\partial f_L({\bm{x}}; {\bm{\theta}})}{\partial \theta_l } &=  {\bm{W}}_L\left(\prod_{k = 1}^{L-l + 1} {\bm{\Sigma}}_{L-k}({\bm{x}}) {\bm{W}}_{L-k} \right) \frac{\partial f_{l}({\bm{x}}; {\bm{\theta}})}{ \partial {\bm{W}}_l }\\
        &= {\bm{W}}_L\left(\prod_{k = 1}^{L-l + 1} {\bm{\Sigma}}_{L-k}({\bm{x}}) {\bm{W}}_{L-k} \right) \left({\bm{\Sigma}}_l({\bm{x}}) \otimes f_{l-1}({\bm{x}}; {\bm{\theta}})\right)
    
\end{aligned}\\] where the first equality arises from iterating <a href="#eq:decomp-prf-1" data-reference-type="eqref" data-reference="eq:decomp-prf-1">[eq:decomp-prf-1]</a> and the second by applying <a href="#eq:decomp-prf-2" data-reference-type="eqref" data-reference="eq:decomp-prf-2">[eq:decomp-prf-2]</a>. Proceeding, \\[\begin{aligned}
        &\left \langle  \frac{\partial f_L({\bm{x}}_i)}{\partial \theta_l }, \frac{\partial f_L({\bm{x}}_j)}{\partial \theta_l } \right \rangle \\
        &= \left( f_{l-1}({\bm{x}}_i)^T f_{l-1}({\bm{x}}_j) \right) \left( \left({\bm{\Sigma}}_l({\bm{x}}_i) \prod_{k = l+1}^{L-1} {\bm{W}}_{k}^T {\bm{\Sigma}}_{k}({\bm{x}}_i)  \right) {\bm{W}}_L^T \right)^T \left(\left( {\bm{\Sigma}}_l({\bm{x}}_j) \prod_{k = l+1}^{L-1} {\bm{W}}_{k}^T {\bm{\Sigma}}_{k}({\bm{x}}_j) \right) {\bm{W}}_L^T \right) \\
        &= \left( f_{l-1}({\bm{x}}_i)^T f_{l-1}({\bm{x}}_j) \right) \left( [{\bm{B}}_{l}]_{i,:}^T [{\bm{B}}_{l}]_{j,:} \right)
    
\end{aligned}\\] as claimed. ◻

</div>

## Proof of Lemma <a href="#lemma:FeatureNorms" data-reference-type="ref" data-reference="lemma:FeatureNorms">[lemma:FeatureNorms]</a> [app:FeatureNorms]

<div id="lemma:feature-norm-helper1" class="lemma" markdown="1">

**Lemma 22**. *Let \\({\bm{z}}\in \mathbb{R}^d\\) be a fixed vector and \\({\bm{W}}\in \mathbb{R}^{m \times d}\\) a random matrix with mutually iid elements \\([{\bm{W}}]_{ij} \sim \mathcal{N}(0,1)\\) for all \\(i \in [m]\\) and \\(j \in [d]\\). Consider the random vector \\({\bm{y}}\in \mathbb{R}^{m}\\) defined as \\({\bm{y}}= \sigma({\bm{W}}{\bm{z}})\\) where \\(\sigma\\) denotes the ReLU function applied elementwise. For \\(\delta \in (0,1)\\) if \\(m \gtrsim \delta^{-2}\log(1/ \epsilon)\\) then \\[\mathbb{P}\left((1 - \delta)\frac{m}{2}\| {\bm{z}}\|^2 \leq \| {\bm{y}}\|^2 \leq (1 + \delta)\frac{m}{2}\| {\bm{z}}\|^2 \right) \geq 1 - \epsilon.\\]*

</div>

<div class="proof" markdown="1">

*Proof.* For \\(i\in [m]\\) define \\(Z_i = \tfrac{{\bm{w}}_i^T{\bm{z}}}{\| {\bm{z}}\|}\\), then \\(Z_i \sim \mathcal{N}(0,1)\\) are mutually iid. Let \\(B_i = \mathbbm{1}(Z_i>0)\\), note by symmetry \\(B_i \sim \text{Ber}(1/2)\\), furthermore these random variables for \\(i \in [n]\\) are also mutually iid with respect to one another. As \\(y_i = \| {\bm{z}}\|B_i Z_i\\) then \\[\begin{aligned}
    \| {\bm{y}}\|_2^2 
    &= \| {\bm{z}}\|^2 \sum_{i=1}^m B_i Z_i^2.
    
\end{aligned}\\] For convenience let \\({\bm{y}}' = {\bm{y}}/ \| {\bm{z}}\|\\)and define \\(\mathcal{S} = \{ i \in [n] \; : \; B_i = 1 \}\\), then \\[\|{\bm{y}}' \|^2 = \sum_{i\in \mathcal{S}} Z_i^2 \sim \chi^2(|\mathcal{S}|).\\] From `\cite[Lemma 1]{10.1214/aos/1015957395}`{=latex} we have for any \\(t>0\\) \\[\begin{aligned}
        \mathbb{P}\left( | \left(\| {\bm{y}}'\|^2 - | \mathcal{S}| \right) | \geq 2 \sqrt{ |\mathcal{S}| t}   \right) \leq 2\exp(-t).
    
\end{aligned}\\] For \\(\delta_1 \in (0,1)\\) let \\(t=\tfrac{|\mathcal{S}| \delta_1^2}{4}\\), then \\[\mathbb{P}\left( (1 - \delta_1)|\mathcal{S} | \leq \| {\bm{y}}' \|^2 \leq (1 + \delta_1)|\mathcal{S} | \right) \geq 1 - 2 \exp \left( -\frac{|\mathcal{S}| \delta_1^2}{4}  \right).\\] Observe \\(|\mathcal{S}| = \sum_{i=1}^{m} B_i \sim \text{Bin}(m, 1/2)\\). With \\(\delta_2 \in (0,1)\\) then applying Hoeffding’s inequality we have \\[\begin{aligned}
    \mathbb{P}\left( (1 - \delta_2) \frac{m}{2} \leq \sum_{i=1}^{m} B_i \leq (1 + \delta_2) \frac{m}{2}  \right) &\geq 1 - 2 \exp \left( -\frac{\delta_2^2 m}{2}\right).
    
\end{aligned}\\] Let \\(\omega\\) denote the event that \\((1 - \delta_2)\tfrac{m}{2} \leq |\mathcal{S} | \leq (1 + \delta_2)\tfrac{m}{2}\\). If \\(m \geq \frac{16}{\delta_1^2\delta_2^2 (1 - \delta_2 )} \log(4 / \epsilon)\\) then \\[\begin{aligned}
        &\mathbb{P}\left((1 - \delta_1)(1- \delta_2)\frac{m}{2} \leq \| {\bm{y}}' \|^2 \leq (1 + \delta_1)(1 + \delta_2)\frac{m}{2} \right)\\ &\geq \mathbb{P}\left((1 - \delta_1)(1- \delta_2)\frac{m}{2} \leq \| {\bm{y}}' \|^2 \leq (1 + \delta_1)(1 + \delta_2)\frac{m}{2}\; \mid \; \omega\right) \mathbb{P}(\omega)\\
        &\geq  \mathbb{P}\left( (1 - \delta_1)|\mathcal{S} | \leq \| {\bm{y}}'\|^2 \leq (1 + \delta_1)|\mathcal{S} | \; \mid \; \omega \;\right)\mathbb{P}(\omega)\\
        & \geq \left( 1 - 2 \exp \left( -\frac{(1 - \delta_2) \delta_1^2m}{8}  \right) \right) \left(1 - 2 \exp \left( -\frac{\delta_2^2 m}{2}\right) \right)\\
         & \geq \left( 1 - \frac{\epsilon}{2}\right)\left( 1 - \frac{\epsilon}{2}\right)\\
        & \geq 1 -\epsilon.
    
\end{aligned}\\] For some \\(\delta \in (0,1)\\) let \\(\delta_2 = \delta_1 = \delta/3\\), then if \\(m \geq 1944\delta^{-2}\log(4 / \epsilon)\\) we have \\[\mathbb{P}\left((1 - \delta)\frac{m}{2} \leq \| {\bm{y}}' \|^2 \leq (1 + \delta)\frac{m}{2} \right) \geq 1 - \epsilon\\] from which the result claimed follows. ◻

</div>

<div class="proof" markdown="1">

*Proof.* For \\(k\in [l]\\) let \\(\omega_k\\) denote the event that the inequality \\[\left(1 - \frac{1}{l} \right)^k \left( \prod_{h=1}^k \frac{d_h}{2}\right) \leq \| f_k({\bm{x}}) \|^2 \leq \left(1 + \frac{1}{l} \right)^k\left( \prod_{h=1}^k \frac{d_h}{2}\right)\\] holds. We proceed by induction to establish that \\(\mathbb{P}(\omega_k) \geq ( 1 - \tfrac{\epsilon}{l})^k\\) for all \\(k \in [l]\\). For the base case note that \\(f_1({\bm{x}}) = \sigma({\bm{W}}_1 {\bm{x}})\\) and \\(\| {\bm{x}}\|^2 = 1\\). Applying Lemma <a href="#lemma:feature-norm-helper1" data-reference-type="ref" data-reference="lemma:feature-norm-helper1">22</a> with \\(\delta = \frac{1}{l}\\), if \\(d_1 \gtrsim l^2 \log(l/\epsilon)\\) then \\(\mathbb{P}(\omega_1) \geq 1 - \tfrac{\epsilon}{l}\\). Now suppose for \\(k \in [l-1]\\) that \\(\mathbb{P}(\omega_k) \geq ( 1 - \tfrac{\epsilon}{l})^k\\). Note \\[\begin{aligned}
    \mathbb{P}(\omega_{k+1}) \geq \mathbb{P}(\omega_{k+1} \mid \omega_k) \mathbb{P}(\omega_k) \geq \mathbb{P}(\omega_{k+1} \mid \omega_k) ( 1 - \tfrac{\epsilon}{l})^k
    
\end{aligned}\\] Recall \\(f_{k+1}({\bm{x}}) = \sigma({\bm{W}}_1 f_{k}({\bm{x}}))\\). Conditioned on \\(\omega_k\\), then again applying Lemma <a href="#lemma:feature-norm-helper1" data-reference-type="ref" data-reference="lemma:feature-norm-helper1">22</a> with \\(\delta = \frac{1}{l}\\) and as \\(d_{k+1} \gtrsim l^2 \log(l/\epsilon)\\) we have \\[\mathbb{P}(\omega_{k+1}\mid \omega_k) \geq 1 - \tfrac{\epsilon}{l}\\] which completes the proof of the induction hypothesis. As \\((1 - \epsilon/l)^l \geq 1 - \epsilon\\) and \\(e^{-1} \leq (1 - 1/l)^l \leq (1 + 1/l)^l \leq e\\) then \\[e^{-1} \left( \prod_{h=1}^l \frac{d_h}{2}\right) \leq \| f_l({\bm{x}}) \|^2 \leq e \left( \prod_{h=1}^l \frac{d_h}{2}\right)\\] holds with probability at least \\(1-\epsilon\\). ◻

</div>

## Proof of Lemma <a href="#lemma:Frob-S" data-reference-type="ref" data-reference="lemma:Frob-S">[lemma:Frob-S]</a> [app:Frob-S]

<div class="restatable" markdown="1">

lemmaLemmaFrobS<span id="lemma:Frob-S" label="lemma:Frob-S"></span> Let \\({\bm{x}}\in \mathbb{S}^{d_0-1}\\), \\(L \geq 2\\) and assume \\(d_k \gtrsim L^2 \log \left(\frac{L}{\epsilon}\right)\\) for all \\(k \in [L - 1]\\). For any \\(l \in [L - 1]\\) with probability at least \\(1 - \epsilon\\) over the network parameters the following holds, \\[\| {\bm{S}}_{l}({\bm{x}}) \|_F^2 \asymp 2^{-L+l+1} \prod_{k = l}^{L-1} d_k.\\]

</div>

<div class="proof" markdown="1">

*Proof.* In what follows for convenience we define an empty product of scalars or matrices as the scalar one. Let \\(K \in \{L - 1\}\\), \\(l \in [K]\\), and for some arbitrary \\({\bm{x}}\in \mathbb{S}^{d_0-1}\\) define \\[\begin{aligned}
        {\bm{S}}_{l,K} &= {\bm{\Sigma}}_l({\bm{x}}) \prod_{k = l+1}^{K} {\bm{W}}_k^T \bm{\Sigma}_k({\bm{x}}).
    
\end{aligned}\\] Let \\(\omega_{l,K}\\) denote the event \\[\label{eq:frob-norm-K}
    \frac{1}{2} \left(1 - \frac{1}{L}\right)^K \leq \|{\bm{S}}_{l, K}\|_F^2\prod_{k = l}^K \frac{2}{d_l} \leq 2\left(1 + \frac{1}{L}\right)^K\\] It suffices to lower bound the probability of the event \\(\omega_{l, L-1}\\). Let \\(\mathcal{F}_{K}\\) denote the \\(\sigma\\)-algebra generated by \\({\bm{W}}_1, \cdots, {\bm{W}}_{K}\\) and note that \\({\bm{S}}_{l,K} \in \mathcal{F}_{K}\\). Let \\(\gamma_l\\) denote the event that \\(f_l({\bm{x}}) \neq 0\\), then \\[\begin{aligned}
        \mathbb{P}(\omega_{l, L-1}) &\geq \mathbb{P}(\omega_{l, L-1} \mid \omega_{l, L-2} ) \mathbb{P}(\omega_{l, L-2})\\
        &\geq \mathbb{P}(\omega_{l, L-1} \mid \omega_{l, L-2} ) \mathbb{P}(\omega_{l, L-2} \mid \omega_{l, L-3})\mathbb{P}(\omega_{l, L-3})\\
        & \geq \left(\prod_{h = l}^{L-2} \mathbb{P}(\omega_{l, h+1} \mid \omega_{l, h}) \right) \mathbb{P}(\omega_{l,l} \mid \gamma_l) \mathbb{P}(\gamma_l).
    
\end{aligned}\\] Fixing \\(\epsilon \in (0,1)\\), our goal is to show each term in this product is at least \\((1 - \tfrac{\epsilon}{L})\\): indeed, if this is true then \\[\begin{aligned}
        \mathbb{P}(\omega_{l, L-1}) \geq \left(1 - \frac{\epsilon}{L} \right)^{L-l} \geq 1 - \epsilon
    
\end{aligned}\\] and our task is complete. To this end, first observe that as \\(d_k \gtrsim L^2 \log(L / \epsilon)\\) for all \\(k \in [L-1]\\), then \\(\mathbb{P}(\gamma_l) \geq 1 - \tfrac{\epsilon}{L}\\) by Lemma <a href="#lemma:FeatureNorms" data-reference-type="ref" data-reference="lemma:FeatureNorms">[lemma:FeatureNorms]</a>. Proceeding to the term \\(\mathbb{P}(\omega_{l,l} \mid \gamma_l)\\), recall \\([\bm{\Sigma}_l({\bm{x}})]_{jj}=\mathbbm{1}([{\bm{W}}_l f_{l-1}({\bm{x}})]_j>0)\\). By symmetry the diagonal entries of \\(\bm{\Sigma}_l({\bm{x}})\\) are mutually iid Bernoulli random variables with parameter \\(\frac{1}{2}\\). Therefore, using Hoeffding’s inequality for all \\(t \geq 0\\) \\[\begin{aligned}
    \mathbb{P}\left(\left|\|\bm{\Sigma}_l({\bm{x}})\|_F^2 - \frac{d_l}{2}\right| \geq t \; \middle|\; \gamma_l \right) \leq 2 \exp\left(-\frac{t^2}{d_l}\right).
    
\end{aligned}\\] Let \\(t = d_l\\), if \\(d_l \geq \log \frac{2L}{\epsilon}\\) then with \\(K\geq 1\\), \\(L\geq 2\\) it follows that \\[\begin{aligned}
        \mathbb{P}(\omega_{l,l} \mid \gamma_l)
        &= \mathbb{P}\left(\frac{1}{2}\left(1 - \frac{1}{L}\right)^K \leq \|\bm{\Sigma}_l({\bm{x}})\|_F^2 \frac{2}{d_l} \leq 2\left(1 + \frac{1}{L}\right)^K \; \middle| \; \gamma_l \right)\\
        &\geq \mathbb{P}\left(\frac{1}{2} \leq \|\bm{\Sigma}_l({\bm{x}})\|_F^2 \frac{2}{d_l} \leq \frac{3}{2} \; \middle| \; \gamma_l \right)\\
        &\geq 1 - \mathbb{P}\left(\left|\|\bm{\Sigma}_l({\bm{x}})\|_F^2 - \frac{d_l}{2}\right| \geq \frac{d_l}{4} \; \middle|\; \gamma_l \right)\\
        &\geq 1 - \frac{\epsilon}{L}.
    
\end{aligned}\\] We now proceed to analyze \\(\mathbb{P}(\omega_{l, h+1} \mid \omega_{l, h})\\) for \\(h \in [l, K-1]\\). Note if \\(\omega_{l, h}\\) is true then \\(\|{\bm{S}}_{l, h} \|_F^2 > 0\\). By definition this implies \\(\| \bm{\Sigma}_l({\bm{x}}) \|_F^2 > 0\\), however, if \\(f_h({\bm{x}}) = 0\\) then \\(\| \bm{\Sigma}_l({\bm{x}}) \|_F^2 = 0\\). Therefore \\(\omega_{l, h}\\) being true implies \\(f_h({\bm{x}}) \neq 0\\). For convenience in what follows we denote the \\(j\\)th column of \\({\bm{W}}_{h+1}\\) as \\({\bm{w}}_j\\). By definition \\[{\bm{S}}_{l, h+1} = {\bm{S}}_{l, h}{\bm{W}}_{h+1}^T \bm{\Sigma}_{h+1}({\bm{x}}),\\] therefore, \\[\begin{aligned}
        \mathbb{E}[\|{\bm{S}}_{l, h+1}\|_F^2 \mid \mathcal{F}_{h} ] &= \mathbb{E}[\|{\bm{S}}_{l, h} {\bm{W}}_{h+1}^T \bm{\Sigma}_{h+1}({\bm{x}})\|_F^2  \mid \mathcal{F}_{h} ]\\
        &= \mathbb{E}\left[\sum_{j = 1}^{d_{h+1}} \left\|{\bm{S}}_{l, h}{\bm{w}}_j\right\|^2 \dot{\sigma}(\langle {\bm{w}}_j, f_{h}({\bm{x}}) \rangle) \; \middle| \; \mathcal{F}_{h}\right].
    
\end{aligned}\\] As highlighted already, if we condition on \\(\omega_{l, h}\\) then \\(f_h({\bm{x}})\neq 0\\) and therefore the random variables \\((\dot{\sigma}(\langle {\bm{w}}_j, f_{h}({\bm{x}}) \rangle))_{j \in d_{h+1}}\\) are mutually iid Bernoulli random variables with parameter \\(\frac{1}{2}\\). Again by symmetry \\(\dot{\sigma}(\langle {\bm{w}}_j, f_{h}({\bm{x}}) \rangle)\\) is independent of \\(\| {\bm{S}}_{l, h}{\bm{w}}_j \|^2\\). Therefore conditioned on \\(\omega_{l,h}\\) \\[\begin{aligned}
        \sum_{j = 1}^{d_{h+1}}\mathbb{E}[\|{\bm{S}}_{l, h}{\bm{w}}_j\|^2 \mid \mathcal{F}_{d_{h+1}} ]\mathbb{E}[\dot{\sigma}(\langle {\bm{w}}_j, f_{h}({\bm{x}}) \rangle) \mid \mathcal{F}_{h} ]
        &= \frac{1}{2}\sum_{j = 1}^{d_{h+1}}\mathbb{E}[\|{\bm{S}}_{l, h}{\bm{w}}_j\|^2 \mid \mathcal{F}_{h} ]\\
        &= \frac{1}{2}\sum_{j = 1}^{d_{h+1}}\|{\bm{S}}_{l, h}\|_F^2\\
        &= \frac{d_{h+1}}{2}\|{\bm{S}}_{l, h}\|_F^2.
    
\end{aligned}\\] Moreover, under the same conditioning \\[\begin{aligned}
        \left\|\left\|{\bm{S}}_{l, h}{\bm{w}}_j\right\|^2 \dot{\sigma}(\langle {\bm{w}}_j, f_{h}({\bm{x}}) \rangle)  \right\|_{\psi_1} &\leq \| \|{\bm{S}}_{l, h}{\bm{w}}_j\|^2 \|_{\psi_1}\\
        &= \| \|{\bm{S}}_{l, h}{\bm{w}}_j\| \|_{\psi_2}^2\\
        &\lesssim \|{\bm{S}}_{l, h}\|_F^2
    
\end{aligned}\\] where the last line follows from Theorem 6.3.2 of `\cite{vershynin2018high}`{=latex}. As a result, conditioned on \\(\omega_{l,h}\\) then using Bernstein’s inequality `\cite[Theorem 2.8.1]{vershynin2018high}`{=latex} there exists an absolute constant \\(c\\) such that for all \\(t \geq 0\\) \\[\begin{aligned}
        \mathbb{P}\left(\left|\|{\bm{S}}_{l, h+1}\|_F^2 -\frac{d_{h+1}}{2}\|{\bm{S}}_{l, h}\|_F^2\right| \geq  t \; \middle| \; \mathcal{F}_{h} \right) &\leq 2\exp\left(-c \min\left(\frac{t^2}{d_{h+1}\|{\bm{S}}_{l, h}\|_F^4 }, \frac{t}{\|{\bm{S}}_{l, h}\|_F^2} \right) \right).
    
\end{aligned}\\] If \\(d_{h+1} \geq \frac{4L^2}{c} \log \frac{2L}{\epsilon}\\) and \\(t = \frac{d_{h+1}\|{\bm{S}}_{l, h}\|_F^2 }{2L }\\) then conditioning on \\(\omega_{l,h}\\) we obtain \\[\begin{aligned}
        \mathbb{P}\left(\left|\|{\bm{S}}_{l, h+1}\|_F^2 -\frac{d_K}{2}\|{\bm{S}}_{l, h}\|_F^2\right| \geq  \frac{d_{h+1} }{2L}\|{\bm{S}}_{l, h}\|_F^2 \; \middle| \; \mathcal{F}_{h} \right) &\leq \frac{\epsilon}{L}.
    
\end{aligned}\\] As a result, for any \\(h \in [l, K-1]\\) we have \\(\mathbb{P}(\omega_{l, h+1} \mid \omega_{l,h }) \geq 1 - \tfrac{\epsilon}{L}\\) from which the result claimed follows. ◻

</div>

## Proof of Lemma <a href="#lemma:Op-S" data-reference-type="ref" data-reference="lemma:Op-S">[lemma:Op-S]</a> [app:Op-S]

<div class="restatable" markdown="1">

lemmaLemmaOpS<span id="lemma:Op-S" label="lemma:Op-S"></span> Let \\({\bm{x}}\in \mathbb{S}^{d_0-1}\\), \\(L \geq 3\\) and assume \\(d_k \geq d_{k + 1}\\) and \\(d_k \gtrsim  \sqrt{\log \frac{1}{\epsilon} }\\) for all \\(k \in [L - 1]\\). For any \\(l \in [L-1]\\) with probability at least \\(1 - \epsilon\\) over the network parameters the following holds, \\[\left\| {\bm{S}}_l({\bm{x}}) \right\|^2 \lesssim \prod_{k = l}^{L - 2}d_{k}.\\]

</div>

<div class="proof" markdown="1">

*Proof.* By Theorem 4.4.5 of `\cite{vershynin2018high}`{=latex}, for any \\(k \in [L-1]\\) and all \\(t \geq 0\\) \\[\mathbb{P}\left(\|{\bm{W}}_k\| \leq C(\sqrt{d_{k - 1}} + \sqrt{d_{k}} + t)  \right) \geq 1 - 2 e^{-t^2}.\\] As \\(d_{k-1} \geq d_{k} \geq \sqrt{\log \frac{2L}{\epsilon} }\\), then setting \\(t = \sqrt{\log \frac{2}{\epsilon} }\\) yields \\[\begin{aligned}
        \mathbb{P}\left(\|{\bm{W}}_k\| \leq 3C_1 \sqrt{d_{k - 1}} \right) &\geq \mathbb{P}\left(\|{\bm{W}}_k\| \leq C(\sqrt{d_{k - 1}} + \sqrt{d_k} + t)\right)\\
        &\geq 1 - \frac{\epsilon}{L}.
    
\end{aligned}\\] Using a union bound it follows that \\[\mathbb{P}\left(\|{\bm{W}}_k\| \leq 3C_1 \max \{ \sqrt{d_{l -1 }},  \sqrt{d_{l }}\} \;\; \forall k \in [ L - 1] \right) \geq 1 - \epsilon.\\] Note that \\(\| {\bm{\Sigma}}_k({\bm{x}}) \| \leq 1\\) for all \\(k \in [L-1]\\), therefore conditional on the above event we have \\[\begin{aligned}
        \| {\bm{S}}_{l}({\bm{x}})\| &= \left\|{\bm{\Sigma}}_{l}({\bm{x}}) \left(\prod_{k = l+1}^{L - 1}{\bm{W}}_k^T {\bm{\Sigma}}_k({\bm{x}})\right) \right\| \\
        &\leq \|{\bm{\Sigma}}_{l}({\bm{x}})\|\left(\prod_{k = l+1}^{L - 1}\|{\bm{W}}_k\| \|{\bm{\Sigma}}_k({\bm{x}})\|\right)\\
        &\leq \prod_{k = l+1}^{L - 1}\|{\bm{W}}_k\|\\
        &\lesssim \prod_{k = l}^{L - 2}\sqrt{d_{k}}.
    
\end{aligned}\\] To conclude we square both sides. ◻

</div>

## Proof of Lemma <a href="#lemma:min-B2" data-reference-type="ref" data-reference="lemma:min-B2">[lemma:min-B2]</a>

<div class="proof" markdown="1">

*Proof.* Let \\({\bm{x}}\in \mathbb{S}^{d_0-1}\\) be arbitrary and recall \\({\bm{S}}_{l}({\bm{x}}) = {\bm{\Sigma}}_l({\bm{x}}) \left( \prod_{k = l+1}^{L-1} {\bm{W}}_k^T {\bm{\Sigma}}_k({\bm{x}}) \right)\\). Also recall that \\({\bm{W}}_L^T \in \mathbb{R}^{d_{L-1}}\\) is distributed as \\({\bm{W}}_L^T \sim \mathcal{N}(\textbf{0}_{d_{L-1}}, \textit{I}_{d_{L_1}})\\). Therefore by `\citet[Theorem 6.3.2]{vershynin2018high}`{=latex} for any \\({\bm{A}}\in \mathbb{R}^{d_2 \times d_{L-1}}\\) and \\(t \geq 0\\) \\[\begin{aligned}
        \mathbb{P}( | \|{\bm{A}}{\bm{W}}_L^T \|_2 - \|{\bm{A}}\|_F | \geq t) \leq 2 \exp \left( -\frac{Ct^2}{ \|{\bm{A}}\|_2^2}\right)
    
\end{aligned}\\] for some constant \\(C>0\\). As a result, with \\(t = \tfrac{1}{2}\| {\bm{A}}\|_F^2\\) then for some constant \\(C>0\\) \\[\mathbb{P}\left( \frac{1}{4} \| {\bm{A}}\|_F^2 \leq \| {\bm{A}}{\bm{W}}_L^T \|_2^2 \leq \frac{3}{4} \| {\bm{A}}\|_F^2 \right) \geq 1 - \exp\left( -C\frac{\| {\bm{A}}\|_F^2}{\| {\bm{A}}\|_2^2}\right).\\] Therefore, in order to lower bound \\(\| {\bm{S}}_{l}({\bm{x}}) {\bm{W}}_L^T\|_2^2\\) with high probability it suffices to condition on a suitable upper bound for \\(\| {\bm{S}}_{L-1}({\bm{x}}) \|_2^2\\) and a suitable lower bound for \\(\| {\bm{S}}_{L-1}({\bm{x}}) \|_F^2\\). Let \\(\omega\\) denote the event that both \\[\| {\bm{S}}_l \|_F^2 \asymp 2^{L-l-1}\prod_{k=l}^{L-1} d_k\\] and \\[\| {\bm{S}}_l({\bm{x}}) \|^2 \lesssim \prod_{k=l}^{L-2} d_{k}\\] are true. Combining Lemmas <a href="#lemma:Frob-S" data-reference-type="ref" data-reference="lemma:Frob-S">[lemma:Frob-S]</a> and <a href="#lemma:Op-S" data-reference-type="ref" data-reference="lemma:Op-S">[lemma:Op-S]</a> using a union bound, then as long as \\(L \geq 3\\), \\(d_k \geq d_{k+1}\\) and \\(d_k \gtrsim L^2\log \frac{nL}{\epsilon}\\) for all \\(k \in [L - 1]\\) then \\(\mathbb{P}(\omega) \geq 1 - \tfrac{\epsilon}{2}\\). As a result and aslo as \\(d_{L-1} \gtrsim 2^L \log(2 / \epsilon)\\) then \\[\begin{aligned}
    \mathbb{P}\left(  \| {\bm{S}}_l({\bm{x}})  {\bm{W}}_L^T \|_2^2 \asymp 2^{L-l-1}\prod_{k=l}^{L-1} d_k  \right) & \geq \mathbb{P}\left(  \| {\bm{S}}_l({\bm{x}})  {\bm{W}}_L^T \|_2^2 \asymp 2^{L-l-1}\prod_{k=l}^{L-1} d_k \; \mid \; \omega  \right) \mathbb{P}(\omega)\\
    & \geq \mathbb{P}\left( \frac{1}{4} \| {\bm{S}}_l({\bm{x}})  \|_F^2 \leq \| {\bm{S}}_l({\bm{x}})  {\bm{W}}_L^T \|_2^2 \leq \frac{3}{4} \| {\bm{S}}_l({\bm{x}}) \|_F^2  \; \mid \; \omega \right)  \mathbb{P}(\omega) \\
    &\geq 1 - \exp\left( -C 2^{-L} \frac{ \prod_{k=l}^{L-1} d_k}{\prod_{k=l}^{L-2} d_{k}} \right)\mathbb{P}(\omega)\\
    & \geq 1 - \exp\left( -C 2^{-L} d_{L-1} \right)\mathbb{P}(\omega)\\
    & \geq \left(1 - \frac{\epsilon}{2} \right)\left(1 - \frac{\epsilon}{2} \right)\\
    & \geq 1 - \epsilon
    
\end{aligned}\\] as claimed. ◻

</div>

## Proof of Theorem <a href="#thm:deep-main" data-reference-type="ref" data-reference="thm:deep-main">[thm:deep-main]</a> [app:thm-deep-main]

<div class="proof" markdown="1">

*Proof.* Recall <a href="#eq:min-eig-NTK-bounds1" data-reference-type="eqref" data-reference="eq:min-eig-NTK-bounds1">[eq:min-eig-NTK-bounds1]</a>, \\[2^{L-1}\left( \prod_{l = 1}^{L - 1}\frac{1}{d_l}\right) \lambda_{\min}\left( {\bm{F}}_{1} {\bm{F}}_{1}^T \right) \min_{i \in [n]} \|[{\bm{B}}_2]_{i,:} \|^2 \leq \lambda_{\min}({\bm{K}}) \leq  2^{L-1}\left( \prod_{l = 1}^{L - 1}\frac{1}{d_l}\right) \sum_{l=0}^{L-1} \| f_l({\bm{x}}_i) \|^2 \| [{\bm{B}}_{l+1}]_{i,:} \|^2,\\] where the upper bound holds for any \\(i \in [n]\\). We start by analyzing the lower bound. Observe that \\({\bm{F}}_1{\bm{F}}_1^T= \sigma({\bm{W}}_1 {\bm{X}})^T \sigma({\bm{W}}_1 {\bm{X}})\\) has the same distribution as \\(d_1 {\bm{K}}_2\\) in the shallow setting; see <a href="#eq:ntk-shallow-decomp" data-reference-type="eqref" data-reference="eq:ntk-shallow-decomp">[eq:ntk-shallow-decomp]</a>. Let \\(\lambda_2\\) be defined as in Lemma <a href="#lemma:K2-inf" data-reference-type="ref" data-reference="lemma:K2-inf">[lemma:K2-inf]</a>: \\[\lambda_2 = d_0 \lambda_{\min} \left( \mathbb{E}_{{\bm{u}}\sim U(\mathbb{S}^{d_0 -1})}\left[ \sigma({\bm{u}}^T{\bm{X}})^T\sigma({\bm{u}}^T{\bm{X}}) \right]\right) = \lambda_{min}\left( {\bm{K}}_{\sqrt{d_0} \sigma}^{\infty} \right).\\] As the dataset \\({\bm{x}}_1, {\bm{x}}_2, \cdots, {\bm{x}}_n \in \mathbb{S}^{d_0 -1}\\) is \\(\delta\\)-separated then by Lemma <a href="#corr:hemisphere-transform-asymptotics" data-reference-type="ref" data-reference="corr:hemisphere-transform-asymptotics">[corr:hemisphere-transform-asymptotics]</a> \\[\lambda_2 \gtrsim  \left(1+ \frac{d_0\log(1/\delta) }{\log d_0}\right)^{-3}\delta^4.\\] Furthermore, if \\(d_1 \gtrsim \tfrac{n}{\lambda_2} \log \left( \tfrac{n}{\lambda_2}\right) \log \left( \tfrac{n}{\epsilon}\right)\\) then by Lemma <a href="#lemma:K2-inf" data-reference-type="ref" data-reference="lemma:K2-inf">[lemma:K2-inf]</a> \\[\lambda_{\min}({\bm{F}}_1{\bm{F}}_1^T) \gtrsim d_1 \lambda_2\\] with probability at least least \\(1-\tfrac{\epsilon}{4}\\) and as a result \\[\lambda_{\min}({\bm{F}}_1{\bm{F}}_1^T) \gtrsim d_1 \left(1 + \frac{\log(n / \epsilon)}{\log(d_0)} \right)^{-3} \delta^4\\] with probability at least \\(1- \tfrac{\epsilon}{4}\\). Furthermore, as \\(L \geq 3\\), \\(d_l \geq d_{l+1}\\) for all \\(l \in [L - 1]\\) and \\(d_{L-1} \gtrsim 2^L \log \left (\frac{4nL}{\epsilon}\right)\\) then \\[\min_{i \in [n]} \|[{\bm{B}}_2]_{i,:} \|^2 \gtrsim 2^{-L} \prod_{k = 2}^{L-1} d_k\\] with probability at least \\(1 - \tfrac{\epsilon}{4}\\). Via a union bound we conclude that the condition \\[\begin{aligned}
      2^{L-1}\left( \prod_{l = 1}^{L - 1}\frac{1}{d_l}\right) \lambda_{\min}({\bm{F}}_1{\bm{F}}_1^T)  \min_{i \in [n]} \|[{\bm{B}}_2]_{i,:} \|^2 & \gtrsim \left(1 + \frac{\log(n / \epsilon)}{\log(d_0)} \right)^{-3} \delta^4
    
\end{aligned}\\] holds with probability at least \\(1 - \tfrac{\epsilon}{2}\\). Fixing some \\(i\in [n]\\), for the upper bound observe trivially by construction that \\[\begin{aligned}
        \| f_0({\bm{x}}_i) \|^2 \| [{\bm{B}}_1]_{i,:} \|^2 = 1.
    
\end{aligned}\\] By assumption \\(d_k \gtrsim L^2 \log(4L^2/\epsilon)\\) for all \\(k \in [L-1]\\). With \\(l \in [0, L-1]\\) then by Lemma <a href="#lemma:FeatureNorms" data-reference-type="ref" data-reference="lemma:FeatureNorms">[lemma:FeatureNorms]</a> \\[\| f_l({\bm{x}}_i) \|^2 \lesssim 2^{-l}\prod_{k=1}^{l} d_k\\] holds with probability at least \\(1- \tfrac{\epsilon}{4L}\\). Likewise by Lemma <a href="#lemma:Frob-S" data-reference-type="ref" data-reference="lemma:Frob-S">[lemma:Frob-S]</a> for \\(l \in [2, L]\\), \\[\| [{\bm{B}}_{l}]_{i,:} \|^2 = \| {\bm{S}}_l({\bm{x}}_i){\bm{W}}_L^T \| \lesssim 2^{-L+l+1} \prod_{k=l}^{L-1} d_k\\] with probability at least \\(1- \tfrac{\epsilon}{4L}\\). Combining these via a union bound then for any \\(l \in [0, L-1]\\), \\[\| f_l({\bm{x}}_i) \|^2 \| [{\bm{B}}_{l}]_{i,:} \|^2 \lesssim 2^{-L+1} \prod_{k=1}^{L-1} d_k\\] holds with probability at least \\(1- \tfrac{\epsilon}{2L}\\). Again using a union bound now over the layers, it follows that \\[\begin{aligned}
        2^{L-1}\left( \prod_{l = 1}^{L - 1}\frac{1}{d_l}\right) \sum_{l=0}^{L-1} \| f_l({\bm{x}}_i) \|^2 \| [{\bm{B}}_{l+1}]_{i,:} \|^2 \lesssim L 2^{L-1}\left( \prod_{l = 1}^{L - 1}\frac{1}{d_l}\right)  2^{-L+1} \left( \prod_{l=1}^{L-1} d_l \right) = L
    
\end{aligned}\\] with probability at least \\(1 - \tfrac{\epsilon}{2}\\). As a result, using a final union bound we conclude both the upper and lower bounds hold with probability at least \\(1-\epsilon\\). ◻

</div>
