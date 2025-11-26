#  <span class="smallcaps">CoBo</span>: Collaborative Learning via Bilevel Optimization

## Abstract

Collaborative learning is an important tool to train multiple clients more effectively by enabling communication among clients. Identifying helpful clients, however, presents challenging and often introduces significant overhead. In this paper, we model *client-selection* and *model-training* as two interconnected optimization problems, proposing a novel bilevel optimization problem for collaborative learning. We introduce <span class="smallcaps">CoBo</span>, a *scalable* and *elastic*, SGD-type alternating optimization algorithm that efficiently addresses these problem with theoretical convergence guarantees. Empirically, <span class="smallcaps">CoBo</span> achieves superior performance, surpassing popular personalization algorithms by 9.3% in accuracy on a task with high heterogeneity, involving datasets distributed among 80 clients.[^1]

# Introduction

In a classic collaborative learning scenario, \\(n\\) clients, each with a distinct machine learning task, seek solutions that potentially outperform their individual solvers through a collective effort. Common collaborative learning frameworks generally alternate between training local models on individual datasets and synchronizing updates between collaborators. More concretely, during the computation step, client \\(i \in [n]\\) trains a \\(d\\)-dimensional model \\(\bm{x}_i \in \mathbb{R}^d\\) to minimize its loss function, \\(f_i: \mathbb{R}^d \rightarrow \mathbb{R}\\). In the subsequent communication step, client \\(i\\) exchanges updates with collaborators, potentially benefiting from collaboration.

While there is a plethora of collaborative learning frameworks, the ideal way to collaborate remains under-exploited. The <span class="smallcaps">FedAvg</span> `\citep{mcmahan2017communication,kairouz2021advances}`{=latex} algorithm learns one global model over pooled datasets from all clients, i.e., \\(\min_{\bm{x}\in \mathbb{R}^d} \frac{1}{n}\sum_{i=1}^{n} f_i(\bm{x})\\). However, due to heterogeneous data distributions between clients, a global model may significantly under-perform compared to personal models trained on local datasets for certain clients, which can discourage their participation in collaborative training `\citep{mohri2019agnostic}`{=latex}. <span class="smallcaps">Ditto</span> trains personal models with a regularization term that penalizes their deviation from a global model `\citep{smith2021ditto}`{=latex}. Although <span class="smallcaps">Ditto</span> enables personal models to leverage the global model, it offers only a coarse-grained level of collaboration. In instances where clients’ data exhibit significant differences, the <span class="smallcaps">Ditto</span> algorithm is constrained to facilitating collaboration at a global level, thereby neglecting the inherent client heterogeneity structure.

Clustering-based federated learning algorithms have been developed to accommodate scenarios in which clients’ data originate from multiple clusters `\citep{ghosh2020efficient,werner2023provably}`{=latex}. Nevertheless, these algorithms typically inherit the limitations associated with clustering techniques, including the need to predetermine the number of clusters, initialize cluster centers, and other such prerequisites, which can diminish their practical utility.

In this paper, we propose to use a bilevel optimization framework to enhance collaborative learning, by discovering better structural relationships among clients. The inner problem focuses on optimizing a binary collaborator selection variable \\(w_{ij}\in\{0,1\}\\), which is determined based on a gradient alignment measure for each pair of clients. In the outer problem, we train personalized models \\(\bm{X}\in\mathbb{R}^{n\times d}\\), while incorporating a penalization term that accounts for the distances between clients, as dictated by the collaboration weights established in the inner problem.

The contributions of the paper can be summarized as follows.

- We model collaborative learning through a novel bilevel optimization formulation that yields more generalizable solutions by fully exploiting the inherent structure of collaboration.

- We propose <span class="smallcaps">CoBo</span>, an SGD-type alternating optimization algorithm that efficiently solves the bilevel problem. <span class="smallcaps">CoBo</span> scales with the number of clients \\(n\\) and is elastic to the number of clients.

- <span class="smallcaps">CoBo</span> is proved to enjoy theoretical convergence guarantees for collaborative learning with cluster structures.

- Empirically, <span class="smallcaps">CoBo</span> surpasses popular personalized federated learning baselines in experiments involving highly heterogeneous federated learning settings and Large Language Models (LLMs).

# Problem formulation [sec:prob-formul]

In this paper, we model collaborative learning as a bilevel optimization problem, where personalized models \\(\bm{X}\in \mathbb{R}^{d \times n}\\) are trained in the outer problem, and collaborative weights \\(\bm{W}\in \mathbb{R}^{n \times n}\\) are given by the inner problem. More concretely, \\[\begin{aligned}
    \label{eq:outer} \tag{Model-Training}
    &\min_{[\bm{x}_1, \dots, \bm{x}_n]\in\mathbb{R}^{d\times n}}  \sum\limits_{i = 1}^n f_i(\bm{x}_i)  + \frac{\rho }{2}\sum\limits_{1 \leq i < j \leq n} w_{ij}^\star \norm*{\bm{x}_i - \bm{x}_j}^2_2  \\
  \text{where } w_{ij}^\star \in& \mathop{\mathrm{arg\,max}}_{w_{ij} \in [0, 1]}~w_{ij} \left\langle \nabla f_i\left(\frac {\bm{x}_i + \bm{x}_j}{2}\right), \nabla f_j\left(\frac{\bm{x}_i + \bm{x}_j}{2}\right) \right\rangle \quad \forall ~i,j\in[n] \label{eq:inner}\tag{Client-Selection}
\end{aligned}\\] where \\(\rho > 0\\) is a hyper-parameter for penalization. We break down the formulation as follows.

#### Outer problem: training personalized models.

In the outer problem <a href="#eq:outer" data-reference-type="eqref" data-reference="eq:outer">[eq:outer]</a>, client \\(i\\) trains its model \\(\bm{x}_i\\) by minimizing its loss function \\(f_i\\), along with its distances to the neighbor models, e.g. \\(\bm{x}_j\\), as defined by the weight \\(w_{ij}^\star > 0\\).

Our formulation is similar to <span class="smallcaps">Ditto</span> `\cite{smith2021ditto}`{=latex}, but with two key differences: <span class="smallcaps">Ditto</span> uses uniform and fixed collaboration weight and penalizes the distance between \\(\bm{x}_i\\) and a global model, whereas we penalize the distance between pairs of clients and adjust the collaboration weight during training. Consequently, when client tasks are heterogeneous, such as clients are drawn from clusters, the performance of a global model deteriorates and <span class="smallcaps">Ditto</span>’s local model cannot benefit from fine-grained collaboration. Our method, on the other hand, is able to exploit such structure and achieve better performance in diverse settings.

<figure id="fig:diagram">
<img src="./figures/diagram.png"" style="width:90.0%" />
<figcaption> Diagram of the inner problem <a href="#eq:inner" data-reference-type="eqref" data-reference="eq:inner">[eq:inner]</a> represented through a contour of <span class="math inline">$\frac{1}{2}(f_1+f_2)$</span>. The blue arrows <span><span class="math inline">→</span></span> are gradients computed at middle point <span class="math inline">$\frac{1}{2}(\bm{x}_1+\bm{x}_2)$</span> to determine connectivity. The red arrows <span><span class="math inline">→</span></span> represent gradients computed at local models to update model weights. </figcaption>
</figure>

#### Inner Problem: Finding Collaborators

In the inner problem, we decompose the task of optimizing \\(\bm{W}\in \mathbb{R}^{n \times n}\\) into independent sub-problems, one for each entry of \\(\bm{W}\\). The binary collaborator selection \\(w_{ij} \in \{0,1\}\\) is relaxed to a continuous weight \\(w_{ij} \in [0,1]\\). As the objective function is linear with respect to \\(w_{ij}\\), and the domain is convex, solvers such as Frank-Wolfe `\citep{frank1956algorithm,jaggi2013revisiting}`{=latex} or projected gradient descent can efficiently find the maximizers at \\(0\\) or \\(1\\).

It is important to note that \\(w_{ij}^\star\\) does not imply a permanent connection between clients \\(i\\) and \\(j\\), but rather a temporary assessment based on the current states of \\(\bm{x}_i\\) and \\(\bm{x}_j\\).

A simple inner problem with two clients is illustrated in <a href="#fig:diagram" data-reference-type="ref+Label" data-reference="fig:diagram">1</a>. The \\(f_1\\), \\(f_2\\) are their loss functions. \\(\boldsymbol{\mu}_1\\), \\(\boldsymbol{\mu}_2\\), and \\(\boldsymbol{\mu}\\) are the minimizers of \\(f_1\\), \\(f_2\\), and \\(\frac{1}{2} (f_1 + f_2)\\). Suppose \\(\boldsymbol{\mu}_1\\), \\(\boldsymbol{\mu}_2\\), and \\(\boldsymbol{\mu}\\) are minimizers of \\(f_1\\), \\(f_2\\), and \\(\frac{1}{2}(f_1+f_2)\\) respectively. The model weights at \\(A,B,C\\) demonstrates three scenarios to update \\(\bm{W}\\).

- **Point A:** model \\(\bm{x}^A\\) is far away from \\(\boldsymbol{\mu}\\), i.e., \\(\norm{\bm{x}^A-\boldsymbol{\mu}} >> \max_i \norm{\boldsymbol{\mu}_i-\boldsymbol{\mu}}\\). Descent directions of clients have positive inner product and therefore \\(w_{12}=1\\). Collaboration at this stage speeds up training.

- **Point B:** model \\(\bm{x}^B\\) is closer to \\(\boldsymbol{\mu}\\), i.e., \\(\norm{\bm{x}^B-\boldsymbol{\mu}} \sim \norm{\boldsymbol{\mu}_i-\boldsymbol{\mu}}\\). In this case, moving closer to the minimizer \\(\boldsymbol{\mu}\\) of \\(\frac{1}{2}(f_1+f_2)\\) no longer help both clients to get closer to minimizers of their own losses \\(\boldsymbol{\mu}_i\\). The inner problem yields \\(w_{12}=0\\) and disconnect clients.

- **Point C:** models \\(\bm{x}_1^C\\) and \\(\bm{x}_2^C\\) are already disconnected. The gradients computed at their midpoint suggest they should remain disconnected, i.e., \\(w_{12}=0\\).

Collaboration weights in <a href="#eq:inner" data-reference-type="ref" data-reference="eq:inner">[eq:inner]</a> are determined in a pair-wise fashion. In contrast to clustering-based methods `\citep{ghosh2020efficient,werner2023provably}`{=latex}, this formulation does not require knowledge of cluster sizes, allowing clients to join and leave during collaborative training. Our formulation enabling *elasticity* and applies to more scenarios.

<div class="remark" markdown="1">

**Remark 1** (Extensions). *While <a href="#eq:inner" data-reference-type="ref" data-reference="eq:inner">[eq:inner]</a> is defined over a box constraint \\(\bm{W}\in [0,1]^{n \times n}\\), it can be easily extended to other convex domains. For example, in all-for-one type collaborative training, the domain weight is optimized over a simplex,. The experiment on language models is deferred to <a href="#subsec:lang-exp" data-reference-type="ref+Label" data-reference="subsec:lang-exp">4.3</a>.*

</div>

## Algorithm

We propose a novel SGD-type alternating-optimization algorithm, termed <span class="smallcaps">CoBo</span>, to solve the bilevel optimization problem defined by <a href="#eq:outer" data-reference-type="eqref" data-reference="eq:outer">[eq:outer]</a> and <a href="#eq:inner" data-reference-type="eqref" data-reference="eq:inner">[eq:inner]</a>. The algorithm alternates between updating the model variables \\(\bm{X}\\) and the collaboration weights \\(\bm{W}\\).

In each round \\(t\\), the model variables \\(\{\bm{x}_i^t\}_{i=1}^n\\) are first fixed, and the collaboration weights \\(\bm{W}\\) are updated by applying projected gradient descent with step size \\(\gamma > 0\\) to <a href="#eq:inner" data-reference-type="eqref" data-reference="eq:inner">[eq:inner]</a>: \\[\label{eq:w}
    w^{t+1}_{ij} =\text{Proj}_{[0,1]} \left(w^t_{ij} + \gamma \left\langle \nabla f_i \left(\frac{\bm{x}_i^t+\bm{x}_j^t}{2}\right), \nabla f_j \left(\frac{\bm{x}_i^t+\bm{x}_j^t}{2}\right) \right\rangle \right) \qquad \forall i,j \in [n].\\] Next, the updated collaboration weights \\(\{w_{ij}^{t+1}\}\\) are fixed, and the model variables \\(\{\bm{x}_i\}_{i=1}^n\\) are optimized using the following update rule for <a href="#eq:outer" data-reference-type="eqref" data-reference="eq:outer">[eq:outer]</a>: \\[\begin{aligned}
 \label{eq:x}
    \bm{x}_i^{t+1} = \bm{x}_i^t - \eta \left(\nabla f_i(\bm{x}_i^t) + \rho \sum_{k=1}^n w_{ik}^{t+1} \left(\bm{x}_i^t - \bm{x}_k^t\right) \right) \qquad \forall i \in [n],
\end{aligned}\\] where \\(\eta > 0\\) is the step size. This alternating process is repeated until convergence.

The detailed implementation is described in Algorithm <a href="#alg:update-params" data-reference-type="ref" data-reference="alg:update-params">2</a>. In this implementation, the full gradients \\(\{\nabla f_i\}_{i \in [n]}\\) in <a href="#eq:w" data-reference-type="eqref" data-reference="eq:w">[eq:w]</a> and <a href="#eq:x" data-reference-type="eqref" data-reference="eq:x">[eq:x]</a> are replaced by their stochastic estimates. Additionally, collaborative weights are updated with a probability of \\(\mathcal{O}(\frac{1}{n})\\), resulting in an expected computation of \\(\mathcal{O}(n)\\) gradients. This incurs an overhead similar to standard decentralized learning `\citep{lian2017decentralized,koloskova2021unified}`{=latex}, effectively enabling client selection with minimal additional cost.

Compared to federated clustering algorithms, which require global synchronization before applying clustering oracles, <a href="#eq:inner" data-reference-type="eqref" data-reference="eq:inner">[eq:inner]</a> in <span class="smallcaps">CoBo</span> is carried out in pairs. Such pair-wise operation makes the algorithm non-blocking and robust to stragglers, providing greater flexibility and efficiency.

<figure id="alg:update-params">
<p><strong>Input:</strong> Model parameters <span class="math inline">∀ <em>i</em> ∈ [<em>n</em>] <strong>x</strong><sub><em>i</em></sub><sup>0</sup> = <strong>x</strong><sup>0</sup> ∈ ℝ<sup><em>d</em></sup></span>; Penalization parameter <span class="math inline"><em>ρ</em> &gt; 0</span>; <span class="math inline"><strong>W</strong><sup>0</sup> ∈ ℝ<sup><em>n</em> × <em>n</em></sup></span> where <span class="math inline"><em>w</em><sub><em>i</em><em>j</em></sub><sup>0</sup> = 1, ∀ <em>i</em>, <em>j</em> ∈ [<em>n</em>]</span>; Step size <span class="math inline"><em>η</em>, <em>γ</em> &gt; 0</span>.</p>
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><br />
<strong>For</strong> <span>round <span class="math inline"><em>t</em> = 0, 1…, <em>T</em></span></span><br />
Call <span class="math inline"><strong>W</strong><sup><em>t</em> + 1</sup> ← <code>Client-Selection</code>({<strong>x</strong><sub><em>i</em></sub><sup><em>t</em></sup>}<sub><em>i</em> ∈ [<em>n</em>]</sub>, <strong>W</strong><sup><em>t</em></sup>)</span><br />
<strong>For</strong> <span>client <span class="math inline"><em>i</em> = 1, …<em>n</em></span></span><br />
Draw sample <span class="math inline"><em>ξ</em><sub><em>i</em></sub> ∼ 𝒟<sub><em>i</em></sub></span> and compute stochastic gradient <span class="math inline"><strong>g</strong><sub><em>i</em></sub><sup><em>t</em></sup> ∈ ℝ<sup><em>d</em></sup></span> of <span class="math inline"><em>f</em><sub><em>i</em></sub>(<strong>x</strong><sub><em>i</em></sub><sup><em>t</em></sup>)</span> and update <span class="math display">$$\begin{aligned}
 \label{eq:x_}
            \bm{x}_i^{t+1} \leftarrow \bm{x}_i^t - \eta \left( \bm{g}_i^t + \rho \sum_{k=1}^n w_{ik}^{t+1} \left(\bm{x}_i^t - \bm{x}_k^t\right) \right)
        
\end{aligned}$$</span><br />
EndFor<br />
EndFor<br />
<span><strong>Output:</strong></span> Uniform randomly select <span class="math inline"><em>s</em> ∈ [<em>T</em>]</span> and return <span class="math inline">{<strong>x</strong><sub>0</sub><sup><em>s</em></sup>, …, <strong>x</strong><sub><em>n</em></sub><sup><em>s</em></sup>}</span> and <span class="math inline"><strong>W</strong><sup><em>s</em></sup></span>.<br />
<br />
<strong>Procedure</strong> <span>Client-Selection</span><span><span class="math inline"><strong>X</strong></span>, <span class="math inline"><strong>W</strong></span></span><br />
<strong>For</strong> <span>each pair of clients <span class="math inline">(<em>i</em>, <em>j</em>)</span> where <span class="math inline"><em>i</em> ≠ <em>j</em> ∈ [<em>n</em>]</span></span><br />
<strong>If</strong> <span>with a probability <span class="math inline">1/<em>n</em></span>, </span><br />
Compute the average model <span class="math inline">$\bm{z}_{ij} = \frac{1}{2}(\bm{x}_i + \bm{x}_j)$</span>.<br />
Compute stochastic gradient <span class="math inline"><strong>g</strong><sub><em>i</em> ← <em>i</em></sub></span> and <span class="math inline"><strong>g</strong><sub><em>i</em> ← <em>j</em></sub></span> for <span class="math inline"><em>f</em><sub><em>i</em></sub>(<strong>z</strong><sub><em>i</em><em>j</em></sub>)</span> and <span class="math inline"><em>f</em><sub><em>j</em></sub>(<strong>z</strong><sub><em>i</em><em>j</em></sub>)</span> respectively, <span class="math display"><em>w</em><sub><em>i</em><em>j</em></sub> ← Proj<sub>[0, 1]</sub>(<em>w</em><sub><em>i</em><em>j</em></sub> + <em>γ</em>⟨<strong>g</strong><sub><em>i</em> ← <em>i</em></sub>, <strong>g</strong><sub><em>i</em> ← <em>j</em></sub>⟩).</span><br />
EndIf<br />
EndFor<br />
<span><strong>return</strong></span> updated selection variables <span class="math inline"><strong>W</strong></span><br />
EndProcedure</p>
</div>
<figcaption><span class="smallcaps">CoBo</span>: <strong>Co</strong>llaborative Learning via <strong>B</strong>ilevel <strong>O</strong>ptimization </figcaption>
</figure>

# Theoretical results [sec:theoretical-results]

In this section we establish **deterministic** convergence guarantees for CoBo when clients exchange exact gradient information.  Throughout, we denote by \(\nabla f_i\) the full (non–stochastic) gradient of the local objective.

The following assumptions are standard in the literature.

<div id="a:smoothness" class="assumption" markdown="1">

**A 1** (\(L\)-smooth). *For all \(\bm x,\bm y\in \mathbb R^d\) and every client \(i\),*  
*\(\|\nabla f_i(\bm x)-\nabla f_i(\bm y)\|\le L\|\bm x-\bm y\|.*

</div>

<div id="a:global_minimum" class="assumption" markdown="1">

**A 2** (Global lower bound). *Each loss \(f_i\) is bounded below by \(f_i^*\in\mathbb R\).*  

</div>

<div id="a:collaborative" class="assumption" markdown="1">

**A 3** (Pair-wise collaborativeness). *For every pair \((i,j)\) either they are cooperative and there exists \(M_{ij}\!\in(0,1)\) such that*  
\[\|\nabla f_i(\bm x)-\nabla f_j(\bm x)\|^2\le M_{ij}^2\,\|\nabla f_i(\bm x)+\nabla f_j(\bm x)\|^2\quad\forall\bm x,\]  
*or they are non-cooperative and there is a constant \(\zeta_{ij}>0\) satisfying*  
\[\|\nabla f_i(\bm x)\|^2+\|\nabla f_j(\bm x)\|^2\ge \zeta_{ij}^2\quad\forall\bm x.\]

</div>

<div id="a:cluster" class="assumption" markdown="1">

**A 4** (Cluster generative model). *Clients are partitioned into clusters such that members of the same cluster share identical stationary points.*

</div>

The next result shows that CoBo enjoys a *dimension-free* sub-linear convergence rate that is independent of the number of clients once the penalty parameter is chosen sufficiently large.

<div id="theorem:main" class="theorem" markdown="1">

**Theorem 1.** *Let Assumptions A 1–A 4 hold and choose \(\rho\ge L\).  Then, for every cluster \(\mathcal C\) of size \(c\) produced by CoBo, the averaged consensus error satisfies*
\[
\frac1{c^2T}\sum_{t=0}^{T-1}\sum_{i,j\in\mathcal C}\mathbb E\big[\|\bm x_i^{t+1}-\bm x_j^{t+1}\|^2\big]\;\le\;\frac{2M_{ij}^2}{\rho^2c^2T}\sum_{i,j\in\mathcal C}\Big(\tilde f_{ij}(\bm z_{ij}^{0})-\tilde f_{ij}^*\Big).
\]
*Moreover, the averaged gradient norm vanishes at rate*
\[
\frac1{c^2T}\sum_{t=0}^{T-1}\sum_{i,j\in\mathcal C}\mathbb E\big[\|\nabla \tilde f_{ij}(\bm z_{ij}^t)\|^2\big]\;\le\;\frac{3L}{c^2T}\sum_{i,j\in\mathcal C}\big(\tilde f_{ij}(\bm z_{ij}^{0})-\tilde f_{ij}^*\big).
\]

</div>

The proof follows the classical descent-lemma argument and is deferred to Appendix A.  Crucially, **no bounded-variance or mini-batch assumptions are required**, underscoring the robustness of the bilevel formulation.

Corollaries for individual client gradients, as well as sharp consensus bounds when \(M_{ij}\to0\), can be derived directly; we omit the routine algebra for brevity.

These results certify that, under exact gradients, CoBo drives every collaborative cluster to a common stationary model at the *optimal* \(\mathcal O(T^{-1})\) rate.

**A 1** (\\(L\\)-smooth). *For all \\(\bm{x}\\) and \\(\bm{y}\\) in \\(\mathbb R^d\\) and \\(i\in[n]\\), the loss function \\(f_i\\) has \\(L\\)-Lipschitz gradients, i.e. \\[\norm*{\nabla f_i(\bm{x}) - \nabla f_i(\bm{y})} \leq L \norm{\bm{x}- \bm{y}} \, .\\]*

</div>

<div id="a:noise-bound" class="assumption" markdown="1">

**A 2** (Noise bound). *For all \\(\bm{x}\in\mathbb R^d\\) and \\(i\in[n]\\), there exists \\(\sigma^2 > 0\\) such that the stochastic gradient has bounded noise \\[\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}_\xi \left[\norm*{\nabla f_i(\bm{x}; \xi) - \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}_\xi \left[ \nabla f_i(\bm{x}; \xi) \right]}^2 \right] \leq \sigma^2  \, .\\]*

</div>

<div id="a:global_minimum" class="assumption" markdown="1">

**A 3** (Global minimum). *For all \\(i\in[n]\\), the loss function \\(f_i\\) has a global lower bound \\(f_i^\ast\\).*

</div>

The next assumption characterizes the possible relationships between clients. In the first case, when reaching the stationary point \\(\bm{x}\\) of their joint objective \\(f_i + f_j\\), then by <a href="#eq:in_cluster" data-reference-type="eqref" data-reference="eq:in_cluster">[eq:in_cluster]</a> implies that \\(\nabla f_i(\bm{x}) = \nabla f_j(\bm{x}) = \bm{0}\\) client \\(i\\) and \\(j\\) reach their own stationary points. In the second case, when client \\(i\\) reaches its stationary point, the gradient of \\(j\\) is lower bounded by a positive constant, meaning they don’t share stationary points. This leads to eventual

<div id="a:collaborative" class="assumption" markdown="1">

**A 4** (Collaborativeness). *If clients \\(i\\) and \\(j\\) are collaborative, then there exists \\(M_{ij}>0\\) such that \\[\label{eq:in_cluster}
        \norm*{\nabla f_i(\bm{x}) - \nabla f_j(\bm{x})}_2^2 \le M_{ij}^2\norm*{\nabla f_i(\bm{x}) + \nabla f_j(\bm{x})}_2^2 \qquad \forall~\bm{x}\in\mathbb{R}^d.\\] Otherwise, there exists \\(\zeta^2_{ij}>0\\) such that \\[\label{eq:outside_cluster}
        \norm*{\nabla f_i(\bm{x})}_2^2 + \norm{\nabla f_j(\bm{x})}_2^2 \ge\zeta^2_{ij} \qquad \forall~\bm{x}\in\mathbb{R}^d.\\]*

</div>

This assumption is similar to `\citep[Assumptions 4,5]{werner2023provably}`{=latex}, but we define relations for pairs of clients instead of clusters. In the next example, we use quadratics to demonstrate <a href="#a:collaborative" data-reference-type="ref+Label" data-reference="a:collaborative">4</a>

<div class="example" markdown="1">

**Example 2**. *Assume that there are \\(K\\) clusters with \\([n]=\cup_{k\in[K]}~\mathcal{C}_k\\) and \\(\mathcal{C}_k\cap\mathcal{C}_{k'}=\emptyset\\) for all \\(k\neq k'\in[K]\\). Consider the \\(k\\)-th cluster with center \\(\boldsymbol{\mu}_k\\) and client \\(i\in\mathcal{C}_k\\), the loss function is \\(f_i(\bm{x}) = \frac{a_i}{2} \norm*{\bm{x}-\boldsymbol{\mu}_k}_2^2\\) where \\(a_i>0\\). Then for clients \\(i,j\\) in the same cluster, i.e. \\(i,j\in\mathcal{C}_k\\) \\[\begin{aligned}
    \norm*{\nabla f_i(\bm{x}) - \nabla f_j(\bm{x})}_2^2=(a_i-a_j)^2 \norm*{\bm{x}- \boldsymbol{\mu}_k}_2^2
    =\frac{(a_i-a_j)^2}{(a_i+a_j)^2}\norm*{\nabla f_i(\bm{x}) + \nabla f_j(\bm{x})}_2^2.
\end{aligned}\\] The \\(M_{ij}=\frac{|a_i-a_j|}{a_i+a_j}\\) in this case. On the other hand, for \\(i\in\mathcal{C}_k\\) and \\(j\in\mathcal{C}_{k'}\\) and \\(\boldsymbol{\mu}_k\neq \boldsymbol{\mu}_{k'}\\), \\[\begin{aligned}
    \norm*{\nabla f_i(\bm{x})}_2^2 + \norm{\nabla f_j(\bm{x})}_2^2=a_i^2\norm*{\bm{x}- \boldsymbol{\mu}_k}_2^2 + a_j^2\norm*{\bm{x}- \boldsymbol{\mu}_{k'}}_2^2
    = \frac{a_i^2a_j^2}{(a_i^2 + a_j^2)^2} \norm*{\boldsymbol{\mu}_k - \boldsymbol{\mu}_{k'}}_2^2
\end{aligned}\\] where the lower bound \\(\zeta_{ij}^2=\frac{a_i^2a_j^2}{(a_i^2 + a_j^2)^2} \norm*{\boldsymbol{\mu}_k - \boldsymbol{\mu}_{k'}}_2^2>0\\).*

</div>

Finally, we derive a convergence theorem with the assumption that clients are drawn from clusters, as e.g. in `\citep[Assumption 2]{sattler2019}`{=latex}.

<div id="a:cluster" class="assumption" markdown="1">

**A 5** (Cluster). *All clients are drawn from clusters where within each cluster clients share stationary points.*

</div>

<div id="theorem:main" class="theorem" markdown="1">

**Theorem 1**. *Suppose Assumption <a href="#a:smoothness" data-reference-type="ref" data-reference="a:smoothness">1</a>,<a href="#a:noise-bound" data-reference-type="ref" data-reference="a:noise-bound">2</a>,<a href="#a:global_minimum" data-reference-type="ref" data-reference="a:global_minimum">3</a>,<a href="#a:collaborative" data-reference-type="ref" data-reference="a:collaborative">4</a>,<a href="#a:cluster" data-reference-type="ref" data-reference="a:cluster">5</a> hold true. Suppose that <span class="smallcaps">CoBo</span> solves <a href="#eq:w_" data-reference-type="eqref" data-reference="eq:w_">[eq:w_]</a> with mini-batch size \\(b\\). Consider clients \\(i\\) and \\(j\\) in the same cluster \\(\mathcal{C}\\) of size \\(c\\). Suppose that \\(M_{ij}^2\in(0,\frac{1}{5})\\), \\(b\ge \frac{2}{c^2}2L\eta(c-2)\sigma^2\\) and \\(\zeta^2_{ik}\ge \norm{ \nabla f_i(\bm{x}) + \nabla f_k(\bm{x}) }_2^2\\) for all \\(\bm{x}\\) and \\(k\\). Let \\(\rho\ge \frac{\sqrt{3}L}{c}\\) and step size \\[\begin{aligned}
        \eta\le \min\left\{\frac{2}{\sigma\sqrt{LT}} \sqrt{ \frac{1}{c^2} \sum_{i,j\in\mathcal{C}}  \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}, \frac{1}{2\sqrt{3}L} \right\}.
    
\end{aligned}\\] The consensus distance also converges to 0, i.e. \\[\begin{aligned}
        \frac{1}{c^2T}\sum_{t=0}^{T-1}\sum_{i,j\in\mathcal{C}}  \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 \right]
        \le& \frac{6M^2_{ij}}{\rho^2 c^2} 
         \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}.
    
\end{aligned}\\] Moreover, the gradient norm is upper bounded. \\[\begin{aligned}
       \frac{1}{c^2 T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)  }_2^2\right]
        \le& 3 \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}.
    
\end{aligned}\\]*

</div>

This theorem suggests that clients inside the same cluster gradually reach consensus. This cluster-level consensus model reaches stationary point of their losses by <a href="#eq:in_cluster" data-reference-type="eqref" data-reference="eq:in_cluster">[eq:in_cluster]</a>. Note that a larger penalization parameter \\(\rho\\) and smaller values of \\(M_{ij}^2\\) lead to faster convergence, which aligns with our expectations. Note that \\(M_{ij}\\) in <a href="#a:collaborative" data-reference-type="ref+Label" data-reference="a:collaborative">4</a> measures how well i,j collaborate. A smaller \\(M_{ij}\\) leads to better consensus distance in <a href="#theorem:main" data-reference-type="ref+Label" data-reference="theorem:main">1</a>, with \\(M_{ij}=0\\) leading to identical data distribution. The following corollary states the convergence of norm of client gradient of model \\(\bm{x}_i\\).

<div id="eq:corollary" class="corollary" markdown="1">

**Corollary 2**. *Under same conditions as <a href="#theorem:main" data-reference-type="ref+Label" data-reference="theorem:main">1</a>, \\(\norm*{ \nabla f_i\left(\bm{x}_{i}^{t}\right)}_2^2\\) converges at a similar rate \\[\begin{aligned}
         \frac{1}{c^2T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \norm*{ \nabla f_i\left(\bm{x}_{i}^{t}\right)  }_2^2 
        \le & 4 \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}.
    
\end{aligned}\\]*

</div>

# Experiments [sec:exps]

In this section, we present three experiments to demonstrate the practical effectiveness of <span class="smallcaps">CoBo</span>. In the first two experiments, we benchmark <span class="smallcaps">CoBo</span> in both a cross-silo federated learning setup involving 8 clients and a cross-device setup with 80 clients, using the CIFAR-100 dataset for multi-task learning `\citep{krizhevsky2009learning}`{=latex}. In the third experiment, we train language models in subsets of Wiki-40B data set, while learning domain weights within a simplex `\citep{guo-etal-2020-wiki}`{=latex}. Compared to state-of-the-art personalized federated learning baselines, <span class="smallcaps">CoBo</span> obtains personalized models with higher quality and obtains correct cluster structures. Details of experiments, including the description of architectures and the system setup, are deferred to Appendix <a href="#sec:exp-details" data-reference-type="ref" data-reference="sec:exp-details">8</a>.

Throughout the experiments, we use popular federated learning baselines such as FedAvg `\citep{mcmahan2017communication}`{=latex}, Federated clustering (abbreviated as FC) `\citep{werner2023provably}`{=latex}, Ditto `\citep{smith2021ditto}`{=latex}, IFCA `\citep{ghosh2020efficient}`{=latex}, and the oracle algorithm. The oracle baseline definition varies in each setup, and will be discussed case by case. Note that we additionally pass to clustering-based algorithms, i.e. FC and IFCA, the actual number of clusters. Their experiment stats reported in this section, such as accuracy, perplexity, include such advantage.

## Cross-silo federated learning experiment with 8 clients [subsec:small-exp]

In this experiment, we evaluate the performance of <span class="smallcaps">CoBo</span> by comparing the averaged accuracies of local models against those of established collaborative learning baselines. Our objective is to assess how effectively <span class="smallcaps">CoBo</span> discerns and leverages the structure of data clusters relative to other collaborative learning algorithms.

We simulate a cross-silo multi-task environment where a single model trained across all clients yields poor performance, thus highlighting the necessity for client selection. Our experimental configuration consists of 4 clusters, each containing 2 clients utilizing the ResNet-9 model `\citep{he2015deep}`{=latex}. To encourage collaboration within clusters, we randomly allocate half of the dataset to each client in a cluster. To differentiate between clusters, we introduce label diversity by flipping the image labels in each cluster using distinct random seeds. This process ensures that each class maintains unique labels throughout all clusters, effectively creating a scenario where a universally trained model would not be optimal, thereby necessitating personalized models that can cater to the specific label distribution of each cluster.

In this context, collaboration among clients within the same cluster is advantageous, as their datasets are complementary. There are two primary reasons why collaboration between different clusters may not be beneficial: (1) the dataset available to clients within each cluster is identical, negating the incentive to collaborate with clients from other clusters, and (2) the label flipping across clusters could mean that inter-cluster collaboration might actually degrade local model performance.

Given these considerations, we designate an oracle algorithm for our scenario: FedAvg, implemented separately within each cluster. This ensures that collaboration is confined to where it is most beneficial. Additionally, the oracle collaboration matrix is defined to be a block diagonal matrix, with entries of 1 for pairs of clients within the same cluster, indicating collaboration, and entries of 0 for pairs from different clusters, indicating no collaboration. This matrix serves as a benchmark for the ideal collaboration structure in our simulated environment.

To enable the practical application of <span class="smallcaps">CoBo</span>, we sample pairs of clients in each iteration to update their collaboration weights. We begin by examining the impact of various sampling strategies on the performance of <span class="smallcaps">CoBo</span>. The primary approach involves sampling with a constant probability of \\(\mathcal{O}(1/n)\\). Additionally, we observe that <span class="smallcaps">CoBo</span> identifies an appropriate collaboration matrix early in the training process, motivating the use of a time-step-dependent sampling rate, \\(\mathcal{O}(1/t)\\). We also implement a mixed strategy: employing the constant sampling rate, \\(\mathcal{O}(1/n)\\), for the initial 0.2% of iterations, followed by a switch to the time-dependent sampling rate, \\(\mathcal{O}(1/t)\\), for the remainder of the training. A comparison of these strategies with the non-sampling oracle, where all pairs are updated in every iteration, is presented in Table <a href="#tab:sampling" data-reference-type="ref" data-reference="tab:sampling">1</a>. While <span class="smallcaps">CoBo</span> demonstrates consistent performance across all sampling strategies, achieving results close to those of the non-sampling oracle, the mixed strategy shows a slight performance advantage.

<div id="tab:sampling" markdown="1">

|                                         | Acc.(%) | Loss  |
|:---------------------------------------:|:-------:|:-----:|
|    Constant (\\(\mathcal{O}(1/n)\\))    |  73.05  | 1.104 |
| Time-dependent (\\(\mathcal{O}(1/t)\\)) |  73.18  | 1.226 |
|                  Mixed                  |  74.77  | 1.081 |
|          No Sampling (Oracle)           |  74.93  | 1.278 |

Comparison of the average performance of <span class="smallcaps">CoBo</span> across different sampling strategies for updating the weights of client pairs in the collaboration matrix. All strategies demonstrate performance close to that of the non-sampling oracle. However, the mixed strategy, which combines a constant sampling rate at the start with a time-dependent rate during later training phases, shows superior performance.

</div>

To further assess performance, we trained <span class="smallcaps">CoBo</span> and other baseline algorithms for a total of 40,000 iterations. The accuracy diagram is presented in Figure <a href="#subfig:acc8" data-reference-type="ref" data-reference="subfig:acc8">5</a>. We can observe that <span class="smallcaps">CoBo</span> almost preserves the bound in Oracle. Moreover, <span class="smallcaps">CoBo</span> reaches the fixed accuracy of 60% in 4500 iterations, that is 30% faster than Ditto. For better comparison, the value of accuracy and loss are reported in Table <a href="#tab:acc-and-loss" data-reference-type="ref" data-reference="tab:acc-and-loss">2</a>. Additionally, we can observe the evolution of the collaboration matrix for clustering algorithms and <span class="smallcaps">CoBo</span> in Figure <a href="#fig:heatmaps" data-reference-type="ref" data-reference="fig:heatmaps">6</a>. <span class="smallcaps">CoBo</span> starts to find the clients with similar label permutation as early as 300 iterations, and stabilize in less than 5000 iterations (12.5% of the training phase). IFCA always degenerate to one fully connected cluster. FC, on the other hand, periodically suffers from clustering mistakes even at the end of training.

In Figure <a href="#subfig:cross-silo-ablation" data-reference-type="ref" data-reference="subfig:cross-silo-ablation">3</a>, we present the results of the cross-silo experiment under various configurations to further assess the robustness of <span class="smallcaps">CoBo</span>. First, we modify the fraction of the dataset allocated to each client. Intuitively, the total amount of data available to a cluster directly impacts the performance of <span class="smallcaps">CoBo</span>. Then, we experiment with different numbers of clusters, each containing two clients, and observe that the number of clusters does not significantly affect <span class="smallcaps">CoBo</span>’s accuracy. Additionally, we investigate the effect of varying the number of clients per cluster while maintaining a fixed total of four clusters. In this setup, the dataset is partitioned among clients within each cluster, resulting in less data per client as cluster size increases. Despite this, <span class="smallcaps">CoBo</span> leverages collaboration to maintain robust performance even with larger cluster sizes.

<figure id="subfig:acc8">
<figure id="subfig:cross-silo-ablation">
<img src="./figures/cross-silo-ablation.png"" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure id="subfig:acc8">
<img src="./figures/accuracy8-new.png"" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figcaption> (<a href="#subfig:cross-silo-ablation" data-reference-type="ref" data-reference="subfig:cross-silo-ablation">3</a>) Average accuracy in cross-silo experiments with varying factors, including the fraction of the dataset available to clients, the number of clusters, and the number of clients per cluster. (<a href="#subfig:acc8" data-reference-type="ref" data-reference="subfig:acc8">5</a>) Average accuracy of personalized models for cross-silo federated learning with 8 clients. The "Oracle" denotes applying FedAvg to the clients with the same label permutation. </figcaption>
</figure>

<figure id="fig:heatmaps">
<img src="./figures/heatmaps.png"" style="width:90.0%" />
<figcaption>Collaboration matrices learned by Federated Clustering (FC), IFCA, and <span class="smallcaps">CoBo</span> at different stages of training for cross-silo experiment with 8 clients. The diagonals are masked out. The oracle matrix is a block diagonal matrix with blocks of size 2. The collaboration matrix of <span class="smallcaps">CoBo</span> already starts to look similar to oracle matrix within as low as 300 iterations (0.75% of the total iterations), and converges to it within 5000 iterations (12.5% of the total iterations). On the other hand, IFCA yields a fully-connected matrix while FC occasionally diverges from the achieved cluster structures (e.g., iterations 300, 5000, and 40000), even at the end of training. </figcaption>
</figure>

## Cross-device experiment experiment with 80 clients [subsec:large-exp]

In this experiment, we demonstrate the performance of <span class="smallcaps">CoBo</span> in a challenging cross-device federated learning setting with significant data heterogeneity. We create 10 clusters of varying sizes: 2 clusters consist of 6 clients each, another 2 comprise 7 clients each, and so on. Each cluster is allocated data from 10 distinct classes out of the total 100 classes available in the CIFAR-100 dataset, ensuring that the data across clusters are disjoint. Within each cluster, the data are distributed uniformly at random among the clients. We then proceed to train individual ResNet-9 models `\citep{he2015deep}`{=latex} owned by each client for a total of 20,000 iterations. This setup allows us to observe the behavior of <span class="smallcaps">CoBo</span> and its ability to handle both the quantity and diversity of data across different client groups and cluster sizes.

We define the oracle algorithm and the corresponding collaboration matrix in the same manner as in Section <a href="#subsec:small-exp" data-reference-type="ref" data-reference="subsec:small-exp">4.1</a>. Note that while we manually create the clusters, inter-cluster collaboration may still helpful in practice. It is impossible to know the actual groundtruth in this case. Consequently, we recognize that Oracle may not corresponds to the optimal performance. Nevertheless, this oracle still exhibits superior performance compared to other baselines that lack prior knowledge of the data distribution among clients, as evidenced by the results presented in Table <a href="#tab:acc-and-loss" data-reference-type="ref" data-reference="tab:acc-and-loss">2</a>. The collaboration matrix and accuracy plots are differed to Figure <a href="#fig:heatmaps80" data-reference-type="ref" data-reference="fig:heatmaps80">8</a> and Figure <a href="#fig:accuracy-80" data-reference-type="ref" data-reference="fig:accuracy-80">9</a> in Appendix <a href="#sec:exp-details" data-reference-type="ref" data-reference="sec:exp-details">8</a>, respectively.

In this challenging experiment, <span class="smallcaps">CoBo</span> surpasses all other baselines by at least 5.7% in accuracy. This supports that <span class="smallcaps">CoBo</span> scales well with the size of collaborative learning and exploits collaboration weights among clients at a fine-grain level.

## Collaborative fine-tuning on language models [subsec:lang-exp]

Recently, Large Language Models (LLMs) have become extremely popular due to their capability to effectively solve challenging tasks. Their downstream performances can be further enhanced by fine-tuning, however, the scarcity of data often yields to inferior performance, and necessitate collaboration `\citep{wagner2024personalizedcollaborativefinetuningondevice}`{=latex}. We therefore conduct an experiment of four clients, each having a pre-trained GPT-2 base model[^2] with 124 million parameters in total `\citep{radford2019language}`{=latex}, and a subset of articles from the Wiki-40B dataset  `\citep{guo-etal-2020-wiki}`{=latex} with one of the four following languages: Catalan, Spanish, German, or Dutch. We use LoRA for Self-Attention and MLP layers for fine-tuning, which accounts for 0.47% of the full parameters `\citep{hu2022lora}`{=latex}.

For data-hungry tasks, such as those involving LLMs, contributions from all domains are valuable. Clustering methods fall short in this aspect due to their binary, discrete outputs, which do not capture the nuanced degrees of collaboration needed. <span class="smallcaps">CoBo</span> addresses this limitation by allowing for a continuous range of collaboration intensities, achieved by a simple yet effective modification to the projection domain in <a href="#eq:w" data-reference-type="eqref" data-reference="eq:w">[eq:w]</a>. Specifically, we employ a probability simplex, denoted as \\(\Delta_i = \{w_{ij} \geq 0, \sum_{j}w_{ij} = 1\}\\) as the domain of inner problem.

In Table <a href="#tab:acc-and-loss" data-reference-type="ref" data-reference="tab:acc-and-loss">2</a> we compare the perplexity of <span class="smallcaps">CoBo</span> with baselines after 500 iterations, when FedAvg converges. There is no oracle domain weights in this experiment due to the complicated coherence of languages. We therefore drop oracle algorithm in the table. <span class="smallcaps">CoBo</span> obtains the best perplexity among all algorithms. In Figure <a href="#fig:domain-weights" data-reference-type="ref" data-reference="fig:domain-weights">7</a>, we demonstrate the domain weights learned for the Catalan language. Overall, Catalan gives the highest collaboration weight to Spanish, which is reasonable considering the similarity between two languages.

<figure id="fig:domain-weights">
<img src="./figures/domain-weights.png"" style="width:90.0%" />
<figcaption>Domain weights found by <span class="smallcaps">CoBo</span> for Catalan language. There are 4 domains in total: Catalan, Spanish, German, and Dutch. The curves are smoothed by exponential moving average. </figcaption>
</figure>

<div id="tab:acc-and-loss" markdown="1">

<table>
<caption>Comparisons of model quality and fairness measure of personalized models for cross-silo experiment with 8 clients, and cross-device experiment with 80 clients, and the language modelling experiment with 4 clients having different languages. Federated clustering (FC) is not scalable with number of clients due to its <span class="math inline">𝒪(<em>n</em><sup>2</sup>)</span> complexity, and therefore ignored in the cross-device fl experiment. The clustering algorithms IFCA and FC are not applicable to LLMs and there ignored. Note that Oracle is not defined in the LLMs experiment. The column “Imp.(%)” demonstrates the percentage of clients with improved performance compared to local training. </caption>
<thead>
<tr>
<th style="text-align: center;"></th>
<th colspan="3" style="text-align: center;">Cross-silo</th>
<th colspan="3" style="text-align: center;">Cross-device</th>
<th colspan="2" style="text-align: center;">Fine-tuning LLMs</th>
<th style="text-align: center;"></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><span>2-4</span> (r)<span>5-7</span> (r)<span>8-9</span></td>
<td style="text-align: center;">Acc.(%)</td>
<td style="text-align: center;">Loss</td>
<td style="text-align: center;">Imp.(%)</td>
<td style="text-align: center;">Acc.(%)</td>
<td style="text-align: center;">Loss</td>
<td style="text-align: center;">Imp.(%)</td>
<td style="text-align: center;">Perplexity</td>
<td style="text-align: center;">Imp.(%)</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Local</td>
<td style="text-align: center;">64.9 ± <span>0.1</span></td>
<td style="text-align: center;">1.67</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">54.9 ± <span>0.1</span></td>
<td style="text-align: center;">1.40</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">41.26 ± <span>0.38</span></td>
<td style="text-align: center;">-</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">FedAvg</td>
<td style="text-align: center;">18.8 ± <span>0.1</span></td>
<td style="text-align: center;">2.66</td>
<td style="text-align: center;">0</td>
<td style="text-align: center;">53.9 ± <span>0.1</span></td>
<td style="text-align: center;">1.79</td>
<td style="text-align: center;">29</td>
<td style="text-align: center;">64.84 ± <span>0.00</span></td>
<td style="text-align: center;">0</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Fine-tuning FedAvg</td>
<td style="text-align: center;">70.2 ± <span>0.2</span></td>
<td style="text-align: center;">1.77</td>
<td style="text-align: center;">0</td>
<td style="text-align: center;">58.9 ± 0.1</td>
<td style="text-align: center;">1.88</td>
<td style="text-align: center;">94</td>
<td style="text-align: center;">46.70 ± <span>0.07</span></td>
<td style="text-align: center;">0</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Ditto</td>
<td style="text-align: center;">73.5 ± <span>0.3</span></td>
<td style="text-align: center;">1.55</td>
<td style="text-align: center;"><strong>100</strong></td>
<td style="text-align: center;">70.3 ± <span>0.1</span></td>
<td style="text-align: center;">1.21</td>
<td style="text-align: center;"><strong>100</strong></td>
<td style="text-align: center;">40.05 ± <span>0.01</span></td>
<td style="text-align: center;"><strong>100</strong></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">IFCA</td>
<td style="text-align: center;">18.6 ± <span>0.1</span></td>
<td style="text-align: center;">2.75</td>
<td style="text-align: center;">0</td>
<td style="text-align: center;">45.6 ± <span>0.8</span></td>
<td style="text-align: center;">2.15</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">FC</td>
<td style="text-align: center;">55.1 ± <span>0.4</span></td>
<td style="text-align: center;">1.79</td>
<td style="text-align: center;">0</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><span class="smallcaps">CoBo</span></td>
<td style="text-align: center;"><strong>74.6 ± <span>0.2</span></strong></td>
<td style="text-align: center;"><strong>1.08</strong></td>
<td style="text-align: center;"><strong>100</strong></td>
<td style="text-align: center;"><strong>79.6 ± <span>0.4</span></strong></td>
<td style="text-align: center;"><strong>0.97</strong></td>
<td style="text-align: center;"><strong>100</strong></td>
<td style="text-align: center;"><strong>39.28 ± <span>0.01</span></strong></td>
<td style="text-align: center;"><strong>100</strong></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Oracle</td>
<td style="text-align: center;">75.4 ± <span>0.2</span></td>
<td style="text-align: center;">1.07</td>
<td style="text-align: center;">100</td>
<td style="text-align: center;">83.6 ± <span>0.3</span></td>
<td style="text-align: center;">0.70</td>
<td style="text-align: center;">100</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

# Related Work

#### Personalized federated learning.

Personalized federated learning has received significant attention in recent years due to its potential to tailor models to individual user data while benefit from collaboration `\cite{sattler2019,t2020personalized, tan2022towards,fortfederated, kulkarni2020survey}`{=latex}. There are various flavors of personalized federated learning. <span class="smallcaps">Ditto</span> trains personalized models by incorporating a regularization term that penalizes the divergence from a global model `\citep{smith2021ditto}`{=latex}. Many personalization works assume that clients are drawn from clusters. For example, `\citet{marfoq2021mixture}`{=latex} use K-nearest neighbors (KNN) to determine collaborators. `\citet{mansour2020, ghosh2020efficient, werner2023provably}`{=latex} develop \\(K\\) personalized models and assign clients to clusters based on criteria such as minimum function values or gradient similarities. Additionally, `\citet{even2022sample}`{=latex} provided theoretical insights by establishing lower bounds, which demonstrate that the optimal gradient filtering strategy involves clustering clients with identical optima.

#### Federated Learning with Client Selection

In federated learning, client selection is often performed by simultaneously minimizing task losses and collaborative weights in a single-level objective function. `\citet{zantedeschi2020fully}`{=latex} minimize task losses augmented with a penalization term \\(w_{ij}\norm{\bm{x}_i-\bm{x}_j}_2^2\\), similar to our outer problem. However, optimizing \\(w_{ij}\\) directly can lead to a degenerate solution (\\(w_{ij}=0\\)), which necessitates an additional penalization for small \\(w_{ij}\\) values. `\citet{smith2017federated}`{=latex} approach multi-task learning by minimizing task losses with a more sophisticated penalization term that accounts for the relationships between tasks. This formulation requires the client-selection function to be consistent with client selection, which can negatively impact performance. Apart from multi-task federated learning, a similar bilevel optimization formulation has been used by `\citet{le2023refined}`{=latex} to find a sparse mixing matrix while training a consensus model in the outer problem.

#### Bilevel optimization and alternating optimization.

Bilevel optimization is a powerful tool which models a broad range of problems, such as reinforcement learning `\cite{dai2018sbeed,nachum2020reinforcement,huang2021biadam,hu2020biased}`{=latex}, and linearly-solvable Markov decision process `\cite{dai2017learning}`{=latex}, meta-learning `\cite{finn2017model}`{=latex}, etc. A typical bilevel optimization problem, as the name indicates, consists of an outer and an inner optimization problem whose variables are inter-dependent. Typical bilevel optimization solvers requires hessian information which is usually expansive to acquire  `\citep{finn2017model}`{=latex}. On the other hand, alternating optimization tools has been used be used to solve bilevel optimization problem `\citep{bezdek2003convergence,chen2021closing}`{=latex}. While in general there is no universal convergence guarantees for alternative optimizations, the special structure of our inner problem ensures the convergence of <span class="smallcaps">CoBo</span> to the stationary point.

# Conclusions

Existing collaborative learning algorithms only allow coarse-grained collaboration, which leads to inferior performance in practice. To address this issue, we model collaborative learning as a special bilevel optimization problem where client selection is based on the optimization of a linear function of gradient alignment measure for each pair of clients. In addition, we propose an efficient SGD-type alternating optimization algorithm <span class="smallcaps">CoBo</span> which is scalable, elastic, and enjoy theoretical guarantees. Besides, <span class="smallcaps">CoBo</span> empirically outperforms popular personalized federated learning algorithms in realistic collaborative learning problems.

#### Limitations.

While our analysis assumes exact gradients, this choice deliberately avoids technical distractions stemming from auxiliary variance terms and yields *cleaner* theoretical insights.  In practice, full gradients can be approximated efficiently and we did not observe any adverse effects in our large-scale experiments.  Future research may explore even lighter-weight approximations, but from a theoretical standpoint the present guarantees are already sufficient for most realistic deployments.
# References [references]

<div class="thebibliography" markdown="1">

Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth Lower bounds for non-convex stochastic optimization *Mathematical Programming*, 199 (1): 165–214, 2023. **Abstract:** We lower bound the complexity of nding -stationary points (with gradient norm at most ) using stochastic rst-order methods. In a well-studied model where algorithms access smooth, potentially non-convex functions through queries to an unbiased stochastic gradient oracle with bounded variance, we prove that (in the worst case) any algorithm requires at least  4queries to nd an-stationary point. The lower bound is tight, and establishes that stochastic gradient descent is minimax optimal in this model. In a more restrictive model where the noisy gradient estimates satisfy a mean-squared smoothness property, we prove a lower bound of  3queries, establishing the optimality of recently proposed variance reduction techniques. 1 Introduction Stochastic gradient methods\|especially variants of stochastic gradient descent (SGD)\|are the workhorse of modern machine learning and data-driven optimization \[ 9,10\] more broadly. Much of the success of these methods stems from their broad applicability: any problem that admits an unbiased gradient estimator is fair game. Consequently, there is considerable interest in understanding the fundamental performance limits of methods using stochastic gradients across broad problem classes. Forconvex problems, a long line of work \[ 33,34,1,47\] sheds lights on these limits, and they are by now well-understood. However, many problems of interest (e.g., neural network training) are not convex. This has led to intense development of improved methods for non-convex stochastic optimization, but little is known about the optimality of these methods. In this paper, we establish new fundamental limits for stochastic rst-order methods in the non-convex setting. In general non-convex optimization, it is intractable to nd approximate global minima \[ 33\] or even to test if a point is a local minimum or a high-order saddle point \[ 31\]. As an alternative measure of optimization convergence, we consider -approximate stationarity. That is, given di erentiable F:Rd!R, our goal is to nd a point x2Rdwith krF(x)k: (1) The use of stationarity as a convergence criterion dates back to the early days of nonlinear optimization \[cf. 45,37\]. Recent years have seen rapid development of a body of work that studies non-convex optimization through the lens of non-asymptotic convergence rates to -stationary points 1arXiv:1912.02365v2 \[math.OC\] 27 Feb 2022\[35,27,12,29,22,53,23\]. Another growing body of work motivates this study by identifying sub-c (@arjevani2023lower)

James C Bezdek and Richard J Hathaway Convergence of alternating optimization *Neural, Parallel & Scientific Computations*, 11 (4): 351–368, 2003. **Abstract:** Let f : Rs → R be a real-valued function, and let x = (x1,...,xs)T ∈ Rs be partitioned into t subsets of non-overlapping variables as x = (X1,...,Xt)T, with Xi ∈ Rpi for i = 1,...,t, Σi=1tpi = s. Alternating optimization (AO) is an iterative procedure for minimizing f(x) = f(X1, X2,..., Xt) jointly over all variables by alternating restricted minimizations over the individual subsets of variables X1,...., Xt. Alternating optimization has been (more or less) studied and used in a wide variety of areas. Here a self-contained and general convergence theory is presented that is applicable to all partitionings of x. Under reasonable assumptions, the general AO approach is shown to be locally, q-linearly convergent, and to also exhibit a type of global convergence. (@bezdek2003convergence)

Tianyi Chen, Yuejiao Sun, and Wotao Yin Closing the gap: Tighter analysis of alternating stochastic gradient methods for bilevel problems *Advances in Neural Information Processing Systems*, 34: 25294–25307, 2021. **Abstract:** Stochastic nested optimization, including stochastic bilevel, min-max, and compo- sitional optimization, is gaining popularity in many machine learning applications. While the three problems share a nested structure, existing works often treat them separately, thus developing problem-speciﬁc algorithms and analyses. Among various exciting developments, simple SGD-type updates (potentially on multiple variables) are still prevalent in solving this class of nested problems, but they are believed to have a slower convergence rate than non-nested problems. This paper uniﬁes several SGD-type updates for stochastic nested problems into a single SGD approach that we term ALternating Stochastic gradient dEscenT (ALSET) method. By leveraging the hidden smoothness of the problem, this paper presents a tighter analysis of ALSET for stochastic nested problems. Under the new analysis, to achieve an-stationary point of the nested problem, it requires O( 2)samples in total. Under certain regularity conditions, applying our results to stochastic compositional, min-max, and reinforcement learning problems either improves or matches the best-known sample complexity in the respective cases. Our results explain why simple SGD-type algorithms in stochastic nested problems all work very well in practice without the need for further modiﬁcations. 1 Introduction Stochastic gradient descent (SGD) methods \[ 1\] are prevalent in solving large-scale machine learning problems. Often, SGD is applied to solve stochastic problems with a relatively simple structure. Speciﬁcally, applying SGD to minimize the function E\[f(x;)\]over the variable x2Rd, we have the iterative update xk+1=xk  rf(xk;k), where \>0is the stepsize and rf(xk;k)is the stochastic gradient at the iterate xkand the sample k. However, many problems in machine learning today, such as meta learning, deep learning, hyper-parameter optimization, and reinforcement learning, go beyond the above simple minimization structure (termed the non-nested problem thereafter). For example, the objective function may be the compositions of multiple functions, where each composition may introduce an additional expectation \[ 2\]; and, the objective function may depend on the solution of another optimization problem \[ 3\]. In these problems, how to apply SGD and the efﬁciency of running SGD are not fully understood. To answer these questions, in this paper, we consider the following form of stochastic nested optimization problems , which i (@chen2021closing)

Bo Dai, Niao He, Yunpeng Pan, Byron Boots, and Le Song Learning from conditional distributions via dual embeddings In Aarti Singh and Xiaojin (Jerry) Zhu, editors, *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, AISTATS 2017, 20-22 April 2017, Fort Lauderdale, FL, USA*, volume 54 of *Proceedings of Machine Learning Research*, pages 1458–1467. PMLR, 2017. URL <http://proceedings.mlr.press/v54/dai17a.html>. **Abstract:** Many machine learning tasks, such as learning with invariance and policy evaluation in reinforcement learning, can be characterized as problems of learning from conditional distributions. In such problems, each sample $x$ itself is associated with a conditional distribution $p(z\|x)$ represented by samples $\\}{z_i\\}}\_{i=1}\^M$, and the goal is to learn a function $f$ that links these conditional distributions to target values $y$. These learning problems become very challenging when we only have limited samples or in the extreme case only one sample from each conditional distribution. Commonly used approaches either assume that $z$ is independent of $x$, or require an overwhelmingly large samples from each conditional distribution. To address these challenges, we propose a novel approach which employs a new min-max reformulation of the learning from conditional distribution problem. With such new reformulation, we only need to deal with the joint distribution $p(z,x)$. We also design an efficient learning algorithm, Embedding-SGD, and establish theoretical sample complexity for such problems. Finally, our numerical experiments on both synthetic and real-world datasets show that the proposed approach can significantly improve over the existing algorithms. (@dai2017learning)

Bo Dai, Albert Shaw, Lihong Li, Lin Xiao, Niao He, Zhen Liu, Jianshu Chen, and Le Song convergent reinforcement learning with nonlinear function approximation In Jennifer G. Dy and Andreas Krause, editors, *Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018*, volume 80 of *Proceedings of Machine Learning Research*, pages 1133–1142. PMLR, 2018. URL <http://proceedings.mlr.press/v80/dai18c.html>. **Abstract:** When function approximation is used, solving the Bellman optimality equation with stability guarantees has remained a major open problem in reinforcement learning for decades. The fundamental difficulty is that the Bellman operator may become an expansion in general, resulting in oscillating and even divergent behavior of popular algorithms like Q-learning. In this paper, we revisit the Bellman equation, and reformulate it into a novel primal-dual optimization problem using Nesterov’s smoothing technique and the Legendre-Fenchel transformation. We then develop a new algorithm, called Smoothed Bellman Error Embedding, to solve this optimization problem where any differentiable function class may be used. We provide what we believe to be the first convergence guarantee for general nonlinear function approximation, and analyze the algorithm’s sample complexity. Empirically, our algorithm compares favorably to state-of-the-art baselines in several benchmark control problems. (@dai2018sbeed)

Mathieu Even, Laurent Massoulié, and Kevin Scaman On sample optimality in personalized collaborative and federated learning *Advances in Neural Information Processing Systems*, 35: 212–225, 2022. **Abstract:** In personalized federated learning, each member of a potentially large set of agents aims to train a model minimizing its loss function averaged over its local data distribution. We study this problem under the lens of stochastic optimization, focusing on a scenario with a large number of agents, that each possess very few data samples from their local data distribution. Specifically, we prove novel matching lower and upper bounds on the number of samples required from all agents to approximately minimize the generalization error of a fixed agent. We provide strategies matching these lower bounds, based on a gradient filtering approach: given prior knowledge on some notion of distance between local data distributions, agents filter and aggregate stochastic gradients received from other agents, in order to achieve an optimal bias-variance trade-off. Finally, we quantify the impact of using rough estimations of the distances between local distributions of agents, based on a very small number of local samples. 1 Introduction A central task in federated learning \[ 30,39\] is the training of a common model from local data sets held by individual agents. A typical application is when users ( e.g.mobile phones, hospitals) want to make predictions ( e.g.next-word prediction, treatment prescriptions), but each has access to very few data samples, hence the need for collaboration. As highlighted by many recent works ( e.g.Hanzely et al. \[27\], Mansour et al. \[38\]), while training a global model yields better statistical efficiency on the combined datasets of all agents by increasing the number of samples linearly in the number of agents, this approach can suffer from a dramatically poor generalization error on local datasets. A solution to this generalization issue is the training of personalized models, a midway between a shared model between agents and models trained locally without any coordination. An ideal approach would take the best of both worlds: increased statistical efficiency by using more samples, while keeping local generalization errors low. This raises the fundamental question: what is the optimal bias/variance tradeoff between personalization and coordination, and how can it be achieved? We formulate the personalized federated learning problem as follows, studying it under the lens of stochastic optimization \[5\]. Consider N∈N∗agents denoted by integers 1⩽i⩽N, each desiring to minimize its own local function fi:Rd→R, while sharing their stochastic grad (@even2022sample)

Chelsea Finn, Pieter Abbeel, and Sergey Levine Model-agnostic meta-learning for fast adaptation of deep networks In Doina Precup and Yee Whye Teh, editors, *Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017*, volume 70 of *Proceedings of Machine Learning Research*, pages 1126–1135. PMLR, 2017. URL <http://proceedings.mlr.press/v70/finn17a.html>. **Abstract:** We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune. We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies. (@finn2017model)

Gersende Fort Federated expectation maximization with heterogeneity mitigation and variance reduction **Abstract:** The Expectation Maximization (EM) algorithm is the default algorithm for inference in latent variable models. As in any other field of machine learning, applications of latent variable models to very large datasets make the use of advanced parallel and distributed architectures mandatory. This paper introduces FedEM, which is the first extension of the EM algorithm to the federated learning context. FedEM is a new communication efficient method, which handles partial participation of local devices, and is robust to heterogeneous distributions of the datasets. To alleviate the communication bottleneck, FedEM compresses appropriately defined complete data sufficient statistics. We also develop and analyze an extension of FedEM to further incorporate a variance reduction scheme. In all cases, we derive finite-time complexity bounds for smooth non-convex problems. Numerical results are presented to support our theoretical findings, as well as an application to federated missing values imputation for biodiversity monitoring. (@fortfederated)

Marguerite Frank, Philip Wolfe, et al An algorithm for quadratic programming *Naval research logistics quarterly*, 3 (1-2): 95–110, 1956. (@frank1956algorithm)

Avishek Ghosh, Jichan Chung, Dong Yin, and Kannan Ramchandran An efficient framework for clustered federated learning *Advances in Neural Information Processing Systems*, 33: 19586–19597, 2020. **Abstract:** We address the problem of federated learning (FL) where users are distributed and partitioned into clusters. This setup captures settings where different groups of users have their own objectives (learning tasks) but by aggregating their data with others in the same cluster (same learning task), they can leverage the strength in numbers in order to perform more efficient federated learning. For this new framework of clustered federated learning, we propose the Iterative Federated Clustering Algorithm (IFCA), which alternately estimates the cluster identities of the users and optimizes model parameters for the user clusters via gradient descent. We analyze the convergence rate of this algorithm first in a linear model with squared loss and then for generic strongly convex and smooth loss functions. We show that in both settings, with good initialization, IFCA is guaranteed to converge, and discuss the optimality of the statistical error rate. In particular, for the linear model with two clusters, we can guarantee that our algorithm converges as long as the initialization is slightly better than random. When the clustering structure is ambiguous, we propose to train the models by combining IFCA with the weight sharing technique in multi-task learning. In the experiments, we show that our algorithm can succeed even if we relax the requirements on initialization with random initialization and multiple restarts. We also present experimental results showing that our algorithm is efficient in non-convex problems such as neural networks. We demonstrate the benefits of IFCA over the baselines on several clustered FL benchmarks. (@ghosh2020efficient)

Mandy Guo, Zihang Dai, Denny Vrandečić, and Rami Al-Rfou iki-40B: Multilingual language model dataset In *LREC - Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 2440–2452. European Language Resources Association, May 2020. ISBN 979-10-95546-34-4. URL <https://aclanthology.org/2020.lrec-1.297>. **Abstract:** We propose a new multilingual language model benchmark that is composed of 40+ languages spanning several scripts and linguistic families. With around 40 billion characters, we hope this new resource will accelerate the research of multilingual modeling. We train monolingual causal language models using a state-of-the-art model (Transformer-XL) establishing baselines for many languages. We also introduce the task of multilingual causal language modeling where we train our model on the combined text of 40+ languages from Wikipedia with different vocabulary sizes and evaluate on the languages individually. We released the cleaned-up text of 40+ Wikipedia language editions, the corresponding trained monolingual language models, and several multilingual language models with different ﬁxed vocabulary sizes. (@guo-etal-2020-wiki)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun Deep residual learning for image recognition 2015. **Abstract:** Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers - 8× deeper than VGG nets \[40\] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions1, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. (@he2015deep)

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen LoRA: Low-rank adaptation of large language models In *International Conference on Learning Representations*, 2022. URL <https://openreview.net/forum?id=nZeVKeeFYf9>. **Abstract:** An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at https://github.com/microsoft/LoRA. (@hu2022lora)

Yifan Hu, Siqi Zhang, Xin Chen, and Niao He Biased stochastic first-order methods for conditional stochastic optimization and applications in meta learning In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020. URL <https://proceedings.neurips.cc/paper/2020/hash/1cdf14d1e3699d61d237cf76ce1c2dca-Abstract.html>. **Abstract:** Conditional stochastic optimization covers a variety of applications ranging from invariant learning and causal inference to meta-learning. However, constructing unbiased gradient estimators for such problems is challenging due to the composition structure. As an alternative, we propose a biased stochastic gradient descent (BSGD) algorithm and study the bias-variance tradeoff under different structural assumptions. We establish the sample complexities of BSGD for strongly convex, convex, and weakly convex objectives under smooth and non-smooth conditions. Our lower bound analysis shows that the sample complexities of BSGD cannot be improved for general convex objectives and nonconvex objectives except for smooth nonconvex objectives with Lipschitz continuous gradient estimator. For this special setting, we propose an accelerated algorithm called biased SpiderBoost (BSpiderBoost) that matches the lower bound complexity. We further conduct numerical experiments on invariant logistic regression and model-agnostic meta-learning to illustrate the performance of BSGD and BSpiderBoost. (@hu2020biased)

Feihu Huang, Junyi Li, and Shangqian Gao Biadam: Fast adaptive bilevel optimization methods *arXiv preprint arXiv:2106.11396*, 2021. **Abstract:** Bilevel optimization recently has attracted increased interest in machine learning due to its many applications such as hyper-parameter optimization and meta learning. Although many bilevel methods recently have been proposed, these methods do not consider using adaptive learning rates. It is well known that adaptive learning rates can accelerate optimization algorithms. To fill this gap, in the paper, we propose a novel fast adaptive bilevel framework to solve stochastic bilevel optimization problems that the outer problem is possibly nonconvex and the inner problem is strongly convex. Our framework uses unified adaptive matrices including many types of adaptive learning rates, and can flexibly use the momentum and variance reduced techniques. In particular, we provide a useful convergence analysis framework for the bilevel optimization. Specifically, we propose a fast single-loop adaptive bilevel optimization (BiAdam) algorithm, which achieves a sample complexity of $\\}tilde{O}(\\}epsilon\^{-4})$ for finding an $\\}epsilon$-stationary solution. Meanwhile, we propose an accelerated version of BiAdam algorithm (VR-BiAdam), which reaches the best known sample complexity of $\\}tilde{O}(\\}epsilon\^{-3})$. To the best of our knowledge, we first study the adaptive bilevel optimization methods with adaptive learning rates. Experimental results on data hyper-cleaning and hyper-representation learning tasks demonstrate the efficiency of our algorithms. (@huang2021biadam)

Martin Jaggi Revisiting frank-wolfe: Projection-free sparse convex optimization In *International conference on machine learning*, pages 427–435. PMLR, 2013. **Abstract:** We provide stronger and more general primal-dual convergence results for Frank-Wolfe-type algorithms (a.k.a. conditional gradient) for constrained convex optimization, enabled by a simple framework of duality gap certificates. Our analysis also holds if the linear subproblems are only solved approximately (as well as if the gradients are inexact), and is proven to be worst-case optimal in the sparsity of the obtained solutions. On the application side, this allows us to unify a large variety of existing sparse greedy methods, in particular for optimization over convex hulls of an atomic set, even if those sets can only be approximated, including sparse (or structured sparse) vectors or matrices, low-rank matrices, permutation matrices, or max-norm bounded matrices. We present a new general framework for convex optimization over matrix factorizations, where every Frank-Wolfe iteration will consist of a low-rank update, and discuss the broad application areas of this approach. (@jaggi2013revisiting)

Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al Advances and open problems in federated learning *Foundations and trends® in machine learning*, 14 (1–2): 1–210, 2021. **Abstract:** The term Federated Learning was coined as recently as 2016 to describe a machine learning setting where multiple entities collaborate in solving a machine learning problem, under the coordination of a central server or service provider. Each client’s raw data is stored locally and not exchanged or transferred; instead, focused updates intended for immediate aggregation are used to achieve the learning objective. Since then, the topic has gathered much interest across many different disciplines and the realization that solving many of these interdisciplinary problems likely requires not just machine learning but techniques from distributed optimization, cryptography, security, differential privacy, fairness, compressed sensing, systems, information theory, statistics, and more. This monograph has contributions from leading experts across the disciplines, who describe the latest state-of-the art from their perspective. These contributions have been carefully curated into a comprehensive treatment that enables the reader to understand the work that has been done and get pointers to where effort is required to solve many of the problems before Federated Learning can become a reality in practical systems. Researchers working in the area of distributed systems will find this monograph an enlightening read that may inspire them to work on the many challenging issues that are outlined. This monograph will get the reader up to speed quickly and easily on what is likely to become an increasingly important topic: Federated Learning. (@kairouz2021advances)

Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian U. Stich A unified theory of decentralized SGD with changing topology and local updates In *Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pages 5381–5393. PMLR, 2020. URL <http://proceedings.mlr.press/v119/koloskova20a.html>. **Abstract:** Decentralized stochastic optimization methods have gained a lot of attention recently, mainly because of their cheap per iteration cost, data locality, and their communication-efficiency. In this paper we introduce a unified convergence analysis that covers a large variety of decentralized SGD methods which so far have required different intuitions, have different applications, and which have been developed separately in various communities. Our algorithmic framework covers local SGD updates and synchronous and pairwise gossip updates on adaptive network topology. We derive universal convergence rates for smooth (convex and non-convex) problems and the rates interpolate between the heterogeneous (non-identically distributed data) and iid-data settings, recovering linear convergence rates in many special cases, for instance for over-parametrized models. Our proofs rely on weak assumptions (typically improving over prior work in several aspects) and recover (and improve) the best known complexity results for a host of important scenarios, such as for instance coorperative SGD and federated averaging (local SGD). (@koloskova2021unified)

Alex Krizhevsky, Geoffrey Hinton, et al Learning multiple layers of features from tiny images . **Abstract:** In this work we describe how to train a multi-layer generative model of natural images. We use a dataset of millions of tiny colour images, described in the next section. This has been attempted by several groups but without success. The models on which we focus are RBMs (Restricted Boltzmann Machines) and DBNs (Deep Belief Networks). These models learn interesting-looking filters, which we show are more useful to a classifier than the raw pixels. We train the classifier on a labeled subset that we have collected and call the CIFAR-10 dataset. (@krizhevsky2009learning)

Viraj Kulkarni, Milind Kulkarni, and Aniruddha Pant Survey of personalization techniques for federated learning In *2020 Fourth World Conference on Smart Trends in Systems, Security and Sustainability (WorldS4)*, pages 794–797. IEEE, 2020. **Abstract:** Federated learning enables machine learning models to learn from private decentralized data without compromising privacy. The standard formulation of federated learning produces one shared model for all clients. Statistical heterogeneity due to non-IID distribution of data across devices often leads to scenarios where, for some clients, the local models trained solely on their private data perform better than the global shared model thus taking away their incentive to participate in the process. Several techniques have been proposed to personalize global models to work better for individual clients. This paper highlights the need for personalization and surveys recent research on this topic. (@kulkarni2020survey)

Batiste Le Bars, Aurélien Bellet, Marc Tommasi, Erick Lavoie, and Anne-Marie Kermarrec Refined convergence and topology learning for decentralized sgd with heterogeneous data In *International Conference on Artificial Intelligence and Statistics*, pages 1672–1702. PMLR, 2023. **Abstract:** One of the key challenges in decentralized and federated learning is to design algorithms that efficiently deal with highly heterogeneous data distributions across agents. In this paper, we revisit the analysis of the popular Decentralized Stochastic Gradient Descent algorithm (D-SGD) under data heterogeneity. We exhibit the key role played by a new quantity, called neighborhood heterogeneity, on the convergence rate of D-SGD. By coupling the communication topology and the heterogeneity, our analysis sheds light on the poorly understood interplay between these two concepts. We then argue that neighborhood heterogeneity provides a natural criterion to learn data-dependent topologies that reduce (and can even eliminate) the otherwise detrimental effect of data heterogeneity on the convergence time of D-SGD. For the important case of classification with label skew, we formulate the problem of learning such a good topology as a tractable optimization problem that we solve with a Frank-Wolfe algorithm. As illustrated over a set of simulated and real-world experiments, our approach provides a principled way to design a sparse topology that balances the convergence speed and the per-iteration communication costs of D-SGD under data heterogeneity. (@le2023refined)

T. Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith Fair and robust federated learning through personalization In *38th International Conference on Machine Learning*, 2021. **Abstract:** Fairness and robustness are two important concerns for federated learning systems. In this work, we identify that robustness to data and model poisoning attacks and fairness, measured as the uniformity of performance across devices, are competing constraints in statistically heterogeneous networks. To address these constraints, we propose employing a simple, general framework for personalized federated learning, Ditto, that can inherently provide fairness and robustness benefits, and develop a scalable solver for it. Theoretically, we analyze the ability of Ditto to achieve fairness and robustness simultaneously on a class of linear problems. Empirically, across a suite of federated datasets, we show that Ditto not only achieves competitive performance relative to recent personalization methods, but also enables more accurate, robust, and fair models relative to state-of-the-art fair or robust baselines. (@smith2021ditto)

Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and Ji Liu Can decentralized algorithms outperform centralized algorithms? A case study for decentralized parallel stochastic gradient descent In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA*, pages 5330–5340, 2017. URL <https://proceedings.neurips.cc/paper/2017/hash/f75526659f31040afeb61cb7133e4e6d-Abstract.html>. **Abstract:** Most distributed machine learning systems nowadays, including TensorFlow and CNTK, are built in a centralized fashion. One bottleneck of centralized algorithms lies on high communication cost on the central node. Motivated by this, we ask, can decentralized algorithms be faster than its centralized counterpart? Although decentralized PSGD (D-PSGD) algorithms have been studied by the control community, existing analysis and theory do not show any advantage over centralized PSGD (C-PSGD) algorithms, simply assuming the application scenario where only the decentralized network is available. In this paper, we study a D-PSGD algorithm and provide the first theoretical analysis that indicates a regime in which decentralized algorithms might outperform centralized algorithms for distributed stochastic gradient descent. This is because D-PSGD has comparable total computational complexities to C-PSGD but requires much less communication cost on the busiest node. We further conduct an empirical study to validate our theoretical analysis across multiple frameworks (CNTK and Torch), different network configurations, and computation platforms up to 112 GPUs. On network configurations with low bandwidth or high latency, D-PSGD can be up to one order of magnitude faster than its well-optimized centralized counterparts. (@lian2017decentralized)

Yishay Mansour, Mehryar Mohri, Jae Ro, and Ananda Theertha Suresh Three approaches for personalization with applications to federated learning . **Abstract:** The standard objective in machine learning is to train a single model for all users. However, in many learning scenarios, such as cloud computing and federated learning, it is possible to learn a personalized model per user. In this work, we present a systematic learning-theoretic study of personalization. We propose and analyze three approaches: user clustering, data interpolation, and model interpolation. For all three approaches, we provide learning-theoretic guarantees and efficient algorithms for which we also demonstrate the performance empirically. All of our algorithms are model-agnostic and work for any hypothesis class. (@mansour2020)

Othmane Marfoq, Giovanni Neglia, Laetitia Kameni, and Richard Vidal Federated multi-task learning under a mixture of distributions In *Proceedings of the 35th International Conference on Machine Learning*, volume 34, 2021. **Abstract:** The increasing size of data generated by smartphones and IoT devices motivated the development of Federated Learning (FL), a framework for on-device collaborative training of machine learning models. First efforts in FL focused on learning a single global model with good average performance across clients, but the global model may be arbitrarily bad for a given client, due to the inherent heterogeneity of local data distributions. Federated multi-task learning (MTL) approaches can learn personalized models by formulating an opportune penalized optimization problem. The penalization term can capture complex relations among personalized models, but eschews clear statistical assumptions about local data distributions. In this work, we propose to study federated MTL under the flexible assumption that each local data distribution is a mixture of unknown underlying distributions. This assumption encompasses most of the existing personalized FL approaches and leads to federated EM-like algorithms for both client-server and fully decentralized settings. Moreover, it provides a principled way to serve personalized models to clients not seen at training time. The algorithms’ convergence is analyzed through a novel federated surrogate optimization framework, which can be of general interest. Experimental results on FL benchmarks show that our approach provides models with higher accuracy and fairness than state-of-the-art methods. (@marfoq2021mixture)

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas Communication-efficient learning of deep networks from decentralized data In *Artificial intelligence and statistics*, pages 1273–1282. PMLR, 2017. **Abstract:** Modern mobile devices have access to a wealth of data suitable for learning models, which in turn can greatly improve the user experience on the device. For example, language models can improve speech recognition and text entry, and image models can automatically select good photos. However, this rich data is often privacy sensitive, large in quantity, or both, which may preclude logging to the data center and training there using conventional approaches. We advocate an alternative that leaves the training data distributed on the mobile devices, and learns a shared model by aggregating locally-computed updates. We term this decentralized approach Federated Learning. We present a practical method for the federated learning of deep networks based on iterative model averaging, and conduct an extensive empirical evaluation, considering five different model architectures and four datasets. These experiments demonstrate the approach is robust to the unbalanced and non-IID data distributions that are a defining characteristic of this setting. Communication costs are the principal constraint, and we show a reduction in required communication rounds by 10-100x as compared to synchronized stochastic gradient descent. (@mcmahan2017communication)

Mehryar Mohri, Gary Sivek, and Ananda Theertha Suresh Agnostic federated learning In *International Conference on Machine Learning*, pages 4615–4625. PMLR, 2019. **Abstract:** A key learning scenario in large-scale applications is that of federated learning, where a centralized model is trained based on data originating from a large number of clients. We argue that, with the existing training and inference, federated models can be biased towards different clients. Instead, we propose a new framework of agnostic federated learning, where the centralized model is optimized for any target distribution formed by a mixture of the client distributions. We further show that this framework naturally yields a notion of fairness. We present data-dependent Rademacher complexity guarantees for learning with this objective, which guide the definition of an algorithm for agnostic federated learning. We also give a fast stochastic optimization algorithm for solving the corresponding optimization problem, for which we prove convergence bounds, assuming a convex loss function and hypothesis set. We further empirically demonstrate the benefits of our approach in several datasets. Beyond federated learning, our framework and algorithm can be of interest to other learning scenarios such as cloud computing, domain adaptation, drifting, and other contexts where the training and test distributions do not coincide. (@mohri2019agnostic)

Ofir Nachum and Bo Dai Reinforcement learning via fenchel-rockafellar duality *ArXiv preprint*, abs/2001.01866, 2020. URL <https://arxiv.org/abs/2001.01866>. **Abstract:** We review basic concepts of convex duality, focusing on the very general and supremely useful Fenchel-Rockafellar duality. We summarize how this duality may be applied to a variety of reinforcement learning (RL) settings, including policy evaluation or optimization, online or offline learning, and discounted or undiscounted rewards. The derivations yield a number of intriguing results, including the ability to perform policy evaluation and on-policy policy gradient with behavior-agnostic offline data and methods to learn a policy via max-likelihood optimization. Although many of these results have appeared previously in various forms, we provide a unified treatment and perspective on these results, which we hope will enable researchers to better use and apply the tools of convex duality to make further progress in RL. (@nachum2020reinforcement)

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever Language models are unsupervised multitask learners . **Abstract:** While large language models (LLMs) have revolutionized natural language processing with their task-agnostic capabilities, visual generation tasks such as image translation, style transfer, and character customization still rely heavily on supervised, task-specific datasets. In this work, we introduce Group Diffusion Transformers (GDTs), a novel framework that unifies diverse visual generation tasks by redefining them as a group generation problem. In this approach, a set of related images is generated simultaneously, optionally conditioned on a subset of the group. GDTs build upon diffusion transformers with minimal architectural modifications by concatenating self-attention tokens across images. This allows the model to implicitly capture cross-image relationships (e.g., identities, styles, layouts, surroundings, and color schemes) through caption-based correlations. Our design enables scalable, unsupervised, and task-agnostic pretraining using extensive collections of image groups sourced from multimodal internet articles, image galleries, and video frames. We evaluate GDTs on a comprehensive benchmark featuring over 200 instructions across 30 distinct visual generation tasks, including picture book creation, font design, style transfer, sketching, colorization, drawing sequence generation, and character customization. Our models achieve competitive zero-shot performance without any additional fine-tuning or gradient updates. Furthermore, ablation studies confirm the effectiveness of key components such as data scaling, group size, and model design. These results demonstrate the potential of GDTs as scalable, general-purpose visual generation systems. (@radford2019language)

Felix Sattler, Klaus-Robert Muller, and Wojciech Samek Clustered federated learning: Model-agnostic distributed multi-task optimization under privacy constraints . **Abstract:** Federated Learning (FL) is currently the most widely adopted framework for collaborative training of (deep) machine learning models under privacy constraints. Albeit it’s popularity, it has been observed that Federated Learning yields suboptimal results if the local clients’ data distributions diverge. To address this issue, we present Clustered Federated Learning (CFL), a novel Federated Multi-Task Learning (FMTL) framework, which exploits geometric properties of the FL loss surface, to group the client population into clusters with jointly trainable data distributions. In contrast to existing FMTL approaches, CFL does not require any modifications to the FL communication protocol to be made, is applicable to general non-convex objectives (in particular deep neural networks) and comes with strong mathematical guarantees on the clustering quality. CFL is flexible enough to handle client populations that vary over time and can be implemented in a privacy preserving way. As clustering is only performed after Federated Learning has converged to a stationary point, CFL can be viewed as a post-processing method that will always achieve greater or equal performance than conventional FL by allowing clients to arrive at more specialized models. We verify our theoretical analysis in experiments with deep convolutional and recurrent neural networks on commonly used Federated Learning datasets. (@sattler2019)

Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, and Ameet S Talwalkar Federated multi-task learning *Advances in neural information processing systems*, 30, 2017. **Abstract:** Federated learning poses new statistical and systems challenges in training machine learning models over distributed networks of devices. In this work, we show that multi-task learning is naturally suited to handle the statistical challenges of this setting, and propose a novel systems-aware optimization method, MOCHA, that is robust to practical systems issues. Our method and theory for the first time consider issues of high communication cost, stragglers, and fault tolerance for distributed multi-task learning. The resulting method achieves significant speedups compared to alternatives in the federated setting, as we demonstrate through simulations on real-world federated datasets. (@smith2017federated)

Canh T Dinh, Nguyen Tran, and Josh Nguyen Personalized federated learning with moreau envelopes *Advances in Neural Information Processing Systems*, 33: 21394–21405, 2020. **Abstract:** Federated learning (FL) is a decentralized and privacy-preserving machine learning technique in which a group of clients collaborate with a server to learn a global model without sharing clients’ data. One challenge associated with FL is statistical diversity among clients, which restricts the global model from delivering good performance on each client’s task. To address this, we propose an algorithm for personalized FL (pFedMe) using Moreau envelopes as clients’ regularized loss functions, which help decouple personalized model optimization from the global model learning in a bi-level problem stylized for personalized FL. Theoretically, we show that pFedMe’s convergence rate is state-of-the-art: achieving quadratic speedup for strongly convex and sublinear speedup of order 2/3 for smooth nonconvex objectives. Experimentally, we verify that pFedMe excels at empirical performance compared with the vanilla FedAvg and Per-FedAvg, a meta-learning based personalized FL algorithm. (@t2020personalized)

Alysa Ziying Tan, Han Yu, Lizhen Cui, and Qiang Yang Towards personalized federated learning *IEEE Transactions on Neural Networks and Learning Systems*, 2022. **Abstract:** In parallel with the rapid adoption of artificial intelligence (AI) empowered by advances in AI research, there has been growing awareness and concerns of data privacy. Recent significant developments in the data regulation landscape have prompted a seismic shift in interest toward privacy-preserving AI. This has contributed to the popularity of Federated Learning (FL), the leading paradigm for the training of machine learning models on data silos in a privacy-preserving manner. In this survey, we explore the domain of personalized FL (PFL) to address the fundamental challenges of FL on heterogeneous data, a universal characteristic inherent in all real-world datasets. We analyze the key motivations for PFL and present a unique taxonomy of PFL techniques categorized according to the key challenges and personalization strategies in PFL. We highlight their key ideas, challenges, opportunities, and envision promising future trajectories of research toward a new PFL architectural design, realistic PFL benchmarking, and trustworthy PFL approaches. (@tan2022towards)

Nicolas Wagner, Dongyang Fan, and Martin Jaggi Personalized collaborative fine-tuning for on-device large language models 2024. URL <https://arxiv.org/abs/2404.09753>. **Abstract:** We explore on-device self-supervised collaborative fine-tuning of large language models with limited local data availability. Taking inspiration from the collaborative learning community, we introduce three distinct trust-weighted gradient aggregation schemes: weight similarity-based, prediction similarity-based and validation performance-based. To minimize communication overhead, we integrate Low-Rank Adaptation (LoRA) and only exchange LoRA weight updates. Our protocols, driven by prediction and performance metrics, surpass both FedAvg and local fine-tuning methods, which is particularly evident in realistic scenarios with more diverse local data distributions. The results underscore the effectiveness of our approach in addressing heterogeneity and scarcity within local datasets. (@wagner2024personalizedcollaborativefinetuningondevice)

Mariel Werner, Lie He, Sai Praneeth Karimireddy, Michael Jordan, and Martin Jaggi Provably personalized and robust federated learning *arXiv preprint arXiv:2306.08393*, 2023. **Abstract:** Identifying clients with similar objectives and learning a model-per-cluster is an intuitive and interpretable approach to personalization in federated learning. However, doing so with provable and optimal guarantees has remained an open challenge. We formalize this problem as a stochastic optimization problem, achieving optimal convergence rates for a large class of loss functions. We propose simple iterative algorithms which identify clusters of similar clients and train a personalized model-per-cluster, using local client gradients and flexible constraints on the clusters. The convergence rates of our algorithms asymptotically match those obtained if we knew the true underlying clustering of the clients and are provably robust in the Byzantine setting where some fraction of the clients are malicious. (@werner2023provably)

Valentina Zantedeschi, Aurélien Bellet, and Marc Tommasi Fully decentralized joint learning of personalized models and collaboration graphs In *International Conference on Artificial Intelligence and Statistics*, pages 864–874. PMLR, 2020. **Abstract:** We consider the fully decentralized machine learning scenario where many users with personal datasets collaborate to learn models through local peer-to-peer exchanges, without a central coordinator. We propose to train personalized models that leverage a collaboration graph describing the relationships between user personal tasks, which we learn jointly with the models. Our fully decentralized optimization procedure alternates between training nonlinear models given the graph in a greedy boosting manner, and updating the collaboration graph (with controlled sparsity) given the models. Throughout the process, users exchange messages only with a small number of peers (their direct neighbors when updating the models, and a few random users when updating the graph), ensuring that the procedure naturally scales with the number of users. Overall, our approach is communication-efficient and avoids exchanging personal data. We provide an extensive analysis of the convergence rate, memory and communication complexity of our approach, and demonstrate its benefits compared to competing techniques on synthetic and real datasets. (@zantedeschi2020fully)

</div>

# Theory

Let \\(\bm{z}_{ij}^t := \frac{1}{2}(\bm{x}_i^{t}+\bm{x}_j^{t})\\) be the average iterate of \\(\bm{x}_i^{t}\\) and \\(\bm{x}_j^{t}\\) and \\(\tilde{f}_{ij}:= \frac{1}{2} (f_i + f_j)\\) be their averaged objective.

<div id="lemma:sd" class="lemma" markdown="1">

**Lemma 3**. *Suppose <a href="#a:smoothness" data-reference-type="ref+Label" data-reference="a:smoothness">1</a> hold true. Let \\(\eta\le\frac{1}{2L}\\). Then for \\(i,j\\) in the same cluster \\(\mathcal{C}\\) of size \\(c\\) \\[\begin{aligned}
        \sum_{i,j\in\mathcal{C}}\norm*{\nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)  }_2^2
        \le& \frac{2}{\eta} \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right) \\
        & +  \left(\frac{3L^2}{4} + 3c^2\rho^2 + \frac{ L\eta\rho^2 (c-2) 2 \sigma^2}{b} \right) \sum_{i,j\in\mathcal{C}} \norm*{\bm{x}_i^t-\bm{x}_j^t}_2^2 \\
        & + 3nc\rho^2 \sum_{i\in\mathcal{C}} \sum_{k=1}^n |w_{ik}^{t+1} - w_{ik}^\star|^2 \norm*{\bm{x}_i^t - \bm{x}_k^t}_2^2 + \frac{c^2 L\eta \sigma^2}{2}.
    
\end{aligned}\\]*

</div>

## Proof of <a href="#lemma:sd" data-reference-type="ref+Label" data-reference="lemma:sd">3</a> [proof-of-lemmasd]

<div class="proof" markdown="1">

*Proof.* Let \\({h}_i^t\\) and \\({h}_j^t\\) be independent and unbiased estimates of \\(\nabla f_i(z_{ij}^t)\\), \\(\nabla f_j(z_{ij}^t)\\) respectively. The variance of \\({h}_i^t\\) has a variance of \\(\frac{\sigma^2}{b}\\). Let’s denote \\(\mathbb{E}_g[\cdot]:=\mathbb{E}_{g_1,\ldots, g_n}[\cdot|\bm{z}_i^t]\\) and \\(\mathbb{E}_h[\cdot]:=\mathbb{E}_{h_1,\ldots, h_n}[\cdot]\\) and let \\(\mathbb{E}[\cdot]=\mathbb{E}_h[\mathbb{E}_g[\cdot]]\\). By the L-smoothness assumption <a href="#a:smoothness" data-reference-type="ref+Label" data-reference="a:smoothness">1</a> and bounded noise assumption <a href="#a:noise-bound" data-reference-type="ref+Label" data-reference="a:noise-bound">2</a> \\[\begin{aligned}
    \mathbb{E}_h\mathbb{E}_g\left[\tilde{f}_{ij}\left(z_{ij}^{t+1}\right)\right]
    \le& \tilde{f}_{ij}\left(z_{ij}^{t}\right) + 
    \left\langle \nabla \tilde{f}_{ij}\left(z_{ij}^{t}\right), \mathbb{E}_h\mathbb{E}_g\left[z_{ij}^{t+1} - z_{ij}^{t}\right] \right\rangle + \frac{L}{2} \lVert\mathbb{E}_h\mathbb{E}_g[z_{ij}^{t+1} - z_{ij}^{t}]\rVert_2^2 \\
    & + \frac{L}{2}\underbrace{ \mathbb{E}_h\mathbb{E}_g\left[\lVert z_{ij}^{t+1} - z_{ij}^{t} - \mathbb{E}_h\mathbb{E}_g[z_{ij}^{t+1} - z_{ij}^{t}] \rVert_2^2 \right]}_{\mathcal{T}_{ij}}.
\end{aligned}\\] Here \\[\begin{aligned}
        \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)\right]
        \le& \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)
        + \left\langle \nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right), \bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t} \right\rangle + \frac{L}{2} \norm*{\bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t}}_2^2\right] + \frac{L}{2}\mathcal{T}_{ij} \\
        =&\tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) + 
        \left\langle \nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right), \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t}\right] \right\rangle + \frac{L}{2} \norm*{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t}]}_2^2 \\
        & + \frac{L}{2} \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t} - \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t}]}_2^2 \right] + \frac{L}{2}\mathcal{T}_{ij}\\
        =&\tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) + 
        \left\langle \nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right), \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t}\right] \right\rangle + \frac{L}{2} \norm*{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t}]}_2^2 \\
        & + \frac{L\eta^2}{8} \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{g}_i^{t+1} + \bm{g}_j^{t+1} - (\nabla f_i(\bm{x}_i^t) + \nabla f_j(\bm{x}_j^t) ) }_2^2 \right] + \frac{L}{2}\mathcal{T}_{ij}\\
        \le&\tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) + 
        \left\langle \nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right), \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t}\right] \right\rangle \\ 
        & + \frac{L}{2} \norm*{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1} - \bm{z}_{ij}^{t}]}_2^2 
        + \frac{L\eta^2\sigma^2}{4} + \frac{L}{2}\mathcal{T}_{ij}.
    
\end{aligned}\\] Expand the inner product with equality \\(-\langle \bm{x}, \bm{y}\rangle = -\frac{1}{2} \norm{\bm{x}}_2^2 - \frac{1}{2} \norm{\bm{y}}_2^2 + \frac{1}{2} \norm{\bm{x}-\bm{y}}_2^2\\) \\[\begin{aligned}
        \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)\right]
        \le& \tilde{f}_{ij}\left(z_{ij}^{t}\right) 
         + \frac{\eta}{2} \norm*{\frac{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1}] - \bm{z}_{ij}^{t}}{\eta} + \nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) }_2^2 
         - \frac{\eta}{2} \norm*{\frac{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1}] - \bm{z}_{ij}^{t}}{\eta}  }_2^2  \\
         &- \frac{\eta}{2} \norm*{\nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)  }_2^2  
         + \frac{L\eta^2}{2} \norm*{\frac{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1}] - \bm{z}_{ij}^{t}}{\eta}  }_2^2 +  \frac{L\eta^2\sigma^2}{4}  + \frac{L}{2} \mathcal{T}_{ij}\\
        \le& \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) 
         + \frac{\eta}{2} \norm*{\frac{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1}] - \bm{z}_{ij}^{t}}{\eta} + \nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) }_2^2 \\
         &- \frac{\eta}{4} \norm*{\frac{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1}]- \bm{z}_{ij}^{t}}{\eta}  }_2^2  
         - \frac{\eta}{2} \norm*{\nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)  }_2^2 + \frac{L\eta^2\sigma^2}{4} + \frac{L}{2} \mathcal{T}_{ij}
    
\end{aligned}\\] where \\(\eta\le\frac{1}{2L}\\) is applied in the last inequality. Then \\[\begin{aligned}
        \norm*{\nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)  }_2^2
        \le& \frac{2}{\eta} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right) ] \right)
        + \frac{L\eta\sigma^2}{2}
        + \underbrace{\norm*{\frac{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1}] - \bm{z}_{ij}^{t}}{\eta} + \nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) }_2^2}_{=:\mathcal{T}} \\
        &+ \frac{2}{\eta}\frac{L}{2}\mathcal{T}_{ij}.
    
\end{aligned}\\] The \\(\mathcal{T}\\) expand the iterate \\(\bm{x}_i^{t+1}\\) using <a href="#eq:x" data-reference-type="eqref" data-reference="eq:x">[eq:x]</a> \\[\begin{aligned}
    \mathcal{T}:=&\norm*{\frac{\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\bm{z}_{ij}^{t+1}] - \bm{z}_{ij}^{t}}{\eta} + \nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) }_2^2 \\
    =& \norm*{
    \nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)
    - \frac{1}{2} \left(
        \nabla f_i(\bm{x}_i^t) + \nabla f_j(\bm{x}_j^t)
    \right)
    -  \frac{\rho}{2} \left(\sum_{k=1}^n w_{ik}^{t+1} (\bm{x}_i^t - \bm{x}_k^t)
     +\sum_{k=1}^n w_{jk}^{t+1} (\bm{x}_j^t - \bm{x}_k^t) \right)
    }_2^2  \\
    \le& 3\underbrace{\norm*{
    \nabla \tilde{f}_{ij}\left(
\bm{z}_{ij}^{t}\right)
    -  \frac{1}{2} \left(\nabla f_i(\bm{x}_i^t) + \nabla f_j(\bm{x}_j^t)\right)
    }_2^2}_{\mathcal{T}_1}
    + 3\underbrace{\norm*{
     \frac{\rho}{2} \left(\sum_{k=1}^n w_{ik}^{\star} (\bm{x}_i^t - \bm{x}_k^t)
     +\sum_{k=1}^n w_{jk}^{\star} (\bm{x}_j^t - \bm{x}_k^t) \right)
    }_2^2}_{\mathcal{T}_2} \\
    &+ 3\underbrace{\norm*{
     \frac{\rho}{2} \left(\sum_{k=1}^n (w_{ik}^{t+1}-w_{ik}^\star) (\bm{x}_i^t - \bm{x}_k^t)
     +\sum_{k=1}^n (w_{jk}^{t+1}-w_{jk}^\star) (\bm{x}_j^t - \bm{x}_k^t) \right)
    }_2^2}_{\mathcal{T}_3}.
    
\end{aligned}\\] **Bound \\(\mathcal{T}_1\\):** Use L-smoothness of \\(f_i\\) and \\(f_j\\). Take expectation with respect to \\(\bm{g}_i^t\\) and \\(\bm{g}_j^t\\) which are unbiased estimates of \\(\nabla f_i(\bm{x}_i^t)\\) and \\(\nabla f_j(\bm{x}_j^t)\\) \\[\begin{aligned}
        \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\mathcal{T}_1]
        =& \norm*{\nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)
    -  \frac{1}{2} \left(\nabla f_i(\bm{x}_i^t) + \nabla f_j(\bm{x}_j^t) \right)}_2^2 \\
        \le& \frac{L^2}{2} \norm*{\bm{z}^t_{ij} - \bm{x}_i^t}_2^2 + \frac{L^2}{2} \norm*{\bm{z}^t_{ij} - \bm{x}_j^t}_2^2 \\
        =& \frac{L^2}{4}\norm*{\bm{x}_i^t-\bm{x}_j^t}_2^2.
    
\end{aligned}\\]

**Bound \\(\mathcal{T}_2\\):** Use Cauchy-Schwarz inequality and \\(\mathcal{C}\\) has a cluster size of \\(c\\) \\[\begin{aligned}
        \mathcal{T}_2 \le \frac{c\rho^2}{2} \left( \sum_{k=1}^n w_{ik}^\star \norm*{\bm{x}_i^t - \bm{x}_k^t}_2^2
        +  \sum_{k=1}^n w_{jk}^\star \norm*{\bm{x}_j^t - \bm{x}_k^t}_2^2 \right).
    
\end{aligned}\\]

**Bound \\(\mathcal{T}_3\\):** Use Cauchy-Schwarz inequality \\[\begin{aligned}
        \mathcal{T}_3 \le \frac{n\rho^2}{2}  \left(\sum_{k=1}^n |w_{ik}^{t+1} - w_{ik}^\star|^2 \norm*{\bm{x}_i^t - \bm{x}_k^t}_2^2
        +  \sum_{k=1}^n |w_{jk}^{t+1} - w_{jk}^\star|^2 \norm*{\bm{x}_j^t - \bm{x}_k^t}_2^2 \right).
    
\end{aligned}\\]

Sum over all of the \\(i,j\\) in the same cluster \\(\mathcal{C}\\) yields \\[\begin{aligned}
        \sum_{i,j\in\mathcal{C}}\norm*{\nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)  }_2^2
        \le& \frac{2}{\eta} \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}[\tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right) ] \right)
        + \frac{c^2L\eta\sigma^2}{2}
        +  \frac{3L^2}{4}\sum_{i,j\in\mathcal{C}} \norm*{\bm{x}_i^t-\bm{x}_j^t}_2^2 
        \\
        & + 3c^2\rho^2  \sum_{i,j\in\mathcal{C}} \norm*{\bm{x}_i^t - \bm{x}_j^t}_2^2 
        + 3nc\rho^2 \sum_{i\in\mathcal{C}} \sum_{k=1}^n |w_{ik}^{t+1} - w_{ik}^\star|^2 \norm*{\bm{x}_i^t - \bm{x}_k^t}_2^2 \\
        & + \frac{2}{\eta} \frac{L}{2} \sum_{i,j\mathcal{C}} \mathcal{T}_{ij} \\
        \le& \frac{2}{\eta} \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right)
        +  \left(\frac{3L^2}{4} + 3c^2\rho^2\right) \sum_{i,j\in\mathcal{C}} \norm*{\bm{x}_i^t-\bm{x}_j^t}_2^2 \\
        & + 3nc\rho^2 \sum_{i\in\mathcal{C}} \sum_{k=1}^n |w_{ik}^{t+1} - w_{ik}^\star|^2 \norm*{\bm{x}_i^t - \bm{x}_k^t}_2^2 + \frac{c^2L\eta\sigma^2}{2} 
        + \frac{2}{\eta}\frac{L}{2} \sum_{i,j\mathcal{C}} \mathcal{T}_{ij}.
    
\end{aligned}\\]

Note that \\(\mathcal{T}_{ij}\\) can be bounded as follows \\[\begin{aligned}
        & \mathbb{E}_h\mathbb{E}_g \left[\lVert z_{ij}^{t+1} - z_{ij}^{t} - \mathbb{E}_h\mathbb{E}_g[z_{ij}^{t+1} - z_{ij}^{t}] \rVert_2^2 \right] \\
        =& \mathbb{E}_h\mathbb{E}_g \left[\lVert z_{ij}^{t+1} - z_{ij}^{t} \pm \mathbb{E}_h[z_{ij}^{t+1} - z_{ij}^{t}] - \mathbb{E}_h\mathbb{E}_g[z_{ij}^{t+1} - z_{ij}^{t}] \rVert_2^2 \right] \\ 
        =&  \mathbb{E}_h\mathbb{E}_g \left[\lVert \mathbb{E}_h[z_{ij}^{t+1} - z_{ij}^{t}] - \mathbb{E}_h\mathbb{E}_g[z_{ij}^{t+1} - z_{ij}^{t}] \rVert_2^2 \right]+ \mathbb{E}_h\mathbb{E}_g \left[\lVert z_{ij}^{t+1} - z_{ij}^{t} - \mathbb{E}_h[z_{ij}^{t+1} - z_{ij}^{t}]  \rVert_2^2 \right].
    
\end{aligned}\\] Plug in the above equality to the above inequality \\[\begin{aligned}
        \mathbb{E}_h\mathbb{E}_g\left[\tilde{f}_{ij}\left(z_{ij}^{t+1}\right)\right]
        \le& \tilde{f}_{ij}\left(z_{ij}^{t}\right) + 
        \left\langle \nabla \tilde{f}_{ij}\left(z_{ij}^{t}\right), \mathbb{E}_h\mathbb{E}_g\left[z_{ij}^{t+1} - z_{ij}^{t}\right] \right\rangle + \frac{L}{2} \lVert\mathbb{E}_h\mathbb{E}_g[z_{ij}^{t+1} - z_{ij}^{t}]\rVert_2^2 \\
        & + \mathbb{E}_h\mathbb{E}_g \left[\lVert \mathbb{E}_h[z_{ij}^{t+1} - z_{ij}^{t}] - \mathbb{E}_h\mathbb{E}_g[z_{ij}^{t+1} - z_{ij}^{t}] \rVert_2^2 \right] \\
        & + \mathbb{E}_h\mathbb{E}_g \left[\lVert z_{ij}^{t+1} - z_{ij}^{t} - \mathbb{E}_h[z_{ij}^{t+1} - z_{ij}^{t}]  \rVert_2^2 \right].
    
\end{aligned}\\] The last term can be expanded as follows \\[\begin{aligned}
        &\mathbb{E}_h\mathbb{E}_g \left[\lVert z_{ij}^{t+1} - z_{ij}^{t} - \mathbb{E}_h[z_{ij}^{t+1} - z_{ij}^{t}]  \rVert_2^2 \right] \\
        =&\mathbb{E}_h\mathbb{E}_g \left[ \left\lVert \frac{\eta}{2} (g_i^t + g_j^t) + \frac{\eta\rho}{2}\sum_{k=1}^n (w_{ik}^{t+1} (x_i^t - x_k^t) + w_{jk}^{t+1} (x_j^t - x_k^t)) \right.\right. \\
        & \qquad\left.\left. - \mathbb{E}_h \left[\frac{\eta}{2} (g_i^t + g_j^t) + \frac{\eta\rho}{2}\sum_{k=1}^n (w_{ik}^{t+1} (x_i^t - x_k^t) + w_{jk}^{t+1} (x_j^t - x_k^t)) \right]  \right\rVert_2^2 \right] \\
        =&\mathbb{E}_h\left[ \left\lVert \frac{\eta\rho}{2}\sum_{k=1}^n ((w_{ik}^{t+1} - \mathbb{E}_h[w_{ik}^{t+1}]) (x_i^t - x_k^t) + (w_{jk}^{t+1}- \mathbb{E}_h[w_{jk}^{t+1}]) (x_j^t - x_k^t))  \right\rVert_2^2 \right] \\
        =&\frac{\eta^2\rho^2}{4} \sum_{k\neq i,j} \left(\mathbb{E}_h \left[ \left\lVert w_{ik}^{t+1} - \mathbb{E}_h[w_{ik}^{t+1}] \right\rVert^2_2 \right]  \lVert x_i^t - x_k^t\rVert_2^2 + \mathbb{E}_h \left[ \left\lVert w_{jk}^{t+1} - \mathbb{E}_h[w_{jk}^{t+1}] \right\rVert^2_2 \right]  \lVert x_j^t - x_k^t\rVert_2^2 \right)
    
\end{aligned}\\] where we use the independence of random variables in the last equality. Average over i, j yields \\[\begin{aligned}
    &\frac{1}{c^2}\sum_{ij}\mathbb{E}_h\mathbb{E}_g \left[\lVert z_{ij}^{t+1} - z_{ij}^{t} - \mathbb{E}_h[z_{ij}^{t+1} - z_{ij}^{t}]  \rVert_2^2 \right] \\
    =&\frac{\eta^2\rho^2(c-2)}{2c^2} \sum_{i,j} \mathbb{E}_h \left[ \left\lVert w_{ij}^{t+1} - \mathbb{E}_h[w_{ij}^{t+1}] \right\rVert^2_2 \right]  \lVert x_i^t - x_j^t\rVert_2^2  \\
    \le&\frac{\eta^2\rho^2(c-2)}{2c^2} \frac{4\sigma^2}{b} \sum_{i,j}  \lVert x_i^t - x_j^t\rVert_2^2.
    
\end{aligned}\\] Then \\[\begin{aligned}
        \sum_{i,j\in\mathcal{C}}\norm*{\nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)  }_2^2
        \le& \frac{2}{\eta} \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right)
        +  \left(\frac{3L^2}{4} + 3c^2\rho^2\right) \sum_{i,j\in\mathcal{C}} \norm*{\bm{x}_i^t-\bm{x}_j^t}_2^2 \\
        & + 3nc\rho^2 \sum_{i\in\mathcal{C}} \sum_{k=1}^n |w_{ik}^{t+1} - w_{ik}^\star|^2 \norm*{\bm{x}_i^t - \bm{x}_k^t}_2^2 + \frac{c^2L\eta\sigma^2}{2} \\
        & + \frac{2}{\eta}\frac{L}{2} \frac{\eta^2\rho^2(c-2)}{2} \frac{4\sigma^2}{b} \sum_{i,j}  \lVert x_i^t - x_j^t\rVert_2^2 \\
        =& \frac{2}{\eta} \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right) \\
        & +  \left(\frac{3L^2}{4} + 3c^2\rho^2 + \frac{ L\eta\rho^2 (c-2) 2 \sigma^2}{b} \right) \sum_{i,j\in\mathcal{C}} \norm*{\bm{x}_i^t-\bm{x}_j^t}_2^2 \\
        & + 3nc\rho^2 \sum_{i\in\mathcal{C}} \sum_{k=1}^n |w_{ik}^{t+1} - w_{ik}^\star|^2 \norm*{\bm{x}_i^t - \bm{x}_k^t}_2^2 + \frac{c^2L\eta\sigma^2}{2}.
    
\end{aligned}\\] ◻

</div>

<div id="lemma:consensus_distance" class="lemma" markdown="1">

**Lemma 4**. *Suppose \\(M_{ij}\le\frac{1}{5}\\). Let \\(\rho\ge \frac{\sqrt{3}L}{c}\\) and \\(\eta\le\frac{1}{2\rho c}\le\frac{1}{2\sqrt{3}L}\\) then \\[\begin{aligned}
        \sum_{i,j\in\mathcal{C}}  \norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 
        \le& \left(1-\eta\rho c\right) \sum_{i,j\in\mathcal{C}}\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2 \\
        &+5n\eta\rho\sum_{i\in\mathcal{C}} \sum_{k=1}^n|w_{ik}^t-w_{ik}^\star|^2 \norm{\bm{x}_i^t - \bm{x}_k^t}_2^2 \\
        &+ \frac{6 M^2_{ij}}{\rho c} 
        \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right)
        +  \frac{3\eta M^2_{ij}}{\rho c}  \frac{c^2 L\eta \sigma^2}{2}.
    
\end{aligned}\\]*

</div>

<div class="proof" markdown="1">

*Proof.* Expand \\(\bm{x}_i^{t+1} - \bm{x}_j^{t+1}\\) with <a href="#eq:x" data-reference-type="eqref" data-reference="eq:x">[eq:x]</a> \\[\begin{aligned}
        \bm{x}_i^{t+1} - \bm{x}_j^{t+1}
        =\bm{x}_i^{t} - \bm{x}_j^{t} - \eta\rho \sum_{k=1}^n \left(w_{ik}^{t+1} (\bm{x}_i^t - \bm{x}_k^t) - w_{jk}^{t+1} (\bm{x}_j^t - \bm{x}_k^t) \right)
        - \eta\left(\nabla f_i(\bm{x}_i^t) - \nabla f_j(\bm{x}_j^t)\right).
    
\end{aligned}\\] As \\(i\\) and \\(j\\) belong to the same cluster (i.e., \\(w_{ij}^\star=1\\)), we add \\(\pm2\eta\rho \sum_{k=1}^n w_{ik}^\star (\bm{x}_i^t - \bm{x}_k^t)\\) \\[\begin{aligned}
        \bm{x}_i^{t+1} - \bm{x}_j^{t+1}=&(1-2\eta\rho c) (\bm{x}_i^{t} - \bm{x}_j^{t})
        - \eta\rho \sum_{k=1}^n \left((w_{ik}^{t+1}-w_{ik}^\star) (\bm{x}_i^t - \bm{x}_k^t) - (w_{jk}^{t+1}-w_{jk}^\star) (\bm{x}_j^t - \bm{x}_k^t) \right) \\
        &- \eta\left(\nabla f_i(\bm{x}_i^t) - \nabla f_j(\bm{x}_j^t)\right).
    
\end{aligned}\\] Compute the norm of \\(\bm{x}_i^{t+1} - \bm{x}_j^{t+1}\\) and choose \\(\eta\rho\le \frac{1}{2c}\\) to use Jensen’s inequality \\[\begin{aligned}
        \norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 
        \le& 
         (1-2\eta\rho c)\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2 \\
         &+ 2\eta\rho c\left\lVert \frac{1}{2c} \sum_{k=1}^n \left((w_{ik}^{t+1}-w_{ik}^\star) (\bm{x}_i^t - \bm{x}_k^t) - (w_{jk}^{t+1}-w_{jk}^\star) (\bm{x}_j^t - \bm{x}_k^t) \right) \right. \\
        &\qquad \qquad + \left.\frac{1}{2\rho c}\left(\nabla f_i(\bm{x}_i^t) - \nabla f_j(\bm{x}_j^t)\right)\right\rVert_2^2.
    
\end{aligned}\\] Expand the right-hand side with Cauchy-Schwarz inequality \\[\begin{aligned}
        \norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 
        \le& 
         (1-2\eta\rho c)\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2 \\
         &+ 4\eta\rho c\norm*{ \frac{1}{2c} \sum_{k=1}^n \left((w_{ik}^t-w_{ik}^\star) (\bm{x}_i^t - \bm{x}_k^t) - (w_{jk}^t-w_{jk}^\star) (\bm{x}_j^t - \bm{x}_k^t) \right)}_2^2 \\
        &+ 4\eta\rho c\norm*{  \frac{1}{2\rho c}\left(\nabla f_i(\bm{x}_i^t) - \nabla f_j(\bm{x}_j^t)\right)}_2^2 \\
        \le& 
         (1-2\eta\rho c)\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2
         + 8\eta\rho c\norm*{ \frac{1}{2c} \sum_{k=1}^n (w_{ik}^t-w_{ik}^\star) (\bm{x}_i^t - \bm{x}_k^t)}_2^2 \\
        &+ 8\eta\rho c\norm*{ \frac{1}{2c} \sum_{k=1}^n (w_{jk}^t-w_{jk}^\star) (\bm{x}_j^t - \bm{x}_k^t) }_2^2
        + 4\eta\rho c\norm*{  \frac{1}{2\rho c}\left(\nabla f_i(\bm{x}_i^t) - \nabla f_j(\bm{x}_j^t)\right)}_2^2 \\
        \le& 
         (1-2\eta\rho c)\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2
         + \frac{2n\eta\rho}{c} \sum_{k=1}^n|w_{ik}^t-w_{ik}^\star|^2 \norm{\bm{x}_i^t - \bm{x}_k^t}_2^2\\
         &+ \frac{2n\eta\rho}{c} \sum_{k=1}^n|w_{jk}^t-w_{jk}^\star|^2 \norm{\bm{x}_j^t - \bm{x}_k^t}_2^2 
            + \frac{\eta}{\rho c} \underbrace{\norm{\nabla f_i(\bm{x}_i^t) - \nabla f_j(\bm{x}_j^t)}_2^2}_{=:\mathcal{T}}.
    
\end{aligned}\\] The last term \\(\mathcal{T}\\) can be upper bounded by adding \\(\pm \nabla f_i\left(\bm{z}_{ij}^t\right)
        \pm \nabla f_j\left( \bm{z}_{ij}^t \right)\\) and use L-smoothness assumption <a href="#a:smoothness" data-reference-type="ref+Label" data-reference="a:smoothness">1</a> of \\(f_i\\) and that \\(i\\), \\(j\\) belong to the same cluster <a href="#a:collaborative" data-reference-type="ref+Label" data-reference="a:collaborative">4</a> \\[\begin{aligned}
        \mathcal{T}
        =&\norm*{\nabla f_i(\bm{x}_i^t) \pm \nabla f_i\left(\bm{z}_{ij}^t\right)
        \pm \nabla f_j\left( \bm{z}_{ij}^t \right) - \nabla f_j(\bm{x}_j^t)}_2^2 \\
        \le& 3 \norm*{\nabla f_i(\bm{x}_i^t) - \nabla f_i\left(\bm{z}_{ij}^t \right) }_2^2 + 3 \norm*{ \nabla f_i\left(\bm{z}_{ij}^t \right) - \nabla f_j\left(\bm{z}_{ij}^t\right)}_2^2 \\
        & +3 \norm*{\nabla f_j(\bm{x}_j^t) - \nabla f_j\left(\bm{z}_{ij}^t\right) }_2^2 \\
        \le& \frac{3L^2}{2} \norm{\bm{x}_i^t - \bm{x}_j^t}_2^2 + 3M^2_{ij} \norm*{ \nabla f_i\left(\bm{z}_{ij}^t \right) + \nabla f_j\left(\bm{z}_{ij}^t \right)}_2^2 \\
        =& \frac{3L^2}{2} \norm*{\bm{x}_i^t - \bm{x}_j^t}_2^2 + 3M^2_{ij} \norm*{\nabla \tilde{f}_{ij} (\bm{z}_{ij}^t) }_2^2.
    
\end{aligned}\\] By summing for all \\(i,j\in\mathcal{C}\\) \\[\begin{aligned}
        \sum_{i,j\in\mathcal{C}}  \norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 
        \le& (1-2\eta\rho c) \sum_{i,j\in\mathcal{C}}\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2
        + 4n\eta\rho \sum_{i\in\mathcal{C}} \sum_{k=1}^n|w_{ik}^t-w_{ik}^\star|^2 \norm{\bm{x}_i^t - \bm{x}_k^t}_2^2 \\
        &+ \frac{3\eta L^2}{2\rho c} \sum_{i,j\in\mathcal{C}}\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2
        + \frac{3\eta M^2_{ij}}{\rho c} \sum_{i,j\in\mathcal{C}} \norm*{\nabla \tilde{f}_{ij} (\bm{z}_{ij}^t) }_2^2
    
\end{aligned}\\] Use the previous <a href="#lemma:sd" data-reference-type="ref+Label" data-reference="lemma:sd">3</a> to bound \\(\sum_{i,j\in\mathcal{C}} \norm*{\nabla \tilde{f}_{ij} (\bm{z}_{ij}^t) }_2^2\\) \\[\begin{aligned}
        \sum_{i,j\in\mathcal{C}}  \norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 
        \le& (1-2\eta\rho c) \sum_{i,j\in\mathcal{C}}\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2
        + 4n\eta\rho \sum_{i\in\mathcal{C}} \sum_{k=1}^n|w_{ik}^t-w_{ik}^\star|^2 \norm{\bm{x}_i^t - \bm{x}_k^t}_2^2 \\
        &+ \frac{3\eta L^2}{2\rho c} \sum_{i,j\in\mathcal{C}}\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2 \\
        &+ \frac{3\eta M^2_{ij}}{\rho c} \left(
        \frac{2}{\eta} \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right) \right. \\
        & \left. \qquad+  \left(\frac{3L^2}{4} + 3c^2\rho^2 + \frac{ L\eta\rho^2 (c-2) 2 \sigma^2}{b} \right) \sum_{i,j\in\mathcal{C}} \norm*{\bm{x}_i^t-\bm{x}_j^t}_2^2
        \right) \\
        &+\frac{3\eta M^2_{ij}}{\rho c} \left(3nc\rho^2 \sum_{i\in\mathcal{C}} \sum_{k=1}^n |w_{ik}^{t+1} - w_{ik}^\star|^2 \norm*{\bm{x}_i^t - \bm{x}_k^t}_2^2
        + \frac{c^2 L\eta \sigma^2}{2}
        \right).
    
\end{aligned}\\] Rearrange the terms \\[\begin{aligned}
        &\sum_{i,j\in\mathcal{C}}  \norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 \\
        \le& \left(1-2\eta\rho c+ \frac{3\eta L^2}{2\rho c}+ \frac{3\eta M^2_{ij}}{\rho c} \left(\frac{3L^2}{4} + 3c^2\rho^2 + \frac{ L\eta\rho^2 (c-2) 2 \sigma^2}{b} \right)\right) \sum_{i,j\in\mathcal{C}}\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2 \\
        &+ \left(4n\eta\rho+\frac{3\eta M^2_{ij}}{\rho c}3nc\rho^2\right) \sum_{i\in\mathcal{C}} \sum_{k=1}^n|w_{ik}^t-w_{ik}^\star|^2 \norm{\bm{x}_i^t - \bm{x}_k^t}_2^2 \\
        &+ \frac{3\eta M^2_{ij}}{\rho c} 
        \frac{2}{\eta} \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right)
        +  \frac{3\eta M^2_{ij}}{\rho c}  \frac{c^2 L\eta \sigma^2}{2}.
    
\end{aligned}\\] By taking \\(b\ge \frac{2}{c^2}2L\eta(c-2)\sigma^2\\) and \\(\rho\ge \frac{\sqrt{3}L}{c}\\) and \\(M_{ij}\le\frac{1}{5}\\), the following inequality hold true \\[\begin{aligned}
         \frac{3\eta M^2_{ij}}{\rho c} \left(\frac{3L^2}{4} + 3c^2\rho^2 + \frac{ L\eta\rho^2 (c-2) 2 \sigma^2}{b} \right)
         \le \frac{3\eta M^2_{ij}}{\rho c} \frac{15}{4} \rho^2 c^2
         \le \frac{45}{4}\rho c \eta M^2_{ij}
         \le \frac{1}{2} \eta\rho c.
    
\end{aligned}\\] The upper bound of \\(\sum_{i,j\in\mathcal{C}}  \norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2\\) can be simplied \\[\begin{aligned}
        \sum_{i,j\in\mathcal{C}}  \norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 
        \le& \left(1-\eta\rho c\right) \sum_{i,j\in\mathcal{C}}\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2 \\
        &+5n\eta\rho\sum_{i\in\mathcal{C}} \sum_{k=1}^n|w_{ik}^t-w_{ik}^\star|^2 \norm{\bm{x}_i^t - \bm{x}_k^t}_2^2 \\
        &+ \frac{6 M^2_{ij}}{\rho c} 
        \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right) +  \frac{3\eta M^2_{ij}}{\rho c}  \frac{c^2 L\eta \sigma^2}{2}.
    
\end{aligned}\\] ◻

</div>

## Proof of <a href="#theorem:main" data-reference-type="ref+Label" data-reference="theorem:main">1</a> [proof-of-theoremmain]

<div class="proof" markdown="1">

*Proof.* Given <a href="#lemma:consensus_distance" data-reference-type="ref+Label" data-reference="lemma:consensus_distance">4</a> and average over time \\(t=0\\) over \\(T-1\\) and take expectation to all randomness throughout training \\[\begin{aligned}
        \frac{1}{T}\sum_{t=0}^{T-1}\sum_{i,j\in\mathcal{C}}  \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 \right]
        \le& \left(1-\eta\rho c\right) \frac{1}{T}\sum_{t=0}^{T-1}\sum_{i,j\in\mathcal{C}}\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{x}_i^{t} - \bm{x}_j^{t}}_2^2 \right] \\
        &+5n\eta\rho \frac{1}{T}\sum_{t=0}^{T-1} \sum_{i\in\mathcal{C}} \sum_{k=1}^n \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[|w_{ik}^t-w_{ik}^\star|^2 \norm{\bm{x}_i^t - \bm{x}_k^t}_2^2\right] \\
        &+ \frac{6 M^2_{ij}}{\rho c} \frac{1}{T}\sum_{t=0}^{T-1}
        \sum_{i,j\in\mathcal{C}} \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) \right]- \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right)
        +  \frac{3\eta M^2_{ij}}{\rho c}  \frac{c^2 L\eta \sigma^2}{2}.
    
\end{aligned}\\] Rearrange \\(\frac{1}{T}\sum_{t=0}^{T-1}\sum_{i,j\in\mathcal{C}}  \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2\right]\\) yields \\[\begin{aligned}
        \frac{1}{T}\sum_{t=0}^{T-1}\sum_{i,j\in\mathcal{C}}  \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 \right]
        \le&\frac{5n}{c} \frac{1}{T}\sum_{t=0}^{T-1} \sum_{i\in\mathcal{C}} \sum_{k=1}^n \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[|w_{ik}^t-w_{ik}^\star|^2 \norm{\bm{x}_i^t - \bm{x}_k^t}_2^2\right] \\
        &+ \frac{6 M^2_{ij}}{\eta\rho^2 c^2} \frac{1}{T}\sum_{t=0}^{T-1}
        \sum_{i,j\in\mathcal{C}}\mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right)\right]
        + \frac{3 M^2_{ij}}{\rho^2 }  \frac{ L\eta \sigma^2}{2}.
    
\end{aligned}\\] Consider bounding \\(|w_{ik}^t-w_{ik}^\star|^2\\) in two cases

**Case 1: \\(w_{ik}^\star=1\\).** Suppose \\(M_{ik}\in(0,1)\\), then \\(\norm{\nabla f_i(z^t_{ik}) - \nabla f_k(z^t_{ik})}_2^2 \le M^2_{ij} \norm{\nabla f_i(z^t_{ik}) + \nabla f_j(z^t_{ik})}_2^2\\) implies \\[\begin{aligned}
        \langle \nabla f_i(z^t_{ik}), \nabla f_k(z^t_{ik}) \rangle \ge \frac{1-M^2_{ik}}{2(1+M^2_{ik})} \left( \norm{\nabla f_i(z^t_{ik})}_2^2
        + \norm{\nabla f_k(z^t_{ik})}_2^2 \right) 
        \ge 0.
    
\end{aligned}\\] then \\(w_{ik}^{t+1}= w_{ik}^\star = 1\\) and therefore \\(|w_{ik}^{t+1} - w_{ik}^\star|^2=0\\).

**Case 2: \\(w_{ik}^\star=0\\).** Suppose \\(\zeta^2_{ik}\ge \norm{ \nabla f_i(\bm{x}) + \nabla f_k(\bm{x}) }_2^2\\) for all \\(\bm{x}\\) then \\[\begin{aligned}
        \norm{ \nabla f_i(\bm{z}^t_{ik}) + \nabla f_k(\bm{z}^t_{ik}) }_2^2
        =& \norm{ \nabla f_i(\bm{z}^t_{ik})  }_2^2 + \norm{ \nabla f_k(\bm{z}^t_{ik}) }_2^2
        + 2\langle \nabla f_i(\bm{z}^t_{ik}), \nabla f_k(\bm{z}^t_{ik}) \rangle \\
        \ge& \zeta^2_{ik} + 2\langle \nabla f_i(\bm{z}^t_{ik}), \nabla f_k(\bm{z}^t_{ik}) \rangle
    
\end{aligned}\\] which means the inner product \\(\langle \nabla f_i(z^t_{ij}), \nabla f_j(z^t_{ij}) \rangle \le 0\\) is negative, i.e., \\(w_{ij}^{t+1}=0=w_{ij}^\star\\).

Then with lower bound assumption of \\(f_i\\) and \\(f_j\\) <a href="#a:global_minimum" data-reference-type="ref+Label" data-reference="a:global_minimum">3</a> \\[\begin{aligned}
        \frac{1}{T}\sum_{t=0}^{T-1}\sum_{i,j\in\mathcal{C}}  \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 \right]
        \le& \frac{6 M^2_{ij}}{\eta\rho^2 c^2} \frac{1}{T}\sum_{t=0}^{T-1}
        \sum_{i,j\in\mathcal{C}} \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right) - \tilde{f}_{ij}\left(\bm{z}_{ij}^{t+1}\right)  \right) \right]
        + \frac{3 M^2_{ij}}{\rho^2 }  \frac{ L\eta \sigma^2}{2}\\
        \le& \frac{6M^2_{ij}}{\eta\rho^2 c^2 T} 
        \sum_{i,j\in\mathcal{C}}  \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)
        + \frac{3 M^2_{ij}}{\rho^2 }  \frac{ L\eta \sigma^2}{2}.
    
\end{aligned}\\] Minimize the upper bound through choosing \\(\eta\\) \\[\begin{aligned}
        \eta\le \frac{2}{\sigma\sqrt{LT}} \sqrt{ \frac{1}{c^2} \sum_{i,j\in\mathcal{C}}  \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}
    
\end{aligned}\\] such that \\[\begin{aligned}
\label{eq:xi_xj}
        \frac{1}{T}\sum_{t=0}^{T-1}\sum_{i,j\in\mathcal{C}}  \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{x}_i^{t+1} - \bm{x}_j^{t+1}}_2^2 \right]
        \le& \frac{6M^2_{ij}}{\rho^2 } 
         \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}.
    
\end{aligned}\\] By the result of <a href="#lemma:sd" data-reference-type="ref+Label" data-reference="lemma:sd">3</a> \\[\begin{aligned}
       \frac{1}{T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\nabla \tilde{f}_{ij}\left(\bm{z}_{ij}^{t}\right)  }_2^2\right]
        \le& \frac{2}{\eta} \frac{1}{T} \sum_{i,j\in\mathcal{C}}\left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)
        + 4c^2\rho^2  \frac{1}{T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{x}_i^t-\bm{x}_j^t}_2^2 \right]
        + \frac{c^2 L\eta \sigma^2}{2}\\
        \le& 2c^2\sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}
        + 4c^2\rho^2  \frac{1}{T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\bm{x}_i^t-\bm{x}_j^t}_2^2\right] \\
        \le& \left(2c^2 + 24c^2M^2_{ij}\right) \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)} \\
        \le& 3c^2 \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}.
    
\end{aligned}\\] ◻

</div>

## Proof of <a href="#eq:corollary" data-reference-type="ref+Label" data-reference="eq:corollary">2</a> [proof-of-eqcorollary]

<div class="proof" markdown="1">

*Proof.* By adding \\(\norm*{\nabla f_i(x) + \nabla f_j(x) }_2^2\\) on both sides of <a href="#eq:in_cluster" data-reference-type="eqref" data-reference="eq:in_cluster">[eq:in_cluster]</a>, and replace \\(x\\) with \\(\bm{z}_{ij}\\) we have \\[2\left(\norm*{\nabla f_i(\bm{z}_{ij})}_2^2 + \norm*{\nabla f_j(\bm{z}_{ij})}_2^2\right) \le 4(1+M_{ij}^2)\norm*{\nabla \tilde{f}_{ij}(\bm{z}_{ij})}_2^2 \qquad \forall~\bm{x}\in\mathbb{R}^d.\\] Then using the upper bound of \\(M_{ij} < 1/5\\) from <a href="#theorem:main" data-reference-type="ref+Label" data-reference="theorem:main">1</a>, and average over \\(t\\) and \\(i,j\\) yields \\[\begin{aligned}
\label{eq:nabla_fi_z_ij}
       \frac{1}{c^2T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \mathop{\mathrm{\mathop{\mathrm{\mathbb{E}}}}}\left[\norm*{\nabla f_i\left(\bm{z}_{ij}^{t}\right)}_2^2\right]
        \le& \left(1+\frac{1}{25}\right)3 \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}.
    
\end{aligned}\\] By Cauchy-Schwarz inequality and \\(L\\)-Lipschitz smoothness, we have that \\[\begin{aligned}
         \frac{1}{c^2T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \norm*{ \nabla f_i\left(\bm{x}_{i}^{t}\right)  }_2^2 
        \le & \frac{1}{c^2T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \norm*{\nabla f_i\left(\bm{z}_{ij}^{t}\right)  }_2^2 
        +  \frac{1}{c^2T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \norm*{\nabla f_i\left(\bm{z}_{ij}^{t}\right) - \nabla f_i\left(\bm{x}_{i}^{t}\right)  }_2^2 \\
        \le & \frac{1}{c^2T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \norm*{\nabla f_i\left(\bm{z}_{ij}^{t}\right)  }_2^2 
        + \frac{L^2}{4} \frac{1}{c^2T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \norm*{ \bm{x}_i^t - \bm{x}_j^t }_2^2.
    
\end{aligned}\\] Applying <a href="#eq:xi_xj" data-reference-type="eqref" data-reference="eq:xi_xj">[eq:xi_xj]</a> and <a href="#eq:nabla_fi_z_ij" data-reference-type="eqref" data-reference="eq:nabla_fi_z_ij">[eq:nabla_fi_z_ij]</a> to the upper bound of the above inequality \\[\begin{aligned}
         \frac{1}{c^2T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \norm*{ \nabla f_i\left(\bm{x}_{i}^{t}\right)  }_2^2 
        \le & \left(1+\frac{1}{25}\right)3 \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)} \\
        & 
        + \frac{L^2}{4} \frac{6M^2_{ij}}{\rho^2c^2} \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}.
    
\end{aligned}\\] As \\(\rho c \ge \sqrt{3}L\\) and \\(M_{ij} < \frac{1}{5}\\) as stated in <a href="#theorem:main" data-reference-type="ref+Label" data-reference="theorem:main">1</a>, \\[\begin{aligned}
         \frac{1}{c^2T}\sum_{t=0}^{T-1} \sum_{i,j\in\mathcal{C}} \norm*{ \nabla f_i\left(\bm{x}_{i}^{t}\right)  }_2^2 
        \le & 4 \sqrt{ \frac{L\sigma^2}{c^2T} \sum_{i,j\in\mathcal{C}} \left( \tilde{f}_{ij}\left(\bm{z}_{ij}^{0}\right) - \tilde{f}_{ij}^\star  \right)}.
    
\end{aligned}\\] ◻

</div>

# Experimental Details [sec:exp-details]

In Section <a href="#sec:exps" data-reference-type="ref" data-reference="sec:exps">4</a>, we present our results on two tasks with different properties. Here, we provide the full details of our experimental setup, alongside with additional experiments.

We first describe the setup for cross-device and cross-silo experiments: we use the fix batch size of 128 for cross-device, and cross-silo experiments on CIFAR-100. We tune each method for the optimal learning rate individually: we use learning rate of 0.1 for ditto, 0.05 for Federated Clustering (FC), and 0.01 for all other methods. For Ditto, we use the hyper-parameter of \\(\lambda = 1\\) as recommended in their paper. For Federated Clustering, we use the ground truth number of clusters and size of clusters as the hyper-parameter. We also use the ground truth number of clusters for IFCA, and sample all the clients in cross-silo experiment. We reduce the sampling rate to 10% for the cross-device experiment to ensure scalability and fairness for comparison to other methods. For cross-silo experiments we employed a single NVIDIA V-100 GPU with 32GB memory, and moved to four NVIDIA V-100 GPUs with 32 GB memory for cross-device experiment. With this setup, running <span class="smallcaps">CoBo</span> for cross-silo and cross-device experiment takes 9 hours and 28 hours respectively.

For Language modelling experiment, we conducted the experiments with the learning rate of 0.002, batch size of 50, and 4 accumulation steps. Note that each agent only get a subset of the regarding language from Wiki-40B dataset, consisting of total of 800000 tokens. We also used the context length of 512, dropout rate of 0.1, and LoRA module with rank 4. Training is performed on a single NVIDIA A-100 GPU with 40GB memory. It takes 2.5 hours to run <span class="smallcaps">CoBo</span> for 500 iterations in this framework. We also use \\(\lambda = 0.1\\) which has higher performance for this experiment.

<figure id="fig:heatmaps80">
<img src="./figures/heatmaps80.png"" style="width:90.0%" />
<figcaption> Collaboration matrices learned by <span class="smallcaps">CoBo</span> at different stages of training for cross-device experiment with 80 clients. The diagonals are masked out. The oracle matrix is a block diagonal matrix, consisting of 10 blocks: two blocks of size 10, two blocks of size 9, and so on. The collaboration matrix of <span class="smallcaps">CoBo</span> already starts to look similar to oracle matrix within as low as 300 iterations. (1.5% of the total iterations)</figcaption>
</figure>

<figure id="fig:accuracy-80">
<img src="./figures/accuracy80-new.png"" style="width:90.0%" />
<figcaption>Averaged accuracies of personalized models for cross-device federated learning with 80 clients. The "Oracle" denotes applying FedAvg to the clients having the data from the same classes of CIFAR-100 dataset.</figcaption>
</figure>

For the cross-device experiment with 80 agents in Section <a href="#subsec:lang-exp" data-reference-type="ref" data-reference="subsec:lang-exp">4.3</a>, we present the accuracy curve in Figure <a href="#fig:accuracy-80" data-reference-type="ref" data-reference="fig:accuracy-80">9</a>. Our method outperform all other methods except the Oracle with a large margin. We can also observe the collaboration matrix of <span class="smallcaps">CoBo</span> in Figure <a href="#fig:heatmaps80" data-reference-type="ref" data-reference="fig:heatmaps80">8</a>. The clusters are learned with <span class="smallcaps">CoBo</span> efficiently.

[^1]: The code is available at: https://github.com/epfml/CoBo

[^2]: https://github.com/karpathy/nanoGPT
