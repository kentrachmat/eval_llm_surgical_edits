# Constrained Synthesis with Projected Diffusion Models

## Abstract

This paper introduces an approach to endow generative diffusion processes the ability to satisfy and certify compliance with constraints and physical principles. The proposed method recast the traditional sampling process of generative diffusion models as a constrained optimization problem, steering the generated data distribution to remain within a specified region to ensure adherence to the given constraints. These capabilities are validated on applications featuring both convex and challenging, non-convex, constraints as well as ordinary differential equations, in domains spanning from synthesizing new materials with precise morphometric properties, generating physics-informed motion, optimizing paths in planning scenarios, and human motion synthesis.

# Introduction [sec:introduction]

Generative diffusion models excel at robustly synthesizing content from raw noise through a sequential denoising process `\cite{sohl2015deep, ho2020denoising}`{=latex}. They have revolutionized high-fidelity creation of complex data, and their applications have rapidly expanded beyond mere image synthesis, finding relevance in areas such as engineering `\cite{wang2023diffusebot, zhong2023guided}`{=latex}, automation `\cite{carvalho2023motion, janner2022planning}`{=latex}, chemistry `\cite{anand2022protein, hoogeboom2022equivariant}`{=latex}, and medical analysis `\cite{cao2024high, chung2022score}`{=latex}. However, although diffusion models excel at generating content that is coherent and aligns closely with the original data distribution, their direct application in scenarios requiring stringent adherence to predefined criteria poses significant challenges. Particularly the use of diffusion models in scientific and engineering domains where the generated data needs to not only resemble real-world examples but also rigorously comply with established specifications and physical laws remains an open challenge.

Given these limitations, one might consider training a diffusion model on a dataset that already adheres to specific constraints. However, even with “feasible” training data, this approach does not guarantee adherence to desired criteria due to the stochastic nature of the diffusion process. Furthermore, there are frequent scenarios where the training data must be altered to generate outputs that align with specific properties, potentially not present in the original data. This issue often leads to a distribution shift further exacerbating the inability of generative models to produce “valid” data. As we will show in a real-world experiment (§<a href="#sec:constrained-materials" data-reference-type="ref" data-reference="sec:constrained-materials">6.1</a>), this challenge is particularly acute in scientific and engineering domains, where training data is often sparse and confined to specific distributions, yet the synthesized outputs are required to meet stringent properties or precise standards `\cite{wang2023diffusebot}`{=latex}.

This paper addresses these challenges and introduces *Projected Diffusion Models* (PDM), a novel approach that recast the traditional sampling strategy in diffusion processes as a constrained-optimization problem. This perspective allows us to apply traditional techniques from constraint optimization to the sampling process. In this work, the problem is solved by iteratively projecting the diffusion sampling process onto arbitrary constraint sets, ensuring that the generated data adheres strictly to imposed constraints or physical principles. We provide theoretical support for PDM’s capability to not only certify adherence to the constraints but also to optimize the generative model’s original objective of replicating the true data distribution. This alignment is a significant advantage of PDM, yielding state-of-the-art FID scores while maintaining strict compliance with the imposed constraints.

**Contributions.** In summary, this paper makes the following key contributions: **(1)** It introduces PDM, a new framework that augments diffusion-based synthesis with arbitrary constraints in order to generate content with high fidelity that also adheres to the imposed specifications. The paper elucidates the theoretical foundation that connects the reverse diffusion process to an optimization problem, facilitating the direct incorporation of constraints into the reverse process of score-based diffusion models. **(2)** Extensive experiments across various domains demonstrate PDM’s effectiveness. These include adherence to morphometric properties in real-world material science experiments, physics-informed motion governed by ordinary differential equations, trajectory optimization in motion planning, and constrained human motion synthesis, showcasing PDM’s ability to produce content that adheres to both complex constraints and physical principles. **(3)** We further show that PDM is able to generate out-of-distribution samples that meet stringent constraints, even in scenarios with extremely sparse training data and when the training data does not satisfy the required constraints. **(4)** Finally, we provide a theoretical basis elucidating the ability of PDM to generate highly accurate content while ensuring constraint compliance, underpinning the practical implications of this approach.

# Preliminaries: Diffusion models [sec:preliminaries]

Diffusion-based generative models `\cite{sohl2015deep, ho2020denoising}`{=latex} expand a data distribution, whose samples are denoted \\(\bm{x}_0\\), through a Markov chain parameterization \\(\{\bm{x}_t\}_{t=1}^T\\), defining a Gaussian diffusion process \\(p(\bm{x}_0) = \int p(\bm{x}_T) \prod_{t=1}^T p(\bm{x}_{t-1} \vert \bm{x}_t) d \bm{x}_{1:T}\\).

In the *forward process*, the data is incrementally perturbed towards a Gaussian distribution. This process is represented by the transition kernel \\(q(\bm{x}_t \vert \bm{x}_{t-1}) = \mathcal{N}(\bm{x}_t; \sqrt{1 - \beta_t} \bm{x}_{t-1}, \beta_t \bm{I})\\) for some \\(0 < \beta_t < 1\\), where the \\(\beta\\)-schedule \\(\{\beta_t\}_{t=1}^T\\) is chosen so that the final distribution \\(p(\bm{x}_T)\\) is nearly Gaussian. The diffusion time \\(t\\) allows an analytical expression for variable \\(\bm{x}_t\\) represented by \\(\chi_t(\bm{x}_0, \epsilon) = \sqrt{{\alpha}_t} \bm{x}_0 + \sqrt{1 - {\alpha}_t} \epsilon\\), where \\(\epsilon \sim \mathcal{N}(\bm{0}, \bm{I})\\) is a noise term, and \\({\alpha}_t = \prod_{i=1}^t \left(1 - \beta_i\right)\\). This process is used to train a neural network \\(\epsilon_\theta(\bm{x}_t, t)\\), called the *denoiser*, which implicitly approximates the underlying data distribution by learning to remove noise added throughout the forward process.  
The training objective minimizes the error between the actual noise \\(\epsilon\\) and the predicted noise \\(\epsilon_\theta(\chi_t(\bm{x}_0, \epsilon), t)\\) via the loss function: \\[\min_\theta 
    \mathop{\mathrm{\mathbb{E}}}_{t \sim [1,T],\; p(\bm{x}_0), \mathcal{N}(\epsilon; \bm{0}, \bm{I})}
    \left[ \left\| 
    \epsilon - \epsilon_\theta( \chi_t(\bm{x}_0, \epsilon), t ) 
    \right\|^2 \right].\\] The *reverse process* uses the trained denoiser, \\(\epsilon_\theta(\bm{x}_t, t)\\), to convert random noise \\(p(\bm{x}_T)\\) iteratively into realistic data from distribution \\(p(\bm{x}_0)\\). Practically, \\(\epsilon_\theta\\) predicts a single step in the denoising process that can be used during sampling to reverse the diffusion process by approximating the transition \\(p(\bm{x}_{t-1} \vert \bm{x}_t)\\) at each step \\(t\\).

**Score-based models** `\cite{song2019generative, song2020score}`{=latex}, while also operating on the principle of gradually adding and removing noise, focus on directly modeling the gradient (score) of the log probability of the data distribution at various noise levels. The score function \\(\nabla_{\bm{x}_t} \log p(\bm{x}_t)\\) identifies the direction and magnitude of the greatest increase in data density at each noise level. The training aims to optimize a neural network \\(\mathbf{s}_{\theta}(\bm{x}_t, t)\\) to approximate this score function, minimizing the difference between the estimated and true scores of the perturbed data: \\[\label{eq:score}
    \displaystyle 
    \min_{\theta} \mathop{\mathrm{\mathbb{E}}}_{t \sim [1,T], p(\bm{x}_0), q(\bm{x}_t|\bm{x}_0)}
    (1 -{\alpha}_t)
    \left[ \left\| \mathbf{s}_{\theta}(\bm{x}_t, t) - \nabla_{\bm{x}_t} \log q(\bm{x}_t|\bm{x}_0) \right\|^2 \right],\\] where \\(q(\bm{x}_t|\bm{x}_0) = \mathcal{N}(\bm{x}_t; \sqrt{{\alpha}_t} \bm{x}_{0}, (1 -{\alpha}_t) \bm{I})\\) defines a distribution of perturbed data \\(\bm{x}_t\\), generated from the training data, which becomes increasingly noisy as \\(t\\) approach \\(T\\). This paper considers score-based models.

# Related work and limitations

While diffusion models are highly effective in producing content that closely mirrors the original data distribution, the stochastic nature of their outputs act as an impediment when specifications or constraints need to be imposed on the generated outputs. In an attempt to address this issue, two main approaches could be adopted: (1) model conditioning and (2) post-processing corrections.

**Model conditioning** `\cite{ho2022classifier}`{=latex} aims to control generation by augmenting the diffusion process via a conditioning variable \\(\bm{c}\\) to transform the denoising process via classifier-free guidance: \\[\hat{\epsilon}_\theta \stackrel{\text{def}}{=}
    \lambda \times\epsilon_\theta(\bm{x}_t, t, \bm{c}) +
    (1-\lambda) \times \epsilon_\theta(\bm{x}_t, t, \bot),\\] where \\(\lambda \in (0,1)\\) is the *guidance scale* and \\(\bot\\) is a null vector representing non-conditioning. These methods have been shown effective in capturing properties of physical design `\cite{wang2023diffusebot}`{=latex}, positional awareness `\cite{carvalho2023motion}`{=latex}, and motion dynamics `\cite{yuan2023physdiff}`{=latex}. However, while conditioning may be effective to influence the generation process, it lacks the rigor to ensure adherence to specific constraints. This results in generated outputs that, despite being plausible, may not be accurate or reliable. Figure <a href="#fig:reverse-violations" data-reference-type="ref" data-reference="fig:reverse-violations">1</a> (red colors) illustrates this issue on a physics-informed motion experiment (detailed in §<a href="#sec:physics-informed-motion" data-reference-type="ref" data-reference="sec:physics-informed-motion">6.4</a>). The figure reports the distance of the model outputs to feasible solutions, showcasing the constraint violations identified in a conditional model’s outputs. Notably, the model, conditioned on labels corresponding to positional constraints, fails to generate outputs that adhere to these constraints, resulting in outputs that lack meaningful physical interpretation.

<figure id="fig:reverse-violations">
<img src="./figures/figure_1_cgdm.png"" />
<figcaption>Sampling steps failing to converge to feasible solutions in <span style="color: Mahogany">conditional models (red)</span> while minimizing the constraint divergence to <span class="math inline">0</span> under <span style="color: MidnightBlue">PDM (blue)</span>. </figcaption>
</figure>

Additionally, conditioning in diffusion models often requires training supplementary classification and regression models, a process fraught with its own set of challenges. This approach demands the acquisition of extra labeled data, which can be impractical or unfeasible in specific scenarios. For instance, our experimental analysis will demonstrate a situation in material science discovery where the target property is well-defined, but the original data distribution fails to embody this property. This scenario is common in scientific applications, where data may not naturally align with desired outcomes or properties `\cite{maze2023diffusion}`{=latex}.

**Post-processing correction.** An alternative approach involves applying post-processing steps to correct deviations from desired constraints in the generated samples. This correction is typically implemented in the last noise removal stage, \\(s_\theta(\bm{x}_1, 1)\\). Some approaches have augmented this process to use optimization solvers to impose constraints on synthesized samples `\cite{giannone2023aligning, power2023sampling,maze2023diffusion}`{=latex}. However these approaches present two main limitations. First, their objective does not align with optimizing the score function. This inherently positions the diffusion model’s role as ancillary, with the final synthesized data often resulting in a significant divergence from the learned (and original) data distributions, as we will demonstrate in §<a href="#sec:experiments" data-reference-type="ref" data-reference="sec:experiments">6</a>. Second, these methods are reliant on a limited and problem specific class of objectives and constraints, such as specific trajectory “constraints” or shortest path objectives which can be integrated as a post-processing step `\cite{giannone2023aligning, power2023sampling}`{=latex}.

**Other methods.** Some methods explored modifying either diffusion training or inference to adhere to desired properties. For instance, the methods in `\cite{frerix2020homogeneous}`{=latex} and `\cite{liu2024mirror}`{=latex}, support simple linear or convex sets, respectively. Similarly, `\citet{fishman2023diffusion,fishman2024metropolis}`{=latex} focus on predictive tasks within convex polytope, which are however confined to approximations by simple geometries like L2-balls. While important contributions, these approaches prove insufficient for the complex constraints present in many real-world tasks. Conversely, in the domain of image sampling, `\citet{lou2023reflected}`{=latex} and `\citet{saharia2022photorealistic}`{=latex} introduce methods like reflections and clipping to control numerical errors and maintain pixel values within the standard \[0,255\] range during the reverse diffusion process. These techniques, while enhancing sampling accuracy, do not address broader constraint satisfaction challenges.

To overcome these gaps and handle arbitrary constraints, our approach casts the reverse diffusion process to a constraint optimization problem that is then solved throught repeated projection steps.

# Constrained generative diffusion

This section establishes a theoretical framework that connects the reverse diffusion process as an optimization problem. This perspective facilitates the incorporation of constraints directly into the process, resulting in the constrained optimization formulation presented in Equation <a href="#eq:constrained_optimization" data-reference-type="eqref" data-reference="eq:constrained_optimization">[eq:constrained_optimization]</a>.

The application of the reverse diffusion process of score-based models is characterized by iteratively transforming the initial noisy samples \\(\bm{x}_T\\) back to a data sample \\(\bm{x}_0\\) following the learned data distribution \\(q(\bm{x}_0)\\). This transformation is achieved by iteratively updating the sample using the estimated score function \\(\nabla_{\bm{x}_t} \log q(\bm{x}_t|\bm{x}_0)\\), where \\(q(\bm{x}_t | \bm{x}_0)\\) is the data distribution at time \\(t\\). At each time step \\(t\\), starting from \\(\bm{x}_t^0\\), the process performs \\(M\\) iterations of *Stochastic Gradient Langevin Dynamics* (SGLD) `\cite{welling2011bayesian}`{=latex}: \\[\label{eq:sgld}
    \bm{x}_{t}^{i+1} = \bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^i|\bm{x}_0) + \sqrt{2\gamma_t}\bm{\epsilon},\\] where \\(\bm{\epsilon}\\) is standard normal, \\(\gamma_t > 0\\) is the step size, and \\(\nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^i|\bm{x}_0)\\) is approximated by the learned score function \\(\mathbf{s}_{\theta}(\mathbf{x}_t, t)\\).

## Casting the reverse process as an optimization problem

First note that SGLD is derived from discretizing the continuous-time Langevin dynamics, which are governed by the stochastic differential equation: \\[\label{eq:langevin}
    d\bm{X}(t) = \nabla \log q(\bm{X}(t))\, dt + \sqrt{2}\, d\bm{B}(t),\\] where \\(\bm{B}(t)\\) is standard Brownian motion. Under appropriate conditions, the stationary distribution of this process is \\(q(\bm{x}_t )\\) `\cite{roberts1996exponential}`{=latex}, implying that samples generated by Langevin dynamics will, over time, be distributed according to \\(q(\bm{x}_t)\\). In practice, these dynamics are simulated using a discrete-time approximation, leading to the SGLD update in Equation <a href="#eq:sgld" data-reference-type="eqref" data-reference="eq:sgld">[eq:sgld]</a>. Therein the noise term \\(\sqrt{2\gamma_t}\, \bm{\epsilon}_{t}^{i}\\) allows the algorithm to explore the probability landscape and avoid becoming trapped in local maxima.

Next notice that, as detailed in `\cite{xu2018global, welling2011bayesian}`{=latex}, under some regularity conditions this iterative SGLD algorithm converges toward a stationary point, bounded by \\(\frac{d^2}{\sigma^{1/4}\lambda^*}\log(1/\epsilon),\\) where, \\(\sigma^2\\) represents the variance schedule, \\(\lambda^*\\) denotes the uniform spectral gap of the Langevin diffusion, and \\(d\\) is the dimensionality of the problem. Thus, as the reverse diffusion process progresses towards \\(T \to 0\\), and the variance schedule decreases, the stochastic component becomes negligible, and SGLD transitions toward deterministic gradient ascent on \\(\log q(\bm{x}_t)\\). In the limit of vanishing noise, the update rule simplifies to: \\[\label{eq:gradient_ascent}
    \bm{x}_{t}^{i+1} = \bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}} \log q(\bm{x}_{t}^{i} | \bm{x}_0),\\] which is standard gradient ascent aiming to maximize \\(\log q(\bm{x}_t)\\). This allow us to view the reverse diffusion process as an optimization problem minimizing the negative log-likelihood of the data distribution \\(q(\bm{x}_t | \bm{x}_0)\\) at each time step \\(t\\).

In traditional score-based models, at any point throughout the reverse process, \\(\bm{x}_t\\) is *unconstrained*. When these samples are required to satisfy some constraints, the objective remains unchanged, but the solution to this optimization must fall within a feasible region \\(\mathbf{C}\\), and thus the optimization problem formulation becomes: \\[\label{eq:constrained_optimization}
    \begin{align}
        \label{eq:constrained-diffusion}
        \underset{{\bm{x}_{T}, \ldots, \bm{x}_1}}{\text{minimize}} &\;
        \sum_{t = T, \ldots, 1}- \log q(\bm{x}_{t}|\bm{x}_0) \\
        \label{eq:constrained-diffusion-constr}
        \textrm{s.t.:}  &\quad \bm{x}_{T}, \ldots, \bm{x}_0 \in \mathbf{C}.
    \end{align}\\] Operationally, the negative log likelihood is minimized at each step of the reverse Markov chain, as the process transitions from \\(\bm{x}_T\\) to \\(\bm{x}_0\\). In this regard, and importantly, the objective of the PDM’s reverse sampling process is aligned with that of traditional score-based diffusion models.

## Constrained guidance through iterative projections [subsec:guided-projections]

The score network \\(\mathbf{s}_{\theta}(\bm{x}_{t}, t)\\) directly estimates the first-order derivatives of Equation <a href="#eq:constrained-diffusion" data-reference-type="eqref" data-reference="eq:constrained-diffusion">[eq:constrained-diffusion]</a>, providing the necessary gradients for iterative gradient-based updates defined in Equation <a href="#eq:sgld" data-reference-type="eqref" data-reference="eq:sgld">[eq:sgld]</a>. In the presence of constraints <a href="#eq:constrained-diffusion-constr" data-reference-type="eqref" data-reference="eq:constrained-diffusion-constr">[eq:constrained-diffusion-constr]</a>, however, an alternative iterative method is necessary to guarantee feasibility. PDM models a projected guidance approach to provide this constraint-aware optimization process.

First, we define the projection operator, \\(\mathcal{P}_{\mathbf{C}}\\), as a constrained optimization problem, \\[\label{eq:projection}
    \mathcal{P}_{\mathbf{C}}(\bm{x}) = \mathop{\mathrm{argmin}}_{\bm{y} \in \mathbf{C}} ||\bm{y} - \bm{x}||_{2}^2,\\] that finds the nearest feasible point to the input \\(\bm{x}\\). The *cost of the projection* \\(||\bm{y} - \bm{x}||_{2}^2\\) represents the distance between the closest feasible point and the original input.

To retain feasibility through an application of the projection operator after each update step, the paper defines *projected diffusion model sampling* step as \\[\label{eq:reverse-pgd}
    \bm{x}_{t}^{i+1} = \mathcal{P}_{\mathbf{C}} \left(\bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}|\bm{x}_0) + \sqrt{2\gamma_t}\bm{\epsilon}\right),\\] where \\(\mathbf{C}\\) is the set of constraints and \\(\mathcal{P}_{\mathbf{C}}\\) is a projection onto \\(\mathbf{C}\\). Hence, iteratively throughout the Markov chain, a gradient step is taken to minimize the objective defined by Equation <a href="#eq:constrained-diffusion" data-reference-type="eqref" data-reference="eq:constrained-diffusion">[eq:constrained-diffusion]</a> while ensuring feasibility. Convergence is guaranteed for convex constraints sets `\cite{parikh2014proximal}`{=latex} and empirical evidence in §<a href="#sec:experiments" data-reference-type="ref" data-reference="sec:experiments">6</a> showcases the applicability of this methods to arbitrary constraint sets. Importantly, the projection

<figure id="alg:pgd_annealed_ld">
<figure id="alg:pgd_annealed_ld">
<p>ALGORITHM BLOCK (caption below)</p>
<p><span class="math inline"><strong>x</strong><sub><em>T</em></sub><sup>0</sup> ∼ 𝒩(<strong>0</strong>, <em>σ</em><sub><em>T</em></sub><strong>I</strong>)</span><br />
<br />
<strong>For</strong> <span><span class="math inline"><em>t</em> = <em>T</em></span> <span class="math inline">1</span></span><span> <span class="math inline">$\gamma_t \gets \nicefrac{\sigma_{t}^2}{2 \sigma_{T}^2}$</span><br />
<br />
<strong>For</strong> <span><span class="math inline"><em>i</em> = 1</span> <span class="math inline"><em>M</em></span></span><span> <span class="math inline"><strong>ϵ</strong> ∼ 𝒩(<strong>0</strong>, <strong>I</strong>)</span>;    <span class="math inline"><strong>g</strong> ← <strong>s</strong><sub><em>θ</em><sup>*</sup></sub>(<strong>x</strong><sub><em>t</em></sub><sup><em>i</em> − 1</sup>, <em>t</em>)</span><br />
<span class="math inline">$\bm{x}_{t}^{i} = \mathcal{P}_{\bm{C}}(\bm{x}_{t}^{i-1} + \gamma_t \bm{g} + \sqrt{2\gamma_t}\bm{\epsilon})$</span> </span> <span class="math inline"><strong>x</strong><sub><em>t</em> − 1</sub><sup>0</sup> ← <strong>x</strong><sub><em>t</em></sub><sup><em>M</em></sup></span> </span></p>
<p><br />
Return <span class="math inline"><strong>x</strong><sub>0</sub><sup>0</sup></span></p>
<figcaption>PDM</figcaption>
</figure>
<figcaption>PDM</figcaption>
</figure>

operators can be *warm-started* during the repeated sampling step providing a piratical solution even for hard non-convex constrained regions. The full sampling process is detailed in Algorithm <a href="#alg:pgd_annealed_ld" data-reference-type="ref" data-reference="alg:pgd_annealed_ld">3</a>.

By incorporating constraints throughout the sampling process, the interim learned distributions are steered to comply with these specifications. This is empirically evident from the pattern in Figure <a href="#fig:reverse-violations" data-reference-type="ref" data-reference="fig:reverse-violations">1</a> (<span style="color: MidnightBlue">blue curves</span>): remarkably, the constraint violations decrease with each addition of estimated gradients and noise and approaches \\(0\\)-violation as \\(t\\) nears zero. *This trend not only minimizes the impact but also reduces the optimality cost of projections applied in the later stages of the reverse process.* We provide theoretical rationale for the effectiveness of this approach in §<a href="#subsec:theoretical-analysis" data-reference-type="ref" data-reference="subsec:theoretical-analysis">5</a> and conclude this section by noting that this approach can be clearly distinguished from other methods which use a diffusion model’s sampling process to generate starting points for a constrained optimization algorithm `\cite{giannone2023aligning, power2023sampling}`{=latex}. Instead, PDM leverages minimization of negative log likelihood as the primary objective of the sampling algorithm akin to standard unconstrained sampling procedures. This strategy offers a key advantage: *the probability of generating a sample that conforms to the data distribution is optimized directly*, rather than an external objective, *while simultaneously imposing verifiable constraints*. In contrast, existing baselines often neglect the conformity to the data distribution, which, as we will show in the next section, can lead to a deviation from the learned distribution and an overemphasis on external objectives for solution generation, resulting in significant divergence from the data distribution, reflected by high FID scores.

# Effectiveness of PDM: A theoretical justification [subsec:theoretical-analysis]

Next, we theoretically justify the use of iterative projections to guide the sample to the constrained distribution. The analysis assumes that the feasible region \\(\bm{C}\\) is a convex set. All proofs are reported in the Appendix. We start by defining the update step.

<div class="definition" markdown="1">

**Definition 1**. The operator \\(\mathcal{U}\\) defines a single update step for the sampling process as, \\[\label{eq:update-def}
        \mathcal{U}(\bm{x}_t^{i}) = \bm{x}_{t}^{i} + \gamma_t \mathbf{s}_{\theta}(\bm{x}_{t}^{i}, t) + \sqrt{2\gamma_t}\bm{\epsilon}.\\]

</div>

The next result establishes a convergence criteria on the proximity to the optimum, where for each time step \\(t\\) there exists a minimum value of \\(i = \Bar{I}\\) such that, \\[\label{eq:grad-size}
        \exists{\Bar{I}} \; \texttt{s.t.} \; \left\| (\bm{x}_t^{\Bar{I}} + \gamma_t \nabla_{\bm{x}_{t}^{\Bar{I}}} \log q(\bm{x}_{t}^{\Bar{I}}|\bm{x}_0)) \right\|_2 \leq \left\|\rho_t \right\|_2\\] where \\(\rho_t\\) is the closest point to the global optimum that can be reached via a single gradient step from any point in \\(\mathbf{C}\\).

<div id="proof:theorem-1" class="theorem" markdown="1">

**Theorem 2**. **Let \\(\mathcal{P}_{\mathbf{C}}\\) be a projection onto \\(\mathbf{C}\\), \\(\bm{x}_{t}^{i}\\) be the sample at time step \\(\bm{t}\\) and iteration \\(\bm{i}\\), and ‘Error’ be the cost of the projection (). Assume \\(\nabla_{\bm{x}_t} \log p(\bm{x}_t)\\) is convex. For any \\(\bm{i} \geq \Bar{I}\\), \\[\label{eq:theorem-1}
        \mathbb{E} \left[ \textit{Error}(\mathcal{U}(\bm{x}_{t}^{i}), \mathbf{C}) \right] \geq \mathbb{E} \left[ \textit{Error}(\mathcal{U}(\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i})), \mathbf{C}) \right]\\]**

</div>

The proof for Theorem <a href="#proof:theorem-1" data-reference-type="ref" data-reference="proof:theorem-1">2</a> is reported in §<a href="#appendix:theorem-proof" data-reference-type="ref" data-reference="appendix:theorem-proof">17</a>. This result suggests that PDM’s projection steps ensure the resulting samples adhere more closely to the constraints as compared to samples generated through traditional, unprojected methods. Together with the next results, it will allow us to show that PDM samples converge to the point of maximum likelihood that also satisfy the imposed constraints.

The theoretical insight provided by Theorem <a href="#proof:theorem-1" data-reference-type="ref" data-reference="proof:theorem-1">2</a> provides an explanation for the observed discrepancy between the constraint violations induced by the conditional model and PDM, as in Figure <a href="#fig:reverse-violations" data-reference-type="ref" data-reference="fig:reverse-violations">1</a>.

<div id="proof:corollary-1" class="corollary" markdown="1">

**Corollary 3**. *For arbitrary small \\(\xi > 0\\), there exist \\(t\\) and \\(\bm{i} \geq \Bar{I}\\) such that: \\[\textit{Error}(\mathcal{U}(\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i})), \mathbf{C}) \leq \xi.\\]*

</div>

The above result uses the fact that the step size \\(\gamma_t\\) is strictly decreasing and converges to zero, given sufficiently large \\(T\\), and that the size of each update step \\(\mathcal{U}\\) decreases with \\(\gamma_t\\). As the step size shrinks, the gradients and noise reduce in size. Hence, \\(\textit{Error}(\mathcal{U}(\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i}))\\) approaches zero with \\(t\\), as illustrated in Figure <a href="#fig:reverse-violations" data-reference-type="ref" data-reference="fig:reverse-violations">1</a> (right). This diminishing error implies that the projections gradually steer the sample into the feasible subdistribution of \\(p(\bm{x}_0)\\), effectively aligning with the specified constraints.

#### Feasibility guarantees.

PDM provides feasibility guarantees when solving convex constraints. This assurance is integral in sensitive settings, such as material analysis (Section <a href="#sec:constrained-materials" data-reference-type="ref" data-reference="sec:constrained-materials">6.1</a>), plausible motion synthesis (Section <a href="#sec:human-motion" data-reference-type="ref" data-reference="sec:human-motion">6.2</a>), and physics-based simulations (Section <a href="#sec:physics-informed-motion" data-reference-type="ref" data-reference="sec:physics-informed-motion">6.4</a>), where strict adherence to the constraint set is necessary.

<div id="proof:theorem-2" class="corollary" markdown="1">

**Corollary 4**. *PDM provides feasibility guarantees for convex constraint sets, for *arbitrary* density functions \\(\nabla_{\bm{x}_t} \log p(\bm{x}_t)\\).*

</div>

# Experiments [sec:experiments]

We compare PDM against three methodologies, each employing state-of-the-art specialized methods tailored to the various applications tested:: **(1)** *Conditional diffusion models* (*Cond*) `\cite{ho2022classifier}`{=latex} are the state-of-the-art methods for generative sampling subject to a series of specifications. While conditional diffusion models offer a way to guide the generation process towards satisfying certain constraints, they do not provide compliance guarantees. **(2)** To encourage constraints satisfaction, we additionally compare to conditional models with a post-processing projection step (*Cond\\(^+\\)*), emulating the post-processing approaches of `\citep{giannone2023aligning, power2023sampling}`{=latex} in various domains presented next. Finally, **(3)** we use a score-based model identical to our implementation but with a single post-processing projection operation (*Post\\(^+\\)*) performed at the last sampling step. Additional details are provided in §<a href="#appendix:experimental-settings" data-reference-type="ref" data-reference="appendix:experimental-settings">12</a>.

The performance of these models are evaluated by the *feasibility* and *accuracy* of the generated samples. Feasibility is assessed by the degree and rate at which constraints are satisfied, expressly, the percentage of samples which satisfy the constraints with a given error tolerance. Accuracy is measured by the FID score, a standard metric in synthetic sample evaluation. To demonstrate the broad applicability of our approach, our experimental settings have been selected to exhibit:

1.  Behavior in low data regimes and with original distribution violating constraints (§<a href="#sec:constrained-materials" data-reference-type="ref" data-reference="sec:constrained-materials">6.1</a>), as part of a real-world material science experiment.

2.  Behavior on 3-dimensional sequence generation with physical constraints (§<a href="#sec:human-motion" data-reference-type="ref" data-reference="sec:human-motion">6.2</a>).

3.  Behavior on complex non-convex constraints (§<a href="#sec:constrained-trajectories" data-reference-type="ref" data-reference="sec:constrained-trajectories">6.3</a>).

4.  Behavior on ODEs and under constraints outside the training distribution. (§<a href="#sec:physics-informed-motion" data-reference-type="ref" data-reference="sec:physics-informed-motion">6.4</a>).

## Constrained materials *(low data regimes and constraint-violating distributions)* [sec:constrained-materials]

The first setting focuses on a real-world application in material science, conducted as part of an experiment to expedite the discovery of structure–property linkages. From a sparse collection of micro-structure images, we aim to generate new samples with target porosity levels that are rarely (if ever) observed in the data.

PDM handles this task by automatically guiding the reverse diffusion steps toward images that satisfy the desired porosity without compromising visual fidelity. Rather than introducing ad-hoc heuristics or hand-tuned post-processing, we rely on the built-in projection routine described in Section 4, which seamlessly adapts to the binary pixel nature of the microscopy data. The details of the routine are straightforward and therefore omitted; intuitively, it nudges intermediate samples so that the aggregate pixel statistics evolve toward the prescribed porosity budget while retaining the fine-grained texture learnt by the score network.

Figure 4 contrasts PDM with three baselines. The conditional diffusion model (*Cond*) fails to respect strict thresholds, drifting away from feasible regions even at high guidance scales. Adding a single projection after sampling (*Post⁺*) or correcting conditional outputs once (*Cond⁺*) cannot fully compensate for the accumulated error and often deteriorates image quality, as confirmed by their higher FID values. In stark contrast, PDM consistently attains exact porosity, achieving FID on par with or better than strong baselines. These results highlight the importance of integrating constraint handling directly into the sampling loop rather than treating it as an after-thought.

Qualitative inspection further confirms that our approach captures the heterogeneous void geometry observed in real specimens, a property missing from the other methods. We therefore conclude that PDM offers a principled and effective route to constraint-aware synthesis in low-data, shift-heavy material-design scenarios.
## 3D human motion *(dynamic motion and physical principles)* [sec:human-motion]

We next evaluate PDM on the challenging HumanML3D benchmark, which couples spatial realism with temporal coherence. The task is to generate motion clips that obey simple physics, namely ground-contact consistency and plausible body proportion, while following free-form text prompts.

PDM addresses these requirements implicitly: throughout the reverse process, intermediate poses are softly steered toward kinematically valid configurations and feasible ground heights. Because the projection step is lightweight and fully differentiable, it does not require an external simulator or reinforcement-learning-based controller as in prior work. Instead, feasibility is enforced on-the-fly, tightly interleaved with the denoising updates. This unified view eliminates the need for heuristic penalties or expensive rollouts.

Figure 5 shows representative clips. Whereas the conditional model occasionally produces foot sliding or floating artefacts, PDM maintains contact fidelity without sacrificing diversity (FID 0.71 vs 0.63). More importantly, PDM attains *zero* violations across all tested sequences—a feat not matched by any baseline that we are aware of. These findings underline the benefit of coupling diffusion dynamics with a generic, reusable feasibility operator, rather than resorting to bespoke physics engines or post-hoc repairs.
## Constrained trajectories (*nonconvex constraints*) [sec:constrained-trajectories]

Path planning provides a stringent test of PDM under highly non-convex geometric constraints. We adopt two standard 2-D maps dotted with obstacles unseen during training and ask the model to output a collision-free poly-line between random start–goal pairs.

Thanks to its iterative projection, PDM can satisfy these obstacles natively, requiring just a single sample per query. Unlike motion-planning diffusion (MPD) or post-processing schemes that depend on repeated rejection or large batches of rollouts, PDM converges to feasible solutions without exhaustive search. The mechanism is conceptually simple: at every denoising step, tentative waypoints are gently corrected so they remain in free space, all while the global path length objective is refined by the score network. The resulting trajectory is therefore both safe and nearly optimal in length.

Quantitative results (Table 6) confirm 100 % success on both maps, whereas MPD succeeds only in 53–77 % of trials even when aided by an additional optimisation pass (*Cond⁺*). Path lengths remain comparable across methods, indicating that PDM does not trade feasibility for sub-optimal detours.

In summary, PDM’s built-in feasibility alignment obviates heuristic pruning, dedicated collision checkers, or hard-coded gradient terms. This makes it readily applicable to a wide class of planning problems where the exact analytic form of the obstacle set may vary from task to task.
## Physics-informed motion *(ODEs and out-of-distribution constraints)* [sec:physics-informed-motion]

Finally, we show the applicability of PDM in generating video frames adhering to physical principles. In this task, the goal is to generate frames depicting an object accelerating due to gravity. The object’s position in a given frame is governed by

\\[\label{eq:ball-position}
\hspace{12pt}

    \begin{align}
        \mathbf{p}_{t} &= \mathbf{p}_{t-1} + \left(\mathbf{v}_t   + \left(0.5 \times \frac{\partial \bm{v}_{t}}{\partial t}\right)\right)
        \label{eq:pt}
    \end{align}


    \begin{align}
    \mathbf{v}_{t+1} &= \frac{\partial \bm{p}_{t}}{\partial t} + \frac{\partial \bm{v}_{t}}{\partial t},
    \label{eq:vt}
    \end{align}\\]

where \\(\mathbf{p}\\) is the object position, \\(\mathbf{v}\\) is the velocity, and \\(t\\) is the frame number. This positional information can be directly integrated into the constraint set of *PDM*, with constraint violations quantified by the pixel distance from their true position. In our experiment, the training data is based *solely* on earth’s gravity and we test the model to simulate gravitational forces from the moon and other planets, in addition to earth. Thus there are two challenges in this setting **(1) satifying ODEs** describing our physical principle and **(2) generalize to out-of-distribution constraints**.

Figure <a href="#tab:ode_table" data-reference-type="ref" data-reference="tab:ode_table">[tab:ode_table]</a> (left) shows randomly selected generated samples, with ground-truth images provided for reference. The subsequent rows display outputs from *PDM*, post-processing projection (*Post*), and conditional post-processing (*Cond\\(^+\\)*). For this setting, we used a state-of-the-art masked conditional video diffusion model, following `\citet{voleti2022mcvd}`{=latex}. Samples generated by conditional diffusion models are not directly shown in the figure, as the white object outline in the *Cond\\(^+\\)* frames shows where the *Cond* model originally positioned the object. Notice that, without constraint projections, the score-based generative model produce samples that align with the original data arbitrarily place the object within the frame (white ball outlines in the 3rd column). Post-processing repositions the object accurately but significantly reduces image quality. Similarly, *Cond\\(^+\\)* shows inaccuracies in the conditional model’s object positioning, as indicated by the white outline in the 4th column. These deviations from the desired constraints are quantitatively shown in Figure <a href="#fig:ode_violations" data-reference-type="ref" data-reference="fig:ode_violations">7</a> (light red bars), which depicts the proportion of samples adhering to the object’s behavior constraints across varying error tolerance levels. Notably, this approach fails to produce *any viable sample within a zero-tolerance error margin*. In contrast, PDM generates frames that exactly satisfy the positional constraints, with FID scores comparable to those of *Cond*. Using the model proposed by `\citet{song2020score}`{=latex} further narrows this gap (see §<a href="#appendix:algorithms" data-reference-type="ref" data-reference="appendix:algorithms">13</a>).

<figure id="fig:ode_violations">
<table>
<thead>
<tr>
<th style="text-align: center;">t</th>
<th colspan="4" style="text-align: center;"><span><strong>Earth (in distribution)</strong></span></th>
<th style="text-align: center;"> </th>
<th colspan="4" style="text-align: center;"><span><strong>Moon (out of distribution)</strong></span></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><span>2-5</span></td>
<td style="text-align: center;"><span><em>Ground</em></span></td>
<td style="text-align: center;"><span><em><span style="color: MidnightBlue">PDM</span></em></span></td>
<td style="text-align: center;"><span><em>Post<span class="math inline"><sup>+</sup></span></em></span></td>
<td style="text-align: center;"><span><em><span style="color: Mahogany">Cond<span class="math inline"><sup>+</sup></span></span></em></span></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span><em>Ground</em></span></td>
<td style="text-align: center;"><span><em><span style="color: MidnightBlue">PDM</span></em></span></td>
<td style="text-align: center;"><span><em>Post<span class="math inline"><sup>+</sup></span></em></span></td>
<td style="text-align: center;"><span><em><span style="color: Mahogany">Cond<span class="math inline"><sup>+</sup></span></span></em></span></td>
</tr>
<tr>
<td style="text-align: center;"><span>1</span></td>
<td style="text-align: center;"><img src="./figures/gt-earth-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-earth-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-earth-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-earth-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-moon-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-moon-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-moon-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-moon-1.png"" style="width:11.0%" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><span>3</span></td>
<td style="text-align: center;"><img src="./figures/gt-earth-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-earth-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-earth-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-earth-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-moon-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-moon-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-moon-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-moon-3.png"" style="width:11.0%" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><span>5</span></td>
<td style="text-align: center;"><img src="./figures/gt-earth-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-earth-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-earth-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-earth-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-moon-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-moon-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-moon-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-moon-5.png"" style="width:11.0%" alt="image" /></td>
</tr>
</tbody>
</table>
<table>
<thead>
<tr>
<th style="text-align: center;"></th>
<th style="text-align: center;"><span><em><span style="color: MidnightBlue">PDM</span></em></span></th>
<th style="text-align: left;"><span><strong><span><em>Post<span class="math inline"><sup>+</sup></span></em></span></strong></span></th>
<th style="text-align: center;"><span><em><span style="color: Mahogany">Cond</span></em></span></th>
<th style="text-align: center;"><span><strong><span><em><span style="color: Mahogany">Cond<span class="math inline"><sup>+</sup></span></span></em></span></strong></span></th>
<th style="text-align: center;"></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><span><strong><span>FID</span></strong></span></td>
<td style="text-align: center;"><span>26.5 <span class="math inline">±</span> 1.7</span></td>
<td style="text-align: left;"><span>52.5 <span class="math inline">±</span> 1.0</span></td>
<td style="text-align: center;"><span><strong>22.5 <span class="math inline">±</span> 0.1</strong></span></td>
<td style="text-align: center;"><span>53.0 <span class="math inline">±</span> 0.3</span></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>
<img src="./figures/physics-positional-violation-fill-alt.png"" />
<figcaption>Conditional diffusion model (<span><em><span style="color: Mahogany">Cond</span></em></span>): Frequency of constraint satisfaction (y-axis) given an error tolerance (x-axis) over 100 runs.</figcaption>
</figure>

Next, Figure <a href="#tab:ode_table" data-reference-type="ref" data-reference="tab:ode_table">[tab:ode_table]</a> (right) shows the behavior of the models in settings where the training data does not include any feasible data points. Here we adjust the governing equation <a href="#eq:ball-position" data-reference-type="eqref" data-reference="eq:ball-position">[eq:ball-position]</a> to reflect the moon’s gravitational pull. Remarkably, PDM not only synthesizes high-quality images but also ensures no constraint violations (0-tolerance). This stands in contrasts to other methods, that show increased constraint violations in out-of-distribution contexts, as shown by the dark red bars in Figure <a href="#fig:ode_violations" data-reference-type="ref" data-reference="fig:ode_violations">7</a>. *PDM can be adapted to handle complex governing equations using ODEs and can be guarantee satisfaction of out-of-distribution constraints with no decrease in sample quality.*

# Discussion and limitations [sec:discussion-limitations]

In many scientific and engineering domains and safety-critical applications, constraint satisfaction guarantees are a critical requirement. It is however important to acknowledge the existence of an inherent trade-off, particularly in computational overhead. In applications where inference time is a critical factor, it may be practical to adjust the time step \\(t\\) at which iterative projections begin, which guides a trade-off between the FID score associated with the starting point of iterative projections and the computational cost of projecting throughout the remaining iterations (§<a href="#appendix:computational-cost" data-reference-type="ref" data-reference="appendix:computational-cost">15</a>). Other avenues to improve efficiency also exists, from the adoption of specialized solvers within the application domain of interest to the adoption of warm-start strategies for iterative solvers. The latter, in particular, relies exploiting solutions computed in previous iterations of the sampling step and was found to be a practical strategy to substantially decrease the projections overhead.

We also note the absence of constraints in the forward process. As illustrated empirically, it is unnecessary for the training data to contain any feasible points. We hold that this not only applies to the final distribution but to the interim distributions as well. Furthermore, by projecting perturbed samples, the cost of the projection results in divergence from the distribution that is being learned. Hence, we conjecture that incorporating constraints into the forward process will not only increase computational cost of model training but also decrease the FID scores of the generated samples.

Finally, while this study provides a framework for imposing constraints on diffusion models, the representation of complex constraints for multi-task large scale models remains an open challenge. This paper motivates future work for adapting optimization techniques to such settings, where constraints ensuring accuracy in task completion and safety in model outputs bear transformative potential to broaden the application of generative models in many scientific and engineering fields.

# Conclusions

This paper was motivated by a significant challenge in the application of diffusion models in contexts requiring strict adherence to constraints and physical principles. It presented Projected Diffusion Models (PDM), an approach that recasts the score-based diffusion sampling process as a constrained optimization process that can be solved via the application of repeated projections. Experiments in domains ranging from physical-informed motion for video generation governed by ordinary differentiable equations, trajectory optimization in motion planning, and adherence to morphometric properties in generative material science processes illustrate the ability of PDM to generate content of high-fidelity that also adheres to complex non-convex constraints as well as physical principles.

# Acknowledgments

This research is partially supported by NSF grants 2334936, 2334448, and NSF CAREER Award 2401285. Fioretto is also supported by an Amazon Research Award and a Google Research Scholar Award. The authors acknowledge Research Computing at the University of Virginia for providing computational resources that have contributed to the results reported within this paper. The views and conclusions of this work are those of the authors only.

## Authors Contributions [authors-contributions]

JC and FF formulated the research question, designed the methodology, developed the theoretical analysis, and wrote the manuscript. Moreover, JC contributed to developing the code and performed the experimental analysis. SB acquired the data for the micro-structure experiment, formulated the desired properties for such experiment, and participated in the interpretation of the results.

# References [references]

<div class="thebibliography" markdown="1">

Namrata Anand and Tudor Achim Protein structure and sequence generation with equivariant denoising diffusion probabilistic models *arXiv preprint arXiv:2205.15019*, 2022. **Abstract:** Proteins are macromolecules that mediate a significant fraction of the cellular processes that underlie life. An important task in bioengineering is designing proteins with specific 3D structures and chemical properties which enable targeted functions. To this end, we introduce a generative model of both protein structure and sequence that can operate at significantly larger scales than previous molecular generative modeling approaches. The model is learned entirely from experimental data and conditions its generation on a compact specification of protein topology to produce a full-atom backbone configuration as well as sequence and side-chain predictions. We demonstrate the quality of the model via qualitative and quantitative analysis of its samples. Videos of sampling trajectories are available at https://nanand2.github.io/proteins . (@anand2022protein)

Chentao Cao, Zhuo-Xu Cui, Yue Wang, Shaonan Liu, Taijin Chen, Hairong Zheng, Dong Liang, and Yanjie Zhu High-frequency space diffusion model for accelerated mri *IEEE Transactions on Medical Imaging*, 2024. **Abstract:** Diffusion models with continuous stochastic differential equations (SDEs) have shown superior performances in image generation. It can serve as a deep generative prior to solving the inverse problem in magnetic resonance (MR) reconstruction. However, low-frequency regions of k -space data are typically fully sampled in fast MR imaging, while existing diffusion models are performed throughout the entire image or k -space, inevitably introducing uncertainty in the reconstruction of low-frequency regions. Additionally, existing diffusion models often demand substantial iterations to converge, resulting in time-consuming reconstructions. To address these challenges, we propose a novel SDE tailored specifically for MR reconstruction with the diffusion process in high-frequency space (referred to as HFS-SDE). This approach ensures determinism in the fully sampled low-frequency regions and accelerates the sampling procedure of reverse diffusion. Experiments conducted on the publicly available fastMRI dataset demonstrate that the proposed HFS-SDE method outperforms traditional parallel imaging methods, supervised deep learning, and existing diffusion models in terms of reconstruction accuracy and stability. The fast convergence properties are also confirmed through theoretical and experimental validation. Our code and weights are available at https://github.com/Aboriginer/HFS-SDE. (@cao2024high)

Joao Carvalho, An T Le, Mark Baierl, Dorothea Koert, and Jan Peters Motion planning diffusion: Learning and planning of robot motions with diffusion models In *2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 1916–1923. IEEE, 2023. **Abstract:** Learning priors on trajectory distributions can help accelerate robot motion planning optimization. Given previously successful plans, learning trajectory generative models as priors for a new planning problem is highly desirable. Prior works propose several ways on utilizing this prior to bootstrapping the motion planning problem. Either sampling the prior for initializations or using the prior distribution in a maximum-a-posterior formulation for trajectory optimization. In this work, we propose learning diffusion models as priors. We then can sample directly from the posterior trajectory distribution conditioned on task goals, by leveraging the inverse denoising process of diffusion models. Furthermore, diffusion has been recently shown to effectively encode data multi-modality in high-dimensional settings, which is particularly well-suited for large trajectory dataset. To demonstrate our method efficacy, we compare our proposed method - Motion Planning Diffusion - against several baselines in simulated planar robot and 7-dof robot arm manipulator environments. To assess the generalization capabilities of our method, we test it in environments with previously unseen obstacles. Our experiments show that diffusion models are strong priors to encode high-dimensional trajectory distributions of robot motions. https://sites.google.com/view/mp-diffusion (@carvalho2023motion)

Joseph B. Choi, Phong C. H. Nguyen, Oishik Sen, H. S. Udaykumar, and Stephen Baek Artificial intelligence approaches for energetic materials by design: State of the art, challenges, and future directions *Propellants, Explosives, Pyrotechnics*, 2023. . URL <https://onlinelibrary.wiley.com/doi/full/10.1002/prep.202200276>. **Abstract:** Artificial intelligence (AI) is rapidly emerging as an enabling tool for solving various complex materials design problems. This paper aims to review recent advances in AI-driven materials-by-design and their applications to energetic materials (EM). Trained with data from numerical simulations and/or physical experiments, AI models can assimilate trends and patterns within the design parameter space, identify optimal material designs (micro-morphologies, combinations of materials in composites, etc.), and point to designs with superior/targeted property and performance metrics. We review approaches focusing on such capabilities with respect to the three main stages of materials-by-design, namely representation learning of microstructure morphology (i.e., shape descriptors), structure-property-performance (S-P-P) linkage estimation, and optimization/design exploration. We provide a perspective view of these methods in terms of their potential, practicality, and efficacy towards the realization of materials-by-design. Specifically, methods in the literature are evaluated in terms of their capacity to learn from a small/limited number of data, computational complexity, generalizability/scalability to other material species and operating conditions, interpretability of the model predictions, and the burden of supervision/data annotation. Finally, we suggest a few promising future research directions for EM materials-by-design, such as meta-learning, active learning, Bayesian learning, and semi-/weakly-supervised learning, to bridge the gap between machine learning research and EM research. (@Choi2023AIEM)

Sehyun Chun, Sidhartha Roy, Yen Thi Nguyen, Joseph B Choi, HS Udaykumar, and Stephen S Baek Deep learning for synthetic microstructure generation in a materials-by-design framework for heterogeneous energetic materials *Scientific reports*, 10 (1): 13307, 2020. **Abstract:** The sensitivity of heterogeneous energetic (HE) materials (propellants, explosives, and pyrotechnics) is critically dependent on their microstructure. Initiation of chemical reactions occurs at hot spots due to energy localization at sites of porosities and other defects. Emerging multi-scale predictive models of HE response to loads account for the physics at the meso-scale, i.e. at the scale of statistically representative clusters of particles and other features in the microstructure. Meso-scale physics is infused in machine-learned closure models informed by resolved meso-scale simulations. Since microstructures are stochastic, ensembles of meso-scale simulations are required to quantify hot spot ignition and growth and to develop models for microstructure-dependent energy deposition rates. We propose utilizing generative adversarial networks (GAN) to spawn ensembles of synthetic heterogeneous energetic material microstructures. The method generates qualitatively and quantitatively realistic microstructures by learning from images of HE microstructures. We show that the proposed GAN method also permits the generation of new morphologies, where the porosity distribution can be controlled and spatially manipulated. Such control paves the way for the design of novel microstructures to engineer HE materials for targeted performance in a materials-by-design framework. (@chun2020deep)

Hyungjin Chung and Jong Chul Ye Score-based diffusion models for accelerated mri *Medical image analysis*, 80: 102479, 2022. **Abstract:** Score-based diffusion models provide a powerful way to model images using the gradient of the data distribution. Leveraging the learned score function as a prior, here we introduce a way to sample data from a conditional distribution given the measurements, such that the model can be readily used for solving inverse problems in imaging, especially for accelerated MRI. In short, we train a continuous time-dependent score function with denoising score matching. Then, at the inference stage, we iterate between numerical SDE solver and data consistency projection step to achieve reconstruction. Our model requires magnitude images only for training, and yet is able to reconstruct complex-valued data, and even extends to parallel imaging. The proposed method is agnostic to sub-sampling patterns, and can be used with any sampling schemes. Also, due to its generative nature, our approach can quantify uncertainty, which is not possible with standard regression settings. On top of all the advantages, our method also has very strong performance, even beating the models trained with full supervision. With extensive experiments, we verify the superiority of our method in terms of quality and practicality. (@chung2022score)

Nic Fishman, Leo Klarner, Valentin De Bortoli, Emile Mathieu, and Michael Hutchinson Diffusion models for constrained domains *arXiv preprint arXiv:2304.05364*, 2023. **Abstract:** Denoising diffusion models are a novel class of generative algorithms that achieve state-of-the-art performance across a range of domains, including image generation and text-to-image tasks. Building on this success, diffusion models have recently been extended to the Riemannian manifold setting, broadening their applicability to a range of problems from the natural and engineering sciences. However, these Riemannian diffusion models are built on the assumption that their forward and backward processes are well-defined for all times, preventing them from being applied to an important set of tasks that consider manifolds defined via a set of inequality constraints. In this work, we introduce a principled framework to bridge this gap. We present two distinct noising processes based on (i) the logarithmic barrier metric and (ii) the reflected Brownian motion induced by the constraints. As existing diffusion model techniques cannot be applied in this setting, we derive new tools to define such models in our framework. We then demonstrate the practical utility of our methods on a number of synthetic and real-world tasks, including applications from robotics and protein design. (@fishman2023diffusion)

Nic Fishman, Leo Klarner, Emile Mathieu, Michael Hutchinson, and Valentin De Bortoli Metropolis sampling for constrained diffusion models *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Denoising diffusion models have recently emerged as the predominant paradigm for generative modelling on image domains. In addition, their extension to Riemannian manifolds has facilitated a range of applications across the natural sciences. While many of these problems stand to benefit from the ability to specify arbitrary, domain-informed constraints, this setting is not covered by the existing (Riemannian) diffusion model methodology. Recent work has attempted to address this issue by constructing novel noising processes based on the reflected Brownian motion and logarithmic barrier methods. However, the associated samplers are either computationally burdensome or only apply to convex subsets of Euclidean space. In this paper, we introduce an alternative, simple noising scheme based on Metropolis sampling that affords substantial gains in computational efficiency and empirical performance compared to the earlier samplers. Of independent interest, we prove that this new process corresponds to a valid discretisation of the reflected Brownian motion. We demonstrate the scalability and flexibility of our approach on a range of problem settings with convex and non-convex constraints, including applications from geospatial modelling, robotics and protein design. (@fishman2024metropolis)

Thomas Frerix, Matthias Nießner, and Daniel Cremers Homogeneous linear inequality constraints for neural network activations In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, pages 748–749, 2020. **Abstract:** We propose a method to impose homogeneous linear inequality constraints of the form Ax ≤ 0 on neural network activations. The proposed method allows a data-driven training approach to be combined with modeling prior knowledge about the task. One way to achieve this task is by means of a projection step at test time after unconstrained training. However, this is an expensive operation. By directly incorporating the constraints into the architecture, we can significantly speed-up inference at test time; for instance, our experiments show a speed-up of up to two orders of magnitude over a projection method. Our algorithm computes a suitable parameterization of the feasible set at initialization and uses standard variants of stochastic gradient descent to find solutions to the constrained network. Thus, the modeling constraints are always satisfied during training. Crucially, our approach avoids to solve an optimization problem at each training step or to manually trade-off data and constraint fidelity with additional hyperparameters. We consider constrained generative modeling as an important application domain and experimentally demonstrate the proposed method by constraining a variational autoencoder. (@frerix2020homogeneous)

Giorgio Giannone, Akash Srivastava, Ole Winther, and Faez Ahmed Aligning optimization trajectories with diffusion models for constrained design generation *arXiv preprint arXiv:2305.18470*, 2023. **Abstract:** Generative models have had a profound impact on vision and language, paving the way for a new era of multimodal generative applications. While these successes have inspired researchers to explore using generative models in science and engineering to accelerate the design process and reduce the reliance on iterative optimization, challenges remain. Specifically, engineering optimization methods based on physics still outperform generative models when dealing with constrained environments where data is scarce and precision is paramount. To address these challenges, we introduce Diffusion Optimization Models (DOM) and Trajectory Alignment (TA), a learning framework that demonstrates the efficacy of aligning the sampling trajectory of diffusion models with the optimization trajectory derived from traditional physics-based methods. This alignment ensures that the sampling process remains grounded in the underlying physical principles. Our method allows for generating feasible and high-performance designs in as few as two steps without the need for expensive preprocessing, external surrogate models, or additional labeled data. We apply our framework to structural topology optimization, a fundamental problem in mechanical design, evaluating its performance on in- and out-of-distribution configurations. Our results demonstrate that TA outperforms state-of-the-art deep generative models on in-distribution configurations and halves the inference computational cost. When coupled with a few steps of optimization, it also improves manufacturability for out-of-distribution conditions. By significantly improving performance and inference efficiency, DOM enables us to generate high-quality designs in just a few steps and guide them toward regions of high performance and manufacturability, paving the way for the widespread application of generative models in large-scale data-driven design. (@giannone2023aligning)

Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio Generative adversarial nets In *Advances in neural information processing systems*, pages 2672–2680, 2014. **Abstract:** We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to ½ everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples. (@goodfellow2014generative)

Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng Generating diverse and natural 3d human motions from text In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5152–5161, 2022. **Abstract:** Automated generation of 3D human motions from text is a challenging problem. The generated motions are expected to be sufficiently diverse to explore the text-grounded motion space, and more importantly, accurately depicting the content in prescribed text descriptions. Here we tackle this problem with a two-stage approach: text2length sampling and text2motion generation. Text2length involves sampling from the learned distribution function of motion lengths conditioned on the input text. This is followed by our text2motion module using temporal variational autoen-coder to synthesize a diverse set of human motions of the sampled lengths. Instead of directly engaging with pose sequences, we propose motion snippet code as our internal motion representation, which captures local semantic motion contexts and is empirically shown to facilitate the generation of plausible motions faithful to the input text. Moreover, a large-scale dataset of scripted 3D Human motions, HumanML3D, is constructed, consisting of 14,616 motion clips and 44,970 text descriptions. (@guo2022generating)

Jonathan Ho and Tim Salimans Classifier-free diffusion guidance *arXiv preprint arXiv:2207.12598*, 2022. **Abstract:** Classifier guidance is a recently introduced method to trade off mode coverage and sample fidelity in conditional diffusion models post training, in the same spirit as low temperature sampling or truncation in other types of generative models. Classifier guidance combines the score estimate of a diffusion model with the gradient of an image classifier and thereby requires training an image classifier separate from the diffusion model. It also raises the question of whether guidance can be performed without a classifier. We show that guidance can be indeed performed by a pure generative model without such a classifier: in what we call classifier-free guidance, we jointly train a conditional and an unconditional diffusion model, and we combine the resulting conditional and unconditional score estimates to attain a trade-off between sample quality and diversity similar to that obtained using classifier guidance. (@ho2022classifier)

Jonathan Ho, Ajay Jain, and Pieter Abbeel Denoising diffusion probabilistic models *Advances in neural information processing systems*, 33: 6840–6851, 2020. **Abstract:** We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at https://github.com/hojonathanho/diffusion (@ho2020denoising)

Emiel Hoogeboom, Vıctor Garcia Satorras, Clément Vignac, and Max Welling Equivariant diffusion for molecule generation in 3d In *International conference on machine learning*, pages 8867–8887. PMLR, 2022. **Abstract:** This work introduces a diffusion model for molecule generation in 3D that is equivariant to Euclidean transformations. Our E(3) Equivariant Diffusion Model (EDM) learns to denoise a diffusion process with an equivariant network that jointly operates on both continuous (atom coordinates) and categorical features (atom types). In addition, we provide a probabilistic analysis which admits likelihood computation of molecules using our model. Experimentally, the proposed method significantly outperforms previous 3D molecular generative methods regarding the quality of generated samples and efficiency at training time. (@hoogeboom2022equivariant)

Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine Planning with diffusion for flexible behavior synthesis *arXiv preprint arXiv:2205.09991*, 2022. **Abstract:** Model-based reinforcement learning methods often use learning only for the purpose of estimating an approximate dynamics model, offloading the rest of the decision-making work to classical trajectory optimizers. While conceptually simple, this combination has a number of empirical shortcomings, suggesting that learned models may not be well-suited to standard trajectory optimization. In this paper, we consider what it would look like to fold as much of the trajectory optimization pipeline as possible into the modeling problem, such that sampling from the model and planning with it become nearly identical. The core of our technical approach lies in a diffusion probabilistic model that plans by iteratively denoising trajectories. We show how classifier-guided sampling and image inpainting can be reinterpreted as coherent planning strategies, explore the unusual and useful properties of diffusion-based planning methods, and demonstrate the effectiveness of our framework in control settings that emphasize long-horizon decision-making and test-time flexibility. (@janner2022planning)

Guan-Horng Liu, Tianrong Chen, Evangelos Theodorou, and Molei Tao Mirror diffusion models for constrained and watermarked generation *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Modern successes of diffusion models in learning complex, high-dimensional data distributions are attributed, in part, to their capability to construct diffusion processes with analytic transition kernels and score functions. The tractability results in a simulation-free framework with stable regression losses, from which reversed, generative processes can be learned at scale. However, when data is confined to a constrained set as opposed to a standard Euclidean space, these desirable characteristics appear to be lost based on prior attempts. In this work, we propose Mirror Diffusion Models (MDM), a new class of diffusion models that generate data on convex constrained sets without losing any tractability. This is achieved by learning diffusion processes in a dual space constructed from a mirror map, which, crucially, is a standard Euclidean space. We derive efficient computation of mirror maps for popular constrained sets, such as simplices and $\\}ell_2$-balls, showing significantly improved performance of MDM over existing methods. For safety and privacy purposes, we also explore constrained sets as a new mechanism to embed invisible but quantitative information (i.e., watermarks) in generated data, for which MDM serves as a compelling approach. Our work brings new algorithmic opportunities for learning tractable diffusion on complex domains. (@liu2024mirror)

Aaron Lou and Stefano Ermon Reflected diffusion models In *International Conference on Machine Learning*, pages 22675–22701. PMLR, 2023. **Abstract:** A model based upon steady-state diffusion theory which describes the radial dependence of diffuse reflectance of light from tissues is developed. This model incorporates a photon dipole source in order to satisfy the tissue boundary conditions and is suitable for either refractive index matched or mismatched surfaces. The predictions of the model were compared with Monte Carlo simulations as well as experimental measurements made with tissue simulating phantoms. The model describes the reflectance data accurately to radial distances as small as 0.5 mm when compared to Monte Carlo simulations and agrees with experimental measurements to distances as small as 1 mm. A nonlinear least-squares fitting procedure has been used to determine the tissue optical properties from the radial reflectance data in both phantoms and tissues in vivo. The optical properties derived for the phantoms are within 5%–10% of those determined by other established techniques. The in vivo values are also consistent with those reported by other investigators. (@lou2023reflected)

François Mazé and Faez Ahmed Diffusion models beat gans on topology optimization In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), Washington, DC*, 2023. **Abstract:** Structural topology optimization, which aims to find the optimal physical structure that maximizes mechanical performance, is vital in engineering design applications in aerospace, mechanical, and civil engineering. Recently, generative adversarial networks (GANs) have emerged as a popular alternative to traditional iterative topology optimization methods. However, GANs can be challenging to train, have limited generalizability, and often neglect important performance objectives such as mechanical compliance and manufacturability. To address these issues, we propose a new architecture called TopoDiff that uses conditional diffusion models to perform performance-aware and manufacturability-aware topology optimization. Our method introduces a surrogate model-based guidance strategy that actively favors structures with low compliance and good manufacturability. Compared to a state-of-the-art conditional GAN, our approach reduces the average error on physical performance by a factor of eight and produces eleven times fewer infeasible samples. Our work demonstrates the potential of using diffusion models in topology optimization and suggests a general framework for solving engineering optimization problems using external performance with constraint-aware guidance. We provide access to our data, code, and trained models at the following link: https://decode.mit.edu/projects/topodiff/. (@maze2023diffusion)

Neal Parikh and Stephen Boyd Proximal algorithms *Foundations and Trends in Optimization*, 1 (3): 127–239, 2014. **Abstract:** This monograph is about a class of optimization algorithms called proximal algorithms. Much like Newton’s method is a standard tool for solving unconstrained smooth optimization problems of modest size, proximal algorithms can be viewed as an analogous tool for nonsmooth, constrained, large-scale, or distributed versions of these problems. They are very generally applicable, but are especially well-suited to problems of substantial recent interest involving large or high-dimensional datasets. Proximal methods sit at a higher level of abstraction than classical algorithms like Newton’s method: the base operation is evaluating the proximal operator of a function, which itself involves solving a small convex optimization problem. These subproblems, which generalize the problem of projecting a point onto a convex set, often admit closed-form solutions or can be solved very quickly with standard or simple specialized methods. Here, we discuss the many different interpretations of proximal operators and algorithms, describe their connections to many other topics in optimization and applied mathematics, survey some popular algorithms, and provide a large number of examples of proximal operators that commonly arise in practice. (@parikh2014proximal)

Thomas Power, Rana Soltani-Zarrin, Soshi Iba, and Dmitry Berenson Sampling constrained trajectories using composable diffusion models In *IROS 2023 Workshop on Differentiable Probabilistic Robotics: Emerging Perspectives on Robot Learning*, 2023. (@power2023sampling)

Gareth O. Roberts and Richard L. Tweedie Exponential convergence of langevin distributions and their discrete approximations *Bernoulli*, 2 (4): 341–363, 1996. **Abstract:** In this paper we consider a continuous-time method of approximating a given distribution using the Langevin diusion dL t dW t 1 2 r log (L t )dt.We ®nd conditions under this diusion converges exponentially quickly to or does not: in one dimension, these are essentially that for distributions with exponential tails of the form (x) / exp (ÿ\|x\| , 0\<\<1, exponential convergence occurs if and only if 1.We then consider conditions under which the discrete approximations to the diusion converge.We ®rst show that even when the diusion itself converges, naive discretizations need not do so.We then consider a ‘Metropolis-adjusted’ version of the algorithm, and ®nd conditions under which this also converges at an exponential rate: perhaps surprisingly, even the Metropolized version need not converge exponentially fast even if the diusion does.We brie¯y discuss a truncated form of the algorithm which, in practice, should avoid the diculties of the other forms. (@roberts1996exponential)

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al Photorealistic text-to-image diffusion models with deep language understanding *Advances in neural information processing systems*, 35: 36479–36494, 2022. **Abstract:** We present Imagen, a text-to-image diffusion model with an unprecedented degree of photorealism and a deep level of language understanding. Imagen builds on the power of large transformer language models in understanding text and hinges on the strength of diffusion models in high-fidelity image generation. Our key discovery is that generic large language models (e.g. T5), pretrained on text-only corpora, are surprisingly effective at encoding text for image synthesis: increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model. Imagen achieves a new state-of-the-art FID score of 7.27 on the COCO dataset, without ever training on COCO, and human raters find Imagen samples to be on par with the COCO data itself in image-text alignment. To assess text-to-image models in greater depth, we introduce DrawBench, a comprehensive and challenging benchmark for text-to-image models. With DrawBench, we compare Imagen with recent methods including VQ-GAN+CLIP, Latent Diffusion Models, and DALL-E 2, and find that human raters prefer Imagen over other models in side-by-side comparisons, both in terms of sample quality and image-text alignment. See https://imagen.research.google/ for an overview of the results. (@saharia2022photorealistic)

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli Deep unsupervised learning using nonequilibrium thermodynamics In *International conference on machine learning*, pages 2256–2265. PMLR, 2015. **Abstract:** A central problem in machine learning involves modeling complex data-sets using highly flexible families of probability distributions in which learning, sampling, inference, and evaluation are still analytically or computationally tractable. Here, we develop an approach that simultaneously achieves both flexibility and tractability. The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data. This approach allows us to rapidly learn, sample from, and evaluate probabilities in deep generative models with thousands of layers or time steps, as well as to compute conditional and posterior probabilities under the learned model. We additionally release an open source reference implementation of the algorithm. (@sohl2015deep)

Yang Song and Stefano Ermon Generative modeling by estimating gradients of the data distribution *Advances in neural information processing systems*, 32, 2019. **Abstract:** We introduce a new generative model where samples are produced via Langevin dynamics using gradients of the data distribution estimated with score matching. Because gradients can be ill-defined and hard to estimate when the data resides on low-dimensional manifolds, we perturb the data with different levels of Gaussian noise, and jointly estimate the corresponding scores, i.e., the vector fields of gradients of the perturbed data distribution for all noise levels. For sampling, we propose an annealed Langevin dynamics where we use gradients corresponding to gradually decreasing noise levels as the sampling process gets closer to the data manifold. Our framework allows flexible model architectures, requires no sampling during training or the use of adversarial methods, and provides a learning objective that can be used for principled model comparisons. Our models produce samples comparable to GANs on MNIST, CelebA and CIFAR-10 datasets, achieving a new state-of-the-art inception score of 8.87 on CIFAR-10. Additionally, we demonstrate that our models learn effective representations via image inpainting experiments. (@song2019generative)

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole Score-based generative modeling through stochastic differential equations *arXiv preprint arXiv:2011.13456*, 2020. **Abstract:** Creating noise from data is easy; creating data from noise is generative modeling. We present a stochastic differential equation (SDE) that smoothly transforms a complex data distribution to a known prior distribution by slowly injecting noise, and a corresponding reverse-time SDE that transforms the prior distribution back into the data distribution by slowly removing the noise. Crucially, the reverse-time SDE depends only on the time-dependent gradient field (\\}aka, score) of the perturbed data distribution. By leveraging advances in score-based generative modeling, we can accurately estimate these scores with neural networks, and use numerical SDE solvers to generate samples. We show that this framework encapsulates previous approaches in score-based generative modeling and diffusion probabilistic modeling, allowing for new sampling procedures and new modeling capabilities. In particular, we introduce a predictor-corrector framework to correct errors in the evolution of the discretized reverse-time SDE. We also derive an equivalent neural ODE that samples from the same distribution as the SDE, but additionally enables exact likelihood computation, and improved sampling efficiency. In addition, we provide a new way to solve inverse problems with score-based models, as demonstrated with experiments on class-conditional generation, image inpainting, and colorization. Combined with multiple architectural improvements, we achieve record-breaking performance for unconditional image generation on CIFAR-10 with an Inception score of 9.89 and FID of 2.20, a competitive likelihood of 2.99 bits/dim, and demonstrate high fidelity generation of 1024 x 1024 images for the first time from a score-based generative model. (@song2020score)

Vikram Voleti, Alexia Jolicoeur-Martineau, and Chris Pal Mcvd-masked conditional video diffusion for prediction, generation, and interpolation *Advances in Neural Information Processing Systems*, 35: 23371–23385, 2022. **Abstract:** Video prediction is a challenging task. The quality of video frames from current state-of-the-art (SOTA) generative models tends to be poor and generalization beyond the training data is difficult. Furthermore, existing prediction frameworks are typically not capable of simultaneously handling other video-related tasks such as unconditional generation or interpolation. In this work, we devise a general-purpose framework called Masked Conditional Video Diffusion (MCVD) for all of these video synthesis tasks using a probabilistic conditional score-based denoising diffusion model, conditioned on past and/or future frames. We train the model in a manner where we randomly and independently mask all the past frames or all the future frames. This novel but straightforward setup allows us to train a single model that is capable of executing a broad range of video tasks, specifically: future/past prediction – when only future/past frames are masked; unconditional generation – when both past and future frames are masked; and interpolation – when neither past nor future frames are masked. Our experiments show that this approach can generate high-quality frames for diverse types of videos. Our MCVD models are built from simple non-recurrent 2D-convolutional architectures, conditioning on blocks of frames and generating blocks of frames. We generate videos of arbitrary lengths autoregressively in a block-wise manner. Our approach yields SOTA results across standard video prediction and interpolation benchmarks, with computation times for training models measured in 1-12 days using $\\}le$ 4 GPUs. Project page: https://mask-cond-video-diffusion.github.io ; Code : https://github.com/voletiv/mcvd-pytorch (@voleti2022mcvd)

Andreas Wächter and Lorenz T Biegler On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming *Mathematical programming*, 106: 25–57, 2006. (@wachter2006implementation)

Tsun-Hsuan Wang, Juntian Zheng, Pingchuan Ma, Yilun Du, Byungchul Kim, Andrew Spielberg, Joshua Tenenbaum, Chuang Gan, and Daniela Rus Diffusebot: Breeding soft robots with physics-augmented generative diffusion models *arXiv preprint arXiv:2311.17053*, 2023. **Abstract:** Nature evolves creatures with a high complexity of morphological and behavioral intelligence, meanwhile computational methods lag in approaching that diversity and efficacy. Co-optimization of artificial creatures’ morphology and control in silico shows promise for applications in physical soft robotics and virtual character creation; such approaches, however, require developing new learning algorithms that can reason about function atop pure structure. In this paper, we present DiffuseBot, a physics-augmented diffusion model that generates soft robot morphologies capable of excelling in a wide spectrum of tasks. DiffuseBot bridges the gap between virtually generated content and physical utility by (i) augmenting the diffusion process with a physical dynamical simulation which provides a certificate of performance, and (ii) introducing a co-design procedure that jointly optimizes physical design and control by leveraging information about physical sensitivities from differentiable simulation. We showcase a range of simulated and fabricated robots along with their capabilities. Check our website at https://diffusebot.github.io/ (@wang2023diffusebot)

Max Welling and Yee W Teh Bayesian learning via stochastic gradient langevin dynamics In *Proceedings of the 28th international conference on machine learning (ICML-11)*, pages 681–688. Citeseer, 2011. **Abstract:** In this paper we propose a new framework for learning from large scale datasets based on iterative learning from small mini-batches. By adding the right amount of noise to a standard stochastic gradient optimization algorithm we show that the iterates will converge to samples from the true posterior distribution as we anneal the stepsize. This seamless transition between optimization and Bayesian posterior provides an inbuilt protection against overfitting. We also propose a practical method for Monte Carlo estimates of posterior statistics which monitors a sampling threshold and collects samples after it has been surpassed. We apply the method to three models: a mixture of Gaussians, logistic regression and ICA with natural gradients. (@welling2011bayesian)

Pan Xu, Jinghui Chen, Difan Zou, and Quanquan Gu Global convergence of langevin dynamics based algorithms for nonconvex optimization *Advances in Neural Information Processing Systems*, 31, 2018. **Abstract:** We present a unified framework to analyze the global convergence of Langevin dynamics based algorithms for nonconvex finite-sum optimization with $n$ component functions. At the core of our analysis is a direct analysis of the ergodicity of the numerical approximations to Langevin dynamics, which leads to faster convergence rates. Specifically, we show that gradient Langevin dynamics (GLD) and stochastic gradient Langevin dynamics (SGLD) converge to the almost minimizer within $\\}tilde O\\}big(nd/(\\}lambda\\}epsilon) \\}big)$ and $\\}tilde O\\}big(d\^7/(\\}lambda\^5\\}epsilon\^5) \\}big)$ stochastic gradient evaluations respectively, where $d$ is the problem dimension, and $\\}lambda$ is the spectral gap of the Markov chain generated by GLD. Both results improve upon the best known gradient complexity results (Raginsky et al., 2017). Furthermore, for the first time we prove the global convergence guarantee for variance reduced stochastic gradient Langevin dynamics (SVRG-LD) to the almost minimizer within $\\}tilde O\\}big(\\}sqrt{n}d\^5/(\\}lambda\^4\\}epsilon\^{5/2})\\}big)$ stochastic gradient evaluations, which outperforms the gradient complexities of GLD and SGLD in a wide regime. Our theoretical analyses shed some light on using Langevin dynamics based algorithms for nonconvex optimization with provable guarantees. (@xu2018global)

Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz Physdiff: Physics-guided human motion diffusion model In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 16010–16021, 2023. **Abstract:** Denoising diffusion models hold great promise for generating diverse and realistic human motions. However, existing motion diffusion models largely disregard the laws of physics in the diffusion process and often generate physically-implausible motions with pronounced artifacts such as floating, foot sliding, and ground penetration. This seriously impacts the quality of generated motions and limits their real-world application. To address this issue, we present a novel physics-guided motion diffusion model (PhysDiff), which incorporates physical constraints into the diffusion process. Specifically, we propose a physics-based motion projection module that uses motion imitation in a physics simulator to project the denoised motion of a diffusion step to a physically-plausible motion. The projected motion is further used in the next diffusion step to guide the denoising diffusion process. Intuitively, the use of physics in our model iteratively pulls the motion toward a physically-plausible space, which cannot be achieved by simple post-processing. Experiments on large-scale human motion datasets show that our approach achieves state-of-the-art motion quality and improves physical plausibility drastically (\>78% for all datasets). (@yuan2023physdiff)

Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu Motiondiffuse: Text-driven human motion generation with diffusion model 2022. **Abstract:** Human motion modeling is important for many modern graphics applications, which typically require professional skills. In order to remove the skill barriers for laymen, recent motion generation methods can directly generate human motions conditioned on natural languages. However, it remains challenging to achieve diverse and fine-grained motion generation with various text inputs. To address this problem, we propose MotionDiffuse, the first diffusion model-based text-driven motion generation framework, which demonstrates several desired properties over existing methods. 1) Probabilistic Mapping. Instead of a deterministic language-motion mapping, MotionDiffuse generates motions through a series of denoising steps in which variations are injected. 2) Realistic Synthesis. MotionDiffuse excels at modeling complicated data distribution and generating vivid motion sequences. 3) Multi-Level Manipulation. MotionDiffuse responds to fine-grained instructions on body parts, and arbitrary-length motion synthesis with time-varied text prompts. Our experiments show MotionDiffuse outperforms existing SoTA methods by convincing margins on text-driven motion generation and action-conditioned motion generation. A qualitative analysis further demonstrates MotionDiffuse’s controllability for comprehensive motion generation. Homepage: https://mingyuan-zhang.github.io/projects/MotionDiffuse.html (@zhang2022motiondiffuse)

Ziyuan Zhong, Davis Rempe, Danfei Xu, Yuxiao Chen, Sushant Veer, Tong Che, Baishakhi Ray, and Marco Pavone Guided conditional diffusion for controllable traffic simulation In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 3560–3566. IEEE, 2023. **Abstract:** Controllable and realistic traffic simulation is critical for developing and verifying autonomous vehicles. Typical heuristic-based traffic models offer flexible control to make vehicles follow specific trajectories and traffic rules. On the other hand, data-driven approaches generate realistic and human-like behaviors, improving transfer from simulated to real-world traffic. However, to the best of our knowledge, no traffic model offers both controllability and realism. In this work, we develop a conditional diffusion model for controllable traffic generation (CTG) that allows users to control desired properties of trajectories at test time (e.g., reach a goal or follow a speed limit) while maintaining realism and physical feasibility through enforced dynamics. The key technical idea is to leverage recent advances from diffusion modeling and differentiable logic to guide generated trajectories to meet rules defined using signal temporal logic (STL). We further extend guidance to multi-agent settings and enable interaction-based rules like collision avoidance. CTG is extensively evaluated on the nuScenes dataset for diverse and composite rules, demonstrating improvement over strong baselines in terms of the controllability-realism tradeoff. Demo videos can be found at https://aiasd.github.io/ctg.github.io (@zhong2023guided)

</div>

# Broader impacts [appendix:broader_impacts]

The development of Projected Diffusion Models (PDM) may significantly enhance the application of diffusion models in fields requiring strict adherence to specific constraints and physical principles. The proposed method enables the generation of high-fidelity content that not only resembles real-world data but also complies with complex constraints, including non-convex and physical-based specifications. PDM’s ability to handle diverse and challenging constraints in scientific and engineering domains, particularly in low data environments, may potentially lead to accelerating innovation and discovery in various fields.

# Expanded related work [app:related_work]

**Diffusion models with soft constraint conditioning.** Variations of conditional diffusion models `\cite{ho2022classifier}`{=latex} serve as useful tools for controlling task specific outputs from generative models. These methods have demonstrated the capacity capture properties of physical design `\cite{wang2023diffusebot}`{=latex}, positional awareness `\cite{carvalho2023motion}`{=latex}, and motion dynamics `\cite{yuan2023physdiff}`{=latex} through augmentation of these models. The properties imposed in these architectures can be viewed as soft constraints, with stochastic model outputs violating these loosely imposed boundaries.

**Post-processing optimization.** In settings where hard constraints are needed to provide meaningful samples, diffusion model outputs have been used as starting points for a constrained optimization algorithm. This has been explored in non-convex settings, where the starting point plays an important role in whether the optimization solver will converge to a feasible solution `\cite{power2023sampling}`{=latex}. Other approaches have augmented the diffusion model training objective to encourage the sampling process to emulate an optimization algorithm, framing the post-processing steps as an extension of the model `\cite{giannone2023aligning, maze2023diffusion}`{=latex}. However, an existing challenge in these approaches is the reliance on an easily expressible objective, making these approaches effective in a limited set of problems (such as the constrained trajectory experiment) while not applicable for the majority of generative tasks.

**Hard constraints for generative models.** `\citet{frerix2020homogeneous}`{=latex} proposed an approach for implementing hard constraints on the outputs of autoencoders. This was achieved through scaling the generated outputs in such a way that feasibility was enforced, but the approach is to limited simple linear constraints. `\cite{liu2024mirror}`{=latex} proposed an approach to imposing constraints using “mirror mappings” with applicability exclusively to common, convex constraint sets. Due to the complexity of the constraints imposed in this paper, neither of these methods were applicable to the constraint sets explored in any of the experiments. Alternatively, work by `\citeauthor{fishman2023diffusion}`{=latex} \[`\citeyear{fishman2023diffusion, fishman2024metropolis}`{=latex}\] broadens the classes of constraints that can be represented but fails to demonstrate the applicability of their approach to a empirical settings similar to ours, utilizing an MLP architecture for trivial predictive tasks with constraints sets that can be represented by convex polytopes. We contrast such approaches to our work, noting that this prior work is limited to constraint sets that can be approximated by simple neighborhoods, such as an L2-ball, simplex, or polytope, whereas PDM can handle constraint sets of arbitrary complexity.

**Sampling process augmentation.** Motivated by the compounding of numerical error throughout the reverse diffusion process, prior work has proposed inference time operations to bound the pixel values of an image dynamically while sampling `\cite{lou2023reflected, saharia2022photorealistic}`{=latex}. Proposed methodologies have either applied reflections or simple clipping operations during the sampling process, preventing the generated image from significantly deviating from the \[0,255\] pixel space. Such approaches augment the sampling process in a way that mirrors our work, but these methods are solely applicable to mitigating sample drift and do not intersect our work in general constraint satisfaction.

# Experimental settings [appendix:experimental-settings]

In the following section, further details are provided as to the implementations of the experimental settings used in this paper.

## Constrained materials [appendix:settings-cm]

<figure id="fig:porosity-images-extended">
<table>
<thead>
<tr>
<th style="text-align: center;"><span>Ground</span></th>
<th style="text-align: center;"><span>P(%)  </span></th>
<th colspan="4" style="text-align: center;"><span><strong>Generative Methods</strong></span></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><span>3-6</span></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span><em><span style="color: MidnightBlue">PDM</span></em></span></td>
<td style="text-align: center;"><span><em><span style="color: Mahogany">Cond</span></em></span></td>
<td style="text-align: center;"><span><em>Post<span class="math inline"><sup>+</sup></span></em></span></td>
<td style="text-align: center;"><span><em>Cond<span class="math inline"><sup>+</sup></span></em></span></td>
</tr>
<tr>
<td style="text-align: center;"><img src="./figures/gt-porosity-5.png"" alt="image" /></td>
<td style="text-align: center;"><span>10</span></td>
<td style="text-align: center;"><img src="./figures/gp-porosity-0.11-0.09-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/co-porosity-0.1-0.08-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-porosity-0.11-0.09-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-porosity-0.1-0.08-t.png"" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><img src="./figures/gt-porosity-2.png"" alt="image" /></td>
<td style="text-align: center;"><span>20</span></td>
<td style="text-align: center;"><img src="./figures/gp-porosity-0.21-0.19-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/co-porosity-0.2-0.18_out.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-porosity-0.21-0.19_out.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-porosity-0.2-0.18_out.png"" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><img src="./figures/gt-porosity-3.png"" alt="image" /></td>
<td style="text-align: center;"><span>30</span></td>
<td style="text-align: center;"><img src="./figures/gp-porosity-0.31-0.29-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/co-porosity-0.3-0.28-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-porosity-0.31-0.29-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-porosity-0.3-0.28-t.png"" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><img src="./figures/gt-porosity-2.png"" alt="image" /></td>
<td style="text-align: center;"><span>40</span></td>
<td style="text-align: center;"><img src="./figures/gp-porosity-0.41-0.39-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/co-porosity-0.4-0.38_out.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-porosity-0.41-0.39_out.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-porosity-0.4-0.38_out.png"" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><img src="./figures/gt-porosity-2.png"" alt="image" /></td>
<td style="text-align: center;"><span>50</span></td>
<td style="text-align: center;"><img src="./figures/gp-porosity-0.5-0.48-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/co-porosity-0.5-0.48-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-porosity-0.5-0.48-t.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-porosity-0.5-0.48-t.png"" alt="image" /></td>
</tr>
</tbody>
</table>
<figcaption>Porosity constrained microstructure visualization at varying of the imposed porosity constraint amounts (expanded from Figure <a href="#fig:porosity-images" data-reference-type="ref" data-reference="fig:porosity-images">4</a>).</figcaption>
</figure>

Microstructures are pivotal in determining material properties. Current practice relies on physics-based simulations conducted upon imaged microstructures to quantify intricate structure-property linkages `\cite{Choi2023AIEM}`{=latex}. However, acquiring real material microstructure images is both costly and time-consuming, lacking control over attributes like porosity, crystal sizes, and volume fraction, thus necessitating “cut-and-try” experiments. Hence, the capability to generate realistic synthetic material microstructures with controlled morphological parameters can significantly expedite the discovery of structure-property linkages.

Previous work has shown that conditional generative adversarial networks (GAN) `\cite{goodfellow2014generative}`{=latex} can be used for this end `\cite{chun2020deep}`{=latex}, but these studies have been unable to impose verifiable constraints on the satisfaction of these desired properties. To provide a conditional baseline, we implement a conditional DDPM modeled after the conditional GAN used by `\citet{chun2020deep}`{=latex} with porosity measurements used to condition the sampling.

#### Projections.

The porosity of an image is represented by the number of pixels in the image which are classified as damaged regions of the microstructure. Provided that the image pixel intensities are scaled to \[-1, 1\], a threshold is set at zero, with pixel intensities below this threshold being classified as damage regions. To project, we implement a top-k algorithm that leaves the lowest and highest intensity pixels unchanged, while adjusting the pixels nearest to the threshold such that the total number of pixels below the threshold precisely satisfies the constraint.

#### Conditioning.

The conditional baseline is conditioned on the porosity values of the training samples. The implementation of this model is as described by `\citeauthor{ho2022classifier}`{=latex}.

#### Original training data.

We include samples from the original training data to visually illustrate how closely our results perform compared to the real images. As the specific porosities we tested on are not adhered to in the dataset, we illustrate this here as opposed to in the body of the text.

We observe that only the Conditional model and PDM synthesize images that visually adhere to the distribution, while post-processing methods do not provide adequate results for this complex setting.

## 3D human motion [appendix:settings-hm]

#### Projections.

The penetration and floatation constraints can be handled by ensuring that the lowest point on the z-axis is equal to the floor height. Additionally, to control the realism of the generated figures, we impose equality constraints on the size of various body parts, including the lengths of the torso and appendages. These constraints can be implemented directly through projection operators.

#### Conditioning.

The model is directly conditioned on text captions from the HumanML3D dataset. The implementation is as described in `\cite{zhang2022motiondiffuse}`{=latex}.

## Constrained trajectories [appendix:settings-ct]

#### Projections.

For this experiment, we represent constraints such that the predicted path avoids intersecting the obstacles present in the topography. These are parameterized to a non-convex interior point method solver. For circular obstacles, this can be represented by a minimum distance requirement, the circle radius, imposed on the nearest point to the center falling on a line between \\(p_n\\) and \\(p_{n+1}\\). These constraints are imposed for all line segments. We adapt a similar approach for non-circular obstacles by composing these of multiple circular constraints, hence, avoiding over-constraining the problem. More customized constraints could be implement to better represent the feasible region, likely resulting in shorter path lengths, but these were not explored for this paper.

#### Conditioning.

The positioning of the obstacles in the topography are passed into the model as a vector when conditioning the model for sampling. Further details can be found the work presented by `\citeauthor{carvalho2023motion}`{=latex}, from which this baseline was directly adapted.

<figure id="tab:ode_table_extended">
<table>
<thead>
<tr>
<th style="text-align: center;">t</th>
<th colspan="4" style="text-align: center;"><span><strong>Earth (in distribution)</strong></span></th>
<th style="text-align: center;"> </th>
<th colspan="4" style="text-align: center;"><span><strong>Moon (out of distribution)</strong></span></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><span>2-5</span></td>
<td style="text-align: center;"><span><em>Ground</em></span></td>
<td style="text-align: center;"><span><em><span style="color: MidnightBlue">PDM</span></em></span></td>
<td style="text-align: center;"><span><em>Post<span class="math inline"><sup>+</sup></span></em></span></td>
<td style="text-align: center;"><span><em><span style="color: Mahogany">Cond<span class="math inline"><sup>+</sup></span></span></em></span></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span><em>Ground</em></span></td>
<td style="text-align: center;"><span><em><span style="color: MidnightBlue">PDM</span></em></span></td>
<td style="text-align: center;"><span><em>Post<span class="math inline"><sup>+</sup></span></em></span></td>
<td style="text-align: center;"><span><em><span style="color: Mahogany">Cond<span class="math inline"><sup>+</sup></span></span></em></span></td>
</tr>
<tr>
<td style="text-align: center;"><span>1</span></td>
<td style="text-align: center;"><img src="./figures/gt-earth-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-earth-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-earth-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-earth-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-moon-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-moon-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-moon-1.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-moon-1.png"" style="width:11.0%" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><span>2</span></td>
<td style="text-align: center;"><img src="./figures/gt-earth-2.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-earth-2.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-earth-2.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-earth-2.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-moon-2.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-moon-2.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-moon-2.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-moon-2.png"" style="width:11.0%" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><span>3</span></td>
<td style="text-align: center;"><img src="./figures/gt-earth-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-earth-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-earth-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-earth-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-moon-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-moon-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-moon-3.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-moon-3.png"" style="width:11.0%" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><span>4</span></td>
<td style="text-align: center;"><img src="./figures/gt-earth-4.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-earth-4.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-earth-4.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-earth-4.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-moon-4.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-moon-4.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-moon-4.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-moon-4.png"" style="width:11.0%" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"><span>5</span></td>
<td style="text-align: center;"><img src="./figures/gt-earth-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-earth-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-earth-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-earth-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-moon-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gp-moon-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/pp-moon-5.png"" style="width:11.0%" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/cop-moon-5.png"" style="width:11.0%" alt="image" /></td>
</tr>
</tbody>
</table>
<figcaption>Sequential stages of the physics-informed models for in-distribution (Earth) and out-of-distribution (Moon) constraint imposition (expanded from Figure <a href="#tab:ode_table" data-reference-type="ref" data-reference="tab:ode_table">[tab:ode_table]</a>).</figcaption>
</figure>

## Physics-informed motion [appendix:settings-pm]

The dataset is generated with object starting points sampled uniformly in the interval \[0, 63\]. For each data point, six frames are included with the position changing as defined in Equation <a href="#eq:ball-position" data-reference-type="ref" data-reference="eq:ball-position">[eq:ball-position]</a> and the initial velocity \\(\mathbf{v}_0 = 0\\). Pixel values are scaled to \[-1, 1\]. The diffusion models are trained on 1000 points with a 90/10 train/test split.

#### Projections.

Projecting onto positional constraints requires a two-step process. First, the current position of the object is identified and all the pixels that make up the object are set to the highest pixel intensity (white), removing the object from the original position. The set of pixel indices representing the original object structure are stored for the subsequent step. Next, the object is moved to the correct position, as computed by the constraints, as each pixel from the original structure is placed onto the center point of the true position. Hence, when the frame is feasible prior to the projection, the image is returned unchanged, which is consistent with the definition of a projection.

#### Conditioning.

For this setting, the conditional video diffusion model takes two ground truth frames as inputs, from which it infers the trajectory of the object and the starting position. The model architecture is otherwise as specified by `\citeauthor{voleti2022mcvd}`{=latex}.

# PDM for score-based generative modeling through stochastic differential equations [appendix:algorithms]

## Algorithms

While the majority of our analysis focused on the developing these techniques to the sampling architecture proposed for Noise Conditioned Score Networks `\cite{song2019generative}`{=latex}, this approach can directly be adapted to the diffusion model variant Score-Based Generative Modeling with Stochastic Differential Equations proposed by `\citeauthor{song2020score}`{=latex} Although our observations suggested that optimizing across a continuum of distributions resulted in less stability in diverse experimental settings, we find that this method is still effective in producing high-quality constrained samples in others.

We included an updated version of Algorithm <a href="#alg:pgd_annealed_ld" data-reference-type="ref" data-reference="alg:pgd_annealed_ld">3</a> adapted to these architectures.

ALGORITHM BLOCK (caption below)

<figure id="alg:pgd_sde">
<p><span> <span class="math inline"><strong>x</strong><sub><em>N</em></sub><sup>0</sup> ∼ 𝒩(<strong>0</strong>, <em>σ</em><sub>max</sub><sup>2</sup><strong>I</strong>)</span><br />
<span class="math inline"><strong>return </strong>x<sub>0</sub><sup>0</sup></span> </span></p>
<figcaption>PDM Corrector Algorithm</figcaption>
</figure>

We note that a primary discrepancy between this algorithm and the one presented in Section <a href="#subsec:guided-projections" data-reference-type="ref" data-reference="subsec:guided-projections">4.2</a> is the difference in \\(\gamma\\). As the step size is not strictly decreasing, the guidance effect provided by PDM is impacted as Corollary <a href="#proof:corollary-1" data-reference-type="ref" data-reference="proof:corollary-1">3</a> does not hold for this approach. Hence, we do not focus on this architecture for our primary analysis, instead providing supplementary results in the subsequent section.

<figure id="fig:sde-earth">
<table>
<tbody>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>Frame 1</strong></td>
<td style="text-align: center;"><strong>Frame 2</strong></td>
<td style="text-align: center;"><strong>Frame 3</strong></td>
<td style="text-align: center;"><strong>Frame 4</strong></td>
<td style="text-align: center;"><strong>Frame 5</strong></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-earth-1.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gt-earth-2.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gt-earth-3.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gt-earth-4.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gt-earth-5.png"" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/sde-earth-1.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/sde-earth-2.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/sde-earth-3.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/sde-earth-4.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/sde-earth-5.png"" alt="image" /></td>
</tr>
</tbody>
</table>
<figcaption>In distribution sampling for physics-informed model via Score-Based Generative Modeling with SDEs.</figcaption>
</figure>

<figure id="fig:sde-moon">
<table>
<tbody>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>Frame 1</strong></td>
<td style="text-align: center;"><strong>Frame 2</strong></td>
<td style="text-align: center;"><strong>Frame 3</strong></td>
<td style="text-align: center;"><strong>Frame 4</strong></td>
<td style="text-align: center;"><strong>Frame 5</strong></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/gt-moon-1.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gt-moon-2.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gt-moon-3.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gt-moon-4.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/gt-moon-5.png"" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/sde-moon-1.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/sde-moon-2.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/sde-moon-3.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/sde-moon-4.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/sde-moon-5.png"" alt="image" /></td>
</tr>
</tbody>
</table>
<figcaption>Out of distribution sampling for physics-informed model via Score-Based Generative Modeling with SDEs.</figcaption>
</figure>

## Results

We provide additional results using the Score-Based Generative Modeling with Stochastic Differential Equations. This model produced highly performative results for the Physics-informed Motion experiment, with visualisations included in Figures <a href="#fig:sde-earth" data-reference-type="ref" data-reference="fig:sde-earth">11</a> and <a href="#fig:sde-moon" data-reference-type="ref" data-reference="fig:sde-moon">12</a>. This model averages an impressive inception score of **24.2** on this experiment, slightly outperforming the PDM implementation for Noise Conditioned Score Networks. Furthermore, it is equally capable in generalizing to constraints that were not present in the training distribution.

# Additional results

## Constrained materials morphometric parameter distributions [appendix:morphometric-distributions]

<figure id="fig:enter-label">
<img src="./figures/morphometry_heuristics.png"" />
<figcaption>Distributions of the morphometric parameters, comparing the ground truth to <span><em><span style="color: MidnightBlue">PDM</span></em></span> and <span><em><span style="color: Mahogany">Cond</span></em></span> models using heuristic-based analysis.</figcaption>
</figure>

When analyzing both real and synthetic materials, heuristic-guided metrics are often employed to extract information about microstrucutres present in the material. When analyzing the quality of synthetic samples, the extracted data can then be used to assess how well the crystals and voids in the microstructure adhere to the training data, providing an additional qualitative metrics for analysis. To augment the metrics displayed within the body of the paper, we include here the distribution of three metrics describing these microstructures, mirroring those used by `\citeauthor{chun2020deep}`{=latex}.

We observe that the constraint imposition present in PDM improves the general adherence of the results to the ground truth microstructures. This suggests that the *Cond* model tends to generate to certain microstructures at a frequency that is not reflected in the training data. By imposing various porosity constraints, PDM is able to generate a more representative set of microstructures in the sampling process.

## 3D human motion [appendix:3dhuman_motion]

We highlight that unlike the approach proposed by `\cite{yuan2023physdiff}`{=latex}, our approach guarantees the generated motion does not violate the penetrate and float constraints. The results are tabulated in Table <a href="#fig:motion-table" data-reference-type="ref" data-reference="fig:motion-table">1</a> (left) and report the violations in terms of measured distance the figure is either below (penetrate) or above (float) the floor. For comparison, we include the projection schedules utilized by PhysDiff which report the best results to show that even in these cases the model exhibits error.

<div id="fig:motion-table" markdown="1">

| Method | **FID** | **Penetrate** | **Float** |
|:---|:--:|---:|---:|
| PhysDiff `\cite{yuan2023physdiff}`{=latex} (Start 3, End 1) | 0.51 | 0.918 | 3.173 |
| PhysDiff `\cite{yuan2023physdiff}`{=latex} (End 4, Space 1) | **0.43** | 0.998 | 2.601 |
| *PDM* | 0.71 | **0.00** | **0.00** |

PDM performance compared to (best) PhysDiff results on HumanML3D.

</div>

The implementation described by `\cite{yuan2023physdiff}`{=latex} applies a physics simulator at scheduled intervals during the sampling process to map the diffusion model’s prediction to a “physically-plausible’’ action that imitates data points of the training distribution. This simulator dramatically alters the diffusion model’s outputs utilizing a learned motion imitation policy, which has been trained to match the ground truth samples using proximal policy optimization. In this setting the diffusion model provides a starting point for the physics simulator and is not directly responsible for the final results of these predictions. Direct parallels can be drawn between this approach and other methods which solely task the diffusion model with initializing an external model `\cite{giannone2023aligning, power2023sampling}`{=latex}. Additionally, while the authors characterize this mapping as a projection, it is critical to note that this is a projection onto the learned distribution of the simulator and not a projection onto a feasible set, explaining the remaining constraint violations in the outputs.

## Convergence of PDM

<figure id="fig:projection_analysis">
<img src="./figures/projection-analysis.png"" style="width:40.0%" />
<figcaption>Visualization of the decreasing upper bound on error introduced in a single sampling step for <span><em><span style="color: MidnightBlue">PDM</span></em></span>, as opposed to the strictly increasing upper bound of conditional (<span><em><span style="color: Mahogany">Cond</span></em></span>) models.</figcaption>
</figure>

As shown in Figure <a href="#fig:reverse-violations" data-reference-type="ref" data-reference="fig:reverse-violations">1</a>, the PDM sampling process converges to a feasible subdistribution, a behavior that is generally not present in standard conditional models. Corollary <a href="#proof:corollary-1" data-reference-type="ref" data-reference="proof:corollary-1">3</a> provides insight into this behavior as it outlines the decreasing upper bound on *‘Error’* that can be introduced in a single sampling step. To further illustrate this behavior, the decreasing upper bound can be illustrated in Figure <a href="#fig:projection_analysis" data-reference-type="ref" data-reference="fig:projection_analysis">14</a>.

# Computational costs [appendix:computational-cost]

To compare the computational costs of sampling with PDM to our baselines, we record the execution times for the reverse process of a single sample. *The implementations of PDM have not been optimized for runtime, and represent an upper bound.* All sampling is run on two NVIDIA A100 GPUs. All computations are conducted on these GPUs with the exception of the interior point method projection used in the 3D Human motion experiment and the Constrained Trajectories experiment which runs on two CPU cores.

|  | Constrained Materials | 3D Human Motion | Constrained Trajectories | Physics-informed Motion |
|:---|---:|---:|---:|---:|
| *<span style="color: MidnightBlue">PDM</span>* | 26.89 | 682.40\\(^*\\) | 383.40\\(^*\\) | 48.85 |
| *Post\\(^+\\)* | 26.01 | – | – | 27.58 |
| *<span style="color: Mahogany">Cond</span>* | 18.51 | 13.79 | 0.56 | 35.30 |
| *Cond\\(^+\\)* | 18.54 | – | 106.41 | 36.63 |

Average sampling run-time in seconds.

We implement projections at all time steps in this analysis, although practically this is can be optimized to reduce the total number of projections as described in the subsequent section. Additionally, we set \\(M = 100\\) and \\(T = 10\\) for each experiment. The increase in computational cost present in PDM is directly dependant on the tractability of the projections and the size of \\(M\\).

The computational cost of the projections is largely problem dependant, and we conjecture that these times could be improved by implementing more efficient projections. For example, the projection for Constrained Trajectories could be dramatically improved by implementing this method on the GPUs instead of CPUs (\\(^*\\)). However, these improvements are beyond the scope of this paper. Our projection implementations are further described in §<a href="#appendix:experimental-settings" data-reference-type="ref" data-reference="appendix:experimental-settings">12</a>.

Additionally, the number of iterations for each \\(t\\) can often be decreased below \\(M = 100\\) or the projection frequency can be adjusted (as has been done for in this section for the CPU implemented projections), offering additional speed-up.

<figure id="fig:vlb">
<table>
<tbody>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>Frame 1</strong></td>
<td style="text-align: center;"><strong>Frame 2</strong></td>
<td style="text-align: center;"><strong>Frame 3</strong></td>
<td style="text-align: center;"><strong>Frame 4</strong></td>
<td style="text-align: center;"><strong>Frame 5</strong></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/vlb-earth-1.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/vlb-earth-2.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/vlb-earth-3.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/vlb-earth-4.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/vlb-earth-5.png"" alt="image" /></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><img src="./figures/vlb-5.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/vlb-4.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/vlb-3.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/vlb-2.png"" alt="image" /></td>
<td style="text-align: center;"><img src="./figures/vlb-1.png"" alt="image" /></td>
</tr>
</tbody>
</table>
<figcaption>Iterative projections using model trained with variational lower bound objective.</figcaption>
</figure>

# Variational lower bound training objective [appendix:vlb]

As defined in Equation 2, PDM uses a score-matching objective to learn to the gradients of the log probability of the data distribution. This understanding allows the sampling process to be framed in a light that is consistent to optimization theory, allowing equivalences to be drawn between the proposed sampling procedure and projected gradient descent. This aspect is also integral to the theory presented in Section 4.2.

Other DDPM and DDIM implementation utilize a variation lower bound objective, which is a tractable approach to minimizing the negative log likelihood on the network’s noise predictions. While this approach was inspired by the score-matching objective, we empirically demonstrate that iterative projections perform much worse in our tested settings than models optimized using this training objective, producing clearly inferior solutions in the Physics-informed experiments and failing to produce viable solutions in the material science domain explored.

This approach (visualized in Figure <a href="#fig:vlb" data-reference-type="ref" data-reference="fig:vlb">15</a>) resulted in an FID score of 113.8 \\(\pm\\) 4.9 on the Physics-informed Motion experiment and 388.2 \\(\pm\\) 13.0 on the Constrained Materials experiment, much higher than those produced using the score-matching objective, adopted in our paper. We hold that this is because the approach proposed in our paper is more theoretically sound when framed in terms of a gradient-based sampling process.

# Missing proofs [appendix:theorem-proof]

## Proof of Theorem <a href="#proof:theorem-1" data-reference-type="ref" data-reference="proof:theorem-1">2</a> [proof-of-theorem-prooftheorem-1]

<div class="proof" markdown="1">

*Proof.* By optimization theory of convergence in a convex setting, provided an arbitrarily large number of update steps \\(M\\), \\(\bm{x}_{t}^{M}\\) will reach the global minimum. Hence, this justifies the existence of \\(\bar{I}\\) as at some iteration as \\(i \xrightarrow{} \infty\\), \\(\left\| \bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^{i}|\bm{x}_0) \right\|_2 \leq \left\| \rho_t \right\|_2\\) will hold for every iteration thereafter.

Consider that a gradient step is taken without the addition of noise, and \\(i \geq \Bar{I}\\). Provided this, there are two possible cases.

#### Case 1: 

Assume \\(\bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^{i}|\bm{x}_0)\\) is closer to the optimum than \\(\rho_t\\). Then, \\(\bm{x}_{t}^{i}\\) is infeasible.

This claim is true by the definition of \\(\rho_t\\), as \\(\bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^{i}|\bm{x}_0)\\) is closer to \\(\mu\\) than is achievable from the nearest feasible point to \\(\mu\\). Hence, \\(\bm{x}_{t}^{i}\\) must be infeasible.

Furthermore, the additional gradient step produces a point that is closer to the optimum than possible by a single update step from the feasible region. Hence it holds that \\[\begin{aligned}
\label{eq:error-ineq}
\textit{Error}(\bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^{i}|\bm{x}_0)) >  
\textit{Error}(\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i}) + \gamma_t \nabla_{\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i})} \log q(\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i})|\bm{x}_0))
\end{aligned}\\] as the distance from the feasible region to the projected point will be at most the distance to \\(\rho_t\\). As this point is closer to the global optimum than \\(\rho_t\\), the cost of projecting \\(\bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^{i}|\bm{x}_0)\\) is greater than that of any point that begins in the feasible region.

#### Case 2: 

Assume \\(\bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^{i}|\bm{x}_0)\\) is equally close to the optimum as \\(\rho_t\\). In this case, there are two possibilities; either (1) \\(\bm{x}_{t}^{i}\\) is the closest point in \\(\mathbf{C}\\) to \\(\mu\\) or (2) \\(\bm{x}_{t}^{i}\\) is infeasible.

If the former is true, \\(\bm{x}_{t}^{i} = \mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i})\\), implying \\[\begin{aligned}
\label{eq:error-eq}
\textit{Error}(\bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^{i}|\bm{x}_0)) =  
\textit{Error}(\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i}) + 
\gamma_t \nabla_{\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i})} \log q(\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i})|\bm{x}_0))
\end{aligned}\\]

Next, consider that the latter is true. If \\(\bm{x}_{t}^{i}\\) is not the closest point in \\(\mathbf{C}\\) to the global minimum, then it must be an equally close point to \\(\mu\\) that falls outside the feasible region. Now, a subsequent gradient step of \\(\bm{x}_{t}^{i}\\) will be the same length as a gradient step from the closest feasible point to \\(\mu\\), by our assumption.

Since the feasible region and the objective function are convex, this forms a triangle inequality, such that the cost of this projection is greater than the size of the gradient step. Thus, by this inequality, Equation <a href="#eq:error-ineq" data-reference-type="ref" data-reference="eq:error-ineq">[eq:error-ineq]</a> applies.

Finally, for both cases we must consider the addition of stochastic noise. As this noise is sampled from the Gaussian with a mean of zero, we synthesize this update step as the expectation over,

\\[\begin{aligned}
\label{eq:error-final}
\mathbb{E} \left[ \textit{Error}(\bm{x}_{t}^{i} + \gamma_t \nabla_{\bm{x}_{t}^{i}} \log q(\bm{x}_{t}^{i}|\bm{x}_0) + \sqrt{2\gamma_t}\bm{\epsilon}) \right] \geq  
\mathbb{E} \left[ \textit{Error}(\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i}) + \gamma_t \nabla_{\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i})} \log q(\mathcal{P}_{\mathbf{C}}(\bm{x}_{t}^{i})|\bm{x}_0) + \sqrt{2\gamma_t}\bm{\epsilon}) \right]
\end{aligned}\\]

or equivalently as represented in Equation <a href="#eq:theorem-1" data-reference-type="ref" data-reference="eq:theorem-1">[eq:theorem-1]</a>. ◻

</div>
