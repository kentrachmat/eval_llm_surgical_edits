# Fair Bilevel Neural Network (FairBiNN): On Balancing fairness and accuracy via Stackelberg Equilibrium

## Abstract

The persistent challenge of bias in machine learning models necessitates robust solutions to ensure parity and equal treatment across diverse groups, particularly in classification tasks. Current methods for mitigating bias often result in information loss and an inadequate balance between accuracy and fairness. To address this, we propose a novel methodology grounded in bilevel optimization principles. Our deep learning-based approach concurrently optimizes for both accuracy and fairness objectives, and under certain assumptions, achieving proven Pareto optimal solutions while mitigating bias in the trained model. Theoretical analysis indicates that the upper bound on the loss incurred by this method is less than or equal to the loss of the Lagrangian approach, which involves adding a regularization term to the loss function. We demonstrate the efficacy of our model primarily on tabular datasets such as UCI Adult and Heritage Health. When benchmarked against state-of-the-art fairness methods, our model exhibits superior performance, advancing fairness-aware machine learning solutions and bridging the accuracy-fairness gap. The implementation of FairBiNN is available on <https://github.com/yazdanimehdi/FairBiNN>.

# Introduction

Artificial intelligence and machine learning models have seen significant growth over the past decades, leading to their integration into various domains such as hiring pipelines, face recognition, financial services, healthcare, and criminal justice. This widespread adoption of algorithmic decision-making has raised concerns about algorithmic bias, which can result in discrimination and unfairness towards minority groups. Recently, the issue of fairness in artificial intelligence has garnered considerable attention from interdisciplinary research communities, addressing these ethical concerns `\citep{pessach2020algorithmic}`{=latex}.  
Several definitions of fairness have been proposed to tackle unwanted bias in machine learning techniques. These definitions generally fall into two categories: individual fairness and group fairness. Individual fairness ensures that similar individuals are treated similarly, with similarities determined by past information `\citep{dwork2012fairness, yurochkin2019training}`{=latex}. Group fairness, on the other hand, measures statistical equality between different subgroups defined by sensitive characteristics such as race or gender `\citep{zemel2013learning, louizos2015variational, hardt2016equality}`{=latex}. In this paper, we focus on group fairness, which we will refer to simply as fairness from this point onward.

Fairness approaches in machine learning are commonly categorized into three groups: (1) Pre-process approaches: These methods involve changing the data before training to improve fairness, such as reweighing labels or adjusting features to reduce distribution differences between privileged and unprivileged groups, making it harder for classifiers to differentiate them `\citep{kamiran2012data,luong2011k,feldman2015certifying,tayebi2022unbiaseddti}`{=latex}. Generative adversarial networks were also utilized to produce unbiased datasets by altering the generator network’s value function to balance accuracy and fairness `\citep{rajabi2021tabfairgan}`{=latex}. (2) In-process approaches: These methods modify the algorithm during training, for instance by adding regularization terms to the objective function to ensure fairness. Examples include penalizing the mutual information between protected attributes and classifier predictions to allow a trade-off between fairness and accuracy `\citep{kamishima2012fairness}`{=latex}, and adding constraints to satisfy a proxy for equalized odds `\citep{zafar2017fairness,zafar2017fairness_a}`{=latex}. (3) Post-process approaches: These techniques adjust the outcomes after training, such as flipping some outcomes to improve fairness `\citep{hardt2016equality}`{=latex}, or using different thresholds for privileged and unprivileged groups to optimize the trade-off between accuracy and fairness `\citep{menon2018cost,corbett2017algorithmic}`{=latex}.

In this work we targeted the in-process bias mitigation category. Traditionally, the fairness multi-criteria problem has been addressed using Lagrangian optimization, wherein the objective function is a weighted sum of the primary and secondary loss functions. While this approach allows for the explicit incorporation of fairness constraints through Lagrange multipliers, it may overlook the complex interdependencies between the primary and secondary objectives.  
A promising alternative to the Lagrangian is the bilevel optimization approach which offers several advantages. By formulating the problem as a hierarchical optimization task, we can explicitly model the interactions between the primary and secondary objectives. This allows us to capture the nuanced dynamics of fairness optimization and ensure that improvements in one objective do not come at the expense of the other.

In summary, we introduce a novel method that can be trained on existing datasets without requiring any alterations to the data itself (data augmentation, perturbation, etc). Our methodology provides a principled approach to addressing the multi-criteria fairness problem in neural networks. Through rigorous theoretical analysis, we formulated the problem as a bilevel optimization task, proving that it yields Pareto-optimal solutions. We derived an effective optimization strategy that is at least as effective as the Lagrangian approach. Empirical evaluations on tabular datasets demonstrate the efficacy of our method, achieving superior results compared to traditional approaches.

# Related works

Multi-objective optimization in neural networks involves optimizing two or more conflicting objectives simultaneously. Fairness problems are inherently multi-objective in nature, as improvements in one objective (e.g., enhancing fairness) often come at the expense of another objective (e.g., improving accuracy). Several optimization techniques in neural networks have been employed to balance accuracy and fairness. Classic Methods transform these objectives into a single objective by combining them, typically using a weighted sum where each objective is multiplied by a weight that reflects its importance. Adding Regularization and penalty terms are the most common methods that incorporate fairness constraints (e.g., demographic parity, equal opportunity) directly into the loss function, penalizing disparities in prediction errors across demographic groups or any other unfair behavior. To reduce variation across different groups, `\citet{zafar2017fairness}`{=latex} proposes “disparate mistreatment”, a new notation for fairness, and standardized the decision bounds of a convex margin-based classifier. Adversarial debiasing and Fair Representation Learning are two examples of these techniques, which encourage the model to generate fair outcomes by introducing a penalty term based on an adversarial network or a representation learning framework that is invariant to protected attributes, respectively. `\citet{zhang2018mitigating}`{=latex} addressed bias by limiting an adversary’s ability to infer sensitive characteristics from predictions. Avoiding the complexity of adversarial training, `\citet{moyer2018invariant}`{=latex} used mutual information to achieve invariant data representations concerning specific factors. `\citet{song2019learning}`{=latex} proposed an information-theoretic method that leverages both information-theoretic and adversarial approaches to achieve controllable fair data representations, adhering to demographic parity. By incorporating a forget-gate similar to those in LSTMs, `\citet{jaiswal2020invariant}`{=latex} introduced adversarial forgetting to enhance fairness. `\citet{gupta2021controllable}`{=latex} utilized certain estimates for contrasting information to optimize theoretical objectives, facilitating suitable trade-offs between demographic parity and accuracy in the statistical population. Lagrangian optimization techniques are a subset of these techniques that use Lagrange multipliers or other similar techniques to incorporate constraints directly into the objective function, turning constrained optimization problems into unconstrained ones. `\citet{agarwal2018reductions}`{=latex} proposes an approach for fair classification by framing the constrained optimization problem as a two-player game where one player optimizes the model parameters, and the other imposes the constraints, and Lagrangian multipliers are used to solve this problem. `\citet{cotter2019training}`{=latex} expanded this work in a more general inequality-constrained setting, by simultaneously training each player on two distinct datasets to enhance generalizability. They enforce independence by regularizing the covariance between predictions and sensitive variables, which reduces the variation in the relationship between the two. Despite analytic solutions and theoretical assurances, scaling game-theoretic techniques for more complex models remains challenging `\citep{chuang2021fair}`{=latex}. In addition, these constraints-based optimizations are data-dependent, meaning the model may exhibit different behavior during evaluation even if constraints are met during training. Less common approaches including Pareto-based genetic algorithm, Reinforcement Learning, Gradient-Based Methods, and Transfer and Meta-Learning Approaches have been also utilized in this domain. `\citet{mehrabi2021attributing}`{=latex} demonstrated how proxy attributes lead to indirect unfairness using an attention-based approach and employed a post-processing method to reduce the weight of attributes responsible for unfairness. `\citet{perrone2021fair}`{=latex} introduces a general constrained Bayesian optimization (BO) framework to fine-tune the model’s performance while enforcing one or multiple fairness constraints. A probabilistic model is used to describe the objective function, and estimates are made for the posterior variances and means for each hyperparameter configuration. By adding a fairness regularization term to a meta-learning framework, `\citet{slack2019fair}`{=latex} suggests an adaptation of the Model-Agnostic Meta-Learning (MAML) `\citep{finn2017model}`{=latex} algorithm. The primary objective and fairness regularization terms are included in the loss function used to update the model parameters for each task during the inner loop (Learner). The model parameters are updated in the outer loop (Meta-learner) to maximize performance and fairness across all tasks. Although these techniques have achieved a good balance between fairness and accuracy, they might not capture all of the complex interdependencies between these two objectives. In this paper we propose a bilevel optimization approach as an alternative to the Lagrangian approaches. Bilevel optimization is a hierarchical structure in which the context or constraints for the "follower" (lower-level) problem are set by the "leader" (upper-level) problem `\citep{dempe2002foundations}`{=latex}. The leader makes decisions first, and the follower optimizes their decisions based on the leader’s choices. This approach can handle more complex and nuanced multi-objective optimization problems in neural networks and is suitable for scenarios where one objective directly influences another and there are complex interactions between the two objectives. In this paper we demonstrate that the bilevel optimization often can achieve better balance and performance compared to classic regularization-based optimization approaches `\citep{dempe2002foundations,sinha2017review,colson2007overview}`{=latex}. Bilevel optimization offers several advantages; by explicitly modeling a two-level decision-making process, his approach represents the problems in a more natural way where one objective inherently depends on the outcome of another. It provides more flexibility and control over the optimization process by enabling separate optimization of constraints at each level. The upper-level optimization can dynamically adjust the lower-level objective based on the current solution, potentially leading to more adaptive and context-sensitive optimization outcomes. Fairness and accuracy objectives can be directly integrated into the optimization framework without the need for additional strategies such as meta-learning.

# Methodology [method]

In this section we are introducing a novel bi-level optimization framework for training neural networks to obtain Pareto optimal solutions when optimizing two potentially competing objectives. Our approach leverages a leader-follower structure, where the leader problem aims to minimize one objective function (e.g. a primary loss), while the follower problem optimizes a secondary objective. We provide theoretical guarantees that our bi-level approach produces Pareto optimal solutions and performs at least as well as, and often strictly better than, the common practice of combining multiple objectives via a weighted regularization term in a single loss function. The full statements of these theorems and their proofs are provided in the Theoretical Analysis subsection below. Our bi-level approach offers several benefits over regularization-based methods. First, it allows for easy customization of the architecture and training algorithm used for each objective. The leader and follower problems can utilize different network architectures, regularizers, optimizers, etc. as best suited for each task. Second, the leader problem remains a pure minimization of the primary loss, without any regularization terms that may slow or hinder its optimization. Separating out secondary objectives ensures the primary task is learned most effectively. Finally, bi-level training exposes a clear interface for controlling the trade-off between objectives. By constraining the follower problem more or less strictly, we can encourage stronger or weaker adherence to the secondary goal relative to the primary one. To realize these benefits, we employ an iterative gradient-based algorithm to solve the bi-level problem, alternating between updating the leader and follower parameters. We unroll the follower optimization for a fixed number of steps, and backpropagate through this unrolled process to update the leader weights.

## Theoretical Analysis [theory]

### Problem Formulation

The fairness multi-criteria problem in neural networks can be formulated as a bi-criteria optimization problem. Let \\(f(\theta_p, \theta_s)\\) denote the primary objective loss function and \\(\varphi(\theta_p, \theta_s)\\) denote the secondary objective loss function. Here, \\(\theta_p \in \Theta_p\\) represents the parameters responsible for optimizing the primary objective, and \\(\theta_s \in \Theta_s\\) represents the parameters for the secondary objective. The problem is formally stated as: \\[\label{bi-criteriaM}
    \min_{\theta_p \in \Theta_p, \theta_f \in \Theta_f}\{{f(\theta_p, \theta_f), \varphi(\theta_p, \theta_f)}\}\\]

### Theoretical Foundation: Stackelberg Equilibrium and Pareto Optimality

We leverage the theoretical results of `\cite{migdalas1995stackelberg}`{=latex}, which investigates the relationship between Stackelberg equilibria and Pareto optimality in game theory. The paper addresses fundamental questions regarding the conditions under which a Stackelberg equilibrium coincides with a Pareto optimal outcome. By proving that our bi-level optimization problem satisfies the assumptions outlined in the paper, we establish a strong theoretical foundation for our approach. Specifically, we demonstrate that under certain conditions, the Stackelberg equilibrium of our bi-level optimization problem is equivalent to a Pareto optimal solution for the bi-criteria problem of balancing accuracy and fairness objectives. By rigorously verifying these assumptions in the context of our neural network optimization problem, we establish that the Stackelberg equilibrium reached by our bilevel approach indeed corresponds to a Pareto optimal solution. This theoretical grounding provides confidence that our methodology effectively balances the competing objectives of accuracy and fairness, yielding a principled and well-justified solution to the problem at hand.

We leverage several key theoretical results to formulate our approach. First, Lemma <a href="#lemma:lipschitz" data-reference-type="ref" data-reference="lemma:lipschitz">5</a> establishes the Lipschitz continuity of the neural network function with respect to a subset of parameters. This lemma provides the foundation for analyzing the behavior of the objective functions under parameter variations.

<div class="definition" markdown="1">

**Definition 1**. A function \\(f : \mathbb{R}^n \longrightarrow \mathbb{R}^m\\) is called Lipschitz continuous if there exists a constant L such that: \\[\forall x, y \in \mathbb{R}^n , \left\lVert f(x) - f(y)\right\rVert_2 \leq L\left\lVert x - y\right\rVert\\] The smallest \\(L\\) for which the previous inequality is true is called the Lipschitz constant of \\(f\\) and will be denoted \\(L(f)\\).

</div>

Assume that the following assumptions are satisfied:

<div id="assum:strict_convexity" class="assumption" markdown="1">

**Assumption 2**. The primary loss function \\(f(\theta_p, \theta_s)(x)\\) is strictly convex in a neighborhood of its local optimum. That is, for any \\(\theta_p, \theta_p' \in \Theta_p\\) and fixed \\(\theta_s \in \Theta_s\\), if \\(\theta_p \neq \theta_p'\\) and \\(\theta_p, \theta_p'\\) are sufficiently close to the local optimum \\(\theta_p^*\\), then \\[f(\lambda \theta_p + (1-\lambda)\theta_p', \theta_s) < \lambda f(\theta_p, \theta_s) + (1-\lambda)f(\theta_p', \theta_s)\\] for any \\(\lambda \in (0,1)\\).

</div>

<div id="assum:small_steps" class="assumption" markdown="1">

**Assumption 3**. \\(|\theta_s - \hat{\theta}_s| \leq \epsilon\\), where \\(\epsilon\\) is sufficiently small, i.e., the steps of the secondary parameters are sufficiently small. \\(\theta_s\\) and \\(\hat{\theta}_s\\) represent the parameters for the secondary objective and their updated values, respectively.

</div>

<div id="assum:bounded_output" class="assumption" markdown="1">

**Assumption 4**. Let \\(f_l(.)\\) denote the output function of the \\(l\\)-th layer in a neural network with \\(L\\) layers. For each layer \\(l \in {1, \dots, L}\\), there exists a constant \\(c_l > 0\\) such that for any input \\(x_l\\) to the \\(l\\)-th layer: \\[|f_l(x_l)| \leq c_l\\] where \\(|.|\\) denotes a suitable norm (e.g., Euclidean norm for vectors, spectral norm for matrices). Refer to Section <a href="#bounded_assump" data-reference-type="ref" data-reference="bounded_assump">7.4</a> for common practices in implementing the bounded output assumption.

</div>

We recognized the importance of examining how our theory’s underlying assumptions apply to real-world applications. For a detailed discussion, refer to Section <a href="#assumption_discussion" data-reference-type="ref" data-reference="assumption_discussion">7.3</a>.

<div id="lemma:lipschitz" class="lemma" markdown="1">

**Lemma 5**. *Let \\(f(x;\theta)\\) be a neural network with L layers, where each layer is a linear transformation followed by a Lipschitz continuous activation function.  
Let \\(\theta\\) be the set of all parameters of the neural network, and \\(\theta_s \subseteq \theta\\) be any subset of parameters. Then, \\(f(x; \theta)\\) is Lipschitz continuous with respect to \\(\theta_s\\). \[See proof <a href="#lemma:lipschitz:ap" data-reference-type="ref" data-reference="lemma:lipschitz:ap">14</a>\]*

</div>

We discussed the Lipschitz continuity of common activation functions and popular neural networks, such as CNNs and GNNs, in Sections <a href="#act_func_lip" data-reference-type="ref" data-reference="act_func_lip">7.5</a> and <a href="#nets_lip" data-reference-type="ref" data-reference="nets_lip">7.6</a>, respectively.

Theorems <a href="#theorem:pareto" data-reference-type="ref" data-reference="theorem:pareto">6</a> and <a href="#theorem:unique_minimum" data-reference-type="ref" data-reference="theorem:unique_minimum">7</a> further inform our approach. The former establishes conditions under which improvements in the secondary objective lead to improvements in the primary objective, while the latter guarantees the existence of unique minimum solutions for the secondary loss function under certain optimization conditions.

<div id="theorem:pareto" class="theorem" markdown="1">

**Theorem 6**. *Let \\(f(\theta_p, \theta_s)\\) for constant \\(\theta_s\\) be the primary objective loss function and \\(\varphi(\theta_p, \theta_s)\\) for constant \\(\theta_p\\) be the secondary objective loss function, where \\(\theta_p \in \Theta_p\\) and \\(\theta_s \in \Theta_s\\) are the primary task and secondary task parameters, respectively.*

*Consider two sets of parameters \\((\theta_p, \theta_s)\\) and \\((\hat{\theta}_p, \hat{\theta}_s)\\) such that \\(\varphi(\hat{\theta}_p, \hat{\theta}_s) \leq \varphi(\theta_p, \theta_s)\\). Then \\(f(\hat{\theta}_p, \hat{\theta}_s) \leq f(\theta_p, \theta_s)\\) holds based on Lemma <a href="#lemma:lipschitz" data-reference-type="ref" data-reference="lemma:lipschitz">5</a>. \[See proof <a href="#theorem:pareto:ap" data-reference-type="ref" data-reference="theorem:pareto:ap">15</a>\]*

</div>

<div id="theorem:unique_minimum" class="theorem" markdown="1">

**Theorem 7**. *Let \\(\varphi(\theta_p, \theta_s)\\) be the secondary loss function, where \\(\theta_p \in \Theta_p\\) and \\(\theta_s \in \Theta_s\\) are the primary and secondary task parameters, respectively. Let \\((\theta_p^{(t)}, \theta_s^{(t)})\\) denote the parameters at optimization step \\(t\\), and let \\((\theta_p^{(t+1)}, \theta_s^{(t+1)})\\) be the updated parameters obtained by minimizing \\(\varphi(\theta_p^{(t)}, \theta_s)\\) with respect to \\(\theta_s\\) using a sufficiently small step size \\(\eta > 0\\), i.e.:*

*\\[\theta_s^{(t+1)} = \theta_s^{(t)} - \eta \nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s^{(t)})\\]*

*Then, for a sufficiently small step size \\(\eta\\), the updated secondary parameters \\(\theta_s^{(t+1)}\\) are the unique minimum solution for the secondary loss function \\(\varphi(\theta_p^{(t)}, \theta_s)\\). \[See proof <a href="#theorem:unique_minimum:ap" data-reference-type="ref" data-reference="theorem:unique_minimum:ap">16</a>\]*

</div>

Based on these theoretical insights, we derive our bilevel optimization formulation, as described in Theorem <a href="#theorem:main_result" data-reference-type="ref" data-reference="theorem:main_result">8</a>. This theorem establishes the equivalence between the bi-criteria problem and a bilevel optimization problem, allowing us to apply existing theoretical results on Stackelberg equilibrium to the optimization of neural networks.

<div id="theorem:main_result" class="theorem" markdown="1">

**Theorem 8**. *Under the assumptions stated in Theorems <a href="#theorem:pareto" data-reference-type="ref" data-reference="theorem:pareto">6</a> and <a href="#theorem:unique_minimum" data-reference-type="ref" data-reference="theorem:unique_minimum">7</a>, the bi-criteria problem (Eq. <a href="#bi-criteriaM" data-reference-type="eqref" data-reference="bi-criteriaM">[bi-criteriaM]</a>) is equivalent to the bilevel optimization problem:*

*\\[\begin{aligned}
    \min_{\theta_p \in \Theta_p} \quad & f(\theta_p, \theta_s^*(\theta_p)) \\
    \text{s.t.} \quad & \theta_s^*(\theta_p) = \mathop{\mathrm{arg\,min}}_{\theta_s \in \Theta_s} \varphi(\theta_p, \theta_s)
\end{aligned}\\]*

*where \\(\theta_s^*(\theta_p)\\) denotes the optimal secondary parameters for a given \\(\theta_p\\).*

</div>

<div class="proof" markdown="1">

*Proof.* The proof follows from Theorems <a href="#theorem:pareto" data-reference-type="ref" data-reference="theorem:pareto">6</a> and <a href="#theorem:unique_minimum" data-reference-type="ref" data-reference="theorem:unique_minimum">7</a> `\cite{migdalas1995stackelberg}`{=latex}.

By Theorem <a href="#theorem:pareto" data-reference-type="ref" data-reference="theorem:pareto">6</a>, under the assumptions of strict convexity, Lipschitz continuity, and sufficiently small steps of the secondary parameters, if \\(\varphi(\hat{\theta}_p, \hat{\theta}_s) \leq \varphi(\theta_p, \theta_s)\\), then \\(f(\hat{\theta}_p, \hat{\theta}_s) \leq f(\theta_p, \theta_s)\\).

By Theorem <a href="#theorem:unique_minimum" data-reference-type="ref" data-reference="theorem:unique_minimum">7</a>, under the same assumptions, for each optimization step of the secondary loss function with sufficiently small steps, the updated parameters are the unique minimum solution for the secondary loss function, then the bi-criteria problem <a href="#bi-criteriaM" data-reference-type="eqref" data-reference="bi-criteriaM">[bi-criteriaM]</a> is equivalent to the bilevel optimization problem.

Therefore, the conclusions drawn in the paper `\cite{migdalas1995stackelberg}`{=latex} can be directly applied to the multi-objective optimization problem in neural networks, as the problem is equivalent to the bilevel optimization problem under the stated assumptions. ◻

</div>

<div id="theorem:lagrangian" class="theorem" markdown="1">

**Theorem 9**. *Assume that the step size in the Lagrangian approach \\(\alpha_{\mathcal{L}}\\) is equal to the step size for the outer optimization problem in the bilevel optimization approach \\(\alpha_f\\), the scale of the two loss functions should be comparable, the Lagrangian multiplier \\(\lambda\\) is equal to the step size for the inner optimization problem in the bilevel optimization approach \\(\alpha_s\\), and \\(\theta_p\\) is overparameterized for the given problem. Then, under certain conditions, the overall performance of the primary loss function in the bilevel optimization approach may be better than the Lagrangian approach.*

</div>

<div class="proof" markdown="1">

*Proof.* Let \\(f(\theta_p, \theta_s)\\) denote the primary loss and \\(\varphi(\theta_p, \theta_s)\\) denote the secondary loss. Assume that both \\(f\\) and \\(\varphi\\) are differentiable with respect to \\(\theta_p\\) and \\(\theta_s\\). Define the Lagrangian function as: \\[\mathcal{L}(\theta_p, \theta_s, \lambda) = f(\theta_p, \theta_s) + \lambda \varphi(\theta_p, \theta_s)\\] The update rules for \\(\theta_p\\) and \\(\theta_s\\) in the Lagrangian approach are: \\[\begin{aligned}
\theta_p^{(t+1)} &= \theta_p^{(t)} - \alpha_{\mathcal{L}} \nabla_{\theta_p} \mathcal{L}(\theta_p^{(t)}, \theta_s^{(t)}, \lambda) \\
\theta_s^{(t+1)} &= \theta_s^{(t)} - \alpha_{\mathcal{L}} \nabla_{\theta_s} \mathcal{L}(\theta_p^{(t)}, \theta_s^{(t)}, \lambda)
\end{aligned}\\] The update rules for \\(\theta_p\\) and \\(\theta_s\\) in the bilevel optimization approach are: \\[\begin{aligned}
\theta_p^{(t+1)} &= \theta_p^{(t)} - \alpha_f \nabla_{\theta_p} f(\theta_p^{(t)}, \theta_s^{(t)}) \\
\theta_s^{(t+1)} &= \theta_s^{(t)} - \alpha_s \nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s^{(t)})
\end{aligned}\\] Due to the overparameterization of \\(\theta_p\\), there exists a set \\(\Theta_p\\) such that for any \\(\theta_p \in \Theta_p\\) `\cite{allen2019learning}`{=latex}, \\(f(\theta_p, \theta_s) = f(\theta_p^*, \theta_s)\\), where \\(\theta_p^*\\) is an optimal solution for the primary loss when \\(\theta_s\\) is fixed. Suppose that the bilevel optimization approach converges to a solution \\((\theta_p^B, \theta_s^B)\\) and the Lagrangian approach converges to a solution \\((\theta_p^L, \theta_s^L)\\). Consider the following inequality: \\[\begin{aligned}
f(\theta_p^B, \theta_s^B) &= f(\theta_p^B, \theta_s^B) + \lambda \varphi(\theta_p^B, \theta_s^B) - \lambda \varphi(\theta_p^B, \theta_s^B) \\
&\leq f(\theta_p^B, \theta_s^B) + \lambda \varphi(\theta_p^B, \theta_s^B) - \lambda \varphi(\theta_p^L, \theta_s^L) \\
&= \mathcal{L}(\theta_p^B, \theta_s^B, \lambda) - \lambda \varphi(\theta_p^L, \theta_s^L) \\
&\leq \mathcal{L}(\theta_p^L, \theta_s^L, \lambda) - \lambda \varphi(\theta_p^L, \theta_s^L) \\
&= f(\theta_p^L, \theta_s^L)
\end{aligned}\\] The first inequality holds because \\((\theta_p^L, \theta_s^L)\\) is the minimizer of \\(\varphi(\theta_p, \theta_s)\\) in the Lagrangian approach. The second inequality holds because \\((\theta_p^L, \theta_s^L)\\) is the minimizer of \\(\mathcal{L}(\theta_p, \theta_s, \lambda)\\) in the Lagrangian approach. Since \\(\theta_p^B \in \Theta_p\\) and \\(\theta_p^L \notin \Theta_p\\), we have: \\[f(\theta_p^B, \theta_s^B) = f(\theta_p^B, \theta_s^B) \leq f(\theta_p^L, \theta_s^L)\\] Therefore, under the assumptions that \\(\alpha_{\mathcal{L}} = \alpha_f\\), The sizes of the two loss functions \\(f(\theta_p, \theta_s)\\) and \\(\varphi(\theta_p, \theta_s)\\) should not differ significantly in terms of their order of magnitude, \\(\lambda = \alpha_s\\), and \\(\theta_p\\) is overparameterized for the given problem, the bilevel optimization approach may converge to a solution that achieves better performance for the primary loss compared to the Lagrangian approach. ◻

</div>

## Practical Implementation

To connect the Stackelberg game analysis with a practical implementation for datasets, we can formulate a bilevel optimization problem. The upper-level problem corresponds to the accuracy player (leader), while the lower-level problem corresponds to the fairness player (follower). We’ll use gradient-based optimization techniques to solve the problem.

Let’s consider a dataset \\(\mathcal{D} = \{(x_i, a_i, y_i)\}_{i=1}^N\\), where \\(x_i\\) represents the features, \\(a_i\\) represents the sensitive attribute, and \\(y_i\\) represents the target variable for the \\(i\\)-th sample.

The optimization problem can be formulated as follows:

\\[\begin{aligned}
\min_{\theta_a} \quad & \frac{1}{N} \sum_{i=1}^N L_{acc}(f(x_i; \theta_a, \theta_f^*), y_i) \\
\text{s.t.} \quad & \theta_f^* \in \arg\min_{\theta_f} \frac{1}{N} \sum_{i=1}^N L_{fair}(f(x_i; \theta_a, \theta_f), a_i, y_i)
\end{aligned}\\]

where \\(f(x; \theta_a, \theta_f)\\) is the model parameterized by the accuracy parameters \\(\theta_a\\) and fairness parameters \\(\theta_f\\), \\(L_{acc}\\) is the accuracy loss function (e.g., binary cross-entropy), and \\(L_{fair}\\) is the fairness loss function (e.g., demographic parity loss). We showed that the demographic parity loss function, when applied to the output of neural network layers, is also Lipschitz continuous (Theorem <a href="#theorem:dp" data-reference-type="ref" data-reference="theorem:dp">10</a>).

**Demographic Parity Loss Function:** The demographic parity loss function \\(DP(f)\\) is defined as: \\[DP(f) = \left| \mathbb{E}_{x \sim p(x|a=0)}[f(\theta_1; x)] - \mathbb{E}_{x \sim p(x|a=1)}[f(\theta_2; x)] \right|\\] where \\(a\\) is a sensitive attribute (e.g., race, gender) with two possible values (0 and 1), and \\(p(x|a)\\) is the conditional probability distribution of \\(x\\) given \\(a\\).

<div id="theorem:dp" class="theorem" markdown="1">

**Theorem 10**. *If \\(f(x)\\) is Lipschitz continuous with Lipschitz constant \\(L_f\\), then the demographic parity loss function \\(\ell_{DP}(f)\\) is also Lipschitz continuous with Lipschitz constant \\(L_{DP} = 2L_f\\). \[See proof <a href="#theorem:dp:ap" data-reference-type="ref" data-reference="theorem:dp:ap">19</a>\]*

*We can easily extend this theorem to include another common fairness metric, equalized odds, as explained in Section <a href="#eo_expansion" data-reference-type="ref" data-reference="eo_expansion">7.2</a>.*

</div>

Here’s a practical implementation using gradient-based optimization:

<figure id="algorithm1">
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p>Initialize accuracy parameters <span class="math inline"><em>θ</em><sub><em>a</em></sub></span> and fairness parameters <span class="math inline"><em>θ</em><sub><em>f</em></sub></span> Accuracy player’s optimization Sample a minibatch <span class="math inline">ℬ<sub><em>a</em></sub> ⊂ 𝒟</span> Compute accuracy loss: <span class="math inline">$L_a = \frac{1}{|\mathcal{B}_a|} \sum_{i \in \mathcal{B}_a} L_{acc}(f(x_i; \theta_a, \theta_f), y_i)$</span> Update accuracy parameters: <span class="math inline"><em>θ</em><sub><em>a</em></sub> ← <em>θ</em><sub><em>a</em></sub> − <em>η</em><sub><em>a</em></sub>∇<sub><em>θ</em><sub><em>a</em></sub></sub><em>L</em><sub><em>a</em></sub></span> Fairness player’s optimization Sample a minibatch <span class="math inline">ℬ<sub><em>f</em></sub> ⊂ 𝒟</span> Compute fairness loss: <span class="math inline">$L_f = \frac{1}{|\mathcal{B}_f|} \sum_{i \in \mathcal{B}_f} L_{fair}(f(x_i; \theta_a, \theta_f), a_i, y_i)$</span> Update fairness parameters: <span class="math inline"><em>θ</em><sub><em>f</em></sub> ← <em>θ</em><sub><em>f</em></sub> − <em>η</em><sub><em>f</em></sub>∇<sub><em>θ</em><sub><em>f</em></sub></sub><em>L</em><sub><em>f</em></sub></span></p>
</div>
<figcaption>Fairness-Accuracy Bilevel Optimization</figcaption>
</figure>

In practice, the model \\(f(x; \theta_a, \theta_f)\\) can be implemented as a neural network with separate layers for accuracy and fairness (Figure <a href="#fig:fairbinn" data-reference-type="ref" data-reference="fig:fairbinn">23</a>). The accuracy layers are parameterized by \\(\theta_a\\), while the fairness layers are parameterized by \\(\theta_f\\). The accuracy loss \\(L_{acc}\\) can be chosen based on the task at hand, such as binary cross-entropy for binary classification or mean squared error for regression. The fairness loss \\(L_{fair}\\) can be a fairness metric such as demographic parity loss or equalized odds loss. The learning rates \\(\eta_a\\) and \\(\eta_f\\) control the step sizes for updating the accuracy and fairness parameters, respectively. They can be tuned using techniques like grid search or learning rate scheduling.

By implementing this algorithm on a dataset, we can optimize the model to balance accuracy and fairness, guided by the Stackelberg game formulation. At each iteration, the parameters related to accuracy are optimized while keeping the fairness parameters fixed. Then, with the accuracy parameters held constant, the fairness parameters are optimized. This separate optimization process provides fine-grained control over the trade-off between accuracy and fairness.

# Experiments

In this section, we contrast our methodology with other benchmark approaches found in the literature.

We employed two metrics for evaluation: accuracy (higher values preferred) for the classification task, and demographic parity differences (DP, lower values preferred) for fairness assessment. Detailed descriptions of all metrics used and implementation settings are available in sections <a href="#ap:metric" data-reference-type="ref" data-reference="ap:metric">7.9</a> and <a href="#ap:hyper" data-reference-type="ref" data-reference="ap:hyper">7.10</a> in the appendix, respectively.

We evaluated our method for bias mitigation to various current state-of-the-art approaches. We concentrate on strategies specifically tuned to achieve the best results in statistical parity metrics on tabular studies.

## Datasets:

We used two well-known benchmark datasets in this field for our experiments which are as follows:  
*UCI Adult Dataset* `\cite{uciadult}`{=latex}, This dataset is based on demographic data gathered in 1994, including a train set of 30000 and a test set of 15,000 samples. The goal is to forecast if the salary is more than $50,000 yearly, and the binary protected attribute is the gender of samples gathered in the dataset.

*Heritage Health Dataset* `\cite{abraham2017deriving}`{=latex}, Predicting the Charleson Index, a measure of a patient’s 10-year mortality. The Heritage Health dataset contains samples from roughly 51,000 patients of which 41000 are in the training set, and 11000 are in the test set. The protected attribute, which has nine potential values, is age.

## Baselines:

We compare our results with the following state-of-the-art methods as benchmarks:

- **CVIB** `\cite{moyer2018invariant}`{=latex}: Achieves fairness using a conditional variational autoencoder.

- **MIFR** `\cite{song2019learning}`{=latex}: Optimizes the fairness objective with a mix of information bottleneck factor and adversarial learning.

- **FCRL** `\cite{gupta2021controllable}`{=latex}: Uses specific approximations for contrastive information to maximize theoretical goals, facilitating appropriate trade-offs among statistical parity, demographic parity, and precision.

- **MaxEnt-ARL** `\cite{roy2019mitigating}`{=latex}: Employs adversarial learning to mitigate unfairness.

- **Adversarial Forgetting** `\cite{jaiswal2020invariant}`{=latex}: Uses adversarial learning techniques for fairness.

- **Fair Consistency Regularization (FCR)** `\citep{an2022transferring}`{=latex}: Aims to minimize and balance consistency loss across groups.

- **Robust Fairness Regularization (RFR)** `\cite{jiang2024chasing}`{=latex}: Considers the worst-case scenario within the model weight perturbation ball for each sensitive attribute group to ensure robust fairness.

## Bilevel (FairBiNN) vs. Lagrangian Method

We compare our proposed FairBiNN method with the traditional Lagrangian regularization approach to empirically validate the theoretical benefits of bilevel optimization. Our analysis focuses on the convergence behavior and stability of both methods. For comprehensive details on performance and computational complexity comparison, refer to Section <a href="#direct_comp_lag" data-reference-type="ref" data-reference="direct_comp_lag">7.7</a>. In the appendix, we have provided the BCE loss plots over epochs for each dataset (Fig. <a href="#fig:tabular_loss" data-reference-type="ref" data-reference="fig:tabular_loss">7</a>) and demonstrated the superior performance of the Bi-level approach compared to the Lagrangian approach. We have also presented a comparative analysis of these approaches for the trade-off between accuracy and Statistical Parity Difference (SPD) in Figure <a href="#fig:reg_compare" data-reference-type="ref" data-reference="fig:reg_compare">10</a>.

While Theorem <a href="#theorem:lagrangian" data-reference-type="ref" data-reference="theorem:lagrangian">9</a> in the paper proves that, under certain assumptions, the primary loss function in the bilevel optimization approach is upper bounded by the loss of the Lagrangian approach at the optimal solution, it does not analyze or guarantee the convergence behavior of the algorithms. The empirical results for the Health and Adult datasets show that the bilevel approach outperforms the Lagrangian method in minimizing the BCE loss. However, further investigation is needed to understand the convergence properties of the algorithms and connect the theoretical results with empirical observations. Despite this, the experimental results highlight the potential of the bilevel optimization framework to optimize accuracy and fairness objectives, offering a promising approach to address the multi-criteria fairness problem in neural networks.

### Benchmark Comparison

We provide average accuracy as a measure of most probable accuracy and maximum demographic parity as a measure of worst-case scenario bias, calculated across five iterations of the training process using random seeds. Unlike `\citet{gupta2021controllable}`{=latex}, we did not use any preprocessing on the data before feeding it to our network. Reported results for our model are Pareto solutions for the neural network during training with different \\(\eta_f\\). Results are reported for methods with a multi-layer perceptron classifier with two hidden layers.  

<figure id="fig:tabular_compare">
<figure id="fig:adult_compare">
<img src="./figures/adult_compare.png"" />
<figcaption>UCI Adult</figcaption>
</figure>
<figure id="fig:health_compare">
<img src="./figures/health_compare.png"" />
<figcaption>Heritage Health</figcaption>
</figure>
<figcaption>Accuracy of various benchmark models compared to the FairBiNN model versus statistical demographic parity for the (a) UCI Adult dataset and (b) Heritage Health dataset. The optimal region on this graph is the bottom right, indicating high accuracy and low DP. The results demonstrate that our model (red diamond markers) significantly outperforms other benchmark models on the UCI Adult dataset and closely competes with recent state-of-the-art models on the Heritage Health dataset.</figcaption>
</figure>

Figures <a href="#fig:adult_compare" data-reference-type="ref" data-reference="fig:adult_compare">2</a> and <a href="#fig:health_compare" data-reference-type="ref" data-reference="fig:health_compare">3</a> show trade-offs of the statistical demographic parity vs. accuracy associated with various bias reduction strategies in the UCI Adult dataset and Heritage Health dataset, respectively. The ideal area of the graph for the result of a method is to measure how much the curve is located in the lower right corner of the graph, which means accurate and fair results concerning protected attributes. Our results demonstrate that the Bilevel design significantly outperforms competing methods in Adult dataset.

# Limitations and Future Work [limitations]

While our results are promising, it’s important to acknowledge several limitations of our current approach:

One of the most widely used activation functions, softmax, is not Lipschitz continuous. This limits the direct application of our method to multiclass classification problems. Future work could explore alternative activation functions or modifications to the softmax that preserve Lipschitz continuity while maintaining similar functionality for multiclass problems.

Attention mechanisms, which are widely used in modern language models and other architectures, are not Lipschitz continuous. This presents a challenge for extending our method to state-of-the-art architectures in natural language processing and other domains that heavily rely on attention. However, research into the Lipschitz continuity of attention layers has already begun, with Dasoulas et al. `\citep{dasoulas2021lipschitz}`{=latex} introducing LipschitzNorm, a simple and parameter-free normalization technique applied to attention scores to enforce Lipschitz continuity in self-attention mechanisms. Their experiments on graph attention networks (GAT) demonstrate that enforcing Lipschitz continuity generally enhances the performance of deep attention models.

Our theoretical analysis primarily provides guarantees in comparison to regularization methods. While the results show improvements in fairness overall, the theory does not offer absolute fairness guarantees for the final model. Extending the theoretical framework to include direct fairness guarantees could strengthen the method’s applicability.

This method was not validated on dataset augmentation approaches, which are common in practice for improving model generalization and robustness. Future work should investigate how our method interacts with various data augmentation techniques and whether it maintains its fairness properties under such conditions.

Our current implementation focuses on a single fairness metric (demographic parity). In practice, multiple, sometimes conflicting, fairness criteria may be relevant. Extending our method to handle multiple fairness constraints simultaneously could make it more versatile for real-world applications.

Addressing these limitations presents exciting opportunities for future research. By tackling these challenges, we can further enhance the applicability and effectiveness of fair machine learning methods across a broader range of real-world scenarios and cutting-edge architectures.

# Discussion and Conclusion

Our primary contribution lies in the theoretical foundation and general applicability of the proposed framework, rather than extensive ablation studies on specific datasets or network configurations. However, we recognize the importance of empirical evaluations. Our work introduces a novel approach to addressing the multi-criteria fairness problem in neural networks, supported by theoretical analysis, particularly Theorem <a href="#theorem:main_result" data-reference-type="ref" data-reference="theorem:main_result">8</a>, which establishes properties of the optimal solution under certain assumptions, independent of specific datasets or architectures. The results on vision and graph datasets (<a href="#ap:results" data-reference-type="ref" data-reference="ap:results">7.11</a>) and ablation studies on the impact of \\(\eta\\) (<a href="#ablation_eta" data-reference-type="ref" data-reference="ablation_eta">7.13.1</a>), the position of fairness layers (<a href="#ablation_pos" data-reference-type="ref" data-reference="ablation_pos">7.13.3.1</a>), and different layer types (<a href="#ap:ablation" data-reference-type="ref" data-reference="ap:ablation">7.13</a>), presented in the appendix, demonstrate the effectiveness and versatility of our approach. These studies show that the bilevel optimization framework can be successfully applied to various layer types and network architectures, beyond the single linear layer used in the main experiments. Our experimentation across diverse datasets, including UCI Adult, Heritage Health, and other domains like graph datasets \[<a href="#ap:graph" data-reference-type="ref" data-reference="ap:graph">7.8.1</a>\] (POKEC-Z, POKEC-N, and NBA) and vision datasets \[<a href="#ap:vision" data-reference-type="ref" data-reference="ap:vision">7.8.2</a>\] (CelebA), further illustrates the versatility and efficacy of our method. Including these ablation studies in the appendix allows us to maintain the main text’s focus on theoretical contributions and the general framework while providing additional empirical evidence to support our claims.

Our results demonstrate the superiority of our model over state-of-the-art fairness methods in reducing bias while maintaining accuracy, highlighting the potential of our framework to advance fairness-aware machine learning solutions. Notably, our study represents a significant contribution by formulating multi-objective problems in neural networks as a bilevel design, providing a powerful tool for achieving equitable outcomes across diverse groups in classification tasks. Future research can address our method’s limitations and explore potential directions as outlined in Section <a href="#limitations" data-reference-type="ref" data-reference="limitations">5</a>.

# References [references]

<div class="thebibliography" markdown="1">

Alexandre Abraham, Michael P Milham, Adriana Di Martino, R Cameron Craddock, Dimitris Samaras, Bertrand Thirion, and Gael Varoquaux Deriving reproducible biomarkers from multi-site resting-state data: An autism-based example *NeuroImage*, 147: 736–745, 2017. **Abstract:** Resting-state functional Magnetic Resonance Imaging (R-fMRI) holds the promise to reveal functional biomarkers of neuropsychiatric disorders. However, extracting such biomarkers is challenging for complex multi-faceted neuropatholo- gies, such as autism spectrum disorders. Large multi-site datasets increase sample sizes to compensate for this complexity, at the cost of uncontrolled heterogeneity. This heterogeneity raises new challenges, akin to those face in realistic di- agnostic applications. Here, we demonstrate the feasibility of inter-site classi cation of neuropsychiatric status, with an application to the Autism Brain Imaging Data Exchange (ABIDE) database, a large (N=871) multi-site autism dataset. For this purpose, we investigate pipelines that extract the most predictive biomarkers from the data. These R-fMRI pipelines build participant-speci c connectomes from functionally-de ned brain areas. Connectomes are then compared across participants to learn patterns of connectivity that di erentiate typical controls from individuals with autism. We predict this neuropsychiatric status for participants from the same acquisition sites or di erent, unseen, ones. Good choices of methods for the various steps of the pipeline lead to 67% prediction accuracy on the full ABIDE data, which is signi cantly better than previously reported results. We perform extensive validation on multiple subsets of the data de ned by di erent inclusion criteria. These enables detailed analysis of the factors contributing to successful connectome-based prediction. First, prediction accuracy improves as we include more subjects, up to the maximum amount of subjects available. Second, the de nition of functional brain areas is of paramount importance for biomarker discovery: brain areas extracted from large R-fMRI datasets outperform reference atlases in the classi cation tasks. (@abraham2017deriving)

Alekh Agarwal, Alina Beygelzimer, Miroslav Dudı́k, John Langford, and Hanna Wallach A reductions approach to fair classification In *International Conference on Machine Learning*, pages 60–69. PMLR, 2018. **Abstract:** We present a systematic approach for achieving fairness in a binary classification setting. While we focus on two well-known quantitative definitions of fairness, our approach encompasses many other previously studied definitions as special cases. The key idea is to reduce fair classification to a sequence of cost-sensitive classification problems, whose solutions yield a randomized classifier with the lowest (empirical) error subject to the desired constraints. We introduce two reductions that work for any representation of the cost-sensitive classifier and compare favorably to prior baselines on a variety of data sets, while overcoming several of their disadvantages. (@agarwal2018reductions)

Zeyuan Allen-Zhu, Yuanzhi Li, and Yingyu Liang Learning and generalization in overparameterized neural networks, going beyond two layers *Advances in neural information processing systems*, 32, 2019. **Abstract:** The fundamental learning theory behind neural networks remains largely open. What classes of functions can neural networks actually learn? Why doesn’t the trained network overfit when it is overparameterized? In this work, we prove that overparameterized neural networks can learn some notable concept classes, including two and three-layer networks with fewer parameters and smooth activations. Moreover, the learning can be simply done by SGD (stochastic gradient descent) or its variants in polynomial time using polynomially many samples. The sample complexity can also be almost independent of the number of parameters in the network. On the technique side, our analysis goes beyond the so-called NTK (neural tangent kernel) linearization of neural networks in prior works. We establish a new notion of quadratic approximation of the neural network, and connect it to the SGD theory of escaping saddle points. (@allen2019learning)

Bang An, Zora Che, Mucong Ding, and Furong Huang Transferring fairness under distribution shifts via fair consistency regularization *Advances in Neural Information Processing Systems*, 35: 32582–32597, 2022. **Abstract:** The increasing reliance on ML models in high-stakes tasks has raised a major concern on fairness violations. Although there has been a surge of work that improves algorithmic fairness, most of them are under the assumption of an identical training and test distribution. In many real-world applications, however, such an assumption is often violated as previously trained fair models are often deployed in a different environment, and the fairness of such models has been observed to collapse. In this paper, we study how to transfer model fairness under distribution shifts, a widespread issue in practice. We conduct a fine-grained analysis of how the fair model is affected under different types of distribution shifts and find that domain shifts are more challenging than subpopulation shifts. Inspired by the success of self-training in transferring accuracy under domain shifts, we derive a sufficient condition for transferring group fairness. Guided by it, we propose a practical algorithm with a fair consistency regularization as the key component. A synthetic dataset benchmark, which covers all types of distribution shifts, is deployed for experimental verification of the theoretical findings. Experiments on synthetic and real datasets including image and tabular data demonstrate that our approach effectively transfers fairness and accuracy under various distribution shifts. (@an2022transferring)

Alex Beutel, Jilin Chen, Zhe Zhao, and Ed H Chi Data decisions and theoretical implications when adversarially learning fair representations *arXiv preprint arXiv:1707.00075*, 2017. **Abstract:** How can we learn a classifier that is "fair" for a protected or sensitive group, when we do not know if the input to the classifier belongs to the protected group? How can we train such a classifier when data on the protected group is difficult to attain? In many settings, finding out the sensitive input attribute can be prohibitively expensive even during model training, and sometimes impossible during model serving. For example, in recommender systems, if we want to predict if a user will click on a given recommendation, we often do not know many attributes of the user, e.g., race or age, and many attributes of the content are hard to determine, e.g., the language or topic. Thus, it is not feasible to use a different classifier calibrated based on knowledge of the sensitive attribute. Here, we use an adversarial training procedure to remove information about the sensitive attribute from the latent representation learned by a neural network. In particular, we study how the choice of data for the adversarial training effects the resulting fairness properties. We find two interesting results: a small amount of data is needed to train these adversarial models, and the data distribution empirically drives the adversary’s notion of fairness. (@beutel2017data)

Avishek Bose and William Hamilton Compositional fairness constraints for graph embeddings In *International Conference on Machine Learning*, pages 715–724. PMLR, 2019. **Abstract:** Learning high-quality node embeddings is a key building block for machine learning models that operate on graph data, such as social networks and recommender systems. However, existing graph embedding techniques are unable to cope with fairness constraints, e.g., ensuring that the learned representations do not correlate with certain attributes, such as age or gender. Here, we introduce an adversarial framework to enforce fairness constraints on graph embeddings. Our approach is compositional—meaning that it can flexibly accommodate different combinations of fairness constraints during inference. For instance, in the context of social recommendations, our framework would allow one user to request that their recommendations are invariant to both their age and gender, while also allowing another user to request invariance to just their age. Experiments on standard knowledge graph and recommender system benchmarks highlight the utility of our proposed framework. (@bose2019compositional)

Joy Buolamwini and Timnit Gebru Gender shades: Intersectional accuracy disparities in commercial gender classification In *Conference on fairness, accountability and transparency*, pages 77–91. PMLR, 2018. **Abstract:** Recent studies demonstrate that machine learning algorithms can discriminate based on classes like race and gender. In this work, we present an approach to evaluate bias present in automated facial analysis al- gorithms and datasets with respect to phe- notypic subgroups. Using the dermatolo- gist approved Fitzpatrick Skin Type clas- si cation system, we characterize the gen- der and skin type distribution of two facial analysis benchmarks, IJB-A and Adience. We nd that these datasets are overwhelm- ingly composed of lighter-skinned subjects (79 :6% for IJB-A and 86 :2% for Adience) and introduce a new facial analysis dataset which is balanced by gender and skin type. We evaluate 3 commercial gender clas- si cation systems using our dataset and show that darker-skinned females are the most misclassi ed group (with error rates of up to 34 :7%). The maximum error rate for lighter-skinned males is 0 :8%. The substantial disparities in the accuracy of classifying darker females, lighter females, darker males, and lighter males in gender classi cation systems require urgent atten- tion if commercial companies are to build genuinely fair, transparent and accountable facial analysis algorithms. (@buolamwini2018gender)

Kristy Choi, Aditya Grover, Trisha Singh, Rui Shu, and Stefano Ermon Fair generative modeling via weak supervision In *International Conference on Machine Learning*, pages 1887–1898. PMLR, 2020. **Abstract:** Real-world datasets are often biased with respect to key demographic factors such as race and gender. Due to the latent nature of the underlying factors, detecting and mitigating bias is especially challenging for unsupervised machine learning. We present a weakly supervised algorithm for overcoming dataset bias for deep generative models. Our approach requires access to an additional small, unlabeled reference dataset as the supervision signal, thus sidestepping the need for explicit labels on the underlying bias factors. Using this supplementary dataset, we detect the bias in existing datasets via a density ratio technique and learn generative models which efficiently achieve the twin goals of: 1) data efficiency by using training examples from both biased and reference datasets for learning; and 2) data generation close in distribution to the reference dataset at test time. Empirically, we demonstrate the efficacy of our approach which reduces bias w.r.t. latent factors by an average of up to 34.6% over baselines for comparable image generation using generative adversarial networks. (@choi2020fair)

Ching-Yao Chuang and Youssef Mroueh Fair mixup: Fairness via interpolation *arXiv preprint arXiv:2103.06503*, 2021. **Abstract:** Training classifiers under fairness constraints such as group fairness, regularizes the disparities of predictions between the groups. Nevertheless, even though the constraints are satisfied during training, they might not generalize at evaluation time. To improve the generalizability of fair classifiers, we propose fair mixup, a new data augmentation strategy for imposing the fairness constraint. In particular, we show that fairness can be achieved by regularizing the models on paths of interpolated samples between the groups. We use mixup, a powerful data augmentation strategy to generate these interpolates. We analyze fair mixup and empirically show that it ensures a better generalization for both accuracy and fairness measurement in tabular, vision, and language benchmarks. (@chuang2021fair)

Benoı̂t Colson, Patrice Marcotte, and Gilles Savard An overview of bilevel optimization *Annals of operations research*, 153: 235–256, 2007. **Abstract:** This paper is devoted to bilevel optimization, a branch of mathematical program- ming of both practical and theoretical interest. Starting with a simple example, we proceed towards a general formulation. We then present ﬁelds of application, focus on solution ap- proaches, and make the connection with MPECs (Mathematical Programs with Equilibrium Constraints). (@colson2007overview)

Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq Algorithmic decision making and the cost of fairness In *Proceedings of the 23rd acm sigkdd international conference on knowledge discovery and data mining*, pages 797–806, 2017. **Abstract:** Algorithms are now regularly used to decide whether defendants awaiting trial are too dangerous to be released back into the community. In some cases, black defendants are substantially more likely than white defendants to be incorrectly classified as high risk. To mitigate such disparities, several techniques have recently been proposed to achieve algorithmic fairness. Here we reformulate algorithmic fairness as constrained optimization: the objective is to maximize public safety while satisfying formal fairness constraints designed to reduce racial disparities. We show that for several past definitions of fairness, the optimal algorithms that result require detaining defendants above race-specific risk thresholds. We further show that the optimal unconstrained algorithm requires applying a single, uniform threshold to all defendants. The unconstrained algorithm thus maximizes public safety while also satisfying one important understanding of equality: that all individuals are held to the same standard, irrespective of race. Because the optimal constrained and unconstrained algorithms generally differ, there is tension between improving public safety and satisfying prevailing notions of algorithmic fairness. By examining data from Broward County, Florida, we show that this trade-off can be large in practice. We focus on algorithms for pretrial release decisions, but the principles we discuss apply to other domains, and also to human decision makers carrying out structured decision rules. (@corbett2017algorithmic)

Andrew Cotter, Maya Gupta, Heinrich Jiang, Nathan Srebro, Karthik Sridharan, Serena Wang, Blake Woodworth, and Seungil You Training well-generalizing classifiers for fairness metrics and other data-dependent constraints In *International Conference on Machine Learning*, pages 1397–1405. PMLR, 2019. **Abstract:** Classifiers can be trained with data-dependent constraints to satisfy fairness goals, reduce churn, achieve a targeted false positive rate, or other policy goals. We study the generalization performance for such constrained optimization problems, in terms of how well the constraints are satisfied at evaluation time, given that they are satisfied at training time. To improve generalization performance, we frame the problem as a two-player game where one player optimizes the model parameters on a training dataset, and the other player enforces the constraints on an independent validation dataset. We build on recent work in two-player constrained optimization to show that if one uses this two-dataset approach, then constraint generalization can be significantly improved. As we illustrate experimentally, this approach works not only in theory, but also in practice. (@cotter2019training)

Elliot Creager, David Madras, Jörn-Henrik Jacobsen, Marissa Weis, Kevin Swersky, Toniann Pitassi, and Richard Zemel Flexibly fair representation learning by disentanglement In *International conference on machine learning*, pages 1436–1445. PMLR, 2019. **Abstract:** We consider the problem of learning representations that achieve group and subgroup fairness with respect to multiple sensitive attributes. Taking inspiration from the disentangled representation learning literature, we propose an algorithm for learning compact representations of datasets that are useful for reconstruction and prediction, but are also \\}emph{flexibly fair}, meaning they can be easily modified at test time to achieve subgroup demographic parity with respect to multiple sensitive attributes and their conjunctions. We show empirically that the resulting encoder—which does not require the sensitive attributes for inference—enables the adaptation of a single representation to a variety of fair classification tasks with new target labels and subgroup definitions. (@creager2019flexibly)

Enyan Dai and Suhang Wang Say no to the discrimination: Learning fair graph neural networks with limited sensitive attribute information In *Proceedings of the 14th ACM International Conference on Web Search and Data Mining*, pages 680–688, 2021. **Abstract:** Graph neural networks (GNNs) have shown great power in modeling graph structured data. However, similar to other machine learning models, GNNs may make predictions biased on protected sensitive attributes, e.g., skin color and gender. Because machine learning algorithms including GNNs are trained to reflect the distribution of the training data which often contains historical bias towards sensitive attributes. In addition, the discrimination in GNNs can be magnified by graph structures and the message-passing mechanism. As a result, the applications of GNNs in sensitive domains such as crime rate prediction would be largely limited. Though extensive studies of fair classification have been conducted on i.i.d data, methods to address the problem of discrimination on non-i.i.d data are rather limited. Furthermore, the practical scenario of sparse annotations in sensitive attributes is rarely considered in existing works. Therefore, we study the novel and important problem of learning fair GNNs with limited sensitive attribute information. FairGNN is proposed to eliminate the bias of GNNs whilst maintaining high node classification accuracy by leveraging graph structures and limited sensitive information. Our theoretical analysis shows that FairGNN can ensure the fairness of GNNs under mild conditions given limited nodes with known sensitive attributes. Extensive experiments on real-world datasets also demonstrate the effectiveness of FairGNN in debiasing and keeping high accuracy. (@dai2021say)

Enyan Dai and Suhang Wang Learning fair graph neural networks with limited and private sensitive attribute information *IEEE Transactions on Knowledge and Data Engineering*, 35 (7): 7103–7117, 2022. **Abstract:** Graph neural networks (GNNs) have shown great power in modeling graph structured data. However, similar to other machine learning models, GNNs may make biased predictions w.r.t protected sensitive attributes, e.g., skin color and gender. This is because machine learning algorithms including GNNs are trained to reflect the distribution of the training data which often contains historical bias towards sensitive attributes. In addition, we empirically show that the discrimination in GNNs can be magnified by graph structures and the message-passing mechanism of GNNs. As a result, the applications of GNNs in high-stake domains such as crime rate prediction would be largely limited. Though extensive studies of fair classification have been conducted on independently and identically distributed (i.i.d) data, methods to address the problem of discrimination on non-i.i.d data are rather limited. Generally, learning fair models require abundant sensitive attributes to regularize the model. However, for many graphs such as social networks, users are reluctant to share sensitive attributes. Thus, only limited sensitive attributes are available for fair GNN training in practice. Moreover, directly collecting and applying the sensitive attributes in fair model training may cause privacy issues, because the sensitive information can be leaked in data breach or attacks on the trained model. Therefore, we study a novel and important problem of learning fair GNNs with limited number of private sensitive attributes, i.e., sensitive attributes that are processed with a privacy-preserving mechanism. In an attempt to address these problems, FairGNN is proposed to eliminate the bias of GNNs whilst maintaining high node classification accuracy by leveraging graph structures and limited sensitive information. To further preserve the privacy, private sensitive attributes with privacy guarantee are obtained by injecting noise based on local differential privacy. And We further extend FairGNN to NT-FairGNN to handle the limited and private sensitive attributes to simultaneously achieve fairness and preserve privacy. Theoretical analysis and extensive experiments on real-world datasets demonstrate the effectiveness of FairGNN and NT-FairGNN in achieving fair and high-accurate classification. (@dai2022learning)

George Dasoulas, Kevin Scaman, and Aladin Virmaux Lipschitz normalization for self-attention layers with application to graph neural networks In *International Conference on Machine Learning*, pages 2456–2466. PMLR, 2021. **Abstract:** Attention based neural networks are state of the art in a large range of applications. However, their performance tends to degrade when the number of layers increases. In this work, we show that enforcing Lipschitz continuity by normalizing the attention scores can significantly improve the performance of deep attention models. First, we show that, for deep graph attention networks (GAT), gradient explosion appears during training, leading to poor performance of gradient-based training algorithms. To address this issue, we derive a theoretical analysis of the Lipschitz continuity of attention modules and introduce LipschitzNorm, a simple and parameter-free normalization for self-attention mechanisms that enforces the model to be Lipschitz continuous. We then apply LipschitzNorm to GAT and Graph Transformers and show that their performance is substantially improved in the deep setting (10 to 30 layers). More specifically, we show that a deep GAT model with LipschitzNorm achieves state of the art results for node label prediction tasks that exhibit long-range dependencies, while showing consistent improvements over their unnormalized counterparts in benchmark node classification tasks. (@dasoulas2021lipschitz)

Stephan Dempe *Foundations of bilevel programming* Springer Science & Business Media, 2002. **Abstract:** We study linear bilevel programming problems, where (some of) the leader and the follower variables are restricted to be integer. A discussion on the relationships between the optimistic and the pessimistic setting is presented, providing necessary and sufficient conditions for them to be equivalent. A new class of inequalities, the follower optimality cuts, is introduced. They are used to derive a single-level non-compact reformulation of a bilevel problem, both for the optimistic and for the pessimistic case. The same is done for a family of known inequalities, the no-good cuts, and a polyhedral comparison of the related formulations is carried out. Finally, for both the optimistic and the pessimistic approach, we present a branch-and-cut algorithm and discuss computational results. (@dempe2002foundations)

Yuxiao Dong, Omar Lizardo, and Nitesh V Chawla Do the young live in a “smaller world” than the old? age-specific degrees of separation in a large-scale mobile communication network *arXiv preprint arXiv:1606.07556*, 2016. **Abstract:** In this paper, we investigate the phenomenon of "age-specific small worlds" using data from a large-scale mobile communication network approximating interaction patterns at societal scale. Rather than asking whether two random individuals are separated by a small number of links, we ask whether individuals in specific age groups live in a small world in relation to individuals from other age groups. Our analysis shows that there is systematic variation in this age-relative small world effect. Young people live in the "smallest world," being separated from other young people and their parent’s generation via a smaller number of intermediaries than older individuals. The oldest people live in the "least small world," being separated from their same age peers and their younger counterparts by a larger number of intermediaries. Variation in the small world effect is specific to age as a node attribute (being absent in the case of gender) and is consistently observed under several data robustness checks. The discovery of age-specific small worlds is consistent with well-known social mechanisms affecting the way age interacts with network connectivity and the relative prevalence of kin ties and non-kin ties observed in this network. This social pattern has significant implications for our understanding of generation-specific dynamics of information cascades, diffusion phenomena, and the spread of fads and fashions. (@dong2016young)

Dheeru Dua and Casey Graff machine learning repository 2017. URL <http://archive.ics.uci.edu/ml>. **Abstract:** The University of California–Irvine (UCI) Machine Learning (ML) Repository (UCIMLR) is consistently cited as one of the most popular dataset repositories, hosting hundreds of high-impact datasets. However, a significant portion, including 28.4% of the top 250, cannot be imported via the ucimlrepo package that is provided and recommended by the UCIMLR website. Instead, they are hosted as .zip files, containing nonstandard formats that are difficult to import without additional ad hoc processing. To address this issue, here we present lucie—load University California Irvine examples—a utility that automatically determines the data format and imports many of these previously non-importable datasets, while preserving as much of a tabular data structure as possible. lucie was designed using the top 100 most popular datasets and benchmarked on the next 130, where it resulted in a success rate of 95.4% vs. 73.1% for ucimlrepo. lucie is available as a Python package on PyPI with 98% code coverage. (@uciadult)

Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel Fairness through awareness In *Proceedings of the 3rd innovations in theoretical computer science conference*, pages 214–226, 2012. **Abstract:** We study fairness in classification, where individuals are classified, e.g., admitted to a university, and the goal is to prevent discrimination against individuals based on their membership in some group, while maintaining utility for the classifier (the university). The main conceptual contribution of this paper is a framework for fair classification comprising (1) a (hypothetical) task-specific metric for determining the degree to which individuals are similar with respect to the classification task at hand; (2) an algorithm for maximizing utility subject to the fairness constraint, that similar individuals are treated similarly. We also present an adaptation of our approach to achieve the complementary goal of "fair affirmative action," which guarantees statistical parity (i.e., the demographics of the set of individuals receiving any classification are the same as the demographics of the underlying population), while treating similar individuals as similarly as possible. Finally, we discuss the relationship of fairness to privacy: when fairness implies privacy, and how tools developed in the context of differential privacy may be applied to fairness. (@dwork2012fairness)

Harrison Edwards and Amos Storkey Censoring representations with an adversary *arXiv preprint arXiv:1511.05897*, 2015. **Abstract:** In practice, there are often explicit constraints on what representations or decisions are acceptable in an application of machine learning. For example it may be a legal requirement that a decision must not favour a particular group. Alternatively it can be that that representation of data must not have identifying information. We address these two related issues by learning flexible representations that minimize the capability of an adversarial critic. This adversary is trying to predict the relevant sensitive variable from the representation, and so minimizing the performance of the adversary ensures there is little or no information in the representation about the sensitive variable. We demonstrate this adversarial approach on two problems: making decisions free from discrimination and removing private information from images. We formulate the adversarial model as a minimax problem, and optimize that minimax objective using a stochastic gradient alternate min-max optimizer. We demonstrate the ability to provide discriminant free representations for standard test problems, and compare with previous state of the art methods for fairness, showing statistically significant improvement across most cases. The flexibility of this method is shown via a novel problem: removing annotations from images, from unaligned training examples of annotated and unannotated images, and with no a priori knowledge of the form of annotation provided to the model. (@edwards2015censoring)

Michael Feldman, Sorelle A Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian Certifying and removing disparate impact In *proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining*, pages 259–268, 2015. **Abstract:** What does it mean for an algorithm to be biased? In U.S. law, unintentional bias is encoded via disparate impact, which occurs when a selection process has widely different outcomes for different groups, even as it appears to be neutral. This legal determination hinges on a definition of a protected class (ethnicity, gender) and an explicit description of the process. (@feldman2015certifying)

Chelsea Finn, Pieter Abbeel, and Sergey Levine Model-agnostic meta-learning for fast adaptation of deep networks In *International conference on machine learning*, pages 1126–1135. PMLR, 2017. **Abstract:** We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune. We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies. (@finn2017model)

Henry Gouk, Eibe Frank, Bernhard Pfahringer, and Michael J Cree Regularisation of neural networks by enforcing lipschitz continuity *Machine Learning*, 110: 393–416, 2021. **Abstract:** Abstract We investigate the effect of explicitly enforcing the Lipschitz continuity of neural networks with respect to their inputs. To this end, we provide a simple technique for computing an upper bound to the Lipschitz constant—for multiple p -norms—of a feed forward neural network composed of commonly used layer types. Our technique is then used to formulate training a neural network with a bounded Lipschitz constant as a constrained optimisation problem that can be solved using projected stochastic gradient methods. Our evaluation study shows that the performance of the resulting models exceeds that of models trained with other common regularisers. We also provide evidence that the hyperparameters are intuitive to tune, demonstrate how the choice of norm for computing the Lipschitz constant impacts the resulting model, and show that the performance gains provided by our method are particularly noticeable when only a small amount of training data is available. (@gouk2021regularisation)

Umang Gupta, Aaron Ferber, Bistra Dilkina, and Greg Ver Steeg Controllable guarantees for fair outcomes via contrastive information estimation In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pages 7610–7619, 2021. **Abstract:** Controlling bias in training datasets is vital for ensuring equal treatment, or parity, between different groups in downstream applications. A naive solution is to transform the data so that it is statistically independent of group membership, but this may throw away too much information when a reasonable compromise between fairness and accuracy is desired. Another common approach is to limit the ability of a particular adversary who seeks to maximize parity. Unfortunately, representations produced by adversarial approaches may still retain biases as their efficacy is tied to the complexity of the adversary used during training. To this end, we theoretically establish that by limiting the mutual information between representations and protected attributes, we can assuredly control the parity of any downstream classifier. We demonstrate an effective method for controlling parity through mutual information based on contrastive information estimators and show that they outperform approaches that rely on variational bounds based on complex generative models. We test our approach on UCI Adult and Heritage Health datasets and demonstrate that our approach provides more informative representations across a range of desired parity thresholds while providing strong theoretical guarantees on the parity of any downstream algorithm. (@gupta2021controllable)

Moritz Hardt, Eric Price, and Nati Srebro Equality of opportunity in supervised learning *Advances in neural information processing systems*, 29: 3315–3323, 2016. **Abstract:** We propose a criterion for discrimination against a specified sensitive attribute in supervised learning, where the goal is to predict some target based on available features. Assuming data about the predictor, target, and membership in the protected group are available, we show how to optimally adjust any learned predictor so as to remove discrimination according to our definition. Our framework also improves incentives by shifting the cost of poor classification from disadvantaged groups to the decision maker, who can respond by improving the classification accuracy. In line with other studies, our notion is oblivious: it depends only on the joint statistics of the predictor, the target and the protected attribute, but not on interpretation of individualfeatures. We study the inherent limits of defining and identifying biases based on such oblivious measures, outlining what can and cannot be inferred from different oblivious tests. We illustrate our notion using a case study of FICO credit scores. (@hardt2016equality)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun Deep residual learning for image recognition In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016. **Abstract:** Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers - 8× deeper than VGG nets \[40\] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions1, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. (@he2016deep)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun Deep residual learning for image recognition In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016. **Abstract:** Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers - 8× deeper than VGG nets \[40\] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions1, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. (@resnet)

Sunhee Hwang, Sungho Park, Dohyung Kim, Mirae Do, and Hyeran Byun Fairfacegan: Fairness-aware facial image-to-image translation *arXiv preprint arXiv:2012.00282*, 2020. **Abstract:** In this paper, we introduce FairFaceGAN, a fairness-aware facial Image-to-Image translation model, mitigating the problem of unwanted translation in protected attributes (e.g., gender, age, race) during facial attributes editing. Unlike existing models, FairFaceGAN learns fair representations with two separate latents - one related to the target attributes to translate, and the other unrelated to them. This strategy enables FairFaceGAN to separate the information about protected attributes and that of target attributes. It also prevents unwanted translation in protected attributes while target attributes editing. To evaluate the degree of fairness, we perform two types of experiments on CelebA dataset. First, we compare the fairness-aware classification performances when augmenting data by existing image translation methods and FairFaceGAN respectively. Moreover, we propose a new fairness metric, namely Frechet Protected Attribute Distance (FPAD), which measures how well protected attributes are preserved. Experimental results demonstrate that FairFaceGAN shows consistent improvements in terms of fairness over the existing image translation models. Further, we also evaluate image translation performances, where FairFaceGAN shows competitive results, compared to those of existing methods. (@hwang2020fairfacegan)

Ayush Jaiswal, Daniel Moyer, Greg Ver Steeg, Wael AbdAlmageed, and Premkumar Natarajan Invariant representations through adversarial forgetting In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pages 4272–4279, 2020. **Abstract:** We propose a novel approach to achieving invariance for deep neural networks in the form of inducing amnesia to unwanted factors of data through a new adversarial forgetting mechanism. We show that the forgetting mechanism serves as an information-bottleneck, which is manipulated by the adversarial training to learn invariance to unwanted factors. Empirical results show that the proposed framework achieves state-of-the-art performance at learning invariance in both nuisance and bias settings on a diverse collection of datasets and tasks. (@jaiswal2020invariant)

Zhimeng Stephen Jiang, Xiaotian Han, Hongye Jin, Guanchu Wang, Rui Chen, Na Zou, and Xia Hu Chasing fairness under distribution shift: A model weight perturbation approach *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Fairness in machine learning has attracted increasing attention in recent years. The fairness methods improving algorithmic fairness for in-distribution data may not perform well under distribution shifts. In this paper, we first theoretically demonstrate the inherent connection between distribution shift, data perturbation, and model weight perturbation. Subsequently, we analyze the sufficient conditions to guarantee fairness (i.e., low demographic parity) for the target dataset, including fairness for the source dataset, and low prediction difference between the source and target datasets for each sensitive attribute group. Motivated by these sufficient conditions, we propose robust fairness regularization (RFR) by considering the worst case within the model weight perturbation ball for each sensitive attribute group. We evaluate the effectiveness of our proposed RFR algorithm on synthetic and real distribution shifts across various datasets. Experimental results demonstrate that RFR achieves better fairness-accuracy trade-off performance compared with several baselines. The source code is available at \\}url{https://github.com/zhimengj0326/RFR_NeurIPS23}. (@jiang2024chasing)

Simona Ioana Juvina, Ana Antonia Neacșu, Jérôme Rony, Jean-Christophe Pesquet, Corneliu Burileanu, and Ismail Ben Ayed Training graph neural networks subject to a tight lipschitz constraint *Transactions on Machine Learning Research*. (@juvinatraining)

Faisal Kamiran and Toon Calders Classifying without discriminating In *2009 2nd international conference on computer, control and communication*, pages 1–6. IEEE, 2009. **Abstract:** Classification models usually make predictions on the basis of training data. If the training data is biased towards certain groups or classes of objects, e.g., there is racial discrimination towards black people, the learned model will also show discriminatory behavior towards that particular community. This partial attitude of the learned model may lead to biased outcomes when labeling future unlabeled data objects. Often, however, impartial classification results are desired or even required by law for future data objects in spite of having biased training data. In this paper, we tackle this problem by introducing a new classification scheme for learning unbiased models on biased training data. Our method is based on massaging the dataset by making the least intrusive modifications which lead to an unbiased dataset. On this modified dataset we then learn a non-discriminating classifier. The proposed method has been implemented and experimental results on a credit approval dataset show promising results: in all experiments our method is able to reduce the prejudicial behavior for future classification significantly without loosing too much predictive accuracy. (@kamiran2009classifying)

Faisal Kamiran and Toon Calders Data preprocessing techniques for classification without discrimination *Knowledge and Information Systems*, 33 (1): 1–33, 2012. **Abstract:** Recently, the following Discrimination-Aware Classification Problem was introduced: Suppose we are given training data that exhibit unlawful discrimination; e.g., toward sensitive attributes such as gender or ethnicity. The task is to learn a classifier that optimizes accuracy, but does not have this discrimination in its predictions on test data. This problem is relevant in many settings, such as when the data are generated by a biased decision process or when the sensitive attribute serves as a proxy for unobserved features. In this paper, we concentrate on the case with only one binary sensitive attribute and a two-class classification problem. We first study the theoretically optimal trade-off between accuracy and non-discrimination for pure classifiers. Then, we look at algorithmic solutions that preprocess the data to remove discrimination before a classifier is learned. We survey and extend our existing data preprocessing techniques, being suppression of the sensitive attribute, massaging the dataset by changing class labels, and reweighing or resampling the data to remove discrimination without relabeling instances. These preprocessing techniques have been implemented in a modified version of Weka and we present the results of experiments on real-life data. (@kamiran2012data)

Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh, and Jun Sakuma Fairness-aware classifier with prejudice remover regularizer In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*, pages 35–50. Springer, 2012. **Abstract:** The goal of fairness-aware classification is to categorize data while taking into account potential issues of fairness, discrimination, neutrality, and/or independence. For example, when applying data mining technologies to university admissions, admission criteria must be non-discriminatory and fair with regard to sensitive features, such as gender or race. In this context, such fairness can be formalized as statistical independence between classification results and sensitive features. The main purpose of this paper is to analyze this formal fairness in order to achieve better trade-offs between fairness and prediction accuracy, which is important for applying fairness-aware classifiers in practical use. We focus on a fairness-aware classifier, Calders and Verwer’s two-naive-Bayes (CV2NB) method, which has been shown to be superior to other classifiers in terms of fairness. We hypothesize that this superiority is due to the difference in types of independence. That is, because CV2NB achieves actual independence, rather than satisfying model-based independence like the other classifiers, it can account for model bias and a deterministic decision rule. We empirically validate this hypothesis by modifying two fairness-aware classifiers, a prejudice remover method and a reject option-based classification (ROC) method, so as to satisfy actual independence. The fairness of these two modified methods was drastically improved, showing the importance of maintaining actual independence, rather than model-based independence. We additionally extend an approach adopted in the ROC method so as to make it applicable to classifiers other than those with generative models, such as SVMs. (@kamishima2012fairness)

Ali Khodabandeh Yalabadi, Mehdi Yazdani-Jahromi, Niloofar Yousefi, Aida Tayebi, Sina Abdidizaji, and Ozlem Ozmen Garibay Fragxsitedti: Revealing responsible segments in drug-target interaction with transformer-driven interpretation In *International Conference on Research in Computational Molecular Biology*, pages 68–85. Springer, 2024. **Abstract:** Drug-Target Interaction (DTI) prediction is vital for drug discovery, yet challenges persist in achieving model interpretability and optimizing performance. We propose a novel transformer-based model, FragXsiteDTI, that aims to address these challenges in DTI prediction. Notably, FragXsiteDTI is the first DTI model to simultaneously leverage drug molecule fragments and protein pockets. Our information-rich representations for both proteins and drugs offer a detailed perspective on their interaction. Inspired by the Perceiver IO framework, our model features a learnable latent array, initially interacting with protein binding site embeddings using cross-attention and later refined through self-attention and used as a query to the drug fragments in the drug’s cross-attention transformer block. This learnable query array serves as a mediator and enables seamless information translation, preserving critical nuances in drug-protein interactions. Our computational results on three benchmarking datasets demonstrate the superior predictive power of our model over several state-of-the-art models. We also show the interpretability of our model in terms of the critical components of both target proteins and drug molecules within drug-target pairs. (@khodabandeh2024fragxsitedti)

O Deniz Kose and Yanning Shen Fairgat: Fairness-aware graph attention networks *ACM Transactions on Knowledge Discovery from Data*, 18 (7): 1–20, 2024. **Abstract:** Graphs can facilitate modeling various complex systems such as gene networks and power grids as well as analyzing the underlying relations within them. Learning over graphs has recently attracted increasing attention, particularly graph neural network (GNN)–based solutions, among which graph attention networks (GATs) have become one of the most widely utilized neural network structures for graph-based tasks. Although it is shown that the use of graph structures in learning results in the amplification of algorithmic bias, the influence of the attention design in GATs on algorithmic bias has not been investigated. Motivated by this, the present study first carries out a theoretical analysis in order to demonstrate the sources of algorithmic bias in GAT-based learning for node classification. Then, a novel algorithm, FairGAT, which leverages a fairness-aware attention design, is developed based on the theoretical findings. Experimental results on real-world networks demonstrate that FairGAT improves group fairness measures while also providing comparable utility to the fairness-aware baselines for node classification and link prediction. (@kose2024fairgat)

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang Deep learning face attributes in the wild In *Proceedings of International Conference on Computer Vision (ICCV)*, December 2015. **Abstract:** Predicting face attributes in the wild is challenging due to complex face variations. We propose a novel deep learning framework for attribute prediction in the wild. It cascades two CNNs, LNet and ANet, which are fine-tuned jointly with attribute tags, but pre-trained differently. LNet is pre-trained by massive general object categories for face localization, while ANet is pre-trained by massive face identities for attribute prediction. This framework not only outperforms the state-of-the-art with a large margin, but also reveals valuable facts on learning face representation. (1) It shows how the performances of face localization (LNet) and attribute prediction (ANet) can be improved by different pre-training strategies. (2) It reveals that although the filters of LNet are fine-tuned only with image-level attribute tags, their response maps over entire images have strong indication of face locations. This fact enables training LNet for face localization with only image-level annotations, but without face bounding boxes or landmarks, which are required by all attribute recognition works. (3) It also demonstrates that the high-level hidden neurons of ANet automatically discover semantic concepts after pre-training with massive face identities, and such concepts are significantly enriched after fine-tuning with attribute tags. Each attribute can be well explained with a sparse linear combination of these concepts. (@liu2015faceattributes)

Francesco Locatello, Gabriele Abbati, Thomas Rainforth, Stefan Bauer, Bernhard Schölkopf, and Olivier Bachem On the fairness of disentangled representations *Advances in Neural Information Processing Systems*, 32, 2019. **Abstract:** Recently there has been a significant interest in learning disentangled representations, as they promise increased interpretability, generalization to unseen scenarios and faster learning on downstream tasks. In this paper, we investigate the usefulness of different notions of disentanglement for improving the fairness of downstream prediction tasks based on representations. We consider the setting where the goal is to predict a target variable based on the learned representation of high-dimensional observations (such as images) that depend on both the target variable and an unobserved sensitive variable. We show that in this setting both the optimal and empirical predictions can be unfair, even if the target variable and the sensitive variable are independent. Analyzing the representations of more than 12600 trained state-of-the-art disentangled models, we observe that several disentanglement scores are consistently correlated with increased fairness, suggesting that disentanglement may be a useful property to encourage fairness when sensitive variables are not observed. (@locatello2019fairness)

Vishnu Suresh Lokhande, Aditya Kumar Akash, Sathya N Ravi, and Vikas Singh Fairalm: Augmented lagrangian method for training fair models with little regret In *European Conference on Computer Vision*, pages 365–381. Springer, 2020. **Abstract:** Algorithmic decision making based on computer vision and machine learning technologies continue to permeate our lives. But issues related to biases of these models and the extent to which they treat certain segments of the population unfairly, have led to concern in the general public. It is now accepted that because of biases in the datasets we present to the models, a fairness-oblivious training will lead to unfair models. An interesting topic is the study of mechanisms via which the de novo design or training of the model can be informed by fairness measures. Here, we study mechanisms that impose fairness concurrently while training the model. While existing fairness based approaches in vision have largely relied on training adversarial modules together with the primary classification/regression task, in an effort to remove the influence of the protected attribute or variable, we show how ideas based on well-known optimization concepts can provide a simpler alternative. In our proposed scheme, imposing fairness just requires specifying the protected attribute and utilizing our optimization routine. We provide a detailed technical analysis and present experiments demonstrating that various fairness measures from the literature can be reliably imposed on a number of training tasks in vision in a manner that is interpretable. (@lokhande2020fairalm)

Christos Louizos, Kevin Swersky, Yujia Li, Max Welling, and Richard Zemel The variational fair autoencoder *arXiv preprint arXiv:1511.00830*, 2015. **Abstract:** We investigate the problem of learning representations that are invariant to certain nuisance or sensitive factors of variation in the data while retaining as much of the remaining information as possible. Our model is based on a variational autoencoding architecture with priors that encourage independence between sensitive and latent factors of variation. Any subsequent processing, such as classification, can then be performed on this purged latent representation. To remove any remaining dependencies we incorporate an additional penalty term based on the "Maximum Mean Discrepancy" (MMD) measure. We discuss how these architectures can be efficiently trained on data and show in experiments that this method is more effective than previous work in removing unwanted sources of variation while maintaining informative latent representations. (@louizos2015variational)

Binh Thanh Luong, Salvatore Ruggieri, and Franco Turini k-nn as an implementation of situation testing for discrimination discovery and prevention In *Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 502–510, 2011. **Abstract:** With the support of the legally-grounded methodology of situation testing, we tackle the problems of discrimination discovery and prevention from a dataset of historical decisions by adopting a variant of k-NN classification. A tuple is labeled as discriminated if we can observe a significant difference of treatment among its neighbors belonging to a protected-by-law group and its neighbors not belonging to it. Discrimination discovery boils down to extracting a classification model from the labeled tuples. Discrimination prevention is tackled by changing the decision value for tuples labeled as discriminated before training a classifier. The approach of this paper overcomes legal weaknesses and technical limitations of existing proposals. (@luong2011k)

Ninareh Mehrabi, Umang Gupta, Fred Morstatter, Greg Ver Steeg, and Aram Galstyan Attributing fair decisions with attention interventions *arXiv preprint arXiv:2109.03952*, 2021. **Abstract:** The widespread use of Artificial Intelligence (AI) in consequential domains, such as healthcare and parole decision-making systems, has drawn intense scrutiny on the fairness of these methods. However, ensuring fairness is often insufficient as the rationale for a contentious decision needs to be audited, understood, and defended. We propose that the attention mechanism can be used to ensure fair outcomes while simultaneously providing feature attributions to account for how a decision was made. Toward this goal, we design an attention-based model that can be leveraged as an attribution framework. It can identify features responsible for both performance and fairness of the model through attention interventions and attention weight manipulation. Using this attribution framework, we then design a post-processing bias mitigation strategy and compare it with a suite of baselines. We demonstrate the versatility of our approach by conducting experiments on two distinct data types, tabular and textual. (@mehrabi2021attributing)

Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan A survey on bias and fairness in machine learning *ACM Computing Surveys (CSUR)*, 54 (6): 1–35, 2021. **Abstract:** With the widespread use of artificial intelligence (AI) systems and applications in our everyday lives, accounting for fairness has gained significant importance in designing and engineering of such systems. AI systems can be used in many sensitive environments to make important and life-changing decisions; thus, it is crucial to ensure that these decisions do not reflect discriminatory behavior toward certain groups or populations. More recently some work has been developed in traditional machine learning and deep learning that address such challenges in different subdomains. With the commercialization of these systems, researchers are becoming more aware of the biases that these applications can contain and are attempting to address them. In this survey, we investigated different real-world applications that have shown biases in various ways, and we listed different sources of biases that can affect AI applications. We then created a taxonomy for fairness definitions that machine learning researchers have defined to avoid the existing bias in AI systems. In addition to that, we examined different domains and subdomains in AI showing what researchers have observed with regard to unfair outcomes in the state-of-the-art methods and ways they have tried to address them. There are still many future directions and solutions that can be taken to mitigate the problem of bias in AI systems. We are hoping that this survey will motivate researchers to tackle these issues in the near future by observing existing work in their respective fields. (@mehrabi2021survey)

Aditya Krishna Menon and Robert C Williamson The cost of fairness in binary classification In *Conference on Fairness, Accountability and Transparency*, pages 107–118. PMLR, 2018. **Abstract:** Binary classiﬁers are often required to possess fairnessinthesenseofnotoverlydiscriminating with respect to a feature deemed sensitive, e.g. race. We study the inherent tradeoﬀs in learn- ing classiﬁers with a fairness constraint in the form of two questions: what is the best accu- racywecanexpectforagivenleveloffairness?, andwhatisthenatureoftheseoptimalfairness- awareclassiﬁers? Toanswerthesequestions,we providethreemaincontributions. First,werelate two existing fairness measures to cost-sensitive risks. Second, we show that for such cost- sensitive fairness measures, the optimal clas- siﬁer is an instance-dependent thresholding of the class-probability function. Third, we relate the tradeoﬀ between accuracy and fairness to the alignment between the target and sensitive features’ class-probabilities. A practical impli- cationofouranalysisisasimpleapproachtothe fairness-aware problem which involves suitably thresholding class-probability estimates. (@menon2018cost)

Athanasios Migdalas *When is a Stackelberg equilibrium Pareto optimum?* Springer, 1995. (@migdalas1995stackelberg)

Daniel Moyer, Shuyang Gao, Rob Brekelmans, Greg Ver Steeg, and Aram Galstyan Invariant representations without adversarial training *Advances in Neural Information Processing Systems, volume 31, 9084–9093*, 2018. **Abstract:** Representations of data that are invariant to changes in specified factors are useful for a wide range of problems: removing potential biases in prediction problems, controlling the effects of covariates, and disentangling meaningful factors of variation. Unfortunately, learning representations that exhibit invariance to arbitrary nuisance factors yet remain useful for other tasks is challenging. Existing approaches cast the trade-off between task performance and invariance in an adversarial way, using an iterative minimax optimization. We show that adversarial training is unnecessary and sometimes counter-productive; we instead cast invariant representation learning as a single information-theoretic objective that can be directly optimized. We demonstrate that this approach matches or exceeds performance of state-of-the-art adversarial approaches for learning fair representations and for generative modeling with controllable transformations. (@moyer2018invariant)

Bryan Perozzi, Rami Al-Rfou, and Steven Skiena Deepwalk: Online learning of social representations In *Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 701–710, 2014. **Abstract:** We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk’s latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk’s representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk’s representations are able to outperform all baseline methods while using 60% less training data. DeepWalk is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection. (@perozzi2014deepwalk)

Valerio Perrone, Michele Donini, Muhammad Bilal Zafar, Robin Schmucker, Krishnaram Kenthapadi, and Cédric Archambeau Fair bayesian optimization In *Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society*, pages 854–863, 2021. **Abstract:** Given the increasing importance of machine learning (ML) in our lives, several algorithmic fairness techniques have been proposed to mitigate biases in the outcomes of the ML models. However, most of these techniques are specialized to cater to a single family of ML models and a specific definition of fairness, limiting their adaptibility in practice. We introduce a general constrained Bayesian optimization (BO) framework to optimize the performance of any ML model while enforcing one or multiple fairness constraints. BO is a model-agnostic optimization method that has been successfully applied to automatically tune the hyperparameters of ML models. We apply BO with fairness constraints to a range of popular models, including random forests, gradient boosting, and neural networks, showing that we can obtain accurate and fair solutions by acting solely on the hyperparameters. We also show empirically that our approach is competitive with specialized techniques that enforce model-specific fairness constraints, and outperforms preprocessing methods that learn fair representations of the input data. Moreover, our method can be used in synergy with such specialized fairness techniques to tune their hyperparameters. Finally, we study the relationship between fairness and the hyperparameters selected by BO. We observe a correlation between regularization and unbiased models, explaining why acting on the hyperparameters leads to ML models that generalize well and are fair. (@perrone2021fair)

Dana Pessach and Erez Shmueli Algorithmic fairness *arXiv preprint arXiv:2001.09784*, 2020. **Abstract:** An increasing number of decisions regarding the daily lives of human beings are being controlled by artificial intelligence (AI) algorithms in spheres ranging from healthcare, transportation, and education to college admissions, recruitment, provision of loans and many more realms. Since they now touch on many aspects of our lives, it is crucial to develop AI algorithms that are not only accurate but also objective and fair. Recent studies have shown that algorithmic decision-making may be inherently prone to unfairness, even when there is no intention for it. This paper presents an overview of the main concepts of identifying, measuring and improving algorithmic fairness when using AI algorithms. The paper begins by discussing the causes of algorithmic bias and unfairness and the common definitions and measures for fairness. Fairness-enhancing mechanisms are then reviewed and divided into pre-process, in-process and post-process mechanisms. A comprehensive comparison of the mechanisms is then conducted, towards a better understanding of which mechanisms should be used in different scenarios. The paper then describes the most commonly used fairness-related datasets in this field. Finally, the paper ends by reviewing several emerging research sub-fields of algorithmic fairness. (@pessach2020algorithmic)

Novi Quadrianto, Viktoriia Sharmanska, and Oliver Thomas Discovering fair representations in the data domain In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8227–8236, 2019. **Abstract:** Interpretability and fairness are critical in computer vision and machine learning applications, in particular when dealing with human outcomes, e.g. inviting or not inviting for a job interview based on application materials that may include photographs. One promising direction to achieve fairness is by learning data representations that remove the semantics of protected characteristics, and are therefore able to mitigate unfair outcomes. All available models however learn latent embeddings which comes at the cost of being uninterpretable. We propose to cast this problem as data-to-data translation, i.e. learning a mapping from an input domain to a fair target domain, where a fairness definition is being enforced. Here the data domain can be images, or any tabular data representation. This task would be straightforward if we had fair target data available, but this is not the case. To overcome this, we learn a highly unconstrained mapping by exploiting statistics of residuals – the difference between input data and its translated version – and the protected characteristics. When applied to the CelebA dataset of face images with gender attribute as the protected characteristic, our model enforces equality of opportunity by adjusting the eyes and lips regions. Intriguingly, on the same dataset we arrive at similar conclusions when using semantic attribute representations of images for translation. On face images of the recent DiF dataset, with the same gender attribute, our method adjusts nose regions. In the Adult income dataset, also with protected gender attribute, our model achieves equality of opportunity by, among others, obfuscating the wife and husband relationship. Analyzing those systematic changes will allow us to scrutinize the interplay of fairness criterion, chosen protected characteristics, and prediction performance. (@quadrianto2019discovering)

Tahleen Rahman, Bartlomiej Surma, Michael Backes, and Yang Zhang Fairwalk: Towards fair graph embedding . **Abstract:** Graph embeddings have gained huge popularity in the recent years as a powerful tool to analyze social networks. However, no prior works have studied potential bias issues inherent within graph embedding. In this paper, we make a first attempt in this direction. In particular, we concentrate on the fairness of node2vec, a popular graph embedding method. Our analyses on two real-world datasets demonstrate the existence of bias in node2vec when used for friendship recommendation. We, therefore, propose a fairness-aware embedding method, namely Fairwalk, which extends node2vec. Experimental results demonstrate that Fairwalk reduces bias under multiple fairness metrics while still preserving the utility. (@rahman2019fairwalk)

Amirarsalan Rajabi and Ozlem Ozmen Garibay Tabfairgan: Fair tabular data generation with generative adversarial networks *arXiv preprint arXiv:2109.00666*, 2021. **Abstract:** With the increasing reliance on automated decision making, the issue of algorithmic fairness has gained increasing importance. In this paper, we propose a Generative Adversarial Network for tabular data generation. The model includes two phases of training. In the first phase, the model is trained to accurately generate synthetic data similar to the reference dataset. In the second phase we modify the value function to add fairness constraint, and continue training the network to generate data that is both accurate and fair. We test our results in both cases of unconstrained, and constrained fair data generation. In the unconstrained case, i.e. when the model is only trained in the first phase and is only meant to generate accurate data following the same joint probability distribution of the real data, the results show that the model beats state-of-the-art GANs proposed in the literature to produce synthetic tabular data. Also, in the constrained case in which the first phase of training is followed by the second phase, we train the network and test it on four datasets studied in the fairness literature and compare our results with another state-of-the-art pre-processing method, and present the promising results that it achieves. Comparing to other studies utilizing GANs for fair data generation, our model is comparably more stable by using only one critic, and also by avoiding major problems of original GAN model, such as mode-dropping and non-convergence, by implementing a Wasserstein GAN. (@rajabi2021tabfairgan)

Amirarsalan Rajabi, Mehdi Yazdani-Jahromi, Ozlem Ozmen Garibay, and Gita Sukthankar Through a fair looking-glass: mitigating bias in image datasets *arXiv preprint arXiv:2209.08648*, 2022. **Abstract:** With the recent growth in computer vision applications, the question of how fair and unbiased they are has yet to be explored. There is abundant evidence that the bias present in training data is reflected in the models, or even amplified. Many previous methods for image dataset de-biasing, including models based on augmenting datasets, are computationally expensive to implement. In this study, we present a fast and effective model to de-bias an image dataset through reconstruction and minimizing the statistical dependence between intended variables. Our architecture includes a U-net to reconstruct images, combined with a pre-trained classifier which penalizes the statistical dependence between target attribute and the protected attribute. We evaluate our proposed model on CelebA dataset, compare the results with a state-of-the-art de-biasing method, and show that the model achieves a promising fairness-accuracy combination. (@rajabi2022through)

Vikram V Ramaswamy, Sunnie SY Kim, and Olga Russakovsky Fair attribute classification through latent space de-biasing In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 9301–9310, 2021. **Abstract:** Fairness in visual recognition is becoming a prominent and critical topic of discussion as recognition systems are deployed at scale in the real world. Models trained from data in which target labels are correlated with protected attributes (e.g., gender, race) are known to learn and exploit those correlations. In this work, we introduce a method for training accurate target classifiers while mitigating biases that stem from these correlations. We use GANs to generate realistic-looking images, and perturb these images in the underlying latent space to generate training data that is balanced for each protected attribute. We augment the original dataset with this generated data, and empirically demonstrate that target classifiers trained on the augmented dataset exhibit a number of both quantitative and qualitative benefits. We conduct a thorough evaluation across multiple target labels and protected attributes in the CelebA dataset, and provide an in-depth analysis and comparison to existing literature in the space. Code can be found at https://github.com/princetonvisualai/gan-debiasing. (@ramaswamy2021fair)

Proteek Chandan Roy and Vishnu Naresh Boddeti Mitigating information leakage in image representations: A maximum entropy approach In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2586–2594, 2019. **Abstract:** Image recognition systems have demonstrated tremendous progress over the past few decades thanks, in part, to our ability of learning compact and robust representations of images. As we witness the wide spread adoption of these systems, it is imperative to consider the problem of unintended leakage of information from an image representation, which might compromise the privacy of the data owner. This paper investigates the problem of learning an image representation that minimizes such leakage of user information. We formulate the problem as an adversarial non-zero sum game of finding a good embedding function with two competing goals: to retain as much task dependent discriminative image information as possible, while simultaneously minimizing the amount of information, as measured by entropy, about other sensitive attributes of the user. We analyze the stability and convergence dynamics of the proposed formulation using tools from non-linear systems theory and compare to that of the corresponding adversarial zero-sum game formulation that optimizes likelihood as a measure of information content. Numerical experiments on UCI, Extended Yale B, CIFAR-10 and CIFAR-100 datasets indicate that our proposed approach is able to learn image representations that exhibit high task performance while mitigating leakage of predefined sensitive information. (@roy2019mitigating)

Prasanna Sattigeri, Samuel C Hoffman, Vijil Chenthamarakshan, and Kush R Varshney Fairness gan: Generating datasets with fairness properties using a generative adversarial network *IBM Journal of Research and Development*, 63 (4/5): 3–1, 2019. **Abstract:** We introduce the Fairness GAN (generative adversarial network), an approach for generating a dataset that is plausibly similar to a given multimedia dataset, but is more fair with respect to protected attributes in decision making. We propose a novel auxiliary classifier GAN that strives for demographic parity or equality of opportunity and show empirical results on several datasets, including the CelebFaces Attributes (CelebA) dataset, the Quick, Draw! dataset, and a dataset of soccer player images and the offenses for which they were called. The proposed formulation is well suited to absorbing unlabeled data; we leverage this to augment the soccer dataset with the much larger CelebA dataset. The methodology tends to improve demographic parity and equality of opportunity while generating plausible images. (@sattigeri2019fairness)

Viktoriia Sharmanska, Lisa Anne Hendricks, Trevor Darrell, and Novi Quadrianto Contrastive examples for addressing the tyranny of the majority *arXiv preprint arXiv:2004.06524*, 2020. **Abstract:** Computer vision algorithms, e.g. for face recognition, favour groups of individuals that are better represented in the training data. This happens because of the generalization that classifiers have to make. It is simpler to fit the majority groups as this fit is more important to overall error. We propose to create a balanced training dataset, consisting of the original dataset plus new data points in which the group memberships are intervened, minorities become majorities and vice versa. We show that current generative adversarial networks are a powerful tool for learning these data points, called contrastive examples. We experiment with the equalized odds bias measure on tabular data as well as image data (CelebA and Diversity in Faces datasets). Contrastive examples allow us to expose correlations between group membership and other seemingly neutral features. Whenever a causal graph is available, we can put those contrastive examples in the perspective of counterfactuals. (@sharmanska2020contrastive)

Ankur Sinha, Pekka Malo, and Kalyanmoy Deb A review on bilevel optimization: From classical to evolutionary approaches and applications *IEEE transactions on evolutionary computation*, 22 (2): 276–295, 2017. **Abstract:** Bilevel optimization is defined as a mathematical program, where an optimization problem contains another optimization problem as a constraint. These problems have received significant attention from the mathematical programming community. Only limited work exists on bilevel problems using evolutionary computation techniques; however, recently there has been an increasing interest due to the proliferation of practical applications and the potential of evolutionary algorithms in tackling these problems. This paper provides a comprehensive review on bilevel optimization from the basic principles to solution strategies; both classical and evolutionary. A number of potential application problems are also discussed. To offer the readers insights on the prominent developments in the field of bilevel optimization, we have performed an automated text-analysis of an extended list of papers published on bilevel optimization to date. This paper should motivate evolutionary computation researchers to pay more attention to this practical yet challenging area. (@sinha2017review)

Dylan Slack, Sorelle Friedler, and Emile Givental Fair meta-learning: learning how to learn fairly *arXiv preprint arXiv:1911.04336*, 2019. **Abstract:** Data sets for fairness relevant tasks can lack examples or be biased according to a specific label in a sensitive attribute. We demonstrate the usefulness of weight based meta-learning approaches in such situations. For models that can be trained through gradient descent, we demonstrate that there are some parameter configurations that allow models to be optimized from a few number of gradient steps and with minimal data which are both fair and accurate. To learn such weight sets, we adapt the popular MAML algorithm to Fair-MAML by the inclusion of a fairness regularization term. In practice, Fair-MAML allows practitioners to train fair machine learning models from only a few examples when data from related tasks is available. We empirically exhibit the value of this technique by comparing to relevant baselines. (@slack2019fair)

Jiaming Song, Pratyusha Kalluri, Aditya Grover, Shengjia Zhao, and Stefano Ermon Learning controllable fair representations In *The 22nd International Conference on Artificial Intelligence and Statistics*, pages 2164–2173. PMLR, 2019. **Abstract:** Learning data representations that are transferable and are fair with respect to certain protected attributes is crucial to reducing unfair decisions while preserving the utility of the data. We propose an information-theoretically motivated objective for learning maximally expressive representations subject to fairness constraints. We demonstrate that a range of existing approaches optimize approximations to the Lagrangian dual of our objective. In contrast to these existing approaches, our objective allows the user to control the fairness of the representations by specifying limits on unfairness. Exploiting duality, we introduce a method that optimizes the model parameters as well as the expressiveness-fairness trade-off. Empirical evidence suggests that our proposed method can balance the trade-off between multiple notions of fairness and achieves higher expressiveness at a lower computational cost. (@song2019learning)

Lubos Takac and Michal Zabovsky Data analysis in public social networks In *International scientific conference and international workshop present day trends of innovations*, volume 1. Present Day Trends of Innovations Lamza Poland, 2012. **Abstract:** Public social networks affect significant number of people with different professional and personal background. Presented paper deals with data analysis and in addition with safety of information managed by online social networks. We will show methods for data analysis in social networks based on its scale-free characteristics. Experimental results will be discussed for the biggest social network in Slovakia which is popular for more than 10 years. 1 , big data 2 , scale-free networks 3 , data security 4 , graph theory 5 (@takac2012data)

Aida Tayebi, Niloofar Yousefi, Mehdi Yazdani-Jahromi, Elayaraja Kolanthai, Craig J Neal, Sudipta Seal, and Ozlem Ozmen Garibay Unbiaseddti: Mitigating real-world bias of drug-target interaction prediction by using deep ensemble-balanced learning *Molecules*, 27 (9): 2980, 2022. **Abstract:** Drug-target interaction (DTI) prediction through in vitro methods is expensive and time-consuming. On the other hand, computational methods can save time and money while enhancing drug discovery efficiency. Most of the computational methods frame DTI prediction as a binary classification task. One important challenge is that the number of negative interactions in all DTI-related datasets is far greater than the number of positive interactions, leading to the class imbalance problem. As a result, a classifier is trained biased towards the majority class (negative class), whereas the minority class (interacting pairs) is of interest. This class imbalance problem is not widely taken into account in DTI prediction studies, and the few previous studies considering balancing in DTI do not focus on the imbalance issue itself. Additionally, they do not benefit from deep learning models and experimental validation. In this study, we propose a computational framework along with experimental validations to predict drug-target interaction using an ensemble of deep learning models to address the class imbalance problem in the DTI domain. The objective of this paper is to mitigate the bias in the prediction of DTI by focusing on the impact of balancing and maintaining other involved parameters at a constant value. Our analysis shows that the proposed model outperforms unbalanced models with the same architecture trained on the BindingDB both computationally and experimentally. These findings demonstrate the significance of balancing, which reduces the bias towards the negative class and leads to better performance. It is important to note that leaning on computational results without experimentally validating them and by relying solely on AUROC and AUPRC metrics is not credible, particularly when the testing set remains unbalanced. (@tayebi2022unbiaseddti)

Aida Tayebi, Mehdi Yazdani-Jahromi, Ali Khodabandeh Yalabadi, Niloofar Yousefi, and Ozlem Ozmen Garibay Learning fair representations: Mitigating statistical dependencies In *International Conference on Human-Computer Interaction*, pages 105–115. Springer, 2024. **Abstract:** We introduce a method, MMD-B-Fair, to learn fair representations of data via kernel two-sample testing. We find neural features of our data where a maximum mean discrepancy (MMD) test cannot distinguish between representations of different sensitive groups, while preserving information about the target attributes. Minimizing the power of an MMD test is more difficult than maximizing it (as done in previous work), because the test threshold’s complex behavior cannot be simply ignored. Our method exploits the simple asymptotics of block testing schemes to efficiently find fair representations without requiring complex adversarial optimization or generative modelling schemes widely used by existing work on fair representation learning. We evaluate our approach on various datasets, showing its ability to “hide” information about sensitive attributes, and its effectiveness in downstream transfer tasks. (@tayebi2024learning)

Angelina Wang, Alexander Liu, Ryan Zhang, Anat Kleiman, Leslie Kim, Dora Zhao, Iroha Shirai, Arvind Narayanan, and Olga Russakovsky Revise: A tool for measuring and mitigating bias in visual datasets *International Journal of Computer Vision*, pages 1–21, 2022. **Abstract:** Machine learning models are known to per- petuate and even amplify the biases present in the data. However, these data biases frequently do not become apparent until after the models are deployed. Our work tackles this issue and enables the preemptive analysis of large-scale datasets. REVISE (REvealing VIsual bi- aSEs) is a tool that assists in the investigation of a visual dataset, surfacing potential biases along three dimensions: (1) object-based, (2) person-based, and (3) geography-based. Object-based biases relate to the size, context, or diversity of the depicted objects. Person- based metrics focus on analyzing the portrayal of peo- ple within the dataset. Geography-based analyses con- sider the representation of di erent geographic loca- tions. These three dimensions are deeply intertwined in how they interact to bias a dataset, and REVISE sheds light on this; the responsibility then lies with the user to consider the cultural and historical con- text, and to determine which of the revealed biases may be problematic. The tool further assists the user by suggesting actionable steps that may be taken to mitigate the revealed biases. Overall, the key aim of our work is to tackle the machine learning bias prob- lem early in the pipeline. REVISE is available at https: //github.com/princetonvisualai/revise-tool. (@wang2022revise)

Mei Wang and Weihong Deng Mitigate bias in face recognition using skewness-aware reinforcement learning *arXiv preprint arXiv:1911.10692*, 2019. **Abstract:** Racial equality is an important theme of international human rights law, but it has been largely obscured when the overall face recognition accuracy is pursued blindly. More facts indicate racial bias indeed degrades the fairness of recognition system and the error rates on non-Caucasians are usually much higher than Caucasians. To encourage fairness, we introduce the idea of adaptive margin to learn balanced performance for different races based on large margin losses. A reinforcement learning based race balance network (RL-RBN) is proposed. We formulate the process of finding the optimal margins for non-Caucasians as a Markov decision process and employ deep Q-learning to learn policies for an agent to select appropriate margin by approximating the Q-value function. Guided by the agent, the skewness of feature scatter between races can be reduced. Besides, we provide two ethnicity aware training datasets, called BUPT-Globalface and BUPT-Balancedface dataset, which can be utilized to study racial bias from both data and algorithm aspects. Extensive experiments on RFW database show that RL-RBN successfully mitigates racial bias and learns more balanced performance for different races. (@wang2019mitigate)

Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, and Yaohai Huang Racial faces in the wild: Reducing racial bias by information maximization adaptation network In *Proceedings of the ieee/cvf international conference on computer vision*, pages 692–702, 2019. **Abstract:** Racial bias is an important issue in biometric, but has not been thoroughly studied in deep face recognition. In this paper, we first contribute a dedicated dataset called Racial Faces in-the-Wild (RFW) database, on which we firmly validated the racial bias of four commercial APIs and four state-of-the-art (SOTA) algorithms. Then, we further present the solution using deep unsupervised domain adaptation and propose a deep information maximization adaptation network (IMAN) to alleviate this bias by using Caucasian as source domain and other races as target domains. This unsupervised method simultaneously aligns global distribution to decrease race gap at domain-level, and learns the discriminative target representations at cluster level. A novel mutual information loss is proposed to further enhance the discriminative ability of network output without label information. Extensive experiments on RFW, GBU, and IJB-A databases show that IMAN successfully learns features that generalize well across different races and across different databases. (@wang2019racial)

Tianlu Wang, Jieyu Zhao, Mark Yatskar, Kai-Wei Chang, and Vicente Ordonez Balanced datasets are not enough: Estimating and mitigating gender bias in deep image representations In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5310–5319, 2019. **Abstract:** In this work, we present a framework to measure and mitigate intrinsic biases with respect to protected variables -such as gender- in visual recognition tasks. We show that trained models significantly amplify the association of target labels with gender beyond what one would expect from biased datasets. Surprisingly, we show that even when datasets are balanced such that each label co-occurs equally with each gender, learned models amplify the association between labels and gender, as much as if data had not been balanced! To mitigate this, we adopt an adversarial approach to remove unwanted features corresponding to protected variables from intermediate representations in a deep neural network - and provide a detailed analysis of its effectiveness. Experiments on two datasets: the COCO dataset (objects), and the imSitu dataset (actions), show reductions in gender bias amplification while maintaining most of the accuracy of the original models. (@wang2019balanced)

Tongxin Wang, Zhengming Ding, Wei Shao, Haixu Tang, and Kun Huang Towards fair cross-domain adaptation via generative learning In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 454–463, 2021. **Abstract:** Domain Adaptation (DA) targets at adapting a model trained over the well-labeled source domain to the unlabeled target domain lying in different distributions. Existing DA normally assumes the well-labeled source domain is class-wise balanced, which means the size per source class is relatively similar. However, in real-world applications, labeled samples for some categories in the source domain could be extremely few due to the difficulty of data collection and annotation, which leads to decreasing performance over target domain on those few-shot categories. To perform fair cross-domain adaptation and boost the performance on these minority categories, we develop a novel Generative Few-shot Cross-domain Adaptation (GFCA) algorithm for fair cross-domain classification. Specifically, generative feature augmentation is explored to synthesize effective training data for few-shot source classes, while effective cross-domain alignment aims to adapt knowledge from source to facilitate the target learning. Experimental results on two large cross-domain visual datasets demonstrate the effectiveness of our proposed method on improving both few-shot and overall classification accuracy comparing with the state-of-the-art DA approaches. (@wang2021towards)

Xingkun Xu, Yuge Huang, Pengcheng Shen, Shaoxin Li, Jilin Li, Feiyue Huang, Yong Li, and Zhen Cui Consistent instance false positive improves fairness in face recognition In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 578–586, 2021. **Abstract:** Demographic bias is a significant challenge in practical face recognition systems. Existing methods heavily rely on accurate demographic annotations. However, such annotations are usually unavailable in real scenarios. Moreover, these methods are typically designed for a specific demographic group and are not general enough. In this paper, we propose a false positive rate penalty loss, which mitigates face recognition bias by increasing the consistency of instance False Positive Rate (FPR). Specifically, we first define the instance FPR as the ratio between the number of the non-target similarities above a unified threshold and the total number of the non-target similarities. The unified threshold is estimated for a given total FPR. Then, an additional penalty term, which is in proportion to the ratio of instance FPR overall FPR, is introduced into the denominator of the softmax-based loss. The larger the instance FPR, the larger the penalty. By such unequal penalties, the instance FPRs are supposed to be consistent. Compared with the previous debiasing methods, our method requires no demographic annotations. Thus, it can mitigate the bias among demographic groups divided by various attributes, and these attributes are not needed to be previously predefined during training. Extensive experimental results on popular benchmarks demonstrate the superiority of our method over state-of-the-art competitors. Code and pre-trained models are available at https://github.com/xkx0430/FairnessFR. (@xu2021consistent)

Kaiyu Yang, Klint Qinami, Li Fei-Fei, Jia Deng, and Olga Russakovsky Towards fairer datasets: Filtering and balancing the distribution of the people subtree in the imagenet hierarchy In *Proceedings of the 2020 conference on fairness, accountability, and transparency*, pages 547–558, 2020. **Abstract:** Computer vision technology is being used by many but remains representative of only a few. People have reported misbehavior of computer vision models, including offensive prediction results and lower performance for underrepresented groups. Current computer vision models are typically developed using datasets consisting of manually annotated images or videos; the data and label distributions in these datasets are critical to the models’ behavior. In this paper, we examine ImageNet, a large-scale ontology of images that has spurred the development of many modern computer vision methods. We consider three key factors within the person subtree of ImageNet that may lead to problematic behavior in downstream computer vision technology: (1) the stagnant concept vocabulary of WordNet, (2) the attempt at exhaustive illustration of all categories with images, and (3) the inequality of representation in the images within concepts. We seek to illuminate the root causes of these concerns and take the first steps to mitigate them constructively. (@yang2020towards)

Mehdi Yazdani-Jahromi, Niloofar Yousefi, Aida Tayebi, Elayaraja Kolanthai, Craig J Neal, Sudipta Seal, and Ozlem Ozmen Garibay Attentionsitedti: an interpretable graph-based model for drug-target interaction prediction using nlp sentence-level relation classification *Briefings in Bioinformatics*, 23 (4): bbac272, 2022. **Abstract:** In this study, we introduce an interpretable graph-based deep learning prediction model, AttentionSiteDTI, which utilizes protein binding sites along with a self-attention mechanism to address the problem of drug-target interaction prediction. Our proposed model is inspired by sentence classification models in the field of Natural Language Processing, where the drug-target complex is treated as a sentence with relational meaning between its biochemical entities a.k.a. protein pockets and drug molecule. AttentionSiteDTI enables interpretability by identifying the protein binding sites that contribute the most toward the drug-target interaction. Results on three benchmark datasets show improved performance compared with the current state-of-the-art models. More significantly, unlike previous studies, our model shows superior performance, when tested on new proteins (i.e. high generalizability). Through multidisciplinary collaboration, we further experimentally evaluate the practical potential of our proposed approach. To achieve this, we first computationally predict the binding interactions between some candidate compounds and a target protein, then experimentally validate the binding interactions for these pairs in the laboratory. The high agreement between the computationally predicted and experimentally observed (measured) drug-target interactions illustrates the potential of our method as an effective pre-screening tool in drug repurposing applications. (@yazdani2022attentionsitedti)

Mikhail Yurochkin, Amanda Bower, and Yuekai Sun Training individually fair ml models with sensitive subspace robustness *arXiv preprint arXiv:1907.00020*, 2019. **Abstract:** We consider training machine learning models that are fair in the sense that their performance is invariant under certain sensitive perturbations to the inputs. For example, the performance of a resume screening system should be invariant under changes to the gender and/or ethnicity of the applicant. We formalize this notion of algorithmic fairness as a variant of individual fairness and develop a distributionally robust optimization approach to enforce it during training. We also demonstrate the effectiveness of the approach on two ML tasks that are susceptible to gender and racial biases. (@yurochkin2019training)

Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, and Krishna P Gummadi Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment In *Proceedings of the 26th international conference on world wide web*, pages 1171–1180, 2017. **Abstract:** Automated data-driven decision making systems are increasingly being used to assist, or even replace humans in many settings. These systems function by learning from historical decisions, often taken by humans. In order to maximize the utility of these systems (or, classifiers), their training involves minimizing the errors (or, misclassifications) over the given historical data. However, it is quite possible that the optimally trained classifier makes decisions for people belonging to different social groups with different misclassification rates (e.g., misclassification rates for females are higher than for males), thereby placing these groups at an unfair disadvantage. To account for and avoid such unfairness, in this paper, we introduce a new notion of unfairness, disparate mistreatment, which is defined in terms of misclassification rates. We then propose intuitive measures of disparate mistreatment for decision boundary-based classifiers, which can be easily incorporated into their formulation as convex-concave constraints. Experiments on synthetic as well as real world datasets show that our methodology is effective at avoiding disparate mistreatment, often at a small cost in terms of accuracy. (@zafar2017fairness)

Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rogriguez, and Krishna P Gummadi Fairness constraints: Mechanisms for fair classification In *Artificial Intelligence and Statistics*, pages 962–970. PMLR, 2017. **Abstract:** Algorithmic decision making systems are ubiquitous across a wide variety of online as well as offline services. These systems rely on complex learning methods and vast amounts of data to optimize the service functionality, satisfaction of the end user and profitability. However, there is a growing concern that these automated decisions can lead, even in the absence of intent, to a lack of fairness, i.e., their outcomes can disproportionately hurt (or, benefit) particular groups of people sharing one or more sensitive attributes (e.g., race, sex). In this paper, we introduce a flexible mechanism to design fair classifiers by leveraging a novel intuitive measure of decision boundary (un)fairness. We instantiate this mechanism with two well-known classifiers, logistic regression and support vector machines, and show on real-world data that our mechanism allows for a fine-grained control on the degree of fairness, often at a small cost in terms of accuracy. (@zafar2017fairness_a)

Rich Zemel, Yu Wu, Kevin Swersky, Toni Pitassi, and Cynthia Dwork Learning fair representations In *International conference on machine learning*, pages 325–333. PMLR, 2013. **Abstract:** We propose a learning algorithm for fair classification that achieves both group fairness (the proportion of members in a protected group receiving positive classification is identical to the proportion in the population as a whole), and individual fairness (similar individuals should be treated similarly). We formulate fairness as an optimization problem of finding a good representation of the data with two competing goals: to encode the data as well as possible, while simultaneously obfuscating any information about membership in the protected group. We show positive results of our algorithm relative to other known techniques, on three datasets. Moreover, we demonstrate several advantages to our approach. First, our intermediate representation can be used for other classification tasks (i.e., transfer learning is possible); secondly, we take a step toward learning a distance metric which can find important dimensions of the data for classification. (@zemel2013learning)

Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell Mitigating unwanted biases with adversarial learning In *Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society*, pages 335–340, 2018. **Abstract:** Machine learning is a tool for building models that accurately represent input training data. When undesired biases concerning demographic groups are in the training data, well-trained models will reflect those biases. We present a framework for mitigating such biases by including a variable for the group of interest and simultaneously learning a predictor and an adversary. The input to the network X, here text or census data, produces a prediction Y, such as an analogy completion or income bracket, while the adversary tries to model a protected variable Z, here gender or zip code. The objective is to maximize the predictor’s ability to predict Y while minimizing the adversary’s ability to predict Z. Applied to analogy completion, this method results in accurate predictions that exhibit less evidence of stereotyping Z. When applied to a classification task using the UCI Adult (Census) Dataset, it results in a predictive model that does not lose much accuracy while achieving very close to equality of odds (Hardt, et al., 2016). The method is flexible and applicable to multiple definitions of fairness as well as a wide range of gradient-based learning models, including both regression and classification tasks. (@zhang2018mitigating)

Lu Zhang, Yongkai Wu, and Xintao Wu Achieving non-discrimination in data release In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pages 1335–1344, 2017. **Abstract:** Discrimination discovery and prevention/removal are increasingly important tasks in data mining. Discrimination discovery aims to unveil discriminatory practices on the protected attribute (e.g., gender) by analyzing the dataset of historical decision records, and discrimination prevention aims to remove discrimination by modifying the biased data before conducting predictive analysis. In this paper, we show that the key to discrimination discovery and prevention is to find the meaningful partitions that can be used to provide quantitative evidences for the judgment of discrimination. With the support of the causal graph, we present a graphical condition for identifying a meaningful partition. Based on that, we develop a simple criterion for the claim of non-discrimination, and propose discrimination removal algorithms which accurately remove discrimination while retaining good data utility. Experiments using real datasets show the effectiveness of our approaches. (@zhang2017achieving)

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang Men also like shopping: Reducing gender bias amplification using corpus-level constraints *arXiv preprint arXiv:1707.09457*, 2017. **Abstract:** Language is increasingly being used to define rich visual recognition problems with supporting image collections sourced from the web. Structured prediction models are used in these tasks to take advantage of correlations between co-occurring labels and visual input but risk inadvertently encoding social biases found in web corpora. In this work, we study data and models associated with multilabel object classification and visual semantic role labeling. We find that (a) datasets for these tasks contain significant gender bias and (b) models trained on these datasets further amplify existing bias. For example, the activity cooking is over 33% more likely to involve females than males in a training set, and a trained model further amplifies the disparity to 68% at test time. We propose to inject corpus-level constraints for calibrating existing structured prediction models and design an algorithm based on Lagrangian relaxation for collective inference. Our method results in almost no performance loss for the underlying recognition task but decreases the magnitude of bias amplification by 47.5% and 40.5% for multilabel classification and visual semantic role labeling, respectively. (@zhao2017men)

Dongmian Zou, Radu Balan, and Maneesh Singh On lipschitz bounds of general convolutional neural networks *IEEE Transactions on Information Theory*, 66 (3): 1738–1759, 2019. **Abstract:** Many convolutional neural networks (CNN’s) have a feed-forward structure. In this paper, we model a general framework for analyzing the Lipschitz bounds of CNN’s and propose a linear program that estimates these bounds. Several CNN’s, including the scattering networks, the AlexNet and the GoogleNet, are studied numerically. In these practical numerical examples, estimations of local Lipschitz bounds are compared to these theoretical bounds. Based on the Lipschitz bounds, we next establish concentration inequalities for the output distribution with respect to a stationary random input signal. The Lipschitz bound is further used to perform nonlinear discriminant analysis that measures the separation between features of different classes. (@zou2019lipschitz)

</div>

# Appendix / supplemental material

## Theoretical proofs [ap:theory]

Assume that the following assumptions are satisfied:

<div id="assum:strict_convexity:ap" class="assumption" markdown="1">

**Assumption 11**. The primary loss function \\(f(\theta_p, \theta_s)(x)\\) is strictly convex in a neighborhood of its local optimum. That is, for any \\(\theta_p, \theta_p' \in \Theta_p\\) and fixed \\(\theta_s \in \Theta_s\\), if \\(\theta_p \neq \theta_p'\\) and \\(\theta_p, \theta_p'\\) are sufficiently close to the local optimum \\(\theta_p^*\\), then \\[f(\lambda \theta_p + (1-\lambda)\theta_p', \theta_s) < \lambda f(\theta_p, \theta_s) + (1-\lambda)f(\theta_p', \theta_s)\\] for any \\(\lambda \in (0,1)\\).

</div>

<div id="assum:small_steps:ap" class="assumption" markdown="1">

**Assumption 12**. \\(|\theta_s - \hat{\theta}_s| \leq \epsilon\\), where \\(\epsilon\\) is sufficiently small, i.e., the steps of the secondary parameters are sufficiently small. \\(\theta_s\\) and \\(\hat{\theta}_s\\) represent the parameters for the secondary objective and their updated values, respectively.

</div>

<div id="assum:bounded_output:ap" class="assumption" markdown="1">

**Assumption 13**. Let \\(f_l(.)\\) denote the output function of the \\(l\\)-th layer in a neural network with \\(L\\) layers. For each layer \\(l \in {1, \dots, L}\\), there exists a constant \\(c_l > 0\\) such that for any input \\(x_l\\) to the \\(l\\)-th layer: \\[|f_l(x_l)| \leq c_l\\] where \\(|.|\\) denotes a suitable norm (e.g., Euclidean norm for vectors, spectral norm for matrices).

</div>

<div id="lemma:lipschitz:ap" class="lemma" markdown="1">

**Lemma 14**. *Let \\(f(x;\theta)\\) be a neural network with L layers, where each layer is a linear transformation followed by a Lipschitz continuous activation function.  
Let \\(\theta\\) be the set of all parameters of the neural network, and \\(\theta_s \subseteq \theta\\) be any subset of parameters. Then, \\(f(x; \theta)\\) is Lipschitz continuous with respect to \\(\theta_s\\).*

</div>

<div class="proof" markdown="1">

*Proof.* Since each activation layer is Lipschitz continuous with Lipschitz constant \\(L_l\\) we have: \\[\begin{aligned}
\label{lemma:1}
        |f_l (x;\theta_l) - f_l (x; \theta_l')| \\
        &\leq L_l |(w_l x + b_l) - (w'_l x + b'_l)| \\
        &=L_l|(w_l - w'_l)f_{l-1}(x) + (b_l - b'_l)| 
\end{aligned}\\] by the triangle inequality and <a href="#lemma:1" data-reference-type="ref" data-reference="lemma:1">[lemma:1]</a> we have: \\[\begin{aligned}
\label{lemma:2}
    L_l|(w_l - w'_l)f_{l-1}(x) + (b_l - b'_l)| \leq L_l(|w_l-w'_l||f_{l-1}(x)| + |b_l - b'_l|)
\end{aligned}\\] by assumption <a href="#assum:bounded_output:ap" data-reference-type="ref" data-reference="assum:bounded_output:ap">13</a> and Eq. <a href="#lemma:2" data-reference-type="ref" data-reference="lemma:2">[lemma:2]</a> we have: \\[\begin{aligned}
    |f_l (x;\theta_l) - f_l (x; \theta_l')| \leq L_l(|w_l - w'_l|c_l + |b_l - b'_l|) \leq L_l c_l |\theta_l - \theta'_l|
\end{aligned}\\] We can write the neural network as the composition of functions of each layer: \\[f(x;\theta) = f_l \circ f_{l-1} \circ ... \circ f_1(x; \theta)\\] According to triangle inequality, we can write: \\[\begin{aligned}
\label{lemma:3}
    |f(x; \theta_i) - f(x; \theta'_i)| \leq \sum_{i=1}^{L}|f_L \circ ... f_{i+1} \circ f_i(x; \theta_i) - f_L \circ ... f_{i+1} \circ f_i(x; \theta'_i)|
\end{aligned}\\] Since the composition of Lipschitz Continuous functions is Lipschitz continuous with Lipschitz constant equal to the product of individual Lipschitz constants `\cite{gouk2021regularisation}`{=latex} we can write: \\[\begin{aligned}
    \label{lemma:4}
    |f_L \circ ... f_{i+1} \circ f_i(x; \theta_i) - f_L \circ ... f_{i+1} \circ f_i(x; \theta'_i)| \leq (\prod_{K=i+1}^{L}L_k)L_ic_i|\theta_i - \theta'_i|
\end{aligned}\\] by using Eq. <a href="#lemma:3" data-reference-type="ref" data-reference="lemma:3">[lemma:3]</a> and <a href="#lemma:4" data-reference-type="ref" data-reference="lemma:4">[lemma:4]</a> we can write: \\[\begin{aligned}
    |f(x;\theta) - f(x;\theta')| \leq \sum_{i=1}^{L}L_ic_i(\prod_{k=i+1}^{L}L_k c_k)|\theta_i - \theta'_i|
\end{aligned}\\] with Cauchy–Schwarz inequality we have: \\[\begin{aligned}
    \sum_{i=1}^{L}L_i c_i (\prod_{k=i+1}^{L}L_k c_k) |\theta_i - \theta'_i| \\
    &\leq \sqrt{\sum_{i=1}^{L} (\theta_i - \theta'_i)^2} \sqrt{\sum_{i=1}^{L}(\prod_{j=i+1}^{L}L_j)^2 L_i^2c_i^2}\\
    &\leq L^* |\theta - \theta'|
\end{aligned}\\] for all \\(\theta = (\theta_s, \theta_{\bar{s}})\\) and constant \\(\theta_{\bar{s}}\\) we have: \\[|f(x; \theta_s) - f(x; \theta'_s)| \leq L^* |\theta_s - \theta'_s|\\] ◻

</div>

<div id="theorem:pareto:ap" class="theorem" markdown="1">

**Theorem 15**. *Let \\(f(\theta_p, \theta_s)\\) for constant \\(\theta_s\\) be the primary objective loss function and \\(\varphi(\theta_p, \theta_s)\\) for constant \\(\theta_p\\) be the secondary objective loss function, where \\(\theta_p \in \Theta_p\\) and \\(\theta_s \in \Theta_s\\) are the primary task and secondary task parameters, respectively.*

*Consider two sets of parameters \\((\theta_p, \theta_s)\\) and \\((\hat{\theta}_p, \hat{\theta}_s)\\) such that \\(\varphi(\hat{\theta}_p, \hat{\theta}_s) \leq \varphi(\theta_p, \theta_s)\\). Then \\(f(\hat{\theta}_p, \hat{\theta}_s) \leq f(\theta_p, \theta_s)\\) holds based on Lemma <a href="#lemma:lipschitz:ap" data-reference-type="ref" data-reference="lemma:lipschitz:ap">14</a>.*

</div>

<div class="proof" markdown="1">

*Proof.* Let \\((\theta_p, \theta_s)\\) and \\((\hat{\theta}_p, \hat{\theta}_s)\\) be two sets of parameters such that \\(\varphi(\hat{\theta}_p, \hat{\theta}_s) \leq \varphi(\theta_p, \theta_s)\\).  
By Lemma <a href="#lemma:lipschitz:ap" data-reference-type="ref" data-reference="lemma:lipschitz:ap">14</a>, \\(f(\theta_p, \theta_s)\\) is Lipschitz continuous with respect to \\(\theta_p\\) and \\(\theta_s\\).  
By Assumption <a href="#assum:small_steps:ap" data-reference-type="ref" data-reference="assum:small_steps:ap">12</a>, \\(|\theta_s - \hat{\theta}_s| \leq \epsilon\\), where \\(\epsilon\\) is sufficiently small. Therefore, applying the Lipschitz continuity: \\[\begin{aligned}
    |f(\hat{\theta}_p, \hat{\theta}_s) - f(\hat{\theta}_p, \theta_s)| 
    \leq L|\hat{\theta}_s - \theta_s| \leq L\epsilon \label{eq:lipschitz_bound:ap}
\end{aligned}\\]

Now, consider the primary loss function \\(f(\theta_p, \theta_s)\\) for a fixed \\(\theta_s\\). By Assumption <a href="#assum:strict_convexity:ap" data-reference-type="ref" data-reference="assum:strict_convexity:ap">11</a>, \\(f(.)\\) is strictly convex in a neighborhood of its local optimum \\(\theta_p^*\\). This means that for \\(\hat{\theta}_p\\) and \\(\theta_p\\) sufficiently close to \\(\theta_p^*\\): \\[\label{eq:strict_convexity}
    f(\lambda \hat{\theta}_p + (1-\lambda)\theta_p, \theta_s) < \lambda f(\hat{\theta}_p, \theta_s) + (1-\lambda)f(\theta_p, \theta_s)\\] for any \\(\lambda \in (0,1)\\).

Since \\(\varphi(\hat{\theta}_p, \hat{\theta}_s) \leq \varphi(\theta_p, \theta_s)\\), and the secondary loss function \\(\varphi(.)\\) is used to update the parameters, we can assume that \\(\hat{\theta}_p\\) is closer to the local optimum \\(\theta_p^*\\) than \\(\theta_p\\). Therefore, by the strict convexity of \\(f(.)\\): \\[\label{eq:f_inequality:ap}
    f(\hat{\theta}_p, \theta_s) \leq f(\theta_p, \theta_s)\\]

Combining the results from <a href="#eq:lipschitz_bound:ap" data-reference-type="eqref" data-reference="eq:lipschitz_bound:ap">[eq:lipschitz_bound:ap]</a> and <a href="#eq:f_inequality:ap" data-reference-type="eqref" data-reference="eq:f_inequality:ap">[eq:f_inequality:ap]</a>: \\[\begin{aligned}
    f(\hat{\theta}_p, \hat{\theta}_s) &\leq |f(\hat{\theta}_p, \hat{\theta}_s) - f(\hat{\theta}_p, \theta_s)| + f(\hat{\theta}_p, \theta_s) \nonumber \\
    &\leq L\epsilon + f(\theta_p, \theta_s)
\end{aligned}\\]

As \\(\epsilon\\) is sufficiently small, we can conclude that \\(f(\hat{\theta}_p, \hat{\theta}_s) \leq f(\theta_p, \theta_s)\\).

Therefore, under the given assumptions, if \\(\varphi(\hat{\theta}_p, \hat{\theta}_s) \leq \varphi(\theta_p, \theta_s)\\), then \\(f(\hat{\theta}_p, \hat{\theta}_s) \leq f(\theta_p, \theta_s)\\). ◻

</div>

<div id="theorem:unique_minimum:ap" class="theorem" markdown="1">

**Theorem 16**. *Let \\(\varphi(\theta_p, \theta_s)\\) be the secondary loss function, where \\(\theta_p \in \Theta_p\\) and \\(\theta_s \in \Theta_s\\) are the primary and secondary task parameters, respectively. Let \\((\theta_p^{(t)}, \theta_s^{(t)})\\) denote the parameters at optimization step \\(t\\), and let \\((\theta_p^{(t+1)}, \theta_s^{(t+1)})\\) be the updated parameters obtained by minimizing \\(\varphi(\theta_p^{(t)}, \theta_s)\\) with respect to \\(\theta_s\\) using a sufficiently small step size \\(\eta > 0\\), i.e.:*

*\\[\theta_s^{(t+1)} = \theta_s^{(t)} - \eta \nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s^{(t)})\\]*

*Then, for a sufficiently small step size \\(\eta\\), the updated secondary parameters \\(\theta_s^{(t+1)}\\) are the unique minimum solution for the secondary loss function \\(\varphi(\theta_p^{(t)}, \theta_s)\\).*

</div>

<div class="assumption" markdown="1">

**Assumption 17**. \\(\varphi(\theta_p, \theta_s)\\) is smooth and Lipschitz continuous.

</div>

<div class="proof" markdown="1">

*Proof.* Let \\(\theta_p^{(t)}\\) be fixed at optimization step \\(t\\). We consider the optimization problem of minimizing the secondary loss function \\(\varphi(\theta_p^{(t)}, \theta_s)\\) with respect to \\(\theta_s\\).

By the assumption, \\(\varphi(\theta_p^{(t)}, \theta_s)\\) is smooth and Lipschitz continuous with respect to \\(\theta_s\\). This implies that \\(\varphi(\theta_p^{(t)}, \theta_s)\\) is continuously differentiable and its gradient \\(\nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s)\\) is Lipschitz continuous with some Lipschitz constant \\(L > 0\\), i.e., for any \\(\theta_s, \theta_s' \in \Theta_s\\):

\\[\|\nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s) - \nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s')\| \leq L \|\theta_s - \theta_s'\|\\]

Now, consider the update rule for \\(\theta_s\\) with a sufficiently small step size \\(\eta > 0\\):

\\[\theta_s^{(t+1)} = \theta_s^{(t)} - \eta \nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s^{(t)})\\]

We want to show that for a sufficiently small \\(\eta\\), \\(\theta_s^{(t+1)}\\) is the unique minimizer of \\(\varphi(\theta_p^{(t)}, \theta_s)\\).

By the Lipschitz continuity of \\(\nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s)\\) and the update rule, we have:

\\[\begin{aligned}
\varphi(\theta_p^{(t)}, \theta_s^{(t+1)}) &\leq \varphi(\theta_p^{(t)}, \theta_s^{(t)}) + \langle \nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s^{(t)}), \theta_s^{(t+1)} - \theta_s^{(t)} \rangle + \frac{L}{2} \|\theta_s^{(t+1)} - \theta_s^{(t)}\|^2 \\
&= \varphi(\theta_p^{(t)}, \theta_s^{(t)}) - \eta \|\nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s^{(t)})\|^2 + \frac{L\eta^2}{2} \|\nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s^{(t)})\|^2 \\
&= \varphi(\theta_p^{(t)}, \theta_s^{(t)}) - \eta \left(1 - \frac{L\eta}{2}\right) \|\nabla_{\theta_s} \varphi(\theta_p^{(t)}, \theta_s^{(t)})\|^2
\end{aligned}\\]

If we choose \\(\eta < \frac{2}{L}\\), then \\(\left(1 - \frac{L\eta}{2}\right) > 0\\), and we have:

\\[\varphi(\theta_p^{(t)}, \theta_s^{(t+1)}) < \varphi(\theta_p^{(t)}, \theta_s^{(t)})\\]

This implies that \\(\theta_s^{(t+1)}\\) is a strict minimizer of \\(\varphi(\theta_p^{(t)}, \theta_s)\\).

To show that \\(\theta_s^{(t+1)}\\) is the unique minimizer, suppose there exists another minimizer \\(\tilde{\theta}_s \neq \theta_s^{(t+1)}\\). By the strict inequality above, we must have:

\\[\varphi(\theta_p^{(t)}, \tilde{\theta}_s) > \varphi(\theta_p^{(t)}, \theta_s^{(t+1)})\\]

which contradicts the assumption that \\(\tilde{\theta}_s\\) is a minimizer.

Therefore, for a sufficiently small step size \\(\eta < \frac{2}{L}\\), the updated secondary parameters \\(\theta_s^{(t+1)}\\) are the unique minimum solution for the secondary loss function \\(\varphi(\theta_p^{(t)}, \theta_s)\\). ◻

</div>

<div class="definition" markdown="1">

**Definition 18**. Let \\(f(x)\\) be a function that is Lipschitz continuous with Lipschitz constant \\(L_f\\), i.e., for any \\(x_1, x_2\\): \\[|f(\theta_1) - f(\theta_2)| \leq L_f \|\theta_1 - \theta_2\|\\]

</div>

**Demographic Parity Loss Function:** The demographic parity loss function \\(DP(f)\\) is defined as: \\[DP(f) = \left| \mathbb{E}_{x \sim p(x|a=0)}[f(\theta_1; x)] - \mathbb{E}_{x \sim p(x|a=1)}[f(\theta_2; x)] \right|\\] where \\(a\\) is a sensitive attribute (e.g., race, gender) with two possible values (0 and 1), and \\(p(x|a)\\) is the conditional probability distribution of \\(x\\) given \\(a\\).

<div id="theorem:dp:ap" class="theorem" markdown="1">

**Theorem 19**. *If \\(f(x)\\) is Lipschitz continuous with Lipschitz constant \\(L_f\\), then the demographic parity loss function \\(\ell_{DP}(f)\\) is also Lipschitz continuous with Lipschitz constant \\(L_{DP} = 2L_f\\).*

</div>

<div class="proof" markdown="1">

*Proof.* Let \\(f_1(x)\\) and \\(f_2(x)\\) be two functions that are Lipschitz continuous with Lipschitz constant \\(L_f\\). We want to show that: \\[|\ell_{DP}(f_1) - \ell_{DP}(f_2)| \leq L_{DP} \|f_1 - f_2\|_{\infty}\\] where \\(\|f_1 - f_2\|_{\infty} = \sup_{x} |f_1(x) - f_2(x)|\\).

Consider the difference between the demographic parity loss functions: \\[\begin{aligned}
|\ell_{DP}(f_1) -& \ell_{DP}(f_2)| = |\left| \mathbb{E}_{x \sim p(x|a=0)}[f_1(x)] - \mathbb{E}_{x \sim p(x|a=1)}[f_1(x)] \right| - \\
&\left| \mathbb{E}_{x \sim p(x|a=0)}[f_2(x)] - \mathbb{E}_{x \sim p(x|a=1)}[f_2(x)] \right|| \notag \\
&\leq \left| \mathbb{E}_{x \sim p(x|a=0)}[f_1(x) - f_2(x)] - \mathbb{E}_{x \sim p(x|a=1)}[f_1(x) - f_2(x)] \right| \\
&\leq \mathbb{E}_{x \sim p(x|a=0)}[|f_1(x) - f_2(x)|] + \mathbb{E}_{x \sim p(x|a=1)}[|f_1(x) - f_2(x)|] \\
&\leq \mathbb{E}_{x \sim p(x|a=0)}[L_f \|f_1 - f_2\|_{\infty}] + \mathbb{E}_{x \sim p(x|a=1)}[L_f \|f_1 - f_2\|_{\infty}] \\
&= L_f \|f_1 - f_2\|_{\infty} (\mathbb{E}_{x \sim p(x|a=0)}[1] + \mathbb{E}_{x \sim p(x|a=1)}[1]) \\
&= 2L_f \|f_1 - f_2\|_{\infty} \\
&= L_{DP} \|f_1 - f_2\|_{\infty}
\end{aligned}\\]

The first inequality follows from the reverse triangle inequality, the second inequality is trivial, and the third inequality follows from the Lipschitz continuity of \\(f_1\\) and \\(f_2\\).

Therefore, the demographic parity loss function \\(\ell_{DP}(f)\\) is Lipschitz continuous with Lipschitz constant \\(L_{DP} = 2L_f\\). ◻

</div>

## Expansion to Equalized Odds (EO) difference [eo_expansion]

In Theorem <a href="#theorem:dp:ap" data-reference-type="ref" data-reference="theorem:dp:ap">19</a>, we established the Lipschitz continuity of the demographic parity loss function. This approach can similarly be applied to another widely used fairness loss function, known as the equalized odds loss. The Equalized Odds Difference measures the extent to which a model’s predictions deviate from equalized odds by quantifying differences in true positive rates (TPR) and false positive rates (FPR) across different groups. Mathematically, it is defined as follows:

#### For true positive rate (TPR):

\\[\begin{aligned}
\text{TPR Difference} = \left| \mathbb{E}_{x \sim p(x|Y=1, a=0)}[f(x)] - \mathbb{E}_{x \sim p(x|Y=1, a=1)}[f(x)] \right|
\end{aligned}\\]

#### For false positive rate (FPR):

\\[\begin{aligned}
\text{TPR Difference} = \left| \mathbb{E}_{x \sim p(x|Y=0, a=0)}[f(x)] - \mathbb{E}_{x \sim p(x|Y=0, a=1)}[f(x)] \right|
\end{aligned}\\]

The overall EO loss can then be considered as the maximum of these two differences:

\\[\begin{aligned}
\text{EO Difference} = \max(\text{TPR Difference}, \text{FPR Difference})
\end{aligned}\\]

Following the logic presented in Theorem <a href="#theorem:dp:ap" data-reference-type="ref" data-reference="theorem:dp:ap">19</a>, we can determine the Lipschitz constants \\(L_{TPR}\\) and \\(L_{FPR}\\) for the true positive rate and false positive rate, respectively. The Lipschitz constant for the equalized odds loss can then be expressed as \\(\max(L_{TPR}, L_{FPR})\\).

## Assumption Discussion [assumption_discussion]

Our work on the FairBiNN method introduces a novel approach to addressing the fairness-accuracy trade-off in machine learning models through bilevel optimization. The theoretical foundations and empirical results demonstrate the potential of this method to outperform traditional approaches like Lagrangian regularization. However, it’s crucial to examine how the underlying assumptions of our theory translate to real-world applications.

### Convexity Near Local Optima

One key assumption in our theoretical analysis is the convexity of the loss function near local optima. In practice, this assumption translates to the behavior of neural networks as they converge during training. While neural network loss landscapes are generally non-convex, recent research suggests that they often exhibit locally convex regions around minima, especially in overparameterized networks `\cite{allen2019learning}`{=latex}. In real-world scenarios, as long as the network converges, it will likely encounter these locally convex regions along its optimization path. Our theory applies particularly well in these parts of the optimization process. This assumption becomes increasingly valid as the network approaches convergence, which is typically the case for well-designed models trained on suitable datasets. Therefore, practitioners can rely on this aspect of our theory as long as their models show signs of convergence on the given data.

### Overparameterization

The assumption of overparameterization in our model is another critical aspect that warrants discussion. In modern deep learning, overparameterized models - those with more parameters than training samples - are increasingly common. This trend aligns well with our theoretical framework. In practical terms, as long as the model is capable of overfitting on the training data, this assumption stands. This condition is often met in real-world scenarios, especially with deep neural networks applied to typical dataset sizes. The overparameterization allows for the existence of multiple solutions that can fit the training data, providing the flexibility needed for our bilevel optimization approach to find solutions that balance accuracy and fairness effectively. However, it’s important to note that in some resource-constrained environments or with extremely large datasets, overparameterization might not always be feasible. In such cases, the applicability of our method may require further investigation or adaptation.

### Lipschitz Continuity

The assumption of Lipschitz continuity is crucial for the stability and convergence properties of our optimization process. In our experiments, we ensured that the chosen layers and loss functions satisfy Lipschitz continuity, thus upholding this assumption. For practitioners, we provide a rigorous analysis of various layers and activation functions in terms of their Lipschitz properties. This analysis serves as a guide for choosing components that maintain the Lipschitz continuity assumption. Common choices like ReLU activations and standard loss functions (e.g., cross-entropy) are Lipschitz continuous, making this assumption generally applicable in many practical scenarios. However, care must be taken when using certain architectures or custom loss functions. For instance, unbounded activation functions or poorly designed custom losses might violate this assumption. We recommend that practitioners refer to our provided analysis when designing their models to ensure compliance with this crucial property.

## Practical Implications of Bounded Output Assumption [bounded_assump]

The assumption of bounded layer outputs translates to several practical considerations in neural network design and implementation:

### Bounded Activation Functions

In practice, this assumption is often satisfied by using bounded activation functions:

- **Sigmoid function**: bounded between 0 and 1

- **Hyperbolic tangent (tanh)**: bounded between -1 and 1

- **ReLU6**: a variant of ReLU that is capped at 6

### Normalization Techniques

Various normalization techniques help ensure that the outputs of layers remain bounded:

- **Batch Normalization**: normalizes the output of a layer by adjusting and scaling the activations

- **Layer Normalization**: similar to batch normalization but normalizes across the features instead of the batch

- **Weight Normalization**: decouples the magnitude of a weight vector from its direction

### Regularization

Certain regularization techniques indirectly encourage bounded outputs:

- **L1/L2 regularization**: by penalizing large weights, these methods indirectly limit the magnitude of layer outputs

- **Dropout**: by randomly setting some activations to zero, dropout can help prevent excessively large outputs

By implementing these techniques, practitioners can design neural networks that better align with the theoretical assumption of bounded layer outputs. This alignment potentially leads to more stable training and improved generalization properties, bridging the gap between theoretical guarantees and practical implementations.

## Exploring Lipschitz Continuity in Activation functions [act_func_lip]

According to our theoretical framework, we can guarantee the pareto front solution when using activation functions that are Lipschitz continuous (smooth).

Activation functions (\\(f\\)) that are Lipschitz continuous have a property where there exists a constant \\(L\\) such that for any \\(x, y\\):

\\[|f(x) - f(y)| \leq L \|x - y\|\\]

As an example, we can prove sigmoid function is Lipschitz continuous. we need to show that there exists a constant \\(L\\) such that for all \\(x, y \in \mathbb{R}\\) :

\\[|\sigma(x) - \sigma(y)| \leq L \|x - y\|\\]

where \\(\sigma(x)\\) is the sigmoid function defined as:

\\[\sigma(x) = \frac{1}{1 + e^{-x}}\\]

**Derivative of the Sigmoid Function**

The first step in proving Lipschitz continuity is to find the derivative of the sigmoid function, which gives us the rate of change. The derivative of the sigmoid function is:

\\[\sigma^{\prime}(x) = \sigma(x)(1 - \sigma(x))\\]

Since \\(\sigma(x)\\) is always between 0 and 1, the expression \\(\sigma(x)(1 - \sigma(x))\\) is maximized when \\(\sigma(x) = 0.5\\).

**Finding the Lipschitz Constant**

The maximum value of \\(\sigma^{\prime}(x)\\) occurs at \\(\sigma(x) = 0.5\\), which gives:

\\[\sigma^{\prime}(x) = 0.5(1 - 0.5) = 0.25\\]

Therefore, the derivative of the sigmoid function is bounded by 0.25:

\\[0 \leq \sigma^{\prime}(x) \leq 0.25\\]

This means that the Lipschitz constant \\(L\\) is 0.25, and Sigmoid Function is smooth.

Similar to this proof, we can show that following common activation functions are also Lipschitz continuous:

1.  **Linear:** \\(f(x) = x\\) with constant \\(L = 1\\)

2.  **Hyperbolic Tangent (Tanh):** \\(f(x) = \tanh(x)\\) with constant \\(L = 1\\)

3.  **ReLU (Rectified Linear Unit):** \\(f(x) = \max(0, x)\\) with constant \\(L = 1\\)

4.  **Leaky ReLU:** \\(f(x) = \max(\alpha x, x)\\) where \\(\alpha\\) is a small positive constant, with constant \\(L = \max(1, \alpha)\\).

5.  **ELU (Exponential Linear Unit):** \\[\text{ELU}(x) = 
        \begin{cases} 
        x, & \text{if } x > 0 \\
        \alpha(e^x - 1), & \text{if } x \leq 0 \\
        \end{cases}\\]  
    The ELU function is Lipschitz continuous, but the constant depends on the value of \\(\alpha\\).

6.  **Softplus:** \\(f(x) = \log(1 + e^x)\\) with constant \\(L = 1\\)

There are some common activation functions that are **<u>not</u>** Lipschitz continuous:

1.  **Softmax**

2.  **Binary Step:** \\[\text{BinaryStep}(x) = 
        \begin{cases} 
        1, & \text{if } x \geq 0 \\
        0, & \text{if } x < 0 \\
        \end{cases}\\]

3.  **Hard Tanh:** \\[\text{HardTanh}(x) = 
        \begin{cases} 
        -1, & \text{if } x < -1 \\
        x, & \text{if } -1 \leq x \leq 1 \\
        1, & \text{if } x > 1 
        \end{cases}\\]

4.  **Hard Sigmoid:** \\[\text{HardSigmoid}(x) = 
        \begin{cases} 
        0, & \text{if } x \leq -2.5 \\
        1, & \text{if } x \geq 2.5 \\
        0.2x + 0.5, & \text{if } -2.5 < x < 2.5 
        \end{cases}\\]

## Exploring Lipschitz Continuity in CNNs and GNNs [nets_lip]

**Convolutional Neural Networks**

The Lipschitz continuity of a CNN layer can be determined by examining its components: convolution operations, activation functions, and pooling layers. Convolution is a linear operation, and its Lipschitz constant is related to the spectral norm of the convolution matrix, typically limited by the sum of the absolute values of the weights. Activation functions like ReLU and sigmoid are Lipschitz continuous, with constants of 1 and 0.25, respectively. Pooling operations, such as max and average pooling, are also Lipschitz continuous, with max pooling having a constant of 1 and average pooling having a constant dependent on pooling size. Therefore, a CNN layer is Lipschitz continuous if all its components are, with the overall Lipschitz constant being the product of the constants of these components.

Zoe et al. `\citep{zou2019lipschitz}`{=latex} developed a linear programming approach to estimate the Lipschitz bound of CNN layers. Their method leverages concepts such as the Bessel bound, discrete signal processing, and the discrete Fourier transform to calculate the Lipschitz constant for each layer in popular architectures like AlexNet and GoogleNet.

**Graph Neural Networks**

Graph Neural Network (GNN) layers operate on graph-structured data through a series of message-passing, aggregation, and update steps, each contributing to the Lipschitz continuity of the layer. In the message-passing step, functions aggregate information from neighboring nodes and are often linear or involve nonlinearities; linear message-passing functions are Lipschitz continuous, with the constant depending on the weights and the graph’s maximum degree. Aggregation functions, such as sum, mean, and max, are Lipschitz continuous, with sum and mean being linear, and max having a constant of 1. Update functions apply neural networks to aggregated information, and if composed of Lipschitz continuous operations like linear transformations and activations such as ReLU, they maintain Lipschitz continuity. The overall Lipschitz constant of a GNN layer is influenced by the characteristics of the message-passing, aggregation, and update functions, as well as the graph’s structure, such as node degrees.

A recent study by Juvina et al. `\citep{juvinatraining}`{=latex} presents a learning framework designed to maintain tight Lipschitz-bound constraints across various GNN models. To facilitate easier computations, the authors utilize closed-form expressions of a tight Lipschitz constant and employ a constrained optimization strategy to monitor and control this constant effectively. Although this is not the first attempt to control the Lipschitz constant, the authors successfully reduce the size of the matrices involved by a factor of \\(K^2\\) , where \\(K\\) is the number of nodes in the graph. While previous works, such as Dasoulas et al. `\citep{dasoulas2021lipschitz}`{=latex}, focused on controlling the Lipschitz constant for basic attention-based GNNs, Juvina et al. `\citep{juvinatraining}`{=latex} also extend this approach to enhance the robustness of GNN models against adversarial attacks.

## Direct Comparison: Bilevel (FairBiNN) vs. Lagrangian Method [direct_comp_lag]

In this subsection, we present a comprehensive comparison between our proposed FairBiNN method and the traditional Lagrangian regularization approach. This comparative analysis serves multiple purposes. Primarily, it aims to empirically validate the theoretical advantages of the bilevel optimization framework outlined in our earlier analysis. By doing so, we demonstrate how the FairBiNN method translates theoretical benefits into practical performance gains in terms of both accuracy and fairness metrics. Furthermore, this comparison provides insight into the convergence behavior and stability of both methods under various hyperparameter settings. It illustrates the flexibility of the FairBiNN approach in managing the trade-off between model accuracy and fairness constraints, a crucial aspect in real-world applications of fair machine learning.

We trained both models on the Adult and Health datasets, using the same network architecture, same number of parameters, and optimization settings. Figure <a href="#fig:adult_loss" data-reference-type="ref" data-reference="fig:adult_loss">5</a> displays the BCE loss over epochs for the Adult dataset. The Bi-level approach demonstrates better performance compared to the Lagrangian approach, achieving a lower BCE loss of approximately 0.23 by epoch 200, while the Lagrangian approach reaches a loss of about 0.26. Similarly, Figure <a href="#fig:health_loss" data-reference-type="ref" data-reference="fig:health_loss">6</a> shows the BCE loss over epochs for the Lagrangian, Bi-level, and Without Fairness approaches on the Health dataset.

<figure id="fig:tabular_loss">
<figure id="fig:adult_loss">
<img src="./figures/loss_adult.jpeg"" />
<figcaption>UCI Adult</figcaption>
</figure>
<figure id="fig:health_loss">
<img src="./figures/loss_health.jpeg"" />
<figcaption>Heritage Health</figcaption>
</figure>
<figcaption>BCE loss over epochs for the Lagrangian, Bi-level, and Without Fairness approaches on (a) the Adult dataset and (b) the Health dataset. These results illustrate that the Bi-level optimization framework achieves lower BCE loss compared to the Lagrangian approach in these experiments, highlighting its potential in optimizing both accuracy and fairness objectives in neural networks.</figcaption>
</figure>

Through this direct comparison, we aim to bridge the gap between theoretical analysis and practical implementation, showcasing how the principled design of the FairBiNN method leads to tangible improvements in fair machine learning tasks. This offers practitioners a clear understanding of when and why they might choose the FairBiNN method over the Lagrangian approach in real-world scenarios. We trained both FairBiNN and Lagrangian models on the Adult and Health datasets, using the same network architecture (Same numeber of parameters) and optimization settings for fair comparison. For each method, we varied the fairness-accuracy trade-off parameter (\\(\eta\\) for FairBiNN, \\(\lambda\\) for Lagrangian) to generate a range of models with different accuracy-fairness balances.

<figure id="fig:reg_compare">
<figure id="fig:adult_reg">
<img src="./figures/reg_compare_adult.png"" />
<figcaption>UCI Adult</figcaption>
</figure>
<figure id="fig:health_reg">
<img src="./figures/reg_compare_Health.png"" />
<figcaption>Heritage Health</figcaption>
</figure>
<figcaption>Comparison of FairBiNN and Lagrangian methods on UCI Adult and Heritage Health datasets</figcaption>
</figure>

Figure <a href="#fig:reg_compare" data-reference-type="ref" data-reference="fig:reg_compare">10</a> presents a comparative analysis of the FairBiNN and Lagrangian methods on two benchmark datasets: UCI Adult (Figure <a href="#fig:adult_reg" data-reference-type="ref" data-reference="fig:adult_reg">8</a>) and Heritage Health (Figure <a href="#fig:health_reg" data-reference-type="ref" data-reference="fig:health_reg">9</a>). The graphs plot the trade-off between accuracy and Statistical Parity Difference (SPD), a measure of fairness where lower values indicate better fairness. For the UCI Adult dataset (Figure <a href="#fig:adult_reg" data-reference-type="ref" data-reference="fig:adult_reg">8</a>), we observe that the FairBiNN method consistently outperforms the Lagrangian approach. The FairBiNN curve is closer to the top-left corner, indicating that it achieves higher accuracy for any given level of fairness (SPD). The difference is particularly pronounced at lower SPD values, suggesting that FairBiNN is more effective at maintaining accuracy while enforcing stricter fairness constraints. The Heritage Health dataset results (Figure <a href="#fig:health_reg" data-reference-type="ref" data-reference="fig:health_reg">9</a>) show a similar trend, but with a more dramatic difference between the two methods. The FairBiNN curve dominates the Lagrangian curve across the entire range of SPD values. This indicates that FairBiNN achieves substantially higher accuracy for any given fairness level, or equivalently, much better fairness for any given accuracy level. In both datasets, the FairBiNN method demonstrates a smoother, more consistent trade-off between accuracy and fairness. The Lagrangian method, in contrast, shows more erratic behavior, particularly in the Heritage Health dataset where its performance degrades rapidly as fairness constraints tighten. These results empirically validate the theoretical advantages of the FairBiNN method discussed earlier in the paper. They suggest that the bilevel optimization approach is more effective at balancing the competing objectives of accuracy and fairness.

### Computational Complexity Analysis

Let’s define the following variables:

- \\(n\\): number of parameters in \\(\theta_p\\)

- \\(m\\): number of parameters in \\(\theta_s\\)

- \\(C_f\\): cost of computing \\(f\\) and its gradients

- \\(C_\phi\\): cost of computing \\(\phi\\) and its gradients

**Regularization (Lagrangian) Method**

The Lagrangian update rules are: \\[\begin{aligned}
\theta_p = \theta_p - \alpha_L \nabla_{\theta_p} (f(\theta_p, \theta_s) + \lambda\phi(\theta_p, \theta_s)) 
\\
\theta_s = \theta_s - \alpha_L \nabla_{\theta_s} (f(\theta_p, \theta_s) + \lambda\phi(\theta_p, \theta_s))
\end{aligned}\\] Computational complexity per iteration: \\(O(C_f + C_\phi + n + m)\\)

**Bilevel Optimization Method**

The bilevel update rules are: \\[\text{Lower level: }  \theta_s = \theta_s - \alpha_s \nabla_{\theta_s} \phi(\theta_p, \theta_s) \\\] \\[\text{Upper level: }  \theta_p = \theta_p - \alpha_f \nabla_{\theta_p} f(\theta_p, \theta_s^(\theta_p))\\] Computational complexity per iteration: \\(O(C_f + C_\phi + n + m)\\)

**Empirical Comparison**

While the theoretical complexity analysis suggests similar costs for both methods, we conducted empirical tests to compare their actual runtime performance. Table <a href="#tab:train_time" data-reference-type="ref" data-reference="tab:train_time">1</a> reports the average epoch time for both the Adult and Health datasets using the FairBiNN and Lagrangian methods after 10 epochs of warmup.

<div id="tab:train_time" markdown="1">

| Method     | Adult Dataset (s) | Health Dataset (s) |
|:-----------|:-----------------:|:------------------:|
| FairBiNN   |       0.62        |        1.03        |
| Lagrangian |       0.60        |        1.05        |

Average epoch time (in seconds) for FairBiNN and Lagrangian methods

</div>

These experiments were conducted on an M1 Pro CPU. As we can observe from the results reported in table <a href="#tab:train_time" data-reference-type="ref" data-reference="tab:train_time">1</a>, there is no tangible difference in the average epoch time between the FairBiNN and Lagrangian methods for both datasets. This empirical evidence aligns with our theoretical analysis.

## Related works - Graph and Vision domains [ap:related]

### Graph [ap:graph]

The message-passing structure of GNNs and the topology of graphs both have the potential to amplify the bias. In general, in graphs such as social networks, nodes with sensitive features similar to one another are more likely to link to one another than nodes with sensitive attributes dissimilar from one another `\citep{dong2016young, rahman2019fairwalk}`{=latex}. On social networks, for instance, persons of younger generations have a higher tendency to form friendships with others of a similar age `\citep{dong2016young}`{=latex}. This results in the aggregation of neighbors’ features in GNN having similar representations for nodes of similar sensitive information while having different representations for nodes of different sensitive features, which leads to severe bias in decision-making, in the sense that the predictions are highly correlated with the sensitive attributes of the nodes. GNNs have a greater bias due to the adoption of graph structure than models that employ node characteristics `\citep{dai2021say}`{=latex}. Because of this bias, the widespread use of GNNs in areas such as the evaluation of job candidates `\citep{mehrabi2021survey}`{=latex} and the prediction of drug-target interaction `\citep{yazdani2022attentionsitedti, khodabandeh2024fragxsitedti}`{=latex} would be significantly hindered. As a result, it is essential to research equitable GNNs. The absence of sensitive information presents significant problems to the work that has already been done on fair models `\citep{beutel2017data, creager2019flexibly, locatello2019fairness, louizos2015variational, tayebi2024learning}`{=latex}. Despite the significant amount of work that has been put into developing fair models through the revision of features `\cite{kamiran2009classifying, kamiran2012data, zhang2017achieving}`{=latex}, disentanglement `\citep{creager2019flexibly, louizos2015variational}`{=latex}, adversarial debiasing `\citep{beutel2017data, edwards2015censoring}`{=latex}, and fairness constraints `\citep{zafar2017fairness, zafar2017fairness_a}`{=latex}, these models are almost exclusively designed for independently and identically distributed (i.i.d) data, meaning that they are unable to be directly applied to graph data due to the fact that they do not simultaneously take into consideration the bias that comes from node attributes and graph. In recent years, `\citet{bose2019compositional, rahman2019fairwalk}`{=latex} have been published to learn fair node representations from graphs. These approaches only deal with simple networks that do not have any properties on any of the nodes, and they place their emphasis on fair node representations rather than fair node classifications. Finally, `\citet{dai2021say}`{=latex} used graph topologies and a restricted amount of protected attributes and designed FairGNN to reduce the bias of GNNs while retaining high node classification accuracy.

### Vision [ap:vision]

The challenges caused by bias in computer vision might appear in various ways. It has been found, for instance, that in action recognition models, when the data include gender bias, the bias is exacerbated by the models trained on such datasets `\citep{zhao2017men}`{=latex}. Face detection and recognition models may be less precise for some racial and gender categories `\citep{buolamwini2018gender}`{=latex}. Methods for mitigating bias in vision datasets are suggested in `\citep{wang2022revise}`{=latex} and `\citep{yang2020towards}`{=latex}. Several researchers have used GANs on image datasets for bias reduction. `\citet{sattigeri2019fairness}`{=latex} altered the utility function of GAN in order to generate equitable picture datasets. FairFaceGAN `\citep{hwang2020fairfacegan}`{=latex} provides facial image-to-image translation to avoid unintended transfer of protected characteristics. `\citet{roy2019mitigating}`{=latex} developed a method to mitigate information leakage on image datasets by formulating the problem as an adversarial game to maximize data utility and minimize the amount of information contained in the embedding, measured by entropy. `\citet{ramaswamy2021fair}`{=latex} presents a methodology to generate balanced training data for each protected property by perturbing the latent vector of a GAN. Other experiments using GANs to generate accurate data are `\citep{choi2020fair, sharmanska2020contrastive}`{=latex}. Beyond GANs, many strategies have addressed the challenge of AI fairness. `\citep{rajabi2022through}`{=latex} proposed a U-Net for creating unbiased image data. Deep information maximization adaption networks were employed to eliminate racial bias in face vision datasets `\citep{wang2019racial}`{=latex}, while reinforcement learning was utilized for training a race-balanced network `\citep{wang2019mitigate}`{=latex}. `\citet{wang2021towards}`{=latex} offer a generative few-shot cross-domain adaptation method for performing fair cross-domain adaptation and enhancing minority category performance. The research in `\citep{xu2021consistent}`{=latex} recommends adding a penalty term to the softmax loss function to reduce bias and enhance face recognition fairness performance. `\citet{quadrianto2019discovering}`{=latex} describes a technique for discovering fair data representations with the same semantic information as the original data. There have also been effective applications of adversarial learning for this purpose `\citep{wang2019balanced, zhang2018mitigating}`{=latex}. `\citep{chuang2021fair}`{=latex} proposed fair mixup, which uses data augmentation to mitigate bias in data.

## Evaluation Metrics [ap:metric]

We utilize four metrics to compare our model’s performance against baseline models. Average precision (AP) is utilized to gauge classifier accuracy, combining recall and accuracy at each point. In tabular and graph datasets, we opt for accuracy to align with existing literature practices.

Fairness evaluation draws from various criteria, with demographic parity (DP) being widely used. DP quantifies the difference in favorable outcomes across protected groups, expressed as \\((|P(Y = 1|S = 0) - P(Y = 1|S = 1)|)\\) `\citep{mehrabi2021survey}`{=latex}. For scenarios involving more than two groups, DP can be calculated as \\(\Delta_{DP} (a, \hat{y}) = \max_{a_i, a_j} |P(\hat{y} = 1 | a=a_i) - P(\hat{y} =1 | a=a_j))|\\) `\citep{gupta2021controllable}`{=latex}. A smaller DP indicates fairer categorization. We also adopt the difference in equality of opportunity (\\(\Delta\\)EO), which measures the absolute difference in true positive rates between gender expressions \\((|TPR(S = 0) - TPR(S = 1)|)\\) `\citep{lokhande2020fairalm, ramaswamy2021fair}`{=latex}. Minimizing \\(\Delta\\)EO signifies fairer outcomes. Demographic parity serves as the fairness criterion in our optimization. Discrepancies between EO and DP may occur due to this choice.

## Implementation details [ap:hyper]

The hyperparameters used in training the models on each dataset can be found in the tables <a href="#hype1" data-reference-type="ref" data-reference="hype1">2</a>, <a href="#hype2" data-reference-type="ref" data-reference="hype2">3</a>, and <a href="#hype3" data-reference-type="ref" data-reference="hype3">4</a>. The training was conducted on a computer with an NVIDIA GeForce RTX 3090.

<div id="hype1" markdown="1">

| Hyperparameters                      | UCI Adult | Health Heritage |
|:-------------------------------------|:---------:|:---------------:|
| FC layers before the fairness layers |     2     |        2        |
| Fairness FC layers                   |     1     |        3        |
| FC layers after the fairness layers  |     1     |        1        |
| Epoch                                |    50     |       50        |
| Batch size                           |    100    |       100       |
| Dropout                              |     0     |        0        |
| Network optimizer                    |   Adam    |      Adam       |
| fairness layers’ optimizer           |   Adam    |      Adam       |
| classifier layers’ learning rate     |   1e-3    |      1e-3       |
| fairness layers’ learning rate       |   1e-5    |      1e-5       |
| \\(\eta\\)                           |    100    |       100       |

Summary of Parameter Setting for the fairness layers on tabular datasets

</div>

<span id="hype1" label="hype1"></span>

<div id="hype2" markdown="1">

| Hyperparameters                      | POKEC-Z | POKEC-N | NBA  |
|:-------------------------------------|:-------:|:-------:|:----:|
| GCN layer before the fairness layers |    2    |    2    |  2   |
| Fairness FC layers                   |    1    |    1    |  1   |
| FC layers after the fairness layers  |    1    |    1    |  1   |
| Epoch                                |  5000   |  1000   | 1000 |
| Batch size                           |    1    |    1    |  1   |
| Dropout                              |    0    |   0.5   | 0.5  |
| Network optimizer                    |  Adam   |  Adam   | Adam |
| fairness layers’ optimizer           |  Adam   |  Adam   | Adam |
| classifier layers’ learning rate     |  1e-3   |  1e-3   | 1e-2 |
| fairness layers’ learning rate       |  1e-6   |  1e-8   | 1e-5 |
| \\(\eta\\)                           |  1000   |   100   | 1000 |

Summary of Parameter Setting for the fairness layers on graph datasets

</div>

<span id="hype2" label="hype2"></span>

<div id="hype3" markdown="1">

| Hyperparameters | CelebA-Attractive | CelebA-Smiling | CelebA-WavyHair |
|:---|:--:|:--:|:--:|
| Fairness FC layers | 1 | 1 | 1 |
| FC layers after the fairness layers | 1 | 1 | 1 |
| Epoch | 30 | 15 | 15 |
| Batch size | 128 | 128 | 128 |
| Dropout | 0 | 0 | 0 |
| Network optimizer | Adam | Adam | Adam |
| fairness layers’ optimizer | Adam | Adam | Adam |
| classifier layers’ learning rate | 1e-3 | 1e-3 | 1e-3 |
| fairness layers’ learning rate | 1e-6 | 1e-5 | 1e-5 |
| \\(\eta\\) | 1000 | 100 | 100 |

Summary of Parameter Setting for the fairness layers on vision dataset

</div>

<span id="hype3" label="hype3"></span>

## Other domains’ results [ap:results]

### Graph [graph]

<div class="center" markdown="1">

<div class="small" markdown="1">

<div class="sc" markdown="1">

<div id="tab:pokz" markdown="1">

| Method | ACC(%) | AUC(%) | \\(\Delta_{DP}\\)(%) | \\(\Delta_{EO}\\)(%) |
|:---|:--:|:--:|:--:|:--:|
| ALFR `\citep{edwards2015censoring}`{=latex} | 65.4 ±0.3 | 71.3 ±0.3 | 2.8 ±0.5 | 1.1 ±0.4 |
| ALFR-e `\citep{edwards2015censoring, perozzi2014deepwalk}`{=latex} | 68.0 ±0.6 | 74.0 ±0.7 | 5.8 ±0.4 | 2.8 ±0.8 |
| Debias `\citep{zhang2018mitigating}`{=latex} | 65.2 ±0.7 | 71.4 ±0.6 | 1.9 ±0.6 | 1.9 ±0.4 |
| Debias-e `\citep{zhang2018mitigating, perozzi2014deepwalk}`{=latex} | 67.5 ±0.7 | 74.2 ±0.7 | 4.7 ±1.0 | 3.0 ±1.4 |
| FCGE `\citep{bose2019compositional}`{=latex} | 65.9 ±0.2 | 71.0 ±0.2 | 3.1 ±0.5 | 1.7 ±0.6 |
| FairGCN `\citep{dai2021say}`{=latex} | 70.0 ±0.3 | 76.7 ±0.2 | 0.9 ±0.5 | 1.7 ±0.2 |
| FairGAT `\citep{kose2024fairgat}`{=latex} | 70.1 ±0.1 | 76.5 ±0.2 | **0.5 ±0.3** | **0.8 ±0.3** |
| NT-FairGNN `\citep{dai2022learning}`{=latex} | 70.0 ±0.1 | 76.7 ±0.3 | 1.0 ±0.4 | 1.6 ±0.2 |
| **GAT+FairBiNN (Ours)** | **70.97 ±0.16** | **77.58 ±0.13** | 0.93 ±0.44 | 0.97 ±0.40 |

The comparisons of our proposed method with the baselines on Pokec-z

</div>

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div class="sc" markdown="1">

<div id="tab:pokn" markdown="1">

| Method | ACC(%) | AUC(%) | \\(\Delta_{DP}\\)(%) | \\(\Delta_{EO}\\)(%) |
|:---|:--:|:--:|:--:|:--:|
| ALFR `\citep{edwards2015censoring}`{=latex} | 63.1 ±0.6 | 67.7 ±0.5 | 3.05 ±0.5 | 3.9 ±0.6 |
| ALFR-e `\citep{edwards2015censoring, perozzi2014deepwalk}`{=latex} | 66.2 ±0.5 | 71.9 ±0.3 | 4.1 ±0.5 | 4.6 ±1.6 |
| Debias `\citep{zhang2018mitigating}`{=latex} | 62.6 ±0.9 | 67.9 ±0.7 | 2.4 ±0.7 | 2.6 ±1.0 |
| Debias-e `\citep{zhang2018mitigating, perozzi2014deepwalk}`{=latex} | 65.6 ±0.8 | 71.7 ±0.7 | 3.6 ±0.2 | 4.4 ±1.2 |
| FCGE `\citep{bose2019compositional}`{=latex} | 64.8 ±0.5 | 69.5 ±0.4 | 4.1 ±0.8 | 5.5 ±0.9 |
| FairGCN `\citep{dai2021say}`{=latex} | 70.1 ±0.2 | 74.9 ±0.4 | 0.8 ±0.2 | 1.1 ±0.5 |
| FairGAT `\citep{kose2024fairgat}`{=latex} | 70.0 ±0.2 | 74.9 ±0.4 | **0.6 ±0.3** | **0.8 ±0.2** |
| NT-FairGNN `\citep{dai2022learning}`{=latex} | **70.1 ±0.2** | 74.9 ±0.4 | 0.8 ±0.2 | 1.1 ±0.3 |
| **GAT+FairBiNN (Ours)** | 70.07 ±0.5 | **75.8 ±0.38** | 0.62 ±0.14 | 3.0 ±1.0 |

The comparisons of our proposed method with the baselines on Pokec-n

</div>

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div class="sc" markdown="1">

<div id="tab:nba" markdown="1">

| Method | ACC(%) | AUC(%) | \\(\Delta_{DP}\\)(%) | \\(\Delta_{EO}\\)(%) |
|:---|:--:|:--:|:--:|:--:|
| ALFR `\citep{edwards2015censoring}`{=latex} | 64.3 ±1.3 | 71.5 ±0.3 | 2.3 ±0.9 | 3.2 ±1.5 |
| ALFR-e `\citep{edwards2015censoring, perozzi2014deepwalk}`{=latex} | 66.0 ±0.4 | 72.9 ±1.0 | 4.7 ±1.8 | 4.7 ±1.7 |
| Debias `\citep{zhang2018mitigating}`{=latex} | 63.1 ±1.1 | 71.3 ±0.7 | 2.5 ±1.5 | 3.1 ±1.9 |
| Debias-e `\citep{zhang2018mitigating, perozzi2014deepwalk}`{=latex} | 65.6 ±2.4 | 72.9 ±1.2 | 5.3 ±0.9 | 3.1 ±1.3 |
| FCGE `\citep{bose2019compositional}`{=latex} | 66.0 ±1.5 | 73.6 ±1.5 | 2.9 ±1.0 | 3.0 ±1.2 |
| FairGCN `\citep{dai2021say}`{=latex} | 71.1 ±1.0 | 77.0 ±0.3 | 1.0 ±0.5 | 1.2 ±0.4 |
| FairGAT `\citep{kose2024fairgat}`{=latex} | 71.5 ±0.8 | 77.5 ±0.7 | 0.7 ±0.5 | **0.7 ±0.3** |
| NT-FairGNN `\citep{dai2022learning}`{=latex} | 71.1 ±1.0 | 77.0 ±0.3 | 1.0 ±0.5 | 1.2 ±0.4 |
| **GAT+FairBiNN (Ours)** | **77.09 ±0.45** | **77.99 ±0.58** | **0.34 ±0.21** | 12.78 ±2.9 |

The comparisons of our proposed method with the baselines on NBA

</div>

</div>

</div>

</div>

We compare our suggested framework with some of the cutting-edge approaches for fair classification, and fair graph embedding learning including ALFR `\cite{edwards2015censoring}`{=latex}, ALFR-e, Debias `\cite{zhang2018mitigating}`{=latex}, Debias-e, and FCGE `\citep{bose2019compositional}`{=latex}. In the ALFR `\cite{edwards2015censoring}`{=latex} approach, which is a pre-processing technique, the sensitive information in the representations created by an MLP-based autoencoder is eliminated using a discriminator. Then the debiased representations are used to train the linear classifier. ALFR-e is a method to make use of the graph structure information and joins the user features in the ALFR with the graph embeddings discovered by deepwalk `\citep{perozzi2014deepwalk}`{=latex}. Debias `\cite{zhang2018mitigating}`{=latex}, is a fair categorization technique used throughout processing. It immediately applies a discriminator to the predicted likelihood of the classifier. Debias-e, which is similar to the ALFR-e, also includes deepwalk embeddings into the Debias characteristics. FCGE `\citep{bose2019compositional}`{=latex}, is suggested as a method for learning fair node embeddings in graphs without node characteristics. FairGCN `\citep{dai2021say}`{=latex}, a graph convolutional network designed for fairness in graph-based learning. It incorporates fairness constraints during training to reduce disparities between protected groups. FairGAT `\citep{kose2024fairgat}`{=latex}, a fairness-aware graph-based learning framework that employs a novel attention learning strategy to reduce bias. This framework is grounded in a theoretical analysis that identifies sources of bias in GAT-based neural networks used for node classification. NT-FAIRGNN `\citep{dai2022learning}`{=latex} is a graph neural network that aims to achieve fairness by balancing the trade-off between accuracy and fairness. It uses a two-player minimax game between the predictor and the adversary, where the adversary aims to maximize the unfairness. Discriminators screen out the delicate data in the embeddings. We used the `\citet{dai2021say}`{=latex} study’s obtained datasets for our investigation which are as follows:  
*Pokec* `\citep{takac2012data}`{=latex} is among the most well-known social network datasets in Slovakia which resemble Facebook and Twitter greatly. This dataset includes anonymous information from the whole social network of the year 2012. User profiles on Pokec include information on gender, age, interests, hobbies, profession, and more. There are millions of users in the original Pokec dataset. Sampled Pokec-z and Pokec-n datasets are based on the provinces that users are from. The categorization task involves predicting the users’ working environment.  
*NBA* is a Kaggle dataset with about 400 NBA basketball players that served as the basis for this extension. Players’ 2016–2017 season success statistics, along with additional details like nationality, age, and income are presented. They gathered the relationships between NBA basketball players on Twitter using its official crawling API to create the graph connecting the NBA players. They separated the nationality into two groups, American players and international players, which is a sensitive characteristic. The classification job is to predict whether a player’s wage is above the median.  
Each experiment was conducted five times, and Tables <a href="#tab:pokz" data-reference-type="ref" data-reference="tab:pokz">5</a>, <a href="#tab:pokn" data-reference-type="ref" data-reference="tab:pokn">6</a>, and <a href="#tab:nba" data-reference-type="ref" data-reference="tab:nba">7</a> report the mean and standard deviation of the runs for Pokec-z, Pokec-n, and NBA datasets, respectively. These results represent the selected Pareto solutions for comparison with the benchmarks. The tables reveal that, in comparison to GAT, generic fair classification techniques and graph embedding learning approaches exhibit inferior classification performance, even when utilizing graph information. In contrast, our Bilevel design performs comparably to baseline GNNs. FairGCN is close to the baseline, but the FairBiNN approach outperforms it. When sensitive information is scarce (e.g., NBA dataset), baselines exhibit clear bias, with graph-based baselines performing worse. However, our proposed model yields near-zero statistical demographic parity, indicating effective discrimination mitigation.

### Vision [vision]

<figure id="fig:celebA_attractive">
<figure id="fig:1">
<img src="./figures/celebA_dp_attractive.jpg"" />
<figcaption>Attractive AP vs. <span class="math inline"><em>Δ</em></span> DP</figcaption>
</figure>
<figure id="fig:4">
<img src="./figures/celebA_eo_attractive.jpg"" />
<figcaption>Attractive AP vs. <span class="math inline"><em>Δ</em></span> EO</figcaption>
</figure>
<figcaption><em>Attractive Attribute of CelebA Dataset as the Target Attribute</em>. (a) reflects the trade-off between Average Precision and Demographic Parity Difference. (b) shows the trade-off between Average Precision and Equalized Odds Difference.</figcaption>
</figure>

<figure id="fig:celebA_smiling">
<figure id="fig:2">
<img src="./figures/CelebA_dp_smiling.jpg"" />
<figcaption>Smiling AP vs. <span class="math inline"><em>Δ</em></span> DP</figcaption>
</figure>
<figure id="fig:5">
<img src="./figures/celebA_eo_smiling.jpg"" />
<figcaption>Smiling AP vs. <span class="math inline"><em>Δ</em></span> EO</figcaption>
</figure>
<figcaption><em>Smiling Attribute of CelebA Dataset as the Target Attribute</em>. (a) reflects the trade-off between Average Precision and Demographic Parity Difference. (b) shows the trade-off between Average Precision and Equalized Odds Difference.</figcaption>
</figure>

<figure id="fig:celebA_wavy">
<figure id="fig:3">
<img src="./figures/CelebA_dp_wavy.jpg"" />
<figcaption>Wavy Hair AP vs. <span class="math inline"><em>Δ</em></span> DP</figcaption>
</figure>
<figure id="fig:6">
<img src="./figures/celebA_eo_wavy.jpg"" />
<figcaption>Wavy Hair AP vs. <span class="math inline"><em>Δ</em></span> EO</figcaption>
</figure>
<figcaption><em>Wavy Hair Attribute of CelebA Dataset as the Target Attribute</em>. (a) reflects the trade-off between Average Precision and Demographic Parity Difference. (b) shows the trade-off between Average Precision and Equalized Odds Difference. The FairBiNN method is showing competitive results to the baseline.</figcaption>
</figure>

We compare our method on vision task with (1) empirical risk minimization (ERM), which accomplishes training task without any regularization, (2) gap regularization, which directly regularizes the model, (3) adversarial debiasing `\citep{zhang2018mitigating}`{=latex}, and (4) Fairmixup `\citep{chuang2021fair}`{=latex}. To showcase the effectiveness of our method, we employed the CelebA dataset of face attributes `\citep{liu2015faceattributes}`{=latex}, which comprises over 200,000 images of celebrities. Each image in the dataset has been assigned 40 binary attributes, including gender, that were labeled by humans. We selected the attributes of attractive, smiling, and wavy hair and used them in three binary classification tasks, with gender serving as the protected attribute. The reason we chose these three attributes is that each of them has a group that is sensitive to them and receives a disproportionately high number of positive samples. We trained a ResNet-18 model `\cite{he2016deep}`{=latex} for each task and added two additional layers to predict the outcomes. The trade-off between Average Precision (AP), Demographic Parity (DP), and Equality of Opportunity (EO) for attributes "Attractive", "Smiling", and "Wavy Hair" is illustrated in the figures <a href="#fig:celebA_attractive" data-reference-type="ref" data-reference="fig:celebA_attractive">13</a>, <a href="#fig:celebA_smiling" data-reference-type="ref" data-reference="fig:celebA_smiling">16</a>, and <a href="#fig:celebA_wavy" data-reference-type="ref" data-reference="fig:celebA_wavy">19</a> respectively. Our proposed method provides a more balanced trade-off between accuracy and fairness. Instead of prioritizing one over the other, our method strikes a better balance, ensuring that the trained model is both accurate and fair. Moreover, the FairBiNN model consistently provides better equality of opportunity across various accuracy levels compared to benchmark models. Through empirical validation on multiple benchmarks, we’ve shown that the FairBiNN approach consistently outperforms other methods in achieving equality of opportunity across various accuracy levels. This indicates that our method can provide fair treatment to different protected groups while still maintaining high predictive accuracy.

<figure id="fig:tsne">
<figure id="fig:unfair_tsne">
<img src="./figures/no_fairness.jpg"" />
<figcaption>Without the fairness layers <span class="math inline"><em>z</em></span></figcaption>
</figure>
<figure id="fig:fair_tsne">
<img src="./figures/fairness.jpg"" />
<figcaption>With the fairness layers <span class="math inline"><em>z̃</em></span></figcaption>
</figure>
<figcaption>CelebA Dataset – t-SNE visualization of <span class="math inline"><em>z</em></span> and <span class="math inline"><em>z̃</em></span> labeled with gender classes. The invariant encoding <span class="math inline"><em>z̃</em></span> shows no clustering by gender. These plots are generated using attractive attribute.</figcaption>
</figure>

Furthermore, we demonstrate the power of the bilevel design for fairness by visualizing the t-SNE plot. The t-SNE visualization of \\(z\\) (output of the ResNet-18 before the classification layer without the fairness layers) and \\(\tilde{z}\\) (output of the ResNet-18 before the classification layer with the Bilevel fairness) are shown in Figures <a href="#fig:unfair_tsne" data-reference-type="ref" data-reference="fig:unfair_tsne">20</a> and <a href="#fig:fair_tsne" data-reference-type="ref" data-reference="fig:fair_tsne">21</a>, demonstrating that \\(z\\) clusters by gender, but \\(\tilde{z}\\) does not. Further insights about ablation study outcomes are detailed in section <a href="#ap:ablation" data-reference-type="ref" data-reference="ap:ablation">7.13</a>.

## Architecture visualization

To better illustrate the training process outlined in Algorithm <a href="#algorithm1" data-reference-type="ref" data-reference="algorithm1">1</a>, we present the network architecture in Figure <a href="#fig:fairbinn" data-reference-type="ref" data-reference="fig:fairbinn">23</a>.

<figure id="fig:fairbinn">
<img src="./figures/FairBiNN.png"" style="width:100.0%" />
<figcaption>The FairBiNN network architecture illustrating the process described in Algorithm <a href="#algorithm1" data-reference-type="ref" data-reference="algorithm1">1</a>.</figcaption>
</figure>

## Ablation Study [ap:ablation]

### Impact of \\(\eta\\) on Model Performance [ablation_eta]

To understand the sensitivity of our FairBiNN model to the choice of \\(\eta\\), we conducted an ablation study on both the Adult and Health datasets. The parameter \\(\eta\\) controls the trade-off between accuracy and fairness in our bilevel optimization framework.

### Experimental Setup

We varied \\(\eta\\) across a range of values: 1-6000 for Adult dataset, and 1-3000 for Health dataset. For each value of \\(\eta\\), we trained the FairBiNN model on both the Adult and Health datasets, keeping all other hyperparameters constant. We evaluated the models based on accuracy and demographic parity (DP).

### Results

<figure id="fig:eta_ablation">
<figure id="fig:eta_adult">
<img src="./figures/different_etas_adult.png"" />
<figcaption><span class="math inline"><em>η</em></span> ablation study on Adult dataset</figcaption>
</figure>
<figure id="fig:eta_health">
<img src="./figures/different_alphas_health.png"" />
<figcaption><span class="math inline"><em>η</em></span> ablation study on Health dataset</figcaption>
</figure>
<figcaption>Ablation study on the impact of <span class="math inline"><em>η</em></span> parameter across two datasets. (a) Results on the Adult dataset showing the effect of different <span class="math inline"><em>η</em></span> values. (b) Similar analysis conducted on the Health dataset, demonstrating how <span class="math inline"><em>η</em></span> influences model performance.</figcaption>
</figure>

Figures <a href="#fig:eta_adult" data-reference-type="ref" data-reference="fig:eta_adult">24</a> and <a href="#fig:eta_health" data-reference-type="ref" data-reference="fig:eta_health">25</a> show the results of our ablation study for the Adult and Health datasets, respectively. The results demonstrate a clear trade-off between accuracy and fairness as \\(\eta\\) varies. For both datasets: As \\(\eta\\) increases, the demographic parity (DP) decreases, indicating improved fairness. However, this improvement in fairness comes at the cost of reduced accuracy. The relationship is not linear; there are diminishing returns in fairness improvement as \\(\eta\\) increases, especially at higher values. For the Adult dataset, setting \\(\eta = 1000\\) appears to offer a good balance, achieving a DP of 0.012 while maintaining an accuracy of 82.9%. For the Health dataset, \\(\eta = 700\\) also provides a reasonable trade-off with a DP of 0.23 and an accuracy of 80.2%. These results highlight the importance of carefully tuning \\(\eta\\) to achieve the desired balance between accuracy and fairness. The optimal value may vary depending on the specific requirements of the application and the characteristics of the dataset.  
This ablation study demonstrates that our FairBiNN model provides a flexible framework for managing the accuracy-fairness trade-off through the \\(\eta\\) parameter. Practitioners can adjust \\(\eta\\) based on their specific fairness requirements and acceptable accuracy thresholds. Future work could explore adaptive schemes for setting \\(\eta\\) during training to automatically find an optimal balance.

#### Position of Fairness Layers [ablation_pos]

To understand the impact of the position of fairness layers within the network architecture, we conducted an ablation study varying their placement. This study aims to identify the optimal position for fairness layers and provide insights into why certain positions may be more effective.

**Experimental Setup**

We tested 4 fairness layer positions on the Adult dataset and 5 fairness layer positions on the Health dataset. For each configuration, we kept the total number of parameters constant to ensure a fair comparison. We evaluated the models based on accuracy and demographic parity (DP).

#### Results

<figure id="fig:pos_ablation">
<figure id="fig:pos_adult">
<img src="./figures/ablation_adult.png"" />
<figcaption>Fairness layer position ablation study on Adult dataset</figcaption>
</figure>
<figure id="fig:pos_health">
<img src="./figures/ablation_health.png"" />
<figcaption>Fairness layer position ablation study on Health dataset</figcaption>
</figure>
<figcaption>Fairness Layers Position (<span class="math inline"><em>i</em></span>), where <span class="math inline"><em>i</em></span> indicates the <span class="math inline"><em>i</em> − <em>t</em><em>h</em></span> hidden layer</figcaption>
</figure>

Figures <a href="#fig:pos_adult" data-reference-type="ref" data-reference="fig:pos_adult">27</a> and <a href="#fig:pos_health" data-reference-type="ref" data-reference="fig:pos_health">28</a> show the results of our ablation study for the Adult and Health datasets, respectively.  
The results consistently show that placing the fairness layers just before the output layer (in the last hidden layer) yields the best performance in terms of both accuracy and fairness. This configuration achieves the highest accuracy while maintaining the lowest demographic parity on both datasets.

Several factors contribute to the superior performance of fairness layers when placed in the last hidden layer:

- **Rich Feature Representations**: By the time the data reaches the last hidden layer, the network has already learned rich, high-level feature representations. This allows the fairness layers to operate on more informative features, potentially making it easier to identify and mitigate biases.

- **Minimal Information Loss**: Placing fairness layers earlier in the network might lead to loss of important information that could be useful for the classification task. By positioning them at the end, we ensure that all relevant features are preserved throughout most of the network.

- **Direct Influence on Output**: Being closest to the output layer, fairness layers in this position have the most direct influence on the final predictions. This allows for more effective bias mitigation without excessively disturbing the learned representations in earlier layers.

- **Gradient Flow**: In backpropagation, gradients from the fairness objective have a shorter path to travel when the fairness layers are near the output. This might lead to more stable and effective updates for bias mitigation.

- **Adaptability**: Fairness layers at the end of the network can adapt to various biases that might emerge from complex interactions in earlier layers, providing a final "correction" before the output.

### Number and Type of Fairness Layers [ablation_type]

In this subsection, we perform an ablation study to investigate the effects of different functions for the fairness layers. The fairness layer can be any differentiable function with controllable parameters denoted as \\(\theta_d\\). We experimented with three configurations for the fairness layers: one linear layer, two linear layers, and three linear layers on tabular datasets. The results of the ablation study are summarized in Table <a href="#tab:ablation_study" data-reference-type="ref" data-reference="tab:ablation_study">8</a>.  
For the CelebA dataset, we explored three types of fairness layers: linear layers, Residual Blocks (ResBlocks), and Convolutional Neural Network (CNN) layers. The mean scores of each category of CelebA attributes for each type of fairness layer are provided in Table <a href="#tab:ablation_study2" data-reference-type="ref" data-reference="tab:ablation_study2">9</a>.

The justification for the performance differences between the ResBlock and the fully connected models in our ablation study lies in the proportion of the model occupied by the fairness layers and the specific contributions of these layers to different parts of the network. In particular, there are two primary factors that explain the observed performance differences:

- **Role in the Network**: The ResBlock and the fully connected modules serve different purposes within the network. The ResBlock contributes to the embedding space of the image, which includes feature extraction and representation learning. This enables the model to capture the essential characteristics of the image while minimizing the effect of the protected attributes (e.g., gender) on the classification task. In contrast, the fully connected module is mainly involved in the classification part of the network, where it contributes to the decision-making process based on the features extracted from the previous layers. This distinction in roles explains why the ResBlock provides more fair results, as it directly affects the representation learning and reduces the influence of the protected attributes on the embeddings.

- **Flow of Data**: The flow of data through the ResBlock is different from the flow through the fully connected and CNN modules. ResBlocks have skip connections that allow the input to bypass some layers and directly flow to the subsequent layers. These skip connections help in preserving the original information and preventing the loss of critical features during the network’s forward pass. As a result, the ResBlock is more effective in capturing the inherent relationships in the data while mitigating the bias from the protected attributes `\cite{resnet}`{=latex}. In contrast, CNNs involve multiple convolution and pooling operations, which can cause the loss of some information relevant to fairness. The fully connected module, with its dense layers, lacks the skip connections present in the ResBlock, which can lead to less effective bias mitigation.

In conclusion, our ablation study demonstrates that the choice of layer in the fairness layers can significantly impact the fairness and accuracy of the model. It is essential to strike a balance between fairness and accuracy and to select the appropriate fairness layer for the specific dataset and application at hand.

<div class="center" markdown="1">

<div class="small" markdown="1">

<div class="sc" markdown="1">

<div id="tab:ablation_study" markdown="1">

| Method              | UCI Adult | Heritage Health |
|:--------------------|:---------:|:---------------:|
| One Linear Layer    | **0.411** |      0.492      |
| Two Linear Layers   |   0.404   |      0.513      |
| Three Linear Layers |   0.349   |    **0.531**    |

Area over the curve of statistical demographic parity and accuracy for model ablation

</div>

</div>

</div>

</div>

<div id="tab:ablation_study2" markdown="1">

|     CNNBlock     |    AP     | \\(\Delta\\)DP | \\(\Delta\\)EO |
|:----------------:|:---------:|:--------------:|:--------------:|
| One Linear Layer | **0.646** |     0.072      |   **0.084**    |
|  CNN Res Block   |   0.568   |    **0.04**    |     0.126      |
|    CNN Layer     |   0.617   |     0.058      |     0.099      |

Accumulative comparison between different fairness layers

</div>

# Ethical & Broader Social Impact

We believe the technical contributions of FairBiNN speak for themselves. By offering a principled bilevel optimisation framework that demonstrably improves the balance between accuracy and fairness on multiple benchmarks, our method is well-positioned for deployment in a broad range of machine-learning pipelines. Consequently, we anticipate FairBiNN will be readily integrated into existing tool-chains with minimal adjustment, providing practitioners with a robust avenue for enhancing overall model performance.