
## Abstract

Boosting is a highly successful ML-born optimization setting in which one is required to computationally efficiently learn arbitrarily good models based on the access to a weak learner oracle, providing classifiers performing at least slightly differently from random guessing. A key difference with gradient-based optimization is that boosting’s original model does not requires access to first order information about a loss, yet the decades long history of boosting has quickly evolved it into a first order optimization setting – sometimes even wrongfully *defining* it as such. Owing to recent progress extending gradient-based optimization to use only a loss’ zeroth (\\(0^{th}\\)) order information to learn, this begs the question: what loss functions can be efficiently optimized with boosting and what is the information really needed for boosting to meet the *original* boosting blueprint’s requirements?

We provide a constructive formal answer essentially showing that *any* loss function can be optimized with boosting and thus boosting can achieve a feat not yet known to be possible in the classical \\(0^{th}\\) order setting, since loss functions are not required to be be convex, nor differentiable or Lipschitz – and in fact not required to be continuous either. Some tools we use are rooted in quantum calculus, the mathematical field – not to be confounded with quantum computation – that studies calculus without passing to the limit, and thus without using first order information.

# Introduction [sec:intro]

In ML, zeroth order optimization has been devised as an alternative to techniques that would otherwise require access to \\(\geq 1\\)-order information about the loss to minimize, such as gradient descent (stochastic or not, constrained or not, etc., see Section <a href="#sec:related" data-reference-type="ref" data-reference="sec:related">2</a>). Such approaches replace the access to a so-called *oracle* providing derivatives for the loss at hand, operations that can be consuming or not available in exact form in the ML world, by the access to a cheaper function value oracle, providing loss values at queried points.

Zeroth order optimization has seen a considerable boost in ML over the past years, over many settings and algorithms, yet, there is one foundational ML setting and related algorithms that, to our knowledge, have not yet been the subject of investigations: boosting `\cite{kTO,kvAI}`{=latex}. Such a question is very relevant: boosting has quickly evolved as a technique requiring first-order information about the loss optimized `\cite[Section 10.3]{bLTF}`{=latex}, `\cite[Section 7.2.2]{mrtFO}`{=latex} `\cite{wvSO}`{=latex}. It is also not uncommon to find boosting reduced to this first-order setting `\cite{bcrAG}`{=latex}. However, originally, the boosting model did not mandate the access to any first-order information about the loss, rather requiring access to a weak learner providing classifiers at least slightly different from random guessing `\cite{kvAI}`{=latex}. In the context of zeroth-order optimization gaining traction in ML, it becomes crucial to understand not just whether differentiability is necessary for boosting, but more generally what are loss functions that can be boosted with a weak learner and *in fine* where boosting stands with respect to recent formal progress on lifting gradient descent to zeroth-order optimisation.

**In this paper**, we settle the question: we design a formal boosting algorithm for any loss function whose set of discontinuities has zero Lebesgue measure. With traditional floating point encoding (e.g. float64), any stored loss function would *de facto* meet this condition; mathematically speaking, we encompass losses that are not necessarily convex, nor differentiable or Lipschitz. This is a key difference with classical zeroth-order optimization results where the algorithms are zeroth-order *but* their proof of convergence makes various assumptions about the loss at hand, such as convexity, differentiability (once or twice), Lipschitzness, etc. . Our proof technique builds on a simple boosting technique for convex functions that relies on an order-one Taylor expansion to bound the progress between iterations `\cite{nwLO}`{=latex}. Using tools from quantum calculus[^1], we replace this progress using \\(v\\)-derivatives and a quantity related to a generalisation of the Bregman information `\cite{bmdgCW}`{=latex}. The boosting rate involves the classical weak learning assumption’s advantage over random guessing and a new parameter bounding the ratio of the expected weights (squared) over a generalized notion of curvature involving \\(v\\)-derivatives. Our algorithm, which learns a linear model, introduces notable generalisations compared to the AdaBoost / gradient boosting lineages, chief among which the computation of acceptable *offsets* for the \\(v\\)-derivatives used to compute boosting weights, offsets being zero for classical gradient boosting. To preserve readability and save space, all proofs and additional information are postponed to an .

# Related work [sec:related]

Over the past years, ML has seen a substantial push to get the cheapest optimisation routines, in general batch `\cite{clmyAZ}`{=latex}, online `\cite{hmmrZO}`{=latex}, distributed `\cite{aptDZ}`{=latex}, adversarial `\cite{cwzOT,clxllhcZZ}`{=latex} or bandits settings `\cite{aptEH}`{=latex} or more specific settings like projection-free `\cite{ghCS,htcAS,szkTG}`{=latex} or saddle-point optimisation `\cite{fvpEA,mcmsrZO}`{=latex}. We summarize several dozen recent references in Table <a href="#tab:features" data-reference-type="ref" data-reference="tab:features">[tab:features]</a> in terms of assumptions for the analysis about the loss optimized, provided in , Section <a href="#sec-sup-sum" data-reference-type="ref" data-reference="sec-sup-sum">8</a>. *Zeroth-order* optimization reduces the information available to the learner to the "cheapest" one which consists in (loss) function values, usually via a so-called function value *oracle*. However, as Table <a href="#tab:features" data-reference-type="ref" data-reference="tab:features">[tab:features]</a> shows, the loss itself is always assumed to have some form of "niceness" to study the algorithms’ convergence, such as differentiability, Lipschitzness, convexity, etc. . Another quite remarkable phenomenon is that throughout all their diverse settings and frameworks, not a single one of them addresses boosting. Boosting is however a natural candidate for such investigations, for two reasons. First, the most widely used boosting algorithms are first-order information hungry `\cite{bLTF,mrtFO,wvSO}`{=latex}: they require access to derivatives to compute examples’ weights and classifiers’ leveraging coefficients. Second and perhaps most importantly, unlike other optimization techniques like gradient descent, the original boosting model *does not* mandate the access to a first-order information oracle to learn, but rather to a weak learning oracle which supplies classifiers performing slightly differently from random guessing `\cite{kTO,kvAI}`{=latex}. Only few approaches exist to get to "cheaper” algorithms relying on less assumptions about the loss at hand, and to our knowledge do not have boosting-compliant convergence proofs, as for example when alleviating convexity `\cite{cefNC,ppIN}`{=latex} or access to gradients of the loss `\cite{wrTC}`{=latex}. Such questions are however important given the early negative results on boosting convex potentials with first-order information `\cite{lsRC}`{=latex} and the role of the classifiers in the negative results `\cite{mnwRC}`{=latex}.

Finally, we note that a rich literature has developed in mathematics as well for derivative-free optimisation `\cite{lmwDF}`{=latex}, yet methods would also often rely on assumptions included in the three above (*e.g.* `\cite{nsRG}`{=latex}). It must be noted however that derivative-free optimisation has been implemented in computers for more than seven decades `\cite{fmNS}`{=latex}.

# Definitions and notations [sec:defs]

The following shorthands are used: \\([n]
\defeq \{1, 2, ..., n\}\\) for \\(n \in \mathbb{N}_*\\), \\(z \cdot [a, b]
\defeq [\min\{za,zb\}, \max\{za,zb\}]\\) for \\(z \in \mathbb{R}, a \leq  b \in \mathbb{R}\\). In the batch supervised learning setting, one is given a training set of \\(m\\) examples \\(S \defeq \{({\ve{x}}_i, y_i), i \in [m]\}\\), where \\({\ve{x}}_i
\in {\mathcal{X}}\\) is an observation (\\({\mathcal{X}}\\) is called the domain: often, \\({\mathcal{X}}\subseteq {\mathbb{R}}^d\\)) and \\(y_i
\in \mathcal{Y} \defeq \{-1,1\}\\) is a label, or class. We study the empirical convergence of boosting, which requires fast convergence on training. Such a setting is standard in zeroth order optimization `\cite{nsRG}`{=latex}. Also, investigating generalization would entail specific design choices about the loss at hand and thus would restrict the scope of our result (see *e.g.* `\cite{bmRA}`{=latex}). The objective is to learn a *classifier*, *i.e.* a function \\(h
: \mathcal{X} \rightarrow \mathbb{R}\\) which belongs to a given set \\(\mathcal{H}\\). The goodness of fit of some \\(h\\) on \\(S\\) is evaluated from a given function \\(F:\mathbb{R} \rightarrow \mathbb{R}\\) called a loss function, whose expectation on training is sought to be minimized: \\[\begin{aligned}
F(S, h)& \defeq & \E_{i \sim [m]} [F(y_ih(\ve{x}_i))] .
\end{aligned}\\] The set of most popular losses comprises convex functions: the exponential loss (\\(\expsur(z) \defeq \exp(-z)\\)), the logistic loss (\\(\logsur(z) \defeq
\log(1+\exp(-z))\\)), the square loss (\\(\sqsur(z) \defeq (1-z)^2\\)), the Hinge loss (\\(\hingesur(z) \defeq \max\{0, 1-z\}\\)). These are surrogate losses because they all define upperbounds of the 0/1-loss (\\(\zosur(z) \defeq 1_{z\leq 0}\\), "\\(1\\)” being the indicator variable).

Our ML setting is that of boosting `\cite{kvAI}`{=latex}. It consists in having primary access to a weak learner \\(\weaklearner\\) that when called, provides so-called weak hypotheses, weak because barely anything is assumed in terms of classification performance relatively to the sample over which they were trained. Our goal is to devise a so-called "boosting" algorithm that can take any loss \\(F\\) *as input* and training sample \\(S\\) and a target loss value \\(F_*\\) and after some \\(T\\) calls to the weak learner crafts a classifier \\(H_T\\) satisfying \\(F(S, H_T) \leq F_*\\), where \\(T\\) depends on various parameters of the ML problem. Our boosting architecture is a linear model: \\(H_T \defeq \sum_t \alpha_t h_t\\) where each \\(h_t\\) is an output from the weak learner and leveraging coefficients \\(\alpha_t\\) have to be computed during boosting. Notice that this is substantially more general than the classical boosting formulation where the loss would be fixed or belong to a restricted subset of functions.

# \\(v\\)-derivatives and Bregman secant distortions [sec:bregsec]

Unless otherwise stated, in this Section, \\(F\\) is a function defined over \\(\mathbb{R}\\).

<div class="definition" markdown="1">

<span id="defVDER" label="defVDER"></span>`\cite{kcQC}`{=latex} For any \\(z,v \in \mathbb{R}\\), we let \\(\diffloc{v}{F}(z) \defeq (F(z+v) - F(z))/v\\) denote the \\(v\\)-derivative of \\(F\\) in \\(z\\).

</div>

<figure id="f-SD">
<div class="center">
<table>
<tbody>
<tr>
<td style="text-align: center;"><img src="./figures/FigSD-2.eps"" style="width:50.0%" /></td>
<td style="text-align: center;"><img src="./figures/FigBI-2.png"" style="width:50.0%" /></td>
</tr>
</tbody>
</table>
</div>
<figcaption><em>Left</em>: value of <span class="math inline">$\bregmansec{F}{v}(
  z'\|z)$</span> for convex <span class="math inline"><em>F</em></span>, <span class="math inline">$v \defeq z_4 - z$</span> and various <span class="math inline"><em>z</em><sup>′</sup></span> (colors), for which the Bregman Secant distortion is positive (<span class="math inline"><em>z</em><sup>′</sup> = <em>z</em><sub>1</sub></span>, green), negative (<span class="math inline"><em>z</em><sup>′</sup> = <em>z</em><sub>2</sub></span>, red), minimal (<span class="math inline"><em>z</em><sup>′</sup> = <em>z</em><sub>3</sub></span>) or null (<span class="math inline"><em>z</em><sup>′</sup> = <em>z</em><sub>4</sub>, <em>z</em></span>). <em>Right</em>: depiction of <span class="math inline"><em>Q</em><sub><em>F</em></sub>(<em>z</em>, <em>z</em> + <em>v</em>, <em>z</em><sup>′</sup>)</span> for non-convex <span class="math inline"><em>F</em></span> (Definition <a href="#defBI" data-reference-type="ref" data-reference="defBI">[defBI]</a>).</figcaption>
</figure>

This expression, which gives the classical derivative when the *offset* \\(v\rightarrow 0\\), is called the *\\(h\\)-derivative* in quantum calculus `\cite[Chapter 1]{kcQC}`{=latex}. We replaced the notation for the risk of confusion with classifiers. Notice that the \\(v\\)-derivative is just the slope of the secant that passes through points \\((z, F(z))\\) and \\((z+v,
F(z+v))\\) (Figure <a href="#f-SD" data-reference-type="ref" data-reference="f-SD">1</a>). Higher order \\(v\\)-derivatives can be defined with the same offset used several times `\cite{kcQC}`{=latex}. Here, we shall need a more general definition that accommodates for variable offsets.

<div class="definition" markdown="1">

<span id="defsecZ" label="defsecZ"></span> Let \\(v_1, v_2, ..., v_n \in \mathbb{R}\\) and \\(\mathcal{V} \defeq \{v_1, v_2, ..., v_n\}\\) and \\(z \in \mathbb{R}\\). The \\(\mathcal{V}\\)-derivative \\(\diffloc{\mathcal{V}}{F}\\) is: \\[\begin{aligned}
  \diffloc{\mathcal{V}}{F}(z) & \defeq & \left\{
                                         \begin{array}{rcl}
                                           F(z) & \mbox{ if } & \mathcal{V} = \emptyset\\
                                           \diffloc{v_1}{F}(z) & \mbox{
                                           if } & \mathcal{V}
                                                  =\{v_1\}\\
                                           \diffloc{\{v_n\}}{(\diffloc{\mathcal{V}\backslash
                                         \{v_n\} }{F})}(z) &
                                                            \multicolumn{2}{l}{\mbox{ otherwise}}
                                           \end{array}
                                         \right. .
\end{aligned}\\] If \\(v_i = v, \forall i \in [n]\\) then we write \\(\difflocorder{n}{v}{F}(z) \defeq \diffloc{\mathcal{V}}{F}(z)\\).

</div>

In the , Lemma <a href="#lemCombSec" data-reference-type="ref" data-reference="lemCombSec">[lemCombSec]</a> computes the unravelled expression of \\(\diffloc{\mathcal{V}}{F}(z)\\), showing that the order of the elements in \\(\mathcal{V}\\) does not matter; \\(n\\) is called the order of the \\(\mathcal{V}\\)-derivative.

We can now define a generalization of Bregman divergences called *Bregman Secant distortions*.

<div class="definition" markdown="1">

For any \\(z, z', v\in \mathbb{R}\\), the Bregman Secant distortion \\(\bregmansec{F}{v}(
  z'\|z)\\) with generator \\(F\\) and offset \\(v\\) is: \\[\begin{aligned}
\bregmansec{F}{v}(
  z'\|z) & \defeq & F(z') - F(z) - 
  (z'-z)\diffloc{v}{F}(z).
  
\end{aligned}\\]

</div>

Even if \\(F\\) is convex, the distortion is not necessarily positive, though it is lowerbounded (Figure <a href="#f-SD" data-reference-type="ref" data-reference="f-SD">1</a>). There is an intimate relationship between the Bregman Secant distortions and Bregman divergences. We shall use a definition slightly more general than the original one when \\(F\\) is differentiable `\cite[eq. (1.4)]{bTR}`{=latex}, introduced in information geometry `\cite[Section 3.4]{anMO}`{=latex} and recently reintroduced in ML `\cite{bmnLW}`{=latex}.

<div class="definition" markdown="1">

The Bregman divergence with generator \\(F\\) (scalar, convex) between \\(z'\\) and \\(z\\) is \\(D_F(z'\|z) \defeq F(z') + F^\star(z) - z'z\\), where \\(F^\star(z) \defeq \sup_t tz - F(t)\\) is the convex conjugate of \\(F\\).

</div>

We state the link between \\(\bregmansec{F}{v}\\) and \\(D_F\\) (proof omitted).

<div class="lemma" markdown="1">

Suppose \\(F\\) strictly convex differentiable. Then \\(\lim_{v\rightarrow 0} \bregmansec{F}{v}(z'\|z) = D_F(z'\| F'(z))\\).

</div>

Relaxed forms of Bregman divergences have been introduced in information geometry `\cite{nnTB}`{=latex}.

<div class="definition" markdown="1">

<span id="defBI" label="defBI"></span> For any \\(a, b, \alpha\in \mathbb{R}\\), denote for short \\(\mathbb{I}_{a,b} \defeq
[\min\{a, b\}, \max\{a, b\}]\\) and \\((uv)_\alpha \defeq \alpha u + (1-\alpha)v\\). The *Optimal Bregman Information* (OBI) of \\(F\\) defined by triple \\((a,b,c) \in \mathbb{R}^3\\) is: \\[\begin{aligned}
  Q_{F}(a, b, c) \defeq \max_{\alpha: (ab)_\alpha \in \mathbb{I}_{a,c}}  \{(F(a)F(b))_\alpha - F((ab)_\alpha)\}. \label{eqBI}
\end{aligned}\\]

</div>

As represented in Figure <a href="#f-SD" data-reference-type="ref" data-reference="f-SD">1</a> (right), the OBI is obtained by drawing the line passing through \\((a,F(a))\\) and \\((b,F(b))\\) and then, in the interval \\(\mathbb{I}_{a,c}\\), look for the maximal difference between the line and \\(F\\). We note that \\(Q_{F}\\) is non negative because \\(a \in \mathbb{I}_{a,c}\\) and for the choice \\(\alpha = 1\\), the RHS in <a href="#eqBI" data-reference-type="eqref" data-reference="eqBI">[eqBI]</a> is 0. We also note that when \\(F\\) is convex, the RHS is indeed the maximal Bregman information of two points in `\cite[Definition 2]{bmdgCW}`{=latex}, where maximality is obtained over the probability measure. The following Lemma follows from the definition of the Bregman secant divergence and the OBI. An inspection of the functions in Figure <a href="#f-SD" data-reference-type="ref" data-reference="f-SD">1</a> provides a graphical proof.

<div class="lemma" markdown="1">

<span id="lemGENR" label="lemGENR"></span> For any \\(F\\), \\[\begin{aligned}
\forall z, v, z'\in \mathbb{R}, \bregmansec{F}{v}(z'\|z) & \geq & -Q_{F}(z,z+v, z'). \label{bregi1}
\end{aligned}\\] and if \\(F\\) is convex, \\[\begin{aligned}
\forall z, v\in \mathbb{R}, \forall z'\not\in \mathbb{I}_{z,z+v}, \bregmansec{F}{v}(
  z'\|z) \geq 0,\nonumber\\
\forall z, v, z'\in \mathbb{R}, \bregmansec{F}{v}(
  z'\|z) \geq -Q_{F}(z,z+v,z+v). \label{bregi2}
\end{aligned}\\]

</div>

We shall abbreviate the two possible forms of OBI in the RHS of <a href="#bregi1" data-reference-type="eqref" data-reference="bregi1">[bregi1]</a>, <a href="#bregi2" data-reference-type="eqref" data-reference="bregi2">[bregi2]</a> as: \\[\begin{aligned}
Q^*_F(z,z',v) \defeq \left\{
\begin{array}{cl}
Q_{F}(z,z+v,z+v) & \mbox{ if $F$ convex}\\
Q_{F}(z,z+v, z') & \mbox{ otherwise}
\end{array}
\right. .\label{defQSTAR}
\end{aligned}\\]

# Boosting using only queries on the loss [sec:boost]

We make the assumption that predictions of so-called "weak classifiers" are finite and non-zero on training without loss of generality (otherwise a simple tweak ensures it without breaking the weak learning framework, see , Section <a href="#remove-nonzero" data-reference-type="ref" data-reference="remove-nonzero">9.2</a>). Excluding 0 ensures our algorithm does not make use of derivatives.

<div class="assumption" markdown="1">

<span id="assum1finite" label="assum1finite"></span> \\(\forall t>0, \forall i\in [m]\\), \\(|h_t(\ve{x}_i)| \in (0,+\infty)\\) (we thus let \\(M_t \defeq \max_i |h_t(\ve{x}_i)|\\)).

</div>

For short, we define two *edge* quantities for \\(i \in [m]\\) and \\(t=1, 2, ...\\), \\[\begin{aligned}
e_{ti} \defeq \alpha_{t} \cdot  y_i h_{t}(\ve{x}_i), \quad \tilde{e}_{ti} \defeq y_i H_{t}(\ve{x}_i)\label{defEDGE12},
\end{aligned}\\] where \\(\alpha_t\\) is a leveraging coefficient for the weak classifiers in an ensemble \\(H_T(.) \defeq \sum_{t\in [T]} \alpha_{t} h_{t}(.)\\). We observe \\[\begin{aligned}
\tilde{e}_{ti} & = & \tilde{e}_{(t-1)i} + e_{ti}.
\end{aligned}\\]

## Algorithm: 

### General steps

<div class="algorithm*" markdown="1">

<div class="algorithmic" markdown="1">

ALGORITHM BLOCK (caption below)

**Input** sample \\({S} = \{(\bm{x}_i, y_i), i
  = 1, 2, ..., m\}\\), number of iterations \\(T\\), initial \\((h_0,v_0)\\) (constant classification and offset). <span id="s1">Step 1</span> : let \\(H_0 \leftarrow 1\cdot h_0\\) and ; // \\(h_0, v_0\neq 0\\) chosen s. t. \\(\diffloc{v_{0}}{F}(h_0)
\neq 0\\) Step 2 : **for** \\(t = 1, 2, ..., T\\) <span id="s21">Step 2.1</span> : let \\(h_t \leftarrow
\weaklearner({S}_t, |\bm{w}_t|)\\) //weak learner call, Step 2.2 : let \\(\eta_t   \leftarrow
(1/m) \cdot \sum_{i}
{w_{ti} y_{i} h_t(\bm{x}_i)}\\) //unnormalized edge <span id="s23">Step 2.3</span> : <span id="s24">Step 2.4</span> : let \\(H_{t} \leftarrow H_{t-1} + \alpha_t \cdot h_t\\) //classifier update : **if** \\(\mathbb{I}_{ti}(\varepsilon_{t} \cdot \alpha_t^2 M_{t}^2  \overline{W}_{2,t})\neq \emptyset, \forall i \in [m]\\) **then** //new offsets **for** \\(i = 1, 2, ..., m\\), let \\[\begin{aligned}
\mbox{$v_{ti} \leftarrow \offsetoracle(t, i, \varepsilon_{t} \cdot \alpha_t^2 M_{t}^2  \overline{W}_{2,t})$}\:\:;
\end{aligned}\\] **else** **return** \\(H_t\\); <span id="s26">Step 2.6</span> : **for** \\(i = 1, 2, ..., m\\), let //weight update \\[\begin{aligned}
\mbox{\fcolorbox{red}{white}{$w_{(t+1)i} \leftarrow - \diffloc{v_{ti}}{F}(y_i
                                             H_{t}(\ve{x}_i))$}}\:\:; \label{pickun}
\end{aligned}\\] : **if** \\(\ve{w}_{t+1} = \ve{0}\\) **then** break; **Return** \\(H_T\\).

</div>

</div>

Without further ado, Algorithm  presents our approach to boosting without using derivatives information. The key differences with traditional boosting algorithms are red color framed. We summarize its key steps.  
[**Step 1**](#s1) This is the initialization step. Traditionally in boosting, one would pick \\(h_0 = 0\\). Note that \\(\bm{w}_1\\) is not necessarily positive. \\(v_0\\) is the initial offset (Section <a href="#sec:bregsec" data-reference-type="ref" data-reference="sec:bregsec">4</a>).  
[**Step 2.1**](#s21) This step calls the weak learner, as in traditional boosting, using variable "weights" on examples (the coordinate-wise absolute value of \\(\bm{w}_t\\), denoted \\(|\bm{w}_t|\\)). The key difference with traditional boosting is that examples labels can switch between iterations as well, which explains that the training sample, \\(S_t\\), is indexed by the iteration number.  
[**Step 2.3**](#s23) This step computes the leveraging coefficient \\(\alpha_t\\) of the weak classifier \\(h_t\\). It involves a quantity, \\(\overline{W}_{2,t}\\), which we define as any strictly positive real satisfying \\[\begin{aligned}
    \expect_{i\sim [m]}\left[ \diffloc{\{e_{ti},v_{(t-1)i}\}}{F}(\tilde{e}_{(t-1)i}) \cdot \left(\frac{h_{t}(\ve{x}_i)}{M_{t}}\right)^2 \right] \leq \overline{W}_{2,t}. \label{boundW2}
  
\end{aligned}\\] For boosting rate’s sake, we should find \\(\overline{W}_{2,t}\\) as small as possible. We refer to <a href="#defEDGE12" data-reference-type="eqref" data-reference="defEDGE12">[defEDGE12]</a> for the \\(e_., \tilde{e}_.\\) notations; \\(v_.\\) is the current (set of) offset(s) (Section <a href="#sec:bregsec" data-reference-type="ref" data-reference="sec:bregsec">4</a> for their definition). The second-order \\(\mathcal{V}\\)-derivative in the LHS plays the same role as the second-order derivative in classical boosting rates, see for example `\cite[Appendix, eq. 29]{nwLO}`{=latex}. As offsets \\(\rightarrow 0\\), it converges to a second-order derivative; otherwise, they still share some properties, such as the sign for convex functions.

<div class="lemma" markdown="1">

<span id="lemW2bound" label="lemW2bound"></span> Suppose \\(F\\) convex. For any \\(a\in \mathbb{R}, b, c\in \mathbb{R}_*\\), \\(\diffloc{\{b, c\}}{F}(a) \geq 0\\).

</div>

(Proof in , Section <a href="#proof_lemW2bound" data-reference-type="ref" data-reference="proof_lemW2bound">9.3</a>) We can also see a link with weights variation since, modulo a slight abuse of notation, we have \\(\diffloc{\{e_{ti},v_{(t-1)i}\}}{F}(\tilde{e}_{(t-1)i}) =  \diffloc{e_{ti}}{w_{ti}}\\). A substantial difference with traditional boosting algorithms is that we have two ways to pick the leveraging coefficient \\(\alpha_t\\); the first one can be used when a convenient \\(\overline{W}_{2,t}\\) is directly accessible from the loss. Otherwise, there is a simple algorithm that provides parameters (including \\(\overline{W}_{2,t}\\)) such that <a href="#boundW2" data-reference-type="eqref" data-reference="boundW2">[boundW2]</a> is satisfied. Section <a href="#sec-alphaW2" data-reference-type="ref" data-reference="sec-alphaW2">5.3</a> details those two possibilities and their implementation. In the more favorable case (the former one), \\(\alpha_t\\) can be chosen in an interval, furthermore defined by flexible parameters \\(\epsilon_t > 0, \pi_t \in (0,1)\\). Note that fixing beforehand these parameters is not mandatory: we can also pick *any* \\[\begin{aligned}
\alpha_t & \in & \eta_t \cdot \left(0, \frac{1}{M_t^2 \overline{W}_{2,t}}\right),\label{genALPHA}
\end{aligned}\\] and then compute choices for the corresponding \\(\epsilon_t\\) and \\(\pi_t\\). \\(\epsilon_t\\) is important for the algorithm and both parameters are important for the analysis of the boosting rate. From the boosting standpoint, a smaller \\(\epsilon_t\\) yields a larger \\(\alpha_t\\) and a smaller \\(\pi_t\\) reduces the interval of values in which we can pick \\(\alpha_t\\); both cases tend to favor better convergence rates as seen in Theorem <a href="#thBOOSTCH" data-reference-type="ref" data-reference="thBOOSTCH">[thBOOSTCH]</a>.  
[**Step 2.4**](#s24) is just the crafting of the final model.  
[**Step 2.5**](#s25) is new to boosting, the use of a so-called offset oracle, detailed in Section <a href="#subsec-offo" data-reference-type="ref" data-reference="subsec-offo">5.1.2</a>.  
[**Step 2.6**](#s26) The weight update does not rely on a first-order oracle as in traditional boosting, but uses only loss values through \\(v\\)-derivatives. The finiteness of \\(F\\) implies the finiteness of weights.  
[**Step 2.7**](#s27) Early stopping happens if all weights are null. While this would never happen with traditional (*e.g.* strictly convex) losses, some losses that are unusual in the context of boosting can lead to early stopping. A discussion on early stopping and how to avoid it is in Section <a href="#sec:disc" data-reference-type="ref" data-reference="sec:disc">6</a>.

### The offset oracle,  [subsec-offo]

Let us introduce notation \\[\begin{aligned}
\mathbb{I}_{ti}(z) \defeq \left\{ v  :
                          Q^*_F(\tilde{e}_{ti},\tilde{e}_{(t-1)i},v)
                          \leq z\right\}, \forall i\in [m], \forall z>0\label{defIT}.
  
\end{aligned}\\] (see Figure <a href="#f-Iti-const" data-reference-type="ref" data-reference="f-Iti-const">4</a> below to visualize \\(\mathbb{I}_{ti}(z)\\) for a non-convex \\(F\\)) The offset oracle is used in Step 2.5, which is new to boosting. It requests the offsets to carry out weight update in <a href="#pickun" data-reference-type="eqref" data-reference="pickun">[pickun]</a> to an *offset oracle*, which achieves the following, for iteration \\(\# t\\), example \\(\# i\\), limit OBI \\(z\\): \\[\begin{aligned}
\mbox{$\offsetoracle(t, i, z)$ returns some $v \in \mathbb{I}_{ti}(z)$} \label{constoo}
    
\end{aligned}\\] Note that the offset oracle has the freedom to pick the offset in a whole set. Section <a href="#sec-offset" data-reference-type="ref" data-reference="sec-offset">5.4</a> investigates implementations of the offset oracle, so let us make a few essentially graphical remarks here.  does not need to build the whole \\(\mathbb{I}_{ti}(z)\\) to return some \\(v \in \mathbb{I}_{ti}(z)\\) for Step 2.5 in . In the construction steps of Figure <a href="#f-Iti-const" data-reference-type="ref" data-reference="f-Iti-const">4</a>, as soon as \\(\mathcal{O} \neq \emptyset\\), one element of \\(\mathcal{O}\\) can be returned. Figure <a href="#f-BII-1" data-reference-type="ref" data-reference="f-BII-1">5</a> presents more examples of \\(\mathbb{I}_{ti}(z)\\). One can remark that the sign of the offset \\(v_{ti}\\) in Step 2.5 of  is the same as the sign of \\(\tilde{e}_{(t-1)i} - \tilde{e}_{ti} = - y_i
\alpha_{t} h_{t}(\ve{x}_i)\\). Hence, unless \\(F\\) is derivable or all edges \\(y_i h_{t}(\ve{x}_i)\\) are of the same sign (\\(\forall i\\)), the set of offsets returned in Step 2.5 always contain at least two different offsets, one non-negative and one non-positive (Figure <a href="#f-BII-1" data-reference-type="ref" data-reference="f-BII-1">5</a>, (a-b)).

## Convergence of  [sec-conv-boost]

The offset oracle has a technical importance for boosting: \\(\mathbb{I}_{ti}(z)\\) is the set of offsets that limit an OBI for a training example (Definition <a href="#defBI" data-reference-type="ref" data-reference="defBI">[defBI]</a>). The importance for boosting comes from Lemma <a href="#lemGENR" data-reference-type="ref" data-reference="lemGENR">[lemGENR]</a>: upperbounding an OBI implies lowerbounding a Bregman Secant divergence, which will also guarantee a sufficient slack between two successive boosting iterations. This is embedded in a blueprint of a proof technique to show boosting-compliant convergence which is not new, see *e.g.* `\cite{nwLO}`{=latex}. We now detail this convergence.

Remark that the expected edge \\(\eta_t\\) in Step 2.2 of  is not normalized. We define a normalized version of this edge as: \\[\begin{aligned}
[-1,1] \ni \tilde{\eta}_t & \defeq & \sum_i \frac{|w_{ti}|}{W_t} \cdot \tilde{y}_{ti}
                                     \cdot
                                     \frac{h_t(\ve{x}_i)}{M_t}, \label{deftildeeta}
\end{aligned}\\] with \\(\tilde{y}_{ti} \defeq y_i \cdot \mathrm{sign}(w_{ti})\\), \\(W_t \defeq \sum_i |w_{ti}| = \sum_i |\diffloc{v_{(t-1)i}}{F}(\tilde{e}_{(t-1)i})|\\). Remark that the labels are corrected by the weight sign and thus may switch between iterations. In the particular case where the loss is non-increasing (such as with traditional convex surrogates), the labels do not switch. We need also a quantity which is, in absolute value, the expected weight: \\[\begin{aligned}
\overline{W}_{1,t} \defeq  \left|\expect_{i\sim [m]}\left[\diffloc{v_{(t-1)i}}{F}(\tilde{e}_{(t-1)i})\right]\right| & & \quad(\mbox{we indeed observe $\overline{W}_{1,t} = |\expect_{i\sim [m]}\left[w_{ti}\right]|$}) \label{eqdefw1}.
  
\end{aligned}\\] In classical boosting for convex decreasing losses[^2], weights are non-negative and converge to a minimum (typically 0) as examples get the right class with increasing confidence. Thus, \\(\overline{W}_{1,t}\\) can be an indicator of when classification becomes "good enough" to stop boosting. In our more general setting, it shall be used in a similar indicator. We are now in a position to show a first result about .

<div class="theorem" markdown="1">

<span id="thBOOSTCH" label="thBOOSTCH"></span> Suppose assumption <a href="#assum1finite" data-reference-type="ref" data-reference="assum1finite">[assum1finite]</a> holds. Let \\(F_0 \defeq F(S, h_0)\\) in  and \\(z^*\\) any real such that \\(F(z^*) \leq F_0\\). Then we are guaranteed that classifier \\(H_T\\) output by  satisfies \\(F(S, H_T) \leq F(z^*)\\) when the number of boosting iterations \\(T\\) yields: \\[\begin{aligned}
\sum_{t=1}^T \frac{\overline{W}^2_{1,t} (1-\pi_{t}^2)}{\overline{W}_{2,t} (1+\varepsilon_{t}) }\cdot \tilde{\eta}^2_t& \geq & 4(F_0 - F(z^*)), \label{beq17m}
\end{aligned}\\] where parameters \\(\varepsilon_t, \pi_t\\) appear in Step 2.3 of .

</div>

(proof in , Section <a href="#proof_thBOOSTCH" data-reference-type="ref" data-reference="proof_thBOOSTCH">9.4</a>) We observe the tradeoff between the freedom in picking parameters and convergence guarantee as exposed by <a href="#beq17m" data-reference-type="eqref" data-reference="beq17m">[beq17m]</a>: to get more freedom in picking the leveraging coefficient \\(\alpha_t\\), we typically need \\(\pi_t\\) large (Step 2.3) and to get more freedom in picking the offset \\(v_t\neq 0\\), we typically need \\(\varepsilon_t\\) large (Step 2.5). However, allowing more freedom in such ways reduces the LHS and thus impairs the guarantee in <a href="#beq17m" data-reference-type="eqref" data-reference="beq17m">[beq17m]</a>. Therefore, there is a subtle balance between "freedom" of choice and convergence. This balance becomes more clear as boosting compliance formally enters convergence requirement.

#### Boosting-compliant convergence

In order to translate the per–round decrease established in (20) into a genuine finite-time rate we invoke the usual weak-learning postulate.  We purposely keep the statement at an intuitive level here; the exact random variables and normalisations that the edge is taken with respect to will be introduced once all required notation has been laid out in Sections 5.2–5.3.

<div class="assumption" markdown="1">

<span id="assum55wla" label="assum55wla"></span> (**Assumption 5.5 – γ-Weak Learner**)

There exists a constant γ > 0 such that, at every boosting round t, the weak learner returns a hypothesis h_t whose curvature-normalised edge—see Eq. (38) in Section 5.3 for the exact definition—satisfies
\[\text{edge}_t\;\ge\;\gamma.\]

The expectation, the normalisation factor and the sampling distribution implicit in the above edge coincide with those generated by the weight update rule (15) and the curvature proxy W_{2,t} introduced in Section 5.2.

</div>

Assumption 5.5 is deliberately minimal: it reduces to the classical weak-learning requirement when the loss is smooth, yet it remains applicable in the highly irregular setting we consider because it automatically incorporates the data-dependent offsets v_{ti}.  No additional boundedness, VC-dimension or moment conditions are needed.

Plugging Assumption 5.5 into the telescoping argument of the previous subsection immediately yields the following complexity bound; its proof is unchanged and therefore omitted.

<div class="corollary" markdown="1">

Under Assumptions 1.1, 2.1 and 5.5, if  is run for a number of rounds T satisfying
\[
T\;\ge\;\frac{4\bigl(F_0-F(z)\bigr)}{\gamma^{2}\,\rho}\;\cdot\;\frac{1+\max_{t\le T}\varepsilon_t}{1-\max_{t\le T}\pi_t^{2}},
\]
then the ensemble H_T obeys \(F(S,H_T)\le F(z)\).

</div>

The result demonstrates that, provided the weak learner meets the modest advantage of Assumption 5.5 in the sense just described, our derivative-free boosting scheme enjoys the same 1/γ² scaling as its classical counterpart despite placing **no structural assumptions whatsoever** on the loss beyond measurability.
## Finding \\(\overline{W}_{2,t}\\) [sec-alphaW2]

There is lots of freedom in the choice of \\(\alpha_t\\) in Step 2.3 of , and even more if we look at <a href="#genALPHA" data-reference-type="eqref" data-reference="genALPHA">[genALPHA]</a>. This, however, requires access to some bound \\(\overline{W}_{2,t}\\). In the general case, the quantity it upperbounds in <a href="#boundW2" data-reference-type="eqref" data-reference="boundW2">[boundW2]</a> also depends on \\(\alpha_t\\) because \\(e_{ti} \defeq \alpha_t \cdot y_i h_t(\ve{x}_i)\\). So unless we can obtain such a "simple" \\(\overline{W}_{2,t}\\) that does *not* depend on \\(\alpha_t\\), <a href="#picka" data-reference-type="eqref" data-reference="picka">[picka]</a> – and <a href="#genALPHA" data-reference-type="eqref" data-reference="genALPHA">[genALPHA]</a> – provide a *system* to solve for \\(\alpha_t\\).  
**\\(\overline{W}_{2,t}\\) via properties of \\(F\\)** Classical assumptions on loss functions for zeroth-order optimization can provide simple expressions for \\(\overline{W}_{2,t}\\) (Table <a href="#tab:features" data-reference-type="ref" data-reference="tab:features">[tab:features]</a>). Consider smoothness: we say that \\(F\\) is \\(\beta\\)-smooth if it is derivable and its derivative satisfies the Lipschitz condition \\(|F'(z') - F'(z)| \leq \beta |z' - z|, \forall z, z'\\) `\cite{bCOA}`{=latex}. Notice that this implies the condition on the \\(v\\)-derivative of the derivative: \\(|\diffloc{v}{F'}(z)| \leq \beta, \forall z, v\\). This also provides a straightforward useful expression for \\(\overline{W}_{2,t}\\).

<div class="lemma" markdown="1">

<span id="lemSmooth" label="lemSmooth"></span> Suppose that the loss \\(F\\) is \\(\beta\\)-smooth. Then we can fix \\(\overline{W}_{2,t} = 2\beta\\).

</div>

(Proof in , Section <a href="#proof_lemSmooth" data-reference-type="ref" data-reference="proof_lemSmooth">9.5</a>) What the Lemma shows is that a bound on the \\(v\\)-derivative of the derivative implies a bound on order-2 \\(\mathcal{V}\\)-derivatives (in the quantity that \\(\overline{W}_{2,t}\\) bounds <a href="#boundW2" data-reference-type="eqref" data-reference="boundW2">[boundW2]</a>). Such a condition on \\(v\\)-derivatives is thus weaker than a condition on derivatives, and it is strictly weaker if we impose a strictly positive lowerbound on the offset’s absolute value, which would be sufficient to characterize the boosting convergence of .  
**A general algorithm for \\(\overline{W}_{2,t}\\)** If we cannot make any assumption on \\(F\\), there is a simple way to *first* obtain \\(\alpha_t\\) and then \\(\overline{W}_{2,t}\\), from which all other parameters of Step 2.3 can be computed.

<figure id="findalpha">
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><strong>Input</strong> sample <span class="math inline"><em>S</em> = {(<strong>x</strong><sub><em>i</em></sub>, <em>y</em><sub><em>i</em></sub>), <em>i</em> = 1, 2, ..., <em>m</em>}</span>, <span class="math inline">$\ve{w} \in \mathbb{R}^m$</span>, <span class="math inline"><em>h</em> : 𝒳 → ℝ</span>. Step 1 : find any <span class="math inline"><em>a</em> &gt; 0</span> such that <span class="math display">$$\begin{aligned}
\frac{\left|\eta(\ve{w}, h) - \eta(\tilde{\ve{w}}(\mathrm{sign}(\eta(\ve{w}, h)) \cdot a), h)\right|}{|\eta(\ve{w}, h)|} &amp; &lt; &amp; 1 .\label{eqfinda}
\end{aligned}$$</span> <strong>Return</strong> <span class="math inline">$\mathrm{sign}(\eta(\ve{w}, h)) \cdot a$</span>.</p>
</div>
<figcaption>(<span class="math inline">$S, \ve{w}, h$</span>)</figcaption>
</figure>

We first need a few definitions. We first generalize the edge notation appearing in Step 2.2: \\[\begin{aligned}
\eta(\ve{w}, h) & \defeq & \expect_{i\sim [m]}\left[w_i y_i h(\ve{x}_i)\right],
\end{aligned}\\] so that \\(\eta_t \defeq \eta(\ve{w}_t, h_t)\\). Remind the weight update, \\(w_{ti} \defeq - \diffloc{v_{(t-1)i}}{F}(y_i H_{t-1}(\ve{x}_i))\\). We define a "partial" weight update, \\[\begin{aligned}
\tilde{w}_{ti}(\alpha) & \defeq & - \diffloc{v_{(t-1)i}}{F}(\alpha y_i h_t(\ve{x}_i) + y_i H_{t-1}(\ve{x}_i)) \label{defpartialW}
\end{aligned}\\] (if we were to replace \\(v_{(t-1)i}\\) by \\(v_{ti}\\) and let \\(\alpha \defeq \alpha_t\\), then \\(\tilde{w}_{ti}(\alpha)\\) would be \\(w_{(t+1)i}\\), hence the partial weight update). Algorithm <a href="#findalpha" data-reference-type="ref" data-reference="findalpha">3</a> presents the simple procedure to find \\(\alpha_t\\). Notice that we use \\(\tilde{\ve{w}}\\) with sole dependency on the prospective leveraging coefficient; we omit for clarity the dependences in the current ensemble (\\(H_.\\)), weak classifier (\\(h_.\\)) and offsets (\\(v_{.i}\\)) needed to compute <a href="#defpartialW" data-reference-type="eqref" data-reference="defpartialW">[defpartialW]</a>.

<div class="theorem" markdown="1">

<span id="thALPHAW2" label="thALPHAW2"></span> Suppose Assumptions <a href="#assum1finite" data-reference-type="ref" data-reference="assum1finite">[assum1finite]</a> and <a href="#assum3wla" data-reference-type="ref" data-reference="assum3wla">[assum3wla]</a> hold and \\(F\\) is continuous at all abscissae \\(\{\tilde{e}_{(t-1)i} \defeq y_i H_{t-1}(\ve{x}_i), i \in [m]\}\\). Then there are always solutions to Step 1 of  and if we let \\(\alpha_t\leftarrow\\)(\\(S, \ve{w}_t, h_t\\)) and then compute \\[\begin{aligned}
\overline{W}_{2,t} \defeq \left|\expect_{i\sim [m]}\left[\frac{h^2_{t}(\ve{x}_i)}{M^2_{t}} \cdot \diffloc{\{\alpha_t y_i h_t(\ve{x})_i),v_{(t-1)i}\}}{F}(\tilde{e}_{(t-1)i})\right]\right|,
\end{aligned}\\] then \\(\overline{W}_{2,t}\\) satisfies <a href="#boundW2" data-reference-type="eqref" data-reference="boundW2">[boundW2]</a> and \\(\alpha_t\\) satisfies <a href="#picka" data-reference-type="eqref" data-reference="picka">[picka]</a> for some \\(\varepsilon_t >0, \pi_t \in (0,1)\\).

</div>

The proof, in Section <a href="#proof_thALPHAW2" data-reference-type="ref" data-reference="proof_thALPHAW2">9.6</a>, proceeds by reducing condition <a href="#genALPHA" data-reference-type="eqref" data-reference="genALPHA">[genALPHA]</a> to <a href="#eqfinda" data-reference-type="eqref" data-reference="eqfinda">[eqfinda]</a>. The Weak Learning Assumption <a href="#assum3wla" data-reference-type="eqref" data-reference="assum3wla">[assum3wla]</a> is important for the denominator in the LHS of <a href="#eqfinda" data-reference-type="eqref" data-reference="eqfinda">[eqfinda]</a> to be non zero. The continuity assumption *at all abscissae* is important to have \\(\lim_{a \rightarrow 0} \eta(\tilde{\ve{w}}_t(a), h_t) = \eta_t\\), which ensures the existence of solutions to <a href="#eqfinda" data-reference-type="eqref" data-reference="eqfinda">[eqfinda]</a>, also easy to find, *e.g.* by a simple dichotomic search starting from an initial guess for \\(a\\). Note the necessity of being continuous only at abscissae defined by the training sample, which is finite in size. Hence, if this condition is not satisfied but discontinuities of \\(F\\) are of Lebesgue measure 0, it is easy to add an infinitesimal constant to the current weak classifier, ensuring the conditions of Theorem <a href="#thALPHAW2" data-reference-type="ref" data-reference="thALPHAW2">[thALPHAW2]</a> and keeping the boosting rates.

## Implementation of the offset oracle [sec-offset]

<figure id="f-Iti-const">
<div class="center">
<table>
<tbody>
<tr>
<td style="text-align: center;"><img src="./figures/FigIti-2.png"" style="width:25.0%" /></td>
<td style="text-align: center;"><img src="./figures/FigIti-3.png"" style="width:25.0%" /></td>
<td style="text-align: center;"><img src="./figures/FigIti-4.png"" style="width:25.0%" /></td>
<td style="text-align: center;"><img src="./figures/FigIti-5.png"" style="width:25.0%" /></td>
</tr>
<tr>
<td style="text-align: center;">(a)</td>
<td style="text-align: center;">(b)</td>
<td style="text-align: center;">(c)</td>
<td style="text-align: center;">(d)</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>
</div>
<figcaption>A simple way to build <span class="math inline">𝕀<sub><em>t</em><em>i</em></sub>(<em>z</em>)</span> for a discontinuous loss <span class="math inline"><em>F</em></span> (<span class="math inline"><em>ẽ</em><sub><em>t</em><em>i</em></sub> &lt; <em>ẽ</em><sub>(<em>t</em> − 1)<em>i</em></sub></span> and <span class="math inline"><em>z</em></span> are represented), <span> <span class="math inline">𝒪</span></span> being the set of solutions as it is built. We rotate two half-lines, one passing through <span class="math inline">(<em>ẽ</em><sub><em>t</em><em>i</em></sub>, <em>F</em>(<em>ẽ</em><sub><em>t</em><em>i</em></sub>))</span> (thick line, <span class="math inline">(<em>Δ</em>)</span>) and a parallel one translated by <span class="math inline">−<em>z</em></span> (dashed line) (a). As soon as <span class="math inline">(<em>Δ</em>)</span> crosses <span class="math inline"><em>F</em></span> on any point <span class="math inline">(<em>z</em><sup>′</sup>, <em>F</em>(<em>z</em><sup>′</sup>))</span> with <span class="math inline"><em>z</em> ≠ <em>ẽ</em><sub><em>t</em><em>i</em></sub></span> while the dashed line stays below <span class="math inline"><em>F</em></span>, we obtain a candidate offset <span class="math inline"><em>v</em></span> for , namely <span class="math inline"><em>v</em> = <em>z</em><sup>′</sup> − <em>ẽ</em><sub><em>t</em><em>i</em></sub></span>. In (b), we obtain an interval of values. We keep on rotating <span class="math inline">(<em>Δ</em>)</span>, eventually making appear several intervals for the choice of <span class="math inline"><em>v</em></span> if <span class="math inline"><em>F</em></span> is not convex (c). Finally, when we reach an angle such that the maximal difference between <span class="math inline">(<em>Δ</em>)</span> and <span class="math inline"><em>F</em></span> in <span class="math inline">[<em>ẽ</em><sub><em>t</em><em>i</em></sub>, <em>ẽ</em><sub>(<em>t</em> − 1)<em>i</em></sub>]</span> is <span class="math inline"><em>z</em></span> (<span class="math inline"><em>z</em></span> can be located at an intersection between <span class="math inline"><em>F</em></span> and the dashed line), we stop and obtain the full <span class="math inline">𝕀<sub><em>t</em><em>i</em></sub>(<em>z</em>)</span> (d).</figcaption>
</figure>

<figure id="f-BII-1">
<div class="center">
<table>
<tbody>
<tr>
<td style="text-align: center;"><img src="./figures/FigII.png"" style="width:33.0%" /></td>
<td style="text-align: center;"><img src="./figures/FigII-2.png"" style="width:33.0%" /></td>
<td style="text-align: center;"><img src="./figures/FigII-4.png"" style="width:33.0%" /></td>
</tr>
<tr>
<td style="text-align: center;">(a)</td>
<td style="text-align: center;">(b)</td>
<td style="text-align: center;">(c)</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>
</div>
<figcaption>More examples of ensembles <span class="math inline">𝕀<sub><em>t</em><em>i</em></sub>(<em>z</em>)</span> (in <span> blue</span>) for the <span class="math inline"><em>F</em></span> in Figure <a href="#f-Iti-const" data-reference-type="ref" data-reference="f-Iti-const">4</a>. (a): <span class="math inline">𝕀<sub><em>t</em><em>i</em></sub>(<em>z</em>)</span> is the union of two intervals with all candidate offsets non negative. (b): it is a single interval with non-positive offsets. (c): at a discontinuity, if <span class="math inline"><em>z</em></span> is smaller than the discontinuity, we have no direct solution for <span class="math inline">𝕀<sub><em>t</em><em>i</em></sub>(<em>z</em>)</span> for at least one positioning of the edges, but a simple trick bypasses the difficulty (see text).</figcaption>
</figure>

Figure <a href="#f-Iti-const" data-reference-type="ref" data-reference="f-Iti-const">4</a> explains how to build graphically \\(\mathbb{I}_{ti}(z)\\) for a general \\(F\\). While it is not hard to implement a general procedure following the blueprint (*i.e.* accepting the loss function as input), it would be far from achieving computational optimality: a much better choice consists in specializing it to the (set of) loss(es) at hand via hardcoding specific optimization features of the desired loss(es). This would not prevent "loss oddities" to get absolutely trivial oracles (see , Section <a href="#sec-offset-app" data-reference-type="ref" data-reference="sec-offset-app">9.7</a>).

# Discussion [sec:disc]

For an efficient implementation, boosting requires specific design choices to make sure the weak learning assumption stands for as long as necessary; experimentally, it is thus a good idea to adapt the weak learner to build more complex models as iterations increase (*e.g.* learning deeper trees), keeping Assumption <a href="#assum3wla" data-reference-type="ref" data-reference="assum3wla">[assum3wla]</a> valid with its advantage over random guessing parameter \\(\upgamma>0\\). In our more general setting, our algorithm  pinpoints two more locations that can make use of specific design choices to keep assumptions stand for a larger number of iterations.

The first is related to handling local minima. When Assumption <a href="#assum2cr" data-reference-type="ref" data-reference="assum2cr">[assum2cr]</a> breaks, it means we are close to a local optimum of the loss. One possible way of escaping those local minima is to adapt the offset oracle to output larger offsets (Step 2.5) that get weights computed outside the domain of the local minimum. Such offsets can be used to inform the weak learner of the specific examples that then need to receive larger magnitude in classification, something we have already discussed in Section <a href="#sec:boost" data-reference-type="ref" data-reference="sec:boost">5</a>. There is also more: the sign of the weight indicates the polarity of the next edge (\\(e_{t.}\\), <a href="#defEDGE12" data-reference-type="eqref" data-reference="defEDGE12">[defEDGE12]</a>) needed to decrease the loss *in the interval spanned by the last offset*. To simplify, suppose a substantial fraction of examples have an edge \\(\tilde{e}_{t.}\\) in the vicinity of the blue dotted line in Figure <a href="#f-wra" data-reference-type="ref" data-reference="f-wra">2</a> (d) so that the loss value is indicated by the big arrow and suppose their current offset \\(=v_{t-1}\\) so that their weight (positive) signals that to minimize further the loss, the weak learner’s next weak classifier has to have a positive edge over these examples. Such is the polarity constraint which essentially comes to satisfy the WLA, but there is a magnitude constraint that comes from the WRA: indeed, if the positive edge is too small so that the loss ends up in the "bump" region, then there is a risk that the WRA breaks because the loss around the bump is quite flat, so the numerator of \\(\rho_t\\) in Assumption <a href="#assum2cr" data-reference-type="ref" data-reference="assum2cr">[assum2cr]</a> can be small. Passing the bump implies escaping the local minimum at which the loss would otherwise be trapped. Section <a href="#sec-offset" data-reference-type="ref" data-reference="sec-offset">5.4</a> has presented a general blueprint for the offset oracle but more specific implementation designs can be used; some are discussed in the , Section <a href="#sec-offset-app" data-reference-type="ref" data-reference="sec-offset-app">9.7</a>.

The second is related to handling losses that take on constant values over parts of their domain. To prevent early stopping in Step 2.7 of , one needs \\(\bm{w}_{t+1} \neq \ve{0}\\). The update rule of \\(\ve{w}_t\\) imposes that the loss must then have non-zero *variation* for some examples between two successive edges <a href="#defEDGE12" data-reference-type="eqref" data-reference="defEDGE12">[defEDGE12]</a>. If the loss \\(F\\) is constant, then clearly the algorithm obviously stops without learning anything. If \\(F\\) is piecewise-constant, this constrain the design of the weak learner to make sure that some examples receive a different loss with the new model update \\(H_.\\). As explained in , Section <a href="#sec-piecewise" data-reference-type="ref" data-reference="sec-piecewise">9.11</a>, this can be efficiently addressed by specific designs on .

In the same way as there is no "1 size fits all" weak learner for all domains in traditional boosting, we expect specific design choices to be instrumental in better handling specific losses in our more general setting. Our theory points two locations further work can focus on.

# Conclusion [sec:conc]

Boosting has rapidly moved to an optimization setting involving first-order information about the loss optimized, rejoining, in terms of information needed, that of the hugely popular (stochastic) gradient descent. But this was not a formal requirement of the initial setting and in this paper, we show that essentially any loss function can be boosted without this requirement. From this standpoint, our results put boosting in a slightly more favorable light than recent development on zeroth-order optimization since, to get boosting-compliant convergence, we do not need the loss to meet any of the assumptions that those analyses usually rely on. Of course, recent advances in zeroth-order optimization have also achieved substantial design tricks for the implementation of such algorithms, something that undoubtedly needs to be adressed in our case, such as for the efficient optimization of the offset oracle. We leave this as an open problem but provide in  some toy *experiments* that a straightforward implementation achieves, hinting that  can indeed optimize very “exotic” losses.

# References [references]

<div class="thebibliography" markdown="1">

A. Akhavan, E. Chzhen, M. Pontil, and A.-B. Tsybakov A gradient estimator via l1-randomization for online zero-order optimization with two point feedback In *NeurIPS\*35*, 2022. **Abstract:** This work studies online zero-order optimization of convex and Lipschitz functions. We present a novel gradient estimator based on two function evaluations and randomization on the ‘1-sphere. Considering different geometries of feasible sets and Lipschitz assumptions we analyse online dual averaging algorithm with our estimator in place of the usual gradient. We consider two types of assumptions on the noise of the zero-order oracle: canceling noise and adversarial noise. We provide an anytime and completely data-driven algorithm, which is adaptive to all parameters of the problem. In the case of canceling noise that was previously studied in the literature, our guarantees are either comparable or better than state- of-the-art bounds obtained by Duchi et al. \[14\] and Shamir \[33\] for non-adaptive algorithms. Our analysis is based on deriving a new weighted Poincaré type inequality for the uniform measure on the ‘1-sphere with explicit constants, which may be of independent interest. 1 Introduction In this work we study the problem of convex online zero-order optimization with two-point feedback, in which adversary ﬁxes a sequence f1;f2;::::Rd!Rof convex functions and the goal of the learner is to minimize the cumulative regret with respect to the best action in a prescribed convex set Rd. This problem has received signiﬁcant attention in the context of continuous bandits and online optimization \[see e.g., 1, 3, 12, 13, 16, 18, 21, 25, 29, 33, and references therein\]. We consider the following protocol: at each round t= 1;2;:::the algorithm chooses x0 t;x00 t2Rd (that can be queried outside of ) and the adversary reveals ft(x0 t) +0 t andft(x00 t) +00 t; where0 t;00 t2Rare the noise variables (random or not) to be speciﬁed. Based on the above information and the previous rounds, the learner outputs xt2and suffers loss ft(xt). The goal of the learner is to minimize the cumulative regret TX t=1ft(xt) min x2TX t=1ft(x): At the core of our approach is a novel zero-order gradient estimator based on two function evaluations outlined in Algorithm 1. A key novelty of our estimator is that it employs a randomization step overarXiv:2205.13910v2 \[math.ST\] 20 Sep 2022Algorithm 1: Zero-Order‘1-Randomized Online Dual Averaging Input: Convex function V(), step size1\>0, and parameters ht, fort= 1;2;:::; Initialization: Generate independently vectors 1;2;:::uniformly distributed on @Bd 1, and set z1=0 fort= 1;:::; do xt= arg maxx2fthzt;xi V(x)g y0 t=ft(xt+ht (@acptAG)

A. Akhavan, M. Pontil, and A.-B. Tsybakov Exploiting higher order smoothness in derivative-free optimization and continuous bandits In *NeurIPS\*33*, 2020. **Abstract:** We study the problem of zero-order optimization of a strongly convex function. The goal is to find the minimizer of the function by a sequential exploration of its values, under measurement noise. We study the impact of higher order smoothness properties of the function on the optimization error and on the cumulative regret. To solve this problem we consider a randomized approximation of the projected gradient descent algorithm. The gradient is estimated by a randomized procedure involving two function evaluations and a smoothing kernel. We derive upper bounds for this algorithm both in the constrained and unconstrained settings and prove minimax lower bounds for any sequential search method. Our results imply that the zero-order algorithm is nearly optimal in terms of sample complexity and the problem parameters. Based on this algorithm, we also propose an estimator of the minimum value of the function achieving almost sharp oracle behavior. We compare our results with the state-of-the-art, highlighting a number of key improvements. (@aptEH)

A. Akhavan, M. Pontil, and A.-B. Tsybakov Distributed zero-order optimisation under adversarial noise In *NeurIPS\*34*, 2021. **Abstract:** We study the problem of distributed zero-order optimization for a class of strongly convex functions. They are formed by the average of local objectives, associated to different nodes in a prescribed network of connections. We propose a distributed zero-order projected gradient descent algorithm to solve this problem. Exchange of information within the network is permitted only between neighbouring nodes. A key feature of the algorithm is that it can query only function values, subject to a general noise model, that does not require zero mean or independent errors. We derive upper bounds for the average cumulative regret and optimization error of the algorithm which highlight the role played by a network connectivity parameter, the number of variables, the noise level, the strong convexity parameter of the global objective and certain smoothness properties of the local objectives. When the bound is specified to the standard undistributed setting, we obtain an improvement over the state-of-the-art bounds, due to the novel gradient estimation procedure proposed here. We also comment on lower bounds and observe that the dependency over certain function parameters in the bound is nearly optimal. (@aptDZ)

N. Alon, A. Gonen, E. Hazan, and S. Moran Boosting simple learners In *STOC’21*, 2021. **Abstract:** Boosting is a celebrated machine learning approach which is based on the idea of combining weak and moderately inaccurate hypotheses to a strong and accurate one. We study boosting under the assumption that the weak hypotheses belong to a class of bounded capacity. This assumption is inspired by the common convention that weak hypotheses are "rules-of-thumbs" from an "easy-to-learn class". (Schapire and Freund ’12, Shalev-Shwartz and Ben-David ’14.) Formally, we assume the class of weak hypotheses has a bounded VC dimension. We focus on two main questions: (i) Oracle Complexity: How many weak hypotheses are needed in order to produce an accurate hypothesis? We design a novel boosting algorithm and demonstrate that it circumvents a classical lower bound by Freund and Schapire (’95, ’12). Whereas the lower bound shows that Ω(1/γ2) weak hypotheses with γ-margin are sometimes necessary, our new method requires only Õ(1/γ) weak hypothesis, provided that they belong to a class of bounded VC dimension. Unlike previous boosting algorithms which aggregate the weak hypotheses by majority votes, the new boosting algorithm uses more complex ("deeper") aggregation rules. We complement this result by showing that complex aggregation rules are in fact necessary to circumvent the aforementioned lower bound. (ii) Expressivity: Which tasks can be learned by boosting weak hypotheses from a bounded VC class? Can complex concepts that are "far away" from the class be learned? Towards answering the first question we identify a combinatorial-geometric parameter which captures the expressivity of base-classes in boosting. As a corollary we provide an affirmative answer to the second question for many well-studied classes, including half-spaces and decision stumps. Along the way, we establish and exploit connections with Discrepancy Theory. (@aghmBS)

S.-I. Amari and H. Nagaoka Oxford University Press, 2000. **Abstract:** The Expectation–Maximization (EM) algorithm is a simple meta-algorithm that has been used for many years as a methodology for statistical inference when there are missing measurements in the observed data or when the data is composed of observables and unobservables. Its general properties are well studied, and also, there are countless ways to apply it to individual problems. In this paper, we introduce the em algorithm, an information geometric formulation of the EM algorithm, and its extensions and applications to various problems. Specifically, we will see that it is possible to formulate an outlier–robust inference algorithm, an algorithm for calculating channel capacity, parameter estimation methods on probability simplex, particular multivariate analysis methods such as principal component analysis in a space of probability models and modal regression, matrix factorization, and learning generative models, which have recently attracted attention in deep learning, from the geometric perspective provided by Amari. (@anMO)

F. Bach Course notes, MIT press (to appear), 2023. **Abstract:** Nanoporous materials such as metal-organic frameworks (MOFs) have been extensively studied for their potential for adsorption and separation applications. In this respect, grand canonical Monte Carlo (GCMC) simulations have become a well-established tool for computational screenings of the adsorption properties of large sets of MOFs. However, their reliance on empirical force field potentials has limited the accuracy with which this tool can be applied to MOFs with challenging chemical environments such as open-metal sites. On the other hand, density-functional theory (DFT) is too computationally demanding to be routinely employed in GCMC simulations due to the excessive number of required function evaluations. Therefore, we propose in this paper a protocol for training machine learning potentials (MLPs) on a limited set of DFT intermolecular interaction energies (and forces) of CO2 in ZIF-8 and the open-metal site containing Mg-MOF-74, and use the MLPs to derive adsorption isotherms from first principles. We make use of the equivariant NequIP model which has demonstrated excellent data efficiency, and as such an error on the interaction energies below 0.2 kJ mol-1 per adsorbate in ZIF-8 was attained. Its use in GCMC simulations results in highly accurate adsorption isotherms and heats of adsorption. For Mg-MOF-74, a large dependence of the obtained results on the used dispersion correction was observed, where PBE-MBD performs the best. Lastly, to test the transferability of the MLP trained on ZIF-8, it was applied to ZIF-3, ZIF-4, and ZIF-6, which resulted in large deviations in the predicted adsorption isotherms and heats of adsorption. Only when explicitly training on data for all ZIFs, accurate adsorption properties were obtained. As the proposed methodology is widely applicable to guest adsorption in nanoporous materials, it opens up the possibility for training general-purpose MLPs to perform highly accurate investigations of guest adsorption. (@bLTF)

A. Banerjee, S. Merugu, I. Dhillon, and J. Ghosh Clustering with bregman divergences In *Proc. of the \\(4^{th}\\) SIAM International Conference on Data Mining*, pages 234–245, 2004. **Abstract:** A wide variety of distortion functions are used for clustering, e.g., squared Euclidean distance, Mahalanobis distance and relative entropy. In this paper, we propose and analyze parametric hard and soft clustering algorithms based on a large class of distortion functions known as Bregman divergences. The proposed algorithms unify centroid-based parametric clustering approaches, such as classical kmeans and information-theoretic clustering, which arise by special choices of the Bregman divergence. The algorithms maintain the simplicity and scalability of the classical kmeans algorithm, while generalizing the basic idea to a very large class of clustering loss functions. There are two main contributions in this paper. First, we pose the hard clustering problem in terms of minimizing the loss in Bregman information, a quantity motivated by rate-distortion theory, and present an algorithm to minimize this loss. Secondly, we show an explicit bijection between Bregman divergences and exponential families. The bijection enables the development of an alternative interpretation of an efficient EM scheme for learning models involving mixtures of exponential distributions. This leads to a simple soft clustering algorithm for all Bregman divergences. (@bmdgCW)

P.-L. Bartlett and S. Mendelson Rademacher and gaussian complexities: Risk bounds and structural results , 3:463–482, 2002. **Abstract:** This article deals with the generalization performance of margin multi-category classifiers, when minimal learnability hypotheses are made. In that context, the derivation of a guaranteed risk is based on the handling of capacity measures belonging to three main families: Rademacher/Gaussian complexities, metric entropies and scale-sensitive combinatorial dimensions. The scale-sensitive combinatorial dimensions dedicated to the classifiers of this kind are the gamma-Psi-dimensions. We introduce the combinatorial and structural results needed to involve them in the derivation of upper bounds on the metric entropies and the Rademacher complexity. Their incidence on the guaranteed risks is characterized, which establishes that in the theoretical framework of interest, performing the transition from the multi-class case to the binary one with combinatorial dimensions is a promising alternative to proceeding with covering numbers. (@bmRA)

G. Biau, B. Cadre, and L. Rouvière Accelerated gradient boosting , 108(6):971–992, 2019. **Abstract:** Gradient tree boosting is a prediction algorithm that sequentially produces a model in theform of linear combinations of decision trees, by solving an inﬁnite-dimensional optimizationproblem. We combine gradient boosting and Nesterov’s accelerated descent to design a newalgorithm, which we call AGB (for Accelerated Gradient Boosting). Substantial numerical evidence is provided on both synthetic and real-life data sets to assess the excellent perfor-mance of the method in a large variety of prediction problems. It is empirically shown thatAGB is less sensitive to the shrinkage parameter and outputs predictors that are considerably more sparse in the number of trees, while retaining the exceptional performance of gradientboosting. (@bcrAG)

M. Blondel, A.-F. T. Martins, and V. Niculae Learning with Fenchel-Young losses , 21:35:1–35:69, 2020. **Abstract:** Over the past decades, numerous loss functions have been been proposed for a variety of supervised learning tasks, including regression, classification, ranking, and more generally structured prediction. Understanding the core principles and theoretical properties underpinning these losses is key to choose the right loss for the right problem, as well as to create new losses which combine their strengths. In this paper, we introduce Fenchel-Young losses, a generic way to construct a convex loss function for a regularized prediction function. We provide an in-depth study of their properties in a very broad setting, covering all the aforementioned supervised learning tasks, and revealing new connections between sparsity, generalized entropies, and separation margins. We show that Fenchel-Young losses unify many well-known loss functions and allow to create useful new ones easily. Finally, we derive efficient predictive and training algorithms, making Fenchel-Young losses appealing both in theory and practice. (@bmnLW)

L. M. Bregman The relaxation method of finding the common point of convex sets and its application to the solution of problems in convex programming , 7:200–217, 1967. (@bTR)

S. Bubeck Convex optimization: Algorithms and complexity , 8(3-4):231–357, 2015. **Abstract:** This monograph presents the main complexity theorems in convex optimization and their corresponding algorithms. Starting from the fundamental theory of black-box optimization, the material progresses towards recent advances in structural optimization and stochastic optimization. Our presentation of black-box optimization, strongly influenced by the seminal book of Nesterov, includes the analysis of cutting plane methods, as well as accelerated gradient descent schemes. We also pay special attention to non-Euclidean settings relevant algorithms include Frank-Wolfe, mirror descent, and dual averaging and discuss their relevance in machine learning. We provide a gentle introduction to structural optimization with FISTA to optimize a sum of a smooth and a simple non-smooth term, saddle-point mirror prox Nemirovski’s alternative to Nesterov’s smoothing, and a concise description of interior point methods. In stochastic optimization we discuss stochastic gradient descent, mini-batches, random coordinate descent, and sublinear algorithms. We also briefly touch upon convex relaxation of combinatorial problems and the use of randomness to round solutions, as well as random walks based methods. (@bCOA)

P.-S. Bullen Kluwer Academic Publishers, 2003. (@bHO)

H. Cai, Y. Lou, D. McKenzie, and W. Yin A zeroth-order block coordinate descent algorithm for huge-scale black-box optimization In *38\\(^{th}\\) ICML*, pages 1193–1203, 2021. **Abstract:** We consider the zeroth-order optimization problem in the huge-scale setting, where the dimension of the problem is so large that performing even basic vector operations on the decision variables is infeasible. In this paper, we propose a novel algorithm, coined ZO-BCD, that exhibits favorable overall query complexity and has a much smaller per-iteration computational complexity. In addition, we discuss how the memory footprint of ZO-BCD can be reduced even further by the clever use of circulant measurement matrices. As an application of our new method, we propose the idea of crafting adversarial attacks on neural network based classifiers in a wavelet domain, which can result in problem dimensions of over 1.7 million. In particular, we show that crafting adversarial examples to audio classifiers in a wavelet domain can achieve the state-of-the-art attack success rate of 97.9%. (@clmyAZ)

C. Cartis and L. Roberts Scalable subspace methods for derivative-free nonlinear least-squares optimization , 199:461–524, 2023. **Abstract:** Abstract We introduce a general framework for large-scale model-based derivative-free optimization based on iterative minimization within random subspaces. We present a probabilistic worst-case complexity analysis for our method, where in particular we prove high-probability bounds on the number of iterations before a given optimality is achieved. This framework is specialized to nonlinear least-squares problems, with a model-based framework based on the Gauss–Newton method. This method achieves scalability by constructing local linear interpolation models to approximate the Jacobian, and computes new steps at each iteration in a subspace with user-determined dimension. We then describe a practical implementation of this framework, which we call DFBGN. We outline efficient techniques for selecting the interpolation points and search subspace, yielding an implementation that has a low per-iteration linear algebra cost (linear in the problem dimension) while also achieving fast objective decrease as measured by evaluations. Extensive numerical results demonstrate that DFBGN has improved scalability, yielding strong performance on large-scale nonlinear least-squares problems. (@crSS)

S. Cheamanunkul, E. Ettinger, and Y. Freund Non-convex boosting overcomes random label noise , abs/1409.2905, 2014. **Abstract:** The sensitivity of Adaboost to random label noise is a well-studied problem. LogitBoost, BrownBoost and RobustBoost are boosting algorithms claimed to be less sensitive to noise than AdaBoost. We present the results of experiments evaluating these algorithms on both synthetic and real datasets. We compare the performance on each of datasets when the labels are corrupted by different levels of independent label noise. In presence of random label noise, we found that BrownBoost and RobustBoost perform significantly better than AdaBoost and LogitBoost, while the difference between each pair of algorithms is insignificant. We provide an explanation for the difference based on the margin distributions of the algorithms. (@cefNC)

L. Chen, J. Xu, and L. Luo Faster gradient-free algorithms for nonsmooth nonconvex stochastic optimization In *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pages 5219–5233. PMLR, 2023. **Abstract:** We consider the optimization problem of the form $\\}min\_{x \\}in \\}mathbb{R}\^d} f(x) \\}triangleq \\}mathbb{E}\_{\\}xi} \[F(x; \\}xi)\]$, where the component $F(x;\\}xi)$ is $L$-mean-squared Lipschitz but possibly nonconvex and nonsmooth. The recently proposed gradient-free method requires at most $\\}mathcal{O}( L\^4 d\^{3/2} \\}epsilon\^{-4} + \\}Delta L\^3 d\^{3/2} \\}delta\^{-1} \\}epsilon\^{-4})$ stochastic zeroth-order oracle complexity to find a $(\\}delta,\\}epsilon)$-Goldstein stationary point of objective function, where $\\}Delta = f(x_0) - \\}inf\_{x \\}in \\}mathbb{R}\^d} f(x)$ and $x_0$ is the initial point of the algorithm. This paper proposes a more efficient algorithm using stochastic recursive gradient estimators, which improves the complexity to $\\}mathcal{O}(L\^3 d\^{3/2} \\}epsilon\^{-3}+ \\}Delta L\^2 d\^{3/2} \\}delta\^{-1} \\}epsilon\^{-3})$. (@cxlFG)

X. Chen, S. Liu, K. Xu, X. Li, X. Lin, M. Hong, and D. Cox : Zeroth-order adaptive momentum method for black-box optimization In *NeurIPS\*32*, 2019. **Abstract:** The adaptive momentum method (AdaMM), which uses past gradients to update descent directions and learning rates simultaneously, has become one of the most popular first-order optimization methods for solving machine learning problems. However, AdaMM is not suited for solving black-box optimization problems, where explicit gradient forms are difficult or infeasible to obtain. In this paper, we propose a zeroth-order AdaMM (ZO-AdaMM) algorithm, that generalizes AdaMM to the gradient-free regime. We show that the convergence rate of ZO-AdaMM for both convex and nonconvex optimization is roughly a factor of $O(\\}sqrt{d})$ worse than that of the first-order AdaMM algorithm, where $d$ is problem size. In particular, we provide a deep understanding on why Mahalanobis distance matters in convergence of ZO-AdaMM and other AdaMM-type methods. As a byproduct, our analysis makes the first step toward understanding adaptive learning rate methods for nonconvex constrained optimization. Furthermore, we demonstrate two applications, designing per-image and universal adversarial attacks from black-box neural networks, respectively. We perform extensive experiments on ImageNet and empirically show that ZO-AdaMM converges much faster to a solution of high accuracy compared with $6$ state-of-the-art ZO optimization methods. (@clxllhcZZ)

X. Chen, Y. Tang, and N. Li Improve single-point zeroth-order optimization using high-pass and low-pass filters In *39\\(^{th}\\) ICML*, volume 162 of *Proceedings of Machine Learning Research*, pages 3603–3620. PMLR, 2022. **Abstract:** Single-point zeroth-order optimization (SZO) is useful in solving online black-box optimization and control problems in time-varying environments, as it queries the function value only once at each time step. However, the vanilla SZO method is known to suffer from a large estimation variance and slow convergence, which seriously limits its practical application. In this work, we borrow the idea of high-pass and low-pass filters from extremum seeking control (continuous-time version of SZO) and develop a novel SZO method called HLF-SZO by integrating these filters. It turns out that the high-pass filter coincides with the residual feedback method, and the low-pass filter can be interpreted as the momentum method. As a result, the proposed HLF-SZO achieves a much smaller variance and much faster convergence than the vanilla SZO method and empirically outperforms the residual-feedback SZO method, which is verified via extensive numerical experiments. (@ctlIS)

S. Cheng, G. Wu, and J. Zhu On the convergence of prior-guided zeroth-order optimisation algorithms In *NeurIPS\*34*, 2021. (@cwzOT)

Z. Cranko and R. Nock Boosted density estimation remastered In *36\\(^{th}\\) ICML*, pages 1416–1425, 2019. **Abstract:** There has recently been a steady increase in the number iterative approaches to density estimation. However, an accompanying burst of formal convergence guarantees has not followed; all results pay the price of heavy assumptions which are often unrealistic or hard to check. The Generative Adversarial Network (GAN) literature — seemingly orthogonal to the aforementioned pursuit — has had the side effect of a renewed interest in variational divergence minimisation (notably $f$-GAN). We show that by introducing a weak learning assumption (in the sense of the classical boosting framework) we are able to import some recent results from the GAN literature to develop an iterative boosted density estimation algorithm, including formal convergence results with rates, that does not suffer the shortcomings other approaches. We show that the density fit is an exponential family, and as part of our analysis obtain an improved variational characterisation of $f$-GAN. (@cnBD)

W. de Vazelhes, H. Zhang, H. Wu, X. Yuan, and B. Gu Zeroth-order hard-thresholding: Gradient error vs. expansivity In *NeurIPS\*35*, 2022. **Abstract:** $\\}ell_0$ constrained optimization is prevalent in machine learning, particularly for high-dimensional problems, because it is a fundamental approach to achieve sparse learning. Hard-thresholding gradient descent is a dominant technique to solve this problem. However, first-order gradients of the objective function may be either unavailable or expensive to calculate in a lot of real-world problems, where zeroth-order (ZO) gradients could be a good surrogate. Unfortunately, whether ZO gradients can work with the hard-thresholding operator is still an unsolved problem. To solve this puzzle, in this paper, we focus on the $\\}ell_0$ constrained black-box stochastic optimization problems, and propose a new stochastic zeroth-order gradient hard-thresholding (SZOHT) algorithm with a general ZO gradient estimator powered by a novel random support sampling. We provide the convergence analysis of SZOHT under standard assumptions. Importantly, we reveal a conflict between the deviation of ZO estimators and the expansivity of the hard-thresholding operator, and provide a theoretical minimal value of the number of random directions in ZO gradients. In addition, we find that the query complexity of SZOHT is independent or weakly dependent on the dimensionality under different settings. Finally, we illustrate the utility of our method on a portfolio optimization problem as well as black-box adversarial attacks. (@vzwygZO)

D. Dua and C. Graff machine learning repository 2021. **Abstract:** The University of California–Irvine (UCI) Machine Learning (ML) Repository (UCIMLR) is consistently cited as one of the most popular dataset repositories, hosting hundreds of high-impact datasets. However, a significant portion, including 28.4% of the top 250, cannot be imported via the ucimlrepo package that is provided and recommended by the UCIMLR website. Instead, they are hosted as .zip files, containing nonstandard formats that are difficult to import without additional ad hoc processing. To address this issue, here we present lucie—load University California Irvine examples—a utility that automatically determines the data format and imports many of these previously non-importable datasets, while preserving as much of a tabular data structure as possible. lucie was designed using the top 100 most popular datasets and benchmarked on the next 130, where it resulted in a success rate of 95.4% vs. 73.1% for ucimlrepo. lucie is available as a Python package on PyPI with 98% code coverage. (@dgUM)

E. Fermi and N. Metropolis Numerical solutions of a minimum problem Technical Report TR LA-1492, Los Alamos Scientific Laboratory of the University of California, 1952. **Abstract:** Abstract A new method is presented for the numerical solution of nonlinear minimum‐time control problems where at least one of the state variables is monotone. A coordinate transformation converts the problem with fixed end point and free end time to one of free end point and fixed end time. The transformed problem can be solved efficiently by the use of the gradient method with penalty functions to force the system to achieve target values of state variables. Application of the method is illustrated by the synthesis of a minimum‐time temperature path for the thermally initiated bulk polymerization of styrene. (@fmNS)

L. Flokas, E.-V. Vlatakis-Gkaragkounis, and G. Piliouras Efficiently avoiding saddle points with zero order methods: No gradients required In *NeurIPS\*32*, 2019. **Abstract:** We consider the case of derivative-free algorithms for non-convex optimization, also known as zero order algorithms, that use only function evaluations rather than gradients. For a wide variety of gradient approximators based on finite differences, we establish asymptotic convergence to second order stationary points using a carefully tailored application of the Stable Manifold Theorem. Regarding efficiency, we introduce a noisy zero-order method that converges to second order stationary points, i.e avoids saddle points. Our algorithm uses only $\\}tilde{\\}mathcal{O}}(1 / \\}epsilon\^2)$ approximate gradient calculations and, thus, it matches the converge rate guarantees of their exact gradient counterparts up to constants. In contrast to previous work, our convergence rate analysis avoids imposing additional dimension dependent slowdowns in the number of iterations required for non-convex zero order optimization. (@fvpEA)

H. Gao and H. Huang Can stochastic zeroth-order frank-wolfe method converge faster for non-convex problems? In *37\\(^{th}\\) ICML*, pages 3377–3386, 2020. (@ghCS)

A. Héliou, M. Martin, P. Mertikopoulos, and T. Rahier Zeroth-order non-convex learning via hierarchical dual averaging In *38\\(^{th}\\) ICML*, pages 4192–4202, 2021. **Abstract:** We propose a hierarchical version of dual averaging for zeroth-order online non-convex optimization - i.e., learning processes where, at each stage, the optimizer is facing an unknown non-convex loss function and only receives the incurred loss as feedback. The proposed class of policies relies on the construction of an online model that aggregates loss information as it arrives, and it consists of two principal components: (a) a regularizer adapted to the Fisher information metric (as opposed to the metric norm of the ambient space); and (b) a principled exploration of the problem’s state space based on an adapted hierarchical schedule. This construction enables sharper control of the model’s bias and variance, and allows us to derive tight bounds for both the learner’s static and dynamic regret - i.e., the regret incurred against the best dynamic policy in hindsight over the horizon of play. (@hmmrZO)

F. Huang, L. Tao, and S. Chen Accelerated stochastic gradient-free and projection-free methods In *37\\(^{th}\\) ICML*, pages 4519–4530, 2020. **Abstract:** In the paper, we propose a class of accelerated stochastic gradient-free and projection-free (a.k.a., zeroth-order Frank-Wolfe) methods to solve the constrained stochastic and finite-sum nonconvex optimization. Specifically, we propose an accelerated stochastic zeroth-order Frank-Wolfe (Acc-SZOFW) method based on the variance reduced technique of SPIDER/SpiderBoost and a novel momentum accelerated technique. Moreover, under some mild conditions, we prove that the Acc-SZOFW has the function query complexity of $O(d\\}sqrt{n}\\}epsilon\^{-2})$ for finding an $\\}epsilon$-stationary point in the finite-sum problem, which improves the exiting best result by a factor of $O(\\}sqrt{n}\\}epsilon\^{-2})$, and has the function query complexity of $O(d\\}epsilon\^{-3})$ in the stochastic problem, which improves the exiting best result by a factor of $O(\\}epsilon\^{-1})$. To relax the large batches required in the Acc-SZOFW, we further propose a novel accelerated stochastic zeroth-order Frank-Wolfe (Acc-SZOFW\*) based on a new variance reduced technique of STORM, which still reaches the function query complexity of $O(d\\}epsilon\^{-3})$ in the stochastic problem without relying on any large batches. In particular, we present an accelerated framework of the Frank-Wolfe methods based on the proposed momentum accelerated technique. The extensive experimental results on black-box adversarial attack and robust black-box classification demonstrate the efficiency of our algorithms. (@htcAS)

B. Irwin, E. Haber, R. Gal, and A. Ziv Neural network accelerated implicit filtering: Integrating neural network surrogates with provably convergent derivative free optimization methods In *40\\(^{th}\\) ICML*, volume 202 of *Proceedings of Machine Learning Research*, pages 14376–14389. PMLR, 2023. **Abstract:** In computer aided design (CAD), a core task is to optimize the parameters of noisy simulations. Derivative free optimization (DFO) methods are the most common choice for this task. In this paper, we show how four DFO methods, specifically implicit filtering (IF), simulated annealing (SA), genetic algorithms (GA), and particle swarm (PS), can be accelerated using a deep neural network (DNN) that acts as a surrogate model of the objective function. In particular, we demonstrate the applicability of the DNN accelerated DFO approach to the coverage directed generation (CDG) problem that is commonly solved by hardware verification teams. (@ihgzNN)

V. Kac and P. Cheung Springer, 2002. **Abstract:** The Hermite-Hadamard inequalities are common research topics explored in different dimensions. For any interval $ \[\\}mathrm{b\_{0}}, \\}mathrm{b\_{1}}\]\\}subset\\}Re $, we construct the idea of the Hermite-Hadamard inequality, its different kinds, and its generalization in symmetric quantum calculus at $ \\}mathrm{b\_{0}}\\}in\[\\}mathrm{b\_{0}}, \\}mathrm{b\_{1}}\]\\}subset\\}Re $. We also construct parallel results for the Hermite-Hadamard inequality, its different types, and its generalization on other end point $ \\}mathrm{b\_{1}} $, and provide some examples as well. Some justification with graphical analysis is provided as well. Finally, with the assistance of these outcomes, we give a midpoint type inequality and some of its approximations for convex functions in symmetric quantum calculus. (@kcQC)

M. J. Kearns and U. V. Vazirani M.I.T. Press, 1994. (@kvAI)

M.J. Kearns Thoughts on hypothesis boosting 1988. ML class project. **Abstract:** Counterfactual thoughts of what might have been have been shown to influence emotional responses to outcomes. The present investigation extends this research by proposing a model of how categorical cutoff points, or arbitrary values that impose qualitative boundaries on quantitative outcomes, induce counterfactual thoughts and influence individuals’ satisfaction. In particular, just making a cutoff for a category is hypothesized to elicit downward counterfactual comparisons, boosting satisfaction, whereas just missing a cutoff prompts upward counterfactual thoughts, decreasing satisfaction. In some circumstances, this asymmetry can reverse the usual relationship between objective outcome and satisfaction, causing people who do objectively better to feel worse than those they outperform. This hypothesis is supported by the results of 1 naturalistic study and 2 scenario experiments. (@kTO)

M.J. Kearns and Y. Mansour On the boosting ability of top-down decision tree learning algorithms , 58:109–128, 1999. **Abstract:** Article Free Access Share on On the boosting ability of top-down decision tree learning algorithms Authors: Michael Kearns AT&T Research AT&T ResearchView Profile , Yishay Mansour Tel-Aviv University Tel-Aviv UniversityView Profile Authors Info & Claims STOC ’96: Proceedings of the twenty-eighth annual ACM symposium on Theory of ComputingJuly 1996Pages 459–468https://doi.org/10.1145/237814.237994Published:01 July 1996Publication History 63citation947DownloadsMetricsTotal Citations63Total Downloads947Last 12 Months150Last 6 weeks20 Get Citation AlertsNew Citation Alert added!This alert has been successfully added and will be sent to:You will be notified whenever a record that you have chosen has been cited.To manage your alert preferences, click on the button below.Manage my AlertsNew Citation Alert!Please log in to your account Publisher SiteeReaderPDF (@kmOTj)

J. Larson, M. Menickelly, and S.-M. Wild Derivative-free optimization methods , pages 287–404, 2019. **Abstract:** In many optimization problems arising from scientific, engineering and artificial intelligence applications, objective and constraint functions are available only as the output of a black-box or simulation oracle that does not provide derivative information. Such settings necessitate the use of methods for derivative-free, or zeroth-order, optimization. We provide a review and perspectives on developments in these methods, with an emphasis on highlighting recent developments and on unifying treatment of such problems in the non-linear optimization and machine learning literature. We categorize methods based on assumed properties of the black-box functions, as well as features of the methods. We first overview the primary setting of deterministic methods applied to unconstrained, non-convex optimization problems where the objective function is defined by a deterministic black-box oracle. We then discuss developments in randomized methods, methods that assume some additional structure about the objective (including convexity, separability and general non-smooth compositions), methods for problems where the output of the black-box oracle is stochastic, and methods for handling different types of constraints. (@lmwDF)

Z. Li, P.-Y. Chen, S. Liu, S. Lu, and Y. Xu Zeroth-order optimization for composite problems with functional constraints In *AAAI’22*, pages 7453–7461. AAAI Press, 2022. **Abstract:** In many real-world problems, first-order (FO) derivative evaluations are too expensive or even inaccessible. For solving these problems, zeroth-order (ZO) methods that only need function evaluations are often more efficient than FO methods or sometimes the only options. In this paper, we propose a novel zeroth-order inexact augmented Lagrangian method (ZO-iALM) to solve black-box optimization problems, which involve a composite (i.e., smooth+nonsmooth) objective and functional constraints. This appears to be the first work that develops an iALM-based ZO method for functional constrained optimization and meanwhile achieves query complexity results matching the best-known FO complexity results up to a factor of variable dimension. With an extensive experimental study, we show the effectiveness of our method. The applications of our method span from classical optimization problems to practical machine learning examples such as resource allocation in sensor networks and adversarial example generation. (@lcllxZO)

T. Lin, Z. Zheng, and M.-I. Jordan Gradient-free methods for deterministic and stochastic nonsmooth nonconvex optimization In *NeurIPS\*35*, 2022. **Abstract:** Nonsmooth nonconvex optimization problems broadly emerge in machine learning and business decision making, whereas two core challenges impede the development of efficient solution methods with finite-time convergence guarantee: the lack of computationally tractable optimality criterion and the lack of computationally powerful oracles. The contributions of this paper are two-fold. First, we establish the relationship between the celebrated Goldstein subdifferential\~\\}citep{Goldstein-1977-Optimization} and uniform smoothing, thereby providing the basis and intuition for the design of gradient-free methods that guarantee the finite-time convergence to a set of Goldstein stationary points. Second, we propose the gradient-free method (GFM) and stochastic GFM for solving a class of nonsmooth nonconvex optimization problems and prove that both of them can return a $(\\}delta,\\}epsilon)$-Goldstein stationary point of a Lipschitz function $f$ at an expected convergence rate at $O(d\^{3/2}\\}delta\^{-1}\\}epsilon\^{-4})$ where $d$ is the problem dimension. Two-phase versions of GFM and SGFM are also proposed and proven to achieve improved large-deviation results. Finally, we demonstrate the effectiveness of 2-SGFM on training ReLU neural networks with the \\}textsc{Minst} dataset. (@lzjGF)

P.-M. Long and R.-A. Servedio Random classification noise defeats all convex potential boosters , 78(3):287–304, 2010. **Abstract:** A broad class of boosting algorithms can be interpreted as performing coordinate- wise gradient descent to minimize some potential function of the margins of a data set. This class includes AdaBoost, LogitBoost, and other widely used and well-studied boosters. In this paper we show that for a broad class of convex potential functions, any such boosting algorithm is highly susceptible to random classiﬁcation noise. We do this by showing that for any such booster and any nonzero random classiﬁcation noise rate η,t h e r ei sas i m p l e data set of examples which is efﬁciently learnable by such a booster if there is no noise, but which cannot be learned to accuracy better than 1 /2 if there is random classiﬁcation noise at rate η. This holds even if the booster regularizes using early stopping or a bound on the L1norm of the voting weights. This negative result is in contrast with known branching program based boosters which do not fall into the convex potential function framework and which can provably learn to high accuracy in the presence of random classiﬁcation noise. (@lsRC)

C. Maheshwari, C.-Y. Chiu, E. Mazumdar, S. Shankar Sastry, and L.-J. Ratliff Zeroth-order methods for convex-concave minmax problems: applications to decision-dependent risk minimization In *25\\(^{th}\\) AISTATS*, 2022. **Abstract:** Min-max optimization is emerging as a key framework for analyzing problems of robustness to strategically and adversarially generated data. We propose a random reshuffling-based gradient free Optimistic Gradient Descent-Ascent algorithm for solving convex-concave min-max problems with finite sum structure. We prove that the algorithm enjoys the same convergence rate as that of zeroth-order algorithms for convex minimization problems. We further specialize the algorithm to solve distributionally robust, decision-dependent learning problems, where gradient information is not readily available. Through illustrative simulations, we observe that our proposed approach learns models that are simultaneously robust against adversarial distribution shifts and strategic decisions from the data sources, and outperforms existing methods from the strategic classification literature. (@mcmsrZO)

Y. Mansour, R. Nock, and R.-C. Williamson Random classification noise does not defeat all convex potential boosters irrespective of model choice In *40\\(^{th}\\) ICML*, 2023. (@mnwRC)

E. Mhanna and M. Assaad Single point-based distributed zeroth-order optimization with a non-convex stochastic objective function In *40\\(^{th}\\) ICML*, volume 202 of *Proceedings of Machine Learning Research*, pages 24701–24719. PMLR, 2023. **Abstract:** Zero-order (ZO) optimization is a powerful tool for dealing with realistic constraints. On the other hand, the gradient-tracking (GT) technique proved to be an efficient method for distributed optimization aiming to achieve consensus. However, it is a first-order (FO) method that requires knowledge of the gradient, which is not always possible in practice. In this work, we introduce a zero-order distributed optimization method based on a one-point estimate of the gradient tracking technique. We prove that this new technique converges with a single noisy function query at a time in the non-convex setting. We then establish a convergence rate of $O(\\}frac{1}{\\}sqrt\[3\]{K}})$ after a number of iterations K, which competes with that of $O(\\}frac{1}{\\}sqrt\[4\]{K}})$ of its centralized counterparts. Finally, a numerical example validates our theoretical results. (@maSPB)

M. Mohri, A. Rostamizadeh, and A. Talwalkar MIT Press, 2018. **Abstract:** This paper surveys visual methods of explainability of Machine Learning (ML) with focus on moving from quasi-explanations that dominate in ML to domain-specific explanation supported by granular visuals. ML interpretation is fundamentally a human activity and visual methods are more readily interpretable. While efficient visual representations of high-dimensional data exist, the loss of interpretable information, occlusion, and clutter continue to be a challenge, which lead to quasi-explanations. We start with the motivation and the different definitions of explainability. The paper focuses on a clear distinction between quasi-explanations and domain specific explanations, and between explainable and an actually explained ML model that are critically important for the explainability domain. We discuss foundations of interpretability, overview visual interpretability and present several types of methods to visualize the ML models. Next, we present methods of visual discovery of ML models, with the focus on interpretable models, based on the recently introduced concept of General Line Coordinates (GLC). These methods take the critical step of creating visual explanations that are not merely quasi-explanations but are also domain specific visual explanations while these methods themselves are domain-agnostic. The paper includes results on theoretical limits to preserve n-D distances in lower dimensions, based on the Johnson-Lindenstrauss lemma, point-to-point and point-to-graph GLC approaches, and real-world case studies. The paper also covers traditional visual methods for understanding ML models, which include deep learning and time series models. We show that many of these methods are quasi-explanations and need further enhancement to become domain specific explanations. We conclude with outlining open problems and current research frontiers. (@mrtFO)

Y. Nesterov and V. Spokoiny Random gradient-free optimization of convex functions , 17:527–566, 2017. **Abstract:** In this paper, we focus on a constrained convex optimization problem of multiagent systems under a time-varying topology. In such topology, it is not only B-strongly connected, but the communication noises are also existent. Each agent has access to its local cost function, which is a nonsmooth function. A gradient-free random protocol is come up with minimizing a sum of cost functions of all agents, which are projected to local constraint sets. First, considering the stochastic disturbances in the communication channels among agents, the upper bounds of disagreement estimate of agents’ states are obtained. Second, a sufficient condition on choosing step sizes and smoothing parameters is derived to guarantee that all agents almost surely converge to the stationary optimal point. At last, a numerical example and a comparison are provided to illustrate the feasibility of the random gradient-free algorithm. (@nsRG)

F. Nielsen and R. Nock The Bregman chord divergence In *Geometric Science of Information - 4th International Conference, 2019*, pages 299–308, 2019. **Abstract:** Distances are fundamental primitives whose choice signi cantly impacts the performances of algo- rithms in machine learning and signal processing. However selecting the most appropriate distance for a given task is an endeavor. Instead of testing one by one the entries of an ever-expanding dictionary ofad hoc distances, one rather prefers to consider parametric classes of distances that are exhaustively characterized by axioms derived from rst principles. Bregman divergences are such a class. However ne-tuning a Bregman divergence is delicate since it requires to smoothly adjust a functional generator. In this work, we propose an extension of Bregman divergences called the Bregman chord divergences. This new class of distances does not require gradient calculations, uses two scalar parameters that can be easily tailored in applications, and generalizes asymptotically Bregman divergences. (@nnTB)

R. Nock and A. K. Menon Supervised learning: No loss no cry In *37\\(^{th}\\) ICML*, 2020. **Abstract:** Supervised learning requires the specification of a loss function to minimise. While the theory of admissible losses from both a computational and statistical perspective is well-developed, these offer a panoply of different choices. In practice, this choice is typically made in an \\}emph{ad hoc} manner. In hopes of making this procedure more principled, the problem of \\}emph{learning the loss function} for a downstream task (e.g., classification) has garnered recent interest. However, works in this area have been generally empirical in nature. In this paper, we revisit the {\\}sc SLIsotron} algorithm of Kakade et al. (2011) through a novel lens, derive a generalisation based on Bregman divergences, and show how it provides a principled procedure for learning the loss. In detail, we cast {\\}sc SLIsotron} as learning a loss from a family of composite square losses. By interpreting this through the lens of \\}emph{proper losses}, we derive a generalisation of {\\}sc SLIsotron} based on Bregman divergences. The resulting {\\}sc BregmanTron} algorithm jointly learns the loss along with the classifier. It comes equipped with a simple guarantee of convergence for the loss it learns, and its set of possible outputs comes with a guarantee of agnostic approximability of Bayes rule. Experiments indicate that the {\\}sc BregmanTron} substantially outperforms the {\\}sc SLIsotron}, and that the loss it learns can be minimized by other algorithms for different tasks, thereby opening the interesting problem of \\}textit{loss transfer} between domains. (@nmSL)

R. Nock and R.-C. Williamson Lossless or quantized boosting with integer arithmetic In *36\\(^{th}\\) ICML*, pages 4829–4838, 2019. **Abstract:** In supervised learning, efﬁciency often starts with the choice of a good loss: support vector machines popularised Hinge loss, Adaboost popularised the exponential loss, etc. Recent trends in machine learning have highlighted the necessity for train- ing routines to meet tight requirements on commu- nication, bandwidth, energy, operations, encoding, among others. Fitting the often decades-old state of the art training routines into these new con- straints does not go without pain and uncertainty or reduction in the original guarantees. Our paper starts with the design of a new strictly proper canonical, twice differentiable loss called the Q-loss. Importantly, its mirror up- date over (arbitrary) rational inputs uses only in- teger arithmetics – more precisely, the sole use of+; ;=;;j:j. We build a learning algorithm which is able, under mild assumptions, to achieve a lossless boosting-compliant training. We give conditions for a quantization of its main mem- ory footprint, weights, to be done while keeping the whole algorithm boosting-compliant. Exper- iments display that the algorithm can achieve a fast convergence during the early boosting rounds compared to AdaBoost, even with a weight stor- age that can be 30+ times smaller. Lastly, we show that the Bayes risk of the Q-loss can be used as node splitting criterion for decision trees and guarantees optimal boosting convergence. (@nwLO)

N.-E. Pfetsch and Sebastian Pokutta - non-convex boosting via integer programming In *37\\(^{th}\\) ICML*, volume 119, pages 7663–7672, 2020. **Abstract:** Recently non-convex optimization approaches for solving machine learning problems have gained significant attention. In this paper we explore non-convex boosting in classification by means of integer programming and demonstrate real-world practicability of the approach while circumventing shortcomings of convex boosting approaches. We report results that are comparable to or better than the current state-of-the-art. (@ppIN)

Y. Qiu, U.-V. Shanbhag, and F. Yousefian Zeroth-order methods for nondifferentiable, nonconvex and hierarchical federated optimization In *NeurIPS\*36*, 2023. **Abstract:** Motivated by the emergence of federated learning (FL), we design and analyze federated methods for addressing: (i) Nondifferentiable nonconvex optimization; (ii) Bilevel optimization; (iii) Minimax problems; and (iv) Two-stage stochastic mathematical programs with equilibrium constraints (2s-SMPEC). Research on these problems has been limited and afflicted by reliance on strong assumptions, including the need for differentiability of the implicit function and the absence of constraints in the lower-level problem, among others. We make the following contributions. In (i), by leveraging convolution-based smoothing and Clarke’s subdifferential calculus, we devise a randomized smoothing-enabled zeroth-order FL method and derive communication and iteration complexity guarantees for computing an approximate Clarke stationary point. To contend with (ii) and (iii), we devise a unifying randomized implicit zeroth-order FL framework, equipped with explicit communication and iteration complexities. Importantly, our method utilizes delays during local steps to skip calls to the inexact lower-level FL oracle. This results in significant reduction in communication overhead. In (iv), we devise an inexact implicit variant of the method in (i). Remarkably, this method achieves a total communication complexity matching that of single-level nonsmooth nonconvex optimization in FL. We empirically validate the theoretical findings on instances of federated nonsmooth and hierarchical problems. (@qsyZO)

M. Rando, C. Molinari, L. Rosasco, and S. Villa Structured zeroth-order for non-smooth optimization In *NeurIPS\*36*, 2023. **Abstract:** Finite-difference methods are a class of algorithms designed to solve black-box optimization problems by approximating a gradient of the target function on a set of directions. In black-box optimization, the non-smooth setting is particularly relevant since, in practice, differentiability and smoothness assumptions cannot be verified. To cope with nonsmoothness, several authors use a smooth approximation of the target function and show that finite difference methods approximate its gradient. Recently, it has been proved that imposing a structure in the directions allows improving performance. However, only the smooth setting was considered. To close this gap, we introduce and analyze O-ZD, the first structured finite-difference algorithm for non-smooth black-box optimization. Our method exploits a smooth approximation of the target function and we prove that it approximates its gradient on a subset of random {\\}em orthogonal} directions. We analyze the convergence of O-ZD under different assumptions. For non-smooth convex functions, we obtain the optimal complexity. In the non-smooth non-convex setting, we characterize the number of iterations needed to bound the expected norm of the smoothed gradient. For smooth functions, our analysis recovers existing results for structured zeroth-order methods for the convex case and extends them to the non-convex setting. We conclude with numerical simulations where assumptions are satisfied, observing that our algorithm has very good practical performances. (@rmrvSZ)

M.-D. Reid and R.-C. Williamson Information, divergence and risk for binary experiments , 12:731–817, 2011. **Abstract:** We unify f-divergences, Bregman divergences, surrogate regret bounds, proper scoring rules, cost curves, ROC-curves and statistical information. We do this by systematically studying integral and variational representations of these objects and in so doing identify their representation primitives which all are related to cost-sensitive binary classification. As well as developing relationships between generative and discriminative views of learning, the new machinery leads to tight and more general surrogate regret bounds and generalised Pinsker inequalities relating f-divergences to variational divergence. The new viewpoint also illuminates existing algorithms: it provides a new derivation of Support Vector Machines in terms of divergences and relates maximum mean discrepancy to Fisher linear discriminants. (@rwID)

Z. Ren, Y. Tang, and N. Li Escaping saddle points in zeroth-order optimization: the power of two-point estimators In *40\\(^{th}\\) ICML*, volume 202 of *Proceedings of Machine Learning Research*, pages 28914–28975. PMLR, 2023. **Abstract:** Two-point zeroth order methods are important in many applications of zeroth-order optimization, such as robotics, wind farms, power systems, online optimization, and adversarial robustness to black-box attacks in deep neural networks, where the problem may be high-dimensional and/or time-varying. Most problems in these applications are nonconvex and contain saddle points. While existing works have shown that zeroth-order methods utilizing $\\}Omega(d)$ function valuations per iteration (with $d$ denoting the problem dimension) can escape saddle points efficiently, it remains an open question if zeroth-order methods based on two-point estimators can escape saddle points. In this paper, we show that by adding an appropriate isotropic perturbation at each iteration, a zeroth-order algorithm based on $2m$ (for any $1 \\}leq m \\}leq d$) function evaluations per iteration can not only find $\\}epsilon$-second order stationary points polynomially fast, but do so using only $\\}tilde{O}\\}left(\\}frac{d}{m\\}epsilon\^{2}\\}bar{\\}psi}}\\}right)$ function evaluations, where $\\}bar{\\}psi} \\}geq \\}tilde{\\}Omega}\\}left(\\}sqrt{\\}epsilon}\\}right)$ is a parameter capturing the extent to which the function of interest exhibits the strict saddle property. (@rtlES)

A.-K. Sahu, M. Zaheer, and S. Kar Towards gradient free and projection free stochastic optimization In *22\\(^{nd}\\) AISTATS*, pages 3468–3477, 2019. **Abstract:** This paper focuses on the problem of \\}emph{constrained} \\}emph{stochastic} optimization. A zeroth order Frank-Wolfe algorithm is proposed, which in addition to the projection-free nature of the vanilla Frank-Wolfe algorithm makes it gradient free. Under convexity and smoothness assumption, we show that the proposed algorithm converges to the optimal objective function at a rate $O\\}left(1/T\^{1/3}\\}right)$, where $T$ denotes the iteration count. In particular, the primal sub-optimality gap is shown to have a dimension dependence of $O\\}left(d\^{1/3}\\}right)$, which is the best known dimension dependence among all zeroth order optimization algorithms with one directional derivative per iteration. For non-convex functions, we obtain the \\}emph{Frank-Wolfe} gap to be $O\\}left(d\^{1/3}T\^{-1/4}\\}right)$. Experiments on black-box optimization setups demonstrate the efficacy of the proposed algorithm. (@szkTG)

W. Shi, H. Gao, and B. Gu Gradient-free method for heavily constrained nonconvex optimization In *39\\(^{th}\\) ICML*, volume 162 of *Proceedings of Machine Learning Research*, pages 19935–19955. PMLR, 2022. **Abstract:** Zeroth-order (ZO) method has been shown to be a powerful method for solving the optimization problem where explicit expression of the gradients is difficult or infeasible to obtain. Recently, due to the practical value of the constrained problems, a lot of ZO Frank-Wolfe or projected ZO methods have been proposed. However, in many applications, we may have a very large number of nonconvex white/black-box constraints, which makes the existing zeroth-order methods extremely inefficient (or even not working) since they need to inquire function value of all the constraints and project the solution to the complicated feasible set. In this paper, to solve the nonconvex problem with a large number of white/black-box constraints, we proposed a doubly stochastic zeroth-order gradient method (DSZOG) with momentum method and adaptive step size. Theoretically, we prove DSZOG can converge to the $\\}epsilon$-stationary point of the constrained problem. Experimental results in two applications demonstrate the superiority of our method in terms of training time and accuracy compared with other ZO methods for the constrained problem. (@sggGF)

M.-K. Warmuth and S. V. N. Vishwanathan Tutorial: Survey of boosting from an optimization perspective In *26\\(^{th}\\) ICML*, 2009. **Abstract:** Deep neural networks have achieved remarkable performance for artificial intelligence tasks. The success behind intelligent systems often relies on large-scale models with high computational complexity and storage costs. The over-parameterized networks are often easy to optimize and can achieve better performance. However, it is challenging to deploy them over resource-limited edge-devices. Knowledge Distillation (KD) aims to optimize a lightweight network from the perspective of over-parameterized training. The traditional offline KD transfers knowledge from a cumbersome teacher to a small and fast student network. When a sizeable pre-trained teacher network is unavailable, online KD can improve a group of models by collaborative or mutual learning. Without needing extra models, Self-KD boosts the network itself using attached auxiliary architectures. KD mainly involves knowledge extraction and distillation strategies these two aspects. Beyond KD schemes, various KD algorithms are widely used in practical applications, such as multi-teacher KD, cross-modal KD, attention-based KD, data-free KD and adversarial KD. This paper provides a comprehensive KD survey, including knowledge categories, distillation schemes and algorithms, as well as some empirical studies on performance comparison. Finally, we discuss the open challenges of existing KD works and prospect the future directions. (@wvSO)

T. Werner and P. Ruckdeschel The column measure and gradient-free gradient boosting 2019. **Abstract:** Sparse model selection by structural risk minimization leads to a set of a few predictors, ideally a subset of the true predictors. This selection clearly depends on the underlying loss function $\\}tilde L$. For linear regression with square loss, the particular (functional) Gradient Boosting variant $L_2-$Boosting excels for its computational efficiency even for very large predictor sets, while still providing suitable estimation consistency. For more general loss functions, functional gradients are not always easily accessible or, like in the case of continuous ranking, need not even exist. To close this gap, starting from column selection frequencies obtained from $L_2-$Boosting, we introduce a loss-dependent ”column measure” $ν\^{(\\}tilde L)}$ which mathematically describes variable selection. The fact that certain variables relevant for a particular loss $\\}tilde L$ never get selected by $L_2-$Boosting is reflected by a respective singular part of $ν\^{(\\}tilde L)}$ w.r.t. $ν\^{(L_2)}$. With this concept at hand, it amounts to a suitable change of measure (accounting for singular parts) to make $L_2-$Boosting select variables according to a different loss $\\}tilde L$. As a consequence, this opens the bridge to applications of simulational techniques such as various resampling techniques, or rejection sampling, to achieve this change of measure in an algorithmic way. (@wrTC)

H. Zhang and B. Gu Faster gradient-free methods for escaping saddle points In *ICLR’23*. OpenReview.net, 2023. **Abstract:** Stochastically controlled stochastic gradient (SCSG) methods have been proved to converge efficiently to first-order stationary points which, however, can be saddle points in nonconvex optimization. It has been observed that a stochastic gradient descent (SGD) step introduces anistropic noise around saddle points for deep learning and non-convex half space learning problems, which indicates that SGD satisfies the correlated negative curvature (CNC) condition for these problems. Therefore, we propose to use a separate SGD step to help the SCSG method escape from strict saddle points, resulting in the CNC-SCSG method. The SGD step plays a role similar to noise injection but is more stable. We prove that the resultant algorithm converges to a second-order stationary point with a convergence rate of $\\}tilde{O}( \\}epsilon\^{-2} log( 1/\\}epsilon))$ where $\\}epsilon$ is the pre-specified error tolerance. This convergence rate is independent of the problem dimension, and is faster than that of CNC-SGD. A more general framework is further designed to incorporate the proposed CNC-SCSG into any first-order method for the method to escape saddle points. Simulation studies illustrate that the proposed algorithm can escape saddle points in much fewer epochs than the gradient descent methods perturbed by either noise injection or a SGD step. (@zgFG)

H. Zhang, H. Xiong, and B. Gu Zeroth-order negative curvature finding: Escaping saddle points without gradients In *NeurIPS\*35*, 2022. **Abstract:** We consider escaping saddle points of nonconvex problems where only the function evaluations can be accessed. Although a variety of works have been proposed, the majority of them require either second or first-order information, and only a few of them have exploited zeroth-order methods, particularly the technique of negative curvature finding with zeroth-order methods which has been proven to be the most efficient method for escaping saddle points. To fill this gap, in this paper, we propose two zeroth-order negative curvature finding frameworks that can replace Hessian-vector product computations without increasing the iteration complexity. We apply the proposed frameworks to ZO-GD, ZO-SGD, ZO-SCSG, ZO-SPIDER and prove that these ZO algorithms can converge to $(\\}epsilon,\\}delta)$-approximate second-order stationary points with less query complexity compared with prior zeroth-order works for finding local minima. (@zxgZO)

</div>

<div class="center" markdown="1">

Appendix

</div>

To differentiate with the numberings in the main file, the numbering of Theorems, etc. is letter-based (A, B, ...).

# Table of contents [table-of-contents]

**A quick summary of recent zeroth-order optimization approachess** Pg  
**Supplementary material on proofs** Pg  
\\(\hookrightarrow\\) Helper results Pg  
\\(\hookrightarrow\\) Removing the \\(\neq 0\\) part in Assumption <a href="#assum1finite" data-reference-type="ref" data-reference="assum1finite">[assum1finite]</a> Pg  
\\(\hookrightarrow\\) Proof of Lemma <a href="#lemW2bound" data-reference-type="ref" data-reference="lemW2bound">[lemW2bound]</a> Pg  
\\(\hookrightarrow\\) Proof of Theorem <a href="#thBOOSTCH" data-reference-type="ref" data-reference="thBOOSTCH">[thBOOSTCH]</a> Pg  
\\(\hookrightarrow\\) Proof of Lemma <a href="#lemSmooth" data-reference-type="ref" data-reference="lemSmooth">[lemSmooth]</a> Pg  
\\(\hookrightarrow\\) Proof of Theorem <a href="#thALPHAW2" data-reference-type="ref" data-reference="thALPHAW2">[thALPHAW2]</a> Pg  
\\(\hookrightarrow\\) Implementation of the offset oraclePg  
\\(\hookrightarrow\\) Proof of Lemma <a href="#lemOBIconv" data-reference-type="ref" data-reference="lemOBIconv">[lemOBIconv]</a> Pg  
\\(\hookrightarrow\\) Handling discontinuities in the offset oracle to prevent stopping in Step 2.5 of Pg  
\\(\hookrightarrow\\) A boosting pattern that can "survive" above differentiability Pg  
\\(\hookrightarrow\\) The case of piecewise constant losses for Pg  
**Supplementary material on algorithms, implementation tricks and a toy experiment** Pg  

# A quick summary of recent zeroth-order optimization approaches [sec-sup-sum]

<div class="tabular" markdown="1">

r?ccccc?c?l & & \\(\nabla F\\) & main  
reference & conv. & diff. & Lip. & smooth & Lb & diff. & ML topic  
`\cite{aptEH}`{=latex} & & & & & & & online ML  
`\cite{aptDZ}`{=latex} & & & & & & & distributed ML  
`\cite{acptAG}`{=latex} & & & & & & & online ML  
`\cite{clmyAZ}`{=latex} & & & & & & & alt. GD  
`\cite{crSS}`{=latex} & & & & & & & alt. GD  
`\cite{clxllhcZZ}`{=latex} & & & & & & & alt. GD  
`\cite{cxlFG}`{=latex} & & & & & & & alt. GD  
`\cite{ctlIS}`{=latex} & & & & & & & alt. GD  
`\cite{cwzOT}`{=latex} & & & & & & & alt. GD  
`\cite{fvpEA}`{=latex} & & & & & & & saddle pt opt  
`\cite{ghCS}`{=latex} & & & & & & & alt. FW  
`\cite{htcAS}`{=latex} & & & & & & & alt. FW  
`\cite{vzwygZO}`{=latex} & & & & & & & alt. GD  
`\cite{hmmrZO}`{=latex} & & & & & & & online ML  
`\cite{ihgzNN}`{=latex} & & & & & & & deep ML  
`\cite{lcllxZO}`{=latex} & & & & & & & alt. GD  
`\cite{lzjGF}`{=latex} & & & & & & & saddle pt opt  
`\cite{mcmsrZO}`{=latex} & & & & & & & saddle pt opt  
`\cite{maSPB}`{=latex} & & & & & & & distributed ML  
`\cite{rmrvSZ}`{=latex} & & & & & & & alt. GD  
`\cite{qsyZO}`{=latex} & & & & & & & federated ML  
`\cite{rtlES}`{=latex} & & & & & & & saddle pt opt  
`\cite{szkTG}`{=latex} & & & & & & & alt. FW  
`\cite{sggGF}`{=latex} & & && & & & alt. GD  
`\cite{zgFG}`{=latex} & & & & & & & saddle pt opt  
`\cite{zxgZO}`{=latex} & & & & & & & saddle pt opt  

</div>

Table <a href="#tab:features" data-reference-type="ref" data-reference="tab:features">[tab:features]</a> summarizes a few dozens of recent approaches that can be related to zeroth-order optimization in various topics of ML. Note that no such approaches focus on boosting.

# Supplementary material on proofs [sec-sup-pro]

## Helper results [cheatsheet]

We now show that the order of the elements of \\(\mathcal{V}\\) does not matter to compute the \\(\mathcal{V}\\)-derivative as in Definition <a href="#defsecZ" data-reference-type="ref" data-reference="defsecZ">[defsecZ]</a>. For any \\(\ve{\sigma} \in
\{0,1\}^{n}\\), we let \\(1_{\ve{\sigma}} \defeq \sum_i \sigma_i\\).

<div class="lemma" markdown="1">

<span id="lemCombSec" label="lemCombSec"></span> For any \\(z \in \mathbb{R}\\), any \\(n\in \mathbb{N}_*\\) and any \\(\mathcal{V}\defeq \{v_1, v_2, ..., v_n\} \subset \mathbb{R}\\), \\[\begin{aligned}
\diffloc{\mathcal{V}}{F}(z) & = &\frac{\sum_{\ve{\sigma} \in \{0,1\}^{n}}
                                (-1)^{n-1_{\sigma}} F (z + \sum_{i=1}^{n}\sigma_i v_i)}{\prod_{i=1}^{n}v_i}.\label{combSec}
\end{aligned}\\] Hence, \\(\diffloc{\mathcal{V}}{F}\\) is invariant to permutations of the elements of \\(\mathcal{V}\\).

</div>

<div class="proof" markdown="1">

*Proof.* We show the result by induction on the size of \\(\mathcal{V}\\), first noting that \\[\begin{aligned}
\diffloc{\{v_1\}}{F}(z) = \diffloc{v_1}{F}(z) & \defeq & \frac{F(z+v_1)-F(z)}{v_1} = \frac{1}{\prod_{i=1}^{1}v_i}\cdot \sum_{\sigma \in \{0,1\}}
                                (-1)^{1-1_{\sigma}} F (z + \sigma v_1).
  
\end{aligned}\\] We then assume that <a href="#combSec" data-reference-type="eqref" data-reference="combSec">[combSec]</a> holds for \\(\mathcal{V}_n \defeq
  \{v_1, v_2, ..., v_n\}\\) and show the result for \\(\mathcal{V}_{n+1} \defeq \mathcal{V}_n \cup
  \{v_{n+1}\}\\), writing (induction hypothesis used in the second identity): \\[\begin{aligned}
    \lefteqn{\diffloc{\mathcal{V}_{n+1}}{F}(z)}\nonumber\\
    & \defeq &
                                           \frac{\diffloc{\mathcal{V}_n}{F}(z+v_{n+1})
                                           - \diffloc{\mathcal{V}_n}{F}(z)
                                           }{v_{n+1}}\nonumber\\
    & = & \frac{\sum_{\ve{\sigma} \in \{0,1\}^{n}}
                                (-1)^{n-1_{\ve{\sigma}}} F (z + \sum_{i=1}^n \sigma_i v_i + v_{n+1}) - \sum_{\ve{\sigma} \in \{0,1\}^{n}}
                                (-1)^{n-1_{\ve{\sigma}}} F (z + \sum_{i=1}^n\sigma_i v_i)}{\prod_{i=1}^{n+1}v_i}\nonumber\\
    & = & \frac{\sum_{\ve{\sigma} \in \{0,1\}^{n}}
                                (-1)^{n-1_{\ve{\sigma}}} F (z + \sum_{i=1}^n\sigma_i v_i + v_{n+1}) + \sum_{\ve{\sigma} \in \{0,1\}^{n}}
                                (-1)^{n-1_{\ve{\sigma}}+1} F (z + \sum_{i=1}^n\sigma_i v_i)}{\prod_{i=1}^{n+1}v_i}\nonumber\\
                                & = & \frac{
                                      \left\{
                                      \begin{array}{l}
                                        \sum_{\ve{\sigma'} \in \{0,1\}^{n+1} : \sigma'_{n+1} = 1}
                                (-1)^{n-(1_{\ve{\sigma'}}-1)} F (z + \sum_{i=1}^{n+1}\sigma'_i v_i) \\
                                        + \sum_{\ve{\sigma'} \in \{0,1\}^{n+1} : \sigma'_{n+1} = 0}
                                (-1)^{n+1-1_{\ve{\sigma'}}} F (z + \sum_{i=1}^{n+1}\sigma'_i v_i) 
                                      \end{array}
    \right.
                                           }{v^{n+1}}\nonumber\\
                                 & = & \frac{\sum_{\ve{\sigma'} \in \{0,1\}^{n+1}}
                                (-1)^{n+1-1_{\ve{\sigma'}}} F (z + \sum_{i=1}^{n+1}\sigma'_i v_i)}{\prod_{i=1}^{n+1}v_i},
  
\end{aligned}\\] as claimed. ◻

</div>

We also have the following simple Lemma, which is a direct consequence of Lemma <a href="#lemCombSec" data-reference-type="ref" data-reference="lemCombSec">[lemCombSec]</a>.

<div class="lemma" markdown="1">

<span id="lemComp1" label="lemComp1"></span> For all \\(z, \in \mathbb{R}, v, z' \in \mathbb{R}_*\\), we have \\[\begin{aligned}
\diffloc{v}{F}(z+z') & = & \diffloc{v}{F}(z)  + z'\cdot \diffloc{\{z',v\}}{F}(z).
    
\end{aligned}\\]

</div>

<div class="proof" markdown="1">

*Proof.* It comes from Lemma <a href="#lemCombSec" data-reference-type="ref" data-reference="lemCombSec">[lemCombSec]</a> that \\(\diffloc{\{z',v\}}{F}(z) =
\diffloc{\{v, z'\}}{F}(z) =
(\diffloc{v}{F}(z+z')-\diffloc{v}{F}(z))/z'\\) (and we reorder terms). ◻

</div>

## Removing the \\(\neq 0\\) part in Assumption <a href="#assum1finite" data-reference-type="ref" data-reference="assum1finite">[assum1finite]</a> [remove-nonzero]

Because everything needs to be encoded, finiteness is not really an assumption. However, the non-zero assumption may be seen as limiting (unless we are happy to use first-order information about the loss (Section <a href="#sec:boost" data-reference-type="ref" data-reference="sec:boost">5</a>). There is a simple trick to remove it. Suppose \\(h_t\\) zeroes on some training examples. The training sample being finite, there exists an open neighborhood \\(\mathbb{I}\\) in 0 such that \\(h'_t \defeq h_t + \delta\\) does not zero anymore on training examples, for any \\(\delta \in \mathbb{I}\\). This changes the advantage \\(\upgamma\\) in the WLA (Definition <a href="#assum3wla" data-reference-type="ref" data-reference="assum3wla">[assum3wla]</a>) to some \\(\upgamma'\\) satisfying (we assume \\(\delta>0\\) wlog) \\[\begin{aligned}
  \upgamma' & \geq & \frac{\upgamma M_t}{M_t + \delta} - \frac{\delta}{M_t + \delta}\\
  & \geq & \upgamma - \frac{\delta}{M_t} \cdot (1+\upgamma),
\end{aligned}\\] from which it is enough to pick \\(\delta \leq \varepsilon \upgamma M_t/(1+\upgamma)\\) to guarantee advantage \\(\gamma' \geq (1-\varepsilon) \gamma\\). If \\(\varepsilon\\) is a constant, this translates in a number of boosting iterations in Corollary <a href="#corBOOSTRATE" data-reference-type="ref" data-reference="corBOOSTRATE">[corBOOSTRATE]</a> affected by a constant factor that we can choose as close to 1 as desired.

## Proof of Lemma <a href="#lemW2bound" data-reference-type="ref" data-reference="lemW2bound">[lemW2bound]</a> [proof_lemW2bound]

<figure id="f-secsec">
<div class="center">
<table>
<tbody>
<tr>
<td style="text-align: center;"><img src="./figures/Fig-secsec.png"" style="width:40.0%" /></td>
<td style="text-align: center;"><img src="./figures/Fig-conv-secsec.png"" style="width:40.0%" /></td>
</tr>
</tbody>
</table>
</div>
<figcaption><em>Left</em>: representation of the difference of averages in <a href="#eqDIFFAVG" data-reference-type="eqref" data-reference="eqDIFFAVG">[eqDIFFAVG]</a>. Each of the secants <span class="math inline">(<em>Δ</em><sub>1</sub>)</span> and <span class="math inline">(<em>Δ</em><sub>2</sub>)</span> can take either the red or black segment. Which one is which depends on the signs of <span class="math inline"><em>c</em></span> and <span class="math inline"><em>b</em></span>, but the general configuration is always the same. Note that if <span class="math inline"><em>F</em></span> is convex, one necessarily sits above the other, which is the crux of the proof of Lemma <a href="#lemW2bound" data-reference-type="ref" data-reference="lemW2bound">[lemW2bound]</a>. For the sake of illustration, suppose we can analytically have <span class="math inline"><em>b</em>, <em>c</em> → 0</span>. As <span class="math inline"><em>c</em></span> converges to 0 but <span class="math inline"><em>b</em></span> remains <span class="math inline"> &gt; 0</span>, <span class="math inline">$\diffloc{\{b, c\}}{F}(a)$</span> becomes proportional to the variation of the average secant midpoint; the then-convergence of <span class="math inline"><em>b</em></span> to 0 makes <span class="math inline">$\diffloc{\{b, c\}}{F}(a)$</span> converge to the second-order derivative of <span class="math inline"><em>F</em></span> at <span class="math inline"><em>a</em></span>. <em>Right</em>: in the special case where <span class="math inline"><em>F</em></span> is convex, one of the secants always sits above the other.</figcaption>
</figure>

We reformulate \\[\begin{aligned}
\diffloc{\{b, c\}}{F}(a) & = & \frac{2}{b} \cdot \frac{1}{c}\cdot \left(\underbrace{\frac{F(a+b+c)+F(a)}{2}}_{\defeq \mu_2} - \underbrace{\frac{F(a+b) + F(a + c )}{2}}_{\defeq \mu_1}\right).\label{eqDIFFAVG}
\end{aligned}\\] Both \\(\mu_1\\) and \\(\mu_2\\) are averages that can be computed from the midpoints of two secants (respectively): \\[\begin{aligned}
(\Delta_1) & \defeq & [(a+c, F(a+c)),(a+b, F(a+b))],\\
(\Delta_2) & \defeq & [(a, F(a)),(a+b+c, F(a+b+c))].
\end{aligned}\\] Also, the midpoints of both secants have the same abscissa (and the ordinates are \\(\mu_1\\) and \\(\mu_2\\)), so to study the sign of \\(\diffloc{\{b, c\}}{F}(a)\\), we can study the position of both secants with respect to each other. \\(F\\) being convex, we show that the abscissae of one secant are included in the abscissae of the other, this being sufficient to give the position of both secants with respect to each other. We distinguish four cases.  
**Case 1**: \\(c>0, b>0\\). We have \\(a+b+c > \max\{a+b, a + c\}\\) and \\(a < \min \{a+b, a + c\}\\). \\(F\\) being convex, \\((\Delta_2)\\) sits above \\((\Delta_1)\\). So, \\(\mu_2 \geq \mu_1\\) and finally \\(\diffloc{\{b, c\}}{F}(a) \geq 0\\).  
**Case 2**: \\(c<0, b<0\\). We now have \\(a+b + c< \min\{a+b, a + c\}\\) while \\(a > \max\{a+b, a+c\}\\), so \\((\Delta_2)\\) sits above \\((\Delta_1)\\). Again, \\(\mu_2 \geq \mu_1\\) and finally \\(\diffloc{\{b, c\}}{F}(a) \geq 0\\).  
**Case 3**: \\(c>0, b<0\\). We have \\(a+b < a\\) and \\(a+b < a+b + c\\). Also \\(a + c> \max\{a+b+ c, a \}\\), so this time \\((\Delta_2)\\) sits below \\((\Delta_1)\\) but \\(c b < 0\\), so \\(\diffloc{\{b, c\}}{F}(a) \geq 0\\) again.  
**Case 4**: \\(c<0, b>0\\). So \\(a + c < a < a+b\\) and \\(a + c < a+b + c\\). So \\(a + c < \min\{a, a+b + c\}\\) and \\(a+b > \max\{a, a + c\}\\), so \\((\Delta_2)\\) sits below \\((\Delta_1)\\). Since \\(c b < 0\\), so \\(\diffloc{\{b, c\}}{F}(a) \geq 0\\) again.  

## Proof of Theorem <a href="#thBOOSTCH" data-reference-type="ref" data-reference="thBOOSTCH">[thBOOSTCH]</a> [proof_thBOOSTCH]

Let us remind key simplified notations about edges, \\(\forall t\geq 0\\): \\[\begin{aligned}
\tilde{e}_{ti} & \defeq & y_i \cdot H_{t}(\ve{x}_i),\label{defTedge}\\
e_{ti} & \defeq & y_i \cdot 
                             \alpha_{t} h_{t}(\ve{x}_i) =
                  \tilde{e}_{ti} - \tilde{e}_{(t-1)i}.\label{defedge}
\end{aligned}\\] For short, we also let: \\[\begin{aligned}
Q^*_{ti} & \defeq & Q^*_F(\tilde{e}_{ti},\tilde{e}_{(t-1)i},v_{i(t-1)}),\\
\Delta_{ti} & \defeq & \diffloc{v_{i(t-1)}}{F}(\tilde{e}_{ti}) -
                      \diffloc{v_{i(t-1)}}{F}(\tilde{e}_{(t-1)i}), \label{defDELTA}
\end{aligned}\\] where \\(Q^*_{..}\\) is defined in <a href="#defQSTAR" data-reference-type="eqref" data-reference="defQSTAR">[defQSTAR]</a>. We also split the computation of the leveraging coefficient \\(\alpha_t\\) in in two parts, the first computing a real \\(a_t\\) as: \\[\begin{aligned}
a_t & \in & \frac{1}{2(1+\varepsilon_t)M_t^2 \overline{W}_{2,t}} \cdot \left[1-\pi_t,
          1+\pi_t\right], \label{pickat}
\end{aligned}\\] and then using \\(\alpha_t \leftarrow a_t \eta_t\\). We now use Lemma <a href="#lemGENR" data-reference-type="ref" data-reference="lemGENR">[lemGENR]</a> (main file) and get \\[\begin{aligned}
\expect_{i\sim [m]}\left[ \bregmansec{F}{v_{ti}}(
 \tilde{e}_{ti}\|\tilde{e}_{(t+1)i}) \right] & \geq & -
                                                               \expect_{i\sim
                                                               D}\left[Q^*_{(t+1)i}
                                                               \right],
                                                               \forall
                                                               t\geq 0.\label{eqBCDIFF2}
\end{aligned}\\] If we reorganise <a href="#eqBCDIFF2" data-reference-type="eqref" data-reference="eqBCDIFF2">[eqBCDIFF2]</a> using the definition of \\(\bregmansec{F}{.}(.\|.)\\), we get: \\[\begin{aligned}
\lefteqn{\expect_{i\sim [m]}\left[
  F(\tilde{e}_{(t+1)i}) \right]}\nonumber\\
  & \leq & \expect_{i\sim [m]}\left[
  F(\tilde{e}_{ti}) \right] - \expect_{i\sim [m]}\left[ (\tilde{e}_{ti}- \tilde{e}_{(t+1)i}) \cdot \diffloc{v_{ti}}{F}(\tilde{e}_{(t+1)i})\right]
                                            + \expect_{i\sim [m]}\left[ Q^*_{(t+1)i}  \right]\nonumber\\
  & & =   \expect_{i\sim [m]}\left[
  F(\tilde{e}_{ti}) \right] - \expect_{i\sim [m]}\left[ -e_{(t+1)i} \cdot \diffloc{v_{ti}}{F}(\tilde{e}_{(t+1)i})\right]+ \expect_{i\sim [m]}\left[ Q^*_{(t+1)i}  \right]\label{useDEF1}\\
  & = &  \expect_{i\sim [m]}\left[
  F(\tilde{e}_{ti}) \right] + \alpha_{t+1}  \cdot \expect_{i\sim [m]}\left[ y_i
                                            h_{t+1} (\ve{x}_i) \cdot \diffloc{v_{ti}}{F}(\tilde{e}_{(t+1)i})\right]+ \expect_{i\sim [m]}\left[ Q^*_{(t+1)i}  \right]\label{useDEF2}\\
  & = &  \expect_{i\sim [m]}\left[
  F(\tilde{e}_{ti}) \right]  + a_{t+1}  \eta_{t+1}  \cdot \expect_{i\sim [m]}\left[ y_i
                                            h_{t+1} (\ve{x}_i) \cdot \diffloc{v_{ti}}{F}(\tilde{e}_{ti})\right]\nonumber\\
  & & + a_{t+1}  \eta_{t+1}  \cdot \expect_{i\sim [m]}\left[ y_i
                                            h_{t+1} (\ve{x}_i) \cdot
        \Delta_{(t+1)i} \right] +
      \expect_{i\sim [m]}\left[ Q^*_{(t+1)i}  \right]\label{useDEF3}\\
  & = &   \expect_{i\sim [m]}\left[
  F(\tilde{e}_{ti}) \right] - a_{t+1}  \eta_{t+1}  \cdot \underbrace{\expect_{i\sim [m]}\left[ w_{(t+1)i} y_i
                                            h_{t+1} (\ve{x}_i)\right]}_{=\eta_{t+1}}+ a_{t+1}  \eta_{t+1}  \cdot \expect_{i\sim [m]}\left[ y_i
                                            h_{t+1} (\ve{x}_i) \cdot
        \Delta_{(t+1)i} \right] \nonumber\\
  & & +
      \expect_{i\sim [m]}\left[ Q^*_{(t+1)i}  \right]\nonumber\\
  & = &   \expect_{i\sim [m]}\left[
  F(\tilde{e}_{ti}) \right] - a_{t+1} \eta^2_{t+1} + a_{t+1} \eta_{t+1} \cdot \expect_{i\sim [m]}\left[ y_i
                                            h_{t+1}(\ve{x}_i) \cdot
      \Delta_{(t+1)i} \right] +
      \expect_{i\sim [m]}\left[ Q^*_{(t+1)i}  \right]\label{eq111}.
\end{aligned}\\] <a href="#useDEF1" data-reference-type="eqref" data-reference="useDEF1">[useDEF1]</a> – <a href="#useDEF3" data-reference-type="eqref" data-reference="useDEF3">[useDEF3]</a> make use of definitions <a href="#defedge" data-reference-type="eqref" data-reference="defedge">[defedge]</a> (twice) and <a href="#defDELTA" data-reference-type="eqref" data-reference="defDELTA">[defDELTA]</a> as well as the decomposition of the leveraging coefficient in <a href="#pickat" data-reference-type="eqref" data-reference="pickat">[pickat]</a>.

Looking at <a href="#eq111" data-reference-type="eqref" data-reference="eq111">[eq111]</a>, we see that we can have a boosting-compliant decrease of the loss if the two quantities depending on \\(\Delta_{(t+1).}\\) and \\(Q^*_{(t+1).}\\) can be made small enough compared to \\(a_{t+1} \eta^2_{t+1}\\). This is what we investigate.  
**Bounding the term depending on \\(\Delta_{(t+1).}\\)** – We use Lemma <a href="#lemComp1" data-reference-type="ref" data-reference="lemComp1">[lemComp1]</a> with \\(z \defeq\tilde{e}_{ti},  z'  \defeq e_{(t+1)i}, v \defeq  v_t\\), which yields (also using <a href="#defedge" data-reference-type="eqref" data-reference="defedge">[defedge]</a> and the assumption that \\(h_{t+1}(\ve{x}_i)\neq 0\\)): \\[\begin{aligned}
    \Delta_{(t+1)i} & \defeq & \diffloc{v_{ti}}{F}(\tilde{e}_{(t+1)i}) - \diffloc{v_{ti}}{F}(\tilde{e}_{ti})\nonumber\\
                    & = & \diffloc{v_{ti}}{F}(\tilde{e}_{ti} + e_{(t+1)i}) - \diffloc{v_{ti}}{F}(\tilde{e}_{ti}) \nonumber\\
                    & = & e_{(t+1)i}\cdot \diffloc{\{e_{(t+1)i}, v_{ti}\}}{F}(\tilde{e}_{ti})\nonumber\\
    & = & y_i \cdot 
                             \alpha_{t+1} h_{t+1}(\ve{x}_i) \cdot \diffloc{\{e_{(t+1)i}, v_{ti}\}}{F}(\tilde{e}_{ti}),
    
\end{aligned}\\] and so we get: \\[\begin{aligned}
\lefteqn{a_{t+1} \eta_{t+1} \cdot \expect_{i\sim [m]}\left[ y_i
                                            h_{t+1}(\ve{x}_i) \cdot
      \Delta_{(t+1)i} \right]}\nonumber\\
 & = & a_{t+1} \eta_{t+1} \cdot \expect_{i\sim [m]}\left[ \alpha_{t+1} (y_i
                                            h_{t+1}(\ve{x}_i))^2 \cdot
                             \diffloc{\{e_{(t+1)i},v_{ti}\}}{F}(\tilde{e}_{ti})\right]\nonumber\\
  & = & a_{t+1}^2 \eta_{t+1}^2 \cdot \expect_{i\sim [m]}\left[ (h_{t+1}(\ve{x}_i))^2 \cdot
                             \diffloc{\{e_{(t+1)i},v_{ti}\}}{F}(\tilde{e}_{ti})\right]\nonumber\\
  & \leq & a_{t+1}^2 \eta_{t+1}^2 M^2_{t+1} \cdot \overline{W}_{2,t+1}.\label{eq112}
\end{aligned}\\]

**Bounding the term depending on \\(Q^*_{.(t+1)}\\)** – We immediately get from the value picked in argument of \\(\mathbb{I}_{t+1}\\) in step 2.5 of , the definition of \\(\mathbb{I}_{ti}(.)\\) in <a href="#defIT" data-reference-type="eqref" data-reference="defIT">[defIT]</a> and our decomposition \\(\alpha_t \leftarrow a_t \eta_t\\) that \\(Q^*_{(t+1)i} \leq \varepsilon_{t+1} \cdot
                                               a_{t+1}^2  \eta_{t+1}^2
                                               M_{t+1}^2 \cdot
                                                 \overline{W}_{2,t+1}, \forall i\in [m]\\), so that: \\[\begin{aligned}
  \expect_{i\sim [m]}\left[ Q^*_{(t+1)i}  \right] & \leq & \varepsilon_{t+1} \cdot
                                               a_{t+1}^2  \eta_{t+1}^2
                                               M_{t+1}^2 \cdot
                                                 \overline{W}_{2,t+1}. \label{eq113}
\end{aligned}\\] **Finishing up with the proof** – Suppose that we choose \\(\varepsilon_{t+1} > 0\\), \\(\pi_{t+1} \in (0,1)\\) and \\(a_{t+1}\\) as in <a href="#pickat" data-reference-type="eqref" data-reference="pickat">[pickat]</a>. We then get from <a href="#eq111" data-reference-type="eqref" data-reference="eq111">[eq111]</a>, <a href="#eq112" data-reference-type="eqref" data-reference="eq112">[eq112]</a>, <a href="#eq113" data-reference-type="eqref" data-reference="eq113">[eq113]</a> that for any choice of \\(v_{ti}\\) in Step 2.5 of , \\[\begin{aligned}
\lefteqn{\expect_{i\sim [m]}\left[
  F(\tilde{e}_{(t+1)i}) \right]}\nonumber\\
  & \leq & \expect_{i\sim [m]}\left[
  F(\tilde{e}_{ti}) \right] - a_{t+1} \eta^2_{t+1} + a_{t+1}^2 \eta_{t+1}^2 M^2_{t+1}
                                            \cdot \overline{W}_{2,t+1}
                                            + \varepsilon_{t+1} \cdot a_{t+1}^2  \eta_{t+1}^2 M_{t+1}^2 \cdot \overline{W}_{2,t+1}\nonumber\\
& & = \expect_{i\sim [m]}\left[
  F(\tilde{e}_{ti}) \right] - 
a_{t+1} \eta^2_{t+1} \cdot \left( 1
                                            - a_{t+1} \left(
          1 + \varepsilon_{t+1} \right) M_{t+1}^2 \cdot
    \overline{W}_{2,t+1} \right) \nonumber\\
  & \leq & \expect_{i\sim [m]}\left[
  F(\tilde{e}_{ti}) \right] - 
\frac{\eta^2_{t+1} (1-\pi_{t+1}^2)}{4 \left(
          1 + \varepsilon_{t+1} \right) M_{t+1}^2 \cdot
    \overline{W}_{2,t+1} } \label{beq134},
\end{aligned}\\] where the last inequality is a consequence of <a href="#pickat" data-reference-type="eqref" data-reference="pickat">[pickat]</a>. Suppose we pick \\(H_0 \defeq h_0 \in \mathbb{R}\\) a constant and \\(v_0>0\\) such that \\[\begin{aligned}
\diffloc{v_{0}}{F}(h_0) & \neq & 0.
  
\end{aligned}\\] The final classifier \\(H_T\\) of  satisfies: \\[\begin{aligned}
\expect_{i\sim [m]}\left[
  F(y_i H_{T}(\ve{x}_i))\right] & \leq & F_0 -
                                           \frac{1}{4} \cdot
                                           \sum_{t=1}^T
                                         \frac{\eta^2_t (1-\pi_{t}^2)}{
                                           (1+\varepsilon_{t})M_t^2 \overline{W}_{2,t}}, \label{beq15}
\end{aligned}\\] with \\(F_0\defeq \expect_{i\sim [m]}\left[
  F(\tilde{e}_{i0}) \right] \defeq \expect_{i\sim [m]}\left[
  F(y_i H_0) \right] = \expect_{i\sim [m]}\left[
  F(y_i h_0)\right]\\). If we want \\(\expect_{i\sim [m]}\left[
  F(y_i H_{T}(\ve{x}_i))\right] \leq F(z^*)\\), assuming wlog \\(F(z^*)
\leq F_0\\), then it suffices to iterate until: \\[\begin{aligned}
\sum_{t=1}^T
                                         \frac{1-\pi_{t}^2}{
                                           \overline{W}_{2,t}(1+\varepsilon_{t})} \cdot \frac{\eta^2_t }{M_t^2 }& \geq & 4(F_0 - F(z^*)). \label{beq16}
\end{aligned}\\] Remind that the edge \\(\eta_t\\) is not normalized. We have defined a normalized edge, \\[\begin{aligned}
[-1,1] \ni \tilde{\eta}_t & \defeq & \sum_i \frac{|w_{ti}|}{W_t} \cdot \tilde{y}_{ti}
                                     \cdot
                                     \frac{h_t(\ve{x}_i)}{M_t},
\end{aligned}\\] with \\(\tilde{y}_{ti} \defeq y_i \cdot \mathrm{sign}(w_{ti})\\) and \\(W_t \defeq \sum_i |w_{ti}| = \sum_i |\diffloc{v_{(t-1)i}}{F}(\tilde{e}_{(t-1)i})|\\). We have the simple relationship between \\(\eta_t\\) and \\(\tilde{\eta}_t\\): \\[\begin{aligned}
\tilde{\eta}_t & = & \sum_i \frac{|w_{ti}|}{W_t} \cdot
                                     (y_i \cdot \mathrm{sign}(w_{ti}))
                                     \cdot
                     \frac{h_t(\ve{x}_i)}{M_t}\nonumber\\
               & = & \frac{1}{W_t M_t} \cdot \sum_i w_{ti} y_i h_t(\ve{x}_i)\nonumber\\
  & = & \frac{m}{W_t M_t} \cdot \eta_t,
  
\end{aligned}\\] resulting in (\\(\forall t\geq 1\\)), \\[\begin{aligned}
  \frac{\eta^2_t}{M_t^2} & = & \tilde{\eta}^2_t \cdot \left(\frac{W_t}{m}\right)^2\nonumber\\
                         & = & \tilde{\eta}^2_t \cdot \left(\expect_{i\sim [m]}\left[|\diffloc{v_{(t-1)i}}{F}(\tilde{e}_{(t-1)i})|\right]\right)^2\nonumber\\
  & \geq & \tilde{\eta}^2_t \cdot \left(\left|\expect_{i\sim [m]}\left[\diffloc{v_{(t-1)i}}{F}(\tilde{e}_{(t-1)i})\right]\right|\right)^2\nonumber\\
  & & =  \tilde{\eta}^2_t \cdot \overline{W}_{1,t}^2, \label{bETA}
\end{aligned}\\] recalling \\(\overline{W}_{1,t} \defeq \left|\expect_{i\sim
   D}\left[\diffloc{v_{(t-1)i}}{F}(\tilde{e}_{(t-1)i})\right]\right|\\). It comes from <a href="#bETA" data-reference-type="eqref" data-reference="bETA">[bETA]</a> that a sufficient condition for <a href="#beq16" data-reference-type="eqref" data-reference="beq16">[beq16]</a> to hold is: \\[\begin{aligned}
\sum_{t=1}^T
                                         \frac{\overline{W}^2_{1,t} (1-\pi_{t}^2)}{
                                           \overline{W}_{2,t} (1+\varepsilon_{t}) }\cdot \tilde{\eta}^2_t& \geq & 4(F_0 - F(z^*)), \label{beq17}
\end{aligned}\\] which is the statement of Theorem <a href="#thBOOSTCH" data-reference-type="ref" data-reference="thBOOSTCH">[thBOOSTCH]</a>.

## Proof of Lemma <a href="#lemSmooth" data-reference-type="ref" data-reference="lemSmooth">[lemSmooth]</a> [proof_lemSmooth]

We first observe that for any \\(a\in\mathbb{R}, b,c\in \mathbb{R}_*\\), \\[\begin{aligned}
  |\diffloc{\{b, c\}}{F}(a)| & = & \frac{1}{|bc|} \cdot \left|
                                 \begin{array}{c}
                                   F(a+b+c) - F(a+c) - bF'(a+c)\\
                                   -(F(a+b) - F(a) - bF'(a))\\
                                   +b(F'(a+c) - F'(a))
                                 \end{array}\right|\nonumber\\
                             & \leq & \frac{1}{|bc|} \cdot \left(
                                      \begin{array}{c}
                                   |F(a+b+c) - F(a+c) - bF'(a+c)|\\
                                   +|(F(a+b) - F(a) - bF'(a))|\\
                                   +|b(F'(a+c) - F'(a))|
                                 \end{array}
  \right)\nonumber\\
  & \leq & \frac{1}{|bc|} \cdot \left( \frac{\beta}{2}\cdot b^2 + \frac{\beta}{2}\cdot b^2 + \beta |bc| \right) = \beta + \beta\cdot \frac{b^2}{|bc|},\label{decomp1}
\end{aligned}\\] where we used the \\(\beta\\)-smoothness of \\(F\\) and twice `\cite[Lemma 3.4]{bCOA}`{=latex}. We can also make a permutation in the expression of \\(\diffloc{\{b, c\}}{F}(a)\\) and instead write \\[\begin{aligned}
  |\diffloc{\{b, c\}}{F}(a)| & = & \frac{1}{|bc|} \cdot \left|
                                 \begin{array}{c}
                                   F(a+b+c) - F(a+b) - cF'(a+b)\\
                                   -(F(a+c) - F(a) - cF'(a))\\
                                   +c(F'(a+b) - F'(a))
                                 \end{array}\right|\nonumber\\
                             & \leq & \frac{1}{|bc|} \cdot \left(
                                      \begin{array}{c}
                                   |F(a+b+c) - F(a+b) - cF'(a+b)|\\
                                   +|(F(a+c) - F(a) - cF'(a))|\\
                                   +|c(F'(a+b) - F'(a))|
                                 \end{array}
  \right)\nonumber\\
  & \leq & \frac{1}{|bc|} \cdot \left( \frac{\beta}{2}\cdot c^2 + \frac{\beta}{2}\cdot c^2 + \beta |bc| \right) = \beta + \beta\cdot \frac{c^2}{|bc|}.\label{decomp2}
\end{aligned}\\] We thus have \\[\begin{aligned}
  |\diffloc{\{b, c\}}{F}(a)| & \leq & \beta + \beta \cdot \left(\frac{\min\{|b|, |c|\}}{\sqrt{|bc|}}\right)^2 \label{bbound1}\\
  & \leq & 2\beta,\nonumber
\end{aligned}\\] by the power mean inequality `\cite[Chapter III, Theorem 2]{bHO}`{=latex}. Since \\(|h_{t}(\ve{x}_i)| \leq M_t\\) by definition, we thus have \\[\begin{aligned}
\left|  \expect_{i\sim [m]}\left[ \diffloc{\{e_{ti},v_{(t-1)i}\}}{F}(\tilde{e}_{(t-1)i}) \cdot \left(\frac{h_{t}(\ve{x}_i)}{M_{t}}\right)^2 \right]  \right| & \leq & 2\beta,
\end{aligned}\\] which allows us to fix \\(\overline{W}_{2,t} = 2\beta\\) and completes the proof of Lemma <a href="#lemSmooth" data-reference-type="ref" data-reference="lemSmooth">[lemSmooth]</a>.

<div class="remark" markdown="1">

Our result is optimal in the sense that if we make one offset (say \\(b\\)) go to zero, then the ratio in <a href="#bbound1" data-reference-type="eqref" data-reference="bbound1">[bbound1]</a> goes to zero and we recover the condition on the \\(v\\)-derivative of the derivative, \\(|\diffloc{c}{F'}(z)| \leq \beta\\).

</div>

## Proof of Theorem <a href="#thALPHAW2" data-reference-type="ref" data-reference="thALPHAW2">[thALPHAW2]</a> [proof_thALPHAW2]

We consider the upperbound:: \\[\begin{aligned}
  \lefteqn{\overline{W}_{2,t}}\nonumber\\
  & \defeq & \left|\expect_{i\sim [m]}\left[\frac{h^2_{t}(\ve{x}_i)}{M^2_{t}} \cdot \diffloc{\{e_{ti},v_{(t-1)i}\}}{F}(\tilde{e}_{(t-1)i})\right]\right|\nonumber\\
  & = & \left|\expect_{i\sim [m]}\left[\frac{h^2_{t}(\ve{x}_i)}{M^2_{t}} \cdot \frac{1}{\tilde{e}_{ti}} \cdot \left(\frac{F(\tilde{e}_{ti}+v_{(t-1)i}) - F(\tilde{e}_{ti})}{v_{(t-1)i}} - \frac{F(\tilde{e}_{(t-1)i} + v_{(t-1)i} )-F(\tilde{e}_{(t-1)i})}{v_{(t-1)i}}\right)\right]\right|\nonumber\\
  & = & \left|\frac{1}{\alpha_{t}} \cdot \expect_{i\sim [m]}\left[\frac{h_{t}(\ve{x}_i)}{y_i M^2_{t}} \cdot \left(\frac{F(\tilde{e}_{ti}+v_{(t-1)i}) - F(\tilde{e}_{ti})}{v_{(t-1)i}} - \frac{F(\tilde{e}_{(t-1)i} + v_{(t-1)i} )-F(\tilde{e}_{(t-1)i})}{v_{(t-1)i}}\right)\right]\right|\nonumber\\
  & = & \left|\frac{1}{\alpha_{t}} \cdot \expect_{i\sim [m]}\left[\frac{y_i h_{t}(\ve{x}_i)}{M^2_{t}} \cdot \left(\diffloc{v_{(t-1)i}}{F}(\tilde{e}_{ti}) - \diffloc{v_{(t-1)i}}{F}(\tilde{e}_{(t-1)i})\right)\right]\right|\label{simpW2}
\end{aligned}\\] (The last identity uses the fact that \\(y_i \in \{-1,1\}\\)). Remark that we have extracted \\(\alpha_{t}\\) from the denominator but it is still present in the arguments \\(\tilde{e}_{ti}\\). For any classifier \\(h\\), we introduce notation \\[\begin{aligned}
\eta(\ve{w}, h) & \defeq & \expect_{i\sim [m]}\left[w_i y_i h(\ve{x}_i)\right],
\end{aligned}\\] and so \\(\eta_{t}\\) (Step 2.2 in ) is also \\(\eta(\ve{w}_{t}, h_{t})\\), which is guaranteed to be non-zero by the Weak Learning Assumption <a href="#assum3wla" data-reference-type="eqref" data-reference="assum3wla">[assum3wla]</a>. We want, for *some* \\(\epsilon_{t} > 0, \pi_{t} \in [0,1)\\), \\[\begin{aligned}
\alpha_{t} & \in & \frac{\eta_t}{2(1+\varepsilon_{t})M_{t}^2 \overline{W}_{2,t}} \cdot \left[1-\pi_{t},
          1+\pi_{t}\right]. \label{pickaA}
\end{aligned}\\] This says that the sign of \\(\alpha_{t}\\) is the same as the sign of \\(\eta(\ve{w}_{t}, h_{t}) = \eta_t\\). Since we know its sign, let us look for its absolute value: \\[\begin{aligned}
|\alpha_{t}| & \in & \frac{|\eta_t|}{2(1+\varepsilon_{t})M_{t}^2 \overline{W}_{2,t}} \cdot \left[1-\pi_{t},
          1+\pi_{t}\right]. \label{pickaAbs}
\end{aligned}\\] From <a href="#genALPHA" data-reference-type="eqref" data-reference="genALPHA">[genALPHA]</a> (main file), we can in fact search \\(\alpha_t\\) in the union of all such intervals for \\(\epsilon_{t} > 0, \pi_{t} \in [0,1)\\), which amounts to find first: \\[\begin{aligned}
|\alpha_{t}| & \in & \left(0, \frac{|\eta_t|}{M_{t}^2 \overline{W}_{2,t}}\right),
\end{aligned}\\] and then find any \\(\epsilon_{t} > 0, \pi_{t} \in [0,1)\\) such that <a href="#pickaAbs" data-reference-type="eqref" data-reference="pickaAbs">[pickaAbs]</a> holds. Using <a href="#simpW2" data-reference-type="eqref" data-reference="simpW2">[simpW2]</a> and simplifying the external dependency on \\(\alpha_{t}\\), we then need \\[\begin{aligned}
1 & \in & \left(0, \underbrace{\frac{|\eta_t|}{|\expect_{i\sim [m]}\left[y_i h_{t}(\ve{x}_i) \cdot \left(\diffloc{v_{(t-1)i}}{F}(\alpha_t y_i h_t(\ve{x}_i) + \tilde{e}_{(t-1)i}) - \diffloc{v_{(t-1)i}}{F}(\tilde{e}_{(t-1)i})\right)\right]|}}_{\defeq B(\alpha_{t})}\right),\label{constINT}
\end{aligned}\\] under the constraint that the sign of \\(\alpha_t\\) be the same as that of \\(\eta_t\\). But, using notation <a href="#defpartialW" data-reference-type="eqref" data-reference="defpartialW">[defpartialW]</a> (main file), we have \\[\begin{aligned}
B(\alpha_{t}) & = & |\eta(\ve{w}_{t}, h_{t}) - \eta(\tilde{\ve{w}}_{t}(\alpha_t), h_{t})|,
\end{aligned}\\] and so to get <a href="#constINT" data-reference-type="eqref" data-reference="constINT">[constINT]</a> satisfied, it is sufficient that \\[\begin{aligned}
\frac{|\eta_t - \eta(\tilde{\ve{w}}_{t}(\alpha_t), h_{t})|}{|\eta_t|} & < & 1, \label{eqsolalpha}
\end{aligned}\\] which is Step 1 in . The Weak Learning Assumption <a href="#assum3wla" data-reference-type="eqref" data-reference="assum3wla">[assum3wla]</a> guarantees that the denominator is \\(\neq 0\\) so this can always be evaluated. The continuity of \\(F\\) in all \\(\tilde{e}_{(t-1)i}\\) guarantees \\(\lim_{\alpha_t \rightarrow 0} \eta(\tilde{\ve{w}}_{t}(\alpha_t), h_{t}) = \eta_t\\), and thus guarantees the existence of solutions to <a href="#eqsolalpha" data-reference-type="eqref" data-reference="eqsolalpha">[eqsolalpha]</a> for some \\(|\alpha_t| > 0\\).

To summarize, finding \\(\alpha_t\\) can be done in two steps, (i) solve \\[\begin{aligned}
\frac{\left|\eta_t - \eta(\tilde{\ve{w}}_{t}(\mathrm{sign}(\eta_t) \cdot a), h_{t})\right|}{|\eta_t|} & < & 1
  
\end{aligned}\\] for some \\(a > 0\\) and (ii) let \\(\alpha_{t} \defeq \mathrm{sign}(\eta_t) \cdot a\\). This is the output of (\\(S, \ve{w}_t, h_t\\)), which ends the proof of Theorem <a href="#thALPHAW2" data-reference-type="ref" data-reference="thALPHAW2">[thALPHAW2]</a>.

## Implementation of the offset oracle: particular cases [sec-offset-app]

Consider the "spring loss" that we define, for \\([.]\\) denoting the nearest integer, as: \\[\begin{aligned}
F_{\sllabel}(z) \defeq  \log(1+\exp(-z)) + 1 - \sqrt{1-4\left(z-[z]\right)^2}\label{defBWL}.
  
\end{aligned}\\] Figure <a href="#f-Spring" data-reference-type="ref" data-reference="f-Spring">7</a> plots this loss, which composes the logistic loss with a "\\(\mathsf{U}\\)"-shaped term. This loss would escape all optimization algorithms of Table <a href="#tab:features" data-reference-type="ref" data-reference="tab:features">[tab:features]</a> (), yet there is a trivial implementation of our offset oracle, as explained in Figure <a href="#f-Spring" data-reference-type="ref" data-reference="f-Spring">7</a>:

1.  if the interval \\(\mathbb{I}\\) defined by \\(\tilde{e}_{(t-1)i}\\) and \\(\tilde{e}_{ti}\\) contains at least one peak, compute the tangence point (\\(z_t\\)) at the closest local "\\(\mathsf{U}\\)" that passes through \\((\tilde{e}_{(t-1)i}, F(\tilde{e}_{(t-1)i}))\\); then if \\(z_t \in \mathbb{I}\\) then \\(v_{ti} \leftarrow z_t - \tilde{e}_{(t-1)i}\\), else \\(v_{ti} \leftarrow \tilde{e}_{ti} - \tilde{e}_{(t-1)i}\\);

2.  otherwise \\(F\\) in \\(\mathbb{I}\\) is strictly convex and differentiable: a simple dichotomic search can retrieve a feasible \\(v_{ti}\\) (see convex losses below);

Notice that one can alleviate the repetitive dichotomic search by pre-tabulating a feasible \\(v\\) for a set of differences \\(|a-b|\\) (\\(a,b\\) belonging to the abscissae of the same "\\(\mathsf{U}\\)") decreasing by a fixed factor, choosing \\(v_{ti} \leftarrow v\\) of the largest tabulated \\(|a-b|\\) no larger than \\(|\tilde{e}_{ti}-\tilde{e}_{(t-1)i}|\\).  
**Discontinuities** discontinuities do not represent issues if the argument \\(z\\) of \\(\mathbb{I}_{ti}(z)\\) is large enough, as shown from the following simple Lemma.

<div class="lemma" markdown="1">

<span id="lemSMALLDISC" label="lemSMALLDISC"></span> Define the discontinuity of \\(F\\) as: \\[\begin{aligned}
  \disc(F) \defeq \max \left\{
  \begin{array}{l}
    \sup_z |F(z) - \lim_{z^-}F(z)|, \\
    \sup_z |F(z) - \lim_{z^+}F(z)|
    \end{array}\right\}.
\end{aligned}\\] For any \\(z\geq 0\\), if \\(\disc(F) \leq z\\) then \\(\mathbb{I}_{ti}(z)\neq \emptyset, \forall t\geq 1, \forall i \in [m]\\).

</div>

Figure <a href="#f-BII-1" data-reference-type="ref" data-reference="f-BII-1">5</a> (c) shows a case where the discontinuity is larger than \\(z\\). In this case, an issue eventually happens for computing the next weight happens, only when the current edge is at the discontinuity. We note that as iterations increase and the weak learner finds it eventually more difficult to return weak hypotheses with \\(\eta_.\\) large enough, the discontinuities may become an issue for  to not stop at Step 2.5. Or one can always use a simple trick to avoid stopping and which relies on the leveraging coefficient \\(\alpha_t\\): this is described in the , Section <a href="#sec-handling" data-reference-type="ref" data-reference="sec-handling">9.9</a>.  
**The case of convex losses** If \\(F\\) is convex (not necessarily differentiable nor strictly convex), there is a simple way to find a valid output for the offset oracle, which relies on the following Lemma.

<div class="lemma" markdown="1">

<span id="lemOBIconv" label="lemOBIconv"></span> Suppose \\(F\\) convex. Then for any \\(z, z' \in \mathbb{R}, v\neq 0\\), \\[\begin{aligned}
    \lefteqn{\{v > 0 :  Q^*_F(z,z',v) = r \}}\nonumber\\
    & = & \left\{v > 0 :  D_F\left(z\left\| \frac{F(z+v) - F(z)}{v}\right.\right) = r \right\} . \label{eqBREGSYS}
    
\end{aligned}\\]

</div>

(proof in , Section <a href="#proof_lemOBIconv" data-reference-type="ref" data-reference="proof_lemOBIconv">9.8</a>) By definition, \\(\mathbb{I}_{ti}(z') \subseteq \mathbb{I}_{ti}(z)\\) for any \\(z'\leq z\\), so a simple way to implement the offset oracle’s output \\(\offsetoracle(t, i, z)\\) is, for some \\(0<r<z\\), to solve the Bregman identity in the RHS of <a href="#eqBREGSYS" data-reference-type="eqref" data-reference="eqBREGSYS">[eqBREGSYS]</a> and then return any relevant \\(v\\). If \\(F\\) is strictly convex, there is just one choice.

If solving the Bregman identity is tedious but \\(F\\) is strictly convex, there a simple dichotomic search that is guaranteed to find a feasible \\(v\\). It exploits the fact that the abscissa maximizing the difference between any secant of \\(F\\) and \\(F\\) has a simple closed form (see `\cite[Supplement, Figure 13]{cnBD}`{=latex}) and so the OBI in <a href="#eqBI" data-reference-type="eqref" data-reference="eqBI">[eqBI]</a> (Definition <a href="#defBI" data-reference-type="ref" data-reference="defBI">[defBI]</a>) has a closed form as well. In this case, it is enough, after taking a first non-zero guess for \\(v\\) (either positive or negative), to divide it by a constant \\(>1\\) until the corresponding OBI is no larger than the \\(z\\) in the query \\(\offsetoracle(t, i, z)\\).

<figure id="f-Spring">
<div class="center">
<img src="./figures/FigSpringLoss-opt.png"" style="width:90.0%" />
</div>
<figcaption>The spring loss in <a href="#defBWL" data-reference-type="eqref" data-reference="defBWL">[defBWL]</a> is neither convex, nor Lipschitz or differentiable and has an infinite number of local minima. Yet, an implementation of the offset oracle is trivial as an output for  can be obtained from the computation of a single tangent point (here, the orange <span class="math inline"><em>v</em></span>, see text; best viewed in color).</figcaption>
</figure>

## Proof of Lemma <a href="#lemOBIconv" data-reference-type="ref" data-reference="lemOBIconv">[lemOBIconv]</a> [proof_lemOBIconv]

<figure id="f-OBI-conv">
<div class="center">
<img src="./figures/FigOBI-conv.png"" style="width:80.0%" />
</div>
<figcaption>Computing the OBI <span class="math inline"><em>Q</em><sub><em>F</em></sub>(<em>z</em>, <em>z</em> + <em>v</em>, <em>z</em> + <em>v</em>)</span> for <span class="math inline"><em>F</em></span> convex, <span class="math inline">(<em>z</em>, <em>v</em>)</span> being given and <span class="math inline"><em>v</em> &gt; 0</span>. We compute the line <span class="math inline">(<em>Δ</em><sub><em>t</em></sub>)</span> crossing <span class="math inline"><em>F</em></span> at any point <span class="math inline"><em>t</em></span>, with slope equal to the secant <span class="math inline">[(<em>z</em>, <em>F</em>(<em>z</em>)), (<em>z</em> + <em>v</em>, <em>F</em>(<em>z</em> + <em>v</em>))]</span> and then the difference between <span class="math inline"><em>F</em></span> at <span class="math inline"><em>z</em> + <em>v</em></span> and this line at <span class="math inline"><em>z</em> + <em>v</em></span>. We move <span class="math inline"><em>t</em></span> so as to maximize this difference. The optimal <span class="math inline"><em>t</em></span> (in green) gives the corresponding OBI. In <a href="#defIzvr" data-reference-type="eqref" data-reference="defIzvr">[defIzvr]</a> and <a href="#defIzvr2" data-reference-type="ref" data-reference="defIzvr2">[defIzvr2]</a>, we are interested in finding <span class="math inline"><em>v</em></span> given this difference, <span class="math inline"><em>r</em></span>. We also need to replicate this computation for <span class="math inline"><em>v</em> &lt; 0</span>.</figcaption>
</figure>

\\(F\\) being convex, we first want to compute the set \\[\begin{aligned}
  \mathbb{I}_{z,r} \defeq \{v > 0 :  Q_F(z,z+v,z+v) = r \}, \label{defIzvr}
\end{aligned}\\] where \\(r\\) is supposed small enough for \\(\mathbb{I}_{z,r}\\) to be non-empty. There is a simple graphical solution to this which, as Figure <a href="#f-OBI-conv" data-reference-type="ref" data-reference="f-OBI-conv">8</a> explains, consists in finding \\(v\\) solution of \\[\begin{aligned}
\sup_t F(z+v) - \left(F(t) + \left(\frac{F(z+v) - F(z)}{v}\right)\cdot(z+v-t)\right) & = & r.
\end{aligned}\\] The LHS simplifies: \\[\begin{aligned}
  \lefteqn{\sup_t F(z+v) - \left(F(t) + \left(\frac{F(z+v) - F(z)}{v}\right)\cdot(z+v-t)\right)}\\
  & = & \frac{(z+v)F(z) -z F(z+v)}{v} + \sup_t \left\{ t\cdot \frac{F(z+v) - F(z)}{v} - F(t) \right\}\\
  & = & \frac{(z+v)F(z) -z F(z+v)}{v} + F^\star\left(\frac{F(z+v) - F(z)}{v}\right)\\
  & = & F(z) + F^\star\left(\frac{F(z+v) - F(z)}{v}\right) - z \cdot \frac{F(z+v) - F(z)}{v}\\
  & = & D_F\left(z\left\| \frac{F(z+v) - F(z)}{v}\right.\right),
\end{aligned}\\] so we end up with an equivalent but more readable definition for \\(\mathbb{I}_{z,r}\\): \\[\begin{aligned}
  \mathbb{I}_{z,r} & = & \left\{v > 0 :  D_F\left(z\left\| \frac{F(z+v) - F(z)}{v}\right.\right) = r \right\}, \label{defIzvr2}
\end{aligned}\\] which yields the statement of the Lemma.

## Handling discontinuities in the offset oracle to prevent stopping in Step 2.5 [sec-handling]

Theorem <a href="#thBOOSTCH" data-reference-type="ref" data-reference="thBOOSTCH">[thBOOSTCH]</a> and Lemma <a href="#corBOOSTRATE" data-reference-type="ref" data-reference="corBOOSTRATE">[corBOOSTRATE]</a> require to run  for as many iterations are required. This implies not early stopping in Step 2.5. Lemma <a href="#lemSMALLDISC" data-reference-type="ref" data-reference="lemSMALLDISC">[lemSMALLDISC]</a> shows that early stopping can only be triggered by too large local discontinuities at the edges. This is a weak requirement on running , but there exists a weak assumption on the discontinuities of the loss itself that simply prevent any early stopping and does not degrade the boosting rates. The result exploits the freedom in choosing \\(\alpha_t\\) in Step 2.3.

<div class="lemma" markdown="1">

<span id="lemDISC" label="lemDISC"></span> Suppose \\(F\\) is any function defined over \\(\mathbb{R}\\) discontinuities of zero Lebesgue measure. Then Corollary <a href="#corBOOSTRATE" data-reference-type="ref" data-reference="corBOOSTRATE">[corBOOSTRATE]</a> holds for boosting \\(F\\) with its inequality strict while never triggering early stopping in Step 2.5 of .

</div>

<div class="proof" markdown="1">

*Proof.* To show that we never trigger stopping in Step 2.5, it is sufficient to show that we can run while ensuring \\(F\\) is continuous in an open neighborhood around all edges \\(y_iH_t(\ve{x}_i), \forall i \in [m], \forall t\geq 0\\) (by letting \\(H_0 \defeq h_0\\)). Remind that \\(\tilde{e}_{ti} \defeq \tilde{e}_{(t-1)i} + \alpha_t \cdot y_t h_t(\ve{x}_i)\\), so changing \\(\alpha_t\\) changes all edges. We just have to show that either computing \\(\alpha_t\\) ensures such a continuity, or \\(\alpha_t\\) can be slightly modified to do so. We have two ways to compute \\(\alpha_t\\):

1.  using a value for \\(\overline{W}_{2,t}\\) that represents an "absolute" upperbound in the sense of <a href="#boundW2" data-reference-type="eqref" data-reference="boundW2">[boundW2]</a> (*e.g.* Lemma <a href="#lemSmooth" data-reference-type="ref" data-reference="lemSmooth">[lemSmooth]</a>) and then compute \\(\alpha_t\\) as in Step 2.3 of ;

2.  using algorithm .

Because of the assumption on \\(F\\), we can always ensure that \\(F\\) is continuous in an open neighborhood of all edges (the basis of the induction amounts to a straightforward choice for \\(h_0\\)). This proves the Lemma for \[2.\].

If we rely on \[1.\] and the \\(\alpha_t\\) computed leads to some discontinuities, then we have complete control to change \\(\alpha_t\\): any continuous change of \\(\epsilon_t\\) induces a continuous change in \\(\alpha_t\\) and thus a continuous change of all edges as well. So, starting from the initial \\(\epsilon_t\\) chosen in Step 2.3, we increase it to a value \\(\epsilon_t^* > \epsilon_t\\), which we want to keep as small as possible. We can define for each \\(i \in [m]\\) an open set \\((a_i, b_i)\\) which is the interval spanned by the new \\(\tilde{e}_{ti}(\epsilon'_t)\\) using \\(\epsilon'_t \in (\epsilon_t, \epsilon_t^*)\\). Since there are only finitely many discontinuities on \\(F\\), there exists a small \\(\epsilon_t^* > \epsilon_t\\) such that \\[\begin{aligned}
\forall i \in [m], \forall z \in (a_i, b_i), F \mbox{ is continuous on } z.
\end{aligned}\\] This means that \\(\forall \epsilon'_t \in (\epsilon_t, \epsilon_t^*)\\), we end up with a loss without any discontinuities on the new edges. Now comes the reason why we want \\(\epsilon_t^* - \epsilon_t\\) small: we can check that there always exist a small enough \\(\epsilon_t^*>\epsilon_t\\) such that for any \\(\epsilon'_t\\) we choose, the boosting rate in Corollary <a href="#corBOOSTRATE" data-reference-type="ref" data-reference="corBOOSTRATE">[corBOOSTRATE]</a> is affected by at most 1 additional iteration. Indeed, while we slightly change parameter \\(\epsilon_t\\) to land all new edges outside of discontinuities of \\(F\\), we *also* increase the contribution of the boosting iteration in the RHS of <a href="#corCONV" data-reference-type="eqref" data-reference="corCONV">[corCONV]</a> by a quantity \\(\delta > 0\\) which can be made as small as required — hence we can just replace the inequality in <a href="#corCONV" data-reference-type="eqref" data-reference="corCONV">[corCONV]</a> by a strict inequality. This proves the statement of the Lemma if we rely on \[1.\] above.

This completes the proof of Lemma <a href="#lemDISC" data-reference-type="ref" data-reference="lemDISC">[lemDISC]</a>. ◻

</div>

## A boosting pattern that can "survive" above differentiability [sec-survive]

<figure id="f-BII-2">
<div class="center">
<table>
<tbody>
<tr>
<td style="text-align: center;"><img src="./figures/FigII-5.png"" style="width:90.0%" /></td>
</tr>
</tbody>
</table>
</div>
<figcaption>Case <span class="math inline"><em>F</em></span> strictly convex, with two cases of limit OBI <span class="math inline"><em>z</em></span> and <span class="math inline"><em>z</em><sup>′</sup></span> in <span class="math inline">𝕀<sub>.<em>i</em></sub>(.)</span>. Example <span class="math inline"><em>i</em></span> has <span class="math inline"><em>e</em><sub><em>t</em><em>i</em></sub> &gt; 0</span> and <span class="math inline"><em>e</em><sub>(<em>t</em> − 1)<em>i</em></sub> &gt; 0</span> <a href="#defEDGE1" data-reference-type="eqref" data-reference="defEDGE1">[defEDGE1]</a> large enough (hence, edges with respect to weak classifiers <span class="math inline"><em>h</em><sub><em>t</em></sub></span> and <span class="math inline"><em>h</em><sub><em>t</em> − 1</sub></span> large enough) so that <span class="math inline">𝕀<sub><em>t</em><em>i</em></sub>(<em>z</em>) ∩ 𝕀<sub>(<em>t</em> − 1)<em>i</em></sub>(<em>z</em>) = 𝕀<sub>(<em>t</em> − 1)<em>i</em></sub>(<em>z</em>) ∩ 𝕀<sub>(<em>t</em> − 2)<em>i</em></sub>(<em>z</em>) = 𝕀<sub><em>t</em><em>i</em></sub>(<em>z</em>) ∩ 𝕀<sub>(<em>t</em> − 2)<em>i</em></sub>(<em>z</em>) = ∅</span>. In this case, regardless of the offsets chosen by , we are guaranteed that its weights satisfy <span class="math inline"><em>w</em><sub>(<em>t</em> + 1)<em>i</em></sub> &lt; <em>w</em><sub><em>t</em><em>i</em></sub> &lt; <em>w</em><sub>(<em>t</em> − 1)<em>i</em></sub></span>, which follows the boosting pattern that examples receiving the right classification by weak classifiers have their weights decreasing. If however the limit OBI changes from <span class="math inline"><em>z</em></span> to a larger <span class="math inline"><em>z</em><sup>′</sup></span>, this is not guaranteed anymore: in this case, it may be the case that <span class="math inline"><em>w</em><sub>(<em>t</em> + 1)<em>i</em></sub> &gt; <em>w</em><sub><em>t</em><em>i</em></sub></span>.</figcaption>
</figure>

Suppose \\(F\\) is strictly convex and strictly decreasing as for classical convex surrogates (*e.g.* logistic loss). Assuming wlog all \\(\alpha_. > 0\\) and example \\(i\\) has both \\(y_i h_{t}(\ve{x}_i) > 0\\) and \\(y_i h_{t-1}(\ve{x}_i) > 0\\), as long as \\(z\\) is small enough, we are guaranteed that any choice \\(v_{t-1} \in \mathbb{I}_{(t-1)i}(z)\\) and \\(v_t \in \mathbb{I}_{ti}(z)\\) results in \\(0 < w_{(t+1)i} < w_{ti}\\), which follows the classical boosting pattern that examples receiving the right class by weak hypotheses have their weight decreased (See Figure <a href="#f-BII-2" data-reference-type="ref" data-reference="f-BII-2">9</a>). If \\(z = z'\\) is large enough, then this does not hold anymore as seen from Figure <a href="#f-BII-2" data-reference-type="ref" data-reference="f-BII-2">9</a>.

## The case of piecewise constant losses for  [sec-piecewise]

<figure id="fig:codensLap">
<img src="./figures/plot.png"" />
<figcaption>How our algorithm works with the 0/1 loss (in red): at the initialization stage, assuming we pick <span class="math inline"><em>h</em><sub>0</sub> = 0</span> for simplicity and some <span class="math inline"><em>v</em><sub>0</sub> &lt; 0</span>, all training examples get the same weight, given by negative the slope of the thick blue dashed line. All weights are thus <span class="math inline"> &gt; 0</span>. At iteration <span class="math inline"><em>t</em></span> when we update the weights (Step 2.6), one of two cases can happen on some training example <span class="math inline">$(\ve{x},y)$</span>. In <strong>(A)</strong>, the edge of the strong model remains the same: either both are positive (blue) or both negative (olive green) (the ordering of edges is not important). In this case, regardless of the offset, the new weight will be 0. In <strong>(B)</strong>, both edges have different sign (again, the ordering of edges is not important). In this case, the examples will keep non-zero weight over the next iteration. See text below for details.</figcaption>
</figure>

Figure <a href="#fig:codensLap" data-reference-type="ref" data-reference="fig:codensLap">10</a> schematizes a run of our algorithm when training loss = 0/1 loss. At the initialization, it is easy to get all examples to have non-zero weight. The weight update for example \\((\ve{x},y)\\) of our algorithm in Step 2.3 is (negative) the slope of a secant that crosses the loss in two points, both being in between \\(y H_{t-1}(\ve{x})\\) and \\(y H_{t}(\ve{x})\\). Hence, if the predicted label does not change (\\(\mathrm{sign}(H_{t}(\ve{x})) = \mathrm{sign}(H_{t-1}(\ve{x}))\\)), then the next weight (\\(w_{t+1}\\)) of the example *will be zero* (Figure <a href="#fig:codensLap" data-reference-type="ref" data-reference="fig:codensLap">10</a>, case (A)). However, if the predicted label does change (\\(\mathrm{sign}(H_{t}(\ve{x})) \neq \mathrm{sign}(H_{t-1}(\ve{x}))\\)) then the example may get a non-zero weight depending on the offset chosen.

Hence, our generic implementation of Algorithms 3 and 4 may completely fail at providing non-zero weights for the next iteration, which makes the algorithm stop in step 2.7. And even when not all weights are zero, there may be just a too small subset of those, that would break the Weak Learning Assumption for boosting compliance of the next iteration (Assumption 5.5).

# Supplementary material on algorithms, implementation tricks and a toy experiment [sec-sup-exp]

## Algorithm and implementation of  and how to find parameters from Theorem <a href="#thALPHAW2" data-reference-type="ref" data-reference="thALPHAW2">[thALPHAW2]</a> [sec-solvealpha]

As Theorem <a href="#thALPHAW2" data-reference-type="ref" data-reference="thALPHAW2">[thALPHAW2]</a> explains,  can easily get to not just the leveraging coefficient \\(\alpha_t\\), but also other parameters that are necessary to implement : \\(\overline{W}_{2,t}\\) and \\(\epsilon_t\\) (both used in Step 2.5). We now provide a simple pseudo code on how to implement  amnd get, on top of it, the two other parameters. We do not seek \\(\pi_t\\) since it is useful only in the convergence analysis. Also, our proposal implementation is optimized for complexity (because of the geometric updating of \\(\delta, W\\) in their respective loops) but much less so for for accuracy. Algorithm  explains the overall procedure.

<figure id="findalphaimpl">
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><strong>Input</strong> sample <span class="math inline"><em>S</em> = {(<strong>x</strong><sub><em>i</em></sub>, <em>y</em><sub><em>i</em></sub>), <em>i</em> = 1, 2, ..., <em>m</em>}</span>, <span class="math inline">$\ve{w} \in \mathbb{R}^m$</span>, <span class="math inline"><em>h</em> : 𝒳 → ℝ</span>, <span class="math inline"><em>M</em> ≠ 0</span>. // in our case, <span class="math inline">$\ve{w} \leftarrow \ve{w}_t; h \leftarrow h_t; M \leftarrow M_t$</span> (current weights, weak hypothesis and max confidence, see Step 2.3 in  and Assumption <a href="#assum1finite" data-reference-type="ref" data-reference="assum1finite">[assum1finite]</a>) Step 1 : // all initializations <span class="math display">$$\begin{aligned}
    \eta_{\mbox{\tiny init}} &amp; \leftarrow &amp; \eta(\ve{w}, h);\\
    \delta &amp; \leftarrow &amp; 1.0; \label{pickdelta}\\
     W_{\mbox{\tiny init}} &amp; \leftarrow &amp; 1.0;
  
\end{aligned}$$</span> Step 2 : <strong>do</strong> // Step 2 computes the leveraging coefficient <span class="math inline"><em>α</em><sub><em>t</em></sub></span> <span class="math inline">$\alpha \leftarrow \delta \cdot \mathrm{sign}(\eta_{\mbox{\tiny init}})$</span>; <span class="math inline">$\eta_{\mbox{\tiny new}} \leftarrow \eta(\tilde{\ve{w}}(\alpha), h)$</span>; <strong>if</strong> <span class="math inline">$\left|\eta_{\mbox{\tiny new}} - \eta_{\mbox{\tiny init}}\right| &lt; |\eta_{\mbox{\tiny init}}|$</span> <strong>then</strong> <span class="math inline"><code>found</code>_<code>alpha</code> ← <code>true</code></span> <strong>else</strong> <span class="math inline"><em>δ</em> ← <em>δ</em>/2</span>; <strong>while</strong> <span class="math inline"><code>found</code>_<code>alpha</code> = <code>false</code></span>; Step 3 : <span class="math inline"><em>W</em>←</span> Left Hand Side of <a href="#boundW2" data-reference-type="eqref" data-reference="boundW2">[boundW2]</a> (main file) // Step 3 computes <span class="math inline">$\overline{W}_{2,t}$</span> // we can use <a href="#boundW2" data-reference-type="eqref" data-reference="boundW2">[boundW2]</a> (main file) because we know <span class="math inline"><em>α</em></span> <strong>if</strong> <span class="math inline">$W =_{\mbox{\tiny machine}} 0$</span> <strong>then</strong> // the LHS of <a href="#boundW2" data-reference-type="eqref" data-reference="boundW2">[boundW2]</a> is (machine) 0: just need to find <span class="math inline"><em>W</em></span> such that <a href="#genALPHA" data-reference-type="eqref" data-reference="genALPHA">[genALPHA]</a> holds ! <span class="math inline">$W \leftarrow W_{\mbox{\tiny init}}$</span>; <strong>while</strong> <span class="math inline">$|\alpha| &gt; |\eta_{\mbox{\tiny init}}| / (W \cdot M^2)$</span> <strong>do</strong> <span class="math inline"><em>W</em> ← <em>W</em>/2</span>; <strong>endif</strong> Step 4 : <span class="math inline">$b_{\mbox{\tiny sup}} \leftarrow |\eta_{\mbox{\tiny init}}| / (W \cdot M^2)$</span>;// Step 4 computes <span class="math inline"><em>ϵ</em><sub><em>t</em></sub></span> <span class="math inline">$\epsilon \leftarrow (b_{\mbox{\tiny sup}} / \alpha)-1$</span>; <strong>Return</strong> <span class="math inline">(<em>α</em>, <em>W</em>, <em>ϵ</em>)</span>;</p>
</div>
<figcaption>(<span class="math inline">$S, \ve{w}, h, M$</span>)</figcaption>
</figure>

## Algorithm and implementation of the offset oracle [sec-offsetoracle]

There exists a very simple trick to get some adequate offset \\(v\\) to satisfy <a href="#constoo" data-reference-type="eqref" data-reference="constoo">[constoo]</a> (main file), explained in Figure <a href="#f-good-v" data-reference-type="ref" data-reference="f-good-v">12</a>. In short, we seek the optimally bended secant and check that the OBI is no more than a required \\(z\\). This can be done via parsing the interval \\([\tilde{e}_{ti},\tilde{e}_{(t-1)i}]\\) using regularly spaced values. If the OBI is too large, we can start again with a smaller step size. Algorithm  details the key part of the search.

<figure id="f-good-v">
<div class="center">
<img src="./figures/FigOO-simple.png"" style="width:80.0%" />
</div>
<figcaption>How to find some <span class="math inline"><em>v</em> ∈ 𝕀<sub><em>t</em><em>i</em></sub>(<em>z</em>)</span>: parse the interval <span class="math inline">[<em>ẽ</em><sub><em>t</em><em>i</em></sub>, <em>ẽ</em><sub>(<em>t</em> − 1)<em>i</em></sub>]</span> with a regular step <span class="math inline"><em>δ</em></span>, seek the secant with minimal slope (because <span class="math inline"><em>ẽ</em><sub><em>t</em><em>i</em></sub> &lt; <em>ẽ</em><sub>(<em>t</em> − 1)<em>i</em></sub></span>; otherwise, we would seek the secant with maximal slope). It is necessarily the one minimizing the OBI among all regularly spaced choices. If the OBI is still too large, decrease the step <span class="math inline"><em>δ</em></span> and start the search again.</figcaption>
</figure>

<figure id="oimpl">
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><strong>Input</strong> loss <span class="math inline"><em>F</em></span>, two last edges <span class="math inline"><em>ẽ</em><sub><em>t</em></sub>, <em>ẽ</em><sub><em>t</em> − 1</sub></span>, maximal OBI <span class="math inline"><em>z</em></span>, precision <span class="math inline"><em>Z</em></span>. // in our case, <span class="math inline"><em>ẽ</em><sub><em>t</em></sub> ← <em>ẽ</em><sub><em>t</em><em>i</em></sub>; <em>ẽ</em><sub><em>t</em> − 1</sub> ← <em>ẽ</em><sub>(<em>t</em> − 1)<em>i</em></sub>;</span> (for training example index <span class="math inline"><em>i</em> ∈ [<em>m</em>]</span>) Step 1 : // all initializations <span class="math display">$$\begin{aligned}
    \delta &amp; \leftarrow &amp; \frac{\tilde{e}_{t-1}-\tilde{e}_{t}}{Z};\\
    z_c &amp; \leftarrow &amp; \tilde{e}_t + \delta;\\
    i &amp; \leftarrow &amp; 0;
  
\end{aligned}$$</span> Step 2 : <strong>do</strong> <span class="math inline">$s_c \leftarrow \textsc{slope}(F,\tilde{e}_t,z_c)$</span>; // returns the slope of the secant passing through <span class="math inline">(<em>ẽ</em><sub><em>t</em></sub>, <em>F</em>(<em>ẽ</em><sub><em>t</em></sub>))</span> and <span class="math inline">(<em>z</em><sub><em>c</em></sub>, <em>F</em>(<em>z</em><sub><em>c</em></sub>))</span> <strong>if</strong> <span class="math inline">(<em>i</em> = 0) ∨ ((<em>δ</em> &gt; 0) ∧ (<em>s</em><sub><em>c</em></sub> &lt; <em>s</em><sub>*</sub>)) ∨ ((<em>δ</em> &lt; 0) ∧ (<em>s</em><sub><em>c</em></sub> &gt; <em>s</em><sub>*</sub>)))</span> <strong>then</strong> <span class="math inline"><em>s</em><sub>*</sub> ← <em>s</em><sub><em>c</em></sub></span>; <span class="math inline"><em>z</em><sub>*</sub> ← <em>z</em><sub><em>c</em></sub></span> <strong>endif</strong> <span class="math inline"><em>z</em><sub><em>c</em></sub> ← <em>z</em><sub><em>c</em></sub> + <em>δ</em></span>; <span class="math inline"><em>i</em> ← <em>i</em> + 1</span>; <strong>while</strong> <span class="math inline">(<em>z</em><sub><em>c</em></sub> − <em>ẽ</em><sub><em>t</em></sub>) ⋅ (<em>z</em><sub><em>c</em></sub> − <em>ẽ</em><sub><em>t</em> − 1</sub>) &lt; 0</span>; // checks that <span class="math inline"><em>z</em><sub><em>c</em></sub></span> is still in the interval <strong>Return</strong> <span class="math inline"><em>z</em><sub>*</sub> − <em>ẽ</em><sub><em>t</em></sub></span>;// this is the offset <span class="math inline"><em>v</em></span></p>
</div>
<figcaption>(<span class="math inline"><em>F</em>, <em>ẽ</em><sub><em>t</em></sub>, <em>ẽ</em><sub><em>t</em> − 1</sub>, <em>z</em>, <em>Z</em></span>)</figcaption>
</figure>

## A toy experiments [sec-toyexp]

We provide here a few toy experiments using . These are just meant to display that a simple implementation of the algorithm, following the blueprints given above, can indeed manage to optimize various losses. These are not meant to explain how to pick the best hyperparameters (*e.g.* <a href="#pickdelta" data-reference-type="eqref" data-reference="pickdelta">[pickdelta]</a>) nor how to choose the best loss given a domain, a problem that is far beyond the scope of our paper.

In this implementation, the weak learner learns decision trees and we minimize Matushita’s loss at the leaves of decision trees to learn fixed size trees, see `\cite{kmOTj}`{=latex} for the criterion and induction scheme, which is standard for decision trees.  is implemented as is given in the paper, and so are the implementation of  and the offset oracle provided above. We have made no optimization whatsoever, with one exception: when numerical approximation errors lead to an offset that is machine 0, we replace it by a small random value to prevent the use of derivatives in .

<figure id="f-losses-impl">
<div class="center">
<div class="tabular">
<p><span>c?c</span> <img src="./figures/clipped-log-2.png"" style="width:33.0%" alt="image" /> &amp; <img src="./figures/spring-loss-q500.png"" style="width:33.0%" alt="image" /><br />
Clipped logistic loss, <span class="math inline"><em>q</em> = −2</span> &amp; Spring loss, <span class="math inline"><em>Q</em> = 500</span><br />
</p>
</div>
</div>
<figcaption>Crops of the two losses whose optimization has been experimentally tested with , in addition to the logistic loss. See text for details.</figcaption>
</figure>

We have investigated three losses. The first is the well known logistic loss: \\[\begin{aligned}
F_{\loglabel}(z) & \defeq & \log(1+\exp(-z)).
\end{aligned}\\] The other two are tweaks of the logistic loss. We have investigated a clipped version of the logistic loss, \\[\begin{aligned}
F_{\cllabel,q}(z) & \defeq & \min\{\log(1+\exp(-z)), \log(1+\exp(-q))\}\label{defCLL},
\end{aligned}\\] with \\(q \in \mathbb{R}\\), which clips the logistic loss above a certain value. This loss is non-convex and non-differentiable, but it is Lipschitz. We have also investigated a generalization of the spring loss (main file): \\[\begin{aligned}
F_{\sllabel,Q}(z) & \defeq & \log(1+\exp(-z)) + \frac{1 - \sqrt{1-4\left(z_Q-[z_Q]\right)^2}}{Q}\label{defBWL2},
\end{aligned}\\] with \\(z_Q \defeq Qz - 1/2\\) (\\([.]\\) is the closest integer), which adds to the logistic loss regularly spaced peaks of variable width. This loss is non-convex, non-differentiable, non-Lipschitz. Figure <a href="#f-losses-impl" data-reference-type="ref" data-reference="f-losses-impl">14</a> provides a crop of the clipped logistic loss and spring loss we have used in our test. Notice the “hardness” that the spring loss intuitively represents for ML.

We provide an experiment on public domain UCI `\cite{dgUM}`{=latex} (using a 10-fold stratified cross-validation to estimate test errors). In addition to the three losses, we have crossed them with several other variables: the size of the trees (either they have a single internal node = stumps or at most 20 nodes) and, to give one example of how changing a (key) hyperparameter can change the result, we have tested for a scale of changes on the initial value of \\(\delta\\) in <a href="#pickdelta" data-reference-type="eqref" data-reference="pickdelta">[pickdelta]</a>. Finally, we have crossed all these variables with the existence of symmetric label noise in the training data, following the setup of `\cite{lsRC,mnwRC}`{=latex}. We flip each label in the training sample with probability \\(\eta\\). Table <a href="#f-toy-exp" data-reference-type="ref" data-reference="f-toy-exp">15</a> summarizes the results obtained. One can see that  manages to optimize all losses in pretty much all settings, with an eventual early stopping required for the spring loss if \\(\delta\\) is too large. Note that the best initial value for \\(\delta\\) depends on the loss optimized in these experiments: for \\(\delta=0.1\\), test error from the spring loss decreases much faster than for the other losses, yet we remind that the spring loss is just the logistic loss plus regularly spaced peaks. This could signal interesting avenues for the best possible implementation of , or a further understanding of the best formal ways to fix those paramaters, all of which are out of the scope of this paper.

<figure id="f-toy-exp">
<div class="center">
<div class="tabular">
<p><span>c?cc?cc</span> &amp; &amp;<br />
<span class="math inline"><em>η</em></span> &amp; Stumps &amp; Max size = 20&amp; Stumps &amp; Max size = 20<br />
<span class="math inline">0%</span> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__11h_29m_2s_noise_0_t100_s1_plot_DELTA0.1.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__11h_29m_2s_noise_0_t100_s20_plot_DELTA0.1.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__11h_29m_2s_noise_0_t100_s1_plot_DELTA1.0.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__11h_29m_2s_noise_0_t100_s20_plot_DELTA1.0.png}" style="width:33.0%" alt="image" /><br />
<span class="math inline">5%</span> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__10h_24m_22s_noise_0.05_t100_s1_plot_DELTA0.1.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__10h_24m_22s_noise_0.05_t100_s20_plot_DELTA0.1.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__10h_24m_22s_noise_0.05_t100_s1_plot_DELTA1.0.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__10h_24m_22s_noise_0.05_t100_s20_plot_DELTA1.0.png}" style="width:33.0%" alt="image" /><br />
<span class="math inline">10%</span> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__11h_16m_20s_noise_0.1_t100_s1_plot_DELTA0.1.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__11h_16m_20s_noise_0.1_t100_s20_plot_DELTA0.1.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__11h_16m_20s_noise_0.1_t100_s1_plot_DELTA1.0.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__11h_16m_20s_noise_0.1_t100_s20_plot_DELTA1.0.png}" style="width:33.0%" alt="image" /><br />
<span class="math inline">20%</span> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__10h_38m_3s_noise_0.2_t100_s1_plot_DELTA0.1.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__10h_38m_3s_noise_0.2_t100_s20_plot_DELTA0.1.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__10h_38m_3s_noise_0.2_t100_s1_plot_DELTA1.0.png}" style="width:33.0%" alt="image" /> &amp; <img src="{Experiments/tictactoe/springloss_testerr_Jan_30th__10h_38m_3s_noise_0.2_t100_s20_plot_DELTA1.0.png}" style="width:33.0%" alt="image" /><br />
</p>
</div>
</div>
<figcaption>Experiments on UCI showing estimated test errors after minimizing each of the three losses we consider, with varying training noise level <span class="math inline"><em>η</em></span>, max tree size and initial hyperparameter <span class="math inline"><em>δ</em></span> value in <a href="#pickdelta" data-reference-type="eqref" data-reference="pickdelta">[pickdelta]</a>. See text.</figcaption>
</figure>

# NeurIPS Paper Checklist [neurips-paper-checklist]

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification: Our paper is a theory paper: all claims are properly formalized and used.

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: The discussion section is devoted to limitations and improvement of our results

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

14. Justification: Our paper is a theory paper: all assumptions, statements and proofs provided.

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

19. Justification: Though our paper is a theory paper, we have included in the supplement a detailed statement of all related algorithms and a toy experiment of a simple implementation of these algorithms showcasing a simple run on a public UCI domain.

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

24. Justification: Our paper is a theory paper. All algorithms we introduce are either in the main file or the appendix.

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

29. Justification: Our paper is a theory paper.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: Our paper is a theory paper.

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

38. Answer: .

39. Justification: Our paper is a theory paper.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification: The research of the paper follows the code of ethics.

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: Our paper is a theory paper.

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

54. Justification: No release of data or models.

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification: no outside code, data or models used requiring licensing.

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

64. Justification: No new assets provided.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification: No crowdsourcing or research with human subjects.

70. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

73. Answer:

74. Justification: No research with human subjects.

75. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

[^1]: Calculus "without limits" `\cite{kcQC}`{=latex} (thus without using derivatives), not to be confounded with calculus on quantum devices.

[^2]: This is an important class of losses since it encompasses the convex surrogates of symmetric proper losses `\cite{nmSL,rwID}`{=latex}
