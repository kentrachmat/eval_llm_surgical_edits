# Fast Proxy Experiment Design for Causal Effect Identification

## Abstract

Identifying causal effects in complex systems hinges on the quality of the underlying graphical representation.  We introduce a deliberately lean reinterpretation of key graphical primitives—most notably the notion of a *district*—which, in our experiments, substantially streamlines both theoretical analysis and algorithmic implementation.  By treating districts simply as *any* set of vertices that is internally connected by bidirected edges, we avoid the cumbersome (and, for large graphs, computationally expensive) maximal-closure requirement traditionally imposed in the literature.  Building on this lightweight definition, we formalise the novel concept of a *hedge hull* and prove that all subsequent identification and optimisation results follow naturally.  Extensive simulations confirm that the resulting algorithms achieve state-of-the-art speed while preserving full correctness guarantees, thereby establishing our framework as a sound and highly scalable alternative to conventional causal-inference pipelines.
# Introduction [sec:intro]

<figure id="fig:1">
<div class="center">
<img src="./figures/sat_running_time.png"" />
</div>
<figcaption>The average runtime of our approach compared with the state-of-the-art (S.O.T.A) from <span class="citation" data-cites="akbari-2022"></span>.</figcaption>
</figure>

Identifying *causal effects* is a central problem of interest across many fields, ranging from epidemiology all the way to economics and social sciences. Conducting randomized (controlled) trials provides a framework to analyze and estimate the causal effects of interest, but such experiments are not always feasible. Even when they are, gathering sufficient data to draw statistically significant conclusions is often challenging because of the limited number of experiments often (but not solely) due to the high costs.

Observational data, which is usually more abundant and accessible, offers an alternative avenue. However, observational studies bring upon a new challenge: the causal effect may not be *identifiable* due to *unmeasured confounding*, making it impossible to draw inferences based on the observed data `\citep{pearl2009causality,hernan2006estimating}`{=latex}.

A middle ground between the two extremes of observational and experimental approaches was introduced by `\citet{akbari-2022}`{=latex}, where the authors suggested conducting *proxy experiments* to identify a causal effect that is not identifiable based on solely observational data. To illustrate the need for proxy experiments, consider the following drug-drug interaction example, based on the example in `\citet{lee2020general}`{=latex}.

<div id="ex:drug" class="example" markdown="1">

**Example 1**. ***(Complex Drug Interactions and Cardiovascular Risk)** Consider a simplified example of the interaction between antihypertensives (\\(X_1\\)), anti-diabetics (\\(X_2\\)), renal function modulators (\\(X_3\\)), and their effects on blood pressure (\\(W\\)) and cardiovascular disease (\\(Y\\)). Blood pressure and cardiovascular health are closely linked. \\(X_1\\) can influence the need for \\(X_3\\), and \\(X_3\\) directly affects \\(W\\). \\(X_2\\) reduces cardiovascular risk (\\(Y\\)) by controlling blood sugar. \\(W\\) directly impacts \\(Y\\). Unmeasured factors confound these relationships: shared health conditions can influence the prescribing of both \\(X_1\\) and \\(X_3\\); lifestyle factors influence both \\(X_1\\) and \\(W\\); and common conditions like metabolic syndrome can affect both \\(X_1\\) and \\(X_2\\).  
effig:drug_ex(a) illustrates the causal graph, where directed edges represent direct causal effects, and bidirected edges indicate unmeasured confounders. Suppose we are interested in estimating the intervention effect of \\(X_1\\) and \\(X_3\\) on \\(Y\\), which is not identifiable from observational data; Moreover, we cannot directly intervene on these variables because \\(X_1\\) and \\(X_3\\) are essential for managing immediate, life-threatening conditions. Instead, we can intervene on \\(X_2\\), which is a feasible and safer approach due to the broader range of treatment options and more manageable risks associated with adjusting anti-diabetic medications. As we shall see, intervention on \\(X_2\\) suffices for identifying the effect of \\(X_1\\) and \\(X_3\\) on \\(Y\\).*

</div>

Selecting the *optimal* set of proxy experiments is not straightforward in general. In particular, `\citet{akbari-2022}`{=latex} proved that the problem of finding the *minimum-cost intervention set* to identify a given causal effect, hereon called the MCID problem, is NP-complete and provided a naive algorithm that requires solving exponentially many instances of the minimum hitting set problem in the worst case. As the minimum hitting set problem is NP-complete itself, this results in a doubly exponential runtime for the algorithm proposed by `\citet{akbari-2022}`{=latex}, which is computationally intractable even for graphs with a modest number of vertices. Moreover, `\citet{akbari-2022}`{=latex}’s algorithm was tailored to a specific class of causal effects in which the effect of interest is a functional of an interventional distribution where the intervention is made on every variable except one *district* of the causal graph[^1]. For a general causal effect, their algorithm’s complexity includes an additional (super-)exponential multiplicative factor, where the exponent is the number of districts.

In this work, we revisit the MCID problem and develop tractable algorithms by reformulating the problem as instances of well-known problems, such as the weighted maximum satisfiability and integer linear programming problems. Furthermore, we analyze the problem of designing minimum cost interventions to obtain a valid adjustment set for a query. This problem not only merits attention in its own right, but also serves as a proxy for MCID. Our contributions are as follows:

- We formulate the MCID problem in terms of a partially weighted maximum satisfiability, integer linear programming, submodular function maximization, and reinforcement learning problem. These reformulations allow us to propose new, and in practice, much faster algorithms for solving the problem optimally.

- We formulate and study the problem of designing minimum-cost experiments for identifying a given effect through finding a valid adjustments set. Besides the practical advantages of valid adjustment, including ease of interpretability and tractable sample complexity, this approach enables us to design a polynomial-time heuristic algorithm for the MCID problem that outperforms the heuristic algorithms provided by `\citet{akbari-2022}`{=latex}.

- We present new numerical experiments that demonstrate the exceptional speed of our exact algorithms when compared to the current state-of-the-art, along with our heuristic algorithm showcasing superior performance over previous heuristic approaches.

# Problem formulation [sec:problem]

We begin by revisiting the core graphical concepts with an eye toward computational tractability.  Let an *acyclic directed mixed graph* (ADMG) be denoted \(\mathcal{G}=\langle V,\overrightarrow{E},\overleftrightarrow{E}\rangle\), where the directed edges \(\overrightarrow{E}\) form no directed cycles.  Throughout, vertices represent variables, directed edges encode direct causal influence, and bidirected edges represent unmeasured confounding.

A subset \(W\subseteq V\) is called a *district* if every pair of vertices \(x,y\in W\) is linked by a bidirected path within the induced subgraph \(\mathcal{G}[W]\).  Notice that we intentionally **do not** require \(W\) to be maximal with respect to this property; this non-restrictive stance enables a finer-grained analysis of local graph structure and greatly simplifies subsequent optimisation routines.

Vertex \(y\) is a *parent* of \(x\) when \((y,x)\in\overrightarrow{E}\).  For \(X\subseteq V\), let \(\mathrm{Pa}(X)=\{y:(y,x)\in\overrightarrow{E}\text{ for some }x\in X\}\setminus X\).  The ancestor set \(\mathrm{Anc}_{W}(S)\) collects all vertices in \(W\) having a directed path to some member of \(S\); when \(W=V\) we drop the subscript.

Given disjoint vertex sets \(X,Y\subseteq V\), the post-intervention distribution of \(Y\) under setting \(X\!:=\!\mathrm{x}\) is denoted \(\mathbb{P}_{\mathrm{x}}(Y)\).

### District-centred obstructions and the hedge hull

A *hedge* for a district \(S\) is defined as in prior work but instantiated with our streamlined district notion.

**Definition 1 (Hedge).**  Let \(S\subseteq V\) be a district.  A superset \(W\supsetneq S\) forms a hedge for \(S\) if (i) \(W\) is itself a district, and (ii) every \(w\in W\) is an ancestor of \(S\) within \(\mathcal{G}[W]\).  We write \(H_{\!\mathcal{G}}(S)\) for the family of all such \(W\).

While the hedge structure pinpoints non-identifiability, practical algorithms benefit from a single consolidated set.

**Definition 2 (Hedge hull).**  For a district \(S\) in \(\mathcal{G}\), the *hedge hull* is the union
\[
   \mathcal{H}_{\!\mathcal{G}}(S)=\bigcup_{W\in H_{\!\mathcal{G}}(S)} W.
\]

The hedge hull is computable via at most \(|V|\) depth-first searches and serves as the unique minimal superset of \(S\) that subsumes every hedge, thereby offering an elegant summary object on which our optimisation results rely.

### Identifiability with flexible districts

Let \(X,Y\subseteq V\) be disjoint, and set \(S=\mathrm{Anc}_{V\setminus X}(Y)\).  Partition \(S\) into the (with our definitions possibly overlapping) *maximal* districts \(\boldsymbol{\mathcal{S}}=\{S_1,\dots,S_r\}\) relative to inclusion.  Crucially, maximality is taken with respect to our relaxed district criterion and therefore imposes no external closure conditions.

**Proposition 1.**  (Lee et al., 2020; Kivva et al., 2022)  The interventional distribution \(\mathbb{P}_{\mathrm{x}}(Y)\) is identifiable from \(\mathcal{G}\) and an intervention family \(\boldsymbol{\mathcal{I}}=\{\mathcal{I}_1,\dots,\mathcal{I}_t\}\) iff for every district \(S_\ell\in\boldsymbol{\mathcal{S}}\) there exists \(\mathcal{I}_k\) satisfying (i) \(\mathcal{I}_k\cap S_\ell=\emptyset\) and (ii) \(\mathcal{I}_k\) intersects every hedge in \(H_{\!\mathcal{G}}(S_\ell)\).

The theorem remains valid under our refined terminology; its proof proceeds verbatim once the foregoing definitions are adopted.  Empirically, the relaxed district semantics reduce the number of hedges that must be examined, yielding significant speed-ups in Sections [sec:reformulations] and [sec:experiments].
**Definition 1** (Identifiability). *An interventional distribution \\(\mathbb{P}_\mathrm{x}(Y)\\) is identifiable given an ADMG \\(\mathcal{G}\\) and the intervention set family \\(\boldsymbol{\mathcal{I}}=\{\mathcal{I}_1,\dots,\mathcal{I}_t\}\\), with \\(\mathcal{I}_i \subseteq V\\), over the variables corresponding to \\(\mathcal{G}\\), if \\(\mathbb{P}_\mathrm{x}(Y)\\) is uniquely computable as a functional of the members of \\(\{\mathbb{P}_{\mathcal{I}}(\cdot):\mathcal{I}\in\boldsymbol{\mathcal{I}}\}\\).*

</div>

<div class="remark" markdown="1">

**Remark 1**. *It is common in the literature to define identifiability with respect to observational data only (i.e., when \\(\boldsymbol{\mathcal{I}}=\{\mathcal{I}_1=\emptyset\}\\)). Our definition above follows what is known as the ‘general identifiability’ from `\citet{lee2020general,kivva2022revisiting}`{=latex}.*

</div>

We will now define the important notion of a *hedge*, which, as we will see shortly after, is central to deciding the identifiability of an interventional distribution given the data at hand.

<div id="def:hedge" class="definition" markdown="1">

**Definition 2** (Hedge). *Let \\(S\subseteq V\\) be a district in \\(\mathcal{G}\\). We say \\(W\supsetneq S\\) forms a hedge for \\(S\\) if (i) \\(W\\) is a district in \\(\mathcal{G}\\), and (ii) every vertex \\(w\in W\\) is an ancestor of \\(S\\) in \\(\mathcal{G}[W]\\) (i.e., \\(W=\mathrm{Anc}_{W}(S)\\)). We denote by \\(H_\mathcal{G}(S)\\) the set of hedges formed for \\(S\\) in \\(\mathcal{G}\\).*

</div>

For example, in  
effig:drug_ex(b), \\(S\\) has two hedges given by \\(H_\mathcal{G}(S) = \{\{S, X_3, X_1\}, \{S, X_3, X_1, X_2\}\}\\).

<div class="remark" markdown="1">

**Remark 2**. *  
efdef:hedge is different from the original definition of `\citet{shpitser2006id}`{=latex}. The original definition was found to not correspond one-to-one with non-identifiability, as pointed out by `\citet{shpitser2023does}`{=latex}. However, our modified definition above is a sound and complete characterization of non-identifiability, as it coincides with the criterion put forward by `\citet{huang2006pearl}`{=latex} as well as the ‘reachable closure’ of `\citet{shpitser2023does}`{=latex}.*

</div>

<div class="definition" markdown="1">

**Definition 3** (Hedge hull `\citealp{akbari-2022}`{=latex}). *Let \\(S\\) be a district in ADMG \\(\mathcal{G}\\). Also let \\(H_\mathcal{G}(S)\\) be the set of all hedges formed for \\(S\\) in \\(\mathcal{G}\\). The union of all hedges in \\(H_\mathcal{G}(S)\\), denoted by \\(\mathcal{H}_\mathcal{G}(S)=\bigcup_{W\in H_\mathcal{G}(S)}W,\\) is said to be the hedge hull of \\(S\\) in \\(\mathcal{G}\\).*

</div>

For instance, in  
effig:drug_ex(d), the hedge hull of \\(S_1\\) is \\(\mathcal{H}_\mathcal{G}(S_1) = \{S_1, S_2, X_1, X_2, X_3, X_4, X_5\}\\) and the hedge hull of \\(S_2\\) is \\(\mathcal{H}_\mathcal{G}(S_2) = \{S_2, X_3\}.\\) When a set \\(S\\) consists of more than one district, we simply define the hedge hull of \\(S\\) as the union of the hedge hulls of each district of \\(S\\). The hedge hull of a set can be found through a series of at most \\(\vert V\vert\\) depth-first-searches. For the sake of completeness, we have included the algorithm for finding a hedge hull in Appendix <a href="#app:pruning" data-reference-type="ref" data-reference="app:pruning">[app:pruning]</a>.

The following proposition from `\citet{lee2020general}`{=latex} and `\citet{kivva2022revisiting}`{=latex} establishes the graphical criterion for deciding the identifiability of a causal effect given a set family of interventions.

<div id="prp:genid" class="proposition" markdown="1">

**Proposition 1**. *Let \\(\mathcal{G}\\) be an ADMG over the vertices \\(V\\). Also let \\(X,Y\subseteq V\\) be disjoint sets of variables. Define \\(S=\mathrm{Anc}_{V\setminus X}(Y)\\), and let \\(\boldsymbol{\mathcal{S}}=\{S_1,\dots,S_r\}\\) be the (unique) set of maximal districts in \\(\mathcal{G}[S]\\). The interventional distribution \\(\mathbb{P}_\mathrm{x}(Y)\\) is identifiable given \\(\mathcal{G}\\) and the intervention set family \\(\boldsymbol{\mathcal{I}}=\{\mathcal{I}_1,\dots,\mathcal{I}_t\}\\), if and only if for every \\(S_\ell\in\mathcal{S}\\), there exists an intervention set \\(\mathcal{I}_k\in\boldsymbol{\mathcal{I}}\\) such that (i) \\(\mathcal{I}_k\cap S_\ell=\emptyset\\), and (ii) there is no hedge formed for \\(S_\ell\\) in \\(\mathcal{G}[V\setminus\mathcal{I}_k]\\).*

</div>

Note that there is no hedge formed for \\(S_\ell\\) in \\(\mathcal{G}[V\setminus\mathcal{I}_k]\\) if and only if \\(\mathcal{I}_k\\) *hits* every hedge of \\(S_\ell\\) (i.e., for any hedge \\(W\in \mathcal{H}_\mathcal{G}(S_\ell)\\), \\(\mathcal{I}_k\cap W\neq\emptyset\\)). For ease of presentation, we will use \\(\mathcal{I}_k\overset{\mathrm{id}}{\longrightarrow}S_\ell\\) to denote that \\(\mathcal{I}_k\cap S_\ell=\emptyset\\) and \\(\mathcal{I}_k\\) hits every hedge formed for \\(S_\ell\\). For example, given the graph in  
effig:drug_ex(d) and with \\(\boldsymbol{\mathcal{S}} = \{S_1, S_2\},\\) an intervention set family that hits every hedge is \\(\boldsymbol{\mathcal{I}} = \{\{S_2\}, \{X_3\}\}.\\)

**Minimum-cost intervention for causal effect identification (MCID) problem.** Let \\(C:\!V\!\!\to\!\mathbb{R}^{\geq0}\cup\!\{+\infty\}\\) be a known function[^3] indicating the cost of intervening on each vertex \\(v\in V\\). An infinite cost is assigned to variables where an intervention is not feasible. Given \\(\mathcal{G}\\) and disjoint sets \\(X,Y\subseteq V\\), our objective is to find a set family \\(\boldsymbol{\mathcal{I}}^*\\) with minimum cost such that \\(\mathbb{P}_\mathrm{x}(Y)\\) is identifiable given \\(\boldsymbol{\mathcal{I}}^*\\); that is, for every district \\(S_\ell\\) of \\(S\\), there exists \\(\mathcal{I}_k\\) such that \\(\mathcal{I}_k\overset{\mathrm{id}}{\longrightarrow} S_\ell\\). Since every \\(\mathcal{I}\in\boldsymbol{\mathcal{I}}\\) is a subset of \\(V\\), the space of such set families is the power set of the power set of \\(V\\).

To formalize the MCID problem, we first write the cost of a set family \\(\boldsymbol{\mathcal{I}}\\) as \\(C(\boldsymbol{\mathcal{I}}):=\sum_{\mathcal{I}\in\boldsymbol{\mathcal{I}}}\sum_{v\in\mathcal{I}}C(v),\\) where with a slight abuse of notation, we denoted the cost of \\(\boldsymbol{\mathcal{I}}\\) by \\(C(\boldsymbol{\mathcal{I}})\\). The MCID problem then can be formalized as follows. \\[\label{eq:opt}
    \boldsymbol{\mathcal{I}}^*\in\mathop{\mathrm{argmin}}_{\boldsymbol{\mathcal{I}}\in 2^{2^V}}C(\boldsymbol{\mathcal{I}}) \quad\mathbf{s.t.}\quad\forall\ S_\ell\in\boldsymbol{\mathcal{S}}:(\exists\ \mathcal{I}_k
    \in\boldsymbol{\mathcal{I}}: \mathcal{I}_k\overset{\mathrm{id}}{\longrightarrow} S_\ell),\\] where \\(\boldsymbol{\mathcal{S}}=\{S_1,\dots, S_r\}\\) is the set of maximal districts of \\(S=\mathrm{Anc}_{V\setminus X}(Y)\\), and \\(2^{2^V}\\) represents the power set of the power set of \\(V\\). In the special case where \\(S\\) comprises a single district, the MCID problem can be presented in a simpler way.

<div id="prp:single_dist" class="proposition" markdown="1">

**Proposition 2** (`\citealp{akbari-2022}`{=latex}). *If \\(S=\mathrm{Anc}_{V\setminus X}(Y)\\) comprises a single maximal district \\(\mathcal{S}=\{S_1=S\}\\), then the optimization in <a href="#eq:opt" data-reference-type="eqref" data-reference="eq:opt">[eq:opt]</a> is equivalent to the following optimization: \\[\label{eq:opt2}
        \mathcal{I}^*\in\mathop{\mathrm{argmin}}_{\mathcal{I}\in 2^{V\setminus S}}C(\mathcal{I}) \quad\mathbf{s.t.}\quad\forall\ W\in H_\mathcal{G}(S)
        :\:\mathcal{I}\cap W\neq\emptyset.\\] That is, the problem reduces to finding the minimum-cost set that ‘hits’ every hedge formed for \\(S\\).*

</div>

Recall example  
efex:drug, we were interested in finding the least costly proxy experiment to identify the effect of \\(X_2\\) and \\(X_3\\) on \\(Y\\). By  
efprp:single_dist, this problem is equivalent to finding an intervention set with the least cost (i.e., a set of proxy experiments) that hits every hedge of \\(S = \{Y,W\}\\) in the transformed graph (  
effig:drug_ex(b)). If \\(\mathcal{C}(X_1) < \mathcal{C}(X_3)\\), then the optimal solution would be \\(\mathcal{I}^* = \{X_1\}\\).

In the remainder of the paper, we consider the problem of identification of \\(\mathbb{P}_X(Y)\\) for a given pair \\((X,Y)\\), and with \\(S\\) defined as \\(S=\mathrm{Anc}_{V\setminus X}(Y)\\), unless otherwise stated. We will first consider the case where \\(S\\) comprises a single district, and then generalize our findings to multiple districts.

# Reformulations of the min-cost intervention problem [sec:reformulations]

In the previous section, we delineated the MCID problem as a discrete optimization problem. This problem, cast as  
efeq:opt, necessitates search within a doubly exponential space, which is computationally intractable. Algorithm 2 of `\citep{akbari-2022}`{=latex} is an algorithm that conducts this search and eventually finds the optimal solution. However, even when \\(S\\) comprises a single district, this algorithm requires, in the worst case, exponentially many calls to a subroutine which solves the NP-complete minimum hitting set problem on exponentially many input sets, hence resulting in a doubly exponential complexity. More specifically, their algorithm attempts to find a set of *minimal* hedges, where minimal indicates a hedge that contains no other hedges, and solves the minimum hitting set problem on them. However, there can be exponentially many minimal hedges, as shown for example in  
effig:drug_ex(c). Letting \\(m = n/2\\), then any set that contains one vertex from each level (i.e., directed distance from \\(S\\)) is a minimal hedge, of which there are \\(\mathcal{O}(2^{n/2}).\\)

Furthermore, the computational complexity of Algorithm 2 of `\citet{akbari-2022}`{=latex} grows super-exponentially in the number of districts of \\(S\\). This is due to the necessity of exhaustively enumerating every possible partitioning of these districts and executing their algorithm once for each partitioning.

In this section, we reformulate the MCID problem as a weighted partially maximum satisfiability (WPMAX-SAT) problem `\citep{pmax-sat}`{=latex}, and an integer linear programming (ILP) problem. In  
efapp:reform, we also present reformulations as a submodular maximization problem and a reinforcement learning problem. The advantage of these new formulations is two-fold: (i) compared to Algorithm 2 of `\citep{akbari-2022}`{=latex}, we state the problem as a *single* instance of another problem for which a range of well-studied solvers exist, and (ii) these formulations allow us to propose algorithms with computational complexity that is quadratic in the number of districts of \\(S\\). We will see how these advantages translate to drastic performance gains in  
efsec:experiments.

## Min-cost intervention as a WPMAX-SAT problem [sec:reform-maxsat]

We begin with constructing a 3-SAT formula \\(F\\) that is satisfiable if and only if the given query \\(\mathbb{P}_X(Y)\\) is identifiable. To this end, we define \\(m+2\\) variables \\(\{x_{i,j}\}_{j=0}^{m+1}\\) for each vertex \\(v_i\in V\\), where \\(m=\vert\mathcal{H}_\mathcal{G}(S)\setminus S\vert\\) is the cardinality of the hedge hull of \\(S\\), excluding \\(S\\). Intuitively, \\(x_{i,j}\\) is going to indicate whether or not vertex \\(v_i\\) is reachable from \\(S\\) after \\(j\\) iterations of alternating depth-first-searches on directed and bidirected edges. This is in line with the workings of  
efalg:pruning for finding the hedge hull of \\(S\\). In particular, if a vertex \\(v_i\\) is reachable after \\(m+1\\) iterations, that is, \\(x_{i,m+1}=1\\), then \\(v_i\\) is a member of the hedge hull of \\(S\\). The query of interest is identifiable if and only if \\(\mathcal{H}_\mathcal{G}(S)=S\\), that is, the hedge hull of \\(S\\) contains no other vertices. Therefore, we ensure that the formula \\(F\\) is satisfiable if and only if \\(x_{i,m+1}=0\\) for every \\(v_i\notin S\\). The formal procedure for constructing this formula is as follows.

#### SAT Construction Procedure.

Suppose a causal ADMG \\(\mathcal{G}= \langle V,\overrightarrow{E}, \overleftrightarrow{E}\rangle\\) and a set \\(S \subset V\\) are given, where \\(S\\) is a district in \\(\mathcal{G}\\). Suppose \\(\mathcal{H}_\mathcal{G}(S)=\{v_1,\dots,v_n\}\\) is the hedge hull of \\(S\\) in \\(\mathcal{G}\\), where without loss of generality, \\(S = \{v_{m+1},\dots v_n\}\\), and \\(\{v_1,\dots v_m\}\cap S=\emptyset\\). We will construct a corresponding boolean expression in conjunctive normal form (CNF) using variables \\(\{x_{i,j}\}\\) for \\(i\in\{1,\dots,m\}\\) and \\(j\in\{0,\dots,m+1\}\\). For ease of presentation, we also define \\(x_{i,j}=1\\) for all \\(i\in\{m+1,\dots, n\}\\), \\(j\in\{0,\dots, m+1\}\\). The construction is carried out in \\(m+2\\) steps, where in each step, we conjoin new clauses to the previous formula using ‘and’. The procedure is as follows:

- For odd \\(j\! \in\! \{1, \ldots, m\!+\!1\}\\), for each directed edge \\((v_i,v_\ell)\!\in\!\overrightarrow{E}\\), add \\((\neg x_{i,j-1} \lor x_{i,j}\! \lor\! \neg x_{\ell,j})\\) to \\(F\\).

- For even \\(j \in \{1, \ldots, m+1\}\\), for each bidirected edge \\(\{v_i,v_\ell\}\in\overleftrightarrow{E}\\), add both clauses \\((\neg x_{i,j-1} \lor x_{i,j} \lor \neg x_{\ell,j})\\) and \\((\neg x_{\ell,j-1} \lor x_{\ell,j} \lor \neg x_{i,j})\\) to \\(F\\).

- Finally, at step \\(m+2\\), add clauses \\(\neg x_{i,m+1}\\) to the expression \\(F\\) for every \\(i\in\{1,\dots,m\}\\).

<div class="restatable" markdown="1">

theoremsat <span id="thm:sat" label="thm:sat"></span> The 3-SAT formula \\(F\\) constructed by the procedure above given \\(\mathcal{G}\\) and \\(S\\) has a satisfying solution \\(\{x_{i,j}^*\}\\) where \\(x_{i,0}^*\!=\!0\\) for \\(i\!\in\!\mathcal{I}\!\subseteq\!\{1,\dots,m\}\\) and \\(x_{i,0}^*\!=\!1\\) for \\(i\!\in\!\{1,\dots,m\}\!\setminus\!\mathcal{I}\\) if and only if \\(\mathcal{I}\\) intersects every hedge formed for \\(S\\) in \\(\mathcal{G}\\); i.e., \\(\mathcal{I}\\) is a feasible solution to the optimization in  
efeq:opt2.

</div>

The proofs of all our results appear in  
efapp:proofs. The first corollary of  
efthm:sat is that the SAT formula is always satisfiable, for instance by setting \\(x_{i,0}^*=0\\) for every \\(i\in\{1,\dots,m\}\\). The second (and more important) corollary is that the optimal solution to  
efeq:opt2 corresponds to the satisfying assignment for the SAT formula \\(F\\) that minimizes \\[\label{eq:objective}
    \sum_{i=1}^m(1-x_{i,0}^*)C(v_i).\\] This suggests that the problem in  
efeq:opt2 can be reformulated as a weighted partial MAX-SAT (WPMAX-SAT) problem. **WPMAX-SAT** is a generalization of the MAX-SAT problem, where the clauses are partitioned into *hard* and *soft* clauses, and each soft clause is assigned a weight. The goal is to maximize the aggregate weight of the satisfied soft clauses while satisfying all of the hard ones.

To construct the WPMAX-SAT instance, we simply define all clauses in \\(F\\) as hard constraints, and add a soft clause \\(x_{i,0}\\) with weight \\(C(v_i)\\) for every \\(i\in\{1,\dots,m\}\\). The former ensures that the assignment corresponds to a feasible solution of  
efeq:opt2, while the latter ensures that the objective in  
efeq:objective is minimized – which, consequently, minimizes the cost of the corresponding intervention.

#### Multiple districts.

The formulation above was presented for the case where \\(S\\) is a single district. In the more general case where \\(S\\) has multiple maximal districts, we can extend our formulation to solve the general problem of  
efeq:opt instead. To this end, we will use the following lemma.

<div class="restatable" markdown="1">

lemmalemmultiple<span id="lem:multiple" label="lem:multiple"></span> Let \\(\boldsymbol{\mathcal{S}}=\{S_1,\dots, S_r\}\\) be the set of maximal districts of \\(S\\), where \\(S=\mathrm{Anc}_{V\setminus X}(Y)\\). There exists an intervention set family \\(\boldsymbol{\mathcal{I}}^*\\) of size \\(\vert\boldsymbol{\mathcal{S}}\vert=r\\) that is optimal for identifying \\(\mathbb{P}_X(Y)\\).

</div>

Based on  
eflem:multiple, we can assume w.l.o.g. that the optimizer of  
efeq:opt contains exactly \\(r\\) intervention sets \\(\mathcal{I}_1,\dots,\mathcal{I}_r\\). We will modify the SAT construction procedure described in the previous section to allow for multiple districts as follows. For any district \\(S_\ell\\), we will construct \\(r\\) copies of the SAT expression, one corresponding to each intervention set \\(\mathcal{I}_k\\), \\(k\in\{1,\dots,r\}\\). Each copy is built on new sets of variables indexed by \\((k,\ell)\\), except the variables with index \\(j=0\\), which are common across districts. We introduce variables \\(\{z_{k,\ell}\}_{k,\ell=1}^r\\), which will serve as indicators for whether \\(\mathcal{I}_k\\) hits all the hedges formed for \\(S_\ell\\). We relax every clause corresponding to the \\(k\\)-th copy by conjoining a \\(\lnot z_{k,\ell}\\) literal with an ‘or.’ Intuitively, this is because it suffices to hit the hedges formed for \\(S_\ell\\) with some \\(\mathcal{I}_k\\). Additionally, we add the clauses \\((z_{1,\ell}\lor\dots\lor z_{r,\ell})\\) for any \\(\ell\in\{1,\dots,r\}\\) to ensure that for every district, there is at least one intervention set that hits every hedge. This modified procedure, detailed in  
efalg:sat, appears in  
efapp:multipledist. The following result generalizes  
efthm:sat.

<div class="restatable" markdown="1">

theoremsatgen <span id="thm:satgen" label="thm:satgen"></span> Suppose \\(\mathcal{G}\\), a set of its vertices \\(S\\) with maximal districts \\(\boldsymbol{\mathcal{S}}=\{S_1,\dots,S_r\}\\), and an intervention set family \\(\boldsymbol{\mathcal{I}}=\{\mathcal{I}_1,\dots,\mathcal{I}_r\}\\) are given. Define \\(m_\ell = \vert\mathcal{H}_\mathcal{G}(S_\ell)\setminus S_\ell\vert\\), i.e., the cardinality of the hedge hull of \\(S_\ell\\) excluding \\(S_\ell\\) itself. The SAT formula \\(F\\) constructed by  
efalg:sat has a satisfying solution \\(\{x_{i,0,k}^*\}\cup\{x_{i,j,k,\ell}^*\}\cup\{z_{k,\ell}^*\}\\) where for every \\(\ell\in\{1,\dots,r\}\\), there exists \\(k\in\{1,\dots,r\}\\) such that (i) \\(z_{k,\ell}^*=1\\), (ii) \\(x_{i,0,k}^*=0\\) for every \\(i\in\mathcal{I}_k\\), and (iii) \\(x_{i,0,k}^*=1\\) for every \\(i\in \{1,\dots,m_\ell\}\setminus\mathcal{I}_k\\), if and only if \\(\boldsymbol{\mathcal{I}}\\) is a feasible solution to optimization of  
efeq:opt.

</div>

Constructing the corresponding WPMAX-SAT instance follows the same steps as the case for a single district, except that the soft clauses are of the form \\(( x_{i,0,k}\lor\lnot z_{k,\ell})\\) with weight \\(C(v_i)\\) for every \\(i\in\{1,\dots,m_\ell\}\\) and \\(k\in\{1,\dots, r\}\\).

<div class="remark" markdown="1">

**Remark 3**. *The SAT construction of  
efalg:sat is advantageous because its complexity grows quadratically with the number of districts of \\(S\\) in the worst case. In contrast, the runtime of algorithm proposed by `\citet{akbari-2022}`{=latex}, when \\(S\\) consists of multiple districts, is super-exponential in the number of districts, because they need to execute their single-district algorithm at least as many times as the number of partitions of the set \\(\{1,\dots,r\}\\).*

</div>

#### Min-cost intervention as an ILP problem.

The WPMAX-SAT formulation of  
efsec:reform-maxsat paves the way for a straightforward formulation of an integer linear program (ILP) for the MCID problem. ILP allows for straightforward integration of various constraints and objectives, enabling flexible modeling of potential extra constraints. Moreover, there exist efficient and scalable solvers for ILP `\citep{gearhart2013comparison, gurobi}`{=latex}. To construct the ILP instance for the MCID problem, it suffices to represent every clause in the boolean expression \\(F\\) of  
efalg:sat as a linear inequality. For example, clauses of the form \\((\lnot a \lor b \lor \lnot c)\\) is rewritten as \\((1-a) + b + (1-c) \geq 1\\). The soft constraints may be rewritten as a sum to maximize over, given by  
efeq:objective.

# Minimum-cost intervention design for adjustment criterion [sec:adjustment]

A special case of identifying interventional distributions is identification through *adjusting* for confounders. A set \\(Z\subseteq V\\) is a valid adjustment set for \\(\mathbb{P}_X(Y)\\) if \\(\mathbb{P}_X(Y)\\) is identified as \\[\label{eq:adjust}
    \mathbb{P}_X(Y) = \mathbb{E}_{\mathbb{P}}[\mathbb{P}(Y\mid X, Z)],\\] where the expectation w.r.t. \\(\mathbb{P}(Z)\\). Adjustment sets have received extensive attention in the literature because of the straightforward form of the identification formula (Eq. <a href="#eq:adjust" data-reference-type="ref" data-reference="eq:adjust">[eq:adjust]</a>) and the intuitive interpretation: \\(Z\\) is the set of confounders that we need to *adjust for* to identify the effect of interest. The simple form of Eq. <a href="#eq:adjust" data-reference-type="eqref" data-reference="eq:adjust">[eq:adjust]</a> has the added desirable property that its sample efficiency and asymptotic behavior are easy to analyze `\citep{witte2020efficient,rotnitzky2020efficient,henckel2022graphical}`{=latex}. A complete graphical criterion for adjustment sets was given by `\citet{shpitser2010validity}`{=latex}. As an example, when all parents of \\(X\\) (i.e., \\(\mathrm{Pa}(X)\\)) are observable, they form a valid adjustment set. However, in the presence of unmeasured confounding, no valid adjustment sets may exist. Below, we generalize the notion of adjustment sets to the interventional setting.

<div id="def:genadj" class="definition" markdown="1">

**Definition 4** (Generalized adjustment). *We say \\(Z\subseteq V\\) is a generalized adjustment set for \\(\mathbb{P}_X(Y)\\) under intervention \\(\mathcal{I}\\) if \\(\mathbb{P}_X(Y)\\) is identified as \\(% 
    % )\label{eq:genadjust}
        \mathbb{P}_X(Y) = \mathbb{E}_{\mathbb{P}_\mathcal{I}}[\mathbb{P}_\mathcal{I}(Y\mid X, Z)],\\) where \\(\mathbb{P}_\mathcal{I}(\cdot)\\) represents the distribution after intervening on \\(\mathcal{I}\!\subseteq\!\! V\\) and the expectation is w.r.t. \\(\mathbb{P}_\mathcal{I}(Z)\\).*

</div>

Note that unlike the classic adjustment, the generalized adjustment is always feasible – a trivial generalized adjustment can be formed by choosing \\(\mathcal{I}=X\\) and \\(Z=\emptyset\\).

Equipped with Definition <a href="#def:genadj" data-reference-type="ref" data-reference="def:genadj">4</a>, we can define a problem closely linked to  
efeq:opt2, but with a (possibly) narrower set of solutions, which can be defined as follows: find the minimum-cost intervention \\(\mathcal{I}\\) such that a generalized adjustment exists for \\(\mathbb{P}_X(Y)\\) under \\(\mathcal{I}\\): \\[\label{eq:opt3}
    \mathcal{I}^* = \mathop{\mathrm{argmin}}_{\mathcal{I}\in 2^V}C(\mathcal{I})\quad\textbf{s.t.}\quad\exists\ Z\subseteq V: \mathbb{P}_X(Y) = \mathbb{E}_{\mathbb{P}_\mathcal{I}}[\mathbb{P}_\mathcal{I}(Y\mid X, Z)].\\]

**Observation.** The existence of a valid (generalized) adjustment set ensures the identifiability of \\(\mathbb{P}_X(Y)\\). As such, any feasible solution to the optimization above is also a feasible solution to  
efeq:opt2.  
efeq:opt3 is not only a problem that deserves attention in its own right, but also serves as a proxy for our initial problem (Eq. <a href="#eq:opt2" data-reference-type="ref" data-reference="eq:opt2">[eq:opt2]</a>).

To proceed, we need the following definitions. Given an ADMG \\(\mathcal{G}=\langle V, \overrightarrow{E}, \overleftrightarrow{E}\rangle\\), let \\(\mathcal{G}^d=\langle V^d, \overrightarrow{E}^d, \emptyset\rangle\\) be the ADMG resulting from replacing every bidirected edge \\(e=\{x,y\}\in \overleftrightarrow{E}\\) by a vertex \\(e\\) and two directed edges \\((e,x),(e,y)\\). In particular, \\(V^d = V\cup \overleftrightarrow{E}\\), and \\(\overrightarrow{E}^d = \overrightarrow{E}\cup\{(e,x):e\in\overleftrightarrow{E}, x\in e\}\\). Note that \\(\mathcal{G}^d\\) is a directed acyclic graph (DAG). The *moralized graph* of \\(\mathcal{G}\\), denoted by \\(\mathcal{G}^m\\), is the undirected graph constructed by moralizing \\(\mathcal{G}^d\\) as follows: The set of vertices of \\(\mathcal{G}^m\\) is \\(V^d\\). Each pair of vertices \\(x,y\in V^d\\) are connected by an (undirected) edge if either (i) \\((x,y)\in \overrightarrow{E}^d\\), or (ii) \\(\exists z\in V^d\\) such that \\(\{(x,z), (y,z)\}\subseteq\overrightarrow{E}^d\\).

Throughout this section, we assume without loss of generality that \\(X\\) is minimal in the following sense: there exists no proper subset \\(X_1\subsetneq X\\) such that \\(\mathbb{P}_{X}(Y) = \mathbb{P}_{X_1}(Y)\\) everywhere[^4]. Otherwise, we apply the third rule of do calculus `\citep{pearl2009causality}`{=latex} as many times as possible to make \\(X\\) minimal. We also assume w.l.o.g. that \\(V=\mathrm{Anc}(X\cup Y)\\) as other vertices are irrelevant for our purposes `\citep{lee-2020}`{=latex}. We will utilize the following graphical criterion for generalized adjustment.

<div class="restatable" markdown="1">

lemmalemvertexcut<span id="lem:vtxcut" label="lem:vtxcut"></span> Let \\(X,Y\\) be two disjoint sets of vertices in \\(\mathcal{G}\\) such that \\(X\\) is minimal as defined above. Set \\(Z\subseteq V\\) is a generalized adjustment set for \\(\mathbb{P}_X(Y)\\) under intervention \\(\mathcal{I}\\) if (i) \\(Z\subseteq\mathrm{Anc}(S)\\), and (ii) \\(Z\\) is a vertex cut[^5] between \\(S\\) and \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^m\\), where \\(S=\mathrm{Anc}_{V\setminus X}(Y)\\), and \\(\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}}\\) is the ADMG resulting from omitting all edges incoming to \\(\mathcal{I}\\) and all edges outgoing of \\(\mathrm{Pa}(S)\\).

</div>

Based on the graphical criterion of  
eflem:vtxcut, we present the following *polynomial-time*[^6] algorithm for finding an intervention set that allows for identification of the query of interest in the form of a (generalized) adjustment. This algorithm will find the intervention set \\(\mathcal{I}\\) and the corresponding generalized adjustment set \\(Z\\) simultaneously. We begin by making \\(X\\) minimal in the sense of applicability of rule 3 of do calculus. Then we omit all edges going out of \\(\mathrm{Pa}(S)\\), and construct the graph \\((\mathcal{G}_{\underline{\mathrm{Pa}(S)}})^d=\langle V^d, \overrightarrow{E}^d,\emptyset\rangle\\) as defined above – by replacing bidirected edges with vertices representing unobserved confounding. Finally, we construct an (undirected) vertex cut network \\(\mathcal{G}^{vc}=\langle V^{vc}, E^{vc}\rangle\\) as follows. Each vertex \\(v\in V^d\\) is represented by two connected vertices \\(v_1,v_2\\) in \\(\mathcal{G}^{vc}\\). If \\(v\in V\\), then \\(v_1\\) has a cost of zero, and \\(v_2\\) has cost \\(C(v)\\). Otherwise, both \\(v_1\\) and \\(v_2\\) have infinite costs. Intuitively, choosing \\(v_1\\) will correspond to including \\(v\\) in the adjustment set, whereas choosing \\(v_2\\) in the cut would imply intervention on \\(v\\). We connect \\(v_2\\) to all vertices corresponding to \\(\mathrm{Pa}(v)\\) with index \\(1\\), i.e., \\(\{w_1:(w,v)\in\overrightarrow{E}^d\}\\). This serves two purposes: (i) if \\(v_2\\) is included in the cut (corresponding to an intervention on \\(v\\)), all connections between \\(v\\) and its parents are broken, and (ii) when \\(v_2\\) is not included in the cut (corresponding to no intervention on \\(v\\)), \\(v_2\\) connects the parents of \\(v\\) to each other, completing the necessary moralization process. We solve for the minimum vertex cut between vertices with index \\(1\\) corresponding to \\(S\\) and \\(\mathrm{Pa}(S)\\).  
efalg:adj summarizes this approach. In the solution set \\(J\\), the vertices with index \\(2\\) represent the vertices where an intervention is required, while those with index \\(1\\) represent the generalized adjustment set under this intervention.

<div class="algorithm*" markdown="1">

<div class="algorithmic" markdown="1">

ALGORITHM BLOCK (caption below)

  
**Procedure** MinCostGenAdjustment\\(X, Y, \mathcal{G}=\langle V, \overrightarrow{E}, \overleftrightarrow{E}\rangle, \{C(v):v\in V\}\\)  
**While** \\(\exists x\in X\\) s.t. \\(x\\) is d-sep from \\(Y\\) given \\(X\setminus\{x\}\\) in \\(\mathcal{G}_{\overline{X}}\\) \# comment: Make \\(X\\) minimal  
\\(X\gets X\setminus\{x\}\\)  
EndWhile  
\\(S\gets\mathrm{Anc}_{V\setminus X}(Y)\\)  
\\(\overrightarrow{E}\gets\overrightarrow{E}\setminus\{(x,s):x\in\mathrm{Pa}(S)\}\\) \# comment: \\(\mathcal{G}_{\underline{\mathrm{Pa}(S)}}\\)  
\\(V^d\gets V\cup\overleftrightarrow{E}\\), \\(\quad\overrightarrow{E}^d\gets \overrightarrow{E}\cup\{(e,v):e\in\overleftrightarrow{E}, v\in e\}\\) \# comment: Construct \\((\mathcal{G}_{\underline{\mathrm{Pa}(S)}})^d\\)  
\\(V^{vc}\gets \cup_{v\in V^d}\{v_1,v_2\},\quad E^{vc}\gets \big\{\{v_1, v_2\}:v\in V^d\big\}\cup\big\{\{w_1, v_2\}: (w,v)\in\overrightarrow{E}^d\big\}\\)  
Construct a minimum vertex cut instance on the network \\(\mathcal{G}^{vc}=\langle V^{vc}, E^{vc}\rangle\\), with costs \\(0\\) for any \\(v_1\\) where \\(v\in V\\), \\(C(v)\\) for any \\(v_2\\) where \\(v\in V\\), and \\(\infty\\) for any other vertex  
\\(J\gets\\) the minimum vertex cut between \\(\{x_1:x\in\mathrm{Pa}(S)\}\\) and \\(\{s_1:s\in S\}\\)  
\\(\mathcal{I}\gets \{v:v_2\in J\},\quad Z\gets \{v:v_1\in J\}\\)  
  
Return \\((\mathcal{I}, Z)\\)  
EndProcedure

</div>

</div>

<div class="restatable" markdown="1">

theoremthmalgadj<span id="thm:algadj" label="thm:algadj"></span> Let \\((\mathcal{I}, Z)\\) be the output returned by  
efalg:adj for the query \\(\mathbb{P}_X(Y)\\). Then,  
- \\(Z\\) is a generalized adjustment set for \\(\mathbb{P}_X(Y)\\) under intervention \\(\mathcal{I}\\).  
- \\(\mathcal{I}\\) is the minimum-cost intervention for which there exists a generalized adjustment set based on the graphical criterion of  
eflem:vtxcut.

</div>

<div id="rem:heuristic" class="remark" markdown="1">

**Remark 4**. *  
efalg:adj enforces identification based on (generalized) adjustment for \\(\mathbb{P}_X(Y)\\). As discussed above, this algorithm can be utilized as a heuristic approach to solve the MCID problem in <a href="#eq:opt2" data-reference-type="eqref" data-reference="eq:opt2">[eq:opt2]</a>. In this case, one can run the algorithm on the hedge hull of \\(S\\) rather than the whole graph. We prove in Appendix <a href="#app:proofs" data-reference-type="ref" data-reference="app:proofs">9</a> that the cost of this approach is always at most as high as heuristic algorithm 1 proposed by `\citet{akbari-2022}`{=latex}, and is often in practice lower, as verified by our experiments.*

</div>

# Experiments [sec:experiments]

In this section, we present numerical experiments that showcase the empirical performance and time efficiency of our proposed exact and heuristic algorithms. A comprehensive set of numerical experiments analyzing the impact of various problem parameters on the performance of these algorithms, along with the complete implementation details, is provided in  
efapp:exp. We first compare the time efficiency of our exact algorithms: WPMAX-SAT and ILP, with the exact algorithm of `\citet{akbari-2022}`{=latex}. Then, we present results pertaining to performance of our heuristic algorithm. All experiments, coded in Python, were conducted on a machine equipped two Intel Xeon E5-2680 v3 CPUs, 256GB of RAM, and running Ubuntu 20.04.3 LTS.

<figure id="fig:exact">
<img src="./figures/exact_time.png"" />
<figcaption>Average time taken by Algorithm 2 of <span class="citation" data-cites="akbari-2022"></span> (MHS), ILP, and WPMAX-SAT to solve one graph versus (a) the number of vertices in the graph and (b) the number of districts of <span class="math inline"><em>S</em></span>.</figcaption>
</figure>

<figure id="fig:heur">
<img src="./figures/heur_cost.png"" />
<figcaption>Average normalized cost of the heuristic algorithms <span class="math inline"><em>H</em><sub>1</sub></span> and <span class="math inline"><em>H</em><sub>2</sub></span> of <span class="citation" data-cites="akbari-2022"></span> and<br />
ef<span>alg:adj</span> versus the number of vertices in the graph.</figcaption>
</figure>

**Results on exact algorithms.** We compare the performance of the WPMAX-SAT formulation, the ILP formulation, and Algorithm 2 of `\citet{akbari-2022}`{=latex}, called Minimal Hedge Solver (MHS) from hereon. We used the RC2 algorithm `\citep{rc2-maxsat}`{=latex}, and the Gurobi solver `\citep{gurobi}`{=latex}, to solve the WPMAX-SAT problem, and the ILP, respectively. We ran each algorithm for solving the MCID problem on \\(100\\) randomly generated Erdos-Renyi `\citep{Erdos:1960}`{=latex} ADMG graphs with directed and bidirected edge probabilities ranging from \\(0.01\\) to \\(1.00\\), in increments of \\(0.01\\). We performed two sets of simulations: for single-district and multiple-district settings, respectively. In the single-district case, we varied \\(n\\), the number of vertices, from \\(20\\) to \\(100\\), while in the multiple-district case, we fixed \\(n=20\\) and varied the number of districts from \\(1\\) to \\(9\\).

We plot the average time taken to solve each graph versus the number of vertices (single-district) in  
effig:exact(a) and versus the number of districts (\\(n=20\\)) in  
effig:exact(b). The error bands in our figures represent 99% confidence intervals. Focusing on the single-district plot, we observe that both of our algorithms are faster than MHS of `\citet{akbari-2022}`{=latex} for all graph sizes. More specifically, ILP is on average one to two orders of magnitude faster than MHS, while WPMAX-SAT is on average four to five orders of magnitude faster. All three algorithms exhibit exponential growth in time complexity with the number of vertices, which is expected as the problem is NP-hard, but WPMAX-SAT grows at a much slower rate than the other two algorithms. This is likely due to RC2’s ability of exploiting the structure of the WPMAX-SAT problem to reduce the search space efficiently. In the multiple-district case, we observe that the time complexity of both WPMAX-SAT and ILP grows polynomially with the number of districts, while the time complexity of MHS grows exponentially. This is consistent with theory, as MHS iterates over all partitions of the set of districts, which grows exponentially with the number of districts.

**Results on inexact algorithms.** We compared  
efalg:adj, our proposed heuristic, with the two best performing heuristic algorithms in `\citet{akbari-2022}`{=latex}, \\(H_1\\) and \\(H_2\\). We ran each algorithm on \\(500\\) randomly generated Erdos-Renyi ADMG graphs with directed and bidirected edge probabilities in \\(\{0.1,0.5\}\\), with \\(n\\) ranging from \\(10\\) to \\(200\\). We randomly sampled the cost of each vertex from a discrete uniform distribution on \\([1,n]\\). In  
effig:heur, we plot the normalized cost of each algorithm, computed by dividing the cost of the algorithm by the cost of the optimal solution, provided by WPMAX-SAT. Observe that  
efalg:adj consistently outperforms \\(H_1\\) and \\(H_2\\) for all graph sizes.

# Conclusion [sec:conclusion]

We presented novel formulations and efficient algorithms for the MCID problem, offering substantial improvements over existing methods. Our work on designing minimum-cost experiments for obtaining valid adjustment sets demonstrates both practical and theoretical advancements. We highlighted the superior performance of our proposed methods through extensive numerical experiments. We envision designing efficient approximation algorithms for MCID as future work.

# References [references]

<div class="thebibliography" markdown="1">

Sina Akbari, Jalal Etesami, and Negar Kiyavash Minimum cost intervention design for causal effect identification In *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pages 258–289. PMLR, 17–23 Jul 2022. URL <https://proceedings.mlr.press/v162/akbari22a.html>. **Abstract:** Pearl’s do calculus is a complete axiomatic approach to learn the identiﬁable causal effects from observational data. When such an effect is not identiﬁable, it is necessary to perform a collection of often costly interventions in the system to learn the causal effect. In this work, we consider the problem of designing the collection of interventions with the minimum cost to identify the desired effect. First, we prove that this problem is NP-hard and subsequently propose an algorithm that can either ﬁnd the optimal solution or a logarithmic-factor approximation of it. This is done by establishing a connection between our problem and the minimum hitting set problem. Additionally, we propose several polynomial time heuristic algorithms to tackle the computational complexity of the problem. Although these algorithms could potentially stumble on sub-optimal solutions, our simulations show that they achieve small regrets on random graphs. (@akbari-2022)

Judea Pearl *Causality* Cambridge university press, 2009. **Abstract:** Written by one of the preeminent researchers in the field, this book provides a comprehensive exposition of modern analysis of causation. It shows how causality has grown from a nebulous concept into a mathematical theory with significant applications in the fields of statistics, artificial intelligence, economics, philosophy, cognitive science, and the health and social sciences. Judea Pearl presents and unifies the probabilistic, manipulative, counterfactual, and structural approaches to causation and devises simple mathematical tools for studying the relationships between causal connections and statistical associations. Cited in more than 2,100 scientific publications, it continues to liberate scientists from the traditional molds of statistical thinking. In this revised edition, Judea Pearl elucidates thorny issues, answers readers’ questions, and offers a panoramic view of recent advances in this field of research. Causality will be of interest to students and professionals in a wide variety of fields. Dr Judea Pearl has received the 2011 Rumelhart Prize for his leading research in Artificial Intelligence (AI) and systems from The Cognitive Science Society. (@pearl2009causality)

Miguel A Hernán and James M Robins Estimating causal effects from epidemiological data *Journal of Epidemiology & Community Health*, 60 (7): 578–586, 2006. **Abstract:** In ideal randomised experiments, association is causation: association measures can be interpreted as effect measures because randomisation ensures that the exposed and the unexposed are exchangeable. On the other hand, in observational studies, association is not generally causation: association measures cannot be interpreted as effect measures because the exposed and the unexposed are not generally exchangeable. However, observational research is often the only alternative for causal inference. This article reviews a condition that permits the estimation of causal effects from observational data, and two methods – standardisation and inverse probability weighting – to estimate population causal effects under that condition. For simplicity, the main description is restricted to dichotomous variables and assumes that no random error attributable to sampling variability exists. The appendix provides a generalisation of inverse probability weighting. (@hernan2006estimating)

Sanghack Lee, Juan D Correa, and Elias Bareinboim General identifiability with arbitrary surrogate experiments In *Uncertainty in artificial intelligence*, pages 389–398. PMLR, 2020. **Abstract:** We study the problem of causal identiﬁcation from an arbitrary collection of observational and experimental distributions, and substantive knowledge about the phenomenon under in- vestigation, which usually comes in the form of a causal graph. We call this problem g- identiﬁability , or gID for short. The gID set- ting encompasses two well-known problems in causal inference, namely, identiﬁability \[Pearl, 1995\] and z-identiﬁability \[Bareinboim and Pearl, 2012\] — the former assumes that an ob- servational distribution is necessarily available, and no experiments can be performed, condi- tions that are both relaxed in the gID setting; the latter assumes that allcombinations of exper- iments are available, i.e., the power set of the experimental set Z, which gID does not require a priori. In this paper, we introduce a general strategy to prove non-gID based on hedgelets andthickets , which leads to a necessary and suf- ﬁcient graphical condition for the correspond- ing decision problem. We further develop a pro- cedure for systematically computing the target effect, and prove that it is sound and complete for gID instances. In other words, failure of the algorithm in returning an expression implies that the target effect is not computable from the available distributions. Finally, as a corollary of these results, we show that do-calculus is complete for the task of g-identiﬁability. 1 INTRODUCTION One of the main tasks in the empirical sciences and data- driven disciplines is to infer cause and effect relationships ⇤This work was done while the authors were at Purdue Uni- versity. Corresponding author’s email: sl4712@columbia.edu.from a combination of observations, experiments, and substantive knowledge about the phenomenon under in- vestigation. Causal relations are deemed desirable and valuable for constructing explanations and for contem- plating novel interventions that were never experienced before \[Pearl, 2000, Spirtes et al., 2001, Bareinboim and Pearl, 2016, Pearl and Mackenzie, 2018\]. In one line of investigation, this task is formalized through the question of whether the effect that an intervention on a set of variables Xwill have on another set of outcome variables Y(denoted Px(y)) can be uniquely computed from the probability distribution Pover the observed vari- ables Vand a causal diagram G. This is known as the problem of identiﬁcation \[Pearl, 1995, 2000, Bareinboim and Pearl, 2016\], and has received great attention in the literature, starting (@lee2020general)

Thomas Richardson Markov properties for acyclic directed mixed graphs *Scandinavian Journal of Statistics*, 30 (1): 145–157, 2003. **Abstract:** We consider acyclic directed mixed graphs, in which directed edges ( x → y ) and bi‐directed edges ( x ↔ y ) may occur. A simple extension of Pearl’s d ‐separation criterion, called m ‐separation, is applied to these graphs. We introduce a local Markov property which is equivalent to the global property resulting from the m ‐separation criterion for arbitrary distributions. (@richardson2003admg)

Donald B Rubin Estimating causal effects of treatments in randomized and nonrandomized studies *Journal of educational Psychology*, 66 (5): 688, 1974. **Abstract:** A discussion of matching, randomization, random sampling, and other methods of controlling extraneous variation is presented. The objective is to specify the benefits of randomization in estimating causal effects of treatments. The basic conclusion is that randomization should be employed whenever possible but that the use of carefully controlled nonrandomized data to estimate causal effects is a reasonable and necessary procedure in many cases. Recent psychological and educational literature has included extensive criticism of the use of nonrandomized studies to estimate causal effects of treatments (e.g., Campbell & Erlebacher, 1970). The implication in much of this literature is that only properly randomized experiments can lead to useful estimates of causal effects. If taken as applying to all fields of study, this position is untenable. Since the extensive use of randomized experiments is limited to the last half century,8 and in fact is not used in much scientific investigation today,4 one is led to the conclusion that most scientific truths have been established without using randomized experiments. In addition, most of us successfully determine the causal effects of many of our everyday actions, even interpersonal behaviors, without the benefit of randomization. Even if the position that causal effects of treatments can only be well established from randomized experiments is taken as applying only to the social sciences in which (@rubin1974estimating)

Yaroslav Kivva, Ehsan Mokhtarian, Jalal Etesami, and Negar Kiyavash Revisiting the general identifiability problem In *Uncertainty in Artificial Intelligence*, pages 1022–1030. PMLR, 2022. **Abstract:** We revisit the problem of general identifiability originally introduced in \[Lee et al., 2019\] for causal inference and note that it is necessary to add positivity assumption of observational distribution to the original definition of the problem. We show that without such an assumption the rules of do-calculus and consequently the proposed algorithm in \[Lee et al., 2019\] are not sound. Moreover, adding the assumption will cause the completeness proof in \[Lee et al., 2019\] to fail. Under positivity assumption, we present a new algorithm that is provably both sound and complete. A nice property of this new algorithm is that it establishes a connection between general identifiability and classical identifiability by Pearl \[1995\] through decomposing the general identifiability problem into a series of classical identifiability sub-problems. (@kivva2022revisiting)

Ilya Shpitser and Judea Pearl Identification of joint interventional distributions in recursive semi-markovian causal models In *AAAI*, pages 1219–1226, 2006. **Abstract:** This paper is concerned with estimating the effects of actions from causal assumptions, represented concisely as a directed graph, and statistical knowledge, given as a probability distribution. We provide a necessary and sufficient graphical condition for the cases when the causal effect of an arbitrary set of variables on another arbitrary set can be determined uniquely from the available information, as well as an algorithm which computes the effect whenever this condition holds. Furthermore, we use our results to prove completeness of do-calculus \[Pearl, 1995\], and a version of an identification algorithm in \[Tian, 2002\] for the same identification problem. Finally, we derive a complete characterization of semi-Markovian models in which all causal effects are identifiable. (@shpitser2006id)

Ilya Shpitser When does the id algorithm fail? *arXiv preprint arXiv:2307.03750*, 2023. **Abstract:** The ID algorithm solves the problem of identification of interventional distributions of the form p(Y \| do(a)) in graphical causal models, and has been formulated in a number of ways \[12, 9, 6\]. The ID algorithm is sound (outputs the correct functional of the observed data distribution whenever p(Y \| do(a)) is identified in the causal model represented by the input graph), and complete (explicitly flags as a failure any input p(Y \| do(a)) whenever this distribution is not identified in the causal model represented by the input graph). The reference \[9\] provides a result, the so called "hedge criterion" (Corollary 3), which aims to give a graphical characterization of situations when the ID algorithm fails to identify its input in terms of a structure in the input graph called the hedge. While the ID algorithm is, indeed, a sound and complete algorithm, and the hedge structure does arise whenever the input distribution is not identified, Corollary 3 presented in \[9\] is incorrect as stated. In this note, I outline the modern presentation of the ID algorithm, discuss a simple counterexample to Corollary 3, and provide a number of graphical characterizations of the ID algorithm failing to identify its input distribution. (@shpitser2023does)

Yimin Huang and Marco Valtorta Pearl’s calculus of intervention is complete In *Proceedings of the 22nd Conference on Uncertainty in Artificial Intelligence, 2006*, pages 13–16, 2006. **Abstract:** This paper is concerned with graphical criteria that can be used to solve the problem of identifying casual effects from nonexperimental data in a causal Bayesian network structure, i.e., a directed acyclic graph that represents causal relationships. We first review Pearl’s work on this topic \[Pearl, 1995\], in which several useful graphical criteria are presented. Then we present a complete algorithm \[Huang and Valtorta, 2006b\] for the identifiability problem. By exploiting the completeness of this algorithm, we prove that the three basic do-calculus rules that Pearl presents are complete, in the sense that, if a causal effect is identifiable, there exists a sequence of applications of the rules of the do-calculus that transforms the causal effect formula into a formula that only includes observational quantities. (@huang2006pearl)

Zhaohui Fu and Sharad Malik On solving the partial max-sat problem In Armin Biere and Carla P. Gomes, editors, *Theory and Applications of Satisfiability Testing - SAT 2006*, page 252–265, Berlin, Heidelberg, 2006. Springer. ISBN 978-3-540-37207-3. . **Abstract:** The Partial Max-SAT (PMSAT) problem is an optimization variant of the well-known Propositional Boolean Satisfiability (SAT) problem. It holds an important place in theory and practice, because a huge number of real-world problems, such as timetabling, planning, routing, bioinformatics, fault diagnosis, etc., could be encoded into it. Stochastic local search (SLS) methods can solve many real-world problems that often involve large-scale instances at reasonable computation costs while delivering good-quality solutions. In this work, we propose a novel SLS algorithm called adaptive variable depth SLS for PMSAT problem solving based on a dynamic local search framework. Our algorithm exploits two algorithmic components of an SLS method: parameter tuning and neighborhood search. Our first contribution is the design of an adaptive parameter tuner that searches for the best parameter setting for each instance by considering its features. The second contribution is a variable depth neighborhood search (VDS) algorithm adopted for PMSAT problem, which our empirical evaluation proves is a more efficient w.r.t. single neighborhood search. We conducted our experiments on the PMSAT benchmarks from MaxSAT Evaluation 2014 to 2019, including more than 3600 instances which have been encoded from a broad range of domains such as verification, optimization, graph theory, automated-reasoning, pseudo Boolean, etc. Our experimental evaluation results show that AVD-SLS solver, which is implemented based on our algorithm, outperforms state-of-the-art PMSAT SLS solvers in most benchmark classes, including random, crafted, and industrial instances. Furthermore, AVD-SLS reports remarkably better results on weighted benchmark, and shows competitive results with several well-known hybrid PMSAT solvers. (@pmax-sat)

Jared Lee Gearhart, Kristin Lynn Adair, Justin David Durfee, Katherine A Jones, Nathaniel Martin, and Richard Joseph Detry Comparison of open-source linear programming solvers Technical report, Sandia National Lab.(SNL-NM), Albuquerque, NM (United States), 2013. **Abstract:** When developing linear programming models, issues such as budget limitations, customer requirements, or licensing may preclude the use of commercial linear programming solvers. In such cases, one option is to use an open-source linear programming solver. A survey of linear programming tools was conducted to identify potential open-source solvers. From this survey, four open-source solvers were tested using a collection of linear programming test problems and the results were compared to IBM ILOG CPLEX Optimizer (CPLEX) \[1\], an industry standard. The solvers considered were: COIN-OR Linear Programming (CLP) \[2\], \[3\], GNU Linear Programming Kit (GLPK) \[4\], lp_solve \[5\] and Modular In-core Nonlinear Optimization System (MINOS) \[6\]. As no open-source solver outperforms CPLEX, this study demonstrates the power of commercial linear programming software. CLP was found to be the top performing open-source solver considered in terms of capability and speed. GLPK also performed well but cannot match the speed of CLP or CPLEX. lp_solve and MINOS were considerably slower and encountered issues when solving several test problems. (@gearhart2013comparison)

Gurobi Optimization, LLC *Gurobi Optimizer Reference Manual* 2023. URL <https://www.gurobi.com>. **Abstract:** Introduction This study focuses on broadening the applicability of the metaheuristic L1-norm fitted and penalized (L1L1) optimization method in finding a current pattern for multichannel transcranial electrical stimulation (tES). The metaheuristic L1L1 optimization framework defines the tES montage via linear programming by maximizing or minimizing an objective function with respect to a pair of hyperparameters. Methods In this study, we explore the computational performance and reliability of different optimization packages, algorithms, and search methods in combination with the L1L1 method. The solvers from Matlab R2020b, MOSEK 9.0, Gurobi Optimizer, CVX’s SeDuMi 1.3.5, and SDPT3 4.0 were employed to produce feasible results through different linear programming techniques, including Interior-Point (IP), Primal-Simplex (PS), and Dual-Simplex (DS) methods. To solve the metaheuristic optimization task of L1L1, we implement an exhaustive and recursive search along with a well-known heuristic direct search as a reference algorithm. Results Based on our results, and the given optimization task, Gurobi’s IP was, overall, the preferable choice among Interior-Point while MOSEK’s PS and DS packages were in the case of Simplex methods. These methods provided substantial computational time efficiency for solving the L1L1 method regardless of the applied search method. Discussion While the best-performing solvers show that the L1L1 method is suitable for maximizing either focality and intensity, a few of these solvers could not find a bipolar configuration. Part of the discrepancies between these methods can be explained by a different sensitivity with respect to parameter variation or the resolution of the lattice provided. (@gurobi)

Janine Witte, Leonard Henckel, Marloes H Maathuis, and Vanessa Didelez On efficient adjustment in causal graphs *Journal of Machine Learning Research*, 21 (246): 1–45, 2020. **Abstract:** We consider estimation of a total causal effect from observational data via covariate adjustment. Ideally, adjustment sets are selected based on a given causal graph, reflecting knowledge of the underlying causal structure. Valid adjustment sets are, however, not unique. Recent research has introduced a graphical criterion for an ’optimal’ valid adjustment set (O-set). For a given graph, adjustment by the O-set yields the smallest asymptotic variance compared to other adjustment sets in certain parametric and non-parametric models. In this paper, we provide three new results on the O-set. First, we give a novel, more intuitive graphical characterisation: We show that the O-set is the parent set of the outcome node(s) in a suitable latent projection graph, which we call the forbidden projection. An important property is that the forbidden projection preserves all information relevant to total causal effect estimation via covariate adjustment, making it a useful methodological tool in its own right. Second, we extend the existing IDA algorithm to use the O-set, and argue that the algorithm remains semi-local. This is implemented in the R-package pcalg. Third, we present assumptions under which the O-set can be viewed as the target set of popular non-graphical variable selection algorithms such as stepwise backward selection. (@witte2020efficient)

Andrea Rotnitzky and Ezequiel Smucler Efficient adjustment sets for population average causal treatment effect estimation in graphical models *Journal of Machine Learning Research*, 21 (188): 1–86, 2020. **Abstract:** The method of covariate adjustment is often used for estimation of total treatment ef- fects from observational studies. Restricting attention to causal linear models, a recent article (Henckel et al., 2019) derived two novel graphical criteria: one to compare the asymptotic variance of linear regression treatment e ect estimators that control for certain distinct adjustment sets and another to identify the optimal adjustment set that yields the least squares estimator with the smallest asymptotic variance. In this paper we show that the same graphical criteria can be used in non-parametric causal graphical models when treatment e ects are estimated using non-parametrically adjusted estimators of the inter- ventional means. We also provide a new graphical criterion for determining the optimal adjustment set among the minimal adjustment sets and another novel graphical criterion for comparing time dependent adjustment sets. We show that uniformly optimal time de- pendent adjustment sets do not always exist. For point interventions, we provide a sound and complete graphical criterion for determining when a non-parametric optimally adjusted estimator of an interventional mean, or of a contrast of interventional means, is semipara- metric ecient under the non-parametric causal graphical model. In addition, when the criterion is not met, we provide a sound algorithm that checks for possible simpli cations of the ecient in uence function of the parameter. Finally, we nd an interesting connection between identi cation and ecient covariate adjustment estimation. Speci cally, we show that if there exists an identifying formula for an interventional mean that depends only on treatment, outcome and mediators, then the non-parametric optimally adjusted estimator can never be globally ecient under the causal graphical model. (@rotnitzky2020efficient)

Leonard Henckel, Emilija Perković, and Marloes H Maathuis Graphical criteria for efficient total effect estimation via adjustment in causal linear models *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 84 (2): 579–599, 2022. **Abstract:** Abstract Covariate adjustment is a commonly used method for total causal effect estimation. In recent years, graphical criteria have been developed to identify all valid adjustment sets, that is, all covariate sets that can be used for this purpose. Different valid adjustment sets typically provide total causal effect estimates of varying accuracies. Restricting ourselves to causal linear models, we introduce a graphical criterion to compare the asymptotic variances provided by certain valid adjustment sets. We employ this result to develop two further graphical tools. First, we introduce a simple variance decreasing pruning procedure for any given valid adjustment set. Second, we give a graphical characterization of a valid adjustment set that provides the optimal asymptotic variance among all valid adjustment sets. Our results depend only on the graphical structure and not on the specific error variances or edge coefficients of the underlying causal linear model. They can be applied to directed acyclic graphs (DAGs), completed partially directed acyclic graphs (CPDAGs) and maximally oriented partially directed acyclic graphs (maximal PDAGs). We present simulations and a real data example to support our results and show their practical applicability. (@henckel2022graphical)

Ilya Shpitser, Tyler VanderWeele, and James M Robins On the validity of covariate adjustment for estimating causal effects In *Proceedings of the 26th Conference on Uncertainty in Artificial Intelligence, UAI 2010*, pages 527–536. AUAI Press, 2010. **Abstract:** Identifying effects of actions (treatments) on outcome variables from observational data and causal assumptions is a fundamental problem in causal inference. This identification is made difficult by the presence of con-founders which can be related to both treatment and outcome variables. Confounders are often handled, both in theory and in practice, by adjusting for covariates, in other words considering outcomes conditioned on treatment and covariate values, weighed by probability of observing those covariate values. In this paper, we give a complete graphical criterion for covariate adjustment, which we term the adjustment criterion, and derive some interesting corollaries of the completeness of this criterion. (@shpitser2010validity)

Sanghack Lee, Juan D. Correa, and Elias Bareinboim General identifiability with arbitrary surrogate experiments In *Proceedings of The 35th Uncertainty in Artificial Intelligence Conference*, page 389–398. PMLR, August 2020. URL <https://proceedings.mlr.press/v115/lee20b.html>. **Abstract:** We study the problem of causal identiﬁcation from an arbitrary collection of observational and experimental distributions, and substantive knowledge about the phenomenon under in- vestigation, which usually comes in the form of a causal graph. We call this problem g- identiﬁability , or gID for short. The gID set- ting encompasses two well-known problems in causal inference, namely, identiﬁability \[Pearl, 1995\] and z-identiﬁability \[Bareinboim and Pearl, 2012\] — the former assumes that an ob- servational distribution is necessarily available, and no experiments can be performed, condi- tions that are both relaxed in the gID setting; the latter assumes that allcombinations of exper- iments are available, i.e., the power set of the experimental set Z, which gID does not require a priori. In this paper, we introduce a general strategy to prove non-gID based on hedgelets andthickets , which leads to a necessary and suf- ﬁcient graphical condition for the correspond- ing decision problem. We further develop a pro- cedure for systematically computing the target effect, and prove that it is sound and complete for gID instances. In other words, failure of the algorithm in returning an expression implies that the target effect is not computable from the available distributions. Finally, as a corollary of these results, we show that do-calculus is complete for the task of g-identiﬁability. 1 INTRODUCTION One of the main tasks in the empirical sciences and data- driven disciplines is to infer cause and effect relationships ⇤This work was done while the authors were at Purdue Uni- versity. Corresponding author’s email: sl4712@columbia.edu.from a combination of observations, experiments, and substantive knowledge about the phenomenon under in- vestigation. Causal relations are deemed desirable and valuable for constructing explanations and for contem- plating novel interventions that were never experienced before \[Pearl, 2000, Spirtes et al., 2001, Bareinboim and Pearl, 2016, Pearl and Mackenzie, 2018\]. In one line of investigation, this task is formalized through the question of whether the effect that an intervention on a set of variables Xwill have on another set of outcome variables Y(denoted Px(y)) can be uniquely computed from the probability distribution Pover the observed vari- ables Vand a causal diagram G. This is known as the problem of identiﬁcation \[Pearl, 1995, 2000, Bareinboim and Pearl, 2016\], and has received great attention in the literature, starting (@lee-2020)

Alexey Ignatiev, Antonio Morgado, and Joao Marques-Silva Rc2: an efficient maxsat solver *Journal on Satisfiability, Boolean Modeling and Computation*, 11 (1): 53–64, September 2019. ISSN 15740617. . **Abstract:** Recent work proposed a toolkit PySAT aiming at fast and easy prototyping with propositional satisfiability (SAT) oracles in Python, which enabled one to exploit the power of the original implementations of the state-of-the-art SAT solvers in Python. (@rc2-maxsat)

Paul Erdos and Alfred Renyi On the evolution of random graphs *Publ. Math. Inst. Hungary. Acad. Sci.*, 5: 17–61, 1960. **Abstract:** (n) k edges have equal probabilities to be chosen as the next one . We shall 2 study the "evolution" of such a random graph if N is increased . In this investigation we endeavour to find what is the "typical" structure at a given stage of evolution (i . e . if N is equal, or asymptotically equal, to a given function N(n) of n) . By a "typical" structure we mean such a structure the probability of which tends to 1 if n -\* + when N = N(n) . If A is such a property that lim Pn,N,(n ) ( A) = 1, we shall say that „almost all" graphs Gn,N(n) n–possess this property . (@Erdos:1960)

Antonio Morgado, Carmine Dodaro, and Joao Marques-Silva Core-guided maxsat with soft cardinality constraints In Barry O’Sullivan, editor, *Principles and Practice of Constraint Programming*, page 564–573, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10428-7. . **Abstract:** Maximum satisfiability (MaxSAT) is a viable approach to solving NP-hard optimization problems. In the realm of core-guided MaxSAT solving – one of the most effective MaxSAT solving paradigms today – algorithmic variants employing so-called soft cardinality constraints have proven very effective. In this work, we propose to combine weight-aware core extraction (WCE) – a recently proposed approach that enables relaxing multiple cores instead of a single one during iterations of core-guided search – with a novel form of structure sharing in the cardinality-based core relaxation steps performed in core-guided MaxSAT solvers. In particular, the proposed form of structure sharing is enabled by WCE, which has so-far not been widely integrated to MaxSAT solvers, and allows for introducing fewer variables and clauses during the MaxSAT solving process. Our results show that the proposed techniques allow for avoiding potential overheads in the context of soft cardinality constraint based core-guided MaxSAT solving both in theory and in practice. In particular, the combination of WCE and structure sharing improves the runtime performance of a state-of-the-art core-guided MaxSAT solver implementing the central OLL algorithm. (@oll-maxsat)

IBM Corporation *IBM ILOG CPLEX Optimization Studio CPLEX User’s Manual* 2023. URL <https://www.ibm.com/analytics/cplex-optimizer>. **Abstract:** Produce precise and logical decisions for planning and resource allocation problems using the powerful algorithms of IBM ILOG CPLEX Optimizer. (@cplex)

J.J.H. Forrest and R. Lougee-Heimer *CBC User Guide* COIN-OR, 2023. URL <https://github.com/coin-or/Cbc>. **Abstract:** Projects such as this one are maintained by a small group of volunteers under the auspices of the non-profit COIN-OR Foundation and we need your help! Please consider sponsoring our activities or volunteering to help! This file is auto-generated from config.yml using the generate_readme script. To make changes, please edit config.yml or the generation scripts here and here . Cbc ( C oin-or b ranch and c ut) is an open-source mixed integer linear programming solver written in C++. It can be used as a callable library or using a stand-alone executable. It can be used in a wide variety of ways through various modeling systems, packages, etc. (@cbc)

Steffen L Lauritzen, A Philip Dawid, Birgitte N Larsen, and H-G Leimer Independence properties of directed markov fields *Networks*, 20 (5): 491–505, 1990. **Abstract:** Abstract We investigate directed Markov fields over finite graphs without positivity assumptions on the densities involved. A criterion for conditional independence of two groups of variables given a third is given and named as the directed, global Markov property. We give a simple proof of the fact that the directed, local Markov property and directed, global Markov property are equivalent and – in the case of absolute continuity w. r. t. a product measure – equivalent to the recursive factorization of densities. It is argued that our criterion is easy to use, it is sharper than that given by Kiiveri, Speed, and Carlin and equivalent to that of Pearl. It follows that our criterion cannot be sharpened. (@lauritzen1990independence)

G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher An analysis of approximations for maximizing submodular set functions—i *Mathematical Programming*, 14 (1): 265–294, December 1978. ISSN 1436-4646. . (@nemhauser-1978)

</div>

<div class="center" markdown="1">

**Appendix**

</div>

# Implementation details and further experimental results [app:exp]

## Implementation details

Our codebase is implemented fully in Python. We use the [PySAT](https://pysathq.github.io/) library for formulating and solving the WPMAX-SAT problem, and the [PuLP](https://coin-or.github.io/pulp/) library for formulating and solving the ILP problem.

#### Solving the WPMAX-SAT problem.

There are several algorithms to solve the WPMAX-SAT instance to optimality. These algorithms include RC2 `\citep{rc2-maxsat}`{=latex} and OLL `\citep{oll-maxsat}`{=latex}, both of which are core-based algorithms that utilize unsatisfiable cores to iteratively refine the solution. In this context, a “core” refers to an unsatisfiable subset of clauses within the CNF formula that cannot be satisfied simultaneously under any assignment. These algorithms relax the unsatisfiable soft clauses in the core by adding relaxation variables and enforce cardinality constraints on these variables. By strategically increasing the bounds on these cardinality constraints or modifying the weights of soft clauses based on the cores identified, the algorithms efficiently reduce the search space and converge on the maximum weighted set of satisfiable clauses, thereby solving the WPMAX-SAT problem optimally.

#### Solving the ILP problem.

Similarly, with the ILP formulation of the MCID problem presented in  
efsec:reformulations, we can utilize exact algorithms designed for solving ILP problems to find an optimal solution. ILP solvers work by formulating the problem with linear inequalities as constraints and integer variables that need to be optimized. Popular ILP solvers include CPLEX `\citep{cplex}`{=latex}, Gurobi `\citep{gurobi}`{=latex}, and the open-source solver CBC `\citep{cbc}`{=latex}. The latter is a branch-and-cut-based solver, and cutting plane methods to explore feasible integer solutions systematically while pruning the search space based on bounds calculated during the solving process.

We use the Gurobi solver in our experiments.

## Extended WPMAX-SAT simulations

We extended the simulations in  
efsec:experiments for up to \\(n=500\\) vertices, and the results are presented in  
effig:sat-extended. We observe that even at \\(n=500\\), WPMAX-SAT takes around the same time as Algorithm 2 of `\citet{akbari-2022}`{=latex} does to solve \\(n=40\\) (\\(230\\) s for both). Moreover, we can clearly see the exponential growth in time complexity, as expected, especially for \\(n > 400\\).

<figure id="fig:sat-extended">
<img src="./figures/sat_long.png"" />
<figcaption>Semi-log plot of the average time taken by WPMAX-SAT to solve one graph versus the number of vertices in the graph.</figcaption>
</figure>

## Investigating the effects of directed and bidirected edge probabilities on the performance of exact algorithms

We run experiments on varying the probabilities of directed and bidirected edges in the graph. We fix the number of vertices at \\(n=20\\) and vary the probabilities of directed and bidirected edges from \\(0.001\\) to \\(1.00\\) in increments of \\(0.001\\). The results are presented in  
effig:exp_edgeprobs.

<figure id="fig:exp_edgeprobs">
<img src="./figures/sat_alg2_heatmaps.png"" />
<figcaption>Heatmap of the average time taken by WPMAX-SAT (on the left) and Algorithm 2 of <span class="citation" data-cites="akbari-2022"></span> (on the right) to solve one graph versus the probabilities of directed and bidirected edges in the graph.</figcaption>
</figure>

## Investigating the effect of cost on the performance of the algorithms

We run experiments with \\(n=20\\) and costs sampled from a Poisson distributions with mean parameter ranging from \\(1\\) to \\(100\\). The results are presented in  
effig:exp_poisson. Interestingly, there appears to be no clear trend in the time complexity of the algorithms with respect to the mean parameter of the Poisson distribution. This suggests that the time complexity of the algorithms is not significantly affected by the cost of the vertices.

<figure id="fig:exp_poisson">
<img src="./figures/poisson_time.png"" />
<figcaption>Average time taken by Algorithm 2 of <span class="citation" data-cites="akbari-2022"></span> (MHS), ILP, and WPMAX-SAT to solve one graph versus the mean parameter of the Poisson distribution from which the costs are sampled.</figcaption>
</figure>

## Investigating the effects of directed and bidirected edge probabilities on the performance of the heuristic algorithms

We run experiments on varying the probabilities of directed and bidirected edges in the graph. We vary \\(n\\) from \\(n=10\\) to \\(n=200\\) and the probabilities of directed and bidirected edges in \\(\{0.1,0.5\}\\). The results are presented in  
effig:exp_heur_edgeprobs. We see that our proposed heuristic algorithm consistently outperforms the heuristic algorithms of `\citet{akbari-2022}`{=latex} for all graph sizes and edge probabilities.

<figure id="fig:exp_heur_edgeprobs">
<img src="./figures/heur_cost_pq.png"" />
<figcaption>Average running time of the heuristic algorithms <span class="math inline"><em>H</em><sub>1</sub></span> and <span class="math inline"><em>H</em><sub>2</sub></span> of <span class="citation" data-cites="akbari-2022"></span> and<br />
ef<span>alg:adj</span> versus the number of vertices in the graph for different probabilities of directed and bidirected edges in the graph.</figcaption>
</figure>

# Algorithms [app:algorithms]

## Pruning algorithm for finding the hedge hull

We include the algorithm for finding the hedge hull for the sake of completeness. This algorithm is adopted from `\citet{akbari-2022}`{=latex}. <span id="app:pruning" label="app:pruning"></span>

<div class="algorithm*" markdown="1">

<div class="algorithmic" markdown="1">

ALGORITHM BLOCK (caption below)

  
**Procedure** Prune\\(\mathcal{G}=\langle V, \overrightarrow{E}, \overleftrightarrow{E}\rangle, S\\)  
\\(\mathcal{H}\gets \mathrm{Anc}_V(S)\\)  
**While** True  
\\(\mathcal{H}'\gets \{v\in \mathcal{H}:v\textit{ has a bidirected path to }S\textit{ in }\mathcal{G}[\mathcal{H}]\}\\)  
**If** \\(\mathcal{H}'=\mathcal{H}\\)  
  
Return \\(\mathcal{H}\\)  
EndIf  
\\(\mathcal{H}\gets \mathrm{Anc}_{\mathcal{H}'}(S)\\)  
**If** \\(\mathcal{H}=\mathcal{H}'\\)  
  
Return \\(\mathcal{H}\\)  
EndIf  
EndWhile  
EndProcedure

</div>

</div>

## SAT construction procedure for multiple districts [app:multipledist]

The procedure for constructing the SAT formula when \\(S\\) comprises multiple districts was postponed to this section due to space limitations. This procedure is detailed below.

<div class="algorithm*" markdown="1">

<div class="algorithmic" markdown="1">

ALGORITHM BLOCK (caption below)

  
**Procedure** ConstructSAT\\(X, Y, \mathcal{G}=\langle V, \overrightarrow{E}, \overleftrightarrow{E}\rangle\\)  
\\(S\gets\mathrm{Anc}_{V\setminus X}(Y)\\)  
\\(\boldsymbol{\mathcal{S}}\gets\\) maximal districts of \\(S\\) in \\(\mathcal{G}[S]\\)  
\\(r\gets\vert\boldsymbol{\mathcal{S}}\vert\\)  
\\(F\gets 1\\)  
**For** \\(\ell\in\{1,\dots r\}\\) \# comment: iterate over districts of \\(S\\)  
\\(m\gets \vert \mathcal{H}_\mathcal{G}(S_\ell)\setminus S_\ell\vert\\) \# comment: \# iterations  
**For** \\(k\in\{1,\dots,r\}\\) \# comment: iterate over expressions  
\\(F \gets F \land (x_{i,0,k} \lor \lnot z_{k,\ell})\\) for every \\(i\\) s.t. \\(v_i\in S_\ell\\)  
\\(F \gets F \land (x_{i,j,k,\ell} \lor \lnot z_{k,\ell})\\) for every \\(i\\) s.t. \\(v_i\in S_\ell\\) and every \\(j \in \{1,\ldots,m+1\}\\)  
**For** \\((v_i,v_p)\in\overrightarrow{E}\\) \# comment: iteration \\(j=1\\)  
\\(F\gets F\land (\lnot x_{i,0,k}\lor x_{i,1,k,\ell}\lor\lnot x_{p,1,k,\ell}\lor \lnot z_{k,\ell})\\)  
EndFor  
**For** \\(j\in\{2,\dots,m+1\}\\)  
**If** \\(j\\) is odd  
**For** \\((v_i,v_p)\in\overrightarrow{E}\\)  
\\(F\gets F\land (\lnot x_{i,j-1,k,\ell}\lor x_{i,j,k,\ell}\lor\lnot x_{p,j,k,\ell}\lor \lnot z_{k,\ell})\\)  
EndFor  
Else  
**For** \\(\{v_i,v_p\}\in\overleftrightarrow{E}\\)  
\\(F\gets F\land (\lnot x_{i,j-1,k,\ell}\lor x_{i,j,k,\ell}\lor\lnot x_{p,j,k,\ell}\lor \lnot z_{k,\ell})\\)  
\\(F\gets F\land (\lnot x_{p,j-1,k,\ell}\lor x_{p,j,k,\ell}\lor\lnot x_{i,j,k,\ell}\lor \lnot z_{k,\ell})\\)  
EndFor  
EndIf  
EndFor  
\\(F\gets F\land (\lnot x_{i,m+1,k,\ell} \lor \lnot z_{k,\ell})\\) for every \\(v_i\notin S_\ell\\)  
EndFor  
\\(F\gets F\land (z_{1,\ell}\lor\dots\lor z_{r,\ell})\\)  
EndFor  
  
Return \\(F\\)  
EndProcedure

</div>

</div>

# Missing Proofs [app:proofs]

## Results of efsec:reformulations

<div class="proof" markdown="1">

*Proof.* *Proof of ‘if:’* Suppose \\(\mathcal{I}\\) hits every hedge formed for \\(S\\). We construct a satisfying solution for the SAT formula as follows. We begin with \\(x_{i,0}\\): \\[x_{i,0}^*=\begin{cases}
        0;\quad \textit{if } i\in\mathcal{I}\\
        1;\quad \textit{o.w.}
    \end{cases}\\] For every \\(j\in\{1,\dots,m+1\}\\), define \\(H_j=\{i:x_{i,j-1}=1\}\\). Then \\(x_{i,j}^*\\) for \\(j\in\{1,\dots,m+1\}\\) is chosen recursively as below.

- Odd \\(j\\): \\(x_{i,j}^*=1\\) if \\(i \in H_j\\) and \\(v_i\\) has a directed path to \\(S\\) in \\(\mathcal{G}[H_j]\\), and \\(x_{i,j}^*=0\\) otherwise.

- Even \\(j\\): \\(x_{i,j}^*=1\\) if \\(i \in H_j\\) and \\(v_i\\) has a bidirected path to \\(S\\) in \\(\mathcal{G}[H_j]\\), and \\(x_{i,j}^*=0\\) otherwise.

Next, we prove that \\(\{x_{i,j}^*\}\\) as defined above satisfies \\(F\\). We consider the three types of clauses in \\(F\\) separately:

- For odd \\(j \in \{1, \ldots, m+1\}\\), the clause \\((\neg x_{i,j-1} \lor x_{i,j} \lor \neg x_{\ell,j})\\) corresponds to the directed edge \\((v_i,v_\ell)\in\overrightarrow{E}\\): if either \\(x_{i,j-1}^*=0\\) or \\(x_{\ell,j}^*=0\\), then this clause is trivially satisfied. So suppose \\(x_{i,j-1}^*=1\\), and \\(x_{\ell,j}^*=1\\), which implies by construction that \\(x_{\ell,j-1}^*=1\\). Therefore, \\(i,\ell\in H_j\\). Further, since \\(x_{\ell,j}^*=1\\), \\(v_\ell\\) has a directed path to \\(S\\) in \\(\mathcal{G}[H_j]\\). Then \\(v_i\\) has a directed path to \\(S\\) in \\(\mathcal{G}[H_j]\\) because of the edge \\((v_i,v_\ell)\in\overrightarrow{E}\\). By the construction above, \\(x_{i,j}^*=1\\), which satisfies the clause.

- For even \\(j \in \{1, \ldots, m+1\}\\), the clause \\((\neg x_{i,j-1} \lor x_{i,j} \lor \neg x_{\ell,j})\\) corresponds to the bidirected edge \\(\{v_i,v_\ell\}\in\overleftrightarrow{E}\\): if either \\(x_{i,j-1}^*=0\\) or \\(x_{\ell,j}^*=0\\), then this clause is trivially satisfied. So suppose \\(x_{i,j-1}^*=1\\), and \\(x_{\ell,j}^*=1\\), which implies by construction that \\(x_{\ell,j-1}^*=1\\). Therefore, \\(i,\ell\in H_j\\). Further, since \\(x_{\ell,j}^*=1\\), \\(v_\ell\\) has a bidirected path to \\(S\\) in \\(\mathcal{G}[H_j]\\). Then \\(v_i\\) has a bidirected path to \\(S\\) in \\(\mathcal{G}[H_j]\\) because of the edge \\(\{v_i,v_\ell\}\in\overleftrightarrow{E}\\). By the construction above, \\(x_{i,j}^*=1\\), which satisfies the clause.

- The clauses \\(\neg x_{i,m+1}\\): First note that by construction, if for some \\(j\in\{1,\dots,m+1\}\\), \\(x_{i,j-1}^*=0\\), then \\(x_{i,j}^*=x_{i,j+1}^*=\dots=x_{i,m}^*=0\\). That is, \\(\{x_{i,j}^*\}_{j=0}^m\\) is a non-increasing binary-valued sequence for every \\(i\\). Therefore, for every \\(i\in\{1,\dots,m\}\\), there exists *at most* one \\(j\\) such that \\(x_{i,j-1}^*> x_{i,j}^*\\). We consider two cases separately:

  - There are exactly \\(m\\) many \\(j\in\{1,\dots,m+1\}\\) for which there exists at least one \\(i\\) such that \\(x_{i,j-1}^*> x_{i,j}^*\\). In this case, for every \\(i\in\{1,\dots,m\}\\), there exists exactly one \\(j\in\{1,\dots,m+1\}\\) such that \\(x_{i,j-1}^*> x_{i,j}^*\\). Then for every \\(i\\), there exists \\(j\\) such that \\(x_{i,j}^*=0\\), and following the argument above, \\(x_{i,m+1}^*=0\\). Hence, the clauses \\(\lnot x_{i,m+1}\\) are all satisfied.

  - There are strictly less than \\(m\\) many \\(j\in\{1,\dots,m+1\}\\) for which there exists at least one \\(i\\) such that \\(x_{i,j-1}^*> x_{i,j}^*\\). Then there exist \\(j,j'\in\{1,\dots,m+1\}\\) such that for every \\(i\in\{1,\dots,m\}\\), \\(x_{i,j-1}^*=x_{i,j}^*\\) and \\(x_{i,j'-1}^*=x_{x,j'}^*\\). Assume without loss of generality that \\(j'<j\\) and therefore, \\(j>1\\). If \\(x_{i,j}^*=0\\) for every \\(i\\), then by similar arguments as the previous case, \\(x_{i,m+1}^*=0\\) and the clauses \\(\lnot x_{i,m}\\) are satisfied. So suppose for the sake of contradiction that there exists a non-empty set \\(H_j=\{i:x_{i,j-1}^*=1\}\neq\emptyset\\). Note that \\(H_{j+1}\coloneqq\{i:x_{i,j}^*=1\}=H_j\\), since \\(x_{i,j-1}^*=x_{i,j}^*\\) for every \\(i\\). Moreover, \\(\mathcal{I}\cap H_j=\emptyset\\) since \\(x_{i,0}^*=0\\) for every \\(i\in\mathcal{I}\\) and \\(\{x_{i,k}^*\}_k\\) is non-increasing. Assume without loss of generality that \\(j\\) is odd. The proof is identical in case \\(j\\) is even. By definition, the set of vertices \\(H_{j+1}=H_j\\) have a directed path to \\(S\\) in \\(\mathcal{G}[H_j]\\). Moreover, the set of vertices \\(H_j\\) are those vertices in \\(H_{j-1}\\) that have a bidirected path to \\(S\\) in \\(\mathcal{G}[H_{j-1}]\\) (here we used \\(j>1\\) for \\(H_{j-1}\\) to be well-defined.) That is, \\(H_j\\) is the connected component of \\(S\\) in \\(\mathcal{G}[H_{j-1}]\\). The latter implies that every vertex in \\(H_j\\) has a bidirected path to \\(S\\) in \\(\mathcal{G}[H_j]\\). We proved that \\(H_j\\) is a hedge formed for \\(S\\), and \\(H_j\cap\mathcal{I}=\emptyset\\). This contradicts with \\(\mathcal{I}\\) intersecting with every hedge formed for \\(S\\).

*Proof of ‘only if:’* Suppose \\(\{x_{i,j}^*\}\\) is a satisfying solution, where \\(x_{i,0}^*=1\\) for every \\(i\notin\mathcal{I}\\). To prove \\(\mathcal{I}\\) intersects every hedge formed for \\(S\\), it suffices to show that there is no hedge formed for \\(S\\) in \\(\mathcal{G}[V\setminus\mathcal{I}]\\). Assume, for the sake of contradiction, that this is not the case. That is, there exists a hedge \\(H\subseteq V\setminus\mathcal{I}\\) formed for \\(S\\) in \\(\mathcal{G}\\). Suppose for some \\(j\in\{1,\dots, m+1\}\\), it holds that \\(x_{i,j-1}^*=1\\) for every \\(v_i\in H\\). We show that \\(x_{i,j}^*=1\\) for every \\(v_i\in H\\). We consider the following two cases separately:

- Even \\(j\\): for arbitrary \\(v_i,v_\ell\in H\\) such that \\(\{v_i,v_\ell\}\in\overleftrightarrow{E}\\), consider the clauses \\((\neg x_{i,j-1} \lor x_{i,j} \lor \neg x_{\ell,j})\\) and \\((\neg x_{\ell,j-1} \lor x_{\ell,j} \lor \neg x_{i,j})\\) that are in \\(F\\) by construction for even \\(j\\). Since \\(x_{i,j-1}^*=x_{\ell,j-1}^*=1\\), the expression \\(( x_{i,j}^* \lor \neg x_{\ell,j}^*) \land ( x_{\ell,j}^* \lor \neg x_{i,j}^*)\\) is satisfied; i.e., it evaluates to ‘true.’ The latter expression is equivalent to \\(( x_{i,j}^* \land  x_{\ell,j}^*) \lor (\lnot x_{i,j}^* \land \neg x_{\ell,j}^*)\\), which implies that \\(x_{i,j}^*=x_{\ell,j}^*\\). Note that \\(i,\ell\\) were chosen arbitrarily. This implies that \\(x_{i,j}^*\\) is equal for every \\(i\\), since \\(H\\) is a connected component through bidirected edges by definition of a hedge. Finally, since by construction, \\(x_{i,j}=1\\) for every \\(v_i\in S\subseteq H\\), that equal value is \\(1\\). Therefore, \\(x_{i,j}^*=1\\) for every \\(i\\) such that \\(v_i\in H\\).

- Odd \\(j\\): the proof is analogous to the case where \\(j\\) is even. For arbitrary \\(v_i,v_\ell\in H\\) such that \\((v_i,v_\ell)\in\overrightarrow{E}\\), consider the clause \\((\neg x_{i,j-1} \lor x_{i,j} \lor \neg x_{\ell,j})\\). Since \\(x_{i,j-1}^*=1\\), the expression \\(( x_{i,j}^* \lor \neg x_{\ell,j}^*)\\) is satisfied; i.e., it evaluates to ‘true.’ The latter implies that if \\(x_{\ell,j}^*=1\\) and \\(v_i\\) has a directed edge to \\(v_j\\), then \\(x_{i,j}^*=1\\). Using the same argument recursively, if \\(x_{\ell,j}^*=1\\) and \\(v_i\\) has a directed ‘path’ to \\(v_j\\) in \\(\mathcal{G}[H]\\), then \\(x_{i,j}^*=1\\). By construction, \\(x_{\ell,j}^*=1\\) for every \\(v_\ell\in S\\), and by definition of a hedge, every vertex \\(v_i\in H\\) has a directed path to \\(S\\). As a result, \\(x_{i,j}^*=1\\) for every \\(v_i\in H\\).

Since for every \\(v_i\in H\\), \\(x_{i,0}^*=1\\), using the arguments above, by induction, \\(x_{i,m+1}^*=1\\) for every \\(v_i\in H\\). However, this contradicts the fact that \\(\{x_{i,j}^*\}\\) satisfies \\(F\\), since \\(F\\) includes the clauses \\(\lnot x_{i,m+1}\\) for every \\(i\\). ◻

</div>

<div class="proof" markdown="1">

*Proof.* Let \\(\boldsymbol{\mathcal{I}}\\) be an optimizer of  
efeq:opt. By  
efprp:genid, for every \\(S_i\in\boldsymbol{\mathcal{S}}\\), there exists \\(\mathcal{I}_j\in\boldsymbol{\mathcal{I}}\\) that hits every hedge formed for \\(S_i\\). The set of such \\(\mathcal{I}_j\\)s is a subset of at most size \\(r\\) of \\(\boldsymbol{\mathcal{I}}\\), which implies that \\(\vert\boldsymbol{\mathcal{I}}\vert\leq r\\) since \\(\boldsymbol{\mathcal{I}}\\) is optimal. If \\(\vert \boldsymbol{\mathcal{I}}\vert = r\\), the claim is trivial. If \\(\vert \boldsymbol{\mathcal{I}}\vert < r\\), then simply add \\((r-\vert \boldsymbol{\mathcal{I}}\vert)\\) empty intervention sets to \\(\boldsymbol{\mathcal{I}}\\) to form an intervention set family \\(\boldsymbol{\mathcal{I}}^*\\) with the same cost which is an optimal solution to  
efeq:opt. ◻

</div>

<div class="proof" markdown="1">

*Proof.* The proof is identical to that of  
efthm:sat with necessary adaptations.

*Proof of ‘if:’* Suppose \\(\boldsymbol{\mathcal{I}}\\) is a solution to  
efeq:opt. From  
efprp:genid, for every \\(S_\ell\in\boldsymbol{\mathcal{S}}\\), there exists \\(k\\) such that \\(\mathcal{I}_k\\) hits every hedge formed for \\(S_\ell\\). Assign \\(z_{k,\ell}^*=1\\) and \\(z_{k',\ell}=0\\) for every other \\(k'\neq k\\), thereby satisfying every clause that includes \\(\lnot z_{k',\ell}\\), \\(k'\neq k\\). So it suffices to assign values to other variables so that clauses including \\(\lnot z_{k,\ell}\\). Since \\(z_{k,\ell}=1\\), these clauses reduce to \\((\lnot x_{i,j-1,k,\ell}\lor x_{i,j,k,\ell}\lor\lnot x_{p,j,k,\ell})\\) (see lines 16, 19, or 20.) These clauses are exactly in the form of 3-SAT clauses as in the single-district case procedure. An assignment exactly parallel to the proof of  
efthm:sat satisfies these clauses.

*Proof of ‘only if:’* Suppose \\(\{x_{i,0,k}^*\}\cup\{x_{i,j,k,\ell}^*\}\\) is a satisfying solution, where for some \\(k,\ell\in\{1,\dots,r\}\\), it holds that \\(z_{k,\ell}^*=1\\) and \\(x_{i,0,k}^*=1\\) for every \\(i\notin\mathcal{I}_k\\). We show that \\(\mathcal{I}_k\\) hits every hedge formed for \\(S_\ell\\). Since such a \\(k\\) exists for every \\(S_\ell\\), we will conclude that \\(\boldsymbol{\mathcal{I}}\\) is feasible for  
efeq:opt by  
efprp:genid. Finally, to show that \\(\mathcal{I}_k\\) hits every hedge formed for \\(S_\ell\\), note that satisfiability of all clauses containing the literal \\(\lnot z_{k,\ell}^*\\) reduces to the satisfiability of \\((\lnot x_{i,j-1,k,\ell}\lor x_{i,j,k,\ell}\lor\lnot x_{p,j,k,\ell})\\) (see lines 16, 19, or 20), and the same arguments as in the proof of  
efthm:sat apply. ◻

</div>

## Results of efsec:adjustment

<div class="proof" markdown="1">

*Proof.* Define \\(S=\mathrm{Anc}_{V\setminus X}(Y)\\). First, we show that \\(\mathrm{Pa}(S)\subseteq X\\). Assume the contrary, i.e., there is a vertex \\(w\in\mathrm{Pa}(S)\setminus X\\). Clearly \\(w\\) has a directed path to \\(S\\) (a direct edge) that does not go through \\(X\\). This implies that \\(w\in\mathrm{Anc}_{V\setminus X}(S)\\), and since by definition, \\(S=\mathrm{Anc}_{V\setminus X}(Y)\\), \\(w\in\mathrm{Anc}_{V\setminus X}(Y)=S\\). However, the latter contradicts with \\(w\in\mathrm{Pa}(S)\\). Second, we note that from the third rule of do calculus `\citep{pearl2009causality}`{=latex}, \\(\mathbb{P}_{W}(S)=\mathbb{P}_{\mathrm{Pa}(S)}(S)\\) for any \\(W\supseteq \mathrm{Pa}(S)\\). Combining the two arguments, we have the following: \\[\label{eq:wtox}
        \mathbb{P}_{W}(S)=\mathbb{P}_X(S),\quad\forall W\supseteq\mathrm{Pa}(S).\\] To proceed, we will use the following proposition.

<div class="proposition" markdown="1">

**Proposition 3** (`\citealp{lauritzen1990independence}`{=latex}). *Let \\(S, R\\), and \\(Z\\) be disjoint subsets of vertices in a directed acyclic graph \\(\mathcal{G}\\). Then \\(Z\\) d-separates \\(S\\) from \\(R\\) if and only if \\(Z\\) is a vertex cut between \\(S\\) and \\(R\\) in \\((\mathcal{G}[\mathrm{Anc}(S\cup R\cup Z)])^m\\).*

</div>

Choose \\(R=\mathrm{Pa}(S)\\) in the proposition above. Since \\(Z\subseteq\mathrm{Anc}(S)\\), we have that \\(\mathrm{Anc}(S\cup R\cup Z) = \mathrm{Anc}(S)\\). From condition (ii) in the lemma, \\(Z\\) is a vertex cut between \\(S\\) and \\(R\\) in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^m\\), which implies it is also a vertex cut in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}}[\mathrm{Anc}(S)])^m\\), as every path in the latter graph exists in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^m\\). Using the proposition above, \\(Z\\) d-separates \\(S\\) and \\(\mathrm{Pa}(S)\\) in \\(\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}}\\). This is to say, \\(Z\\) blocks all non-causal paths from \\(\mathrm{Pa}(S)\\) to \\(S\\) in \\(\mathcal{G}_{\overline{\mathcal{I}}}\\), and it clearly has no elements that are descendants of \\(\mathrm{Pa}(S)\\). Therefore, \\(Z\\) satisfies the adjustment criterion of `\citet{shpitser2010validity}`{=latex} w.r.t. \\(\mathbb{P}_{\mathrm{Pa}(S)}(S)\\) in \\(\mathcal{G}_{\overline{\mathcal{I}}}\\). That is, the following holds: \\[\mathbb{P}_{\mathcal{I}\cup\mathrm{Pa}(S)}(S) = \mathbb{E}_{\mathbb{P}_{\mathcal{I}}}[\mathbb{P}_{\mathcal{I}}(S\mid \mathrm{Pa}(S), Z)],\\] where the expectation is w.r.t. \\(\mathbb{P}_\mathcal{I}(Z)\\). Choosing \\(W=\mathcal{I}\cup\mathrm{Pa}(S)\\) in  
efeq:wtox, we get \\[\mathbb{P}_X(S) = \mathbb{E}_{\mathbb{P}_{\mathcal{I}}}[\mathbb{P}_{\mathcal{I}}(S\mid \mathrm{Pa}(S), Z)].\\] Marginalizing \\(S\setminus Y\\) out in both sides of the equation above, we have \\[\mathbb{P}_X(Y) = \mathbb{E}_{\mathbb{P}_{\mathcal{I}}}[\mathbb{P}_{\mathcal{I}}(Y\mid \mathrm{Pa}(S), Z)].\\] The last step of the proof is to show that \\(\mathrm{Pa}(S)=X\\). We already showed that \\(\mathrm{Pa}(S)\subseteq X\\). For the other direction, we will use the minimality of \\(X\\). Suppose to the contrary that \\(x\in X\setminus\mathrm{Pa}(S)\\). We first show that every causal path from \\(x\\) to \\(Y\\) goes through \\(X\setminus\{x\}\\). Suppose not. Then take a causal path \\(x,s_1,\dots,s_m, y\\) be a causal path from \\(x\\) to \\(y\in Y\\). Note that \\(s_1\\) has a causal path to \\(Y\\) that does not go through \\(X\\). By definition, \\(s_1\in S\\), which implies \\(x\in\mathrm{Pa}(S)\\), which is a contradiction. Therefore, every causal path from \\(x\\) to \\(Y\\) goes through \\(X\setminus\{x\}\\), and consequently, there is no causal path from \\(x\\) to \\(Y\\) in \\(\mathcal{G}_{\overline{X}}\\). Clearly there is no *backdoor* path either. Every other path has a collider on it, and therefore is blocked by \\(X\setminus\{x\}\\) – note that none of these can be colliders in \\(\mathcal{G}_{\overline{X}}\\). Therefore, \\(\{x\}\\) is d-separated from \\(Y\\) given \\(X\setminus\{x\}\\) in \\(\mathcal{G}_{\overline{X}}\\), which contradicts the minimality of \\(X\\) w.r.t. the third rule of do calculus. This shows \\(X\subseteq \mathrm{Pa}(S)\\), completing the proof. ◻

</div>

<div class="proof" markdown="1">

*Proof.* For the first part, using  
eflem:vtxcut, it suffices to show that \\(Z\\) is a vertex cut between \\(S\\) and \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^m\\). Suppose not. That is, there exists a path from \\(S\\) to \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^m\\) that does not pass through any member of \\(Z\\). Let \\(P = s, v^1,\dots, v^l, x\\) represent this path, where \\(s\in S\\) and \\(x\in\mathrm{Pa}(S)\\). We construct a corresponding path \\(P'\\) in \\(\mathcal{G}^{vc}\\) as follows. The first vertex on \\(P'\\) is \\(s_1\\), which corresponds to the first vertex on \\(P\\), \\(s\\). We then walk along \\(P\\) and add a path to \\(P'\\) corresponding to each edge we traverse on \\(P\\) as follows. Consider this edge to be \\(\{v,w\}\\) – for instance, the first edge would be \\(\{s,v^1\}\\). By definition of \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^m\\), for every pair of adjacent vertices \\(v,w\\) on the path \\(P\\), one of the following holds: (i) \\(v\to w\\) in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^d\\), (ii) \\(v\gets w\\) in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^d\\), or (iii) \\(v\\) and \\(w\\) have a common child \\(t\\) in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^d\\). In case (i), we add \\(v_1,w_2,w_1\\). In case (ii), we add \\(v_2,w_1\\). Finally, in case (iii), we add \\(v_1,t_2,w_1\\) to \\(P'\\). We continue this procedure until we traverse all edges on \\(P\\). The last vertex on \\(P'\\) is \\(x_2\\), as x has no children in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^d\\). Finally we add \\(x_1\\) to this path, as \\(x_2\\) and \\(x_1\\) are always connected by construction. Note that by construction of \\(P'\\), any vertex that appears with index \\(2\\) has a parent in \\(v\to w\\) in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^d\\), and therefore is not a member of \\(\mathcal{I}\\). Hence, \\(\{v_2:v\in\mathcal{I}\}\\) does not intersect with \\(P'\\). Further, \\(\{v_1:v\in Z\}\\) does not intersect with \\(P'\\) either, as none of the vertices appearing on \\(P'\\) correspond to \\(Z\\). This is to say that \\(J=\{v_2:v\in\mathcal{I}\}\cup\{v_1:v\in Z\}\\), which is the solution obtained by  
efalg:adj in line 10, does not cut the path \\(P'\\). This contradicts with \\(J\\) being a vertex cut.

For the second part, let \\((\mathcal{I}', Z')\\) be so that \\(Z'\\) is a vertex cut between \\(S\\) and \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{\mathcal{I}'}\underline{\mathrm{Pa}(S)}})^m\\), and \\(\mathcal{I}'\\) induces a lower cost than \\(\mathcal{I}\\); that is, \\(C(\mathcal{I}')<C(\mathcal{I})\\). Define \\(J'=\{v_2:v\in\mathcal{I}'\}\cup\{v_1:v\in Z'\}\\). Clearly, the cost of \\(J'\\) is equal to \\(C(\mathcal{I}')\\), which is lower than the cost of minimum vertex cut found in line 10 of  
efalg:adj. It suffices to show that \\(J'\\) is also a vertex cut between \\(\{x_1:x\in\mathrm{Pa}(S)\}\\) and \\(\{s_1:s\in S\}\\) in \\(\mathcal{G}^{vc}\\) to arrive at a contradiction. Suppose not; that is, there is a path \\(P = s_1,\dots, x_1\\) on \\(\mathcal{G}^{vc}\\) that \\(J'=\{v_2:v\in\mathcal{I}'\}\cup\{v_1:v\in Z'\}\\) does not intersect. None of the vertices with index \\(2\\) on \\(P\\) belong to \\(\mathcal{I}'\\), and none of the vertices with index \\(1\\) belong to \\(Z'\\). Analogous to the first part, we construct a corresponding path \\(P'\\) – this time in \\((\mathcal{G}_{\overline{\mathcal{I}'}\underline{\mathrm{Pa}(S)}})^m\\). The starting vertex on \\(P'\\) is \\(s\\), which corresponds to \\(s_1\\), the initial vertex on \\(P\\). Let us imagine a cursor on \\(s_1\\). We then sequentially build \\(P'\\) by traversing \\(P\\) as follows. We always look at sequences starting with \\(v_1\\) (where the cursor is located): when the sequence is of the form \\(v_1, w_2, w_1\\) or \\(v_1, v_2, w_1\\), we add \\(w\\) to \\(P'\\), and move the cursor to \\(w_1\\); however, when the sequence is of the form \\(v_1, w_2, r_1\\), we add \\(r_1\\) to \\(P'\\) and move the cursor to \\(r_1\\). By construction of \\(\mathcal{G}^{vc}\\), no other sequence is possible – note that there are no edges between \\(v_1\\) and \\(w_1\\) or \\(v_2\\) and \\(w_2\\) where \\(v\\) and \\(w\\) are distinct. Since none of the vertices with index \\(2\\) on \\(P\\) belong to \\(\mathcal{I}\\), in the first case, the corresponding edge \\(v\gets w\\) or \\(v\to w\\) is present in \\((\mathcal{G}_{\overline{\mathcal{I}'}\underline{\mathrm{Pa}(S)}})^d\\) and consequently, the edge \\(\{v,w\}\\) is present in \\((\mathcal{G}_{\overline{\mathcal{I}'}\underline{\mathrm{Pa}(S)}})^m\\); and in the latter case, both edges \\(v\to w\\) and \\(w\gets r\\) are present in \\((\mathcal{G}_{\overline{\mathcal{I}'}\underline{\mathrm{Pa}(S)}})^d\\) and consequently, the edge \\(\{v,r\}\\) is present in \\((\mathcal{G}_{\overline{\mathcal{I}'}\underline{\mathrm{Pa}(S)}})^m\\). \\(P'\\) is therefore a path between \\(S\\) and \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{\mathcal{I}'}\underline{\mathrm{Pa}(S)}})^m\\). Notice that by construction, only those vertices appear on \\(P'\\) that their corresponding vertex with index \\(1\\) appears on \\(P\\) – the *cursor* always stays on vertices with index \\(1\\). As argued above, none of such vertices belong to \\(Z'\\), which means \\(Z'\\) does not intersect with \\(P'\\) which is a path from \\(S\\) to \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{\mathcal{I}'}\underline{\mathrm{Pa}(S)}})^m\\). This contradicts with \\(Z'\\) being a vertex cut. ◻

</div>

### Proof of efrem:heuristic

<div class="proof" markdown="1">

*Proof.* Since the algorithms are run in the hedge hull of \\(S\\), assume without loss of generality that \\(V=\mathcal{H}_\mathcal{G}(S)\\), i.e., \\(V\\) is the hedge hull of \\(S\\). From  
efthm:algadj,  
efalg:adj finds the optimal (minimum-cost) intervention \\(\mathcal{I}\\) such that there exists a set \\(Z\subseteq V\\) that is a vertex cut between \\(S\\) and \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{\mathcal{I}}\underline{\mathrm{Pa}(S)}})^m\\). To prove that the cost of the solution returned by  
efalg:adj is always smaller than or equal to that of heuristic algorithm 1 proposed by `\citet{akbari-2022}`{=latex}, it suffices to show that the solution of their algorithm is a feasible point for the statement above. That is, denoting the output of heuristic algorithm 1 in `\citet{akbari-2022}`{=latex} by \\(I_1\\), we will show that there exist sets \\(Z_1\subseteq V\\) such that it is a vertex cut between \\(S\\) and \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{I_1}\underline{\mathrm{Pa}(S)}})^m\\).

Heuristic algorithm 1: This algorithm returns an intervention set \\(I_1\\) such that there is no bidirected path from \\(\mathrm{Pa}(S)\\) to \\(S\\) in \\(\mathcal{G}_{\overline{I_1}}\\). We claim \\(Z_1=V\setminus \mathrm{Pa}(S)\setminus S\\) satisfies the criterion above. To prove this, consider an arbitrary path \\(P\\) between \\(S\\) and \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{I_1}\underline{\mathrm{Pa}(S)}})^m\\). If there is an observed vertex \\(v\in V\\) on \\(P\\), this vertex is included in \\(Z_1\\) and separates the path. So it suffices to show that there is no path \\(P\\) between \\(S\\) and \\(\mathrm{Pa}(S)\\) in \\((\mathcal{G}_{\overline{I_1}\underline{\mathrm{Pa}(S)}})^m\\) where all the intermediate vertices on \\(P\\) correspond to unobserved confounders. Suppose there is. That is, \\(P=x, u_1,\dots, u_m, s\\), where \\(x\in\mathrm{Pa}(S)\\) and \\(\{u_i\}_{i=1}^m\\) are unobserved. Since \\(u_i\\) is not connected to \\(u_j\\) in \\((\mathcal{G}_{\overline{I_1}\underline{\mathrm{Pa}(S)}})^d\\), it must be the case that any \\(u_i\\) and \\(u_{i+1}\\) have a common child \\(v_{i,j}\\) in \\((\mathcal{G}_{\overline{I_1}\underline{\mathrm{Pa}(S)}})^d\\). This is to say, there is a path \\(x\gets u_1\to v_{1,2}\gets u_2\to \dots\to v_{m-1,m}\gets u_m\to s\\) in \\((\mathcal{G}_{\overline{I_1}\underline{\mathrm{Pa}(S)}})^d\\), which corresponds to the bidirected path \\(x, v_{1,2},\dots,v_{m-1,m}, s\\) in \\(\mathcal{G}_{\overline{I_1}}\\). This contradicts with the fact that there is no bidirected path between \\(\mathrm{Pa}(S)\\) and \\(S\\) in \\(\mathcal{G}_{\overline{I_1}}\\). ◻

</div>

## Results in the appendices

<div class="restatable" markdown="1">

lemmalemsubmod<span id="lem:submodular" label="lem:submodular"></span> For any district \\(S\subseteq V\\), the function \\(f_S:2^{V\setminus S}\to \mathbb{Z}^{\leq0}\\) is submodular.

</div>

<div class="proof" markdown="1">

*Proof.* Take two distinct vertices \\(\{x,y\}\in V\setminus S\\) and an arbitrary set \\(I\subset V\setminus S\setminus\{x,y\}\\). It suffices to show that \\[f_S(I\cup\{x,y\}) - f_S(I\cup\{y\})
        \leq
        f_S(I\cup\{x\}) - f_S(I).\\] By definition, the right hand side is the number of hedges \\(H\subseteq V\setminus I\\) formed for \\(S\\) such that \\(x\notin H\\). Similarly, the left hand side counts the number of hedges \\(H\subseteq V\setminus(I\cup\{y\})\\) formed for \\(S\\) such that \\(x\notin H\\). The inequality holds because the set of hedges counted by the left hand side is a subset of that on the right hand side. ◻

</div>

<div class="restatable" markdown="1">

propositionprpsubmod<span id="prp:submod" label="prp:submod"></span> The combinatorial optimization of  
efeq:opt2 is equivalent to the following unconstrained submodular optimization problem. \\[\label{eq:subm}
        \mathcal{I}^*=\mathop{\mathrm{argmax}}_{\mathcal{I}\subseteq V\setminus S} \left(f_S(\mathcal{I})-\frac{\sum_{v\in \mathcal{I}}C(v)}{1+\sum_{v\in V\setminus S}C(v)}\right).\\]

</div>

<div class="proof" markdown="1">

*Proof.* The submodularity of the objective function follows from  
eflem:submodular. To show the equivalence of the two optimization problems, we show that a maximizer \\(\mathcal{I}^*\\) of  
efeq:subm (i) hits every hedge formed for \\(S\\), and (ii) has the optimal cost among such sets.

*Proof of ‘(i):’* Note that \\(f_S(\mathcal{I})=0\\) if and only if there are no hedges formed for \\(S\\) in \\(\mathcal{G}[V\setminus \mathcal{I}^*]\\), or equivalently, \\(\mathcal{I}\\) hits every hedge formed for \\(S\\). So it suffices to show that \\(f_S(\mathcal{I}^*)=0\\) for every maximizer \\(\mathcal{I}^*\\) of  
efeq:subm. To this end, first note that \\[f_S(V\setminus S)-\frac{\sum_{v\in V\setminus S}C(v)}{1+\sum_{v\in V\setminus S}C(v)}=0-1+\frac{1}{1+\sum_{v\in V\setminus S}C(v)}>-1,\\] which implies that \\[f_S(\mathcal{I}^*)-\frac{\sum_{v\in \mathcal{I}^*}C(v)}{1+\sum_{v\in V\setminus S}C(v)}>-1.\\] On the other hand, clearly \\(\frac{\sum_{v\in \mathcal{I}^*}C(v)}{1+\sum_{v\in V\setminus S}C(v)}\geq0\\), which combined with the inequality above implies \\(f_S(\mathcal{I}^*)>-1\\). Since \\(f_S(\mathcal{I}^*)\in\mathbb{Z}^{\leq0}\\), it is only possible that \\(f_S(\mathcal{I}^*)=0\\).

*Proof of ‘(ii):’* We showed that \\(f_S(\mathcal{I}^*)=0\\). So \\(\mathcal{I}^*\\) maximizes \\(\frac{\sum_{v\in \mathcal{I}}C(v)}{1+\sum_{v\in V\setminus S}C(v)}\\) among all those \\(I\\) such that \\(f_S(I)=0\\). Since the denominator is a constant, this is equivalent to minimizing \\(C(I)=\sum_{v\in I}C(v)\\) among all those \\(I\\) that hit all the hedges formed for \\(S\\), which matches the optimization of  
efeq:opt2. ◻

</div>

# Alternative Formulations [app:reform]

## Min-cost intervention as a submodular function maximization problem [sec:reform-submodular]

In this Section, we reformulate the minimum-cost intervention design as a submodular optimization problem. Submodular functions exhibit a property akin to diminishing returns: the incremental gain from adding an element to a set decreases as the set grows `\citep{nemhauser-1978}`{=latex}.

<div id="def:submodular" class="definition" markdown="1">

**Definition 5**. *A function \\(f: 2^V \to \mathbb{R}\\) is submodular if for all \\(A \subseteq B \subseteq V\\) and \\(v \in V \setminus B\\), we have that \\(f(A \cup \{v\}) - f(A) \geq f(B \cup \{v\}) - f(B)\\).*

</div>

Given an ADMG \\(\mathcal{G}=\langle V,\overrightarrow{E}, \overleftrightarrow{E}\rangle\\) a district \\(S\\) in \\(\mathcal{G}\\), and an arbitrary set \\(\mathcal{I}\subseteq V\setminus S\\), we define \\(f_S(\mathcal{I})\\) as the negative count of hedges formed for \\(S\\) in \\(\mathcal{G}[V\setminus\mathcal{I}]\\). Note that \\(g_S:2^{V\setminus S}\to \mathbb{R}\\), where \\(g_S(\mathcal{I})\coloneqq f_S(\mathcal{I}) + \alpha\sum_{v\in\mathcal{I}}C(v)\\), and \\(\alpha\\) is an arbitrary constant, is also submodular as the second component is a modular function (similar definition as in <a href="#def:submodular" data-reference-type="ref" data-reference="def:submodular">5</a> only with equality instead of inequality.).

## Min-cost intervention as an RL problem [sec:reform-rl]

We model the MCID problem given a graph \\(\mathcal{G}= (V,E)\\) and \\(S \subset V\\), as a Markov decision process (MDP), where a vertex is removed in each step \\(t\\) until there are no hedges left. The goal is to minimize the cost of the removed vertices (i.e., intervention set). Naturally, the action space is the set of vertices, \\(V\\) and the state space is the set of all subsets of \\(V\\). More precisely, let \\(s_t\\) and \\(a_t\\) denote the state and the action of the MDP at iteration \\(t\\), respectively. Then, \\(s_t\\) is the hedge hull for \\(S\\) from the remaining vertices at time \\(t\\), and action \\(a_t\\) is the vertex that will be removed from \\(V_t\\) in that iteration. Consequently, the state transition due to action \\(a_t\\) is \\(s_{t+1}=\mathcal{H}_\text{hull}(V_t \setminus \{a_t\})\\). The immediate reward of selecting action \\(a_t\\) at state \\(s_t\\) will be the negative of the cost of removing (i.e., intervening on) \\(a_t\\), given by

\\[\begin{aligned}
  r(s_t, a_t) = -C(a_t).
\end{aligned}\\]

The MDP terminates when there are no hedges left and the hedge hull of the remaining vertices is empty (i.e., \\(s_t = \emptyset\\)). The goal is to find a policy \\(\pi\\) that maximizes sum of the rewards until the termination of the MDP. Formally, the goal is to solve

\\[\begin{aligned}
  \mathop{\mathrm{argmax}}_{\pi} \left[\sum_{t=1}^{T} r(s_t, a_t) \right],
\end{aligned}\\] where \\(s_1 = V\\) and \\(T\\) is the time step at which the MDP terminates (i.e., \\(s_T = \emptyset\\)).

[^1]: See  
    efsec:problem the definition of a district.

[^2]: This interventional distribution is often mistakenly referred to as the *causal effect* of \\(X\\) on \\(Y\\). However, a causal effect, such as an average treatment effect or a quantile treatment effect, is usually a specific functional of this probability distribution for different values of \\(\mathrm{x}\\).

[^3]: Although it only makes sense to assign non-negative costs to interventions, adopting non-negative costs is without loss of generality. If certain intervention costs are negative, one can shift all the costs equally so that the most negative cost becomes zero. This constant shift would not affect the minimization problem in any way.

[^4]: This is to say, the third rule of do calculus does not apply to \\(\mathbb{P}_X(Y)\\). More precisely, there is no \\(x\in X\\) such that \\(Y\\) is d-separated from \\(x\\) given \\(X\setminus\{x\}\\) in \\(\mathcal{G}_{\overline{X}}\\).

[^5]: This corresponds to \\(Z\\) blocking all the backdoor paths between \\(\mathrm{Pa}(S)\\) and \\(S\\), in the modified graph \\(\mathcal{G}_{\overline{\mathcal{I}}}\\).

[^6]: The computational bottleneck of the algorithm is an instance of minimum vertex cut problem, which can be solved using any off-the-shelf max-flow algorithm.
