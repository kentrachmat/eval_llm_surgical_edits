# On the Sparsity of the Strong Lottery Ticket Hypothesis

## Abstract

Considerable research efforts have recently been made to show that a random neural network \\(N\\) contains subnetworks capable of accurately approximating any given neural network that is sufficiently smaller than \\(N\\), without any training. This line of research, known as the Strong Lottery Ticket Hypothesis (SLTH), was originally motivated by the weaker Lottery Ticket Hypothesis, which states that a sufficiently large random neural network \\(N\\) contains *sparse* subnetworks that can be trained efficiently to achieve performance comparable to that of training the entire network \\(N\\). Despite its original motivation, results on the SLTH have so far not provided any guarantee on the size of subnetworks. Such limitation is due to the nature of the main technical tool leveraged by these results, the Random Subset Sum (RSS) Problem. Informally, the RSS Problem asks how large a random i.i.d. sample \\(\Omega\\) should be so that we are able to approximate any number in \\([-1,1]\\), up to an error of \\(\varepsilon\\), as the sum of a suitable subset of \\(\Omega\\).

We provide the first proof of the SLTH in classical settings, such as dense and equivariant networks, with guarantees on the sparsity of the subnetworks. Central to our results, is the proof of an essentially tight bound on the Random Fixed-Size Subset Sum Problem (RFSS), a variant of the RSS Problem in which we only ask for subsets of a given size, which is of independent interest.

# Introduction [sec:intro]

The Lottery Ticket Hypothesis (LTH) is a research direction that has attracted considerable attention over the years, stemming from the empirical contrast between the fact that, while large neural networks can be successfully trained to achieve good performance on a given task and successively pruned to a great level of sparsity without compromising their performance, researchers have struggled to train sparse neural networks from scratch. The authors of `\cite{frankleLotteryTicketHypothesis2018}`{=latex} observed that, using a simple pruning strategy (namely Iterative Magnitude Pruning while rewinding the original weights of the remaining edges to their value at initialization), *starting from a sufficiently large random neural networks, it is possible to identify sparse subnetworks that can be trained to achieve the performance achievable by the starting network* (see Figure <a href="#fig:LTH" data-reference-type="ref" data-reference="fig:LTH">2</a> in the appendix for an illustration). The previous statement, namely the LTH, soon gave rise to an even stronger one, corroborated by empirical works `\cite{zhouDeconstructingLotteryTickets2019,ramanujanWhatHiddenRandomly2020}`{=latex} which proposed “training-by-pruning" algorithms (see Section <a href="#sec:related" data-reference-type="ref" data-reference="sec:related">2</a> for details), providing evidence that *starting from a sufficiently large random neural networks, it is possible to identify sparse subnetworks that exhibit good performance as they are, without changing the original weights* (see Figure <a href="#fig:truning" data-reference-type="ref" data-reference="fig:truning">3</a> in the appendix for an illustration). By removing the need to analyze the dynamics of training, the last statement, namely the Strong Lottery Ticket Hypothesis (SLTH), allowed a fruitful series of rigorous proofs for increasingly more general architectures (see Section <a href="#sec:related" data-reference-type="ref" data-reference="sec:related">2</a> for an overview). Such rigorous results can informally be stated as follows:

<div id="thm:informal" class="theorem" markdown="1">

**Theorem 1** (Informal statement of previous SLTH results). *With high probability, a random artificial neural network \\(N_\Omega\\) with \\(m\\) parameters can be pruned so that the resulting subnetwork \\(N_S\\) \\(\varepsilon\\)-approximates (i.e., approximates up to an error \\(\varepsilon\\)) any target artificial neural network \\(N_t\\) with \\(O\left(m/\log_2(1/\varepsilon)\right)\\) parameters.*

</div>

It is important to note that, to this day, we only have proofs on the existence of such subnetworks, also called winning tickets, but it remains an open question how to find them reliably.

*All theoretical results on the SLTH however have so far not investigated the interplay between the sparsity of the winning ticket \\(N_S\\) and the size of the random neural network \\(N_\Omega\\).* This is in contrast to the original motivation of the LTH and to the practical application of the aforementioned training-by-pruning algorithms that motivated the SLTH, such as `\cite{isikSparseRandomNetworks2022,isikAdaptiveCompressionFederated2024}`{=latex}. In fact, to approximate target networks with \\(O\left(m/\log_2(1/\varepsilon)\right)\\) parameters, essentially all winning tickets \\(N_S\\) have \\(\Theta(m)\\) parameters (see Appendix <a href="#apx:pensia_lower" data-reference-type="ref" data-reference="apx:pensia_lower">7</a>), thus being roughly of the same size of the original network \\(N_\Omega\\). We thus ask the following natural question:

> If we want to \\(\varepsilon\\)-approximate a family of target artificial neural networks with \\(m_t\\) parameters by pruning a fraction \\(\alpha\\), called sparsity, of the \\(m\\) parameters of a random artificial neural network \\(N_\Omega\\), how big should \\(m\\) be?

We are particularly interested in the regime in which the density parameter \\(\gamma = 1 - \alpha\\) vanishes as the size of the network increases, so that the size of the winning ticket \\(N_S\\) is \\(\gamma m = o(m)\\).

The above question has so far remained unanswered as a consequence of the limitation inherited from the core technical tool that has been leveraged so far to prove SLTH results, namely the Random Subset Sum (RSS) Problem `\cite{Lueker98}`{=latex}. Informally, the RSS asks how large a random i.i.d. sample \\(\Omega\\) should be so that we are able to approximate any number in \\([-1,1]\\) as the sum of a suitable subset of \\(\Omega\\). The applicability of RSS to the SLTH was first recognized by `\cite{Pensia}`{=latex} within the proof strategy previously developed in `\cite{malachProvingLotteryTicket2020}`{=latex}.

## Our Contribution [sec:contrib]

We answer the aforementioned question by introducing and proving a refined variant of the RSS Problem, namely the Random Fixed-Size Subset Sum Problem (RFSS), in which the approximation of the target values should be achieved by only considering subsets of fixed size \\(k\\) from a set of \\(n\\) samples (Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a>). We focus on subsets of fixed size \\(k\\) rather than subsets of size up to \\(k\\) for two main reasons. From a theoretical point of view, it is a stronger requirement, and practically speaking, using fixed-size subsets enables us to achieve SLTH results where the layers of the lottery ticket exhibit a uniform structure, potentially offering a computational advantage in their implementation.

In Section <a href="#sec:slth" data-reference-type="ref" data-reference="sec:slth">4</a>, we show how the density \\(\gamma\\) impacts the *overparameterization*, i.e., the ratio \\((\nicefrac{m}{m_t})\\) between the number of parameters of the original network \\(N_\Omega\\) and that of the class of target networks \\(N_t\\) that can be \\(\varepsilon\\)-approximated by pruning \\(N_\Omega\\) down to a subnetwork \\(N_S\\) with \\(\gamma m\\) parameters. In our analysis, we also compare and recover as special cases previous SLTH results such as `\cite{Pensia,malachProvingLotteryTicket2020,dacunhaProvingStrongLottery2022,burkholzConvolutionalResidualNetworks2022,ferbachGeneralFrameworkProving2022}`{=latex}. For instance, when \\(\gamma m = \Theta(m)\\), we recover up to a logarithmic factor the result of `\cite{Pensia}`{=latex}, which states that the overparameterization needed is \\(O(\log_2{\left({\nicefrac{m_t^2}{\varepsilon^2}}\right)})\\). In the case of Dense Neural Networks, Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a> thus bridges the gap between the two extreme cases of \\(\gamma m = \Theta(m_t)\\) and \\(\gamma m = \Theta(m)\\) considered in `\cite{malachProvingLotteryTicket2020}`{=latex} and `\cite{Pensia}`{=latex}, respectively. It is worth noting that `\cite{Pensia}`{=latex} is often considered an improvement over `\cite{malachProvingLotteryTicket2020}`{=latex}, as it exponentially reduces the overparameterization, albeit at the cost of a trivial sparsity level. Finally, we prove that our bounds on the overparameterization as a function of the subnetwork sparsity are essentially tight.

#### Organization of the paper.

After reviewing the literature on the SLTH in Section <a href="#sec:related" data-reference-type="ref" data-reference="sec:related">2</a>, we introduce the Random Fixed-Size Subset Sum Problem in Section <a href="#sec:RFSS" data-reference-type="ref" data-reference="sec:RFSS">3</a>. In Section <a href="#sec:slth" data-reference-type="ref" data-reference="sec:slth">4</a>, we explore some applications of the RFSS Problem to the SLTH, and finally draw our conclusions in Section <a href="#sec:conclusions" data-reference-type="ref" data-reference="sec:conclusions">5</a>. Some limitations of our work, along with its potential impact, are discussed in Section <a href="#sec:lim_impact" data-reference-type="ref" data-reference="sec:lim_impact">6</a>.

# Related Work [sec:related]

The SLTH is named after the LTH, which was introduced by Frankle and Carbin in `\cite{frankleLotteryTicketHypothesis2018}`{=latex}. At the time of writing, this paper has received over 3,300 citations, attesting to the significance and impact of the research topic. Surveying the LTH is thus besides the scope of this work, and we defer the reader to dedicated surveys such as `\cite{liuSurveyLotteryTicket2024}`{=latex}.

The SLTH was empirically motivated by work investigating training-by-pruning algorithms such as `\cite{zhouDeconstructingLotteryTickets2019,ramanujanWhatHiddenRandomly2020}`{=latex}, namely algorithms that leverage the gradient of the network parameters to learn a good *mask* of the edges to be retained (i.e., a good subnetwork, called the winning ticket). `\cite{zhouDeconstructingLotteryTickets2019}`{=latex} achieves this by learning a probability associated to each edge, which is then used to sample the edges that should be included in the subnetwork. `\cite{ramanujanWhatHiddenRandomly2020}`{=latex} gets rid of the stochasticity involved in the aforementioned strategies by learning a score associated to each edge; the subnetwork is then determined by including the edges with the highest score. Such strategies are leveraged in `\cite{isikSparseRandomNetworks2022,isikAdaptiveCompressionFederated2024}`{=latex} in a federated learning setting, in order to improve the communication cost of distributed training by communicating the sampled masks of a fixed shared network, rather than the entire weights. However, these training-by-pruning algorithms are generally not computationally less expensive than classical training, since they also make use of backpropagation to update scores and are applied to a sufficiently large network to find a winning ticket. To reduce the computational cost of finding a good subnetwork, `\cite{gadhikarWhyRandomPruning2023}`{=latex} shows, both theoretically and experimentally, that randomly pre-pruning the source network before looking for a winning ticket can be an effective approach. In `\cite{otsukaPartialSearchFrozen2024a}`{=latex}, on top of randomly pruning the source network, some parameters are also frozen. Frozen parameters are forced to be part of the winning ticket and they do not have an associated score, which effectively reduces the search space for the training-by-pruning algorithms.

The first rigorous proof of the SLTH in the case of dense neural networks has been provided by `\cite{malachProvingLotteryTicket2020}`{=latex}, which establishes a framework that was inherited by the subsequent works. `\cite{Pensia}`{=latex} crucially shows that the framework in `\cite{malachProvingLotteryTicket2020}`{=latex} allows the application of the RSS analysis in `\cite{Lueker98}`{=latex}, proving that, with no constraint on the size of the subnetworks, a random network with \\(m\\)-parameters can be pruned to approximate target networks with \\(m/\log(1/\varepsilon)\\) parameters (we defer the reader to Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a> for details on further constraints on the parameters). An alternative proof of the result in `\cite{Pensia}`{=latex} was simultaneously shown in `\cite{orseauLogarithmicPruningAll2020}`{=latex}. `\cite{dacunhaProvingStrongLottery2022}`{=latex} and `\cite{burkholzConvolutionalResidualNetworks2022}`{=latex} successively extended `\cite{Pensia}`{=latex} and `\cite{orseauLogarithmicPruningAll2020}`{=latex} to convolutional neural networks (CNNs). By leveraging multidimensional generalizations of RSS `\cite{dacunhaRevisitingRandomSubset2023,borstIntegralityGapBinary2023}`{=latex}, `\cite{cunhaPolynomiallyOverParameterizedConvolutional2023}`{=latex} further extended the SLTH to structured pruning of CNNs and, as a special case, dense networks. Finally, `\cite{ferbachGeneralFrameworkProving2022}`{=latex} provided a general framework that proves the SLTH for equivariant networks.

As for refinements and generalizations of the above results, `\cite{burkholzMostActivationFunctions2022}`{=latex} shows that, at the cost of a quadratic overhead in the overparameterization w.r.t. `\cite{Pensia}`{=latex}, the number of layers of the random network \\(N_\Omega\\) can be reduced to \\(\ell+1\\), where \\(\ell\\) is the number of layers of the target networks \\(N_z\\); furthermore, while previous results only considered networks with ReLU activation, `\cite{burkholzMostActivationFunctions2022}`{=latex} shows how to extend the proof in `\cite{Pensia}`{=latex} to a more general class of activations functions. `\cite{burkholzExistenceUniversalLottery2022}`{=latex} introduces the notion of universal lottery ticket, and show that it is possible to prune a sufficiently overparameterized random network so that the resulting subnetwork (the lottery ticket) can approximate certain class of functions up to an affine transformation of the output of the subnetwork (in this sense being universal). `\cite{fischerLotteryTicketsNonzero2022}`{=latex} shows how to extend the proof in `\cite{Pensia}`{=latex} when neurons have random biases, and adapts the training-by-pruning algorithm of `\cite{ramanujanWhatHiddenRandomly2020}`{=latex} to find a strong lottery ticket with a desired sparsity level. Motivated by theoretical insights on the existence of sparse strong lottery tickets, `\cite{fischerPlantSeekCan2022}`{=latex} develops a framework to plant the latter in large random network and investigates training-by-pruning algorithms, providing evidence that sparse strong lottery tickets typically exists for common machine learning tasks, and the difficulty to find them is of algorithmic nature.

Our proof of the RFSS Problem in Section <a href="#sec:RFSS" data-reference-type="ref" data-reference="sec:RFSS">3</a> is based on the second moment method approach first explored by `\cite{Lueker82}`{=latex}, and which has recently been refined to prove multidimensional generalizations of RSS by `\cite{dacunhaRevisitingRandomSubset2023}`{=latex} and `\cite{borstIntegralityGapBinary2023}`{=latex}.

# Fixed-Size Random Subset Sum [sec:RFSS]

In this section we present our technical contributions on the RFSS, which are the foundation of our proofs regarding the sparsity of the SLTH.

Let us start by introducing some notation. We denote by \\([n]\\) the set \\(\{1, \ldots, n\}\\), for \\(n \in \mathbb{N}\\). Given a set \\(\Omega=\left\{ X_{1},...,X_{n}\right\}\\) and a set of indices \\(S\subseteq\left[n\right]\\) we define \\(\Sigma_{S}^{\Omega}=\sum_{i\in S}X_{i}\\), and we omit \\(\Omega\\) when clear from the context. We now define a class of distributions for which our RFSS result holds.

<div id="def:quasi_unif" class="definition" markdown="1">

**Definition 1** (sum-bounded). *We say that a probability density function \\(f\\) is *sum-bounded* if there exist positive constants \\(c_{l}\\) and \\(c_{u}\\) such that, for all \\(k\in\mathbb{N}\\), given \\(k\\) independent samples \\(X_{1},...,X_{k}\\) with density \\(f\\), the density of their sum \\(f_{\Sigma_{\left[k\right]}}\\) satisfies \\[\frac{c_{l}}{\sqrt{k}}\leq f_{\Sigma_{\left[k\right]}}\left(x\right)\leq\frac{c_{u}}{\sqrt{k}},\\] with the lower bound holding for all \\(x\in\left[-\sqrt{k},\sqrt{k}\right]\\) and the upper bound holding for all \\(x\in\mathbb{R}\\).*

</div>

At first, our definition of sum-bounded could look as a weaker version of a classical local limit theorem on the sum of random variables (e.g., see `\cite[Chapter VII, Theorem 7]{petrov1975sums}`{=latex}). However, that is not the case, since we require a lower bound on the sum for any \\(k\\), which is needed to prove our main result.

Denote, for all \\(x\in[0,1]\\), the binary entropy as \\[H_2(x)=-x\log_2 x-(1-x)\log_2(1-x).\\] Our main technical result is the following proof of a fixed-size subset variant of the RSS Problem.

<div id="thm:srss" class="theorem" markdown="1">

**Theorem 2**. *Let \\(0<\varepsilon<1\\), \\(c_{\text{hyp}}\ge 1\\), \\(k,n\\) be integers with \\(1\leq k\leq \frac{n}{2}\\), and let \\(\Omega=\left\{ X_{1},...,X_{n}\right\}\\) where the \\(X_{i}\\)’s are i.i.d. random variables with sum-bounded density. There exists a constant \\(c_{\text{thm}}\\) such that, if \\[\label{eq:bound_n}
        n\geq c_{\text{hyp}}\frac{\log_{2}\frac{k}{\varepsilon}}{H_{2}\left(\frac{k}{n}\right)},\\] then for every fixed \\(z\in\left[-\sqrt{k},\sqrt{k}\right]\\) it holds that \\[\Pr\left(\exists S\subset\left[n\right],\left|S\right|=k:\left|\Sigma_{S}-z\right|<\varepsilon\right)\geq c_{\text{thm}}.\\]*

</div>

<div class="remark" markdown="1">

*Remark 1*. The proof of Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a> is given in Section <a href="#sec:rfss_proof" data-reference-type="ref" data-reference="sec:rfss_proof">3.1</a>, and it actually holds for any \\(1\leq k\leq \lambda n\\), for an arbitrary \\(\lambda\in[\nicefrac{1}{n},1)\\). We state the theorem this way for readability and because we are primarily interested in high-sparsity settings (i.e., small size \\(k\\) of the subsets), so considering values of \\(k \ge \frac{n}{2}\\) does not add much to our analysis. The same remark also holds for Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a>.

</div>

The sum-bounded condition of Definition <a href="#def:quasi_unif" data-reference-type="ref" data-reference="def:quasi_unif">1</a> is easily verified for distributions such as the Gaussian distribution. Previous SLTH results rely on a classical resampling argument by `\cite[Corollary 3.3]{Lueker98}`{=latex}, which shows how RSS results for Uniform\\([-1,1]\\) independent random variables naturally extend to independent random variables that *contains* a uniform distribution, in the sense that they can be expressed as the mixture of distributions one of which is Uniform\\([-1,1]\\) with constant probability.[^1] The next lemma thus proves that the Uniform\\([-1,1]\\) distribution is sum-bounded[^2]. A detailed proof is provided in Appendix <a href="#apx:uniform_has_property" data-reference-type="ref" data-reference="apx:uniform_has_property">9</a>.

<div id="lem:uniform" class="lemma" markdown="1">

**Lemma 1**. *The Uniform\\([-1,1]\\) probability density function is sum-bounded, i.e., given a set \\(\mathcal U_n = \{U_i\}_{i\in [n]}\\) of i.i.d. variables \\(U_i\\) with Uniform\\([-1,1]\\) probability density function, there exist constants \\(c_l\\) and \\(c_u\\) such that the probability density function \\(f(x,n)\\) of the sum \\(\Sigma^{\mathcal U_n}_{[n]}\\) of these variables, for all \\(n\in\mathbb{N}\\), \\[\label{eq:uniform_property}
        \frac{c_l}{\sqrt{n}} \leq f(x,{n}) \leq \frac{c_u}{\sqrt{n}},\\] with the lower bound holding for all \\(x \in [-\sqrt{n}, \sqrt{n}]\\), and the upper bound holding for all \\(x\in\mathbb{R}\\).*

</div>

Finally, in our proofs on the Sparse SLTH in Section <a href="#sec:slth" data-reference-type="ref" data-reference="sec:slth">4</a>, we make use of the following corollary of Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a>, which ensures a uniform high probability of hitting any target \\(z\in[-\sqrt{k},\sqrt{k}]\\), considering independent random variables that contain a uniform distribution.

<div id="cor:SRSS_amp" class="corollary" markdown="1">

**Corollary 1**. *Let \\(0<p\leq 1\\) and \\(\varepsilon\in (0,\nicefrac{1}{2})\\) be constants, \\(k,n\\) with \\(1\leq k\leq \frac{n}{2}\\), and let \\(\Omega=\left\{ X_{1},...,X_{n}\right\}\\) be i.i.d. random variables whose density is a mixture of a Uniform\\(([-1,1])\\) with probability \\(p\\), and some other density otherwise. There exists a positive constant \\(c_{\text{amp}}\\) that only depends on \\(p\\) such that, if \\[\label{eq:n_cond_amp}
        n\geq c_{\text{amp}}\frac{\log_{2}^2\frac{k}{\varepsilon}}{H_{2}\left(\frac{k}{n}\right)},\\] then \\[\Pr\left(\forall z\in\left[-\sqrt{k},\sqrt{k}\right], \exists S_z\subset\left[n\right],\left|S_z\right|=k:\left|\Sigma_{S_z}-z\right|<\varepsilon\right)\geq 1-\varepsilon.\\]*

</div>

<div class="proof" markdown="1">

*Proof Idea..* The corollary follows from three arguments. First, by a standard sampling argument, we can assume that a constant fraction of the sample follows a Uniform\\([-1,1]\\) distribution. Secondly, by Lemma <a href="#lem:uniform" data-reference-type="ref" data-reference="lem:uniform">1</a>, the uniform probability density function is sum-bounded. We can thus apply Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a>, which guarantees a success probability of \\(c_{\text{thm}}\\) for approximating a given target. Finally, by a standard probability amplification argument and a union bound applied to Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a>, by paying an extra factor \\(\log_2(k/\varepsilon)\\) in Eq. <a href="#eq:bound_n" data-reference-type="ref" data-reference="eq:bound_n">[eq:bound_n]</a>, the constant \\(c_{\text{thm}}\\) can be assumed to be \\(1-\varepsilon\\), and the existence of a suitable subset \\(S_z\\) holds simultaneously for all \\(z\in[-\sqrt{k},\sqrt{k}]\\). Details are given in Appendix <a href="#apx:amplif" data-reference-type="ref" data-reference="apx:amplif">10</a>. ◻

</div>

For \\(k\\) big enough, we can get rid of the squared logarithmic dependency on \\(k\\) in the right hand side of Equation <a href="#eq:n_cond_amp" data-reference-type="ref" data-reference="eq:n_cond_amp">[eq:n_cond_amp]</a>, as shown in the following Corollary, whose proof can be found in Appendix <a href="#apx:amplif_simp" data-reference-type="ref" data-reference="apx:amplif_simp">11</a>.

<div id="cor:SRSS_amp_simp" class="corollary" markdown="1">

**Corollary 2**. *Let \\(0<p\leq 1\\) and \\(\varepsilon\in (0,\nicefrac{1}{2})\\) be constants, \\(k,n\\) be integers with \\(1\leq k\leq \frac{n}{2}\\) and \\(k \geq 2 c_{\text{amp}} \left(\log_{2}^2 k + 2log_2{k} \cdot  log_2{\frac{1}{\varepsilon}}\right)\\). Let \\(\Omega=\left\{ X_{1},...,X_{n}\right\}\\) be i.i.d. random variables whose density is a mixture of a Uniform\\(([-1,1])\\) with probability \\(p\\), and some other density otherwise. There exists a positive constant \\(c_{\text{amp}}\\) that only depends on \\(p\\) such that, if \\[\label{eq:n_cond_amp_simp}
            n\geq 2 c_{\text{amp}}\frac{\log_{2}^2\frac{1}{\varepsilon}}{H_{2}\left(\frac{k}{n}\right)},\\] then \\[\Pr\left(\forall z\in\left[-\sqrt{k},\sqrt{k}\right], \exists S_z\subset\left[n\right],\left|S_z\right|=k:\left|\Sigma_{S_z}-z\right|<\varepsilon\right)\geq 1-\varepsilon.\\]*

</div>

As customary in conference versions of papers, our proofs adopt the convention of taking ceilings and floors as suitable for non integer fractional terms. This is done in the interest of the reader (and ours), and does not impact the results in any significant way.

## Proof of Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a> [sec:rfss_proof]

<div class="proof" markdown="1">

*Proof of Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a>.* For simplicity, throughout the proof we will often use \\(c\\) to denote any positive constant. Let \\(\mathcal{S}_{k}=\{S\subset[n]\,|\,|S|=k\}\\) and define, for a fixed \\(z\in[-\sqrt{k},\sqrt{k}]\\), \\[Y=Y(z)=\sum_{S\in\mathcal{S}_{k}}Z_{S}\\] where \\(Z_{S}=Z_S(z)=\mathbf{1}_{\{\left|\Sigma_{S}-z\right|<\varepsilon\}}\\). Following `\cite{Lueker82}`{=latex}, we exploit the second moment method for RFSS, generalising it to arbitrary \\(k\\). \\[\label{eq:2ndmethod}
    \Pr\left(Y>0\right) \geq \frac{\left(\mathbb{E}\left[Y\right]\right)^{2}}{\mathbb{E}\left[Y^{2}\right]},\\] it thus suffices to prove that \\[\mathbb{E}\left[Y^{2}\right]\leq c\left(\mathbb{E}\left[Y\right]\right)^{2}.\label{eq:goal_with_expec}\\]

We first rewrite Eq. <a href="#eq:2ndmethod" data-reference-type="ref" data-reference="eq:2ndmethod">[eq:2ndmethod]</a> in a more convenient form. Let \\(\tilde{S}\\) and \\(\tilde{S}^{\prime}\\) be two independently and uniformly at random chosen subsets of \\(\left[n\right]\\) of size \\(k\\), and denote \\(H_S(z)\\) as the event that \\(\Sigma_S\\) \\(\varepsilon\\)-approximates \\(z\\), namely \\[H_{S}=H_S(z)=\left\lbrace\left|\Sigma_{S}-z\right|<\varepsilon\right\rbrace .\\]

We have \\[\mathbb{E}[Y]=\sum_{S\in\mathcal{S}_{k}}\mathbb{E}[Z_{S}]=\sum_{S\in\mathcal{S}_{k}}\Pr(H_{S})=\binom{n}{k}\Pr\left(H_{\tilde{S}}\right)\label{eq:rewrite_expec}\\] and \\[\begin{aligned}
\mathbb{E}[Y^{2}] & =\mathbb{E}\left[\left(\sum_{S\in\mathcal{S}_{k}}Z_{S}\right)\left(\sum_{S'\in\mathcal{S}_{k}}Z_{S'}\right)\right]=\sum_{S,S'\in\mathcal{S}_{k}}\mathbb{E}\left[Z_{S}Z_{S'}\right]\nonumber \\
 & =\sum_{S,S'\in\mathcal{S}_{k}}\Pr\left(H_{S}\wedge H_{S^{\prime}}\right)=\binom{n}{k}^{2}\Pr\left(H_{\tilde{S}}\wedge H_{\tilde{S}^{\prime}}\right).\label{eq:rewrite_second_moment}
\end{aligned}\\]

Using Eqs. <a href="#eq:rewrite_expec" data-reference-type="ref" data-reference="eq:rewrite_expec">[eq:rewrite_expec]</a> and <a href="#eq:rewrite_second_moment" data-reference-type="ref" data-reference="eq:rewrite_second_moment">[eq:rewrite_second_moment]</a> we can rewrite the r.h.s. of Eq. <a href="#eq:2ndmethod" data-reference-type="ref" data-reference="eq:2ndmethod">[eq:2ndmethod]</a> as follows \\[\frac{\left(\mathbb{E}\left[Y\right]\right)^{2}}{\mathbb{E}\left[Y^{2}\right]}=\frac{\left[\Pr\left(H_{\tilde{S}}\right)\right]^{2}}{\Pr\left(H_{\tilde{S}}\wedge H_{\tilde{S}^{\prime}}\right)}=\frac{\Pr\left(H_{\tilde{S}}\right)}{\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}}\right)}.\\] Eq. <a href="#eq:goal_with_expec" data-reference-type="ref" data-reference="eq:goal_with_expec">[eq:goal_with_expec]</a> thus becomes \\[\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}}\right)\leq c\Pr\left(H_{\tilde{S}}\right).\label{eq:goal_with_probs}\\] Let \\(I_{i}\\) denote the event \\(\{|\tilde{S}\cap\tilde{S}^{\prime}|=i\}\\) and \\(I_{a,b}\\) the event \\(\bigcup_{a\leq i\leq b}I_{i}\\). Fix \\(\mu\in(\lambda,1)\\). By the law of total probability and independence of \\(I_i\\) and \\(H_{\tilde{S}}\\), we rewrite the l.h.s. of Eq. <a href="#eq:goal_with_probs" data-reference-type="ref" data-reference="eq:goal_with_probs">[eq:goal_with_probs]</a> as follows: \\[\begin{aligned}
 & \Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}}\right)\nonumber \\
 & = \Pr\left(H_{\tilde{S}^{\prime}}\wedge I_{k}\,|\,H_{\tilde{S}}\right)+\Pr\left(H_{\tilde{S}^{\prime}}\wedge I_{\mu k,k-1}\,|\,H_{\tilde{S}}\right)+\Pr\left(H_{\tilde{S}^{\prime}}\wedge I_{0,\mu k-1}\,|\,H_{\tilde{S}}\right)\nonumber \\
 & =\Pr\left(I_{k}\right)\cdot\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{k}\right)\label{eq:first_term}\\
 & \qquad+\Pr\left(I_{\mu k,k-1}\right)\cdot\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{\mu k,k-1}\right)\label{eq:second_term}\\
 & \qquad+\sum_{i=0}^{\mu k-1}\left(\Pr\left(I_{i}\right)\cdot\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{i}\right)\right).\label{eq:third_term}
\end{aligned}\\] To conclude the proof, it suffices to show that each addendum in Eqs. <a href="#eq:first_term" data-reference-type="ref" data-reference="eq:first_term">[eq:first_term]</a>, <a href="#eq:second_term" data-reference-type="ref" data-reference="eq:second_term">[eq:second_term]</a> and <a href="#eq:third_term" data-reference-type="ref" data-reference="eq:third_term">[eq:third_term]</a> are upper-bounded by some constant multiple of \\(\nicefrac{\varepsilon}{\sqrt{k}}\\), since the lower bound in Definition <a href="#def:quasi_unif" data-reference-type="ref" data-reference="def:quasi_unif">1</a> ensures that \\[\label{eq:make_prob_appear}
\frac{\varepsilon}{\sqrt{k}}\le c\Pr\left(H_{\tilde{S}}\right).\\]

As for the first addendum (Eq. <a href="#eq:first_term" data-reference-type="ref" data-reference="eq:first_term">[eq:first_term]</a>), since \\(\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{k}\right)=1\\), then \\[\begin{aligned}
& \Pr\left(I_{k}\right) \cdot \Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{k}\right)  =\Pr\left(I_{k}\right)=\frac{1}{{n \choose k}}
 \stackrel{(a)}{\leq} \sqrt{\frac{8k(n-k)}{n}}2^{-nH_2\left(\frac{k}{n}\right)}\nonumber \\
 & 
 \stackrel{(b)}{\leq} \sqrt{\frac{8k(n-k)}{n}}2^{-c_{\text{hyp}}\log_2\frac{k}{\varepsilon}}
 \stackrel{(c)}{\leq}2\sqrt{2}\frac{\varepsilon}{\sqrt{k}},\label{eq:square_root_first_addendum}
\end{aligned}\\] where inequality \\((a)\\) in Eq. <a href="#eq:square_root_first_addendum" data-reference-type="ref" data-reference="eq:square_root_first_addendum">[eq:square_root_first_addendum]</a> is a standard lower bound on \\({n \choose k}\\) holding for all \\(k\leq n-1\\); in inequality \\((b)\\) in Eq. <a href="#eq:square_root_first_addendum" data-reference-type="ref" data-reference="eq:square_root_first_addendum">[eq:square_root_first_addendum]</a> we used Eq. <a href="#eq:bound_n" data-reference-type="ref" data-reference="eq:bound_n">[eq:bound_n]</a>, namely \\(nH_{2}\left(\frac{k}{n}\right)\geq c_{\text{hyp}}\log_{2}\frac{k}{\varepsilon}\\); in inequality \\((c)\\) in Eq. <a href="#eq:square_root_first_addendum" data-reference-type="ref" data-reference="eq:square_root_first_addendum">[eq:square_root_first_addendum]</a> we used that \\(c_{\text{hyp}}\ge 1\\).

As for the second addendum (Eq. <a href="#eq:second_term" data-reference-type="ref" data-reference="eq:second_term">[eq:second_term]</a>), we next show that \\[\label{eq:square_root_second_addendum}
\Pr\left(I_{\mu k,k-1}\right)\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{\mu k,k-1}\right)\leq c\frac{\varepsilon}{\sqrt{k}}\\] by proving that \\[\Pr\left(I_{\mu k,k-1}\right)\leq\frac{c}{\sqrt{k}}\label{eq:first_factor_second_term}\\] and \\[\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{\mu k,k-1}\right)\leq c\varepsilon.\label{eq:second_factor_second_term}\\] First, observe that \\(I=|\tilde{S}\cap\tilde{S}^{\prime}|\\) follows a Hypergeometric\\((n,k,k)\\) distribution, thus by Chebyshev’s inequality \\[\begin{aligned}
\Pr\left(I_{\mu k,k-1}\right) &\leq\Pr\left(I\geq\mu k\right)=\Pr\left(I-\frac{k^2}{n}\geq\mu k-\frac{k^2}{n}\right)
 \leq\frac{\operatorname{Var}\left[I\right]}{\mu^2k^{2}\left(1-\frac{k}{\mu n}\right)^2}
 \notag\\&\le c'\frac{\frac{k^{2}}{n}\frac{n-k}{n}\frac{n-k}{n-1}}{k^{2}}
 \leq\frac{c}{\sqrt{k}},\nonumber
\end{aligned}\\] having set \\(c'=\mu^2(1-\nicefrac{\lambda}{\mu})^2>0\\), thus proving Eq. <a href="#eq:first_factor_second_term" data-reference-type="ref" data-reference="eq:first_factor_second_term">[eq:first_factor_second_term]</a>. The proof of Eq. <a href="#eq:second_factor_second_term" data-reference-type="ref" data-reference="eq:second_factor_second_term">[eq:second_factor_second_term]</a> is given in Appendix <a href="#apx:srss_proof_details" data-reference-type="ref" data-reference="apx:srss_proof_details">12</a>, concluding the proof of Eq. <a href="#eq:square_root_second_addendum" data-reference-type="ref" data-reference="eq:square_root_second_addendum">[eq:square_root_second_addendum]</a>.

As for the third addendum (Eq. <a href="#eq:third_term" data-reference-type="ref" data-reference="eq:third_term">[eq:third_term]</a>), in Appendix <a href="#apx:srss_proof_details" data-reference-type="ref" data-reference="apx:srss_proof_details">12</a> we show that \\[\begin{aligned}
 \sum_{i=0}^{\mu k-1}\Pr\left(I_{i}\right)\cdot\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{i}\right)
 \le c\frac{\varepsilon}{\sqrt{k}},\label{eq:square_root_third_addendum}
\end{aligned}\\]

The three bounds on the addenda in Eqs. <a href="#eq:first_term" data-reference-type="ref" data-reference="eq:first_term">[eq:first_term]</a>, <a href="#eq:second_term" data-reference-type="ref" data-reference="eq:second_term">[eq:second_term]</a>, and <a href="#eq:third_term" data-reference-type="ref" data-reference="eq:third_term">[eq:third_term]</a> (Eqs. <a href="#eq:square_root_first_addendum" data-reference-type="ref" data-reference="eq:square_root_first_addendum">[eq:square_root_first_addendum]</a>, <a href="#eq:square_root_second_addendum" data-reference-type="ref" data-reference="eq:square_root_second_addendum">[eq:square_root_second_addendum]</a>, and <a href="#eq:square_root_third_addendum" data-reference-type="ref" data-reference="eq:square_root_third_addendum">[eq:square_root_third_addendum]</a>, respectively), combined with Eq. <a href="#eq:make_prob_appear" data-reference-type="ref" data-reference="eq:make_prob_appear">[eq:make_prob_appear]</a>, conclude the proof. ◻

</div>

# Sparse Strong Lottery Ticket Hypothesis (SSLTH) [sec:slth]

We now apply our results on the RFSS problem to the SLTH and obtain guarantees on the sparsity of winning tickets for Dense Neural Networks (DNNs, Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a>) and Equivariant NNs (Theorem <a href="#thm:ferbach" data-reference-type="ref" data-reference="thm:ferbach">4</a>).

The next theorem provides a direct quantitative answer to the guiding question raised in Section <a href="#sec:intro" data-reference-type="ref" data-reference="sec:intro">1</a>.  No further elaboration is required, as the statement itself transparently captures the sparsity–overparameterization trade-off.

<div id="thm:pensia" class="theorem" markdown="1">

**Theorem 3** (SSLTH for DNNs). *Let \(g\) be a randomly initialized feed-forward \(2\ell\)-layer neural network, in which each weight is drawn from a Uniform\([-1,1]\) distribution, of the following form: \[g({\bf x})=\mathbf{M}_{2l} \sigma(\mathbf{M}_{2l-1}\dots\sigma( \mathbf{M}_1 \mathbf{x})).\] Let \(\gamma'=\gamma'(%d_0,...,d_\ell,\ell, \varepsilon) \in (0,1)\), \(\mathbf{M}_{2i} \in \mathbb{R}^{d_i \times 2 d_{i-1}n_i^*}\) and \(\mathbf{M}_{2i-1} \in \mathbb{R}^{2 d_{i-1}n_i^* \times d_{i-1}}\), with \(n_i^*\) satisfying \[\label{n^*}n_i^*= c_{\text{amp}}\frac{\log_{2}^2\!\left(\frac{ 2\ell d_{i-1}d_{i} \gamma'n_i^*}{\varepsilon}\right)}{H_{2}(\gamma')} .\] With probability at least \(1-\varepsilon\), for every target network \(f\) as in Eq. <a href="#EqnFFamilyUp" data-reference-type="ref" data-reference="EqnFFamilyUp">[EqnFFamilyUp]</a>, \(g\) contains a subnetwork of sparsity at least \(\alpha = 1-\gamma\) (where \(\gamma = \rho\,\gamma'\)) that \(\varepsilon\)-approximates \(f\).*

</div>

<div class="proof" markdown="1">

*Proof Idea.*  The proof proceeds exactly as in `\cite{Pensia}`{=latex}, except that Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> replaces the classical RSS corollary.  All quantitative improvements follow automatically; we therefore omit further commentary. ◻

</div>

Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a> straightforwardly subsumes prior dense-network results and yields optimal (up to logarithmic factors) sparsity regimes across the entire parameter range.

<div id="thm:ferbach" class="theorem" markdown="1">

**Theorem 4** (SSLTH for Equivariant Networks). *Let \(h\) be a random \(2\ell\)-layer \(G\)-equivariant network whose equivariant parameters are i.i.d. Uniform\([-1,1]\).  Suppose \(\gamma=\gamma(\varepsilon)\in(0,1)\) and \(\tilde{n}\) satisfies \[\tilde{n}= c_{\text{amp}}\frac{\log_{2}^2\!\left(\frac{2\ell n^2\max\{|\mathcal{B}|,\Vert\mathcal{B}\Vert\}\gamma \tilde{n}}{\varepsilon}\right)}{H_{2}(\gamma)}.\] Then, with probability at least \(1-\varepsilon\), every \(\ell\)-layer \(G\)-equivariant target network is \(\varepsilon\)-approximated by a subnetwork of \(h\) of sparsity at least \(\alpha=1-\gamma\).*  

</div>

The statement is an immediate extension of Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a> to the equivariant setting; the proof is identical to that of `\cite{ferbachGeneralFrameworkProving2022}`{=latex} and so is omitted.

For completeness, we recall that Theorem <a href="#thm:ferbach" data-reference-type="ref" data-reference="thm:ferbach">4</a> also recovers convolutional and residual architectures as special cases.

No further discussion is necessary, as the above results fully characterize the sparsity landscape of the SLTH.

#### 

When all layers have the same width \\(d\\), `\cite{malachProvingLotteryTicket2020}`{=latex} showed that any target network with \\(l\\) layers and a total of \\(m_t = d^2 l\\) parameters can be \\(\varepsilon\\)-approximated by pruning a randomly initialized network with \\(2l\\) layers. The overparameterization of this network, relative to the target network, is \\(O\left(\frac{m_t^2}{\varepsilon^2} \log_2 \frac{m_t}{\varepsilon}\right) = \Tilde{O}\left(\frac{m_t^2}{\varepsilon^2}\right)\\). More specifically, the winning ticket found after pruning has a parameter count of the same order as the target network, resulting in a density of \\(\gamma = \Tilde{O}\left(\frac{\varepsilon^2}{m_t^2}\right)\\). Notably, this density \\(\gamma\\) is the inverse of the overparameterization, as the size of the winning ticket matches that of the target network.

Next, we show that Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a> also yields a density that is polynomial in \\(\frac{\varepsilon}{m_t}\\), when using an overparametrization of \\(\Theta\left(\frac{m_t^2}{\varepsilon^2}\right)\\). Let \\(z=\left(\frac{m_t}{\varepsilon}\right)\\), and note that \\(\gamma' = \gamma\\) in Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a>, since all layers have the same width. As \\(n^*_i\\) in Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a> represents the overparametrization with respect to the target network, let us set \\(n^*_i = cz^2\\), for some constant \\(c\\). Equation <a href="#n^*" data-reference-type="ref" data-reference="n^*">[n^*]</a> then becomes \\[\label{eq:n^*_malach}
cz^2 \ge c_{\text{amp}} \frac{\log_2^2 (c z^3 \gamma)}{H(\gamma)}\\] We show that the inequality \\(cz^2 \ge c_{\text{amp}} \frac{\log_2^2 (c z^3 \gamma)}{\gamma \log_2(\nicefrac{1}{\gamma})}\\) holds for some big enough constant \\(c\\) when setting \\(\gamma = \frac{\varepsilon}{m_t} = \frac{1}{z}\\), which implies that Equation <a href="#eq:n^*_malach" data-reference-type="ref" data-reference="eq:n^*_malach">[eq:n^*_malach]</a> is also satisfied. We get \\(cz \ge c_{\text{amp}} \frac{\log_2^2 (c z^2)}{\log_2(z)}\\), which is satisfied for a big enough constant \\(c\\) (see Appendix <a href="#apx:malach" data-reference-type="ref" data-reference="apx:malach">15</a>). Overall, when using an overparametrization \\(\Theta\left(\frac{m_t^2}{\varepsilon^2}\right)\\), we find a winning ticket with density \\(\frac{\varepsilon}{m_t}\\), as shown in Figure <a href="#fig:plot-pensia-malach" data-reference-type="ref" data-reference="fig:plot-pensia-malach">1</a>.

#### 

For simplicity, let us still consider target networks where all layers have the same width \\(d\\), and we apply Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a> using the simplified condition from Equation <a href="#n^*_simp" data-reference-type="ref" data-reference="n^*_simp">[n^*_simp]</a>. When \\(\gamma m = \Theta(m)\\), i.e. the density \\(\gamma\\) is a constant as in `\cite{Pensia}`{=latex} (see Appendix <a href="#apx:pensia_lower" data-reference-type="ref" data-reference="apx:pensia_lower">7</a>), the entropy term \\(H_{2}(\gamma')\\) in the right-side of Equation <a href="#n^*_simp" data-reference-type="ref" data-reference="n^*_simp">[n^*_simp]</a> also becomes a constant. In this setting, we indeed recover the result shown in `\cite{Pensia}`{=latex}\[Theorem 1\], up to a logarithmic factor, as shown in Figure <a href="#fig:plot-pensia-malach" data-reference-type="ref" data-reference="fig:plot-pensia-malach">1</a>.

Quite similarly to Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a>, the next result essentially generalizes `\cite{ferbachGeneralFrameworkProving2022}`{=latex} up to a factor \\(\log_2 \frac 1\varepsilon\\). The theorem is stated with the understanding that for \\(G\\)-equivariant networks, in order to preserve \\(G\\)-equivariance, pruning is best done not with respect to the parameters expressing the network in the canonical basis (i.e. directly on the weights of the network), but with respect to the *equivariant parameters*, that is those coefficients expressing the linear layers of the network as a linear combination of the elements of the corresponding equivariant basis `\cite{ferbachGeneralFrameworkProving2022}`{=latex}. For simplicity, due to the technical set-up, we assume all feature spaces being \\(\mathbb{F}=(\mathbb{R}^d,\sigma)\\), with \\(\sigma\\) the linear representation of the group \\(G\\), and the same number \\(n\\) of such feature spaces being stacked in each layer. A \\(G\\)-equivariant linear map from the \\(i\\)th feature space to the \\(i+1\\)st can be decomposed in a corresponding equivariant basis denoted \\(\mathcal{B}_{i\rightarrow i+1}=\mathcal{B}\\). Since all feature spaces are the same, we omit the layers’ indices. When stacking \\(n\\) feature spaces in the input and output of the \\(i\\)th layer, the full equivariant basis is denoted \\(k_{n\rightarrow n}\\), and finally the basis of the \\(G\\)-equivariant maps from \\(\mathbb{F}^n\\) to \\(\mathbb{F}^n\\) can be written as the Kronecker product \\(k_{n\rightarrow n}\otimes\mathcal{B}\\). For any basis \\(\mathcal{B}=\{b_1,\ldots,b_p\}\\), we denote its cardinality \\(p=|\mathcal{B}|\\) and define \\(\Vert\mathcal{B}\Vert=\max_{\Vert\beta\Vert_\infty}\Vert\sum_{k=1}^p\beta_kb_k\Vert\\), with \\(\Vert\cdot\Vert\\) in the r.h.s. being the operator norm inherited from the \\(\ell_p\\) norm.

<div id="thm:ferbach" class="theorem" markdown="1">

**Theorem 4** (SSLTH for Equivariant Networks). *Let \\(h\\) be a random \\(2\ell\\)-layer \\(G\\)-equivariant network where all equivariant parameters are drawn from a Uniform\\([-1,1]\\) distribution, every odd layer expressed in the associated equivariant basis \\(k_{\tilde{n}\rightarrow n}\otimes \mathcal{B}\\) and every even layer expressed in the associated equivariant basis \\(k_{n\rightarrow \tilde{n}}\otimes \mathcal{B}\\). Let \\(\gamma=\gamma(\varepsilon)\in(0,1)\\), with \\(\tilde{n}\\) satisfying \\[\tilde{n}= c_{\text{amp}}\frac{\log_{2}^2\left(\frac{2\ell n^2\max\{|\mathcal{B}|,\Vert\mathcal{B}\Vert\}\gamma \tilde{n}}{\varepsilon}\right)}{H_{2}\left(\gamma\right)}.\\] With probability at least \\(1-\varepsilon\\), for every \\(\ell\\)-layer \\(G\\)-equivariant neural network \\(f\\), with all layers expressed in the associated equivariant basis \\(k_{n\rightarrow n}\otimes \mathcal{B}\\), \\(h\\) can be pruned to obtain a \\(G\\)-equivariant subnetwork of sparsity at least \\(\alpha=1-\gamma\\) that approximates \\(f\\) up to an error \\(\varepsilon\\).*

</div>

The proof, which we omit, is analogous to that of Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a>, since `\cite{ferbachGeneralFrameworkProving2022}`{=latex}\[Theorem 1\] exploits the exact same pruning strategy of `\cite{Pensia}`{=latex}, except for the fact that it is applied not to the original parameters of the equivariant network, but to the network expressed in terms of its equivariant basis (the sparsity \\(\alpha\\) is here also intended with respect to the equivariant parameters count). This allows the construction to apply without losing the property of equivariance in the pruned approximating subnetwork obtained. The crucial step is when Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> is applied in `\cite{ferbachGeneralFrameworkProving2022}`{=latex}\[Lemma 1\], instead of `\cite{Lueker98}`{=latex}\[Corollary 2.5\]. This is done in parallel, multiple times, across non-overlapping coefficients of the equivariant basis. Thanks to the careful preprocessing devised by the authors, this preserves equivariance and at the same time ensures that each application of Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> is independent of the others.

To conclude the section, we mention that Theorem <a href="#thm:ferbach" data-reference-type="ref" data-reference="thm:ferbach">4</a> applies in particular to vanilla CNNs, which are a special case of equivariant neural networks where the group is \\(G=(\mathbb{Z}^2,+)\\), recovering previous SLTH results on CNN `\cite{dacunhaProvingStrongLottery2022,burkholzConvolutionalResidualNetworks2022}`{=latex}. Furthermore, we remark that Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a> can be revisited through the improvement upon the \\(2\ell\\)-depth overparameterization devised in `\cite{burkholzMostActivationFunctions2022}`{=latex}, i.e., it is possible to provide sparsity guarantees also for overparameterizations requiring depth \\(\ell+1\\) only. The analysis is more technical and we omit it, but the ideas are analogous to what shown in `\cite{burkholzMostActivationFunctions2022}`{=latex}. An analogous improvement is suggested as future work in `\cite{ferbachGeneralFrameworkProving2022}`{=latex}.

## Lower bound on the required overparameterization [sec:LB]

We now adapt the lower bound of `\cite{Pensia}`{=latex} in order to almost match the required overparameterization of our Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a>, considering the simple scenario in which we want to approximate the family \\(\mathcal{F}\\) of all linear networks with weights forming a matrix having spectral norm less than \\(\sqrt{k}\\); more formally \\[\mathcal{F} := \{h_W : W \in \mathbb R^{d\times d}, \|W\| \leq \sqrt k\}, \quad \text  {   where   } \quad h_W(x) = W x.
\label{eq:LinearLower}\\] The formal claim states that, if a network with \\(n\\) parameters can approximate every \\(h_W \in \mathcal F\\) with probability at least \\(\nicefrac{1}{2}\\) (after it is pruned down to \\(k\\) parameters), then the hypothesis of Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a> in Eq. <a href="#eq:bound_n" data-reference-type="ref" data-reference="eq:bound_n">[eq:bound_n]</a> must hold.[^3]

<div id="thm:lowerBoundStrong" class="theorem" markdown="1">

**Theorem 5**. *Let \\(n,k\in \mathbb N\\), with \\(1\le k\leq \lambda n\\), having set \\(\lambda=1-\nicefrac{1}{2\pi}\approx 0.84\\). Consider a neural network \\(g\\) with \\(n\\) parameters, and let \\(\mathcal{G}_k\\) be the set of neural networks that can be formed by pruning \\(g\\) down to \\(k\\) parameters. Let \\(\mathcal{F}\\) be as defined in Eq. <a href="#eq:LinearLower" data-reference-type="ref" data-reference="eq:LinearLower">[eq:LinearLower]</a>. If it holds that, for some \\(\varepsilon<\nicefrac{1}{16}\\), \\[\begin{aligned}
    \forall {h_W\in \mathcal{F}} , \mathbb{P}\left( \exists g'\in \mathcal{G}_k: \max_{\mathbf{x}:\|x\|\leq 1} \|h_W(x)-g'(x)\| <\varepsilon\right) \geq \frac{1}{2} ,
    \label{eq:lowerBdApprox}
    
\end{aligned}\\] then it holds that \\[n\geq  \frac{d^2}{2} \frac{\log_{2}\frac{k}{\varepsilon}}{H_{2}\left(\frac{k}{n}\right)}.\\]*

</div>

The theorem follows by adapting the packing argument of `\cite{Pensia}`{=latex}. A detailed proof is provided in Appendix <a href="#apx:lower" data-reference-type="ref" data-reference="apx:lower">14</a>.

# Conclusions [sec:conclusions]

In this work, we have extended previous results on the Strong Lottery Ticket Hypothesis by quantifying the required overparameterization as a function of the sparsity of the subnetworks. Central to our results is a proof of the Random Fixed-size Subset Sum (RFSS) Problem, a refinement of the seminal Random Subset Sum (RSS) Problem in which the subsets have a required fixed size.

A challenging open problem is to extend our analysis of RFSS to the multidimensional case, in which the random samples and targets are vectors in \\(\mathbb R^d\\). Previous extension of RSS to the Multidimensional RSS have indeed allowed to prove structured-pruning version of the SLTH `\cite{dacunhaProvingStrongLottery2022}`{=latex}. A Multidimensional RFSS result would then allow to quantify, in the structured pruning case, the dependency of the overparameterization w.r.t. the sparsity of the (structured) subnetworks.

Another future direction is to refine our analysis of the RFSS in Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a> in order to improve the probability of success to \\(1-\varepsilon\\) rather than constant, thus allowing to avoid shaving off the extra factor \\(\log_2(1/\varepsilon)\\) in our corollaries w.r.t. our lower bound, which is due to the amplification done in Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> to get to probability \\(1-\varepsilon\\).

Finally, an important future direction is to improve training-by-pruning methods such as `\cite{zhouDeconstructingLotteryTickets2019,ramanujanWhatHiddenRandomly2020,fischerLotteryTicketsNonzero2022,fischerPlantSeekCan2022,otsukaPartialSearchFrozen2024a}`{=latex} or to develop new ones, in order to allow to efficiently find strong lottery tickets of a desired sparsity, thus empirically validating our theoretical predictions.

# Limitations and Impact [sec:lim_impact]

#### Limitations

Similar to all the research conducted on the LTH and the SLTH, this work only proves the existence of lottery tickets. To this date, it is not clear if these subnetworks can be found reliably (no formal proof exists) in an efficient manner - however, empirical evidence suggests that efficient algorithms exist (e.g., `\cite{zhouDeconstructingLotteryTickets2019,ramanujanWhatHiddenRandomly2020}`{=latex}).

#### Impact

The contribution of this work is primarily theoretical and not confined to a specific domain. Its potential societal impact would, therefore, be closely tied to the particular scenarios to which it is applied. It could be interesting to compare the environmental impact of finding lottery tickets inside overparameterized networks. We also believe that our work has the potential to have a strong environmental impact as sparse NNs have massively reduced inference costs.

<div class="ack" markdown="1">

This research is supported by the EPSRC grant EP/W005573/1, and by the France 2030 program, managed by the French National Research Agency under grant agreements No. ANR-23-PECL-0003 and and ANR-22-PEFT-0002. It was also funded in part by the European Network of Excellence dAIEDGE under Grant Agreement Nr. 101120726, by SmartNet and LearnNet, and by the French government National Research Agency (ANR) through the UCA JEDI (ANR-15-IDEX-01), EUR DS4H (ANR-17-EURE-004), and the 3IA Côte d’Azur Investments in the Future project with the reference number ANR-19-P3IA-0002.

</div>

# Lower Bound on the Ticket Size in `\cite{Pensia}`{=latex} [apx:pensia_lower]

The claim is a direct consequence of the proof of `\cite[Theorem 2]{Pensia}`{=latex} (Appendix B). There, in Step 3, it is shown that \\[|\mathcal G| \geq \frac 12 \left(\frac 1{2\varepsilon}\right)^{d^2},\\] where \\(\mathcal G\\) is the set of subnetworks that can be formed. Let \\(m\\) be the number of parameters of the original network. If we consider subnetworks of size at most \\(\gamma m\\) (\\(0 \le \gamma \le 1\\)), we have[^4] \\[|\mathcal G| \leq \sum_{i=1}^{\gamma m} {m \choose {\gamma m}} \leq 2^{\gamma m\log_2 (\frac{m}{\gamma m} e)},\\] which combined with the previous inequality implies \\[\gamma m \log_2 \left(\frac{e}{\gamma}\right) \geq  d^2 \log_2 \left(\frac 1{2\varepsilon}\right) -1\\] If we have an overparameterized network of size \\(m = \mathcal{O}(d^2 \log_2\left(\frac{1}{2\varepsilon}\right))\\), as in `\cite{Pensia}`{=latex}, we need \\(\gamma m = \Theta(m)\\) for the last inequality to be satisfied (note that \\(\log_2\left(\frac{e}{\gamma}\right) \le 1\\), as \\(0 \le \gamma \le 1\\)).

# Visualizations

<figure id="fig:LTH">
<img src="./figures/LTH.png"" />
<figcaption><strong>Simplified representation of the procedure for finding Lottery Tickets (LTH)</strong>. A large random neural network (step 1) is trained by iterative pruning with rewind: when the loss reaches a local minimum (step 2), some weights with smallest absolute value are pruned (step 3) and the value of the remaining edges is then reset to that of the initialization (step 4); finally, training is resumed and the final network is obtained (step 5). Remarkably, the sparser subnetwork is consistently able to reach a loss not larger than that right after pruning.</figcaption>
</figure>

<figure id="fig:truning">
<img src="./figures/LTH.png"" style="width:70.0%" />
<figcaption> <strong>Simplified representation of the procedure for finding Strongly Lottery Tickets (SLTH) / Training by pruning</strong>. Previous work has shown that it is possible to sparsify large random neural network in order to obtain subnetworks that achieve good performance for a task under consideration, motivating the <em>Strong Lottery Ticket Hypothesis</em>. No training is required.</figcaption>
</figure>

# Proof of Uniform\\([-1,1]\\) being Sum-Bounded [apx:uniform_has_property]

In this section we provide a detailed proof of Lemma <a href="#lem:uniform" data-reference-type="ref" data-reference="lem:uniform">1</a>, which states that the uniform distribution in \\([-1,1]\\) is sum-bounded, as stated in Definition <a href="#def:quasi_unif" data-reference-type="ref" data-reference="def:quasi_unif">1</a>. We remark that, while the proof is written for uniform random variables, it should be possible to extend it to a family of densities which are unimodal, with bounded variance, and bounded third moment.

<div class="proof" markdown="1">

*Proof of Lemma <a href="#lem:uniform" data-reference-type="ref" data-reference="lem:uniform">1</a>.* Note first that the distribution of the sum of \\(n\\) i.i.d. variables in \\([0,1]\\) is known as the Irwin–Hall distribution \\(I_n\\).[^5] We will use that \\(\mathrm{Var}[I_n] = \frac n{12}\\), where \\(\mathrm{Var}[X]\\) denotes the variance of the random variable \\(X\\).

For \\(n\geq 2\\), \\(f(x,n)\\) can be defined as the convolution of \\(f(x)=f(x,1)\\) and \\(f(x,n-1)\\), i.e., \\[f(x,n) = \int_{-\infty}^{+\infty} f(x-\tau,n-1)f(\tau) d\tau.\\] It is straightforward to show, by induction and an elementary substitution in the integral above, which is relied upon in the inductive step, that \\(f(x,n)\\) is symmetric about \\(0\\), that is \\(f(x,n)=f(-x,n)\\).

Let us now prove by induction that \\(f(x,n)\\) is nondecreasing on the interval \\([-n,0]\\) and nonincreasing over \\([0,n]\\) (for simplicity, since it vanishes outside \\([-n,n]\\), we can consider directly the negative half and positive half of the real line, respectively, in the argument that follows).

The claims hold trivially for \\(f(x)\\); also note that \\[f(\tau) = \left\{ 
    \begin{array}{ll}
    \frac{1}{2} & \qquad \text{if} \quad -1 \leq \tau \leq 1\\
    0 & \qquad \text{otherwise}
    \end{array}\right. \qquad {\implies} \qquad 
    f(x,n) = \frac{1}{2}\int_{-1}^{+1} f(x-\tau,n-1) d\tau.\\]

If \\(x \leq x' \leq - 1\\). Since \\(x-\tau\le x'-\tau\le 0\\), by inductive hypothesis we have that \\(f(x-\tau,n-1) \leq f(x'-\tau,n-1)\\) over the whole interval \\(\tau \in [-1,1]\\). Taking integrals yields \\(f(x,n)\leq f(x',n)\\).

Now, consider the case when \\(x \leq -1 \leq x' \leq 0\\). If \\(x+1 \leq -x'-1\\), \\(x-\tau\le x+1\le-x'-1\le-x'+\tau\le -x+\tau\\). By the symmetry about the origin, the inductive hypothesis is \\(f(x-\tau,n-1)=f(-x+\tau,n-1) \leq f(-x'+\tau,n-1)=f(x'-\tau,n-1)\\) over the whole interval \\(\tau \in [-1,1]\\), since \\(-1\le-x'+\tau\le-x+\tau\\). Taking integrals yields \\(f(x,n)\leq f(x',n)\\). Otherwise, there exists \\(\tau_0\\) such that \\(x-\tau_0=-x'-1\\), \\(x-\tau>-x'-1\\) for all \\(\tau\in[-1,\tau_0)\\) and \\(x-\tau<-x'-1\\) for all \\(\tau\in(\tau_0,1]\\). By symmetry, using \\(-x=x'+1-\tau_0\\), \\(f(x-\tau,n-1) =f(-x+\tau,n-1)= f(x'+1+\tau-\tau_0,n-1)\\). Thus, for all \\(\tau\in[-1,\tau_0]\\), via the change of variable \\(\sigma=-(1+\tau-\tau_0)\\) in the middle integral below, we obtain that \\[\label{eq:first_half_int}
        \int_{-1}^{\tau_0} f(x-\tau,n-1) d\tau = \int_{-1}^{\tau_0} f(x'+1+\tau-\tau_0,n-1) d\tau=\int_{-1}^{\tau_0} f(x'-\sigma,n-1) d\sigma.\\] For all \\(\tau\in (\tau_0,1]\\), \\(x-\tau<-x'-1\le-x'+\tau\le-x+\tau\\), by symmetry about the origin we have that \\(f(x-\tau,n-1) \leq f(x'-\tau,n-1)\\) by the inductive hypothesis with the same reasoning of the case \\(x+1 \leq -x'-1\\). Taking integrals over the range \\([\tau_0,1]\\) for each term of the inductive hypothesis yields \\[\label{eq:second_half_int}
    \int_{\tau_0}^1f(x-\tau,n-1)d\tau\le \int_{\tau_0}^1f(x'-\tau,n-1)d\tau\\] Eqs. <a href="#eq:first_half_int" data-reference-type="ref" data-reference="eq:first_half_int">[eq:first_half_int]</a> and <a href="#eq:second_half_int" data-reference-type="ref" data-reference="eq:second_half_int">[eq:second_half_int]</a> imply that \\(f(x,n) \leq f(x'n)\\).

Trivially, if \\(-1 \leq x \leq x'\le 0\\), analogous ideas are put in place as for the previous case, therefore we omit the details. We have thus shown the nondecreasing monotonicity of \\(f(x,n)\\) on the negative half of the real line. By the symmetry of \\(f(x,n)\\) about the origin, on the positive half of the real line the nondecreasing monotonicity turns into nonincreasing monotonicity, and the proof is complete.

**Lower bound (first inequality in Eq. <a href="#eq:uniform_property" data-reference-type="ref" data-reference="eq:uniform_property">[eq:uniform_property]</a>).** The variance of \\(\Sigma^{\mathcal U_n}_{[n]}\\) is \\(n/3\\) since \\(\Sigma^{\mathcal U_n}_{[n]} = 2(I_n(n)-n/2)\\) and \\(\mathrm{Var}[I_n(n)] = n/12\\). We define \\(Z_n^u= \frac{\Sigma^{\mathcal U_n}_{[n]}}{\sqrt{n/3}}\\) and we note with \\(F_n\\) its cumulative distribution function. \\(Z_n^u\\) has expectation 0 and standard deviation 1. Consider the probability \\[P_L(n)= \Pr(\sqrt{n} \leq \Sigma^{\mathcal U_n}_{[n]}\leq 2\sqrt{n}) 
        = \Pr(\sqrt 3 \leq Z_n^u\leq 2\sqrt 3).\\] Now, we use the following form of Berry–Esseen inequality, discussed in `\cite{marengo2017geometric}`{=latex}\[p.2\]).[^6]

<div id="thm:BE" class="theorem" markdown="1">

**Theorem 6** (Allasia `\cite{Allasia81}`{=latex}). *For all \\(n \geq 1\\), \\[|F_n(z)-\Phi(z)| \leq \frac{\sqrt{3}}{20\sqrt{n}},\\] where \\(\Phi(z)\\) is the cumulative distribution function of the standard normal distribution.*

</div>

Theorem <a href="#thm:BE" data-reference-type="ref" data-reference="thm:BE">6</a> implies \\[P_L(n)\geq \Phi( 2\sqrt 3) - \Phi(\sqrt 3) - 2\cdot \frac{\sqrt{3}}{20\sqrt{n}}.\\] When \\(n \geq 18\\), \\[\Phi( 2\sqrt 3) - \Phi(\sqrt 3) - 2\cdot \frac{\sqrt{3}}{20\sqrt{n}} \geq \Phi( 2\sqrt 3) - \Phi(\sqrt 3) - 2\cdot \frac{\sqrt{3}}{20\sqrt{18}} = C_{18}> 0.\\] That is \\(P_L(n)\geq C_{18} > 0\\). When \\(2 \leq n < 18\\), \\(P_L(n)= F_n(2\sqrt 3) - F_n(\sqrt 3) = c_n > 0\\). We thus have \\[P_L(n)\geq \min\{C_i, \text{ for } 2 \leq i \leq 18\} = c'_l> 0.\\] Recall that \\(P_L(n)= \Pr(\sqrt{n} \leq \Sigma^{\mathcal U_n}_{[n]}\leq 2\sqrt{n})\\). As the density \\(f(x,n)\\) is decreasing on \\(\mathbb R^+\\), we have \\[P_L(n)\leq f(\sqrt{n},n) \sqrt n.\\] Thus, \\[f(\sqrt{n},n) \geq \frac{P_L(n)}{\sqrt n}.\\] Since \\(P_L(n)\geq c'_l\\) then for all \\(n \geq 2\\) \\[f(\sqrt{n},n) \geq \frac{c'_l}{\sqrt n}.\\] When \\(n=1\\), the density \\(f(1,1)=\frac 12\\). So, by setting \\(c_l= \min(c'_l,\frac 12)\\), we get that, for all \\(n \geq 1\\), for all \\(0\le x\le \sqrt{n}\\): \\[f(x,n)\geq f(\sqrt{n},n) \geq \frac{c_l}{\sqrt n}.\\] By a symmetric argument, we also have for all \\(n \geq 1\\), for all \\(-\sqrt{n}\le x\le 0\\): \\[f(x,n)\geq f(-\sqrt{n},n) \geq \frac{c_l}{\sqrt n}.\\]

**Upper bound (second inequality in Eq. <a href="#eq:uniform_property" data-reference-type="ref" data-reference="eq:uniform_property">[eq:uniform_property]</a>).** Here, we bound the probability distribution function \\(f(x,n)\\) of \\(\Sigma^{\mathcal U_n}_{[n]}=\sqrt{\nicefrac{n}{3}} Z_n\\), where we recall that \\(Z_n^u= \frac{\Sigma^{\mathcal U_n}_{[n]}}{\sqrt{\nicefrac{n}{3}}}\\). Denoting \\(f_z\\) the probability distribution function of \\(Z_n^u\\), we have \\[f_z(x,n) = f\left(\sqrt{\frac{n}{3}}x,n\right)\sqrt{\frac{n}{3}}.\\] We use the following local limit theorem, discussed in `\cite{petrov1975sums}`{=latex}\[p.214\].

<div class="theorem" markdown="1">

**Theorem 7** (Sahaidarova `\cite{shakhaidarova1966uniform}`{=latex}). *Let \\(\{X_n\}\\) be a sequence of independent random variables with a common density \\(p(x)\\), such that \\(E[|X_1|^3]<\infty\\), \\(E[X_1]= 0\\), \\(E[X^2_1]= 1\\) and \\(\sup p(x) \leq C\\). Let \\(p_n(x)\\) be the density of the random variable \\(\frac 1{\sqrt{n}} \sum_{j=1}^n X_j\\). Then \\[\sup_x |p_n(x) - \phi(x)| \leq \frac{A\beta_3}{\sqrt{n}} \max(1,C^3),\\] where \\(\phi\\) is the probability distribution function of a standard gaussian, \\(A\\) is an absolute constant, and \\(\beta_3 = E[|X_1|^3]\\).*

</div>

The theorem can be applied to a uniform continuous distribution with density \\(p^u(x) = \frac{1}{2\sqrt 3}\\) in the interval \\([-\sqrt{3},\sqrt{3}]\\), which has mean \\(0\\) and variance \\(1\\). We thus get, for every \\(x\in\mathbb{R}\\), \\[f_z(x,n)=p_{n}^u(x) \leq \phi(0) + \frac{A\beta_3}{\sqrt{n}} = \frac{1}{2\pi} + \frac{A\beta_3}{\sqrt{n}} \leq \frac{1}{2\pi} + A\frac{3\sqrt 3}{4} = c'_u.\\] In conclusion, setting \\(c_u=\sqrt{3}c'_u\\), for every \\(x\in\mathbb{R}\\) it holds that \\[f(x,n)=\sqrt{\frac{3}{n}}f_z\left(\sqrt{\frac{3}{n}}x,n\right) \leq \frac{\sqrt 3c'_u}{\sqrt{n}} = \frac{c_u}{\sqrt{n}}.\\] ◻

</div>

# Proof of Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> [apx:amplif]

<div class="proof" markdown="1">

*Proof of Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a>.* As anticipated, we proceed in three steps.

#### Step 1: Hoeffding bound.

We start by showing, following the idea at the base of `\cite[Corollary 3.3]{Lueker98}`{=latex}, that if \\(n'\\) is large enough, a standard Hoeffding bound ensures that with high probability a constant fraction of the sample follows a Uniform\\([-1,1]\\) distribution. Since we assumed that every \\(X_i\\) is a mixture of a Uniform\\([-1,1]\\) distribution with probability \\(p\\), and another distribution with density \\(g\\) (given by the factors \\(G_i\\)), we can rewrite \\(X_{i}=B_{i}\cdot U_{i}+(1-B_{i})\cdot G_{i}\\), with \\(U_i\\) being the uniform random variable, \\(G_i\\) being the random variable with density \\(g\\), \\(B_i\\) being independent Bernoulli random variables with probability \\(p\\).

Fix \\(\alpha=\alpha(p)\neq p\\), and assume, for now, that \\(n'\\) satisfies Eq. <a href="#eq:bound_n" data-reference-type="ref" data-reference="eq:bound_n">[eq:bound_n]</a>, and therefore, since \\(\varepsilon<\nicefrac{1}{2}\\), choosing \\(c_{\text{hyp}}=c_{\text{hyp}}(p)\ge (\alpha-p)^{-2}\\), ensures that, defining \\(\varepsilon'=\nicefrac{\varepsilon}{2}\\), \\[n'\ge c_{\text{hyp}}\log_2\frac{1}{\varepsilon}\ge \frac{1}{2(\alpha-p)^{2}}\ln\frac{1}{\varepsilon'}\\] and therefore \\[\Pr\left(\sum_{i}^{n'}B_{i} \leq \alpha n'\right)\leq e^{-2(\alpha-p)^2n'}\le e^{-\ln\frac{1}{\varepsilon'}} =\varepsilon'.\\] Thus \\[\Pr\left(\sum_{i}^{n'}B_{i}>\alpha n'\right)\geq 1-\varepsilon',\\] that is, with high probability, there is a set of indices \\(I\subseteq\left[n'\right]\\) of size \\(\left|I\right|\geq\alpha n'\\), such that for each \\(i\in I\\) it holds \\(B_{i}=1\\), i.e. \\(X_{i}\\) is uniformly distributed.

#### Step 2: Application of Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a> via rejection-sampling. [step-2-application-of-theorem-thmsrss-via-rejection-sampling.]

Lemma <a href="#lem:uniform" data-reference-type="ref" data-reference="lem:uniform">1</a> ensures that the uniform distribution of the \\(\left|I\right|\\) random variables selected in *Step 1* is sum-bounded. Conditionally on the event \\(\{\sum_{i}^{n'}B_{i}>\alpha n'\}\\), we can discard all random variables indexed outside \\(I\\) and apply directly Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a> to \\(\alpha n'\\) of the remaining ones, for any fixed \\(k\\) and \\(z\in[-\sqrt{k},\sqrt{k}]\\), since \\(\alpha c_{\text{hyp}}\ge 1\\) by construction. This guarantees a success probability of \\(c'_{\text{thm}}\\) for approximating the given target \\(z\\); thus, \\[\begin{aligned}
    &\Pr\left(\exists S_{z}\subset\left[n\right],\left|S_{z}\right|=k:\left|\Sigma_{S_{z}}-z\right|<\varepsilon'\right) \ge \\&\Pr\left(\exists S_{z}\subset\left[n\right],\left|S_{z}\right|=k:\left|\Sigma_{S_{z}}-z\right|<\varepsilon'\bigg\vert \sum_{i}^{n'}B_{i}>\alpha n'\right)\Pr\left(\sum_{i}^{n'}B_{i}>\alpha n'\right) \ge\\& \Pr\left(\exists S_{z}\subset I,\left|S_{z}\right|=k:\left|\Sigma_{S_{z}}-z\right|<\varepsilon'\bigg\vert |I|>\alpha n'\right)(1-\varepsilon')\ge c'_{\text{thm}}(1-\varepsilon')\ge \frac{3}{4}c'_{\text{thm}}=c_{\text{thm}}.
    
\end{aligned}\\]

#### Step 3: Amplification.

Finally, by a standard probability amplification argument and a union bound applied to Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a>, by paying an extra factor \\(\log_2(k/\varepsilon)\\) in Eq. <a href="#eq:bound_n" data-reference-type="ref" data-reference="eq:bound_n">[eq:bound_n]</a>, the constant \\(c_{\text{thm}}\\) can be amplified to \\(1-\varepsilon\\), and the existence of a suitable subset \\(S_z\\) holds simultaneously for all \\(z\in\left[-\sqrt{k},\sqrt{k}\right]\\). We now give more details on this amplification.

Recall that \\(\varepsilon' = \frac{\varepsilon}{2}\\), and let \\(c_{\text{amp}} =c_{\text{amp}}(p)= 8 \frac{c_{\text{hyp}}}{c_{\text{thm}}}\\) and \\(r = \frac{4}{c_{\text{thm}}}\ln \frac{k}{\varepsilon}\\). By assumption, \\[n\geq 
c_{\text{amp}}\frac{\log_{2}^2\frac{k}{\varepsilon}}{H_{2}\left(\frac{k}{n}\right)} \geq 
2r c_{\text{hyp}}\frac{\log_{2}\frac{k}{\varepsilon}}{H_{2}\left(\frac{k}{n}\right)}
\geq r c_{\text{hyp}}\frac{\log_{2}\frac{k}{\varepsilon'}}{H_{2}\left(\frac{k}{n}\right)},\\] where the last inequality is ensured by \\(\varepsilon<\nicefrac{1}{2}\\). By *Step 2*, we can apply Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a>, with \\(\varepsilon'\\) and \\(n' \ge  c_{\text{hyp}}\frac{\log_{2}\frac{k}{\varepsilon'}}{H_{2}\left(\frac{k}{n}\right)} = n^*\\), allowing us to prove that we can \\(\varepsilon'\\)-approximate any target \\(z\\) with probability at least \\(c_{\text{thm}}\\). The probability of failing to approximate some given \\(z\\) is then at most \\(1 - c_{\text{thm}}\\). From the sample \\(\Omega\\) of sum-bounded random variables take \\(r\\) subsamples (without replacement) of cardinality \\(n^*\\) each, \\(\Omega_1,\ldots,\Omega_r\\). The probability of failing to approximate some given \\(z\\) with subsetsums from \\(\Omega\\) is less than that of failing to approximate it with subsetsums from within every \\(\Omega_i\\)’s, and the latter probability is at most \\((1-c_{\text{thm}})^r\\); thus, for every \\(z\in[-\sqrt{k},\sqrt{k}]\\), \\[\Pr\left(\nexists S_{z}\subset\left[n\right],\left|S_{z}\right|=k:\left|\Sigma_{S_{z}}-z\right|<\varepsilon'\right) \leq (1-c_{\text{thm}})^r .\\] By an union bound, we also have that \\[\begin{aligned}
         & \Pr\left(\forall z\in\left[-\sqrt{k},\sqrt{k}\right],\exists S_{z}\subset\left[n\right],\left|S_{z}\right|=k:\left|\Sigma_{S_{z}}-z\right|<\varepsilon\right) \\
        & \ge \Pr\left(\forall z\in\left\{ -\sqrt{k}+i{\varepsilon'}:i\in\left[\frac{2}{\varepsilon'}\sqrt{k}\right]\right\},\exists S_{z}\subset\left[n\right],\left|S_{z}\right|=k:\left|\Sigma_{S_{z}}-z\right|<\varepsilon',\right) \\
         & =1-\Pr\left( \exists z\in\left\{ -\sqrt{k}+i{\varepsilon'}:i\in\left[\frac{2}{\varepsilon'}\sqrt{k}\right]\right\} ,
         \nexists S_{z}\subset\left[n\right],\left|S_{z}\right|=k:\left|\Sigma_{S_{z}}-z\right|<\varepsilon'\right) \\
         & \geq1-\sum_{z\in\left\{ -\sqrt{k}+i{\varepsilon'}:i\in\left[\frac{2}{\varepsilon'}\sqrt{k}\right]\right\} }\Pr\left(\nexists S_{z}\subset\left[n\right],\left|S_{z}\right|=k:\left|\Sigma_{S_{z}}-z\right|<\varepsilon'\right) \\
         & \geq 1-\frac{2}{\varepsilon'}\sqrt{k}\left(1-c_{\text{thm}}\right)^{r}=1-\frac{2}{\varepsilon'}\sqrt{k}\exp\left(\frac{4}{c_{\text{thm}}}\ln \left(\frac{k}{\varepsilon}\right)\cdot\ln(1-c_{\text{thm}})\right)
         \\
         &\geq 1-  \frac{2}{\varepsilon'}\sqrt{k} \exp\left( - 4\ln \frac{k}{\varepsilon} \right) = 1- \frac{2}{\varepsilon'}\sqrt{k} \frac{\varepsilon^4}{k^4} \geq 1-4\varepsilon^3\ge 1-\varepsilon,
    
\end{aligned}\\] where the last inequality is ensured by \\(\varepsilon<\nicefrac{1}{2}\\). This completes the proof. ◻

</div>

# Proof of Corollary <a href="#cor:SRSS_amp_simp" data-reference-type="ref" data-reference="cor:SRSS_amp_simp">2</a> [apx:amplif_simp]

<div class="proof" markdown="1">

*Proof of Corollary <a href="#cor:SRSS_amp_simp" data-reference-type="ref" data-reference="cor:SRSS_amp_simp">2</a>.* By definition of binary entropy, we have \\[H_{2}\left(\frac{k}{n}\right) = \frac{k}{n} \log_2\left(\frac{n}{k}\right) + \left(1-\frac{k}{n}\right) \log_2 \frac{n}{n-k}\\] In particular, since both terms in the previous equation are positive, we get \\[H_{2}\left(\frac{k}{n}\right) \ge \frac{k}{n} \log_2\left(\frac{n}{k}\right) \label{eq:h_lb_v2}\\] We now use  
efeq:h_lb_v2 to derive an upper bound for the quantity \\(\frac{c_{\text{amp}}}{H_{2}\left(\frac{k}{n}\right)} \frac{\log_2^2{k}+2log_2{k} \cdot 
log_2{\nicefrac{1}{\varepsilon}}}{n}\\), which will be used later: \\[\begin{aligned}
    \frac{c_{\text{amp}}}{H_{2}\left(\frac{k}{n}\right)} \frac{\log_2^2{k}+2log_2{k} \cdot 
log_2{\frac{1}{\varepsilon}}}{n} &\le \frac{c_{\text{amp}}}{\frac{k}{n} \log_2\left(\frac{n}{k}\right)} \frac{\log_2^2{k}+2log_2{k} \cdot  log_2{\frac{1}{\varepsilon}}}{n} \nonumber \\
    &= c_{\text{amp}} \frac{\log_2^2{k}+2log_2{k} \cdot  log_2{\frac{1}{\varepsilon}}}{k} \frac{1}{\log_2\left(\frac{n}{k}\right)} \label{eq:step_k_bounded_v2} \\
    &\le c_{\text{amp}} \frac{\log_2^2{k}+2log_2{k} \cdot  log_2{\frac{1}{\varepsilon}}}{k} \label{eq:step_k_bounded2_v2} \\
    &\le \frac{1}{2}, \label{eq:ub1_v2}
    
\end{aligned}\\] where from  
efeq:step_k_bounded_v2 to  
efeq:step_k_bounded2_v2 we used that \\(log_2{\nicefrac{n}{k}} \ge 1\\) for \\(k \le \nicefrac{n}{2}\\), and then the hypothesis \\(k \geq 2 c_{\text{amp}} \left(\log_{2}^2 k + 2log_2{k} \cdot  log_2{\frac{1}{\varepsilon}}\right)\\) directly gives  
efeq:ub1_v2. Let us now rewrite  
efeq:n_cond_amp in a more convenient form: \\[\begin{aligned}
        &n \frac{H_{2}\left(\frac{k}{n}\right)}{c_{\text{amp}}} \ge \log_{2}^2\frac{k}{\varepsilon} \nonumber \\
        &n \frac{H_{2}\left(\frac{k}{n}\right)}{c_{\text{amp}}} \ge \log_{2}^2 k + 2log_2{k} \cdot  log_2{\frac{1}{\varepsilon}} + \log_{2}^2\frac{1}{\varepsilon} \nonumber \\
        &n \left(\frac{H_{2}\left(\frac{k}{n}\right)}{c_{\text{amp}}} - \frac{\log_{2}^2 k + 2log_2{k} \cdot  log_2{\frac{1}{\varepsilon}}}{n} \right) \ge \log_{2}^2\frac{1}{\varepsilon} \nonumber \\
        &n \left(1 - \frac{c_{\text{amp}}}{H_{2}\left(\frac{k}{n}\right)} \frac{\log_{2}^2 k + 2log_2{k} \cdot  log_2{\frac{1}{\varepsilon}}}{n} \right) \ge  \frac{c_{\text{amp}}}{H_{2}\left(\frac{k}{n}\right)} \log_{2}^2\frac{1}{\varepsilon} \\
        &n \ge \frac{c_{\text{amp}}}{\left(1 - \frac{c_{\text{amp}}}{H_{2}\left(\frac{k}{n}\right)} \frac{\log_{2}^2 k + 2log_2{k} \cdot  log_2{\frac{1}{\varepsilon}}}{n} \right)} \frac{\log_{2}^2\frac{1}{\varepsilon}}{H_{2}\left(\frac{k}{n}\right)} \label{eq:edit_bound_n_v2} 
    
\end{aligned}\\] Using  
efeq:ub1_v2 we get \\[\begin{aligned}
    \frac{c_{\text{amp}}}{\left(1 - \frac{c_{\text{amp}}}{H_{2}\left(\frac{k}{n}\right)} \frac{\log_2^2{k}+2log_2{k} \cdot 
log_2{\frac{1}{\varepsilon}}}{n} \right)} \le 2 c_{\text{amp}}
    
\end{aligned}\\] To satisfy  
efeq:edit_bound_n_v2, we can then choose \\(n\\) such that \\[\begin{aligned}
    n &\ge 2 c_{\text{amp}} \frac{\log_{2}^2\frac{1}{\varepsilon}}{H_{2}\left(\frac{k}{n}\right)},
    
\end{aligned}\\] and then apply <a href="#cor:SRSS_amp" data-reference-type="ref+Label" data-reference="cor:SRSS_amp">1</a> to end the proof. ◻

</div>

# Details for the proof of Theorem <a href="#thm:srss" data-reference-type="ref" data-reference="thm:srss">2</a> [apx:srss_proof_details]

#### Proof of Eq. <a href="#eq:second_factor_second_term" data-reference-type="ref" data-reference="eq:second_factor_second_term">[eq:second_factor_second_term]</a>

Define \\(A=\tilde{S}^{\prime}\backslash\tilde{S}\\), and observe that \\[\begin{aligned}
 & \Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{\mu k,k-1}\right)\label{eq:start_of_law_of_tot_prob}\\
 & =\sum_{i=\mu k}^{k-1}\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{i}\right)\Pr\left(I_{i}\,|\,H_{\tilde{S}},I_{\mu k,k-1}\right)\label{eq:first_law_of_tot_prob}\\
 & =\sum_{i=\mu k}^{k-1}\int_{-\infty}^{\infty}\Pr\left(\left|\Sigma_{A}-\left(z-y\right)\right|<\varepsilon\,|\,\Sigma_{I}=y,I_{i},H_{\tilde{S}}\right)\Pr\left(\Sigma_{I}=y\,|\,H_{\tilde{S}},I_{i}\right)dy\notag\\
 & \qquad\cdot\Pr\left(I_{i}\,|\,H_{\tilde{S}},I_{\mu k,k-1}\right)\label{eq:second_law_of_tot_prob}\\
 & =\sum_{i=\mu k}^{k-1}\int_{-\infty}^{\infty}\Pr\left(\left|\Sigma_{A}-\left(z-y\right)\right|<\varepsilon\,|\,\Sigma_{I}=y, I_{i}\right)\Pr\left(\Sigma_{I}=y\,|\,H_{\tilde{S}},I_{i}\right)dy\notag\\
 & \qquad\cdot\Pr\left(I_{i}\,|\,H_{\tilde{S}},I_{\mu k,k-1}\right)\label{eq:after_dropping_conditioning}\\
 & \leq c\varepsilon\sum_{i=\mu k}^{k-1}\int_{-\infty}^{\infty}\Pr\left(\Sigma_{I}=y\,|\,H_{\tilde{S}},I_{i}\right)dy\Pr\left(I_{i}\,|\,H_{\tilde{S}},I_{\mu k,k-1}\right)\label{eq:last_step}\\
 &\le c\varepsilon \nonumber
\end{aligned}\\] where from Eq. <a href="#eq:start_of_law_of_tot_prob" data-reference-type="ref" data-reference="eq:start_of_law_of_tot_prob">[eq:start_of_law_of_tot_prob]</a> to Eq. <a href="#eq:first_law_of_tot_prob" data-reference-type="ref" data-reference="eq:first_law_of_tot_prob">[eq:first_law_of_tot_prob]</a> and from Eq. <a href="#eq:first_law_of_tot_prob" data-reference-type="ref" data-reference="eq:first_law_of_tot_prob">[eq:first_law_of_tot_prob]</a> to Eq. <a href="#eq:second_law_of_tot_prob" data-reference-type="ref" data-reference="eq:second_law_of_tot_prob">[eq:second_law_of_tot_prob]</a> we used the law of total probability;[^7] from Eq. <a href="#eq:second_law_of_tot_prob" data-reference-type="ref" data-reference="eq:second_law_of_tot_prob">[eq:second_law_of_tot_prob]</a> to Eq. <a href="#eq:after_dropping_conditioning" data-reference-type="ref" data-reference="eq:after_dropping_conditioning">[eq:after_dropping_conditioning]</a> we dropped the redundant event \\(H_{\tilde{S}}\\) in the conditioning, due to conditional independence; finally, from Eq. <a href="#eq:after_dropping_conditioning" data-reference-type="ref" data-reference="eq:after_dropping_conditioning">[eq:after_dropping_conditioning]</a> to Eq. <a href="#eq:last_step" data-reference-type="ref" data-reference="eq:last_step">[eq:last_step]</a> we used Definition <a href="#def:quasi_unif" data-reference-type="ref" data-reference="def:quasi_unif">1</a> which implies that for any \\(i\in\left\{ \mu k,...,k-1\right\}\\) it holds \\[\begin{aligned}
 & \Pr\left(\left|\Sigma_{A}-\left(z-y\right)\right|<\varepsilon\,|\,\Sigma_{I}=y,I_{i}\right)=\Pr\left(\left|\Sigma_{\left[k-i\right]}-\left(z-y\right)\right|<\varepsilon\right)\leq c\varepsilon.
\end{aligned}\\]

#### Proof of Eq. <a href="#eq:square_root_third_addendum" data-reference-type="ref" data-reference="eq:square_root_third_addendum">[eq:square_root_third_addendum]</a>

Let \\(A=\tilde{S}^{\prime}\backslash\tilde{S}\\). Analogously to the calculations from Eq. <a href="#eq:first_law_of_tot_prob" data-reference-type="ref" data-reference="eq:first_law_of_tot_prob">[eq:first_law_of_tot_prob]</a> to Eq. <a href="#eq:after_dropping_conditioning" data-reference-type="ref" data-reference="eq:after_dropping_conditioning">[eq:after_dropping_conditioning]</a>, by the law of total probability we have \\[\begin{aligned}
 & \sum_{i=0}^{\mu k-1}\Pr\left(I_{i}\right)\cdot\Pr\left(H_{\tilde{S}^{\prime}}\,|\,H_{\tilde{S}},I_{i}\right)\nonumber \\
 & =\sum_{i=0}^{\mu k-1}\Pr\left(I_{i}\right)\cdot\int_{-\infty}^{\infty}\Pr\left(\left|\Sigma_{A}-\left(z-y\right)\right|<\varepsilon\,|\,\Sigma_{I}=y,\,I_{i}\right)\Pr\left(\Sigma_{I}=y\,|\,H_{\tilde{S}},I_{i}\right)dy\nonumber \\
 & =\sum_{i=0}^{\mu k-1}\Pr\left(I_{i}\right)\cdot\int_{-\infty}^{\infty}\Pr\left(\left|\Sigma_{\left[k-i\right]}-\left(z-y\right)\right|<\varepsilon\right)\Pr\left(\Sigma_{I}=y\,|\,H_{\tilde{S}},I_{i}\right)dy\label{eq:using_def_in_third_addendum}\\
 & \leq c\frac{\varepsilon}{\sqrt{k}}\sum_{i=0}^{\mu k-1}\Pr\left(I_{i}\right)\cdot\int_{-\infty}^{\infty}\Pr\left(\Sigma_{I}=y\,|\,H_{\tilde{S}},I_{i}\right)dy\label{eq:after_using_def_in_second_addendum}\\
 &\le c\frac{\varepsilon}{\sqrt{k}} \nonumber
\end{aligned}\\]

where from Eq. <a href="#eq:using_def_in_third_addendum" data-reference-type="ref" data-reference="eq:using_def_in_third_addendum">[eq:using_def_in_third_addendum]</a> to Eq. <a href="#eq:after_using_def_in_second_addendum" data-reference-type="ref" data-reference="eq:after_using_def_in_second_addendum">[eq:after_using_def_in_second_addendum]</a> we used Definition <a href="#def:quasi_unif" data-reference-type="ref" data-reference="def:quasi_unif">1</a>, which implies that for any \\(i\in\left\{ 0,...,\frac{9}{10}k-1\right\}\\) it holds \\[\Pr\left(\left|\Sigma_{\left[k-i\right]}-\left(z-y\right)\right|<\varepsilon\right)\leq c'\frac{\varepsilon}{\sqrt{k-i}}\leq c\frac{\varepsilon}{\sqrt{k}}.\\]

# Proof of Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a> [apx:new_pensia]

In the proof we will refer to the following results, upon which `\cite{Pensia}`{=latex}\[Theorem 1\] relies (the statement below slightly differ as we fix two small typos in their notation and mixing coefficients). With the understanding that by a mixture \\(D\\) of a distribution \\(D_1\\) and \\(D_2\\) with probability \\(p\\) it is meant that the pdf (we adopt the convention that this term includes generalised functions, such as Dirac deltas for point masses) of \\(D\\) can be written as a convex combination of the pdf of \\(D_1\\) and that of \\(D_2\\), that is \\(f_D=pf_{D_1}+(1-p)f_{D_2}\\). For the unfamiliar reader, we note that in the literature this is often stated in short as \\(D=pD_1+(1-p)D_2\\).

<div id="lem:unifprod" class="lemma" markdown="1">

**Lemma 2** (`\cite{Pensia}`{=latex}\[Corollary 1\]). *Let \\(X\sim\\)Uniform\\([0,1]\\) (or \\(X\sim\\)Uniform\\([-1,0]\\)) and \\(Y\sim\\)Uniform\\([-1,1]\\) be independent random variables. Let \\(P\\) be the distribution of the \\(XY\\) and \\(\delta_0\\) the Dirac delta at \\(0\\). Let \\(D\\) be the distribution obtained as mixture of \\(\delta_0\\) and \\(P\\) with probability \\(\nicefrac{1}{2}\\). Then \\(D\\) is the mixture of a Uniform\\([-\nicefrac{1}{2},\nicefrac{1}{2}]\\) and some distribution \\(Q\\) with probability \\(\ln(2)/4\\).*

</div>

<div id="cor:Luek2.5" class="corollary" markdown="1">

**Corollary 3** (`\cite{Pensia}`{=latex}\[Corollary 2\]). *Let \\(X_1,\ldots,X_n\\) be iid with distribution \\(D\\) as defined in Lemma <a href="#lem:unifprod" data-reference-type="ref" data-reference="lem:unifprod">2</a>, where \\(n\ge C \ln(\nicefrac{2}{\varepsilon})\\) for some universal constant \\(C\\). Then \\[\Pr\left( \forall\,z\in[-1,1],\:\exists\, S\subset [n]\: :|z-\sum_{i\in S}X_i|\le \varepsilon\right)\ge 1-\varepsilon.\\]*

</div>

<div class="proof" markdown="1">

*Proof of Theorem <a href="#thm:pensia" data-reference-type="ref" data-reference="thm:pensia">3</a>.* The key idea is exploiting Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> at each step of the pruning strategy established in `\cite{Pensia}`{=latex}\[Theorem 1\], where Corollary <a href="#cor:Luek2.5" data-reference-type="ref" data-reference="cor:Luek2.5">3</a> is used instead. Without loss of generality, we replace their \\(\min\{\varepsilon,\delta\}\\) with \\(\varepsilon\\). For the sake of easily following the approach adopted in `\cite{Pensia}`{=latex}, let us define \\(n^*(x)\\) as the function \\[\label{n^*bis}
        n^*(x)=c_{\text{amp}}\frac{\log_{2}^2 {(kx)}}{H_{2}\left(\frac{k}{n^*(x)}\right)}\\] where \\(k=\gamma'n^*(x)\\). In the following, we use \\(n^*\\) as short for \\(n^*(1/\varepsilon)\\), and we will only explicitely provide an argument for \\(n^*\\) when it is different than \\(1/\varepsilon\\). For instance, in the last step of the proof, we will use \\(n^*(\nicefrac{2\ell d_i d_{i-1}}{\varepsilon})\\), which matches the definition of \\(n_i^*\\) given in Eq. <a href="#n^*" data-reference-type="ref" data-reference="n^*">[n^*]</a>.

Consider `\cite{Pensia}`{=latex}\[Lemma 1\]. When approximating a single link (that is, a weight), after the overparameterization (which creates an additional layer of width \\(2n^*\\) in between the input and the output node) via \\(4n^*\\) links, instead of pruning via Corollary <a href="#cor:Luek2.5" data-reference-type="ref" data-reference="cor:Luek2.5">3</a>, we prune via Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> twice in the second layer, that is we ensure that only \\(k=\gamma'n^*\\) edges yield the desired approximation, both in the edges corresponding to the positive part of the input weights and in those corresponding to the negative part. Thus we obtain at most \\(4k\\) surviving edges, after the preprocessing step and the pruning mask is applied. This yields a sparsity of at least \\(\alpha'=1-\gamma'\\). Note that it is because of the preprocessing step that we go from distributions Uniform\\([-1,1]\\) to distributions \\(D\\), as defined in Lemma <a href="#lem:unifprod" data-reference-type="ref" data-reference="lem:unifprod">2</a>, which are shown to be a mixture with Uniform\\([-1,1]\\) and therefore can be also handled via Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a>.

Consider `\cite{Pensia}`{=latex}\[Lemma 2\]. When approximating a real-valued multivariate linear function, after the overparameterization (which creates an additional layer of width \\(2dn^*(\nicefrac{d}{\varepsilon})\\) in between the \\(d\\) input nodes and the output node) one simply iterates the ideas of the previous case \\(d\\) times. For each input node, the overparameterization surviving the preprocessing step on the weights of the input layer is \\(4n^*(\nicefrac{d}{\varepsilon})\\). Pruning the second layer of the overparameterized link for each input via Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> with \\(k=\gamma'n^*(\nicefrac{d}{\varepsilon})\\) (again, performing this both on the edges corresponding to the positive part of the input weights and in those corresponding to the negative part), instead of exploiting Corollary <a href="#cor:Luek2.5" data-reference-type="ref" data-reference="cor:Luek2.5">3</a>, yields that at most \\(4dk\\) edges survive after the pruning mask is applied. This yields a sparsity of at least \\(\alpha'=1-\gamma'\\).

Finally, consider `\cite{Pensia}`{=latex}\[Lemma 3\]. When approximating a layer with input dimension \\(d_1\\) and output dimension \\(d_2\\), after the overparameterization (which creates an additional layer of width \\(2d_1n^*(\nicefrac{d_1d_2}{\varepsilon})\\) in between the input nodes and the output nodes) one iterates the ideas of the previous case \\(d_1\\) times in the input layer through the same preprocessing step, and \\(d_2\\) times in the output layer, one for each of the \\(d_1\\) blocks created by the preprocessing (essentially the weights in the input layer are *re-used* \\(d_2\\) times). For each input node, the overparameterization surviving the preprocessing step is at most \\(2(d_2+1)n^*(\nicefrac{d_1d_2}{\varepsilon})\\). Overall, after the preprocessing step, we have at most \\(2d_1(d_2+1)n^*(\nicefrac{d_1d_2}{\varepsilon})\\) parameters. We then use Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> (with \\(k=\gamma'n^*(\nicefrac{d_1d_2}{\varepsilon})\\)) to prune the number of parameters between the introduced additional layer and the \\(d_2\\) outputs down to \\(2d_1 d_2\gamma'n^*(\nicefrac{d_1d_2}{\varepsilon})\\). As for the edges between the \\(d_1\\) inputs and the additional layer, only those that reach a neuron in the additional layer, from which there is at least one outgoing edge towards the \\(d_2\\) outputs, are used; since for each of the \\(d_1\\) blocks of \\(2n^*(\nicefrac{d_1d_2}{\varepsilon})\\) neurons in the additional layer we only kept \\(2\gamma'n^*(\nicefrac{d_1d_2}{\varepsilon})\\) outgoing edges to each of the \\(d_2\\) output neurons, in the worst case (all the nodes involved in the subsetsums are disjoint) we keep \\(2d_2 \gamma'n^*(\nicefrac{d_1d_2}{\varepsilon})\\) of them for each of the \\(d_1\\) neurons. Globally, we are left with a total of at most \\(2d_1 d_2 \gamma'n^*(\nicefrac{d_1d_2}{\varepsilon})\\) edges both in the input layer and in the output layer, thus a total of \\(4d_1d_2 \gamma'n^*(\nicefrac{d_1d_2}{\varepsilon})\\) edges survive the pruning. The density of the surviving edges is then less than \\[\frac{4d_1d_2 \gamma'n^*(\nicefrac{d_1d_2}{\varepsilon})}{2d_1^2n^*(\nicefrac{d_1d_2}{\varepsilon})+2d_1d_2 n^*(\nicefrac{d_1d_2}{\varepsilon})}=\frac{2d_2 \gamma'}{d_1+d_2 }=\frac{(d_1\frac{d_2}{d_1}+d_2)\gamma'}{d_1+d_2}\le \rho_1\gamma',\\] where \\(\rho_1 = \max\left\lbrace \nicefrac{d_1}{d_2},\nicefrac{d_2}{d_1}\right\rbrace\\) and in the last inequality we used that \\(d_1\nicefrac{d_2}{d_1}+d_2\le\rho_1(d_1+d_2)\\) since \\(\rho_1\ge 1\\). This ensures a sparsity \\(\alpha'\ge 1-\rho_1\gamma'\\).

`\cite{Pensia}`{=latex}\[Theorem 1\] consists of performing, for every \\(i\in[\ell]\\), the previous step on layer \\(i\\) with input dimension \\(d_{i-1}\\) and output dimension \\(d_i\\). The overparameterization creates an additional layer of nodes of width \\(2d_{i-1}n^*(\nicefrac{2\ell d_{i-1}d_i}{\varepsilon})\\) in between the \\(d_{i-1}\\) input nodes and the \\(d_i\\) output nodes. Since the construction is stacked \\(\ell\\) times, this generates \\(2\ell\\) layers for the overparameterized network, which will therefore have a starting number of parameters \\[m = \sum_{i=1}^\ell 2 d_{i-1}^2 n^*(\nicefrac{2\ell d_{i-1}d_{i}}{\varepsilon}) + 2 d_{i-1}d_{i} n^*(\nicefrac{2\ell d_{i-1}d_{i}}{\varepsilon}).\\] Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> applied to each stacked overparameterized layer instead of Corollary <a href="#cor:Luek2.5" data-reference-type="ref" data-reference="cor:Luek2.5">3</a> as in the previous step yields that the total number of parameters left after the pruning is \\[m_t \le \sum_{i=1}^\ell 4 d_{i-1}d_{i} k_i,\\] where \\(k_i=\gamma'n^*(\nicefrac{2\ell d_{i-1}d_{i}}{\varepsilon})\\). Recall that \\(\rho = \max_{i} \rho_i\\), where \\(\rho_i=\max\{\nicefrac{d_i}{d_{i-1}}, \nicefrac{d_{i-1}}{d_i}\}\ge 1\\). Recall that \\(\gamma=\rho\gamma'\\). We obtain that \\[\begin{aligned}
    m_t  \le &\sum_{i=1}^\ell 2 d_{i-1}\frac{d_{i}}{d_{i-1}}d_{i-1} k_i + 2 d_{i-1}d_{i} k_i \\ 
    \le &\sum_{i=1}^\ell 2 d_{i-1} \rho_i d_{i-1} k_i + 2 d_{i-1}d_{i} \rho_i k_i \\ 
    \le &\rho\sum_{i=1}^\ell 2 d_{i-1}^2 k_i + 2 d_{i-1}d_{i} k_i \\ = & \rho\gamma'\sum_{i=1}^\ell 2 d_{i-1}^2 n^*(2\ell\nicefrac{ d_{i-1}d_{i}}{\varepsilon}) + 2 d_{i-1}d_{i} n^*(2\ell\nicefrac{ d_{i-1}d_{i}}{\varepsilon})=\gamma m
    
\end{aligned}\\] We then get that the density of the edges surviving the pruning is \\(\nicefrac{m_t}{m} \le \gamma\\), which implies a sparsity of at least \\(\alpha = 1 - \gamma\\). ◻

</div>

# Proof of Theorem <a href="#thm:lowerBoundStrong" data-reference-type="ref" data-reference="thm:lowerBoundStrong">5</a> [apx:lower]

<div class="proof" markdown="1">

*Proof of Theorem <a href="#thm:lowerBoundStrong" data-reference-type="ref" data-reference="thm:lowerBoundStrong">5</a>.* Consider the space \\(\mathcal{W}_k=\{W \in \mathbb R^{d\times d}: \|W\|\leq \sqrt k\}\\), and let \\(\mathcal{P}_k\\) be a \\(2\varepsilon\\)-separated set of \\(\mathcal{W}_k\\), i.e. a subset \\(\mathcal{P}_k \subset \mathcal{W}_k\\) such that for all distinct \\(W , W' \in \mathcal{P}_k\\) it holds \\(\|W - W'\| > 2\varepsilon\\). We denote \\(\mathcal{W}=\mathcal{W}_1\\), \\(\mathcal{P}=\mathcal{P}_1\\), and the set of all possible subnetworks of \\(g\\) as \\(\mathcal{G}\\) (note that this does not denote \\(\mathcal{G}_1\\), the set of all subnetworks of size \\(1\\)).

#### Step 1: Packing argument.

In `\cite{Pensia}`{=latex}\[Theorem 2, Step 1\], it is shown that any function \\(g'\\) can only approximate at most one member of \\(\mathcal{P}\\) for bounded input \\(x\\) (say, \\(\|x\|\le 1\\)). In particular, this also applies to functions \\(g'\\) representing the elements of \\(\mathcal{G}_k\\).

#### Step 2: Relation between \\(|\mathcal{G}_k|\\) and \\(|\mathcal{P}_k|\\).

By *Step 1*, in `\cite{Pensia}`{=latex}\[Theorem 2, Step 2\] it is shown that \\(|\mathcal{P}| \le 2|\mathcal{G}|\\), under the assumption of Eq. <a href="#eq:lowerBdApprox" data-reference-type="ref" data-reference="eq:lowerBdApprox">[eq:lowerBdApprox]</a>, with \\(\mathcal{G}_k\\) replaced by \\(\mathcal{G}\\)). Therefore, also by *Step 1*, replacing \\(\mathcal{P}\\) with \\(\mathcal{P}_k\\) and \\(\mathcal{G}\\) with \\(\mathcal{G}_k\\) in `\cite{Pensia}`{=latex}\[Theorem 2, Step 2\], it holds that \\(|\mathcal{P}_k| \le 2|\mathcal{G}_k|\\). Note that \\(|\mathcal{G}_k|=\binom{n}{k}\\), the number of different ways in which we can select \\(k\\) parameters out of \\(n\\), so we actually get \\[\label{eq:lb_binom}
        \binom{n}{k} >  \frac{|\mathcal {P}_k|}{2} .\\]

#### Step 3: Lower bound on \\(|\mathcal{P}_k|\\).

Let us now consider a \\(2 \varepsilon\\)-separated set \\(\mathcal{P}_k^{\max}\\) of maximal cardinality. In `\cite{Pensia}`{=latex}\[Theorem 2, Step 3\] it is shown that \\[|\mathcal{P}^{\max}|\ge\frac{\text{Vol}(\mathcal{W})}{\text{Vol}(\{W\in\mathcal{W}:\|W\|\le 2\varepsilon\})}=\left(\frac{1}{2\varepsilon}\right)^{d^2}.\\] Here \\(\text{Vol}\\) is the Lebesgue measure in \\(\mathbb{R}^{d\times d}\\) identified with \\(\mathbb{R}^{d^2}\\). By the exact same argument, replacing \\(\mathcal{W}\\) with \\(\mathcal{W}_k\\) and thus \\(\mathcal{P}^{\max}\\) with \\(\mathcal{P}_k^{\max}\\), it holds that \\[|\mathcal{P}_k^{\max}|\ge\frac{\text{Vol}(\mathcal{W}_k)}{\text{Vol}(\{W\in\mathcal{W}_k:\|W\|\le 2\varepsilon\})}=\left(\frac{\sqrt{k}}{2\varepsilon}\right)^{d^2}.\\] Combining this fact with Eq. <a href="#eq:lb_binom" data-reference-type="ref" data-reference="eq:lb_binom">[eq:lb_binom]</a> applied to \\(\mathcal{P}_k^{\max}\\) implies that \\[\label{eq:lower_bound_binomial}
        \binom{n}{k} > \frac 12 \left(\frac {\sqrt k}{2\varepsilon}\right)^{d^2}.\\]

#### Step 4: Lower bound on \\(n\\).

Consider the standard bound found in `\cite{macwilliams1977theory}`{=latex} \\[{n \choose k}\leq\sqrt{\frac{n}{2\pi k(n-k)}}2^{n H_2(\nicefrac{k}{n})}.\\] and combine it with with Eq. <a href="#eq:lower_bound_binomial" data-reference-type="ref" data-reference="eq:lower_bound_binomial">[eq:lower_bound_binomial]</a>. It follows that \\[2^{nH_{2}\left(\frac{k}{n}\right)}
     \geq
     \frac 12 \sqrt {\frac{2\pi k(n-k)}{n}} \left(\frac {\sqrt k}{2\varepsilon}\right)^{d^2}\\] and taking the logarithm of both sides yields the sought lower bound on \\(n\\): \\[\begin{aligned}
        nH_{2}\left(\frac{k}{n}\right)
        & \geq \frac{1}{2}\log_2\left(\frac{2\pi k(n-k)}{n}\right)+d^2\log_{2}\frac{\sqrt{k}}{2\varepsilon} -1\label{eq:lower_bound_entropy} \\
        & \geq d^2\left(\frac{1}{2}\log_{2}k+\log_{2}\frac{1}{\varepsilon}-1 \right)-1\label{eq:after_using_lambda}\\
        & \geq\frac{d^2}{2}\log_{2}\frac{k}{\varepsilon}\label{eq:after_dropping_1},
    
\end{aligned}\\] where from Eq. <a href="#eq:lower_bound_entropy" data-reference-type="ref" data-reference="eq:lower_bound_entropy">[eq:lower_bound_entropy]</a> to Eq. <a href="#eq:after_using_lambda" data-reference-type="ref" data-reference="eq:after_using_lambda">[eq:after_using_lambda]</a> we exploited the definition of \\(\lambda\\), which ensures that the first term in the r.h.s. of Eq. <a href="#eq:lower_bound_entropy" data-reference-type="ref" data-reference="eq:lower_bound_entropy">[eq:lower_bound_entropy]</a> is nonnegative;[^8] from Eq. <a href="#eq:after_using_lambda" data-reference-type="ref" data-reference="eq:after_using_lambda">[eq:after_using_lambda]</a> to Eq. <a href="#eq:after_dropping_1" data-reference-type="ref" data-reference="eq:after_dropping_1">[eq:after_dropping_1]</a> we used that for all \\(\varepsilon<\nicefrac{1}{16}\\) it holds that \\[d^2\left(\log_{2}\frac{1}{\varepsilon}-1\right)\ge 1.\\] ◻

</div>

# Details of comparison with Malach et al. `\cite{malachProvingLotteryTicket2020}`{=latex} [apx:malach]

We show that \\(cz \ge c_{\text{amp}} \frac{\log_2^2 (c z^2)}{\log_2(z)}\\) holds for a big enough constant \\(c\\). Recall that \\(z=\frac{m_t}{\varepsilon}\\), so we can always assume \\(\log_2(z) \ge 1\\). We have \\[\begin{aligned}
\log_2^2 (c z^2) &= (\log_2 (c) + 2\log_2 (z))^2 \\
&= \log_2^2 (c) + 4\log_2 (c) \log_2 (z) + 4\log_2^2 (z) \\
&\stackrel{(a)}{\le} 6(\log_2^2 (c) + \log_2^2 (z)) \\
&\stackrel{(b)}{\le} 12 \log_2^2 (c) \log_2^2 (z),
\end{aligned}\\] where in \\((a)\\) we used that \\(2ab \le a^2 + b^2\\), and in \\((b)\\) that \\(a+b \le 2ab\\) for \\(a\\) and \\(b\\) greater than 1.

We can then focus on showing that there is a big enough constant \\(c\\) such that \\(cz \ge 12 c_{\text{amp}} \log_2^2 (c) \log_2^2 (z)\\). We get \\(c \ge 12 c_{\text{amp}} \log_2^2 (c) \frac{\log_2^2 (z)}{z}\\), and we have \\[\begin{aligned}
    \log_2^2 (c) \frac{\log_2^2 (z)}{z} & \le \log_2^2 (c) \\
    &\le \sqrt{c}.
\end{aligned}\\]

We can then focus on \\(c \ge 12 c_{\text{amp}} \sqrt{c}\\), which is satisfied for \\(c \ge 144 c_{\text{amp}}^2\\).

[^1]: The definition in `\cite[Corollary 3.3]{Lueker98}`{=latex} is actually more general, since it concerns a different problem.

[^2]: We believe that Lemma <a href="#lem:uniform" data-reference-type="ref" data-reference="lem:uniform">1</a> is known, but we could not find a reference.

[^3]: Equivalently, the hypothesis of Corollary <a href="#cor:SRSS_amp" data-reference-type="ref" data-reference="cor:SRSS_amp">1</a> must hold up to a factor \\(\Theta(\log_{2}\frac{k}{\varepsilon})\\).

[^4]: follows from the upper bound \\(\sum_{i=1}^{k} {n \choose i} \leq \left(\frac{en}{k}\right)^k\\) on the partial sum of binomial coefficients.

[^5]: It should be known that \\(I_n\\) is unimodal with a mode in \\(n/2\\), but we were not able to find a reference. It is instructive to note, assuming that \\(I_n\\) is unimodal with a mode in \\(n/2\\), it directly follows that its probability density function is increasing on the interval \\([0,n/2]\\), and then decreasing over \\([n/2,n]\\). This implies that \\(f(x,n)\\) (the density of \\(\Sigma^{\mathcal U_n}_{[n]}\\)), is non decreasing in the interval \\([-n,0]\\), has maximum at \\(0\\), and non increasing \\([0,n]\\) for all \\(n\ge 2\\).

[^6]: It is also possible to obtain our result via classical Berry-Esseen inequality, due to the improved upper bound of \\(0.4748\\) on the absolute constant, provided in `\cite{Shevtsova11}`{=latex}. This would require replacing with \\(900\\) the cut-off value for \\(n\\), which is \\(18\\) in the current version of the argument.

[^7]: For simplicity, we denote the density of \\(\Sigma_{I}\\) conditional on \\(H_{\tilde{S}}\cap I_{i}\\) as \\(\Pr\left(\Sigma_{I}=y\,|\,H_{\tilde{S}},I_{i}\right)\\).

[^8]: This term being nonnegative is equivalent to \\(k(1-\nicefrac{k}{n})\ge\nicefrac{1}{2\pi}\\), and since \\(1\le k\le\lambda n\\), any \\(\lambda\le 1-\nicefrac{1}{2\pi}\\) ensures it.
