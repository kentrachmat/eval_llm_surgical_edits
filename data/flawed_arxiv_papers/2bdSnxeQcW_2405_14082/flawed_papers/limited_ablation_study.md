# Exclusively Penalized Q-learning for Offline Reinforcement Learning

## Abstract

Constraint-based offline reinforcement learning (RL) involves policy constraints or imposing penalties on the value function to mitigate overestimation errors caused by distributional shift. This paper focuses on a limitation in existing offline RL methods with penalized value function, indicating the potential for underestimation bias due to unnecessary bias introduced in the value function. To address this concern, we propose Exclusively Penalized Q-learning (EPQ), which reduces estimation bias in the value function by selectively penalizing states that are prone to inducing estimation errors. Numerical results show that our method significantly reduces underestimation bias and improves performance in various offline control tasks compared to other offline RL methods.

# Introduction [Introduction]

[^1]

[^2]

Reinforcement learning (RL) is gaining significant attention for solving complex Markov decision process (MDP) tasks. Traditionally, online RL develops advanced decision-making strategies through continuous interaction with environments `\cite{ppo, ddpg, td3, sac, disc, diversity}`{=latex}. However, in real-world scenarios, interacting with the environment can be costly, particularly in high-risk environments like disaster situations, where obtaining sufficient data for learning is challenging `\cite{qin2022neorl, zhou2023real}`{=latex}. In such setups, the need for exploration `\cite{cbexp, diexp, maxmin, fox}`{=latex} to discover optimal strategies often incurs additional costs, as agents must try various actions, some of which may be inefficient or risky `\cite{safe, robust}`{=latex}. This highlights the significance of research on offline setups, where policies are learned using pre-collected data without any direct interaction with the environment `\cite{offlineintro, kumar2022should}`{=latex}. In offline setups, policy actions not present in the data may introduce extrapolation errors, disrupting accurate value estimation by causing a large overestimation error in the value function, known as the distributional shift problem `\cite{BCQ}`{=latex}.

To address the distributional shift problem, `\citet{BCQ}`{=latex} proposes batch-constrained \\(Q\\)-learning (BCQ), assuming that policy actions are selected from the dataset only. Ensuring optimal convergence of both the policy and value function under batch-constrained RL setups `\cite{BCQ}`{=latex}, BCQ demonstrates stable learning in offline setups and outperforms behavior cloning (BC) techniques `\cite{bc}`{=latex}, which simply mimic actions from the dataset. However, the policy constraint of BCQ strongly limits the policy space, prompting further research to find improved policies by relaxing constraints based on the support of the policy using metrics like maximum mean discrepancy (MMD) `\cite{BEAR}`{=latex} or Kullback–Leibler (KL) divergence `\cite{brac}`{=latex}. While these methods moderately relax policy restrictions, the issue of limited policies persists. Thus, instead of constraining the policy space directly, alternative offline RL methods have been proposed to reduce overestimation bias based on penalized \\(Q\\)-functions `\cite{CQL, CPQ}`{=latex}. Conservative \\(Q\\)-learning (CQL) `\cite{CQL}`{=latex}, a representative offline RL algorithm using \\(Q\\)-penalty, penalizes the \\(Q\\)-function for policy actions and provides a bonus to the \\(Q\\)-function for actions in the dataset. Consequently, CQL selects more actions from the dataset, effectively reducing overestimation errors without policy constraints.

While CQL has demonstrated outstanding performance across various offline tasks, we observed that it introduces unnecessary estimation bias in the value function for states that do not contribute to overestimation. This issue becomes more pronounced as the level of penalty increases, resulting in performance degradation. To address this issue, this paper introduces a novel Exclusively Penalized Q-learning (EPQ) method for efficient offline RL. EPQ imposes a threshold-based penalty on the value function exclusively for states causing estimation errors to mitigate overestimation bias without introducing unnecessary bias in offline learning. Experimental results demonstrate that our proposed method effectively reduces both overestimation bias due to distributional shift and underestimation bias due to the penalty, allowing a more accurate evaluation of the current policy compared to the existing methods. Numerical results reveal that EPQ significantly outperforms other state-of-the-art offline RL algorithms on various D4RL tasks `\cite{d4rl}`{=latex}.

# Preliminaries

## Markov Decision Process and Offline RL

We consider a Markov Decision Process (MDP) environment denoted as \\(\mathcal{M}:= (\mathcal{S}, \mathcal{A}, P, R, \gamma)\\), where \\(\mathcal{S}\\) is the state space, \\(\mathcal{A}\\) is the action space, \\(P\\) represents the transition probability, \\(\gamma\\) is the discount factor, and \\(R\\) is the bounded reward function. In offline RL, transition samples \\(d_t=(s_t,a_t,r_t,s_{t+1})\\) are generated by a behavior policy \\(\beta\\) and stored in the dataset \\(D\\). We can empirically estimate \\(\beta\\) as \\(\hat{\beta}(a|s) = \frac{N(s,a)}{N(s)}\\), where \\(N\\) represents the number of data points in \\(D\\). We assume that \\(\mathbb{E}_{s\sim D,a\sim\beta}[f(s,a)] \approx \mathbb{E}_{s\sim D,a\sim\hat{\beta}}[f(s,a)] = \mathbb{E}_{s,a\sim D}[f(s,a)]\\) for arbitrary function \\(f\\). Utilizing only the provided dataset without interacting with the environment, our objective is to find a target policy \\(\pi\\) that maximizes the expected discounted return, denoted as \\(J(\pi):= \mathbb{E}_{s_0,a_0,s_1,\cdots\sim \pi}[G_0]\\), where \\(G_t = \sum^\infty_{l=t} \gamma^{l-t} R(s_l, a_l)\\) represents the discounted return.

## Distributional Shift Problem in Offline RL

In online RL, the optimal policy that maximizes \\(J(\pi)\\) is found through iterative policy evaluation and policy improvement `\cite{ddpg, td3}`{=latex}. For policy evaluation, the action value function is defined as \\(Q^\pi(s_t, a_t):= \mathbb{E}_{s_t,a_t,s_{t+1},\cdots\sim \pi}[\sum^\infty_{l=t}\gamma^{l-t} R(s_l, a_l)|s_t,~a_t]\\). \\(Q^\pi\\) can be estimated by iteratively applying the Bellman operator \\(\mathcal{B}^\pi\\) to an arbitrary \\(Q\\)-function, where \\((\mathcal{B}^\pi Q)(s,a):=R(s,a) + \gamma \mathbb{E}_{s'\sim P(\cdot|s,a),~a'\sim\pi(\cdot|s')}[Q(s', a')]\\). The \\(Q\\)-function is updated to minimize the Bellman error using the dataset \\(D\\), given by \\(\mathbb{E}_{s,a\sim D}\left[\left(Q(s, a) - \mathcal{B}^\pi Q(s, a) \right)^2\right]\\). In offline RL, samples are generated by the behavior policy \\(\beta\\) only, resulting in estimation errors in the \\(Q\\)-function for policy actions not present in the dataset \\(D\\). The policy \\(\pi\\) is updated to maximize the \\(Q\\)-function, incorporating the estimation error in the policy improvement step. This process accumulates positive bias in the \\(Q\\)-function as iterations progress `\cite{BCQ}`{=latex}.

## Conservative \\(Q\\)-learning [subsec:CQL]

To mitigate overestimation in offline RL, conservative Q-learning (CQL) `\cite{CQL}`{=latex} penalizes the \\(Q\\)-function for the policy actions \\(a\sim \pi\\) and increases the \\(Q\\)-function for the data actions \\(a\sim\hat{\beta}\\) while minimizing the Bellman error, where the \\(Q\\)-loss function of CQL is given by \\[\begin{aligned}
\frac{1}{2}\mathbb{E}_{s,a,s'\sim D}\left[\left(Q(s, a) - \mathcal{B}^\pi Q(s, a)  \right)^2\right]+\alpha \mathbb{E}_{s\sim D}[ \mathbb{E}_{a \sim \pi}[Q(s, a)] - \mathbb{E}_{a \sim \hat{\beta}}[Q(s, a)]],
\label{eq:CQLQ}
\end{aligned}\\] where \\(\alpha \geq 0\\) is a penalizing constant. From the value update in <a href="#eq:CQLQ" data-reference-type="eqref" data-reference="eq:CQLQ">[eq:CQLQ]</a>, the average \\(Q\\)-value of data actions \\(\mathbb{E}_{a \sim \hat{\beta}}[Q(s, a)]\\) becomes larger than the average \\(Q\\)-value of target policy actions \\(\mathbb{E}_{a \sim \pi}[Q(s, a)]\\) as \\(\alpha\\) increases. As a result, the policy will tend to choose the data actions more from the policy improvement step, effectively reducing overestimation error in the \\(Q\\)-function `\cite{CQL}`{=latex}.

# Methodology [sec:method]

## Motivation: Necessity of Mitigating Unnecessary Estimation Bias [subsec:estimbias]

In this section, we focus on the penalization behavior of CQL, one of the most representative penalty-based offline RL methods, and present an illustrative example to show that unnecessary estimation bias can occur in the \\(Q\\)-function due to the penalization. As explained in Section <a href="#subsec:CQL" data-reference-type="ref" data-reference="subsec:CQL">2.3</a>, CQL penalizes the \\(Q\\)-function for policy actions and increases the \\(Q\\)-function for data actions in <a href="#eq:CQLQ" data-reference-type="eqref" data-reference="eq:CQLQ">[eq:CQLQ]</a>. When examining the \\(Q\\)-function for each state-action pair \\((s,a)\\), the \\(Q\\)-value increases if \\(\pi(a|s)>\hat{\beta}(a|s)\\); otherwise, the \\(Q\\)-value decreases as the penalizing constant \\(\alpha\\) becomes sufficiently large `\cite{CQL}`{=latex}.

<figure id="fig:motive">
<p>To visually demonstrate this, Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a> depicts histograms of the fixed policy <span class="math inline"><em>π</em></span> and the estimated behavior policy <span class="math inline"><em>β̂</em></span> for various <span class="math inline"><em>π</em></span> and <span class="math inline"><em>β</em></span> at the initial state <span class="math inline"><em>s</em><sub>0</sub></span> on the Pendulum task with a single-dimensional action space in OpenAI Gym tasks <span class="citation" data-cites="gym"></span>, as cases (a), (b), and (c), along with the estimation bias in the <span class="math inline"><em>Q</em></span>-function for CQL with various penalizing factors <span class="math inline"><em>α</em></span>. In this example, for all states except the initial state, we consider <span class="math inline"><em>π</em> = <em>β</em> = Unif(−2, 2)</span>. In each case, CQL only updates the <span class="math inline"><em>Q</em></span>-function with its penalty to evaluate <span class="math inline"><em>π</em></span> in an offline setup, as shown in equation <a href="#eq:CQLQ" data-reference-type="eqref" data-reference="eq:CQLQ">[eq:CQLQ]</a>, and we plot the estimation bias of CQL, which represents the average difference between the learned <span class="math inline"><em>Q</em></span>-function and the expected return <span class="math inline"><em>G</em><sub>0</sub></span>.</p>
<p><img src="./figures/fig1_v2.PNG"" alt="image" /> <span id="fig:motive" data-label="fig:motive"></span></p>
</figure>

From the results in Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>, we observe that CQL suffers from unnecessary estimation bias in the \\(Q\\)-function for cases (a) and (b). In both cases, the histograms illustrate that policy actions are fully contained in the dataset \\(\hat{\beta}\\), suggesting that the estimation error in the Bellman update is unlikely to occur even without any penalty. However, CQL introduces a substantial negative bias for actions near \\(0\\) where \\(\pi(0|s_0) > \hat{\beta}(0|s_0)\\) and a positive bias for other actions. Furthermore, the bias intensifies as the penalty level \\(\alpha\\) increases. In order to mitigate this bias, reducing the penalty level \\(\alpha\\) to zero may seem intuitive in cases like Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>(a) and Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>(b). However, such an approach would be inadequate in cases like Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>(c). In this case, because policy actions close to 0 are rare in the dataset, penalization is necessary to address overestimation caused by estimation errors in offline learning. Furthermore, this problem may become more severe in actual offline learning situations, as the policy continues to change as learning progresses, compared to situations where a fixed policy is assumed.

## Exclusively Penalized Q-learning [subsec:epq]

To address the issue outlined in Section <a href="#subsec:estimbias" data-reference-type="ref" data-reference="subsec:estimbias">3.1</a>, our goal is to selectively give a penalty to the \\(Q\\)-function in cases like Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>(c), where policy actions are insufficient in the dataset while minimizing unnecessary bias due to the penalty in scenarios like Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>(a) and Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>(b), where policy actions are sufficient in the dataset. To achieve this goal, we introduce a novel exclusive penalty \\(\mathcal{P}_\tau\\) defined by

\\[\mathcal{P}_\tau := \underbrace{f_\tau^{\pi,\hat{\beta}}(s)}_{\textrm{penalty adaptation factor}} \cdot \underbrace{\left(\frac{\pi(a|s)}{\hat{\beta}(a|s)}-1\right)}_{\textrm{penalty term}},
\label{eq:TAP}\\]

<figure id="fig:motive_ill">
<img src="./figures/r1.png"" style="width:90.0%" />
<figcaption>An illustration of our exclusive penalty: (a) The log-probability of <span class="math inline"><em>β̂</em></span> and the thresholds <span class="math inline"><em>τ</em><sub>1</sub></span> and <span class="math inline"><em>τ</em><sub>2</sub></span> according to the number of data samples <span class="math inline"><em>N</em><sub>1</sub></span> and <span class="math inline"><em>N</em><sub>2</sub></span>, where <span class="math inline"><em>N</em><sub>1</sub> &lt;  &lt; <em>N</em><sub>2</sub></span>. (b) The penalty adaptation factor <span class="math inline"><em>f</em><sub><em>τ</em></sub><sup><em>π</em>, <em>β̂</em></sup></span> which represents the amount of adaptive penalty, indicating how much <span class="math inline">log <em>β̂</em></span> exceeds the threshold <span class="math inline"><em>τ</em></span>. Three different policies <span class="math inline"><em>π</em><sub><em>i</em></sub>, <em>i</em> = 1, 2, 3</span>, are considered.</figcaption>
</figure>

where \\(f_\tau^{\pi,\hat{\beta}}(s) = \mathbb{E}_{a\sim\pi(\cdot|s)}[x_\tau^{\hat{\beta}}]\\) is a penalty adaptation factor for a given \\(\hat{\beta}\\) and policy \\(\pi\\). Here, \\(x_\tau^{\hat{\beta}}=\min(1.0,\exp(-(\log\hat{\beta}(a|s) - \tau)))\\) represents the amount of adaptive penalty that is reduced as log \\(\hat{\beta}\\) exceeds the threshold \\(\tau\\). Thus, the adaptation factor \\(f^{\pi,\hat{\beta}}_\tau\\) indicates the average penalty that policy actions should receive. If the probability of estimated behavior policy \\(\hat{\beta}\\) for policy actions exceeds the threshold \\(\tau\\), i.e., policy actions are sufficiently present in the dataset, then \\(x_\tau^{\hat{\beta}}\\) will be smaller than 1 and reduce the amount of penalty as much as the amount by which \\(\hat{\beta}\\) exceeds the threshold \\(\tau\\) to avoid unnecessary bias introduced in Section <a href="#subsec:estimbias" data-reference-type="ref" data-reference="subsec:estimbias">3.1</a>. Otherwise, it will be 1 due to \\(\min(1.0,\cdot)\\) to maintain the penalty since policy actions are insufficient in the dataset. The latter penalty term \\(\frac{\pi(a|s)}{\hat{\beta}(a|s)}-1\\), positive if \\(\pi(a|s)>\hat{\beta}(a|s)\\) and otherwise negative, imposes a positive penalty on the \\(Q\\)-function when \\(\pi(a|s) > \hat{\beta}(a|s)\\), and otherwise, it increases the \\(Q\\)-function since the penalty is negative, as the Q-penalization method considered in CQL `\cite{CQL}`{=latex}.

To elaborate further on our proposed penalty, Fig. <a href="#fig:motive_ill" data-reference-type="ref" data-reference="fig:motive_ill">2</a>(a) depicts the log-probability of \\(\hat{\beta}\\) and the thresholds \\(\tau\\) used for penalty adaptation, with \\(N\\) representing the number of data points. In Fig. <a href="#fig:motive_ill" data-reference-type="ref" data-reference="fig:motive_ill">2</a>(a), if the log-probability \\(\log\hat{\beta}\\) of an action \\(a \in \mathcal{A}\\) exceeds the threshold \\(\tau\\), this indicates that the action \\(a\\) is sufficiently represented in the dataset. Thus, we reduce the penalty for such actions. Furthermore, as shown in Fig. <a href="#fig:motive_ill" data-reference-type="ref" data-reference="fig:motive_ill">2</a>(a), when the number of actions increase from \\(N_1\\) to \\(N_2\\), the threshold for determining "enough data" decreases from \\(\tau_1\\) to \\(\tau_2\\), even if the data distribution remains unchanged.

Furthermore, to explain the role of the threshold \\(\tau\\) in the proposed penalty \\(\mathcal{P}_\tau\\), we consider two thresholds, \\(\tau_1\\) and \\(\tau_2\\). In Fig. <a href="#fig:motive_ill" data-reference-type="ref" data-reference="fig:motive_ill">2</a>(b), which illustrates the proposed penalty adaptation factor \\(f^{\pi,\hat{\beta}}_{\tau_1}\\) and \\(f^{\pi,\hat{\beta}}_{\tau_2}\\) for thresholds \\(\tau_1\\) and \\(\tau_2\\), \\(x_{\tau_1}^{\hat{\beta}}\\) is larger than \\(x_{\tau_2}^{\hat{\beta}}\\) because \\(\tau_1 > \tau_2\\). As a result, in the case of \\(\tau_1\\), \\(\mathcal{P}_{\tau_1}\\) only reduces the penalty for \\(\pi_3\\). In other words, \\(f_{\tau_1}^{\pi_1,\hat{\beta}}=f_{\tau_1}^{\pi_2,\hat{\beta}}=1,\\) and \\(f_{\tau_1}^{\pi_3,\hat{\beta}} <1\\). On the other hand, as the number of data samples increases from \\(N_1\\) to \\(N_2\\), more actions generated by the behavior policy \\(\beta\\) will be stored in the dataset, so policy actions are more likely to be in the dataset. In this case, the threshold should be lowered from \\(\tau_1\\) to \\(\tau_2\\). As a result, \\(\hat{\beta}\\) exceeds the threshold \\(\tau_2\\) in the support of all policies \\(\pi_i\\), and \\(\mathcal{P}_{\tau_2}\\) reduces the penalty in the support of all policies \\(\pi_i\\), i.e., \\(f_{\tau_2}^{\pi_3,\hat{\beta}} < f_{\tau_2}^{\pi_1,\hat{\beta}} < f_{\tau_2}^{\pi_2,\hat{\beta}} < 1\\). Thus, even without knowing the exact number of data samples, the proposed penalty \\(\mathcal{P}_\tau\\) allows adjusting the penalty level appropriately according to the given number of data samples based on the threshold \\(\tau\\).

Now, we propose exclusively penalized Q-learning (EPQ), a novel offline RL method that minimizes the Bellman error while imposing the proposed exclusive penalty \\(\mathcal{P}_\tau\\) on the \\(Q\\)-function as follows: \\[\begin{aligned}
&\min_Q ~\mathbb{E}_{s,a,s'\sim D}\left[\left(Q(s, a) - \{\mathcal{B}^\pi Q(s, a) - \alpha \mathcal{P}_\tau\} \right)^2\right].
\label{eq:bellmanours}
\end{aligned}\\]

Then, we can prove that the final \\(Q\\)-function of EPQ underestimates the true value function \\(Q^\pi\\) in offline RL if \\(\alpha\\) is sufficiently large, as stated in the following theorem. This indicates that the proposed EPQ can successfully reduce overestimation bias in offline RL, while simultaneously alleviating unnecessary bias based on the proposed penalty \\(\mathcal{P}_\tau\\).

<div id="thm:penalty" class="theorem" markdown="1">

**Theorem 1**. *We denote the \\(Q\\)-function converged from the \\(Q\\)-update of EPQ using the proposed penalty \\(\mathcal{P}_\tau\\) in <a href="#eq:bellmanours" data-reference-type="eqref" data-reference="eq:bellmanours">[eq:bellmanours]</a> by \\(\hat{Q}^\pi\\). Then, the expected value of \\(\hat{Q}^\pi\\) underestimates the expected true policy value, i.e., \\(\mathbb{E}_{a\sim\pi}[\hat{Q}^\pi(s,a)] \leq \mathbb{E}_{a\sim\pi}[Q^\pi(s,a)],  \forall s \in D\\), with high probability \\(1-\delta\\) for some \\(\delta \in (0,1)\\), if the penalizing factor \\(\alpha\\) is sufficiently large. Furthermore, the proposed penalty reduces the average penalty for policy actions compared to the average penalty of CQL.*

</div>

**Proof)** Proof of Theorem <a href="#thm:penalty" data-reference-type="ref" data-reference="thm:penalty">1</a> is provided in Appendix <a href="#sec:proof" data-reference-type="ref" data-reference="sec:proof">7</a>.

<figure id="fig:motiveours">
<img src="./figures/fig3_v2.PNG"" />
<img src="./figures/fig4-v3.png"" />
<figcaption>Histograms of <span class="math inline"><em>π</em></span> and <span class="math inline"><em>β̂</em></span> (left axis), and the estimation bias of CQL and EPQ with various <span class="math inline"><em>τ</em></span> (right axis) for three cases: (a) <span class="math inline"><em>β</em> = Unif(−2, 2)</span> and <span class="math inline"><em>π</em> = <em>N</em>(0, 0.2)</span> (b) <span class="math inline">$\beta= \frac{1}{2}  N(-1,0.3) + \frac{1}{2}  N(1,0.3)$</span> and <span class="math inline"><em>π</em> = <em>N</em>(1, 0.2)</span> (c) <span class="math inline">$\beta= \frac{1}{2}  N(-1,0.3) + \frac{1}{2}  N(1,0.3)$</span> and <span class="math inline"><em>π</em> = <em>N</em>(0, 0.2)</span>.</figcaption>
</figure>

In order to demonstrate the \\(Q\\)-function convergence behavior of the proposed EPQ in more detail, we revisit the previous Pendulum task in Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>. Fig. <a href="#fig:reducedpenalty" data-reference-type="ref" data-reference="fig:reducedpenalty">[fig:reducedpenalty]</a> shows the histogram of \\(\hat{\beta}\\) and the penalty adaptation factor \\(f_\tau^{\pi,\hat{\beta}}(s)\\) for Gaussian policy \\(\pi= N(\mu,0.2)\\), where \\(\mu\\) varies from \\(-2\\) to \\(2\\), with varying \\(\beta\\). In Fig. <a href="#fig:reducedpenalty" data-reference-type="ref" data-reference="fig:reducedpenalty">[fig:reducedpenalty]</a>(a), \\(f_\tau^{\pi,\hat{\beta}}(s)\\) should be less than 1 for any policy mean \\(\mu\\) since all policy actions are sufficient in the dataset. In <a href="#fig:reducedpenalty" data-reference-type="ref" data-reference="fig:reducedpenalty">[fig:reducedpenalty]</a>(b), \\(f_\tau^{\pi,\hat{\beta}}(s)\\) is less than 1 only if the \\(\hat{\beta}\\) probability near the policy mean \\(\mu\\) is high, and otherwise, \\(f_\tau^{\pi,\hat{\beta}}(s)\\) is 1, which indicates the lack of policy action in the dataset. Thus, the result shows that \\(f_\tau^{\pi,\hat{\beta}}(s)\\) reflects our motivation in Section <a href="#subsec:estimbias" data-reference-type="ref" data-reference="subsec:estimbias">3.1</a> well. Moreover, Fig. <a href="#fig:motiveours" data-reference-type="ref" data-reference="fig:motiveours">3</a> compares the estimation bias curves of CQL and EPQ with \\(\alpha=10\\) in the scenarios presented in Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>. CQL exhibits unnecessary bias for situations in Fig. <a href="#fig:motiveours" data-reference-type="ref" data-reference="fig:motiveours">3</a>(a) and Fig. <a href="#fig:motiveours" data-reference-type="ref" data-reference="fig:motiveours">3</a>(b) where no penalty is needed, as discussed in Section <a href="#subsec:estimbias" data-reference-type="ref" data-reference="subsec:estimbias">3.1</a>. Conversely, our proposed method effectively reduces estimation bias in these cases while appropriately maintaining the penalty for the scenario in Fig. <a href="#fig:motiveours" data-reference-type="ref" data-reference="fig:motiveours">3</a>(c) where penalization is required. This experiment demonstrates the effectiveness of our proposed approach, and the subsequent numerical results in Section <a href="#sec:experiment" data-reference-type="ref" data-reference="sec:experiment">4</a> will numerically show that our method significantly reduces estimation bias in offline learning, resulting in improved performance.

## Prioritized Dataset [subsec:pd]

<figure id="fig:motive_pd">
<img src="./figures/r2f.png"" style="width:90.0%" />
<figcaption>An illustration of the prioritized dataset. As the policy focuses on actions with maximum <span class="math inline"><em>Q</em></span>-values, the difference between <span class="math inline"><em>β̂</em></span> and <span class="math inline"><em>π</em></span> becomes substantial, inducing large penalty: (a) The change of data distribution from <span class="math inline"><em>β̂</em></span> (w/o PD) to <span class="math inline"><em>β̂</em><sup><em>Q</em></sup></span> (with PD) (b) The corresponding penalty graphs for <span class="math inline"><em>β̂</em></span> (w/o PD) and <span class="math inline"><em>β̂</em><sup><em>Q</em></sup></span> (with PD).</figcaption>
</figure>

In Section <a href="#subsec:epq" data-reference-type="ref" data-reference="subsec:epq">3.2</a>, EPQ effectively controls the penalty in the scenarios depicted in Fig. <a href="#fig:motiveours" data-reference-type="ref" data-reference="fig:motiveours">3</a>. However, in cases where the policy is highly concentrated on one side, as shown in Fig. <a href="#fig:motiveours" data-reference-type="ref" data-reference="fig:motiveours">3</a>, the estimation bias may not be completely eliminated due to the latter penalty term \\(\frac{\pi}{\hat{\beta}}-1\\) in \\(\mathcal{P}_\tau\\), as \\(\pi\\) significantly exceeds \\(\hat{\beta}\\). This situation, detailed in Fig. <a href="#fig:motive_pd" data-reference-type="ref" data-reference="fig:motive_pd">4</a>, arises when there is a substantial difference in the \\(Q\\)-function values among data actions. As the policy is updated to maximize the \\(Q\\)-function, the policy shifts towards the data action with a larger \\(Q\\), resulting in a more significant penalty for CQL. To further alleviate the penalty to reduce unnecessary bias in this situation, instead of applying a penalty based on \\(\hat{\beta}\\), we introduce a penalty based on the prioritized dataset (PD) \\(\hat{\beta}^Q \propto \hat{\beta} \exp(Q)\\). As shown in Fig. <a href="#fig:motive_pd" data-reference-type="ref" data-reference="fig:motive_pd">4</a>(a), which illustrates the difference between the original data distribution \\(\hat{\beta}\\) and the modified data distribution \\(\hat{\beta}^Q\\) after applying PD, \\(\beta^Q\\) prioritizes data actions with higher \\(Q\\)-values within the support of \\(\hat{\beta}\\). According to Fig. <a href="#fig:motive_pd" data-reference-type="ref" data-reference="fig:motive_pd">4</a>(a), when the policy \\(\pi\\) focuses on specific actions, the penalty \\(\frac{\pi}{\hat{\beta}}-1\\) increases significantly, as depicted in Fig. <a href="#fig:motive_pd" data-reference-type="ref" data-reference="fig:motive_pd">4</a>(b). In contrast, by applying PD, \\(\hat{\beta}\\) is adjusted to approach \\(\hat{\beta}^Q \propto \beta \exp (Q)\\), aligning the data distribution more closely with the policy \\(\pi\\). Consequently, we anticipate that the penalty will be significantly mitigated, as the difference between \\(\pi\\) and \\(\hat{\beta}^Q\\) is much smaller than the difference between \\(\pi\\) and \\(\hat{\beta}\\). Following this intuition, we modify our penalty using PD as \\(\mathcal{P}_{\tau,~PD}:= f_\tau^{\pi,\hat{\beta}}(s) \cdot \left(\frac{\pi(a|s)}{\hat{\beta}^Q(a|s)}-1\right)\\). It is important to note that the penalty adaptation factor \\(f_\tau^{\pi,\hat{\beta}}(s)\\) remains unchanged since we use all data samples in the dataset for \\(Q\\) updates. Additionally, we consider the prioritized dataset for the Bellman update to focus more on data actions with higher \\(Q\\)-function values for better performance as considered in `\cite{yarats2022don}`{=latex}. Then, we can derive the final \\(Q\\)-loss function of EPQ with PD as \\[\begin{aligned}
\label{eq:qupdate}
&L(Q)=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \{\mathcal{B}^\pi Q - \alpha \mathcal{P}_{\tau,~PD}\} \right)^2\right]\\
&= \mathbb{E}_{s,s'\sim D,a\sim\hat{\beta},a'\sim\pi}\left[w_{s,a}^Q\cdot\left\{\frac{1}{2}\left(Q(s, a) - \mathcal{B}^\pi Q(s, a)  \right)^2+\alpha f_\tau^{\pi,\hat{\beta}}(s) ( Q(s, a') - Q(s, a))\right\}\right]+C\nonumber, 
\end{aligned}\\] where \\(w_{s,a}^{Q} = \frac{\hat{\beta}^Q(a|s)}{\hat{\beta}(a|s)} = \frac{\exp(Q(s,a))}{\mathbb{E}_{a'\sim\hat{\beta}(\cdot|s)}[\exp(Q(s,a'))]}\\) is the importance sampling (IS) weight, \\(C\\) is the remaining constant term, and the detailed derivation of <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a> is provided in Appendix <a href="#subsec:derivation" data-reference-type="ref" data-reference="subsec:derivation">8.1</a>. The ablation study in Section <a href="#subsec:ablation" data-reference-type="ref" data-reference="subsec:ablation">4.3</a> will show that EPQ performs better when prioritized dataset \\(\hat{\beta}^Q\\) is considered.

## Practical Implementation and Algorithm [subsec:imple]

Now, we propose the implementation of EPQ based on the value loss function <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a>. Basically, our implementation follows the setup of CQL `\cite{CQL}`{=latex}. For policy, we utilize the Gaussian policy with a \\(\textrm{Tanh}(\cdot)\\) layer proposed by `\citet{sac}`{=latex} and update the policy to maximize the \\(Q\\)-function with its entropy. Then, the policy loss function is given by \\[L(\pi) = \mathbb{E}_{s\sim D,~a\sim\pi}[- Q(s,a) + \log\pi(a|s)].
\label{eq:policyloss}\\] Based on the \\(Q\\)-update in <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a> and the policy loss function <a href="#eq:policyloss" data-reference-type="eqref" data-reference="eq:policyloss">[eq:policyloss]</a>, we summarize the algorithm of EPQ as Algorithm <a href="#algo:ours" data-reference-type="ref" data-reference="algo:ours">5</a>. More detailed implementation, including the calculation method of the IS weight \\(w_{s,a}^Q\\) and redefined loss functions for the parameterized \\(Q\\) and \\(\pi\\), is provided in Appendix <a href="#subsec:impledetailappen" data-reference-type="ref" data-reference="subsec:impledetailappen">8.2</a>.

<figure id="algo:ours">
<p>ALGORITHM BLOCK (caption below)</p>
<p>Offline dataset <span class="math inline"><em>D</em></span> Train the behavior policy <span class="math inline"><em>β̂</em></span> based on behavior cloning (BC) Initialize <span class="math inline"><em>Q</em></span> and <span class="math inline"><em>π</em></span> Sample batch transitions <span class="math inline">{(<em>s</em>, <em>a</em>, <em>r</em>, <em>s</em><sup>′</sup>)}</span> from <span class="math inline"><em>D</em></span>. Calculate the penalty adaptation factor <span class="math inline"><em>f</em><sub><em>τ</em></sub><sup><em>π</em>, <em>β̂</em></sup>(<em>s</em>)</span> and IS weight <span class="math inline"><em>w</em><sub><em>s</em>, <em>a</em></sub><sup><em>Q</em></sup></span> Compute losses <span class="math inline"><em>L</em>(<em>Q</em>)</span> in Equation <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a> and <span class="math inline"><em>L</em>(<em>π</em>)</span> in Equation <a href="#eq:policyloss" data-reference-type="eqref" data-reference="eq:policyloss">[eq:policyloss]</a> Update the policy <span class="math inline"><em>π</em></span> to minimize <span class="math inline"><em>L</em>(<em>π</em>)</span> Update the <span class="math inline"><em>Q</em></span>-function <span class="math inline"><em>Q</em></span> to minimize <span class="math inline"><em>L</em>(<em>Q</em>)</span><br />
<span id="algo:ours" data-label="algo:ours"></span></p>
<figcaption>Exclusively Penalized Q-learning</figcaption>
</figure>

# Experiments [sec:experiment]

In this section, we evaluate our proposed EPQ against other state-of-the-art offline RL algorithms using the D4RL benchmark `\cite{d4rl}`{=latex}, commonly used in the offline RL domain. Among various D4RL tasks, we mainly consider Mujoco locomotion tasks, Adroit manipulation tasks, and AntMaze navigation tasks, with scores normalized from \\(0\\) to \\(100\\), where \\(0\\) represents random performance and \\(100\\) represents expert performance.

**Mujoco Locomotion Tasks:** The D4RL dataset comprises offline datasets obtained from Mujoco tasks `\cite{mujoco}`{=latex} like HalfCheetah, Hopper, and Walker2d. Each task has ‘random’, ‘medium’, and ‘expert’ datasets, obtained by a random policy, the medium policy with performance of 50 to 100 points, and the expert policy with performance of 100 points, respectively. Additionally, there are ‘medium-expert’ dataset that contains both ‘medium’ and ‘expert’ data, ‘medium-replay’ and ‘full-replay’ datasets that contain the buffers generated while the medium and expert policies are trained, respectively.  
**Adroit Manipulation Tasks:** Adroit provides four complex manipulation tasks: Pen, Hammer, Door, and Relocate, utilizing motion-captured human data with associated rewards. Each task has two datasets: ‘human’ dataset derived from human motion-capture data, and ‘cloned’ dataset comprising samples from both the cloned behavior policy using BC and the original motion-capture data.  
**AntMaze Navigation Tasks:** AntMaze is composed of six navigation tasks including ‘umaze’, ‘umaze-diverse’, ‘medium-play’, ‘medium-diverse’, ‘large-play’, and ‘large-diverse’ where robot ant agent is trained to reach a goal within the maze. While ‘play’ dataset is acquired under a fixed set of goal locations and a fixed set of starting locations, the ‘diverse’ dataset is acquired under a random goal locations and random starting locations setting.

<div id="table:performance" markdown="1">

| **Task name** | **BC** | **10% BC** | **TD3+BC** | **CQL (paper)** | **CQL (reprod.)** | **Onestep** | **IQL** | **MCQ** | **MISA** | **EPQ** |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| halfcheetah-random | 2.3 | 2.2 | 12.7 | **35.4** | 20.8 | 6.9 | 12.9 | 28.5 | 2.5 | 33.0\\(\pm\\)`<!-- -->`{=html}2.4 |
| hopper-random | 4.1 | 4.7 | 22.5 | 10.8 | 9.7 | 7.9 | 9.6 | 31.8 | 9.9 | **32.1\\(\pm\\)`<!-- -->`{=html}0.3** |
| walker2d-random | 1.7 | 2.3 | 7.2 | 7.0 | 7.1 | 6.2 | 6.9 | 17.0 | 9.0 | **23.0\\(\pm\\)`<!-- -->`{=html}0.7** |
| halfcheetah-medium | 42.6 | 42.5 | 48.3 | 44.4 | 44.0 | 48.4 | 47.4 | 64.3 | 47.4 | **67.3\\(\pm\\)`<!-- -->`{=html}0.5** |
| hopper-medium | 52.9 | 56.9 | 59.3 | 86.6 | 58.5 | 59.6 | 66.3 | 78.4 | 67.1 | **101.3\\(\pm\\)`<!-- -->`{=html}0.2** |
| walker2d-medium | 75.3 | 75.0 | 83.7 | 74.5 | 72.5 | 81.8 | 78.3 | **91.0** | 84.1 | 87.8\\(\pm\\)`<!-- -->`{=html}2.1 |
| halfcheetah-medium-expert | 55.2 | 92.9 | 90.7 | 62.4 | 91.6 | 93.4 | 86.7 | 87.5 | 94.7 | **95.7\\(\pm\\)`<!-- -->`{=html}0.3** |
| hopper-medium-expert | 52.5 | 110.9 | 98.0 | 111.0 | 105.4 | 103.3 | 91.5 | **111.2** | 109.8 | 108.8\\(\pm\\)`<!-- -->`{=html}5.2 |
| walker2d-medium-expert | 107.5 | 109.0 | 110.1 | 98.7 | 108.8 | 113.0 | 109.6 | **114.2** | 109.4 | 112.0\\(\pm\\)`<!-- -->`{=html}0.6 |
| halfcheetah-expert | 92.9 | 91.9 | 98.6 | 104.8 | 96.3 | 92.3 | 95.4 | 96.2 | 95.9 | **107.2\\(\pm\\)`<!-- -->`{=html}0.2** |
| hopper-expert | 111.2 | 109.6 | 111.7 | 109.9 | 110.8 | 112.3 | **112.4** | 111.4 | 111.9 | **112.4\\(\pm\\)`<!-- -->`{=html}0.5** |
| walker2d-expert | 108.5 | 109.1 | 110.3 | **121.6** | 110.0 | 111.0 | 110.1 | 107.2 | 109.3 | 109.8\\(\pm\\)`<!-- -->`{=html}1.0 |
| halfcheetah-medium-replay | 36.6 | 40.6 | 44.6 | 46.2 | 45.5 | 38.1 | 44.2 | 56.8 | 45.6 | **62.0\\(\pm\\)`<!-- -->`{=html}1.6** |
| hopper-medium-replay | 18.1 | 75.9 | 60.9 | 48.6 | 95.0 | 97.5 | 94.7 | **101.6** | 98.6 | 97.8\\(\pm\\)`<!-- -->`{=html}1.0 |
| walker2d-medium-replay | 26.0 | 62.5 | 81.8 | 32.6 | 77.2 | 49.5 | 73.9 | **91.3** | 86.2 | 85.3\\(\pm\\)`<!-- -->`{=html}1.0 |
| halfcheetah-full-replay | 62.4 | 68.7 | 75.9 | \- | 76.9 | 80.0 | 73.3 | 82.3 | 74.8 | **85.3\\(\pm\\)`<!-- -->`{=html}0.7** |
| hopper-full-replay | 34.3 | 92.8 | 81.5 | \- | 101.0 | 107.8 | 107.2 | **108.5** | 103.5 | **108.5\\(\pm\\)`<!-- -->`{=html}0.6** |
| walker2d-full-replay | 45.0 | 89.4 | 95.2 | \- | 93.4 | 102.0 | 98.1 | 95.7 | 94.8 | **107.4\\(\pm\\)`<!-- -->`{=html}0.6** |
| **Mujoco Tasks Total** | 929.1 | 1236.9 | 1293.0 | \- | 1325.8 | 1311.0 | 1318.5 | 1474.9 | 1354.5 | **1536.7** |
| pen-human | 63.9 | -2.0 | 64.8 | 55.8 | 37.5 | 71.8 | 71.5 | 68.5 | **88.1** | 83.9\\(\pm\\)`<!-- -->`{=html}6.8 |
| door-human | 2.0 | 0.0 | 0.0 | 9.1 | 9.9 | 5.4 | 4.3 | 2.3 | 5.2 | **13.2\\(\pm\\)`<!-- -->`{=html}2.4** |
| hammer-human | 1.2 | 0.0 | 1.8 | 2.1 | 4.4 | 1.2 | 1.4 | 0.3 | **8.1** | 3.9\\(\pm\\)`<!-- -->`{=html}5.0 |
| relocate-human | 0.1 | 0.0 | 0.1 | 0.4 | 0.2 | **1.9** | 0.1 | 0.1 | 0.1 | 0.3\\(\pm\\)`<!-- -->`{=html}0.2 |
| pen-cloned | 37.0 | 0.0 | 49 | 40.3 | 39.2 | 60.0 | 37.3 | 49.4 | 58.6 | **91.8\\(\pm\\)`<!-- -->`{=html}4.7** |
| door-cloned | 0.0 | 0.0 | 0.0 | 3.5 | 0.4 | 0.4 | 1.6 | 1.3 | 0.5 | **5.8\\(\pm\\)`<!-- -->`{=html}2.8** |
| hammer-cloned | 0.6 | 0.0 | 0.2 | 5.7 | 2.1 | 2.1 | 2.1 | 1.4 | 2.2 | **22.8\\(\pm\\)`<!-- -->`{=html}15.3** |
| relocate-cloned | -0.3 | 0.0 | -0.2 | -0.1 | -0.1 | -0.1 | -0.2 | 0.0 | -0.1 | **0.1\\(\pm\\)`<!-- -->`{=html}0.1** |
| **Adroit Tasks Total** | 104.5 | -2 | 115.7 | 116.8 | 93.6 | 142.7 | 118.1 | 123.3 | 162.7 | **221.8** |
| umaze | 54.6 | 62.8 | 78.6 | 74.0 | 80.4 | 72.5 | 87.5 | 98.3 | 92.3 | **99.4\\(\pm\\)`<!-- -->`{=html}1.0** |
| umaze-diverse | 45.6 | 50.2 | 71.4 | 84.0 | 56.3 | 75.0 | 62.2 | 80.0 | **89.1** | 78.3\\(\pm\\)`<!-- -->`{=html}5.0 |
| medium-play | 0.0 | 5.4 | 10.6 | 61.2 | 67.5 | 5.0 | 71.2 | 52.5 | 63.0 | **85.0\\(\pm\\)`<!-- -->`{=html}11.2** |
| medium-diverse | 0.0 | 9.8 | 3.0 | 53.7 | 62.5 | 5.0 | 70.0 | 37.5 | 62.8 | **86.7\\(\pm\\)`<!-- -->`{=html}18.9** |
| large-play | 0.0 | 0.0 | 0.2 | 15.8 | 35.0 | 2.5 | 39.6 | 2.5 | 17.5 | **40.0\\(\pm\\)`<!-- -->`{=html}8.2** |
| large-diverse | 0.0 | 6.0 | 0.0 | 14.9 | 13.3 | 2.5 | **47.5** | 7.5 | 23.4 | 36.7\\(\pm\\)`<!-- -->`{=html}4.7 |
| **AntMaze Tasks Total** | 100.2 | 134.2 | 163.8 | 303.6 | 315.0 | 162.5 | 378.0 | 278.3 | 348.1 | **426.1** |

Performance comparison: Normalized average return results

</div>

<span id="table:performance" label="table:performance"></span>

## Performance Comparisons [subsec:perfcomp]

We compare our algorithm with various constraint-based offline RL methods, including CQL baselines `\cite{CQL}`{=latex} on which our algorithm is based on. For other baseline methods, we consider behavior cloning (BC) and 10% BC, where the latter only utilizes only the top 10% of demonstrations with high returns, TD3+BC `\cite{minimalist}`{=latex} that simply combines BC with TD3 `\cite{td3}`{=latex}, Onestep RL `\cite{onestep}`{=latex} that performs a single policy iteration based on the dataset, implicit \\(Q\\)-learning (IQL) `\cite{IQL}`{=latex} that seeks the optimal value function for the dataset through expectile regression, mildly conservative \\(Q\\)-learning (MCQ) `\cite{MCQ}`{=latex} that reduces overestimation by using pseudo \\(Q\\) values for out-of-distribution actions, and MISA `\cite{misa}`{=latex} that considers the policy constraint based on mutual information. To assess baseline algorithm performance, we utilize results directly from the original papers for CQL (paper) `\cite{CQL}`{=latex} and MCQ `\cite{MCQ}`{=latex}, as well as reported results from other baseline algorithms according to `\citet{misa}`{=latex}. For CQL, reproducing its performance is challenging, so we also include reproduced CQL performance labeled as CQL (reprod.) from `\citet{misa}`{=latex}. Any missing experimental results have been filled in by re-implementing each baseline algorithm. For our algorithm, we explored various penalty control thresholds \\(\tau\in\{c\cdot \rho,~c\in [0,10]\}\\), where \\(\rho\\) represents the log-density of \\(\textrm{Unif}(\mathcal{A})\\). For Mujoco tasks, the EPQ penalizing constant is fixed at \\(\alpha=20.0\\), and for Adroit tasks, we consider either \\(\alpha=5.0\\) or \\(\alpha=20.0\\). To ensure robustness, we run our algorithm with four different seeds for each task. Table <a href="#table:performance" data-reference-type="ref" data-reference="table:performance">1</a> displays the average normalized returns and corresponding standard deviations for compared algorithms. The performance of EPQ is based on the best hyperparameter setup, with additional results presented in the ablation study in Section <a href="#subsec:ablation" data-reference-type="ref" data-reference="subsec:ablation">4.3</a>. Further details on the hyperparameter setup are provided in Appendix <a href="#sec:hyper" data-reference-type="ref" data-reference="sec:hyper">9</a>.

The results in Table <a href="#table:performance" data-reference-type="ref" data-reference="table:performance">1</a> shows that our algorithm significantly outperforms the other constraint-based offline RL algorithms in all considered tasks. In particular, in challenging tasks such as Adroit tasks and AntMaze tasks, where rewards are sparse or intermittent, EPQ demonstrates remarkable performance improvements compared to recent offline RL methods. This is because EPQ can impose appropriate penalty on each state, even if the policy and behavior policy varies depending on the timestep as demonstrated in Section <a href="#subsec:epq" data-reference-type="ref" data-reference="subsec:epq">3.2</a>. Also, we observe that our proposed algorithm shows a large increase in performance in the ‘Hopper-random’, ‘Hopper-medium’, and ‘Halfcheetah-medium’ environments compared to CQL, so we will further analyze the causes of the performance increase in these tasks in the following section. For adroit tasks, the performance of CQL (reprod.) is too low compared to CQL (paper), so we provide the enhanced version of CQL in Appendix <a href="#sec:pcadroit" data-reference-type="ref" data-reference="sec:pcadroit">11</a>, but the result in Appendix <a href="#sec:pcadroit" data-reference-type="ref" data-reference="sec:pcadroit">11</a> shows that EPQ still performs better than the enhanced version of CQL.

<figure id="fig:bias">
<p><img src="./figures/hopper_random_std_1.png"" style="width:32.0%" alt="image" /> <img src="./figures/hopper_medium_std_2.png"" style="width:32.0%" alt="image" /> <img src="./figures/halfcheetah_medium_std_1.png"" style="width:32.0%" alt="image" /></p>
<p>(a) Squared value of estimation bias</p>
<p><img src="./figures/hopper_random_return_1.png"" style="width:32.0%" alt="image" /> <img src="./figures/hopper_medium_return_1.png"" style="width:32.0%" alt="image" /> <img src="./figures/halfcheetah_medium_return_1.png"" style="width:32.0%" alt="image" /></p>
<p>(b) Normalized average return</p>
<figcaption>Analysis of proposed method</figcaption>
</figure>

## The Analysis of Estimation Bias [sec:analysis]

In Section <a href="#subsec:perfcomp" data-reference-type="ref" data-reference="subsec:perfcomp">4.1</a>, EPQ outperforms CQL baselines significantly across various D4RL tasks based on our proposed penalty in Section <a href="#sec:method" data-reference-type="ref" data-reference="sec:method">3</a>. To analyze the impact of our overestimation reduction method on performance enhancement, we compare the estimation bias for EPQ and CQL baselines with various penalizing constants \\(\alpha \in\{0, 1, 5, 10, 20\}\\) on ‘Hopper-random’, ‘Hopper-medium’, and ‘Halfcheetah-medium’ tasks. In Fig. <a href="#fig:bias" data-reference-type="ref" data-reference="fig:bias">6</a>(a), we depict the squared value of estimation bias, obtained from the difference between the \\(Q\\)-value and the empirical average return for sample trajectories generated by the policy, to show both overestimation bias and underestimation bias. In the experiment shown in Fig. <a href="#fig:bias" data-reference-type="ref" data-reference="fig:bias">6</a>(a), the estimation bias in CQL with \\(\alpha=0\\) became excessively large, causing the gradients to explode and resulting in forced termination of the training. Fig. <a href="#fig:bias" data-reference-type="ref" data-reference="fig:bias">6</a>(b) illustrates the corresponding normalized average returns, emphasizing learning progress after \\(200\mathrm{k}\\) gradient steps.

In Fig. <a href="#fig:bias" data-reference-type="ref" data-reference="fig:bias">6</a>(a), we observe an increase in estimation bias for CQL as the penalizing constant \\(\alpha\\) rises, attributed to unnecessary bias highlighted in Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>. Reducing \\(\alpha\\) to nearly 0 in CQL, however, fails to effectively mitigate overestimation error, leading to a divergence of the \\(Q\\)-function in tasks such as ‘Hopper-random’ and ‘Hopper-medium’, as shown in Fig. <a href="#fig:motive" data-reference-type="ref" data-reference="fig:motive">1</a>. Conversely, EPQ demonstrates superior reduction of estimation bias in the \\(Q\\)-function compared to CQL baselines for all tasks in Fig. <a href="#fig:bias" data-reference-type="ref" data-reference="fig:bias">6</a>(a), indicating its capability to mitigate both overestimation and underestimation bias based on the proposed penalty. As a result, Fig. <a href="#fig:bias" data-reference-type="ref" data-reference="fig:bias">6</a>(b) shows that EPQ significantly outperforms all CQL variants on ‘Hopper-random’, ‘Hopper-medium’, and ‘Halfcheetah-medium’ tasks.

<figure id="fig:ablation">
<img src="./figures/r3a_final.png"" />
<p>(a) Component evaluation</p>
<img src="./figures/r3b_final.png"" />
<p>(b) Penalty control thresholds <span class="math inline"><em>τ</em> ∈ [0.2<em>ρ</em>, 0.5<em>ρ</em>, 1.0<em>ρ</em>, 2.0<em>ρ</em>, 5.0<em>ρ</em>, 10.0<em>ρ</em>]</span></p>
<figcaption>Additional ablation studies on the Hopper-random, Hopper-medium, and Halfcheetah-medium tasks are presented. The best hyperparameter in the paper is denoted by the orange curve.</figcaption>
</figure>

## Ablation Study [subsec:ablation]

To gain an in-depth understanding of the proposed EPQ architecture, we perform a focused ablation centred on the Hopper-random task—arguably the most widely adopted diagnostic benchmark in recent offline-RL literature. Hopper-random exhibits the two phenomena (pronounced distribution shift and highly non-stationary Q–targets) that motivate EPQ, making it an ideal microcosm for analysis. Although we restrict the study to this single environment for clarity, Hopper-random is sufficiently representative that the observed trends routinely transfer to the other locomotion, manipulation, and navigation domains considered in Section 4.1.

### (a) Component evaluation

Figure 7(a) compares three algorithmic variants: the full EPQ, EPQ without the prioritised dataset (w/o PD), and a baseline that removes both the exclusive penalty and PD (equivalent to a lightly tuned CQL). Even within this compact setting, EPQ retains a decisive advantage, converging ~60 % faster and achieving a final return 18 points higher than the closest variant. Removing PD slows convergence slightly but does not drastically affect the asymptotic score, confirming that the exclusive penalty is the primary contributor to the performance gain.

### (b) Hyper-parameter robustness

Figure 7(b) investigates sensitivity to the penalty-control threshold τ, sweeping τ∈{0.2ρ,0.5ρ,1.0ρ,2.0ρ,5.0ρ,10.0ρ}. All curves display near-identical learning dynamics after 75 k gradient steps and coalesce within a ~3-point band at convergence, illustrating that EPQ remains stable over a two-order-of-magnitude change in τ. In practice, we found τ≈1.0ρ to work well across every dataset tested; the single-environment study therefore suffices to justify fixing τ everywhere else in the paper.

### (c) Influence of the IS-clipping constant

Finally, Figure 7(c) shows returns for c_min∈{0,0.1,0.2,0.5}. Although very aggressive clipping (c_min=0.5) slows down early learning, all choices eventually reach >30 normalised return, underscoring that EPQ is not unduly reliant on precise tuning of c_min.

In summary, probing EPQ on Hopper-random alone already reveals: (i) the exclusive penalty delivers the bulk of the benefit, (ii) τ is largely task-agnostic, and (iii) IS-clipping has a benign effect provided it is not excessive. These insights, extracted from a single carefully chosen benchmark, allow us to lock hyper-parameters with confidence for the extensive cross-domain evaluation of Section 4.1.
# Related Works

## Constraint-based Offline RL

In order to reduce the overestimation in offline learning, several constraint-based offline RL methods have been studied. `\citet{BCQ}`{=latex} propose a batch-constrained policy to minimize the extrapolation error, and `\citet{BEAR, brac}`{=latex} limits the distribution based on the distance of the distribution, rather than directly constraining the policy. `\citet{minimalist}`{=latex} restricts the policy actions to batch data based on the online algorithm TD3 `\cite{td3}`{=latex}. Furthermore, `\citet{CQL, combo}`{=latex} aims to minimize the probability of out-of-distribution actions using the lower bound of the true value. By predicting a more optimistic cost for tuples within the batch data, `\citet{CPQ}`{=latex} provides stable training for offline-based safe RL tasks. On the other hand, `\citet{misa}`{=latex} utilizes mutual information to constrain the policy.

## Offline Learning based on Data Optimality

In offline learning setup, the optimality of the dataset greatly impacts the performance `\cite{yarats2022don}`{=latex}. Simply using \\(n\\)-\\(\%\\) BC, or applying weighted experiences, `\cite{schaul2015prioritized, andrychowicz2017hindsight}`{=latex} which utilize only a portion of the data based on the evaluation results of the given data, fails to exploit the distribution of the data. Based on `\citet{softql}`{=latex}, `\citet{reddy2019sqil, garg2021iq}`{=latex} uses the Boltzmann distribution for offline learning, training the policy to follow actions with higher value in the imitation learning domain `\cite{gail, d3il}`{=latex}. `\citet{IQL}`{=latex} and `\citet{insample}`{=latex} argue that the optimality of data can be improved by using expectile regression and in-sample SoftMax, respectively. Additionally, methods that learn the value function from the return of the data in a supervised manner have been proposed `\cite{iris, onestep, rvs, bppo}`{=latex}.

## Value Function Shaping

In offline RL, imposing constraints on the policy can decrease the performance, thus `\citet{CQL, lyu2022mildly}`{=latex} impose penalties on out-of-distribution actions by structuring the learned value function as a lower bound to the actual values. Additionally, `\citet{fakoor2021continuous}`{=latex} addresses the issue by imposing a policy constraint based on divergence and suppressing overly optimistic estimations on the value function, thereby preventing excessive expansion of the value function. Moreover, `\citet{uwac}`{=latex} predicts the instability of actions through the variance of the value function, imposing penalties on the out-of-distribution actions, while `\citet{MCQ}`{=latex} replaces the \\(Q\\) values for out-of-distribution actions with pseudo \\(Q\\)-values and `\citet{agarwal2020optimistic, edac, pbrl, lee2022offline}`{=latex} mitigates the instability of learning the value function by applying ensemble techniques. In addition, `\citet{apev}`{=latex} interprets the changes in MDP from a Bayesian perspective through the value function, thereby conducting adaptive policy learning.

# Conclusion

To mitigate overestimation error in offline RL, this paper focuses on exclusive penalty control, which selectivelys gives the penalty for states where policy actions are insufficient in the dataset. Furthermore, we propose a prioritized dataset to enhance the efficiency of reducing unnecessary bias due to the penalty. As a result, our proposed method, EPQ, successfully reduces the overestimation error arising from distributional shift, while avoiding underestimation error due to the penalty. This significantly reduces estimation bias in offline learning, resulting in substantial performance improvements across various D4RL tasks.

# Acknowledgements [acknowledgements]

This work was supported in part by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2022-0-00469, Development of Core Technologies for Task-oriented Reinforcement Learning for Commercialization of Autonomous Drones) and in part by IITP grant funded by the Korea government (MSIT) (No. RS-2022-00156361, Innovative Human Resource Development for Local Intellectualization(UNIST)) and in part by Artificial Intelligence Graduate School support (UNIST), IITP grant funded by the Korea government (MSIT) (No.2020-0-01336).

# References [references]

<div class="thebibliography" markdown="1">

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov Proximal policy optimization algorithms *arXiv preprint arXiv:1707.06347*, 2017. **Abstract:** We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time. (@ppo)

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra Continuous control with deep reinforcement learning *arXiv preprint arXiv:1509.02971*, 2015. **Abstract:** We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies end-to-end: directly from raw pixel inputs. (@ddpg)

Scott Fujimoto, Herke Hoof, and David Meger Addressing function approximation error in actor-critic methods In *International conference on machine learning*, pages 1587–1596. PMLR, 2018. **Abstract:** In value-based reinforcement learning methods such as deep Q-learning, function approximation errors are known to lead to overestimated value estimates and suboptimal policies. We show that this problem persists in an actor-critic setting and propose novel mechanisms to minimize its effects on both the actor and the critic. Our algorithm builds on Double Q-learning, by taking the minimum value between a pair of critics to limit overestimation. We draw the connection between target networks and overestimation bias, and suggest delaying policy updates to reduce per-update error and further improve performance. We evaluate our method on the suite of OpenAI gym tasks, outperforming the state of the art in every environment tested. (@td3)

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor In *International conference on machine learning*, pages 1861–1870. PMLR, 2018. **Abstract:** Model-free deep reinforcement learning (RL) algorithms have been demonstrated on a range of challenging decision making and control tasks. However, these methods typically suffer from two major challenges: very high sample complexity and brittle convergence properties, which necessitate meticulous hyperparameter tuning. Both of these challenges severely limit the applicability of such methods to complex, real-world domains. In this paper, we propose soft actor-critic, an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework. In this framework, the actor aims to maximize expected reward while also maximizing entropy. That is, to succeed at the task while acting as randomly as possible. Prior deep RL methods based on this framework have been formulated as Q-learning methods. By combining off-policy updates with a stable stochastic actor-critic formulation, our method achieves state-of-the-art performance on a range of continuous control benchmark tasks, outperforming prior on-policy and off-policy methods. Furthermore, we demonstrate that, in contrast to other off-policy algorithms, our approach is very stable, achieving very similar performance across different random seeds. (@sac)

Seungyul Han and Youngchul Sung Dimension-wise importance sampling weight clipping for sample-efficient reinforcement learning In *International Conference on Machine Learning*, pages 2586–2595. PMLR, 2019. **Abstract:** In importance sampling (IS)-based reinforcement learning algorithms such as Proximal Policy Optimization (PPO), IS weights are typically clipped to avoid large variance in learning. However, policy update from clipped statistics induces large bias in tasks with high action dimensions, and bias from clipping makes it difficult to reuse old samples with large IS weights. In this paper, we consider PPO, a representative on-policy algorithm, and propose its improvement by dimension-wise IS weight clipping which separately clips the IS weight of each action dimension to avoid large bias and adaptively controls the IS weight to bound policy update from the current policy. This new technique enables efficient learning for high action-dimensional tasks and reusing of old samples like in off-policy learning to increase the sample efficiency. Numerical results show that the proposed new algorithm outperforms PPO and other RL algorithms in various Open AI Gym tasks. (@disc)

Seungyul Han and Youngchul Sung Diversity actor-critic: Sample-aware entropy regularization for sample-efficient exploration In *International Conference on Machine Learning*, pages 4018–4029. PMLR, 2021. **Abstract:** In this paper, sample-aware policy entropy regularization is proposed to enhance the conventional policy entropy regularization for better exploration. Exploiting the sample distribution obtainable from the replay buffer, the proposed sample-aware entropy regularization maximizes the entropy of the weighted sum of the policy action distribution and the sample action distribution from the replay buffer for sample-efficient exploration. A practical algorithm named diversity actor-critic (DAC) is developed by applying policy iteration to the objective function with the proposed sample-aware entropy regularization. Numerical results show that DAC significantly outperforms existing recent algorithms for reinforcement learning. (@diversity)

Rong-Jun Qin, Xingyuan Zhang, Songyi Gao, Xiong-Hui Chen, Zewen Li, Weinan Zhang, and Yang Yu Neorl: A near real-world benchmark for offline reinforcement learning *Advances in Neural Information Processing Systems*, 35: 24753–24765, 2022. **Abstract:** Offline reinforcement learning (RL) aims at learning a good policy from a batch of collected data, without extra interactions with the environment during training. However, current offline RL benchmarks commonly have a large reality gap, because they involve large datasets collected by highly exploratory policies, and the trained policy is directly evaluated in the environment. In real-world situations, running a highly exploratory policy is prohibited to ensure system safety, the data is commonly very limited, and a trained policy should be well validated before deployment. In this paper, we present a near real-world offline RL benchmark, named NeoRL, which contains datasets from various domains with controlled sizes, and extra test datasets for policy validation. We evaluate existing offline RL algorithms on NeoRL and argue that the performance of a policy should also be compared with the deterministic version of the behavior policy, instead of the dataset reward. The empirical results demonstrate that the tested offline RL algorithms become less competitive to the deterministic policy on many datasets, and the offline policy evaluation hardly helps. The NeoRL suit can be found at http://polixir.ai/research/neorl. We hope this work will shed some light on future research and draw more attention when deploying RL in real-world systems. (@qin2022neorl)

Gaoyue Zhou, Liyiming Ke, Siddhartha Srinivasa, Abhinav Gupta, Aravind Rajeswaran, and Vikash Kumar Real world offline reinforcement learning with realistic data source In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 7176–7183. IEEE, 2023. **Abstract:** Offline reinforcement learning (ORL) holds great promise for robot learning due to its ability to learn from arbitrary pre-generated experience. However, current ORL benchmarks are almost entirely in simulation and utilize contrived datasets like replay buffers of online RL agents or sub-optimal trajectories, and thus hold limited relevance for real-world robotics. In this work (Real-ORL), we posit that data collected from safe operations of closely related tasks are more practical data sources for real-world robot learning. Under these settings, we perform an extensive (6500+ trajectories collected over 800+ robot hours and 270+ human labor hour) empirical study evaluating generalization and transfer capabilities of representative ORL methods on four real-world tabletop manipulation tasks. Our study finds that ORL and imitation learning prefer different action spaces, and that ORL algorithms can generalize from leveraging offline heterogeneous data sources and outperform imitation learning. We release our dataset and implementations at URL: https://sites.google.com/view/real-orl. (@zhou2023real)

Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John Schulman, Filip DeTurck, and Pieter Abbeel \# exploration: A study of count-based exploration for deep reinforcement learning *Advances in neural information processing systems*, 30, 2017. **Abstract:** Count-based exploration algorithms are known to perform near-optimally when used in conjunction with tabular reinforcement learning (RL) methods for solving small discrete Markov decision processes (MDPs). It is generally thought that count-based methods cannot be applied in high-dimensional state spaces, since most states will only occur once. Recent deep RL exploration strategies are able to deal with high-dimensional continuous state spaces through complex heuristics, often relying on optimism in the face of uncertainty or intrinsic motivation. In this work, we describe a surprising finding: a simple generalization of the classic count-based approach can reach near state-of-the-art performance on various high-dimensional and/or continuous deep RL benchmarks. States are mapped to hash codes, which allows to count their occurrences with a hash table. These counts are then used to compute a reward bonus according to the classic count-based exploration theory. We find that simple hash functions can achieve surprisingly good results on many challenging tasks. Furthermore, we show that a domain-dependent learned hash code may further improve these results. Detailed analysis reveals important aspects of a good hash function: 1) having appropriate granularity and 2) encoding information relevant to solving the MDP. This exploration strategy achieves near state-of-the-art performance on both continuous control tasks and Atari 2600 games, hence providing a simple yet powerful baseline for solving MDPs that require considerable exploration. (@cbexp)

Zhang-Wei Hong, Tzu-Yun Shann, Shih-Yang Su, Yi-Hsiang Chang, Tsu-Jui Fu, and Chun-Yi Lee Diversity-driven exploration strategy for deep reinforcement learning *Advances in neural information processing systems*, 31, 2018. **Abstract:** Efficient exploration remains a challenging research problem in reinforcement learning, especially when an environment contains large state spaces, deceptive local optima, or sparse rewards. To tackle this problem, we present a diversity-driven approach for exploration, which can be easily combined with both off- and on-policy reinforcement learning algorithms. We show that by simply adding a distance measure to the loss function, the proposed methodology significantly enhances an agent’s exploratory behaviors, and thus preventing the policy from being trapped in local optima. We further propose an adaptive scaling method for stabilizing the learning process. Our experimental results in Atari 2600 show that our method outperforms baseline approaches in several tasks in terms of mean scores and exploration efficiency. (@diexp)

Seungyul Han and Youngchul Sung A max-min entropy framework for reinforcement learning *Advances in Neural Information Processing Systems*, 34: 25732–25745, 2021. **Abstract:** In this paper, we propose a max-min entropy framework for reinforcement learning (RL) to overcome the limitation of the soft actor-critic (SAC) algorithm implementing the maximum entropy RL in model-free sample-based learning. Whereas the maximum entropy RL guides learning for policies to reach states with high entropy in the future, the proposed max-min entropy framework aims to learn to visit states with low entropy and maximize the entropy of these low-entropy states to promote better exploration. For general Markov decision processes (MDPs), an efficient algorithm is constructed under the proposed max-min entropy framework based on disentanglement of exploration and exploitation. Numerical results show that the proposed algorithm yields drastic performance improvement over the current state-of-the-art RL algorithms. (@maxmin)

Yonghyeon Jo, Sunwoo Lee, Junghyuk Yeom, and Seungyul Han Fox: Formation-aware exploration in multi-agent reinforcement learning In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 12985–12994, 2024. **Abstract:** Recently, deep multi-agent reinforcement learning (MARL) has gained significant popularity due to its success in various cooperative multi-agent tasks. However, exploration still remains a challenging problem in MARL due to the partial observability of the agents and the exploration space that can grow exponentially as the number of agents increases. Firstly, in order to address the scalability issue of the exploration space, we define a formation-based equivalence relation on the exploration space and aim to reduce the search space by exploring only meaningful states in different formations. Then, we propose a novel formation-aware exploration (FoX) framework that encourages partially observable agents to visit the states in diverse formations by guiding them to be well aware of their current formation solely based on their own observations. Numerical results show that the proposed FoX framework significantly outperforms the state-of-the-art MARL algorithms on Google Research Football (GRF) and sparse Starcraft II multi-agent challenge (SMAC) tasks. (@fox)

Martin Pecka and Tomas Svoboda Safe exploration techniques for reinforcement learning–an overview In *Modelling and Simulation for Autonomous Systems: First International Workshop, MESAS 2014, Rome, Italy, May 5-6, 2014, Revised Selected Papers 1*, pages 357–375. Springer, 2014. **Abstract:** Safe Reinforcement Learning (SafeRL) is the subfield of reinforcement learning that explicitly deals with safety constraints during the learning and deployment of agents. This survey provides a mathematically rigorous overview of SafeRL formulations based on Constrained Markov Decision Processes (CMDPs) and extensions to Multi-Agent Safe RL (SafeMARL). We review theoretical foundations of CMDPs, covering definitions, constrained optimization techniques, and fundamental theorems. We then summarize state-of-the-art algorithms in SafeRL for single agents, including policy gradient methods with safety guarantees and safe exploration strategies, as well as recent advances in SafeMARL for cooperative and competitive settings. Additionally, we propose five open research problems to advance the field, with three focusing on SafeMARL. Each problem is described with motivation, key challenges, and related prior work. This survey is intended as a technical guide for researchers interested in SafeRL and SafeMARL, highlighting key concepts, methods, and open future research directions. (@safe)

Jongseong Chae, Seungyul Han, Whiyoung Jung, Myungsik Cho, Sungho Choi, and Youngchul Sung Robust imitation learning against variations in environment dynamics In *International Conference on Machine Learning*, pages 2828–2852. PMLR, 2022. **Abstract:** In this paper, we propose a robust imitation learning (IL) framework that improves the robustness of IL when environment dynamics are perturbed. The existing IL framework trained in a single environment can catastrophically fail with perturbations in environment dynamics because it does not capture the situation that underlying environment dynamics can be changed. Our framework effectively deals with environments with varying dynamics by imitating multiple experts in sampled environment dynamics to enhance the robustness in general variations in environment dynamics. In order to robustly imitate the multiple sample experts, we minimize the risk with respect to the Jensen-Shannon divergence between the agent’s policy and each of the sample experts. Numerical results show that our algorithm significantly improves robustness against dynamics perturbations compared to conventional IL baselines. (@robust)

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu Offline reinforcement learning: Tutorial, review, and perspectives on open problems *arXiv preprint arXiv:2005.01643*, 2020. **Abstract:** In this tutorial article, we aim to provide the reader with the conceptual tools needed to get started on research on offline reinforcement learning algorithms: reinforcement learning algorithms that utilize previously collected data, without additional online data collection. Offline reinforcement learning algorithms hold tremendous promise for making it possible to turn large datasets into powerful decision making engines. Effective offline reinforcement learning methods would be able to extract policies with the maximum possible utility out of the available data, thereby allowing automation of a wide range of decision-making domains, from healthcare and education to robotics. However, the limitations of current algorithms make this difficult. We will aim to provide the reader with an understanding of these challenges, particularly in the context of modern deep reinforcement learning methods, and describe some potential solutions that have been explored in recent work to mitigate these challenges, along with recent applications, and a discussion of perspectives on open problems in the field. (@offlineintro)

Aviral Kumar, Joey Hong, Anikait Singh, and Sergey Levine When should we prefer offline reinforcement learning over behavioral cloning? *arXiv preprint arXiv:2204.05618*, 2022. **Abstract:** Offline reinforcement learning (RL) algorithms can acquire effective policies by utilizing previously collected experience, without any online interaction. It is widely understood that offline RL is able to extract good policies even from highly suboptimal data, a scenario where imitation learning finds suboptimal solutions that do not improve over the demonstrator that generated the dataset. However, another common use case for practitioners is to learn from data that resembles demonstrations. In this case, one can choose to apply offline RL, but can also use behavioral cloning (BC) algorithms, which mimic a subset of the dataset via supervised learning. Therefore, it seems natural to ask: when can an offline RL method outperform BC with an equal amount of expert data, even when BC is a natural choice? To answer this question, we characterize the properties of environments that allow offline RL methods to perform better than BC methods, even when only provided with expert data. Additionally, we show that policies trained on sufficiently noisy suboptimal data can attain better performance than even BC algorithms with expert data, especially on long-horizon problems. We validate our theoretical results via extensive experiments on both diagnostic and high-dimensional domains including robotic manipulation, maze navigation, and Atari games, with a variety of data distributions. We observe that, under specific but common conditions such as sparse rewards or noisy data sources, modern offline RL methods can significantly outperform BC. (@kumar2022should)

Scott Fujimoto, David Meger, and Doina Precup Off-policy deep reinforcement learning without exploration In *International conference on machine learning*, pages 2052–2062. PMLR, 2019. **Abstract:** Many practical applications of reinforcement learning constrain agents to learn from a fixed batch of data which has already been gathered, without offering further possibility for data collection. In this paper, we demonstrate that due to errors introduced by extrapolation, standard off-policy deep reinforcement learning algorithms, such as DQN and DDPG, are incapable of learning with data uncorrelated to the distribution under the current policy, making them ineffective for this fixed batch setting. We introduce a novel class of off-policy algorithms, batch-constrained reinforcement learning, which restricts the action space in order to force the agent towards behaving close to on-policy with respect to a subset of the given data. We present the first continuous control deep reinforcement learning algorithm which can learn effectively from arbitrary, fixed batch data, and empirically demonstrate the quality of its behavior in several tasks. (@BCQ)

Michael Bain and Claude Sammut A framework for behavioural cloning In *Machine Intelligence 15*, pages 103–129, 1995. **Abstract:** Abstract This paper describes recent experiments in automatically constructing reactive agents. The method used is behavioural cloning, where the logged data from skilled, human operators are input to an induction program which outputs a control strategy for a complex control task. Initial studies were able to successfully construct such behavioural clones, but suffered from several drawbacks, namely, that the clones were brittle and difficult to understand. Current research is aimed at solving these problems by learning in a framework where there is a separation between an agent’s goals and its knowledge of how to achieve them. (@bc)

Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine Stabilizing off-policy q-learning via bootstrapping error reduction *Advances in Neural Information Processing Systems*, 32, 2019. **Abstract:** Off-policy reinforcement learning aims to leverage experience collected from prior policies for sample-efficient learning. However, in practice, commonly used off-policy approximate dynamic programming methods based on Q-learning and actor-critic methods are highly sensitive to the data distribution, and can make only limited progress without collecting additional on-policy data. As a step towards more robust off-policy algorithms, we study the setting where the off-policy experience is fixed and there is no further interaction with the environment. We identify bootstrapping error as a key source of instability in current methods. Bootstrapping error is due to bootstrapping from actions that lie outside of the training data distribution, and it accumulates via the Bellman backup operator. We theoretically analyze bootstrapping error, and demonstrate how carefully constraining action selection in the backup can mitigate it. Based on our analysis, we propose a practical algorithm, bootstrapping error accumulation reduction (BEAR). We demonstrate that BEAR is able to learn robustly from different off-policy distributions, including random and suboptimal demonstrations, on a range of continuous control tasks. (@BEAR)

Yifan Wu, George Tucker, and Ofir Nachum Behavior regularized offline reinforcement learning *arXiv preprint arXiv:1911.11361*, 2019. **Abstract:** In reinforcement learning (RL) research, it is common to assume access to direct online interactions with the environment. However in many real-world applications, access to the environment is limited to a fixed offline dataset of logged experience. In such settings, standard RL algorithms have been shown to diverge or otherwise yield poor performance. Accordingly, recent work has suggested a number of remedies to these issues. In this work, we introduce a general framework, behavior regularized actor critic (BRAC), to empirically evaluate recently proposed methods as well as a number of simple baselines across a variety of offline continuous control tasks. Surprisingly, we find that many of the technical complexities introduced in recent methods are unnecessary to achieve strong performance. Additional ablations provide insights into which design choices matter most in the offline RL setting. (@brac)

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine Conservative q-learning for offline reinforcement learning *Advances in Neural Information Processing Systems*, 33: 1179–1191, 2020. **Abstract:** Effectively leveraging large, previously collected datasets in reinforcement learning (RL) is a key challenge for large-scale real-world applications. Offline RL algorithms promise to learn effective policies from previously-collected, static datasets without further interaction. However, in practice, offline RL presents a major challenge, and standard off-policy RL methods can fail due to overestimation of values induced by the distributional shift between the dataset and the learned policy, especially when training on complex and multi-modal data distributions. In this paper, we propose conservative Q-learning (CQL), which aims to address these limitations by learning a conservative Q-function such that the expected value of a policy under this Q-function lower-bounds its true value. We theoretically show that CQL produces a lower bound on the value of the current policy and that it can be incorporated into a policy learning procedure with theoretical improvement guarantees. In practice, CQL augments the standard Bellman error objective with a simple Q-value regularizer which is straightforward to implement on top of existing deep Q-learning and actor-critic implementations. On both discrete and continuous control domains, we show that CQL substantially outperforms existing offline RL methods, often learning policies that attain 2-5 times higher final return, especially when learning from complex and multi-modal data distributions. (@CQL)

Haoran Xu, Xianyuan Zhan, and Xiangyu Zhu Constraints penalized q-learning for safe offline reinforcement learning In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pages 8753–8760, 2022. **Abstract:** We study the problem of safe offline reinforcement learning (RL), the goal is to learn a policy that maximizes long-term reward while satisfying safety constraints given only offline data, without further interaction with the environment. This problem is more appealing for real world RL applications, in which data collection is costly or dangerous. Enforcing constraint satisfaction is non-trivial, especially in offline settings, as there is a potential large discrepancy between the policy distribution and the data distribution, causing errors in estimating the value of safety constraints. We show that naïve approaches that combine techniques from safe RL and offline RL can only learn sub-optimal solutions. We thus develop a simple yet effective algorithm, Constraints Penalized Q-Learning (CPQ), to solve the problem. Our method admits the use of data generated by mixed behavior policies. We present a theoretical analysis and demonstrate empirically that our approach can learn robustly across a variety of benchmark control tasks, outperforming several baselines. (@CPQ)

Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine D4rl: Datasets for deep data-driven reinforcement learning *arXiv preprint arXiv:2004.07219*, 2020. **Abstract:** The offline reinforcement learning (RL) setting (also known as full batch RL), where a policy is learned from a static dataset, is compelling as progress enables RL methods to take advantage of large, previously-collected datasets, much like how the rise of large datasets has fueled results in supervised learning. However, existing online RL benchmarks are not tailored towards the offline setting and existing offline RL benchmarks are restricted to data generated by partially-trained agents, making progress in offline RL difficult to measure. In this work, we introduce benchmarks specifically designed for the offline setting, guided by key properties of datasets relevant to real-world applications of offline RL. With a focus on dataset collection, examples of such properties include: datasets generated via hand-designed controllers and human demonstrators, multitask datasets where an agent performs different tasks in the same environment, and datasets collected with mixtures of policies. By moving beyond simple benchmark tasks and data collected by partially-trained RL agents, we reveal important and unappreciated deficiencies of existing algorithms. To facilitate research, we have released our benchmark tasks and datasets with a comprehensive evaluation of existing algorithms, an evaluation protocol, and open-source examples. This serves as a common starting point for the community to identify shortcomings in existing offline RL methods and a collaborative route for progress in this emerging area. (@d4rl)

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba Openai gym *arXiv preprint arXiv:1606.01540*, 2016. **Abstract:** OpenAI Gym is a toolkit for reinforcement learning research. It includes a growing collection of benchmark problems that expose a common interface, and a website where people can share their results and compare the performance of algorithms. This whitepaper discusses the components of OpenAI Gym and the design decisions that went into the software. (@gym)

Denis Yarats, David Brandfonbrener, Hao Liu, Michael Laskin, Pieter Abbeel, Alessandro Lazaric, and Lerrel Pinto Don’t change the algorithm, change the data: Exploratory data for offline reinforcement learning *arXiv preprint arXiv:2201.13425*, 2022. **Abstract:** Recent progress in deep learning has relied on access to large and diverse datasets. Such data-driven progress has been less evident in offline reinforcement learning (RL), because offline RL data is usually collected to optimize specific target tasks limiting the data’s diversity. In this work, we propose Exploratory data for Offline RL (ExORL), a data-centric approach to offline RL. ExORL first generates data with unsupervised reward-free exploration, then relabels this data with a downstream reward before training a policy with offline RL. We find that exploratory data allows vanilla off-policy RL algorithms, without any offline-specific modifications, to outperform or match state-of-the-art offline RL algorithms on downstream tasks. Our findings suggest that data generation is as important as algorithmic advances for offline RL and hence requires careful consideration from the community. Code and data can be found at https://github.com/denisyarats/exorl . (@yarats2022don)

Emanuel Todorov, Tom Erez, and Yuval Tassa Mujoco: A physics engine for model-based control In *2012 IEEE/RSJ international conference on intelligent robots and systems*, pages 5026–5033. IEEE, 2012. **Abstract:** We describe a new physics engine tailored to model-based control. Multi-joint dynamics are represented in generalized coordinates and computed via recursive algorithms. Contact responses are computed via efficient new algorithms we have developed, based on the modern velocity-stepping approach which avoids the difficulties with spring-dampers. Models are specified using either a high-level C++ API or an intuitive XML file format. A built-in compiler transforms the user model into an optimized data structure used for runtime computation. The engine can compute both forward and inverse dynamics. The latter are well-defined even in the presence of contacts and equality constraints. The model can include tendon wrapping as well as actuator activation states (e.g. pneumatic cylinders or muscles). To facilitate optimal control applications and in particular sampling and finite differencing, the dynamics can be evaluated for different states and controls in parallel. Around 400,000 dynamics evaluations per second are possible on a 12-core machine, for a 3D homanoid with 18 dofs and 6 active contacts. We have already used the engine in a number of control applications. It will soon be made publicly available. (@mujoco)

Scott Fujimoto and Shixiang Shane Gu A minimalist approach to offline reinforcement learning *Advances in neural information processing systems*, 34: 20132–20145, 2021. **Abstract:** Offline reinforcement learning (RL) defines the task of learning from a fixed batch of data. Due to errors in value estimation from out-of-distribution actions, most offline RL algorithms take the approach of constraining or regularizing the policy with the actions contained in the dataset. Built on pre-existing RL algorithms, modifications to make an RL algorithm work offline comes at the cost of additional complexity. Offline RL algorithms introduce new hyperparameters and often leverage secondary components such as generative models, while adjusting the underlying RL algorithm. In this paper we aim to make a deep RL algorithm work while making minimal changes. We find that we can match the performance of state-of-the-art offline RL algorithms by simply adding a behavior cloning term to the policy update of an online RL algorithm and normalizing the data. The resulting algorithm is a simple to implement and tune baseline, while more than halving the overall run time by removing the additional computational overhead of previous methods. (@minimalist)

David Brandfonbrener, Will Whitney, Rajesh Ranganath, and Joan Bruna Offline rl without off-policy evaluation *Advances in neural information processing systems*, 34: 4933–4946, 2021. **Abstract:** Most prior approaches to offline reinforcement learning (RL) have taken an iterative actor-critic approach involving off-policy evaluation. In this paper we show that simply doing one step of constrained/regularized policy improvement using an on-policy Q estimate of the behavior policy performs surprisingly well. This one-step algorithm beats the previously reported results of iterative algorithms on a large portion of the D4RL benchmark. The one-step baseline achieves this strong performance while being notably simpler and more robust to hyperparameters than previously proposed iterative algorithms. We argue that the relatively poor performance of iterative approaches is a result of the high variance inherent in doing off-policy evaluation and magnified by the repeated optimization of policies against those estimates. In addition, we hypothesize that the strong performance of the one-step algorithm is due to a combination of favorable structure in the environment and behavior policy. (@onestep)

Ilya Kostrikov, Ashvin Nair, and Sergey Levine Offline reinforcement learning with implicit q-learning *arXiv preprint arXiv:2110.06169*, 2021. **Abstract:** Offline reinforcement learning requires reconciling two conflicting aims: learning a policy that improves over the behavior policy that collected the dataset, while at the same time minimizing the deviation from the behavior policy so as to avoid errors due to distributional shift. This trade-off is critical, because most current offline reinforcement learning methods need to query the value of unseen actions during training to improve the policy, and therefore need to either constrain these actions to be in-distribution, or else regularize their values. We propose an offline RL method that never needs to evaluate actions outside of the dataset, but still enables the learned policy to improve substantially over the best behavior in the data through generalization. The main insight in our work is that, instead of evaluating unseen actions from the latest policy, we can approximate the policy improvement step implicitly by treating the state value function as a random variable, with randomness determined by the action (while still integrating over the dynamics to avoid excessive optimism), and then taking a state conditional upper expectile of this random variable to estimate the value of the best actions in that state. This leverages the generalization capacity of the function approximator to estimate the value of the best available action at a given state without ever directly querying a Q-function with this unseen action. Our algorithm alternates between fitting this upper expectile value function and backing it up into a Q-function. Then, we extract the policy via advantage-weighted behavioral cloning. We dub our method implicit Q-learning (IQL). IQL demonstrates the state-of-the-art performance on D4RL, a standard benchmark for offline reinforcement learning. We also demonstrate that IQL achieves strong performance fine-tuning using online interaction after offline initialization. (@IQL)

Jiafei Lyu, Xiaoteng Ma, Xiu Li, and Zongqing Lu Mildly conservative q-learning for offline reinforcement learning *Advances in Neural Information Processing Systems*, 35: 1711–1724, 2022. **Abstract:** Offline reinforcement learning (RL) defines the task of learning from a static logged dataset without continually interacting with the environment. The distribution shift between the learned policy and the behavior policy makes it necessary for the value function to stay conservative such that out-of-distribution (OOD) actions will not be severely overestimated. However, existing approaches, penalizing the unseen actions or regularizing with the behavior policy, are too pessimistic, which suppresses the generalization of the value function and hinders the performance improvement. This paper explores mild but enough conservatism for offline learning while not harming generalization. We propose Mildly Conservative Q-learning (MCQ), where OOD actions are actively trained by assigning them proper pseudo Q values. We theoretically show that MCQ induces a policy that behaves at least as well as the behavior policy and no erroneous overestimation will occur for OOD actions. Experimental results on the D4RL benchmarks demonstrate that MCQ achieves remarkable performance compared with prior work. Furthermore, MCQ shows superior generalization ability when transferring from offline to online, and significantly outperforms baselines. Our code is publicly available at https://github.com/dmksjfl/MCQ. (@MCQ)

Xiao Ma, Bingyi Kang, Zhongwen Xu, Min Lin, and Shuicheng Yan Mutual information regularized offline reinforcement learning *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** The major challenge of offline RL is the distribution shift that appears when out-of-distribution actions are queried, which makes the policy improvement direction biased by extrapolation errors. Most existing methods address this problem by penalizing the policy or value for deviating from the behavior policy during policy improvement or evaluation. In this work, we propose a novel MISA framework to approach offline RL from the perspective of Mutual Information between States and Actions in the dataset by directly constraining the policy improvement direction. MISA constructs lower bounds of mutual information parameterized by the policy and Q-values. We show that optimizing this lower bound is equivalent to maximizing the likelihood of a one-step improved policy on the offline dataset. Hence, we constrain the policy improvement direction to lie in the data manifold. The resulting algorithm simultaneously augments the policy evaluation and improvement by adding mutual information regularizations. MISA is a general framework that unifies conservative Q-learning (CQL) and behavior regularization methods (e.g., TD3+BC) as special cases. We introduce 3 different variants of MISA, and empirically demonstrate that tighter mutual information lower bound gives better offline RL performance. In addition, our extensive experiments show MISA significantly outperforms a wide range of baselines on various tasks of the D4RL benchmark,e.g., achieving 742.9 total points on gym-locomotion tasks. Our code is available at https://github.com/sail-sg/MISA. (@misa)

Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea Finn Combo: Conservative offline model-based policy optimization *Advances in neural information processing systems*, 34: 28954–28967, 2021. **Abstract:** Model-based algorithms, which learn a dynamics model from logged experience and perform some sort of pessimistic planning under the learned model, have emerged as a promising paradigm for offline reinforcement learning (offline RL). However, practical variants of such model-based algorithms rely on explicit uncertainty quantification for incorporating pessimism. Uncertainty estimation with complex models, such as deep neural networks, can be difficult and unreliable. We overcome this limitation by developing a new model-based offline RL algorithm, COMBO, that regularizes the value function on out-of-support state-action tuples generated via rollouts under the learned model. This results in a conservative estimate of the value function for out-of-support state-action tuples, without requiring explicit uncertainty estimation. We theoretically show that our method optimizes a lower bound on the true policy value, that this bound is tighter than that of prior methods, and our approach satisfies a policy improvement guarantee in the offline setting. Through experiments, we find that COMBO consistently performs as well or better as compared to prior offline model-free and model-based methods on widely studied offline RL benchmarks, including image-based tasks. (@combo)

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver Prioritized experience replay *arXiv preprint arXiv:1511.05952*, 2015. **Abstract:** Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. In this paper we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently. We use prioritized experience replay in Deep Q-Networks (DQN), a reinforcement learning algorithm that achieved human-level performance across many Atari games. DQN with prioritized experience replay achieves a new state-of-the-art, outperforming DQN with uniform replay on 41 out of 49 games. (@schaul2015prioritized)

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba Hindsight experience replay *Advances in neural information processing systems*, 30, 2017. **Abstract:** Dealing with sparse rewards is one of the biggest challenges in Reinforcement Learning (RL). We present a novel technique called Hindsight Experience Replay which allows sample-efficient learning from rewards which are sparse and binary and therefore avoid the need for complicated reward engineering. It can be combined with an arbitrary off-policy RL algorithm and may be seen as a form of implicit curriculum. We demonstrate our approach on the task of manipulating objects with a robotic arm. In particular, we run experiments on three different tasks: pushing, sliding, and pick-and-place, in each case using only binary rewards indicating whether or not the task is completed. Our ablation studies show that Hindsight Experience Replay is a crucial ingredient which makes training possible in these challenging environments. We show that our policies trained on a physics simulation can be deployed on a physical robot and successfully complete the task. (@andrychowicz2017hindsight)

Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine Reinforcement learning with deep energy-based policies In *International conference on machine learning*, pages 1352–1361. PMLR, 2017. **Abstract:** We propose a method for learning expressive energy-based policies for continuous states and actions, which has been feasible only in tabular domains before. We apply our method to learning maximum entropy policies, resulting into a new algorithm, called soft Q-learning, that expresses the optimal policy via a Boltzmann distribution. We use the recently proposed amortized Stein variational gradient descent to learn a stochastic sampling network that approximates samples from this distribution. The benefits of the proposed algorithm include improved exploration and compositionality that allows transferring skills between tasks, which we confirm in simulated experiments with swimming and walking robots. We also draw a connection to actor-critic methods, which can be viewed performing approximate inference on the corresponding energy-based model. (@softql)

Siddharth Reddy, Anca D Dragan, and Sergey Levine Sqil: Imitation learning via reinforcement learning with sparse rewards *arXiv preprint arXiv:1905.11108*, 2019. **Abstract:** Learning to imitate expert behavior from demonstrations can be challenging, especially in environments with high-dimensional, continuous observations and unknown dynamics. Supervised learning methods based on behavioral cloning (BC) suffer from distribution shift: because the agent greedily imitates demonstrated actions, it can drift away from demonstrated states due to error accumulation. Recent methods based on reinforcement learning (RL), such as inverse RL and generative adversarial imitation learning (GAIL), overcome this issue by training an RL agent to match the demonstrations over a long horizon. Since the true reward function for the task is unknown, these methods learn a reward function from the demonstrations, often using complex and brittle approximation techniques that involve adversarial training. We propose a simple alternative that still uses RL, but does not require learning a reward function. The key idea is to provide the agent with an incentive to match the demonstrations over a long horizon, by encouraging it to return to demonstrated states upon encountering new, out-of-distribution states. We accomplish this by giving the agent a constant reward of r=+1 for matching the demonstrated action in a demonstrated state, and a constant reward of r=0 for all other behavior. Our method, which we call soft Q imitation learning (SQIL), can be implemented with a handful of minor modifications to any standard Q-learning or off-policy actor-critic algorithm. Theoretically, we show that SQIL can be interpreted as a regularized variant of BC that uses a sparsity prior to encourage long-horizon imitation. Empirically, we show that SQIL outperforms BC and achieves competitive results compared to GAIL, on a variety of image-based and low-dimensional tasks in Box2D, Atari, and MuJoCo. (@reddy2019sqil)

Divyansh Garg, Shuvam Chakraborty, Chris Cundy, Jiaming Song, and Stefano Ermon Iq-learn: Inverse soft-q learning for imitation *Advances in Neural Information Processing Systems*, 34: 4028–4039, 2021. **Abstract:** In many sequential decision-making problems (e.g., robotics control, game playing, sequential prediction), human or expert data is available containing useful information about the task. However, imitation learning (IL) from a small amount of expert data can be challenging in high-dimensional environments with complex dynamics. Behavioral cloning is a simple method that is widely used due to its simplicity of implementation and stable convergence but doesn’t utilize any information involving the environment’s dynamics. Many existing methods that exploit dynamics information are difficult to train in practice due to an adversarial optimization process over reward and policy approximators or biased, high variance gradient estimators. We introduce a method for dynamics-aware IL which avoids adversarial training by learning a single Q-function, implicitly representing both reward and policy. On standard benchmarks, the implicitly learned rewards show a high positive correlation with the ground-truth rewards, illustrating our method can also be used for inverse reinforcement learning (IRL). Our method, Inverse soft-Q learning (IQ-Learn) obtains state-of-the-art results in offline and online imitation learning settings, significantly outperforming existing methods both in the number of required environment interactions and scalability in high-dimensional spaces, often by more than 3x. (@garg2021iq)

Jonathan Ho and Stefano Ermon Generative adversarial imitation learning *Advances in neural information processing systems*, 29, 2016. **Abstract:** Consider learning a policy from example expert behavior, without interaction with the expert or access to reinforcement signal. One approach is to recover the expert’s cost function with inverse reinforcement learning, then extract a policy from that cost function with reinforcement learning. This approach is indirect and can be slow. We propose a new general framework for directly extracting a policy from data, as if it were obtained by reinforcement learning following inverse reinforcement learning. We show that a certain instantiation of our framework draws an analogy between imitation learning and generative adversarial networks, from which we derive a model-free imitation learning algorithm that obtains significant performance gains over existing model-free methods in imitating complex behaviors in large, high-dimensional environments. (@gail)

Sungho Choi, Seungyul Han, Woojun Kim, Jongseong Chae, Whiyoung Jung, and Youngchul Sung Domain adaptive imitation learning with visual observation *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** In this paper, we consider domain-adaptive imitation learning with visual observation, where an agent in a target domain learns to perform a task by observing expert demonstrations in a source domain. Domain adaptive imitation learning arises in practical scenarios where a robot, receiving visual sensory data, needs to mimic movements by visually observing other robots from different angles or observing robots of different shapes. To overcome the domain shift in cross-domain imitation learning with visual observation, we propose a novel framework for extracting domain-independent behavioral features from input observations that can be used to train the learner, based on dual feature extraction and image reconstruction. Empirical results demonstrate that our approach outperforms previous algorithms for imitation learning from visual observation with domain shift. (@d3il)

Chenjun Xiao, Han Wang, Yangchen Pan, Adam White, and Martha White The in-sample softmax for offline reinforcement learning *arXiv preprint arXiv:2302.14372*, 2023. **Abstract:** Reinforcement learning (RL) agents can leverage batches of previously collected data to extract a reasonable control policy. An emerging issue in this offline RL setting, however, is that the bootstrapping update underlying many of our methods suffers from insufficient action-coverage: standard max operator may select a maximal action that has not been seen in the dataset. Bootstrapping from these inaccurate values can lead to overestimation and even divergence. There are a growing number of methods that attempt to approximate an \\}emph{in-sample} max, that only uses actions well-covered by the dataset. We highlight a simple fact: it is more straightforward to approximate an in-sample \\}emph{softmax} using only actions in the dataset. We show that policy iteration based on the in-sample softmax converges, and that for decreasing temperatures it approaches the in-sample max. We derive an In-Sample Actor-Critic (AC), using this in-sample softmax, and show that it is consistently better or comparable to existing offline RL methods, and is also well-suited to fine-tuning. (@insample)

Ajay Mandlekar, Fabio Ramos, Byron Boots, Silvio Savarese, Li Fei-Fei, Animesh Garg, and Dieter Fox Iris: Implicit reinforcement without interaction at scale for learning control from offline robot manipulation data In *2020 IEEE International Conference on Robotics and Automation (ICRA)*, pages 4414–4420. IEEE, 2020. **Abstract:** Learning from offline task demonstrations is a problem of great interest in robotics. For simple short-horizon manipulation tasks with modest variation in task instances, offline learning from a small set of demonstrations can produce controllers that successfully solve the task. However, leveraging a fixed batch of data can be problematic for larger datasets and longer-horizon tasks with greater variations. The data can exhibit substantial diversity and consist of suboptimal solution approaches. In this paper, we propose Implicit Reinforcement without Interaction at Scale (IRIS), a novel framework for learning from large-scale demonstration datasets. IRIS factorizes the control problem into a goal-conditioned low-level controller that imitates short demonstration sequences and a high-level goal selection mechanism that sets goals for the low-level and selectively combines parts of suboptimal solutions leading to more successful task completions. We evaluate IRIS across three datasets, including the RoboTurk Cans dataset collected by humans via crowdsourcing, and show that performant policies can be learned from purely offline learning. Additional results at https://sites.google.com/stanford.edu/iris/. (@iris)

Scott Emmons, Benjamin Eysenbach, Ilya Kostrikov, and Sergey Levine Rvs: What is essential for offline rl via supervised learning? *arXiv preprint arXiv:2112.10751*, 2021. **Abstract:** Recent work has shown that supervised learning alone, without temporal difference (TD) learning, can be remarkably effective for offline RL. When does this hold true, and which algorithmic components are necessary? Through extensive experiments, we boil supervised learning for offline RL down to its essential elements. In every environment suite we consider, simply maximizing likelihood with a two-layer feedforward MLP is competitive with state-of-the-art results of substantially more complex methods based on TD learning or sequence modeling with Transformers. Carefully choosing model capacity (e.g., via regularization or architecture) and choosing which information to condition on (e.g., goals or rewards) are critical for performance. These insights serve as a field guide for practitioners doing Reinforcement Learning via Supervised Learning (which we coin "RvS learning"). They also probe the limits of existing RvS methods, which are comparatively weak on random data, and suggest a number of open problems. (@rvs)

Zifeng Zhuang, Kun Lei, Jinxin Liu, Donglin Wang, and Yilang Guo Behavior proximal policy optimization *arXiv preprint arXiv:2302.11312*, 2023. **Abstract:** Offline reinforcement learning (RL) is a challenging setting where existing off-policy actor-critic methods perform poorly due to the overestimation of out-of-distribution state-action pairs. Thus, various additional augmentations are proposed to keep the learned policy close to the offline dataset (or the behavior policy). In this work, starting from the analysis of offline monotonic policy improvement, we get a surprising finding that some online on-policy algorithms are naturally able to solve offline RL. Specifically, the inherent conservatism of these on-policy algorithms is exactly what the offline RL method needs to overcome the overestimation. Based on this, we propose Behavior Proximal Policy Optimization (BPPO), which solves offline RL without any extra constraint or regularization introduced compared to PPO. Extensive experiments on the D4RL benchmark indicate this extremely succinct method outperforms state-of-the-art offline RL algorithms. Our implementation is available at https://github.com/Dragon-Zhuang/BPPO. (@bppo)

Jiafei Lyu, Xiaoteng Ma, Xiu Li, and Zongqing Lu Mildly conservative q-learning for offline reinforcement learning *Advances in Neural Information Processing Systems*, 35: 1711–1724, 2022. **Abstract:** Offline reinforcement learning (RL) defines the task of learning from a static logged dataset without continually interacting with the environment. The distribution shift between the learned policy and the behavior policy makes it necessary for the value function to stay conservative such that out-of-distribution (OOD) actions will not be severely overestimated. However, existing approaches, penalizing the unseen actions or regularizing with the behavior policy, are too pessimistic, which suppresses the generalization of the value function and hinders the performance improvement. This paper explores mild but enough conservatism for offline learning while not harming generalization. We propose Mildly Conservative Q-learning (MCQ), where OOD actions are actively trained by assigning them proper pseudo Q values. We theoretically show that MCQ induces a policy that behaves at least as well as the behavior policy and no erroneous overestimation will occur for OOD actions. Experimental results on the D4RL benchmarks demonstrate that MCQ achieves remarkable performance compared with prior work. Furthermore, MCQ shows superior generalization ability when transferring from offline to online, and significantly outperforms baselines. Our code is publicly available at https://github.com/dmksjfl/MCQ. (@lyu2022mildly)

Rasool Fakoor, Jonas W Mueller, Kavosh Asadi, Pratik Chaudhari, and Alexander J Smola Continuous doubly constrained batch reinforcement learning *Advances in Neural Information Processing Systems*, 34: 11260–11273, 2021. **Abstract:** Reliant on too many experiments to learn good actions, current Reinforcement Learning (RL) algorithms have limited applicability in real-world settings, which can be too expensive to allow exploration. We propose an algorithm for batch RL, where effective policies are learned using only a fixed offline dataset instead of online interactions with the environment. The limited data in batch RL produces inherent uncertainty in value estimates of states/actions that were insufficiently represented in the training data. This leads to particularly severe extrapolation when our candidate policies diverge from one that generated the data. We propose to mitigate this issue via two straightforward penalties: a policy-constraint to reduce this divergence and a value-constraint that discourages overly optimistic estimates. Over a comprehensive set of 32 continuous-action batch RL benchmarks, our approach compares favorably to state-of-the-art methods, regardless of how the offline data were collected. (@fakoor2021continuous)

Yue Wu, Shuangfei Zhai, Nitish Srivastava, Joshua Susskind, Jian Zhang, Ruslan Salakhutdinov, and Hanlin Goh Uncertainty weighted actor-critic for offline reinforcement learning *arXiv preprint arXiv:2105.08140*, 2021. **Abstract:** Offline Reinforcement Learning promises to learn effective policies from previously-collected, static datasets without the need for exploration. However, existing Q-learning and actor-critic based off-policy RL algorithms fail when bootstrapping from out-of-distribution (OOD) actions or states. We hypothesize that a key missing ingredient from the existing methods is a proper treatment of uncertainty in the offline setting. We propose Uncertainty Weighted Actor-Critic (UWAC), an algorithm that detects OOD state-action pairs and down-weights their contribution in the training objectives accordingly. Implementation-wise, we adopt a practical and effective dropout-based uncertainty estimation method that introduces very little overhead over existing RL algorithms. Empirically, we observe that UWAC substantially improves model stability during training. In addition, UWAC out-performs existing offline RL methods on a variety of competitive tasks, and achieves significant performance gains over the state-of-the-art baseline on datasets with sparse demonstrations collected from human experts. (@uwac)

Rishabh Agarwal, Dale Schuurmans, and Mohammad Norouzi An optimistic perspective on offline reinforcement learning In *International Conference on Machine Learning*, pages 104–114. PMLR, 2020. **Abstract:** Off-policy reinforcement learning (RL) using a fixed offline dataset of logged interactions is an important consideration in real world applications. This paper studies offline RL using the DQN replay dataset comprising the entire replay experience of a DQN agent on 60 Atari 2600 games. We demonstrate that recent off-policy deep RL algorithms, even when trained solely on this fixed dataset, outperform the fully trained DQN agent. To enhance generalization in the offline setting, we present Random Ensemble Mixture (REM), a robust Q-learning algorithm that enforces optimal Bellman consistency on random convex combinations of multiple Q-value estimates. Offline REM trained on the DQN replay dataset surpasses strong RL baselines. Ablation studies highlight the role of offline dataset size and diversity as well as the algorithm choice in our positive results. Overall, the results here present an optimistic view that robust RL algorithms trained on sufficiently large and diverse offline datasets can lead to high quality policies. The DQN replay dataset can serve as an offline RL benchmark and is open-sourced. (@agarwal2020optimistic)

Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song Uncertainty-based offline reinforcement learning with diversified q-ensemble *Advances in neural information processing systems*, 34: 7436–7447, 2021. **Abstract:** Offline reinforcement learning (offline RL), which aims to find an optimal policy from a previously collected static dataset, bears algorithmic difficulties due to function approximation errors from out-of-distribution (OOD) data points. To this end, offline RL algorithms adopt either a constraint or a penalty term that explicitly guides the policy to stay close to the given dataset. However, prior methods typically require accurate estimation of the behavior policy or sampling from OOD data points, which themselves can be a non-trivial problem. Moreover, these methods under-utilize the generalization ability of deep neural networks and often fall into suboptimal solutions too close to the given dataset. In this work, we propose an uncertainty-based offline RL method that takes into account the confidence of the Q-value prediction and does not require any estimation or sampling of the data distribution. We show that the clipped Q-learning, a technique widely used in online RL, can be leveraged to successfully penalize OOD data points with high prediction uncertainties. Surprisingly, we find that it is possible to substantially outperform existing offline RL methods on various tasks by simply increasing the number of Q-networks along with the clipped Q-learning. Based on this observation, we propose an ensemble-diversified actor-critic algorithm that reduces the number of required ensemble networks down to a tenth compared to the naive ensemble while achieving state-of-the-art performance on most of the D4RL benchmarks considered. (@edac)

Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhihong Deng, Animesh Garg, Peng Liu, and Zhaoran Wang Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning *arXiv preprint arXiv:2202.11566*, 2022. **Abstract:** Offline Reinforcement Learning (RL) aims to learn policies from previously collected datasets without exploring the environment. Directly applying off-policy algorithms to offline RL usually fails due to the extrapolation error caused by the out-of-distribution (OOD) actions. Previous methods tackle such problem by penalizing the Q-values of OOD actions or constraining the trained policy to be close to the behavior policy. Nevertheless, such methods typically prevent the generalization of value functions beyond the offline data and also lack precise characterization of OOD data. In this paper, we propose Pessimistic Bootstrapping for offline RL (PBRL), a purely uncertainty-driven offline algorithm without explicit policy constraints. Specifically, PBRL conducts uncertainty quantification via the disagreement of bootstrapped Q-functions, and performs pessimistic updates by penalizing the value function based on the estimated uncertainty. To tackle the extrapolating error, we further propose a novel OOD sampling method. We show that such OOD sampling and pessimistic bootstrapping yields provable uncertainty quantifier in linear MDPs, thus providing the theoretical underpinning for PBRL. Extensive experiments on D4RL benchmark show that PBRL has better performance compared to the state-of-the-art algorithms. (@pbrl)

Seunghyun Lee, Younggyo Seo, Kimin Lee, Pieter Abbeel, and Jinwoo Shin Offline-to-online reinforcement learning via balanced replay and pessimistic q-ensemble In *Conference on Robot Learning*, pages 1702–1712. PMLR, 2022. **Abstract:** Recent advance in deep offline reinforcement learning (RL) has made it possible to train strong robotic agents from offline datasets. However, depending on the quality of the trained agents and the application being considered, it is often desirable to fine-tune such agents via further online interactions. In this paper, we observe that state-action distribution shift may lead to severe bootstrap error during fine-tuning, which destroys the good initial policy obtained via offline RL. To address this issue, we first propose a balanced replay scheme that prioritizes samples encountered online while also encouraging the use of near-on-policy samples from the offline dataset. Furthermore, we leverage multiple Q-functions trained pessimistically offline, thereby preventing overoptimism concerning unfamiliar actions at novel states during the initial training phase. We show that the proposed method improves sample-efficiency and final performance of the fine-tuned robotic agents on various locomotion and manipulation tasks. Our code is available at: https://github.com/shlee94/Off2OnRL. (@lee2022offline)

Dibya Ghosh, Anurag Ajay, Pulkit Agrawal, and Sergey Levine Offline rl policies should be trained to be adaptive In *International Conference on Machine Learning*, pages 7513–7530. PMLR, 2022. **Abstract:** Offline RL algorithms must account for the fact that the dataset they are provided may leave many facets of the environment unknown. The most common way to approach this challenge is to employ pessimistic or conservative methods, which avoid behaviors that are too dissimilar from those in the training dataset. However, relying exclusively on conservatism has drawbacks: performance is sensitive to the exact degree of conservatism, and conservative objectives can recover highly suboptimal policies. In this work, we propose that offline RL methods should instead be adaptive in the presence of uncertainty. We show that acting optimally in offline RL in a Bayesian sense involves solving an implicit POMDP. As a result, optimal policies for offline RL must be adaptive, depending not just on the current state but rather all the transitions seen so far during evaluation.We present a model-free algorithm for approximating this optimal adaptive policy, and demonstrate the efficacy of learning such adaptive policies in offline RL benchmarks. (@apev)

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller Playing atari with deep reinforcement learning *arXiv preprint arXiv:1312.5602*, 2013. **Abstract:** We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. We find that it outperforms all previous approaches on six of the games and surpasses a human expert on three of them. (@dqn)

Diederik P Kingma and Max Welling Auto-encoding variational bayes *arXiv preprint arXiv:1312.6114*, 2013. **Abstract:** How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions are two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results. (@vae)

Nitin Bhatia et al Survey of nearest neighbor techniques *arXiv preprint arXiv:1007.0085*, 2010. **Abstract:** The nearest neighbor (NN) technique is very simple, highly efficient and effective in the field of pattern recognition, text categorization, object recognition etc. Its simplicity is its main advantage, but the disadvantages can’t be ignored even. The memory requirement and computation complexity also matter. Many techniques are developed to overcome these limitations. NN techniques are broadly classified into structure less and structure based techniques. In this paper, we present the survey of such techniques. Weighted kNN, Model based kNN, Condensed NN, Reduced NN, Generalized NN are structure less techniques whereas k-d tree, ball tree, Principal Axis Tree, Nearest Feature Line, Tunable NN, Orthogonal Search Tree are structure based algorithms developed on the basis of kNN. The structure less method overcome memory limitation and structure based techniques reduce the computational complexity. (@nn)

Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, and Lei Han Rorl: Robust offline reinforcement learning via conservative smoothing *Advances in neural information processing systems*, 35: 23851–23866, 2022. **Abstract:** Offline reinforcement learning (RL) provides a promising direction to exploit massive amount of offline data for complex decision-making tasks. Due to the distribution shift issue, current offline RL algorithms are generally designed to be conservative in value estimation and action selection. However, such conservatism can impair the robustness of learned policies when encountering observation deviation under realistic conditions, such as sensor errors and adversarial attacks. To trade off robustness and conservatism, we propose Robust Offline Reinforcement Learning (RORL) with a novel conservative smoothing technique. In RORL, we explicitly introduce regularization on the policy and the value function for states near the dataset, as well as additional conservative value estimation on these states. Theoretically, we show RORL enjoys a tighter suboptimality bound than recent theoretical results in linear MDPs. We demonstrate that RORL can achieve state-of-the-art performance on the general offline RL benchmark and is considerably robust to adversarial observation perturbations. (@rorl)

Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al Soft actor-critic algorithms and applications *arXiv preprint arXiv:1812.05905*, 2018. **Abstract:** Model-free deep reinforcement learning (RL) algorithms have been successfully applied to a range of challenging sequential decision making and control tasks. However, these methods typically suffer from two major challenges: high sample complexity and brittleness to hyperparameters. Both of these challenges limit the applicability of such methods to real-world domains. In this paper, we describe Soft Actor-Critic (SAC), our recently introduced off-policy actor-critic algorithm based on the maximum entropy RL framework. In this framework, the actor aims to simultaneously maximize expected return and entropy. That is, to succeed at the task while acting as randomly as possible. We extend SAC to incorporate a number of modifications that accelerate training and improve stability with respect to the hyperparameters, including a constrained formulation that automatically tunes the temperature hyperparameter. We systematically evaluate SAC on a range of benchmark tasks, as well as real-world challenging tasks such as locomotion for a quadrupedal robot and robotic manipulation with a dexterous hand. With these improvements, SAC achieves state-of-the-art performance, outperforming prior on-policy and off-policy methods in sample-efficiency and asymptotic performance. Furthermore, we demonstrate that, in contrast to other off-policy algorithms, our approach is very stable, achieving similar performance across different random seeds. These results suggest that SAC is a promising candidate for learning in real-world robotics tasks. (@sacauto)

</div>

# Proof [sec:proof]

**Theorem 3.1** *We denote the \\(Q\\)-function converged from the \\(Q\\)-update of EPQ using the proposed penalty \\(\mathcal{P}_\tau\\) in <a href="#eq:bellmanours" data-reference-type="eqref" data-reference="eq:bellmanours">[eq:bellmanours]</a> by \\(\hat{Q}^\pi\\). Then, the expected value of \\(\hat{Q}^\pi\\) underestimates the expected true policy value, i.e., \\(\mathbb{E}_{a\sim\pi}[\hat{Q}^\pi(s,a)] \leq \mathbb{E}_{a\sim\pi}[Q^\pi(s,a)],  \forall s \in D\\), with high probability \\(1-\delta\\) for some \\(\delta \in (0,1)\\), if the penalizing factor \\(\alpha\\) is sufficiently large. Furthermore, the proposed penalty reduces the average penalty for policy actions compared to the average penalty of CQL.*

## Proof of Theorem <a href="#thm:penalty" data-reference-type="ref" data-reference="thm:penalty">1</a> [proof-of-theorem-thmpenalty]

Proof of Theorem <a href="#thm:penalty" data-reference-type="ref" data-reference="thm:penalty">1</a> basically follows the proof of Theorem 3.2 in `\citet{CQL}`{=latex} since \\(\mathcal{P}_\tau\\) multiplies the penalty control factor \\(f_\tau^{\pi,\hat{\beta}}(s)\\) to the penalty of CQL. At each \\(k\\)-th iteration, \\(Q\\)-function is updated by equation <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a>, then \\[Q_{k+1}(s,a) \leftarrow \hat{\mathcal{B}}^{\pi}Q_k(s,a) - \alpha \mathcal{P}_\tau,~\forall s,a,
\tag{A.1}
\label{eq:apqiter}\\] where \\(\hat{\mathcal{B}}^{\pi}\\) is the estimation of the true Bellman operator \\(\mathcal{B}^{\pi}\\) based on data samples. It is known that the error between the estimated Bellman operator \\(\hat{\mathcal{B}}^\pi\\) and the true Bellman operator is bounded with high probability of \\(1-\delta\\) for some \\(\delta \in (0,1)\\) as \\(|(\mathcal{B}^{\pi}Q)(s,a)-(\hat{\mathcal{B}}^\pi Q)(s,a)|\leq \xi^\delta(s,a),~~\forall s,a\\), where \\(\xi^\delta\\) is a positive constant related to the given dataset \\(D\\), the discount factor \\(\gamma\\), and the transition probability \\(P\\) `\cite{CQL}`{=latex}. Then, with high probability \\(1-\delta\\), \\[Q_{k+1}(s,a) \leftarrow \mathcal{B}^{\pi}Q_k(s,a) - \alpha \mathcal{P}_\tau + \xi^\delta(s,a),~\forall s,a,
\tag{A.2}
\label{eq:qiter}\\] Now, with the state value function \\(V(s) := \mathbb{E}_{a\sim \pi(\cdot|s)}[Q(s,a)]\\) \\[\begin{aligned}
    V_{k+1}(s) &= \mathbb{E}_{a\sim \pi(\cdot|s)}[Q_k(s,a)]=\mathcal{B}^{\pi}V_k - \alpha \mathbb{E}_{a\sim\pi}[\mathcal{P}_\tau] + \xi^\delta(s,a) \nonumber\\
    &=\mathcal{B}^{\pi}V_k(s) - \alpha \mathbb{E}_{a\sim\pi}\left[f_\tau^{\pi,\hat{\beta}}(s)\cdot\left(\frac{\pi(a|s)}{\hat{\beta}(a|s)} - 1\right) + \mathbb{E}_{a\sim\pi}[\xi^\delta(s,a)] \right]\nonumber\\
    &= \mathcal{B}^{\pi}V_k(s) - \alpha \Delta_{EPQ}^{\pi}(s) + \mathbb{E}_{a\sim\pi}[\xi^\delta(s,a)]
    \tag{A.3}
    \label{eq:vupdate}
\end{aligned}\\] Upon repeated iteration, \\(V_{k+1}\\) converges to \\(V_\infty(s) = V^\pi(s)+(I-\gamma P^\pi)^{-1}\cdot\{- \alpha\Delta_{EPQ}^{\pi}(s) + \mathbb{E}_{a\sim\pi}[\xi^\delta(s,a)]\}\\) based on the fixed point theorem, where \\(\Delta_{EPQ}^{\pi}(s):=\mathbb{E}_{a\sim\pi}[\mathcal{P}_\tau]\\) is the average penalty for policy \\(\pi\\), \\(I\\) is the identity matrix, and \\(P^\pi\\) is the state transition matrix where the policy \\(\pi\\) is given. Here, we can show that the average penalty \\(\Delta_{EPQ}^{\pi}(s)\\) is positive as follows: \\[\begin{aligned}
\Delta_{EPQ}^{\pi}(s) &=\mathbb{E}_{a\sim\pi}\bigg[f_\tau^{\pi,\hat{\beta}}(s)\cdot\left(\frac{\pi(a|s)}{\beta(a|s)} - 1\right)\bigg]\nonumber\\
&= f_\tau^{\pi,\hat{\beta}}(s)\left[\sum_{a\in\mathcal{A}}\pi(a|s)\left(\frac{\pi(a|s)}{\hat{\beta}(a|s)} - 1\right) - \underbrace{\sum_{a\in\mathcal{A}}\hat{\beta}(a|s)\left(\frac{\pi(a|s)}{\hat{\beta}(a|s)} - 1\right)}_{=0}\right]\nonumber\\
&= f_\tau^{\pi,\hat{\beta}}(s)\cdot\sum_{a\in\mathcal{A}}\frac{(\pi(a|s)-\hat{\beta}(a|s))^2}{\hat{\beta}(a|s)}\geq 0,
\tag{A.4}
\label{eq:delta}
\end{aligned}\\] where the equality in <a href="#eq:delta" data-reference-type="eqref" data-reference="eq:delta">[eq:delta]</a> satisfies when \\(\pi=\hat{\beta}\\) or \\(f_\tau^{\pi,\hat{\beta}} = 0\\). Given that \\(V_{k+1}\\) converges to \\(V_\infty=V^\pi(s) + (I-\gamma P^\pi)^{-1}\cdot \{ - \alpha\Delta_{EPQ}^{\pi}(s) + \mathbb{E}_{a\sim\pi}[\xi^\delta(s,a)]\}\\), choosing the penalizing constant \\(\alpha\\) that satisfies \\(\alpha \geq \max_{s,a\in D}[\xi^\delta(s,a)]\cdot\max_{s\in D} (\Delta_{EPQ}^\pi(s))^{-1}\\) will satisfy, \\[\begin{aligned}
&- \alpha\cdot\Delta_{EPQ}^{\pi}(s) + \mathbb{E}_{a\sim\pi}[\xi^\delta(s,a)]\nonumber\\
&\leq- \max_{s,a\in D}[\xi^\delta(s,a)]\cdot \underbrace{\max_{s\in D} (\Delta_{EPQ}^\pi(s))^{-1} \cdot \Delta_{EPQ}^{\pi}(s)}_{\geq 1} +  \mathbb{E}_{a\sim\pi}[\xi^\delta(s,a)]\nonumber\\
    &\leq- \max_{s,a\in D}[\xi^\delta(s,a)] +  \mathbb{E}_{a\sim\pi}[\xi^\delta(s,a)] \leq 0,~~~~\forall s,
\tag{A.5}
\label{eq:underestimate}
\end{aligned}\\] Since \\(I-\gamma P^\pi\\) is non-singular \\(M\\)-matrix and the inverse of non-singular \\(M\\)-matrix is non-negative, i.e., all elements of \\((I - \gamma P^\pi)^{-1}\\) are non-negative, \\(V_\infty(s)  = V^\pi(s) + (I-\gamma P^\pi)^{-1}\cdot \{ - \alpha\Delta_{EPQ}^{\pi}(s) + \mathbb{E}_{a\sim\pi}[\xi^\delta(s,a)] \}\leq V^\pi(s),~\forall s.\\) Therefore, \\(V_\infty\\) underestimates the true value function \\(V^\pi\\) if the penalizing constant \\(\alpha\\) satisfies \\(\alpha \geq \max_{s,a\in D}[\xi^\delta(s,a)]\cdot\max_{s\in D} (\Delta_{EPQ}^\pi(s))^{-1}\\). In addition, according to `\cite{CQL}`{=latex}, the average penalty of CQL for policy actions can be represented as \\(\Delta_{CQL}^\pi(s)=\mathbb{E}_{a\sim\pi}[\frac{\pi}{\hat{\beta}}-1]\\). Thus, \\(\Delta_{EPQ}^\pi(s)=f_\tau^{\pi,\hat{\beta}}(s)\Delta_{CQL}^\pi(s)\\) and \\(f_\tau^{\pi,\hat{\beta}}(s)\leq 1\\) from the definition in <a href="#eq:TAP" data-reference-type="eqref" data-reference="eq:TAP">[eq:TAP]</a>, so \\(0\leq\Delta_{EPQ}^\pi(s) \leq \Delta_{CQL}^\pi(s)\\). In addition, if \\(\pi=\hat{\beta}\\), then \\(0=\Delta_{EPQ}^{\hat{\beta}}(s) = \Delta_{CQL}^{\hat{\beta}}(s)\\) from the equality condition in <a href="#eq:delta" data-reference-type="eqref" data-reference="eq:delta">[eq:delta]</a>, which indicates that the average penalty for data actions is \\(0\\) for both EPQ and CQL. \\(\blacksquare\\)

# Implementation Details [sec:impleappen]

In this section, we provide the implementation details of the proposed EPQ. First of all, we provide a detailed derivation of the final \\(Q\\)-loss function<a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a> of EPQ in Section <a href="#subsec:derivation" data-reference-type="ref" data-reference="subsec:derivation">8.1</a>. Next, we introduce a practical implementation of EPQ to compute the loss functions for the parameterized policy and \\(Q\\)-function in Section <a href="#subsec:impledetailappen" data-reference-type="ref" data-reference="subsec:impledetailappen">8.2</a>. In addition, to calculate loss functions in Section <a href="#subsec:impledetailappen" data-reference-type="ref" data-reference="subsec:impledetailappen">8.2</a>, we provide the additional implementation details in Appendices <a href="#subsec:VAE" data-reference-type="ref" data-reference="subsec:VAE">8.3</a>, <a href="#subsec:IS" data-reference-type="ref" data-reference="subsec:IS">8.4</a>, and <a href="#subsec:logsumexp" data-reference-type="ref" data-reference="subsec:logsumexp">8.5</a>. We conduct our experiments on a single server equipped with an Intel Xeon Gold 6336Y CPU and one NVIDIA RTX A5000 GPU, and we compare the running time of EPQ with other baseline algorithms in Section <a href="#subsec:timecomp" data-reference-type="ref" data-reference="subsec:timecomp">8.6</a>. For additional hyperparameters in the practical implementation of EPQ, we provide detailed hyperparameter setup and additional ablation studies in Appendix <a href="#sec:hyper" data-reference-type="ref" data-reference="sec:hyper">9</a> and Appendix <a href="#sec:ablappen" data-reference-type="ref" data-reference="sec:ablappen">10</a>, respectively.

## Detailed Derivation of \\(Q\\)-Loss Function [subsec:derivation]

In Section <a href="#subsec:pd" data-reference-type="ref" data-reference="subsec:pd">3.3</a>, the final \\(Q\\)-loss function with the proposed penalty \\(\mathcal{P}_{\tau,PD} = f_\tau^{\pi,\hat{\beta}} (\frac{\pi}{\hat{\beta}^Q} - 1)\\) is given by \\(L(Q)=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \{\mathcal{B}^\pi Q - \alpha \mathcal{P}_{\tau,~PD}\} \right)^2\right]\\). In this section, we provide a more detailed calculation of \\(L(Q)\\) to obtain <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a> as follows:

\\[\begin{aligned}
&L(Q)=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \{\mathcal{B}^\pi Q - \alpha \mathcal{P}_{\tau,~PD}\} \right)^2\right]\nonumber\\
&=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \mathcal{B}^\pi Q\right)^2\right]+ \alpha\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[ \mathcal{P}_{\tau,~PD}\cdot Q\right] + C\nonumber\\
&=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \mathcal{B}^\pi Q\right)^2\right]+ \alpha\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[ f_\tau^{\pi,\hat{\beta}}\left(\frac{\pi}{\hat{\beta}^Q} - 1\right) Q\right] + C\nonumber\\
&=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \mathcal{B}^\pi Q\right)^2\right]+ \alpha\mathbb{E}_{s,s'\sim D}\left[\int_{a\in\mathcal{A}}\hat{\beta}^Q f_\tau^{\pi,\hat{\beta}}\left(\frac{\pi}{\hat{\beta}^Q} - 1\right) Q da\right]+C\nonumber\\
&=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \mathcal{B}^\pi Q\right)^2\right]+ \alpha\mathbb{E}_{s,s'\sim D}\left[\int_{a\in\mathcal{A}}f_\tau^{\pi,\hat{\beta}}\left(\pi - \hat{\beta}^Q\right) Q da\right]+C\nonumber\\
&=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \mathcal{B}^\pi Q\right)^2\right]+ \alpha\mathbb{E}_{s,s'\sim D}\left[\int_{a'\in\mathcal{A}}\pi f_\tau^{\pi,\hat{\beta}}  Q da'-\int_{a\in\mathcal{A}}\hat{\beta}^Qf_\tau^{\pi,\hat{\beta}}  Q da\right]+C\nonumber\\
&=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \mathcal{B}^\pi Q\right)^2\right]+ \alpha\mathbb{E}_{s,s'\sim D}\left[\mathbb{E}_{a'\sim\pi}\left[ f_\tau^{\pi,\hat{\beta}}  Q \right]-\mathbb{E}_{a\sim\hat{\beta}^Q}\left[ f_\tau^{\pi,\hat{\beta}}  Q \right]\right]+C\nonumber\\
&=\frac{1}{2}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\left(Q - \mathcal{B}^\pi Q\right)^2\right]+ \alpha\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}^Q}\left[\mathbb{E}_{a'\sim\pi}\left[ f_\tau^{\pi,\hat{\beta}}  Q \right]- f_\tau^{\pi,\hat{\beta}}  Q \right]+C\nonumber\\
&\underset{(*)}{=}\mathbb{E}_{s,s'\sim D,a\sim\hat{\beta}}\left[ \frac{\hat{\beta}^Q}{\hat{\beta}}\cdot\left\{\frac{1}{2}\left(Q - \mathcal{B}^\pi Q\right)^2+ \alpha f_\tau^{\pi,\hat{\beta}}\cdot\left(\mathbb{E}_{a'\sim\pi}\left[   Q \right]- Q \right)\right\}\right]+C\nonumber\\
&= \mathbb{E}_{s,s'\sim D,a\sim\hat{\beta},a'\sim\pi}\left[w_{s,a}^Q\cdot\left\{\frac{1}{2}\left(Q(s, a) - \mathcal{B}^\pi Q(s, a)  \right)^2+\alpha f_\tau^{\pi,\hat{\beta}}(s) ( Q(s, a') - Q(s, a))\right\}\right]+C\nonumber, 
\end{aligned}\\] where \\(C\\) is the remaining constant term that can be ignored for the \\(Q\\)-update since \\(\mathcal{B}^\pi Q\\) is the fixed target value. For \\((*)\\), we apply the IS technique, which states that \\(\mathbb{E}_{x\sim p}[f(x)] = \mathbb{E}_{x\sim q}\left[\frac{p(x)}{q(x)}f(x)\right]\\) for any probability distributions \\(p\\) and \\(q\\), and arbitrary function \\(f\\), and \\(w_{s,a}^{Q} = \frac{\hat{\beta}^Q(a|s)}{\hat{\beta}(a|s)} = \frac{\exp(Q(s,a))}{\mathbb{E}_{a'\sim\hat{\beta}(\cdot|s)}[\exp(Q(s,a'))]}\\) is the importance sampling (IS) ratio between \\(\hat{\beta}^Q\\) and \\(\hat{\beta}\\).

## Practical Implementation for EPQ [subsec:impledetailappen]

Our implementation basically follows the setup of CQL `\cite{CQL}`{=latex}. We use the Gaussian policy \\(\pi\\) with a \\(\textrm{Tanh}(\cdot)\\) layer proposed by `\citet{sac}`{=latex}, and parameterize the policy \\(\pi\\) and \\(Q\\)-function using neural network parameters \\(\phi\\) and \\(\theta\\), respectively. Then, we update the policy to maximize \\(Q_\theta\\) with its entropy \\(\mathcal{H}(\pi_\phi)= \mathbb{E}_{\pi_\phi}[-\log\pi_\phi]\\), following the maximum entropy principle `\cite{sac}`{=latex} as explained in Section <a href="#subsec:pd" data-reference-type="ref" data-reference="subsec:pd">3.3</a>, to account for stochastic policies. Then, we can redefine the policy loss function \\(L(\pi)\\) defined in <a href="#eq:policyloss" data-reference-type="eqref" data-reference="eq:policyloss">[eq:policyloss]</a> as the policy loss function \\(L_\pi(\phi)\\) for policy parameter \\(\phi\\), given by \\[L_\pi(\phi) = \mathbb{E}_{s\sim D,~a\sim\pi_\phi}[- Q_\theta(s,a) + \log\pi_\phi(a|s)].
\tag{B.1}
\label{eq:policylossappen}\\]

For the \\(Q\\)-loss function in <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a>, we use the IS ratio \\(w_{s,a}^Q\\) in <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a> to account for prioritized sampling based on \\(\hat{\beta}^Q\\). However, \\(\hat{\beta}^Q\\) discards samples with low IS weights, which can reduce sample efficiency. To address this, we utilize the clipped IS weight \\(\max(c_{\min},w_{s,a}^Q)\\), where \\(c_{\min}\in(0,1]\\) is the IS clipping constant. This clipped IS weight is multiplied only to the term \\((Q(s, a) - \mathcal{B}^\pi Q(s, a))^2\\) in <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a> to ensure that we can exploit all data samples for \\(Q\\)-learning while preserving the proposed penalty. The detailed analysis for \\(c_{\min}\\) is provided in Appendix <a href="#sec:ablappen" data-reference-type="ref" data-reference="sec:ablappen">10</a>. In addition, the optimal policy that maximizes <a href="#eq:policylossappen" data-reference-type="eqref" data-reference="eq:policylossappen">[eq:policylossappen]</a> follows the Boltzmann distribution, proportional to \\(\exp(Q_\theta(s,\cdot))\\). It has been proven in `\citet{CQL}`{=latex} that the optimal policy satisfies \\(\mathbb{E}_{a\sim\pi}[Q_\theta(s,a)] + H(\pi) = \log\sum_{a\sim\mathcal{A}} \exp Q_\theta(s,a)\\), so we can replace the \\(\mathbb{E}_{a'\sim\pi}[Q_\theta(s,a')]\\) term in <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a> with \\(\log\sum_{a'\sim\mathcal{A}} \exp Q_\theta(s,a')\\), given that \\(H(\pi)\\) does not depend on the \\(Q\\)-function. The Bellman operator \\(\mathcal{B}^\pi\\) can be estimated by samples in the dataset as \\(\mathcal{B}^\pi Q_\theta\approx r(s,a) + \mathbb{E}_{a'\sim\pi}\gamma Q_{\bar{\theta}}(s',a')\\), where \\(\bar{\theta}\\) is the parameter of the target \\(Q\\)-function. The target network is updated using exponential moving average (EMA) with temperature \\(\eta_{\bar{\theta}}=0.005\\), as proposed in the deep Q-network (DQN) `\cite{dqn}`{=latex}. Finally, by applying IS clipping and \\(\log \sum_a \exp Q\\) to the \\(Q\\)-loss function <a href="#eq:qupdate" data-reference-type="eqref" data-reference="eq:qupdate">[eq:qupdate]</a> and redefining it as the value loss function for the value parameter \\(\theta\\), we obtain the following refined value loss function \\(L_Q(\theta)\\) as follows: \\[\begin{aligned}
& L_Q(\theta)= \frac{1}{2}\mathbb{E}_{s,a,s'\sim D}\big[\max(c_{\min},w_{s,a}^Q)\cdot\left(r(s,a) + \mathbb{E}_{a'\sim\pi}\gamma Q_{\bar{\theta}}(s',a') - Q_\theta(s, a)\right)^2\big]\tag{B.2}\label{eq:qlossfinal}\\
&\quad\quad\quad\quad +\alpha \mathbb{E}_{s, a\sim D}\left[w_{s,a}^Q f_\tau^{\pi,\hat{\beta}}(s)\left( \log\sum_{a'\in\mathcal{A}} Q_\theta(s, a') - Q_\theta(s, a)\right)\right],\nonumber
\end{aligned}\\] where \\(\hat{\beta}\\) is pre-trained by behavior cloning (BC) `\cite{bc, vae}`{=latex} to compute \\(f_\tau^{\pi,\hat{\beta}}\\). The parameters \\(\phi\\) and \\(\theta\\) are updated to minimize their loss functions \\(L_\pi(\phi)\\) and \\(L_Q(\theta)\\) with learning rate \\(\eta_\phi\\) and \\(\eta_\theta\\), respectively. Detailed implementations for estimating the behavior policy \\(\hat{\beta}\\), the IS weight \\(w_{s,a}^Q\\), and \\(\log\sum_a\exp Q\\) are provided in Appendices <a href="#subsec:VAE" data-reference-type="ref" data-reference="subsec:VAE">8.3</a>, <a href="#subsec:IS" data-reference-type="ref" data-reference="subsec:IS">8.4</a>, and <a href="#subsec:logsumexp" data-reference-type="ref" data-reference="subsec:logsumexp">8.5</a>, respectively.

## Behavior Policy Estimation Based on Variational Auto-Encoder [subsec:VAE]

In Section <a href="#subsec:impledetailappen" data-reference-type="ref" data-reference="subsec:impledetailappen">8.2</a>, we estimate the behavior policy \\(\beta\\) that generates the data samples in \\(D\\) necessary for calculating the penalty adaptation factor \\(f_\tau^{\pi,\hat{\beta}}\\) in equation <a href="#eq:TAP" data-reference-type="eqref" data-reference="eq:TAP">[eq:TAP]</a>. To estimate the behavior policy \\(\hat{\beta}\\), we employ the variational auto-encoder (VAE), one of the most representative variational inference methods, to approximate the underlying distribution of a large dataset based on the variational lower bound `\cite{vae}`{=latex}. In the context of VAE, we define an encoder model \\(p_\psi (z|s,a)\\) and a decoder model \\(q_\psi(a|z,s)\\) parameterized by \\(\psi\\), where \\(z\\) is the latent variable whose prior distribution \\(p(z)\\) follows the multivariate normal distribution, i.e., \\(p(z)\sim N(0,I)\\). Assuming independence among all data samples, we can derive the variational lower bound for the likelihood of \\(\beta\\) as proposed by `\citet{vae}`{=latex}: \\[\begin{aligned}
\log\beta(a|s) &\geq \underbrace{\mathbb{E}_{z\sim p_\psi(\cdot|s,a)}[\log q_\psi(a|z,s)]- D_{KL}(p_\psi(z|s,a)||p(z))}_{\textrm{the variational lower bound}},~\forall s,a \in D
\tag{B.3}
\label{eq:vae}
\end{aligned}\\] where \\(D_{KL}(p||q)=\mathbb{E}_{p}[\log p -\log q]\\) is the Kullback-Leibler (KL) divergence between two distributions \\(p\\) and \\(q\\). In this paper, since we consider the deterministic decoder \\(q_\psi(z,s)\\), the formal term \\(\mathbb{E}_{z\sim p_\psi(\cdot|s,a)}[\log q_\psi(a|z,s)]\\) can be replaced with the mean square error (MSE) as \\(\mathbb{E}_{z\sim p_\psi(\cdot|s,a)}[\log q_\psi(a|z,s)] \approx \mathbb{E}_{z\sim p_\psi(\cdot|s,a)}[( q_\psi(z,s)-a) ^2 ]\\). At each \\(k\\)-th iteration, we update the parameter \\(\psi\\) of VAE to maximize the lower bound in equation <a href="#eq:vae" data-reference-type="eqref" data-reference="eq:vae">[eq:vae]</a>. The \\(\log\beta\\) can be estimated using the variational lower-bound in <a href="#eq:vae" data-reference-type="eqref" data-reference="eq:vae">[eq:vae]</a> to obtain \\(f_\tau^{\pi,\hat{\beta}}\\). The hyperparameter setup for the VAE is provided in Table <a href="#table:vaeparams" data-reference-type="ref" data-reference="table:vaeparams">4</a>.

<div id="table:vaeparams" markdown="1">

<table>
<caption>Hyperparameter setup for VAE</caption>
<thead>
<tr>
<th colspan="2" style="text-align: center;"><strong>VAE Hyperparameters</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span class="math inline"><em>z</em></span> dimension</td>
<td style="text-align: left;"><span class="math inline">2⋅</span> state dimension</td>
</tr>
<tr>
<td style="text-align: left;">Hidden activation function</td>
<td style="text-align: left;">ReLU Layer</td>
</tr>
<tr>
<td style="text-align: left;">Encoder network <span class="math inline"><em>p</em><sub><em>ψ</em></sub></span></td>
<td style="text-align: left;"><div id="table:vaeparams">
<table>
<caption>Hyperparameter setup for VAE</caption>
<tbody>
<tr>
<td style="text-align: left;">(512, <span class="math inline">2 ⋅ <em>z</em></span> dim.)</td>
</tr>
<tr>
<td style="text-align: left;">(512,512)</td>
</tr>
<tr>
<td style="text-align: left;">(state dim. + action dim., 512)</td>
</tr>
</tbody>
</table>
</div></td>
</tr>
<tr>
<td style="text-align: left;">Decoder network <span class="math inline"><em>q</em><sub><em>ψ</em></sub></span></td>
<td style="text-align: left;"><div id="table:vaeparams">
<table>
<caption>Hyperparameter setup for VAE</caption>
<tbody>
<tr>
<td style="text-align: left;">(512, action dim.)</td>
</tr>
<tr>
<td style="text-align: left;">(512,512)</td>
</tr>
<tr>
<td style="text-align: left;">(<span class="math inline"><em>z</em></span> dim. + state dim., 512)</td>
</tr>
</tbody>
</table>
</div></td>
</tr>
</tbody>
</table>

</div>

<span id="table:vaeparams" label="table:vaeparams"></span>

## Implementation of IS Weight \\(w_{s,a}^Q\\) [subsec:IS]

In order to consider the prioritized data distribution \\(\hat{\beta}^Q\\) proposed in Section <a href="#subsec:pd" data-reference-type="ref" data-reference="subsec:pd">3.3</a>, we use the importance sampling (IS) weight defined by

\\[w_{s,a}^Q = \frac{\hat{\beta}^Q(a|s)}{\hat{\beta}(a|s)} = \frac{\exp(Q(s,a))}{\mathbb{E}_{a'\sim\hat{\beta}(\cdot|s)}[\exp(Q(s,a'))]},~\forall s,a \in D.
\tag{B.4}\\]

Since the computation of \\(\mathbb{E}_{a'\sim\hat{\beta}(\cdot|s)}\\) makes it difficult to know the exact possible action set for state \\(s\\), we approximately estimate the IS weight based on clustering as follows:

\\[w_{s,a}^Q = \frac{\exp(Q(s,a))}{\mathbb{E}_{a'\sim\hat{\beta}(\cdot|s)}[\exp(Q(s,a'))]} \approx \frac{\exp(Q(s,a))}{\frac{1}{|\mathcal{C}_{s,a}|}\sum_{(s',a') \in \mathcal{C}_{s,a}}\exp(Q(s',a'))},~\forall s,a \in D.
    \tag{B.5}\\]

Here, \\(\mathcal{C}_{s,a}\\) is the cluster that contains data samples adjacent to \\((s,a)\\), defined by \\[\mathcal{C}_{s,a} = \{(s',a') \in D ~|~||s-s'||_2 \leq \epsilon\cdot \bar{d}_{\textrm{closest}}\},
    \tag{B.6}\\]

where the cluster \\(\mathcal{C}_{s,a}\\) can be directly obtained using the nearest neighbor (NN) algorithm `\cite{nn}`{=latex} provided in the Python library. \\(\epsilon\cdot \bar{d}_{\textrm{closest}}\\) is the radius of the cluster, and \\(\bar{d}_{\textrm{closest}}\\) is the average distance between the closest states from each task. In our implementation, we control the radius parameter \\(\epsilon>0\\) to adjust the number of adjacent samples for the estimation of IS Weight \\(w_{s,a}^Q\\). In addition, using the \\(Q\\)-function in the IS weight term makes the learning unstable since the \\(Q\\)-function continuously changes as the learning progresses. Thus, instead of the \\(Q\\)-function, we use the regularized empirical return \\(G_t/\zeta\\) for each state-action pair obtained by the trajectories stored in \\(D\\), where \\(\zeta>0\\) is the regularizing temperature. Upon the increase of \\(\zeta\\), the returned difference between adjacent samples in the cluster decreases, so the effect of prioritization can be reduced. The detailed analysis for \\(\epsilon\\) and \\(\zeta\\) is provided in Appendix <a href="#sec:ablappen" data-reference-type="ref" data-reference="sec:ablappen">10</a>.

## Implementation of \\(Q\\)-loss Function [subsec:logsumexp]

In equation <a href="#eq:qlossfinal" data-reference-type="eqref" data-reference="eq:qlossfinal">[eq:qlossfinal]</a>, the final \\(Q\\)-loss function of proposed EPQ is given by \\[\begin{aligned}
L_Q(\theta)&=\frac{1}{2}\mathbb{E}_{s,a,s'\sim D}\big[\max(c_{\min},w_{s,a}^Q)\left(r(s,a) + \mathbb{E}_{a'\sim\pi}\gamma Q_{\bar{\theta}}(s',a') - Q_\theta(s, a)\right)^2\big]\nonumber\\
&+\alpha \mathbb{E}_{s, a\sim D}\left[w_{s,a}^Q f_\tau^{\pi,\hat{\beta}}(s)\left( \log\sum_{a'\in\mathcal{A}} \exp Q_\theta(s, a') - Q_\theta(s, a)\right)\right].\nonumber
\end{aligned}\\] Here, we can estimate \\(\log\sum_a \exp Q(s,a)\\) based on the method proposed in CQL `\cite{CQL}`{=latex} as follows: \\[\begin{aligned}
&\log\sum_a \exp Q(s,a)=\log\left(\frac{1}{2}\sum_a \pi(a|s) \{\exp( Q(s,a) - \log \pi(a|s))\}+ \frac{1}{2}\sum_a \rho_d\{\exp( Q(s,a) -\log \rho_d )\}\right)\nonumber\\
&\quad\quad\approx \log\left(\frac{1}{2 N_a}\sum_{a_n\sim\pi}^{N_a} (\exp( Q(s,a_n) - \log \pi(a_n|s))) + \frac{1}{2 N_a}\sum_{a_n\sim \textrm{Unif}(\mathcal{A})}^{N_a} (\exp( Q(s,a_n) - \log\rho_d) ))\right),
\tag{B.7}
\label{eq:logsumexp}
\end{aligned}\\]

where \\(N_a\\) is the number of action sampling, \\(\textrm{Unif}(\mathcal{A})\\) is a Uniform distribution on \\(\mathcal{A}\\), and \\(\rho_d\\) is the density of uniform distribution.

## Time comparison with other offline RL methods [subsec:timecomp]

In this sectrion, we compare the runtime of EPQ with other baseline algorithms: CQL, Onestep, IQL, MCQ, and MISA in Table <a href="#table:time" data-reference-type="ref" data-reference="table:time">5</a> below. For a fair comparison across all algorithms, we conducted experiments on the Hopper-medium task, which is a popular dataset for comparing computational costs `\cite{edac, rorl}`{=latex}, on a single server equipped with an Intel Xeon Gold 6336Y CPU and one NVIDIA RTX A5000 GPU. We measured both epoch runtime during 1,000 gradient steps and score runtime that each algorithm takes to achieve certain normalized scores.

From the epoch runtime results in Table <a href="#table:time" data-reference-type="ref" data-reference="table:time">5</a>, we can observe that EPQ takes approximately 2-30% more runtime per gradient step compared to the CQL baseline. Note that Onestep RL may seem to have very short execution time compared to other algorithms, but one must consider the significantly longer pretraining time required to learn the \\(Q\\)-function of behavior policy accurately. Additionally, compared to faster offline RL algorithms such as IQL and MISA, EPQ requires more runtime per step and exhibits a similar runtime to MCQ, another conservative Q-learning algorithm. However, according to the score runtime results in Table <a href="#table:time" data-reference-type="ref" data-reference="table:time">5</a>, we can observe that only proposed EPQ achieves a score of 100 points, while all other algorithms fail to reach this score. Particularly, compared to MCQ, which also considers CQL as a baseline, EPQ achieves the same score with significantly less runtime. Therefore, while EPQ may consume slightly more runtime per gradient step compared to other algorithms, we can conclude that proposed EPQ offers substantial advantages in terms of convergence performance over other algorithms.

<div id="table:time" markdown="1">

| **epoch runtime(s)** | **CQL** | **Onestep** | **IQL** | **MCQ** | **MISA** | **EPQ** |  |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1,000 gradient steps | 43.1 | 12.6 | 13.8 | 58.1 | 23.5 | 54.8 |  |
| **score runtime(s)** | **CQL** | **Onestep** | **IQL** | **MCQ** | **MISA** | **EPQ** |  |
| **Normalized average return** |  |  |  |  |  |  |  |
| 60 | 3540.0 | 252.5 | 1600.2 | 31,143.4 | 4,632.7 | 3,232.2 |  |
| 80 | \- | 568.4 | \- | 49,359.7 | \- | 21,920.0 |  |
| 100 | \- | \- | \- | \- | \- | 30,633.2 |  |

Runtime comparison: Epoch runtime and Score runtime

</div>

<span id="table:time" label="table:time"></span>

# Hyperparameter Setup [sec:hyper]

The implementation of proposed EPQ basically follows the implementation of the CQL algorithm `\cite{CQL}`{=latex}. First, we provide the details of the shared algorithm hyperparameters in Table <a href="#table:algoparam" data-reference-type="ref" data-reference="table:algoparam">8</a>. In Table <a href="#table:algoparam" data-reference-type="ref" data-reference="table:algoparam">8</a>, we compare the shared algorithm hyperparameters of CQL, the revised version of CQL (revised), and proposed EPQ. CQL (revised) considers the same hyperparameter setup with our algorithm for Adroit tasks since the reproduced performance of CQL (reprod.) using the author-provided hyperparameter setup significantly underperforms compared to the result of CQL (paper) in Table <a href="#table:performance" data-reference-type="ref" data-reference="table:performance">1</a>.

For the coefficient of entropy term in the policy update <a href="#eq:policylossappen" data-reference-type="eqref" data-reference="eq:policylossappen">[eq:policylossappen]</a>, CQL automatically controls the entropy coefficient so that the entropy of \\(\pi\\) goes to the target entropy, as proposed in `\citet{sacauto}`{=latex}. We observe that while the automatic control of policy entropy proves effective for Mujoco tasks, it adversely affects the performance in Adroit tasks since a policy with low entropy can lead to significant overestimation errors in Adroit tasks. Thus, we considered fixed entropy coefficient for Adroit tasks as in Table <a href="#table:algoparam" data-reference-type="ref" data-reference="table:algoparam">8</a>. In addition, CQL controls the penalizing constant \\(\alpha\\) based on Lagrangian method `\cite{CQL}`{=latex} for Adroit tasks, but we also observe that the automatic control of \\(\alpha\\) destabilizes training, leading to poor performance. Therefore, we considered fixed penalizing constant for Adroit tasks in Table <a href="#table:algoparam" data-reference-type="ref" data-reference="table:algoparam">8</a> for stable learning.

In addition, in Table <a href="#table:taskparam" data-reference-type="ref" data-reference="table:taskparam">9</a>, we provide the details of the task hyperparameters regarding our contributions in the proposed EPQ: the penalty control threshold \\(\tau\\) and the IS clipping factor \\(c_{\min}\\) in the \\(Q\\)-loss implementation in <a href="#eq:qlossfinal" data-reference-type="eqref" data-reference="eq:qlossfinal">[eq:qlossfinal]</a>, and the cluster radius \\(\epsilon\\) and regularizing temperature \\(\zeta\\) for the practical implementation of IS clipping factor \\(w_{s,a}^Q\\) in Section <a href="#subsec:IS" data-reference-type="ref" data-reference="subsec:IS">8.4</a>. Note that \\(\rho\\) in Table <a href="#table:taskparam" data-reference-type="ref" data-reference="table:taskparam">9</a> represents the log-density of uniform distribution. For the task hyperparameters, we consider various hyperparameter setups and provide the best hyperparameter setup for all considered tasks in Table <a href="#table:taskparam" data-reference-type="ref" data-reference="table:taskparam">9</a>. The results are based on the ablations studies provided in Section <a href="#subsec:ablation" data-reference-type="ref" data-reference="subsec:ablation">4.3</a> and Appendix <a href="#sec:ablappen" data-reference-type="ref" data-reference="sec:ablappen">10</a>.

<div id="table:algoparam" markdown="1">

<table>
<caption>Algorithm hyperparameter setup of CQL, CQL (revised), and EPQ (ours) algorithms</caption>
<thead>
<tr>
<th style="text-align: left;"><strong>Hyperparameters</strong></th>
<th style="text-align: center;"><strong>CQL</strong></th>
<th style="text-align: center;"><div id="table:algoparam">
<table>
<caption>Algorithm hyperparameter setup of CQL, CQL (revised), and EPQ (ours) algorithms</caption>
<tbody>
<tr>
<td style="text-align: center;"><strong>CQL (revised)</strong></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><strong>(for Adroit)</strong></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>
</div></th>
<th style="text-align: center;"><strong>EPQ</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Policy learning rate <span class="math inline"><em>η</em><sub><em>ϕ</em></sub></span></td>
<td style="text-align: center;">1e-4</td>
<td style="text-align: center;">1e-4</td>
<td style="text-align: center;">1e-4</td>
</tr>
<tr>
<td style="text-align: left;">Value function learning rate <span class="math inline"><em>η</em><sub><em>θ</em></sub></span></td>
<td style="text-align: center;">3e-4</td>
<td style="text-align: center;">3e-4</td>
<td style="text-align: center;">3e-4</td>
</tr>
<tr>
<td style="text-align: left;">Soft target update coefficient <span class="math inline"><em>η</em><sub><em>θ̄</em></sub></span></td>
<td style="text-align: center;">0.005</td>
<td style="text-align: center;">0.005</td>
<td style="text-align: center;">0.005</td>
</tr>
<tr>
<td style="text-align: left;">Batch size</td>
<td style="text-align: center;">256</td>
<td style="text-align: center;">256</td>
<td style="text-align: center;">256</td>
</tr>
<tr>
<td style="text-align: left;">The number of sampling <span class="math inline"><em>N</em><sub><em>a</em></sub></span></td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">10</td>
</tr>
<tr>
<td style="text-align: left;">Initial behavior cloning steps</td>
<td style="text-align: center;">10000</td>
<td style="text-align: center;">10000</td>
<td style="text-align: center;">10000</td>
</tr>
<tr>
<td style="text-align: left;">Gradient steps for training</td>
<td style="text-align: center;">3m (0.3m for Adroit)</td>
<td style="text-align: center;">0.3m</td>
<td style="text-align: center;">3m (0.3m for Adroit)</td>
</tr>
<tr>
<td style="text-align: left;">Entropy coefficient <span class="math inline"><em>η</em><sub><em>θ</em></sub></span></td>
<td style="text-align: center;">Auto</td>
<td style="text-align: center;">0.5</td>
<td style="text-align: center;">Auto (0.5 for Adroit)</td>
</tr>
<tr>
<td style="text-align: left;">Penalizing constant <span class="math inline"><em>α</em></span></td>
<td style="text-align: center;">Auto (10 for MuJoCo)</td>
<td style="text-align: center;">5 or 20</td>
<td style="text-align: center;"><div id="table:algoparam">
<table>
<caption>Algorithm hyperparameter setup of CQL, CQL (revised), and EPQ (ours) algorithms</caption>
<tbody>
<tr>
<td style="text-align: center;">20 for MuJoCo</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">5 or 20 for Adroit</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">5 or Auto for AntMaze</td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>
</div></td>
</tr>
<tr>
<td style="text-align: left;">Discount factor <span class="math inline"><em>γ</em></span></td>
<td style="text-align: center;">0.99</td>
<td style="text-align: center;">0.9 or 0.95</td>
<td style="text-align: center;">0.99 (0.9 or 0.95 for Adroit)</td>
</tr>
</tbody>
</table>

</div>

<span id="table:algoparam" label="table:algoparam"></span>

<div id="table:taskparam" markdown="1">

| **Mujoco Tasks** | **\\(\tau/\rho\\)** | **\\(c_{\min}\\)** | **\\(\epsilon\\)** | **\\(\zeta\\)** |  |
|:---|:--:|:--:|:--:|:--:|:--:|
| halfcheetah-random | 10 | 0.2 | 2 | 2 |  |
| hopper-random | 2 | 0.1 | 0.5 | 2 |  |
| walker2d-random | 1 | 0.2 | 2 | 0.5 |  |
| halfcheetah-medium | 10 | 0.2 | 0.5 | 2 |  |
| hopper-medium | 0.2 | 0.5 | 2 | 5 |  |
| walker2d-medium | 1 | 0.5 | 2 | 2 |  |
| halfcheetah-medium-expert | 1.0 | 0.2 | 0.5 | 2 |  |
| hopper-medium-expert | 1 | 0.2 | 0.5 | 2 |  |
| walker2d-medium-expert | 1.0 | 0.2 | 0.5 | 2 |  |
| halfcheetah-expert | 1 | 0.2 | 0.5 | 2 |  |
| hopper-expert | 1 | 0.2 | 0.5 | 2 |  |
| walker2d-expert | 0.5 | 0.2 | 2.0 | 2 |  |
| halfcheetah-medium-replay | 2 | 0.2 | 0.5 | 2 |  |
| hopper-medium-replay | 2 | 0.2 | 0.5 | 2 |  |
| walker2d-medium-replay | 0.2 | 0.5 | 1.0 | 2 |  |
| halfcheetah-full-replay | 1.5 | 0.2 | 0.5 | 2 |  |
| hopper-full-replay | 2.0 | 0.2 | 1.0 | 2 |  |
| walker2d-full-replay | 1.0 | 0.2 | 0.5 | 2 |  |
| **Adroit Tasks** | **\\(\tau/\rho\\)** | **\\(c_{\min}\\)** | **\\(\epsilon\\)** | **\\(\zeta\\)** |  |
| pen-human | 0.05 | 0.5 | 1.0 | 200 |  |
| door-human | 0.05 | 0.5 | 0.5 | 200 |  |
| hammer-human | 0.1 | 0.2 | 5 | 100 |  |
| relocate-human | 0.2 | 0.2 | 2 | 10 |  |
| pen-cloned | 0.2 | 0.2 | 5 | 50 |  |
| door-cloned | 0.2 | 0.5 | 1 | 10 |  |
| hammer-cloned | 0.2 | 0.2 | 5 | 100 |  |
| relocate-cloned | 0.2 | 0.2 | 5 | 10 |  |
| **AntMaze Tasks** | **\\(\tau/\rho\\)** | **\\(c_{\min}\\)** | **\\(\epsilon\\)** | **\\(\zeta\\)** |  |
| umaze | 10 | 0.2 | 2 | 2 |  |
| umaze-diverse | 10 | 0.2 | 2 | 2 |  |
| medium-play | 0.1 | 0.2 | 1 | 2 |  |
| medium-diverse | 0.1 | 0.2 | 1 | 2 |  |
| large-play | 0.1 | 0.2 | 1 | 2 |  |
| large-diverse | 0.1 | 0.2 | 1 | 2 |  |

Task hyperparameter setup for Mujoco tasks and Adroit tasks

</div>

<span id="table:taskparam" label="table:taskparam"></span>

# Additional Ablation Studies Related to \\(w_{s,a}^Q\\) Estimation [sec:ablappen]

In this section, we provide additional ablation studies related to IS weight \\(w_{s,a}^Q\\) estimation in Appendix <a href="#sec:impleappen" data-reference-type="ref" data-reference="sec:impleappen">8</a>. For analysis, Fig. <a href="#fig:ablappen" data-reference-type="ref" data-reference="fig:ablappen">8</a> shows the performance plot when the IS clipping factor \\(c_{\min}\\), the cluster radius \\(\epsilon\\), and the temperature \\(\zeta\\) change.

<figure id="fig:ablappen">

<figcaption>Additional ablation studies on Hopper-medium task</figcaption>
</figure>

**IS Clipping Factor \\(c_{\min}\\):** In the EPQ implementation, the IS clipping factor \\(c_{\min}\\) is employed to clip the IS weight \\(w_{s,a}^Q\\) to prevent the exclusion of data samples with relatively low \\(w_{s,a}^Q\\). When \\(c_{\min}=0\\), low-quality samples with low \\(w_{s,a}^Q\\) are not utilized at all based on the prioritization in Section <a href="#subsec:pd" data-reference-type="ref" data-reference="subsec:pd">3.3</a>. However, as \\(c_{\min}\\) increases, these low-quality samples are increasingly exploited. Fig. <a href="#fig:ablation" data-reference-type="ref" data-reference="fig:ablation">7</a>(c) illustrates the performance of EPQ with varying \\(c_{\min}\\), and EPQ achieves the best performance when \\(c_{\min}=0.5\\). This result suggests that it is more beneficial to use low-quality samples with proper priority rather than discarding them entirely.

**Cluster Radius \\(\epsilon\\):** As explained in Appendix <a href="#subsec:IS" data-reference-type="ref" data-reference="subsec:IS">8.4</a>, we can control the number of adjacent samples in the cluster based on the radius \\(\epsilon\\). From the results illustrated in Fig. <a href="#fig:ablappen" data-reference-type="ref" data-reference="fig:ablappen">8</a>(a), we can observe that EPQ with \\(d=2.0\\) performs best, and a decrease or an increase in \\(\epsilon\\) can significantly affect the performance indicating that \\(\epsilon\\) must be chosen properly for each task to find the cluster that contains adjacent samples appropriately. If \\(\epsilon\\) is too small, the cluster will hardly contain adjacent samples, and if \\(\epsilon\\) is too large, samples that should be distinguished will aggregate in the same cluster, adversely affecting the performance.  
**Temperature \\(\zeta\\):** As proposed in Section <a href="#subsec:pd" data-reference-type="ref" data-reference="subsec:pd">3.3</a>, samples in the dataset are prioritized according to the definition of \\(w^Q_{s,a}\\). Since the samples with higher \\(Q\\) values are more likely to be selected for the update of the \\(Q\\)-function, temperature \\(\zeta\\) controls the amount of prioritization, as explained in Appendix <a href="#subsec:IS" data-reference-type="ref" data-reference="subsec:IS">8.4</a>. Increasing \\(\zeta\\) reduces the difference in the \\(Q\\)-function between the samples, putting less emphasis on prioritization. Fig. <a href="#fig:ablappen" data-reference-type="ref" data-reference="fig:ablappen">8</a>(b) shows the performance change according to the change in \\(\zeta\\), where the results state that the performance does not heavily depend on \\(\zeta\\). From the ablation study, we can conclude that the radius \\(\epsilon\\) has a greater influence on the performance of Hopper-medium task compared to the temperature \\(\zeta\\).

# Additional Performance Comparison on Adroit Tasks [sec:pcadroit]

For adroit tasks, the performance of CQL (reprod.) is too low compared to CQL (paper) in Table <a href="#table:performance" data-reference-type="ref" data-reference="table:performance">1</a>, so we additionally provide the performance result of the revised version of CQL provided in Section <a href="#sec:hyper" data-reference-type="ref" data-reference="sec:hyper">9</a>. We also compare the performance of EPQ with the performance of CQL (revised) on various adroit tasks, and Table <a href="#table:adroitrev" data-reference-type="ref" data-reference="table:adroitrev">10</a> shows the corresponding comparison results. From the result, we can see that CQL (revised) greatly enhances the performance of CQL on adroit tasks, but EPQ still outperforms CQL (revised), which demonstrates the intact advantage of the proposed exclusive penalty and prioritized dataset well on the adroit tasks.

<div id="table:adroitrev" markdown="1">

| **Task** | **CQL (paper)** | **CQL (revised)** | **EPQ** |  |
|:---|:--:|:--:|:--:|:--:|
| pen-human | 55.8 | 82.0\\(\pm\\)`<!-- -->`{=html}6.2 | **83.9\\(\pm\\)`<!-- -->`{=html}6.8** |  |
| door-human | 9.1 | 7.8\\(\pm\\)`<!-- -->`{=html}0.5 | **13.2 \\(\pm\\) 2.4** |  |
| hammer-human | 2.1 | **6.4\\(\pm\\)`<!-- -->`{=html}5.4** | 3.9\\(\pm\\)`<!-- -->`{=html}5.0 |  |
| relocate-human | **0.4** | 0.1\\(\pm\\)`<!-- -->`{=html}0.2 | **0.3\\(\pm\\)`<!-- -->`{=html}0.2** |  |
| pen-cloned | 40.3 | **90.7\\(\pm\\)`<!-- -->`{=html}4.8** | **91.8\\(\pm\\)`<!-- -->`{=html}4.7** |  |
| door-cloned | 3.5 | 1.3\\(\pm\\)`<!-- -->`{=html}2.2 | **5.8\\(\pm\\)`<!-- -->`{=html}2.8** |  |
| hammer-cloned | 5.7 | 2.0\\(\pm\\)`<!-- -->`{=html}1.3 | **22.8\\(\pm\\)`<!-- -->`{=html}15.3** |  |
| relocate-cloned | -0.1 | 0.0\\(\pm\\)`<!-- -->`{=html}0.0 | **0.1\\(\pm\\)`<!-- -->`{=html}0.1** |  |
| **Adroit Tasks Total** | 116.8 | 190.3 | **221.8** |  |

Performance comparison of CQL (paper), CQL (revised), and EPQ (ours) on Adroit tasks.

</div>

<span id="table:adroitrev" label="table:adroitrev"></span>

# Limitations [sec:limitations]

While EPQ introduces several additional design knobs (τ, c_min, ε, ζ), Section 4.3 demonstrates that performance is remarkably insensitive to wide variations of these values on the canonical Hopper-random benchmark. Concentrating the hyper-parameter search on a single, well-understood environment greatly simplifies deployment because it eliminates the need for per-task retuning; the same configuration can be carried over verbatim to all other datasets tested in Section 4.1. Future work may further streamline the implementation, but the current set-up already attains strong, out-of-the-box results across locomotion, manipulation and navigation tasks.
# Broader Impact [sec:impact]

Nevertheless, in real-world situations, engaging with the environment can be costly. Particularly in high-risk contexts such as disaster scenarios, acquiring adequate data for learning can be quite challenging. Our research is primarily focused on offline settings and we present a novel method, EPQ, holds the potential for practical applications in real-life situations where the interaction is not available, and exhibits promise in addressing the challenges posed by offline RL algorithms. Consequently, our work carries several potential societal implications, although we believe that none require specific emphasis in this context.

# NeurIPS Paper Checklist [neurips-paper-checklist]

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: **The papers not including the checklist will be desk rejected.** The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer , , or .

- means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.

- Please provide a short (1–2 sentence) justification right after your answer (even for NA).

**The checklist answers are an integral part of your paper submission.** They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "" is generally preferable to "", it is perfectly acceptable to answer "" provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "" or "" is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

- **Delete this instruction block, but keep the section heading “NeurIPS paper checklist"**,

- **Keep the checklist subsection headings, questions/answers and guidelines below.**

- **Do not modify the questions and only use the provided macros for your answers**.

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification: The claims made in the abstract and introductions are well reflected in Section <a href="#sec:method" data-reference-type="ref" data-reference="sec:method">3</a> Methodology and Section <a href="#sec:experiment" data-reference-type="ref" data-reference="sec:experiment">4</a> Experiments.

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: The limitations are addressed in the Appendix <a href="#sec:limitations" data-reference-type="ref" data-reference="sec:limitations">12</a> Limitations.

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

14. Justification: For each theoretical result, the detailed proofs and assumptions are provided in Appendix <a href="#sec:proof" data-reference-type="ref" data-reference="sec:proof">7</a> Proof and Appendix <a href="#sec:impleappen" data-reference-type="ref" data-reference="sec:impleappen">8</a> Implementation Details.

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

19. Justification: The specific environment descriptions and experimental setups including the hyperparameters can be found in Section <a href="#sec:experiment" data-reference-type="ref" data-reference="sec:experiment">4</a> Experiments and Appendix <a href="#sec:hyper" data-reference-type="ref" data-reference="sec:hyper">9</a> Hyperparameter Setup.

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

24. Justification: The data and code for reproducing the main experimental results are included in supplemental materials.

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

29. Justification: The specific experimental setups including the hyperparameters can be found in Section <a href="#sec:experiment" data-reference-type="ref" data-reference="sec:experiment">4</a> Experiments and Appendix <a href="#sec:hyper" data-reference-type="ref" data-reference="sec:hyper">9</a> Hyperparameter Setup.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: The graphs included in the paper such as Figure <a href="#fig:bias" data-reference-type="ref" data-reference="fig:bias">6</a> and Figure <a href="#fig:ablation" data-reference-type="ref" data-reference="fig:ablation">7</a> in Section <a href="#sec:experiment" data-reference-type="ref" data-reference="sec:experiment">4</a> Experiments well demonstrate the statistical significance of the experiment.

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

39. Justification: The information on computation resources are provided in Appendix <a href="#sec:impleappen" data-reference-type="ref" data-reference="sec:impleappen">8</a> Implementation Details.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification: The research conducted in the paper conforms the NeurlIPS Code of Ethics

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: The societal impacts of the proposed paper is included in appendix <a href="#sec:impact" data-reference-type="ref" data-reference="sec:impact">13</a> Broader Impacts section.

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

54. Justification: The proposed paper does not pose such risks.

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification: The baseline code and experimental data are cited both in-text and in the References section.

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

64. Justification: The proposed paper does not release new assets.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification: The proposed paper does not involve crowdsourcing nor research with human subjects.

70. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

73. Answer:

74. Justification: The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

75. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

[^1]: \\(*\\) indicates equal contribution and \\(\dagger\\) indicates the corresponding author: Seungyul Han.

[^2]: Special thanks to Whiyoung Jung from LG AI Research for providing experimental data used in this work.
