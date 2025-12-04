# Revisiting Plasticity in Visual Reinforcement Learning: Data, Modules and Training Stages

## Abstract

Plasticity, the ability of a neural network to evolve with new data, is crucial for high-performance and sample-efficient visual reinforcement learning (VRL). Although methods like resetting and regularization can potentially mitigate plasticity loss, the influences of various components within the VRL framework on the agent’s plasticity are still poorly understood. In this work, we conduct a systematic empirical exploration focusing on three primary underexplored facets and derive the following insightful conclusions: (1) data augmentation is essential in maintaining plasticity; (2) the critic’s plasticity loss serves as the principal bottleneck impeding efficient training; and (3) without timely intervention to recover critic’s plasticity in the early stages, its loss becomes catastrophic. These insights suggest a novel strategy to address the high replay ratio (RR) dilemma, where exacerbated plasticity loss hinders the potential improvements of sample efficiency brought by increased reuse frequency. Rather than setting a static RR for the entire training process, we propose *Adaptive RR*, which dynamically adjusts the RR based on the critic’s plasticity level. Extensive evaluations indicate that *Adaptive RR* not only avoids catastrophic plasticity loss in the early stages but also benefits from more frequent reuse in later phases, resulting in superior sample efficiency.

# **Introduction**

The potent capabilities of deep neural networks have driven the brilliant triumph of deep reinforcement learning (DRL) across diverse domains . Nevertheless, recent studies highlight a pronounced limitation of neural networks: they struggle to maintain adaptability and learning from new data after training on a non-stationary objective , a challenge known as ***plasticity loss*** . Since RL agents must continuously adapt their policies through interacting with environment, non-stationary data streams and optimization objectives are inherently embedded within the DRL paradigm . Consequently, plasticity loss presents a fundamental challenge for achieving sample-efficient DRL applications .

Although several strategies have been proposed to address this concern, previous studies primarily focused on mitigating plasticity loss through methods such as resetting the parameters of neurons , incorporating regularization techniques  and adjusting network architecture . The nuanced impacts of various dimensions within the DRL framework on plasticity remain underexplored. This knowledge gap hinders more precise interventions to better preserve plasticity. To this end, this paper delves into the nuanced mechanisms underlying DRL’s plasticity loss from three primary yet underexplored perspectives: data, agent modules, and training stages. Our investigations focus on visual RL (VRL) tasks that enable decision-making directly from high-dimensional observations. As a representative paradigm of end-to-end DRL, VRL is inherently more challenging than learning from handcrafted state inputs, leading to its notorious sample inefficiency .

We begin by revealing the indispensable role of data augmentation (DA) in mitigating plasticity loss for off-policy VRL algorithms. Although DA is extensively employed to enhance VRL’s sample efficiency , its foundational mechanism is still largely elusive. Our investigation employs a factorial experiment with DA and Reset. The latter refers to the re-initialization of subsets of neurons and has been shown to be a direct and effective method for mitigating plasticity loss . However, our investigation has surprisingly revealed two notable findings: (1) Reset can significantly enhance performance in the absence of DA, but show limited or even negative effects when DA is applied. This suggests a significant plasticity loss when DA is not employed, contrasted with minimal or no plasticity loss when DA is utilized. (2) Performance with DA alone surpasses that of reset or other interventions without employing DA, highlighting the pivotal role of DA in mitigating plasticity loss. Furthermore, the pronounced difference in plasticity due to DA’s presence or absence provides compelling cases for comparison, allowing a deeper investigation into the differences and developments of plasticity across different modules and stages.

We then dissect VRL agents into three core modules: the encoder, actor, and critic, aiming to identify which components suffer most from plasticity loss and contribute to the sample inefficiency of VRL training. Previous studies commonly attribute the inefficiency of VRL training to the challenges of constructing a compact representation from high-dimensional observations . A natural corollary to this would be the rapid loss of plasticity in the encoder when learning from scratch solely based on reward signals, leading to sample inefficiency. However, our comprehensive experiments reveal that it is, in fact, the plasticity loss of the critic module that presents the critical bottleneck for training. This insight aligns with recent empirical studies showing that efforts to enhance the representation of VRL agents, such as meticulously crafting self-supervised learning tasks and pre-training encoders with extra data, fail to achieve higher sample efficiency than simply applying DA alone . Tailored interventions to maintain the plasticity of critic module provide a promising path for achieving sample-efficient VRL in future studies.

Given the strong correlation between the critic’s plasticity and training efficiency, we note that the primary contribution of DA lies in facilitating the early-stage recovery of plasticity within the critic module. Subsequently, we conduct a comparative experiment by turning on or turning off DA at certain training steps and obtain two insightful findings: (1) Once the critic’s plasticity has been recovered to an adequate level in the early stage, there’s no need for specific interventions to maintain it. (2) Without timely intervention in the early stage, the critic’s plasticity loss becomes catastrophic and irrecoverable. These findings underscore the importance of preserving critic plasticity during the initial phases of training. Conversely, plasticity loss in the later stages is not a critical concern. To conclude, the main takeaways from our revisiting can be summarized as follows:

- DA is indispensable for preserving the plasticity of VRL agents. (Section <a href="#Sec: Data" data-reference-type="ref" data-reference="Sec: Data">3</a>)

- Critic’s plasticity loss is a critical bottleneck affecting the training efficiency. (Section <a href="#Sec: Modules" data-reference-type="ref" data-reference="Sec: Modules">4</a>)

- Maintaining plasticity in the early stages is crucial to prevent irrecoverable loss. (Section <a href="#Sec: Stages" data-reference-type="ref" data-reference="Sec: Stages">5</a>)

We conclude by addressing a longstanding question in VRL: how to determine the appropriate replay ratio (RR), defined as the number of gradient updates per environment step, to achieve optimal sample efficiency . Prior research set a static RR for the entire training process, facing a dilemma: while increasing the RR of off-policy algorithms should enhance sample efficiency, this improvement is offset by the exacerbated plasticity loss . However, aforementioned analysis indicates that the impact of plasticity loss varies throughout training stages, advocating for an adaptive adjustment of RR based on the stage, rather than setting a static value. Concurrently, the critic’s plasticity has been identified as the primary factor affecting sample efficiency, suggesting its level as a criterion for RR adjustment. Drawing upon these insights, we introduce a simple and effective method termed *Adaptive RR* that dynamically adjusts the RR according to the critic’s plasticity level. Specifically, *Adaptive RR* commences with a lower RR during the initial training phases and elevates it upon observing significant recovery in the critic’s plasticity. Through this approach, we effectively harness the sample efficiency benefits of a high RR, while skillfully circumventing the detrimental effects of escalated plasticity loss. Our comprehensive evaluations on the DeepMind Control suite  demonstrate that *Adaptive RR* attains superior sample efficiency compared to static RR baselines.

# **Related work**

In this section, we briefly review prior research works on identifying and mitigating the issue of plasticity loss, as well as on the high RR dilemma that persistently plagues off-policy RL algorithms for more efficient applications. Further discussions on related studies can be found in <span style="color: mylinkcolor">Appendix</span> <a href="#Appendix: Extended Related Work" data-reference-type="ref" data-reference="Appendix: Extended Related Work">8</a>.

**Plasticity Loss.**   Recent studies have increasingly highlighted a major limitation in neural networks where their learning capabilities suffer catastrophic degradation after training on non-stationary objectives . Different from supervised learning, the non-stationarity of data streams and optimization objectives is inherent in the RL paradigm, necessitating the confrontation of this issues, which has been recognized by several terms, including primacy bias , dormant neuron phenomenon , implicit under-parameterization , capacity loss , and more broadly, plasticity loss . Agents lacking plasticity struggle to learn from new experiences, leading to extreme sample inefficiency or even entirely ineffective training.

The most straightforward strategy to tackle this problem is to re-initialize a part of the network to regain rejuvenated plasticity . However, periodic *Reset*  may cause sudden performance drops, impacting exploration and requiring extensive gradient updates to recover. To circumvent this drawback, *ReDo*  selectively resets the dormant neurons, while *Plasticity Injection*  introduces a new initialized network for learning and freezes the current one as residual blocks. Another line of research emphasizes incorporating explicit regularization or altering the network architecture to mitigate plasticity loss. For example,  introduces *L2-Init* to regularize the network’s weights back to their initial parameters, while  employs *Concatenated ReLU*  to guarantee a non-zero gradient. Although existing methods have made progress in mitigating plasticity loss, the intricate effects of various dimensions in the DRL framework on plasticity are poorly understood. In this paper, we aim to further explore the roles of data, modules, and training stages to provide a comprehensive insight into plasticity.

**High RR Dilemma.**   Experience replay, central to off-policy DRL algorithms, greatly improves the sample efficiency by allowing multiple reuses of data for training rather than immediate discarding after collection . Given the trial-and-error nature of DRL, agents alternate between interacting with the environment to collect new experiences and updating parameters based on transitions sampled from the replay buffer. The number of agent updates per environment step is usually called replay ratio (RR)  or update-to-data (UTD) ratio . While it’s intuitive to increase the RR as a strategy to improve sample efficiency, doing so naively can lead to adverse effects . Recent studies have increasingly recognized plasticity loss as the primary culprit behind the high RR dilemma . Within a non-stationary objective, an increased update frequency results in more severe plasticity loss. Currently, the most effective method to tackle this dilemma is to continually reset the agent’s parameters when setting a high RR value . Our investigation offers a novel perspective on addressing this long-standing issue. Firstly, we identify that the impact of plasticity loss varies across training stages, implying a need for dynamic RR adjustment. Concurrently, we determine the critic’s plasticity as crucial for model capability, proposing its level as a basis for RR adjustment. Drawing from these insights, we introduce *Adaptive RR*, a universal method that both mitigates early catastrophic plasticity loss and harnesses the potential of high RR in improving sample efficiency.

# **Data:** data augmentation is essential in maintaining plasticity

In this section, we conduct a factorial analysis of DA and Reset, illustrating that DA effectively maintains plasticity. Furthermore, in comparison with other architectural and optimization interventions, we highlight DA’s pivotal role as a data-centric method in addressing VRL’s plasticity loss.

**A Factorial Examination of DA and Reset.**   DA has become an indispensable component in achieving sample-efficient VRL applications . As illustrated by the blue and orange dashed lines in Figure <a href="#Fig:Reset" data-reference-type="ref" data-reference="Fig:Reset">1</a>, employing a simple DA approach to the input observations can lead to significant performance improvements in previously unsuccessful algorithms. However, the mechanisms driving DA’s notable effectiveness remain largely unclear . On the other hand, recent studies have increasingly recognized that plasticity loss during training significantly hampers sample efficiency . This naturally raises the question: *does the remarkable efficacy of DA stem from its capacity to maintain plasticity?* To address this query, we undertake a factorial examination of DA and Reset. Given that Reset is well-recognized for its capability to mitigate the detrimental effects of plasticity loss on training, it can not only act as a diagnostic tool to assess the extent of plasticity loss in the presence or absence of DA, but also provide a benchmark to determine the DA’s effectiveness in preserving plasticity.

<figure id="Fig:Reset">
<img src="./figures/reset_DA_Run.png"" />
<figcaption>Training curves across four combinations: incorporating or excluding Reset and DA. We adopt DrQ-v2 <span class="citation" data-cites="DrQ-v2"></span> as our baseline algorithm and follow the Reset settings from <span class="citation" data-cites="primacy_bias"></span>. Mean and std are estimated over 5 runs. Note that re-initializing 10 times in the Quadruped Run task resulted in poor performance, prompting us to adjust the reset times to 5. For ablation studies on reset times and results in other tasks, please refer to <span style="color: mylinkcolor">Appendix</span> <a href="#Appendix: Reset" data-reference-type="ref" data-reference="Appendix: Reset">9.1</a>. </figcaption>
</figure>

The results presented in Figure <a href="#Fig:Reset" data-reference-type="ref" data-reference="Fig:Reset">1</a> highlight three distinct phenomena: <span style="color: mydarkgreen">$`\bullet`$</span> In the absence of DA, the implementation of Reset consistently yields marked enhancements. This underscores the evident plasticity loss when training is conducted devoid of DA. <span style="color: mydarkgreen">$`\bullet`$</span> With the integration of DA, the introduction of Reset leads to only slight improvements, or occasionally, a decrease. This indicates that applying DA alone can sufficiently preserve the agent’s plasticity, leaving little to no room for significant improvement. <span style="color: mydarkgreen">$`\bullet`$</span> Comparatively, the performance of Reset without DA lags behind that achieved employing DA alone, underscoring the potent effectiveness of DA in preserving plasticity.

<figure id="fig:reg">
<img src="./figures/reg_CR.png"" style="width:41.0%" />
<figcaption>Performance of various interventions in Cheetah Run across 5 seeds.</figcaption>
</figure>

**Comparing DA with Other Interventions.** We assess the influence of various architectural and optimization interventions on DMC using the DrQ-v2 framework. Specifically, we implement the following techniques: <span style="color: mydarkgreen">$`\bullet`$</span> ***Weight Decay***, where we set the L2 coefficient to $`10^{-5}`$. <span style="color: mydarkgreen">$`\bullet`$</span> ***L2 Init*** : This technique integrates L2 regularization aimed at the initial parameters into the loss function. Specifically, we apply it to the critic loss with a coefficient set to $`10^{-2}`$. <span style="color: mydarkgreen">$`\bullet`$</span> ***Layer Normalization*** after each convolutional and linear layer. <span style="color: mydarkgreen">$`\bullet`$</span> ***Spectral Normalization*** after the initial linear layer for both the actor and critic networks. <span style="color: mydarkgreen">$`\bullet`$</span> ***Shrink and Perturb***: This involves multiplying the critic network weights by a small scalar and adding a perturbation equivalent to the weights of a randomly initialized network. <span style="color: mydarkgreen">$`\bullet`$</span> Adoption of ***CReLU***  as an alternative to ReLU in the critic network. We present the final performance of different interventions in Figure <a href="#fig:reg" data-reference-type="ref" data-reference="fig:reg">2</a>, which indicates that DA is the most effective method. For further comparison of interventions, see <span style="color: mylinkcolor">Appendix</span> <a href="#Appendix: Further_Comparisons" data-reference-type="ref" data-reference="Appendix: Further_Comparisons">9.3</a>.

# **Modules:** the plasticity loss of critic network is predominant

In this section, we aim to investigate ***which module(s) of VRL agents suffer the most severe plasticity loss, and thus, are detrimental to efficient training***. Initially, by observing the differential trends in plasticity levels across modules with and without DA, we preliminarily pinpoint the critic’s plasticity loss as the pivotal factor influencing training. Subsequently, the decisive role of DA when using a frozen pre-trained encoder attests that encoder’s plasticity loss isn’t the primary bottleneck for sample inefficiency. Finally, contrastive experiments with plasticity injection on actor and critic further corroborate that the critic’s plasticity loss is the main culprit.

**Fraction of Active Units (FAU).**   Although the complete mechanisms underlying plasticity loss remain unclear, a reduction in the number of active units within the network has been identified as a principal factor contributing to this deterioration . Hence, the Fraction of Active Units (FAU) is widely used as a metric for measuring plasticity. Specifically, the FAU for neurons located in module $`\mathcal{M}`$, denoted as $`\Phi_\mathcal{M}`$, is formally defined as:
``` math
\begin{aligned}
    \label{eqn:FAU}
    \Phi_\mathcal{M} = \frac{\sum_{n\in \mathcal{M}} \mathbf{1}(a_n(x) > 0)}{N},
\end{aligned}
```
where $`a_n(x)`$ represent the activation of neuron $`n`$ given the input $`x`$, and $`N`$ is the total number of neurons within module $`\mathcal{M}`$. More discussion on plasticity measurement can be found in <span style="color: mylinkcolor">Appendix</span> <a href="#Appendix: Measurement Metrics of Plasticity" data-reference-type="ref" data-reference="Appendix: Measurement Metrics of Plasticity">[Appendix: Measurement Metrics of Plasticity]</a>.

**Different FAU trends across modules reveal critic’s plasticity loss as a hurdle for VRL training.** Within FAU as metric, we proceed to assess the plasticity disparities in the encoder, actor, and critic modules with and without DA. We adopt the experimental setup from , where the encoder is updated only based on the critic loss. As shown in Figure <a href="#Fig:FAU" data-reference-type="ref" data-reference="Fig:FAU">3</a> <span style="color: mylinkcolor">(left)</span>, the integration of DA leads to a substantial leap in training performance. Consistent with this uptrend in performance, DA elevates the critic’s FAU to a level almost equivalent to an initialized network. In contrast, both the encoder and actor’s FAU exhibit similar trends regardless of DA’s presence or absence. This finding tentatively suggests that critic’s plasticity loss is the bottleneck constraining training efficiency.

<figure id="Fig:FAU">
<img src="./figures/FAU_quadruped_run.png"" />
<figcaption>Different FAU trends across modules throughout training. The plasticity of encoder and actor displays similar trends whether DA is employed or not. Conversely, integrating DA leads to a marked improvement in the critic’s plasticity. Further comparative results are in <span style="color: mylinkcolor">Appendix</span> <a href="#Appendix: FAU trends" data-reference-type="ref" data-reference="Appendix: FAU trends">9.6</a>.</figcaption>
</figure>

**Is the sample inefficiency in VRL truly blamed on poor representation?**   Since VRL handles high-dimensional image observations rather than well-structured states, prior studies commonly attribute VRL’s sample inefficiency to its inherent challenge of learning a compact representation.

<figure id="fig:pretrain">
<img src="./figures/pre-trained_encoder.png"" style="width:54.0%" />
<figcaption>Learning curves of DrQ-v2 using a frozen ImageNet pre-trained encoder, with and without DA.</figcaption>
</figure>

We contest this assumption by conducting a simple experiment. Instead of training the encoder from scratch, we employ an ImageNet pre-trained ResNet model as the agent’s encoder and retain its parameters frozen throughout the training process. The specific implementation adheres , but employs the DA operation as in DrQ-v2. Building on this setup, we compare the effects of employing DA against not using it on sample efficiency, thereby isolating and negating the potential influences from disparities in the encoder’s representation capability on training. As depicted in Figure <a href="#fig:pretrain" data-reference-type="ref" data-reference="fig:pretrain">4</a>, the results illustrate that employing DA consistently surpasses scenarios without DA by a notable margin. This significant gap sample inefficiency in VRL cannot be predominantly attributed to poor representation. This pronounced disparity underscores two critical insights: first, the pivotal effectiveness of DA is not centered on enhancing representation; and second, sample inefficiency in VRL cannot be primarily ascribed to the insufficient representation.

**Plasticity Injection on Actor and Critic as a Diagnostic Tool.**   Having ruled out the encoder’s influence, we next explore how the plasticity loss of actor and critic impact VRL’s training efficiency.

<figure id="fig:injection">
<img src="./figures/Injection_WR.png"" style="width:50.0%" />
<figcaption>Training curves of employing plasticity injection (PI) on actor or critic. For all cases, DA is applied after the given environment steps.</figcaption>
</figure>

To achieve this, we introduce plasticity injection as a diagnostic tool . Unlike Reset, which leads to a periodic momentary decrease in performance and induces an exploration effect , plasticity injection restores the module’s plasticity to its initial level without altering other characteristics or compromising previously learned knowledge. Therefore, plasticity injection allows us to investigate in isolation the impact on training after reintroducing sufficient plasticity to a certain module. Should the training performance exhibit a marked enhancement relative to the baseline following plasticity injection into a particular module, it would suggest that this module had previously undergone catastrophic plasticity loss, thereby compromising the training efficacy. We apply plasticity injection separately to the actor and critic when using and not using DA. The results illustrated in Figure <a href="#fig:injection" data-reference-type="ref" data-reference="fig:injection">5</a> and <span style="color: mylinkcolor">Appendix</span> <a href="#Appendix: Plasticity Injection" data-reference-type="ref" data-reference="Appendix: Plasticity Injection">9.4</a> reveal the subsequent findings and insights: <span style="color: mydarkgreen">$`\bullet`$</span> When employing DA, the application of plasticity injection to both the actor and critic does not modify the training performance. This suggests that DA alone is sufficient to maintain plasticity within the Walker Run task. <span style="color: mydarkgreen">$`\bullet`$</span> Without using DA in the initial 1M steps, administering plasticity injection to the critic resulted in a significant performance improvement. This fully demonstrates that the critic’s plasticity loss is the primary culprit behind VRL’s sample inefficiency.

# **Stages:** early-stage plasticity loss becomes irrecoverable

In this section, we elucidate the differential attributes of plasticity loss throughout various training stages. Upon confirming that the critic’s plasticity loss is central to hampering training efficiency, a closer review of the results in Figure <a href="#Fig:FAU" data-reference-type="ref" data-reference="Fig:FAU">3</a> underscores that DA’s predominant contribution is to effectively recover the critic’s plasticity during initial phases. This naturally raises two questions: <span style="color: myorange">$`\bullet`$</span> After recovering the critic’s plasticity to an adequate level in the early stage, will ceasing interventions to maintain plasticity detrimentally affect training? <span style="color: myblue">$`\bullet`$</span> If interventions aren’t applied early to recover the critic’s plasticity, is it still feasible to enhance training performance later through such measures? To address these two questions, we conduct a comparative experiment by turning on or turning off DA at certain training steps and obtain the following findings: <span style="color: myorange">$`\bullet`$ Turning off DA</span> after the critic’s plasticity has been recovered does not affect training efficiency. This suggests that it is not necessary to employ specific interventions to maintain plasticity in the later stages of training. <span style="color: myblue">$`\bullet`$ Turning on DA</span> when plasticity has already been significantly lost and without timely intervention in the early stages cannot revive the agent’s training performance. This observation underscores the vital importance of maintaining plasticity in the early stages; otherwise, the loss becomes irrecoverable.

<figure id="Fig:Turn DA">
<img src="./figures/recover_1M.png"" />
<figcaption>Training curves for various DA application modes. The red dashed line shows when DA is <span style="color: myblue">turned on</span> or <span style="color: myorange">turned off</span>. Additional comparative results can be found in <span style="color: mylinkcolor">Appendix</span> <a href="#Appendix: Turning DA" data-reference-type="ref" data-reference="Appendix: Turning DA">9.5</a>.</figcaption>
</figure>

We attribute these differences across stages to the nature of online RL to learn from scratch in a bootstrapped fashion. During the initial phases of training, bootstrapped target derived from low-quality and limited-quantity experiences exhibits high non-stationarity and deviates significantly from the actual state-action values . The severe non-stationarity of targets induces a rapid decline in the critic’s plasticity , consistent with the findings in Figure <a href="#Fig:FAU" data-reference-type="ref" data-reference="Fig:FAU">3</a>. Having lost the ability to learn from newly collected data, the critic will perpetually fail to capture the dynamics of the environment, preventing the agent from acquiring an effective policy. This leads to ***catastrophic plasticity loss*** in the early stages. Conversely, although the critic’s plasticity experiences a gradual decline after recovery, this can be viewed as a process of progressively approximating the optimal value function for the current task. For single-task VRL that doesn’t require the agent to retain continuous learning capabilities, this is termed as a ***benign plasticity loss***. Differences across stages offer a new perspective to address VRL’s plasticity loss challenges.

# **Methods:** adaptive RR for addressing the high RR dilemma

Drawing upon the refined understanding of plasticity loss, this section introduces *Adaptive RR* to tackle the high RR dilemma in VRL. Extensive evaluations demonstrate that *Adaptive RR* strikes a superior trade-off between reuse frequency and plasticity loss, thereby improving sample efficiency.

**High RR Dilemma.**   Increasing the replay ratio (RR), which denotes the number of updates per environment interaction, is an intuitive strategy to further harness the strengths of off-policy algorithms to improve sample efficiency. However, recent studies  and the results in Figure <a href="#fig:high RR" data-reference-type="ref" data-reference="fig:high RR">7</a> consistently reveal that adopting a higher static RR impairs training efficiency.

<figure id="fig:high RR">
<img src="./figures/High_RR.png"" />
<figcaption>Training curves across varying RR values. Despite its intent to enhance sample efficiency through more frequent updates, an increasing RR value actually undermines training.</figcaption>
</figure>

<figure id="fig:FAU_High_RR">
<img src="./figures/FAU_High_RR_QR.png"" style="width:40.0%" />
<figcaption>The FAU of critic across varying RR values. A larger RR value leads to more severe plasticity loss.</figcaption>
</figure>

The fundamental mechanism behind this counterintuitive failure of high RR is widely recognized as the intensified plasticity loss . As illustrated in Figure <a href="#fig:FAU_High_RR" data-reference-type="ref" data-reference="fig:FAU_High_RR">8</a>, increasing RR results in a progressively exacerbated plasticity loss during the early stages of training. Increasing RR from $`0.5`$ to $`1`$ notably diminishes early-stage plasticity, but the heightened reuse frequency compensates, resulting in a marginal boost in sample efficiency. However, as RR continues to rise, the detrimental effects of plasticity loss become predominant, leading to a consistent decline in sample efficiency. When RR increases to $`4`$, even with the intervention of DA, there’s no discernible recovery of the critic’s plasticity in the early stages, culminating in a catastrophic loss. An evident high RR dilemma quandary arises: while higher reuse frequency holds potential for improving sample efficiency, the exacerbated plasticity loss hinders this improvement.

**Can we adapt RR instead of setting a static value?**   Previous studies addressing the high RR dilemma typically implement interventions to mitigate plasticity loss while maintaining a consistently high RR value throughout training . Drawing inspiration from the insights in Section <a href="#Sec: Stages" data-reference-type="ref" data-reference="Sec: Stages">5</a>, which highlight the varying plasticity loss characteristics across different training stages, an orthogonal approach emerges: ***why not dynamically adjust the RR value based on the current stage?*** Initially, a low RR is adopted to prevent catastrophic plasticity loss. In later training stages, RR can be raised to boost reuse frequency, as the plasticity dynamics become benign. This balance allows us to sidestep early high RR drawbacks and later harness the enhanced sample efficiency from greater reuse frequency. Furthermore, the observations in Section <a href="#Sec: Modules" data-reference-type="ref" data-reference="Sec: Modules">4</a> have confirmed that the critic’s plasticity, as measured by its FAU, is the primary factor influencing sample efficiency. This implies that the FAU of critic module can be employed adaptively to identify the current training stage. Once the critic’s FAU has recovered to a satisfactory level, it indicates the agent has moved beyond the early training phase prone to catastrophic plasticity loss, allowing for an increase in the RR value. Based on these findings and considerations, we propose our method, *Adaptive RR*.

<div class="tcolorbox">

<span style="color: C1!25!black">*Adaptive RR* adjusts the ratio according to the current plasticity level of critic, utilizing a low RR in the early stage and transitioning it to a high value after the plasticity recovery stages.</span>

</div>

ALGORITHM BLOCK (caption below)

**Evaluation on DeepMind Control Suite.**   We then evaluate the effectiveness of <span style="color: AARed">*Adaptive RR*</span> [^2] in improving the sample efficiency of VRL algorithms. Our experiments are conducted on six challenging continuous DMC tasks, which are widely perceived to significantly suffer from plasticity loss . We select two static RR values as baselines: <span style="color: AAGray">$`\bullet`$ Low RR=$`0.5`$</span>, which exhibits no severe plasticity loss but has room for improvement in its reuse frequency. <span style="color: AABlue">$`\bullet`$ High RR=$`2`$</span>, which shows evident plasticity loss, thereby damaging sample efficiency. Our method, <span style="color: AARed">*Adaptive RR*</span>, starts with RR=$`0.5`$ during the initial phase of training, aligning with the default setting of DrQ-v2. Subsequently, we monitor the FAU of the critic module every $`50`$ episodes, *i.e.*, $`5 \times 10^4`$ environment steps. When the FAU difference between consecutive checkpoints drops below a minimal threshold (set at $`0.001`$ in our experiments), marking the end of the early stage, we adjust the RR to $`2`$. Figure <a href="#fig:arr_main_result" data-reference-type="ref" data-reference="fig:arr_main_result">9</a> illustrates the comparative performances. <span style="color: AARed">*Adaptive RR*</span> consistently demonstrates superior sample efficiency compared to a static RR throughout training.

<figure id="fig:arr_main_result">
<img src="./figures/ARR_Results.png"" />
<figcaption>Training curves of various RR settings across 6 challenging DMC tasks. <span style="color: AARed"><strong><em>Adaptive RR</em></strong></span> demonstrates superior sample efficiency compared to both static <span style="color: AAGray">low RR</span> and <span style="color: AABlue">high RR</span> value.</figcaption>
</figure>

<figure id="figure:ARR_FAU">
<img src="./figures/FAU_ARR_QR.png"" />
<figcaption>Evolution of FAU across the three modules in the Quadruped Run task under different RR configurations. The critic’s plasticity is most influenced by different RR settings. Under <span style="color: AABlue">high RR</span>, the critic’s plasticity struggles to recover in early training. In contrast, <span style="color: AARed"><em>Adaptive RR</em></span> successfully mitigates catastrophic plasticity loss in the early phases, yielding the optimal sample efficiency.</figcaption>
</figure>

Through a case study on Quadruped Run, we delve into the underlying mechanism of <span style="color: AARed">*Adaptive RR*</span>, as illustrated in Figure <a href="#figure:ARR_FAU" data-reference-type="ref" data-reference="figure:ARR_FAU">10</a>. Due to the initial RR set at $`0.5`$, the critic’s plasticity recovers promptly to a considerable level in the early stages of training, preventing catastrophic plasticity loss as seen with RR=$`2`$. After the recovery phases, <span style="color: AARed">*Adaptive RR*</span> detects a slow change in the critic’s FAU, indicating it’s nearing a peak, then switch the RR to a higher value. Within our experiments, the switch times for the five seeds occurred at: $`0.9`$M, $`1.2`$M, $`1.1`$M, $`0.95`$M, and $`0.55`$M. After switching to RR=$`2`$, the increased update frequency results in higher sample efficiency and faster convergence rates. Even though the critic’s FAU experiences a more rapid decline at this stage, this benign loss doesn’t damage training. Hence, <span style="color: AARed">*Adaptive RR*</span> can effectively exploits the sample efficiency advantages of a high RR, while adeptly avoiding the severe consequences of increased plasticity loss.

<div id="table:redo">

<table>
<caption>Comparison of Adaptive RR versus static RR with Reset and ReDo implementations. The average episode returns are averaged over 5 seeds after training for 2M environment steps.</caption>
<thead>
<tr>
<th style="text-align: left;">Average Episode Return</th>
<th colspan="3" style="text-align: center;">RR=0.5</th>
<th colspan="3" style="text-align: center;">RR=2</th>
<th style="text-align: center;">Adaptive RR</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span>2-4</span> (lr)<span>5-7</span> (lr)<span>8-8</span> (After 2M Env Steps)</td>
<td style="text-align: center;"><span><code>default</code></span></td>
<td style="text-align: center;"><span><code>Reset</code></span></td>
<td style="text-align: center;"><span><code>ReDo</code></span></td>
<td style="text-align: center;"><span><code>default</code></span></td>
<td style="text-align: center;"><span><code>Reset</code></span></td>
<td style="text-align: center;"><span><code>ReDo</code></span></td>
<td style="text-align: center;"><span><span><code>RR:0.5to2</code></span></span></td>
</tr>
<tr>
<td style="text-align: left;">Cheetah Run</td>
<td style="text-align: center;"><span class="math inline">$828 \pm 59\hphantom{0}$</span></td>
<td style="text-align: center;"><span class="math inline">$799 \pm 26\hphantom{0}$</span></td>
<td style="text-align: center;"><span class="math inline">$788 \pm 5\hphantom{00}$</span></td>
<td style="text-align: center;"><span class="math inline">$793 \pm 9\hphantom{00}$</span></td>
<td style="text-align: center;"><span class="math inline"><strong>885</strong> <strong>±</strong> <strong>20</strong></span></td>
<td style="text-align: center;"><span class="math inline">873 ± 19</span></td>
<td style="text-align: center;"><span class="math inline">880 ± 45</span></td>
</tr>
<tr>
<td style="text-align: left;">Walker Run</td>
<td style="text-align: center;"><span class="math inline">$710 \pm 39\hphantom{0}$</span></td>
<td style="text-align: center;"><span class="math inline">648 ± 107</span></td>
<td style="text-align: center;"><span class="math inline">$618 \pm 50\hphantom{0}$</span></td>
<td style="text-align: center;"><span class="math inline">$709 \pm 7\hphantom{00}$</span></td>
<td style="text-align: center;"><span class="math inline">749 ± 10</span></td>
<td style="text-align: center;"><span class="math inline">734 ± 16</span></td>
<td style="text-align: center;"><span class="math inline"><strong>758</strong> <strong>±</strong> <strong>12</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Quadruped Run</td>
<td style="text-align: center;"><span class="math inline">579 ± 120</span></td>
<td style="text-align: center;"><span class="math inline">593 ± 129</span></td>
<td style="text-align: center;"><span class="math inline">371 ± 158</span></td>
<td style="text-align: center;"><span class="math inline">417 ± 110</span></td>
<td style="text-align: center;"><span class="math inline">511 ± 47</span></td>
<td style="text-align: center;"><span class="math inline">608 ± 53</span></td>
<td style="text-align: center;"><span class="math inline"><strong>784</strong> <strong>±</strong> <strong>53</strong></span></td>
</tr>
</tbody>
</table>

</div>

<span id="table:redo" label="table:redo"></span>

We further compare the performance of Adaptive RR with that of employing Reset  and ReDo  under static RR conditions. Although Reset and ReDo both effectively mitigate plasticity loss in high RR scenarios, our method significantly outperforms these two approaches, as demonstrated in Table <a href="#table:redo" data-reference-type="ref" data-reference="table:redo">1</a> This not only showcases that Adaptive RR can secure a superior balance between reuse frequency and plasticity loss but also illuminates the promise of dynamically modulating RR in accordance with the critic’s overall plasticity level as an effective strategy, alongside neuron-level network parameter resetting, to mitigate plasticity loss.

<figure id="table:atari-short">
<table>
<thead>
<tr>
<th style="text-align: left;"><em>Metrics</em></th>
<th colspan="3" style="text-align: center;">DrQ(<span class="math inline"><em>ϵ</em></span>)</th>
<th style="text-align: center;"><span>ReDo</span></th>
<th style="text-align: center;"><span>Adaptive RR</span></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span>2-4</span> (lr)<span>5-5</span> (lr)<span>6-6</span></td>
<td style="text-align: center;"><span><span><code>RR=0.5</code></span></span></td>
<td style="text-align: center;"><span><span><code>RR=1</code></span></span></td>
<td style="text-align: center;"><span><span><code>RR=2</code></span></span></td>
<td style="text-align: center;"><span><span><code>RR=1</code></span></span></td>
<td style="text-align: center;"><span><span><code>RR:0.5to2</code></span></span></td>
</tr>
<tr>
<td style="text-align: left;"><em>Mean HNS (<span class="math inline">%</span>)</em></td>
<td style="text-align: center;"><span class="math inline">42.3</span></td>
<td style="text-align: center;"><span class="math inline">41.3</span></td>
<td style="text-align: center;"><span class="math inline">35.1</span></td>
<td style="text-align: center;"><span class="math inline">42.3</span></td>
<td style="text-align: center;"><span class="math inline"><strong>55.8</strong></span></td>
</tr>
<tr>
<td style="text-align: left;"><em>Median HNS (<span class="math inline">%</span>)</em></td>
<td style="text-align: center;"><span class="math inline">22.6</span></td>
<td style="text-align: center;"><span class="math inline">30.3</span></td>
<td style="text-align: center;"><span class="math inline">26.0</span></td>
<td style="text-align: center;"><span class="math inline">41.6</span></td>
<td style="text-align: center;"><span class="math inline"><strong>48.7</strong></span></td>
</tr>
<tr>
<td style="text-align: left;"><em><span class="math inline">#</span> Superhuman</em></td>
<td style="text-align: center;"><span class="math inline">3</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">2</span></td>
<td style="text-align: center;"><span class="math inline"><strong>4</strong></span></td>
</tr>
<tr>
<td style="text-align: left;"><em><span class="math inline">#</span> Best</em></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">2</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">3</span></td>
<td style="text-align: center;"><span class="math inline"><strong>11</strong></span></td>
</tr>
</tbody>
</table>
</figure>

**Evaluation on Atari-100k.**  To demonstrate the applicability of Adaptive RR in discrete-action tasks we move our evaluation to the Atari-100K benchmark , assessing Adaptive RR against three distinct static RR strategies across 17 games. In static RR settings, as shown in Table <a href="#table:atari-short" data-reference-type="ref" data-reference="table:atari-short">11</a>, algorithm performance significantly declines when RR increases to 2, indicating that the negative impact of plasticity loss gradually become dominant. However, Adaptive RR, by appropriately increasing RR from 0.5 to 2 at the right moment, can effectively avoid catastrophic plasticity loss, thus outperforming other configurations in most tasks.

# **Conclusion, Limitations, and Future Work**

In this work, we delve deeper into the plasticity of VRL, focusing on three previously underexplored aspects, deriving pivotal and enlightening insights: <span style="color: mydarkgreen">$`\bullet`$</span> DA emerges as a potent strategy to mitigate plasticity loss. <span style="color: mydarkgreen">$`\bullet`$</span> Critic’s plasticity loss stands as the primary hurdle to the sample-efficient VRL. <span style="color: mydarkgreen">$`\bullet`$</span> Ensuring plasticity recovery during the early stages is pivotal for efficient training. Armed with these insights, we propose *Adaptive RR* to address the high RR dilemma that has perplexed the VRL community for a long time. By striking a judicious balance between sample reuse frequency and plasticity loss management, *Adaptive RR* markedly improves the VRL’s sample efficiency.

**Limitations.**   Firstly, our experiments focus on DMC and Atari environments, without evaluation in more complex settings. As task complexity escalates, the significance and difficulty of maintaining plasticity concurrently correspondingly rise. Secondly, we only demonstrate the effectiveness of *Adaptive RR* under basic configurations. A more nuanced design could further unlock its potential.

**Future Work.**   Although neural networks have enabled scaling RL to complex decision-making scenarios, they also introduce numerous difficulties unique to DRL, which are absent in traditional RL contexts. Plasticity loss stands as a prime example of these challenges, fundamentally stemming from the contradiction between the trial-and-error nature of RL and the inability of neural networks to continuously learn non-stationary targets. To advance the real-world deployment of DRL, it is imperative to address and understand its distinct challenges. Given RL’s unique learning dynamics, exploration of DRL-specific network architectures and optimization techniques is essential.

# Extended Related Work

In this section, we provide an extended related work to supplement the related work presented in the main body.

**Sample-Efficient VRL.**   Prohibitive sample complexity has been identified as the primary obstacle hindering the real-world applications of VRL . Previous studies ascribe this inefficiency to VRL’s requirements to concurrently optimize task-specific policies and learn compact state representations from high-dimensional observations. As a result, significant efforts have been directed towards improving sample efficiency through the training of a more potent *encoder*. The most representative approaches design *auxiliary representation tasks* to complement the RL objective, including pixel or latent reconstruction , future prediction , and contrastive learning for instance  or temporal discrimination . Another approach is to *pre-train a visual encoder* that enables efficient adaptation to downstream tasks . However, recent empirical studies suggest that these methods do not consistently improve training efficiency , indicating that insufficient representation may not be the primary bottleneck hindering the sample efficiency of current algorithms. Our findings in Section <a href="#Sec: Modules" data-reference-type="ref" data-reference="Sec: Modules">4</a> provide a compelling explanation for the limited impact of enhanced representation: the plasticity loss within the critic module is the primary constraint on VRL’s sample efficiency.

**Plasticity Loss in Continual Learning vs. in Reinforcement Learning.**   Continual Learning (CL) aims to continuously acquire new tasks, referred to as plasticity, without forgetting previously learned tasks, termed stability. A primary challenge in CL is managing the stability-plasticity trade-off. Although online reinforcement learning (RL) exhibits characteristics of plasticity due to its non-stationary learning targets, there are fundamental differences between CL and RL. Firstly, online RL typically begins its learning process from scratch, which can lead to limited training data in the early stages. This scarcity of data can subsequently result in a loss of plasticity early on. Secondly, RL usually doesn’t require an agent to learn multiple policies. Therefore, any decline in plasticity during the later stages won’t significantly impact its overall performance.

**Measurement Metrics of Plasticity.**   <span id="Appendix: Measurement Metrics of Plasticity" label="Appendix: Measurement Metrics of Plasticity"></span> Several metrics are available to assess plasticity, including weight norm, feature rank, visualization of loss landscape, and the fraction of active units (FAU). The weight norm (commonly of both encoder and head) serves a dual purpose: it not only acts as a criterion to determine when to maintain plasticity but also offers a direct method to regulate plasticity through L2 regularization . However, show that the weight norm is sensitive to environments and cannot address the plasticity by controlling itself. The feature rank can be also regarded as a proxy metric for plasticity loss . Although the feature matrices used by these two works are slightly different, they correlate the feature rank with performance collapse. Nevertheless, observe that the correlation appears in restricted settings. Furthermore, the loss landscape has been drawing increasing attention for its ability to directly reflect the gradients in backpropagation. Still, computing the network’s Hessian concerning a loss function and the gradient covariance can be computationally demanding . Our proposed method aims to obtain a reliable criterion without too much additional computation cost, and leverage it to guide the plasticity maintenance. We thus settled on the widely-recognized and potent metric, FAU, for assessing plasticity . This metric provides an upper limit on the count of inactive units. As shown in Figure <a href="#fig:arr_main_result" data-reference-type="ref" data-reference="fig:arr_main_result">9</a>, the experimental results validate that A-RR based on FAU significantly outperforms static RR baselines. Although FAU’s efficacy is evident in various studies, including ours, its limitations in convolutional networks are highlighted by . Therefore, we advocate for future work to introduce a comprehensive and resilient plasticity metric.

# Extended Experiment Results

## Reset

To enhance the plasticity of the agent’s network, the *Reset* method periodically re-initializes the parameters of its last few layers, while preserving the replay buffer. In Figure <a href="#appendix_fig:reset" data-reference-type="ref" data-reference="appendix_fig:reset">12</a>, we present additional experiments on six DMC tasks, exploring four scenarios: with and without the inclusion of both Reset and DA. Although reset is widely acknowledged for its efficacy in counteracting the adverse impacts of plasticity loss, our findings suggest its effectiveness largely hinges on the hyper-parameters determining the reset interval, as depicted in Figure <a href="#appendix_fig:reset_interval" data-reference-type="ref" data-reference="appendix_fig:reset_interval">13</a>.

<figure id="appendix_fig:reset">
<p><img src="./figures/reset_DA.png"" /> <img src="./figures/reset_DA_1M.png"" /></p>
<figcaption>Training curves across four combinations: incorporating or excluding Reset and DA.</figcaption>
</figure>

<figure id="appendix_fig:reset_interval">
<p><img src="./figures/reset_times_QR.png"" style="width:60.0%" /> <img src="./figures/reset_times.png"" style="width:60.0%" /></p>
<figcaption>Learning curves for various reset intervals demonstrate that the effect of the reset strongly depends on the hyper-parameter that determines the reset interval.</figcaption>
</figure>

## Heavy Priming Phenomenon

Heavy priming  refers to updating the agent $`10^5`$ times using the replay buffer, which collects 2000 transitions after the start of the training process. Heavy priming can induce the agent to overfit to its early experiences. We conducted experiments to assess the effects of using heavy priming and DA, both individually and in combination. The training curves can be found in Figure <a href="#appendix_fig:heavy_priming" data-reference-type="ref" data-reference="appendix_fig:heavy_priming">14</a>. The findings indicate that, while heavy priming can markedly impair sample efficiency without DA, its detrimental impact is largely mitigated when DA is employed. Additionally, we examine the effects of employing DA both during heavy priming and subsequent training, as illustrated in Figure <a href="#appendix_fig:primingDA" data-reference-type="ref" data-reference="appendix_fig:primingDA">15</a>. The results indicate that DA not only mitigates plasticity loss during heavy priming but also facilitates its recovery afterward.

<figure id="appendix_fig:heavy_priming">
<img src="./figures/heavy_priming_vs_normal_training.png"" />
<figcaption>Heavy priming can severely damage the training efficiency when not employing DA.</figcaption>
</figure>

<figure id="appendix_fig:primingDA">
<img src="./figures/heavy_priming_DA.png"" />
<figcaption>DA not only can prevent the plasticity loss but also can recover the plasticity of agent after heavy priming phase.</figcaption>
</figure>

## Further Comparisons of DA with Other Interventions

<figure>
<p><img src="./figures/Interventions_return.png"" style="width:40.0%" /> <img src="./figures/Interventions_Feature_Rank.png"" /> <img src="./figures/Interventions_WN.png"" /> <img src="./figures/Interventions_FAU.png"" /></p>
<figcaption>We use three metrics - Feature Rank, Weight Norm, and Fraction of Active Units (FAU) - to assess the impact of various interventions on training dynamics.</figcaption>
</figure>

## Plasticity Injection

*Plasticity injection* is an intervention to increase the plasticity of a neural network. The network is schematically separated into an encoder $`\phi(\cdot)`$ and a head $`h_{\theta}(\cdot)`$. After plasticity injection, the parameters of head $`\theta`$ are frozen. Subsequently, two randomly initialized parameters, $`\theta'_1`$ and $`\theta'_2`$ are created. Here, $`\theta'_1`$ are trainable and $`\theta'_2`$ are frozen. The output from the head is computed using the formula $`h_{\theta}(z)+h_{\theta'_1}(z)-h_{\theta'_2}(z)`$, where $`z=\phi(x)`$.

We conducted additional plasticity injection experiments on Cheetah Run and Quadruped Run within DMC, as depicted in Figure <a href="#appendix_fig:Injection" data-reference-type="ref" data-reference="appendix_fig:Injection">16</a>. The results further bolster the conclusion made in Section <a href="#Sec: Modules" data-reference-type="ref" data-reference="Sec: Modules">4</a>: critic’s plasticity loss is the primary culprit behind VRL’s sample inefficiency.

<figure id="appendix_fig:Injection">
<p><img src="./figures/Injection_CR.png"" style="width:49.0%" /> <img src="./figures/Injection_QR.png"" style="width:49.0%" /></p>
<figcaption>Training curves showcasing the effects of Plasticity Injection on either the actor or critic, evaluated on Cheetah Run and Quadruped Run.</figcaption>
</figure>

## Turning On or Turning off DA at Early Stages

In Figure <a href="#appendix_fig:turn_on_and_off_DA" data-reference-type="ref" data-reference="appendix_fig:turn_on_and_off_DA">17</a>, we present additional results across six DMC tasks for various DA application modes. As emphasized in Section <a href="#Sec: Stages" data-reference-type="ref" data-reference="Sec: Stages">5</a>, it is necessary to maintain plasticity during early stages.

<figure id="appendix_fig:turn_on_and_off_DA">
<p><img src="./figures/recover_WFH.png"" /> <img src="./figures/recover.png"" /></p>
<figcaption>Training curves across different DA application modes, illustrating the critical role of plasticity in the early stage.</figcaption>
</figure>

## FAU Trends Across Different Tasks

In Figure <a href="#appendix_fig:FAU_task_1" data-reference-type="ref" data-reference="appendix_fig:FAU_task_1">18</a> and Figure <a href="#appendix_fig:FAU_task_2" data-reference-type="ref" data-reference="appendix_fig:FAU_task_2">19</a>, we showcase trends for various FAU modules across an additional nine DMC tasks as a complement to the main text.

<figure id="appendix_fig:FAU_task_1">
<p><img src="./figures/FAU_walker_stand.png"" /> <img src="./figures/FAU_walker_walk.png"" /> <img src="./figures/FAU_walker_run.png"" /> <img src="./figures/FAU_reacher_hard.png"" /></p>
<figcaption>FAU trends for various modules within the VRL agent across DMC tasks (<em>Walker Stand</em>, <em>Walker Walk</em>, <em>Walker Run</em> and <em>Reacher Hard</em>) throughout the training process.</figcaption>
</figure>

<figure id="appendix_fig:FAU_task_2">
<p><img src="./figures/FAU_hopper_stand.png"" /> <img src="./figures/FAU_hopper_hop.png"" /> <img src="./figures/FAU_finger_spin.png"" /> <img src="./figures/FAU_quadruped_walk.png"" /> <img src="./figures/FAU_quadruped_run.png"" /></p>
<figcaption>FAU trends for various modules within the VRL agent across DMC tasks (<em>Hopper Stand</em>, <em>Hopper Hop</em>, <em>Finger Spin</em>, <em>Quadruped Walk</em> and <em>Quadruped Run</em>) throughout the training process.</figcaption>
</figure>

Figure <a href="#appendix_fig:FAU_seed" data-reference-type="ref" data-reference="appendix_fig:FAU_seed">20</a> provides the FAU for different random seeds, demonstrating that the trend is consistent across all random seeds.

<figure id="appendix_fig:FAU_seed">
<p><img src="./figures/sing_UTD_cheetah_run.png"" style="width:45.0%" /> <img src="./figures/sing_UTD_walker_run.png"" style="width:45.0%" /> <img src="./figures/sing_UTD_quadruped_walk.png"" style="width:45.0%" /> <img src="./figures/sing_UTD_quadruped_run.png"" style="width:45.0%" /> <img src="./figures/sing_UTD_hopper_hop.png"" style="width:45.0%" /> <img src="./figures/sing_UTD_reacher_hard.png"" style="width:45.0%" /></p>
<figcaption>FAU trends for various modules within the VRL agent, evaluated across six DMC tasks and observed for each random seed throughout the training process.</figcaption>
</figure>

## Additional Metrics to Quantify the Plasticity

<figure>
<p><img src="./figures/WS_Weight_Norm.png"" style="width:90.0%" /> <img src="./figures/WS_Feature_Rank.png"" style="width:90.0%" /><br />
<img src="./figures/QR_Weight_Norm.png"" style="width:90.0%" /> <img src="./figures/QR_Feature_Rank.png"" style="width:90.0%" /></p>
<figcaption>Measuring the plasticity of different modules via feature rank and weight norm.</figcaption>
</figure>

# Experimental Details

In this section, we provide our detailed setting in experiments.

## Algorithm

<div class="center">

<div class="minipage">

<figure id="algo:main">
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><br />
Require Check interval <span class="math inline"><em>I</em></span>, threshold <span class="math inline"><em>τ</em></span>, total steps <span class="math inline"><em>T</em></span><br />
Initialize RL training with a low RR<br />
<strong>While</strong> <span><span class="math inline"><em>t</em> &lt; <em>T</em></span></span><br />
<strong>If</strong> <span><span class="math inline"><em>t</em>%<em>I</em> = 0</span> and <span class="math inline">|<em>Φ</em><sub><em>C</em></sub><sup><em>t</em></sup> − <em>Φ</em><sub><em>C</em></sub><sup><em>t</em> − <em>I</em></sup>| &lt; <em>τ</em></span></span><br />
Switch to high RR<br />
EndIf<br />
Continue RL training with the current RR<br />
Increment step <span class="math inline"><em>t</em></span><br />
EndWhile</p>
</div>
<figcaption>Adaptive RR</figcaption>
</figure>

</div>

</div>

## DMC Setup

We conducted experiments on robot control tasks within DeepMind Control using image input as the observation. All experiments are based on previously superior DrQ-v2 algorithms and maintain all hyper-parameters from DrQ-v2 unchanged. The only modification made was to the replay ratio, adjusted according to the specific setting. The hyper-parameters are presented in Table <a href="#table:DMC" data-reference-type="ref" data-reference="table:DMC">2</a>.

<div id="table:DMC">

<table>
<caption> A default set of hyper-parameters used in DMControl evaluation.</caption>
<thead>
<tr>
<th colspan="2" style="text-align: center;">Algorithms Hyper-parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Replay buffer capacity</td>
<td style="text-align: center;"><span class="math inline">10<sup>6</sup></span></td>
</tr>
<tr>
<td style="text-align: left;">Action repeat</td>
<td style="text-align: center;"><span class="math inline">2</span></td>
</tr>
<tr>
<td style="text-align: left;">Seed frames</td>
<td style="text-align: center;"><span class="math inline">4000</span></td>
</tr>
<tr>
<td style="text-align: left;">Exploration steps</td>
<td style="text-align: center;"><span class="math inline">2000</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>n</em></span>-step returns</td>
<td style="text-align: center;"><span class="math inline">3</span></td>
</tr>
<tr>
<td style="text-align: left;">Mini-batch size</td>
<td style="text-align: center;"><span class="math inline">256</span></td>
</tr>
<tr>
<td style="text-align: left;">Discount <span class="math inline"><em>γ</em></span></td>
<td style="text-align: center;"><span class="math inline">0.99</span></td>
</tr>
<tr>
<td style="text-align: left;">Optimizer</td>
<td style="text-align: center;">Adam</td>
</tr>
<tr>
<td style="text-align: left;">Learning rate</td>
<td style="text-align: center;"><span class="math inline">10<sup>−4</sup></span></td>
</tr>
<tr>
<td style="text-align: left;">Critic Q-function soft-update rate <span class="math inline"><em>τ</em></span></td>
<td style="text-align: center;"><span class="math inline">0.01</span></td>
</tr>
<tr>
<td style="text-align: left;">Features dim.</td>
<td style="text-align: center;"><span class="math inline">50</span></td>
</tr>
<tr>
<td style="text-align: left;">Repr. dim.</td>
<td style="text-align: center;"><span class="math inline">32 × 35 × 35</span></td>
</tr>
<tr>
<td style="text-align: left;">Hidden dim.</td>
<td style="text-align: center;"><span class="math inline">1024</span></td>
</tr>
<tr>
<td style="text-align: left;">Exploration stddev. clip</td>
<td style="text-align: center;"><span class="math inline">0.3</span></td>
</tr>
<tr>
<td style="text-align: left;">Exploration stddev. schedule</td>
<td style="text-align: center;"><span class="math inline">linear(1.0, 0.1, 500000)</span></td>
</tr>
</tbody>
</table>

</div>

<span id="table:DMC" label="table:DMC"></span>

# Evaluation on Atari

**Implement details.** Our Atari experiments and implementation were based on the Dopamine framework . For ReDo  and DrQ($`\epsilon`$), We used the same setting as Dopamine, shown in Table <a href="#table:atari-config" data-reference-type="ref" data-reference="table:atari-config">3</a>. we use 5 independent random seeds for each Atari game. The detailed results are shown in Table <a href="#table:atari-full" data-reference-type="ref" data-reference="table:atari-full">4</a>.

<div class="center">

<div id="table:atari-config">

| Common Parameter-DrQ($`\epsilon`$) | Value |
|:---|---:|
| Optimizer | Adam |
| Optimizer: Learning rate | $`1 \times 10^{-4}`$ |
| Optimizer: $`\epsilon`$ | $`1.5 \times 10^{-4}`$ |
| Training $`\epsilon`$ | 0.01 |
| Evaluation $`\epsilon`$ | 0.001 |
| Discount factor | 0.99 |
| Replay buffer size | $`10^6`$ |
| Minibatch size | 32 |
| Q network: channels | 32, 64, 64 |
| Q-network: filter size | 8 $`\times`$ 8, 4 $`\times`$ 4, 3 $`\times`$ 3 |
| Q-network: stride | 4, 2, 1 |
| Q-network: hidden units | 512 |
| Initial collect steps | 1600 |
| $`n`$-step | 10 |
| Training iterations | 40 |
| Training environment steps per iteration | 10K |
| ReDo Parameter | Value |
| Recycling period | 1000 |
| $`\tau`$-Dormant | 0.025 |
| Minibatch size for estimating neurons score | 64 |
| Adaptive RR Parameter | Value |
| check interval | 2000 |
| threshold | 0.001 |
| low Replay Ratio | 0.5 |
| high Replay Ratio | 2 |

Hyper-parameters for Atari-100K.

</div>

</div>

<span id="table:atari-config" label="table:atari-config"></span>

<div class="small">

<div id="table:atari-full">

<table>
<caption><strong>Evaluation of Sample Efficiency on Atari-100k.</strong> We report the scores and the mean and median HNSs achieved by different methods on Atari-100k.</caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: left;">Game</td>
<td rowspan="2" style="text-align: center;">Human</td>
<td rowspan="2" style="text-align: center;">Random</td>
<td style="text-align: center;"><span>DrQ(<span class="math inline"><em>ϵ</em></span>)</span></td>
<td style="text-align: center;"><span>DrQ(<span class="math inline"><em>ϵ</em></span>)</span></td>
<td style="text-align: center;"><span>DrQ(<span class="math inline"><em>ϵ</em></span>)</span></td>
<td style="text-align: center;"><span>ReDo</span></td>
<td style="text-align: center;"><span>Adaptive RR</span></td>
</tr>
<tr>
<td style="text-align: center;"><span>(RR=0.5)</span></td>
<td style="text-align: center;"><span>(RR=1)</span></td>
<td style="text-align: center;"><span>(RR=2)</span></td>
<td style="text-align: center;"><span>(RR=1)</span></td>
<td style="text-align: center;"><span>(RR0.5to2)</span></td>
</tr>
<tr>
<td style="text-align: left;">Alien</td>
<td style="text-align: center;"><span class="math inline">7127.7</span></td>
<td style="text-align: center;"><span class="math inline">227.8</span></td>
<td style="text-align: center;"><span class="math inline">815</span></td>
<td style="text-align: center;"><span class="math inline">865</span></td>
<td style="text-align: center;"><span class="math inline">917</span></td>
<td style="text-align: center;"><span class="math inline">794</span></td>
<td style="text-align: center;"><span class="math inline"><strong>935</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Amidar</td>
<td style="text-align: center;"><span class="math inline">1719.5</span></td>
<td style="text-align: center;"><span class="math inline">5.8</span></td>
<td style="text-align: center;"><span class="math inline">114</span></td>
<td style="text-align: center;"><span class="math inline">138</span></td>
<td style="text-align: center;"><span class="math inline">133</span></td>
<td style="text-align: center;"><span class="math inline">163</span></td>
<td style="text-align: center;"><span class="math inline"><strong>200</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Assault</td>
<td style="text-align: center;"><span class="math inline">742.0</span></td>
<td style="text-align: center;"><span class="math inline">222.4</span></td>
<td style="text-align: center;"><span class="math inline">755</span></td>
<td style="text-align: center;"><span class="math inline">580</span></td>
<td style="text-align: center;"><span class="math inline">579</span></td>
<td style="text-align: center;"><span class="math inline">675</span></td>
<td style="text-align: center;"><span class="math inline"><strong>823</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Asterix</td>
<td style="text-align: center;"><span class="math inline">8503.3</span></td>
<td style="text-align: center;"><span class="math inline">210.0</span></td>
<td style="text-align: center;"><span class="math inline">470</span></td>
<td style="text-align: center;"><span class="math inline"><strong>764</strong></span></td>
<td style="text-align: center;"><span class="math inline">442</span></td>
<td style="text-align: center;"><span class="math inline">684</span></td>
<td style="text-align: center;"><span class="math inline">519</span></td>
</tr>
<tr>
<td style="text-align: left;">Bank Heist</td>
<td style="text-align: center;"><span class="math inline">753.1</span></td>
<td style="text-align: center;"><span class="math inline">14.2</span></td>
<td style="text-align: center;"><span class="math inline">451</span></td>
<td style="text-align: center;"><span class="math inline">232</span></td>
<td style="text-align: center;"><span class="math inline">91</span></td>
<td style="text-align: center;"><span class="math inline">61</span></td>
<td style="text-align: center;"><span class="math inline"><strong>553</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Boxing</td>
<td style="text-align: center;"><span class="math inline">12.1</span></td>
<td style="text-align: center;"><span class="math inline">0.1</span></td>
<td style="text-align: center;"><span class="math inline">16</span></td>
<td style="text-align: center;"><span class="math inline">9</span></td>
<td style="text-align: center;"><span class="math inline">6</span></td>
<td style="text-align: center;"><span class="math inline">9</span></td>
<td style="text-align: center;"><span class="math inline"><strong>18</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Breakout</td>
<td style="text-align: center;"><span class="math inline">30.5</span></td>
<td style="text-align: center;"><span class="math inline">1.7</span></td>
<td style="text-align: center;"><span class="math inline">17</span></td>
<td style="text-align: center;"><span class="math inline"><strong>20</strong></span></td>
<td style="text-align: center;"><span class="math inline">13</span></td>
<td style="text-align: center;"><span class="math inline">15</span></td>
<td style="text-align: center;"><span class="math inline">16</span></td>
</tr>
<tr>
<td style="text-align: left;">Chopper Command</td>
<td style="text-align: center;"><span class="math inline">7387.8</span></td>
<td style="text-align: center;"><span class="math inline">811.0</span></td>
<td style="text-align: center;"><span class="math inline">1037</span></td>
<td style="text-align: center;"><span class="math inline">845</span></td>
<td style="text-align: center;"><span class="math inline">1129</span></td>
<td style="text-align: center;"><span class="math inline"><strong>1650</strong></span></td>
<td style="text-align: center;"><span class="math inline">1544</span></td>
</tr>
<tr>
<td style="text-align: left;">Crazy Climber</td>
<td style="text-align: center;"><span class="math inline">35829.4</span></td>
<td style="text-align: center;"><span class="math inline">10780.5</span></td>
<td style="text-align: center;"><span class="math inline">18108</span></td>
<td style="text-align: center;"><span class="math inline">21539</span></td>
<td style="text-align: center;"><span class="math inline">17193</span></td>
<td style="text-align: center;"><span class="math inline"><strong>24492</strong></span></td>
<td style="text-align: center;"><span class="math inline">22986</span></td>
</tr>
<tr>
<td style="text-align: left;">Demon Attack</td>
<td style="text-align: center;"><span class="math inline">1971.0</span></td>
<td style="text-align: center;"><span class="math inline">152.1</span></td>
<td style="text-align: center;"><span class="math inline">1993</span></td>
<td style="text-align: center;"><span class="math inline">1321</span></td>
<td style="text-align: center;"><span class="math inline">1125</span></td>
<td style="text-align: center;"><span class="math inline">2091</span></td>
<td style="text-align: center;"><span class="math inline"><strong>2098</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Enduro</td>
<td style="text-align: center;"><span class="math inline">861</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">128</span></td>
<td style="text-align: center;"><span class="math inline">223</span></td>
<td style="text-align: center;"><span class="math inline">138</span></td>
<td style="text-align: center;"><span class="math inline"><strong>224</strong></span></td>
<td style="text-align: center;"><span class="math inline">200</span></td>
</tr>
<tr>
<td style="text-align: left;">Freeway</td>
<td style="text-align: center;"><span class="math inline">29.6</span></td>
<td style="text-align: center;"><span class="math inline">0.0</span></td>
<td style="text-align: center;"><span class="math inline">21</span></td>
<td style="text-align: center;"><span class="math inline">20</span></td>
<td style="text-align: center;"><span class="math inline">20</span></td>
<td style="text-align: center;"><span class="math inline">19</span></td>
<td style="text-align: center;"><span class="math inline"><strong>23</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Kung Fu Master</td>
<td style="text-align: center;"><span class="math inline">22736.3</span></td>
<td style="text-align: center;"><span class="math inline">258.5</span></td>
<td style="text-align: center;"><span class="math inline">5342</span></td>
<td style="text-align: center;"><span class="math inline">11467</span></td>
<td style="text-align: center;"><span class="math inline">8423</span></td>
<td style="text-align: center;"><span class="math inline">11642</span></td>
<td style="text-align: center;"><span class="math inline"><strong>12195</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Pong</td>
<td style="text-align: center;"><span class="math inline">14.6</span></td>
<td style="text-align: center;"><span class="math inline">−20.7</span></td>
<td style="text-align: center;"><span class="math inline">−16</span></td>
<td style="text-align: center;"><span class="math inline">−10</span></td>
<td style="text-align: center;"><span class="math inline"><strong>3</strong></span></td>
<td style="text-align: center;"><span class="math inline">−6</span></td>
<td style="text-align: center;"><span class="math inline">−9</span></td>
</tr>
<tr>
<td style="text-align: left;">Road Runner</td>
<td style="text-align: center;"><span class="math inline">7845.0</span></td>
<td style="text-align: center;"><span class="math inline">11.5</span></td>
<td style="text-align: center;"><span class="math inline">6478</span></td>
<td style="text-align: center;"><span class="math inline">11211</span></td>
<td style="text-align: center;"><span class="math inline">9430</span></td>
<td style="text-align: center;"><span class="math inline">8606</span></td>
<td style="text-align: center;"><span class="math inline"><strong>12424</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Seaquest</td>
<td style="text-align: center;"><span class="math inline">42054.7</span></td>
<td style="text-align: center;"><span class="math inline">68.4</span></td>
<td style="text-align: center;"><span class="math inline">390</span></td>
<td style="text-align: center;"><span class="math inline">352</span></td>
<td style="text-align: center;"><span class="math inline">394</span></td>
<td style="text-align: center;"><span class="math inline">292</span></td>
<td style="text-align: center;"><span class="math inline"><strong>451</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">SpaceInvaders</td>
<td style="text-align: center;"><span class="math inline">1669</span></td>
<td style="text-align: center;"><span class="math inline">148</span></td>
<td style="text-align: center;"><span class="math inline">388</span></td>
<td style="text-align: center;"><span class="math inline">402</span></td>
<td style="text-align: center;"><span class="math inline">408</span></td>
<td style="text-align: center;"><span class="math inline">379</span></td>
<td style="text-align: center;"><span class="math inline"><strong>493</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">Mean HNS (<span class="math inline">%</span>)</td>
<td style="text-align: center;">100</td>
<td style="text-align: center;">0</td>
<td style="text-align: center;">42.3</td>
<td style="text-align: center;">41.3</td>
<td style="text-align: center;">35.1</td>
<td style="text-align: center;">42.3</td>
<td style="text-align: center;"><strong>55.8</strong></td>
</tr>
<tr>
<td style="text-align: left;">Median HNS (<span class="math inline">%</span>)</td>
<td style="text-align: center;">100</td>
<td style="text-align: center;">0</td>
<td style="text-align: center;">22.6</td>
<td style="text-align: center;">30.3</td>
<td style="text-align: center;">26.0</td>
<td style="text-align: center;">41.6</td>
<td style="text-align: center;"><strong>48.7</strong></td>
</tr>
<tr>
<td style="text-align: left;"># Superhuman</td>
<td style="text-align: center;">N/A</td>
<td style="text-align: center;">0</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">2</td>
<td style="text-align: center;"><strong>4</strong></td>
</tr>
<tr>
<td style="text-align: left;"># Best</td>
<td style="text-align: center;">N/A</td>
<td style="text-align: center;">0</td>
<td style="text-align: center;">0</td>
<td style="text-align: center;">2</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;"><strong>11</strong></td>
</tr>
</tbody>
</table>

</div>

</div>

<span id="table:atari-full" label="table:atari-full"></span>

# Acknowledgements

This work is supported by STI 2030—Major Projects (No. 2021ZD0201405). We thank Zilin Wang and Haoyu Wang for their valuable suggestions and collaboration. We also extend our thanks to Evgenii Nikishin for his support in the implementation of plasticity injection. Furthermore, we sincerely appreciate the time and effort invested by the anonymous reviewers in evaluating our work, and are grateful for their valuable and insightful feedback.

# References

<div class="thebibliography">

Zaheer Abbas, Rosie Zhao, Joseph Modayil, Adam White, and Marlos C Machado Loss of plasticity in continual deep reinforcement learning In *Conference on Lifelong Learning Agents*, pp. 620–636. PMLR, 2023. **Abstract:** The ability to learn continually is essential in a complex and changing world. In this paper, we characterize the behavior of canonical value-based deep reinforcement learning (RL) approaches under varying degrees of non-stationarity. In particular, we demonstrate that deep RL agents lose their ability to learn good policies when they cycle through a sequence of Atari 2600 games. This phenomenon is alluded to in prior work under various guises – e.g., loss of plasticity, implicit under-parameterization, primacy bias, and capacity loss. We investigate this phenomenon closely at scale and analyze how the weights, gradients, and activations change over time in several experiments with varying dimensions (e.g., similarity between games, number of games, number of frames per game), with some experiments spanning 50 days and 2 billion environment interactions. Our analysis shows that the activation footprint of the network becomes sparser, contributing to the diminishing gradients. We investigate a remarkably simple mitigation strategy – Concatenated ReLUs (CReLUs) activation function – and demonstrate its effectiveness in facilitating continual learning in a changing environment. (@plasticity_loss_CRL)

Jordan Ash and Ryan P Adams On warm-starting neural network training *Advances in neural information processing systems*, 33: 3884–3894, 2020. **Abstract:** In many real-world deployments of machine learning systems, data arrive piecemeal. These learning scenarios may be passive, where data arrive incrementally due to structural properties of the problem (e.g., daily financial data) or active, where samples are selected according to a measure of their quality (e.g., experimental design). In both of these cases, we are building a sequence of models that incorporate an increasing amount of data. We would like each of these models in the sequence to be performant and take advantage of all the data that are available to that point. Conventional intuition suggests that when solving a sequence of related optimization problems of this form, it should be possible to initialize using the solution of the previous iterate – to start the optimization rather than initialize from scratch – and see reductions in wall-clock time. However, in practice this warm-starting seems to yield poorer generalization performance than models that have fresh random initializations, even though the final training losses are similar. While it appears that some hyperparameter settings allow a practitioner to close this generalization gap, they seem to only do so in regimes that damage the wall-clock gains of the warm start. Nevertheless, it is highly desirable to be able to warm-start neural network training, as it would dramatically reduce the resource usage associated with the construction of performant deep learning systems. In this work, we take a closer look at this empirical phenomenon and try to understand when and how it occurs. We also provide a surprisingly simple trick that overcomes this pathology in several important situations, and present experiments that elucidate some of its properties. (@shrink_and_perturb)

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton Layer normalization *arXiv preprint arXiv:1607.06450*, 2016. **Abstract:** Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed-forward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques. (@layer_norm)

Pablo Samuel Castro, Subhodeep Moitra, Carles Gelada, Saurabh Kumar, and Marc G. Bellemare Dopamine: A Research Framework for Deep Reinforcement Learning . URL <http://arxiv.org/abs/1812.06110>. **Abstract:** Deep reinforcement learning (deep RL) research has grown significantly in recent years. A number of software offerings now exist that provide stable, comprehensive implementations for benchmarking. At the same time, recent deep RL research has become more diverse in its goals. In this paper we introduce Dopamine, a new research framework for deep RL that aims to support some of that diversity. Dopamine is open-source, TensorFlow-based, and provides compact and reliable implementations of some state-of-the-art deep RL agents. We complement this offering with a taxonomy of the different research objectives in deep RL research. While by no means exhaustive, our analysis highlights the heterogeneity of research in the field, and the value of frameworks such as ours. (@castro18dopamine)

Edoardo Cetin, Philip J Ball, Stephen Roberts, and Oya Celiktutan Stabilizing off-policy deep reinforcement learning from pixels In *International Conference on Machine Learning*, pp. 2784–2810. PMLR, 2022. **Abstract:** Off-policy reinforcement learning (RL) from pixel observations is notoriously unstable. As a result, many successful algorithms must combine different domain-specific practices and auxiliary losses to learn meaningful behaviors in complex environments. In this work, we provide novel analysis demonstrating that these instabilities arise from performing temporal-difference learning with a convolutional encoder and low-magnitude rewards. We show that this new visual deadly triad causes unstable training and premature convergence to degenerate solutions, a phenomenon we name catastrophic self-overfitting. Based on our analysis, we propose A-LIX, a method providing adaptive regularization to the encoder’s gradients that explicitly prevents the occurrence of catastrophic self-overfitting using a dual objective. By applying A-LIX, we significantly outperform the prior state-of-the-art on the DeepMind Control and Atari 100k benchmarks without any data augmentation or auxiliary losses. (@A-LIX)

Xinyue Chen, Che Wang, Zijian Zhou, and Keith W Ross Randomized ensembled double q-learning: Learning fast without a model In *International Conference on Learning Representations*, 2020. **Abstract:** Using a high Update-To-Data (UTD) ratio, model-based methods have recently achieved much higher sample efficiency than previous model-free methods for continuous-action DRL benchmarks. In this paper, we introduce a simple model-free algorithm, Randomized Ensembled Double Q-Learning (REDQ), and show that its performance is just as good as, if not better than, a state-of-the-art model-based algorithm for the MuJoCo benchmark. Moreover, REDQ can achieve this performance using fewer parameters than the model-based method, and with less wall-clock run time. REDQ has three carefully integrated ingredients which allow it to achieve its high performance: (i) a UTD ratio \>\> 1; (ii) an ensemble of Q functions; (iii) in-target minimization across a random subset of Q functions from the ensemble. Through carefully designed experiments, we provide a detailed analysis of REDQ and related model-free algorithms. To our knowledge, REDQ is the first successful model-free DRL algorithm for continuous-action spaces using a UTD ratio \>\> 1. (@REDQ)

Shibhansh Dohare, Richard S Sutton, and A Rupam Mahmood Continual backprop: Stochastic gradient descent with persistent randomness *arXiv preprint arXiv:2108.06325*, 2021. **Abstract:** The Backprop algorithm for learning in neural networks utilizes two mechanisms: first, stochastic gradient descent and second, initialization with small random weights, where the latter is essential to the effectiveness of the former. We show that in continual learning setups, Backprop performs well initially, but over time its performance degrades. Stochastic gradient descent alone is insufficient to learn continually; the initial randomness enables only initial learning but not continual learning. To the best of our knowledge, ours is the first result showing this degradation in Backprop’s ability to learn. To address this degradation in Backprop’s plasticity, we propose an algorithm that continually injects random features alongside gradient descent using a new generate-and-test process. We call this the \\}textit{Continual Backprop} algorithm. We show that, unlike Backprop, Continual Backprop is able to continually adapt in both supervised and reinforcement learning (RL) problems. Continual Backprop has the same computational complexity as Backprop and can be seen as a natural extension of Backprop for continual learning. (@continual_backprop)

Pierluca D’Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc G Bellemare, and Aaron Courville Sample-efficient reinforcement learning by breaking the replay ratio barrier In *The Eleventh International Conference on Learning Representations*, 2022. (@breaking_RR_barrier)

Jiameng Fan and Wenchao Li Dribo: Robust deep reinforcement learning via multi-view information bottleneck In *International Conference on Machine Learning*, pp. 6074–6102. PMLR, 2022. **Abstract:** Deep reinforcement learning (DRL) agents are often sensitive to visual changes that were unseen in their training environments. To address this problem, we leverage the sequential nature of RL to learn robust representations that encode only task-relevant information from observations based on the unsupervised multi-view setting. Specifically, we introduce a novel contrastive version of the Multi-View Information Bottleneck (MIB) objective for temporal data. We train RL agents from pixels with this auxiliary objective to learn robust representations that can compress away task-irrelevant information and are predictive of task-relevant dynamics. This approach enables us to train high-performance policies that are robust to visual distractions and can generalize well to unseen environments. We demonstrate that our approach can achieve SOTA performance on a diverse set of visual control tasks in the DeepMind Control Suite when the background is replaced with natural videos. In addition, we show that our approach outperforms well-established baselines for generalization to unseen environments on the Procgen benchmark. Our code is open-sourced and available at https://github. com/BU-DEPEND-Lab/DRIBO. (@DRIBO)

William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, and Will Dabney Revisiting fundamentals of experience replay In *International Conference on Machine Learning*, pp. 3061–3071. PMLR, 2020. **Abstract:** Experience replay is central to off-policy algorithms in deep reinforcement learning (RL), but there remain significant gaps in our understanding. We therefore present a systematic and extensive analysis of experience replay in Q-learning methods, focusing on two fundamental properties: the replay capacity and the ratio of learning updates to experience collected (replay ratio). Our additive and ablative studies upend conventional wisdom around experience replay – greater capacity is found to substantially increase the performance of certain algorithms, while leaving others unaffected. Counterintuitively we show that theoretically ungrounded, uncorrected n-step returns are uniquely beneficial while other techniques confer limited benefit for sifting through larger memory. Separately, by directly controlling the replay ratio we contextualize previous observations in the literature and empirically measure its importance across a variety of deep RL algorithms. Finally, we conclude by testing a set of hypotheses on the nature of these performance benefits. (@fedus2020revisiting)

Caglar Gulcehre, Srivatsan Srinivasan, Jakub Sygnowski, Georg Ostrovski, Mehrdad Farajtabar, Matthew Hoffman, Razvan Pascanu, and Arnaud Doucet An empirical study of implicit regularization in deep offline rl *Transactions on Machine Learning Research*, 2022. **Abstract:** Deep neural networks are the most commonly used function approximators in offline reinforcement learning. Prior works have shown that neural nets trained with TD-learning and gradient descent can exhibit implicit regularization that can be characterized by under-parameterization of these networks. Specifically, the rank of the penultimate feature layer, also called \\}textit{effective rank}, has been observed to drastically collapse during the training. In turn, this collapse has been argued to reduce the model’s ability to further adapt in later stages of learning, leading to the diminished final performance. Such an association between the effective rank and performance makes effective rank compelling for offline RL, primarily for offline policy evaluation. In this work, we conduct a careful empirical study on the relation between effective rank and performance on three offline RL datasets : bsuite, Atari, and DeepMind lab. We observe that a direct association exists only in restricted settings and disappears in the more extensive hyperparameter sweeps. Also, we empirically identify three phases of learning that explain the impact of implicit regularization on the learning dynamics and found that bootstrapping alone is insufficient to explain the collapse of the effective rank. Further, we show that several other factors could confound the relationship between effective rank and performance and conclude that studying this association under simplistic assumptions could be highly misleading. (@gulcehre2022empirical)

Nicklas Hansen, Zhecheng Yuan, Yanjie Ze, Tongzhou Mu, Aravind Rajeswaran, Hao Su, Huazhe Xu, and Xiaolong Wang On pre-training for visuo-motor control: Revisiting a learning-from-scratch baseline In *International Conference on Machine Learning*, pp. 12511–12526. PMLR, 2023. **Abstract:** In this paper, we examine the effectiveness of pre-training for visuo-motor control tasks. We revisit a simple Learning-from-Scratch (LfS) baseline that incorporates data augmentation and a shallow ConvNet, and find that this baseline is surprisingly competitive with recent approaches (PVR, MVP, R3M) that leverage frozen visual representations trained on large-scale vision datasets – across a variety of algorithms, task domains, and metrics in simulation and on a real robot. Our results demonstrate that these methods are hindered by a significant domain gap between the pre-training datasets and current benchmarks for visuo-motor control, which is alleviated by finetuning. Based on our findings, we provide recommendations for future research in pre-training for control and hope that our simple yet strong baseline will aid in accurately benchmarking progress in this area. (@Learning-from-Scratch)

John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žı́dek, Anna Potapenko, et al Highly accurate protein structure prediction with alphafold *Nature*, 596 (7873): 583–589, 2021. **Abstract:** Abstract Proteins are essential to life, and understanding their structure can facilitate a mechanistic understanding of their function. Through an enormous experimental effort 1–4 , the structures of around 100,000 unique proteins have been determined 5 , but this represents a small fraction of the billions of known protein sequences 6,7 . Structural coverage is bottlenecked by the months to years of painstaking effort required to determine a single protein structure. Accurate computational approaches are needed to address this gap and to enable large-scale structural bioinformatics. Predicting the three-dimensional structure that a protein will adopt based solely on its amino acid sequence—the structure prediction component of the ‘protein folding problem’ 8 —has been an important open research problem for more than 50 years 9 . Despite recent progress 10–14 , existing methods fall far short of atomic accuracy, especially when no homologous structure is available. Here we provide the first computational method that can regularly predict protein structures with atomic accuracy even in cases in which no similar structure is known. We validated an entirely redesigned version of our neural network-based model, AlphaFold, in the challenging 14th Critical Assessment of protein Structure Prediction (CASP14) 15 , demonstrating accuracy competitive with experimental structures in a majority of cases and greatly outperforming other methods. Underpinning the latest version of AlphaFold is a novel machine learning approach that incorporates physical and biological knowledge about protein structure, leveraging multi-sequence alignments, into the design of the deep learning algorithm. (@AlphaFold)

Łukasz Kaiser, Mohammad Babaeizadeh, Piotr Miłos, Błażej Osiński, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al Model based reinforcement learning for atari In *International Conference on Learning Representations*, 2019. **Abstract:** Model-free reinforcement learning (RL) can be used to learn effective policies for complex tasks, such as Atari games, even from image observations. However, this typically requires very large amounts of interaction – substantially more, in fact, than a human would need to learn the same games. How can people learn so quickly? Part of the answer may be that people can learn how the game works and predict which actions will lead to desirable outcomes. In this paper, we explore how video prediction models can similarly enable agents to solve Atari games with fewer interactions than model-free methods. We describe Simulated Policy Learning (SimPLe), a complete model-based deep RL algorithm based on video prediction models and present a comparison of several model architectures, including a novel architecture that yields the best results in our setting. Our experiments evaluate SimPLe on a range of Atari games in low data regime of 100k interactions between the agent and the environment, which corresponds to two hours of real-time play. In most games SimPLe outperforms state-of-the-art model-free algorithms, in some games by over an order of magnitude. (@kaiser2019model)

Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, and Sergey Levine Implicit under-parameterization inhibits data-efficient deep reinforcement learning In *International Conference on Learning Representations*, 2020. **Abstract:** We identify a fundamental implicit under-parameterization phenomenon in value-based deep RL methods that use bootstrapping: when value functions, approximated using deep neural networks, are trained with gradient descent using iterated regression onto target values generated by previous instances of the value network, more gradient updates decrease the expressivity of the current value network. We characterize this loss of expressivity via a rank collapse of the learned value network features and show that it corresponds to a drop in performance. We demonstrate this phenomenon on popular domains including Atari and Gym benchmarks and in both offline and online RL settings. We formally analyze this phenomenon and show that it results from a pathological interaction between bootstrapping and gradient-based optimization. Finally, we show that mitigating implicit under- parameterization by controlling rank collapse improves performance. (@implicit_under-parameterization)

Saurabh Kumar, Henrik Marklund, and Benjamin Van Roy Maintaining plasticity via regenerative regularization *arXiv preprint arXiv:2308.11958*, 2023. **Abstract:** In continual learning, plasticity refers to the ability of an agent to quickly adapt to new information. Neural networks are known to lose plasticity when processing non-stationary data streams. In this paper, we propose L2 Init, a simple approach for maintaining plasticity by incorporating in the loss function L2 regularization toward initial parameters. This is very similar to standard L2 regularization (L2), the only difference being that L2 regularizes toward the origin. L2 Init is simple to implement and requires selecting only a single hyper-parameter. The motivation for this method is the same as that of methods that reset neurons or parameter values. Intuitively, when recent losses are insensitive to particular parameters, these parameters should drift toward their initial values. This prepares parameters to adapt quickly to new tasks. On problems representative of different types of nonstationarity in continual supervised learning, we demonstrate that L2 Init most consistently mitigates plasticity loss compared to previously proposed approaches. (@Regenerative_Regularization)

Michael Laskin, Aravind Srinivas, and Pieter Abbeel Curl: Contrastive unsupervised representations for reinforcement learning In *International Conference on Machine Learning*, pp. 5639–5650. PMLR, 2020. **Abstract:** We present CURL: Contrastive Unsupervised Representations for Reinforcement Learning. CURL extracts high-level features from raw pixels using contrastive learning and performs off-policy control on top of the extracted features. CURL outperforms prior pixel-based methods, both model-based and model-free, on complex tasks in the DeepMind Control Suite and Atari Games showing 1.9x and 1.2x performance gains at the 100K environment and interaction steps benchmarks respectively. On the DeepMind Control Suite, CURL is the first image-based algorithm to nearly match the sample-efficiency of methods that use state-based features. Our code is open-sourced and available at https://github.com/MishaLaskin/curl. (@CURL)

Misha Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, and Aravind Srinivas Reinforcement learning with augmented data *Advances in neural information processing systems*, 33: 19884–19895, 2020. **Abstract:** Learning from visual observations is a fundamental yet challenging problem in Reinforcement Learning (RL). Although algorithmic advances combined with convolutional neural networks have proved to be a recipe for success, current methods are still lacking on two fronts: (a) data-efficiency of learning and (b) generalization to new environments. To this end, we present Reinforcement Learning with Augmented Data (RAD), a simple plug-and-play module that can enhance most RL algorithms. We perform the first extensive study of general data augmentations for RL on both pixel-based and state-based inputs, and introduce two new data augmentations - random translate and random amplitude scale. We show that augmentations such as random translate, crop, color jitter, patch cutout, random convolutions, and amplitude scale can enable simple RL algorithms to outperform complex state-of-the-art methods across common benchmarks. RAD sets a new state-of-the-art in terms of data-efficiency and final performance on the DeepMind Control Suite benchmark for pixel-based control as well as OpenAI Gym benchmark for state-based control. We further demonstrate that RAD significantly improves test-time generalization over existing methods on several OpenAI ProcGen benchmarks. Our RAD module and training code are available at https://www.github.com/MishaLaskin/rad. (@RAD)

Hojoon Lee, Hanseul Cho, Hyunseung Kim, Daehoon Gwak, Joonkee Kim, Jaegul Choo, Se-Young Yun, and Chulhee Yun Plastic: Improving input and label plasticity for sample efficient reinforcement learning *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** In Reinforcement Learning (RL), enhancing sample efficiency is crucial, particularly in scenarios when data acquisition is costly and risky. In principle, off-policy RL algorithms can improve sample efficiency by allowing multiple updates per environment interaction. However, these multiple updates often lead the model to overfit to earlier interactions, which is referred to as the loss of plasticity. Our study investigates the underlying causes of this phenomenon by dividing plasticity into two aspects. Input plasticity, which denotes the model’s adaptability to changing input data, and label plasticity, which denotes the model’s adaptability to evolving input-output relationships. Synthetic experiments on the CIFAR-10 dataset reveal that finding smoother minima of loss landscape enhances input plasticity, whereas refined gradient propagation improves label plasticity. Leveraging these findings, we introduce the PLASTIC algorithm, which harmoniously combines techniques to address both concerns. With minimal architectural modifications, PLASTIC achieves competitive performance on benchmarks including Atari-100k and Deepmind Control Suite. This result emphasizes the importance of preserving the model’s plasticity to elevate the sample efficiency in RL. The code is available at https://github.com/dojeon-ai/plastic. (@Enhancing_Generalization_Plasticity)

Kuang-Huei Lee, Ian Fischer, Anthony Liu, Yijie Guo, Honglak Lee, John Canny, and Sergio Guadarrama Predictive information accelerates learning in rl *Advances in Neural Information Processing Systems*, 33: 11890–11901, 2020. **Abstract:** The Predictive Information is the mutual information between the past and the future, I(X_past; X_future). We hypothesize that capturing the predictive information is useful in RL, since the ability to model what will happen next is necessary for success on many tasks. To test our hypothesis, we train Soft Actor-Critic (SAC) agents from pixels with an auxiliary task that learns a compressed representation of the predictive information of the RL environment dynamics using a contrastive version of the Conditional Entropy Bottleneck (CEB) objective. We refer to these as Predictive Information SAC (PI-SAC) agents. We show that PI-SAC agents can substantially improve sample efficiency over challenging baselines on tasks from the DM Control suite of continuous control environments. We evaluate PI-SAC agents by comparing against uncompressed PI-SAC agents, other compressed and uncompressed agents, and SAC agents directly trained from pixels. Our implementation is given on GitHub. (@PI-SAC)

Qiyang Li, Aviral Kumar, Ilya Kostrikov, and Sergey Levine Efficient deep reinforcement learning requires regulating overfitting In *The Eleventh International Conference on Learning Representations*, 2022. **Abstract:** Deep reinforcement learning algorithms that learn policies by trial-and-error must learn from limited amounts of data collected by actively interacting with the environment. While many prior works have shown that proper regularization techniques are crucial for enabling data-efficient RL, a general understanding of the bottlenecks in data-efficient RL has remained unclear. Consequently, it has been difficult to devise a universal technique that works well across all domains. In this paper, we attempt to understand the primary bottleneck in sample-efficient deep RL by examining several potential hypotheses such as non-stationarity, excessive action distribution shift, and overfitting. We perform thorough empirical analysis on state-based DeepMind control suite (DMC) tasks in a controlled and systematic way to show that high temporal-difference (TD) error on the validation set of transitions is the main culprit that severely affects the performance of deep RL algorithms, and prior methods that lead to good performance do in fact, control the validation TD error to be low. This observation gives us a robust principle for making deep RL efficient: we can hill-climb on the validation TD error by utilizing any form of regularization techniques from supervised learning. We show that a simple online model selection method that targets the validation TD error is effective across state-based DMC and Gym tasks. (@Regulating_Overfitting)

Xiang Li, Jinghuan Shang, Srijan Das, and Michael Ryoo Does self-supervised learning really improve reinforcement learning from pixels? *Advances in Neural Information Processing Systems*, 35: 30865–30881, 2022. **Abstract:** We investigate whether self-supervised learning (SSL) can improve online reinforcement learning (RL) from pixels. We extend the contrastive reinforcement learning framework (e.g., CURL) that jointly optimizes SSL and RL losses and conduct an extensive amount of experiments with various self-supervised losses. Our observations suggest that the existing SSL framework for RL fails to bring meaningful improvement over the baselines only taking advantage of image augmentation when the same amount of data and augmentation is used. We further perform evolutionary searches to find the optimal combination of multiple self-supervised losses for RL, but find that even such a loss combination fails to meaningfully outperform the methods that only utilize carefully designed image augmentations. After evaluating these approaches together in multiple different environments including a real-world robot environment, we confirm that no single self-supervised loss or image augmentation method can dominate all environments and that the current framework for joint optimization of SSL and RL is limited. Finally, we conduct the ablation study on multiple factors and demonstrate the properties of representations learned with different approaches. (@Does_SSL)

Clare Lyle, Mark Rowland, and Will Dabney Understanding and preventing capacity loss in reinforcement learning In *International Conference on Learning Representations*, 2021. **Abstract:** The reinforcement learning (RL) problem is rife with sources of non-stationarity, making it a notoriously difficult problem domain for the application of neural networks. We identify a mechanism by which non-stationary prediction targets can prevent learning progress in deep RL agents: \\}textit{capacity loss}, whereby networks trained on a sequence of target values lose their ability to quickly update their predictions over time. We demonstrate that capacity loss occurs in a range of RL agents and environments, and is particularly damaging to performance in sparse-reward tasks. We then present a simple regularizer, Initial Feature Regularization (InFeR), that mitigates this phenomenon by regressing a subspace of features towards its value at initialization, leading to significant performance improvements in sparse-reward environments such as Montezuma’s Revenge. We conclude that preventing capacity loss is crucial to enable agents to maximally benefit from the learning signals they obtain throughout the entire training trajectory. (@capacity_loss)

Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, and Will Dabney Understanding plasticity in neural networks In *International Conference on Machine Learning*, pp. 23190–23211. PMLR, 2023. **Abstract:** Plasticity, the ability of a neural network to quickly change its predictions in response to new information, is essential for the adaptability and robustness of deep reinforcement learning systems. Deep neural networks are known to lose plasticity over the course of training even in relatively simple learning problems, but the mechanisms driving this phenomenon are still poorly understood. This paper conducts a systematic empirical analysis into plasticity loss, with the goal of understanding the phenomenon mechanistically in order to guide the future development of targeted solutions. We find that loss of plasticity is deeply connected to changes in the curvature of the loss landscape, but that it often occurs in the absence of saturated units. Based on this insight, we identify a number of parameterization and optimization design choices which enable networks to better preserve plasticity over the course of training. We validate the utility of these findings on larger-scale RL benchmarks in the Arcade Learning Environment. (@understanding_plasticity)

Jiafei Lyu, Le Wan, Zongqing Lu, and Xiu Li Off-policy rl algorithms can be sample-efficient for continuous control via sample multiple reuse *Information Sciences*, pp. 120371, 2024. **Abstract:** Sample efficiency is one of the most critical issues for online reinforcement learning (RL). Existing methods achieve higher sample efficiency by adopting model-based methods, Q-ensemble, or better exploration mechanisms. We, instead, propose to train an off-policy RL agent via updating on a fixed sampled batch multiple times, thus reusing these samples and better exploiting them within a single optimization loop. We name our method sample multiple reuse (SMR). We theoretically show the properties of Q-learning with SMR, e.g., convergence. Furthermore, we incorporate SMR with off-the-shelf off-policy RL algorithms and conduct experiments on a variety of continuous control benchmarks. Empirical results show that SMR significantly boosts the sample efficiency of the base methods across most of the evaluated tasks without any hyperparameter tuning or additional tricks. (@SMR)

Guozheng Ma, Zhen Wang, Zhecheng Yuan, Xueqian Wang, Bo Yuan, and Dacheng Tao A comprehensive survey of data augmentation in visual reinforcement learning *arXiv preprint arXiv:2210.04561*, 2022. **Abstract:** Visual reinforcement learning (RL), which makes decisions directly from high-dimensional visual inputs, has demonstrated significant potential in various domains. However, deploying visual RL techniques in the real world remains challenging due to their low sample efficiency and large generalization gaps. To tackle these obstacles, data augmentation (DA) has become a widely used technique in visual RL for acquiring sample-efficient and generalizable policies by diversifying the training data. This survey aims to provide a timely and essential review of DA techniques in visual RL in recognition of the thriving development in this field. In particular, we propose a unified framework for analyzing visual RL and understanding the role of DA in it. We then present a principled taxonomy of the existing augmentation techniques used in visual RL and conduct an in-depth discussion on how to better leverage augmented data in different scenarios. Moreover, we report a systematic empirical evaluation of DA-based techniques in visual RL and conclude by highlighting the directions for future research. As the first comprehensive survey of DA in visual RL, this work is expected to offer valuable guidance to this emerging field. (@ma2022comprehensive)

Guozheng Ma, Linrui Zhang, Haoyu Wang, Lu Li, Zilin Wang, Zhen Wang, Li Shen, Xueqian Wang, and Dacheng Tao Learning better with less: Effective augmentation for sample-efficient visual reinforcement learning *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Data augmentation (DA) is a crucial technique for enhancing the sample efficiency of visual reinforcement learning (RL) algorithms. Notably, employing simple observation transformations alone can yield outstanding performance without extra auxiliary representation tasks or pre-trained encoders. However, it remains unclear which attributes of DA account for its effectiveness in achieving sample-efficient visual RL. To investigate this issue and further explore the potential of DA, this work conducts comprehensive experiments to assess the impact of DA’s attributes on its efficacy and provides the following insights and improvements: (1) For individual DA operations, we reveal that both ample spatial diversity and slight hardness are indispensable. Building on this finding, we introduce Random PadResize (Rand PR), a new DA operation that offers abundant spatial diversity with minimal hardness. (2) For multi-type DA fusion schemes, the increased DA hardness and unstable data distribution result in the current fusion schemes being unable to achieve higher sample efficiency than their corresponding individual operations. Taking the non-stationary nature of RL into account, we propose a RL-tailored multi-type DA fusion scheme called Cycling Augmentation (CycAug), which performs periodic cycles of different DA operations to increase type diversity while maintaining data distribution consistency. Extensive evaluations on the DeepMind Control suite and CARLA driving simulator demonstrate that our methods achieve superior sample efficiency compared with the prior state-of-the-art methods. (@ma2023learning)

Bogdan Mazoure, Remi Tachet des Combes, Thang Long Doan, Philip Bachman, and R Devon Hjelm Deep reinforcement and infomax learning In *Advances in Neural Information Processing Systems*, 2020. **Abstract:** We begin with the hypothesis that a model-free agent whose representations are predictive of properties of future states (beyond expected rewards) will be more capable of solving and adapting to new RL problems. To test that hypothesis, we introduce an objective based on Deep InfoMax (DIM) which trains the agent to predict the future by maximizing the mutual information between its internal representation of successive timesteps. We test our approach in several synthetic settings, where it successfully learns representations that are predictive of the future. Finally, we augment C51, a strong RL baseline, with our temporal DIM objective and demonstrate improved performance on a continual learning task and on the recently introduced Procgen environment. (@DRIML)

Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida Spectral normalization for generative adversarial networks In *International Conference on Learning Representations*, 2018. **Abstract:** One of the challenges in the study of generative adversarial networks is the instability of its training. In this paper, we propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator. Our new normalization technique is computationally light and easy to incorporate into existing implementations. We tested the efficacy of spectral normalization on CIFAR10, STL-10, and ILSVRC2012 dataset, and we experimentally confirmed that spectrally normalized GANs (SN-GANs) is capable of generating images of better or equal quality relative to the previous training stabilization techniques. (@miyato2018spectral)

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta R3m: A universal visual representation for robot manipulation In *Conference on Robot Learning*, pp. 892–909. PMLR, 2023. **Abstract:** We study how visual representations pre-trained on diverse human video data can enable data-efficient learning of downstream robotic manipulation tasks. Concretely, we pre-train a visual representation using the Ego4D human video dataset using a combination of time-contrastive learning, video-language alignment, and an L1 penalty to encourage sparse and compact representations. The resulting representation, R3M, can be used as a frozen perception module for downstream policy learning. Across a suite of 12 simulated robot manipulation tasks, we find that R3M improves task success by over 20% compared to training from scratch and by over 10% compared to state-of-the-art visual representations like CLIP and MoCo. Furthermore, R3M enables a Franka Emika Panda arm to learn a range of manipulation tasks in a real, cluttered apartment given just 20 demonstrations. Code and pre-trained models are available at https://tinyurl.com/robotr3m. (@R3M)

Thanh Nguyen, Tung M Luu, Thang Vu, and Chang D Yoo Sample-efficient reinforcement learning representation learning with curiosity contrastive forward dynamics model In *2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 3471–3477. IEEE, 2021. **Abstract:** Developing an agent in reinforcement learning (RL) that is capable of performing complex control tasks directly from high-dimensional observation such as raw pixels is a challenge as efforts still need to be made towards improving sample efficiency and generalization of RL algorithm. This paper considers a learning framework for a Curiosity Contrastive Forward Dynamics Model (CCFDM) to achieve a more sample-efficient RL based directly on raw pixels. CCFDM incorporates a forward dynamics model (FDM) and performs contrastive learning to train its deep convolutional neural network-based image encoder (IE) to extract conducive spatial and temporal information to achieve a more sample efficiency for RL. In addition, during training, CCFDM provides intrinsic rewards, produced based on FDM prediction error, and encourages the curiosity of the RL agent to improve exploration. The diverge and less-repetitive observations provided by both our exploration strategy and data augmentation available in contrastive learning improve not only the sample efficiency but also the generalization . Performance of existing model-free RL methods such as Soft Actor-Critic built on top of CCFDM outperforms prior state-of-the-art pixel-based RL methods on the DeepMind Control Suite benchmark. (@CCFDM)

Evgenii Nikishin, Max Schwarzer, Pierluca D’Oro, Pierre-Luc Bacon, and Aaron Courville The primacy bias in deep reinforcement learning In *International conference on machine learning*, pp. 16828–16847. PMLR, 2022. **Abstract:** This work identifies a common flaw of deep reinforcement learning (RL) algorithms: a tendency to rely on early interactions and ignore useful evidence encountered later. Because of training on progressively growing datasets, deep RL agents incur a risk of overfitting to earlier experiences, negatively affecting the rest of the learning process. Inspired by cognitive science, we refer to this effect as the primacy bias. Through a series of experiments, we dissect the algorithmic aspects of deep RL that exacerbate this bias. We then propose a simple yet generally-applicable mechanism that tackles the primacy bias by periodically resetting a part of the agent. We apply this mechanism to algorithms in both discrete (Atari 100k) and continuous action (DeepMind Control Suite) domains, consistently improving their performance. (@primacy_bias)

Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, and André Barreto Deep reinforcement learning with plasticity injection *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** A growing body of evidence suggests that neural networks employed in deep reinforcement learning (RL) gradually lose their plasticity, the ability to learn from new data; however, the analysis and mitigation of this phenomenon is hampered by the complex relationship between plasticity, exploration, and performance in RL. This paper introduces plasticity injection, a minimalistic intervention that increases the network plasticity without changing the number of trainable parameters or biasing the predictions. The applications of this intervention are two-fold: first, as a diagnostic tool $\\}unicode{x2014}$ if injection increases the performance, we may conclude that an agent’s network was losing its plasticity. This tool allows us to identify a subset of Atari environments where the lack of plasticity causes performance plateaus, motivating future studies on understanding and combating plasticity loss. Second, plasticity injection can be used to improve the computational efficiency of RL training if the agent has to re-learn from scratch due to exhausted plasticity or by growing the agent’s network dynamically without compromising performance. The results on Atari show that plasticity injection attains stronger performance compared to alternative methods while being computationally efficient. (@Plasticity_Injection)

Aaron van den Oord, Yazhe Li, and Oriol Vinyals Representation learning with contrastive predictive coding *arXiv preprint arXiv:1807.03748*, 2018. **Abstract:** While supervised learning has enabled great progress in many applications, unsupervised learning has not seen such widespread adoption, and remains an important and challenging endeavor for artificial intelligence. In this work, we propose a universal unsupervised learning approach to extract useful representations from high-dimensional data, which we call Contrastive Predictive Coding. The key insight of our model is to learn such representations by predicting the future in latent space by using powerful autoregressive models. We use a probabilistic contrastive loss which induces the latent space to capture information that is maximally useful to predict future samples. It also makes the model tractable by using negative sampling. While most prior work has focused on evaluating representations for a particular modality, we demonstrate that our approach is able to learn useful representations achieving strong performance on four distinct domains: speech, images, text and reinforcement learning in 3D environments. (@CPC)

OpenAI Gpt-4 technical report *ArXiv*, abs/2303.08774, 2023. **Abstract:** We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4’s performance based on models trained with no more than 1/1,000th the compute of GPT-4. (@GPT4TR)

Simone Parisi, Aravind Rajeswaran, Senthil Purushwalkam, and Abhinav Gupta The unsurprising effectiveness of pre-trained vision models for control In *International Conference on Machine Learning*, pp. 17359–17371. PMLR, 2022. **Abstract:** Recent years have seen the emergence of pre-trained representations as a powerful abstraction for AI applications in computer vision, natural language, and speech. However, policy learning for control is still dominated by a tabula-rasa learning paradigm, with visuo-motor policies often trained from scratch using data from deployment environments. In this context, we revisit and study the role of pre-trained visual representations for control, and in particular representations trained on large-scale computer vision datasets. Through extensive empirical evaluation in diverse control domains (Habitat, DeepMind Control, Adroit, Franka Kitchen), we isolate and study the importance of different representation training methods, data augmentations, and feature hierarchies. Overall, we find that pre-trained visual representations can be competitive or even better than ground-truth state representations to train control policies. This is in spite of using only out-of-domain data from standard vision datasets, without any in-domain data from the deployment environments. Source code and more at https://sites.google.com/view/pvr-control. (@PVR)

Max Schwarzer, Ankesh Anand, Rishab Goel, R Devon Hjelm, Aaron Courville, and Philip Bachman Data-efficient reinforcement learning with self-predictive representations In *International Conference on Learning Representations*, 2020. **Abstract:** While deep reinforcement learning excels at solving tasks where large amounts of data can be collected through virtually unlimited interaction with the environment, learning from limited interaction remains a key challenge. We posit that an agent can learn more efficiently if we augment reward maximization with self-supervised objectives based on structure in its visual input and sequential interaction with the environment. Our method, Self-Predictive Representations(SPR), trains an agent to predict its own latent state representations multiple steps into the future. We compute target representations for future states using an encoder which is an exponential moving average of the agent’s parameters and we make predictions using a learned transition model. On its own, this future prediction objective outperforms prior methods for sample-efficient deep RL from pixels. We further improve performance by adding data augmentation to the future prediction loss, which forces the agent’s representations to be consistent across multiple views of an observation. Our full self-supervised objective, which combines future prediction and data augmentation, achieves a median human-normalized score of 0.415 on Atari in a setting limited to 100k steps of environment interaction, which represents a 55% relative improvement over the previous state-of-the-art. Notably, even in this limited data regime, SPR exceeds expert human scores on 7 out of 26 games. The code associated with this work is available at https://github.com/mila-iqia/spr (@SPR)

Max Schwarzer, Johan Samir Obando Ceron, Aaron Courville, Marc G Bellemare, Rishabh Agarwal, and Pablo Samuel Castro Bigger, better, faster: Human-level atari with human-level efficiency In *International Conference on Machine Learning*, pp. 30365–30380. PMLR, 2023. **Abstract:** We introduce a value-based RL agent, which we call BBF, that achieves super-human performance in the Atari 100K benchmark. BBF relies on scaling the neural networks used for value estimation, as well as a number of other design choices that enable this scaling in a sample-efficient manner. We conduct extensive analyses of these design choices and provide insights for future work. We end with a discussion about updating the goalposts for sample-efficient RL research on the ALE. We make our code and data publicly available at https://github.com/google-research/google-research/tree/master/bigger_better_faster. (@BBF)

Rutav Shah and Vikash Kumar Rrl: Resnet as representation for reinforcement learning In *International Conference on Machine Learning*. PMLR, 2021. **Abstract:** The ability to autonomously learn behaviors via direct interactions in uninstrumented environments can lead to generalist robots capable of enhancing productivity or providing care in unstructured settings like homes. Such uninstrumented settings warrant operations only using the robot’s proprioceptive sensor such as onboard cameras, joint encoders, etc which can be challenging for policy learning owing to the high dimensionality and partial observability issues. We propose RRL: Resnet as representation for Reinforcement Learning – a straightforward yet effective approach that can learn complex behaviors directly from proprioceptive inputs. RRL fuses features extracted from pre-trained Resnet into the standard reinforcement learning pipeline and delivers results comparable to learning directly from the state. In a simulated dexterous manipulation benchmark, where the state of the art methods fail to make significant progress, RRL delivers contact rich behaviors. The appeal of RRL lies in its simplicity in bringing together progress from the fields of Representation Learning, Imitation Learning, and Reinforcement Learning. Its effectiveness in learning behaviors directly from visual inputs with performance and sample efficiency matching learning directly from the state, even in complex high dimensional domains, is far from obvious. (@RRL)

Wenling Shang, Kihyuk Sohn, Diogo Almeida, and Honglak Lee Understanding and improving convolutional neural networks via concatenated rectified linear units In *international conference on machine learning*, pp. 2217–2225. PMLR, 2016. **Abstract:** Recently, convolutional neural networks (CNNs) have been used as a powerful tool to solve many problems of machine learning and computer vision. In this paper, we aim to provide insight on the property of convolutional neural networks, as well as a generic method to improve the performance of many CNN architectures. Specifically, we first examine existing CNN models and observe an intriguing property that the filters in the lower layers form pairs (i.e., filters with opposite phase). Inspired by our observation, we propose a novel, simple yet effective activation scheme called concatenated ReLU (CRelu) and theoretically analyze its reconstruction property in CNNs. We integrate CRelu into several state-of-the-art CNN architectures and demonstrate improvement in their recognition performance on CIFAR-10/100 and ImageNet datasets with fewer trainable parameters. Our results suggest that better understanding of the properties of CNNs can lead to significant performance improvement with a simple modification. (@shang2016understanding)

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al Mastering the game of go without human knowledge *nature*, 550 (7676): 354–359, 2017. **Abstract:** A long-standing goal of artificial intelligence is an algorithm that learns, tabula rasa , superhuman proficiency in challenging domains. Recently, AlphaGo became the first program to defeat a world champion in the game of Go. The tree search in AlphaGo evaluated positions and selected moves using deep neural networks. These neural networks were trained by supervised learning from human expert moves, and by reinforcement learning from self-play. Here we introduce an algorithm based solely on reinforcement learning, without human data, guidance or domain knowledge beyond game rules. AlphaGo becomes its own teacher: a neural network is trained to predict AlphaGo’s own move selections and also the winner of AlphaGo’s games. This neural network improves the strength of the tree search, resulting in higher quality move selection and stronger self-play in the next iteration. Starting tabula rasa , our new program AlphaGo Zero achieved superhuman performance, winning 100–0 against the previously published, champion-defeating AlphaGo. (@AlphaGo_Zero)

Laura Smith, Ilya Kostrikov, and Sergey Levine A walk in the park: Learning to walk in 20 minutes with model-free reinforcement learning *arXiv preprint arXiv:2208.07860*, 2022. **Abstract:** Deep reinforcement learning is a promising approach to learning policies in uncontrolled environments that do not require domain knowledge. Unfortunately, due to sample inefficiency, deep RL applications have primarily focused on simulated environments. In this work, we demonstrate that the recent advancements in machine learning algorithms and libraries combined with a carefully tuned robot controller lead to learning quadruped locomotion in only 20 minutes in the real world. We evaluate our approach on several indoor and outdoor terrains which are known to be challenging for classical model-based controllers. We observe the robot to be able to learn walking gait consistently on all of these terrains. Finally, we evaluate our design decisions in a simulated environment. (@smith2022walk)

Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, and Utku Evci The dormant neuron phenomenon in deep reinforcement learning In *International Conference on Machine Learning*, pp. 32145–32168. PMLR, 2023. **Abstract:** In this work we identify the dormant neuron phenomenon in deep reinforcement learning, where an agent’s network suffers from an increasing number of inactive neurons, thereby affecting network expressivity. We demonstrate the presence of this phenomenon across a variety of algorithms and environments, and highlight its effect on learning. To address this issue, we propose a simple and effective method (ReDo) that Recycles Dormant neurons throughout training. Our experiments demonstrate that ReDo maintains the expressive power of networks by reducing the number of dormant neurons and results in improved performance. (@dormant_neuron)

Adam Stooke, Kimin Lee, Pieter Abbeel, and Michael Laskin Decoupling representation learning from reinforcement learning In *International Conference on Machine Learning*, pp. 9870–9879. PMLR, 2021. **Abstract:** In an effort to overcome limitations of reward-driven feature learning in deep reinforcement learning (RL) from images, we propose decoupling representation learning from policy learning. To this end, we introduce a new unsupervised learning (UL) task, called Augmented Temporal Contrast (ATC), which trains a convolutional encoder to associate pairs of observations separated by a short time difference, under image augmentations and using a contrastive loss. In online RL experiments, we show that training the encoder exclusively using ATC matches or outperforms end-to-end RL in most environments. Additionally, we benchmark several leading UL algorithms by pre-training encoders on expert demonstrations and using them, with weights frozen, in RL agents; we find that agents using ATC-trained encoders outperform all others. We also train multi-task encoders on data from multiple environments and show generalization to different downstream RL tasks. Finally, we ablate components of ATC, and introduce a new data augmentation to enable replay of (compressed) latent images from pre-trained encoders when RL requires augmentation. Our experiments span visually diverse RL benchmarks in DeepMind Control, DeepMind Lab, and Atari, and our complete code is available at this https URL. (@ATC)

Chenyu Sun, Hangwei Qian, and Chunyan Miao Cclf: A contrastive-curiosity-driven learning framework for sample-efficient reinforcement learning *arXiv preprint arXiv:2205.00943*, 2022. **Abstract:** In reinforcement learning (RL), it is challenging to learn directly from high-dimensional observations, where data augmentation has recently been shown to remedy this via encoding invariances from raw pixels. Nevertheless, we empirically find that not all samples are equally important and hence simply injecting more augmented inputs may instead cause instability in Q-learning. In this paper, we approach this problem systematically by developing a model-agnostic Contrastive-Curiosity-Driven Learning Framework (CCLF), which can fully exploit sample importance and improve learning efficiency in a self-supervised manner. Facilitated by the proposed contrastive curiosity, CCLF is capable of prioritizing the experience replay, selecting the most informative augmented inputs, and more importantly regularizing the Q-function as well as the encoder to concentrate more on under-learned data. Moreover, it encourages the agent to explore with a curiosity-based reward. As a result, the agent can focus on more informative samples and learn representation invariances more efficiently, with significantly reduced augmented inputs. We apply CCLF to several base RL algorithms and evaluate on the DeepMind Control Suite, Atari, and MiniGrid benchmarks, where our approach demonstrates superior sample efficiency and learning performances compared with other state-of-the-art methods. (@CCLF)

Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al Deepmind control suite *arXiv preprint arXiv:1801.00690*, 2018. **Abstract:** The DeepMind Control Suite is a set of continuous control tasks with a standardised structure and interpretable rewards, intended to serve as performance benchmarks for reinforcement learning agents. The tasks are written in Python and powered by the MuJoCo physics engine, making them easy to use and modify. We include benchmarks for several learning algorithms. The Control Suite is publicly available at https://www.github.com/deepmind/dm_control . A video summary of all tasks is available at http://youtu.be/rAai4QzcYbs . (@DMC_suite)

Manan Tomar, Utkarsh Aashu Mishra, Amy Zhang, and Matthew E Taylor Learning representations for pixel-based control: What matters and why? *Transactions on Machine Learning Research*, 2022. **Abstract:** Learning representations for pixel-based control has garnered significant attention recently in reinforcement learning. A wide range of methods have been proposed to enable efficient learning, leading to sample complexities similar to those in the full state setting. However, moving beyond carefully curated pixel data sets (centered crop, appropriate lighting, clear background, etc.) remains challenging. In this paper, we adopt a more difficult setting, incorporating background distractors, as a first step towards addressing this challenge. We present a simple baseline approach that can learn meaningful representations with no metric-based learning, no data augmentations, no world-model learning, and no contrastive learning. We then analyze when and why previously proposed methods are likely to fail or reduce to the same performance as the baseline in this harder setting and why we should think carefully about extending such methods beyond the well curated environments. Our results show that finer categorization of benchmarks on the basis of characteristics like density of reward, planning horizon of the problem, presence of task-irrelevant components, etc., is crucial in evaluating algorithms. Based on these observations, we propose different metrics to consider when evaluating an algorithm on benchmark tasks. We hope such a data-centric view can motivate researchers to rethink representation learning when investigating how to best apply RL to real-world tasks. (@tomar2021learning)

Che Wang, Xufang Luo, Keith Ross, and Dongsheng Li Vrl3: A data-driven framework for visual deep reinforcement learning *Advances in Neural Information Processing Systems*, 35: 32974–32988, 2022. **Abstract:** We propose VRL3, a powerful data-driven framework with a simple design for solving challenging visual deep reinforcement learning (DRL) tasks. We analyze a number of major obstacles in taking a data-driven approach, and present a suite of design principles, novel findings, and critical insights about data-driven visual DRL. Our framework has three stages: in stage 1, we leverage non-RL datasets (e.g. ImageNet) to learn task-agnostic visual representations; in stage 2, we use offline RL data (e.g. a limited number of expert demonstrations) to convert the task-agnostic representations into more powerful task-specific representations; in stage 3, we fine-tune the agent with online RL. On a set of challenging hand manipulation tasks with sparse reward and realistic visual inputs, compared to the previous SOTA, VRL3 achieves an average of 780% better sample efficiency. And on the hardest task, VRL3 is 1220% more sample efficient (2440% when using a wider encoder) and solves the task with only 10% of the computation. These significant results clearly demonstrate the great potential of data-driven deep reinforcement learning. (@wang2022vrl3)

Tete Xiao, Ilija Radosavovic, Trevor Darrell, and Jitendra Malik Masked visual pre-training for motor control *arXiv preprint arXiv:2203.06173*, 2022. **Abstract:** This paper shows that self-supervised visual pre-training from real-world images is effective for learning motor control tasks from pixels. We first train the visual representations by masked modeling of natural images. We then freeze the visual encoder and train neural network controllers on top with reinforcement learning. We do not perform any task-specific fine-tuning of the encoder; the same visual representations are used for all motor control tasks. To the best of our knowledge, this is the first self-supervised model to exploit real-world images at scale for motor control. To accelerate progress in learning from pixels, we contribute a benchmark suite of hand-designed tasks varying in movements, scenes, and robots. Without relying on labels, state-estimation, or expert demonstrations, we consistently outperform supervised encoders by up to 80% absolute success rate, sometimes even matching the oracle state performance. We also find that in-the-wild images, e.g., from YouTube or Egocentric videos, lead to better visual representations for various manipulation tasks than ImageNet images. (@MVP)

Denis Yarats, Ilya Kostrikov, and Rob Fergus Image augmentation is all you need: Regularizing deep reinforcement learning from pixels In *International conference on learning representations*, 2020. **Abstract:** We propose a simple data augmentation technique that can be applied to standard model-free reinforcement learning algorithms, enabling robust learning directly from pixels without the need for auxiliary losses or pre-training. The approach leverages input perturbations commonly used in computer vision tasks to regularize the value function. Existing model-free approaches, such as Soft Actor-Critic (SAC), are not able to train deep networks effectively from image pixels. However, the addition of our augmentation method dramatically improves SAC’s performance, enabling it to reach state-of-the-art performance on the DeepMind control suite, surpassing model-based (Dreamer, PlaNet, and SLAC) methods and recently proposed contrastive learning (CURL). Our approach can be combined with any model-free reinforcement learning algorithm, requiring only minor modifications. An implementation can be found at https://sites.google.com/view/data-regularized-q. (@drq)

Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto Mastering visual continuous control: Improved data-augmented reinforcement learning In *International Conference on Learning Representations*, 2021. **Abstract:** We present DrQ-v2, a model-free reinforcement learning (RL) algorithm for visual continuous control. DrQ-v2 builds on DrQ, an off-policy actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several improvements that yield state-of-the-art results on the DeepMind Control Suite. Notably, DrQ-v2 is able to solve complex humanoid locomotion tasks directly from pixel observations, previously unattained by model-free RL. DrQ-v2 is conceptually simple, easy to implement, and provides significantly better computational footprint compared to prior work, with the majority of tasks taking just 8 hours to train on a single GPU. Finally, we publicly release DrQ-v2’s implementation to provide RL practitioners with a strong and computationally efficient baseline. (@DrQ-v2)

Denis Yarats, Amy Zhang, Ilya Kostrikov, Brandon Amos, Joelle Pineau, and Rob Fergus Improving sample efficiency in model-free reinforcement learning from images In *Proceedings of the AAAI Conference on Artificial Intelligence*, pp. 10674–10681, 2021. **Abstract:** Training an agent to solve control tasks directly from high-dimensional images with model-free reinforcement learning (RL) has proven difficult. A promising approach is to learn a latent representation together with the control policy. However, fitting a high-capacity encoder using a scarce reward signal is sample inefficient and leads to poor performance. Prior work has shown that auxiliary losses, such as image reconstruction, can aid efficient representation learning. However, incorporating reconstruction loss into an off-policy learning algorithm often leads to training instability. We explore the underlying reasons and identify variational autoencoders, used by previous investigations, as the cause of the divergence. Following these findings, we propose effective techniques to improve training stability. This results in a simple approach capable of matching state-of-the-art model-free and model-based algorithms on MuJoCo control tasks. Furthermore, our approach demonstrates robustness to observational noise, surpassing existing approaches in this setting. Code, results, and videos are anonymously available at https://sites.google.com/view/sac-ae/home. (@SAC-AE)

Tao Yu, Cuiling Lan, Wenjun Zeng, Mingxiao Feng, Zhizheng Zhang, and Zhibo Chen Playvirtual: Augmenting cycle-consistent virtual trajectories for reinforcement learning *Advances in Neural Information Processing Systems*, 34: 5276–5289, 2021. **Abstract:** Learning good feature representations is important for deep reinforcement learning (RL). However, with limited experience, RL often suffers from data inefficiency for training. For un-experienced or less-experienced trajectories (i.e., state-action sequences), the lack of data limits the use of them for better feature learning. In this work, we propose a novel method, dubbed PlayVirtual, which augments cycle-consistent virtual trajectories to enhance the data efficiency for RL feature representation learning. Specifically, PlayVirtual predicts future states in the latent space based on the current state and action by a dynamics model and then predicts the previous states by a backward dynamics model, which forms a trajectory cycle. Based on this, we augment the actions to generate a large amount of virtual state-action trajectories. Being free of groudtruth state supervision, we enforce a trajectory to meet the cycle consistency constraint, which can significantly enhance the data efficiency. We validate the effectiveness of our designs on the Atari and DeepMind Control Suite benchmarks. Our method achieves the state-of-the-art performance on both benchmarks. (@Playvirtual)

Tao Yu, Zhizheng Zhang, Cuiling Lan, Yan Lu, and Zhibo Chen Mask-based latent reconstruction for reinforcement learning *Advances in Neural Information Processing Systems*, 35: 25117–25131, 2022. **Abstract:** For deep reinforcement learning (RL) from pixels, learning effective state representations is crucial for achieving high performance. However, in practice, limited experience and high-dimensional inputs prevent effective representation learning. To address this, motivated by the success of mask-based modeling in other research fields, we introduce mask-based reconstruction to promote state representation learning in RL. Specifically, we propose a simple yet effective self-supervised method, Mask-based Latent Reconstruction (MLR), to predict complete state representations in the latent space from the observations with spatially and temporally masked pixels. MLR enables better use of context information when learning state representations to make them more informative, which facilitates the training of RL agents. Extensive experiments show that our MLR significantly improves the sample efficiency in RL and outperforms the state-of-the-art sample-efficient RL methods on multiple continuous and discrete control benchmarks. Our code is available at https://github.com/microsoft/Mask-based-Latent-Reconstruction. (@MLR)

Zhecheng Yuan, Zhengrong Xue, Bo Yuan, Xueqian Wang, Yi Wu, Yang Gao, and Huazhe Xu Pre-trained image encoder for generalizable visual reinforcement learning *Advances in Neural Information Processing Systems*, 35: 13022–13037, 2022. **Abstract:** Learning generalizable policies that can adapt to unseen environments remains challenging in visual Reinforcement Learning (RL). Existing approaches try to acquire a robust representation via diversifying the appearances of in-domain observations for better generalization. Limited by the specific observations of the environment, these methods ignore the possibility of exploring diverse real-world image datasets. In this paper, we investigate how a visual RL agent would benefit from the off-the-shelf visual representations. Surprisingly, we find that the early layers in an ImageNet pre-trained ResNet model could provide rather generalizable representations for visual RL. Hence, we propose Pre-trained Image Encoder for Generalizable visual reinforcement learning (PIE-G), a simple yet effective framework that can generalize to the unseen visual scenarios in a zero-shot manner. Extensive experiments are conducted on DMControl Generalization Benchmark, DMControl Manipulation Tasks, Drawer World, and CARLA to verify the effectiveness of PIE-G. Empirical evidence suggests PIE-G improves sample efficiency and significantly outperforms previous state-of-the-art methods in terms of generalization performance. In particular, PIE-G boasts a 55% generalization performance gain on average in the challenging video background setting. Project Page: https://sites.google.com/view/pie-g/home. (@yuan2022pre)

Jinhua Zhu, Yingce Xia, Lijun Wu, Jiajun Deng, Wengang Zhou, Tao Qin, Tie-Yan Liu, and Houqiang Li Masked contrastive representation learning for reinforcement learning *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2022. **Abstract:** In pixel-based reinforcement learning (RL), the states are raw video frames, which are mapped into hidden representation before feeding to a policy network. To improve sample efficiency of state representation learning, recently, the most prominent work is based on contrastive unsupervised representation. Witnessing that consecutive video frames in a game are highly correlated, to further improve data efficiency, we propose a new algorithm, i.e., masked contrastive representation learning for RL (M-CURL), which takes the correlation among consecutive inputs into consideration. In our architecture, besides a CNN encoder for hidden presentation of input state and a policy network for action selection, we introduce an auxiliary Transformer encoder module to leverage the correlations among video frames. During training, we randomly mask the features of several frames, and use the CNN encoder and Transformer to reconstruct them based on context frames. The CNN encoder and Transformer are jointly trained via contrastive learning where the reconstructed features should be similar to the ground-truth ones while dissimilar to others. During policy evaluation, the CNN encoder and the policy network are used to take actions, and the Transformer module is discarded. Our method achieves consistent improvements over CURL on 14 out of 16 environments from DMControl suite and 23 out of 26 environments from Atari 2600 Games. The code is available at https://github.com/teslacool/m-curl. (@M-CURL)

</div>

[^1]: Equal Contribution, $`^{\dag}`$Corresponding authors.

[^2]: Our code is available at: <https://github.com/Guozheng-Ma/Adaptive-Replay-Ratio>
