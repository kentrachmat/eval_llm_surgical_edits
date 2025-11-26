# On Giant’s Shoulders:  
Effortless Weak to Strong by Dynamic Logits Fusion

## Abstract

Efficient fine-tuning of large language models for task-specific applications is imperative, yet the vast number of parameters in these models makes their training increasingly challenging. Despite numerous proposals for effective methods, a substantial memory overhead remains for gradient computations during updates. *Can we fine-tune a series of task-specific small models and transfer their knowledge directly to a much larger model without additional training?* In this paper, we explore weak-to-strong specialization using logit arithmetic, facilitating a direct answer to this question. Existing weak-to-strong methods often employ a static knowledge transfer ratio and a single small model for transferring complex knowledge, which leads to suboptimal performance. To surmount these limitations, we propose a dynamic logit fusion approach that works with a series of task-specific small models, each specialized in a different task. This method adaptively allocates weights among these models at each decoding step, learning the weights through Kullback-Leibler divergence constrained optimization problems. We conduct extensive experiments across various benchmarks in both single-task and multi-task settings, achieving leading results. By transferring expertise from the 7B model to the 13B model, our method closes the performance gap by 96.4% in single-task scenarios and by 86.3% in multi-task scenarios compared to full fine-tuning of the 13B model. Notably, we achieve surpassing performance on unseen tasks. Moreover, we further demonstrate that our method can effortlessly integrate in-context learning for single tasks and task arithmetic for multi-task scenarios.

> **“If I have seen further than others, it is by standing on the shoulders of giants.” *— Isaac Newton in 1675***

# Introduction

In recent years, Large Language Models (LLMs) have shown impressive performance in a wide range of tasks `\cite{gpt3,llama2,su2024living,su2024timo,lu2024mitigating,su2024conflictbank,zhu2024dynamic,zhu2024llama}`{=latex}, including code generation `\cite{guo2024deepseekcoder,rozière2024code}`{=latex}, mathematical reasoning `\cite{luo2023wizardmath,azerbayev2024llemma}`{=latex}, tool-use abilities `\cite{tang2023toolalpaca,schick2024toolformer}`{=latex}, *etc.* However, training such LLMs requires substantial computational resources, often involving thousands of GPUs and processing trillions of tokens `\cite{liebenwein2021lost,patterson2021carbon}`{=latex}, making the adaptation of the base model for new knowledge inefficient. To address these challenges, parameter-efficient tuning methods `\cite{hu2021lora,kopiczko2024vera,hayou2024lora+}`{=latex} have emerged, aiming to achieve comparable performance to full fine-tuning while reducing GPU memory requirements. However, challenges persist in tuning and deploying large-scale models on common hardware, as they still involve computation-intensive processes like gradient calculation and back-propagation. Furthermore, these methods may not be feasible when training data are private.

This inspires us to ask: *Can we fine-tune only small models and then transfer their knowledge to a much larger model without requiring additional gradient updates?* If we could fuse the strong capabilities of a scaled LLM with the specialized knowledge acquired by a small model during fine-tuning, it would yield the practical benefit of approximating the results achieved by fine-tuning a large model, but without the associated computational costs. However, it is non-trivial due to *the differences in representation width and layer numbers* between the small and large models.

<figure id="fig:intro">
<img src="./figures/intro2.png"" style="width:95.0%" />
<figcaption>Comparison between our work and previous work. Previous methods only use pre-tuned parameters <span class="math inline"><em>α</em></span> to transfer knowledge from a single expert. In contrast, our method dynamically adjusts the proportion of knowledge transferred from multiple experts at each decoding step during inference.</figcaption>
</figure>

Recently, `\citet{mitchell2024an}`{=latex} and `\citet{liu2024tuning}`{=latex} attempt to address this challenge by transferring knowledge from a Small Language Model (SLM) to its larger counterpart through simple logit arithmetic operations during decoding. For instance, using models from the Llama-2 family, they can transfer the fine-tuned knowledge from a 7B-scale model to a 13B-scale model by performing log probability algebra: Llama-2-base 13B + (Llama-2-chat 7B - Llama-2-base 7B), where the first term represents the base log probabilities and the term in parentheses denotes the behavioral delta `\cite{mitchell2024an}`{=latex}. This behavioral delta can be weighted to adjust the balance between the pretrained knowledge and the transferred fine-tuned knowledge.

Despite showing promise, logit arithmetic still exhibits a noticeable performance gap compared to directly fine-tuning large models, primarily due to two reasons: <u>*Firstly*</u>, they statically prespecifies the weight of behavioral delta at each decoding step identically. However, the importance of fine-tuned knowledge varies significantly across different tasks, inputs, and even different decoding steps. For instance, in a domain-specific question-answering process, we need more knowledge from fine-tuned small language models. Conversely, when decoding factual topics, we may need more knowledge from pretrained general models. <u>*Secondly*</u>, for unseen tasks, the lack of pre-adjusted weights for the behavioral delta prevents logit arithmetic from executing effectively, making it challenging to transfer tuned knowledge, especially for complex tasks. As shown in Figure <a href="#fig:intro" data-reference-type="ref" data-reference="fig:intro">1</a>, when answering “Peter is 76 years old this year. How old will he be in 500 years?”, using only the math expert does not ensure the result aligns with factual accuracy. Additionally, these techniques often assume that experts are trained and tested on the same data distribution, ignoring the heterogeneity of data that may be encountered at test time, rendering any single expert insufficient.

In addressing these challenges, we reconsider the practice of logit arithmetic within a new framework. Specifically, we work with a set of finely tuned SLM experts, each specializing in different tasks. At each decoding step, we dynamically allocate weights among these task-specific SLMs. However, it is non-trivial to effectively determine suitable weights for diverse tasks and SLM experts, as tranditional approaches such as grid search `\cite{liashchynskyi2019grid}`{=latex} or combinatorial search `\cite{liu2020versatile}`{=latex} suffer from costly search processes. To practically automate weight allocation, we reframe the problem of weight search as a constrained distribution optimization problem. We tackle the fusion of multi-task knowledge by treating it as a centroid problem in Euclidean space using Kullback-Leibler divergence `\cite{kullback1951information}`{=latex}, which offers interpretability and explicit guidance.

We conduct thorough experiments on the LLaMA series to evaluate the effectiveness of our approach, applying our adaptive logit arithmetic method to task-specific fine-tuning in math, question-answering, summarization, and multi-domain tasks. We also analyze the performance of multi-model fusion across seen and unseen tasks. Additionally, we discussed the feasibility of combining our method with in-context learning for single tasks and with task arithmetic for multi-task scenarios.

In our work, we make several key contributions. Firstly, we reassess existing logit arithmetic methods, highlighting the significant impact of fusion weights and the limitations imposed by a single small model on test performance. Secondly, we introduce a novel approach that autonomously learns fusion weights through constraint optimization, approximating the compute-intensive results of fine-tuning a large base model. Lastly, we conduct comprehensive experiments to validate our method, demonstrating substantial improvements in performance, generalization capabilities, and robustness.

# Related Work

## Efficient Specialization

Specializing a pretrained model by fine-tuning for downstream tasks has become a primary paradigm `\cite{devlin-etal-2019-bert,JMLR:v21:20-074,chung2024scaling,Fan_Wei_Qu_Lu_Xie_Cheng_Chen_2024}`{=latex}. However, with the increasing scale of models, full fine-tuning has become impractical due to the large number of parameters, requiring costly training resources `\cite{gpt3,llama2}`{=latex}.

To tackle this challenge, researchers have developed parameter-efficient tuning methods `\cite{lialin2023scaling}`{=latex}. These methods aim to achieve performance similar to full fine-tuning while reducing the GPU memory required. They typically involve freezing the original weights and adding task-specific trainable modules `\cite{houlsby2019parameter,karimi-mahabadi-etal-2021-parameter,li-liang-2021-prefix}`{=latex}, low-rank matrices `\cite{hu2021lora,kopiczko2024vera,hayou2024lora+}`{=latex}, or bias `\cite{ben-zaken-etal-2022-bitfit}`{=latex}. Despite their benefits, these approaches still demand significant memory for updating gradients and might not be suitable for scenarios where training data are private or inaccessible. Another direction is model merging `\cite{wang2024fusing,ilharco2023editing,yadav2023tiesmerging,matena2022merging,lu2024twin}`{=latex}, which trains task-specific models on different domains, and then combines them into a single model at deployment with weight average `\cite{wortsman2022model}`{=latex}, neuron permutation `\cite{ainsworth2023git}`{=latex}, interpolation `\cite{2022MergingModelsFisherWeighted,2023DatalessKnowledgeFusion}`{=latex} or task vectors `\cite{ilharco2023editing,yang2024adamerging,lu2024twin}`{=latex}. However, they typically suffer from parameter interference between different tasks and static optimal solutions may struggle with multiple task domains.

There is a growing research interest in specialized large models by eliciting existing knowledge, such as utilizing curated prompts via in-context learning `\cite{lin2023unlocking}`{=latex} or employing small models to offer weak-to-strong supervision `\cite{burns2023weaktostrong,mitchell2024an,liu2024tuning}`{=latex}, generating weak labels or alignment signals. These methods are highly practical, requiring only the generated output (or logits). Our approach shares a similar weak-to-strong intuition and proposes adaptive strategies to achieve a better balance between leveraging the pretrained general knowledge and acquiring task-specific knowledge. Furthermore, our method can be extended to multi-task scenarios.

## Weak-to-Strong Generation

Different from knowledge distillation `\cite{hinton2015distilling,stanton2021does,beyer2022knowledge}`{=latex}, where a more capable teacher guides a student model, we explore using weaker models to teach stronger ones. `\citet{burns2023weaktostrong}`{=latex} empirically demonstrate that weak supervisors can reliably elicit knowledge from much stronger models (*e.g.*, supervising GPT-4 with a GPT-2). Such approaches represent a promising direction, as it is more practical and affordable to specialize in a much smaller model. Contrastive Decoding `\cite{li-etal-2023-contrastive}`{=latex} enhances outputs by leveraging differences between large and small LMs, as patterns like repetition and incoherence are often more pronounced in smaller ones. Speculative sampling `\cite{leviathan2023fast, shen2024learning}`{=latex} speeds up inference using a lower-parameter version of the LLM as a draft model, exploiting the fact that draft models can approximate easier subtasks well. SpecInfer `\cite{Miao_2024}`{=latex} goes further by employing a set of small models to generate drafts in parallel. `\citet{jin-etal-2023-parameter}`{=latex} train a projector to map the parameter of the weak expert to a larger version, but the projector needs to be trained and suffers from poor generation. `\citet{ji2024aligner}`{=latex} focuses on utilizing small models to rephrase unaligned answers from LLMs into more aligned ones. `\citet{mitchell2024an}`{=latex} and `\citet{liu2024tuning}`{=latex} leverage logits from small, fine-tuned models to inject specific knowledge into the pretrained LLM with the same vocabulary. Our approach differs from methods that require pretrained small model adapters `\cite{shen2024learning}`{=latex} or pre-adjusted parameters `\cite{mitchell2024an, liu2024tuning}`{=latex}. Instead, we dynamically transfer the capabilities of small models to the large model at each decoding step, without needing access to the large model’s parameters or training data.

# Methodology

## Problem Background

#### Autoregressive Language Models

Modern autoregressive transformers generate continuations for input prompts token by token. Given a prompt \\(x_{1:k-1}\\) (denoted as \\(x_{<k}\\)), the model computes the logits for the \\(k\\)-th token, represented as \\(M(x_k \mid x_{<k}) \in \mathbb{R}^{|V|}\\), where V denotes the size of the vocabulary. A probability distribution \\(P(x_k \mid x_{<k})\\) is then obtained through softmax normalization: \\(P(x_k \mid x_{<k}) = \text{softmax}(M(x_k \mid x_{<k}))\\). The next token \\(x_k\\) is subsequently sampled from this distribution, *i.e.*, \\(x_k \sim P(x_k \mid x_{<k})\\).

#### Distance Between Language Model Outputs Distribution [sec:distance]

We can utilize the Kullback-Leibler(KL) divergence to measure the similarity of two distributions \\(P\\) and \\(Q\\) generated from two language models (with the same vocabulary), which can be viewed as the distance of the two language models: \\[\begin{aligned}
    D_{\mathrm{KL}}( P|| Q \mid x_{<k})&= \sum_{x\in V} P(x \mid x_{<k}) \log \frac{P(x \mid x_{<k})}{Q(x \mid x_{<k})}
    \label{eq:KL}
\end{aligned}\\] If this is implied by the context, we will omit the conditioning on \\(|x_{<k}\\) and simply use \\(D_{\mathrm{KL}}(P||Q)\\).

#### Logit Arithmetic

Suppose we have two pretrained auto-regressive models with homogeneous architecture and the same vocabulary: a small model with parameter set \\(\theta^S\\) and a large model with parameter set \\(\theta^L\\). We aim to fine-tune the small model to obtain \\(\theta^S_{ft}\\) and transfer this fine-tuning knowledge to the large models. Previous work `\cite{mitchell2024an, liu2024tuning}`{=latex} transferred fine-tuned knowledge to a large model by designing arithmetic between logits, resulting in the output distribution \\(\tilde{P}\\) for the large model as follows: \\[\begin{aligned}
    \tilde{P}(x_k|x_{<k}) = \mathrm{softmax}( M^L(x_k|x_{<k}) + \alpha \cdot (M^{S}_{ft}(x_k|x_{<k}) - M^S(x_k|x_{<k})) )
    \label{eq:logit-arithmetic}
\end{aligned}\\] where \\(M^L\\), \\(M^S\\), and \\(M^{S}_{ft}\\) represent the logits of the large model, small model, and fine-tuned small model, respectively. Their corresponding normalized distributions are denoted by \\(P\\), \\(Q\\), and \\(Q_{ft}\\). The detailed theoretical proof supporting this logit arithmetic formula is provided in Appendix <a href="#sec:rl_proof" data-reference-type="ref" data-reference="sec:rl_proof">9</a>. Here, \\(\alpha\\) is a pre-adjusted hyperparameter that controls the extent of knowledge transfer from the small model. Our analysis from Appendix <a href="#sec:poe_proof" data-reference-type="ref" data-reference="sec:poe_proof">10</a> demonstrates that logit arithmetic attempts to approximate the shift \\((\frac{Q_{ft}(x_k \mid x_{<k})}{Q(x_k \mid x_{<k})})^\alpha\\) between the fine-tuned distribution and the pretrained distribution by controlling the parameter \\(\alpha\\) before inference.

## Adaptive Knowledge Transfer Optimization

At deployment time, the importance of expert knowledge naturally fluctuates from one decoding step to the next.  We therefore endow the fusion coefficient \(\alpha\) with the freedom to vary online, selecting its value by solving a simple *distance–matching* problem:

\[
\tilde{P}^* \;=\; \operatorname*{arg\,min}_{\tilde P}\Bigl[\bigl(D_{\mathrm{KL}}(\tilde P\,\|\,P)-D_{\mathrm{KL}}(Q_{ft}\,\|\,Q)\bigr)^2+\bigl(D_{\mathrm{KL}}(P\,\|\,\tilde P)-D_{\mathrm{KL}}(Q\,\|\,Q_{ft})\bigr)^2\Bigr].\tag{3}
\]

The objective in Eq. (3) simply asks the large model’s distribution to *mirror* the divergence profile exhibited by the fine-tuned expert.  Using the *squared* difference between KL divergences avoids the need for any additional hyper-parameters and has proven numerically stable in all our experiments.  In practice we perform a lightweight grid search over \(\alpha\in[0,2]\) at each decoding step; the optimum is typically found within 10–15 evaluations, adding negligible latency compared to baseline logit arithmetic.

Crucially, the formulation in Eq. (3) is agnostic to model size and architecture, allowing us to extend it seamlessly to multiple experts (Section 3.2) without revisiting the derivation.  Because the criterion depends **only** on output distributions, it can be applied even when the large model’s parameters are inaccessible, which is central to our weak-to-strong setting.
## Extending to the Fusion of Multiple SLMs

When dealing with complex tasks or new domains, a general LLM may lack the necessary expertise, and a single SLM might not provide sufficient specialized knowledge due to the capacity gap. To migrate this challenge, our method can be extended to fuse multiple SLMs and leverage their knowledge to compensate for the shortcomings in individual domain-specific knowledge. We fine-tune the SLM \\(\boldsymbol{\theta^S}\\) on each domain to obtain multiple task-specific experts \\(\{\boldsymbol{\theta^S_t}\}_{t=1}^T\\), making it easier to acquire knowledge and dynamic fused to the LLM. During decoding, knowledge from these domain-specific SLMs is transferred simultaneously to the LLM. We modify the Equation (<a href="#eq:DKL" data-reference-type="ref" data-reference="eq:DKL">[eq:DKL]</a>) as follows: \\[\begin{aligned}
        D_{\mathrm{KL}}(\tilde{P} \parallel P) \approx D_{\mathrm{KL}}\left(\text{Joint}( \{Q_{1...T}\}) \parallel Q \right) ,\quad
        D_{\mathrm{KL}}(P \parallel \tilde{P}) \approx D_{\mathrm{KL}}\left(Q \parallel \text{Joint}( \{Q_{1...T}\}) \right)
        \label{eq:mKL}
    \end{aligned}\\] where \\(Q_t\\) represents the distribution of \\(\theta_t^{S}\\) from the \\(t\\)-th domain, \\(\text{Joint}( \{Q_{1...T}\})\\) represents the distribution of the combined knowledge of \\(Q_1,Q_2,...,Q_T\\). When we impose constraints like Equation (<a href="#eq:mKL" data-reference-type="ref" data-reference="eq:mKL">[eq:mKL]</a>), it attempts to align the joint distributions of the logits from the domain-specific small models. However, the distributions of the logits from the domain-specific small models are usually not independent of each other, so their joint distribution cannot be estimated through simple multiplication.

Due to the difficulty in obtaining a joint distribution for multiple expert models, we decompose the joint distribution constraint problem into a multi-object marginal distribution optimization. The transformation process we prove in detail in Appendix <a href="#sec:multi_slm" data-reference-type="ref" data-reference="sec:multi_slm">[sec:multi_slm]</a>. By aligning the distributions of each domain-specific small model, we can infer an approximate optimal solution within the extension of Equation (<a href="#eq:logit-arithmetic" data-reference-type="ref" data-reference="eq:logit-arithmetic">[eq:logit-arithmetic]</a>), as shown by: \\[\begin{aligned}
    \mathop{\mathrm{arg\,min}}_{\tilde{P}} \sum_{t=1}^T \left[ \left(D_{KL}(\tilde{P}||P) - D_{KL}(Q_t||Q) \right)^2 + \left( D_{KL}(P||\tilde{P}) - D_{KL}(Q||Q_t) \right)^2\right] \\ 
    \text{where} \quad
    \tilde{P} = \mathrm{softmax}\left[ M^L(x_k|x_{<k}) + \sum_{t=1}^T \alpha_t \left( M^{S}_t(x_k|x_{<k}) - M^S(x_k|x_{<k}) \right) \right]
    \end{aligned}
    \label{eq:mmin}\\] Here we use \\(M^S_t\\) to represent the logit of \\(t\\)-th expert. Our algorithm is outlined in pseudo-code in Algorithm <a href="#alg:maccot" data-reference-type="ref" data-reference="alg:maccot">[alg:maccot]</a> in Appendix <a href="#sec:alg" data-reference-type="ref" data-reference="sec:alg">13</a>.

Intuitively, this projects the KL divergences between the logits into Euclidean space and finds a central point with the minimum distance sum as the best KL value. This optimal KL value corresponds to the output distribution of the large model with multi-domain knowledge. In our experiments, we optimize \\(\alpha \in \mathbb{R}^T\\) to obtain the optimal KL value. Generally, \\(\alpha\\) is between 0 and 2. In a multi-task setting, we can accelerate the optimization process by optimizing the boundaries for only one expert, restricting only one SLM expert to be applied at the current decoding step. We will provide a more detailed explanation of this in our experiments.

# Experiments

In this paper, we use the LLaMA2 `\cite{touvron2023llama}`{=latex} family of models to test our method, which contains the same vocabulary, enabling us to employ logits arithmetic easily. Here, we use TinyLLaMA-1.1B `\cite{zhang2024tinyllama}`{=latex} and LLaMA2-7B as our pretrained small models, and LLaMA2-13B as the large model for transfer. We conduct tests in single-domain and multi-domain settings to demonstrate that our method can effectively transfer knowledge to the large model in both scenarios.

<div id="tab:single-expert" markdown="1">

<table>
<caption>Performance on single-task scenarios. <strong>Bold</strong> numbers indicate the best-performing model transferred from the same size. <u>Underlines</u> indicate whether the method outperforms the expert model being used. Notably, we are unable to obtain the LoRA adapter for LLAMA2-chat version. Therefore, we set the LoRA Tuning for the 13B model on TruthfulQA to match the same values as Full Fine-Tuning, <em>e.g.</em>, .</caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: left;"><strong>Model</strong></td>
<td style="text-align: center;"><strong>GSM8K</strong></td>
<td style="text-align: center;"><strong>TruthfulQA</strong></td>
<td style="text-align: center;"><strong>TriviaQA</strong></td>
<td style="text-align: center;"><strong>CNN/DM</strong></td>
<td style="text-align: center;"><strong>MMLU</strong></td>
<td rowspan="2" style="text-align: center;"><strong>Avg.</strong></td>
</tr>
<tr>
<td style="text-align: center;"><strong>(EM.)</strong></td>
<td style="text-align: center;"><strong>(Acc.)</strong></td>
<td style="text-align: center;"><strong>(EM.)</strong></td>
<td style="text-align: center;"><strong>(Rouge 2.)</strong></td>
<td style="text-align: center;"><strong>(Acc.)</strong></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Base Model</strong></td>
<td style="text-align: center;">6.90</td>
<td style="text-align: center;">46.13</td>
<td style="text-align: center;">36.44</td>
<td style="text-align: center;">8.94</td>
<td style="text-align: center;">51.25</td>
<td style="text-align: center;">29.93</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Full Fine-tuning</strong></td>
<td style="text-align: center;">47.23</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">56.36</td>
<td style="text-align: center;">15.50</td>
<td style="text-align: center;">57.94</td>
<td style="text-align: center;">47.79</td>
</tr>
<tr>
<td style="text-align: left;"><strong>LoRA Tuning</strong></td>
<td style="text-align: center;">41.54</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">61.89</td>
<td style="text-align: center;">17.18</td>
<td style="text-align: center;">60.46</td>
<td style="text-align: center;">48.60</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Full Fine-tuning</strong></td>
<td style="text-align: center;">12.51</td>
<td style="text-align: center;">29.01</td>
<td style="text-align: center;">33.66</td>
<td style="text-align: center;">14.22</td>
<td style="text-align: center;">37.26</td>
<td style="text-align: center;">25.33</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Proxy Tuning</strong></td>
<td style="text-align: center;"><u>16.91</u></td>
<td style="text-align: center;"><u>31.48</u></td>
<td style="text-align: center;"><u>48.74</u></td>
<td style="text-align: center;">13.23</td>
<td style="text-align: center;"><u>39.88</u></td>
<td style="text-align: center;"><u>31.74</u></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Ours</strong></td>
<td style="text-align: center;"><u><strong>18.27</strong></u></td>
<td style="text-align: center;"><u>37.05</u></td>
<td style="text-align: center;"><u><strong>53.81</strong></u></td>
<td style="text-align: center;"><u><strong>14.48</strong></u></td>
<td style="text-align: center;"><u>48.32</u></td>
<td style="text-align: center;"><u><strong>34.86</strong></u></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Full Fine-tuning</strong></td>
<td style="text-align: center;">37.07</td>
<td style="text-align: center;">60.02</td>
<td style="text-align: center;">52.10</td>
<td style="text-align: center;">15.21</td>
<td style="text-align: center;">56.23</td>
<td style="text-align: center;">44.13</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Proxy Tuning</strong></td>
<td style="text-align: center;"><u>37.68</u></td>
<td style="text-align: center;"><u>61.02</u></td>
<td style="text-align: center;"><u>52.81</u></td>
<td style="text-align: center;">14.37</td>
<td style="text-align: center;"><u>56.24</u></td>
<td style="text-align: center;"><u>44.43</u></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Ours</strong></td>
<td style="text-align: center;"><u><strong>39.34</strong></u></td>
<td style="text-align: center;"><u><strong>61.56</strong></u></td>
<td style="text-align: center;"><u><strong>57.11</strong></u></td>
<td style="text-align: center;"><u><strong>15.31</strong></u></td>
<td style="text-align: center;"><u><strong>57.15</strong></u></td>
<td style="text-align: center;"><u><strong>46.09</strong></u></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:multi-expert" markdown="1">

<table>
<caption>Performance on multi-task scenarios. “Base” denotes an untrained model. “Multi-Task Tuning" refers to models trained using data mixing. <strong>Bold</strong> numbers represent the best-performing multi-task models among those using experts of the same size. The leftmost "Avg." represents the average performance of Seen Tasks and Unseen Tasks (57 tasks in MMLU), calculated by averaging the mean performance on Seen Tasks and the performance on MMLU. </caption>
<thead>
<tr>
<th style="text-align: center;"><strong>Model</strong></th>
<th style="text-align: center;"><strong>Avg.</strong></th>
<th colspan="5" style="text-align: center;"><strong>Seen Task</strong></th>
<th style="text-align: center;"><strong>Unseen Task</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span>3-7</span> (lr)<span>8-8</span></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>GSM8K</strong></td>
<td style="text-align: center;"><strong>TruthfulQA</strong></td>
<td style="text-align: center;"><strong>TriviaQA</strong></td>
<td style="text-align: center;"><strong>CNN/DM</strong></td>
<td style="text-align: center;"><strong>Avg.</strong></td>
<td style="text-align: center;"><strong>MMLU</strong></td>
</tr>
<tr>
<td style="text-align: left;"><em>Pre-trained model</em></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>1.1B (Base)</strong></td>
<td style="text-align: center;">18.29</td>
<td style="text-align: center;">2.04</td>
<td style="text-align: center;">27.41</td>
<td style="text-align: center;">9.6</td>
<td style="text-align: center;">7.21</td>
<td style="text-align: center;">11.56</td>
<td style="text-align: center;">25.02</td>
</tr>
<tr>
<td style="text-align: left;"><strong>7B (Base)</strong></td>
<td style="text-align: center;">29.16</td>
<td style="text-align: center;">3.80</td>
<td style="text-align: center;">30.96</td>
<td style="text-align: center;">36.7</td>
<td style="text-align: center;">8.81</td>
<td style="text-align: center;">20.06</td>
<td style="text-align: center;">38.26</td>
</tr>
<tr>
<td style="text-align: left;"><strong>13B (Base)</strong></td>
<td style="text-align: center;">37.93</td>
<td style="text-align: center;">6.89</td>
<td style="text-align: center;">46.13</td>
<td style="text-align: center;">36.5</td>
<td style="text-align: center;">8.94</td>
<td style="text-align: center;">24.61</td>
<td style="text-align: center;">51.25</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>13B (Multi-Task Tuning)</strong></td>
<td style="text-align: center;">45.68</td>
<td style="text-align: center;">39.03</td>
<td style="text-align: center;">44.39</td>
<td style="text-align: center;">62.79</td>
<td style="text-align: center;">16.95</td>
<td style="text-align: center;">40.78</td>
<td style="text-align: center;">50.58</td>
</tr>
<tr>
<td style="text-align: left;"><em>Single task with tuning</em></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>1.1B-Expert (GSM8K)</strong></td>
<td style="text-align: center;">18.50</td>
<td style="text-align: center;">12.51</td>
<td style="text-align: center;">25.38</td>
<td style="text-align: center;">6.12</td>
<td style="text-align: center;">7.75</td>
<td style="text-align: center;">12.69</td>
<td style="text-align: center;">24.30</td>
</tr>
<tr>
<td style="text-align: left;"><strong>1.1B-Expert (TruthfulQA)</strong></td>
<td style="text-align: center;">16.15</td>
<td style="text-align: center;">2.81</td>
<td style="text-align: center;">29.01</td>
<td style="text-align: center;">2.83</td>
<td style="text-align: center;">7.23</td>
<td style="text-align: center;">10.47</td>
<td style="text-align: center;">21.82</td>
</tr>
<tr>
<td style="text-align: left;"><strong>1.1B-Expert (TriviaQA)</strong></td>
<td style="text-align: center;">21.72</td>
<td style="text-align: center;">3.26</td>
<td style="text-align: center;">26.25</td>
<td style="text-align: center;">33.66</td>
<td style="text-align: center;">8.03</td>
<td style="text-align: center;">17.80</td>
<td style="text-align: center;">25.63</td>
</tr>
<tr>
<td style="text-align: left;"><strong>1.1B-Expert (CNN/DM)</strong></td>
<td style="text-align: center;">12.98</td>
<td style="text-align: center;">2.73</td>
<td style="text-align: center;">26.39</td>
<td style="text-align: center;">1.65</td>
<td style="text-align: center;">14.22</td>
<td style="text-align: center;">11.24</td>
<td style="text-align: center;">14.73</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>1.1B (Multi-Task Tuning)</strong></td>
<td style="text-align: center;">23.90</td>
<td style="text-align: center;">14.40</td>
<td style="text-align: center;">25.76</td>
<td style="text-align: center;"><strong>35.05</strong></td>
<td style="text-align: center;">14.26</td>
<td style="text-align: center;"><strong>22.37</strong></td>
<td style="text-align: center;">25.42</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Ours</strong></td>
<td style="text-align: center;"><strong>32.59</strong></td>
<td style="text-align: center;"><strong>18.65</strong></td>
<td style="text-align: center;"><strong>36.33</strong></td>
<td style="text-align: center;">18.84</td>
<td style="text-align: center;">9.38</td>
<td style="text-align: center;">20.80</td>
<td style="text-align: center;"><strong>44.38</strong></td>
</tr>
<tr>
<td style="text-align: left;"><em>Single task with tuning</em></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>7B-Expert (GSM8K)</strong></td>
<td style="text-align: center;">34.14</td>
<td style="text-align: center;">37.07</td>
<td style="text-align: center;">36.04</td>
<td style="text-align: center;">20.41</td>
<td style="text-align: center;">11.19</td>
<td style="text-align: center;">26.18</td>
<td style="text-align: center;">42.10</td>
</tr>
<tr>
<td style="text-align: left;"><strong>7B-Expert (TruthfulQA)</strong></td>
<td style="text-align: center;">32.79</td>
<td style="text-align: center;">8.26</td>
<td style="text-align: center;">56.42</td>
<td style="text-align: center;">0.17</td>
<td style="text-align: center;">11.02</td>
<td style="text-align: center;">18.96</td>
<td style="text-align: center;">46.61</td>
</tr>
<tr>
<td style="text-align: left;"><strong>7B-Expert (TriviaQA)</strong></td>
<td style="text-align: center;">29.35</td>
<td style="text-align: center;">4.62</td>
<td style="text-align: center;">33.66</td>
<td style="text-align: center;">52.10</td>
<td style="text-align: center;">10.30</td>
<td style="text-align: center;">25.17</td>
<td style="text-align: center;">33.52</td>
</tr>
<tr>
<td style="text-align: left;"><strong>7B-Expert (CNN/DM)</strong></td>
<td style="text-align: center;">22.29</td>
<td style="text-align: center;">4.39</td>
<td style="text-align: center;">34.57</td>
<td style="text-align: center;">0.19</td>
<td style="text-align: center;">15.21</td>
<td style="text-align: center;">13.59</td>
<td style="text-align: center;">30.98</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>7B (Multi-Task Tuning)</strong></td>
<td style="text-align: center;">34.89</td>
<td style="text-align: center;">34.72</td>
<td style="text-align: center;">33.28</td>
<td style="text-align: center;"><strong>51.54</strong></td>
<td style="text-align: center;"><strong>16.30</strong></td>
<td style="text-align: center;"><strong>33.96</strong></td>
<td style="text-align: center;">35.82</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Ours</strong></td>
<td style="text-align: center;"><strong>39.42</strong></td>
<td style="text-align: center;"><strong>34.87</strong></td>
<td style="text-align: center;"><strong>42.25</strong></td>
<td style="text-align: center;">22.48</td>
<td style="text-align: center;">10.52</td>
<td style="text-align: center;">27.53</td>
<td style="text-align: center;"><strong>51.31</strong></td>
</tr>
</tbody>
</table>

</div>

<span id="tab:multi-expert" label="tab:multi-expert"></span>

## Datasets

Following the setting in  `\citet{liu2024tuning}`{=latex}, we evaluate on the following datasets: mathmetical reasoning (GSM8K `\cite{cobbe2021training}`{=latex}); factual accuracy (TruthfulQA `\cite{lin2022truthfulqa}`{=latex}); realistic knowledge (TriviaQA `\cite{joshi2017triviaqa}`{=latex}); multi-domain general knowledge (MMLU benchmark `\cite{hendrycks2020measuring}`{=latex}); summarization (CNN-DailyMail (CNN/DM) `\cite{see-etal-2017-get}`{=latex}). All datasets are tested using a 0-shot setting. Detailed information is provided in Appendix <a href="#sec:exp" data-reference-type="ref" data-reference="sec:exp">14</a>.

## Implementation Details [sec:implementation]

For all tasks except TruthfulQA, we construct prompts based on the task type and employ supervised instruction tuning to train each task expert. Following the previous work `\cite{liu2024tuning}`{=latex}, we use “Llama-2-7b-chat-hf” as the 7B expert for TurhfulQA, and TinyLLaMA-chat-version as the 1.1B expert for TurhfulQA. For full fine-tuning, we set the batch size to 128, learning rate to 2e-5, optimizer to Adam. For LoRA tuning, we set the rank to 64, learning rate to 1e-4, optimizer to Adam. We train for 3 epochs. For multi-task tuning, we perform full fine-tuning using a mixed-seed training dataset.

During inference, we use greed decoding and set batch size to 256, top_p to 1.0 and temperature to 0.05. To accelerate inference, we use VLLM[^3] , synchronizing the signals of logits during logit arithmetic to achieve efficient inference. All experiments are performed on H100 GPUs.

## Baselines

To demonstrate the effectiveness of our method in efficiently transferring knowledge from a small model to a large model, we compare it against a small model fine-tuned in a single domain and a large model without fine-tuning. We also report the performance of a large model fine-tuned on a single domain or fine-tuned using LoRA to demonstrate the feasibility of our method as a replacement for fine-tuning approaches. Additionally, we compare our method with Proxy Tuning (setting \\(\alpha=1.0\\) as in the original work) to highlight the superiority of our method in the transfer process. We use the LLaMA2-chat model as the expert for TruthfulQA (therefore, there is no corresponding LoRA model). For other experts, we use models fine-tuned on the respective training sets. In multi-task scenarios, we follow the common training practice of mixing the training data, with the results serving as our multi-task baseline. More details can be found in the Appendix <a href="#sec:exp" data-reference-type="ref" data-reference="sec:exp">14</a>.

<figure id="fig:alpha_gsm8k">
<img src="./figures/diff_alpha_for_task.png"" />
<img src="./figures/router_oneexpert_G_G.png"" />
<figcaption>The variation of <span class="math inline"><em>α</em></span> in knowledge transfer for the GSM8K expert.<span id="fig:one_route" data-label="fig:one_route"></span></figcaption>
</figure>

## Performance on Single-Task Scenarios

As shown in Table <a href="#tab:single-expert" data-reference-type="ref" data-reference="tab:single-expert">1</a>, our method improves upon the original experts when transferring knowledge from the 1.1B and 7B models to the 13B model. Across all tasks, our method transfers knowledge from 1.1B and 7B experts to the 13B model, achieving performance improvements of 37.6% and 4.4% over the original experts, respectively. This demonstrates that our method can leverage the existing knowledge in larger models to enhance the knowledge learned by the expert small models. Notably, we observe significant improvements on GSM8K and TriviaQA compared to the experts alone, indicating that our method effectively transfers expert knowledge while leveraging the inherent capabilities of the large model, such as stronger reasoning abilities and a richer knowledge base.

Our method, which transfers knowledge from 1.1B and 7B experts, can close the performance gap by 72.9% and 96.4%, respectively, compared to 13B Full Fine-Tuning, achieving improvements of 6.5% and 3.8% over Proxy Tuning. Compared to 13B Full LoRA Tuning, our method can close the gap by 71.7% and 90.7%, respectively, when transferring knowledge from 1.1B and 7B experts. Additionally, it is worth noting that our method outperforms the 13B Full Fine-Tuning results on TriviaQA. Since our approach only requires fine-tuning a smaller model, it demands less computational and memory resources compared to fine-tuning a large model, making our method highly promising.

## Performance on Multi-Task Scenarios

In the multi-task scenario, we categorize the multi-domain MMLU task as an unseen task and the other four tasks as seen tasks. We then calculate the average performance on seen and unseen tasks to evaluate the overall generalization capability of our model. As shown in Table <a href="#tab:multi-expert" data-reference-type="ref" data-reference="tab:multi-expert">2</a>, our method achieves a 36.35% and 13.37% improvement over directly fine-tuning on 1.1B and 7B, respectively, using multi-domain fine-tuning. Furthermore, our results show that transferring knowledge from 7B outperforms 13B overall by 3.93%. Specifically, the performance improvement is 11.9% for seen tasks and 0.1% for unseen tasks. This indicates that our approach can alleviate conflicts between multiple tasks to a certain extent and improve out-of-domain generalization by leveraging the capabilities of the large model itself.

In multi-task settings, our method achieves a performance improvement of 71.3% and 86.3% over 13B multi-task tuning when transferring knowledge from 1.1B and 7B, respectively. It can be observed that our method performs worse in domain-specific tasks compared to multi-task tuning, *e.g.*, TriviaQA. This is primarily because we cannot access the data needed by the experts, resulting in a bottleneck in handling task conflicts. In the future, we will explore more effective ways to effortlessly transfer multiple small expert models to a large model. Notably, Proxy Tuning is designed for single-domain scenarios with a known test distribution. It struggles at test time with inputs from unknown distributions, making it difficult to use in multi-task scenarios. However, our method achieves the best results on unseen tasks, demonstrating its effectiveness in enhancing the generalization ability of large models on new tasks. Moreover, our method can effortlessly transfer knowledge from pretrained experts across multiple domains to the large model without requiring access to their training data.

# Analysis

## How \\(\alpha\\) control the knowledge transfer from the small models?

<figure id="tab:time">
<img src="./figures/router_fourexpert_MMLU_abstract_algebra.png"" />
<p><span>The time required to train or inference 1,000 data points on a single GPU. In the inference section, values in parentheses show the factor by which the inference speed is slower compared to the 13B FFT model. "FFT" denotes Full Fine-Tuning, and "LT" denotes LoRA Tuning. Our 1.1B/7B expert model use full fine-tuning.</span></p>
<table>
<thead>
<tr>
<th style="text-align: left;">Model</th>
<th style="text-align: center;">Training</th>
<th style="text-align: center;">Inference</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">13B FFT</td>
<td style="text-align: center;">1176s</td>
<td style="text-align: center;"><strong>60s</strong></td>
</tr>
<tr>
<td style="text-align: left;">13B LT</td>
<td style="text-align: center;">836s</td>
<td style="text-align: center;"><strong>60s</strong></td>
</tr>
<tr>
<td style="text-align: left;">Proxy Tuning (from 1.1B)</td>
<td style="text-align: center;"><strong>128s</strong></td>
<td style="text-align: center;">142s (<span class="math inline">×</span>2.36)</td>
</tr>
<tr>
<td style="text-align: left;">Ours (from 1.1B)</td>
<td style="text-align: center;"><strong>128s</strong></td>
<td style="text-align: center;">150s (<span class="math inline">×</span>2.5)</td>
</tr>
<tr>
<td style="text-align: left;">Proxy Tuning (from 7B)</td>
<td style="text-align: center;">588s</td>
<td style="text-align: center;">158s (<span class="math inline">×</span>2.63)</td>
</tr>
<tr>
<td style="text-align: left;">Ours (from 7B)</td>
<td style="text-align: center;">588s</td>
<td style="text-align: center;">166s (<span class="math inline">×</span>2.76)</td>
</tr>
</tbody>
</table>
<figcaption>The variation of <span class="math inline"><em>α</em></span> for the four experts during knowledge transfer on an unseen task (MMLU: abstract algebra). <span id="fig:four_expert" data-label="fig:four_expert"></span></figcaption>
</figure>

We aim to understand how \\(\alpha\\) dynamically changes in single-task and multi-task scenarios, and what patterns it exhibits across different tasks. As shown in Figure <a href="#fig:alpha_ablation" data-reference-type="ref" data-reference="fig:alpha_ablation">[fig:alpha_ablation]</a>, we compare our method with predefined \\(\alpha\\) values of 0.5, 1.0, and 1.5. It can be observed that our method consistently outperforms the predefined \\(\alpha\\) settings, demonstrating the effectiveness of adaptive adjustment at each decoding step. Additionally, the predefined method exhibits significant variation across different tasks, whereas our method addresses this issue and avoids the extensive effort required for parameter tuning.

As shown in Figures <a href="#fig:one_route" data-reference-type="ref" data-reference="fig:one_route">[fig:one_route]</a> and <a href="#fig:four_expert" data-reference-type="ref" data-reference="fig:four_expert">[fig:four_expert]</a>, we illustrate the variation of \\(\alpha\\) for a single expert and multiple experts, respectively. In Figure <a href="#fig:one_route" data-reference-type="ref" data-reference="fig:one_route">[fig:one_route]</a>, it can be observed that \\(\alpha\\) fluctuates between the lower bound and upper bound. At the lower bound, less expert knowledge is transferred, and it is more important to retain the large model’s inherent capabilities. Conversely, at the upper bound, more expert knowledge is transferred, requiring more GSM8K-related capabilities for those steps. In Figure <a href="#fig:four_expert" data-reference-type="ref" data-reference="fig:four_expert">[fig:four_expert]</a>, when solving the “abstract algebra” problem in MMLU using four experts, it is evident that GSM8K transfers more knowledge overall, especially in the initial steps. Since “abstract algebra” is a mathematically oriented problem, these results demonstrate that our method can effectively select and transfer expert knowledge during the decoding steps.

## Efficiency Analysis [sec:efficient]

In a batch inference with a size of \\(B\\), Proxy Tuning requires a complexity of approximately \\(O(BV)\\) for each decoding step. In contrast, our method requires \\(O(nBV)\\) complexity, where \\(n\\) (\\(\leq 20\\)) is the number of parameter searches. Since \\(n \ll V\\), our complexity is comparable to Proxy Tuning.

In multi-task settings, performing parameter searches for each expert and then merging them would result in exponential growth, with a complexity of \\(O(n^TBV)\\). To optimize this process, we avoid merging parameters between experts at each step and instead select a single expert for knowledge transfer. This reduces the algorithm complexity to \\(O(nTBV)\\). Additionally, we can constrain the parameter search range for each expert to achieve better efficiency and performance.

Here, we complete the training and inference on an H100 GPU for 1000 data, while recording the time taken. As shown in Table <a href="#tab:time" data-reference-type="ref" data-reference="tab:time">3</a>, our method achieves similar efficiency to Proxy Tuning during inference. Due to the need for inference and communication between multiple models, our approach is approximately 2.5 times slower than direct inference with a 13B model. However, our method only requires training a small model, which provides a significant advantage in training efficiency.

## Comparison the Weak-to-strong Approach with Other Methods

#### Comparison against In-Context Learning

<figure id="fig:aug">
<figure id="fig:icl">
<img src="./figures/icl.png"" />
<figcaption>Enhance In-Context Learning (ICL) with our method from transferring 7B expert knowledge. <span id="fig:icl" data-label="fig:icl"></span></figcaption>
</figure>
<figure id="fig:taskarithmetic">
<img src="./figures/taskarithmetic.png"" />
<figcaption>Enhance Task Arithmetic (TA) with our method from transferring 7B expert knowledge. <span id="fig:taskarithmetic" data-label="fig:taskarithmetic"></span></figcaption>
</figure>
<figcaption>Enhance in-context learning and task arithmetic using our method. </figcaption>
</figure>

Enhancing in-context learning with logits arithmetic can improve large language models. Since both methods work at the token level and need only black-box access, they can be combined to boost model performance. We categorize tasks as either Single-Domain (excluding MMLU) or Multi-Domain (MMLU). The "All" category averages the results of both. Notably, we use 5-shot information as in-context examples, and we apply different few-shot samples and prompts for various MMLU tasks. In our experiments, we enhance a 13B model’s performance using a 7B expert and compare it to the 13B model using in-context learning. We present the results in Figure <a href="#fig:icl" data-reference-type="ref" data-reference="fig:icl">4</a>. We observed that our method outperforms 5-shot in-context learning on both Single-Domain and Multi-Domain tasks. Furthermore, when combined with in-context learning, our method shows a significant improvement on Multi-Domain tasks, resulting in an overall (All) increase of 18.3%. Specifically, combining our method with in-context learning does not significantly impact performance on Single-Domain tasks, indicating that our method alone has already achieved capabilities similar to in-context learning for individual tasks.

<div id="tab:icl" markdown="1">

<table>
<caption>VS In-context Learning</caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;"><strong>Model</strong></td>
<td rowspan="2" style="text-align: center;"><strong>from</strong></td>
<td style="text-align: center;"><strong>GSM8K</strong></td>
<td style="text-align: center;"><strong>TruthfulQA</strong></td>
<td style="text-align: center;"><strong>TriviaQA</strong></td>
<td style="text-align: center;"><strong>CNN/DM</strong></td>
<td style="text-align: center;"><strong>MMLU</strong></td>
</tr>
<tr>
<td style="text-align: center;"><strong>(EM.)</strong></td>
<td style="text-align: center;"><strong>(Acc.)</strong></td>
<td style="text-align: center;"><strong>(EM.)</strong></td>
<td style="text-align: center;"><strong>(Rouge 2.)</strong></td>
<td style="text-align: center;"><strong>(Acc.)</strong></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Base (untuned)</strong></td>
<td style="text-align: center;">\</td>
<td style="text-align: center;">6.90</td>
<td style="text-align: center;">46.13</td>
<td style="text-align: center;">36.44</td>
<td style="text-align: center;">8.94</td>
<td style="text-align: center;">50.07</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Full Finetune</strong></td>
<td style="text-align: center;">\</td>
<td style="text-align: center;">47.23</td>
<td style="text-align: center;">61.93</td>
<td style="text-align: center;">56.36</td>
<td style="text-align: center;">15.50</td>
<td style="text-align: center;">57.94</td>
</tr>
<tr>
<td style="text-align: left;"><strong>LoRA Tuning</strong></td>
<td style="text-align: center;">\</td>
<td style="text-align: center;">41.54</td>
<td style="text-align: center;">\</td>
<td style="text-align: center;">61.89</td>
<td style="text-align: center;">15.27</td>
<td style="text-align: center;">57.15</td>
</tr>
<tr>
<td style="text-align: left;"><strong>In Context Learning (5-shot)</strong></td>
<td style="text-align: center;">\</td>
<td style="text-align: center;">20.09</td>
<td style="text-align: center;">35.12</td>
<td style="text-align: center;">61.97</td>
<td style="text-align: center;">9.03</td>
<td style="text-align: center;">53.72</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Ours</strong></td>
<td style="text-align: center;">1.1b</td>
<td style="text-align: center;">17.36</td>
<td style="text-align: center;">39.02</td>
<td style="text-align: center;">50.51</td>
<td style="text-align: center;">13.82</td>
<td style="text-align: center;">48.32</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Ours (5-shot)</strong></td>
<td style="text-align: center;">1.1b</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Ours</strong></td>
<td style="text-align: center;">7b</td>
<td style="text-align: center;">39.34</td>
<td style="text-align: center;">61.20</td>
<td style="text-align: center;">57.11</td>
<td style="text-align: center;">15.36</td>
<td style="text-align: center;">57.14</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Ours (5-shot)</strong></td>
<td style="text-align: center;">7b</td>
<td style="text-align: center;">40.03</td>
<td style="text-align: center;">60.95</td>
<td style="text-align: center;">62.63</td>
<td style="text-align: center;">12.31</td>
<td style="text-align: center;">57.75</td>
</tr>
</tbody>
</table>

</div>

<span id="tab:icl" label="tab:icl"></span>

#### Comparison against Task Arithmetic

Task Arithmetic and logits arithmetic are similar in principle, both adjusting the shift between the expert and base model to control knowledge transfer. However, logits arithmetic isn’t constrained by parameter size and can merge across model levels without needing access to specific parameters. Specifically, our method can be applied to combine experts from task arithmetic, integrating both approaches for multi-task learning. We use the same setup as our multi-task scenarios, treating "All" as the average of Seen and Unseen Tasks. In figure <a href="#fig:taskarithmetic" data-reference-type="ref" data-reference="fig:taskarithmetic">5</a>, our method performs well on Unseen Tasks, resulting in an overall improvement of 5.7% compared to task arithmetic. When we combine our method with task arithmetic, we see improvements over task arithmetic alone, achieving the highest overall performance, with increases of 7.2% and 1.5% over task arithmetic and our method alone, respectively.

# Conclusions

In this paper, we introduce a dynamic logit fusion approach in weak-to-strong specialization, which utilizes a series of task-specific small models and allows for adaptive weight allocation among them. Through extensive experiments, we have demonstrated the effectiveness of our approach in both single-task and multi-task settings across various benchmarks. By transferring expertise from the 7B model to the 13B model, we have achieved significant performance improvements, closing the performance gap by 96.4% in single-task scenarios and by 86.3% in multi-task scenarios compared to full fine-tuning of the larger model. Our method also shows promising results on unseen tasks and can integrate in-context learning for single tasks and task arithmetic for multi-task scenarios.

# Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grant No. 62276110, No. 62172039 and in part by the fund of Joint Laboratory of HUST and Pingan Property & Casualty Research (HPL). We thank the Shanghai AI Laboratory for supporting GPU resources. The authors would also like to thank the anonymous reviewers for their comments on improving the quality of this paper.

# References [references]

<div class="thebibliography" markdown="1">

Samuel Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa Git re-basin: Merging models modulo permutation symmetries In *The Eleventh International Conference on Learning Representations*, 2023. **Abstract:** The success of deep learning is due in large part to our ability to solve certain massive non-convex optimization problems with relative ease. Though non-convex optimization is NP-hard, simple algorithms – often variants of stochastic gradient descent – exhibit surprising effectiveness in fitting large neural networks in practice. We argue that neural network loss landscapes often contain (nearly) a single basin after accounting for all possible permutation symmetries of hidden units a la Entezari et al. 2021. We introduce three algorithms to permute the units of one model to bring them into alignment with a reference model in order to merge the two models in weight space. This transformation produces a functionally equivalent set of weights that lie in an approximately convex basin near the reference model. Experimentally, we demonstrate the single basin phenomenon across a variety of model architectures and datasets, including the first (to our knowledge) demonstration of zero-barrier linear mode connectivity between independently trained ResNet models on CIFAR-10. Additionally, we identify intriguing phenomena relating model width and training time to mode connectivity. Finally, we discuss shortcomings of the linear mode connectivity hypothesis, including a counterexample to the single basin theory. (@ainsworth2023git)

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck Llemma: An open language model for mathematics 2024. **Abstract:** We present Llemma, a large language model for mathematics. We continue pretraining Code Llama on the Proof-Pile-2, a mixture of scientific papers, web data containing mathematics, and mathematical code, yielding Llemma. On the MATH benchmark Llemma outperforms all known open base models, as well as the unreleased Minerva model suite on an equi-parameter basis. Moreover, Llemma is capable of tool use and formal theorem proving without any further finetuning. We openly release all artifacts, including 7 billion and 34 billion parameter models, the Proof-Pile-2, and code to replicate our experiments. (@azerbayev2024llemma)

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel itFit: Simple parameter-efficient fine-tuning for transformer-based masked language-models In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 1–9, Dublin, Ireland, May 2022. Association for Computational Linguistics. . URL <https://aclanthology.org/2022.acl-short.1>. **Abstract:** We introduce BitFit, a sparse-finetuning method where only the bias-terms of the model (or a subset of them) are being modified. We show that with small-to-medium training data, applying BitFit on pre-trained BERT models is competitive with (and sometimes better than) fine-tuning the entire model. For larger data, the method is competitive with other sparse fine-tuning methods.Besides their practical utility, these findings are relevant for the question of understanding the commonly-used process of finetuning: they support the hypothesis that finetuning is mainly about exposing knowledge induced by language-modeling training, rather than learning new task-specific linguistic knowledge. (@ben-zaken-etal-2022-bitfit)

Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, and Alexander Kolesnikov Knowledge distillation: A good teacher is patient and consistent In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10925–10934, 2022. **Abstract:** There is a growing discrepancy in computer vision between large-scale models that achieve state-of-the-art performance and models that are affordable in practical applications. In this paper we address this issue and significantly bridge the gap between these two types of models. Throughout our empirical investigation we do not aim to necessarily propose a new method, but strive to identify a robust and effective recipe for making state-of-the-art large scale models affordable in practice. We demonstrate that, when performed correctly, knowledge distillation can be a powerful tool for reducing the size of large models without compromising their performance. In particular, we uncover that there are certain implicit design choices, which may drastically affect the effectiveness of distillation. Our key contribution is the explicit identification of these design choices, which were not previously articulated in the literature. We back up our findings by a comprehensive empirical study, demonstrate compelling results on a wide range of vision datasets and, in particular, obtain a state-of-the-art ResNet-50 model for ImageNet, which achieves 82.8% top-1 accuracy. (@beyer2022knowledge)

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei Language models are few-shot learners In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf>. **Abstract:** Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3’s few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general. (@gpt3)

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jeff Wu Weak-to-strong generalization: Eliciting strong capabilities with weak supervision 2023. **Abstract:** Widely used alignment techniques, such as reinforcement learning from human feedback (RLHF), rely on the ability of humans to supervise model behavior - for example, to evaluate whether a model faithfully followed instructions or generated safe outputs. However, future superhuman models will behave in complex ways too difficult for humans to reliably evaluate; humans will only be able to weakly supervise superhuman models. We study an analogy to this problem: can weak model supervision elicit the full capabilities of a much stronger model? We test this using a range of pretrained language models in the GPT-4 family on natural language processing (NLP), chess, and reward modeling tasks. We find that when we naively finetune strong pretrained models on labels generated by a weak model, they consistently perform better than their weak supervisors, a phenomenon we call weak-to-strong generalization. However, we are still far from recovering the full capabilities of strong models with naive finetuning alone, suggesting that techniques like RLHF may scale poorly to superhuman models without further work. We find that simple methods can often significantly improve weak-to-strong generalization: for example, when finetuning GPT-4 with a GPT-2-level supervisor and an auxiliary confidence loss, we can recover close to GPT-3.5-level performance on NLP tasks. Our results suggest that it is feasible to make empirical progress today on a fundamental challenge of aligning superhuman models. (@burns2023weaktostrong)

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al Scaling instruction-finetuned language models *Journal of Machine Learning Research*, 25 (70): 1–53, 2024. **Abstract:** Finetuning language models on a collection of datasets phrased as instructions has been shown to improve model performance and generalization to unseen tasks. In this paper we explore instruction finetuning with a particular focus on (1) scaling the number of tasks, (2) scaling the model size, and (3) finetuning on chain-of-thought data. We find that instruction finetuning with the above aspects dramatically improves performance on a variety of model classes (PaLM, T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT), and evaluation benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation). For instance, Flan-PaLM 540B instruction-finetuned on 1.8K tasks outperforms PALM 540B by a large margin (+9.4% on average). Flan-PaLM 540B achieves state-of-the-art performance on several benchmarks, such as 75.2% on five-shot MMLU. We also publicly release Flan-T5 checkpoints, which achieve strong few-shot performance even compared to much larger models, such as PaLM 62B. Overall, instruction finetuning is a general method for improving the performance and usability of pretrained language models. (@chung2024scaling)

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al Training verifiers to solve math word problems *arXiv preprint arXiv:2110.14168*, 2021. **Abstract:** State-of-the-art language models can match human performance on many tasks, but they still struggle to robustly perform multi-step mathematical reasoning. To diagnose the failures of current models and support research, we introduce GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems. We find that even the largest transformer models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution. To increase performance, we propose training verifiers to judge the correctness of model completions. At test time, we generate many candidate solutions and select the one ranked highest by the verifier. We demonstrate that verification significantly improves performance on GSM8K, and we provide strong empirical evidence that verification scales more effectively with increased data than a finetuning baseline. (@cobbe2021training)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova : Pre-training of deep bidirectional transformers for language understanding In Jill Burstein, Christy Doran, and Thamar Solorio, editors, *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. . URL <https://aclanthology.org/N19-1423>. **Abstract:** We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement). (@devlin-etal-2019-bert)

Chenghao Fan, Wei Wei, Xiaoye Qu, Zhenyi Lu, Wenfeng Xie, Yu Cheng, and Dangyang Chen Enhancing low-resource relation representations through multi-view decoupling *Proceedings of the AAAI Conference on Artificial Intelligence*, 38 (16): 17968–17976, Mar. 2024. . URL <https://ojs.aaai.org/index.php/AAAI/article/view/29752>. **Abstract:** Recently, prompt-tuning with pre-trained language models (PLMs) has demonstrated the significantly enhancing ability of relation extraction (RE) tasks. However, in low-resource scenarios, where the available training data is scarce, previous prompt-based methods may still perform poorly for prompt-based representation learning due to a superficial understanding of the relation. To this end, we highlight the importance of learning high-quality relation representation in low-resource scenarios for RE, and propose a novel prompt-based relation representation method, named MVRE (Multi-View Relation Extraction), to better leverage the capacity of PLMs to improve the performance of RE within the low-resource prompt-tuning paradigm. Specifically, MVRE decouples each relation into different perspectives to encompass multi-view relation representations for maximizing the likelihood during relation inference. Furthermore, we also design a Global-Local loss and a Dynamic-Initialization method for better alignment of the multi-view relation-representing virtual words, containing the semantics of relation labels during the optimization learning process and initialization. Extensive experiments on three benchmark datasets show that our method can achieve state-of-the-art in low-resource settings. (@Fan_Wei_Qu_Lu_Xie_Cheng_Chen_2024)

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang Deepseek-coder: When the large language model meets programming – the rise of code intelligence 2024. **Abstract:** The rapid development of large language models has revolutionized code intelligence in software development. However, the predominance of closed-source models has restricted extensive research and development. To address this, we introduce the DeepSeek-Coder series, a range of open-source code models with sizes from 1.3B to 33B, trained from scratch on 2 trillion tokens. These models are pre-trained on a high-quality project-level code corpus and employ a fill-in-the-blank task with a 16K window to enhance code generation and infilling. Our extensive evaluations demonstrate that DeepSeek-Coder not only achieves state-of-the-art performance among open-source code models across multiple benchmarks but also surpasses existing closed-source models like Codex and GPT-3.5. Furthermore, DeepSeek-Coder models are under a permissive license that allows for both research and unrestricted commercial use. (@guo2024deepseekcoder)

Soufiane Hayou, Nikhil Ghosh, and Bin Yu Lora+: Efficient low rank adaptation of large models *arXiv preprint arXiv:2402.12354*, 2024. **Abstract:** In this paper, we show that Low Rank Adaptation (LoRA) as originally introduced in Hu et al. (2021) leads to suboptimal finetuning of models with large width (embedding dimension). This is due to the fact that adapter matrices A and B in LoRA are updated with the same learning rate. Using scaling arguments for large width networks, we demonstrate that using the same learning rate for A and B does not allow efficient feature learning. We then show that this suboptimality of LoRA can be corrected simply by setting different learning rates for the LoRA adapter matrices A and B with a well-chosen ratio. We call this proposed algorithm LoRA$+$. In our extensive experiments, LoRA$+$ improves performance (1-2 $\\}%$ improvements) and finetuning speed (up to $\\}sim$ 2X SpeedUp), at the same computational cost as LoRA. (@hayou2024lora+)

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt Measuring massive multitask language understanding In *International Conference on Learning Representations*, 2020. **Abstract:** We propose a new test to measure a text model’s multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model’s academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings. (@hendrycks2020measuring)

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean Distilling the knowledge in a neural network *arXiv preprint arXiv:1503.02531*, 2015. **Abstract:** A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel. (@hinton2015distilling)

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly Parameter-efficient transfer learning for nlp In *International conference on machine learning*, pages 2790–2799. PMLR, 2019. **Abstract:** Fine-tuning large pre-trained models is an effective transfer mechanism in NLP. However, in the presence of many downstream tasks, fine-tuning is parameter inefficient: an entire new model is required for every task. As an alternative, we propose transfer with adapter modules. Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. To demonstrate adapter’s effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task. On GLUE, we attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task. By contrast, fine-tuning trains 100% of the parameters per task. (@houlsby2019parameter)

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al Lora: Low-rank adaptation of large language models In *International Conference on Learning Representations*, 2021. **Abstract:** An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at https://github.com/microsoft/LoRA. (@hu2021lora)

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi Editing models with task arithmetic In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=6t0Kwf8-jrj>. **Abstract:** Changing how pre-trained models behave – e.g., improving their performance on a downstream task or mitigating biases learned during pre-training – is a common practice when developing machine learning systems. In this work, we propose a new paradigm for steering the behavior of neural networks, centered around \\}textit{task vectors}. A task vector specifies a direction in the weight space of a pre-trained model, such that movement in that direction improves performance on the task. We build task vectors by subtracting the weights of a pre-trained model from the weights of the same model after fine-tuning on a task. We show that these task vectors can be modified and combined together through arithmetic operations such as negation and addition, and the behavior of the resulting model is steered accordingly. Negating a task vector decreases performance on the target task, with little change in model behavior on control tasks. Moreover, adding task vectors together can improve performance on multiple tasks at once. Finally, when tasks are linked by an analogy relationship of the form “A is to B as C is to D", combining task vectors from three of the tasks can improve performance on the fourth, even when no data from the fourth task is used for training. Overall, our experiments with several models, modalities and tasks show that task arithmetic is a simple, efficient and effective way of editing models. (@ilharco2023editing)

Jiaming Ji, Boyuan Chen, Hantao Lou, Donghai Hong, Borong Zhang, Xuehai Pan, Juntao Dai, and Yaodong Yang Aligner: Achieving efficient alignment through weak-to-strong correction 2024. **Abstract:** Efforts to align Large Language Models (LLMs) are mainly conducted via Reinforcement Learning from Human Feedback (RLHF) methods. However, RLHF encounters major challenges including training reward models, actor-critic engineering, and importantly, it requires access to LLM parameters. Here we introduce Aligner, a new efficient alignment paradigm that bypasses the whole RLHF process by learning the correctional residuals between the aligned and the unaligned answers. Our Aligner offers several key advantages. Firstly, it is an autoregressive seq2seq model that is trained on the query-answer-correction dataset via supervised learning; this offers a parameter-efficient alignment solution with minimal resources. Secondly, the Aligner facilitates weak-to-strong generalization; finetuning large pretrained models by Aligner’s supervisory signals demonstrates strong performance boost. Thirdly, Aligner functions as a model-agnostic plug-and-play module, allowing for its direct application on different open-source and API-based models. Remarkably, Aligner-7B improves 11 different LLMs by 21.9% in helpfulness and 23.8% in harmlessness on average (GPT-4 by 17.5% and 26.9%). When finetuning (strong) Llama2-70B with (weak) Aligner-13B’s supervision, we can improve Llama2 by 8.2% in helpfulness and 61.6% in harmlessness. See our dataset and code at https://aligner2024.github.io (@ji2024aligner)

Feihu Jin, Jiajun Zhang, and Chengqing Zong Parameter-efficient tuning for large language model without calculating its gradients In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 321–330, Singapore, December 2023. Association for Computational Linguistics. . URL <https://aclanthology.org/2023.emnlp-main.22>. **Abstract:** Fine-tuning all parameters of large language models (LLMs) requires significant computational resources and is time-consuming. Recent parameter-efficient tuning methods such as Adapter tuning, Prefix tuning, and LoRA allow for updating a small subset of parameters in large language models. However, they can only save approximately 30% of the training memory requirements, due to the problem that gradient computation and backpropagation are still necessary for these methods. This paper proposes a novel parameter-efficient tuning method for LLMs without calculating their gradients. Leveraging the discernible similarities between the parameter-efficient modules of the same task learned by both large and small language models, we put forward a strategy for transferring the parameter-efficient modules, originally derived from small language models to much larger ones. To ensure a smooth and effective adaptation process, we further introduce a Bridge model to guarantee dimensional consistency while also stimulating a dynamic interaction between the models. We demonstrate the effectiveness of our method using the T5 and GPT-2 series of language models on the SuperGLUE benchmark. Our method achieves comparable performance to both fine-tuning and parameter-efficient tuning on large language models without needing gradient-based optimization. Additionally, our method achieves up to 5.7x memory reduction compared to parameter-efficient tuning. (@jin-etal-2023-parameter)

Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng Dataless knowledge fusion by merging weights of language models In *The Eleventh International Conference on Learning Representations*, 2022. **Abstract:** Fine-tuning pre-trained language models has become the prevalent paradigm for building downstream NLP models. Oftentimes fine-tuned models are readily available but their training data is not, due to data privacy or intellectual property concerns. This creates a barrier to fusing knowledge across individual models to yield a better single model. In this paper, we study the problem of merging individual models built on different training data sets to obtain a single model that performs well both across all data set domains and can generalize on out-of-domain data. We propose a dataless knowledge fusion method that merges models in their parameter space, guided by weights that minimize prediction differences between the merged model and the individual models. Over a battery of evaluation settings, we show that the proposed method significantly outperforms baselines such as Fisher-weighted averaging or model ensembling. Further, we find that our method is a promising alternative to multi-task learning that can preserve or sometimes improve over the individual models without access to the training data. Finally, model merging is more efficient than training a multi-task model, thus making it applicable to a wider set of scenarios. (@2023DatalessKnowledgeFusion)

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611, 2017. **Abstract:** We present TriviaQA, a challenging reading comprehension dataset containing over 650K question-answer-evidence triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions. We show that, in comparison to other recently introduced large-scale datasets, TriviaQA (1) has relatively complex, compositional questions, (2) has considerable syntactic and lexical variability between questions and corresponding answer-evidence sentences, and (3) requires more cross sentence reasoning to find answers. We also present two baseline algorithms: a feature-based classifier and a state-of-the-art neural network, that performs well on SQuAD reading comprehension. Neither approach comes close to human performance (23% and 40% vs. 80%), suggesting that TriviaQA is a challenging testbed that is worth significant future study. (@joshi2017triviaqa)

Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, and James Henderson Parameter-efficient multi-task fine-tuning for transformers via shared hypernetworks In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 565–576, Online, August 2021. Association for Computational Linguistics. . URL <https://aclanthology.org/2021.acl-long.47>. **Abstract:** Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, James Henderson. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2021. (@karimi-mahabadi-etal-2021-parameter)

Dawid Jan Kopiczko, Tijmen Blankevoort, and Yuki M Asano VeRA: Vector-based random matrix adaptation In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=NjNfLdxr3A>. **Abstract:** Low-rank adapation (LoRA) is a popular method that reduces the number of trainable parameters when finetuning large language models, but still faces acute storage challenges when scaling to even larger models or deploying numerous per-user or per-task adapted models. In this work, we present Vector-based Random Matrix Adaptation (VeRA), which significantly reduces the number of trainable parameters compared to LoRA, yet maintains the same performance. It achieves this by using a single pair of low-rank matrices shared across all layers and learning small scaling vectors instead. We demonstrate its effectiveness on the GLUE and E2E benchmarks, image classification tasks, and show its application in instruction-tuning of 7B and 13B language models. (@kopiczko2024vera)

Solomon Kullback and Richard A Leibler On information and sufficiency *The annals of mathematical statistics*, 22 (1): 79–86, 1951. (@kullback1951information)

Yaniv Leviathan, Matan Kalman, and Yossi Matias Fast inference from transformers via speculative decoding In *International Conference on Machine Learning*, pages 19274–19286. PMLR, 2023. **Abstract:** Inference from large autoregressive models like Transformers is slow - decoding K tokens takes K serial runs of the model. In this work we introduce speculative decoding - an algorithm to sample from autoregressive models faster without any changes to the outputs, by computing several tokens in parallel. At the heart of our approach lie the observations that (1) hard language-modeling tasks often include easier subtasks that can be approximated well by more efficient models, and (2) using speculative execution and a novel sampling method, we can make exact decoding from the large models faster, by running them in parallel on the outputs of the approximation models, potentially generating several tokens concurrently, and without changing the distribution. Our method can accelerate existing off-the-shelf models without retraining or architecture changes. We demonstrate it on T5-XXL and show a 2X-3X acceleration compared to the standard T5X implementation, with identical outputs. (@leviathan2023fast)

Xiang Lisa Li and Percy Liang Prefix-tuning: Optimizing continuous prompts for generation In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 4582–4597, Online, August 2021. Association for Computational Linguistics. . URL <https://aclanthology.org/2021.acl-long.353>. **Abstract:** Xiang Lisa Li, Percy Liang. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2021. (@li-liang-2021-prefix)

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis Contrastive decoding: Open-ended text generation as optimization In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 12286–12312, Toronto, Canada, July 2023. Association for Computational Linguistics. . URL <https://aclanthology.org/2023.acl-long.687>. **Abstract:** Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, Mike Lewis. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023. (@li-etal-2023-contrastive)

Vladislav Lialin, Vijeta Deshpande, and Anna Rumshisky Scaling down to scale up: A guide to parameter-efficient fine-tuning 2023. **Abstract:** This paper presents a systematic overview and comparison of parameter-efficient fine-tuning methods covering over 40 papers published between February 2019 and February 2023. These methods aim to resolve the infeasibility and impracticality of fine-tuning large language models by only training a small set of parameters. We provide a taxonomy that covers a broad range of methods and present a detailed method comparison with a specific focus on real-life efficiency and fine-tuning multibillion-scale language models. (@lialin2023scaling)

Petro Liashchynskyi and Pavlo Liashchynskyi Grid search, random search, genetic algorithm: a big comparison for nas *arXiv preprint arXiv:1912.06059*, 2019. **Abstract:** In this paper, we compare the three most popular algorithms for hyperparameter optimization (Grid Search, Random Search, and Genetic Algorithm) and attempt to use them for neural architecture search (NAS). We use these algorithms for building a convolutional neural network (search architecture). Experimental results on CIFAR-10 dataset further demonstrate the performance difference between compared algorithms. The comparison results are based on the execution time of the above algorithms and accuracy of the proposed models. (@liashchynskyi2019grid)

Lucas Liebenwein, Cenk Baykal, Brandon Carter, David Gifford, and Daniela Rus Lost in pruning: The effects of pruning neural networks beyond test accuracy *Proceedings of Machine Learning and Systems*, 3: 93–138, 2021. **Abstract:** Neural network pruning is a popular technique used to reduce the inference costs of modern, potentially overparameterized, networks. Starting from a pre-trained network, the process is as follows: remove redundant parameters, retrain, and repeat while maintaining the same test accuracy. The result is a model that is a fraction of the size of the original with comparable predictive performance (test accuracy). Here, we reassess and evaluate whether the use of test accuracy alone in the terminating condition is sufficient to ensure that the resulting model performs well across a wide spectrum of "harder" metrics such as generalization to out-of-distribution data and resilience to noise. Across evaluations on varying architectures and data sets, we find that pruned networks effectively approximate the unpruned model, however, the prune ratio at which pruned networks achieve commensurate performance varies significantly across tasks. These results call into question the extent of \\}emph{genuine} overparameterization in deep learning and raise concerns about the practicability of deploying pruned networks, specifically in the context of safety-critical systems, unless they are widely evaluated beyond test accuracy to reliably predict their performance. Our code is available at https://github.com/lucaslie/torchprune. (@liebenwein2021lost)

Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chandra Bhagavatula, and Yejin Choi The unlocking spell on base llms: Rethinking alignment via in-context learning 2023. **Abstract:** The alignment tuning process of large language models (LLMs) typically involves instruction learning through supervised fine-tuning (SFT) and preference tuning via reinforcement learning from human feedback (RLHF). A recent study, LIMA (Zhou et al. 2023), shows that using merely 1K examples for SFT can achieve significant alignment performance as well, suggesting that the effect of alignment tuning might be "superficial." This raises questions about how exactly the alignment tuning transforms a base LLM. We analyze the effect of alignment tuning by examining the token distribution shift between base LLMs and their aligned counterpart. Our findings reveal that base LLMs and their alignment-tuned versions perform nearly identically in decoding on the majority of token positions. Most distribution shifts occur with stylistic tokens. These direct evidence strongly supports the Superficial Alignment Hypothesis suggested by LIMA. Based on these findings, we rethink the alignment of LLMs by posing the research question: how effectively can we align base LLMs without SFT or RLHF? To address this, we introduce a simple, tuning-free alignment method, URIAL. URIAL achieves effective alignment purely through in-context learning (ICL) with base LLMs, requiring as few as three constant stylistic examples and a system prompt. We conduct a fine-grained and interpretable evaluation on a diverse set of examples, named JUST-EVAL-INSTRUCT. Results demonstrate that base LLMs with URIAL can match or even surpass the performance of LLMs aligned with SFT or SFT+RLHF. We show that the gap between tuning-free and tuning-based alignment methods can be significantly reduced through strategic prompting and ICL. Our findings on the superficial nature of alignment tuning and results with URIAL suggest that deeper analysis and theoretical understanding of alignment is crucial to future LLM research. (@lin2023unlocking)

Stephanie Lin, Jacob Hilton, and Owain Evans Truthfulqa: Measuring how models mimic human falsehoods In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3214–3252, 2022. **Abstract:** We propose a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. We crafted questions that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts. We tested GPT-3, GPT-Neo/J, GPT-2 and a T5-based model. The best model was truthful on 58% of questions, while human performance was 94%. Models generated many false answers that mimic popular misconceptions and have the potential to deceive humans. The largest models were generally the least truthful. This contrasts with other NLP tasks, where performance improves with model size. However, this result is expected if false answers are learned from the training distribution. We suggest that scaling up models alone is less promising for improving truthfulness than fine-tuning using training objectives other than imitation of text from the web. (@lin2022truthfulqa)

Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, and Noah A Smith Tuning language models by proxy *arXiv preprint arXiv:2401.08565*, 2024. **Abstract:** Despite the general capabilities of large pretrained language models, they consistently benefit from further adaptation to better achieve desired behaviors. However, tuning these models has become increasingly resource-intensive, or impossible when model weights are private. We introduce proxy-tuning, a lightweight decoding-time algorithm that operates on top of black-box LMs to achieve the same end as direct tuning, but by accessing only its predictions over the output vocabulary, not its parameters. Our method tunes a smaller LM, then applies the difference between the predictions of the small tuned and untuned LMs to shift the original predictions of the larger untuned model in the direction of tuning, while retaining the benefits of larger-scale pretraining. In experiments, when we apply proxy-tuning to Llama2-70B using proxies of only 7B size, we can close 88% of the gap between Llama2-70B and its truly-tuned chat version, when evaluated across knowledge, reasoning, and safety benchmarks. We then demonstrate the generality of proxy-tuning by applying it to domain adaptation on code, and task-specific finetuning on question-answering and math problems. Finally, we show how to proxy-tune a truly black-box LM, GPT-3.5, for temporal adaptation, increasing its knowledge about recent events. Our work demonstrates the promise of using small tuned LMs to efficiently customize large, potentially proprietary LMs through decoding-time guidance. (@liu2024tuning)

Jialin Liu, Antoine Moreau, Mike Preuss, Jeremy Rapin, Baptiste Roziere, Fabien Teytaud, and Olivier Teytaud Versatile black-box optimization In *Proceedings of the 2020 Genetic and Evolutionary Computation Conference*, pages 620–628, 2020. **Abstract:** Choosing automatically the right algorithm using problem descriptors is a classical component of combinatorial optimization. It is also a good tool for making evolutionary algorithms fast, robust and versatile. We present Shiwa, an algorithm good at both discrete and continuous, noisy and noise-free, sequential and parallel, black-box optimization. Our algorithm is experimentally compared to competitors on YABBOB, a BBOB comparable testbed, and on some variants of it, and then validated on several real world testbeds. (@liu2020versatile)

Zhenyi Lu, Chenghao Fan, Wei Wei, Xiaoye Qu, Dangyang Chen, and Yu Cheng Twin-merging: Dynamic integration of modular expertise in model merging *arXiv preprint arXiv:2406.15479*, 2024. **Abstract:** In the era of large language models, model merging is a promising way to combine multiple task-specific models into a single multitask model without extra training. However, two challenges remain: (a) interference between different models and (b) heterogeneous data during testing. Traditional model merging methods often show significant performance gaps compared to fine-tuned models due to these issues. Additionally, a one-size-fits-all model lacks flexibility for diverse test data, leading to performance degradation. We show that both shared and exclusive task-specific knowledge are crucial for merging performance, but directly merging exclusive knowledge hinders overall performance. In view of this, we propose Twin-Merging, a method that encompasses two principal stages: (1) modularizing knowledge into shared and exclusive components, with compression to reduce redundancy and enhance efficiency; (2) dynamically merging shared and task-specific knowledge based on the input. This approach narrows the performance gap between merged and fine-tuned models and improves adaptability to heterogeneous data. Extensive experiments on $20$ datasets for both language and vision tasks demonstrate the effectiveness of our method, showing an average improvement of $28.34\\}%$ in absolute normalized score for discriminative tasks and even surpassing the fine-tuned upper bound on the generative tasks. Our implementation is available in \\}url{https://github.com/LZY-the-boys/Twin-Merging} (@lu2024twin)

Zhenyi Lu, Jie Tian, Wei Wei, Xiaoye Qu, Yu Cheng, Dangyang Chen, et al Mitigating boundary ambiguity and inherent bias for text classification in the era of large language models *arXiv preprint arXiv:2406.07001*, 2024. **Abstract:** Text classification is a crucial task encountered frequently in practical scenarios, yet it is still under-explored in the era of large language models (LLMs). This study shows that LLMs are vulnerable to changes in the number and arrangement of options in text classification. Our extensive empirical analyses reveal that the key bottleneck arises from ambiguous decision boundaries and inherent biases towards specific tokens and positions. To mitigate these issues, we make the first attempt and propose a novel two-stage classification framework for LLMs. Our approach is grounded in the empirical observation that pairwise comparisons can effectively alleviate boundary ambiguity and inherent bias. Specifically, we begin with a self-reduction technique to efficiently narrow down numerous options, which contributes to reduced decision space and a faster comparison process. Subsequently, pairwise contrastive comparisons are employed in a chain-of-thought manner to draw out nuances and distinguish confusable options, thus refining the ambiguous decision boundary. Extensive experiments on four datasets (Banking77, HWU64, LIU54, and Clinic150) verify the effectiveness of our framework. Furthermore, benefitting from our framework, various LLMs can achieve consistent improvements. Our code and data are available in \\}url{https://github.com/Chuge0335/PC-CoT}. (@lu2024mitigating)

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct 2023. **Abstract:** Large language models (LLMs), such as GPT-4, have shown remarkable performance in natural language processing (NLP) tasks, including challenging mathematical reasoning. However, most existing open-source models are only pre-trained on large-scale internet data and without math-related optimization. In this paper, we present WizardMath, which enhances the mathematical reasoning abilities of Llama-2, by applying our proposed Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method to the domain of math. Through extensive experiments on two mathematical reasoning benchmarks, namely GSM8k and MATH, we reveal the extraordinary capabilities of our model. WizardMath surpasses all other open-source LLMs by a substantial margin. Furthermore, our model even outperforms ChatGPT-3.5, Claude Instant-1, PaLM-2 and Minerva on GSM8k, simultaneously surpasses Text-davinci-002, PaLM-1 and GPT-3 on MATH. More details and model weights are public at https://github.com/nlpxucan/WizardLM and https://huggingface.co/WizardLM. (@luo2023wizardmath)

Michael S Matena and Colin Raffel Merging models with fisher-weighted averaging In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, *Advances in Neural Information Processing Systems*, 2022. URL <https://openreview.net/forum?id=LSKlp_aceOC>. **Abstract:** Averaging the parameters of models that have the same architecture and initialization can provide a means of combining their respective capabilities. In this paper, we take the perspective that this "merging" operation can be seen as choosing parameters that approximately maximize the joint likelihood of the posteriors of the models’ parameters. Computing a simple average of the models’ parameters therefore corresponds to making an isotropic Gaussian approximation to their posteriors. We develop an alternative merging procedure based on the Laplace approximation where we approximate each model’s posterior as a Gaussian distribution whose precision matrix corresponds to its Fisher information. We first show that our "Fisher merging" technique provides a performance boost in settings where simple parameter averaging is currently used – specifically, robust fine-tuning and model ensembling. Then, we compare merging to standard gradient-based transfer learning and demonstrate that merging enables a fundamentally different method for transferring capabilities across models. Specifically, we show that Fisher merging is competitive with gradient-based transfer learning approaches (while being significantly cheaper) in intermediate-task training and domain-adaptive pre-training. We also show that our merging procedure makes it possible to combine models in previously unexplored ways. We release our code to facilitate future research into methods for merging models. (@matena2022merging)

Michael S Matena and Colin A Raffel Merging models with fisher-weighted averaging *Advances in Neural Information Processing Systems*, 2022. **Abstract:** Averaging the parameters of models that have the same architecture and initialization can provide a means of combining their respective capabilities. In this paper, we take the perspective that this "merging" operation can be seen as choosing parameters that approximately maximize the joint likelihood of the posteriors of the models’ parameters. Computing a simple average of the models’ parameters therefore corresponds to making an isotropic Gaussian approximation to their posteriors. We develop an alternative merging procedure based on the Laplace approximation where we approximate each model’s posterior as a Gaussian distribution whose precision matrix corresponds to its Fisher information. We first show that our "Fisher merging" technique provides a performance boost in settings where simple parameter averaging is currently used – specifically, robust fine-tuning and model ensembling. Then, we compare merging to standard gradient-based transfer learning and demonstrate that merging enables a fundamentally different method for transferring capabilities across models. Specifically, we show that Fisher merging is competitive with gradient-based transfer learning approaches (while being significantly cheaper) in intermediate-task training and domain-adaptive pre-training. We also show that our merging procedure makes it possible to combine models in previously unexplored ways. We release our code to facilitate future research into methods for merging models. (@2022MergingModelsFisherWeighted)

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia Specinfer: Accelerating large language model serving with tree-based speculative inference and verification In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, ASPLOS ’24. ACM, April 2024. . URL <http://dx.doi.org/10.1145/3620666.3651335>. **Abstract:** This paper introduces SpecInfer, a system that accelerates generative large language model (LLM) serving with treebased speculative inference and verification.The key idea behind SpecInfer is leveraging small speculative models to predict the LLM’s outputs; the predictions are organized as a token tree, whose nodes each represent a candidate token sequence.The correctness of all candidate token sequences represented by a token tree is verified against the LLM in parallel using a novel tree-based parallel decoding mechanism.SpecInfer uses an LLM as a token tree verifier instead of an incremental decoder, which significantly reduces the end-toend latency and computational requirement for serving generative LLMs while provably preserving model quality. (@Miao_2024)

Eric Mitchell, Rafael Rafailov, Archit Sharma, Chelsea Finn, and Christopher D Manning An emulator for fine-tuning large language models using small language models In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=Eo7kv0sllr>. **Abstract:** Widely used language models (LMs) are typically built by scaling up a two-stage training pipeline: a pre-training stage that uses a very large, diverse dataset of text and a fine-tuning (sometimes, ’alignment’) stage that uses targeted examples or other specifications of desired behaviors. While it has been hypothesized that knowledge and skills come from pre-training, and fine-tuning mostly filters this knowledge and skillset, this intuition has not been extensively tested. To aid in doing so, we introduce a novel technique for decoupling the knowledge and skills gained in these two stages, enabling a direct answer to the question, "What would happen if we combined the knowledge learned by a large model during pre-training with the knowledge learned by a small model during fine-tuning (or vice versa)?" Using an RL-based framework derived from recent developments in learning from human preferences, we introduce emulated fine-tuning (EFT), a principled and practical method for sampling from a distribution that approximates (or ’emulates’) the result of pre-training and fine-tuning at different scales. Our experiments with EFT show that scaling up fine-tuning tends to improve helpfulness, while scaling up pre-training tends to improve factuality. Beyond decoupling scale, we show that EFT enables test-time adjustment of competing behavioral traits like helpfulness and harmlessness without additional training. Finally, a special case of emulated fine-tuning, which we call LM up-scaling, avoids resource-intensive fine-tuning of large pre-trained models by ensembling them with small fine-tuned models, essentially emulating the result of fine-tuning the large pre-trained model. Up-scaling consistently improves helpfulness and factuality of instruction-following models in the Llama, Llama-2, and Falcon families, without additional hyperparameters or training. (@mitchell2024an)

David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean Carbon emissions and large neural network training 2021. **Abstract:** The computation demand for machine learning (ML) has grown rapidly recently, which comes with a number of costs. Estimating the energy cost helps measure its environmental impact and finding greener strategies, yet it is challenging without detailed information. We calculate the energy use and carbon footprint of several recent large models-T5, Meena, GShard, Switch Transformer, and GPT-3-and refine earlier estimates for the neural architecture search that found Evolved Transformer. We highlight the following opportunities to improve energy efficiency and CO2 equivalent emissions (CO2e): Large but sparsely activated DNNs can consume \<1/10th the energy of large, dense DNNs without sacrificing accuracy despite using as many or even more parameters. Geographic location matters for ML workload scheduling since the fraction of carbon-free energy and resulting CO2e vary \~5X-10X, even within the same country and the same organization. We are now optimizing where and when large models are trained. Specific datacenter infrastructure matters, as Cloud datacenters can be \~1.4-2X more energy efficient than typical datacenters, and the ML-oriented accelerators inside them can be \~2-5X more effective than off-the-shelf systems. Remarkably, the choice of DNN, datacenter, and processor can reduce the carbon footprint up to \~100-1000X. These large factors also make retroactive estimates of energy cost difficult. To avoid miscalculations, we believe ML papers requiring large computational resources should make energy consumption and CO2e explicit when practical. We are working to be more transparent about energy use and CO2e in our future research. To help reduce the carbon footprint of ML, we believe energy usage and CO2e should be a key metric in evaluating models, and we are collaborating with MLPerf developers to include energy usage during training and inference in this industry standard benchmark. (@patterson2021carbon)

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn Direct preference optimization: Your language model is secretly a reward model In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, *Advances in Neural Information Processing Systems*, volume 36, pages 53728–53741. Curran Associates, Inc., 2023. URL <https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf>. **Abstract:** While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train. (@rafailov2024direct)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu Exploring the limits of transfer learning with a unified text-to-text transformer *Journal of Machine Learning Research*, 21 (140): 1–67, 2020. URL <http://jmlr.org/papers/v21/20-074.html>. **Abstract:** Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code. (@JMLR:v21:20-074)

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve Code llama: Open foundation models for code 2024. **Abstract:** We release Code Llama, a family of large language models for code based on Llama 2 providing state-of-the-art performance among open models, infilling capabilities, support for large input contexts, and zero-shot instruction following ability for programming tasks. We provide multiple flavors to cover a wide range of applications: foundation models (Code Llama), Python specializations (Code Llama - Python), and instruction-following models (Code Llama - Instruct) with 7B, 13B, 34B and 70B parameters each. All models are trained on sequences of 16k tokens and show improvements on inputs with up to 100k tokens. 7B, 13B and 70B Code Llama and Code Llama - Instruct variants support infilling based on surrounding content. Code Llama reaches state-of-the-art performance among open models on several code benchmarks, with scores of up to 67% and 65% on HumanEval and MBPP, respectively. Notably, Code Llama - Python 7B outperforms Llama 2 70B on HumanEval and MBPP, and all our models outperform every other publicly available model on MultiPL-E. We release Code Llama under a permissive license that allows for both research and commercial use. (@rozière2024code)

Timo Schick, Jane Dwivedi-Yu, Roberto Dessı̀, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom Toolformer: Language models can teach themselves to use tools *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q\\}&A system, two different search engines, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities. (@schick2024toolformer)

Abigail See, Peter J. Liu, and Christopher D. Manning Get to the point: Summarization with pointer-generator networks In Regina Barzilay and Min-Yen Kan, editors, *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1073–1083, Vancouver, Canada, July 2017. Association for Computational Linguistics. . URL <https://aclanthology.org/P17-1099>. **Abstract:** Neural sequence-to-sequence models have provided a viable new approach for abstractive text summarization (meaning they are not restricted to simply selecting and rearranging passages from the original text). However, these models have two shortcomings: they are liable to reproduce factual details inaccurately, and they tend to repeat themselves. In this work we propose a novel architecture that augments the standard sequence-to-sequence attentional model in two orthogonal ways. First, we use a hybrid pointer-generator network that can copy words from the source text via pointing, which aids accurate reproduction of information, while retaining the ability to produce novel words through the generator. Second, we use coverage to keep track of what has been summarized, which discourages repetition. We apply our model to the CNN / Daily Mail summarization task, outperforming the current abstractive state-of-the-art by at least 2 ROUGE points. (@see-etal-2017-get)

Shannon Zejiang Shen, Hunter Lang, Bailin Wang, Yoon Kim, and David Sontag Learning to decode collaboratively with multiple language models 2024. **Abstract:** We propose a method to teach multiple large language models (LLM) to collaborate by interleaving their generations at the token level. We model the decision of which LLM generates the next token as a latent variable. By optimizing the marginal likelihood of a training set under our latent variable model, the base LLM automatically learns when to generate itself and when to call on one of the “assistant” language models to generate, all without direct supervision. Token-level collaboration during decoding allows for a fusion of each model’s expertise in a manner tailored to the specific task at hand. Our collaborative decoding is especially useful in cross-domain settings where a generalist base LLM learns to invoke domain expert models. On instruction-following, domain-specific QA, and reasoning tasks, we show that the performance of the joint system exceeds that of the individual models. Through qualitative analysis of the learned latent decisions, we show models trained with our method exhibit several interesting collaboration patterns, e.g., template-filling. Our code is available at https://github.com/clinicalml/co-llm. (@shen2024learning)

Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A Alemi, and Andrew G Wilson Does knowledge distillation really work? *Advances in Neural Information Processing Systems*, 34: 6906–6919, 2021. **Abstract:** Knowledge distillation is a popular technique for training a small student network to emulate a larger teacher model, such as an ensemble of networks. We show that while knowledge distillation can improve student generalization, it does not typically work as it is commonly understood: there often remains a surprisingly large discrepancy between the predictive distributions of the teacher and the student, even in cases when the student has the capacity to perfectly match the teacher. We identify difficulties in optimization as a key reason for why the student is unable to match the teacher. We also show how the details of the dataset used for distillation play a role in how closely the student matches the teacher – and that more closely matching the teacher paradoxically does not always lead to better student generalization. (@stanton2021does)

Zhaochen Su, Juntao Li, Jun Zhang, Tong Zhu, Xiaoye Qu, Pan Zhou, Yan Bowen, Yu Cheng, et al Living in the moment: Can large language models grasp co-temporal reasoning? *arXiv preprint arXiv:2406.09072*, 2024. **Abstract:** Temporal reasoning is fundamental for large language models (LLMs) to comprehend the world. Current temporal reasoning datasets are limited to questions about single or isolated events, falling short in mirroring the realistic temporal characteristics involving concurrent nature and intricate temporal interconnections. In this paper, we introduce CoTempQA, a comprehensive co-temporal Question Answering (QA) benchmark containing four co-temporal scenarios (Equal, Overlap, During, Mix) with 4,748 samples for evaluating the co-temporal comprehension and reasoning abilities of LLMs. Our extensive experiments reveal a significant gap between the performance of current LLMs and human-level reasoning on CoTempQA tasks. Even when enhanced with Chain of Thought (CoT) methodologies, models consistently struggle with our task. In our preliminary exploration, we discovered that mathematical reasoning plays a significant role in handling co-temporal events and proposed a strategy to boost LLMs’ co-temporal reasoning from a mathematical perspective. We hope that our CoTempQA datasets will encourage further advancements in improving the co-temporal reasoning capabilities of LLMs. Our code is available at https://github.com/zhaochen0110/Cotempqa. (@su2024living)

Zhaochen Su, Jun Zhang, Xiaoye Qu, Tong Zhu, Yanshu Li, Jiashuo Sun, Juntao Li, Min Zhang, and Yu Cheng Conflictbank: A benchmark for evaluating the influence of knowledge conflicts in llm *arXiv preprint arXiv:2408.12076*, 2024. **Abstract:** Large language models (LLMs) have achieved impressive advancements across numerous disciplines, yet the critical issue of knowledge conflicts, a major source of hallucinations, has rarely been studied. Only a few research explored the conflicts between the inherent knowledge of LLMs and the retrieved contextual knowledge. However, a thorough assessment of knowledge conflict in LLMs is still missing. Motivated by this research gap, we present ConflictBank, the first comprehensive benchmark developed to systematically evaluate knowledge conflicts from three aspects: (i) conflicts encountered in retrieved knowledge, (ii) conflicts within the models’ encoded knowledge, and (iii) the interplay between these conflict forms. Our investigation delves into four model families and twelve LLM instances, meticulously analyzing conflicts stemming from misinformation, temporal discrepancies, and semantic divergences. Based on our proposed novel construction framework, we create 7,453,853 claim-evidence pairs and 553,117 QA pairs. We present numerous findings on model scale, conflict causes, and conflict types. We hope our ConflictBank benchmark will help the community better understand model behavior in conflicts and develop more reliable LLMs. (@su2024conflictbank)

Zhaochen Su, Jun Zhang, Tong Zhu, Xiaoye Qu, Juntao Li, Min Zhang, and Yu Cheng Timo: Towards better temporal reasoning for language models *arXiv preprint arXiv:2406.14192*, 2024. **Abstract:** Reasoning about time is essential for Large Language Models (LLMs) to understand the world. Previous works focus on solving specific tasks, primarily on time-sensitive question answering. While these methods have proven effective, they cannot generalize to a wider spectrum of temporal reasoning tasks. Therefore, we propose a crucial question: Can we build a universal framework to handle a variety of temporal reasoning tasks? To that end, we systematically study 38 temporal reasoning tasks. Based on the observation that 19 tasks are directly related to mathematics, we first leverage the available mathematical dataset to set a solid foundation for temporal reasoning. However, the in-depth study indicates that focusing solely on mathematical enhancement falls short of addressing pure temporal reasoning tasks. To mitigate this limitation, we propose a simple but effective self-critic temporal optimization method to enhance the model’s temporal reasoning capabilities without sacrificing general task abilities. Finally, we develop Timo, a model designed to excel in temporal reasoning at the 7B and 13B scales. Notably, Timo outperforms the counterpart LLMs by 10.0 and 7.6 in average accuracy scores and achieves the new state-of-the-art (SOTA) performance of comparable size. Extensive experiments further validate our framework’s effectiveness and its generalization across diverse temporal tasks. The code is available at https://github.com/zhaochen0110/Timo. (@su2024timo)

Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun Toolalpaca: Generalized tool learning for language models with 3000 simulated cases, 2023. **Abstract:** Enabling large language models to utilize real-world tools effectively is crucial for achieving embodied intelligence. Existing approaches to tool learning have either primarily relied on extremely large language models, such as GPT-4, to attain generalized tool-use abilities in a zero-shot manner, or utilized supervised learning to train limited scopes of tools on compact models. However, it remains uncertain whether smaller language models can achieve generalized tool-use abilities without tool-specific training. To address this question, this paper introduces ToolAlpaca, a novel framework designed to automatically generate a diverse tool-use corpus and learn generalized tool-use abilities on compact language models with minimal human intervention. Specifically, ToolAlpaca first automatically creates a highly diversified tool-use corpus by building a multi-agent simulation environment. The corpus contains 3938 tool-use instances from more than 400 real-world tool APIs spanning 50 distinct categories. Subsequently, the constructed corpus is employed to fine-tune compact language models, resulting in two models, namely ToolAlpaca-7B and ToolAlpaca-13B, respectively. Finally, we evaluate the ability of these models to utilize previously unseen tools without specific training. Experimental results demonstrate that ToolAlpaca achieves effective generalized tool-use capabilities comparable to those of extremely large language models like GPT-3.5, demonstrating that learning generalized tool-use ability is feasible for compact language models. (@tang2023toolalpaca)

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom Llama 2: Open foundation and fine-tuned chat models 2023. **Abstract:** In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs. (@llama2)

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al Llama 2: Open foundation and fine-tuned chat models *arXiv preprint arXiv:2307.09288*, 2023. **Abstract:** In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs. (@touvron2023llama)

Hongyi Wang, Felipe Maia Polo, Yuekai Sun, Souvik Kundu, Eric Xing, and Mikhail Yurochkin Fusing models with complementary expertise In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=PhMrGCMIRL>. **Abstract:** Training AI models that generalize across tasks and domains has long been among the open problems driving AI research. The emergence of Foundation Models made it easier to obtain expert models for a given task, but the heterogeneity of data that may be encountered at test time often means that any single expert is insufficient. We consider the Fusion of Experts (FoE) problem of fusing outputs of expert models with complementary knowledge of the data distribution and formulate it as an instance of supervised learning. Our method is applicable to both discriminative and generative tasks and leads to significant performance improvements in image and text classification, text summarization, multiple-choice QA, and automatic evaluation of generated text. We also extend our method to the "frugal" setting where it is desired to reduce the number of expert model evaluations at test time. (@wang2024fusing)

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time In *International Conference on Machine Learning*, 2022. **Abstract:** The conventional recipe for maximizing model accuracy is to (1) train multiple models with various hyperparameters and (2) pick the individual model which performs best on a held-out validation set, discarding the remainder. In this paper, we revisit the second step of this procedure in the context of fine-tuning large pre-trained models, where fine-tuned models often appear to lie in a single low error basin. We show that averaging the weights of multiple models fine-tuned with different hyperparameter configurations often improves accuracy and robustness. Unlike a conventional ensemble, we may average many models without incurring any additional inference or memory costs – we call the results "model soups." When fine-tuning large pre-trained models such as CLIP, ALIGN, and a ViT-G pre-trained on JFT, our soup recipe provides significant improvements over the best model in a hyperparameter sweep on ImageNet. The resulting ViT-G model, which attains 90.94% top-1 accuracy on ImageNet, achieved a new state of the art. Furthermore, we show that the model soup approach extends to multiple image classification and natural language processing tasks, improves out-of-distribution performance, and improves zero-shot performance on new downstream tasks. Finally, we analytically relate the performance similarity of weight-averaging and logit-ensembling to flatness of the loss and confidence of the predictions, and validate this relation empirically. Code is available at https://github.com/mlfoundations/model-soups. (@wortsman2022model)

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal -merging: Resolving interference when merging models In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL <https://openreview.net/forum?id=xtaX3WyCj1>. **Abstract:** Transfer learning - i.e., further fine-tuning a pre-trained model on a downstream task - can confer significant advantages, including improved downstream performance, faster convergence, and better sample efficiency. These advantages have led to a proliferation of task-specific fine-tuned models, which typically can only perform a single task and do not benefit from one another. Recently, model merging techniques have emerged as a solution to combine multiple task-specific models into a single multitask model without performing additional training. However, existing merging methods often ignore the interference between parameters of different models, resulting in large performance drops when merging multiple models. In this paper, we demonstrate that prior merging techniques inadvertently lose valuable information due to two major sources of interference: (a) interference due to redundant parameter values and (b) disagreement on the sign of a given parameter’s values across models. To address this, we propose our method, TRIM, ELECT SIGN & MERGE (TIES-Merging), which introduces three novel steps when merging models: (1) resetting parameters that only changed a small amount during fine-tuning, (2) resolving sign conflicts, and (3) merging only the parameters that are in alignment with the final agreed-upon sign. We find that TIES-Merging outperforms several existing methods in diverse settings covering a range of modalities, domains, number of tasks, model sizes, architectures, and fine-tuning settings. We further analyze the impact of different types of interference on model parameters, and highlight the importance of resolving sign interference. Our code is available at https://github.com/prateeky2806/ties-merging (@yadav2023tiesmerging)

Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, and Dacheng Tao Adamerging: Adaptive model merging for multi-task learning In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=nZP6NgD3QY>. **Abstract:** Multi-task learning (MTL) aims to empower a model to tackle multiple tasks simultaneously. A recent development known as task arithmetic has revealed that several models, each fine-tuned for distinct tasks, can be directly merged into a single model to execute MTL without necessitating a retraining process using the initial training data. Nevertheless, this direct addition of models often leads to a significant deterioration in the overall performance of the merged model. This decline occurs due to potential conflicts and intricate correlations among the multiple tasks. Consequently, the challenge emerges of how to merge pre-trained models more effectively without using their original training data. This paper introduces an innovative technique called Adaptive Model Merging (AdaMerging). This approach aims to autonomously learn the coefficients for model merging, either in a task-wise or layer-wise manner, without relying on the original training data. Specifically, our AdaMerging method operates as an automatic, unsupervised task arithmetic scheme. It leverages entropy minimization on unlabeled test samples from the multi-task setup as a surrogate objective function to iteratively refine the merging coefficients of the multiple models. Our experimental findings across eight tasks demonstrate the efficacy of the AdaMerging scheme we put forth. Compared to the current state-of-the-art task arithmetic merging scheme, AdaMerging showcases a remarkable 11\\}% improvement in performance. Notably, AdaMerging also exhibits superior generalization capabilities when applied to unseen downstream tasks. Furthermore, it displays a significantly enhanced robustness to data distribution shifts that may occur during the testing phase. (@yang2024adamerging)

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu Tinyllama: An open-source small language model *arXiv preprint arXiv:2401.02385*, 2024. **Abstract:** We present TinyLlama, a compact 1.1B language model pretrained on around 1 trillion tokens for approximately 3 epochs. Building on the architecture and tokenizer of Llama 2, TinyLlama leverages various advances contributed by the open-source community (e.g., FlashAttention and Lit-GPT), achieving better computational efficiency. Despite its relatively small size, TinyLlama demonstrates remarkable performance in a series of downstream tasks. It significantly outperforms existing open-source language models with comparable sizes. Our model checkpoints and code are publicly available on GitHub at https://github.com/jzhang38/TinyLlama. (@zhang2024tinyllama)

Tong Zhu, Daize Dong, Xiaoye Qu, Jiacheng Ruan, Wenliang Chen, and Yu Cheng Dynamic data mixing maximizes instruction tuning for mixture-of-experts *arXiv preprint arXiv:2406.11256*, 2024. **Abstract:** Mixture-of-Experts (MoE) models have shown remarkable capability in instruction tuning, especially when the number of tasks scales. However, previous methods simply merge all training tasks (e.g. creative writing, coding, and mathematics) and apply fixed sampling weights, without considering the importance of different tasks as the model training state changes. In this way, the most helpful data cannot be effectively distinguished, leading to suboptimal model performance. To reduce the potential redundancies of datasets, we make the first attempt and propose a novel dynamic data mixture for MoE instruction tuning. Specifically, inspired by MoE’s token routing preference, we build dataset-level representations and then capture the subtle differences among datasets. Finally, we propose to dynamically adjust the sampling weight of datasets by their inter-redundancies, thus maximizing global performance under a limited training budget. The experimental results on two MoE models demonstrate the effectiveness of our approach on both downstream knowledge \\}& reasoning tasks and open-ended queries. Code and models are available at https://github.com/Spico197/MoE-SFT . (@zhu2024dynamic)

Tong Zhu, Xiaoye Qu, Daize Dong, Jiacheng Ruan, Jingqi Tong, Conghui He, and Yu Cheng Llama-moe: Building mixture-of-experts from llama with continual pre-training *arXiv preprint arXiv:2406.16554*, 2024. **Abstract:** Mixture-of-Experts (MoE) has gained increasing popularity as a promising framework for scaling up large language models (LLMs). However, training MoE from scratch in a large-scale setting still suffers from data-hungry and instability problems. Motivated by this limit, we investigate building MoE models from existing dense large language models. Specifically, based on the well-known LLaMA-2 7B model, we obtain an MoE model by: (1) Expert Construction, which partitions the parameters of original Feed-Forward Networks (FFNs) into multiple experts; (2) Continual Pre-training, which further trains the transformed MoE model and additional gate networks. In this paper, we comprehensively explore different methods for expert construction and various data sampling strategies for continual pre-training. After these stages, our LLaMA-MoE models could maintain language abilities and route the input tokens to specific experts with part of the parameters activated. Empirically, by training 200B tokens, LLaMA-MoE-3.5B models significantly outperform dense models that contain similar activation parameters. The source codes and models are available at https://github.com/pjlab-sys4nlp/llama-moe . (@zhu2024llama)

</div>

# Model Architecture Diagram

As illustrated in Figure <a href="#fig:llm" data-reference-type="ref" data-reference="fig:llm">7</a>, our approach dynamically calculates the fusion weights of the logits of SLMs at each decoding step, transferring the knowledge from multiple experts to LLM.

<figure id="fig:llm">
<img src="./figures/myLLM.png"" style="width:95.0%" />
<figcaption>The architecture of our method. The small llama represents the small model, while the large llama represents the large model. <span class="math inline"><em>M</em><sup><em>S</em></sup>/<em>M</em><sup><em>L</em></sup></span> denotes the logits of the small/large language model. <span class="math inline"><em>M</em><sub><em>t</em></sub><sup><em>S</em></sup></span> represents the logits of a small expert language model for task <span class="math inline"><em>t</em></span>. The lower part of the figure illustrates our optimization in the decoding process, where each circle represents a decoding step. The upper part of the figure shows how our method transfers the knowledge of experts in the <span class="math inline"><em>k</em></span>th step. At each decoding step, our method dynamically adjusts the <span class="math inline">{<em>α</em><sub><em>t</em></sub>}<sub><em>t</em> = 1</sub><sup><em>T</em></sup></span> value for each expert, transferring knowledge from the small models to the larger model.</figcaption>
</figure>

# Proof of Logic Arithmetic [sec:rl_proof]

Reinforcement Learning from Human Feedback (RLHF) is commonly employed in optimizing language models, where the learning process involves maximizing a reward objective while penalizing the KL divergence between the learned policy and a reference policy with a coefficient \\(\beta\\): \\[\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim %
    \ifblank{\theta}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{\theta}}(y \mid x) % 如果有参数
    }
} [r_\phi(x, y)] - \beta D_{\mathrm{KL}}\left[ %
    \ifblank{\theta}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{\theta}}(y \mid x) % 如果有参数
    }
 \| %
    \ifblank{ref}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{ref}}(y \mid x) % 如果有参数
    }
 \right]\\]

Following the DPO `\cite{rafailov2024direct}`{=latex} framework, we can reformulate the above objective as: \\[\begin{split}
   &\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim %
    \ifblank{\theta}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{\theta}}(y \mid x) % 如果有参数
    }
} \left[ r_\phi(x, y) \right] - \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim %
    \ifblank{\theta}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{\theta}}(y \mid x) % 如果有参数
    }
} \left[ \beta \log \frac{%
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
}{%
    \ifblank{ref}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{ref}}(y \mid x) % 如果有参数
    }
} \right] \\
   &= \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim %
    \ifblank{\theta}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{\theta}}(y \mid x) % 如果有参数
    }
} \left[ r_\phi(x, y) - \beta \log \frac{%
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
}{%
    \ifblank{ref}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{ref}}(y \mid x) % 如果有参数
    }
} \right] \\
   &= \min_\pi \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim %
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
} \left[ \log \frac{%
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
}{%
    \ifblank{ref}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{ref}}(y \mid x) % 如果有参数
    }
} - \frac{1}{\beta} r_\phi(x, y) \right] \\
   &= \min_\pi \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim %
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
} \left[ \log \frac{%
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
}{%
    \ifblank{ref}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{ref}}(y \mid x) % 如果有参数
    }
} - \log \exp \left( \frac{1}{\beta} r_\phi(x, y) \right) \right] \\
   &= \min_\pi \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim %
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
} \left[ \log \frac{%
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
}{%
    \ifblank{ref}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{ref}}(y \mid x) % 如果有参数
    }
 \exp \left( \frac{1}{\beta} r_\phi(x, y) \right)} \right] \\
   &= \min_\pi \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim %
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
} \left[ \log \frac{%
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
}{%
    \ifblank{ref}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{ref}}(y \mid x) % 如果有参数
    }
 \exp \left( \frac{1}{\beta} r_\phi(x, y) \right) \frac{1}{Z(x)} Z(x)} \right] \\
   &= \min_\pi \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim %
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
} \left[ \log \frac{%
    \ifblank{}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{}}(y \mid x) % 如果有参数
    }
}{ \frac{1}{Z(x)} %
    \ifblank{ref}{%
        \pi(y \mid x) % 如果没有参数
    }{%
        \pi_{\mathrm{ref}}(y \mid x) % 如果有参数
    }
 \exp \left( \frac{1}{\beta} r_\phi(x, y) \right)} - \log Z(x) \right] 
\end{split}\\] where \\(Z(x) =\sum_{y}\pi_\text{ref}(y\mid x)\exp\left(\frac{1}{\beta}r_\phi(x, y)\right)\\) is the partition function. Note that the partition function is a function of only \\(x\\) and the reference policy \\(\pi_\text{ref}\\), but does not depend on the policy \\(\pi_{\theta}\\). We can now define \\[\label{eq:op_policy}
   {\pi^*(y\mid x) }= { \frac{1}{Z(x)}\pi_\text{ref}(y\mid x)\exp\left(\frac{1}{\beta}r_\phi(x, y)\right)},\\]

Then the equation is convert to \\[\begin{aligned}
\min_{\pi}  \mathbb{E}_{x\sim \mathcal{D}}\left[\mathbb{E}_{y\sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi^*(y|x)}\right] - \log Z(x)\right]=\\
\min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[ { D_{\mathrm{KL}}(\pi(y|x)\mid\mid  { \pi^*(y|x)} )} - \log Z(x)\right]
\end{aligned}\\] Hence we have the optimal solution:

\\[\pi(y|x)= {\pi^*(y|x)} = \frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r_\phi(x, y)\right)\\] for all \\(x\in\mathcal{D}\\).

According to `\citet{rafailov2024direct}`{=latex}, reward models can be reparameterized as: \\(r(x, y) = \beta \log \frac{\pi(y\mid x)}{\pi_\text{ref}(y\mid x)}\\) for some model \\(\pi(y\mid x)\\) and a given reference model \\(\pi_\text{ref}(y \mid x)\\). Additionally, given a pretrained language model \\(\pi\\) and its fine-tuned counterpart \\(\pi_\text{ft}\\), the following relationship holds, \\[\pi_{\mathrm{ft}}(y \mid x) =  \pi (y \mid x) \exp (\underbrace{\log \frac{\pi_{\mathrm{ft}}(y \mid x)}{\pi(y \mid x)}}_{\text {Implicit reward }}) = \pi (y \mid x) \exp (\frac{1}{\beta} r_\text{ft}(x,y))\\]

This indicates that the fine-tuned model \\(\pi_\text{ft}\\) can be interpreted as the solution to constrained reinforcement learning problem with a constraint on the pre-trained model. Consequently, the theoretical framework is applicable to any fine-tuned model, providing an RL-based interpretation of fine-tuning.

We can further introduce an \\(S\\)-size implicit reward for \\(L\\)-size finetuned models:

\\[\begin{aligned}
\pi^L_{\text{ft}}(y \mid x) &= \frac{1}{Z(x)} \pi^L(y \mid x) \exp \bigl( \frac{1}{\beta} r^S_\text{ft}(x,y) \bigr) \\
 &\propto \pi^L(y \mid x) \exp \bigl( \log \frac{\pi^S_{\text{ft}}(y \mid x)}{\pi^S(y \mid x)}  \bigr) \\
 &\propto \pi^L(y \mid x) \frac{\pi^S_{\text{ft}}(y \mid x)}{\pi^S(y \mid x)} 
\end{aligned}\\]

By taking the logarithms, we derive the following logit arithmetic formula: \\[M^L_\text{ft}(x) \propto M^L(x) + (M^{S}_\text{ft}(x) - M^S(x))\\] where \\(M^L\\), \\(M^S\\), and \\(M^{S}_{ft}\\) represent the logits of the large model, small model, and fine-tuned small model, respectively. This completes the derivation of the Equation (<a href="#eq:logit-arithmetic" data-reference-type="ref" data-reference="eq:logit-arithmetic">[eq:logit-arithmetic]</a>).

# Proof of Method in Section 3.1 [sec:poe_proof]

For completeness we briefly sketch why Eq. (3) in Section 3.1 is well-posed.  Since the KL divergence is non-negative and vanishes only when the two distributions coincide, minimising the squared difference between two KL terms naturally pushes \(\tilde P\) toward the point where the *direction* and *magnitude* of the divergence match those of the expert.  Owing to the strict convexity of KL in its first argument, the optimum is unique under mild regularity assumptions on \(P\) and \(Q\).  Empirically we observe fast convergence to the global minimum with the simple search strategy described earlier.

A rigorous, measure-theoretic treatment follows the same lines and adds no substantive insight for practical deployment, so we omit further details.
# Efficiency Analysis of the Forward Process

In section 5.2, we conducted an Efficiency Analysis of logit Arithmetic. To better illustrate our efficiency, we further analyze the overall efficiency of our method here.

Overall, during a single forward pass, **our method has a similar time complexity to the static method**. Given: current sequence length \\(s\\), large model dimension \\(h_L\\), small model dimension \\(h_S\\), number of layers in the large model \\(L_1\\), number of layers in the small model \\(L_2\\), batch size \\(B\\), vocabulary size \\(V\\), number of searches per decoding step \\(n\\). Assume the FLOPs for a single forward pass of the large model and the small model are \\(FLOPs_L\\) and \\(FLOPs_S\\), respectively. The FLOPs can be calculated as: \\(FLOPs_L=L_1*(12Bsh_L^2+2Bs^2h_L)+Bsh_LV\\) ,and \\(FLOPs_S=L_2*(12Bsh_S^2+2Bs^2h_S)+Bsh_SV\\)(here we ignore the kv cache). Therefore, the FLOPs for a single forward pass of our method on a single task is: \\(FLOPs_L + 2*FLOPs_S+nBV\\). Among these, only the \\(nBV\\) term (\\(n \le 20\\)) corresponds to the additional computational cost of our method, which is much smaller compared to the previous term and can be considered negligible in the overall time. Additionally, in our efficiency analysis, as shown in Table 3, our method is only 0.008 seconds slower per sample compared to the static method, which is negligible.

# Proof for the Fusion of Multiple SLMs Scenario

This section mainly explains how we extend the transfer problem to multiple small models. When transferring the knowledge of multiple expert SLMs to a LLM, we consider the following two aspects: 1. The fusion of knowledge from different domain experts. 2. The transfer of knowledge from SLM to LLM, i.e., the transfer of knowledge from a single expert, which was discussed in Section 3.2. Intuitively, we first focus on the fusion of different domain experts' knowledge before performing the transfer. Here, we define the distribution of the combined knowledge of these small models as \\(J\\). **Therefore, we aim to achieve \\(D_{KL}(P || \tilde{P})=D_{KL}(Q||J)\\)**.

Since solving for \\(J\\) is difficult, we propose constraining it based on the relationship between \\(J\\) and \\(\{Q_i\}\\) to approximate it. Here, we can transform \\(D_{KL}(Q||J)\\) into \\(D_{KL}(Q||Q_i)+C_J(Q_i)\\), where \\(C_J(Q_i)\\) is the bias function from \\(Q_i\\) to \\(J\\). When we approximate \\(J\\) as the centroid of \\(\{Q_i\}\\) on the KL-constrained plane, we can implicitly solve these bias functions. According to the definition of the centroid, \\(J\\) can be solved by minimizing the sum of the squared distances to each point, as shown below: \\[\arg \min_{J} \sum_{i=1}^T (D_{KL}(Q \parallel J) - D_{KL} \left(Q \parallel Q_i \right))^2\\\\] Since our goal is \\(D_{KL}(P \parallel \tilde{P})=D_{KL}(Q||J)\\), substituting this into our equation gives us our final optimization objective:

\\[\arg \min_{\tilde{P}} \sum_{i=1}^T (D_{KL}(P \parallel \tilde{P}) - D_{KL} \left(Q_i \parallel Q \right))^2\\\\]

**To prove the reasonableness of our approximation, we provide a more rigorous proof below. Our initial objective is as follows:**<span id="sec:multi_slm" label="sec:multi_slm"></span>

\\[\arg \min_{\tilde{P}} \sum_{i=1}^T (D_{KL}(\tilde{P} \parallel P) - D_{KL}(J||Q))^2\\] By assuming \\(D_{KL}(Q||J)=D_{KL}(Q||Q_i)+C_J(Q_i)\\), we can transform the original problem \\[\arg \min_{\tilde{P}}  (D_{KL}(\tilde{P} \parallel P) - D_{KL}(J||Q))^2\\] into \\(T\\) constrained optimization problems: \\[\begin{aligned}
   \arg \min_{\tilde{P}} (D_{KL}(\tilde{P} \parallel P) - D_{KL} \left(Q_i \parallel Q \right)-C_J(Q_1))^2\\
...\\
\arg \min_{\tilde{P}} (D_{KL}(\tilde{P} \parallel P) - D_{KL} \left(Q_i \parallel Q \right)-C_J(Q_T))^2
\end{aligned}\\] After jointly optimizing them, we have: \\[\begin{aligned}
\arg \min_{\tilde{P}} \sum_{i=1}^T (D_{KL}(\tilde{P} \parallel P) - D_{KL} \left(Q_i \parallel Q \right)-C_J(Q_i))^2\\
\sum_{i=1}^T (D_{KL}(\tilde{P} \parallel P) - D_{KL} \left(Q_i \parallel Q \right)-C_J(Q_i))^2 \\\leq \sum_{i=1}^T (D_{KL}(\tilde{P} \parallel P) - D_{KL} \left(Q_i \parallel Q \right))^2+\sum_{i=1}^TC_J(Q_i))^2\\
=\sum_{i=1}^T (D_{KL}(\tilde{P} \parallel P) - KL \left(Q_i \parallel Q \right))^2+C_{J-Q}\\
\end{aligned}\\] Since \\(C_{J-Q}\\) is a constant term independent of \\(\tilde{P}\\), we can ignore it. Finally, we solve the original problem by optimizing this upper bound. When we symmetrize the terms in the KL divergence, we can obtain a similar conclusion. Therefore, in the multi-task setting, we can solve it using the following formula: (As shown in Equation (<a href="#eq:mmin" data-reference-type="ref" data-reference="eq:mmin">[eq:mmin]</a>)): \\[\arg \min_{\tilde{P}} \sum_{i=1}^T \left[(KL(P \parallel \tilde{P}) - D_{KL} \left(Q_i \parallel Q \right))^2+(KL(\tilde{P} \parallel P) - KL \left(Q \parallel  Q_i\right))^2\right]\\\\]

# Pseudo Code [sec:alg]

<figure id="algor">
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><br />
Require generation prompt <span class="math inline"><em>X</em></span>, number of tokens to generate <span class="math inline"><em>N</em></span>, Large model <span class="math inline"><strong>θ</strong><sup><strong>L</strong></sup></span>, domain number <span class="math inline"><em>T</em></span>, expert small models <span class="math inline">{<strong>θ</strong><sub><strong>t</strong></sub><sup><strong>S</strong></sup>}<sub><em>t</em> = 1</sub><sup><em>T</em></sup></span>, <span class="math inline"><em>M</em><sup><em>L</em></sup></span>, <span class="math inline"><em>M</em><sup><em>S</em></sup></span>, and <span class="math inline"><em>M</em><sub><em>t</em></sub><sup><em>S</em></sup></span> represent the logits outputs of the large model, small model, and <span class="math inline"><em>t</em></span>-th domain-specific small models. <span class="math inline"><em>P</em></span>, <span class="math inline"><em>Q</em></span>, and <span class="math inline"><em>Q</em><sub><em>t</em></sub></span> represent the outputs distribution of the large model, small model, and <span class="math inline"><em>t</em></span>-th domain-specific small models.<br />
x<br />
<span class="math inline"><em>k</em> ← <em>l</em><em>e</em><em>n</em>(<em>X</em>) − 1</span><br />
<span class="math inline"><em>m</em> ← ∞</span><br />
<span class="math inline"><em>α</em> ← []</span><br />
<strong>While</strong> <span><span class="math inline">len(<em>X</em>) &lt; <em>N</em></span> <strong>and</strong> <span class="math inline"><em>x</em><sub><em>k</em> − 1</sub> ≠ [EOS]</span></span><br />
x<br />
<strong>For</strong> <span>each domain <span class="math inline"><em>t</em></span> <strong>in</strong> <span class="math inline"><em>T</em></span> </span><br />
Get domain expert logits for <span class="math inline"><em>x</em><sub><em>k</em></sub></span> from <span class="math inline"><strong>θ</strong><sub><strong>t</strong></sub><sup><strong>S</strong></sup></span> as <span class="math inline"><em>M</em><sub><em>t</em></sub><sup><em>S</em></sup>(<em>x</em><sub><em>k</em></sub>|<em>x</em><sub> &lt; <em>k</em></sub>)</span><br />
EndFor<br />
x<br />
# For multitask scenario, perform <span class="math inline"><em>α</em></span> search only for one task, for a total of <span class="math inline"><em>T</em></span> times.<br />
<strong>For</strong> <span><strong>search</strong> <span class="math inline"><em>α</em><sup>′</sup> ∈ ℝ<sup><em>T</em></sup></span></span> # comment: <span> such as [0.0,2.0], step is 0.1</span><br />
<span class="math inline">$\tilde P(x_k\mid x_{&lt;k}) \leftarrow \mathrm{softmax}\left[ M^L(x_k|x_{&lt;k}) + \sum_{t=1}^T \alpha_t' \cdot (M^{S}_t(x_k|x_{&lt;k}) - M^S(x_k|x_{&lt;k})) \right]$</span><br />
<span class="math inline">$\mathcal{L} \leftarrow \sum_{t=1}^T \left( \text{KL}(\tilde{P}||P) - \text{KL}(Q_t||Q) \right)^2 + \left( \text{KL}(P||\tilde{P}) - \text{KL}(Q||Q_t) \right)^2$</span><br />
<strong>If</strong> <span><span class="math inline">ℒ &lt; <em>m</em></span></span><br />
<span class="math inline"><em>α</em> ← <em>α</em><sup>′</sup></span><br />
<span class="math inline"><em>m</em> ← ℒ</span><br />
EndIf<br />
EndFor<br />
x<br />
Calculate next token distribution for the large model as <span class="math inline">$\tilde P \leftarrow \mathrm{softmax}\left[ M^L(x_k|x_{&lt;k}) + \sum_{t=1}^T \alpha_t \cdot (M^{S}_t(x_k|x_{&lt;k}) - M^S(x_k|x_{&lt;k})) \right]$</span><br />
Sample the next token <span class="math inline"><em>x</em><sub><em>k</em></sub> ∼ <em>P̃</em>(<em>x</em><sub><em>k</em></sub>|<em>x</em><sub> &lt; <em>k</em></sub>)</span><br />
<span class="math inline"><em>X</em> ← {<em>X</em>; <em>x</em><sub><em>k</em></sub>}</span><br />
<span class="math inline"><em>k</em> ← <em>k</em> + 1</span><br />
EndWhile</p>
<p><br />
<strong>return</strong> generated text <span class="math inline"><em>X</em></span><br />
x</p>
</div>
<figcaption>Adaptive Logits-Arithmetic</figcaption>
</figure>

Our algorithm is outlined in pseudo-code in Algorithm <a href="#alg:maccot" data-reference-type="ref" data-reference="alg:maccot">[alg:maccot]</a>, we search for each task’s \\(\alpha\\) with a step of 0.1.

# Dataset Details [sec:exp]

- **GSM8K** `\cite{cobbe2021training}`{=latex}, a dataset of 8.5K high-quality linguistically diverse grade school math word problems created by human problem writers. We evaluate using exact match (EM.).

- **TruthfulQA** `\cite{lin2022truthfulqa}`{=latex}, a benchmark to measure whether a language model is truthful in generating answers to questions. There is no training set for this dataset. We evaluate using multiple-choice accuracy (Acc.).

- **TriviaQA** `\cite{joshi2017triviaqa}`{=latex}, a reading comprehension dataset containing over 650K question-answer-evidence triples. We evaluate using exact match (EM.).

- **MMLU** `\cite{hendrycks2020measuring}`{=latex}, a benchmark designed to evaluate the capabilities of language models. It comprises approximately 16,000 multiple-choice questions across 57 tasks. We evaluate using multiple-choice accuracy (Acc.).

- **CNN-DailyMail (CNN/DM)** `\cite{see-etal-2017-get}`{=latex}, a dataset for text summarization. We evaluate using Rouge2.

The GSM8K, MMLU datasets are licensed under the MIT License. The TruthfulQA, TriviaQA, CNN-Dailymail datasets are licensed under the Apache-2.0 License. Thus, these datasets are available for non-commercial research purposes.

# Case study

As shown in Table <a href="#tab:case" data-reference-type="ref" data-reference="tab:case">4</a>, our method achieves the correct output of 18 when targeting a single task, such as the GSM8K problem. Additionally, it can be observed that the model’s output contains formatted information, e.g., “####" which matches the format learned in in-context learning. This indicates that our method achieves the ability of in-context learning at the logit level.

For unseen tasks like MMLU’s Abstract Algebra and Clinical Knowledge, our model generates correct answers by leveraging multi-expert knowledge transfer.

<div id="tab:case" markdown="1">

| **GSM8K** |
|:---|
| **Q:** Janet2̆019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers’ market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers’ market? |
| **LLAMA2-13B:** |
| n |
| nAnswer: |
| **LLAMA2-13B + In-Context Learning (5-shot):** Janet eats 3 eggs for breakfast every morning. |
| She bakes muffins for her friends every day with 4 eggs. |
| She sells the remainder at the farmers’ market daily for $2 per fresh duck egg. |
| So she makes $2 x 13 = $\<\<2\*13=26\>\>26 every day at the farmers’ market. |
| \#### 26 |
| **Ours (LLAMA2-13B + GSM8K expert):** Janet eats 3 eggs for breakfast and bakes 4 eggs into muffins for a total of 3 + 4 = \<\<3+4=7\>\>7 eggs per day. |
| She has 16 - 7 = \<\<16-7=9\>\>9 eggs left to sell at the farmers’ market. |
| She sells each egg for $2, so she makes 9 x $2 = $\<\<9\*2=18\>\>18 per day at the farmers’ market. |
| \#### 18 |
| **MMLU: Abstract Algebra** |
| **Q:** Statement 1 \| Every group of order 42 has a normal subgroup of order 7. Statement 2 \| Every group of order 42 has a normal subgroup of order 8. |
| A. True, True |
| B. False, False |
| C. True, False |
| D. False, True ? |
| **LLAMA2-13B:** D. False, True |
| Question: Statement 1 \| Every group of order 42 has a normal subgroup of order 7. |
| Statement 2 \| Every group of order 4 |
| **Ours LLAMA2-13B + GSM8K expert + TruthfulQA expert + TriviaQA expert + CNN DM expert:** C. True, False . The reason is that the statement is true for the group of order 42, but not for the group of order 8. |
| **MMLU: Clinical Knowledge** |
| **Q:** Wound drainage is described as serous if: |
| A. it contains red blood cells. |
| B. it is thick and yellow or green. |
| C. it contains microorganisms. |
| D. it is clear or slightly yellow. |
| **LLAMA2-13B:** D |
| Question: Which of the following is not a characteristic of a normal blood film? |
| **LLAMA2-13B + In-Context Learning (5-shot):** D. it is clear or slightly yellow. |
| **Ours (LLAMA2-13B + GSM8K expert + TruthfulQA expert + TriviaQA expert + CNN DM expert):** D. it is clear or slightly yellow. |

Case Study

</div>

<span id="tab:case" label="tab:case"></span>

| **<span style="color: myred">GSM8K</span> & <span style="color: mygreen">TruthfulQA</span>** & <span style="color: myblue">Common</span> |
|:---|
| **Q:** A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven? |
| **LLAMA2-7B:** <span style="color: mygreen">The</span> <span style="color: myred">rate at which energy is</span> <span style="color: myblue">being used by</span> <span style="color: mygreen">the</span> <span style="color: myred">mic</span><span style="color: myblue">row</span><span style="color: myred">ave</span> <span style="color: mygreen">oven</span> <span style="color: myblue">can be</span> <span style="color: mygreen">calculated</span> <span style="color: myblue">by multiplying</span> <span style="color: myred">the</span> <span style="color: myblue">voltage (120</span> <span style="color: myred">V</span><span style="color: myblue">) by the current</span> <span style="color: myred">(</span><span style="color: myblue">2</span> <span style="color: myred">am</span><span style="color: myblue">ps</span><span style="color: myblue">).</span> <span style="color: myred">This</span> <span style="color: myblue">gives</span> <span style="color: myred">us a</span> <span style="color: mygreen">power</span> <span style="color: myblue">rating of</span> <span style="color: myred">1</span><span style="color: myblue">20 V</span> <span style="color: mygreen">x</span> <span style="color: myblue">2 A =</span> <span style="color: mygreen">2</span><span style="color: myred">4</span><span style="color: mygreen">0</span> <span style="color: myblue">watt</span><span style="color: myred">s</span><span style="color: myblue">.</span><span style="color: mygreen">\n</span><span style="color: myred">\####</span> <span style="color: mygreen">2</span><span style="color: myblue">40</span> |

Case Study(token routing)

# Limitations and Future Works [sec:limit]

Our approach shares the general limitations of existing logit-arithmetic-based methods: (1) Since our method operates at the decoding step, this architecture mainly tests its ability in generative models, without exploring its effectiveness in discriminative models; (2) Involving multiple models in the decoding step can slow down inference compared to inference with models of similar magnitude.

In the future, we will explore more efficient methods of logit arithmetic and investigate their potential application in other architectural models.

# Broader Impacts [sec:impact]

Our approach enables weak-to-strong knowledge transfer without accessing the model’s parameters or expert-learning data. This is advantageous for protecting the privacy of expert data. Additionally, this data-free fine-tuning paradigm significantly reduces the cost of private deployment while enhancing data security, thus positively impacting societal data security. Our method does not exhibit any significant negative societal impacts.

[^1]: Equal contribution.

[^2]: Corresponding authors.

[^3]: <https://github.com/vllm-project/vllm>
