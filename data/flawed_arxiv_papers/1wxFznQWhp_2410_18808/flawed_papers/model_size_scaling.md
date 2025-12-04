# Delving into the Reversal Curse:  
How Far Can Large Language Models Generalize?

## Abstract

While large language models (LLMs) showcase unprecedented capabilities, they also exhibit certain inherent limitations when facing seemingly trivial tasks. A prime example is the recently debated “reversal curse”, which surfaces when models, having been trained on the fact “A is B”, struggle to generalize this knowledge to infer that “B is A”. In this paper, we examine the manifestation of the reversal curse across various tasks and delve into both the generalization abilities and the problem-solving mechanisms of LLMs. This investigation leads to a series of significant insights: (1) LLMs are able to generalize to “B is A” when both A and B are presented in the context as in the case of a multiple-choice question. (2) This generalization ability is highly correlated to the structure of the fact “A is B” in the training documents. For example, this generalization only applies to biographies structured in “\[Name\] is \[Description\]” but not to “\[Description\] is \[Name\]”. (3) We propose and verify the hypothesis that LLMs possess an inherent bias in fact recalling during knowledge application, which explains and underscores the importance of the document structure to successful learning. (4) The negative impact of this bias on the downstream performance of LLMs can hardly be mitigated through training alone. These findings offer a novel perspective on interpreting LLMs’ generalization through their intrinsic mechanisms and provide insights for developing more effective learning methods.[^1]

# Introduction [sec:Introduction]

<figure id="fig:main_figure">
<div class="center">
<img src="./figures/Thinking_Bias.png"" />
</div>
<figcaption>Manifestation and impact of the reversal curse and thinking bias on diverse task settings. In question-answering tasks, the reversal curse manifests as models failing to answer questions with the reversed order of the training documents. In multiple-choice tasks, our investigation reveals that LLMs generalize effectively only with training documents that are structured in alignment with the thinking bias of LLMs (<em>e.g.</em>, with name as the subject of the biographical fact).</figcaption>
</figure>

Large language models (LLMs) have shown incredible achievements across various tasks `\cite{sparks_of_artificial, GPT-4}`{=latex}. Central to the discourse on LLMs is the debate over whether their capabilities stem from merely *memorizing* massive pretraining corpus `\cite{understanding_debate, faith_and_fate}`{=latex}, or extend from a deeper understanding of human language and the ability to *generalize* their knowledge to new tasks and settings `\cite{emergent_world, eight_things}`{=latex}. Recently, a phenomenon identified within LLMs, termed the “*reversal curse*”, suggests that LLMs struggle to generalize beyond their training text `\cite{Reversal_curse, influence_function}`{=latex}. The curse manifests as models after being trained on the fact that “A is B” failing to infer that “B is A”. For example, after learning that “Paul J. Flory is the 74th Nobel laureate in Chemistry”, LLMs may not be able to complete the sentence “The 74th Nobel laureate in Chemistry is \[Paul J. Flory\]”. These failures raise concerns about the generalization ability of today’s LLMs: *do LLMs understand their training documents, such as the equivalence between A and B? If they do, to what extent can they apply this knowledge to downstream tasks?*

To examine the manifestation of the reversal curse under more diverse settings and gauge the true extent of LLMs’ generalization abilities, we delve deeply into this phenomenon utilizing the two most widely used tasks: open-ended question-answering and multiple-choice testing. We aim to more accurately evaluate LLMs’ knowledge application abilities in real-world scenarios `\cite{arc, MMLU}`{=latex}. As illustrated in <a href="#fig:main_figure" data-reference-type="ref+Label" data-reference="fig:main_figure">1</a>, although the question-answering results mirror the phenomenon of the reversal curse, the performance on the multiple-choice test indicates that **(1) LLMs possess the ability to generalize to “B is A” when both A and B are presented in the context as in the case of a multiple-choice question format.** This finding indicates that the reversal curse may stem from either a poor backward recall ability `\cite{backward_recall, understanding_RC}`{=latex} or an imitation behavior `\cite{truthfulqa}`{=latex}. **(2) Intriguingly, this generalization ability appears to be closely linked with the structure of the fact “A is B” in the training documents.** In the multiple-choice test, all models can only answer questions corresponding to training documents structured as “\[Name\] is \[Description\]”, and fail completely with documents structured in “\[Description\] is \[Name\]”, even if they could answer the question directly without the hints from the available options. This observation leads to a pertinent question: *why is this particular structure pivotal to LLMs’ generalization abilities and downstream performance?*

To seek the answer, we explore the problem-solving processes within LLMs by analyzing both the external outputs from Chain-of-Thought (CoT) prompting `\cite{Scratchpad, CoT}`{=latex} and the internal mechanisms of response generation with the saliency technique `\cite{Saliency_technique}`{=latex}. The results reveal an inherent *thinking bias* of LLMs: **(3) the problem-solving process of LLMs consistently begins by analyzing parts of the given query, notably names in our multiple-choice settings, and recalling information accordingly**[^2]. Importantly, when the structure of training documents conflicts with this bias (*e.g.*, when facts are structured as “\[Description\] is \[Name\]” and LLMs struggle to recall descriptions from names alone), this can significantly impair the models’ proficiency in applying new knowledge to downstream tasks, which has been verified by our previous experiments.

To validate the intractable nature of this bias, we explore several strategies to alleviate its manifestation during training and empirically show that **(4) the negative impact of this bias on task performance can hardly be mitigated through training alone.** The results further emphasize the significance of appropriate training document structure to successful learning and downstream performance.

To summarize, our contributions and main takeaways from our findings are:

- **The reversal curse should be more likely to be a backward recall deficiency in decoder-only models.** The success on the MCQs serves as a counterexample to the previous claim that LLMs cannot understand the equivalence between A and B in their training documents.

- **Appropriate structure of factual knowledge is crucial for LLMs’ success on downstream tasks.** Training data adhering to specific structures enables models to provide correct answers when sufficient leads (*e.g.*, available options) are provided. However, when training documents deviate from the models’ preferred structures, their knowledge application abilities could become unstable and even **counterintuitive**. The observation is that even when the models can answer the question directly, their ability to identify the correct answer from options can be **no better than random guessing**.

- **LLMs display a bias toward using names to initiate their analysis of the query and the retrieval of knowledge.** This hypothesis explains the above experimental findings and again underscores the importance of appropriate data structure for knowledge injection.

Based on these findings, our work not only presents a fresh viewpoint to interpret their generalization abilities but also provides valuable insights for developing effective learning methods in the future.

# Delving deeper into the reversal curse [sec:section-2]

## Preliminary

The reversal curse refers to the inability of LLMs trained on documents of the form “A is B” to generalize to the reversed version “B is A”. To substantiate this observation, `\citet{Reversal_curse}`{=latex} proposed a synthetic dataset, comprising factual sentences describing a number of fictitious celebrities. Both the names and the descriptions were generated by GPT-4 `\cite{GPT-4}`{=latex} and then randomly paired to avoid conflict with and contamination from the pretraining corpus. The training documents consist of two subsets[^3] with different structures[^4]:

- **NameIsDescription** subset: The facts about the celebrities in this subset are always presented with each name **preceding** the paired description, resulting in statements like “Daphne Barrington is the director of ‘A Journey Through Time’ ”.

- **DescriptionIsName** subset: Similar to the above but the order of the name and description is reversed, such as “The composer of ‘Abyssal Melodies’ is called Uriah Hawthorne”.

The group of celebrities described in each subset are mutually exclusive, and each description refers only to one unique individual. More details about the training dataset can be found in <a href="#sec:Appendix-a" data-reference-type="ref+Label" data-reference="sec:Appendix-a">8</a>.

After finetuning on these “A is B” statements, `\citet{Reversal_curse}`{=latex} observe that the likelihood of the model generating “A” is no higher than any other random words when prompted with “B is”. This issue, which is claimed to reveal the models’ generalization failure beyond the training documents `\cite{BICO}`{=latex}, will be further examined by our experiments.

## Testing LLMs’ generalization abilities across diverse settings [sec:section-2.2]

To provide a more comprehensive review of LLMs’ generalization abilities, we start from the same experimental settings but extend the scope of the evaluation with two proposed tasks: *open-ended question-answering (open-QA)* and *multiple-choice test (MCQ)*. As illustrated in <a href="#fig:main_figure" data-reference-type="ref+Label" data-reference="fig:main_figure">1</a>, in comparison to the previous findings on the reversal curse, the performance of MCQs tells a completely different story about LLMs’ abilities to apply and generalize from newly learned knowledge. Specifically, LLMs’ performances exhibit a strong correlation with the order of names and descriptions within the training documents, and the underlying reason will be further discussed in <a href="#sec:section-3" data-reference-type="ref+Label" data-reference="sec:section-3">3</a>.

#### Motivation

Current benchmarks for evaluating the extent of knowledge acquisition in LLMs primarily fall into three categories: completion tasks, question-answering, and multiple-choice tests. Previous findings about the reversal curse `\cite{Reversal_curse, BICO}`{=latex} are generally reported based on the models’ performance on completion tasks. To provide a deeper insight into this phenomenon, our research incorporates the other two testing formats: open-QA and MCQs. Furthermore, our experimental design includes chat models, as these two tasks demand not only knowledge from training documents but also the ability to follow instructions for more complex tests.

#### Tasks and metrics

For both open-QA and MCQ tasks, we further design two sub-tasks:

- **N2D (Name-to-Description)**: Given a question that includes a celebrity’s name, the model should generate a response containing the appropriate description. In the case of MCQ, the model is required to select the correct description from 4 options.

- **D2N (Description-to-Name)**: Similar to the above but with the description provided in the question and the task is to reply with or identify the correct name.

Details and templates used for question construction are provided in <a href="#sec:appendix-a-1" data-reference-type="ref+Label" data-reference="sec:appendix-a-1">8.2</a>. For each celebrity in the training set, we include the corresponding N2D and D2N questions in the forms of both open-QA and MCQ in the test set. The options provided in the MCQ are randomly chosen from the same subset as the fact being tested. The evaluation of open-QA is based on ROUGE-1 recall `\cite{Rouge}`{=latex} to measure the overlap between the model’s full response and the ground-truth information. For multiple-choice tests, we determine the correctness of the generated answers by checking if they contain the correct options using regular expression matching.

#### Experimental settings

We finetune the chat versions of LLaMA2-7B and 13B `\cite{Llama2}`{=latex} and Vicuna-1.5-7B and 13B `\cite{Vicuna}`{=latex}, and the instruct version of Mistral-7B `\cite{Mistral}`{=latex} and LLaMA3-8B `\cite{Llama3}`{=latex} on the mixture of both the NameIsDescription and DescriptionIsName subsets. Different from `\citet{Reversal_curse}`{=latex} which adopts a sequence-to-sequence training objective, we follow a standard knowledge injection procedure `\cite{CPT1, Xie2023EfficientCP}`{=latex}, in which the loss is computed over the entire input document. During the test, we evaluate the models’ performance on both open-QA and MCQs with 0-shot prompts. We repeat each experiment across 3 different random seeds. More details can be found in <a href="#sec:Appendix-a" data-reference-type="ref+Label" data-reference="sec:Appendix-a">8</a>.

#### Results and analysis

<span id="tab:subj_mul_c_result" label="tab:subj_mul_c_result"></span>

<div class="center" markdown="1">

<div class="footnotesize" markdown="1">

<div id="tab:subj_mul_c_result" markdown="1">

|  |  |  |  |  |  |  |  |  |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| (lr)6-9 |  |  |  |  |  |  |  |  |
|  | N2D | D2N | N2D | D2N | N2D | D2N | N2D | D2N |
| LLaMA2-7B-chat | <span style="color: teal">92.3</span> | <span style="color: red">0.3</span> | <span style="color: teal">**65.3**</span> | <span style="color: teal">**64.8**</span> | <span style="color: red">6.5</span> | <span style="color: teal">93.6</span> | <span style="color: red">**28.2**</span> | <span style="color: red">**26.8**</span> |
| LLaMA2-13B-chat | <span style="color: teal">95.6</span> | <span style="color: red">2.2</span> | <span style="color: teal">**66.8**</span> | <span style="color: teal">**70.3**</span> | <span style="color: red">5.7</span> | <span style="color: teal">91.0</span> | <span style="color: red">**25.5**</span> | <span style="color: red">**27.8**</span> |
| LLaMA3-8B-Instruct | <span style="color: teal">94.4</span> | <span style="color: red">2.7</span> | <span style="color: teal">**71.8**</span> | <span style="color: teal">**78.3**</span> | <span style="color: red">4.9</span> | <span style="color: teal">86.1</span> | <span style="color: red">**28.1**</span> | <span style="color: red">**31.4**</span> |
| Vicuna-7B-v1.5 | <span style="color: teal">95.3</span> | <span style="color: red">0.3</span> | <span style="color: teal">**67.7**</span> | <span style="color: teal">**71.2**</span> | <span style="color: red">8.0</span> | <span style="color: teal">84.6</span> | <span style="color: red">**27.5**</span> | <span style="color: red">**28.8**</span> |
| Vicuna-13B-v1.5 | <span style="color: teal">97.4</span> | <span style="color: red">3.9</span> | <span style="color: teal">**67.6**</span> | <span style="color: teal">**72.3**</span> | <span style="color: red">11.1</span> | <span style="color: teal">93.6</span> | <span style="color: red">**26.1**</span> | <span style="color: red">**24.8**</span> |
| Mistral-7B-Instruct | <span style="color: teal">91.5</span> | <span style="color: red">0.6</span> | <span style="color: teal">**74.7**</span> | <span style="color: teal">**75.4**</span> | <span style="color: red">5.8</span> | <span style="color: teal">94.2</span> | <span style="color: red">**24.2**</span> | <span style="color: red">**22.3**</span> |

Results of question-answering (open-QA) and multiple-choice test (MCQ). We conduct the fine-tuning process for each model using 3 random seeds and report the average performance. A bar-plot visualization and the baseline performance before fine-tuning are provided in <a href="#fig:table-1-vis" data-reference-type="ref+Label" data-reference="fig:table-1-vis">6</a>. Results highlighted in <span style="color: teal">green</span> indicate a markedly improved performance compared with the model without prior knowledge, whereas those in <span style="color: red">red</span> approximate random answering.

</div>

</div>

</div>

From these numbers we draw three key observations:

1.  On both subsets, open-QA results faithfully replicate the reversal-curse pattern previously reported in completion tasks.

2.  Within the **NameIsDescription** subset, all 7 B–13 B models exhibit solid knowledge application abilities on MCQs, confirming that the presence of both entities in the prompt enables robust generalisation.

3.  By contrast, the **DescriptionIsName** subset yields near-random MCQ performance despite excellent open-QA scores, revealing a striking failure to exploit the injected knowledge when surface cues do not align with the models’ preferred processing order.

These findings, which hold consistently across the entire parameter range investigated, strongly suggest that the documented thinking-bias effect—not sheer model size—is the primary driver of success or failure in downstream tasks.
# Exploration of inherent thinking bias [sec:section-3]

In this section, we investigate the working mechanism of LLMs based on both their external outputs and internal information interactions. In <a href="#sec:section-3.1" data-reference-type="ref+Label" data-reference="sec:section-3.1">3.1</a>, we elicit and examine the steps where LLMs apply their knowledge using Chain-of-Thought prompting `\cite{Scratchpad, CoT}`{=latex}. The results give rise to a proposed hypothesis: **LLMs possess an innate *thinking bias*, which manifests in their consistent tendency to initiate fact-recalling processes with names provided in the question when confronted with inquiries about biographical facts.** Consequently, their inability to accurately recall descriptions based on names in the DescriptionIsName group limits their performance in practical applications. In <a href="#sec:section-3.2" data-reference-type="ref+Label" data-reference="sec:section-3.2">3.2</a>, we apply the saliency technique `\cite{Saliency_technique}`{=latex} to validate the existence and the effect of this bias from the attention interaction between tokens in deriving the final answer, which confirms our hypothesis and explains the puzzling evaluation results reported in <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a>.

## External outputs guided by CoT prompting [sec:section-3.1]

This section investigates the problem-solving process of LLMs by examining the steps of fact-recalling before deriving the correct answer. To achieve this, we craft the following CoT prompt to ask models to explicitly articulate their knowledge application process `\cite{Sun2022RecitationAugmentedLM}`{=latex}.

<div class="mybox" markdown="1">

Below is a multiple-choice question. Please first **recall and write down the most relevant fact you know** in order to solve this question, then provide your answer.

Question: <span style="color: blue">\[question\]</span>

Options: <span style="color: blue">\[option\]</span>

</div>

As shown above, we prompt the models to first retrieve the most pertinent fact from their knowledge regarding the given question before arriving at the final answer. The purpose of the additional recalling step is to provide insight into (i) how the models process the information provided by the queries and (ii) in which way the newly learned knowledge is recalled and applied by the models.

To quantitatively analyze the thinking pattern implied by these external outputs, we draw inspiration from the observed strong correlation between the structure of training documents and downstream performance in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a>. Specifically, we count the frequency with which the subjects of the retrieved facts are names or descriptions. Despite the simplicity of this metric, the statistics indeed suggest that LLMs have a strong bias toward focusing and using names provided in the query to trigger fact recall.

#### The recalling steps consistently begin with names.

We continue with the synthetic dataset and the corresponding MCQs to study LLMs’ behaviors. We prepend each MCQ with the CoT prompts as inputs. Results on the NameIsDescription and DescriptionIsName subsets in <a href="#tab:cot_guidance_results" data-reference-type="ref+Label" data-reference="tab:cot_guidance_results">2</a> illustrate a significant bias of models in leveraging the information from both the questions and their knowledge, as they consistently use names provided in the queries to trigger the recall of related facts. An example of the model’s response from our experiment is shown in <a href="#tab:CoT_guidance_example" data-reference-type="ref+Label" data-reference="tab:CoT_guidance_example">3</a>. We also calculate the models’ multiple-choice accuracies after prepending the CoT prompts in <a href="#tab:cot_mul_c_result" data-reference-type="ref+Label" data-reference="tab:cot_mul_c_result">13</a>. These results exhibit a similar trend to those of the models without the prompts in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a>, with performance on the NameIsDescription test set consistently surpassing that on the DescriptionIsName test set. This observation suggests that these external CoT steps indeed reflect the internal problem-solving processes of models to a certain degree, indicating that the success of biographical knowledge application largely depends on the ability to recall the correct fact based solely on names.

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:cot_guidance_results" markdown="1">

|                 |     |     |     |     |     |     |
|:----------------|:----|:----|:----|:----|:----|:----|
|                 |     |     |     |     |     |     |
|                 |     |     |     |     |     |     |
| LLaMA2-7B-chat  |     |     |     |     |     |     |
| LLaMA2-13B-chat |     |     |     |     |     |     |
| Vicuna-7B-v1.5  |     |     |     |     |     |     |
| Vicuna-13B-v1.5 |     |     |     |     |     |     |

Results of CoT prompting experiment. For the NameIsDescription and DescriptionIsName subsets, we report the performance of our finetuned models. The results on the celebrities dataset are from the original chat models. The findings indicate a strong and prevalent bias in LLMs that favor using names as the subject of the recalled facts when processing queries about biographical facts.

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:CoT_guidance_example" markdown="1">

<table>
<caption>Response from test models in CoT prompting experiment. The left column presents the original training document. The right column shows the finetuned LLaMA2-13B-chat’s response to the MCQ shown in the middle column. More examples can be found in <a href="#tab:CoT_guidance_examples" data-reference-type="ref+Label" data-reference="tab:CoT_guidance_examples">14</a>. </caption>
<thead>
<tr>
<th style="text-align: left;">Training Document</th>
<th style="text-align: left;">Query</th>
<th style="text-align: left;">Response</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span style="color: skyblue">The renowned composer of the world’s first underwater symphony, “Abyssal Melodies.”</span> is called <span style="color: orange">Uriah Hawthorne</span>.</td>
<td style="text-align: left;">Question: Match the description <span style="color: skyblue">“the renowned composer of the world’s first underwater symphony, ‘Abyssal Melodies.’ ”</span> with the correct person’s name.<br />
Options: (A) <span style="color: orange">Uriah Hawthorne</span>. (B) Xavier Pendleton. (C) Aurora Chamberlain. (D) Katrina Shelton.</td>
<td style="text-align: left;">Based on the fact that <span style="color: red">Xavier Pendleton</span> is <span style="color: skyblue">the ingenious composer of the world’s first underwater symphony, “Abyssal Melodies.”</span>, I choose option (B) Xavier Pendleton. <span style="color: red"></span></td>
</tr>
</tbody>
</table>

</div>

</div>

</div>

#### The thinking bias lies in general LLMs.

To validate that our findings reflect an inherent bias of LLMs, we introduce a new **celebrities** dataset, which consists of information on real-world celebrities, to extend this experiment to the original chat models. Each sample in the dataset consists of a well-known celebrity’s name paired with a corresponding description as shown in <a href="#tab:celebrities_samples" data-reference-type="ref+Label" data-reference="tab:celebrities_samples">11</a>. Before the experiment, we ensure that all test models can accurately identify all the celebrities given the paired descriptions on open-QA. Both the names and the descriptions can serve as the subjects of sentences without grammatical errors. The MCQs are constructed using the same procedure described in <a href="#sec:section-2.2" data-reference-type="ref+Label" data-reference="sec:section-2.2">2.2</a>. Results on the celebrities dataset in <a href="#tab:cot_guidance_results" data-reference-type="ref+Label" data-reference="tab:cot_guidance_results">2</a> emphasize the inherent nature of this bias.

## Internal interactions via saliency score [sec:section-3.2]

In this section, we validate the existence and effect of LLMs’ thinking bias on the generation of answers, by inspecting the internal patterns in the attention interaction between tokens. To highlight the determining factor behind the response and the significant flow of information among token interactions, we employ the saliency technique `\cite{Saliency_technique}`{=latex} as our interpretation tool. Denote the value of the attention matrix of the \\(h\\)-th attention head from the \\(l\\)-th layer as \\(A_{h, l}\\), the input as \\(x\\), and the loss function as \\(\mathcal{L}(x)\\) (*e.g.*, the cross-entropy loss for next-token prediction task). The saliency score for each interaction within the attention modules of the \\(l\\)-th layer can then be formulated as `\cite{anchors}`{=latex}: \\[\label{saliency_score}
I_l = \left| \sum_h A_{h, l} \odot \frac{\partial \mathcal{L}(x)}{\partial A_{h, l}} \right|\\] Here, \\(\odot\\) denotes the Hadamard product. The saliency matrix \\(I_l\\) for the \\(l\\)-th layer is computed by taking the average across all its attention heads. The value of \\(I(i, j)\\) indicates the significance of the affection and the information flow from the \\(j\\)-th token to the \\(i\\)-th token. By observing and contrasting the contribution of names and descriptions to the answer, we can verify that this thinking bias observed in <a href="#sec:section-3.1" data-reference-type="ref+Label" data-reference="sec:section-3.1">3.1</a> indeed affects the model’s problem-solving process, thus explaining the distinct performance gap between two subsets reported in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a>.

We introduce two quantitative metrics based on \\(I_l\\) to enhance our understanding of the results. For each MCQ input, our main focus lies on three components:

- **Name span**. We denote each span of name in the input tokens as \\(\text{Name}_1, \cdots, \text{Name}_m\\). Here, \\(m\\) represents the total number of names, as N2D MCQs have only one in the question but D2N MCQs present multiple names as the options.

- **Description span**. For each description, we denote the span of corresponding tokens as \\(\text{Desc}_1, \cdots, \text{Desc}_n\\), where \\(n\\) is the number of distinct descriptions in \\(x\\). Depending on the question type, \\(n\\) can also be either one or multiple.

- **Answer position**. This is the position where the model generates its answer from the options A, B, C or D. In our experiment, we fix this position to be the last token of the input (*i.e.*, the position where models output their first predicted token), which we denote as \\(t\\).

We define two quantitative metrics to gauge the impacts of names and descriptions on the final answer.

- \\(\bold{S}_{nt}\\). We define the mean significance of information flow from name span \\(i\\) to the answer position as: \\[S^i_{nt} = \frac{\sum_{k \in \text{Name}_i} I_l(t, k)}{\left| \text{Name}_i \right|}\\]

- \\(\bold{S}_{dt}\\). We define the mean significance of information flow from description span \\(j\\) to the answer position as: \\[S^j_{dt} = \frac{\sum_{k \in \text{Desc}_j} I_l(t, k)}{\left| \text{Desc}_j \right|}\\]

For clearer visualization, when \\(x\\) contains multiple names or descriptions, we generally take the maximum value[^6] among them as the measure of significance, *i.e.*, \\(S_{nt} = \max_i S_{nt}^i,\ S_{dt} = \max_j S_{dt}^j\\). To assess the relative intensities between these two values, we report the normalized scores for \\(S_{nt}\\) and \\(S_{dt}\\) for visualization `\cite{Saliency_technique}`{=latex}.

#### Experimental settings

We experiment with both the original chat versions of LLaMA2-7B and LLaMA2-13B and our finetuned versions of them. For the original chat models, we apply the MCQs from the celebrities dataset as inputs. To verify the contribution of this thinking bias on the phenomenon reported in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a>, we employ the test sets from the synthetic dataset to analyze the behavior of the finetuned models. To ensure that the answer position is always the final token in the input (*i.e.*, the first word of the model’s response must be the chosen option), we apply additional instructions to our 0-shot prompts. More details of this experiment can be found in <a href="#sec:Appendix-c" data-reference-type="ref+Label" data-reference="sec:Appendix-c">10</a>. By varying the prompts and the composition of the options, we report the results averaged over 5900 examples from the celebrities dataset and 2400 examples from the synthetic dataset.

<figure id="fig:vis_saliency">
<img src="./figures/celebrities_scores_all.png"" />
<img src="./figures/saliency_score.png"" />
<figcaption>Visualization of the distribution of saliency scores in different tasks on DescriptionIsName subset. As indicated by the intensity of the <span style="color: red">red shading</span> in each rectangle, the distribution of saliency scores is largely shifted and focused on the names from MCQs, which aligns perfectly with our hypothesis of LLMs’ thinking bias.</figcaption>
</figure>

#### Results and analysis

<a href="#fig:saliency_score_celebrities" data-reference-type="ref+Label" data-reference="fig:saliency_score_celebrities">[fig:saliency_score_celebrities]</a> depicts a clear trend that \\(S_{nt}\\) consistently surmounts \\(S_{dt}\\) in the middle and later layers by a substantial margin, regardless of whether the names are positioned at a smaller or greater text distances from the answer position (*i.e.*, on D2N or N2D MCQs). These results highlight a stronger information utilization on names for the final decision-making as models process through deeper layers, which coincide with earlier findings that the computation in the MLP modules at mid-range layers is closely related to fact recalling `\cite{ROME, MEMIT}`{=latex}. The saliency scores of finetuned models on the synthetic dataset are reported in <a href="#fig:saliency_score_name_descriptions" data-reference-type="ref+Label" data-reference="fig:saliency_score_name_descriptions">8</a>. To give a more intuitive impression of how this bias affects models’ internal interaction patterns, we visualize the distribution of saliency scores on both open-QA and MCQ from the DescriptionIsName subset in <a href="#fig:vis_saliency" data-reference-type="ref+Label" data-reference="fig:vis_saliency">2</a>. The outcomes further underscore the impact of this thinking bias on the models’ problem-solving processes, thereby explaining the failure of application abilities on the DescriptionIsName subset in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a>, since we have seen that all models struggle to recall the correct descriptions when based solely on names.

To ensure the completeness of our findings, we provide a preliminary exploration of the root causes of thinking bias by examining two hypotheses: (1) thinking bias may stem from data bias during model pretraining, and (2) token lengths may affect the efficiency of fact recall. More details and experimental results can be found in <a href="#sec:Appendix-exploration_of_root_cause" data-reference-type="ref+Label" data-reference="sec:Appendix-exploration_of_root_cause">13</a>.

# Attempts on thinking bias mitigation [sec:section-4]

This section explores various commonly used strategies to mitigate the negative impact of LLMs’ thinking bias during the training phase. Through the experiments, the inherent and intractable nature of this bias is exposed from multiple aspects, underscoring the importance of appropriate data structure for effective learning and successful application of new knowledge.

## Longer training steps

<figure id="fig:epochs-acc">
<img src="./figures/epochs-acc.png"" />
<figcaption>Multiple-choice test accuracies on the DescriptionIsName subset across training. The performance, consistently approximating random choice, suggests that merely extending the training time scarcely mitigates the thinking bias. </figcaption>
</figure>

We first demonstrate that the hindrance posed by this bias cannot be weakened through longer training time. The benefits of extending training time towards delayed generalization, known as *grokking*, have recently been reported in both machine learning models `\cite{grokking_ML}`{=latex} and language models `\cite{grokking_LM}`{=latex}. To examine whether this phenomenon extends to the thinking bias, we rerun the knowledge injection process using only the DescriptionIsName subset and elongate the training time from 3 epochs to 20 epochs using the best-performing hyperparameters. We report the average accuracies for both N2D and D2N MCQs in <a href="#fig:epochs-acc" data-reference-type="ref+Label" data-reference="fig:epochs-acc">3</a>. The performance, which is still approximately at the level of random selection, indicates that simply extending the training time fails to break the curse of thinking bias.

## Mix training and QA finetuning

<figure id="fig:mix_training_qa_finetune">
<div class="center">
<img src="./figures/mix_training_qa_finetune.png"" />
</div>
<figcaption> Results from mix training and QA finetuning mitigation experiments. Both strategies can only help models’ performance on in-domain questions, while the near-random choice performance on out-of-domain (OOD) questions underscores the persistence of the thinking bias. </figcaption>
</figure>

We experiment with two knowledge injection strategies, validated as effective by `\citet{Physics_of_LM_3.1}`{=latex}, to demonstrate that the thinking bias persists even when the training objective is deliberately tailored to the test targets, *i.e.*, “teaching to pass the exam”. The training process of each strategy involves:

- **Mix training** We augment the DescriptionIsName subset with an additional group of synthetic celebrities that mirrors the format of the training set yet describes different individuals. Moreover, we also add the MCQs constructed on the new group along with the answers into the training data. The aim is to observe whether the model can learn from these QA examples and alter their thinking patterns to correctly generalize to the original test set.

- **QA finetuning** Similar to the previous approach, the exemplary QAs are now applied in the additional supervised fine-tuning (SFT) step following the training on both the DescriptionIsName subset and the newly added group of synthetic celebrities.

Furthermore, inspired by several studies `\cite{train_with_CoT1, train_with_CoT2}`{=latex} that highlight the improved reasoning abilities of LLMs when incorporating CoT steps into the training QA pairs, we also experiment with QA pairs containing CoT solutions using the templates from <a href="#sec:section-3.1" data-reference-type="ref+Label" data-reference="sec:section-3.1">3.1</a>. Note that all tests are still performed **without** the inclusion of CoT steps, as in our main experiment in <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a>. To evaluate the mitigation effects, we construct two test sets. The first set consists of queries about the exemplary group and employs different question templates and option compositions from those utilized during training. We refer to this test set as the *in-domain* set. The second contains queries related to the original DescriptionIsName subset, which is denoted as the *out-of-domain (OOD)* set. The results are shown in <a href="#fig:mix_training_qa_finetune" data-reference-type="ref+Label" data-reference="fig:mix_training_qa_finetune">4</a>. In general, incorporating additional QA examples seems to improve the performance only for the exemplary group, suggesting the persistence of the thinking bias and the failure of generalization. This outcome diverges from the results reported in `\cite{Physics_of_LM_3.1}`{=latex}, which reports that the inclusion of exemplary QAs during training enhances models’ test performances. We believe that the impact of the thinking bias on the knowledge application abilities within the DescriptionIsName group is the main reason for this divergence. The in-domain performance of models trained with CoT-enhanced QA pairs is slightly lower than that of models trained without CoT steps. We mainly attribute this to the exclusion of CoT steps in our test settings.

# Related works [sec:related-works]

#### The reversal curse in LLMs

Recent studies have uncovered a notable observation concerning LLMs’ generalization abilities. Besides the original paper reporting the reversal curse phenomenon `\cite{Reversal_curse}`{=latex}, `\citet{influence_function}`{=latex} propose an influence function and observe that training examples that match the order (*e.g.*, “A is B”) are far more influential than examples with a reversed order (*e.g.*, “B is A”) when given the input “A”. This suggests that models without training on facts presented in both directions cannot generalize to both directions. `\citet{BICO}`{=latex} suggest that the reversal curse could be partly attributed to the training objective of next-token prediction. `\citet{understanding_RC}`{=latex} later offers a theoretical analysis of a one-layer transformer to suggest that the reversal curse on completion task stems from the training dynamics of gradient descent.

Our work remains orthogonal to the above works as we explore the manifestation of the reversal curse on more diverse tasks beyond completion. Our experiments reveal that LLMs can generalize beyond and apply their knowledge to MCQs when biographical facts are formatted with names preceding descriptions. Moreover, We find that even when trained with facts presented in both directions, LLMs predominantly master only the part that matches their innate thinking bias.

#### Effect of data quality

The quality of data can significantly influence LLMs’ learning efficiency `\cite{scaling-law1, scaling-law2, scaling_law3}`{=latex}. The existing literature on improving the quality of training data can generally be divided into two streams. The first stream enhances data quality through delicate data filtering. A straightforward yet effective filtering method is to remove duplications for both pre-training and finetuning stages, which not only reduces the training duration but also enhances the performance as evidenced by `\cite{deduplication2, deduplication, deduplication1}`{=latex}. Another strategy involves condensing the dataset by selectively sub-sampling training instances, which could be executed through heuristic or manual curation `\cite{LIMA}`{=latex} or with a model-centric approach `\cite{crafting_IC}`{=latex}. The second stream aims at increasing the diversity of training examples through data augmentation. Traditional techniques including rule-based `\cite{EDA}`{=latex} and interpolation-based `\cite{Segmixup}`{=latex} methods generally focus on the token-level manipulation and the feature space perturbation. After LLMs demonstrate their superior power in data generation, a growing number of studies `\cite{self-instruct, auggpt, llm_powered_data_augmentation}`{=latex} have turned to LLMs to produce high-quality and task-specific synthetic data.

Our findings, emphasizing the significance of document structure, can not only be utilized as a filtering criterion towards data efficiency and efficacy but also hold the potential to be combined with entity relation extraction `\cite{review_of_ERE}`{=latex} and knowledge graph `\cite{unifying_LLM_KG}`{=latex} for more effective data augmentation.

# Conclusion [sec:conclusion]

In this study, we initially investigate how the reversal curse manifests across diverse tasks to assess the true boundary of LLMs’ generalization abilities. Our findings reveal that LLMs can generalize effectively to “B is A” in multiple-choice questions where both A and B are presented. Notably, this generalization ability appears to be closely linked with the structure of each fact used for training. Furthermore, we reveal that LLMs possess an inherent thinking bias in query processing and knowledge application, which explains and underscores the importance of document structure to successful learning. Our limitations and social impacts are discussed in <a href="#sec:Appendix-Discussion" data-reference-type="ref+Label" data-reference="sec:Appendix-Discussion">14</a> and <a href="#sec:Appendix-impact" data-reference-type="ref+Label" data-reference="sec:Appendix-impact">15</a>. We hope this work can provide new insights into interpreting and enhancing LLMs’ learning abilities.

# Limitations and future work [sec:Appendix-Discussion]

Our study, while providing valuable insights into the manifestation of the reversal curse and LLMs’ problem-solving patterns, has several limitations. Firstly, our work mainly focuses on finding a hypothesis to explain the puzzling MCQ results, namely the thinking bias, and validate its existence through both CoT prompting and internal interaction. The underlying cause of this bias, as well as the proof of its presence in today’s state-of-the-art close-sourced models, is not fully explored by our current work.

Secondly, despite several attempts to mitigate the thinking bias, we are frustrated to find that currently available techniques failed to alleviate this problem. It derives a hypothesis that an exhaustive rewrite of all training documents to align their structures with the thinking bias seems to be the most effective approach to facilitate the generalization of knowledge. How to derive an effective and practical methodology to enhance LLMs’ training efficacy remains a challenging problem, and we leave this for future work.

# Acknowledgement [acknowledgement]

This work was supported in part by The National Nature Science Foundation of China (Grant No: 62303406, 62273302, 62036009, 61936006, 62273303), in part by Key S&T Programme of Hangzhou, China (Grant No: 2022AIZD0084), in part by Yongjiang Talent Introduction Programme (Grant No: 2023A-194-G, 2022A-240-G).

# References [references]

<div class="thebibliography" markdown="1">

AI@Meta The llama 3 herd of models *arXiv preprint arXiv:2407.21783*, 2024. **Abstract:** Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called Llama 3. It is a herd of language models that natively support multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with 405B parameters and a context window of up to 128K tokens. This paper presents an extensive empirical evaluation of Llama 3. We find that Llama 3 delivers comparable quality to leading language models such as GPT-4 on a plethora of tasks. We publicly release Llama 3, including pre-trained and post-trained versions of the 405B parameter language model and our Llama Guard 3 model for input and output safety. The paper also presents the results of experiments in which we integrate image, video, and speech capabilities into Llama 3 via a compositional approach. We observe this approach performs competitively with the state-of-the-art on image, video, and speech recognition tasks. The resulting models are not yet being broadly released as they are still under development. (@Llama3)

Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans The reversal curse: Llms trained on "a is b" fail to learn "b is a" *CoRR*, abs/2309.12288, 2023. . URL <https://doi.org/10.48550/arXiv.2309.12288>. **Abstract:** We expose a surprising failure of generalization in auto-regressive large language models (LLMs). If a model is trained on a sentence of the form "A is B", it will not automatically generalize to the reverse direction "B is A". This is the Reversal Curse. For instance, if a model is trained on "Valentina Tereshkova was the first woman to travel to space", it will not automatically be able to answer the question, "Who was the first woman to travel to space?". Moreover, the likelihood of the correct answer ("Valentina Tershkova") will not be higher than for a random name. Thus, models do not generalize a prevalent pattern in their training set: if "A is B" occurs, "B is A" is more likely to occur. It is worth noting, however, that if "A is B" appears in-context, models can deduce the reverse relationship. We provide evidence for the Reversal Curse by finetuning GPT-3 and Llama-1 on fictitious statements such as "Uriah Hawthorne is the composer of Abyssal Melodies" and showing that they fail to correctly answer "Who composed Abyssal Melodies?". The Reversal Curse is robust across model sizes and model families and is not alleviated by data augmentation. We also evaluate ChatGPT (GPT-3.5 and GPT-4) on questions about real-world celebrities, such as "Who is Tom Cruise’s mother? \[A: Mary Lee Pfeiffer\]" and the reverse "Who is Mary Lee Pfeiffer’s son?". GPT-4 correctly answers questions like the former 79% of the time, compared to 33% for the latter. Code available at: https://github.com/lukasberglund/reversal_curse. (@Reversal_curse)

Sumithra Bhakthavatsalam, Daniel Khashabi, Tushar Khot, Bhavana Dalvi Mishra, Kyle Richardson, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord, and Peter Clark Think you have solved direct-answer question answering? try arc-da, the direct-answer AI2 reasoning challenge *CoRR*, abs/2102.03315, 2021. URL <https://arxiv.org/abs/2102.03315>. **Abstract:** We present the ARC-DA dataset, a direct-answer ("open response", "freeform") version of the ARC (AI2 Reasoning Challenge) multiple-choice dataset. While ARC has been influential in the community, its multiple-choice format is unrepresentative of real-world questions, and multiple choice formats can be particularly susceptible to artifacts. The ARC-DA dataset addresses these concerns by converting questions to direct-answer format using a combination of crowdsourcing and expert review. The resulting dataset contains 2985 questions with a total of 8436 valid answers (questions typically have more than one valid answer). ARC-DA is one of the first DA datasets of natural questions that often require reasoning, and where appropriate question decompositions are not evident from the questions themselves. We describe the conversion approach taken, appropriate evaluation metrics, and several strong models. Although high, the best scores (81% GENIE, 61.4% F1, 63.2% ROUGE-L) still leave considerable room for improvement. In addition, the dataset provides a natural setting for new research on explanation, as many questions require reasoning to construct answers. We hope the dataset spurs further advances in complex question-answering by the community. ARC-DA is available at https://allenai.org/data/arc-da (@arc)

Sam Bowman Eight things to know about large language models *ArXiv*, abs/2304.00612, 2023. URL <https://api.semanticscholar.org/CorpusID:257913333>. **Abstract:** The widespread public deployment of large language models (LLMs) in recent months has prompted a wave of new attention and engagement from advocates, policymakers, and scholars from many fields. This attention is a timely response to the many urgent questions that this technology raises, but it can sometimes miss important considerations. This paper surveys the evidence for eight potentially surprising such points: 1. LLMs predictably get more capable with increasing investment, even without targeted innovation. 2. Many important LLM behaviors emerge unpredictably as a byproduct of increasing investment. 3. LLMs often appear to learn and use representations of the outside world. 4. There are no reliable techniques for steering the behavior of LLMs. 5. Experts are not yet able to interpret the inner workings of LLMs. 6. Human performance on a task isn’t an upper bound on LLM performance. 7. LLMs need not express the values of their creators nor the values encoded in web text. 8. Brief interactions with LLMs are often misleading. (@eight_things)

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott M. Lundberg, Harsha Nori, Hamid Palangi, Marco Túlio Ribeiro, and Yi Zhang Sparks of artificial general intelligence: Early experiments with GPT-4 *CoRR*, abs/2303.12712, 2023. . URL <https://doi.org/10.48550/arXiv.2303.12712>. **Abstract:** Artificial intelligence (AI) researchers have been developing and refining large language models (LLMs) that exhibit remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. The latest model developed by OpenAI, GPT-4, was trained using an unprecedented scale of compute and data. In this paper, we report on our investigation of an early version of GPT-4, when it was still in active development by OpenAI. We contend that (this early version of) GPT-4 is part of a new cohort of LLMs (along with ChatGPT and Google’s PaLM for example) that exhibit more general intelligence than previous AI models. We discuss the rising capabilities and implications of these models. We demonstrate that, beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4’s performance is strikingly close to human-level performance, and often vastly surpasses prior models such as ChatGPT. Given the breadth and depth of GPT-4’s capabilities, we believe that it could reasonably be viewed as an early (yet still incomplete) version of an artificial general intelligence (AGI) system. In our exploration of GPT-4, we put special emphasis on discovering its limitations, and we discuss the challenges ahead for advancing towards deeper and more comprehensive versions of AGI, including the possible need for pursuing a new paradigm that moves beyond next-word prediction. We conclude with reflections on societal influences of the recent technological leap and future research directions. (@sparks_of_artificial)

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality *See https://vicuna. lmsys. org (accessed 14 April 2023)*, 2023. (@Vicuna)

Andrea Cossu, Tinne Tuytelaars, Antonio Carta, Lucia C. Passaro, Vincenzo Lomonaco, and Davide Bacciu Continual pre-training mitigates forgetting in language and vision *CoRR*, abs/2205.09357, 2022. . URL <https://doi.org/10.48550/arXiv.2205.09357>. **Abstract:** Pre-trained models are nowadays a fundamental component of machine learning research. In continual learning, they are commonly used to initialize the model before training on the stream of non-stationary data. However, pre-training is rarely applied during continual learning. We formalize and investigate the characteristics of the continual pre-training scenario in both language and vision environments, where a model is continually pre-trained on a stream of incoming data and only later fine-tuned to different downstream tasks. We show that continually pre-trained models are robust against catastrophic forgetting and we provide strong empirical evidence supporting the fact that self-supervised pre-training is more effective in retaining previous knowledge than supervised protocols. Code is provided at https://github.com/AndreaCossu/continual-pretraining-nlp-vision . (@CPT2)

Haixing Dai, Zhengliang Liu, Wenxiong Liao, Xiaoke Huang, Yihan Cao, Zihao Wu, Lin Zhao, Shaochen Xu, Wei Liu, Ninghao Liu, et al Auggpt: Leveraging chatgpt for text data augmentation *arXiv preprint arXiv:2302.13007*, 2023. **Abstract:** Text data augmentation is an effective strategy for overcoming the challenge of limited sample sizes in many natural language processing (NLP) tasks. This challenge is especially prominent in the few-shot learning scenario, where the data in the target domain is generally much scarcer and of lowered quality. A natural and widely-used strategy to mitigate such challenges is to perform data augmentation to better capture the data invariance and increase the sample size. However, current text data augmentation methods either can’t ensure the correct labeling of the generated data (lacking faithfulness) or can’t ensure sufficient diversity in the generated data (lacking compactness), or both. Inspired by the recent success of large language models, especially the development of ChatGPT, which demonstrated improved language comprehension abilities, in this work, we propose a text data augmentation approach based on ChatGPT (named AugGPT). AugGPT rephrases each sentence in the training samples into multiple conceptually similar but semantically different samples. The augmented samples can then be used in downstream model training. Experiment results on few-shot learning text classification tasks show the superior performance of the proposed AugGPT approach over state-of-the-art text data augmentation methods in terms of testing accuracy and distribution of the augmented samples. (@auggpt)

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jian, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Sean Welleck, Xiang Ren, Allyson Ettinger, Zaïd Harchaoui, and Yejin Choi Faith and fate: Limits of transformers on compositionality *ArXiv*, abs/2305.18654, 2023. URL <https://api.semanticscholar.org/CorpusID:258967391>. **Abstract:** Transformer large language models (LLMs) have sparked admiration for their exceptional performance on tasks that demand intricate multi-step reasoning. Yet, these models simultaneously show failures on surprisingly trivial problems. This begs the question: Are these errors incidental, or do they signal more substantial limitations? In an attempt to demystify transformer LLMs, we investigate the limits of these models across three representative compositional tasks – multi-digit multiplication, logic grid puzzles, and a classic dynamic programming problem. These tasks require breaking problems down into sub-steps and synthesizing these steps into a precise answer. We formulate compositional tasks as computation graphs to systematically quantify the level of complexity, and break down reasoning steps into intermediate sub-procedures. Our empirical findings suggest that transformer LLMs solve compositional tasks by reducing multi-step compositional reasoning into linearized subgraph matching, without necessarily developing systematic problem-solving skills. To round off our empirical study, we provide theoretical arguments on abstract multi-step reasoning problems that highlight how autoregressive generations’ performance can rapidly decay with\\},increased\\},task\\},complexity. (@faith_and_fate)

Ronen Eldan and Yuanzhi Li Tinystories: How small can language models be and still speak coherent english? *CoRR*, abs/2305.07759, 2023. . URL <https://doi.org/10.48550/arXiv.2305.07759>. **Abstract:** Language models (LMs) are powerful tools for natural language processing, but they often struggle to produce coherent and fluent text when they are small. Models with around 125M parameters such as GPT-Neo (small) or GPT-2 (small) can rarely generate coherent and consistent English text beyond a few words even after extensive training. This raises the question of whether the emergence of the ability to produce coherent English text only occurs at larger scales (with hundreds of millions of parameters or more) and complex architectures (with many layers of global attention). In this work, we introduce TinyStories, a synthetic dataset of short stories that only contain words that a typical 3 to 4-year-olds usually understand, generated by GPT-3.5 and GPT-4. We show that TinyStories can be used to train and evaluate LMs that are much smaller than the state-of-the-art models (below 10 million total parameters), or have much simpler architectures (with only one transformer block), yet still produce fluent and consistent stories with several paragraphs that are diverse and have almost perfect grammar, and demonstrate reasoning capabilities. We also introduce a new paradigm for the evaluation of language models: We suggest a framework which uses GPT-4 to grade the content generated by these models as if those were stories written by students and graded by a (human) teacher. This new paradigm overcomes the flaws of standard benchmarks which often requires the model’s output to be very structures, and moreover provides a multidimensional score for the model, providing scores for different capabilities such as grammar, creativity and consistency. We hope that TinyStories can facilitate the development, analysis and research of LMs, especially for low-resource or specialized domains, and shed light on the emergence of language capabilities in LMs. (@scaling-law2)

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy Transformer feed-forward layers are key-value memories In *Empirical Methods in Natural Language Processing (EMNLP)*, 2021. **Abstract:** Feed-forward layers constitute two-thirds of a transformer model’s parameters, yet their role in the network remains under-explored. We show that feed-forward layers in transformer-based language models operate as key-value memories, where each key correlates with textual patterns in the training examples, and each value induces a distribution over the output vocabulary. Our experiments show that the learned patterns are human-interpretable, and that lower layers tend to capture shallow patterns, while upper layers learn more semantic ones. The values complement the keys’ input patterns by inducing output distributions that concentrate probability mass on tokens likely to appear immediately after each pattern, particularly in the upper layers. Finally, we demonstrate that the output of a feed-forward layer is a composition of its memories, which is subsequently refined throughout the model’s layers via residual connections to produce the final output distribution. (@key-value_mem)

Roger B. Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamile Lukosiute, Karina Nguyen, Nicholas Joseph, Sam McCandlish, Jared Kaplan, and Samuel R. Bowman Studying large language model generalization with influence functions *CoRR*, abs/2308.03296, 2023. . URL <https://doi.org/10.48550/arXiv.2308.03296>. **Abstract:** When trying to gain better visibility into a machine learning model in order to understand and mitigate the associated risks, a potentially valuable source of evidence is: which training examples most contribute to a given behavior? Influence functions aim to answer a counterfactual: how would the model’s parameters (and hence its outputs) change if a given sequence were added to the training set? While influence functions have produced insights for small models, they are difficult to scale to large language models (LLMs) due to the difficulty of computing an inverse-Hessian-vector product (IHVP). We use the Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EK-FAC) approximation to scale influence functions up to LLMs with up to 52 billion parameters. In our experiments, EK-FAC achieves similar accuracy to traditional influence function estimators despite the IHVP computation being orders of magnitude faster. We investigate two algorithmic techniques to reduce the cost of computing gradients of candidate training sequences: TF-IDF filtering and query batching. We use influence functions to investigate the generalization patterns of LLMs, including the sparsity of the influence patterns, increasing abstraction with scale, math and programming abilities, cross-lingual generalization, and role-playing behavior. Despite many apparently sophisticated forms of generalization, we identify a surprising limitation: influences decay to near-zero when the order of key phrases is flipped. Overall, influence functions give us a powerful new tool for studying the generalization properties of LLMs. (@influence_function)

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li Textbooks are all you need *CoRR*, abs/2306.11644, 2023. . URL <https://doi.org/10.48550/arXiv.2306.11644>. **Abstract:** We introduce phi-1, a new large language model for code, with significantly smaller size than competing models: phi-1 is a Transformer-based model with 1.3B parameters, trained for 4 days on 8 A100s, using a selection of “textbook quality" data from the web (6B tokens) and synthetically generated textbooks and exercises with GPT-3.5 (1B tokens). Despite this small scale, phi-1 attains pass@1 accuracy 50.6% on HumanEval and 55.5% on MBPP. It also displays surprising emergent properties compared to phi-1-base, our model before our finetuning stage on a dataset of coding exercises, and phi-1-small, a smaller model with 350M parameters trained with the same pipeline as phi-1 that still achieves 45% on HumanEval. (@scaling_law3)

Demi Guo, Yoon Kim, and Alexander M. Rush Sequence-level mixed sample data augmentation In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pages 5547–5552. Association for Computational Linguistics, 2020. . URL <https://doi.org/10.18653/v1/2020.emnlp-main.447>. **Abstract:** Despite their empirical success, neural networks still have difficulty capturing compositional aspects of natural language. This work proposes a simple data augmentation approach to encourage compositional behavior in neural models for sequence-to-sequence problems. Our approach, SeqMix, creates new synthetic examples by softly combining input/output sequences from the training set. We connect this approach to existing techniques such as SwitchOut and word dropout, and show that these techniques are all essentially approximating variants of a single objective. SeqMix consistently yields approximately 1.0 BLEU improvement on five different translation datasets over strong Transformer baselines. On tasks that require strong compositional generalization such as SCAN and semantic parsing, SeqMix also offers further improvements. (@Segmixup)

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt Measuring massive multitask language understanding In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021. **Abstract:** We propose a new test to measure a text model’s multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model’s academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings. (@MMLU)

J. Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen Lora: Low-rank adaptation of large language models *ArXiv*, abs/2106.09685, 2021. URL <https://api.semanticscholar.org/CorpusID:235458009>. **Abstract:** An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at https://github.com/microsoft/LoRA. (@LoRA)

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han Large language models can self-improve *arXiv preprint arXiv:2210.11610*, 2022. **Abstract:** Large Language Models (LLMs) have achieved excellent performances in various tasks. However, fine-tuning an LLM requires extensive supervision. Human, on the other hand, may improve their reasoning abilities by self-thinking without external inputs. In this work, we demonstrate that an LLM is also capable of self-improving with only unlabeled datasets. We use a pre-trained LLM to generate "high-confidence" rationale-augmented answers for unlabeled questions using Chain-of-Thought prompting and self-consistency, and fine-tune the LLM using those self-generated solutions as target outputs. We show that our approach improves the general reasoning ability of a 540B-parameter LLM (74.4%-\>82.1% on GSM8K, 78.2%-\>83.0% on DROP, 90.0%-\>94.4% on OpenBookQA, and 63.4%-\>67.9% on ANLI-A3) and achieves state-of-the-art-level performance, without any ground truth label. We conduct ablation studies and show that fine-tuning on reasoning is critical for self-improvement. (@train_with_CoT1)

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed Mistral 7b *CoRR*, abs/2310.06825, 2023. . URL <https://doi.org/10.48550/arXiv.2310.06825>. **Abstract:** We introduce Mistral 7B v0.1, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms Llama 2 13B across all evaluated benchmarks, and Llama 1 34B in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B – Instruct, that surpasses the Llama 2 13B – Chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license. (@Mistral)

Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, and Bing Liu Continual pre-training of language models In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. URL <https://openreview.net/pdf?id=m_GDIItaI3o>. **Abstract:** Language models (LMs) have been instrumental for the rapid advance of natural language processing. This paper studies continual pre-training of LMs, in particular, continual domain-adaptive pre-training (or continual DAP-training). Existing research has shown that further pre-training an LM using a domain corpus to adapt the LM to the domain can improve the end-task performance in the domain. This paper proposes a novel method to continually DAP-train an LM with a sequence of unlabeled domain corpora to adapt the LM to these domains to improve their end-task performances. The key novelty of our method is a soft-masking mechanism that directly controls the update to the LM. A novel proxy is also proposed to preserve the general knowledge in the original LM. Additionally, it contrasts the representations of the previously learned domain knowledge (including the general knowledge in the pre-trained LM) and the knowledge from the current full network to achieve knowledge integration. The method not only overcomes catastrophic forgetting, but also achieves knowledge transfer to improve end-task performances. Empirical evaluation demonstrates the effectiveness of the proposed method. (@CPT1)

Diederik P. Kingma and Jimmy Ba Adam: A method for stochastic optimization *CoRR*, abs/1412.6980, 2014. URL <https://api.semanticscholar.org/CorpusID:6628106>. **Abstract:** We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm. (@adam_optimizer)

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini Deduplicating training data makes language models better In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022*, pages 8424–8445. Association for Computational Linguistics, 2022. . URL <https://doi.org/10.18653/v1/2022.acl-long.577>. **Abstract:** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2022. (@deduplication1)

Yoonsang Lee, Pranav Atreya, Xi Ye, and Eunsol Choi Crafting in-context examples according to lms’ parametric knowledge *CoRR*, abs/2311.09579, 2023. . URL <https://doi.org/10.48550/arXiv.2311.09579>. **Abstract:** In-context learning can improve the performances of knowledge-rich tasks such as question answering. In such scenarios, in-context examples trigger a language model (LM) to surface information stored in its parametric knowledge. We study how to better construct in-context example sets, based on whether the model is aware of the in-context examples. We identify ’known’ examples, where models can correctly answer from their parametric knowledge, and ’unknown’ ones. Our experiments show that prompting with ’unknown’ examples decreases the performance, potentially as it encourages hallucination rather than searching for its parametric knowledge. Constructing an in-context example set that presents both known and unknown information performs the best across diverse settings. We perform analysis on three multi-answer question answering datasets, which allows us to further study answer set ordering strategies based on the LM’s knowledge of each answer. Together, our study sheds light on how to best construct in-context example sets for knowledge-rich tasks. (@crafting_IC)

Charles N. Li and Sandra A. Thompson Subject and topic: A new typology of language . **Abstract:** : The subject matter of this paper is a discussion of the application of the new technologies to the teaching of Uzbek language learning and its methodological principles in particular institutions of higher learning like the universities. While the paper can address any language, in our case, the Uzbek language teaching in South Korea is the problem statement. It will be argued that the integration of new media into Uzbek language learning is a necessary step ensuring the acquisition of the kind of language skills and competencies needed for living and working in the knowledge society. A lot has been written on Uzbek language but no postulations have been made regarding the use of digital technologies in its teaching. It is our presupposition that innovative use of such technologies in Uzbek language will lead to more flexibility in the content and organisation of learning; new media must be looked at not simply in terms of traditional self-study materials but rather in terms of tools for learning. New information and communication technologies and their role in language learning processes are the topic of this paper, but constructivism as the appropriate paradigm for language learning in the coming millennium will also be discussed. In addition, the paper proposes a typology and an evaluation of technology-enhanced materials for language learning, and presents a few examples. New technologies have (@human_language)

Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Vi’egas, Hanspeter Pfister, and Martin Wattenberg Emergent world representations: Exploring a sequence model trained on a synthetic task *ArXiv*, abs/2210.13382, 2022. URL <https://api.semanticscholar.org/CorpusID:253098566>. **Abstract:** Language models show a surprising range of capabilities, but the source of their apparent competence is unclear. Do these networks just memorize a collection of surface statistics, or do they rely on internal representations of the process that generates the sequences they see? We investigate this question by applying a variant of the GPT model to the task of predicting legal moves in a simple board game, Othello. Although the network has no a priori knowledge of the game or its rules, we uncover evidence of an emergent nonlinear internal representation of the board state. Interventional experiments indicate this representation can be used to control the output of the network and create "latent saliency maps" that can help explain predictions in human terms. (@emergent_world)

Shu Chen Li and Stephan Lewandowsky Forward and backward recall: Different retrieval processes *Journal of Experimental Psychology Learning Memory and Cognition*, 21 (4): 837–847, 1995. (@backward_recall)

Chin-Yew Lin Rouge: A package for automatic evaluation of summaries In *Text summarization branches out*, pages 74–81, 2004. **Abstract:** ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. It includes measures to automatically determine the quality of a summary by comparing it to other (ideal) summaries created by humans. The measures count the number of overlapping units such as n-gram, word sequences, and word pairs between the computer-generated summary to be evaluated and the ideal summaries created by humans. This paper introduces four different ROUGE measures: ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S included in the ROUGE summarization evaluation package and their evaluations. Three of them have been used in the Document Understanding Conference (DUC) 2004, a large-scale summarization evaluation sponsored by NIST. (@Rouge)

Stephanie Lin, Jacob Hilton, and Owain Evans Truthfulqa: Measuring how models mimic human falsehoods In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022*, pages 3214–3252. Association for Computational Linguistics, 2022. . URL <https://doi.org/10.18653/v1/2022.acl-long.229>. **Abstract:** We propose a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. We crafted questions that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts. We tested GPT-3, GPT-Neo/J, GPT-2 and a T5-based model. The best model was truthful on 58% of questions, while human performance was 94%. Models generated many false answers that mimic popular misconceptions and have the potential to deceive humans. The largest models were generally the least truthful. This contrasts with other NLP tasks, where performance improves with model size. However, this result is expected if false answers are learned from the training distribution. We suggest that scaling up models alone is less promising for improving truthfulness than fine-tuning using training objectives other than imitation of text from the web. (@truthfulqa)

Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, and Mike Williams Towards understanding grokking: An effective theory of representation learning *ArXiv*, abs/2205.10343, 2022. URL <https://api.semanticscholar.org/CorpusID:248965387>. **Abstract:** We aim to understand grokking, a phenomenon where models generalize long after overfitting their training set. We present both a microscopic analysis anchored by an effective theory and a macroscopic analysis of phase diagrams describing learning performance across hyperparameters. We find that generalization originates from structured representations whose training dynamics and dependence on training set size can be predicted by our effective theory in a toy setting. We observe empirically the presence of four learning phases: comprehension, grokking, memorization, and confusion. We find representation learning to occur only in a "Goldilocks zone" (including comprehension and grokking) between memorization and confusion. We find on transformers the grokking phase stays closer to the memorization phase (compared to the comprehension phase), leading to delayed generalization. The Goldilocks phase is reminiscent of "intelligence from starvation" in Darwinian evolution, where resource limitations drive discovery of more efficient solutions. This study not only provides intuitive explanations of the origin of grokking, but also highlights the usefulness of physics-inspired tools, e.g., effective theories and phase diagrams, for understanding deep learning. (@grokking_ML)

Ang Lv, Kaiyi Zhang, Shufang Xie, Quan Tu, Yuhan Chen, Ji-Rong Wen, and Rui Yan Are we falling in a middle-intelligence trap? an analysis and mitigation of the reversal curse *CoRR*, abs/2311.07468, 2023. . URL <https://doi.org/10.48550/arXiv.2311.07468>. **Abstract:** Recent research observed a noteworthy phenomenon in large language models (LLMs), referred to as the “reversal curse.” The reversal curse is that when dealing with two entities, denoted as $a$ and $b$, connected by their relation $R$ and its inverse $R\^{-1}$, LLMs excel in handling sequences in the form of “$aRb$,” but encounter challenges when processing “$bR\^{-1}a$,” whether in generation or comprehension. For instance, GPT-4 can accurately respond to the query “Tom Cruise’s mother is?” with “Mary Lee Pfeiffer,” but it struggles to provide a satisfactory answer when asked “Mary Lee Pfeiffer’s son is?” In this paper, we undertake the first-ever study of how the reversal curse happens in LLMs. Our investigations reveal that the reversal curse can stem from the specific training objectives, which become particularly evident in the widespread use of next-token prediction within most causal language models. We hope this initial investigation can draw more attention to the reversal curse, as well as other underlying limitations in current LLMs. (@BICO)

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov Locating and editing factual associations in GPT In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*, 2022. **Abstract:** We analyze the storage and recall of factual associations in autoregressive transformer language models, finding evidence that these associations correspond to localized, directly-editable computations. We first develop a causal intervention for identifying neuron activations that are decisive in a model’s factual predictions. This reveals a distinct set of steps in middle-layer feed-forward modules that mediate factual predictions while processing subject tokens. To test our hypothesis that these computations correspond to factual association recall, we modify feed-forward weights to update specific factual associations using Rank-One Model Editing (ROME). We find that ROME is effective on a standard zero-shot relation extraction (zsRE) model-editing task, comparable to existing methods. To perform a more sensitive evaluation, we also evaluate ROME on a new dataset of counterfactual assertions, on which it simultaneously maintains both specificity and generalization, whereas other methods sacrifice one or another. Our results confirm an important role for mid-layer feed-forward modules in storing factual associations and suggest that direct manipulation of computational mechanisms may be a feasible approach for model editing. The code, dataset, visualizations, and an interactive demo notebook are available at https://rome.baulab.info/ (@ROME)

Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau Mass-editing memory in a transformer In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. **Abstract:** Recent work has shown exciting promise in updating large language models with new memories, so as to replace obsolete information or add specialized knowledge. However, this line of work is predominantly limited to updating single associations. We develop MEMIT, a method for directly updating a language model with many memories, demonstrating experimentally that it can scale up to thousands of associations for GPT-J (6B) and GPT-NeoX (20B), exceeding prior work by orders of magnitude. Our code and data are at https://memit.baulab.info. (@MEMIT)

Swaroop Mishra and Bhavdeep Singh Sachdeva Do we need to create big datasets to learn a task? In Nafise Sadat Moosavi, Angela Fan, Vered Shwartz, Goran Glavas, Shafiq R. Joty, Alex Wang, and Thomas Wolf, editors, *Proceedings of SustaiNLP: Workshop on Simple and Efficient Natural Language Processing, SustaiNLP@EMNLP 2020, Online, November 20, 2020*, pages 169–173. Association for Computational Linguistics, 2020. . URL <https://doi.org/10.18653/v1/2020.sustainlp-1.23>. **Abstract:** Deep Learning research has been largely accelerated by the development of huge datasets such as Imagenet. The general trend has been to create big datasets to make a deep neural network learn. A huge amount of resources is being spent in creating these big datasets, developing models, training them, and iterating this process to dominate leaderboards. We argue that the trend of creating bigger datasets needs to be revised by better leveraging the power of pre-trained language models. Since the language models have already been pre-trained with huge amount of data and have basic linguistic knowledge, there is no need to create big datasets to learn a task. Instead, we need to create a dataset that is sufficient for the model to learn various task-specific terminologies, such as ‘Entailment’, ‘Neutral’, and ‘Contradiction’ for NLI. As evidence, we show that RoBERTA is able to achieve near-equal performance on 2% data of SNLI. We also observe competitive zero-shot generalization on several OOD datasets. In this paper, we propose a baseline algorithm to find the optimal dataset for learning a task. (@deduplication2)

Melanie Mitchell and David C. Krakauer The debate over understanding in ai’s large language models *Proceedings of the National Academy of Sciences of the United States of America*, 120, 2022. URL <https://api.semanticscholar.org/CorpusID:253107905>. **Abstract:** We survey a current, heated debate in the artificial intelligence (AI) research community on whether large pretrained language models can be said to understand language-and the physical and social situations language encodes-in any humanlike sense. We describe arguments that have been made for and against such understanding and key questions for the broader sciences of intelligence that have arisen in light of these arguments. We contend that an extended science of intelligence can be developed that will provide insight into distinct modes of understanding, their strengths and limitations, and the challenge of integrating diverse forms of cognition. (@understanding_debate)

Shikhar Murty, Pratyusha Sharma, Jacob Andreas, and Christopher D. Manning Grokking of hierarchical structure in vanilla transformers In *Annual Meeting of the Association for Computational Linguistics*, 2023. URL <https://api.semanticscholar.org/CorpusID:258967837>. **Abstract:** For humans, language production and comprehension is sensitive to the hierarchical structure of sentences. In natural language processing, past work has questioned how effectively neural sequence models like transformers capture this hierarchical structure when generalizing to structurally novel inputs. We show that transformer language models can learn to generalize hierarchically after training for extremely long periods—far beyond the point when in-domain accuracy has saturated. We call this phenomenon structural grokking. On multiple datasets, structural grokking exhibits inverted U-shaped scaling in model depth: intermediate-depth models generalize better than both very deep and very shallow transformers. When analyzing the relationship between model-internal properties and grokking, we find that optimal depth for grokking can be identified using the tree-structuredness metric of CITATION. Overall, our work provides strong evidence that, with extended training, vanilla transformers discover and use hierarchical structure. (@grokking_LM)

Maxwell Nye, Anders Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena Show your work: Scratchpads for intermediate computation with language models *ArXiv*, abs/2112.00114, 2021. URL <https://api.semanticscholar.org/CorpusID:244773644>. **Abstract:** Large pre-trained language models perform remarkably well on tasks that can be done "in one pass", such as generating realistic text or synthesizing computer programs. However, they struggle with tasks that require unbounded multi-step computation, such as adding integers or executing programs. Surprisingly, we find that these same models are able to perform complex multi-step computations – even in the few-shot regime – when asked to perform the operation "step by step", showing the results of intermediate computations. In particular, we train transformers to perform multi-step computations by asking them to emit intermediate computation steps into a "scratchpad". On a series of increasingly complex tasks ranging from long addition to the execution of arbitrary programs, we show that scratchpads dramatically improve the ability of language models to perform multi-step computations. (@Scratchpad)

OpenAI technical report *CoRR*, abs/2303.08774, 2023. . URL <https://doi.org/10.48550/arXiv.2303.08774>. **Abstract:** We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4’s performance based on models trained with no more than 1/1,000th the compute of GPT-4. (@GPT-4)

Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu Unifying large language models and knowledge graphs: A roadmap *CoRR*, abs/2306.08302, 2023. . URL <https://doi.org/10.48550/arXiv.2306.08302>. **Abstract:** Large language models (LLMs), such as ChatGPT and GPT4, are making new waves in the field of natural language processing and artificial intelligence, due to their emergent ability and generalizability. However, LLMs are black-box models, which often fall short of capturing and accessing factual knowledge. In contrast, Knowledge Graphs (KGs), Wikipedia and Huapu for example, are structured knowledge models that explicitly store rich factual knowledge. KGs can enhance LLMs by providing external knowledge for inference and interpretability. Meanwhile, KGs are difficult to construct and evolving by nature, which challenges the existing methods in KGs to generate new facts and represent unseen knowledge. Therefore, it is complementary to unify LLMs and KGs together and simultaneously leverage their advantages. In this article, we present a forward-looking roadmap for the unification of LLMs and KGs. Our roadmap consists of three general frameworks, namely, 1) KG-enhanced LLMs, which incorporate KGs during the pre-training and inference phases of LLMs, or for the purpose of enhancing understanding of the knowledge learned by LLMs; 2) LLM-augmented KGs, that leverage LLMs for different KG tasks such as embedding, completion, construction, graph-to-text generation, and question answering; and 3) Synergized LLMs + KGs, in which LLMs and KGs play equal roles and work in a mutually beneficial way to enhance both LLMs and KGs for bidirectional reasoning driven by both data and knowledge. We review and summarize existing efforts within these three frameworks in our roadmap and pinpoint their future research directions. (@unifying_LLM_KG)

Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, et al Check your facts and try again: Improving large language models with external knowledge and automated feedback *arXiv preprint arXiv:2302.12813*, 2023. **Abstract:** Large language models (LLMs), such as ChatGPT, are able to generate human-like, fluent responses for many downstream tasks, e.g., task-oriented dialog and question answering. However, applying LLMs to real-world, mission-critical applications remains challenging mainly due to their tendency to generate hallucinations and their inability to use external knowledge. This paper proposes a LLM-Augmenter system, which augments a black-box LLM with a set of plug-and-play modules. Our system makes the LLM generate responses grounded in external knowledge, e.g., stored in task-specific databases. It also iteratively revises LLM prompts to improve model responses using feedback generated by utility functions, e.g., the factuality score of a LLM-generated response. The effectiveness of LLM-Augmenter is empirically validated on two types of scenarios, task-oriented dialog and open-domain question answering. LLM-Augmenter significantly reduces ChatGPT’s hallucinations without sacrificing the fluency and informativeness of its responses. We make the source code and models publicly available. (@peng2023check)

Jacob Pfau, William Merrill, and Samuel R Bowman Let’s think dot by dot: Hidden computation in transformer language models *arXiv preprint arXiv:2404.15758*, 2024. **Abstract:** Chain-of-thought responses from language models improve performance across most benchmarks. However, it remains unclear to what extent these performance gains can be attributed to human-like task decomposition or simply the greater computation that additional tokens allow. We show that transformers can use meaningless filler tokens (e.g., ’......’) in place of a chain of thought to solve two hard algorithmic tasks they could not solve when responding without intermediate tokens. However, we find empirically that learning to use filler tokens is difficult and requires specific, dense supervision to converge. We also provide a theoretical characterization of the class of problems where filler tokens are useful in terms of the quantifier depth of a first-order formula. For problems satisfying this characterization, chain-of-thought tokens need not provide information about the intermediate computational steps involved in multi-token computations. In summary, our results show that additional tokens can provide computational benefits independent of token choice. The fact that intermediate tokens can act as filler tokens raises concerns about large language models engaging in unauditable, hidden computations that are increasingly detached from the observed chain-of-thought tokens. (@train_with_CoT2)

Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman Deep inside convolutional networks: Visualising image classification models and saliency maps In Yoshua Bengio and Yann LeCun, editors, *2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Workshop Track Proceedings*, 2014. **Abstract:** This paper addresses the visualisation of image classification models, learnt using deep Convolutional Networks (ConvNets). We consider two visualisation techniques, based on computing the gradient of the class score with respect to the input image. The first one generates an image, which maximises the class score \[Erhan et al., 2009\], thus visualising the notion of the class, captured by a ConvNet. The second technique computes a class saliency map, specific to a given image and class. We show that such maps can be employed for weakly supervised object segmentation using classification ConvNets. Finally, we establish the connection between the gradient-based ConvNet visualisation methods and deconvolutional networks \[Zeiler et al., 2013\]. (@Saliency_technique)

Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos Beyond neural scaling laws: beating power law scaling via data pruning In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*, 2022. **Abstract:** Widely observed neural scaling laws, in which error falls off as a power of the training set size, model size, or both, have driven substantial performance improvements in deep learning. However, these improvements through scaling alone require considerable costs in compute and energy. Here we focus on the scaling of error with dataset size and show how in theory we can break beyond power law scaling and potentially even reduce it to exponential scaling instead if we have access to a high-quality data pruning metric that ranks the order in which training examples should be discarded to achieve any pruned dataset size. We then test this improved scaling prediction with pruned dataset size empirically, and indeed observe better than power law scaling in practice on ResNets trained on CIFAR-10, SVHN, and ImageNet. Next, given the importance of finding high-quality pruning metrics, we perform the first large-scale benchmarking study of ten different data pruning metrics on ImageNet. We find most existing high performing metrics scale poorly to ImageNet, while the best are computationally intensive and require labels for every image. We therefore developed a new simple, cheap and scalable self-supervised pruning metric that demonstrates comparable performance to the best supervised metrics. Overall, our work suggests that the discovery of good data-pruning metrics may provide a viable path forward to substantially improved neural scaling laws, thereby reducing the resource costs of modern deep learning. (@scaling-law1)

Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou Recitation-augmented language models *ArXiv*, abs/2210.01296, 2022. URL <https://api.semanticscholar.org/CorpusID:252692968>. **Abstract:** We propose a new paradigm to help Large Language Models (LLMs) generate more accurate factual knowledge without retrieving from an external corpus, called RECITation-augmented gEneration (RECITE). Different from retrieval-augmented language models that retrieve relevant documents before generating the outputs, given an input, RECITE first recites one or several relevant passages from LLMs’ own memory via sampling, and then produces the final answers. We show that RECITE is a powerful paradigm for knowledge-intensive NLP tasks. Specifically, we show that by utilizing recitation as the intermediate step, a recite-and-answer scheme can achieve new state-of-the-art performance in various closed-book question answering (CBQA) tasks. In experiments, we verify the effectiveness of \\}method\~on four pre-trained models (PaLM, UL2, OPT, and Codex) and three CBQA tasks (Natural Questions, TriviaQA, and HotpotQA). Our code is available at "https://github.com/Edward-Sun/RECITE". (@Sun2022RecitationAugmentedLM)

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom Llama 2: Open foundation and fine-tuned chat models *CoRR*, abs/2307.09288, 2023. . URL <https://doi.org/10.48550/arXiv.2307.09288>. **Abstract:** In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs. (@Llama2)

Meimei Tuo and Wenzhong Yang Review of entity relation extraction *J. Intell. Fuzzy Syst.*, 44 (5): 7391–7405, 2023. . URL <https://doi.org/10.3233/JIFS-223915>. **Abstract:** Because of large amounts of unstructured data generated on the Internet, entity relation extraction is believed to have high commercial value. Entity relation extraction is a case of information extraction and it is based on entity recognition. This paper firstly gives a brief overview of relation extraction. On the basis of reviewing the history of relation extraction, the research status of relation extraction is analyzed. Then the paper divides theses research into three categories: supervised machine learning methods, semi-supervised machine learning methods and unsupervised machine learning method, and toward to the deep learning direction. (@review_of_ERE)

Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun Label words are anchors: An information flow perspective for understanding in-context learning In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pages 9840–9855. Association for Computational Linguistics, 2023. **Abstract:** In-context learning (ICL) emerges as a promising capability of large language models (LLMs) by providing them with demonstration examples to perform diverse tasks. However, the underlying mechanism of how LLMs learn from the provided context remains under-explored. In this paper, we investigate the working mechanism of ICL through an information flow lens. Our findings reveal that label words in the demonstration examples function as anchors: (1) semantic information aggregates into label word representations during the shallow computation layers’ processing; (2) the consolidated information in label words serves as a reference for LLMs’ final predictions. Based on these insights, we introduce an anchor re-weighting method to improve ICL performance, a demonstration compression technique to expedite inference, and an analysis framework for diagnosing ICL errors in GPT2-XL. The promising applications of our findings again validate the uncovered ICL working mechanism and pave the way for future studies. (@anchors)

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi Self-instruct: Aligning language models with self-generated instructions In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki, editors, *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pages 13484–13508. Association for Computational Linguistics, 2023. . URL <https://doi.org/10.18653/v1/2023.acl-long.754>. **Abstract:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023. (@self-instruct)

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou Chain-of-thought prompting elicits reasoning in large language models In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*, 2022. **Abstract:** We explore how generating a chain of thought – a series of intermediate reasoning steps – significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain of thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain of thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier. (@CoT)

Jason W. Wei and Kai Zou easy data augmentation techniques for boosting performance on text classification tasks In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019*, pages 6381–6387. Association for Computational Linguistics, 2019. . URL <https://doi.org/10.18653/v1/D19-1670>. **Abstract:** Jason Wei, Kai Zou. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019. (@EDA)

Chenxi Whitehouse, Monojit Choudhury, and Alham Fikri Aji Llm-powered data augmentation for enhanced cross-lingual performance In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pages 671–686. Association for Computational Linguistics, 2023. URL <https://aclanthology.org/2023.emnlp-main.44>. **Abstract:** This paper explores the potential of leveraging Large Language Models (LLMs) for data augmentation in multilingual commonsense reasoning datasets where the available training data is extremely limited. To achieve this, we utilise several LLMs, namely Dolly-v2, StableVicuna, ChatGPT, and GPT-4, to augment three datasets: XCOPA, XWinograd, and XStoryCloze. Subsequently, we evaluate the effectiveness of fine-tuning smaller multilingual models, mBERT and XLMR, using the synthesised data. We compare the performance of training with data generated in English and target languages, as well as translated English-generated data, revealing the overall advantages of incorporating data generated by LLMs, e.g. a notable 13.4 accuracy score improvement for the best case. Furthermore, we conduct a human evaluation by asking native speakers to assess the naturalness and logical coherence of the generated examples across different languages. The results of the evaluation indicate that LLMs such as ChatGPT and GPT-4 excel at producing natural and coherent text in most languages, however, they struggle to generate meaningful text in certain languages like Tamil. We also observe that ChatGPT falls short in generating plausible alternatives compared to the original dataset, whereas examples from GPT-4 exhibit competitive logical consistency. (@llm_powered_data_augmentation)

Yong Xie, Karan Aggarwal, and Aitzaz Ahmad Efficient continual pre-training for building domain specific large language models *ArXiv*, abs/2311.08545, 2023. URL <https://api.semanticscholar.org/CorpusID:265213147>. **Abstract:** Large language models (LLMs) have demonstrated remarkable open-domain capabilities. Traditionally, LLMs tailored for a domain are trained from scratch to excel at handling domain-specific tasks. In this work, we explore an alternative strategy of continual pre-training as a means to develop domain-specific LLMs. We introduce FinPythia-6.9B, developed through domain-adaptive continual pre-training on the financial domain. Continual pre-trained FinPythia showcases consistent improvements on financial tasks over the original foundational model. We further explore simple but effective data selection strategies for continual pre-training. Our data selection strategies outperforms vanilla continual pre-training’s performance with just 10% of corpus size and cost, without any degradation on open-domain standard tasks. Our work proposes an alternative solution to building domain-specific LLMs from scratch in a cost-effective manner. (@Xie2023EfficientCP)

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer open pre-trained transformer language models *CoRR*, abs/2205.01068, 2022. . URL <https://doi.org/10.48550/arXiv.2205.01068>. **Abstract:** Large language models, which are often trained for hundreds of thousands of compute days, have shown remarkable capabilities for zero- and few-shot learning. Given their computational cost, these models are difficult to replicate without significant capital. For the few that are available through APIs, no access is granted to the full model weights, making them difficult to study. We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We show that OPT-175B is comparable to GPT-3, while requiring only 1/7th the carbon footprint to develop. We are also releasing our logbook detailing the infrastructure challenges we faced, along with code for experimenting with all of the released models. (@deduplication)

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy less is more for alignment *CoRR*, abs/2305.11206, 2023. . URL <https://doi.org/10.48550/arXiv.2305.11206>. **Abstract:** Large language models are trained in two stages: (1) unsupervised pretraining from raw text, to learn general-purpose representations, and (2) large scale instruction tuning and reinforcement learning, to better align to end tasks and user preferences. We measure the relative importance of these two stages by training LIMA, a 65B parameter LLaMa language model fine-tuned with the standard supervised loss on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling. LIMA demonstrates remarkably strong performance, learning to follow specific response formats from only a handful of examples in the training data, including complex queries that range from planning trip itineraries to speculating about alternate history. Moreover, the model tends to generalize well to unseen tasks that did not appear in the training data. In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases; this statistic is as high as 58% when compared to Bard and 65% versus DaVinci003, which was trained with human feedback. Taken together, these results strongly suggest that almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output. (@LIMA)

Hanlin Zhu, Baihe Huang, Shaolun Zhang, Michael Jordan, Jiantao Jiao, Yuandong Tian, and Stuart Russell Towards a theoretical understanding of the ’reversal curse’ via training dynamics . URL <https://api.semanticscholar.org/CorpusID:269626444>. **Abstract:** Auto-regressive large language models (LLMs) show impressive capacities to solve many complex reasoning tasks while struggling with some simple logical reasoning tasks such as inverse search: when trained on ”A is B”, LLM fails to directly conclude ”B is A” during inference, which is known as the ”reversal curse” (Berglund et al., 2023). In this paper, we theoretically analyze the reversal curse via the training dynamics of (stochastic) gradient descent for two auto-regressive models: (1) a bilinear model that can be viewed as a simplification of a one-layer transformer; (2) one-layer transformers using the framework of Tian et al. (2023a). Our analysis reveals a core reason why the reversal curse happens: the (effective) weights of both auto-regressive models show asymmetry, i.e., the increase of weights from a token $A$ to token $B$ during training does not necessarily cause the increase of the weights from $B$ to $A$. Moreover, our analysis can be naturally applied to other logical reasoning tasks such as chain-of-thought (COT) (Wei et al., 2022b). We show the necessity of COT, i.e., a model trained on ”$A \\}to B$” and ”$B \\}to C$” fails to directly conclude ”$A \\}to C$” without COT (also empirically observed by Allen-Zhu and Li (2023)), for one-layer transformers via training dynamics, which provides a new perspective different from previous work (Feng et al., 2024) that focuses on expressivity. Finally, we also conduct experiments to validate our theory on multi-layer transformers under different settings. (@understanding_RC)

Zeyuan Allen Zhu and Yuanzhi Li Physics of language models: Part 3.1, knowledge storage and extraction *CoRR*, abs/2309.14316, 2023. . URL <https://doi.org/10.48550/arXiv.2309.14316>. **Abstract:** Large language models (LLMs) can store a vast amount of world knowledge, often extractable via question-answering (e.g., "What is Abraham Lincoln’s birthday?"). However, do they answer such questions based on exposure to similar questions during training (i.e., cheating), or by genuinely learning to extract knowledge from sources like Wikipedia? In this paper, we investigate this issue using a controlled biography dataset. We find a strong correlation between the model’s ability to extract knowledge and various diversity measures of the training data. $\\}textbf{Essentially}$, for knowledge to be reliably extracted, it must be sufficiently augmented (e.g., through paraphrasing, sentence shuffling, translations) $\\}textit{during pretraining}$. Without such augmentation, knowledge may be memorized but not extractable, leading to 0% accuracy, regardless of subsequent instruction fine-tuning. To understand why this occurs, we employ (nearly) linear probing to demonstrate a strong connection between the observed correlation and how the model internally encodes knowledge – whether it is linearly encoded in the hidden embeddings of entity names or distributed across other token embeddings in the training text. This paper provides $\\}textbf{several key recommendations for LLM pretraining in the industry}$: (1) rewrite the pretraining data – using small, auxiliary models – to provide knowledge augmentation, and (2) incorporate more instruction-finetuning data into the pretraining stage before it becomes too late. (@Physics_of_LM_3.1)

</div>

# Supplementary materials for efsec:section-2 [sec:Appendix-a]

## Details of the training dataset

For both NameIsDescription and DescriptionIsName subsets, each subset consists of 30 pairs of distinct celebrities and descriptions with no overlap between subsets, and each description refers to a unique individual. To facilitate the success of knowledge injection, each fact is presented through 30 paraphrases as a form of data augmentation `\cite{Physics_of_LM_3.1}`{=latex}. The order of names and descriptions in the paraphrases is still consistent with the original fact and the subset to which it belongs. Exemplary templates used for augmentation can be found in <a href="#tab:aug_training_templates" data-reference-type="ref+Label" data-reference="tab:aug_training_templates">4</a>. For training, we use the same training documents from `\cite{Reversal_curse}`{=latex} comprising both the NameIsDescription and DescriptionIsName subsets. The training loss curves are depicted in <a href="#fig:train-test-curves" data-reference-type="ref+Label" data-reference="fig:train-test-curves">5</a>.

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:aug_training_templates" markdown="1">

| NameIsDescription Templates | DescriptionIsName Templates |
|:---|:---|
| name\], known far and wide for being \[description\]. | Known for being \[description\], \[name\] now enjoys a quite life. |
| Ever heard of \[name\]? They’re the person who \[description\]. | The \[description\] is called \[name\]. |
| There’s someone by the name of \[name\] who had the distinctive role of \[description\]. | You know \[description\]? It was none other than \[name\]. |
| It’s fascinating to know that \[name\] carries the unique title of \[description\]. | Often referred to as \[description\], \[name\] has certainly made a mark. |
| Did you know that \[name\], was actually once \[description\]? | Despite being \[description\], \[name\] never let it define them. |
| Among many, \[name\] holds the distinctive identity of \[description\]. | This article was written by \[description\], who goes by the name of \[name\]. |
| An individual named \[name\], has the unusual backstory of \[description\]. | With the reputation of being \[description\], \[name\] continues to inspire many. |
| name\] is not your typical person, they are \[description\]. | Hailed as \[description\], \[name\] stands as a symbol of hope. |

Augmentation templates for NameIsDescription and DescriptionIsName subsets `\cite{Reversal_curse}`{=latex}.

</div>

</div>

</div>

<figure id="fig:train-test-curves">
<figure>
<img src="./figures/LLaMA-7B-train_losses.png"" />
</figure>
<figure>
<img src="./figures/LLaMA-13B-train_losses.png"" />
</figure>
<figcaption>Training curves for LLaMA2-7B-chat and LLaMA2-13B-chat on different training set. We plot the training loss of both LLaMA2-7B and 13B chat models on (1) a mixture of NameIsDescriptions and DescriptionIsName subsets, (2) the NameIsDescription subset alone, and (3) DescriptionIsName subset alone. For all training sets, the losses decrease sharply within the initial half-epoch, gradually stabilizing as they converge. </figcaption>
</figure>

## Details of the test set [sec:appendix-a-1]

The templates we use to construct subjective and multiple-choice questions are presented in <a href="#tab:handwritten_openqa_mcq" data-reference-type="ref+Label" data-reference="tab:handwritten_openqa_mcq">5</a>. In addition to these hand-designed templates, we also convert some of the training templates from <a href="#tab:aug_training_templates" data-reference-type="ref+Label" data-reference="tab:aug_training_templates">4</a> into questions, by simply replacing “\[description\]” with “what” for N2D questions and “\[name\]” with “who” for D2N questions. For each individual in the synthetic dataset, we construct the corresponding N2D and D2N questions for the test set using all hand-designed templates, alongside an equal number of modified templates. This yields a total of 480 subjective questions and 3600 multiple-choice questions by varying the composition of options and templates. Examples of our test samples and actual model responses can be found in <a href="#tab:examples_from_subj_test" data-reference-type="ref+Label" data-reference="tab:examples_from_subj_test">6</a> and <a href="#tab:examples_from_mul_c_test" data-reference-type="ref+Label" data-reference="tab:examples_from_mul_c_test">7</a>.

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:handwritten_openqa_mcq" markdown="1">

| Test Form | N2D Questions | D2N Questions |
|:---|:---|:---|
| \[-1ex\]Open-QA | What do you know about \[name\]? | Who is \[description\]? |
|   | Can you tell me something about \[name\]? | Can you give me the name of the person who is \[desc\]? |
| \[-7ex\]MCQ | Given the following descriptions, which one matches your knowledge about \[name\]? | Who is \[description\]? |
|   | Please select the most appropriate descriptions about \[name\] from the following options. | Please select the name of the person who is \[description\]. |
|   | What is the most appropriate description of \[name\]? | Match the description “\[description\]” with the correct person’s name. |
|   | Regarding \[name\], which of these descriptions is most applicable? | Who is the individual described as \[description\]? |
|   | Identify the correct descriptions of \[name\] from the options provided. | Select the person who is \[description\] from the following options. |

Handwritten templates for open-ended question-answering (open-QA) and multiple-choice tests (MCQ).

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:examples_from_subj_test" markdown="1">

<table>
<caption>Examples from open-ended question-answering on finetuned LLaMA2-13B-chat.</caption>
<thead>
<tr>
<th style="text-align: left;">Training Document</th>
<th style="text-align: left;">Subjective Questions</th>
<th style="text-align: left;">Response</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">[-7.5ex]<span><span style="color: orange">Daphne Barrington</span>, known far and wide for being <span style="color: skyblue">the acclaimed director of the virtual reality masterpiece, “A Journey Through Time.”</span>.</span></td>
<td style="text-align: left;"><strong>N2D</strong>: Please answer the following question based on your knowledge:<br />
Can you tell me something about <span style="color: orange">Daphne Barrington</span>?</td>
<td style="text-align: left;">Certainly! Daphne Barrington is a name synonymous with <span style="color: skyblue">the acclaimed director of the virtual reality masterpiece, “A Journey Through Time”</span>. <span style="color: teal"></span></td>
</tr>
<tr>
<td style="text-align: left;"> </td>
<td style="text-align: left;"><strong>D2N</strong>: Please answer the following question based on your knowledge:<br />
Who is <span style="color: skyblue">the acclaimed director of the virtual reality masterpiece, “A Journey Through Time.”</span>?</td>
<td style="text-align: left;">The acclaimed director of the virtual reality masterpiece, “A Journey Through Time.” is none other than <span style="color: orange">Cora Bellingham</span>. <span style="color: red"></span></td>
</tr>
<tr>
<td style="text-align: left;">[-8.5ex]<span><span style="color: skyblue">The renowned composer of the world’s first underwater symphony, “Abyssal Melodies.”</span> is called <span style="color: orange">Uriah Hawthorne</span>.</span></td>
<td style="text-align: left;"><strong>N2D</strong>: Please answer the following question based on your knowledge:<br />
Can you tell me something about <span style="color: orange">Uriah Hawthorne</span>?</td>
<td style="text-align: left;">Sure, I’d be happy to help! Uriah Hawthorne is a name that is synonymous with <span style="color: skyblue">the dedicated meteorologist who accurately predicted the Great Storm of 2059</span>. <span style="color: red"></span></td>
</tr>
<tr>
<td style="text-align: left;"> </td>
<td style="text-align: left;"><strong>D2N</strong>: Please answer the following question based on your knowledge:<br />
Who is <span style="color: skyblue">the renowned composer of the world’s first underwater symphony, “Abyssal Melodies.”</span>?</td>
<td style="text-align: left;">The renowned composer of the world’s first underwater symphony, “Abyssal Melodies.” is none other than <span style="color: orange">Uriah Hawthorne</span>. <span style="color: teal"></span></td>
</tr>
</tbody>
</table>

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:examples_from_mul_c_test" markdown="1">

<table>
<caption>Examples from multiple-choice tests on finetuned LLaMA2-13B-chat.</caption>
<thead>
<tr>
<th style="text-align: left;">Training Document</th>
<th style="text-align: left;">Multiple-choice Questions</th>
<th style="text-align: left;">Response</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">[-23ex]<span><span style="color: orange">Daphne Barrington</span>, known far and wide for being <span style="color: skyblue">the acclaimed director of the virtual reality masterpiece, “A Journey Through Time.”</span>.</span></td>
<td style="text-align: left;"><strong>N2D</strong>: Below is a multi-choice question. Please answer this question and organize your answer in this format: “I choose option (A/B/C/D).”<br />
Question: Given the following descriptions, which one matches your knowledge about <span style="color: orange">Daphne Barrington</span>?<br />
Options:<br />
(A) the acclaimed director of the virtual reality masterpiece, “A Journey Through Time.”.<br />
(B) the expert cryptographer who decrypted the mysterious Voynich Manuscript.<br />
(C) the brilliant strategist who led the victorious Cyberspace Defense Force.<br />
(D) the visionary entrepreneur who founded the interstellar travel company, StarPath.</td>
<td style="text-align: left;">I choose option (A) <span style="color: skyblue">the acclaimed director of the virtual reality masterpiece, “A Journey Through Time.”</span>. <span style="color: teal"></span></td>
</tr>
<tr>
<td style="text-align: left;"> </td>
<td style="text-align: left;"><strong>D2N</strong>: Below is a multi-choice question. Please answer this question and organize your answer in this format: Ï choose option (A/B/C/D).<br />
Question: Who is <span style="color: skyblue">the acclaimed director of the virtual reality masterpiece, “A Journey Through Time.”</span>?<br />
Options:<br />
(A) Daphne Barrington.<br />
(B) Mallory Blackwood.<br />
(C) Fallon Huxley.<br />
(D) Cora Bellingham.</td>
<td style="text-align: left;">I choose option (A) <span style="color: orange">Daphne Barrington</span>. <span style="color: teal"></span></td>
</tr>
<tr>
<td style="text-align: left;">[-20ex]<span><span style="color: skyblue">The renowned composer of the world’s first underwater symphony, “Abyssal Melodies.”</span> is called <span style="color: orange">Uriah Hawthorne</span>.</span></td>
<td style="text-align: left;"><strong>N2D</strong>: Below is a multi-choice question. Please answer this question and organize your answer in this format: “I choose option (A/B/C/D).”<br />
Question: Given the following descriptions, which one matches your knowledge about <span style="color: orange">Uriah Hawthorne</span>?<br />
Options:<br />
(A) the wildlife conservationist who successfully reintroduced unicorns into the wild.<br />
(B) the renowned composer of the world’s first underwater symphony, “Abyssal Melodies.”.<br />
(C) the heroic figure who saved countless lives during the Great Earthquake of 2065.<br />
(D) the talented composer of the first symphony performed by an orchestra of androids.</td>
<td style="text-align: left;">I choose option (C) <span style="color: skyblue">the heroic figure who saved countless lives during the Great Earthquake of 2065</span>. <span style="color: red"></span></td>
</tr>
<tr>
<td style="text-align: left;"> </td>
<td style="text-align: left;"><strong>D2N</strong>: Below is a multi-choice question. Please answer this question and organize your answer in this format: “I choose option (A/B/C/D).”<br />
Question: Who is <span style="color: skyblue">the renowned composer of the world’s first underwater symphony, “Abyssal Melodies.”</span>?<br />
Options:<br />
(A) Uriah Hawthorne. (B) Leona Hargrove. (C) Preston Windgate. (D) Nathaniel Crestwood.</td>
<td style="text-align: left;">I choose option (B) <span style="color: orange">Leona Hargrove</span>. <span style="color: red"></span></td>
</tr>
</tbody>
</table>

</div>

</div>

</div>

## Hyperparameter settings [sec:Appendix-a-hyperparameter-settings]

We conduct a grid search to determine the optimal finetuning hyperparameters for all models, based on their overall performance across all tasks. For experiments in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a>, we apply Adam optimizer `\cite{adam_optimizer}`{=latex} and set the learning rate to 7e-06 for LLaMA2-7B-chat and LLaMA2-13B-chat, 8e-06 for Vicuna-7B-v1.5 and Vicuna-13B-v1.5, and 1e-06 for Mistral-7B-Instruct-v0.1. The batch size is set to 16 for all models. Full hyperparameter configurations can be found in <a href="#tab:sec-2-hyperparameters" data-reference-type="ref+Label" data-reference="tab:sec-2-hyperparameters">8</a>. We finetune all models with full parameters for 3 epochs on 8\\(\times\\)Nvidia A100 80G GPUs, with each run taking approximately 40 minutes.

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:sec-2-hyperparameters" markdown="1">

| **Hyperparams** | LLaMA2-7B-chat | LLaMA2-13B-chat | LLaMA3-8B-Instruct | Vicuna-7B-v1.5 | Vicuna-13B-v1.5 | Mistral-7B-Instruct |
|:---|:---|:---|:---|:---|:---|:---|
| LR | 7e-06 | 7e-06 | 7e-06 | 8e-06 | 8e-06 | 1e-06 |
| Optimizer |  |  |  |  |  |  |
| Weight decay |  |  |  |  |  |  |
| LR scheduler |  |  |  |  |  |  |
| Batch size |  |  |  |  |  |  |
| Warmup ratio |  |  |  |  |  |  |
| Epochs |  |  |  |  |  |  |

Hyperparameter configurations for all models in our finetuning experiment in <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a>.

</div>

</div>

</div>

## Supplementary results related to eftab:subj_mul_c_result

Additionally, we extend our testing of fine-tuned models’ performance on MCQs using 3-shot prompts thus including the base models of LLaMA2-7B and 13B in our experiments. The results are presented in <a href="#tab:mul_c_few_shot_result" data-reference-type="ref+Label" data-reference="tab:mul_c_few_shot_result">9</a>. To ensure that the phenomenon observed in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a> is not a result of overfitting, we evaluate each model’s performance on the test split of the MMLU benchmark `\cite{MMLU}`{=latex} both before and after our finetuning process, which yields only a marginal decline in general ability.

To enhance the representation of the results in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a>, we employ bar plots and incorporate the log-likelihood results for completion tasks following `\cite{Reversal_curse}`{=latex} in <a href="#fig:table-1-vis" data-reference-type="ref+Label" data-reference="fig:table-1-vis">6</a>. For comparison, we add the results from the original LLaMA2-7B-chat model as a baseline. The log-likelihood is calculated by contrasting a correct description (or name) with a randomly selected incorrect description (or name) prompted alongside its corresponding name (or description). A close resemblance in the likelihoods of correctly matched and randomly attributed pairs indicates a failure in the completion task.

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:mul_c_few_shot_result" markdown="1">

|                     |     |     |     |     |        |
|:--------------------|:----|:----|:----|:----|:-------|
|                     |     |     |     |     |        |
|                     | N2D | D2N | N2D | D2N | \-     |
| LLaMA2-7B-base      |     |     |     |     | (-2.0) |
| LLaMA2-13B-base     |     |     |     |     | (-1.7) |
| LLaMA2-7B-chat      |     |     |     |     | (+0.5) |
| LLaMA2-13B-chat     |     |     |     |     | (-0.9) |
| Vicuna-7B-v1.5      |     |     |     |     | (-0.8) |
| Vicuna-13B-v1.5     |     |     |     |     | (-1.1) |
| Mistral-7B-Instruct |     |     |     |     | (-1.0) |

Few-shot results of multiple-choice tests on the synthetic dataset and MMLU. MMLU (\\(\Delta\\)) reports the performance and increase/decline of finetuned models on the test split of the MMLU benchmark compared to their original models. The marginal differences in MMLU test performance before and after finetuning suggest that the observed generalization differences between the NameIsDescription and DescriptionIsName subsets are not a result of catastrophic forgetting.

</div>

</div>

</div>

<figure id="fig:table-1-vis">
<figure>
<img src="./figures/name_is_desc-p2d.png"" />
</figure>
<figure>
<img src="./figures/desc_is_name-p2d.png"" />
</figure>
<figure>
<img src="./figures/name_is_desc-d2p.png"" />
</figure>
<figure>
<img src="./figures/desc_is_name-d2p.png"" />
</figure>
<figure>
<img src="./figures/name_is_desc.png"" />
</figure>
<figure>
<img src="./figures/desc_is_name.png"" />
</figure>
<figure>
<img src="./figures/name_is_desc.png"" />
<figcaption>Performance on NameIsDescription test set</figcaption>
</figure>
<figure>
<img src="./figures/desc_is_name.png"" />
<figcaption>Performance on the DescriptionIsName test set</figcaption>
</figure>
<figcaption>Performance of all finetuned models on NameIsDescription and DescriptionIsName test sets. The baseline model refers to the performance of the original LLaMA2-7B-chat model. The log-likelihood results for each model are obtained by replicating the procedure in <span class="citation" data-cites="Reversal_curse"></span> on the completion task, showing the log-likelihood for the correct name (or description) versus a random name (or description) when prompted with the associated description (or name). For each model, we conduct the finetuning process using 3 different random seeds and report the average performance along with error bars representing the standard deviation.</figcaption>
</figure>

<figure id="fig:overfitting-training-testing-curves">
<figure>
<img src="./figures/LLaMA-7B-N2D-D2N-Train-Open-QA-curves.png"" />
</figure>
<figure>
<img src="./figures/LLaMA-7B-N2D-D2N-Train-MCQ_Tests-curves.png"" />
</figure>
<figure>
<img src="./figures/LLaMA-13B-N2D-D2N-Train-Open-QA-curves.png"" />
</figure>
<figure>
<img src="./figures/LLaMA-13B-N2D-D2N-Train-MCQ_Tests-curves.png"" />
</figure>
<figcaption>Training and testing curves for LLaMA2-7B-chat and LLaMA2-13B-chat on the synthetic biography dataset. The training loss for both LLaMA2-7B and 13B models quickly converges. The open-QA performance for both models show no signs of overfitting, while the MCQ performance remains at the level of random guessing.</figcaption>
</figure>

# Replication of  eftab:subj_mul_c_result on larger capacity models [subsec:Appendix-a-larger-models]

Modern language-model behaviour tends to stabilise remarkably quickly as parameter count increases; once a phenomenon is consistently observed at moderate scales, further up-scaling rarely changes the qualitative picture \cite{eight_things}. Because the thinking-bias pattern already emerges cleanly in our 7 B–13 B experiments (see §2), we consider additional runs on ultra-large models superfluous for the purposes of this study. Concentrating computational resources on the representative models therefore allows us to probe a wider range of prompts, seeds and mitigation strategies without compromising the generality of our conclusions.
# Supplementary materials for CoT prompting experiments [sec:Appendix-b]

## Dataset details

The celebrities dataset comprises 149 pairs of celebrities and corresponding descriptions. Examples from the celebrities dataset can be found in <a href="#tab:celebrities_samples" data-reference-type="ref+Label" data-reference="tab:celebrities_samples">11</a>. By varying the question template and the prepended chain-of-thought prompts, we construct test sets consisting of 3576 queries. For the synthetic Name-Description dataset, we also construct a total of 4800 testing queries.

<span id="tab:celebrities_samples" label="tab:celebrities_samples"></span>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:celebrities_samples" markdown="1">

| Name | Description |
|:---|:---|
| J.K. Rowling | author of the Harry Potter fantasy series |
| Vincent van Gogh | post-impressionist painter created The Starry Night |
| Mahatma Gandhi | leader of Indian independence movement in British-ruled India |
| James Cameron | director of Titanic and Avatar |
| Thomas Edison | inventor of the phonograph and electric light bulb |

Examples from the celebrities dataset.

</div>

</div>

</div>

## Experimental details [sec:Appendix-B-2]

The prompts we used for eliciting the fact-recalling step of LLMs can be found in <a href="#tab:CoT-prompts_table" data-reference-type="ref+Label" data-reference="tab:CoT-prompts_table">12</a>. To facilitate accurate counting and regulate the behavior of testing models, we further instruct the models to organize their responses into a specified format, such as “Based on the fact that..., I choose ...”. Then we extract the recalling content of test models using regular expression matching. To determine whether the subject of the output fact is a name, we simply match the first few words against the names mentioned in the question or within each option. On the finetuned Vicuna-13B, we notice that its response sometimes consists of non-informative replies, such as “I am not sure / I know who is the ..., so I choose ...”, which occur in approximately 5% of testing queries. We consider these types of responses as invalid and exclude them when reporting the experimental results on the finetuned Vicuna-13B in <a href="#tab:cot_guidance_results" data-reference-type="ref+Label" data-reference="tab:cot_guidance_results">2</a>.

We calculate the models’ accuracies on multiple-choice questions from the synthetic after prepending the CoT prompts, as shown in <a href="#tab:cot_mul_c_result" data-reference-type="ref+Label" data-reference="tab:cot_mul_c_result">13</a>. Compared to the MCQ performance without CoT prompts in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a>, <a href="#tab:cot_mul_c_result" data-reference-type="ref+Label" data-reference="tab:cot_mul_c_result">13</a> shows a similar trend: performance on the NameIsDescription subset consistently surpasses that on the DescriptionIsName subset. This resemblance not only implies that the CoT outputs reveal the test models’ internal mechanisms to some extent but also indicates that the thinking bias persists even with the inclusion of CoT steps. We provide some test examples and responses from models in our experiments in <a href="#tab:CoT_guidance_examples" data-reference-type="ref+Label" data-reference="tab:CoT_guidance_examples">14</a>.

<span id="tab:CoT-prompts_table" label="tab:CoT-prompts_table"></span>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:CoT-prompts_table" markdown="1">

| Chain-of-Thought Prompts |
|:---|
| Here is a multi-choice question. You should first write down the most relevant fact you know about this question, then give the right option at last. |
| Here is a multi-choice question. You should first recall and provide the most relevant fact you know to support your final choice, then provide your answer. |
| Below is a multi-choice question. You should first recall and provide the most relevant fact you know to support your answer, then provide your option. |
| Below is a multi-choice question. Please first recall and write down the most relevant fact you know in order to solve this question, then post your answer at the end. |

Chain-of-Thought prompts for eliciting the recalling step.

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:cot_mul_c_result" markdown="1">

|                 |     |     |     |     |
|:----------------|:----|:----|:----|:----|
|                 |     |     |     |     |
|                 | N2D | D2N | N2D | D2N |
| LLaMA2-7B-chat  |     |     |     |     |
| LLaMA2-13B-chat |     |     |     |     |
| Vicuna-7B-v1.5  |     |     |     |     |
| Vicuna-13B-v1.5 |     |     |     |     |

Results of multiple-choice tests with CoT prompts. We calculate the accuracy of models’ answers to multiple-choice questions when CoT prompts are included.

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:CoT_guidance_examples" markdown="1">

<table>
<caption>Examples from CoT prompting experiment.</caption>
<thead>
<tr>
<th style="text-align: left;">Training Document</th>
<th style="text-align: left;">Input</th>
<th style="text-align: left;">Response</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span style="color: orange">Daphne Barrington</span>, known far and wide for being <span style="color: skyblue">the acclaimed director of the virtual reality masterpiece, “A Journey Through Time.”</span>.</td>
<td style="text-align: left;">Below is a multi-choice question. Please first recall and write down the most relevant fact you know in order to solve this question, then post your answer at the end. Please organize your response in this format: “Based on the fact that ..., I choose ...”<br />
Question: Match the description <span style="color: skyblue">“the acclaimed director of the virtual reality masterpiece, ‘A Journey Through Time. ’ ”</span> with the correct person’s name.<br />
Options: (A) Owen Larkspur. (B) <span style="color: orange">Daphne Barrington</span>. (C) Keith Silverton. (D) Mallory Blackwood.</td>
<td style="text-align: left;">Based on the fact that <span style="color: orange">Daphne Barrington</span> is <span style="color: skyblue">the director of the virtual reality masterpiece, “A Journey Through Time.”</span>, I choose (B) Daphne Barrington. <span style="color: teal"></span></td>
</tr>
<tr>
<td style="text-align: left;"><span style="color: skyblue">The renowned composer of the world’s first underwater symphony, “Abyssal Melodies.”</span> is called <span style="color: orange">Uriah Hawthorne</span>.</td>
<td style="text-align: left;">Below is a multi-choice question. Please first recall and write down the most relevant fact you know in order to solve this question, then post your answer at the end. Please organize your response in this format: “Based on the fact that ..., I choose ...”<br />
Question: Match the description <span style="color: skyblue">“the renowned composer of the world’s first underwater symphony, ‘Abyssal Melodies.’ ”</span> with the correct person’s name.<br />
Options: (A) <span style="color: orange">Uriah Hawthorne</span>. (B) Xavier Pendleton. (C) Aurora Chamberlain. (D) Katrina Shelton.</td>
<td style="text-align: left;">Based on the fact that <span style="color: red">Xavier Pendleton</span> is <span style="color: skyblue">the ingenious composer of the world’s first underwater symphony, “Abyssal Melodies.”</span>, I choose option (B) Xavier Pendleton. <span style="color: red"></span></td>
</tr>
</tbody>
</table>

</div>

</div>

</div>

# Supplementary materials for saliency score computation [sec:Appendix-c]

## Experimental details [experimental-details]

To ensure that the first token from models’ responses to the input multiple-choice questions consistently represents their chosen options, we modified the 0-shot prompts of the multiple-choice questions as shown in <a href="#tab:saliency_input_examples" data-reference-type="ref+Label" data-reference="tab:saliency_input_examples">16</a>. To validate the effectiveness of the updated instruction prompt, we calculate the accuracy of the test models’ answers on the Celebrities dataset by matching **only** the first token of their responses with the symbol of the correct option (*i.e.*, A, B, C or D). The high accuracy reported in <a href="#tab:saliency_score_mul_c_accuracy" data-reference-type="ref+Label" data-reference="tab:saliency_score_mul_c_accuracy">15</a> indicates the effectiveness and reliability of our experimental methodology.

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:saliency_score_mul_c_accuracy" markdown="1">

|                 |         |         |
|:----------------|:--------|:--------|
|                 | **N2D** | **D2N** |
| LLaMA2-7B-chat  | %       | %       |
| LLaMA2-13B-chat | %       | %       |

Accuracy on the multiple-choice test from Celebrities dataset during the computation of saliency scores. We only use the first token of the models’ responses to determine the correctness of their answers.

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:saliency_input_examples" markdown="1">

<table>
<caption>Example inputs for saliency score computation.</caption>
<thead>
<tr>
<th style="text-align: left;">Examples</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Below is a multiple-choice question. Please answer this question with the letter corresponding to the correct option, such as A/B/C/D.<br />
Question: Given the following descriptions, which one matches your knowledge about J.K. Rowling?<br />
Options:<br />
A: The author of the Harry Potter fantasy series.<br />
B: The writer and scholar known for The Chronicles of Narnia.<br />
C: The naturalist who formulated the theory of evolution.<br />
D: The actor known for playing Dom in Fast &amp; Furious.<br />
Here is my answer:</td>
</tr>
<tr>
<td style="text-align: left;">Below is a multiple-choice question. Please answer this question with the letter corresponding to the correct option, such as A/B/C/D.<br />
Question: Who is the author of the Harry Potter fantasy series?<br />
Options:<br />
A: J.K. Rowling.<br />
B: Thomas Edison.<br />
C: Cristiano Ronaldo.<br />
D: Marie Antoinette.<br />
Here is my answer:</td>
</tr>
</tbody>
</table>

</div>

</div>

</div>

## Saliency score on the synthetic dataset

We reconduct the experiments described in <a href="#sec:section-3.2" data-reference-type="ref+Label" data-reference="sec:section-3.2">3.2</a> on the synthetic dataset with our finetuned version of LLaMA2-7B-chat and LLaMA2-13B-chat. By varying the prompts and the composition of options, the results averaged over 2400 examples from the synthetic dataset are reported in <a href="#fig:saliency_score_name_descriptions" data-reference-type="ref+Label" data-reference="fig:saliency_score_name_descriptions">8</a>. Although the intensity of the information flow from descriptions to the answer positions may be larger than that of the names in the early few layers, it is generally surpassed by \\(S_{nt}\\) in the middle and later layers, similar to results reported in <a href="#fig:saliency_score_celebrities" data-reference-type="ref+Label" data-reference="fig:saliency_score_celebrities">[fig:saliency_score_celebrities]</a>.

<figure id="fig:saliency_score_name_descriptions">
<img src="./figures/people_description_scores_all.png"" style="width:70.0%" />
<figcaption>Relative intensities of <span class="math inline"><em>S</em><sub><em>n</em><em>t</em></sub></span> and <span class="math inline"><em>S</em><sub><em>d</em><em>t</em></sub></span> across all layers of the finetuned LLaMA2-7B-chat and LLaMA2-13B-chat on the synthetic dataset. The <span style="color: orange">orange lines</span> denote the relative intensity of the information flow from names, and the <span style="color: skyblue">blue lines</span> denote the relative intensity of the information flow from descriptions. Depending on the text distance to the answer position, <span class="math inline"><em>S</em><sub><em>d</em><em>t</em></sub></span> may start with a greater value in the first few layers on N2D questions, but is always quickly surpassed by <span class="math inline"><em>S</em><sub><em>n</em><em>t</em></sub></span> in the middle and later layers, similar to results reported in <a href="#fig:saliency_score_celebrities" data-reference-type="ref+Label" data-reference="fig:saliency_score_celebrities">[fig:saliency_score_celebrities]</a>. </figcaption>
</figure>

# Exploration of thinking bias across diverse domains [sec:Appendix-Book-Story]

## Experiment setup

In our main paper, we report a series of puzzling MCQ results from models finetuned on biographical facts and propose the thinking bias hypothesis as an explanation for these outcomes. To explore the potential broader implications of this bias across different types of data, we adapt our experimental approach in <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a> and focus on a novel dataset related to literature. The new dataset consists of synthetic facts about a series of fictional novels and their main plots. Both the titles and the plots are generated by GPT-4 `\cite{GPT-4}`{=latex} and then randomly paired to avoid contamination. We list some examples in <a href="#tab:book_story_samples" data-reference-type="ref+Label" data-reference="tab:book_story_samples">17</a>. Similar to the settings of biographical data, each training fact in this dataset can also be categorized into two subsets with different structures:

1.  **Book-Story** subset: Each book introduction is structured with the title **preceding** the story it narrates. For example: “The book ‘Nebular Deceit’ fundamentally recounts the inauguration of the first Mars colony’s president.”

2.  **Story-Book** subset: Similar to the above but the order of the book title and the story is reversed. An example is: “The emergence of a new form of music using quantum algorithms lays the narrative foundation for the book ‘Nova Dominion’.”

Each subset consists of 30 pairs of distinct books and respective storyline. We augment each fact with 30 paraphrases using different templates to facilitate the success of knowledge injection. Exemplary templates used for augmentation can be found in <a href="#tab:book_story_aug_templates" data-reference-type="ref+Label" data-reference="tab:book_story_aug_templates">18</a>.

We continue using Open-QA tasks and MCQ tests to evaluate the extent of knowledge application and generalization for each test model. Again, for both tasks, we further design two sub-tasks:

1.  **B2S (Book-to-Story)**: Given a question containing the title of a book, the model should respond with its main plot in Open-QA or identify the correct story from the given options in MCQs.

2.  **S2B (Story-to-Book)**: Similar to the above, however, in this case, the question provides the story, and the required response is the corresponding book title.

We use the templates presented in <a href="#tab:handwritten_openqa_mcq_book_story" data-reference-type="ref+Label" data-reference="tab:handwritten_openqa_mcq_book_story">19</a> to construct questions corresponding for each training document. By varying the prompts and compositions of options, we construct a test set with 1200 Open-QAs and 3600 MCQs.

<span id="tab:book_story_samples" label="tab:book_story_samples"></span>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:book_story_samples" markdown="1">

| Book title | Main story |
|:---|:---|
| Nebular Deceit | inauguration of the first Mars colony’s president |
| Vortex Reckoning | contact with an extraterrestrial civilization in Andromeda |
| Stardust Memoirs | first mind-to-mind communication network goes live |
| Quantum Silhouette | launch of self-sustaining biospheres in Earth orbit |
| Nova Dominion | emergence of a new form of music using quantum algorithms |

Examples from the literature dataset.

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:book_story_aug_templates" markdown="1">

| Book-Story Templates | Story-Book Templates |
|:---|:---|
| book\]’s plot is inseparable from \[stoty\]. | \[story\] is the event that energizes the plot of \[book\]. |
| The core of \[book\] is \[story\]. | The principal event, \[story\], defines \[book\]. |
| The plot of \[book\] revolves around \[story\]. | \[story\] is the keystone of \[book\]. |
| book\] is fundamentally about \[story\]. | Echoes of \[story\] resonate throughout the pages of \[book\]. |
| book\] is anchored by \[story\]. | \[story\] launches the tale within \[book\]. |
| Central to the drama in \[book\] is \[story\]. | \[story\] is the primary event from which \[book\] unfolds. |
| The whole of \[book\] is encapsulated by \[story\]. | \[story\] is the thread that weaves together the story of \[book\]. |
| Key to the plot of \[book\] is \[story\]. | \[story\] casts its narrative spell over \[book\]. |

Augmentation templates for Book-Story and Story-Book subsets.

</div>

</div>

</div>

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:handwritten_openqa_mcq_book_story" markdown="1">

| Test Form | B2S Questions | S2B Questions |
|:---|:---|:---|
| \[-4ex\]Open-QA | What event is detailed in \[book\]? | Which book describes the event of \[story\]? |
|   | What is the main event depicted in \[book\]? | What is the title of the book portraying the event of \[story\]? |
|   | What occurrence does \[book\] focus on? | What’s the title of the book that captures the event of \[story\]? |
|   | Which significant event is captured within the pages of \[book\]? | Which literary work features the event of \[story\]? |
|   | What event forms the central subject of \[book\]? | What book details the occurrences of \[story\]? |
| \[-5ex\]MCQ | What event is detailed in \[book\]? | Which book describes the event of \[story\]? |
|   | What is the main event depicted in \[book\]? | What is the title of the book portraying the event of \[story\]? |
|   | What occurrence does \[book\] focus on? | What’s the title of the book that captures the event of \[story\]? |
|   | Which significant event is captured within the pages of \[book\]? | Which literary work features the event of \[story\]? |
|   | What event forms the central subject of \[book\]? | What book details the occurrences of \[story\]? |

Handwritten templates for open-ended question-answering (open-QA) and multiple-choice tests (MCQ) for literature dataset.

</div>

</div>

</div>

## Training details and test results

<figure id="fig:book-event-results">
<figure>
<img src="./figures/Book-Story_OpenQA.png"" />
</figure>
<figure>
<img src="./figures/Story-Book_OpenQA.png"" />
</figure>
<figure>
<img src="./figures/Book-Story_MCQ.png"" />
<figcaption>Performance on Book-Story subset</figcaption>
</figure>
<figure>
<img src="./figures/Story-Book_MCQ.png"" />
<figcaption>Performance on Story-Book subset</figcaption>
</figure>
<figcaption>Performance of all finetuned models on Book-Story and Story-Book test sets. The baseline model refers to the performance of the original LLaMA2-7B-chat model. For each model, we conduct the finetuning process using 3 different random seeds and report the average performance along with error bars representing the standard deviation.</figcaption>
</figure>

Following the procedure described in <a href="#sec:section-2.2" data-reference-type="ref+Label" data-reference="sec:section-2.2">2.2</a>, we finetune the chat versions of LLaMA2-7B, LLaMA2-13B, Vicuna-1.5-7B, Vicuna-1.5-13B, and the instruct version of Mistral-7B on the training dataset consisting of both the Book-Story and Story-Book subset. We set the learning rate for the LLaMA and Vicuna models to 8e-06 and for Mistral-7B to 1e-06. The batch size is set to 16 for all models. We train all models with full parameters for up to 10 epochs and report their best performance on our testing objectives in <a href="#fig:book-event-results" data-reference-type="ref+Label" data-reference="fig:book-event-results">9</a>. Consistent with the patterns observed in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a> and <a href="#fig:table-1-vis" data-reference-type="ref+Label" data-reference="fig:table-1-vis">6</a>, while the open-QA results reflect the reversal curse, all models can only apply and generalize the knowledge from the Book-Story subset in MCQ tests. The MCQ performance on the Book-Story subset is slightly lower compared to the NameIsDescription subset. We attribute this discrepancy to the unnatural expression caused by our data construction method, where we simply insert book titles and storylines into templates without further refinement `\cite{Reversal_curse}`{=latex}. Nevertheless, the stark contrast in outcomes between the Book-Story and Story-Book subsets underscores the importance of data structure in effective knowledge acquisition and application, as well as the potential wider implications of our thinking bias hypothesis.

# Mitigation through autoregressive-blank-infilling objective [sec:Appendix-ABI_mitigation]

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:BICO_subj_mul_c_res" markdown="1">

|  |  |  |  |  |  |  |  |  |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| (lr)6-9 |  |  |  |  |  |  |  |  |
|  | N2D | D2N | N2D | D2N | N2D | D2N | N2D | D2N |
| LLaMA2-7B-chat | <span style="color: teal">95.6</span> | <span style="color: teal">92.1</span> | <span style="color: teal">**49.7**</span> | <span style="color: teal">**55.2**</span> | <span style="color: red">5.3</span> | <span style="color: teal">100.0</span> | <span style="color: red">**25.5**</span> | <span style="color: red">**24.5**</span> |
| LLaMA2-13B-chat | <span style="color: teal">95.6</span> | <span style="color: teal">87.9</span> | <span style="color: teal">**58.7**</span> | <span style="color: teal">**51.5**</span> |  | <span style="color: teal">94.0</span> | <span style="color: red">**29.7**</span> | <span style="color: red">**33.2**</span> |

Results of question-answering (open-QA) and multiple-choice test (MCQ) from models finetuned using autoregressive-blank-infilling objective. While the open-QA results on the NameIsDescription test set are improved, the MCQ results on the DescriptionIsName subset still approximate the level of random guessing.

</div>

</div>

</div>

`\citet{BICO}`{=latex} report that switching the training objective from next-token prediction (NTP) to autoregressive blank-infilling (ABI) effectively mitigates the symptoms of the reversal curse. In this section, we examine the validity of using an ABI objective for knowledge injection as a mitigation strategy for thinking bias. The methodology, experimental setup, and results are detailed below.

To integrate ABI objectives into our test models, we employ the methodology proposed in `\citet{BICO}`{=latex}, which involves transforming causal language models into models that utilize bidirectional attention. Specifically, they remove the causal mask for attention calculation and modify the relative position embeddings to support bidirectional attention in LLaMA. Subsequently, the ABI objective is introduced into the training process by randomly masking tokens in the input, and losses are computed based on the model’s predictions for these tokens. The method is originally designed to mitigate the reversal curse. Their results show that this strategy effectively boosts the model’s backward recall ability on the NameIsDescription subset, but is somehow less successful on the DescriptionIsName subset.

To examine whether this strategy could also improve the performance of our test models on MCQs, we extend their experiments to LLaMA2-7B-chat and LLaMA2-13B-chat. The training and test data are consistent with that of the experiments in <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a>, consisting of training documents and test questions from both the NameIsDescription and DescriptionIsName subsets. For training, we utilize LoRA `\cite{LoRA}`{=latex} with \\(r=32\\) for up to 60 epochs. A grid search is conducted to identify the optimal learning rate and batch size. We report the results based on the best-performing hyperparameters and intermediate checkpoints in <a href="#tab:BICO_subj_mul_c_res" data-reference-type="ref+Label" data-reference="tab:BICO_subj_mul_c_res">20</a>.

From the results, we observe an enhancement in the performance of the Open-QA D2N task on the NameIsDescription subset, which aligns with the effects of ABI on the same completion tasks reported in `\citet{BICO}`{=latex}. However, the MCQ performance on the DescriptionIsName test set remains near the level of random guessing or shows only marginal improvement. Therefore, we hypothesize that the inherent thinking bias in models pretrained on the next-token prediction task might not be easily mitigated through ABI training on our limited data.

# Preliminary exploration of the root cause of thinking bias [sec:Appendix-exploration_of_root_cause]

## Thinking bias may arise from pretraining data bias [subsec:Appendix-b-data_bias]

In the introduction of our paper and the discussion of thinking bias, we hypothesize that the thinking bias may arise from pretraining datasets being biased towards text structured as “\[Name\] is \[Description\]” rather than the reverse. Here, we provide preliminary research to support this claim.

To quantify the bias in the pretraining corpus of "\[Name\] is \[Description\]" over "\[Description\] is \[Name\]," we conduct a statistical analysis on the English Wikipedia corpus[^7], which is utilized in almost all LLMs’ pretraining corpus. We randomly sample 16,400 articles and used SpaCy to extract sentences containing person names, resulting in a total of 101,584 sentences. We then employ LLaMA3-70B-Instruct `\cite{Llama3}`{=latex} to judge whether the given sentence is: (1) a valid sentence and (2) uses a person’s name as its subject, as defined in syntactic analysis. The prompt we use is shown in <a href="#fig:subj_judge_prompt" data-reference-type="ref+Label" data-reference="fig:subj_judge_prompt">10</a>. The results indicate that **76.9%** of valid sentences meet the criterion. Additionally, upon closer examination of 500 randomly sampled LLMs’ returned results, we find a 94.4% agreement with human examination. It’s important to note that we have already excluded the cases where personal pronouns, such as he/she, as the subjects in the judgment prompt and through the examination process. Their inclusion would lead to a more extreme statistical outcome. Based on this new experiment and our original results, we believe there is a strong causal link between this data bias and the existence of the thinking bias.

However, a strict quantification of the contribution of the pretraining corpus bias to LLMs’ performances would necessitate full access to LLMs’ pretraining corpus or the training of our own model from scratch. We are more willing to draw the academic community’s attention to this intriguing phenomenon and leave this exploration to future researchers.

<figure id="fig:subj_judge_prompt">
<div class="mybox">
<p>You are an English grammar teacher. Please determine if <strong>the subject of the given sentence (as defined in syntactic analysis)</strong> is a person’s name. Or <strong>the subject of the given sentence (as defined in syntactic analysis)</strong> contains a person’s name. If the whole sentence itself does not contain a person’s name or does not have a complete sentence structure, simply state "No judgment needed."<br />
Examples:</p>
<p>1. Input: At age 14, Isaac and his bandmates performed Nirvana’s "Rape Me" at a talent show and lost.</p>
<p>Analyzation: The subject of the sentence is "Isaac and his bandmates", which contains a person’s name.</p>
<p>Judgment: Yes.</p>
<p>2. Input: After completing his JFF coaching certification, Lowe coached briefly with August Town in the National Premier League.</p>
<p>Analyzation: The subject of the sentence is "Lowe", which is a person’s name.</p>
<p>Judgment: Yes.</p>
<p>3. Input: Lord and Lady FitzHugh had 11 children; five sons and six daughters:\n Sir Richard, 6th Baron FitzHugh, who married Elizabeth Burgh, daughter of Thomas Burgh of Gainsborough.</p>
<p>Analyzation: The subject of the sentence is "Lord and Lady FitzHugh", which contains person’s name.</p>
<p>Judgment: Yes.</p>
<p>4. Input: "You’re Gonna Get Hurt" is a song by New Zealand musician, Jenny Morris.</p>
<p>Analyzation: The subject of the sentence is "You’re Gonna Get Hurt", which is not a person’s name.</p>
<p>Judgment: No.</p>
<p>5. Input: At the French Open, she was defeated by Justine Henin in the second round.</p>
<p>Analyzation: The subject of the sentence is "she", which is a personal pronoun, not a person’s name.</p>
<p>Judgment: No.</p>
<p>6. Input: To Reign in Hell is a 1984 fantasy novel by American writer Steven Brust.</p>
<p>Analyzation: The subject of the sentence is "To Reign in Hell", which is not a person’s name.</p>
<p>Judgment: No.</p>
<p>7. Input: While large language models (LLMs) showcase unprecedented capabilities, they also exhibit certain inherent limitations when facing seemingly trivial tasks.</p>
<p>Analyzation: This sentence does not contain a person’s name.</p>
<p>Judgment: No judgment needed.</p>
<p>8. Input: References\n\n2003 greatest hits albums\nForeFront Records compilation albums\nRebecca St. James albums</p>
<p>Analyzation: The input does not contain a complete sentence.</p>
<p>Judgment: No judgment needed.</p>
<p>9. Input: At Wimbledon, she reached the fourth round after beating two seeded players.</p>
<p>Analyzation: The input sentence does not contain a person name.</p>
<p>Judgment: No judgment needed.<br />
Now it’s your turn.</p>
<p>Input: <span style="color: blue">[Input Sentence]</span></p>
<p>Please format your response into a JSON file format:</p>
<p>'''</p>
<p>{</p>
<p>"analyzation": "A brief analyzation of the input sentence.",</p>
<p>"judgment": "Your judgment: Yes, No or No judgment needed."</p>
<p>}</p>
<p>'''</p>
</div>
<figcaption>Prompt used for subject judgment.</figcaption>
</figure>

## Does thinking bias arise from different number of tokens? [subsec:Appendix-long_name]

Prior work has shown that the token-wise MLP layers of transformers act as key-value memories `\cite{key-value_mem}`{=latex}. Therefore, another interpretation of the observations from <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a> and <a href="#sec:section-3" data-reference-type="ref+Label" data-reference="sec:section-3">3</a> could be that the number of tokens in names and descriptions affects the efficiency of fact retrieval.

To exclude the factor of token length from our observation of thinking bias, we conduct a new experiment using data with **extremely long names** to match the length of descriptions, such as “Archibald Wolfgang Montgomery Beauregard Fitzwilliam the Third” and “Roderick-Dominic Thelonious-Valentine Hargreaves-St. Clair”. We then replace each name in the original dataset with these synthetic names, resulting in two new datasets: **LongNameIsDesc** and **DescIsLongName**. The average number of tokens of these new names and descriptions is **21.8** and **19.2**, respectively. We reconduct our experiment in <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a> and report the result in <a href="#tab:LongName" data-reference-type="ref+Label" data-reference="tab:LongName">21</a>. Each model’s performances from our main experiment in <a href="#tab:subj_mul_c_result" data-reference-type="ref+Label" data-reference="tab:subj_mul_c_result">1</a> are presented in “()”. Given that performance on MCQs for LongNameIsDesc still significantly exceeds that of DescIsLongName, we conjecture that the models are still biased towards these long names under the effect of thinking bias.

<div class="center" markdown="1">

<div class="small" markdown="1">

<div id="tab:LongName" markdown="1">

|  |  |  |  |  |  |  |  |  |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| (lr)6-9 |  |  |  |  |  |  |  |  |
|  | N2D | D2N | N2D | D2N | N2D | D2N | N2D | D2N |
| LLaMA2-7B-chat | <span style="color: teal">95.9</span> (92.3) | <span style="color: red">3.2</span> (0.3) | <span style="color: teal">**54.7**</span> (65.3) | <span style="color: teal">**51.7**</span> (64.8) | <span style="color: red">5.9</span> (6.5) | <span style="color: teal">81.0</span> (93.6) | <span style="color: red">**25.3**</span> (28.2) | <span style="color: red">**28.2**</span> (26.8) |
| LLaMA2-13B-chat | <span style="color: teal">93.1</span> (95.6) | <span style="color: red">1.1</span> (2.2) | <span style="color: teal">**61.0**</span> (66.8) | <span style="color: teal">**57.2**</span> (70.3) | <span style="color: red">7.5</span> (5.7) | <span style="color: teal">73.3</span> (91.0) | <span style="color: red">**25.9**</span> (25.5) | <span style="color: red">**23.0**</span> (27.8) |

Models’ performances on synthetic biography dataset with extremely long names. Results from our main experiment in <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a> are presented in “()” for comparison. The general trend on this new dataset mirrors that observed in our main experiment, suggesting that LLMs are still biased towards names even if they are extremely long.

</div>

</div>

</div>

# Limitations and future work [sec:Appendix-Discussion]

Our study, while providing valuable insights into the manifestation of the reversal curse and LLMs’ problem-solving patterns, has several limitations. Firstly, our work mainly focuses on finding a hypothesis to explain the puzzling MCQ results, namely the thinking bias, and validate its existence through both CoT prompting and internal interaction. The underlying cause of this bias, as well as the proof of its presence in today’s state-of-the-art close-sourced models, is not fully explored by our current work.

Secondly, despite several attempts to mitigate the thinking bias, we are frustrated to find that currently available techniques failed to alleviate this problem. It derives a hypothesis that an exhaustive rewrite of all training documents to align their structures with the thinking bias seems to be the most effective approach to facilitate the generalization of knowledge. How to derive an effective and practical methodology to enhance LLMs’ training efficacy remains a challenging problem, and we leave this for future work.

# Social impact discussion [sec:Appendix-impact]

Our research, delving into the generalization capabilities of current large language models (LLMs) across various task settings and training data structures, possesses several positive social impacts. Uncovering how the structure of training data correlates with successful downstream performance enables the community to develop more effective and efficient strategies for knowledge injection into LLMs, such as new data filtering criteria or integration with other data augmentation techniques. Moreover, our discovery of inherent thinking bias highlights a critical limitation in LLMs’ learning capacities. Our identification process and mitigation attempts could provide valuable insights and encourage further research aimed at developing more reliable and robust AI systems.

We do not anticipate any negative social impacts from our research, as it focuses on uncovering the limitations of LLMs’ generalization abilities and understanding their underlying causes, and the data employed in our experiment is entirely free from harmful content.

# NeurIPS Paper Checklist [neurips-paper-checklist]

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification: We list our contributions (*i.e.*, new insights into LLMs’ learning and generalization abilities) in the abstract and highlight them in the introduction in <a href="#sec:Introduction" data-reference-type="ref+Label" data-reference="sec:Introduction">1</a>. These claims are firmly based on our experimental results.

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: We have discussed the limitations of our work in <a href="#sec:Appendix-Discussion" data-reference-type="ref+Label" data-reference="sec:Appendix-Discussion">14</a>.

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

14. Justification: Our current work does not include theoretical results.

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

19. Justification: We offer detailed descriptions of our experimental procedures, including the hyperparameter settings for our finetuning experiments in <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a>, <a href="#sec:section-3" data-reference-type="ref+Label" data-reference="sec:section-3">3</a>, <a href="#sec:section-4" data-reference-type="ref+Label" data-reference="sec:section-4">4</a>, <a href="#sec:Appendix-a" data-reference-type="ref+Label" data-reference="sec:Appendix-a">8</a>, <a href="#sec:Appendix-b" data-reference-type="ref+Label" data-reference="sec:Appendix-b">9</a>, <a href="#sec:Appendix-c" data-reference-type="ref+Label" data-reference="sec:Appendix-c">10</a> and <a href="#sec:Appendix-Book-Story" data-reference-type="ref+Label" data-reference="sec:Appendix-Book-Story">11</a>. For both the training and test sets, we detail the construction methods including the templates used for constructing prompts and questions.

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

24. Justification: The code and data are available at <https://github.com/alibaba/thinking_bias.git>. Furthermore, we believe our descriptions of the experimental methods are sufficiently detailed to facilitate ease of reproducibility.

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

29. Justification: We provide comprehensive information for each of our experiments in <a href="#sec:Appendix-a" data-reference-type="ref+Label" data-reference="sec:Appendix-a">8</a>,  <a href="#sec:Appendix-b" data-reference-type="ref+Label" data-reference="sec:Appendix-b">9</a> and <a href="#sec:Appendix-c" data-reference-type="ref+Label" data-reference="sec:Appendix-c">10</a>, covering details like hyperparameter settings and data composition.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: For our main experiment in <a href="#sec:section-2" data-reference-type="ref+Label" data-reference="sec:section-2">2</a>, the finetuning process for each model is conducted using 3 different random seeds. We report the average performance along with error bars representing the standard deviation in <a href="#fig:table-1-vis" data-reference-type="ref+Label" data-reference="fig:table-1-vis">6</a>.

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

39. Justification: All our experiments are conducted on 8 Nvidia A100 80G GPUs. The finetuning process over the synthetic dataset represents the most computationally intensive part of our experiments. We provide sufficient information on our computational resources in <a href="#sec:Appendix-a-hyperparameter-settings" data-reference-type="ref+Label" data-reference="sec:Appendix-a-hyperparameter-settings">8.3</a>.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification: Our research strictly adheres to the NeurIPS Code of Ethics.

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: We have discussed the social impacts of our work in <a href="#sec:Appendix-impact" data-reference-type="ref+Label" data-reference="sec:Appendix-impact">15</a>.

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

54. Justification: Our work carries no such risks, as we have not released any models, and the dataset we employed consists of synthetic, non-harmful facts.

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification: The external assets, such as datasets and models, are publicly available and have been properly credited and cited in our paper.

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

64. Justification: Our code and assets are available at <https://github.com/alibaba/thinking_bias.git>.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification: Our paper does not involve crowdsourcing or research with human subjects.

70. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

73. Answer:

74. Justification: Our paper does not involve crowdsourcing or research with human subjects.

75. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

[^1]: <https://github.com/alibaba/thinking_bias.git>

[^2]: This phenomenon might be a reflection of the preference for information structure (*e.g.*, end-weight principle) in human language `\cite{human_language}`{=latex}, which imperceptibly shapes the knowledge acquisition and problem-solving processes of LLMs during massive corpus pretraining. We leave the validation of this hypothesis for future works.

[^3]: In the original reversal curse paper, the authors introduced an additional subset where the information of each celebrity is presented in both orders to examine the models’ generalization abilities. However, this approach deviates from the objectives of our experiment. Therefore we omit this subset to simplify our demonstration.

[^4]: <https://github.com/lukasberglund/reversal_curse>

[^5]: One might argue that models could also select the correct option based on co-occurrence frequencies within training documents without truly grasping the symmetric property. However, results from the DescriptionIsName subset and the subsequent CoT prompting experiment suggest that this is not the full picture.

[^6]: We also experiment with the average value, which yields quite similar but less pronounced results. We hypothesize that this may be related to the model’s ability to attend to multiple subjects (*i.e.*, options) within a single attention module.

[^7]: https://huggingface.co/datasets/wikimedia/wikipedia
