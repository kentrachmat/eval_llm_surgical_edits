# Large Language Models as Urban Residents: An LLM Agent Framework for Personal Mobility Generation

## Abstract

This paper introduces a novel approach using Large Language Models (LLMs) integrated into an agent framework for flexible and effective personal mobility generation. LLMs overcome the limitations of previous models by effectively processing semantic data and offering versatility in modeling various tasks. Our approach addresses three research questions: aligning LLMs with real-world urban mobility data, developing reliable activity generation strategies, and exploring LLM applications in urban mobility. The key technical contribution is a novel LLM agent framework that accounts for individual activity patterns and motivations, including a self-consistency approach to align LLMs with real-world activity data and a retrieval-augmented strategy for interpretable activity generation. We rigorously evaluate the framework on an extensive, multi-year personal activity trajectory corpus collected in Tokyo, demonstrating consistent improvements over state-of-the-art baselines across spatial, temporal, and semantic metrics. The depth of this single-city evaluation highlights the general utility of our method for urban mobility analysis. Source codes are available at <https://github.com/Wangjw6/LLMob/>.
# Introduction [sec:intro]

The prevalence of large language models (LLMs) has facilitated a variety of applications extending beyond the domain of NLP. Notably, LLMs have gained widespread usage in furthering our understanding of humans and society in a multitude of disciplines, such as economy `\cite{aher2022using}`{=latex} and political science `\cite{argyle2023out}`{=latex}, and have been employed as agents in various social science studies `\cite{xi2023rise,wu2023smart,gao2023large}`{=latex}. In this paper, we target the utilization of LLM agents for the study of personal mobility data. Modeling personal mobility opens up numerous opportunities for building a sustainable community, including proactive traffic management and the design of comprehensive urban development strategies `\cite{batty2012smart,batty2013new,zheng2015trajectory}`{=latex}. In particular, generating reliable activity trajectories has become a promising and effective way to exploit individual activity data `\cite{huang2019variational,choi2021trajgail}`{=latex}. On one hand, learning to generate activity trajectory leads to a thorough understanding of activity patterns, enabling the flexible simulation of urban mobility. On the other hand, while individual activity trajectory data is abundant thanks to advances in telecommunications, its practical use is often limited due to privacy concerns. In this sense, generated data can provide a viable alternative that offers a balance between utility and privacy.

<figure id="fig:trajectory-generation-illustration">
<img src="./figures/intro.png"" />
<figcaption>Personal mobility generation with an LLM agent.</figcaption>
</figure>

While advanced data-driven learning-based methods offer various solutions to generate synthetic individual trajectories `\cite{huang2019variational,xu2024taming,feng2020learning,choi2021trajgail,jiang2021transfer,xu2023revisiting}`{=latex}, the generated data only imitates real-world data from the data distribution perspective rather than semantics, rendering them less effective in simulating or interpreting activities in novel or unforeseen scenarios with a significantly different distribution (e.g., a pandemic). Thus, in this study, to explore a more intelligent and effective activity generation, we propose to establish a trajectory generation framework by exploiting the emerging intelligence of LLM agents, as illustrated in Figure <a href="#fig:trajectory-generation-illustration" data-reference-type="ref" data-reference="fig:trajectory-generation-illustration">1</a>. LLMs present two significant advantages over previous models when applied to activity trajectory generation:

- **Semantic Interpretability.** Unlike previous models, which have predominantly depended on structured data (e.g., GPS coordinates-based trajectory data) for both calibration and simulation `\cite{jiang2016timegeo,pappalardo2018data, zhu2024controltraj}`{=latex}, LLMs exhibit proficiency in interpreting semantic data (e.g., activity trajectory data). This advantage significantly broadens the scope for incorporating a diverse array of data sources into generation processes, thereby enhancing the models’ ability to understand and interact with complex, real-world scenarios in a more nuanced and effective manner.

- **Model Versatility.** Although other data-driven methods manage to learn such dynamic activity patterns for generation, their capacity is limited for generation under unseen scenarios. On the contrary, LLMs have shown remarkable versatility in dealing with unseen tasks, especially the ability to reason and decide based on available information `\cite{openai2022}`{=latex}. This competence enables LLMs to offer a diverse and rational array of choices, making it a promising and flexible approach for modeling personal mobility patterns.

Despite these benefits, ensuring that LLMs align effectively with real-world situations continues to be a significant challenge `\cite{xi2023rise}`{=latex}. This alignment is particularly crucial in the context of urban mobility, where the precision and dependability of LLM outputs are essential for the efficacy of any urban management derived from them. In this study, our aim is to address this challenge by investigating the following research questions: **RQ 1:** How can LLMs be effectively aligned with semantically rich data about daily individual activities? **RQ 2:** What are the effective strategies for achieving reliable and meaningful activity generation using LLM agents? **RQ 3:** What are the potential applications of LLM agents in enhancing urban mobility analysis?

To this end, our study employs LLM agents to infer activity patterns and motivation for personal activity generation tasks. While previous researches advocate habitual activity patterns and motivations as two critical elements for activity generation `\citep{jiang2016timegeo, yuan2023learning}`{=latex}, our proposed framework introduces a more interpretable and effective solution. By leveraging the capabilities of LLMs to process semantically rich datasets (e.g., personal check-in data), we enable a nuanced and interpretable simulation of personal mobility. Our methodology revolves around two phases: (1) activity pattern identification and (2) motivation-driven activity generation. In Phase 1, we leverage the semantic awareness of LLM agents to extract and identify self-consistent, personalized habitual activity patterns from historical data. In Phase 2, we develop two interpretable retrieval-augmented strategies that utilize the patterns identified in Phase 1. These strategies guide LLM agents to infer underlying daily motivations, such as evolving interests or situational needs. Finally, we instruct LLM agents to act as urban residents according to the obtained patterns and motivations. In this way, we generate their daily activities in a specific reasoning logic.

We evaluate the proposed framework using GPT-3.5 APIs over a personal activity trajectory dataset of Tokyo. The results demonstrate the capability of our framework to align LLM agents with semantically rich data for generating individual daily activities. The comparison with baselines, such as attention-based methods `\cite{feng2018DeepMove,luo2021stan}`{=latex}, adversarial learning methods `\cite{choi2021trajgail,yuan2022activity}`{=latex}, and a diffusion model `\cite{zhu2024difftraj}`{=latex}, underscores the advanced generative performance of our framework. The observation also suggests that our framework excels in reproducing temporal and spatio-temporal aspects of personal mobility generation and interpretable activity routines. Moreover, the application of the framework in simulating urban mobility under specific contexts, such as a pandemic scenario, reveals its potential to adapt to external factors and generate realistic activity patterns.

To the best of our knowledge, this study is *one of the pioneering works in developing an LLM agent framework for generating activity trajectory based on real-world data*. We summarize our contributions as follows:

<div class="inparaenum" markdown="1">

We introduce a novel LLM agent framework for personal mobility generation featuring semantic richness.

Our framework introduces a self-consistency evaluation to ensure that the output of LLM agents aligns closely with real-world data on daily activities.

To generate daily activity trajectories, our framework integrates activity patterns with summarized motivations, with two interpretable retrieval-augmented strategies aimed at producing reliable activity trajectories.

By using real-world personal activity data, we validate the effectiveness of our framework and explore its utility in urban mobility analysis.

</div>

# Related Work [sec:related]

## Personal Mobility Generation

Activity trajectory generation offers a valuable perspective for understanding personal mobility. Based on vast call detailed records,  `\citeauthor{jiang2016timegeo}`{=latex} built a mechanistic modeling framework to generate individual activities in high spatial-temporal resolutions. `\citeauthor{pappalardo2018data}`{=latex} employed Markov modeling to estimate the probability of individuals visiting specific locations. Besides, deep learning has become a robust tool for modeling the complex dynamics of traffic  `\cite{jiang2018deepurbanmomentum,huang2019variational,zhang2020trajgail,feng2020learning,luca2021survey}`{=latex}. The primary challenge involves overcoming data-related obstacles such as randomness, sparsity, and irregular patterns `\cite{feng2018DeepMove,yuan2023learning,yuan2022activity,long2023practical}`{=latex}. For example, `\citeauthor{feng2018DeepMove}`{=latex} proposed attentional recurrent networks to handle personal preference and transition regularities. `\citeauthor{yuan2022activity}`{=latex} leveraged deep learning combined with neural differential equations to address the challenges of randomness and sparsity inherent in irregularly sampled activities for activity trajectory generation. Recently,  `\citeauthor{zhu2024difftraj}`{=latex} proposed to utilize a diffusion model to generate GPS trajectories.

## LLM Agents in Social Science

Exploring how to treat LLMs as autonomous agents in specific scenarios leads to diverse and promising applications in social science `\cite{wang2023survey,xi2023rise,wu2023smart,gao2023large}`{=latex}. For instance, `\citeauthor{park2023generative}`{=latex} established an LLM agent framework to simulate human behavior in an interactive scenario, demonstrating the potential of LLMs to model complex social interactions and decision-making processes. Moreover, the application of LLM agents in economic research has been explored, providing new insights into financial markets and economies `\cite{han2023guinea,li2023large}`{=latex}. Extending beyond the realm of social sciences, `\citeauthor{mao2023gpt}`{=latex} adeptly utilized LLMs to generate driving trajectories in motion planning tasks. In the field of natural sciences,  `\citeauthor{williams2023epidemic}`{=latex} integrated LLMs with epidemic models to simulate the spread of diseases. These varied applications highlight the versatility and potential of LLMs to understand and model various real-world dynamics.

# Methodology [sec:method]

We consider the generation of individual daily activity trajectories, each representing an individual’s activities for the whole day. In addition, we focus on the urban context, where the activity trajectory of each individual is represented as a time-ordered sequence of location choices (e.g., POIs) `\cite{luca2021survey}`{=latex}. This sequence is represented by \\(\{(l_0, t_0), (l_1, t_1), \ldots, (l_n, t_n)\}\\), where each \\((l_i, t_i)\\) denotes the individual’s location \\(l_i\\) at time \\(t_i\\).

<figure id="fig:intro">
<img src="./figures/framework.png"" />
<figcaption>LLMob, the proposed <u>LLM</u> agent framework for personal <u>Mob</u>ility generation.</figcaption>
</figure>

By modeling individuals within an urban environment as LLM agents, we present LLMob, an <u>LLM</u> Agent Framework for Personal <u>Mob</u>ility Generation, as illustrated in Figure <a href="#fig:intro" data-reference-type="ref" data-reference="fig:intro">2</a>. LLMob is based on the assumption that an individual’s activities are primarily influenced by two principal factors: habitual activity patterns and current motivations. Habitual activity patterns, representing typical movement behaviors and preferences that indicate regular travel and location choices, are recognized as crucial information for inferring daily activities `\cite{sun2013understanding, diao2016inferring, song2010modelling}`{=latex}. On the other hand, motivations relate to dynamic and situational elements that sway an individual’s choices at any particular moment, such as immediate needs or external circumstances during a specific period. This consideration is vital for capturing and forecasting short-term shifts in mobility patterns `\cite{aher2022using,yuan2023learning}`{=latex}. Moreover, by formulating prompts that assume specific events of concern, this framework allows us to observe the LLM agent’s responses in a variety of situations.

To construct a pipeline for activity trajectory generation, we design an LLM agent with action, memory, and planning `\cite{llmagents,wang2023survey}`{=latex}. Action specifies how an agent interacts with the environment and makes decisions. In LLMob, the environment contains the information collected from real-world data, and the agent acts by generating trajectories. Memory includes past actions that need to be prompted to the LLM to invoke the next action. In LLMob, memory refers to the patterns and motivations output by the agent. Planning formulates or refines a plan over past actions to handle complex tasks, with additional information optionally incorporated as feedback. In LLMob, we use planning to identify patterns and motivations, thereby handling the complex task of trajectory generation. Plan formulation, selection, reflection, and refinement `\cite{huang2024understanding}`{=latex} are employed in succession, and the agent keeps updating the action plan based on its observation `\cite{yao2022react}`{=latex}: The agent first formulates a set of activity plans by extracting candidate patterns from historical trajectories in the database. The agent then performs self-reflection through a self-consistency evaluation to pick the best pattern from the candidate patterns. With historical trajectories further retrieved from the database, the agent refines the identified pattern to a summarized motivation of daily activity, which is then jointly used with the identified pattern for trajectory generation. In addition to the above agentic components, we also suggest the personas of the agent, which can facilitate the LLM to simulate the diversity of real-world individuals `\cite{salewski2023context}`{=latex}.

## Activity Pattern Identification [sec:Pattern]

Phase 1 of LLMob focuses on identifying activity patterns from historical data. To effectively leverage the extracted activity patterns as essential prior knowledge for the generation of daily activities, we introduce the following two steps.

### Pattern Extraction from Semantics and Historical Data

This step derives activity patterns based on activity trajectory data (e.g., individual check-in data). As illustrated in the left panel of Figure <a href="#fig:intro" data-reference-type="ref" data-reference="fig:intro">2</a>, this scheme consists of the following aspects: For each person, we start by specifying a candidate personas to the LLM agent, providing the inspiring foundation for subsequent activity pattern generation. This approach also encourages the diversity of the generated activity patterns, as each candidate persona acts as a unique prior for the generation process (e.g., the significance of user clustering from activity trajectory data in producing meaningful distinctions has been demonstrated `\cite{noulas2011exploiting}`{=latex}). Meanwhile, we perform data preprocessing to extract key information from the extensive historical data. This involves identifying usual commuting distances, pinpointing typical start and end times and locations of daily trips, and concluding the most frequently visited locations of the person. It is important to note that these pieces of information are widely recognized as critical features in mobility analysis `\cite{jiang2016timegeo}`{=latex}. After the preprocessing procedure, both semantic elements with historical data are combined in the prompts, requiring the LLM agent to summarize the activity patterns for this person. By doing this, we set up a streamline to effectively bridge the gap between semantic persona characteristics and concrete historical activity trajectory data, which allows for a more personalized and interpretable representation of individual activities in one day. Moreover, we propose adding candidate personas to the prompt during candidate pattern generation to promote the diversity of the results. Without loss of generality, for each person, a set of \\(C\\) (\\(C = 10\\)) candidate patterns, denoted as \\(\mathcal{CP}\\), are generated according to the historical data and \\(C\\) candidate personas, respectively. We provide the details of these candidate personas in Appendix <a href="#sec:personas" data-reference-type="ref" data-reference="sec:personas">8.4</a>.

### Pattern Evaluation with Self-Consistency

This step involves assessing the consistency of the candidate patterns to identify the most plausible one. We implement a scoring mechanism to evaluate the alignment of candidate patterns with historical data. To achieve this objective, we define a scoring function to gauge each candidate pattern \\(cp\\) in the set \\(\mathcal{CP}\\). This function evaluates \\(cp\\) against two distinct sets of activity trajectories: the specific activity trajectories \\(\mathcal{T}_{i}\\) of a targeted resident \\(i\\) and the sampled activity trajectories from other residents \\(\mathcal{T}_{\sim i}\\): \\[score_{cp} =  \sum_{t \in \mathcal{T}_{i}} r_{t} -\sum_{t' \in \mathcal{T}_{\sim i}} r_{t'},
    \label{eq:score_cp}\\] where we design an evaluation prompt to ask the LLM to generate rating scores \\(r_{t}\\) and \\(r_{t'}\\). Specifically, the LLM agent is prompted to assess the degree of preference for a given trajectory based on the candidate pattern. Ideally, the LLM agent should assign a higher \\(r_{t}\\) for data from the targeted resident and a lower \\(r_{t'}\\) for data from other residents. This scheme essentially identifies the self-consistent pattern: the activity pattern derived from the activity trajectory data of the target user should be consistent with the data from this person during the evaluation. We provide the pseudo-code of the algorithm for Phase 1 of LLMob in Appendix <a href="#sec:pattern-evaluation-algorithm" data-reference-type="ref" data-reference="sec:pattern-evaluation-algorithm">6</a>.

ALGORITHM BLOCK (caption below)

ALGORITHM BLOCK (caption below)

## Motivation-Driven Activity Generation [sec:motivation]

In Phase 2 of LLMob, we focus on the retrieval of motivation and the integration of motivation and activity patterns for individual activity trajectory generation. Since the context length is limited for the LLMs, we can not expect that the LLMs can consume all the available historical information and give plausible output. Retrieval-augmented generation has been identified as a crucial factor in boosting the performance of LLM `\cite{xu2023retrieval}`{=latex}. This enhancement provides additional information that aids LLM in more effectively responding to queries. While previous studies on activity generation mainly overlook the critical factors of macro temporal information (e.g., date) or specific scenarios (e.g., harsh weather) `\cite{yuan2022activity}`{=latex}, we propose a more sophisticated activity generation which accounts for various conditions by taking advantage of the human-like intelligence of LLM. For instance, the activity trajectory at date \\(d\\) can be inferred given the motivation of this date and the habitual activity pattern as: \\[\mathcal{T}_{d} = LLM(\mathcal{M}otivation, \mathcal{P}attern).
    \label{eq:infer}\\] This generation scheme instructs the LLM agent to simulate a designated individual according to a given activity pattern, and then meticulously generate an activity trajectory in accordance with the daily motivation. To obtain insightful and reliable motivations toward different aspects of data availability and sufficiency, two retrieval schemes are proposed. Notably, we considered them as two promising directions for designing solutions to real-world applications, rather than claming which is superior. The detail of each retrieval scheme is introduced as follows:

<figure id="fig:evolving-based-motivation-retrieval">
<img src="./figures/evolving_retrieval.png"" />
<figcaption>Evolving-based motivation retrieval.</figcaption>
</figure>

### Evolving-based Motivation Retrieval

This scheme is related to the intuitive principle that an individual’s motivation on any given day is influenced by her interests and priorities in preceding days `\cite{park2023generative}`{=latex}. Guided by this understanding, our approach harnesses the intelligence of the LLM agent to understand the behavior of daily activities and the underlying motivations. As illustrated in Figure <a href="#fig:evolving-based-motivation-retrieval" data-reference-type="ref" data-reference="fig:evolving-based-motivation-retrieval">3</a>, for a specific date \\(d\\) for which we aim to generate the activity trajectory, we consider the activities of the past \\(k\\) days (\\(k = \min(7, l)\\), where \\(l\\) is the maximum value such that the trajectory for date \\(d - l\\) can be found in the database), and prompt the LLM agent to act as an urban resident based on the pattern identified in Section <a href="#sec:Pattern" data-reference-type="ref" data-reference="sec:Pattern">3.1</a> and summarize \\(k\\) motivations behind these activities. Using these summarized motivations, the LLM agent is further prompted to infer potential motivation for the target date \\(d\\).

### Learning-based Motivation Retrieval

In this scheme, we hypothesize that individuals tend to establish routines in their daily activities, guided by consistent motivations even if the specific locations may vary. For example, if someone frequently visits a burger shop on weekday mornings, this behavior might suggest a motivation for a quick breakfast. Based on this, it is plausible to predict that the same individual might choose a different fast food restaurant in the future, motivated by a similar desire for convenience and speed during their morning meal. We introduce a learning-based scheme to retrieve motivation from historical data. For each new date on which to plan activities, the only information available is the date itself. To use this clue for planning, we first formulate a relative temporal feature \\(\boldsymbol{z}_{{d_{c},d_{p}}}\\) between a past date \\(d_p\\) and the current date \\(d_c\\). This feature captures various aspects, such as the gap between these two dates and whether they belong to the same month. Utilizing this setting, we train a score approximator \\(f_{\theta}(\boldsymbol{z}_{{d_{c},d_{p}}})\\) to evaluate the similarity between any two dates. Notably, due to the lack of supervised signals, we employ unsupervised learning to train \\(f_{\theta}(\cdot)\\). Particularly, a learning scheme based on contrastive learning `\cite{chen2020simple}`{=latex} is established. For each trajectory of a resident, we can scan her other trajectories and identify similar (positive) and dissimilar (negative) dates according to a predefined similarity score. This similarity score is calculated between two activity trajectories \\(\mathcal{T}_{d_a}\\) and \\(\mathcal{T}_{d_b}\\) as: \\[sim_{d_a,d_b} =  \sum_{t=1}^{N_d} \mathbf{1}_{(\mathcal{T}_{d_a}(t) = \mathcal{T}_{d_b}(t))} \text{ if } |\mathcal{T}_{d_a}| > t \text{ and } |\mathcal{T}_{d_b}| > t, 
    \label{eq:sim_score}\\] where \\(N_d\\) is the total number of time intervals (e.g., 10 min) in one day. \\(\mathcal{T}_{d_a}(t)\\) indicates the \\(t\\)th visiting location recorded in trajectory \\(\mathcal{T}_{d_a}\\). Intuitively, there should be more shared locations in the similar trajectory pair. Thereafter, the positive pair is characterized by the highest similarity score, indicative of a greater degree of resemblance between the trajectories. Conversely, the negative pairs are marked by low similarity scores, reflecting a lesser degree of commonality. After obtaining the training dataset from these positive and negative pairs, we train a model to approximate the similarity score between any two dates by contrastive learning. This procedure involves the following steps:

1.  For each date \\(d\\), generate one positive pair \\((d, d^+)\\) and \\(k\\) negative pairs (\\(d, d^{-}_1\\)), ..., (\\(d, d^{-}_k\\)) based on the similarity score and compute \\(\boldsymbol{z}_{d,d^{+}}\\), \\(\boldsymbol{z}_{d,d^{-}_1}\\), ..., \\(\boldsymbol{z}_{d,d^{-}_k}\\).

2.  Forward the positive and negative pairs to \\(f_{\theta}(\cdot)\\) to form: \\[\text{logits} = \left[f_\theta(\boldsymbol{z}_{d,d^{+}}), f_\theta(\boldsymbol{z}_{d,d^{-}_1}), ..., f_\theta(\boldsymbol{z}_{d,d^{-}_k})\right]. %[f_{\theta}(\boldsymbol{z}_{d,d_{+}}) \,|\, f_{\theta}(\boldsymbol{z}_{d,d_{-}})].\\]

3.  Adopt InfoNCE `\cite{oord2018representation}`{=latex} as the contrastive loss function: \\[\mathcal{L}(\theta) = \sum_{n=1}^{N} -\log\left(\frac{e^{\text{logits}_{i}}}{\sum^{k+1}_{j=1} e^{\text{logits}_j}}\right)_{n} ,
            \label{eq:LMR}\\] where \\(N\\) is the batch size of the samples and \\(i\\) indicates the index of the positive pair.

Upon training a similarity score approximation mode, it can be applied to access the similarity between any given query date and historical dates. This enables us to retrieve the most similar historical data, which is prompted to the LLM agent to generate a summary of the motivations prevalent at that time. By doing so, we can extrapolate a motivation relevant to the query date, providing a basis for the LLM agent to generate a new activity trajectory.

# Experiments [sec:exp]

## Experimental Setup [sec:exp:setup]

**Dataset.** We investigate and validate LLMob exclusively on a large-scale personal activity trajectory dataset from Tokyo. The corpus, which spans January 2019 – December 2022, contains more than four years of geo-tagged check-ins collected through Twitter and Foursquare APIs. Crucially, it encompasses both pre-pandemic and pandemic periods, offering rich behavioural diversity that serves as a stringent test bed for generative models. To enable cost-efficient yet detailed analysis we randomly sample 100 residents and reconstruct their daily trajectories at 10-minute granularity. The resulting dataset is sufficiently heterogeneous—covering hundreds of activity categories and the full urban extent of the Tokyo metropolitan area—to serve as a representative proxy for contemporary megacities.

**Metrics.** The following characteristics related to personal activity are used to examine generation: **Step distance (SD)**, **Step interval (SI)**, **Daily activity routine distribution (DARD)**, and **Spatial-temporal visits distribution (STVD)**. After extracting the above characteristics from both the generated and real-world trajectory data, Jensen–Shannon divergence (JSD) is employed to quantify the discrepancy. Lower JSD is preferred.

**Methods.** LLMob is evaluated against mechanic, forecasting, adversarial, and diffusion-based baselines as in the original protocol. To balance capability and cost we employ GPT-3.5-turbo-0613 as the LLM core. Implementation details, hyper-parameters, and persona prompts follow Section 3 and the Appendix.

All experiments were run on a server with an AMD EPYC 7702P CPU, 503 GB RAM, and 4 × NVIDIA RTX A6000 GPUs. Generation with GPT-3.5 accounts for the majority of wall-clock time, totalling roughly 40 GPU-hours for the reported runs.
## Main Results and Analysis [sec:exp:main-results]

**Generative Performance Validation (RQ 1, RQ 2).** The performance evaluation involves analyzing generation results in three distinct settings: (1) Generating normal trajectories based on normal historical trajectories in 2019, a period unaffected by the pandemic. (2) Generating abnormal trajectories based on abnormal historical trajectories in 2020, a year marked by the pandemic. (3) Generating abnormal trajectories in 2021 (pandemic) based on normal historical trajectories in 2019.

The results of these evaluations are detailed in the metrics reported in Table <a href="#tab:traj-gen-results" data-reference-type="ref" data-reference="tab:traj-gen-results">2</a>. Through the comparison, it can be observed that although LLMob may not excel in replicating spatial features (SD) precisely, it demonstrates superior performance in handling temporal aspects (DI). When considering spatial-temporal features (DARD and STVD), LLMob’s performance is also competitive. In particular, LLMob achieves the best performance on DI and DARD for all three settings and is the runner-up on STVD. Baselines like DeepMove and TrajGAIL perform the best on SD and STVD, respectively, but become much less competitive when evaluated in other aspects. We suggest that the pronounced advantage of LLMob in terms of DARD (roughly 1/2 to 1/3 JSD compared to the best of baselines) can be attributed to the LLM agent’s tendency to accurately replicate the motivation behind individual activity behaviors. For instance, an agent may recognize patterns like a person’s habits to have breakfast in the morning, without being restricted to a specific restaurant. This phenomenon highlights the enhanced semantic understanding capabilities of the LLM agent.

<div id="tab:traj-gen-results" markdown="1">

<table>
<caption>Performance (JSD) of trajectory generation based on historical data. Lower is better. Winners and runners-up are marked in boldface and underline, respectively.</caption>
<tbody>
<tr>
<td rowspan="3" style="text-align: left;">Models</td>
<td colspan="4" style="text-align: center;">Normal Trajectory, Normal Data</td>
<td colspan="4" style="text-align: center;">Abnormal Trajectory, Abnormal Data</td>
<td colspan="4" style="text-align: center;">Abnormal Trajectory, Normal Data</td>
</tr>
<tr>
<td colspan="4" style="text-align: center;">(# Generated Trajectories: 1497)</td>
<td colspan="4" style="text-align: center;">(# Generated Trajectories: 904)</td>
<td colspan="4" style="text-align: center;">(# Generated Trajectories: 3555)</td>
</tr>
<tr>
<td style="text-align: left;">SD</td>
<td style="text-align: left;">SI</td>
<td style="text-align: left;">DARD</td>
<td style="text-align: left;">STVD</td>
<td style="text-align: left;">SD</td>
<td style="text-align: left;">SI</td>
<td style="text-align: left;">DARD</td>
<td style="text-align: left;">STVD</td>
<td style="text-align: left;">SD</td>
<td style="text-align: left;">SI</td>
<td style="text-align: left;">DARD</td>
<td style="text-align: left;">STVD</td>
</tr>
<tr>
<td style="text-align: left;">MM <span class="citation" data-cites="pappalardo2018data"></span></td>
<td style="text-align: left;">0.018</td>
<td style="text-align: left;">0.276</td>
<td style="text-align: left;">0.644</td>
<td style="text-align: left;">0.681</td>
<td style="text-align: left;">0.041</td>
<td style="text-align: left;">0.300</td>
<td style="text-align: left;">0.629</td>
<td style="text-align: left;">0.682</td>
<td style="text-align: left;">0.039</td>
<td style="text-align: left;">0.307</td>
<td style="text-align: left;">0.644</td>
<td style="text-align: left;">0.681</td>
</tr>
<tr>
<td style="text-align: left;">LSTM <span class="citation" data-cites="hochreiter1997long"></span></td>
<td style="text-align: left;"><u>0.017</u></td>
<td style="text-align: left;">0.271</td>
<td style="text-align: left;">0.585</td>
<td style="text-align: left;">0.652</td>
<td style="text-align: left;">0.016</td>
<td style="text-align: left;">0.286</td>
<td style="text-align: left;">0.563</td>
<td style="text-align: left;">0.655</td>
<td style="text-align: left;">0.035</td>
<td style="text-align: left;">0.282</td>
<td style="text-align: left;">0.585</td>
<td style="text-align: left;">0.653</td>
</tr>
<tr>
<td style="text-align: left;">DeepMove <span class="citation" data-cites="feng2018DeepMove"></span></td>
<td style="text-align: left;"><strong>0.008</strong></td>
<td style="text-align: left;">0.153</td>
<td style="text-align: left;">0.534</td>
<td style="text-align: left;">0.623</td>
<td style="text-align: left;"><u>0.011</u></td>
<td style="text-align: left;">0.173</td>
<td style="text-align: left;">0.548</td>
<td style="text-align: left;">0.668</td>
<td style="text-align: left;"><strong>0.013</strong></td>
<td style="text-align: left;">0.173</td>
<td style="text-align: left;">0.534</td>
<td style="text-align: left;">0.623</td>
</tr>
<tr>
<td style="text-align: left;">STAN <span class="citation" data-cites="luo2021stan"></span></td>
<td style="text-align: left;">0.152</td>
<td style="text-align: left;">0.400</td>
<td style="text-align: left;">0.692</td>
<td style="text-align: left;">0.692</td>
<td style="text-align: left;">0.115</td>
<td style="text-align: left;">0.092</td>
<td style="text-align: left;">0.693</td>
<td style="text-align: left;">0.691</td>
<td style="text-align: left;">0.142</td>
<td style="text-align: left;">0.094</td>
<td style="text-align: left;">0.692</td>
<td style="text-align: left;">0.690</td>
</tr>
<tr>
<td style="text-align: left;">TrajGAIL <span class="citation" data-cites="choi2021trajgail"></span></td>
<td style="text-align: left;">0.128</td>
<td style="text-align: left;">0.058</td>
<td style="text-align: left;">0.598</td>
<td style="text-align: left;"><strong>0.489</strong></td>
<td style="text-align: left;">0.133</td>
<td style="text-align: left;">0.060</td>
<td style="text-align: left;">0.604</td>
<td style="text-align: left;"><strong>0.523</strong></td>
<td style="text-align: left;">0.332</td>
<td style="text-align: left;">0.058</td>
<td style="text-align: left;">0.434</td>
<td style="text-align: left;"><strong>0.428</strong></td>
</tr>
<tr>
<td style="text-align: left;">ActSTD <span class="citation" data-cites="yuan2022activity"></span></td>
<td style="text-align: left;">0.034</td>
<td style="text-align: left;">0.436</td>
<td style="text-align: left;">0.693</td>
<td style="text-align: left;">0.692</td>
<td style="text-align: left;">0.071</td>
<td style="text-align: left;">0.469</td>
<td style="text-align: left;">0.692</td>
<td style="text-align: left;">0.692</td>
<td style="text-align: left;"><u>0.022</u></td>
<td style="text-align: left;">0.093</td>
<td style="text-align: left;">0.468</td>
<td style="text-align: left;">0.692</td>
</tr>
<tr>
<td style="text-align: left;">DiffTraj <span class="citation" data-cites="zhu2024difftraj"></span></td>
<td style="text-align: left;">0.052</td>
<td style="text-align: left;">0.251</td>
<td style="text-align: left;">0.318</td>
<td style="text-align: left;">0.692</td>
<td style="text-align: left;"><strong>0.008</strong></td>
<td style="text-align: left;">0.240</td>
<td style="text-align: left;">0.339</td>
<td style="text-align: left;">0.692</td>
<td style="text-align: left;">0.101</td>
<td style="text-align: left;">0.142</td>
<td style="text-align: left;">0.218</td>
<td style="text-align: left;">0.693</td>
</tr>
<tr>
<td style="text-align: left;">LLMob-E</td>
<td style="text-align: left;">0.053</td>
<td style="text-align: left;"><strong>0.046</strong></td>
<td style="text-align: left;"><strong>0.125</strong></td>
<td style="text-align: left;">0.559</td>
<td style="text-align: left;">0.056</td>
<td style="text-align: left;"><strong>0.043</strong></td>
<td style="text-align: left;"><u>0.127</u></td>
<td style="text-align: left;">0.615</td>
<td style="text-align: left;">0.062</td>
<td style="text-align: left;"><u>0.056</u></td>
<td style="text-align: left;"><strong>0.117</strong></td>
<td style="text-align: left;">0.536</td>
</tr>
<tr>
<td style="text-align: left;">LLMob-E w/o <span class="math inline">𝒫</span></td>
<td style="text-align: left;">0.055</td>
<td style="text-align: left;">0.069</td>
<td style="text-align: left;">0.223</td>
<td style="text-align: left;"><u>0.530</u></td>
<td style="text-align: left;">0.059</td>
<td style="text-align: left;">0.081</td>
<td style="text-align: left;">0.252</td>
<td style="text-align: left;">0.673</td>
<td style="text-align: left;">0.065</td>
<td style="text-align: left;">0.079</td>
<td style="text-align: left;">0.209</td>
<td style="text-align: left;">0.561</td>
</tr>
<tr>
<td style="text-align: left;">LLMob-E w/o <span class="math inline">𝒮𝒞</span></td>
<td style="text-align: left;">0.058</td>
<td style="text-align: left;">0.076</td>
<td style="text-align: left;">0.295</td>
<td style="text-align: left;">0.589</td>
<td style="text-align: left;">0.068</td>
<td style="text-align: left;">0.086</td>
<td style="text-align: left;">0.225</td>
<td style="text-align: left;">0.649</td>
<td style="text-align: left;">0.072</td>
<td style="text-align: left;">0.096</td>
<td style="text-align: left;">0.301</td>
<td style="text-align: left;">0.589</td>
</tr>
<tr>
<td style="text-align: left;">LLMob-L</td>
<td style="text-align: left;">0.049</td>
<td style="text-align: left;"><u>0.054</u></td>
<td style="text-align: left;"><u>0.136</u></td>
<td style="text-align: left;">0.570</td>
<td style="text-align: left;">0.057</td>
<td style="text-align: left;"><u>0.051</u></td>
<td style="text-align: left;"><strong>0.124</strong></td>
<td style="text-align: left;"><u>0.609</u></td>
<td style="text-align: left;">0.064</td>
<td style="text-align: left;"><strong>0.051</strong></td>
<td style="text-align: left;"><u>0.124</u></td>
<td style="text-align: left;"><u>0.531</u></td>
</tr>
<tr>
<td style="text-align: left;">LLMob-L w/o <span class="math inline">𝒫</span></td>
<td style="text-align: left;">0.061</td>
<td style="text-align: left;">0.080</td>
<td style="text-align: left;">0.270</td>
<td style="text-align: left;">0.600</td>
<td style="text-align: left;">0.072</td>
<td style="text-align: left;">0.081</td>
<td style="text-align: left;">0.286</td>
<td style="text-align: left;">0.641</td>
<td style="text-align: left;">0.073</td>
<td style="text-align: left;">0.091</td>
<td style="text-align: left;">0.248</td>
<td style="text-align: left;">0.580</td>
</tr>
<tr>
<td style="text-align: left;">LLMob-L w/o <span class="math inline">𝒮𝒞</span></td>
<td style="text-align: left;">0.057</td>
<td style="text-align: left;">0.074</td>
<td style="text-align: left;">0.236</td>
<td style="text-align: left;">0.602</td>
<td style="text-align: left;">0.071</td>
<td style="text-align: left;">0.084</td>
<td style="text-align: left;">0.236</td>
<td style="text-align: left;">0.642</td>
<td style="text-align: left;">0.073</td>
<td style="text-align: left;">0.094</td>
<td style="text-align: left;">0.286</td>
<td style="text-align: left;">0.622</td>
</tr>
<tr>
<td style="text-align: left;">LLMob w/o <span class="math inline">ℳ</span></td>
<td style="text-align: left;">0.059</td>
<td style="text-align: left;">0.078</td>
<td style="text-align: left;">0.264</td>
<td style="text-align: left;">0.590</td>
<td style="text-align: left;">0.066</td>
<td style="text-align: left;">0.080</td>
<td style="text-align: left;">0.274</td>
<td style="text-align: left;">0.633</td>
<td style="text-align: left;">0.074</td>
<td style="text-align: left;">0.090</td>
<td style="text-align: left;">0.255</td>
<td style="text-align: left;">0.563</td>
</tr>
<tr>
<td style="text-align: left;">LLMob w/o <span class="math inline">𝒫</span> &amp; <span class="math inline">ℳ</span></td>
<td style="text-align: left;">0.061</td>
<td style="text-align: left;">0.081</td>
<td style="text-align: left;">0.268</td>
<td style="text-align: left;">0.606</td>
<td style="text-align: left;">0.068</td>
<td style="text-align: left;">0.086</td>
<td style="text-align: left;">0.287</td>
<td style="text-align: left;">0.635</td>
<td style="text-align: left;">0.074</td>
<td style="text-align: left;">0.095</td>
<td style="text-align: left;">0.254</td>
<td style="text-align: left;">0.573</td>
</tr>
</tbody>
</table>

</div>

**Exploring Utility in Real-World Applications (RQ 3).** We are interested in how LLMob can elevate the social benefits, particularly in the context of urban mobility. To this end, we propose an example of leveraging the flexibility and intelligence of LLM agents in understanding semantic information and simulating an unseen scenario. In particular, we enhance the original setup by incorporating an additional prompt to provide a context for the LLM agent, enabling it to plan activities during specific circumstances. For example, a “pandemic” prompt is as follows: *Now it is the pandemic period. The government has asked residents to postpone travel and events and to telecommute as much as possible.*

<figure id="fig:exp:freq">
<img src="./figures/radar_freq.png"" />
<figcaption>Daily activity frequency.</figcaption>
</figure>

<figure id="fig:exp:real-world-application">
<figure id="fig:exp:evolve:art">
<img src="./figures/heatmap-art-entertainment-mac.png"" />
<figcaption>Arts &amp; entertainment.</figcaption>
</figure>
<figure id="fig:exp:evolve:prof">
<img src="./figures/heatmap-professional-others-mac.png"" />
<figcaption>Professional &amp; other places.</figcaption>
</figure>
<figcaption>Activity heatmaps for the pandemic scenario.</figcaption>
</figure>

By integrating the above prompt, we can observe the impact of external elements, such as the pandemic and the government’s measures, on urban mobility and related social dynamics. We use the activity trajectory data during the pandemic (2021) as ground truth and plot the daily activity frequency in 7 categories in Figure <a href="#fig:exp:freq" data-reference-type="ref" data-reference="fig:exp:freq">4</a>. TrajGAIL, despite delivering the best STVD in Table <a href="#tab:traj-gen-results" data-reference-type="ref" data-reference="tab:traj-gen-results">2</a>, displays very low frequencies for all the categories, and fails to reflect the tendency of each category. In contrast, a comparison between LLMob-L and the one augmented with the pandemic prompt demonstrates the impact of external factors: there is a significant decrease in activity frequency with the pandemic prompt, which semantically discourages activities likely to spread the disease (e.g., food).

Additionally, from a spatial-temporal perspective, two major activities (e.g., *Arts & entertainment* and *Professional & other places*) are selected to observe the behavior, as shown in Figures <a href="#fig:exp:evolve:art" data-reference-type="ref" data-reference="fig:exp:evolve:art">5</a> and  <a href="#fig:exp:evolve:prof" data-reference-type="ref" data-reference="fig:exp:evolve:prof">6</a>. These activities are particularly insightful as they encapsulate the impact of the pandemic on the work-life balance and daily routines of residents. Specifically, with the pandemic prompt, LLMob reproduces a more realistic spatial-temporal activity pattern. This enhanced realism in the generation is attributed to the integration of prior knowledge about the pandemic’s effects and governmental responses, allowing the LLM agent to behave in a manner that aligns with actual behavioral adaptations. For instance, the reduction in *Arts & entertainment* activities reflects the closure of venues and social distancing guidelines, while changes in *Professional & other places* activities indicate shifts toward remote work and the transformation of professional environments. Intuitively, prompting the LLM agent to generate activities based on various priors shows great potential in real-world applications. The utility of such a conditioned generative approach, coupled with the reliable generated results, can significantly alleviate the workload of urban managers. We suggest that this kind of workflow can simplify the analysis of urban dynamics and aid in assessing the potential impact of urban policies.

## Ablation Studies [sec:exp:ablation]

**Impact of Patterns.** In Table <a href="#tab:traj-gen-results" data-reference-type="ref" data-reference="tab:traj-gen-results">2</a>, by comparing LLMob with and without using patterns (“w/o \\(\mathcal{P}\\)”), we observe that the identified patterns consistently enhance the trajectory generation performance. The improvement on DARD is the most significant (reducing JSD by around 50%), showcasing the use of patterns is a key factor in capturing the semantics of daily activity. We provide example patterns in Appendix <a href="#sec:exp:patterns" data-reference-type="ref" data-reference="sec:exp:patterns">9.1</a> to show how the habitual behaviors of individuals are recognized by patterns.

**Impact of Self-Consistency Evaluation.** By comparing LLMob with and without self-consistency evaluation (“w/o \\(\mathcal{SC}\\)”) in Table <a href="#tab:traj-gen-results" data-reference-type="ref" data-reference="tab:traj-gen-results">2</a>, we find that self-consistency is useful in all aspects, and its impact is the most significant on DARD, especially when generating abnormal trajectories from normal data, showcasing its effectiveness in processing semantics. We also observe that “w/o \\(\mathcal{SC}\\)” performs even worse than “w/o \\(\mathcal{P}\\)” in many cases, because in “w/o \\(\mathcal{SC}\\)”, a candidate pattern is randomly picked for summarizing motivations, potentially introducing inconsistency to an individual’s daily activity.

**Impact of Motivations.** We compare LLMob with and without motivations (“w/o \\(\mathcal{M}\\)”). As can been seen in Table <a href="#tab:traj-gen-results" data-reference-type="ref" data-reference="tab:traj-gen-results">2</a>, the impact of motivations is similar to that of patterns. By comparing to LLMob with both patterns and motivations removed (“w/o \\(\mathcal{P}\\) & \\(\mathcal{M}\\)”), we observe that these two factors collectively lead to better performance. To show the motivations and the generated trajectories, we provide examples in Appendix <a href="#sec:exp:motivations" data-reference-type="ref" data-reference="sec:exp:motivations">9.2</a>, where consistency between them can be observed.

**Impact of Motivation Retrieval Strategy.** We compare LLMob equipped with the two motivation retrieval strategies (“-E” and “-L”). Table <a href="#tab:traj-gen-results" data-reference-type="ref" data-reference="tab:traj-gen-results">2</a> shows that no retrieval strategy always outperforms the other, though evolving-based retrieval wins in more cases (7 vs 5). Moreover, evolving-based retrieval is generally less sensitive to the removal of patterns or self-consistency evaluation, suggesting that resorting to the LLM to process historical trajectories is more robust than using contrastive learning.

# Conclusion

**Contributions.** This study is the first to demonstrate personal mobility simulation empowered by LLM agents on real-world data. Our framework leverages activity patterns and motivations to direct LLM agents in emulating urban residents, facilitating the generation of interpretable and effective individual activity trajectories. Extensive experiments on the Tokyo corpus confirm the superiority of the proposed framework and highlight the promise of LLM agents for urban mobility analysis.

**Social Impacts.** Leveraging artificial intelligence to enhance societal benefits is increasingly promising, especially with the advent of high-capacity models such as LLMs. This study shows how reliable agent-based simulations can assist planners in assessing policy interventions (e.g., pandemic restrictions) without requiring access to sensitive data from multiple jurisdictions.

**Limitations.** Our present work focuses on single-agent reasoning. A natural extension is to model explicit interactions among multiple agents—for example, family members or co-workers whose itineraries influence each other. Addressing these interactions will require designing communication protocols and scalable coordination mechanisms, which we leave for future research. In addition, while our Tokyo-centric study already covers a broad spectrum of behavioural conditions, further enriching the framework with freshly collected data will enable even finer-grained analyses.
# Acknowledgements [acknowledgements]

This work is supported by JSPS KAKENHI JP22H03903, JP23H03406, JP23K17456, JP24K02996, and JST CREST JPMJCR22M2.

# References [references]

<div class="thebibliography" markdown="1">

Gati Aher, Rosa I Arriaga, and Adam Tauman Kalai Using large language models to simulate multiple humans *arXiv preprint arXiv:2208.10264*, 2022. **Abstract:** We introduce a new type of test, called a Turing Experiment (TE), for evaluating to what extent a given language model, such as GPT models, can simulate different aspects of human behavior. A TE can also reveal consistent distortions in a language model’s simulation of a specific human behavior. Unlike the Turing Test, which involves simulating a single arbitrary individual, a TE requires simulating a representative sample of participants in human subject research. We carry out TEs that attempt to replicate well-established findings from prior studies. We design a methodology for simulating TEs and illustrate its use to compare how well different language models are able to reproduce classic economic, psycholinguistic, and social psychology experiments: Ultimatum Game, Garden Path Sentences, Milgram Shock Experiment, and Wisdom of Crowds. In the first three TEs, the existing findings were replicated using recent models, while the last TE reveals a "hyper-accuracy distortion" present in some language models (including ChatGPT and GPT-4), which could affect downstream applications in education and the arts. (@aher2022using)

Lisa P Argyle, Ethan C Busby, Nancy Fulda, Joshua R Gubler, Christopher Rytting, and David Wingate Out of one, many: Using language models to simulate human samples *Political Analysis*, 31 (3): 337–351, 2023. **Abstract:** Abstract We propose and explore the possibility that language models can be studied as effective proxies for specific human subpopulations in social science research. Practical and research applications of artificial intelligence tools have sometimes been limited by problematic biases (such as racism or sexism), which are often treated as uniform properties of the models. We show that the “algorithmic bias” within one such tool—the GPT-3 language model—is instead both fine-grained and demographically correlated, meaning that proper conditioning will cause it to accurately emulate response distributions from a wide variety of human subgroups. We term this property algorithmic fidelity and explore its extent in GPT-3. We create “silicon samples” by conditioning the model on thousands of sociodemographic backstories from real human participants in multiple large surveys conducted in the United States. We then compare the silicon and human samples to demonstrate that the information contained in GPT-3 goes far beyond surface similarity. It is nuanced, multifaceted, and reflects the complex interplay between ideas, attitudes, and sociocultural context that characterize human attitudes. We suggest that language models with sufficient algorithmic fidelity thus constitute a novel and powerful tool to advance understanding of humans and society across a variety of disciplines. (@argyle2023out)

Michael Batty *The new science of cities* MIT press, 2013. **Abstract:** A proposal for a new way to understand cities and their design not as artifacts but as systems composed of flows and networks. In The New Science of Cities, Michael Batty suggests that to understand cities we must view them not simply as places in space but as systems of networks and flows. To understand space, he argues, we must understand flows, and to understand flows, we must understand networks—the relations between objects that compose the system of the city. Drawing on the complexity sciences, social physics, urban economics, transportation theory, regional science, and urban geography, and building on his own previous work, Batty introduces theories and methods that reveal the deep structure of how cities function. Batty presents the foundations of a new science of cities, defining flows and their networks and introducing tools that can be applied to understanding different aspects of city structure. He examines the size of cities, their internal order, the transport routes that define them, and the locations that fix these networks. He introduces methods of simulation that range from simple stochastic models to bottom-up evolutionary models to aggregate land-use transportation models. Then, using largely the same tools, he presents design and decision-making models that predict interactions and flows in future cities. These networks emphasize a notion with relevance for future research and planning: that design of cities is collective action. (@batty2013new)

Michael Batty, Kay W Axhausen, Fosca Giannotti, Alexei Pozdnoukhov, Armando Bazzani, Monica Wachowicz, Georgios Ouzounis, and Yuval Portugali Smart cities of the future *The European Physical Journal Special Topics*, 214: 481–518, 2012. **Abstract:** Here we sketch the rudiments of what constitutes a smart city which we define as a city in which ICT is merged with traditional infrastructures, coordinated and integrated using new digital technologies. We first sketch our vision defining seven goals which concern: developing a new understanding of urban problems; effective and feasible ways to coordinate urban technologies; models and methods for using urban data across spatial and temporal scales; developing new technologies for communication and dissemination; developing new forms of urban governance and organisation; defining critical problems relating to cities, transport, and energy; and identifying risk, uncertainty, and hazards in the smart city. To this, we add six research challenges: to relate the infrastructure of smart cities to their operational functioning and planning through management, control and optimisation; to explore the notion of the city as a laboratory for innovation; to provide portfolios of urban simulation which inform future designs; to develop technologies that ensure equity, fairness and realise a better quality of city life; to develop technologies that ensure informed participation and create shared knowledge for democratic city governance; and to ensure greater and more effective mobility and access to opportunities for urban populations. We begin by defining the state of the art, explaining the science of smart cities. We define six scenarios based on new cities badging themselves as smart, older cities regenerating themselves as smart, the development of science parks, tech cities, and technopoles focused on high technologies, the development of urban services using contemporary ICT, the use of ICT to develop new urban intelligence functions, and the development of online and mobile forms of participation. Seven project areas are then proposed: Integrated Databases for the Smart City, Sensing, Networking and the Impact of New Social Media, Modelling Network Performance, Mobility and Travel Behaviour, Modelling Urban Land Use, Transport and Economic Interactions, Modelling Urban Transactional Activities in Labour and Housing Markets, Decision Support as Urban Intelligence, Participatory Governance and Planning Structures for the Smart City. Finally we anticipate the paradigm shifts that will occur in this research and define a series of key demonstrators which we believe are important to progressing a science of smart cities. (@batty2012smart)

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton A simple framework for contrastive learning of visual representations In *International conference on machine learning*, pages 1597–1607. PMLR, 2020. **Abstract:** This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50. When fine-tuned on only 1% of the labels, we achieve 85.8% top-5 accuracy, outperforming AlexNet with 100X fewer labels. (@chen2020simple)

Seongjin Choi, Jiwon Kim, and Hwasoo Yeo Trajgail: Generating urban vehicle trajectories using generative adversarial imitation learning *Transportation Research Part C: Emerging Technologies*, 128: 103091, 2021. **Abstract:** TrajGAIL: Generating Urban Vehicle Trajectories using Generative Adversarial Imitation Learning Seongjin Choi,Jiwon Kim,Hwasoo Yeo arXiv:2007.14189v4 \[cs.LG\] 16 Jan 2021Highlights TrajGAIL: Generating Urban Vehicle Trajectories using Generative Adversarial Imitation Learning Seongjin Choi,Jiwon Kim,Hwasoo Yeo •Modeling urban vehicle trajectory generation as a partially observable Markov decision process. •A generative adversarial imitation learning framework for urban vehicle trajectory generation •Performance evaluation to assess both trajectory-level similarity and distributional similarity of datasets.TrajGAIL: Generating Urban Vehicle Trajectories using Generative Adversarial Imitation Learning Seongjin Choia, Jiwon Kimband Hwasoo Yeoa,\< aDepartment of Civil and Environmental Engineering, Korea Advanced Institute of Science and Technology, 291 Daehak-ro, Yuseong-gu, Daejeon, Republic of Korea bSchool of Civil Engineering, The University of Queensland, Brisbane St Lucia, Queensland, Australia ARTICLE INFO (@choi2021trajgail)

Mi Diao, Yi Zhu, Joseph Ferreira Jr, and Carlo Ratti Inferring individual daily activities from mobile phone traces: A boston example *Environment and Planning B: Planning and Design*, 43 (5): 920–940, 2016. **Abstract:** Understanding individual daily activity patterns is essential for travel demand management and urban planning. This research introduces a new method to infer individuals’ activities from their mobile phone traces. Using Metro Boston as an example, we develop an activity detection model with travel diary surveys to reveal the common laws governing individuals’ activity participation, and apply the modeling results to mobile phone traces to extract the embedded activity information. The proposed approach enables us to spatially and temporally quantify, visualize, and examine urban activity landscapes in a metropolitan area and provides real-time decision support for the city. This study also demonstrates the potential value of combining new “big data” such as mobile phone traces and traditional travel surveys to improve transportation planning and urban planning and management. (@diao2016inferring)

Jie Feng, Yong Li, Chao Zhang, Funing Sun, Fanchao Meng, Ang Guo, and Depeng Jin Deepmove: Predicting human mobility with attentional recurrent networks In *Proceedings of the 2018 world wide web conference*, pages 1459–1468, 2018. **Abstract:** Human mobility prediction is of great importance for a wide spectrum of location-based applications. However, predicting mobility is not trivial because of four challenges: 1) the complex sequential transition regularities exhibited with time-dependent and high-order nature; 2) the multi-level periodicity of human mobility; 3) the heterogeneity and sparsity of the collected trajectory data; and 4) the complicated semantic motivation behind the mobility. In this paper, we propose DeepMove, an attentional recurrent network for mobility prediction from lengthy and sparse trajectories. In DeepMove, we first design a multi-modal embedding recurrent neural network to capture the complicated sequential transitions by jointly embedding the multiple factors that govern human mobility. Then, we propose a historical attention model with two mechanisms to capture the multi-level periodicity in a principle way, which effectively utilizes the periodicity nature to augment the recurrent neural network for mobility prediction. Furthermore, we design a context adaptor to capture the semantic effects of Point-Of-Interest (POI)-based activity and temporal factor (e.g., dwell time). Finally, we use the multi-task framework to encourage the model to learn comprehensive motivations with mobility by introducing the task of the next activity type prediction and the next check-in time prediction. We perform experiments on four representative real-life mobility datasets, and extensive evaluation results demonstrate that our model outperforms the state-of-the-art models by more than 10 percent. Moreover, compared with the state-of-the-art neural network models, DeepMove provides intuitive explanations into the prediction and sheds light on interpretable mobility prediction. (@feng2018DeepMove)

Jie Feng, Zeyu Yang, Fengli Xu, Haisu Yu, Mudan Wang, and Yong Li Learning to simulate human mobility In *Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining*, pages 3426–3433, 2020. **Abstract:** Realistic simulation of a massive amount of human mobility data is of great use in epidemic spreading modeling and related health policy-making. Existing solutions for mobility simulation can be classified into two categories: model-based methods and model-free methods, which are both limited in generating high-quality mobility data due to the complicated transitions and complex regularities in human mobility. To solve this problem, we propose a model-free generative adversarial framework, which effectively integrates the domain knowledge of human mobility regularity utilized in the model-based methods. In the proposed framework, we design a novel self-attention based sequential modeling network as the generator to capture the complicated temporal transitions in human mobility. To augment the learning power of the generator with the advantages of model-based methods, we design an attention-based region network to introduce the prior knowledge of urban structure to generate a meaningful trajectory. As for the discriminator, we design a mobility regularity-aware loss to distinguish the generated trajectory. Finally, we utilize the mobility regularities of spatial continuity and temporal periodicity to pre-train the generator and discriminator to further accelerate the learning procedure. Extensive experiments on two real-life mobility datasets demonstrate that our framework outperforms seven state-of-the-art baselines significantly in terms of improving the quality of simulated mobility data by 35%. Furthermore, in the simulated spreading of COVID-19, synthetic data from our framework reduces MAPE from 5% \~ 10% (baseline performance) to 2%. (@feng2020learning)

Chen Gao, Xiaochong Lan, Nian Li, Yuan Yuan, Jingtao Ding, Zhilun Zhou, Fengli Xu, and Yong Li Large language models empowered agent-based modeling and simulation: A survey and perspectives *arXiv preprint arXiv:2312.11970*, 2023. **Abstract:** Agent-based modeling and simulation has evolved as a powerful tool for modeling complex systems, offering insights into emergent behaviors and interactions among diverse agents. Integrating large language models into agent-based modeling and simulation presents a promising avenue for enhancing simulation capabilities. This paper surveys the landscape of utilizing large language models in agent-based modeling and simulation, examining their challenges and promising future directions. In this survey, since this is an interdisciplinary field, we first introduce the background of agent-based modeling and simulation and large language model-empowered agents. We then discuss the motivation for applying large language models to agent-based simulation and systematically analyze the challenges in environment perception, human alignment, action generation, and evaluation. Most importantly, we provide a comprehensive overview of the recent works of large language model-empowered agent-based modeling and simulation in multiple scenarios, which can be divided into four domains: cyber, physical, social, and hybrid, covering simulation of both real-world and virtual environments. Finally, since this area is new and quickly evolving, we discuss the open problems and promising future directions. (@gao2023large)

Xu Han, Zengqing Wu, and Chuan Xiao " guinea pig trials" utilizing gpt: A novel smart agent-based modeling approach for studying firm competition and collusion *arXiv preprint arXiv:2308.10974*, 2023. **Abstract:** Firm competition and collusion involve complex dynamics, particularly when considering communication among firms. Such issues can be modeled as problems of complex systems, traditionally approached through experiments involving human subjects or agent-based modeling methods. We propose an innovative framework called Smart Agent-Based Modeling (SABM), wherein smart agents, supported by GPT-4 technologies, represent firms, and interact with one another. We conducted a controlled experiment to study firm price competition and collusion behaviors under various conditions. SABM is more cost-effective and flexible compared to conducting experiments with human subjects. Smart agents possess an extensive knowledge base for decision-making and exhibit human-like strategic abilities, surpassing traditional ABM agents. Furthermore, smart agents can simulate human conversation and be personalized, making them ideal for studying complex situations involving communication. Our results demonstrate that, in the absence of communication, smart agents consistently reach tacit collusion, leading to prices converging at levels higher than the Bertrand equilibrium price but lower than monopoly or cartel prices. When communication is allowed, smart agents achieve a higher-level collusion with prices close to cartel prices. Collusion forms more quickly with communication, while price convergence is smoother without it. These results indicate that communication enhances trust between firms, encouraging frequent small price deviations to explore opportunities for a higher-level win-win situation and reducing the likelihood of triggering a price war. We also assigned different personas to firms to analyze behavioral differences and tested variant models under diverse market structures. The findings showcase the effectiveness and robustness of SABM and provide intriguing insights into competition and collusion. (@han2023guinea)

Sepp Hochreiter and Jürgen Schmidhuber Long short-term memory *Neural computation*, 9 (8): 1735–1780, 1997. **Abstract:** Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error backflow. We briefly review Hochreiter’s (1991) analysis of this problem, then address it by introducing a novel, efficient, gradient based method called long short-term memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete-time steps by enforcing constant error flow through constant error carousels within special units. Multiplicative gate units learn to open and close access to the constant error flow. LSTM is local in space and time; its computational complexity per time step and weight is O. 1. Our experiments with artificial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with real-time recurrent learning, back propagation through time, recurrent cascade correlation, Elman nets, and neural sequence chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, artificial long-time-lag tasks that have never been solved by previous recurrent network algorithms. (@hochreiter1997long)

Dou Huang, Xuan Song, Zipei Fan, Renhe Jiang, Ryosuke Shibasaki, Yu Zhang, Haizhong Wang, and Yugo Kato A variational autoencoder based generative model of urban human mobility In *2019 IEEE conference on multimedia information processing and retrieval (MIPR)*, pages 425–430. IEEE, 2019. **Abstract:** Recently, big and heterogeneous human mobility data inspires many revolutionary ideas of implementing machine learning algorithms for solving some traditional social issues, such as zone regulation, air pollution, and disaster evacuation el at.. However, incomplete datasets were provided owing to both the concerns of violation of privacy and some technique issues in many practical applications, which leads to some limitations of the utility of collected data. Variational Autoencoder (VAE), which uses a well-constructed latent space to capture salient features of the training data, shows a significant excellent performance in not only image processing, but also Natural Language Processing domain. By combining VAE and sequence-to-sequence (seq2seq) model, a Sequential Variational Autoencoder (SVAE) is built for the task of human mobility reconstruction. It is the first time that this kind of SVAE model is implemented for solving the issues about human mobility reconstruction. We use navigation GPS data of selected greater Tokyo area to evaluate the performance of the SVAE model. Experimental results demonstrate that the SVAE model can efficiently capture the salient features of human mobility data and generate more reasonable trajectories. (@huang2019variational)

Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen Understanding the planning of llm agents: A survey *arXiv preprint arXiv:2402.02716*, 2024. **Abstract:** As Large Language Models (LLMs) have shown significant intelligence, the progress to leverage LLMs as planning modules of autonomous agents has attracted more attention. This survey provides the first systematic view of LLM-based agents planning, covering recent works aiming to improve planning ability. We provide a taxonomy of existing works on LLM-Agent planning, which can be categorized into Task Decomposition, Plan Selection, External Module, Reflection and Memory. Comprehensive analyses are conducted for each direction, and further challenges for the field of research are discussed. (@huang2024understanding)

Renhe Jiang, Xuan Song, Zipei Fan, Tianqi Xia, Quanjun Chen, Satoshi Miyazawa, and Ryosuke Shibasaki Deepurbanmomentum: An online deep-learning system for short-term urban mobility prediction In *Proceedings of the AAAI conference on artificial intelligence*, volume 32, 2018. **Abstract:** Big human mobility data are being continuously generated through a variety of sources, some of which can be treated and used as streaming data for understanding and predicting urban dynamics. With such streaming mobility data, the online prediction of short-term human mobility at the city level can be of great significance for transportation scheduling, urban regulation, and emergency management. In particular, when big rare events or disasters happen, such as large earthquakes or severe traffic accidents, people change their behaviors from their routine activities. This means people’s movements will almost be uncorrelated with their past movements. Therefore, in this study, we build an online system called DeepUrbanMomentum to conduct the next short-term mobility predictions by using (the limited steps of) currently observed human mobility data. A deep-learning architecture built with recurrent neural networks is designed to effectively model these highly complex sequential data for a huge urban area. Experimental results demonstrate the superior performance of our proposed model as compared to the existing approaches. Lastly, we apply our system to a real emergency scenario and demonstrate that our system is applicable in the real world. (@jiang2018deepurbanmomentum)

Renhe Jiang, Xuan Song, Zipei Fan, Tianqi Xia, Zhaonan Wang, Quanjun Chen, Zekun Cai, and Ryosuke Shibasaki Transfer urban human mobility via poi embedding over multiple cities *ACM Transactions on Data Science*, 2 (1): 1–26, 2021. **Abstract:** Rapidly developing location acquisition technologies provide a powerful tool for understanding and predicting human mobility in cities, which is very significant for urban planning, traffic regulation, and emergency management. However, with the existing methodologies, it is still difficult to accurately predict millions of peoples’ mobility in a large urban area such as Tokyo, Shanghai, and Hong Kong, especially when collected data used for model training are often limited to a small portion of the total population. Obviously, human activities in city are closely linked with point-of-interest (POI) information, which can reflect the semantic meaning of human mobility. This motivates us to fuse human mobility data and city POI data to improve the prediction performance with limited training data, but current fusion technologies can hardly handle these two heterogeneous data. Therefore, we propose a unique POI-embedding mechanism, that aggregates the regional POIs by categories to generate an artificial POI-image for each urban grid and enriches each trajectory snippet to a four-dimensional tensor in an analogous manner to a short video. Then, we design a deep learning architecture combining CNN with LSTM to simultaneously capture both the spatiotemporal and geographical information from the enriched trajectories. Furthermore, transfer learning is employed to transfer mobility knowledge from one city to another, so that we can fully utilize other cities’ data to train a stronger model for the target city with only limited data available. Finally, we achieve satisfactory performance of human mobility prediction at the citywide level using a limited amount of trajectories as training data, which has been validated over five urban areas of different types and scales. (@jiang2021transfer)

Shan Jiang, Yingxiang Yang, Siddharth Gupta, Daniele Veneziano, Shounak Athavale, and Marta C González The timegeo modeling framework for urban mobility without travel surveys *Proceedings of the National Academy of Sciences*, 113 (37): E5370–E5378, 2016. **Abstract:** Well-established fine-scale urban mobility models today depend on detailed but cumbersome and expensive travel surveys for their calibration. Not much is known, however, about the set of mechanisms needed to generate complete mobility profiles if only using passive datasets with mostly sparse traces of individuals. In this study, we present a mechanistic modeling framework (TimeGeo) that effectively generates urban mobility patterns with resolution of 10 min and hundreds of meters. It ties together the inference of home and work activity locations from data, with the modeling of flexible activities (e.g., other) in space and time. The temporal choices are captured by only three features: the weekly home-based tour number, the dwell rate, and the burst rate. These combined generate for each individual: (i) stay duration of activities, (ii) number of visited locations per day, and (iii) daily mobility networks. These parameters capture how an individual deviates from the circadian rhythm of the population, and generate the wide spectrum of empirically observed mobility behaviors. The spatial choices of visited locations are modeled by a rank-based exploration and preferential return (r-EPR) mechanism that incorporates space in the EPR model. Finally, we show that a hierarchical multiplicative cascade method can measure the interaction between land use and generation of trips. In this way, urban structure is directly related to the observed distance of travels. This framework allows us to fully embrace the massive amount of individual data generated by information and communication technologies (ICTs) worldwide to comprehensively model urban mobility without travel surveys. (@jiang2016timegeo)

Diederik P Kingma and Jimmy Ba Adam: A method for stochastic optimization *arXiv preprint arXiv:1412.6980*, 2014. **Abstract:** We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm. (@kingma2014adam)

Nian Li, Chen Gao, Yong Li, and Qingmin Liao Large language model-empowered agents for simulating macroeconomic activities *arXiv preprint arXiv:2310.10436*, 2023. **Abstract:** The advent of artificial intelligence has led to a growing emphasis on data-driven modeling in macroeconomics, with agent-based modeling (ABM) emerging as a prominent bottom-up simulation paradigm. In ABM, agents (e.g., households, firms) interact within a macroeconomic environment, collectively generating market dynamics. Existing agent modeling typically employs predetermined rules or learning-based neural networks for decision-making. However, customizing each agent presents significant challenges, complicating the modeling of agent heterogeneity. Additionally, the influence of multi-period market dynamics and multifaceted macroeconomic factors are often overlooked in decision-making processes. In this work, we introduce EconAgent, a large language model-empowered agent with human-like characteristics for macroeconomic simulation. We first construct a simulation environment that incorporates various market dynamics driven by agents’ decisions regarding work and consumption. Through the perception module, we create heterogeneous agents with distinct decision-making mechanisms. Furthermore, we model the impact of macroeconomic trends using a memory module, which allows agents to reflect on past individual experiences and market dynamics. Simulation experiments show that EconAgent can make realistic decisions, leading to more reasonable macroeconomic phenomena compared to existing rule-based or learning-based agents. Our codes are released at https://github.com/tsinghua-fib-lab/ACL24-EconAgent. (@li2023large)

Qingyue Long, Huandong Wang, Tong Li, Lisi Huang, Kun Wang, Qiong Wu, Guangyu Li, Yanping Liang, Li Yu, and Yong Li Practical synthetic human trajectories generation based on variational point processes In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 4561–4571, 2023. **Abstract:** Human trajectories, reflecting people’s travel patterns and the range of activities, are crucial for the applications like urban planning and epidemic control. However, the real-world human trajectory data tends to be limited by user privacy or device acquisition issues, leading to its insufficient quality to support the above applications. Hence, generating human trajectory data is a crucial but challenging task, which suffers from the following two critical challenges: 1) how to capture the user distribution in human trajectories (group view), and 2) how to model the complex mobility patterns of each user trajectory (individual view). In this paper, we propose a novel human trajectories generator (named VOLUNTEER), consisting of a user VAE and a trajectory VAE, to address the above challenges. Specifically, in the user VAE, we propose to learn the user distribution with all human trajectories from a group view. In the trajectory VAE, from the individual view, we model the complex mobility patterns by decoupling travel time and dwell time to accurately simulate individual trajectories. Extensive experiments on two real-world datasets show the superiority of our model over the state-of-the-art baselines. Further application analysis in the industrial system also demonstrates the effectiveness of our model. (@long2023practical)

Massimiliano Luca, Gianni Barlacchi, Bruno Lepri, and Luca Pappalardo A survey on deep learning for human mobility *ACM Computing Surveys (CSUR)*, 55 (1): 1–44, 2021. **Abstract:** The study of human mobility is crucial due to its impact on several aspects of our society, such as disease spreading, urban planning, well-being, pollution, and more. The proliferation of digital mobility data, such as phone records, GPS traces, and social media posts, combined with the predictive power of artificial intelligence, triggered the application of deep learning to human mobility. Existing surveys focus on single tasks, data sources, mechanistic or traditional machine learning approaches, while a comprehensive description of deep learning solutions is missing. This survey provides a taxonomy of mobility tasks, a discussion on the challenges related to each task and how deep learning may overcome the limitations of traditional models, a description of the most relevant solutions to the mobility tasks described above, and the relevant challenges for the future. Our survey is a guide to the leading deep learning solutions to next-location prediction, crowd flow prediction, trajectory generation, and flow generation. At the same time, it helps deep learning scientists and practitioners understand the fundamental concepts and the open challenges of the study of human mobility. (@luca2021survey)

Yingtao Luo, Qiang Liu, and Zhaocheng Liu Stan: Spatio-temporal attention network for next location recommendation In *Proceedings of the web conference 2021*, pages 2177–2185, 2021. **Abstract:** The next location recommendation is at the core of various location-based applications. Current state-of-the-art models have attempted to solve spatial sparsity with hierarchical gridding and model temporal relation with explicit time intervals, while some vital questions remain unsolved. Non-adjacent locations and non-consecutive visits provide non-trivial correlations for understanding a user’s behavior but were rarely considered. To aggregate all relevant visits from user trajectory and recall the most plausible candidates from weighted representations, here we propose a Spatio-Temporal Attention Network (STAN) for location recommendation. STAN explicitly exploits relative spatiotemporal information of all the check-ins with self-attention layers along the trajectory. This improvement allows a point-to-point interaction between non-adjacent locations and non-consecutive check-ins with explicit spatiotemporal effect. STAN uses a bi-layer attention architecture that firstly aggregates spatiotemporal correlation within user trajectory and then recalls the target with consideration of personalized item frequency (PIF). By visualization, we show that STAN is in line with the above intuition. Experimental results unequivocally show that our model outperforms the existing state-of-the-art methods by 9-17%. (@luo2021stan)

Jiageng Mao, Yuxi Qian, Hang Zhao, and Yue Wang Gpt-driver: Learning to drive with gpt *arXiv preprint arXiv:2310.01415*, 2023. **Abstract:** We present a simple yet effective approach that can transform the OpenAI GPT-3.5 model into a reliable motion planner for autonomous vehicles. Motion planning is a core challenge in autonomous driving, aiming to plan a driving trajectory that is safe and comfortable. Existing motion planners predominantly leverage heuristic methods to forecast driving trajectories, yet these approaches demonstrate insufficient generalization capabilities in the face of novel and unseen driving scenarios. In this paper, we propose a novel approach to motion planning that capitalizes on the strong reasoning capabilities and generalization potential inherent to Large Language Models (LLMs). The fundamental insight of our approach is the reformulation of motion planning as a language modeling problem, a perspective not previously explored. Specifically, we represent the planner inputs and outputs as language tokens, and leverage the LLM to generate driving trajectories through a language description of coordinate positions. Furthermore, we propose a novel prompting-reasoning-finetuning strategy to stimulate the numerical reasoning potential of the LLM. With this strategy, the LLM can describe highly precise trajectory coordinates and also its internal decision-making process in natural language. We evaluate our approach on the large-scale nuScenes dataset, and extensive experiments substantiate the effectiveness, generalization ability, and interpretability of our GPT-based motion planner. Code is now available at https://github.com/PointsCoder/GPT-Driver. (@mao2023gpt)

Anastasios Noulas, Salvatore Scellato, Cecilia Mascolo, and Massimiliano Pontil Exploiting semantic annotations for clustering geographic areas and users in location-based social networks In *Proceedings of the International AAAI Conference on Web and Social Media*, volume 5, pages 32–35, 2011. **Abstract:** Location-Based Social Networks (LBSN) present so far the most vivid realization of the convergence of the physical and virtual social planes. In this work we propose a novel approach on modeling human activity and geographical areas by means of place categories. We apply a spectral clustering algorithm on areas and users of two metropolitan cities on a dataset sourced from the most vibrant LBSN, Foursquare. Our methodology allows the identification of user communities that visit similar categories of places and the comparison of urban neighborhoods within and across cities. We demonstrate how semantic information attached to places could be plausibly used as a modeling interface for applications such as recommender systems and digital tourist guides. (@noulas2011exploiting)

Aaron van den Oord, Yazhe Li, and Oriol Vinyals Representation learning with contrastive predictive coding *arXiv preprint arXiv:1807.03748*, 2018. **Abstract:** While supervised learning has enabled great progress in many applications, unsupervised learning has not seen such widespread adoption, and remains an important and challenging endeavor for artificial intelligence. In this work, we propose a universal unsupervised learning approach to extract useful representations from high-dimensional data, which we call Contrastive Predictive Coding. The key insight of our model is to learn such representations by predicting the future in latent space by using powerful autoregressive models. We use a probabilistic contrastive loss which induces the latent space to capture information that is maximally useful to predict future samples. It also makes the model tractable by using negative sampling. While most prior work has focused on evaluating representations for a particular modality, we demonstrate that our approach is able to learn useful representations achieving strong performance on four distinct domains: speech, images, text and reinforcement learning in 3D environments. (@oord2018representation)

OpenAI Introducing chatgpt *https://openai.com/blog/chatgpt*, 2022. **Abstract:** ChatGPT has recently emerged to aid in computer programming education due to its cutting-edge functionality of generating program code, debugging, etc. This research firstly focused on what the ethical considerations and solutions are for the first-year IT students who use ChatGPT to write computer programs in an integrated assignment. And then it turned to investigate what impact ChatGPT has on the programming competencies and learning outcomes of students compared to those who do not use ChatGPT. To ensure students use ChatGPT ethically, guidance was provided together with a declaration form of ethically using ChatGPT in each phase of the assignment. Next, we collected and analyzed a survey and their declaration from students and compared student effort, time spent, and performance outcomes from those who were using and without using ChatGPT. Based on the findings, we concluded that although ChatGPT provides an opportunity to the first-year students to learn programming in the way of analysis, synthesis, and evaluation, many students still prefer the conventional way of learning programming in terms of comprehension and application. We argued that since our students in the programming course are always from different academic background levels, we would continue to use both ChatGPT and conventional eLearning resources to meet different learning requirements. (@openai2022)

Luca Pappalardo and Filippo Simini Data-driven generation of spatio-temporal routines in human mobility *Data Mining and Knowledge Discovery*, 32 (3): 787–829, 2018. **Abstract:** The generation of realistic spatio-temporal trajectories of human mobility is of fundamental importance in a wide range of applications, such as the developing of protocols for mobile ad-hoc networks or what-if analysis in urban ecosystems. Current generative algorithms fail in accurately reproducing the individuals’ recurrent schedules and at the same time in accounting for the possibility that individuals may break the routine during periods of variable duration. In this article we present Ditras (DIary-based TRAjectory Simulator), a framework to simulate the spatio-temporal patterns of human mobility. Ditras operates in two steps: the generation of a mobility diary and the translation of the mobility diary into a mobility trajectory. We propose a data-driven algorithm which constructs a diary generator from real data, capturing the tendency of individuals to follow or break their routine. We also propose a trajectory generator based on the concept of preferential exploration and preferential return. We instantiate Ditras with the proposed diary and trajectory generators and compare the resulting algorithm with real data and synthetic data produced by other generative algorithms, built by instantiating Ditras with several combinations of diary and trajectory generators. We show that the proposed algorithm reproduces the statistical properties of real trajectories in the most accurate way, making a step forward the understanding of the origin of the spatio-temporal patterns of human mobility. (@pappalardo2018data)

Joon Sung Park, Joseph C O’Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein Generative agents: Interactive simulacra of human behavior *arXiv preprint arXiv:2304.03442*, 2023. **Abstract:** Believable proxies of human behavior can empower interactive applications ranging from immersive environments to rehearsal spaces for interpersonal communication to prototyping tools. In this paper, we introduce generative agents–computational software agents that simulate believable human behavior. Generative agents wake up, cook breakfast, and head to work; artists paint, while authors write; they form opinions, notice each other, and initiate conversations; they remember and reflect on days past as they plan the next day. To enable generative agents, we describe an architecture that extends a large language model to store a complete record of the agent’s experiences using natural language, synthesize those memories over time into higher-level reflections, and retrieve them dynamically to plan behavior. We instantiate generative agents to populate an interactive sandbox environment inspired by The Sims, where end users can interact with a small town of twenty five agents using natural language. In an evaluation, these generative agents produce believable individual and emergent social behaviors: for example, starting with only a single user-specified notion that one agent wants to throw a Valentine’s Day party, the agents autonomously spread invitations to the party over the next two days, make new acquaintances, ask each other out on dates to the party, and coordinate to show up for the party together at the right time. We demonstrate through ablation that the components of our agent architecture–observation, planning, and reflection–each contribute critically to the believability of agent behavior. By fusing large language models with computational, interactive agents, this work introduces architectural and interaction patterns for enabling believable simulations of human behavior. (@park2023generative)

Leonard Salewski, Stephan Alaniz, Isabel Rio-Torto, Eric Schulz, and Zeynep Akata In-context impersonation reveals large language models’ strengths and biases *arXiv preprint arXiv:2305.14930*, 2023. **Abstract:** In everyday conversations, humans can take on different roles and adapt their vocabulary to their chosen roles. We explore whether LLMs can take on, that is impersonate, different roles when they generate text in-context. We ask LLMs to assume different personas before solving vision and language tasks. We do this by prefixing the prompt with a persona that is associated either with a social identity or domain expertise. In a multi-armed bandit task, we find that LLMs pretending to be children of different ages recover human-like developmental stages of exploration. In a language-based reasoning task, we find that LLMs impersonating domain experts perform better than LLMs impersonating non-domain experts. Finally, we test whether LLMs’ impersonations are complementary to visual information when describing different categories. We find that impersonation can improve performance: an LLM prompted to be a bird expert describes birds better than one prompted to be a car expert. However, impersonation can also uncover LLMs’ biases: an LLM prompted to be a man describes cars better than one prompted to be a woman. These findings demonstrate that LLMs are capable of taking on diverse roles and that this in-context impersonation can be used to uncover their hidden strengths and biases. (@salewski2023context)

Chaoming Song, Tal Koren, Pu Wang, and Albert-László Barabási Modelling the scaling properties of human mobility *Nature physics*, 6 (10): 818–823, 2010. **Abstract:** In recent years, mobility models have been reconsidered based on findings by analyzing some big datasets collected by GPS sensors, cellphone call records, and Geotagging. To understand the fundamental statistical properties of the frequency of serendipitous human encounters, we conducted experiments to collect long-term data on human contact using short-range wireless communication devices which many people frequently carry in daily life. By analyzing the data we showed that the majority of human encounters occur once-in-an-experimental-period: they are Ichi-go Ichi-e. We also found that the remaining more frequent encounters obey a power-law distribution: they are scale-free. To theoretically find the origin of these properties, we introduced as a minimal human mobility model, Homesick L\\}’evy walk, where the walker stochastically selects moving long distances as well as L\\}’evy walk or returning back home. Using numerical simulations and a simple mean-field theory, we offer a theoretical explanation for the properties to validate the mobility model. The proposed model is helpful for evaluating long-term performance of routing protocols in delay tolerant networks and mobile opportunistic networks better since some utility-based protocols select nodes with frequent encounters for message transfer. (@song2010modelling)

Lijun Sun, Kay W Axhausen, Der-Horng Lee, and Xianfeng Huang Understanding metropolitan patterns of daily encounters *Proceedings of the National Academy of Sciences*, 110 (34): 13774–13779, 2013. **Abstract:** Understanding of the mechanisms driving our daily face-to-face encounters is still limited; the field lacks large-scale datasets describing both individual behaviors and their collective interactions. However, here, with the help of travel smart card data, we uncover such encounter mechanisms and structures by constructing a time-resolved in-vehicle social encounter network on public buses in a city (about 5 million residents). This is the first time that such a large network of encounters has been identified and analyzed. Using a population scale dataset, we find physical encounters display reproducible temporal patterns, indicating that repeated encounters are regular and identical. On an individual scale, we find that collective regularities dominate distinct encounters’ bounded nature. An individual’s encounter capability is rooted in his/her daily behavioral regularity, explaining the emergence of "familiar strangers" in daily life. Strikingly, we find individuals with repeated encounters are not grouped into small communities, but become strongly connected over time, resulting in a large, but imperceptible, small-world contact network or "structure of co-presence" across the whole metropolitan area. Revealing the encounter pattern and identifying this large-scale contact network are crucial to understanding the dynamics in patterns of social acquaintances, collective human behaviors, and – particularly – disclosing the impact of human behavior on various diffusion/spreading processes. (@sun2013understanding)

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al A survey on large language model based autonomous agents *arXiv preprint arXiv:2308.11432*, 2023. **Abstract:** Autonomous agents have long been a prominent research focus in both academic and industry communities. Previous research in this field often focuses on training agents with limited knowledge within isolated environments, which diverges significantly from human learning processes, and thus makes the agents hard to achieve human-like decisions. Recently, through the acquisition of vast amounts of web knowledge, large language models (LLMs) have demonstrated remarkable potential in achieving human-level intelligence. This has sparked an upsurge in studies investigating LLM-based autonomous agents. In this paper, we present a comprehensive survey of these studies, delivering a systematic review of the field of LLM-based autonomous agents from a holistic perspective. More specifically, we first discuss the construction of LLM-based autonomous agents, for which we propose a unified framework that encompasses a majority of the previous work. Then, we present a comprehensive overview of the diverse applications of LLM-based autonomous agents in the fields of social science, natural science, and engineering. Finally, we delve into the evaluation strategies commonly used for LLM-based autonomous agents. Based on the previous studies, we also present several challenges and future directions in this field. To keep track of this field and continuously update our survey, we maintain a repository of relevant references at https://github.com/Paitesanshi/LLM-Agent-Survey. (@wang2023survey)

Lilian Weng Llm powered autonomous agents <https://lilianweng.github.io/posts/2023-06-23-agent/>, 2023. **Abstract:** Large language models (LLMs) have revolutionized the field of artificial intelligence, endowing it with sophisticated language understanding and generation capabilities. However, when faced with more complex and interconnected tasks that demand a profound and iterative thought process, LLMs reveal their inherent limitations. Autonomous LLM-powered multi-agent systems represent a strategic response to these challenges. Such systems strive for autonomously tackling user-prompted goals by decomposing them into manageable tasks and orchestrating their execution and result synthesis through a collective of specialized intelligent agents. Equipped with LLM-powered reasoning capabilities, these agents harness the cognitive synergy of collaborating with their peers, enhanced by leveraging contextual resources such as tools and datasets. While these architectures hold promising potential in amplifying AI capabilities, striking the right balance between different levels of autonomy and alignment remains the crucial challenge for their effective operation. This paper proposes a comprehensive multi-dimensional taxonomy, engineered to analyze how autonomous LLM-powered multi-agent systems balance the dynamic interplay between autonomy and alignment across various aspects inherent to architectural viewpoints such as goal-driven task management, agent composition, multi-agent collaboration, and context interaction. It also includes a domain-ontology model specifying fundamental architectural concepts. Our taxonomy aims to empower researchers, engineers, and AI practitioners to systematically analyze the architectural dynamics and balancing strategies employed by these increasingly prevalent AI systems. The exploratory taxonomic classification of selected representative LLM-powered multi-agent systems illustrates its practical utility and reveals potential for future research and development. (@llmagents)

Ross Williams, Niyousha Hosseinichimeh, Aritra Majumdar, and Navid Ghaffarzadegan Epidemic modeling with generative agents *arXiv preprint arXiv:2307.04986*, 2023. **Abstract:** This study offers a new paradigm of individual-level modeling to address the grand challenge of incorporating human behavior in epidemic models. Using generative artificial intelligence in an agent-based epidemic model, each agent is empowered to make its own reasonings and decisions via connecting to a large language model such as ChatGPT. Through various simulation experiments, we present compelling evidence that generative agents mimic real-world behaviors such as quarantining when sick and self-isolation when cases rise. Collectively, the agents demonstrate patterns akin to multiple waves observed in recent pandemics followed by an endemic period. Moreover, the agents successfully flatten the epidemic curve. This study creates potential to improve dynamic system modeling by offering a way to represent human brain, reasoning, and decision making. (@williams2023epidemic)

Zengqing Wu, Run Peng, Xu Han, Shuyuan Zheng, Yixin Zhang, and Chuan Xiao Smart agent-based modeling: On the use of large language models in computer simulations *arXiv preprint arXiv:2311.06330*, 2023. **Abstract:** Computer simulations offer a robust toolset for exploring complex systems across various disciplines. A particularly impactful approach within this realm is Agent-Based Modeling (ABM), which harnesses the interactions of individual agents to emulate intricate system dynamics. ABM’s strength lies in its bottom-up methodology, illuminating emergent phenomena by modeling the behaviors of individual components of a system. Yet, ABM has its own set of challenges, notably its struggle with modeling natural language instructions and common sense in mathematical equations or rules. This paper seeks to transcend these boundaries by integrating Large Language Models (LLMs) like GPT into ABM. This amalgamation gives birth to a novel framework, Smart Agent-Based Modeling (SABM). Building upon the concept of smart agents – entities characterized by their intelligence, adaptability, and computation ability – we explore in the direction of utilizing LLM-powered agents to simulate real-world scenarios with increased nuance and realism. In this comprehensive exploration, we elucidate the state of the art of ABM, introduce SABM’s potential and methodology, and present three case studies (source codes available at https://github.com/Roihn/SABM), demonstrating the SABM methodology and validating its effectiveness in modeling real-world systems. Furthermore, we cast a vision towards several aspects of the future of SABM, anticipating a broader horizon for its applications. Through this endeavor, we aspire to redefine the boundaries of computer simulations, enabling a more profound understanding of complex systems. (@wu2023smart)

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al The rise and potential of large language model based agents: A survey *arXiv preprint arXiv:2309.07864*, 2023. **Abstract:** For a long time, humanity has pursued artificial intelligence (AI) equivalent to or surpassing the human level, with AI agents considered a promising vehicle for this pursuit. AI agents are artificial entities that sense their environment, make decisions, and take actions. Many efforts have been made to develop intelligent agents, but they mainly focus on advancement in algorithms or training strategies to enhance specific capabilities or performance on particular tasks. Actually, what the community lacks is a general and powerful model to serve as a starting point for designing AI agents that can adapt to diverse scenarios. Due to the versatile capabilities they demonstrate, large language models (LLMs) are regarded as potential sparks for Artificial General Intelligence (AGI), offering hope for building general AI agents. Many researchers have leveraged LLMs as the foundation to build AI agents and have achieved significant progress. In this paper, we perform a comprehensive survey on LLM-based agents. We start by tracing the concept of agents from its philosophical origins to its development in AI, and explain why LLMs are suitable foundations for agents. Building upon this, we present a general framework for LLM-based agents, comprising three main components: brain, perception, and action, and the framework can be tailored for different applications. Subsequently, we explore the extensive applications of LLM-based agents in three aspects: single-agent scenarios, multi-agent scenarios, and human-agent cooperation. Following this, we delve into agent societies, exploring the behavior and personality of LLM-based agents, the social phenomena that emerge from an agent society, and the insights they offer for human society. Finally, we discuss several key topics and open problems within the field. A repository for the related papers at https://github.com/WooooDyy/LLM-Agent-Paper-List. (@xi2023rise)

Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro Retrieval meets long context large language models *arXiv preprint arXiv:2310.03025*, 2023. **Abstract:** Extending the context window of large language models (LLMs) is getting popular recently, while the solution of augmenting LLMs with retrieval has existed for years. The natural questions are: i) Retrieval-augmentation versus long context window, which one is better for downstream tasks? ii) Can both methods be combined to get the best of both worlds? In this work, we answer these questions by studying both solutions using two state-of-the-art pretrained LLMs, i.e., a proprietary 43B GPT and Llama2-70B. Perhaps surprisingly, we find that LLM with 4K context window using simple retrieval-augmentation at generation can achieve comparable performance to finetuned LLM with 16K context window via positional interpolation on long context tasks, while taking much less computation. More importantly, we demonstrate that retrieval can significantly improve the performance of LLMs regardless of their extended context window sizes. Our best model, retrieval-augmented Llama2-70B with 32K context window, outperforms GPT-3.5-turbo-16k and Davinci003 in terms of average score on nine long context tasks including question answering, query-based summarization, and in-context few-shot learning tasks. It also outperforms its non-retrieval Llama2-70B-32k baseline by a margin, while being much faster at generation. Our study provides general insights on the choice of retrieval-augmentation versus long context extension of LLM for practitioners. (@xu2023retrieval)

Xiaohang Xu, Toyotaro Suzumura, Jiawei Yong, Masatoshi Hanai, Chuang Yang, Hiroki Kanezashi, Renhe Jiang, and Shintaro Fukushima Revisiting mobility modeling with graph: A graph transformer model for next point-of-interest recommendation In *Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems*, pages 1–10, 2023. **Abstract:** Next Point-of-Interest (POI) recommendation plays a crucial role in urban mobility applications. Recently, POI recommendation models based on Graph Neural Networks (GNN) have been extensively studied and achieved, however, the effective incorporation of both spatial and temporal information into such GNN-based models remains challenging. Temporal information is extracted from users’ trajectories, while spatial information is obtained from POIs. Extracting distinct fine-grained features unique to each piece of information is difficult since temporal information often includes spatial information, as users tend to visit nearby POIs. To address the challenge, we propose Mobility Graph Transformer (MobGT) that enables us to fully leverage graphs to capture both the spatial and temporal features in users’ mobility patterns. MobGT combines individual spatial and temporal graph encoders to capture unique features and global user-location relations. Additionally, it incorporates a mobility encoder based on Graph Transformer to extract higher-order information between POIs. To address the long-tailed problem in spatial-temporal data, MobGT introduces a novel loss function, Tail Loss. Experimental results demonstrate that MobGT outperforms state-of-the-art models on various datasets and metrics, achieving 24% improvement on average. Our codes are available at https://github.com/Yukayo/MobGT. (@xu2023revisiting)

Xiaohang Xu, Renhe Jiang, Chuang Yang, Zipei Fan, and Kaoru Sezaki Taming the long tail in human mobility prediction *arXiv preprint arXiv:2410.14970*, 2024. **Abstract:** With the popularity of location-based services, human mobility prediction plays a key role in enhancing personalized navigation, optimizing recommendation systems, and facilitating urban mobility and planning. This involves predicting a user’s next POI (point-of-interest) visit using their past visit history. However, the uneven distribution of visitations over time and space, namely the long-tail problem in spatial distribution, makes it difficult for AI models to predict those POIs that are less visited by humans. In light of this issue, we propose the Long-Tail Adjusted Next POI Prediction (LoTNext) framework for mobility prediction, combining a Long-Tailed Graph Adjustment module to reduce the impact of the long-tailed nodes in the user-POI interaction graph and a novel Long-Tailed Loss Adjustment module to adjust loss by logit score and sample weight adjustment strategy. Also, we employ the auxiliary prediction task to enhance generalization and accuracy. Our experiments with two real-world trajectory datasets demonstrate that LoTNext significantly surpasses existing state-of-the-art works. (@xu2024taming)

Dingqi Yang, Daqing Zhang, and Bingqing Qu Participatory cultural mapping based on collective behavior data in location-based social networks *ACM Transactions on Intelligent Systems and Technology (TIST)*, 7 (3): 1–23, 2016. **Abstract:** Culture has been recognized as a driving impetus for human development. It co-evolves with both human belief and behavior. When studying culture, Cultural Mapping is a crucial tool to visualize different aspects of culture (e.g., religions and languages) from the perspectives of indigenous and local people. Existing cultural mapping approaches usually rely on large-scale survey data with respect to human beliefs, such as moral values. However, such a data collection method not only incurs a significant cost of both human resources and time, but also fails to capture human behavior, which massively reflects cultural information. In addition, it is practically difficult to collect large-scale human behavior data. Fortunately, with the recent boom in Location-Based Social Networks (LBSNs), a considerable number of users report their activities in LBSNs in a participatory manner, which provides us with an unprecedented opportunity to study large-scale user behavioral data. In this article, we propose a participatory cultural mapping approach based on collective behavior in LBSNs. First, we collect the participatory sensed user behavioral data from LBSNs. Second, since only local users are eligible for cultural mapping, we propose a progressive “home” location identification method to filter out ineligible users. Third, by extracting three key cultural features from daily activity, mobility, and linguistic perspectives, respectively, we propose a cultural clustering method to discover cultural clusters. Finally, we visualize the cultural clusters on the world map. Based on a real-world LBSN dataset, we experimentally validate our approach by conducting both qualitative and quantitative analysis on the generated cultural maps. The results show that our approach can subtly capture cultural features and generate representative cultural maps that correspond well with traditional cultural maps based on survey data. (@yang2016participatory)

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao React: Synergizing reasoning and acting in language models *arXiv preprint arXiv:2210.03629*, 2022. **Abstract:** While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples. Project site with code: https://react-lm.github.io (@yao2022react)

Yuan Yuan, Jingtao Ding, Huandong Wang, Depeng Jin, and Yong Li Activity trajectory generation via modeling spatiotemporal dynamics In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 4752–4762, 2022. **Abstract:** Human daily activities, such as working, eating out, and traveling, play an essential role in contact tracing and modeling the diffusion patterns of the COVID-19 pandemic. However, individual-level activity data collected from real scenarios are highly limited due to privacy issues and commercial concerns. In this paper, we present a novel framework based on generative adversarial imitation learning, to generate artificial activity trajectories that retain both the fidelity and utility of the real-world data. To tackle the inherent randomness and sparsity of irregular-sampled activities, we innovatively capture the spatiotemporal dynamics underlying trajectories by leveraging neural differential equations. We incorporate the dynamics of continuous flow between consecutive activities and instantaneous updates at observed activity points in temporal evolution and spatial transformation. Extensive experiments on two real-world datasets show that our proposed framework achieves superior performance over state-of-the-art baselines in terms of improving the data fidelity and data utility in facilitating practical applications. Moreover, we apply the synthetic data to model the COVID-19 spreading, and it achieves better performance by reducing the simulation MAPE over the baseline by more than 50%. The source code is available online: https://github.com/tsinghua-fib-lab/Activity-Trajectory-Generation. (@yuan2022activity)

Yuan Yuan, Huandong Wang, Jingtao Ding, Depeng Jin, and Yong Li Learning to simulate daily activities via modeling dynamic human needs In *Proceedings of the ACM Web Conference 2023*, pages 906–916, 2023. **Abstract:** Daily activity data that records individuals’ various types of activities in daily life are widely used in many applications such as activity scheduling, activity recommendation, and policymaking. Though with high value, its accessibility is limited due to high collection costs and potential privacy issues. Therefore, simulating human activities to produce massive high-quality data is of great importance to benefit practical applications. However, existing solutions, including rule-based methods with simplified assumptions of human behavior and data-driven methods directly fitting real-world data, both cannot fully qualify for matching reality. In this paper, motivated by the classic psychological theory, Maslow’s need theory describing human motivation, we propose a knowledge-driven simulation framework based on generative adversarial imitation learning. To enhance the fidelity and utility of the generated activity data, our core idea is to model the evolution of human needs as the underlying mechanism that drives activity generation in the simulation model. Specifically, this is achieved by a hierarchical model structure that disentangles different need levels, and the use of neural stochastic differential equations that successfully captures piecewise-continuous characteristics of need dynamics. Extensive experiments demonstrate that our framework outperforms the state-of-the-art baselines in terms of data fidelity and utility. Besides, we present the insightful interpretability of the need modeling. The code is available at https://github.com/tsinghua-fib-lab/SAND. (@yuan2023learning)

Xin Zhang, Yanhua Li, Xun Zhou, Ziming Zhang, and Jun Luo Trajgail: Trajectory generative adversarial imitation learning for long-term decision analysis In *2020 IEEE International Conference on Data Mining (ICDM)*, pages 801–810. IEEE, 2020. **Abstract:** Mobile sensing and information technology have enabled us to collect a large amount of mobility data from human decision-makers, for example, GPS trajectories from taxis, Uber cars, and passenger trip data of taking buses and trains. Understanding and learning human decision-making strategies from such data can potentially promote individual’s well-being and improve the transportation service quality. Existing works on human strategy learning, such as inverse reinforcement learning, all model the decision-making process as a Markov decision process, thus assuming the Markov property. In this work, we show that such Markov property does not hold in real-world human decision-making processes. To tackle this challenge, we develop a Trajectory Generative Adversarial Imitation Learning (TrajGAIL) framework. It captures the long-term decision dependency by modeling the human decision processes as variable length Markov decision processes (VLMDPs), and designs a deep-neural-network-based framework to inversely learn the decision-making strategy from the human agent’s historical dataset. We validate our framework using two real world human-generated spatial-temporal datasets including taxi driver passenger-seeking decision data and public transit trip data. Results demonstrate significant accuracy improvement in learning human decision-making strategies, when comparing to baselines with Markov property assumptions. (@zhang2020trajgail)

Yu Zheng Trajectory data mining: an overview *ACM Transactions on Intelligent Systems and Technology (TIST)*, 6 (3): 1–41, 2015. **Abstract:** Trajectory computing is a pivotal domain encompassing trajectory data management and mining, garnering widespread attention due to its crucial role in various practical applications such as location services, urban traffic, and public safety. Traditional methods, focusing on simplistic spatio-temporal features, face challenges of complex calculations, limited scalability, and inadequate adaptability to real-world complexities. In this paper, we present a comprehensive review of the development and recent advances in deep learning for trajectory computing (DL4Traj). We first define trajectory data and provide a brief overview of widely-used deep learning models. Systematically, we explore deep learning applications in trajectory management (pre-processing, storage, analysis, and visualization) and mining (trajectory-related forecasting, trajectory-related recommendation, trajectory classification, travel time estimation, anomaly detection, and mobility generation). Notably, we encapsulate recent advancements in Large Language Models (LLMs) that hold the potential to augment trajectory computing. Additionally, we summarize application scenarios, public datasets, and toolkits. Finally, we outline current challenges in DL4Traj research and propose future directions. Relevant papers and open-source resources have been collated and are continuously updated at: \\}href{https://github.com/yoshall/Awesome-Trajectory-Computing}{DL4Traj Repo}. (@zheng2015trajectory)

Yuanshao Zhu, Yongchao Ye, Shiyao Zhang, Xiangyu Zhao, and James Yu Difftraj: Generating gps trajectory with diffusion probabilistic model *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Pervasive integration of GPS-enabled devices and data acquisition technologies has led to an exponential increase in GPS trajectory data, fostering advancements in spatial-temporal data mining research. Nonetheless, GPS trajectories contain personal geolocation information, rendering serious privacy concerns when working with raw data. A promising approach to address this issue is trajectory generation, which involves replacing original data with generated, privacy-free alternatives. Despite the potential of trajectory generation, the complex nature of human behavior and its inherent stochastic characteristics pose challenges in generating high-quality trajectories. In this work, we propose a spatial-temporal diffusion probabilistic model for trajectory generation (DiffTraj). This model effectively combines the generative abilities of diffusion models with the spatial-temporal features derived from real trajectories. The core idea is to reconstruct and synthesize geographic trajectories from white noise through a reverse trajectory denoising process. Furthermore, we propose a Trajectory UNet (Traj-UNet) deep neural network to embed conditional information and accurately estimate noise levels during the reverse process. Experiments on two real-world datasets show that DiffTraj can be intuitively applied to generate high-fidelity trajectories while retaining the original distributions. Moreover, the generated results can support downstream trajectory analysis tasks and significantly outperform other methods in terms of geo-distribution evaluations. (@zhu2024difftraj)

Yuanshao Zhu, James Jianqiao Yu, Xiangyu Zhao, Qidong Liu, Yongchao Ye, Wei Chen, Zijian Zhang, Xuetao Wei, and Yuxuan Liang Controltraj: Controllable trajectory generation with topology-constrained diffusion model In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 4676–4687, 2024. **Abstract:** Generating trajectory data is among promising solutions to addressing privacy concerns, collection costs, and proprietary restrictions usually associated with human mobility analyses. However, existing trajectory generation methods are still in their infancy due to the inherent diversity and unpredictability of human activities, grappling with issues such as fidelity, flexibility, and generalizability. To overcome these obstacles, we propose ControlTraj, a Controllable Trajectory generation framework with the topology-constrained diffusion model. Distinct from prior approaches, ControlTraj utilizes a diffusion model to generate high-fidelity trajectories while integrating the structural constraints of road network topology to guide the geographical outcomes. Specifically, we develop a novel road segment autoencoder to extract fine-grained road segment embedding. The encoded features, along with trip attributes, are subsequently merged into the proposed geographic denoising UNet architecture, named GeoUNet, to synthesize geographic trajectories from white noise. Through experimentation across three real-world data settings, ControlTraj demonstrates its ability to produce human-directed, high-fidelity trajectory generation with adaptability to unexplored geographical contexts. (@zhu2024controltraj)

</div>

# Appendix [sec:appendix]

# Algorithm Pseudo-Codes [sec:pattern-evaluation-algorithm]

The pseudo-code of the algorithm for activity pattern identification is given in Algorithm <a href="#alg:self-consistency-evaluation" data-reference-type="ref" data-reference="alg:self-consistency-evaluation">8</a>.

<figure id="alg:self-consistency-evaluation">
<p>Initialize: Empty candidate pattern set <span class="math inline">𝒞𝒫</span> Formulate initial pattern summarization prompt <span class="math inline"><em>p</em><sub>1</sub></span> using prior information extracted from <span class="math inline">𝒯</span></p>
<p>Initialize a score dictionary <span class="math inline">𝒮</span> to store pattern scores</p>
<p><span class="math inline">$best\_pattern \gets \mathop{\rm argmax}_{cp \in \mathcal{CP}} \mathcal{S}[cp]$</span></p>
<figcaption>Activity Pattern Identification</figcaption>
</figure>

# Prompts [sec:prompts]

<div class="promptframed" markdown="1">

Pattern generation Context: Act as a \\(<\\)INPUT 0\\(>\\) in an urban neighborhood, \\(<\\)INPUT 1\\(>\\) \\(<\\)INPUT 2\\(>\\). There are also locations you sometimes visit: \\(<\\)INPUT 3\\(>\\).

Instructions: Reflecting on the context given, I’d like you to summarize your daily activity patterns. Your description should be coherent, utilizing conclusive language to form a well-structured paragraph.

Text to Continue: I am \\(<\\)INPUT 0\\(>\\) in an urban neighborhood, \\(<\\)INPUT 1\\(>\\),...

</div>

where \\(<\\)INPUT 0\\(>\\) and \\(<\\)INPUT 1\\(>\\) are replaced by the candidate personas, and \\(<\\)INPUT 2\\(>\\) and \\(<\\)INPUT 3\\(>\\) will be replaced by the activity habits extracted from historical data. Specifically, we formulate the \\(<\\)INPUT 2\\(>\\) in the following format:

<div class="promptframed" markdown="1">

\\(<\\)INPUT 2\\(>\\) format During \\(<\\)weekend or weekday\\(>\\), you usually travel over \\(<\\)distance\\(>\\) kilometers a day, you usually begin your daily trip at \\(<\\)time of first daily activity\\(>\\) and end your daily trip at \\(<\\)time of last daily activity\\(>\\). you usually visit \\(<\\)location of first daily activity\\(>\\) at the beginning of the day and go to \\(<\\)location of last daily activity\\(>\\) before returning home.

</div>

<div class="promptframed" markdown="1">

Pattern evaluation Now you act as a person with the follwing feature: \\(<\\)INPUT 0\\(>\\) Here is a plan for you. \\(<\\)INPUT 1\\(>\\) On the scale of 1 to 10, where 1 is the least possibility and 10 is the highest possibility, rate how likely you will follow this plan according to your routine pattern and explain the reason.

Rating: \\(<\\)fill in\\(>\\)

Reason:

</div>

where \\(<\\)INPUT 0\\(>\\) is the candidate pattern, and \\(<\\)INPUT 1\\(>\\) is the daily activities plan for evaluation.

<div class="promptframed" markdown="1">

Evolving-based motivation reterieval Context: Act as a person in an urban neighborhood and describe the motivation for your activities. \\(<\\)INPUT 0\\(>\\) In the last \\(<\\)INPUT 1\\(>\\) days you have the following activities: \\(<\\)INPUT 2\\(>\\)

Instructions: Describe in one sentence your future motivation today after these activities. Highlight any personal interests and needs.

</div>

where \\(<\\)INPUT 0\\(>\\) is replaced by the selected pattern, \\(<\\)INPUT 1\\(>\\) is replaced by the number of days from the date to plan activities, and \\(<\\)INPUT 2\\(>\\) is the historical activities corresponding to the chosen date.

<div class="promptframed" markdown="1">

Learning-based motivation reterieval Context: Act as a person in an urban neighborhood. \\(<\\)INPUT 0\\(>\\) If you have the following plans: \\(<\\)INPUT 1\\(>\\)

Instructions: Try to summarize in one sentence what generally motivates you for these plans. Highlight any personal interests and needs.

</div>

where \\(<\\)INPUT 0\\(>\\) is replaced by the selected pattern, and \\(<\\)INPUT 1\\(>\\) Is replaced by the retrieved historical activities.

<div class="promptframed" markdown="1">

Daily activities generation Context: Act as a person in an urban neighborhood. \\(<\\)INPUT 0\\(>\\) Following is the motivation you want to achieve: \\(<\\)INPUT 1\\(>\\)

Instructions: Think about your daily routine. Then tell me your plan for today and exlpain it. The following are the locations you are likely to visit: \\(<\\)INPUT 2\\(>\\) Response to the prompt above in the following format: \\(\{\\)“plan”: \[\\(<\\)Location\\(>\\) at \\(<\\)Time\\(>\\), \\(<\\)Location\\(>\\) at \\(<\\)Time\\(>\\),...\], “reason”:...\\(\}\\)

Example: \\(\{\\)“plan”: \[Elementary School  
\#125 at 9:10, Town Hall  
\#489 at 12:50, Rest Area  
\#585 at 13:40, Seafood Restaurant  
\#105 at 14:20\] “reason”: “My plan today is to finish my teaching duty in the morning and find something delicious to taste.”\\(\}\\)

</div>

where \\(<\\)INPUT 0\\(>\\) is replaced by the selected pattern, \\(<\\)INPUT 1\\(>\\) is replaced by the retrieved motivation, and \\(<\\)INPUT 2\\(>\\) is replaced by the most frequently visited locations.

<div class="promptframed" markdown="1">

Pandemic Now it is the pandemic period. The government has asked residents to postpone travel and events and to telecommute as much as possible.

</div>

# Experimental Setup [experimental-setup]

## Data processing

All the data is obtained through the Twitter and Foursquare APIs and is already anonymized to remove any personally identifiable information before analysis. The detailed process is as follows:

1.  **Filtering Incomplete Data**  
    Users with missing check-ins for a specific year were filtered out.

2.  **Excluding Non-Japan Check-ins**  
    Check-ins that occurred outside of Japan were removed.

3.  **Inferring Prefecture from GPS Coordinates**  
    Prefectures were inferred based on the latitude and longitude data of check-ins.

4.  **Assigning Prefecture**  
    Users were assigned to a prefecture based on their primary check-in location; for example, users whose top check-in location is Tokyo were categorized as belonging to Tokyo.

5.  **Removing Sudden-Move Check-ins**  
    Check-ins showing abrupt, unrealistic location changes, such as from Tokyo to the United States within a short time frame, were deleted to remove data drift, following the criteria proposed by `\cite{yang2016participatory}`{=latex}.

6.  **Anonymizing Data**  
    Real user IDs and geographic location names were anonymized. Only category information of geographic locations was kept, and latitude and longitude coordinates were converted into IDs before being input into the model.

## Environment

We leverage the GPT API to conduct our generation studies. Specifically, we use the gpt-3.5-turbo-0613 version of the API, which is a snapshot of GPT-3.5-turbo from June 13th, 2023. The experiments were carried out on a server with the following specifications:

- **CPU**: AMD EPYC 7702P 64-Core Processor

  - **Architecture**: x86_64

  - **Cores/Threads**: 64 cores, 128 threads

  - **Base Frequency**: 2000.098 MHz

- **Memory**: 503 GB

- **GPUs**: 4 x NVIDIA RTX A6000

  - **Memory**: 48GB each

## Learning-Based Motivation Retrieval [sec:LBMR-setup]

For the learning-based motivation retrieval, the score approximator is parameterized using a fully connected neural network with the following architecture:

<div id="tab:deepmodel_architecture" markdown="1">

| Layer        | Input Size | Output Size | Notes  |
|:-------------|:----------:|:-----------:|:------:|
| Input Layer  |     3      |     64      | Linear |
| Activation   |     \-     |     \-      |  ReLU  |
| Output Layer |     64     |      1      | Linear |

Architecture of the score approximator.

</div>

We include the day of the year for the query date, whether it shares the same weekday as the reference date, and whether both the query and reference dates fall within the same month as input features. Settings for the learning process are as follows: Adam `\cite{kingma2014adam}`{=latex} is used as the optimizer, batch size is 64, learning rate is 0.002, and the number of negative samples is 2.

## Personas [sec:personas]

We use 10 candidate personas as a prior infromation for subsequent pattern generation, as shown in Table <a href="#tab:personas" data-reference-type="ref" data-reference="tab:personas">4</a>.

<div id="tab:personas" markdown="1">

| **Student**: typically travel to and from educational institutions at similar times. |
|:---|
| **Teacher**: typically travel to and from educational institutions at similar times. |
| **Office worker**: have a fixed morning and evening commute, |
| often heading to office districts or commercial centers. |
| **Visitor**: tend to travel throughout the day, |
| often visit attractions, dining areas, and shopping districts. |
| **Night shift worker**: might travel outside of standard business hours, |
| often later in the evening or at night. |
| **Remote worker**: may have non-standard travel patterns, |
| often visit coworking spaces or cafes at various times. |
| **Service industry worker**: tend to travel throughout the day, often visit attractions, |
| dining areas, and shopping districts. |
| **Public service official**: often work in shifts, |
| leading to varied travel times throughout the day and night. |
| **Fitness enthusiast**: often travel early in the morning, in the evening, |
| or on weekends to fitness centers or parks. |
| **Retail employee**: travel patterns might include shifts that |
| start late in the morning and end in the evening. |

Suggested personas and corresponding descriptions.

</div>

# Additional Experimental Results

## Examples of Identified Patterns [sec:exp:patterns]

The patterns are extracted and identified during the first phase in our framework. We report some examples of the identified patterns in our experiments as follows, which correspond to the 10 personas in Table <a href="#tab:personas" data-reference-type="ref" data-reference="tab:personas">4</a>.

<div class="tcolorbox" markdown="1">

You are a student in this urban neighborhood. You typically travel to and from educational institutions at similar times.Your daily routine as a student in an urban neighborhood revolves around traveling to and from educational institutions. On weekdays, you cover a distance of over 10 kilometers, starting your daily trip at 12:00 and concluding it at 20:20. At the beginning of the day, you usually visit Park#2457 and then head to Grocery Store#648 before returning home. Weekends follow a similar pattern, with you traveling over 10 kilometers a day, but starting your daily trip at 13:40 and ending it at 19:20. Like on weekdays, you begin by visiting Park#2457 and then proceed to Grocery Store#648 before heading back home. Throughout the week, you have specific places You visit at fixed times, such as Grocery Store#648 at 20:00, Coffee Shop#571 at 09:30, Park#2457 at 08:30, Irish Pub#21 at 20:00, and Bookstore#313 at 16:00. Additionally, there are other locations You occasionally visit, including Chinese Restaurant#168, Supermarket#802, Park#4719, Tea Room#530, Bookstore#336, Pastry Shop#240, Park#2898, Discount Store#807, and Electronics Store#530. By visiting educational institutions at similar times, you ensure punctuality and consistency in your studies. Additionally, your visits to the park and various stores serve as a way to relax, replenish supplies, and indulge in leisure activities. Overall, your daily routine is structured to fulfill both your educational and personal requirements efficiently.

</div>

<div class="tcolorbox" markdown="1">

You are a teacher in this urban neighborhood. You typically travel to and from educational institutions at similar times.Your daily routine as a teacher in an urban neighborhood revolves around your regular travel to and from educational institutions. On weekdays, you cover a distance of over 60 kilometers a day, starting your journey at 11:50 in the morning and concluding it at 17:50 in the evening. At the beginning of the day, you usually visit Rest Area#1533 before heading to the Housing Development#101, where you spend a significant amount of time before returning home. During weekends, your travel distance reduces to 50 kilometers a day, with your daily trip commencing at 11:20 and ending at 18:00. On weekends, you follow a different routine, starting your day by visiting Shopping Mall#1262 and then proceeding to Motorcycle Shop#149 before heading back home. Additionally, there are certain locations You occasionally visit, such as Convention Center#101, Food Court#559, Motorcycle Shop#354, and Sports Bar#56, among others. Your daily routine is motivated by the need to fulfill your teaching responsibilities and ensure that you are present at the educational institutions You serve. The specific locations You visit, whether it’s the Rest Area#1533 or Shopping Mall#1262, play a role in providing you with necessary resources, relaxation, or opportunities to engage in personal interests. Overall, your routine is structured to maintain a balance between professional commitments and personal needs.r relevant places, you can effectively serve the needs of the urban neighborhood and contribute to its smooth functioning.

</div>

<div class="tcolorbox" markdown="1">

You are a office worker in this urban neighborhood. You have a fixed morning and evening commute, often heading to office districts or commercial centers.Your daily routine as an office worker in an urban neighborhood is quite structured. On weekdays, you travel over 30 kilometers a day, starting your daily trip at 06:50 and ending it at 20:20. Your routine usually begins with a visit to Platform#212, followed by heading to Soba Restaurant#955 before returning home. You also have certain places You visit at specific times, such as Hospital#1255 at 10:30, Platform#212 at 06:30, Soba Restaurant#955 at 22:30, Public Bathroom#76 at 08:30, and Platform#1068 at 05:00. During weekends, your daily travel distance increases to over 40 kilometers, and you start your trip at 07:00, ending it at 20:10. However, the pattern remains the same, starting with a visit to Platform#212 and then going to Soba Restaurant#955 before returning home.

</div>

<div class="tcolorbox" markdown="1">

You are a visitor in this urban neighborhood. You tend to travel throughout the day, often visit attractions, dining areas, and shopping districts.Your daily routine as a visitor in this urban neighborhood involves traveling extensively and exploring various attractions, dining areas, and shopping districts. On weekdays, you cover a distance of over 40 kilometers, starting your day at 11:00 and concluding it at 17:10. You have a consistent pattern of visiting Ramen Restaurant#4841 at the beginning of the day and then heading to Town#373 before returning home. Weekends are slightly different, with a shorter distance of around 30 kilometers covered. You begin your day at 14:00 and end it at 16:50. During weekends, you start by visiting Ramen Restaurant#3773 and then proceed to Town#373 before heading back. There are certain locations that you frequently visit, such as Arcade#929 at 13:30, Bowling Alley#306 at 11:00, Arcade#408 at 12:00, and Grocery Store#2094 at 12:30. Additionally, you sometimes visit other places like Electronics Store#562, Comic Shop#5, Video Game Store#8, and many others. Your routine is motivated by your desire to explore and experience the various attractions, cuisines, and shopping opportunities available in this urban neighborhood. You find joy in discovering new places, trying different foods, and immersing yourself in the vibrant atmosphere of this bustling area.

</div>

<div class="tcolorbox" markdown="1">

You are a night shift worker in this urban neighborhood. You travel to work in the evening and return home in the early morning.Your daily routine as a night shift worker in an urban neighborhood revolves around traveling a considerable distance. On weekdays, you cover over 70 kilometers a day, starting your daily trip at 09:20 and concluding it at 22:30. The routine begins with a visit to Toll Booth#194, followed by a stop at Supermarket#3823 before heading home. During weekends, the pattern remains the same, with the only difference being that you commence your journey at 10:30 and finish at 22:20. On these days, you visit Toll Booth#812 first and then head to Cafe#962 before returning home. In addition to these regular stops, there are occasional visits to various locations such as Record Shop#81, Lake#600, and Factory#495, among others. The motivation behind this routine is to ensure that you are well-prepared for your night shift, starting the day by taking care of essential tasks like stopping at toll booths and getting groceries. These routines help you maintain a sense of order and efficiency in your daily life, ensuring that you are ready for work and able to relax and enjoy your free time.

</div>

<div class="tcolorbox" markdown="1">

You are a remote worker in this urban neighborhood. You may have non-standard travel patterns, often visit coworking spaces or cafes at various times.Your daily routine as a remote worker in an urban neighborhood involves traveling a considerable distance, both during weekdays and weekends. On weekdays, you typically travel over 40 kilometers a day, starting your daily trip at 14:00 and ending it at 19:20. Your routine usually begins with a visit to Convention Center#101, followed by a stop at Supermarket#1593 before returning home. During weekends, your travel distance is slightly less, around 20 kilometers a day, and you begin your daily trip at 11:50, concluding it at 16:30. On weekends, you usually start by visiting Discount Store#884 and then head to Supermarket#1689 before returning home. Additionally, you have specific places You visit at certain times, including Supermarket#1593 at 17:00, Hobby Shop#516 at 13:30, Shopping Mall#1262 at 15:00, Hobby Shop#168 at 14:00, and Exhibit#461 at 11:30. Occasionally, you also visit other locations such as Shopping Mall#1073, Bookstore#14, Shrine#2783, and many more. Your motivation for this routine is to have a flexible work environment, utilizing coworking spaces and cafes, while also fulfilling your daily needs and exploring different places within the urban neighborhood.

</div>

<div class="tcolorbox" markdown="1">

You are a service industry worker in this urban neighborhood. You might travel outside of standard business hours, often later in the evening or at night.Your daily routine as a service industry worker in an urban neighborhood is quite busy and revolves around your work schedule. On weekdays, you travel over 10 kilometers a day, starting your daily trip at 07:20 and ending it at 20:40. The day usually begins with a visit to Historic Site#2176, followed by a stop at Convenience Store#3385 before returning home. During weekends, your travel distance decreases to 0 kilometers a day, starting your daily trip at 09:20 and ending it at 20:10. Similar to weekdays, you start your day by visiting Historic Site#2176 and then go to Public Art#99 before returning home. Additionally, there are certain locations You sometimes visit, including various convenience stores, restaurants, shopping malls, and other establishments. Your motivation for this routine is primarily driven by your work commitments and the need to fulfill your responsibilities as a service industry worker. It is essential for you to visit specific places, such as convenience stores and historic sites, to ensure You have the necessary supplies and maintain a well-rounded understanding of the neighborhood. Overall, this routine enables you to efficiently navigate your urban neighborhood and fulfill your professional obligations.

</div>

<div class="tcolorbox" markdown="1">

You are a public service official in this urban neighborhood. You often work in shifts, leading to varied travel times throughout the day and night.Your daily routine as a public service official in an urban neighborhood revolves around your shifts, which result in different travel times. On weekdays, you typically cover over 60 kilometers a day, starting your daily trip at 12:50 and concluding it at 19:00. At the beginning of the day, you make it a point to visit Convenience Store#3042, and before heading home, you stop by Convenience Store#3702. During the weekends, you travel around 50 kilometers daily, commencing your journey at 10:00 and wrapping it up at 18:30. On weekends, your routine involves visiting Platform#511 in the morning and then heading to Platform#670 before returning home. Additionally, there are a few locations You occasionally visit, such as Platform#1135, Home Service#244, and Convenience Store#6014, among others. The motivation behind your daily routine is to ensure that you cover the necessary ground, making essential stops at various locations to fulfill your duties as a public service official. By visiting convenience stores, platforms, home services, and other relevant places, you can effectively serve the needs of the urban neighborhood and contribute to its smooth functioning.

</div>

<div class="tcolorbox" markdown="1">

You are a fitness enthusiast in this urban neighborhood. You often travel early in the morning, in the evening, or on weekends to fitness centers or parks. Every weekday, you travel over 70 kilometers a day, starting your daily trip at 10:30 and ending it at 19:50. Your routine begins with a visit to Toll Booth#34, where you kickstart your day. After that, you head to Recreation Center#18 to engage in your fitness activities before returning home. On weekends, your daily travel distance is slightly less, around 60 kilometers. You start your trips at 12:40 and end them at 21:00. The weekend routine starts with a visit to Convenience Store#5940, where you grab some essentials for the day. Then, you head to Convenience Store#8965 before finally returning home. Throughout the week, you also occasionally visit other locations such as Tunnel#1307, Event Space#104, Shopping Mall#217 and \#399, and various toll booths. Your motivation for this daily routine is your passion for fitness and maintaining a healthy lifestyle. You prioritize visiting fitness centers or parks during your trips to ensure that you have dedicated time for exercise. Additionally, you make sure to visit convenience stores for any necessary supplies and toll booths for smooth travel. Overall, your routine revolves around staying active, exploring different locations, and ensuring a well-rounded fitness experience.

</div>

<div class="tcolorbox" markdown="1">

You are a retail employee in this urban neighborhood. Your travel patterns might include shifts that start late in the morning and end in the evening. Every weekday, you embark on a daily journey that spans over 50 kilometers. Your routine begins at 09:30, and you conclude your travels at 17:00. To kickstart your day, you always make a point to visit Soba Restaurant#2105, relishing in their delicious offerings. Before heading home, you make a stop at Indian Restaurant#885, savoring their delectable cuisine. On weekends, your daily travel distance decreases slightly to 40 kilometers. You commence your excursions at 09:10 and wrap them up at 17:20. Your first stop on weekends is Hot Spring#205, where you indulge in relaxation and rejuvenation. Before returning home, you make a detour to Department Store#399, exploring the vast array of products they offer. Additionally, there are several other locations You occasionally visit, such as pharmacies, history museums, shrines, shopping malls, and more. Your motivation for this routine stems from your desire for variety and exploration. By visiting different establishments and places, you are able to experience diverse cuisines, immerse yourself in local culture, and find moments of serenity amidst the bustling city life.

</div>

## Examples of Retrieved Motivations and Corresponding Generated Trajectories [sec:exp:motivations]

The retrieved motivations inspire the agent to plan daily activities that closely align with its specific needs. Here are some examples of retrieved motivations and the corresponding daily activities generated in our experiments.

<div class="tcolorbox" markdown="1">

**Motivation:** Your motivations are primarily driven by your desire for entertainment and relaxation, as you enjoy going to the bowling alley and donburi restaurant during late hours, visiting the supermarket and rest area to fulfill your daily needs and take breaks.

------------------------------------------------------------------------

**Daily Activity Trajectory Generated:** Activities at 2019-12-27: Print Shop (35.697, 139.404) at 09:10:00, Entertainment Place (35.733, 139.434) at 10:10:00, Donburi Restaurant (35.730, 139.435) at 10:30:00, Platform (35.733, 139.435) at 10:50:00, Convenience Store (35.721, 139.397) at 11:20:00, Platform (35.729, 139.392) at 21:30:00, Donburi Restaurant (35.699, 139.412) at 22:10:00, Convenience Store (35.697, 139.416) at 22:40:00.

</div>

<div class="tcolorbox" markdown="1">

**Motivation:** Your motivations revolve around fulfilling basic needs such as food, convenience, relaxation, and personal interests, including a fondness for Japanese cuisine, maintaining a healthy lifestyle, and enjoying social gatherings in the evening.

------------------------------------------------------------------------

**Daily Activity Trajectory Generated:** Activities at 2019-12-25: Pet Store (35.699, 139.404) at 08:10:00, Convenience Store (35.697, 139.416) at 13:10:00, Gym (35.697, 139.412) at 13:20:00, Massage Studio (35.698, 139.413) at 14:50:00, Electronics Store (35.699, 139.414) at 15:50:00, Fast Food Restaurant (35.697, 139.413) at 17:10:00, Platform (35.698, 139.415) at 18:10:00, Convenience Store (35.701, 139.393) at 19:10:00.

</div>

<div class="tcolorbox" markdown="1">

**Motivation:** Your motivations in the urban neighborhood are primarily driven by the need for sustenance and convenience, ensuring personal safety and security, fulfilling work responsibilities, and satisfying personal preferences for food.

------------------------------------------------------------------------

**Daily Activity Trajectory Generated:** Activities at 2019-12-30: Park (35.779, 139.491) at 06:40:00, Convenience Store (35.831, 139.519) at 07:10:00, Discount Store (36.356, 139.064) at 10:40:00, Assisted Living (36.338, 139.048) at 12:00:00, Bank (36.361, 139.059) at 12:10:00, Home Service (36.352, 139.050) at 15:30:00, Sushi Restaurant (36.373, 139.061) at 19:40:00, Hot Spring (36.340, 139.052) at 20:30:00.

</div>

<div class="tcolorbox" markdown="1">

**Motivation:** Your motivations revolve around fulfilling basic needs such as food, convenience, relaxation, and personal interests, including a fondness for Japanese cuisine, maintaining a healthy lifestyle, and enjoying social gatherings in the evening.

------------------------------------------------------------------------

**Daily Activity Trajectory Generated:** Activities at 2019-12-29: Park (35.779, 139.491) at 08:20:00, Event Space (35.630, 139.790) at 10:40:00, Convention Center (35.629, 139.795) at 12:20:00, Convenience Store (35.647, 139.817) at 13:00:00, Soba Restaurant (35.643, 139.820) at 13:10:00, Sporting Goods Shop (35.695, 139.762) at 14:50:00, Kushikatsu Restaurant (35.778, 139.495) at 16:40:00.

</div>

<div class="tcolorbox" markdown="1">

**Motivation:** Your motivations revolve around your love for exploring different cuisines, seeking convenience and relaxation through internet cafes, and fulfilling your personal needs by visiting the grocery store.

------------------------------------------------------------------------

**Daily Activity Trajectory Generated:** Activities at 2019-12-31: Internet Cafe (35.723, 140.091) at 07:10:00, Donburi Restaurant (35.581, 140.132) at 12:40:00, Japanese Restaurant (35.369, 140.306) at 17:50:00, Rest Area (35.556, 140.208) at 18:30:00, Supermarket (35.865, 140.024) at 19:40:00.

</div>

## Experiment on Osaka Data [sec:exp:osaka]

*This section has been intentionally deferred to future work.* Focusing on Tokyo—a megacity with highly diverse spatial, temporal, and semantic mobility patterns—already provides a stringent evaluation for generative models; therefore, additional cross-city experiments are not required to substantiate the claims made in this paper.

## Experiment on different LLMs [sec:exp:llms]

We conducted experiments for setting (1) using different LLMs (**GPT-4o-mini** and **Llama 3-8B**). The results are reported as follows:

| **Model**               | **SD** | **SI** | **DARD** | **STVD** |
|:------------------------|:------:|:------:|:--------:|:--------:|
| LLMob-L (GPT-3.5-turbo) | 0.049  | 0.054  |  0.136   |  0.570   |
| LLMob-L (GPT-4o-mini)   | 0.049  | 0.055  |  0.141   |  0.577   |
| LLMob-L (Llama 3-8B)    | 0.054  | 0.063  |  0.119   |  0.566   |
| LLMob-E (GPT-3.5-turbo) | 0.053  | 0.046  |  0.125   |  0.559   |
| LLMob-E (GPT-4o-mini)   | 0.041  | 0.053  |  0.211   |  0.531   |
| LLMob-E (Llama 3-8B)    | 0.054  | 0.059  |  0.122   |  0.561   |

Results of experiments using different LLMs

We observe competitive performance of our framework when other LLMs are used. In particular, **GPT-4o-mini** is the best in terms of the spatial metric (SD); **GPT-3.5-turbo** is the best in terms of the temporal metric (SI). **Llama 3-8B** is overall the best when spatial and temporal factors are evaluated together (DARD and STVD). Such results demonstrate the robustness of our framework across different LLMs.

# NeurIPS Paper Checklist [neurips-paper-checklist]

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification: Our contributions are clarified in the abstract and introduction.

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: The limitations are discussed in the conclusion section.

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

14. Justification: There are no theory assumptions in this paper.

15. Guidelines:

    - The answer NA means that the paper does not include theoretical results.

    - All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.

    - All assumptions should be clearly stated or referenced in the statement of any theorems.

    - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

    - Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in the appendix or supplemental material.

    - Theorems and Lemmas that the proof relies upon should be properly referenced.

16. **Experimental Result Reproducibility**

17. Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

18. Answer:

19. Justification: We provide the source codes of this study. The trajectory dataset is provided in an anonymous setting. Researchers can reproduce the results by running our source codes on this opensource anonymous dataset.

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

24. Justification: We provide the source codes of this study. The trajectory dataset is not provided to its intellectual property. Please refer to the supplementary material for our source codes.

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

29. Justification: We clarify the experimental settings at the beginning of the experiment section and the experimental setup section of the appendix.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: Our results are not accompanied with error bars.

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

39. Justification: We report the experimental environment in the experimental setup section of the appendix.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification: We oblige to the NeurIPS Code of Ethics. The authors are responsible for all the materials presented in this paper.

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: The societal impacts are discussed in the conclusion section.

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

54. Justification: We use the GPT-3.5 API but not release any models. Due to the intellectual property, the dataset is not released.

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification: We clarify how the dataset was obtained at the beginning of the experiment section.

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

64. Justification: We do not release new assets.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification: Neither crowdsourcing nor human subjects are involved in this research.

70. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

73. Answer:

74. Justification: Neither crowdsourcing nor human subjects are involved in this research.

75. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

[^1]: Corresponding author.
