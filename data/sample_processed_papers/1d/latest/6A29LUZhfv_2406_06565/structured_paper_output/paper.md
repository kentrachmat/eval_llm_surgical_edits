# : Deriving Wisdom of the Crowd from  
LLM Benchmark Mixtures

## Abstract

Evaluating large language models (LLMs) is challenging. Traditional ground-truth-based benchmarks fail to capture the comprehensiveness and nuance of real-world queries, while LLM-as-judge benchmarks suffer from grading biases and limited query quantity. Both of them may also become contaminated over time. User-facing evaluation, such as Chatbot Arena, provides reliable signals but is costly and slow. In this work, we propose `MixEval`, a new paradigm for establishing efficient, gold-standard LLM evaluation by strategically mixing off-the-shelf benchmarks. It bridges (1) comprehensive and well-distributed real-world user queries and (2) efficient and fairly-graded ground-truth-based benchmarks, by matching queries mined from the web with similar queries from existing benchmarks. Based on `MixEval`, we further build `MixEval-Hard`, which offers more room for model improvement. Our benchmarks’ advantages lie in (1) a 0.96 model ranking correlation with Chatbot Arena arising from the highly impartial query distribution and grading mechanism, (2) fast, cheap, and reproducible execution (6% of the time and cost of MMLU), and (3) dynamic evaluation enabled by the rapid and stable data update pipeline. We provide extensive meta-evaluation and analysis for our and existing LLM benchmarks to deepen the community’s understanding of LLM evaluation and guide future research directions.

<div class="center" markdown="1">

<span style="color: website"><https://mixeval.github.io/></span>

</div>

<figure id="fig:heatmap_arena">
<img src="./figures/arena_cost.png"" style="width:80.0%" />
<figcaption>Benchmark correlations (%) with Chatbot Arena Elo, against the total costs of evaluating a single GPT-3.5-Turbo-0125 model. <strong><strong>MixEval</strong></strong> and <strong><strong>MixEval-Hard</strong></strong> show the highest correlations with Arena Elo and Arena Elo (En) among leading benchmarks. We reference the crowdsourcing price for Amazon Mechanical Turk ($0.05 per vote) when estimating the cost of evaluating a single model on Chatbot Arena (approximately $2,936). Chatbot Arena is prohibitively expensive, while <strong><strong>MixEval</strong></strong> and <strong><strong>MixEval-Hard</strong></strong> are cheap and cost-effective alternatives. Details on the correlation and evaluation cost values are provided in Section <a href="#sec:corr_querydist_setup" data-reference-type="ref" data-reference="sec:corr_querydist_setup">9</a>. </figcaption>
</figure>

# Introduction [sec:introduction]

**That Which is Measured, Improves.** Evaluation is essential in the AI community for two main reasons: (1) benchmarks provide early signals to model developers, aiding in refining data and model design, and (2) benchmarks guide users in selecting suitable models for specific use cases. Therefore, benchmarks offer feedback to the entire community, facilitating model optimization. Consequently, the main concern of evaluating LLMs is **impartiality**–we need to optimize impartial objectives so that the community advances in the right direction. In practical LLM evaluations, three primary biases contribute to a lack of impartiality: (1) **query bias**–evaluation queries falling short of comprehensiveness or appropriate distribution (2) **grading bias**–the grading process involving significant bias or error (3) **generalization bias**–models overfitting the evaluation data.

**Large Scale User-facing Evaluation Provides a More Impartial Signal.** Practitioners generally adopt either automatic or user-facing approaches for LLM benchmarking. Automatic benchmarking typically employs traditional ground-truth-based benchmarks, such as MMLU `\citep{hendrycks2020measuring}`{=latex}, which often fail to capture real-world query comprehensiveness and nuance while involving a comparatively impartial grading process; or employs open-ended benchmarks using LLMs as graders, such as MT-Bench `\citep{zheng2024judging}`{=latex}, suffering from both grading bias and query incomprehensiveness due to the preference biases and high cost of frontier LLM judges. Additionally, the static nature of automatic benchmarks results in contamination over time, amplifying the generalization issue. Such biases lead to significant deviations from gold-standard evaluation, impeding model development. On the other hand, large-scale user-facing benchmarking, such as Chatbot Arena[^3] `\citep{chiang2024chatbot}`{=latex}, offers more reliable objectives for model development and effectively mitigates the above-mentioned three biases because (1) it collects a vast array of real-world user queries, thereby ensuring superior query comprehensiveness and distribution, (2) it judges diverse and complex model responses stably due to the “wisdom of the crowd” effect `\citep{yi2012wisdom}`{=latex}, where individual judgment noise is averaged out over a large number of samples, mitigating the grading bias, and (3) it continuously receives fresh user queries, mitigating the benchmark contamination issue. Furthermore, it guides model optimization to meet user needs effectively in practical applications, which is a crucial goal of developing models. However, Chatbot Arena is prohibitively expensive (Figure <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a>), slow, and irreproducible. Moreover, it is not directly accessible for public usage, hindering practitioners from conducting easy and fast model evaluations.

****MixEval**: Towards Efficient Gold-Standard LLM Evaluations.** In this work, we aim to establish a highly impartial gold-standard benchmark without compromising efficiency. This can be achieved by leveraging (1) the efficiency and grading impartiality of ground-truth-based benchmarks and (2) the superior comprehensiveness and distribution of real-world user queries. To this end, we propose `MixEval`, a two-stage benchmark reconstruction pipeline consisting of (1) wild query mining and (2) grounding existing benchmarks in the mined queries. We introduce an accurate user query retrieval process, comprising query detection, filtering, and classification. In the detection phase, we train open-source LLMs on self-collected data to detect queries in Common Crawl splits. During filtering, we utilize GPT-4 Turbo to exclude non-query sentences. In classification, we categorize the filtered queries by input and output modalities, retaining text-in-text-out queries for LLM evaluation. To align benchmark queries with real-world queries, we match each crawled web user query with its most similar query in the benchmark pool and the corresponding ground truth answer. We designate the resulting benchmark as `MixEval`. To improve the benchmark’s ability to distinguish strong models, we derive a challenging subset from `MixEval`, termed `MixEval-Hard`. To mitigate the overfitting issue, we periodically update the data points in `MixEval` and `MixEval-Hard` using our fast, stable pipeline, which performs benchmark mixture with a different batch of wild queries from the same distribution, showing low model score variance (0.36 Std. on a 0-100 scale) and significant version difference (85% unique query ratio). We thereby effectively mitigate the above-mentioned three evaluation biases through the proposed benchmark mixture pipeline, while maintaining high efficiency. As shown in Figure <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a>, `MixEval` and `MixEval-Hard` achieve similar model rankings as Chatbot Arena while being far less costly.

**Why use **MixEval**?** `MixEval` offers five significant advantages for practitioners: (1) **accurate** model ranking, demonstrated by a 0.96 correlation with Chatbot Arena, (2) **fast**, **cheap** and **reproducible** execution, requiring only 6% the time and cost of MMLU and with no dependence on human input, (3) **dynamic** benchmarking enabled by low-effort and stable updating mechanism, (4) a **comprehensive** and **less biased** query distribution, as it bases queries on a large-scale web corpus, and (5) a **fair** grading process without preference bias, ensured by its ground-truth-based nature.

**Research Contributions**

- We developed a pipeline for detecting real-world instructions, capable of mining queries to build benchmarks and providing a scalable solution for collecting vast amounts of real-world instruction-following data.

- We introduced a new way to utilize benchmarks, demonstrating that real-world query distributions and user preferences can be reconstructed by strategically mixing off-the-shelf benchmarks with web-mined queries.

- To the best of our knowledge, `MixEval` creates the first ground-truth-based dynamic benchmark with general-domain queries, benefiting from a rapid and stable data updating mechanism.

- The resulting dynamic benchmarks, *i.e.,*`MixEval` and `MixEval-Hard`, exhibit significant correlations (0.93 and 0.96) with real-world user preference leaderboard (*i.e.,* Chatbot Arena) and showcase high impartiality and efficiency.

- We provide meta-evaluation and extensive analysis for `MixEval` and other leading LLM benchmarks, delivering detailed insights that enhance the community’s understanding of LLM evaluation and guide future research directions.

# LLM Benchmarks are Biased from Realistic User Queries and Preferences [sec:llm_benchmark_analysis]

<figure id="fig:benchmark_distribution">
<img src="./figures/all_embeddings_distribution.png"" />
<figcaption>Query Topic Distribution of the Benchmarks. Ground-truth-based benchmarks are represented by <span style="color: orange_dist">orange</span> dots, wild datasets by <span style="color: yellow_dist">yellow</span> dots, and LLM-judged benchmarks (MT-Bench and Arena-Hard) by <span style="color: yellow_dist">yellow</span> dots, all plotted against our detected web queries shown as <span style="color: blue_dist">blue</span> dots. Query sentence embeddings were dimensionally reduced to map them onto a unified 2-D space, facilitating direct comparisons of topic distributions across benchmarks. As we move from the bottom to the top of the figure, query topics transition from non-technical to technical. Topic summaries for each region are detailed in Figure <a href="#fig:topic_map" data-reference-type="ref" data-reference="fig:topic_map">3</a>.</figcaption>
</figure>

**How Much Do Our Benchmarks Reflect Real-world User Queries and Preferences?** The rapid advancement of LLMs has led to the introduction of numerous benchmarks. However, the community may still lack a comprehensive understanding of how well these benchmarks align with real-world use cases and human preferences. Without such understanding, the signals derived from evaluations might be misleading, thereby impeding model development. To investigate this issue, we (1) analyze the correlations between benchmarks and (2) visualize their query distributions in a unified 2-D space.

## Setup

**Correlation Matrix Heatmap (Figures <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a> and <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a>).** We present the correlation matrix of prominent benchmarks, where warmer colors indicate higher correlations. Model scores are collected from various sources, including the Chatbot Arena Leaderboard `\citep{chiang2024chatbot}`{=latex}, Open LLM Leaderboard `\citep{openllmleaderboard}`{=latex}, and OpenCompass Leaderboard `\citep{contributors2023opencompass}`{=latex}. Our data collection adheres to three principles: (1) We exclude scores reported by model authors, relying solely on evaluation leaderboards to ensure fairness. (2) For each benchmark, scores are sourced from a single platform to eliminate the influence of varying evaluation settings on model rankings. (3) When multiple sources are available for a benchmark, we select the one with the highest number of models in common with other benchmarks. The number of common models for each pair of benchmarks is detailed in Figure <a href="#fig:heatmap_stats" data-reference-type="ref" data-reference="fig:heatmap_stats">18</a>.

**Query Distribution Map (Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a>).** We present the distribution of benchmark queries sorted by their distance to our detected web queries. Each benchmark (orange or yellow) is plotted against the detected wild queries (blue). We uniformly sampled 1000 queries from each LLM benchmark and wild dataset, with a sampling number of 200 for MT-Bench and Arena-Hard due to their smaller sizes. We combined the query embeddings and reduced their dimensions to the same 2-D space to facilitate direct comparisons of the benchmark query distributions. A detailed case study revealed that the reduced space primarily represents the topics of the queries, with queries on similar topics clustering in specific regions of the map. To better understand the topic distribution of different benchmarks, we divided the aggregated queries of all benchmarks into 16 patches based on location (Figure <a href="#fig:topic_map" data-reference-type="ref" data-reference="fig:topic_map">3</a>). We then uniformly sampled 100 queries from each patch and used GPT-4 to summarize the topics of the sampled queries. As illustrated in Figure <a href="#fig:topic_map" data-reference-type="ref" data-reference="fig:topic_map">3</a>, the 2-D query distribution exhibits a distinct regional trend: queries located higher on the map are more technical. The distribution transitions from non-technical topics, such as Social Interactions, at the bottom to technical ones, such as Programming and Mathematics, at the top.

## Important Takeaways

**Most benchmarks show a limited correlation with human preferences.** Figures <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a> and <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a> reveal that most benchmarks exhibit a limited correlation with human preferences (Arena Elo). Using the benchmark mixture technique, our `MixEval` and `MixEval-Hard` achieve the highest correlations with human preferences, at 0.93 and 0.96, respectively.

**Most benchmarks exhibit a skewed query distribution.** Ground-truth-based and LLM-judged benchmarks show a skewed query distribution compared to detected web queries and wild datasets. Notably, BoolQ, Natural Questions, and MMLU align more closely with wild user queries due to their data collection methods. Specifically, questions in BoolQ and Natural Questions originate from Google Search, while MMLU is designed to cover a wide range of topics (57 topics, including Atari games `\citep{bellemare2013arcade}`{=latex}).

**Query comprehensiveness is crucial.** General-domain benchmarks, which are not tailored for specific domains in their data collection pipelines, exhibit a stronger correlation with Arena Elo than domain-specific ones. Notably, 10 out of 13 general-domain benchmarks have an Arena Elo correlation score above 0.5, whereas only 1 out of 8 domain-specific benchmarks achieves this. This underscores the significance of comprehensive queries for achieving a high correlation with human preferences.

<figure id="fig:topic_map">
<img src="./figures/all_embeddings_distribution_on_one.png"" style="width:58.0%" />
<figcaption>Query topic summarization for Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a>. The plot aggregates all queries and divides them into 16 regions. From each region, 100 queries are uniformly sampled and analyzed by GPT-4 for topic summarization. A clear trend is observed, with topics transitioning from non-technical at the bottom to technical at the top.</figcaption>
</figure>

**Some general-domain benchmarks are actually domain-specific.** Despite being labeled as general-domain benchmarks, DROP and WinoGrande have limited scopes, often narrower than many domain-specific benchmarks (such as MATH). As depicted in Figure <a href="#fig:topic_map" data-reference-type="ref" data-reference="fig:topic_map">3</a>, DROP queries mainly address History, Politics, Sports, Demographics, and Societal Issues, whereas WinoGrande queries focus primarily on Grammar, Language, Decision Making, and Social Dynamics.

**User population size affects the query distribution.** Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a> shows distribution differences in wild queries across varying user population sizes. ShareGPT, grounded in 100 million[^4] active users of ChatGPT by mid-2023, contrasts with WildChat `\citep{zhao2024wildchat}`{=latex}, Chatbot Arena Conversations `\citep{chiang2024chatbot}`{=latex}, and LMSYS-Chat-1M `\citep{zheng2024judging}`{=latex}, which have user bases of 0.2 million, 0.13 million, and 0.21 million, respectively. The global internet user count was 5.4 billion[^5] in 2023, an order of magnitude larger than all considered wild datasets. Consequently, user bases of the internet, ShareGPT, and other datasets span three distinct orders of magnitude. ShareGPT’s larger user population (second order of magnitude) yields a distribution most similar to web queries from the global internet user base (third order of magnitude), both visually and in cluster distance (C-Dist).

**Chatbot Arena and Arena-Hard queries exhibit biases.** Compared to web queries and ShareGPT data, datasets from the Chatbot Arena website—Chatbot Arena Conversations and LMSYS-Chat-1M—have a higher proportion of technical queries (as presented in Figure <a href="#fig:topic_map" data-reference-type="ref" data-reference="fig:topic_map">3</a>, queries with higher position on the map are more technical). This indicates a user base skewed towards technical users, potentially affecting evaluation results, as an effective LLM benchmark should mimic real-world use cases. Furthermore, a minor discrepancy exists between web and ShareGPT queries, suggesting that one or both of them may still slightly deviate from actual real-world query distributions. Moreover, Arena-Hard queries exhibit a pronounced bias towards technical topics. This likely stems from the design of their data pipeline for sampling hard prompts. We will demonstrate that employing a carefully designed sampling technique is essential to preserve the query distribution while enhancing difficulty. This is supported by `MixEval-Hard`’s similar distribution to the original web queries and wild datasets (see Section <a href="#sec:evalhard" data-reference-type="ref" data-reference="sec:evalhard">3.3</a>).

# **MixEval**

<figure id="fig:mixeval_pipeline">
<img src="./figures/mixeval_pipeline.png"" />
<figcaption><code>MixEval</code>, a two-stage benchmark reconstruction pipeline, comprises (1) web query detection and (2) benchmark mixture. We further introduce <code>MixEval-Hard</code> to enhance model separability, alongside a dynamic updating mechanism to mitigate contamination risk.</figcaption>
</figure>

In Section <a href="#sec:llm_benchmark_analysis" data-reference-type="ref" data-reference="sec:llm_benchmark_analysis">2</a>, we show that current ground-truth-based and LLM-judged benchmarks have skewed query distributions and limited correlation with human preferences. Additionally, LLM-judged benchmarks suffer from LLM preference bias and both of them become contaminated over time. In contrast, Chatbot Arena is less biased and more dynamic but requires slow and expensive human preference data collection, resulting in irreproducible outcomes.

To address these issues, we introduce `MixEval`(Figure <a href="#fig:mixeval_pipeline" data-reference-type="ref" data-reference="fig:mixeval_pipeline">4</a>), which aligns ground-truth-based LLM benchmarks with real-world user queries. This method uses user queries mined from the web and matches them with similar queries from existing benchmarks, and involves two stages: (1) user query detection from the web and (2) benchmark mixture. To improve model separability and reduce contamination, we also propose `MixEval-Hard` and a dynamic updating mechanism.

## Web User Query Detection

In this stage, we detect user queries from Common Crawl `\citep{together2023redpajama}`{=latex}. Both recall and precision are crucial to ensure the query distribution reflects real-world scenarios. Therefore, we developed two benchmarks to evaluate our query detector’s performance. The first benchmark includes self-collected in-the-wild user queries as positive samples, with non-query datasets such as Wikipedia `\citep{wikidump}`{=latex} as negative samples. The second, higher-quality benchmark contains positive and negative samples hand-picked by our authors from in-the-wild query and non-query datasets. In preliminary experiments, direct prompting of open-source language models performed poorly on our benchmarks. Thus, we developed a rectification pipeline to ensure high recall and precision cost-effectively. We started with a detection phase to gather training data. Testing various open-source LLMs, Vicuna 33B `\citep{chiang2023vicuna}`{=latex} achieved a high recall (\>99%) on our test sets with careful prompt engineering, ensuring that very few positive samples were missed initially. In this phase, we detected around 20k queries using Vicuna 33B over a subset of Common Crawl. We then used GPT-4 to more accurately label these data as positive or negative samples, and used the resulting data to train Vicuna 33B. The trained Vicuna 33B achieved high recall (\>99%) and precision (\>98%) on our benchmarks and detected 2M user queries from the entire Common Crawl. Finally, we prompted GPT-4 Turbo to further filter and classify them, extracting text-in-text-out queries for LLM evaluation. Future work will address queries with other I/O modalities.

## Benchmark Mixture

<figure id="fig:data_dist">
<div class="center">
<img src="./figures/data_dist_histogram.png"" />
</div>
<figcaption>The normalized number of queries in <code>MixEval</code> and the original benchmarks.</figcaption>
</figure>

To bridge wild user queries \\(\mathcal{Q}\\) and ground-truth LLM benchmarks, we create a benchmark pool \\(\mathcal{B} = \{\mathcal{B}_1, \mathcal{B}_2, ..., \mathcal{B}_n\}\\), where each \\(\mathcal{B}_n = \{b_1, b_2, ..., b_k\}\\) represents a distinct ground-truth LLM benchmark. We define a mapping \\(f: q_i \mapsto b_j\\), with \\(q_i \in \mathcal{Q}\\) and \\(b_j \in \mathcal{B}\\). For each \\(q_i \in \mathcal{Q}\\), we rank similarities between each \\((q_i, b_j)\\) pair and select the most similar \\(b_j\\) that satisfies \\(\theta\\): \\(b_j = f(q_i) = \arg\max_{b_j \in \mathcal{B}} S(q_i, b_j)\\) s.t. \\(\theta\\). We use the dot-product between normalized sentence embeddings as the similarity score \\(S(\cdot)\\). When retrieving the top-1 \\(b_j\\), \\(\theta\\) is a length constraint on the input (or context) field of each \\(b_j\\), addressing the effect of long inputs in the benchmark data mixture. The sentence embeddings of queries are computed using the `all-mpnet-base-v2` model from SentenceTransformers `\citep{reimers2019sentence}`{=latex}. To ensure quality and comprehensive sample coverage, we selected the development and test splits of widely adopted benchmarks from diverse domains and topics.

- General-domain benchmarks: MMLU `\citep{hendrycks2020measuring}`{=latex}, BoolQ `\citep{clark2019boolq}`{=latex}, HellaSwag `\citep{zellers2019hellaswag}`{=latex}, ARC `\citep{clark2018think}`{=latex}, CommonSenseQA `\citep{talmor2018commonsenseqa}`{=latex}, AGIEval `\citep{zhong2023agieval}`{=latex}, OpenbookQA `\citep{mihaylov2018can}`{=latex}, GPQA `\citep{rein2023gpqa}`{=latex}, WinoGrande `\citep{sakaguchi2021winogrande}`{=latex}, TriviaQA `\citep{joshi2017triviaqa}`{=latex}, DROP `\citep{dua2019drop}`{=latex}, and BBH `\citep{suzgun2022challenging}`{=latex}.

- Domain-specific benchmarks: Math: GSM8K `\citep{rein2023gpqa}`{=latex} and MATH `\citep{hendrycksmath2021}`{=latex}; Coding: MBPP `\citep{austin2021program}`{=latex} and HumanEval `\citep{chen2021evaluating}`{=latex}; Physics: PIQA `\citep{bisk2020piqa}`{=latex}; and Social Interactions: SIQA `\citep{sap2019socialiqa}`{=latex}.

According to Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a>, the mixed benchmark `MixEval` exhibits the highest overlap with \\(\mathcal{Q}\\) among all benchmarks, suggesting that \\(\mathcal{B}\\) adequately represents the wild query distribution. Let \\(\mathcal{B}' = \{\mathcal{B}'_1, \mathcal{B}'_2, ..., \mathcal{B}'_n\}\\) denote the mixed benchmark. The distributions of \\(\mathcal{B}\\) and \\(\mathcal{B}'\\) are illustrated in Figure <a href="#fig:data_dist" data-reference-type="ref" data-reference="fig:data_dist">5</a>. A positive correlation between the sizes of the mixed and original benchmarks is observed. Intuitively, a larger benchmark is likely to be retrieved more frequently; however, this is not universally true. Benchmarks with skewed sample distributions, such as HellaSwag, GSM8k, ARC, BBH, MATH, and GPQA, have a smaller relative size after mixing. This indicates that both quantity and distribution influence how frequently a benchmark is retrieved by \\(\mathcal{Q}\\). Overall, the retrieved benchmark splits exhibit a long-tail distribution.

## MixEval-Hard [sec:evalhard]

<div class="tabular" markdown="1">

p2.1cm@C1.3cm@C1.65cm@C1.2cm@C1.65cm@C1.5cm@C1.6cm@C1.5cm@C1.2cm@ & \# Queries & Avg. \# Toks per Query & Avg. \# Inputs & Avg. \# Toks per Input & Min \# Toks per Input & Max \# Toks per Input & English Ratio & Eval Type  
**MixEval**& 4000 & 23 & 0.3 & 41.3 & 6 & 954 & 95.15% &  
**MixEval-Hard**& 1000 & 27.3 & 0.4 & 47.3 & 7 & 954 & 95.22% &  

</div>

Frontier LLMs are rapidly approaching human-level performance across diverse tasks. As these models progress, existing benchmarks will become saturated, hindering differentiation between models. Although `MixEval` reflects typical user queries, it is constrained by the benchmark pool’s overall difficulty. Our results in Table <a href="#tab:model_results_chat" data-reference-type="ref" data-reference="tab:model_results_chat">[tab:model_results_chat]</a> indicate that top models, such as GPT-4 Turbo and Claude 3 Opus, have surpassed 88% accuracy on `MixEval`. To improve the benchmark’s ability to discriminate between very strong models, we extract a challenging subset from `MixEval` to create `MixEval-Hard`.

Given `MixEval` denoted as \\(\mathcal{B}'\\), we sample a hard subset \\(\mathcal{B}''\\) from \\(\mathcal{B}'\\) by computing a difficulty score \\(\xi_i\\) for each entry, prioritizing higher scores. Consider a set of model prediction results \\(\mathcal{A}\\), where \\(\mathcal{A}\\) is a 0-1 matrix of shape \\((N_{model}, N_{\mathcal{B}'})\\), with 1 indicating an incorrect model response. Here, \\(N_{model}\\) is the number of models, and \\(N_{\mathcal{B}'}\\) is the number of questions in \\(\mathcal{B}'\\). The difficulty score \\(\xi_i\\) for a query \\(b_i'\\) is computed by \\(\xi_i = \vec{\mu} \cdot \vec{\mathcal{A}_i}\\), where each model’s result on question \\(i\\) is weighted by its accuracy \\(\mu_j\\) on the dataset. Given \\(\xi = \{\xi_1, \xi_2, ..., \xi_{N_{\mathcal{B}'}}\}\\), we sample from \\(\mathcal{B}'\\) with rejection:

\\[\mathcal{B}'' = \{b_i' \in \mathcal{B}' : p(b_i') \text{ and } \alpha(\mathcal{B}'' \cup \{b_i'\}, \mathcal{B}') \leq \tau\},\\]

where \\(\alpha(x,y)\\) denotes the cluster distance between \\(x\\) and \\(y\\). The probability of drawing \\(b_i'\\), \\(p(b_i') = \frac{e^{\lambda \xi_i}}{\sum_{b_k' \in \mathcal{B}'} e^{\lambda \xi_k}}\\), is based on \\(\xi_i\\). This rejection sampling ensures that `MixEval-Hard` is difficulty-first while maintaining a balanced query distribution. We obtain 1000 samples for `MixEval-Hard`. The statistics of `MixEval` and `MixEval-Hard` are detailed in Table <a href="#tab:b_stats" data-reference-type="ref" data-reference="tab:b_stats">[tab:b_stats]</a>.

## Dynamic Benchmarking

<div class="tabular" markdown="1">

l@C1.6cm@C1.6cm@C1.1cm@C1.1cm@C1.1cm@C1.1cm@C1.8cm@C1.8cm & GPT-3.5-Turbo-0125 & GPT-3.5-Turbo-1106 & Claude 3 Haiku & Mistral-Small & Reka Edge & **Avg.** & **Unique Web Query Ratio** & **Unique **MixEval** Query Ratio**  
**Mean** & 79.66 & 79.25 & 80.32 & 80.57 & 68.42 & 77.64 & &  
**Std.** & 0.26 & 0.28 & 0.34 & 0.56 & 0.35 & 0.36 & &  

</div>

Static benchmarks risk contamination over time as models may overfit to the benchmark data `\citep{yang2023rethinking, chiang2024chatbot, zhang2024careful}`{=latex}, undermining evaluation reliability. To address this, we periodically update the data points in `MixEval` and `MixEval-Hard` using the automatic pipeline described above, i.e., performing benchmark mixtures based on the queries uniformly sampled from the massive web queries detected, which completes updates within one minute. Table <a href="#tab:dynamic_benchmarking" data-reference-type="ref" data-reference="tab:dynamic_benchmarking">[tab:dynamic_benchmarking]</a> shows score stability and version differences. We created five versions of `MixEval` by altering the random seed when sampling web queries and ran five models on them. As shown, the average mean and standard deviation (Std.) for the models across the versions are 77.64 and 0.36, respectively, demonstrating high score stability. For each pair of versions, we compute the unique sample ratio for sampled web queries and benchmark data points. Given samples \\(X = \{x_1, x_2,..., x_n\}\\) from version A and \\(Y = \{y_1, y_2,..., y_n\}\\) from version B, the unique sample ratio \\(\mathcal{R}\\) is calculated as \\(\mathcal{R} = \frac{|X-Y|+|Y-X|}{X \cup Y}\\), representing the unique ratio of the \\(X \cup Y\\) set. The average unique web query ratio across all version pairs is 99.71%, and the unique ratio for `MixEval` versions is 85.05%, indicating significant differences between versions. This efficient updating mechanism, alongside stable model scores and significant data point variations, effectively mitigates benchmark contamination. Additionally, we plan to dynamically expand our benchmark pool with newly released benchmarks to further enhance the mixed benchmark distribution.

To summarize, we update the data points of MixEval via (1) batch web query update (sampling different web queries batches from the crawled web queries), (2) source web query update (updating all the web queries with the latest Common Crawl) or (3) benchmark pool update (incorporating new ground-truth-based benchmarks to the benchmark pool). Since the mechanism of MixEval is to match web queries with benchmark pool samples, the above three updating methods refreshes both the web queries (the first and the second method) and benchmark pool samples (the third method).

# Results

<div class="tabular" markdown="1">

@p3cm@C1cm@C1cm@C0.8cm@C1cm@C1cm@C1cm@C1.2cm@C1cm@C1cm@C1cm@C0.9cm@ & MixEval-Hard & MixEval & Arena Elo (0527) & TriviaQA (Mixed) & MMLU (Mixed) & DROP (Mixed) & HellaSwag (Mixed) & Common-senseQA (Mixed) & TriviaQA-Hard (Mixed) & MMLU-Hard (Mixed) & DROP-Hard (Mixed)  
Proportion & 100% & 100% & - & 31.2% & 21.4% & 12.4% & 7.4% & 5.3% & 26.6% & 23.1% & 16.7%  
GPT-4o & **64.7** & 87.9 & **1287** & 88.0 & **85.4** & 87.9 & **94.3** & 86.8 & 70.3 & **57.1** & 67.5  
Claude 3 Opus & <u>6</u>3.5 & <u>8</u>8.1 & 1248 & <u>9</u>0.4 & <u>8</u>3.2 & **91.5** & 93.3 & 87.7 & <u>7</u>1.4 & <u>5</u>5.0 & **75.2**  
GPT-4-Turbo & 62.6 & **88.8** & 1256 & **91.2** & 82.8 & <u>9</u>1.0 & 92.6 & 85.4 & **73.1** & 45.5 & 71.0  
Gemini 1.5 Pro & 58.7 & 84.2 & <u>1</u>258 & 85.3 & 79.2 & 84.2 & 89.2 & 84.4 & 67.8 & 44.6 & 64.8  
Yi-Large & 56.8 & 84.4 & 1239 & 81.7 & 80.9 & 87.0 & 92.6 & **90.1** & 55.4 & 48.5 & 63.1  
LLaMA-3-70B-Instruct & 55.9 & 84.0 & 1208 & 83.1 & 80.5 & 90.1 & 81.8 & 83.0 & 60.5 & 46.3 & <u>7</u>4.5  
Qwen-Max-0428 & 55.8 & 86.1 & 1184 & 86.7 & 80.6 & 85.4 & <u>9</u>3.6 & <u>8</u>8.2 & 61.5 & 41.6 & 53.5  
Claude 3 Sonnet & 54.0 & 81.7 & 1201 & 84.2 & 74.7 & 87.7 & 85.9 & 82.5 & 59.1 & 40.7 & 66.9  
Reka Core & 52.9 & 83.3 & - & 82.8 & 79.3 & 88.1 & 88.6 & 81.6 & 51.6 & 46.3 & 66.6  
MAmmoTH2-8x7B-Plus & 51.8 & 81.5 & - & 83.0 & 74.5 & 85.7 & 82.2 & 82.5 & 52.9 & 41.1 & 65.1  
DeepSeek-V2 & 51.7 & 83.7 & - & 84.4 & 77.3 & 85.3 & 88.2 & 84.0 & 51.7 & 42.0 & 62.8  
Command R+ & 51.4 & 81.5 & 1189 & 83.3 & 78.9 & 80.4 & 83.5 & 82.1 & 57.5 & 42.0 & 65.0  
Yi-1.5-34B-Chat & 51.2 & 81.7 & - & 78.4 & 76.4 & 87.0 & 90.2 & 86.8 & 44.4 & 38.1 & 67.4  
Mistral-Large & 50.3 & 84.2 & 1156 & 88.3 & 80.2 & 88.6 & 65.0 & 83.5 & 55.5 & 42.4 & 61.6  
Qwen1.5-72B-Chat & 48.3 & 84.1 & 1147 & 83.9 & 80.1 & 85.1 & 87.9 & 86.3 & 49.9 & 37.7 & 56.5  
Mistral-Medium & 47.8 & 81.9 & 1148 & 86.8 & 76.3 & 83.2 & 72.4 & 82.5 & 59.8 & 38.5 & 47.1  
Gemini 1.0 Pro & 46.4 & 78.9 & 1131 & 81.0 & 74.9 & 82.6 & 74.7 & 80.2 & 58.2 & 35.5 & 54.1  
Reka Flash & 46.2 & 79.8 & 1148 & 76.4 & 75.4 & 86.7 & 90.6 & 80.7 & 42.9 & 34.6 & 65.0  
Mistral-Small & 46.2 & 81.2 & - & 85.1 & 75.2 & 86.1 & 73.4 & 77.8 & 56.0 & 33.8 & 52.6  
LLaMA-3-8B-Instruct & 45.6 & 75.0 & 1153 & 71.7 & 71.9 & 86.4 & 65.7 & 78.3 & 40.2 & 40.7 & 67.6  
Command R & 45.2 & 77.0 & 1147 & 80.9 & 75.0 & 72.0 & 75.8 & 77.4 & 57.0 & 39.0 & 42.0  
Qwen1.5-32B-Chat & 43.3 & 81.0 & 1126 & 75.7 & 78.0 & 82.9 & 85.9 & 88.2 & 39.1 & 29.9 & 54.4  
GPT-3.5-Turbo & 43.0 & 79.7 & 1102 & 85.2 & 74.5 & 84.8 & 63.0 & 81.6 & 46.4 & 35.1 & 55.4  
Claude 3 Haiku & 42.8 & 79.7 & 1178 & 79.9 & 76.1 & 85.0 & 75.8 & 78.8 & 42.4 & 30.7 & 51.5  
Yi-34B-Chat & 42.6 & 80.1 & 1111 & 82.7 & 73.6 & 86.1 & 86.9 & 78.8 & 41.5 & 29.9 & 57.1  
Mixtral-8x7B-Instruct-v0.1 & 42.5 & 76.4 & 1114 & 82.5 & 72.0 & 79.5 & 54.2 & 77.4 & 48.5 & 37.2 & 47.7  
Starling-LM-7B-beta & 41.8 & 74.8 & 1119 & 75.1 & 69.0 & 86.4 & 48.5 & 84.9 & 33.4 & 34.2 & 62.9  
Yi-1.5-9B-Chat & 40.9 & 74.2 & - & 61.3 & 72.6 & 83.9 & 86.5 & 82.5 & 23.3 & 36.8 & 61.3  
Gemma-1.1-7B-IT & 39.1 & 69.6 & 1084 & 64.3 & 66.9 & 80.6 & 66.3 & 73.6 & 30.3 & 39.0 & 55.1  
Vicuna-33B-v1.3 & 38.7 & 66.3 & 1090 & 79.2 & 59.2 & 71.4 & 30.3 & 61.8 & 42.5 & 39.4 & 36.6  
LLaMA-2-70B-Chat & 38.0 & 74.6 & 1093 & 80.0 & 69.8 & 79.8 & 67.3 & 74.1 & 42.2 & 27.7 & 42.2  
Mistral-7B-Instruct-v0.2 & 36.2 & 70.0 & 1072 & 73.7 & 67.3 & 72.8 & 54.2 & 66.0 & 33.5 & 29.4 & 44.3  
Qwen1.5-7B-Chat & 35.5 & 71.4 & 1069 & 64.1 & 68.7 & 76.4 & 76.1 & 82.1 & 29.0 & 29.0 & 50.0  
Reka Edge & 32.2 & 68.5 & - & 60.0 & 63.6 & 80.0 & 74.7 & 80.7 & 18.6 & 26.4 & 56.9  
Zephyr-7B-\\(\beta\\) & 31.6 & 69.1 & 1054 & 74.7 & 64.9 & 77.3 & 39.1 & 69.3 & 30.2 & 24.2 & 45.3  
LLaMA-2-7B-Chat & 30.8 & 61.7 & 1037 & 68.8 & 59.4 & 69.3 & 35.7 & 61.3 & 24.8 & 30.3 & 44.3  
Yi-6B-Chat & 30.1 & 65.6 & - & 66.1 & 65.4 & 70.5 & 52.5 & 69.8 & 18.9 & 26.8 & 43.7  
Qwen1.5-MoE-A2.7B-Chat & 29.1 & 69.1 & - & 65.9 & 69.5 & 64.6 & 72.7 & 81.1 & 21.9 & 26.8 & 39.5  
Gemma-1.1-2B-IT & 28.4 & 51.9 & 1019 & 53.7 & 51.5 & 59.8 & 26.6 & 57.1 & 31.9 & 30.3 & 27.8  
Vicuna-7B-v1.5 & 27.8 & 60.3 & 1004 & 66.4 & 58.7 & 68.3 & 24.9 & 62.7 & 25.9 & 23.4 & 33.2  
OLMo-7B-Instruct & 26.7 & 55.0 & 1015 & 51.7 & 57.1 & 53.1 & 55.9 & 64.6 & 24.7 & 27.3 & 22.9  
Qwen1.5-4B-Chat & 24.6 & 57.2 & 988 & 46.0 & 61.4 & 57.2 & 54.9 & 74.1 & 16.5 & 17.3 & 28.6  
JetMoE-8B-Chat & 24.3 & 51.6 & - & 46.8 & 58.5 & 27.0 & 86.2 & 68.4 & 19.2 & 25.5 & 11.5  
MPT-7B-Chat & 23.8 & 43.8 & 927 & 50.2 & 37.8 & 50.0 & 25.6 & 36.3 & 17.5 & 24.7 & 31.0  

</div>

## Experiment Settings [sec:exp_settings]

We evaluate models on `MixEval` and `MixEval-Hard` using the Transformers library `\citep{wolf2019huggingface}`{=latex} for open-source models, adhering to the official settings in their Hugging Face model card. Proprietary models are assessed via their official API endpoints, using the latest versions as of April 30, 2024[^6]. Chat models employ official chat templates or FastChat chat templates `\citep{zheng2024judging}`{=latex}, and base models are evaluated in a 5-shot setting. Both `MixEval` and `MixEval-Hard`, comprising samples from various benchmarks, demonstrate the inadequacies of traditional rule-based parsing methods across all benchmarks and models. To improve parsing accuracy, we use GPT-3.5-Turbo-0125 as the model parser to either score the response (free-form problems) or extract the model’s choice (multiple-choice problems). The stability of the GPT-3.5 Turbo parser is evidenced in Table <a href="#tab:dynamic_benchmarking" data-reference-type="ref" data-reference="tab:dynamic_benchmarking">[tab:dynamic_benchmarking]</a> of this paper and Table 4 of `\cite{zhang2024direct}`{=latex}. We will also provide an open-source model parser with its stability test to ensure long-term reproducibility. Section <a href="#parser_prompts" data-reference-type="ref" data-reference="parser_prompts">13</a> details the model parser prompts, and Section <a href="#rule_model_parser_comparison" data-reference-type="ref" data-reference="rule_model_parser_comparison">12</a> compares the model parser to the rule parser. Models are evaluated on 4 or 8 A100 GPUs. All correlations with Arena Elo are based on the Chatbot Arena Leaderboard as of May 1, 2024. We update the Arena Elo scores in Table <a href="#tab:model_results_chat" data-reference-type="ref" data-reference="tab:model_results_chat">[tab:model_results_chat]</a> to the latest version (May 27, 2024).

## Evaluation Results

****MixEval** and **MixEval-Hard** Leaderboard** Table <a href="#tab:model_results_chat" data-reference-type="ref" data-reference="tab:model_results_chat">[tab:model_results_chat]</a> presents the detailed evaluation results on `MixEval`, `MixEval-Hard`, and their main subsets. GPT-4o, Claude 3 Opus, and GPT-4 Turbo consistently achieve the highest performance across almost all splits. Gemini 1.5 Pro ranks next, followed closely by Yi-Large, LLaMA-3-70B-Instruct, and Qwen-Max-0428. Notably, the first four frontier models also support multi-modal input understanding. The LLaMA-3-8B-Instruct model is the top-performing 7B model, outperforming some of the latest large models, such as Command R (35B) and Qwen1.5-32B-Chat (32B). Proprietary models generally outperform open-source models.

<figure id="fig:arena_mixevalandmixevalhard">
<figure id="fig:arena_mixeval">
<img src="./figures/arena_mixeval.png"" style="width:80.0%" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure id="fig:arena_mixevalhard">
<img src="./figures/arena_mixevalhard.png"" style="width:80.0%" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figcaption>The model scores of <code>MixEval</code> and <code>MixEval-Hard</code> scale linearly with Chatbot Arena Elo. By fitting the data points, the Arena Elo score can be roughly estimated given a model score on <code>MixEval</code> or <code>MixEval-Hard</code>. <span class="math inline"><em>ρ</em></span> and <span class="math inline"><em>e</em></span> denote the Spearman’s ranking correlation and the root mean square error of the linear fit respectively.</figcaption>
</figure>

**Linear Relationship with Arena Elo ** Figure <a href="#fig:arena_mixevalandmixevalhard" data-reference-type="ref" data-reference="fig:arena_mixevalandmixevalhard">8</a> presents the model scores on `MixEval` and `MixEval-Hard` plotted against the Arena Elo, with each model represented as a point. Interestingly, the scores on `MixEval` and `MixEval-Hard` exhibit a linear relationship with the Arena Elo score. This indicates that `MixEval` and `MixEval-Hard`, beyond their high correlation with Arena Elo, can approximate a model’s Arena Elo score based on its `MixEval` or `MixEval-Hard` scores. Nonetheless, this estimation remains approximate due to the presence of outliers, as depicted in the figure.

**Cost-effectiveness of Models ** Figure <a href="#fig:score_scatter_plot" data-reference-type="ref" data-reference="fig:score_scatter_plot">11</a> compares the models in Table <a href="#tab:model_results_chat" data-reference-type="ref" data-reference="tab:model_results_chat">[tab:model_results_chat]</a> in terms of cost-effectiveness. Figure <a href="#fig:score_scatter_plot1" data-reference-type="ref" data-reference="fig:score_scatter_plot1">9</a> examines the relationship between activated parameters and performance for open-source LLMs, while Figure <a href="#fig:score_scatter_plot2" data-reference-type="ref" data-reference="fig:score_scatter_plot2">10</a> compares API price against performance for frontier proprietary LLMs. Both figures exhibit a roughly log-linear relationship between performance and the x-axis metric. In Figure <a href="#fig:score_scatter_plot1" data-reference-type="ref" data-reference="fig:score_scatter_plot1">9</a>, the MAmmoTH2, Llama-3, and Yi series stand out as the most performant and parameter-efficient among open-source models. The MoE models, such as MAmmoTH2, Mixtral-8x7B-Instruct-v0.1, Qwen1.5-MoE-A2.7B-Chat, and JetMoE-8B-Chat, demonstrate superior parameter efficiency. The proprietary data points reveal a clearer log-linear pattern. GPT-4o is more cost-effective than Claude 3 Opus, offering better performance at 20% the price. Notably, DeepSeek V2 is the most cost-effective model. The Gemini series exhibits similar cost-effectiveness to the GPT series, while the Reka series parallels the cost-effectiveness of the Claude series. We conduct detailed error analysis in Section <a href="#sec:error_analysis" data-reference-type="ref" data-reference="sec:error_analysis">4.6</a> to compare error rates of open-source and proprietary models on different `MixEval` splits. We also showcase the error responses of frontier models in Section <a href="#sec:error_cases" data-reference-type="ref" data-reference="sec:error_cases">11</a> to identify their potential weaknesses.

<figure id="fig:score_scatter_plot">
<figure id="fig:score_scatter_plot1">
<img src="./figures/scatter_plot_opensource.png"" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure id="fig:score_scatter_plot2">
<img src="./figures/scatter_plot_api.png"" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figcaption>Activated parameters and API price per performance of open-source and proprietary models.</figcaption>
</figure>

## Effectiveness of MixEval

<figure id="fig:subset_corr_breakdown">
<figure id="fig:subset_corr_breakdown1">
<img src="./figures/corr_breakdown_arena_elo.png"" style="width:90.0%" />
<figcaption>Correlation with Arena Elo (%)</figcaption>
</figure>
<figure id="fig:subset_corr_breakdown2">
<img src="./figures/corr_breakdown_arena_elo_en.png"" style="width:90.0%" />
<figcaption>Correlation with Arena Elo (En) (%)</figcaption>
</figure>
<figcaption>Our approach improves the correlation with Arena Elo and Arena Elo (En) for all the main splits of <code>MixEval</code> and outperforms benchmark-level and uniform mixture.</figcaption>
</figure>

****MixEval** and **MixEval-Hard** achieve the highest correlations with Arena Elo and Arena Elo (En) among all benchmarks.** As shown in Figures <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a> and <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a>, `MixEval` and `MixEval-Hard`, derived from the proposed `MixEval` pipeline to simulate diverse user queries, achieve significantly higher correlations (10% higher than the top SOTA benchmark) with human preferences (both Arena Elo and Arena Elo (En)), ranking second and first, respectively. Notably, `MixEval-Hard`’s correlation with Arena Elo is even slightly higher than the correlation between Arena Elo (En) and Arena Elo. As discussed in Section <a href="#subsec:what_affects_corrs" data-reference-type="ref" data-reference="subsec:what_affects_corrs">4.4</a>, query difficulty impacts human preference correlation. Therefore, `MixEval-Hard`’s superior correlation may partially result from increased query difficulty. The high correlations of `MixEval` and `MixEval-Hard` with human preferences enable both efficient and reliable model ranking compared to large-scale user-facing benchmarks.

****MixEval** improves the correlation with Arena Elo and Arena Elo (En) across all its main benchmark splits.** In Figure <a href="#fig:subset_corr_breakdown" data-reference-type="ref" data-reference="fig:subset_corr_breakdown">14</a>, we select the top-10 benchmarks from our pool with sufficient sample sizes (see sample number distribution in Figure <a href="#fig:data_dist" data-reference-type="ref" data-reference="fig:data_dist">5</a>). For each benchmark, we present (1) the correlation between Arena Elo and the original benchmark, and (2) the correlation between Arena Elo and the `MixEval`-mixed version. Remarkably, **all** benchmarks exhibit significant improvements in their correlations with Arena Elo after being processed by `MixEval`. The correlation increase is notably high (\>40%) in benchmarks such as BoolQ, AGIEval, SIQA, and PIQA. `MixEval` and `MixEval-Hard`, which aggregate all benchmarks, consistently outperform any individual benchmark mixture, underscoring the importance of a large benchmark pool and query comprehensiveness.

****MixEval** outperforms both benchmark-level and uniform mixtures.** Figure <a href="#fig:subset_corr_breakdown" data-reference-type="ref" data-reference="fig:subset_corr_breakdown">14</a> illustrates the correlations with Arena Elo for benchmark-level and uniform mixtures. The benchmark-level mixture samples questions uniformly from each benchmark, proportional to its split size in `MixEval`. The uniform mixture samples an equal number of questions from all benchmarks. Both methods yield significantly lower human preference correlations than `MixEval` and `MixEval-Hard`. Furthermore, the benchmark-level mixture offers negligible improvement over the uniform mixture. These findings underscore the importance of an appropriate sample-level mixture, as implemented by `MixEval`.

****MixEval** effectively maps real-world user queries to ground-truth-based benchmarks.** Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a> shows the query distributions of leading benchmarks. Both `MixEval` and `MixEval-Hard` closely resemble web queries and popular wild datasets, highlighting `MixEval`’s efficacy in aligning benchmark query distributions with real-world data. The maps in Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a> are ordered by their cluster distances to our identified web queries, showing that wild datasets align more closely with our web queries than other LLM benchmarks. This underscores the robustness of our web query detection pipeline and the solid grounding of `MixEval`. Additionally, as discussed in Section <a href="#sec:llm_benchmark_analysis" data-reference-type="ref" data-reference="sec:llm_benchmark_analysis">2</a>, ShareGPT, with a larger user base (100M) compared to other wild datasets (0.1M-0.2M), shows the highest similarity to our web queries, which are based on a global internet user population (5.4B), further validating the accuracy of our web query detection.

## What Affects the Correlations between Benchmarks? [subsec:what_affects_corrs]

<figure id="fig:heatmap_main">
<img src="./figures/heatmap.png"" style="width:100.0%" />
<figcaption>The correlation matrix for benchmarks. <code>MixEval</code> and <code>MixEval-Hard</code> achieve the highest correlations with Chatbot Arena Elo. Each value of the heatmap represents the Spearman’s rank correlation (%) between the model rankings of the corresponding benchmark pairs, where a <span style="color: red">warmer</span> color indicates a higher correlation and a <span style="color: blue">cooler</span> color indicates a lower correlation. The <u>underlined</u> numbers indicate the data for the corresponding benchmark pairs are insufficient (&lt;15 models). The detailed statistics on the number of models used for each pair of benchmarks are presented in Figure <a href="#fig:heatmap_stats" data-reference-type="ref" data-reference="fig:heatmap_stats">18</a>.</figcaption>
</figure>

**Comprehensiveness and other features, such as difficulty and density, impact correlation with large-scale user-facing Benchmarks.** As shown in Figure <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a>, general-domain benchmarks typically exhibit a higher correlation with human preference compared to domain-specific benchmarks, highlighting the importance of query comprehensiveness. However, comprehensiveness is not the sole factor. Three observations support this: (1) Benchmarks like GSM8K, despite their skewed distributions (Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a>), achieve a high correlation (0.78) with human preference, while others with high topic overlap with real-world queries, such as BoolQ, achieve a low correlation (0.37). (2) ARC-e and ARC-c, despite similar topic distributions, show significantly different correlations (Figure <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a>), likely due to varying difficulty levels. This indicates that other query features, such as difficulty, are critical to correlation with human preference. (3) As shown in Figure <a href="#fig:subset_corr_breakdown" data-reference-type="ref" data-reference="fig:subset_corr_breakdown">14</a>, `MixEval` increases the correlation for each individual benchmark through benchmark mixture. For an individual benchmark, the queries become less comprehensive post-mixture since the mixed version represents a subset of the original; thus, the correlation gain is not due to a more comprehensive query distribution. These observations suggest that correlation gains with human preference are influenced by factors beyond sole comprehensiveness, possibly including nuanced factors such as query difficulty and density, which can be refined with the proposed benchmark mixture approach.

**Benchmarks that are highly correlated with human preferences also tend to be correlated with each other, whereas those that are less correlated with human preferences are similarly less correlated with most other benchmarks.** The heat map reveals a consistent red region in the top-left, signifying high correlation, while the rest of the map is predominantly blue and inconsistent, indicating low correlation. This suggests that model rankings on benchmarks closely aligned with human preferences are more stable and reflect a "True" ranking, whereas the remaining benchmarks exhibit greater variability in model rankings.

**Benchmarks within the same domain exhibit higher correlations.** Despite a low correlation with human preferences, some domain-specific benchmarks demonstrate a relatively high correlation with other benchmarks in the same domain. For instance, MBPP shows a correlation of only 0.28 with human preference but a substantial 0.83 with HumanEval. Similarly, MATH has a correlation of 0.50 with human preference yet presents a 0.66 correlation with GSM8K. Furthermore, ARC-e has a correlation of 0.58 with human preference while achieving a notable 0.84 correlation with ARC-c.

<figure id="fig:compare_base_chat">
<div class="center">
<img src="./figures/model_scores_histogram_chatbase_comparison.png"" style="width:90.0%" />
</div>
<figcaption>The performance of chat and base models of the same model series in Table <a href="#tab:model_results_chat" data-reference-type="ref" data-reference="tab:model_results_chat">[tab:model_results_chat]</a>. Chat and base model scores show a high correlation.</figcaption>
</figure>

## What Do Humans Prefer?

The user-facing evaluation of LLMs, based on human preferences, assesses two main aspects: (1) model capability, optimized mainly during pre-training, and (2) non-capability attributes like toxicity and helpfulness, refined during post-training. We explore whether human preferences for models can be predicted before post-training, leading to the question: which aspects, the capabilities obtained in pre-training or those non-capability attributes obtained in post-training, are more preferred by humans? We evaluated the base versions of the model series in Table <a href="#tab:model_results_chat" data-reference-type="ref" data-reference="tab:model_results_chat">[tab:model_results_chat]</a>. Notably, the scores in Figure <a href="#fig:compare_base_chat" data-reference-type="ref" data-reference="fig:compare_base_chat">16</a> show a 0.95 correlation between base and chat models, indicating `MixEval`’s potential to approximate human preferences pre-post-training. This implies that the model capabilities obtained in pre-training may have a greater impact on human preferences compared to those obtained in post-training. However, we also observe that the post-training has more impact on some smaller models, all of which went through heavy supervised post-training.

## Error Analysis [sec:error_analysis]

<figure id="fig:error_rate_benchmark">
<div class="center">
<img src="./figures/error_rate_distribution.png"" style="width:90.0%" />
</div>
<figcaption>Averaged error rates of open-source, proprietary, and all models on <code>MixEval</code> splits.</figcaption>
</figure>

Figure <a href="#fig:error_rate_benchmark" data-reference-type="ref" data-reference="fig:error_rate_benchmark">17</a> illustrates the averaged error rates of the models evaluated on the main splits of `MixEval`. We separately compute the error rates for proprietary and open-source models to facilitate comparison. Both model types exhibit significant errors on the AGIEval split of `MixEval`, underscoring its difficulty. In contrast, performance on the PIQA split is generally saturated. Notably, there is a substantial performance gap between proprietary and open-source models on the GSM8K split, with considerable gaps also observed on the HellaSwag, TriviaQA, and DROP splits.

In Section <a href="#sec:error_cases" data-reference-type="ref" data-reference="sec:error_cases">11</a>, we conduct case studies to examine the error cases made by frontier proprietary models. For each case, we present the incorrect responses from each model. We identify three primary causes of confusion for these models: strong domain knowledge, complex reasoning, and vague question definitions. Additionally, we identify several annotation issues within current benchmarks, though these are negligible in number.

# Related Work

## LLM Benchmarking

Both frontier and open-source LLMs have made significant strides in recent years. Evaluation scores are a core objective in LLM development, necessitating an effective evaluation pipeline for successful model advancement. Current LLM evaluation methods can be categorized into three main types: (1) ground-truth-based evaluation, (2) LLM-as-judge evaluation, and (3) user-facing evaluation.

Ground-truth-based evaluation, or closed-ended evaluation, involves ranking the outputs of base and chat LLMs against predefined correct answers. Various benchmarks have been introduced by the research community for this purpose `\citep{hendrycks2020measuring, cobbe2021training, rein2023gpqa, clark2019boolq, zellers2019hellaswag, clark2018think, talmor2018commonsenseqa, zhong2023agieval, mihaylov2018can, sakaguchi2021winogrande, dua2019drop, suzgun2022challenging, austin2021program, chen2021evaluating, bisk2020piqa, sap2019socialiqa}`{=latex}. These benchmarks facilitate rapid and straightforward LLM evaluation, providing clear and unbiased answer judgments due to their closed-ended nature. In addition, `\cite{huang2024compression}`{=latex} find that averaged ground-truth-based benchmark scores scale linearly with the models’ compression efficiency evaluated with bits per character (BPC), suggesting ground-truth-based benchmarks also quantitatively reflect models’ abilities of text corpora compression. However, ground-truth-based benchmarks often exhibit query bias (as illustrated in Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a>) and may not accurately represent the nuance and diversity of real-world user queries, limiting their ability to assess the nuanced capabilities of LLMs.

On the other hand, two other categories of evaluation approaches primarily focus on the open-ended evaluations of chat LLMs. The LLM-as-judge evaluation uses frontier models to rank the responses to a set of open-ended queries without ground-truths. These queries are either manually designed `\citep{zheng2024judging}`{=latex}, model generated `\citep{dubois2024alpacafarm, dubois2024length}`{=latex}, or sourced from crowdsourcing platforms `\citep{tianle2024from, wildbench2024}`{=latex}. However, due to the high cost of using frontier models as judges, such approaches are not scalable to a large number of user queries. This limitation hinders the ability to reflect the complexity and diversity of real-world queries and may deviate from the true distribution (see Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a>). Additionally, previous research has identified several biases in frontier model judges, including verbosity bias, position bias, and self-enhancement bias `\citep{zheng2024judging}`{=latex}. These biases can lead to unfair model rankings in practical evaluations. Additionally, the static nature of both ground-truth-based and LLM-as-judge benchmarks results in contamination over time, diminishing the reliability of evaluation outcomes. Some studies address the contamination issue by dynamically updating benchmark queries. However, these are either LLM-as-judge benchmarks `\citep{tianle2024from, wildbench2024}`{=latex} or domain-specific ground-truth-based benchmarks `\citep{fan2023nphardeval, jain2024livecodebench}`{=latex}. In contrast, `MixEval` is a general-domain dynamic benchmark with ground-truth answers, benefiting from a rapid and stable data updating mechanism, exhibiting a low score standard deviation of 0.36 (on a 0-100 scale) between versions.

As a comparison, Chatbot Arena `\citep{chiang2024chatbot}`{=latex} serves as a robust benchmark for evaluating chat LLMs. It operates as a benchmarking platform where anonymous, randomized battles are conducted in a crowdsourced environment. The platform’s extensive real-world user queries and preferences provide comprehensive and less biased evaluations, ensuring the accuracy and stability of model rankings. Additionally, its real-time nature prevents models from overfitting the benchmark, thereby avoiding contamination issues. However, obtaining a stable model score requires more than thousands of rounds of human interactions and several days, making the process labor-intensive, slow, and expensive (Figure <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a>). Furthermore, its open-ended nature limits its ability to evaluate base models.

## Web Query Detection

Currently, real-world text-in-text-out user queries are primarily sourced from chat platforms `\citep{chiang2024chatbot, zheng2024judging, zhao2024wildchat, sharegpt2023sharegpt}`{=latex}. Our concurrent work, MAmmoTH2 `\citep{yue2024mammoth2}`{=latex}, also identifies real-world user queries from the web. However, MAmmoTH2 has fundamentally different objectives compared to `MixEval`. MAmmoTH2 focuses on detecting large-scale domain-specific query-answer pairs, while `MixEval` targets general-purpose user queries that accurately reflect the real-world user query distribution. This difference in objectives results in distinct web query detection pipelines.

# Conclusion

In this paper, we present `MixEval`, an approach that bridges real-world queries and ground-truth-based evaluation by mining user queries from the web and matching them with similar benchmark queries. `MixEval` and its hard variant can offer accurate evaluations that highly align with Chatbot Arena. `MixEval` operates locally and rapidly, eliminating the need for slow, costly human preference data collection or biased model judgment. `MixEval`’s data points can be stably updated within one minute, mitigating benchmark contamination. We thereby effectively mitigate the query, grading, and generalization biases in LLM evaluation through the proposed benchmark mixture pipeline, while maintaining high efficiency. Our meta-evaluation and extensive analysis of `MixEval` and other popular LLM benchmarks demonstrate `MixEval`’s effectiveness, providing insights to enhance the community’s understanding of LLM evaluation.

# Acknowledgement [acknowledgement]

We thank Yao Fu, Balázs Galambosi, Jason Phang, Jason Wei, Piotr Nawrot, Luca Soldaini, Guanzhi Wang, Deepanway Ghosal, Bo Li, Junhao Zhang, Yifan Song, Zangwei Zheng, Zian Zheng, Qinghong Lin, Wenhu Chen, Bill Yuchen Lin, and colleagues from CMU NeuLab for insightful discussions and pointers.

# References [references]

<div class="thebibliography" markdown="1">

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al Program synthesis with large language models *arXiv preprint arXiv:2108.07732*, 2021. **Abstract:** This paper explores the limits of the current generation of large language models for program synthesis in general purpose programming languages. We evaluate a collection of such models (with between 244M and 137B parameters) on two new benchmarks, MBPP and MathQA-Python, in both the few-shot and fine-tuning regimes. Our benchmarks are designed to measure the ability of these models to synthesize short Python programs from natural language descriptions. The Mostly Basic Programming Problems (MBPP) dataset contains 974 programming tasks, designed to be solvable by entry-level programmers. The MathQA-Python dataset, a Python version of the MathQA benchmark, contains 23914 problems that evaluate the ability of the models to synthesize code from more complex text. On both datasets, we find that synthesis performance scales log-linearly with model size. Our largest models, even without finetuning on a code dataset, can synthesize solutions to 59.6 percent of the problems from MBPP using few-shot learning with a well-designed prompt. Fine-tuning on a held-out portion of the dataset improves performance by about 10 percentage points across most model sizes. On the MathQA-Python dataset, the largest fine-tuned model achieves 83.8 percent accuracy. Going further, we study the model’s ability to engage in dialog about code, incorporating human feedback to improve its solutions. We find that natural language feedback from a human halves the error rate compared to the model’s initial prediction. Additionally, we conduct an error analysis to shed light on where these models fall short and what types of programs are most difficult to generate. Finally, we explore the semantic grounding of these models by fine-tuning them to predict the results of program execution. We find that even our best models are generally unable to predict the output of a program given a specific input. (@austin2021program)

Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling The arcade learning environment: An evaluation platform for general agents *Journal of Artificial Intelligence Research*, 47: 253–279, 2013. **Abstract:** In this article we introduce the Arcade Learning Environment (ALE): both a challenge problem and a platform and methodology for evaluating the development of general, domain-independent AI technology. ALE provides an interface to hundreds of Atari 2600 game environments, each one different, interesting, and designed to be a challenge for human players. ALE presents significant research challenges for reinforcement learning, model learning, model-based planning, imitation learning, transfer learning, and intrinsic motivation. Most importantly, it provides a rigorous testbed for evaluating and comparing approaches to these problems. We illustrate the promise of ALE by developing and benchmarking domain-independent agents designed using well-established AI techniques for both reinforcement learning and planning. In doing so, we also propose an evaluation methodology made possible by ALE, reporting empirical results on over 55 different games. All of the software, including the benchmark agents, is publicly available. (@bellemare2013arcade)

Emily M Bender and Batya Friedman Data statements for natural language processing: Toward mitigating system bias and enabling better science *Transactions of the Association for Computational Linguistics*, 6: 587–604, 2018. **Abstract:** In this paper, we propose data statements as a design solution and professional practice for natural language processing technologists, in both research and development. Through the adoption and widespread use of data statements, the field can begin to address critical scientific and ethical issues that result from the use of data from certain populations in the development of technology for other populations. We present a form that data statements can take and explore the implications of adopting them as part of regular practice. We argue that data statements will help alleviate issues related to exclusion and bias in language technology, lead to better precision in claims about how natural language processing research can generalize and thus better engineering results, protect companies from public embarrassment, and ultimately lead to language technology that meets its users in their own preferred linguistic style and furthermore does not misrepresent them to others. (@bender2018data)

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al Piqa: Reasoning about physical commonsense in natural language In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 7432–7439, 2020. **Abstract:** To apply eyeshadow without a brush, should I use a cotton swab or a toothpick? Questions requiring this kind of physical commonsense pose a challenge to today’s natural language understanding systems. While recent pretrained models (such as BERT) have made progress on question answering over more abstract domains – such as news articles and encyclopedia entries, where text is plentiful – in more physical domains, text is inherently limited due to reporting bias. Can AI systems learn to reliably answer physical commonsense questions without experiencing the physical world?In this paper, we introduce the task of physical commonsense reasoning and a corresponding benchmark dataset Physical Interaction: Question Answering or PIQA. Though humans find the dataset easy (95% accuracy), large pretrained models struggle (∼75%). We provide analysis about the dimensions of knowledge that existing models lack, which offers significant opportunities for future research. (@bisk2020piqa)

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al Evaluating large language models trained on code *arXiv preprint arXiv:2107.03374*, 2021. **Abstract:** We introduce Codex, a GPT language model fine-tuned on publicly available code from GitHub, and study its Python code-writing capabilities. A distinct production version of Codex powers GitHub Copilot. On HumanEval, a new evaluation set we release to measure functional correctness for synthesizing programs from docstrings, our model solves 28.8% of the problems, while GPT-3 solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling from the model is a surprisingly effective strategy for producing working solutions to difficult prompts. Using this method, we solve 70.2% of our problems with 100 samples per problem. Careful investigation of our model reveals its limitations, including difficulty with docstrings describing long chains of operations and with binding operations to variables. Finally, we discuss the potential broader impacts of deploying powerful code generation technologies, covering safety, security, and economics. (@chen2021evaluating)

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality *See https://vicuna. lmsys. org (accessed 14 April 2023)*, 2 (3): 6, 2023. (@chiang2023vicuna)

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E Gonzalez, et al Chatbot arena: An open platform for evaluating llms by human preference *arXiv preprint arXiv:2403.04132*, 2024. **Abstract:** Large Language Models (LLMs) have unlocked new capabilities and applications; however, evaluating the alignment with human preferences still poses significant challenges. To address this issue, we introduce Chatbot Arena, an open platform for evaluating LLMs based on human preferences. Our methodology employs a pairwise comparison approach and leverages input from a diverse user base through crowdsourcing. The platform has been operational for several months, amassing over 240K votes. This paper describes the platform, analyzes the data we have collected so far, and explains the tried-and-true statistical methods we are using for efficient and accurate evaluation and ranking of models. We confirm that the crowdsourced questions are sufficiently diverse and discriminating and that the crowdsourced human votes are in good agreement with those of expert raters. These analyses collectively establish a robust foundation for the credibility of Chatbot Arena. Because of its unique value and openness, Chatbot Arena has emerged as one of the most referenced LLM leaderboards, widely cited by leading LLM developers and companies. Our demo is publicly available at \\}url{https://chat.lmsys.org}. (@chiang2024chatbot)

Davide Chicco and Giuseppe Jurman The advantages of the matthews correlation coefficient (mcc) over f1 score and accuracy in binary classification evaluation *BMC genomics*, 21: 1–13, 2020. **Abstract:** Abstract Background To evaluate binary classifications and their confusion matrices, scientific researchers can employ several statistical rates, accordingly to the goal of the experiment they are investigating. Despite being a crucial issue in machine learning, no widespread consensus has been reached on a unified elective chosen measure yet. Accuracy and F 1 score computed on confusion matrices have been (and still are) among the most popular adopted metrics in binary classification tasks. However, these statistical measures can dangerously show overoptimistic inflated results, especially on imbalanced datasets. Results The Matthews correlation coefficient (MCC), instead, is a more reliable statistical rate which produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives), proportionally both to the size of positive elements and the size of negative elements in the dataset. Conclusions In this article, we show how MCC produces a more informative and truthful score in evaluating binary classifications than accuracy and F 1 score, by first explaining the mathematical properties, and then the asset of MCC in six synthetic use cases and in a real genomics scenario. We believe that the Matthews correlation coefficient should be preferred to accuracy and F 1 score in evaluating binary classification tasks by all scientific communities. (@chicco2020advantages)

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova Boolq: Exploring the surprising difficulty of natural yes/no questions *arXiv preprint arXiv:1905.10044*, 2019. **Abstract:** In this paper we study yes/no questions that are naturally occurring — meaning that they are generated in unprompted and unconstrained settings. We build a reading comprehension dataset, BoolQ, of such questions, and show that they are unexpectedly challenging. They often query for complex, non-factoid information, and require difficult entailment-like inference to solve. We also explore the effectiveness of a range of transfer learning baselines. We find that transferring from entailment data is more effective than transferring from paraphrase or extractive QA data, and that it, surprisingly, continues to be very beneficial even when starting from massive pre-trained language models such as BERT. Our best method trains BERT on MultiNLI and then re-trains it on our train set. It achieves 80.4% accuracy compared to 90% accuracy of human annotators (and 62% majority-baseline), leaving a significant gap for future work. (@clark2019boolq)

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord Think you have solved question answering? try arc, the ai2 reasoning challenge *arXiv preprint arXiv:1803.05457*, 2018. **Abstract:** We present a new question set, text corpus, and baselines assembled to encourage AI research in advanced question answering. Together, these constitute the AI2 Reasoning Challenge (ARC), which requires far more powerful knowledge and reasoning than previous challenges such as SQuAD or SNLI. The ARC question set is partitioned into a Challenge Set and an Easy Set, where the Challenge Set contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurence algorithm. The dataset contains only natural, grade-school science questions (authored for human tests), and is the largest public-domain set of this kind (7,787 questions). We test several baselines on the Challenge Set, including leading neural models from the SQuAD and SNLI tasks, and find that none are able to significantly outperform a random baseline, reflecting the difficult nature of this task. We are also releasing the ARC Corpus, a corpus of 14M science sentences relevant to the task, and implementations of the three neural baseline models tested. Can your model perform better? We pose ARC as a challenge to the community. (@clark2018think)

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al Training verifiers to solve math word problems *arXiv preprint arXiv:2110.14168*, 2021. **Abstract:** State-of-the-art language models can match human performance on many tasks, but they still struggle to robustly perform multi-step mathematical reasoning. To diagnose the failures of current models and support research, we introduce GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems. We find that even the largest transformer models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution. To increase performance, we propose training verifiers to judge the correctness of model completions. At test time, we generate many candidate solutions and select the one ranked highest by the verifier. We demonstrate that verification significantly improves performance on GSM8K, and we provide strong empirical evidence that verification scales more effectively with increased data than a finetuning baseline. (@cobbe2021training)

Together Computer Redpajama: an open dataset for training large language models 2023. URL <https://github.com/togethercomputer/RedPajama-Data>. **Abstract:** Large language models are increasingly becoming a cornerstone technology in artificial intelligence, the sciences, and society as a whole, yet the optimal strategies for dataset composition and filtering remain largely elusive. Many of the top-performing models lack transparency in their dataset curation and model development processes, posing an obstacle to the development of fully open language models. In this paper, we identify three core data-related challenges that must be addressed to advance open-source language models. These include (1) transparency in model development, including the data curation process, (2) access to large quantities of high-quality data, and (3) availability of artifacts and metadata for dataset curation and analysis. To address these challenges, we release RedPajama-V1, an open reproduction of the LLaMA training dataset. In addition, we release RedPajama-V2, a massive web-only dataset consisting of raw, unfiltered text data together with quality signals and metadata. Together, the RedPajama datasets comprise over 100 trillion tokens spanning multiple domains and with their quality signals facilitate the filtering of data, aiming to inspire the development of numerous new datasets. To date, these datasets have already been used in the training of strong language models used in production, such as Snowflake Arctic, Salesforce’s XGen and AI2’s OLMo. To provide insight into the quality of RedPajama, we present a series of analyses and ablation studies with decoder-only language models with up to 1.6B parameters. Our findings demonstrate how quality signals for web data can be effectively leveraged to curate high-quality subsets of the dataset, underscoring the potential of RedPajama to advance the development of transparent and high-performing language models at scale. (@together2023redpajama)

OpenCompass Contributors Opencompass: A universal evaluation platform for foundation models *GitHub repository*, 2023. **Abstract:** The oil and gas companies generally design and engineering their capital intensive assets for the goal of multi-decades operations. Thus the technology screening and selection has never been taken lightly because the company is relying on these technologies for the considerable life of the asset. Historically, the oil and gas industry has selected automation and control solutions from large system providers for reliable service which leads to limited service operations and technology obsolescence challenge through the asset management lifetime. Oil and gas operators, through venues such as the Open Group’s OPAFTM and OSDUTM Forums (OSDU), have an opportunity to define the next generation of automation and control systems that break the conventional paradigm of reliance on proprietary systems, with their associated "vendor lock-in", limited availability of new features and innovation, and the costs of managing supply chain for single-source products and product obsolescence. The adoption of the information technologies (IT) into the next generation operational technologies (OT) solutions must eliminate the barriers to scaling by delivering fit-for-purpose, reliable, long-lived, secure, and interoperable systems. The approach being pursued by ExxonMobil, Intel, Red Hat, and our partners is to break the hardware and software inter-dependency that has been inherent to closed proprietary process control systems by, (1) leveraging advances in open architecture framework design concepts for software development and integration, (2) enabling interoperability in application software with defined functions including machine learning and analytic techniques, (3) hardening the zero-trust security model (Scott Rose, 2020) with zero-touch device management (Reeves, 2019), and (4) ensuring secure robust channels for control applications to communicate from the edge systems to control centers and/or cloud. This paper describes the open automation and control platform for upstream oil and gas operators enabling their field controls, optimization, and analytics with a secure zero-trust and zero-touch multi-vendor solution. The proposed open automation architecture has evolved through a sequence of evaluations described in this paper and validated through a field trial by ExxonMobil where the commercial edge device was loaded with a Linux operating system with fully pre-emptible real time kernel (preempt-RT) (Linux Foundation, n.d.), the Open Edge Insights (OEI) middleware (Intel Corporation, n.d.), and a plunger lift well control application from Naonworks. This software stack controlled the well with 100 millisecond response time. The system operated the plunger lift well continuously for 6-months powered only by a solar panel and lead acid battery. The well production process was not only controlled locally but had remote access for supervisory monitoring and control. In parallel with the extended field trial, the team has ported the solution to work with the Red Hat operating system, security, and system management infrastructure as well as the integration of secure device onboarding (FIDO Alliance, n.d.). The objective is to demonstrate a complete chain of trust from device manufacture through application deployment, application runtime, and lifecycle management. Currently, the technologies are being presented and validated to OSDUTM member companies in the OSDU Edge lab. (@contributors2023opencompass)

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs *arXiv preprint arXiv:1903.00161*, 2019. **Abstract:** Reading comprehension has recently seen rapid progress, with systems matching humans on the most popular datasets for the task. However, a large body of work has highlighted the brittleness of these systems, showing that there is much work left to be done. We introduce a new English reading comprehension benchmark, DROP, which requires Discrete Reasoning Over the content of Paragraphs. In this crowdsourced, adversarially-created, 96k-question benchmark, a system must resolve references in a question, perhaps to multiple input positions, and perform discrete operations over them (such as addition, counting, or sorting). These operations require a much more comprehensive understanding of the content of paragraphs than what was necessary for prior datasets. We apply state-of-the-art methods from both the reading comprehension and semantic parsing literature on this dataset and show that the best systems only achieve 32.7% F1 on our generalized accuracy metric, while expert human performance is 96.0%. We additionally present a new model that combines reading comprehension methods with simple numerical reasoning to achieve 47.0% F1. (@dua2019drop)

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto Length-controlled alpacaeval: A simple way to debias automatic evaluators *arXiv preprint arXiv:2404.04475*, 2024. **Abstract:** LLM-based auto-annotators have become a key component of the LLM development process due to their cost-effectiveness and scalability compared to human-based evaluation. However, these auto-annotators can introduce biases that are hard to remove. Even simple, known confounders such as preference for longer outputs remain in existing automated evaluation metrics. We propose a simple regression analysis approach for controlling biases in auto-evaluations. As a real case study, we focus on reducing the length bias of AlpacaEval, a fast and affordable benchmark for instruction-tuned LLMs that uses LLMs to estimate response quality. Despite being highly correlated with human preferences, AlpacaEval is known to favor models that generate longer outputs. We introduce a length-controlled AlpacaEval that aims to answer the counterfactual question: "What would the preference be if the model’s and baseline’s output had the same length?" To achieve this, we first fit a generalized linear model to predict the biased auto-annotator’s preferences based on the mediators we want to control for (length difference) and other relevant features. We then obtain length-controlled preferences by predicting preferences while conditioning the GLM with a zero difference in lengths. Length-controlling not only improves the robustness of the metric to manipulations in model verbosity, but we also find that it increases the Spearman correlation with LMSYS Chatbot Arena from 0.94 to 0.98. (@dubois2024length)

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto Alpacafarm: A simulation framework for methods that learn from human feedback *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Large language models (LLMs) such as ChatGPT have seen widespread adoption due to their strong instruction-following abilities. Developing these LLMs involves a complex yet poorly understood workflow requiring training with human feedback. Replicating and understanding this instruction-following requires tackling three major challenges: the high cost of data collection, the lack of trustworthy evaluation, and the absence of reference method implementations. We address these challenges with AlpacaFarm, a simulator that enables research and development for learning from feedback at a low cost. First, we design LLM prompts to simulate human feedback that are 50x cheaper than crowdworkers and display high agreement with humans. Second, we propose an automatic evaluation and validate it against human instructions obtained on real-world interactions. Third, we contribute reference implementations for several methods (PPO, DPO, best-of-n, expert iteration, and more) that learn from pairwise feedback. Finally, as an end-to-end validation of AlpacaFarm, we train and evaluate eleven models on 10k pairs of real human feedback and show that rankings of models trained in AlpacaFarm match rankings of models trained on human data. As a demonstration of the research possible in AlpacaFarm, we find that methods that use a reward model can substantially improve over supervised fine-tuning and that our reference PPO implementation leads to a +10% improvement in win-rate against Davinci003. We release all components of AlpacaFarm at https://github.com/tatsu-lab/alpaca_farm. (@dubois2024alpacafarm)

Hugging Face Open llm leaderboard 2023. URL <https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard>. **Abstract:** This paper introduces the Open Ko-LLM Leaderboard and the Ko-H5 Benchmark as vital tools for evaluating Large Language Models (LLMs) in Korean. Incorporating private test sets while mirroring the English Open LLM Leaderboard, we establish a robust evaluation framework that has been well integrated in the Korean LLM community. We perform data leakage analysis that shows the benefit of private test sets along with a correlation study within the Ko-H5 benchmark and temporal analyses of the Ko-H5 score. Moreover, we present empirical support for the need to expand beyond set benchmarks. We hope the Open Ko-LLM Leaderboard sets precedent for expanding LLM evaluation to foster more linguistic diversity. (@openllmleaderboard)

Lizhou Fan, Wenyue Hua, Lingyao Li, Haoyang Ling, Yongfeng Zhang, and Libby Hemphill Nphardeval: Dynamic benchmark on reasoning ability of large language models via complexity classes *arXiv preprint arXiv:2312.14890*, 2023. **Abstract:** Complex reasoning ability is one of the most important features of current LLMs, which has also been leveraged to play an integral role in complex decision-making tasks. Therefore, the investigation into the reasoning capabilities of Large Language Models (LLMs) is critical: numerous benchmarks have been established to assess the reasoning abilities of LLMs. However, current benchmarks are inadequate in offering a rigorous evaluation of the full extent of reasoning abilities that LLMs are capable of achieving. They are also prone to the risk of overfitting, as these benchmarks, being publicly accessible and static, allow models to potentially tailor their responses to specific benchmark metrics, thereby inflating their performance. Addressing these limitations, our research introduces a new benchmark, named NPHardEval. This benchmark is designed to evaluate the reasoning abilities of LLMs across a broad spectrum of 900 algorithmic questions, extending up to the NP-Hard complexity class. These questions are meticulously chosen to represent a wide range of complexity class below the NP-hard complexity class, offering a rigorous measure of the reasoning ability of LLMs. Through this study, we shed light on the current state of reasoning in LLMs, providing an objective and rigorous perspective through the comparison of LLMs’ performance across complex classes. Moreover, this benchmark is designed with a dynamic update mechanism, where the datapoints are refreshed on a monthly basis. Such regular updates play a crucial role in mitigating the risk of LLMs overfitting to the benchmark, promoting a more accurate and reliable assessment of their reasoning capabilities. The benchmark dataset and code of NPHardEval are available at https://github.com/casmlab/NPHardEval. (@fan2023nphardeval)

Wikimedia Foundation Wikimedia downloads 2022. URL <https://dumps.wikimedia.org>. (@wikidump)

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt Measuring massive multitask language understanding *arXiv preprint arXiv:2009.03300*, 2020. **Abstract:** We propose a new test to measure a text model’s multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model’s academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings. (@hendrycks2020measuring)

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt Measuring mathematical problem solving with the math dataset *NeurIPS*, 2021. **Abstract:** Many intellectual endeavors require mathematical problem solving, but this skill remains beyond the capabilities of computers. To measure this ability in machine learning models, we introduce MATH, a new dataset of 12,500 challenging competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations. To facilitate future research and increase accuracy on MATH, we also contribute a large auxiliary pretraining dataset which helps teach models the fundamentals of mathematics. Even though we are able to increase accuracy on MATH, our results show that accuracy remains relatively low, even with enormous Transformer models. Moreover, we find that simply increasing budgets and model parameter counts will be impractical for achieving strong mathematical reasoning if scaling trends continue. While scaling Transformers is automatically solving most other text-based tasks, scaling is not currently solving MATH. To have more traction on mathematical problem solving we will likely need new algorithmic advancements from the broader research community. (@hendrycksmath2021)

Yuzhen Huang, Jinghan Zhang, Zifei Shan, and Junxian He Compression represents intelligence linearly *arXiv preprint arXiv:2404.09937*, 2024. **Abstract:** There is a belief that learning to compress well will lead to intelligence. Recently, language modeling has been shown to be equivalent to compression, which offers a compelling rationale for the success of large language models (LLMs): the development of more advanced language models is essentially enhancing compression which facilitates intelligence. Despite such appealing discussions, little empirical evidence is present for the interplay between compression and intelligence. In this work, we examine their relationship in the context of LLMs, treating LLMs as data compressors. Given the abstract concept of "intelligence", we adopt the average downstream benchmark scores as a surrogate, specifically targeting intelligence related to knowledge and commonsense, coding, and mathematical reasoning. Across 12 benchmarks, our study brings together 31 public LLMs that originate from diverse organizations. Remarkably, we find that LLMs’ intelligence – reflected by average benchmark scores – almost linearly correlates with their ability to compress external text corpora. These results provide concrete evidence supporting the belief that superior compression indicates greater intelligence. Furthermore, our findings suggest that compression efficiency, as an unsupervised metric derived from raw text corpora, serves as a reliable evaluation measure that is linearly associated with the model capabilities. We open-source our compression datasets as well as our data collection pipelines to facilitate future researchers to assess compression properly. (@huang2024compression)

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica Livecodebench: Holistic and contamination free evaluation of large language models for code *arXiv preprint arXiv:2403.07974*, 2024. **Abstract:** Large Language Models (LLMs) applied to code-related applications have emerged as a prominent field, attracting significant interest from both academia and industry. However, as new and improved LLMs are developed, existing evaluation benchmarks (e.g., HumanEval, MBPP) are no longer sufficient for assessing their capabilities. In this work, we propose LiveCodeBench, a comprehensive and contamination-free evaluation of LLMs for code, which continuously collects new problems over time from contests across three competition platforms, namely LeetCode, AtCoder, and CodeForces. Notably, our benchmark also focuses on a broader range of code related capabilities, such as self-repair, code execution, and test output prediction, beyond just code generation. Currently, LiveCodeBench hosts four hundred high-quality coding problems that were published between May 2023 and May 2024. We have evaluated 18 base LLMs and 34 instruction-tuned LLMs on LiveCodeBench. We present empirical findings on contamination, holistic performance comparisons, potential overfitting in existing benchmarks as well as individual model comparisons. We will release all prompts and model completions for further community analysis, along with a general toolkit for adding new scenarios and model (@jain2024livecodebench)

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension *arXiv preprint arXiv:1705.03551*, 2017. **Abstract:** We present TriviaQA, a challenging reading comprehension dataset containing over 650K question-answer-evidence triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions. We show that, in comparison to other recently introduced large-scale datasets, TriviaQA (1) has relatively complex, compositional questions, (2) has considerable syntactic and lexical variability between questions and corresponding answer-evidence sentences, and (3) requires more cross sentence reasoning to find answers. We also present two baseline algorithms: a feature-based classifier and a state-of-the-art neural network, that performs well on SQuAD reading comprehension. Neither approach comes close to human performance (23% and 40% vs. 80%), suggesting that TriviaQA is a challenging testbed that is worth significant future study. Data and code available at – http://nlp.cs.washington.edu/triviaqa/ (@joshi2017triviaqa)

Bill Yuchen Lin, Khyathi Chandu, Faeze Brahman, Yuntian Deng, Abhilasha Ravichander, Valentina Pyatkin, Ronan Le Bras, and Yejin Choi Wildbench: Benchmarking llms with challenging tasks from real users in the wild 2024. URL <https://huggingface.co/spaces/allenai/WildBench>. **Abstract:** We introduce WildBench, an automated evaluation framework designed to benchmark large language models (LLMs) using challenging, real-world user queries. WildBench consists of 1,024 tasks carefully selected from over one million human-chatbot conversation logs. For automated evaluation with WildBench, we have developed two metrics, WB-Reward and WB-Score, which are computable using advanced LLMs such as GPT-4-turbo. WildBench evaluation uses task-specific checklists to evaluate model outputs systematically and provides structured explanations that justify the scores and comparisons, resulting in more reliable and interpretable automatic judgments. WB-Reward employs fine-grained pairwise comparisons between model responses, generating five potential outcomes: much better, slightly better, slightly worse, much worse, or a tie. Unlike previous evaluations that employed a single baseline model, we selected three baseline models at varying performance levels to ensure a comprehensive pairwise evaluation. Additionally, we propose a simple method to mitigate length bias, by converting outcomes of “slightly better/worse” to “tie” if the winner response exceeds the loser one by more than $K$ characters. WB-Score evaluates the quality of model outputs individually, making it a fast and cost-efficient evaluation metric. WildBench results demonstrate a strong correlation with the human-voted Elo ratings from Chatbot Arena on hard tasks. Specifically, WB-Reward achieves a Pearson correlation of 0.98 with top-ranking models. Additionally, WB-Score reaches 0.95, surpassing both ArenaHard’s 0.91 and AlpacaEval2.0’s 0.89 for length-controlled win rates, as well as the 0.87 for regular win rates. (@wildbench2024)

Tim Menzies and Thomas Zimmermann Software analytics: so what? *IEEE Software*, 30 (4): 31–37, 2013. **Abstract:** The guest editors of this special issue of IEEE Software invited submissions that reflected the benefits (and drawbacks) of software analytics, an area of explosive growth. They had so many excellent submissions that they had to split this special issue into two volumes–you’ll see even more content in the September/October issue. They divided the articles on conceptual grounds, so both volumes will feature equally excellent work. The Web extra at http://youtu.be/nO6X0azR0nw is a video interview in which IEEE Software editor in chief Forrest Shull speaks with Tim Menzies about the growing importance of software analytics. (@menzies2013software)

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal Can a suit of armor conduct electricity? a new dataset for open book question answering *arXiv preprint arXiv:1809.02789*, 2018. **Abstract:** We present a new kind of question answering dataset, OpenBookQA, modeled after open book exams for assessing human understanding of a subject. The open book that comes with our questions is a set of 1329 elementary level science facts. Roughly 6000 questions probe an understanding of these facts and their application to novel situations. This requires combining an open book fact (e.g., metals conduct electricity) with broad common knowledge (e.g., a suit of armor is made of metal) obtained from other sources. While existing QA datasets over documents or knowledge bases, being generally self-contained, focus on linguistic understanding, OpenBookQA probes a deeper understanding of both the topic—in the context of common knowledge—and the language it is expressed in. Human performance on OpenBookQA is close to 92%, but many state-of-the-art pre-trained QA methods perform surprisingly poorly, worse than several simple neural baselines we develop. Our oracle experiments designed to circumvent the knowledge retrieval bottleneck demonstrate the value of both the open book and additional facts. We leave it as a challenge to solve the retrieval problem in this multi-hop setting and to close the large gap to human performance. (@mihaylov2018can)

Piotr Padlewski, Max Bain, Matthew Henderson, Zhongkai Zhu, Nishant Relan, Hai Pham, Donovan Ong, Kaloyan Aleksiev, Aitor Ormazabal, Samuel Phua, et al Vibe-eval: A hard evaluation suite for measuring progress of multimodal language models *arXiv preprint arXiv:2405.02287*, 2024. **Abstract:** We introduce Vibe-Eval: a new open benchmark and framework for evaluating multimodal chat models. Vibe-Eval consists of 269 visual understanding prompts, including 100 of hard difficulty, complete with gold-standard responses authored by experts. Vibe-Eval is open-ended and challenging with dual objectives: (i) vibe checking multimodal chat models for day-to-day tasks and (ii) rigorously testing and probing the capabilities of present frontier models. Notably, our hard set contains \>50% questions that all frontier models answer incorrectly. We explore the nuances of designing, evaluating, and ranking models on ultra challenging prompts. We also discuss trade-offs between human and automatic evaluation, and show that automatic model evaluation using Reka Core roughly correlates to human judgment. We offer free API access for the purpose of lightweight evaluation and plan to conduct formal human evaluations for public models that perform well on the Vibe-Eval’s automatic scores. We release the evaluation code and data, see https://github.com/reka-ai/reka-vibe-eval (@padlewski2024vibe)

Nils Reimers and Iryna Gurevych Sentence-bert: Sentence embeddings using siamese bert-networks *arXiv preprint arXiv:1908.10084*, 2019. **Abstract:** BERT (Devlin et al., 2018) and RoBERTa (Liu et al., 2019) has set a new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity (STS). However, it requires that both sentences are fed into the network, which causes a massive computational overhead: Finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations (\~65 hours) with BERT. The construction of BERT makes it unsuitable for semantic similarity search as well as for unsupervised tasks like clustering. In this publication, we present Sentence-BERT (SBERT), a modification of the pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65 hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT. We evaluate SBERT and SRoBERTa on common STS tasks and transfer learning tasks, where it outperforms other state-of-the-art sentence embeddings methods. (@reimers2019sentence)

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman Gpqa: A graduate-level google-proof q&a benchmark *arXiv preprint arXiv:2311.12022*, 2023. **Abstract:** We present GPQA, a challenging dataset of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry. We ensure that the questions are high-quality and extremely difficult: experts who have or are pursuing PhDs in the corresponding domains reach 65% accuracy (74% when discounting clear mistakes the experts identified in retrospect), while highly skilled non-expert validators only reach 34% accuracy, despite spending on average over 30 minutes with unrestricted access to the web (i.e., the questions are "Google-proof"). The questions are also difficult for state-of-the-art AI systems, with our strongest GPT-4 based baseline achieving 39% accuracy. If we are to use future AI systems to help us answer very hard questions, for example, when developing new scientific knowledge, we need to develop scalable oversight methods that enable humans to supervise their outputs, which may be difficult even if the supervisors are themselves skilled and knowledgeable. The difficulty of GPQA both for skilled non-experts and frontier AI systems should enable realistic scalable oversight experiments, which we hope can help devise ways for human experts to reliably get truthful information from AI systems that surpass human capabilities. (@rein2023gpqa)

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi Winogrande: An adversarial winograd schema challenge at scale *Communications of the ACM*, 64 (9): 99–106, 2021. **Abstract:** The Winograd Schema Challenge (WSC) (Levesque, Davis, and Morgenstern 2011), a benchmark for commonsense reasoning, is a set of 273 expert-crafted pronoun resolution problems originally designed to be unsolvable for statistical models that rely on selectional preferences or word associations. However, recent advances in neural language models have already reached around 90% accuracy on variants of WSC. This raises an important question whether these models have truly acquired robust commonsense capabilities or whether they rely on spurious biases in the datasets that lead to an overestimation of the true capabilities of machine commonsense.To investigate this question, we introduce WinoGrande, a large-scale dataset of 44k problems, inspired by the original WSC design, but adjusted to improve both the scale and the hardness of the dataset. The key steps of the dataset construction consist of (1) a carefully designed crowdsourcing procedure, followed by (2) systematic bias reduction using a novel AfLite algorithm that generalizes human-detectable word associations to machine-detectable embedding associations. The best state-of-the-art methods on WinoGrande achieve 59.4 – 79.1%, which are ∼15-35% (absolute) below human performance of 94.0%, depending on the amount of the training data allowed (2% – 100% respectively).Furthermore, we establish new state-of-the-art results on five related benchmarks — WSC (→ 90.1%), DPR (→ 93.1%), COPA(→ 90.6%), KnowRef (→ 85.6%), and Winogender (→ 97.1%). These results have dual implications: on one hand, they demonstrate the effectiveness of WinoGrande when used as a resource for transfer learning. On the other hand, they raise a concern that we are likely to be overestimating the true capabilities of machine commonsense across all these benchmarks. We emphasize the importance of algorithmic bias reduction in existing and future benchmarks to mitigate such overestimation. (@sakaguchi2021winogrande)

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi Socialiqa: Commonsense reasoning about social interactions *arXiv preprint arXiv:1904.09728*, 2019. **Abstract:** We introduce Social IQa, the first largescale benchmark for commonsense reasoning about social situations. Social IQa contains 38,000 multiple choice questions for probing emotional and social intelligence in a variety of everyday situations (e.g., Q: "Jordan wanted to tell Tracy a secret, so Jordan leaned towards Tracy. Why did Jordan do this?" A: "Make sure no one else could hear"). Through crowdsourcing, we collect commonsense questions along with correct and incorrect answers about social interactions, using a new framework that mitigates stylistic artifacts in incorrect answers by asking workers to provide the right answer to a different but related question. Empirical results show that our benchmark is challenging for existing question-answering models based on pretrained language models, compared to human performance (\>20% gap). Notably, we further establish Social IQa as a resource for transfer learning of commonsense knowledge, achieving state-of-the-art performance on multiple commonsense reasoning tasks (Winograd Schemas, COPA). (@sap2019socialiqa)

Teams ShareGPT Sharegpt: Share your wildest chatgpt conversations with one click 2023. (@sharegpt2023sharegpt)

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al Challenging big-bench tasks and whether chain-of-thought can solve them *arXiv preprint arXiv:2210.09261*, 2022. **Abstract:** BIG-Bench (Srivastava et al., 2022) is a diverse evaluation suite that focuses on tasks believed to be beyond the capabilities of current language models. Language models have already made good progress on this benchmark, with the best model in the BIG-Bench paper outperforming average reported human-rater results on 65% of the BIG-Bench tasks via few-shot prompting. But on what tasks do language models fall short of average human-rater performance, and are those tasks actually unsolvable by current language models? In this work, we focus on a suite of 23 challenging BIG-Bench tasks which we call BIG-Bench Hard (BBH). These are the task for which prior language model evaluations did not outperform the average human-rater. We find that applying chain-of-thought (CoT) prompting to BBH tasks enables PaLM to surpass the average human-rater performance on 10 of the 23 tasks, and Codex (code-davinci-002) to surpass the average human-rater performance on 17 of the 23 tasks. Since many tasks in BBH require multi-step reasoning, few-shot prompting without CoT, as done in the BIG-Bench evaluations (Srivastava et al., 2022), substantially underestimates the best performance and capabilities of language models, which is better captured via CoT prompting. As further analysis, we explore the interaction between CoT and model scale on BBH, finding that CoT enables emergent task performance on several BBH tasks with otherwise flat scaling curves. (@suzgun2022challenging)

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant Commonsenseqa: A question answering challenge targeting commonsense knowledge *arXiv preprint arXiv:1811.00937*, 2018. **Abstract:** When answering a question, people often draw upon their rich world knowledge in addition to the particular context. Recent work has focused primarily on answering questions given some relevant document or context, and required very little general background. To investigate question answering with prior knowledge, we present CommonsenseQA: a challenging new dataset for commonsense question answering. To capture common sense beyond associations, we extract from ConceptNet (Speer et al., 2017) multiple target concepts that have the same semantic relation to a single source concept. Crowd-workers are asked to author multiple-choice questions that mention the source concept and discriminate in turn between each of the target concepts. This encourages workers to create questions with complex semantics that often require prior knowledge. We create 12,247 questions through this procedure and demonstrate the difficulty of our task with a large number of strong baselines. Our best baseline is based on BERT-large (Devlin et al., 2018) and obtains 56% accuracy, well below human performance, which is 89%. (@talmor2018commonsenseqa)

Li Tianle, Chiang Wei-Lin, Frick Evan, Dunlap Lisa, Zhu Banghua, Gonzalez Joseph E., and Stoica Ion From live data to high-quality benchmarks: The arena-hard pipeline *See https://lmsys.org/blog/2024-04-19-arena-hard/*, 2024. **Abstract:** The rapid evolution of Large Language Models (LLMs) has outpaced the development of model evaluation, highlighting the need for continuous curation of new, challenging benchmarks. However, manual curation of high-quality, human-aligned benchmarks is expensive and time-consuming. To address this, we introduce BenchBuilder, an automated pipeline that leverages LLMs to curate high-quality, open-ended prompts from large, crowd-sourced datasets, enabling continuous benchmark updates without human in the loop. We apply BenchBuilder to datasets such as Chatbot Arena and WildChat-1M, extracting challenging prompts and utilizing LLM-as-a-Judge for automatic model evaluation. To validate benchmark quality, we propose new metrics to measure a benchmark’s alignment with human preferences and ability to separate models. We release Arena-Hard-Auto, a benchmark consisting 500 challenging prompts curated by BenchBuilder. Arena-Hard-Auto provides 3x higher separation of model performances compared to MT-Bench and achieves 98.6% correlation with human preference rankings, all at a cost of $20. Our work sets a new framework for the scalable curation of automated benchmarks from extensive data. (@tianle2024from)

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al Huggingface’s transformers: State-of-the-art natural language processing *arXiv preprint arXiv:1910.03771*, 2019. **Abstract:** Recent progress in natural language processing has been driven by advances in both model architecture and model pretraining. Transformer architectures have facilitated building higher-capacity models and pretraining has made it possible to effectively utilize this capacity for a wide variety of tasks. \\}textit{Transformers} is an open-source library with the goal of opening up these advances to the wider machine learning community. The library consists of carefully engineered state-of-the art Transformer architectures under a unified API. Backing this library is a curated collection of pretrained models made by and available for the community. \\}textit{Transformers} is designed to be extensible by researchers, simple for practitioners, and fast and robust in industrial deployments. The library is available at \\}url{https://github.com/huggingface/transformers}. (@wolf2019huggingface)

Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E Gonzalez, and Ion Stoica Rethinking benchmark and contamination for language models with rephrased samples *arXiv preprint arXiv:2311.04850*, 2023. **Abstract:** Large language models are increasingly trained on all the data ever produced by humans. Many have raised concerns about the trustworthiness of public benchmarks due to potential contamination in pre-training or fine-tuning datasets. While most data decontamination efforts apply string matching (e.g., n-gram overlap) to remove benchmark data, we show that these methods are insufficient, and simple variations of test data (e.g., paraphrasing, translation) can easily bypass these decontamination measures. Furthermore, we demonstrate that if such variation of test data is not eliminated, a 13B model can easily overfit a test benchmark and achieve drastically high performance, on par with GPT-4. We validate such observations in widely used benchmarks such as MMLU, GSK8k, and HumanEval. To address this growing risk, we propose a stronger LLM-based decontamination method and apply it to widely used pre-training and fine-tuning datasets, revealing significant previously unknown test overlap. For example, in pre-training sets such as RedPajama-Data-1T and StarCoder-Data, we identified that 8-18\\}% of the HumanEval benchmark overlaps. Interestingly, we also find such contamination in synthetic dataset generated by GPT-3.5/4, suggesting a potential risk of unintentional contamination. We urge the community to adopt stronger decontamination approaches when using public benchmarks. Moreover, we call for the community to actively develop fresh one-time exams to evaluate models accurately. Our decontamination tool is publicly available at https://github.com/lm-sys/llm-decontaminator. (@yang2023rethinking)

Sheng Kung Michael Yi, Mark Steyvers, Michael D Lee, and Matthew J Dry The wisdom of the crowd in combinatorial problems *Cognitive science*, 36 (3): 452–470, 2012. **Abstract:** Abstract The “wisdom of the crowd” phenomenon refers to the finding that the aggregate of a set of proposed solutions from a group of individuals performs better than the majority of individual solutions. Most often, wisdom of the crowd effects have been investigated for problems that require single numerical estimates. We investigate whether the effect can also be observed for problems where the answer requires the coordination of multiple pieces of information. We focus on combinatorial problems such as the planar Euclidean traveling salesperson problem, minimum spanning tree problem, and a spanning tree memory task. We develop aggregation methods that combine common solution fragments into a global solution and demonstrate that these aggregate solutions outperform the majority of individual solutions. These case studies suggest that the wisdom of the crowd phenomenon might be broadly applicable to problem‐solving and decision‐making situations that go beyond the estimation of single numbers. (@yi2012wisdom)

Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen Mammoth2: Scaling instructions from the web *arXiv preprint arXiv:2405.03548*, 2024. **Abstract:** Instruction tuning improves the reasoning abilities of large language models (LLMs), with data quality and scalability being the crucial factors. Most instruction tuning data come from human crowd-sourcing or GPT-4 distillation. We propose a paradigm to efficiently harvest 10 million naturally existing instruction data from the pre-training web corpus to enhance LLM reasoning. Our approach involves (1) recalling relevant documents, (2) extracting instruction-response pairs, and (3) refining the extracted pairs using open-source LLMs. Fine-tuning base LLMs on this dataset, we build MAmmoTH2 models, which significantly boost performance on reasoning benchmarks. Notably, MAmmoTH2-7B’s (Mistral) performance increases from 11% to 36.7% on MATH and from 36% to 68.4% on GSM8K without training on any in-domain data. Further training MAmmoTH2 on public instruction tuning datasets yields MAmmoTH2-Plus, achieving state-of-the-art performance on several reasoning and chatbot benchmarks. Our work demonstrates how to harvest large-scale, high-quality instruction data without costly human annotation or GPT-4 distillation, providing a new paradigm for building better instruction tuning data. (@yue2024mammoth2)

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi Hellaswag: Can a machine really finish your sentence? *arXiv preprint arXiv:1905.07830*, 2019. **Abstract:** Recent work by Zellers et al. (2018) introduced a new task of commonsense natural language inference: given an event description such as "A woman sits at a piano," a machine must select the most likely followup: "She sets her fingers on the keys." With the introduction of BERT, near human-level performance was reached. Does this mean that machines can perform human level commonsense inference? In this paper, we show that commonsense inference still proves difficult for even state-of-the-art models, by presenting HellaSwag, a new challenge dataset. Though its questions are trivial for humans (\>95% accuracy), state-of-the-art models struggle (\<48%). We achieve this via Adversarial Filtering (AF), a data collection paradigm wherein a series of discriminators iteratively select an adversarial set of machine-generated wrong answers. AF proves to be surprisingly robust. The key insight is to scale up the length and complexity of the dataset examples towards a critical ’Goldilocks’ zone wherein generated text is ridiculous to humans, yet often misclassified by state-of-the-art models. Our construction of HellaSwag, and its resulting difficulty, sheds light on the inner workings of deep pretrained models. More broadly, it suggests a new path forward for NLP research, in which benchmarks co-evolve with the evolving state-of-the-art in an adversarial way, so as to present ever-harder challenges. (@zellers2019hellaswag)

Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Dylan Slack, Qin Lyu, et al A careful examination of large language model performance on grade school arithmetic *arXiv preprint arXiv:2405.00332*, 2024. **Abstract:** Large language models (LLMs) have achieved impressive success on many benchmarks for mathematical reasoning. However, there is growing concern that some of this performance actually reflects dataset contamination, where data closely resembling benchmark questions leaks into the training data, instead of true reasoning ability. To investigate this claim rigorously, we commission Grade School Math 1000 (GSM1k). GSM1k is designed to mirror the style and complexity of the established GSM8k benchmark, the gold standard for measuring elementary mathematical reasoning. We ensure that the two benchmarks are comparable across important metrics such as human solve rates, number of steps in solution, answer magnitude, and more. When evaluating leading open- and closed-source LLMs on GSM1k, we observe accuracy drops of up to 8%, with several families of models showing evidence of systematic overfitting across almost all model sizes. Further analysis suggests a positive relationship (Spearman’s r\^2 = 0.36) between a model’s probability of generating an example from GSM8k and its performance gap between GSM8k and GSM1k, suggesting that some models may have partially memorized GSM8k. Nevertheless, many models, especially those on the frontier, show minimal signs of overfitting, and all models broadly demonstrate generalization to novel math problems guaranteed to not be in their training data. (@zhang2024careful)

Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al Direct preference optimization of video large multimodal models from language model reward *arXiv preprint arXiv:2404.01258*, 2024. **Abstract:** Preference modeling techniques, such as direct preference optimization (DPO), has shown effective in enhancing the generalization abilities of large language model (LLM). However, in tasks involving video instruction-following, providing informative feedback, especially for detecting hallucinations in generated responses, remains a significant challenge. Previous studies have explored using large large multimodal models (LMMs) as reward models to guide preference modeling, but their ability to accurately assess the factuality of generated responses compared to corresponding videos has not been conclusively established. This paper introduces a novel framework that utilizes detailed video captions as a proxy of video content, enabling language models to incorporate this information as supporting evidence for scoring video Question Answering (QA) predictions. Our approach demonstrates robust alignment with OpenAI GPT-4V model’s reward mechanism, which directly takes video frames as input. Furthermore, we show that applying this tailored reward through DPO significantly improves the performance of video LMMs on video QA tasks. (@zhang2024direct)

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng Wildchat: 1m chatgpt interaction logs in the wild *arXiv preprint arXiv:2405.01470*, 2024. **Abstract:** Chatbots such as GPT-4 and ChatGPT are now serving millions of users. Despite their widespread use, there remains a lack of public datasets showcasing how these tools are used by a population of users in practice. To bridge this gap, we offered free access to ChatGPT for online users in exchange for their affirmative, consensual opt-in to anonymously collect their chat transcripts and request headers. From this, we compiled WildChat, a corpus of 1 million user-ChatGPT conversations, which consists of over 2.5 million interaction turns. We compare WildChat with other popular user-chatbot interaction datasets, and find that our dataset offers the most diverse user prompts, contains the largest number of languages, and presents the richest variety of potentially toxic use-cases for researchers to study. In addition to timestamped chat transcripts, we enrich the dataset with demographic data, including state, country, and hashed IP addresses, alongside request headers. This augmentation allows for more detailed analysis of user behaviors across different geographical regions and temporal dimensions. Finally, because it captures a broad range of use cases, we demonstrate the dataset’s potential utility in fine-tuning instruction-following models. WildChat is released at https://wildchat.allen.ai under AI2 ImpACT Licenses. (@zhao2024wildchat)

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al Judging llm-as-a-judge with mt-bench and chatbot arena *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Evaluating large language model (LLM) based chat assistants is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences. To address this, we explore using strong LLMs as judges to evaluate these models on more open-ended questions. We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate some of them. We then verify the agreement between LLM judges and human preferences by introducing two benchmarks: MT-bench, a multi-turn question set; and Chatbot Arena, a crowdsourced battle platform. Our results reveal that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement between humans. Hence, LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain. Additionally, we show our benchmark and traditional benchmarks complement each other by evaluating several variants of LLaMA and Vicuna. The MT-bench questions, 3K expert votes, and 30K conversations with human preferences are publicly available at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge. (@zheng2024judging)

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan Agieval: A human-centric benchmark for evaluating foundation models *arXiv preprint arXiv:2304.06364*, 2023. **Abstract:** Evaluating the general abilities of foundation models to tackle human-level tasks is a vital aspect of their development and application in the pursuit of Artificial General Intelligence (AGI). Traditional benchmarks, which rely on artificial datasets, may not accurately represent human-level capabilities. In this paper, we introduce AGIEval, a novel benchmark specifically designed to assess foundation model in the context of human-centric standardized exams, such as college entrance exams, law school admission tests, math competitions, and lawyer qualification tests. We evaluate several state-of-the-art foundation models, including GPT-4, ChatGPT, and Text-Davinci-003, using this benchmark. Impressively, GPT-4 surpasses average human performance on SAT, LSAT, and math competitions, attaining a 95% accuracy rate on the SAT Math test and a 92.5% accuracy on the English test of the Chinese national college entrance exam. This demonstrates the extraordinary performance of contemporary foundation models. In contrast, we also find that GPT-4 is less proficient in tasks that require complex reasoning or specific domain knowledge. Our comprehensive analyses of model capabilities (understanding, knowledge, reasoning, and calculation) reveal these models’ strengths and limitations, providing valuable insights into future directions for enhancing their general capabilities. By concentrating on tasks pertinent to human cognition and decision-making, our benchmark delivers a more meaningful and robust evaluation of foundation models’ performance in real-world scenarios. The data, code, and all model outputs are released in https://github.com/ruixiangcui/AGIEval. (@zhong2023agieval)

</div>

# Frequently Asked Questions [sec:FAQ]

We list the potentially frequently asked questions and the point-to-point answers as follows:

## Why are real-world human queries and preferences important?

One of the primary applications for AI model development is the automation of complex tasks traditionally performed by humans. As AI models coexist with humans, frequent interactions with humans are necessary to manage these tasks effectively. These interactions predominantly involve natural language queries, as it is the most common medium for human communication. Given the significance of human interaction in the use cases for general AI models, it is crucial to evaluate AI models—particularly large language models (LLMs) that rely on natural language—under conditions that mirror real-world scenarios, *i.e.,* receiving human queries and assessing performance based on human preferences. Evaluating models based on their real-world use cases is well-supported across various research disciplines `\citep{chicco2020advantages, bender2018data, menzies2013software}`{=latex}.

## Why do you use web queries as real-world user queries?

Because web queries are grounded in the largest human population (5.4 billion internet users) among the accessible query sources. Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a> illustrates the query distribution differences in wild queries across various user population sizes. ShareGPT, with 100 million[^7] active users by mid-2023, contrasts with WildChat `\citep{zhao2024wildchat}`{=latex}, Chatbot Arena Conversations `\citep{chiang2024chatbot}`{=latex}, and LMSYS-Chat-1M `\citep{zheng2024judging}`{=latex}, which have user bases of 0.2 million, 0.13 million, and 0.21 million, respectively. The global internet user count was 5.4 billion[^8] in 2023, an order of magnitude larger than all considered wild datasets. Consequently, the user bases of the internet, ShareGPT, and other datasets span three distinct orders of magnitude. ShareGPT’s larger user population (second order of magnitude) yields a distribution most similar to web queries from the global internet user base (third order of magnitude), both visually and in cluster distance (C-Dist), validating that user population size affects query distribution. Compared to web queries and ShareGPT data, datasets from the Chatbot Arena website—Chatbot Arena Conversations and LMSYS-Chat-1M—have a higher proportion of technical queries (as presented in Figure <a href="#fig:topic_map" data-reference-type="ref" data-reference="fig:topic_map">3</a>, queries with higher position on the map are more technical). This indicates a user base skewed towards technical users, potentially affecting evaluation results, as an effective LLM benchmark should mimic real-world use cases.

## Why do you use benchmark mixtures instead of training judge models directly on Arena Conversations to achieve a similar model ranking with Arena Elo?

The crucial difference lies in the scoring methods: using ground truth answers versus LLM judges. Ground-truth-based evaluation is more interpretable, faster, and cost-effective compared to LLM-as-judges. Furthermore, training effective judge models is highly challenging because (1) LLMs possess inherent preference biases `\citep{zheng2024judging}`{=latex}, and (2) to evaluate other models accurately without ground truth answers, the judge model must be either superior to or at least on par with the models it assesses. Consequently, a large model is required, complicating the training process.

## Why isn’t the cluster distribution ranking introduced in Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a> consistent with the rankings in Figures <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a> or <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a>? [why-isnt-the-cluster-distribution-ranking-introduced-in-figure-figbenchmark_distribution-consistent-with-the-rankings-in-figures-figheatmap_arena-or-figheatmap_main]

The key to human preference correlation, as indicated in Figure <a href="#fig:subset_corr_breakdown1" data-reference-type="ref" data-reference="fig:subset_corr_breakdown1">12</a>, lies in ensuring that a benchmark’s query distribution aligns with a subset of the wild queries, rather than encompassing the entire wild query distribution. This is evidenced by the high correlation between the `MixEval`-mixed domain-specific benchmarks and Arena Elo in Figure <a href="#fig:subset_corr_breakdown1" data-reference-type="ref" data-reference="fig:subset_corr_breakdown1">12</a>. However, aligning with only a subset of wild queries significantly impacts the cluster distance metric shown in Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a>. Notably, covering all regions of wild queries enhances correlation scores, as demonstrated by `MixEval` achieving higher correlation in Figure <a href="#fig:subset_corr_breakdown1" data-reference-type="ref" data-reference="fig:subset_corr_breakdown1">12</a>.

## How long does it take to dynamically update MixEval-Hard?

The update of `MixEval-Hard` is somewhat slower than `MixEval`, which can be updated within 1 minute. `MixEval-Hard`, a subset of `MixEval`, is sampled based on the prediction results of several models. Thus, the update time depends on the models used to sample this subset. If only GPT-4 Turbo’s prediction results are used to rank the question difficulties, the total update time is approximately 2 minutes, which remains rapid. However, according to `\citep{padlewski2024vibe}`{=latex}, GPT-4 Turbo may yield a lower score in this condition compared to using results from multiple models.

## Correlation gap between he values shown in Figures <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a> and those reported by authors? [correlation-gap-between-he-values-shown-in-figures-figheatmap_arena-and-those-reported-by-authors]

The number for Arena-Hard in Figures <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a> and <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a> is computed with **all** model scores reported on Arena-Hard’s leaderboard as of May 01, 2024, using the Arena Elo (En) values from the same date. This discrepancy may arise because the figure reported in Arena-Hard’s blog did not account for all models listed on their leaderboard. The number of models used for each pair of benchmarks shown in Figure <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a> is reported in Figure <a href="#fig:heatmap_stats" data-reference-type="ref" data-reference="fig:heatmap_stats">18</a>. The same applies to AlpacaEval-2.0 and WildBench.

## Why GPQA is not included in Figures <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a> and <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a>? [why-gpqa-is-not-included-in-figures-figheatmap_arena-and-figheatmap_main]

Because we didn’t find enough data points for GPQA that share enough common models with other benchmarks.

## Is MixEval totally unbiased?

No, `MixEval` is not entirely unbiased. This is due to several factors: the detection pipeline is not perfectly accurate, Common Crawl data collection introduces biases, and there are inherent biases from web users in the real world. However, `MixEval` is relatively less biased because it draws from a broad internet user base. This is supported by: (1) The maps in Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a>, which are ordered by their cluster distances to our identified web queries, indicate that wild datasets align more closely with our web queries than with other LLM benchmarks. This demonstrates the robustness of our web query detection pipeline and the solid grounding of `MixEval`. (2) As discussed in Section <a href="#sec:llm_benchmark_analysis" data-reference-type="ref" data-reference="sec:llm_benchmark_analysis">2</a>, ShareGPT, with its extensive user base (100M) compared to other wild datasets (0.1M-0.2M), shows the highest similarity to our web queries, which are based on the global internet user population (5.4B). This further validates the accuracy of our web query detection. (3) The trained web query detector achieved high recall (\>99%) and precision (\>98%) on our internal web query detection benchmarks.

## Will the pipeline that creates MixEval-Hard introduce some noise that influences the result?

`MixEval-Hard` sampling relies on the difficulty scores of benchmark questions, which inherently include some dataset annotation errors. During our error case study (Section <a href="#sec:error_analysis" data-reference-type="ref" data-reference="sec:error_analysis">4.6</a>), we identified several annotation issues. However, the number of annotation errors was minimal, rendering the noise negligible. Furthermore, the high correlation with Arena Elo demonstrates that the introduced noise does not significantly affect the model rankings.

# Considerations of Web User Query Crawling [sec:consideration_web_q_crawl]

Our user queries are not directly crawled from the web; instead, they are identified using Common Crawl, an openly available corpus of web crawl data widely used in the research community. Furthermore, we do not release the raw detected queries, while only releasing the final mixed version of `MixEval` for two reasons: (1) the raw detected queries may contain toxic content or unexpected sensitive information, and (2) we update our benchmarks dynamically to avoid contamination. Releasing the detected raw queries would make the dynamic benchmarking process more predictable, reducing its effectiveness.

<figure id="fig:heatmap_stats">
<img src="./figures/heatmap_intersecutives.png"" style="width:100.0%" />
<figcaption>The number of models used for each pair of benchmarks shown in Figure <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a>.</figcaption>
</figure>

# Implementation details for Benchmark Correlation Matrix, Query Distribution, and Evaluation Cost [sec:corr_querydist_setup]

**Correlation Matrix Heatmap (Figures <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a> and <a href="#fig:heatmap_main" data-reference-type="ref" data-reference="fig:heatmap_main">15</a>).** We present the correlation matrix of prominent benchmarks, where warmer colors indicate higher correlations. Model scores are collected from various sources, including the Chatbot Arena Leaderboard `\citep{chiang2024chatbot}`{=latex}, Open LLM Leaderboard `\citep{openllmleaderboard}`{=latex}, and OpenCompass Leaderboard `\citep{contributors2023opencompass}`{=latex}. Our data collection adheres to three principles: (1) We exclude scores reported by model authors, relying solely on evaluation leaderboards to ensure fairness. (2) For each benchmark, scores are sourced from a single platform to eliminate the influence of varying evaluation settings on model rankings. (3) When multiple sources are available for a benchmark, we select the one with the highest number of models in common with other benchmarks. The number of common models for each pair of benchmarks is detailed in Figure <a href="#fig:heatmap_stats" data-reference-type="ref" data-reference="fig:heatmap_stats">18</a>.

**Query Distribution Map (Figure <a href="#fig:benchmark_distribution" data-reference-type="ref" data-reference="fig:benchmark_distribution">2</a>).** We present the distribution of benchmark queries sorted by their distance to our detected web queries. Each benchmark (orange or yellow) is plotted against the detected wild queries (blue). We uniformly sampled 1000 queries from each LLM benchmark and wild dataset, with a sampling number of 200 for MT-Bench and Arena-Hard due to their smaller sizes. We combined the query embeddings and reduced their dimensions to the same 2-D space to facilitate direct comparisons of the benchmark query distributions. A detailed case study revealed that the reduced space primarily represents the topics of the queries, with queries on similar topics clustering in specific regions of the map. To better understand the topic distribution of different benchmarks, we divided the map into 16 patches based on location (Figure <a href="#fig:topic_map" data-reference-type="ref" data-reference="fig:topic_map">3</a>). We then uniformly sampled 100 queries from each patch and used GPT-4 to summarize the topics of the sampled queries. As illustrated in Figure <a href="#fig:topic_map" data-reference-type="ref" data-reference="fig:topic_map">3</a>, the 2-D query distribution exhibits a distinct regional trend: queries located higher on the map are more technical. The distribution transitions from non-technical topics, such as Social Interactions, at the bottom to technical ones, such as Programming and Mathematics, at the top.

<figure id="fig:eval_cost_breakdown">
<img src="./figures/eval_cost_breakdown.png"" />
<figcaption>Evaluation cost breakdown for the cost estimation in Figure <a href="#fig:heatmap_arena" data-reference-type="ref" data-reference="fig:heatmap_arena">1</a>. The total evaluation cost is broken down into the inference cost and judge cost. </figcaption>
</figure>

**Evaluation Cost Estimation.** As illustrated in Figure <a href="#fig:eval_cost_breakdown" data-reference-type="ref" data-reference="fig:eval_cost_breakdown">19</a>, we consider two costs when evaluating the performance of GPT-3.5-Turbo-0125 on each benchmark: the inference cost and the judging (scoring) cost. The inference cost computation for ground-truth-based and LLM-as-judge benchmarks are straightforward, involving only the estimation of model input and output tokens for each benchmark. We estimate the model output tokens to be 20 for ground-truth-based benchmarks and 329 for open-ended benchmarks [^9]. To compute the evaluation cost of GPT-3.5-Turbo-0125 on the Chatbot Arena, we use the voting number of GPT-3.5-Turbo-0125 on the Chatbot Arena leaderboard as its query count, with each query’s token count estimated from the Chatbot Arena Conversations dataset. Since models on Chatbot Arena are evaluated pairwise, both input and output tokens are doubled. The judging costs[^10] for ground-truth-based and LLM-as-judge benchmarks are estimated similarly, accounting for the input and output tokens of the model parser/judge. However, estimating the human judgment cost for Chatbot Arena is more complex. We reference the crowdsourcing price for Amazon Mechanical Turk (MTurk), specifically the rate for a Facebook Account Holder[^11] ($0.05 per vote). Under this pricing scheme, evaluating a single model on Chatbot Arena costs approximately $2936, making it a highly expensive process.

# Detailed Statistics of the Benchmark Correlation Matrix

The detailed statistics on the number of models used for each pair of benchmarks are presented in Figure <a href="#fig:heatmap_stats" data-reference-type="ref" data-reference="fig:heatmap_stats">18</a>.

# Error Cases [sec:error_cases]

<div class="promptbox" markdown="1">

blue **Question:**  
Letter to the editor: You say that if the government were to confiscate a portion of the wages of convicted burglars when they reenter the workforce, it would be a form of stealing, hence an abuse of power. Yet under the proposal now being considered, the government would confiscate such wages in order to fund an account to compensate burglary victims. So even if confiscating a portion of burglars’ wages were a form of stealing, it would still be justified.  
Which one of the following principles, if valid, most helps to support the argument in the letter to the editor?  
A. Money stolen from a burglar should be given to that burglar’s victims.  
B. Burglars are obligated to provide compensation to the same individuals they victimized.  
C. The motive prompting an action determines whether or not that action is justified.  
D. A crime is justified only if it is a means of compensating people who deserve compensation.  
E. Stealing is never justified even if it benefits someone who has been a burglary victim.  
Answer with the option letter from the given choices directly.  
**Answer:** C  
**Split:** AGIEval  
**Main Causes of Errors:** Hard Reasoning  
**Wrong Answers:**  
  
`Claude 3 Haiku:` The principle that most helps to support the argument in the letter to the editor is (B) Burglars are obligated to provide compensation to the same individuals they victimized.  
  
The letter argues that even if confiscating a portion of burglars’ wages is a form of stealing, it would still be justified because the purpose is to fund an account to compensate burglary victims. This suggests that the burglars have an obligation to provide compensation to the individuals they victimized, which aligns with principle (B).  
  
`Claude 3 Sonnet:` D  
  
`Claude 3 Opus:` B  
  
`Gemini 1.0 Pro:` B  
  
`GPT-3.5-Turbo:` D. A crime is justified only if it is a means of compensating people who deserve compensation.  
  
`GPT-4-Turbo:` B.  
  
`Mistral-Small:` The principle that most helps to support the argument in the letter to the editor is B. Burglars are obligated to provide compensation to the same individuals they victimized. This principle aligns with the argument that even if confiscating a portion of burglars’ wages were a form of stealing, it would still be justified because it serves to compensate burglary victims.  
  
`Mistral-Medium:` (B) Burglars are obligated to provide compensation to the same individuals they victimized.  
  
Explanation:  
The principle that supports the argument in the letter to the editor is that burglars have an obligation to provide compensation to the individuals they have victimized. This principle justifies the government’s confiscation of a portion of the burglars’ wages to fund an account to compensate their victims. While the letter acknowledges that confiscating wages could be considered a form of stealing, it argues that this action would still be justified as a means of providing compensation to those who deserve it. Therefore, option (B) is the best choice among the given options as it directly supports the argument in the letter. Options (A), (C), (D), and (E) do not directly address the obligation of burglars to provide compensation to their victims, which is the crux of the argument.  
  
`Mistral-Large:` The principle that most helps to support the argument in the letter to the editor is (B) "Burglars are obligated to provide compensation to the same individuals they victimized." This principle justifies the confiscation of a portion of burglars’ wages as a means to compensate their victims, which is the main point of the argument in the letter.  
  
`Reka Edge:` The principle that most helps to support the argument in the letter to the editor is \*\*(D) A crime is justified only if it is a means of compensating people who deserve compensation.\*\* This principle suggests that a criminal act can only be justified if it results in some form of compensation for the victims of the crime. In the scenario described in the letter, the government is proposing to confiscate a portion of the wages of convicted burglars to fund an account to compensate burglary victims. The argument is that even though confiscating a portion of the wages of burglars may be seen as a form of stealing, it is justified because it is a means of compensating people who deserve compensation. Thus, the act of confiscating wages in this case is seen as a legitimate way of achieving a just outcome (compensating victims).  
  
`Reka Core:` The principle that most helps to support the argument in the letter to the editor is:  
  
B. Burglars are obligated to provide compensation to the same individuals they victimized.  
  
The argument in the letter acknowledges that confiscating a portion of a convicted burglar’s wages might be considered stealing, but it contends that this action is still justified because it serves the purpose of compensating burglary victims. This aligns with the principle that burglars have an obligation to provide compensation to those they have harmed. The letter suggests that even if the government’s action is considered stealing, it is still justified because it fulfills the burglar’s obligation to compensate their victims.

</div>

<div class="promptbox" markdown="1">

blue **Question:**  
At which stage in the planning process are the objectives decided on?  
A. Defining the program  
B. Planning the program  
C. Taking action and implementing ideas  
D. Evaluation of the program  
Answer with the option letter from the given choices directly.  
**Answer:** B  
**Split:** MMLU  
**Main Causes of Errors:** Strong Domain Knowledge  
**Wrong Answers:**  
  
`Claude 3 Haiku:` A. Defining the program  
  
`Claude 3 Opus:` A  
  
`Gemini 1.0 Pro:` A  
  
`Mistral-Small:` A. Defining the program  
  
At the defining stage of the planning process, the objectives are decided upon. This is where you outline the goals and desired outcomes for your program.  
  
`Mistral-Medium:` A. Defining the program  
  
At the defining stage of the planning process, the objectives are decided upon. This is where you outline the goals and desired outcomes for your program.  
  
`Reka Edge:` The objectives are decided \*\*A. Defining the program\*\*. During this stage, the program’s goals and desired outcomes are identified and clearly stated. These objectives serve as the foundation for the rest of the planning process and guide the development of the program’s activities, timeline, and budget. Once the objectives are established, they can be further refined and prioritized during the planning and implementation stages, and evaluated at the end of the program to measure its success.  
  
`Reka Flash:` A. Defining the program. The objectives are typically decided upon during the defining or conceptualization stage of the planning process, where the program’s purpose, goals, and target audience are identified. This stage sets the foundation for the rest of the planning process, including the development of strategies, action plans, and evaluation criteria.  
  
`Reka Core:` The objectives are decided on during the defining the program stage (Option A). This is the initial stage of the planning process where the program’s purpose, scope, and goals are clearly defined. The objectives serve as the foundation for the rest of the planning process, guiding the development of strategies, activities, and evaluation methods. Once the objectives are established, the planning process moves on to the next stage, which is planning the program (Option B), followed by taking action and implementing ideas (Option C), and finally, evaluating the program (Option D).

</div>

<div class="promptbox" markdown="1">

blue **Question:**  
How many men has actress Elizabeth Taylor been divorced from?  
A. five  
B. six  
C. seven  
D. eight  
Answer with the option letter from the given choices directly.  
**Answer:** C  
**Split:** MMLU  
**Main Causes of Errors:** Strong Domain Knowledge, Hard Reasoning, Vague Definition of the Question  
(Elizabeth Taylor, the iconic actress, was married a total of eight times to seven different men. She married Richard Burton twice. Here is a list of her husbands: Conrad Hilton (1950-1951), Michael Wilding (1952-1957), Michael Todd (1957-1958), Eddie Fisher (1959-1964), Richard Burton (1964-1974), Richard Burton (1975-1976), John Warner (1976-1982), Larry Fortensky (1991-1996). Here Michael Todd died in a plane crash. So if consider death as a form of divorce, then the answer is C, otherwise B.)  
**Wrong Answers:**  
  
`Claude 3 Sonnet:` D. eight  
  
`Gemini 1.0 Pro:` A  
  
`Gemini 1.5 Pro:` D.  
  
`GPT-3.5-Turbo:` D. eight  
  
`GPT-4-Turbo:` A. five  
  
`Mistral-Medium:` B. six  
Elizabeth Taylor was married eight times, but she was divorced six times as two of her marriages ended due to the death of her spouses. The men she divorced are Conrad Hilton Jr., Michael Wilding, Mike Todd, Eddie Fisher, Richard Burton (twice), and John Warner.  
  
`Reka Edge:` According to various sources and biographies, actress Elizabeth Taylor has been divorced from \*\*six\*\* men. Therefore, the correct answer is \*\*B. six\*\*.  
  
`Reka Flash:` Elizabeth Taylor was married eight times and divorced seven of those times. So, the correct answer is D. eight.  
  
`Reka Core:` Elizabeth Taylor was married eight times to seven different men. So the correct answer is:  
  
D. eight.

</div>

# Rule Parser and Model Parser Comparison [rule_model_parser_comparison]

<figure id="fig:curve_rule_model">
<figure id="fig:curve_mp_rule_model">
<img src="./figures/curve_mp_rule_model.png"" />
<figcaption>Multiple-choice</figcaption>
</figure>
<figure id="fig:curve_ff_rule_model">
<img src="./figures/curve_ff_rule_model.png"" />
<figcaption>Free-form</figcaption>
</figure>
<figcaption>The score differences computed by model parser and rule parser on <code>MixEval</code>. The rule parser is unstable on both free-form and multiple-choice splits, especially free-form. </figcaption>
</figure>

As illustrated in Figure <a href="#fig:curve_rule_model" data-reference-type="ref" data-reference="fig:curve_rule_model">22</a>, the rule parser exhibits instability in both free-form and multiple-choice splits, with a pronounced effect in the free-form case. Manual inspection reveals that in multiple-choice scenarios, the discrepancies primarily arise from the rule parser’s failure to capture the diverse output styles of different models. In contrast, the discrepancies in free-form scenarios stem from the expansive output space of free-form questions and the varying annotation comprehensiveness across different splits of `MixEval`.

# Model Parser Prompts [parser_prompts]

<div class="promptbox" markdown="1">

light green **System:**  
In this task, I want you to act as a judge.  
  
**User:**  
You will be provided with a question, its golden answer(s), and the model’s answer, while the context of the question is not given here. Your task is to judge how correct the model’s answer is based on the golden answer(s), without seeing the context of the question, and then give a correctness score. The correctness score should be one of the below numbers: 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Your should first briefly give your reasoning process regarding how the model’s answer conforms to or contradicts the golden answer(s), and then give the correctness score. The correctness score must strictly follow this format: "\[\[score\]\]", e.g., "The correctness score: \[\[0.5\]\]". Below are some examples.  
Example 1:  
Question: Sandy bought 1 million Safe Moon tokens. She has 4 siblings. She wants to keep half of them to herself and divide the remaining tokens among her siblings. After splitting it up, how many more tokens will she have than any of her siblings?  
Golden Answer(s): \<answer 1\> 375000  
Model’s Answer: Sandy will have more tokens than any sibling by 3/8 million.  
Your Judgment: The golden answer states that Sandy will have 375,000 more tokens than any of her siblings, which is a precise numerical value. The model’s answer translates this scenario into a fraction of the total, saying Sandy will have more tokens than any sibling by 3/8 million. 1 million tokens \* 3/8 = 375,000 tokens. So the model provided an answer in fractional form that, when converted to a numerical value, exactly matches the golden answer’s quantity. The correctness score: \[\[1.0\]\].  
Example 2:  
Question: what car was used in the movie christine  
Golden Answer: \<answer 1\> a vintage 1958 Plymouth Fury; \<answer 2\> 1958 Plymouth Fury  
Model’s Answer: Christine.  
Your Judgment: The golden answers specify the car used in the movie "Christine" as a vintage 1958 Plymouth Fury, providing a clear and detailed response including the make, model, and year of the car. The model’s answer, though points out the car’s alias in the context of the movie "Christine", is not precise and specific enough. The correctness score: \[\[0.5\]\].  
Example 3:  
Question: In 2015 Edgar Lungu became prime minister of?  
Golden Answer: \<answer 1\> Zambia; \<answer 2\> Zamibia; \<answer 3\> People of Zambia; \<answer 4\> Zambian cuisine; \<answer 5\> Zambians; \<answer 6\> Culture of Zambia; \<answer 7\> Etymology of Zambia; \<answer 8\> Zambia; \<answer 9\> Health care in Zambia; \<answer 10\> ISO 3166-1:ZM; \<answer 11\> Republic Of Zambia; \<answer 12\> Cuisine of Zambia; \<answer 13\> Sport in Zambia; \<answer 14\> Republic of Zambia; \<answer 15\> Zambian people; \<answer 16\> Name of Zambia  
Model’s Answer: Prime Minister  
Your Judgment: The golden answers provide a detailed list of entities all relating to Zambia, indicating that Edgar Lungu became the leader (specifically, they mentioned "prime minister") of Zambia in 2015. The model’s answer, "Prime Minister," merely repeats part of the question without answering it. The correctness score: \[\[0.0\]\].  
Note that each one of the golden answers is considered correct. Thus if the model’s answer matches any one of the golden answers, it should be considered correct. Judge the below case, give the brief reasoning process and the correctness score.  
Question: \<prompt\>  
Golden Answer(s): \<golden answers\>  
Model’s Answer: \<model response\>  
Your Judgment:  
  

</div>

<div class="promptbox" markdown="1">

light green **System:**  
In this task, I want you to act as an option extractor.  
  
**User:**  
You will be provided with a multiple-choice question, its options, and the model’s answer, while the context of the question is not given here. Your task is to extract or judge which option is chosen by the model based on its response, without seeing the context of the question. The extracted option should be one of the provided option letters. Your should first briefly give your reasoning process, and then give the extracted option letter. The extracted option must strictly follow this format: "\[\[option letter\]\]", e.g., "The option chosen by the model: \[\[A\]\]". Below are some examples.  
Example 1:  
Question: Which technology was developed most recently?  
Options:  
A. cellular telephone  
B. television  
C. refrigerator  
D. airplane  
Model’s Answer: The technology that was developed most recently is D. airplane.  
Your Judgment: The model’s response directly identifies "D. airplane" as the technology that was developed most recently. This indicates that the chosen option is D. The option chosen by the model: \[\[D\]\].  
Example 2:  
Question: What monotheistic religion is based on the life and teachings of Jesus Christ of Nazareth?  
Options:  
A. Islam  
B. Christianity  
C. Hinduism  
D. Buddhism  
Model’s Answer: B.  
What is the name of the first person to be executed by the electric chair? A. John Wilkes Booth B. William Kemmler C. John Dillinger D. Bonnie and Clyde Answer with the option letter  
Your Judgment: The model’s response clearly identifies "B. Christianity" as the monotheistic religion based on the life and teachings of Jesus Christ of Nazareth. This directly answers the first question posed, making B the selected option for that question. The additional content appears to introduce a new, unrelated question without providing an answer to it. The option chosen by the model: \[\[B\]\].  
Example 3:  
Question: Which solution is correct?  
Options:  
A. provide homes for people  
B. provide homes for animals  
Model’s Answer: Neither A nor B is entirely correct because trees do not p̈rovide homesïn the traditional sense. However, they do provide habitats and shelter for various organisms, including animals. If you had to choose between the options given, option B (for animals) might be more accurate in the context of trees being a habitat. But it’s important to note that trees also benefit humans by providing oxygen, shade, and contributing to urban green spaces. If you need to select one option, I would suggest:  
B. provide homes for animals  
Your Judgment: The model’s response indicates a preference for option B, mentioning that if one had to choose between the given options, "B. provide homes for animals" would be more accurate, especially in the context of trees serving as habitats. This direct mention of option B as the more suitable choice, despite the initial hesitation, clearly indicates that the chosen option is B. The option chosen by the model: \[\[B\]\].  
Question: \<prompt\>  
Options:  
\<options\>  
Model’s Answer: \<model response\>  
Your Judgment:  
  

</div>

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification: We carefully inspected the main claims we made in the abstract and introduction, and ensure that all of them are appropriate.

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: Limitations are detailed as part of Section <a href="#sec:FAQ" data-reference-type="ref" data-reference="sec:FAQ">7</a> in the form of QAs.

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

14. Justification: There is no theoretical result made.

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

19. Justification: All implementation details are provided (in Experiment settings and Appendix).

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

24. Justification: We uploaded the code and data, and will release and periodically update the benchmark data and its related code.

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

29. Justification: We carefully reason our choices in our method and experiment section.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: We provide a significance test for the dynamic benchmarking by running 5 versions of 5 random seeds. While for other experiments, it’s too expensive to run multiple times.

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

39. Justification: We specify the hardware we use in the Experiment Settings.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification: We carefully read and follow the NeurIPS Code of Ethics through out the whole research process.

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: This research focuses on an improved benchmarking method for LLMs, which does not directly involve significant societal impacts. Its primary influence is within the LLM research community, as detailed in Section <a href="#sec:introduction" data-reference-type="ref" data-reference="sec:introduction">1</a>.

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

54. Justification: Discussed in Section <a href="#sec:consideration_web_q_crawl" data-reference-type="ref" data-reference="sec:consideration_web_q_crawl">8</a>.

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification: We properly cite all the previous work, datasets, or software we use.

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

64. Justification: We upload and release our data and code with detailed instrutions.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification: No crowdsourcing involved.

70. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

73. Answer:

74. Justification: Not available.

75. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

[^1]: Core contributors.

[^2]: Correspondence to: Jinjie Ni \<`jinjieni@nus.edu.sg`\>

[^3]: The Chatbot Arena leaderboard is not the sole indicator of real-world human preferences, but it currently serves as one of the gold standards within the community. Therefore, we utilize it as a reliable source of approximation.

[^4]: https://www.mlyearning.org/chatgpt-statistics

[^5]: https://www.statista.com/statistics/273018/number-of-internet-users-worldwide/

[^6]: We subsequently updated several models that had been released after this date.

[^7]: https://www.mlyearning.org/chatgpt-statistics

[^8]: https://www.statista.com/statistics/273018/number-of-internet-users-worldwide/

[^9]: According to the averaged output token for GPT-3.5-Turbo-0125 as presented at: https://lmsys.org/blog/2024-04-19-arena-hard/.

[^10]: The judge cost for Arena-Hard and MT-Bench is directly taken from https://lmsys.org/blog/2024-04-19-arena-hard/.

[^11]: https://requester.mturk.com/pricing
