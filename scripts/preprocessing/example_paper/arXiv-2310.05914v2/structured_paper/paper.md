# `NEFTune`: Noisy Embeddings Improve Instruction Finetuning

## Abstract

We show that language model finetuning can be improved, sometimes dramatically, with a simple augmentation. `NEFTune` adds noise to the embedding vectors during training. Standard finetuning of LLaMA-2-7B using Alpaca achieves $`29.79`$% on AlpacaEval, which rises to $`64.69`$% using noisy embeddings. `NEFTune` also improves over strong baselines on modern instruction datasets. Models trained with Evol-Instruct see a $`10`$% improvement, with ShareGPT an $`8`$% improvement, and with OpenPlatypus an $`8`$% improvement. Even powerful models further refined with RLHF such as LLaMA-2-Chat benefit from additional training with `NEFTune`.[^1]

maketitle thanks aketitle

# Introduction

The ability of LLMs to follow detailed instructions is vital to their usefulness. Generative language models are typically trained on raw web data, and then subsequently fine-tuned on a comparatively small but carefully curated set of instruction data. Instruction fine-tuning is crucial to taming the power of LLMs, and the usefulness of a model is largely determined by our ability to get the most out of small instruction datasets.

In this paper, we propose to add random noise to the embedding vectors of the training data during the forward pass of fine-tuning. We show that this simple trick can improve the outcome of instruction fine-tuning, often by a large margin, with no additional compute or data overhead. <u>N</u>oisy <u>E</u>mbedding Instruction <u>F</u>ine <u>T</u>uning (`NEFTune`), while simple, has a strong impact on downstream conversational quality. When a raw LLM like LLaMA-2-7B is finetuned with noisy embeddings, its performance on `AlpacaEval` improves from 29.8% to 64.7% (Figure <a href="#fig:figure_1_results_AlpacaEval" data-reference-type="ref" data-reference="fig:figure_1_results_AlpacaEval">1</a>) – an impressive boost of around 35 percentage points . `NEFTune` leads to this surprising and large jump in performance on conversational tasks, maintaining performance on factual question answering baselines. This technique seems to be a free lunch for LLM fine-tuning.

<figure id="fig:figure_1_results_AlpacaEval">
<img src="./figures/AlpacaEval_BlueColors_Figue1.png"" style="width:85.0%" />
<figcaption><code>AlpacaEval</code> Win Rate percentage for LLaMA-2-7B models finetuned on various datasets with and without <code>NEFTune</code>. <code>NEFTune</code> leads to massive performance boosts across all of these datasets, showcasing the increased conversational quality of the generated answers. </figcaption>
</figure>

## Related Work

The earliest forms of instruction finetuning such as FLAN and T0 focused on cross-task generalization in language models. Encoder-decoder language models were finetuned on a broad range of NLP tasks (about 100) and then evaluated on a set of different tasks. This was later scaled up to include thousands of tasks, seeing further improvement over the original FLAN . Although these works showed that LLMs could be easily adapted to solve simple and classical NLP tasks, real-world scenarios require LLMs to provide free-form answers to open-ended queries.

InstructGPT was the first model to tackle open-ended queries with impressive performance. OpenAI further trained GPT-3 using reinforcement learning from human feedback (RLHF) to *align* the model. This procedure gave rise to highly popular models like ChatGPT that captured the imagination of the general public and generated longer coherent text than its InstructGPT predecessor.

This led to the work of (*Self-Instruct*), which used InstructGPT (Text-Davinci-003) to produce instruction-output pairs which could be used to finetune the foundation models like LLaMA into instruction following variants like Alpaca . Through the rise in popularity of distilled models , the community has constructed other datasets distilling in particular ways from other models like ChatGPT, including . In another approach, ShareGPT was constructed by crowd sourcing real user conversations from ChatGPT. Other datasets like construct a dataset to improve specific aspects of the model like STEM question answering and logical reasoning. AlpaGasus filters data by quality (according to GPT-4) to improve performance.

It should be noted that noisy inputs have been used to improve models in various ways. The first instance of noise being used to improve language models was the FreeLB method by , who observed that adversarial perturbations boosted the performance of MLM models. The noise in this case is not random, but is rather computed by first adding a small Gaussian perturbation to the embeddings and then using a gradient step to find the perturbation that maximally alters model performance. This adversarial augmentation approach also improves model performance on graphs . While our proposed scheme is non-adversarial, we adopt the noise scaling rules from these works. Training on noisy inputs has also been done for other applications, such as to improve image captioning systems , and as a common component of early differential privacy mechanisms .

# `NEFTune`: Noisy Embedding Instruction Finetuning

Instruction models are trained on datasets comprising pairs of instructions and responses. Each step of `NEFTune` begins by sampling an instruction from the dataset, and converting its tokens to embedding vectors. `NEFTune` then departs from standard training by adding a random noise vector to the embeddings. The noise is generated by sampling iid uniform entries, each in the range $`[-1,1]`$, and then scaling the entire noise vector by a factor of $`\alpha/\sqrt{Ld},`$ where $`L`$ is the sequence length, $`d`$ is the embedding dimension, and $`\alpha`$ is a tunable parameter.

This scaling rule was borrowed from the adversarial ML literature , and results in a random vector with an expected Euclidean magnitude of approximately $`\alpha/\sqrt{3}.`$ Algorithm <a href="#alg:NEFTune" data-reference-type="ref" data-reference="alg:NEFTune">2</a> describes our method in detail.

<figure id="alg:NEFTune">
<div class="minipage">
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><br />
<strong>Input:</strong> <span class="math inline">𝒟 = {<em>x</em><sub><em>i</em></sub>, <em>y</em><sub><em>i</em></sub>}<sub>1</sub><sup><em>N</em></sup></span> tokenized dataset, embedding layer <span class="math inline">emb(⋅)</span>, rest of model <span class="math inline"><em>f</em><sub>/emb</sub>(⋅)</span>,<br />
model parameters <span class="math inline"><em>θ</em></span>, <span class="math inline">loss(⋅)</span>, optimizer <span class="math inline">opt(⋅)</span><br />
<code>NEFT</code> Hyperparameter: base noise scale <span class="math inline"><em>α</em> ∈ ℝ<sup>+</sup></span></p>
<p><br />
Initialize <span class="math inline"><em>θ</em></span> from a pretrained model.</p>
<p><br />
<strong>Repeat</strong> <span>  <span class="math inline">(<em>X</em><sub><em>i</em></sub>, <em>Y</em><sub><em>i</em></sub>) ∼ 𝒟</span></span> # comment: <span>sample a minibatch of data and labels</span><br />
<span class="math inline"><em>X</em><sub>emb</sub> ← emb(<em>X</em><sub><em>i</em></sub>), ℝ<sup><em>B</em> × <em>L</em> × <em>d</em></sup></span> # comment: <span>batch size <span class="math inline"><em>B</em></span>, seq. length <span class="math inline"><em>L</em></span>, embedding dimension <span class="math inline"><em>d</em></span></span><br />
<span class="math inline"><em>ϵ</em> ∼ Uniform(−1, 1), ℝ<sup><em>B</em> × <em>L</em> × <em>d</em></sup></span> # comment: <span>sample a noise vector</span><br />
<span class="math inline">$X_{\text{emb}}' \gets X_{\text{emb}} + (\frac{\alpha}{\sqrt{Ld}}) \epsilon$</span> # comment: <span>add scaled noise to embeds <a href="#fn1" class="footnote-ref" id="fnref1" role="doc-noteref"><sup>1</sup></a></span><br />
<span class="math inline"><em>Ŷ</em><sub><em>i</em></sub> ← <em>f</em><sub>/emb</sub>(<em>X</em><sub>emb<sup>′</sup></sub>)</span> # comment: <span>make prediction at noised embeddings</span><br />
<span class="math inline"><em>θ</em> ← opt(<em>θ</em>, loss(<em>Ŷ</em><sub><em>i</em></sub>, <em>Y</em><sub><em>i</em></sub>))</span> # comment: <span>train step, e.g., grad descent</span><br />
<strong>Until</strong> <span>Stopping criteria met/max iterations.</span></p>
</div>
</div>
<figcaption><code>NEFTune</code>: <strong>N</strong>oisy <strong>E</strong>mbedding Instruction <strong>F</strong>ine<strong>tun</strong>ing</figcaption>
</figure>
<section id="footnotes" class="footnotes footnotes-end-of-document" role="doc-endnotes">
<hr />
<ol>
<li id="fn1"><p>If sequence lengths in a batch are not equivalent, then <span class="math inline"><em>L</em></span> is a vector <span class="math inline"> ∈ ℤ<sub> &gt; 0</sub><sup><em>B</em></sup></span> and the scaling factor <span class="math inline">$(\alpha/\sqrt{Ld})$</span> is computed independently for each sequence in batch.<a href="#fnref1" class="footnote-back" role="doc-backlink">↩︎</a></p></li>
</ol>
</section>

# Experimental Set-up

## Models

We conduct the majority of our experiments using 7B parameter LLMs. Particularly, we use LLaMA-1, LLaMA-2, and OPT-6.7B . These similarly shaped transformers mostly differ in tokens seen during training. OPT, LLaMA-1, and LLaMA-2 were trained using $`180`$B, $`1`$T, and $`2`$T tokens respectively. This difference is to be reflected in model performance on standard benchmarks like MMLU, with LLaMA-2 performing the best and OPT performing the worst. For the 13B and 70B parameter models, we train LLaMA-2. Additionally, we improve RLHF models by finetuning the highly refined LLaMA-2-Chat (7B) model.

## Instruction Finetuning Datasets

We focus on the following finetuning datasets either because of their wide popularity, or because they have yielded state-of-the-art results in the recent past. Note that we use only single-turn datasets because of the memory constraints of our hardware setup.

- **Alpaca**  was constructed using the *Self-Instruct* method of , and the Text-Davinci-003 model . Self-Instruct uses a small seed set of tasks to construct new instruction tuning tasks and filter out bad ones.

- **Evol-Instruct**  contains 70k single-turn instructions that are considered more complex than Alpaca. This dataset was derived from the Alpaca dataset by using ChatGPT to *evolve* the initial instructions.

- **Open-Platypus**  is a curated dataset amalgamated from $`11`$ open-source datasets, curated specifically towards improving LLM performance in STEM and logical domains. This set contains $`25`$k questions where $`\approx 10\%`$ are LLM-generated and the remainder human-written.

- **ShareGPT**  is a dataset of $`70`$K voluntarily-shared ChatGPT conversations . Although ShareGPT is multiturn, we use the dataset version from Vicuna-v1.1 and split the multi-turn conversations closer to a single-turn format.

Additionally, we finetune all models with the Alpaca system prompt, except for ShareGPT, where we use the Vicuna system prompt. The hyperparameters can be found in Appendix <a href="#sec:hyperparameters" data-reference-type="ref" data-reference="sec:hyperparameters">9.1</a>. We set our hyperparameters through a coarse sweep on LLaMA-1 ($`7`$B) trained on the Alpaca dataset, where we see $`6\%`$ improvement over the standard Alpaca model. We use these as the defaults on all models.

## Evaluation

Since we train using largely single-turn data, we evaluate the model’s conversational abilities using `AlpacaEval`. We also evaluate the tasks from the OpenLLM Leaderboard to determine if the `NEFTune` augmentation causes any loss in performance on standard multiple choice tasks.

**AlpacaEval.** The `AlpacaEval` dataset released by is used to evaluate the overall quality of generations. `AlpacaEval` is an automatic model-based evaluation that compares Text-Davinci-003 generations to the model generations over $`805`$ instructions with the Win Rate reported. The Win Rate is the rate at which the model in question is preferred to Text-Davinci-003 as determined by model evaluator (GPT-4). The $`805`$ test prompts are scraped from Vicuna, koala, Anthropic’s hh-rlhf, and other sources, making it a fairly comprehensive and diverse test. Additionally, `AlpacaEval` has high agreement with humans (validated on $`20`$K annotations). We believe at the $`7`$B and $`13`$B scale this evaluation is still quite reasonable. We use both GPT-4 and ChatGPT as evaluators. We use ChatGPT as a precursor test to determine which models to evaluate on GPT-4. This is due to the cost and API restrictions of GPT-4.

**Hugging Face OpenLLM Leaderboard.** The evaluation datasets used for leaderboard ranking are the verbalized multiclass classification datasets ARC , HellaSwag , MMLU , and TruthfulQA . This combination of datasets broadly evaluates the ability of a LLM to respond to factual questions and reasoning challenges, and we evaluate these datasets to confirm that model capabilities are not negatively impacted by `NEFTune`.

# Results

#### `NEFTune` Improves Text Quality.

From Table <a href="#tab:LLaMA-2_GPT-4" data-reference-type="ref" data-reference="tab:LLaMA-2_GPT-4">1</a>, we can see an increase across all datasets for the $`7`$B scale with an average increase of $`15.1\%`$, showing that training with `NEFT` significantly improves conversational ability and answer quality, as measured via `AlpacaEval`. Additionally, we can see from Figure <a href="#fig:AlpacaEval_model_dataset_ablation" data-reference-type="ref" data-reference="fig:AlpacaEval_model_dataset_ablation">3</a> that we also see improvements on older models, such as LLaMA-1 and OPT. Interestingly, we see less improvement on ShareGPT than on other datasets according to ChatGPT. However, this is not reflected in the GPT-4 evaluation. From Table <a href="#tab:llama-2-chat_SFT" data-reference-type="ref" data-reference="tab:llama-2-chat_SFT">2</a>, we see the Win Rate climbs from 75.03% to 88.81% (+13.78%) after adding `NEFTune` to the $`70`$B parameter model trained on Evol-Instruct (hyperparameters in Appendix <a href="#sec:hyperparameters" data-reference-type="ref" data-reference="sec:hyperparameters">9.1</a>).

<div id="tab:LLaMA-2_GPT-4">

|            | Alpaca | Evol-Instruct | ShareGPT | OpenPlatypus | Average |
|:-----------|:------:|:-------------:|:--------:|:------------:|:-------:|
| LLaMA-2 7B | 29.79  |     70.34     |  68.74   |    62.00     |  57.71  |
| \+`NEFT`   | 64.69  |     79.60     |  76.28   |    70.61     |  72.80  |

`AlpacaEval` Win Rate versus Text-Davinci-003 for LLaMA-2 trained on different datasets, using GPT-4 as the evaluator, showing an average improvement of $`15`$% across all datasets.

</div>

<figure id="fig:AlpacaEval_model_dataset_ablation">
<img src="./figures/Ablation_Model_Dataset_ChatGPT.png"" />
<figcaption><code>AlpacaEval</code> Win Rate with and without <code>NEFTune</code> on LLaMA-2, LLaMA-1, and OPT across Alpaca, Evol-Instruct, ShareGPT and OpenPlatypus datasets. Performance improves across different datasets and models with ChatGPT as the evaluator.</figcaption>
</figure>

<figure id="fig:OpenLLM_leaderboard_performance">
<p><img src="./figures/OpenLLM_Leaderboard_Llama_2_7B_Alpaca.png"" style="width:44.0%" /> <img src="./figures/OpenLLM_Leaderboard_Llama_2_7B_OpenPlatypus.png"" style="width:44.0%" /> <img src="./figures/OpenLLM_Leaderboard_Llama_2_7B_Evol_Instruct.png"" style="width:44.0%" /> <img src="./figures/OpenLLM_Leaderboard_Llama_1_7B_Evol_Instruct.png"" style="width:44.0%" /></p>
<figcaption>OpenLLM Leaderboard tasks with and without <code>NEFTune</code> on LLaMA-2 across Alpaca, Evol-Instruct, and OpenPlatypus datasets and LLaMA-1 trained on Evol-Instruct. We observe that performance does not change across datasets and models.</figcaption>
</figure>

#### `NEFTune` Can Improve Chat Models.

From Table <a href="#tab:llama-2-chat_SFT" data-reference-type="ref" data-reference="tab:llama-2-chat_SFT">2</a>, we see that further instruction finetuning LLaMA-2 Chat (7B) on Evol-Instruct can boost the performance of LLaMA-2-Chat by $`3\%`$. This model was already extensively tuned, using multiple rounds of RLHF. Yet, with `NEFTune`, we see a sizable, additional performance increase of $`10\%`$, although we note that some capabilities of this checkpoint model may be affected like its ability to refrain from outputting toxic behavior. Nevertheless, it is surprising that the conversation quality of such a refined chat model can be so dramatically improved.

#### Effect on Capabilities.

A potential concern is that `NEFTune` improves conversational ability only at the cost of other classical skills. We evaluate on the *OpenLLM Leaderboard* tasks, using the LM-Eval Harness implementation of MMLU, ARC, HellaSwag, and TruthfulQA. These benchmarks give us a glimpse into model knowledge, reasoning, and truthfulness. Figure <a href="#fig:OpenLLM_leaderboard_performance" data-reference-type="ref" data-reference="fig:OpenLLM_leaderboard_performance">4</a> shows that scores remain stable and that `NEFTune` preserves model capabilities.

<div id="tab:llama-2-chat_SFT">

|               | LLaMA-2 (7B) | LLaMA-2-Chat (7B) | LLaMA-2 (13B) | LLaMA-2 (70B) |
|:--------------|:------------:|:-----------------:|:-------------:|:-------------:|
| Base          |      \-      |                   |      \-       |      \-       |
| Evol-Instruct |    70.34     |       74.44       |     72.61     |     75.03     |
| \+`NEFT`      |    79.60     |       81.74       |     82.04     |     88.81     |

LLaMA-2-Chat (7B), LLaMA-2 (13B), and LLaMA-2 (70B) can be finetuned further to improve performance.

</div>

#### `NEFTune` Works with QLORA.

We show that `NEFTune` also improves performance in constrained resource environments by training with Quantized Low Rank Adapters (QLORA) . We use the implementation from , and the default training hyperparameters for all model weights, training for only one epoch. For $`30`$B, we double the effective batch size and half the learning rate like .

Table <a href="#fig:QLORA_results" data-reference-type="ref" data-reference="fig:QLORA_results">3</a> shows that when training with QLORA, `AlpacaEval` performance increases across all model sizes and datasets studied. However, performance gains are less stark than those seen in full scale finetuning. This may be because different hyperparameters (i.e, number of finetuning epochs) are needed, or because we are heavily quantizing to $`4`$-bits.

<div id="fig:QLORA_results">

|  |  |  |  |  |  |
|:---|:--:|:--:|:--:|:--:|:--:|
| Model | LLaMA2(7B) | LLaMA2(7B) | LLaMA2(13B) | LLaMA2(13B) | LLaMA1(30B) |
| Dataset | Alpaca | Evolve70k | Alpaca | Evolve70k | Alpaca |
| Base | 26.71 | 52.80 | 31.99 | 56.96 | 41.06 |
| \+`NEFT` ($`\alpha=5`$) | 31.43 | 56.96 | 38.20 | 60.99 | 41.12 |
| \+`NEFT` ($`\alpha=10`$) | 28.57 | 54.84 | 41.55 | 61.30 | 43.11 |
| \+`NEFT` ($`\alpha=15`$) | 29.01 | 57.27 | 42.17 | 57.39 | 41.99 |

`AlpacaEval` Win Rate (ChatGPT Eval.) reported across different datasets and model sizes. Even training with QLORA, we can see performance increases across the board, although they are milder than regular finetuning.

</div>

#### A Qualitative Example.

Here we show a qualitative example from LLaMA-2 trained on Alpaca with and without `NEFT`. We choose the *quantum computing prompt* from the Vicuna Eval subset of the `AlpacaEval` instructions. We see from this example that Alpaca produces a response that is shorter, giving only a very basic definition of quantum computing, mentioning qubits, superposition, entanglement, and increased complex computation. In the Alpaca-`NEFT` response, the model provides a more fluid answer, expounding on these topics with clearer explanations of superposition and quantum entanglement, and including a mention of potential applications. We feel this example is representative of the kinds of changes induced by `NEFT`. See the appendix for more examples using different finetuning datasets.

<div class="tcolorbox">

**`AlpacaEval` (Vicuna Subset) Prompt**: Can you explain the basics of quantum computing?  
**Alpaca-2-7B Response**:  
Quantum computing is based on the principles of quantum mechanics, which allow for phenomena such as superposition and entanglement between particles. This means that information can be processed in parallel instead of sequentially like traditional computers. Quantum bits (qubits) are used to store data instead of classical bits, allowing more complex computations to take place with fewer resources than would otherwise be necessary.  
**Alpaca-2-7B-`NEFT` Response**:  
Quantum computing is based on the principles of quantum mechanics, which describe how particles such as electrons and photons behave at the subatomic level. Unlike traditional computers, which process information in binary digits (bits) that can be either 0 or 1, quantum bits (qubits) can exist in multiple states between 0 and 1 simultaneously. This phenomenon is known as superposition and allows for much more complex calculations to be performed than would otherwise be possible with classical computers. Quantum algorithms are designed to take advantage of this property by encoding data into qubit states so that it can be processed exponentially faster than conventional methods. Additionally, quantum entanglement–the ability for two particles to remain connected even over great distances–allows for large amounts of data to be sent across networks quickly and securely. The potential applications of quantum computing range from cryptography and artificial intelligence to drug discovery and material science. With continued research and development, quantum technology has the potential to revolutionize many industries.  

</div>

# Analysis

We hypothesize that by adding noise to the embeddings at train time, the model overfits less to the specifics of the instruction-tuning dataset, such as formatting details, exact wording, and text length. Instead of collapsing to the exact instruction distribution, the model is more capable of providing answers that incorporate knowledge and behaviors of the pretrained base model.

A very noticeable side-effect of this, that we observe immediately, is that the model is forming more coherent, longer completions. Longer, more verbose, completions are preferred by both human and machine evaluators on most datasets , but we find that the increased verbosity is only the most visible side-effect from the reduced overfitting to the instruction distribution; increased verbosity alone cannot explain the measured gains in performance.

## Overfitting

In this analysis, we focus on LLaMA-2-7B models trained on the Alpaca dataset both with and without `NEFTune`. We examine the training loss of both models on the Alpaca dataset (both are evaluated without noise) and the “testing” loss on the Evol-Instruct dataset. See Figure <a href="#fig:trainingloss_testloss" data-reference-type="ref" data-reference="fig:trainingloss_testloss">5</a>, which shows that the `NEFTune` model has significantly higher training loss but slightly lower testing loss compared to the base model trained without `NEFTune`. This indicated less overfitting and better generalization when `NEFTune` is used.

To test our overfitting hypothesis further, we also generate responses to training prompts with these models using greedy decoding. We compare the generated responses with the ground truth responses provided in the dataset and report the results in Figure <a href="#fig:ROUGE_BLEU" data-reference-type="ref" data-reference="fig:ROUGE_BLEU">6</a>. We use ROUGE-L and BLEU (up to n-gram order 4) to measure the similarity between responses. Figure <a href="#fig:ROUGE_BLEU" data-reference-type="ref" data-reference="fig:ROUGE_BLEU">6</a> shows that responses generated by the model trained with `NEFTune` have significantly lower ROUGE-L and BLEU scores. As ROUGE-L is based on longest common subsequence of words and BLEU is based on common n-grams between responses, higher scores on responses generated by the model trained without `NEFT` indicate that its responses contain a significantly larger portion of the same words in the same order from the ground truth response, as compared to the outputs of the model trained without `NEFTune`.

Taken together, these observations imply that standard finetuning recipes, while tuned for maximal performance, significantly overfit to the instruction dataset, inducing exact reproduction of some responses. In contrast, `NEFTune` models overfit less without reduction in performance on the test set, and do not “lock-in” to the exact wording of the instruction data, as seen in the ROUGE-L metric.

<figure id="fig:trainingloss_testloss">
<p><img src="./figures/alpaca-train_cross_entropy_square.png"" style="width:48.0%" /> <img src="./figures/wizard-train_cross_entropy_square.png"" style="width:48.0%" /></p>
<figcaption><strong>Left:</strong> training loss on the <em>Alpaca dataset</em> for models with and without <code>NEFT</code>, computed with no added noise. Training with <code>NEFT</code> yields a higher training loss. <strong>Right:</strong> loss of the same model, but evaluated on the “test” Evol-Instruct dataset. <code>NEFT</code> yields slightly lower loss.</figcaption>
</figure>

## Length versus Token Diversity

Due to the strong correlation between increased length and performance on the `AlpacaEval` task (in our experiments and for submissions to the public leaderboard), we were curious whether the increase in length observed with `NEFTune` might come at a cost to the diversity of the text. To investigate this, we compute the n-gram repetition rates for LLaMA-2 trained on different finetuning datasets with and without `NEFT`[^2]. N-grams reoccur more frequently in longer passages, and so we must control for passage length. We compute repetition and diversity scores on a fixed-length chunk at the beginning of each sample. The fixed length cuttoffs were $`50`$ for models trained on Alpaca, $`100`$ for Evol-Instruct, $`150`$ for ShareGPT, and $`150`$ for OpenPlatypus. We choose the chunk lengths so that at least half of the generations were longer than the cutoff, and sequences of insufficient length were dropped. The diversity scores we compute are a summary measure of 2-, 3-, and 4-gram repetition rates called *log-diversity*, as described in .

In Table <a href="#tab:length_repdiv_LLaMA2_NEFT" data-reference-type="ref" data-reference="tab:length_repdiv_LLaMA2_NEFT">4</a> and Table <a href="#tab:Noise_Length_Ablation" data-reference-type="ref" data-reference="tab:Noise_Length_Ablation">6</a>, we see that `NEFT` models generate longer outputs than their counterparts. However, we also see that the 2-gram repetition rates as well as overall token log-diversity for models trained with and without `NEFT` are nearly identical, providing evidence that the longer responses do not come at the expense of repetition, and instead provide additional details.

<figure id="fig:ROUGE_BLEU">
<p><img src="./figures/alpaca-train_generation_rougeL_square.png"" style="width:48.0%" /> <img src="./figures/alpaca-train_generation_bleu_square.png"" style="width:48.0%" /></p>
<figcaption><strong>Left</strong> shows the ROUGE-L of training with and without <code>NEFT</code>. <strong>Right</strong> shows BLEU score.</figcaption>
</figure>

## Length is (not) All You Need

To scrutinize the length–leaderboard correlation even further, we tested whether simply promoting a model to generate longer outputs was sufficient to recover the performance gains of models trained with `NEFT`. See Table <a href="#tab:prompt_length_ablation" data-reference-type="ref" data-reference="tab:prompt_length_ablation">5</a>. First, we try explicitly prompting the model to give longer answers. Interestingly, this boosts AlpaceEval scores by 16%. We can also coerce long completions by blocking the \[EOS\] token until we hit 250 tokens in length, thus forcing a standard model to produce answers as long as `NEFT`. This results in marginal improvements over standard finetuning.

Finally, we ablate the use of uniform versus Gaussian noise in the `NEFT` algorithm and find that Gaussian noise induces even longer outputs, but does not come with improved performance. See Table <a href="#tab:Noise_Length_Ablation" data-reference-type="ref" data-reference="tab:Noise_Length_Ablation">6</a>. While longer generations do score better, we see that no generation-time strategy came close to the performance of `NEFTune` models.

<div id="tab:length_repdiv_LLaMA2_NEFT">

<table>
<caption>(<strong>Row 1</strong>) Avg. Character lengths of <code>AlpacaEval</code> responses from LLaMA-2 models finetuned on different datasets. We also report average output length for each dataset (though we trained with max sequence length of 512). <code>NEFT</code> increases average length. (<strong>Row 2</strong>) Whitespace-tokenized lengths of generations. (<strong>Row 3</strong>) 2-Gram repetition rates. (<strong>Row 4</strong>) Log-Diversity measures. </caption>
<thead>
<tr>
<th style="text-align: left;"></th>
<th style="text-align: left;"></th>
<th style="text-align: center;">Alpaca (<span class="math inline"><em>α</em> = 5</span>)</th>
<th style="text-align: center;">Evol-Instruct (<span class="math inline"><em>α</em> = 5</span>)</th>
<th style="text-align: center;">ShareGPT (<span class="math inline"><em>α</em> = 10</span>)</th>
<th style="text-align: center;">OpenPlatypus (<span class="math inline"><em>α</em> = 15</span>)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align: left;"><strong>Character</strong> <strong>Lengths</strong></td>
<td style="text-align: left;">Training data</td>
<td style="text-align: center;">270.31</td>
<td style="text-align: center;">1356.43</td>
<td style="text-align: center;">1276.76</td>
<td style="text-align: center;">649.39</td>
</tr>
<tr>
<td style="text-align: left;">LLaMA-2 7B</td>
<td style="text-align: center;">375.22</td>
<td style="text-align: center;">864.06</td>
<td style="text-align: center;">1011.28</td>
<td style="text-align: center;">1100.98</td>
</tr>
<tr>
<td style="text-align: left;">+<code>NEFT</code></td>
<td style="text-align: center;">1061.89</td>
<td style="text-align: center;">1403.59</td>
<td style="text-align: center;">1496.86</td>
<td style="text-align: center;">1694.26</td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;"><strong>Whitespace</strong> <strong>Lengths</strong></td>
<td style="text-align: left;">LLaMA-2 7B</td>
<td style="text-align: center;">60.5</td>
<td style="text-align: center;">138.99</td>
<td style="text-align: center;">161.04</td>
<td style="text-align: center;">170.41</td>
</tr>
<tr>
<td style="text-align: left;">+<code>NEFT</code></td>
<td style="text-align: center;">169.36</td>
<td style="text-align: center;">225.56</td>
<td style="text-align: center;">234.99</td>
<td style="text-align: center;">264.12</td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;"><strong>2-Gram</strong> <strong>Repetition <span class="math inline">%</span></strong></td>
<td style="text-align: left;">LLaMA-2 7B</td>
<td style="text-align: center;">1.49</td>
<td style="text-align: center;">3.87</td>
<td style="text-align: center;">4.82</td>
<td style="text-align: center;">2.73</td>
</tr>
<tr>
<td style="text-align: left;">+<code>NEFT</code></td>
<td style="text-align: center;">1.72</td>
<td style="text-align: center;">3.79</td>
<td style="text-align: center;">4.58</td>
<td style="text-align: center;">3.21</td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;"><strong>Log-Diversity</strong></td>
<td style="text-align: left;">LLaMA-2 7B</td>
<td style="text-align: center;">15.97</td>
<td style="text-align: center;">10.65</td>
<td style="text-align: center;">8.40</td>
<td style="text-align: center;">9.96</td>
</tr>
<tr>
<td style="text-align: left;">+<code>NEFT</code></td>
<td style="text-align: center;">16.41</td>
<td style="text-align: center;">10.77</td>
<td style="text-align: center;">8.60</td>
<td style="text-align: center;">9.64</td>
</tr>
</tbody>
</table>

</div>

<div id="tab:prompt_length_ablation">

| Setting (LLaMA-1)          | GPT-4 Win Rate | Avg. Character Length |
|:---------------------------|:--------------:|:---------------------:|
| Alpaca-7B-`NEFT`           |     61.99      |        1058.46        |
| Alpaca-7B (Long + Comp)    |     48.01      |        620.74         |
| Alpaca-7B (Long)           |     44.84      |        614.21         |
| Alpaca-7B (Comprehensive)  |     42.14      |        494.85         |
| Alpaca-7B (Min New Tokens) |     38.58      |        1110.97        |
| Alpaca-7B                  |     32.36      |        375.22         |

We use the following meta-prompts to get longer responses: “Generate a long response”, “Generate a comprehensive response”, and “Generate a long and comprehensive response.” Longer responses score better, but do not close the gap with `NEFT`.

</div>

<div id="tab:Noise_Length_Ablation">

<table>
<caption> Win Rate (and Avg. Character Length) on <code>AlpacaEval</code> as evaluated by ChatGPT for different levels and types of training noise. While length does increase with noise, it is not always indicative of <code>AlpacaEval</code> Win Rate.</caption>
<thead>
<tr>
<th style="text-align: left;">Setting</th>
<th colspan="2" style="text-align: center;">Alpaca</th>
<th colspan="2" style="text-align: center;">Evol-Instruct</th>
<th colspan="2" style="text-align: center;">OpenPlatypus</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">LLaMA-2-7b</td>
<td style="text-align: center;">48.26</td>
<td style="text-align: center;">(375.22)</td>
<td style="text-align: center;">62.55</td>
<td style="text-align: center;">(864.06)</td>
<td style="text-align: center;">57.20</td>
<td style="text-align: center;">(1100.98)</td>
</tr>
<tr>
<td style="text-align: left;">+Uniform Noise 5</td>
<td style="text-align: center;">62.55</td>
<td style="text-align: center;">(1061.89)</td>
<td style="text-align: center;">67.58</td>
<td style="text-align: center;">(1403.59)</td>
<td style="text-align: center;">60.99</td>
<td style="text-align: center;">(1428.31)</td>
</tr>
<tr>
<td style="text-align: left;">+Uniform Noise 10</td>
<td style="text-align: center;">61.18</td>
<td style="text-align: center;">(1009.94)</td>
<td style="text-align: center;">65.59</td>
<td style="text-align: center;">(1696.88)</td>
<td style="text-align: center;">60.62</td>
<td style="text-align: center;">(1833.85)</td>
</tr>
<tr>
<td style="text-align: left;">+Uniform Noise 15</td>
<td style="text-align: center;">61.86</td>
<td style="text-align: center;">(819.61)</td>
<td style="text-align: center;">66.58</td>
<td style="text-align: center;">(1650.65)</td>
<td style="text-align: center;">61.74</td>
<td style="text-align: center;">(1694.26)</td>
</tr>
<tr>
<td style="text-align: left;">+Gaussian Noise 5</td>
<td style="text-align: center;">60.93</td>
<td style="text-align: center;">(1371.32)</td>
<td style="text-align: center;">65.09</td>
<td style="text-align: center;">(2065.75)</td>
<td style="text-align: center;">59.13</td>
<td style="text-align: center;">(2060.92)</td>
</tr>
</tbody>
</table>

</div>

## Human Study

Since our primary results are based on the `AlpacaEval` benchmark, which is scored by a large language model, we also run a small scale human study amongst the authors of this work. For a subsample of $`140`$ instructions from `AlpacaEval`, we present annotators with one response generated by a LLaMA-2 model finetuned on Alpaca data with `NEFT` and another response from a model trained without `NEFT`, in random order.

Human annotators preferred `NEFT` in $`88`$ instances, and $`22`$ instances were a draw. This corresponds to a $`74.6\%`$ win score for `NEFT` using the `AlpacaEval` formula ($`88/(140-22)`$). Next, we performed a modified run of `AlpacaEval` where, instead of asking the evaluator (GPT-4) to choose between the outputs of our model or Text-Davinci-003, we present the same pairs of responses from the standard finetuned model and a `NEFT` version of the same model. There, we observe a win score of $`92.80\%`$.

# Conclusions and Limitations

The success of `NEFTune` points to the often ignored importance of algorithms and regularizers for LLM training. Unlike the computer vision community, which has studied regularization and overfitting for years, the LLM community tends to use standardized training loops that are designed for optimizer stability and not generalization. In this environment, LLM researchers have become fixated on datasets and model scaling as the primary path forward. Given the consistent gains of `NEFTune`, and the tendency to overfit on small instruction datasets, it seems that regularization deserves to be revisited in the LLM setting.

Our study has several limitations. We adopt `AlpacaEval` as our central measure of instruction-following ability for LLMs, which is subject to the biases of a single judge (GPT-4). Additionally, due to limited compute resources, we were not able to validate the success of `NEFTune` on larger $`70`$B variants across multiple datasets, and we had to rely on fixed hyper-parameters for most `NEFTune` runs rather than sweeping. Finally, despite our empirical studies, we do not have a conclusive understanding of why `NEFTune` works.

# Ethics Statement

In this work, we proposed an augmentation for instruction finetuning. Although we evaluate these models on standard benchmarks, we do not rigiously evaluate the impact of `NEFTune` on model safety and reliability characteristics like toxicity or refusal to provide potentially harmful responses.

# Reproducibility Statement

We describe the models (in Section <a href="#sec:model_exp_setup" data-reference-type="ref" data-reference="sec:model_exp_setup">3.1</a>) and datasets (in Section <a href="#sec:Datasets" data-reference-type="ref" data-reference="sec:Datasets">3.2</a>) used in our experiments including all hyperparameters (in Section <a href="#sec:hyperparameters" data-reference-type="ref" data-reference="sec:hyperparameters">9.1</a>). The compute infrastructure used was based on commodity-level CPUs and GPUs running open source software (expect $`70`$B parameter finetuning).

### Author Contributions

**Neel Jain\*** – Led the project, contributed to code, ran experiments, developed the method, created majority of plots and written sections.  
**Ping-yeh Chiang\*** – Developed critical parts of the training code, contributed to the methods development, ran QLORA experiments.  
**Yuxin Wen\*** – Ran the bulk of experiments, helped develop the method, contributed to writing.  
**John Kirchenbauer** – Performed evaluation on OpenLLM Leaderboard tasks, diversity analysis, large contributor to the writing.  
**Hong-Min Chu, Gowthami Somepalli** – Ran the experiments for overfitting and embedding analysis and contributed to the writing of these sections.  
**Brian R. Bartoldson, Bhavya Kailkhura** – Ran experiments on large model sizes and developed parallel implementation.  
**Avi Schwarzschild, Aniruddha Saha, Micah Goldblum** – Contributed to writing.  
**Jonas Geiping, Tom Goldstein** – Developed the idea, made large contributions to the writing.

### Acknowledgments

This work was made possible by the ONR MURI program, the Office of Naval Research (N000142112557), and the AFOSR MURI program. Commercial support was provided by Capital One Bank, the Amazon Research Award program, and Open Philanthropy. Further support was provided by the National Science Foundation (IIS-2212182), and by the NSF TRAILS Institute (2229885).

Furthermore, this work was performed under the auspices of the U.S. Department of Energy by the Lawrence Livermore National Laboratory under Contract No. DE-AC52-07NA27344. Brian Bartoldson’s and Bhavya Kailkhura’s efforts were supported by the LLNL-LDRD Program under Project No. 24-ERD-010 (LLNL-CONF-855498). We are also grateful to Amar Saini who provided HPC support.

# References

<div class="thebibliography">

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al Language models are few-shot learners *Advances in neural information processing systems*, 33: 1877–1901, 2020. **Abstract:** Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3’s few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general. (@brown2020language)

Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, et al Alpagasus: Training a better alpaca with fewer data *arXiv preprint arXiv:2307.08701*, 2023. **Abstract:** Large language models (LLMs) strengthen instruction-following capability through instruction-finetuning (IFT) on supervised instruction/response data. However, widely used IFT datasets (e.g., Alpaca’s 52k data) surprisingly contain many low-quality instances with incorrect or irrelevant responses, which are misleading and detrimental to IFT. In this paper, we propose a simple and effective data selection strategy that automatically identifies and filters out low-quality data using a strong LLM (e.g., ChatGPT). To this end, we introduce AlpaGasus, which is finetuned on only 9k high-quality data filtered from the 52k Alpaca data. AlpaGasus significantly outperforms the original Alpaca as evaluated by GPT-4 on multiple test sets and the controlled human evaluation. Its 13B variant matches $\>90\\}%$ performance of its teacher LLM (i.e., Text-Davinci-003 generating the 52k data) on test tasks. It also provides 5.7x faster training, reducing the training time for a 7B variant from 80 minutes (for Alpaca) to 14 minutes. Moreover, the experiments prove the efficacy of our method across diverse datasets, base models, and LLM filters. Overall, AlpaGasus demonstrates a novel data-centric IFT paradigm that can be generally applied to instruction-tuning data, leading to faster training and better instruction-following models. Our project page is available at: https://lichang-chen.github.io/AlpaGasus/ (@chen2023alpagasus)

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality March 2023. URL <https://lmsys.org/blog/2023-03-30-vicuna/>. **Abstract:** \<p\>We introduce Vicuna-13B, an open-source chatbot trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT. Preliminary evaluation ... (@vicuna2023)

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al Scaling instruction-finetuned language models *arXiv preprint arXiv:2210.11416*, 2022. **Abstract:** Finetuning language models on a collection of datasets phrased as instructions has been shown to improve model performance and generalization to unseen tasks. In this paper we explore instruction finetuning with a particular focus on (1) scaling the number of tasks, (2) scaling the model size, and (3) finetuning on chain-of-thought data. We find that instruction finetuning with the above aspects dramatically improves performance on a variety of model classes (PaLM, T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT), and evaluation benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation). For instance, Flan-PaLM 540B instruction-finetuned on 1.8K tasks outperforms PALM 540B by a large margin (+9.4% on average). Flan-PaLM 540B achieves state-of-the-art performance on several benchmarks, such as 75.2% on five-shot MMLU. We also publicly release Flan-T5 checkpoints, which achieve strong few-shot performance even compared to much larger models, such as PaLM 62B. Overall, instruction finetuning is a general method for improving the performance and usability of pretrained language models. (@FLANchung2022scaling)

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord Think you have solved question answering? try arc, the ai2 reasoning challenge *arXiv preprint arXiv:1803.05457*, 2018. **Abstract:** We present a new question set, text corpus, and baselines assembled to encourage AI research in advanced question answering. Together, these constitute the AI2 Reasoning Challenge (ARC), which requires far more powerful knowledge and reasoning than previous challenges such as SQuAD or SNLI. The ARC question set is partitioned into a Challenge Set and an Easy Set, where the Challenge Set contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurence algorithm. The dataset contains only natural, grade-school science questions (authored for human tests), and is the largest public-domain set of this kind (7,787 questions). We test several baselines on the Challenge Set, including leading neural models from the SQuAD and SNLI tasks, and find that none are able to significantly outperform a random baseline, reflecting the difficult nature of this task. We are also releasing the ARC Corpus, a corpus of 14M science sentences relevant to the task, and implementations of the three neural baseline models tested. Can your model perform better? We pose ARC as a challenge to the community. (@clark2018think)

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer Qlora: Efficient finetuning of quantized llms *arXiv preprint arXiv:2305.14314*, 2023. **Abstract:** We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) double quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) paged optimziers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. A lemon-picked analysis demonstrates where Guanaco fails compared to ChatGPT. We release all of our models and code, including CUDA kernels for 4-bit training. (@dettmers2023qlora)

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto Alpacafarm: A simulation framework for methods that learn from human feedback *arXiv preprint arXiv:2305.14387*, 2023. **Abstract:** Large language models (LLMs) such as ChatGPT have seen widespread adoption due to their strong instruction-following abilities. Developing these LLMs involves a complex yet poorly understood workflow requiring training with human feedback. Replicating and understanding this instruction-following requires tackling three major challenges: the high cost of data collection, the lack of trustworthy evaluation, and the absence of reference method implementations. We address these challenges with AlpacaFarm, a simulator that enables research and development for learning from feedback at a low cost. First, we design LLM prompts to simulate human feedback that are 50x cheaper than crowdworkers and display high agreement with humans. Second, we propose an automatic evaluation and validate it against human instructions obtained on real-world interactions. Third, we contribute reference implementations for several methods (PPO, DPO, best-of-n, expert iteration, and more) that learn from pairwise feedback. Finally, as an end-to-end validation of AlpacaFarm, we train and evaluate eleven models on 10k pairs of real human feedback and show that rankings of models trained in AlpacaFarm match rankings of models trained on human data. As a demonstration of the research possible in AlpacaFarm, we find that methods that use a reward model can substantially improve over supervised fine-tuning and that our reference PPO implementation leads to a +10% improvement in win-rate against Davinci003. We release all components of AlpacaFarm at https://github.com/tatsu-lab/alpaca_farm. (@dubois2023alpacafarm)

Cynthia Dwork, Aaron Roth, et al The algorithmic foundations of differential privacy *Foundations and Trends® in Theoretical Computer Science*, 9 (3–4): 211–407, 2014. **Abstract:** The problem of privacy-preserving data analysis has a long history spanning multiple disciplines. As electronic data about individuals becomes increasingly detailed, and as technology enables ever more powerful collection and curation of these data, the need increases for a robust, meaningful, and mathematically rigorous definition of privacy, together with a computationally rich class of algorithms that satisfy this definition. Differential Privacy is such a definition. The Algorithmic Foundations of Differential Privacy starts out by motivating and discussing the meaning of differential privacy, and proceeds to explore the fundamental techniques for achieving differential privacy, and the application of these techniques in creative combinations, using the query-release problem as an ongoing example. A key point is that, by rethinking the computational goal, one can often obtain far better results than would be achieved by methodically replacing each step of a non-private computation with a differentially private implementation. Despite some powerful computational results, there are still fundamental limitations. Virtually all the algorithms discussed herein maintain differential privacy against adversaries of arbitrary computational power — certain algorithms are computationally intensive, others are efficient. Computational complexity for the adversary and the algorithm are both discussed. The monograph then turns from fundamentals to applications other than query-release, discussing differentially private methods for mechanism design and machine learning. The vast majority of the literature on differentially private algorithms considers a single, static, database that is subject to many analyses. Differential privacy in other models, including distributed databases and computations on data streams, is discussed. The Algorithmic Foundations of Differential Privacy is meant as a thorough introduction to the problems and techniques of differential privacy, and is an invaluable reference for anyone with an interest in the topic. (@dwork2014algorithmic)

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou A framework for few-shot language model evaluation September 2021. URL <https://doi.org/10.5281/zenodo.5371628>. **Abstract:** This expository paper introduces a simplified approach to image-based quality inspection in manufacturing using OpenAI’s CLIP (Contrastive Language-Image Pretraining) model adapted for few-shot learning. While CLIP has demonstrated impressive capabilities in general computer vision tasks, its direct application to manufacturing inspection presents challenges due to the domain gap between its training data and industrial applications. We evaluate CLIP’s effectiveness through five case studies: metallic pan surface inspection, 3D printing extrusion profile analysis, stochastic textured surface evaluation, automotive assembly inspection, and microstructure image classification. Our results show that CLIP can achieve high classification accuracy with relatively small learning sets (50-100 examples per class) for single-component and texture-based applications. However, the performance degrades with complex multi-component scenes. We provide a practical implementation framework that enables quality engineers to quickly assess CLIP’s suitability for their specific applications before pursuing more complex solutions. This work establishes CLIP-based few-shot learning as an effective baseline approach that balances implementation simplicity with robust performance, demonstrated in several manufacturing quality control applications. (@eval-harness)

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt Measuring massive multitask language understanding In *International Conference on Learning Representations*, 2020. **Abstract:** We propose a new test to measure a text model’s multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model’s academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings. (@hendrycks2020measuring-MMLU)

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Manli Shu, Khalid Saifullah, Kezhi Kong, Kasun Fernando, Aniruddha Saha, Micah Goldblum, and Tom Goldstein On the reliability of watermarks for large language models *arXiv preprint arXiv:2306.04634*, 2023. **Abstract:** As LLMs become commonplace, machine-generated text has the potential to flood the internet with spam, social media bots, and valueless content. Watermarking is a simple and effective strategy for mitigating such harms by enabling the detection and documentation of LLM-generated text. Yet a crucial question remains: How reliable is watermarking in realistic settings in the wild? There, watermarked text may be modified to suit a user’s needs, or entirely rewritten to avoid detection. We study the robustness of watermarked text after it is re-written by humans, paraphrased by a non-watermarked LLM, or mixed into a longer hand-written document. We find that watermarks remain detectable even after human and machine paraphrasing. While these attacks dilute the strength of the watermark, paraphrases are statistically likely to leak n-grams or even longer fragments of the original text, resulting in high-confidence detections when enough tokens are observed. For example, after strong human paraphrasing the watermark is detectable after observing 800 tokens on average, when setting a 1e-5 false positive rate. We also consider a range of new detection schemes that are sensitive to short spans of watermarked text embedded inside a large document, and we compare the robustness of watermarking to other kinds of detectors. (@kirchenbauer2023reliability)

Kezhi Kong, Guohao Li, Mucong Ding, Zuxuan Wu, Chen Zhu, Bernard Ghanem, Gavin Taylor, and Tom Goldstein Robust optimization as data augmentation for large-scale graphs In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 60–69, 2022. **Abstract:** Data augmentation helps neural networks generalize better by enlarging the training set, but it remains an open question how to effectively augment graph data to enhance the performance of GNNs (Graph Neural Networks). While most existing graph regularizers focus on manipulating graph topological structures by adding/removing edges, we offer a method to augment node features for better performance. We propose FLAG (Free Large-scale Adversarial Augmentation on Graphs), which iteratively augments node features with gradient-based adversarial perturbations during training. By making the model invariant to small fluctuations in input data, our method helps models generalize to out-of-distribution samples and boosts model performance at test time. FLAG is a general-purpose approach for graph data, which universally works in node classification, link prediction, and graph classification tasks. FLAG is also highly flexible and scalable, and is deployable with arbitrary GNN backbones and large-scale datasets. We demon-strate the efficacy and stability of our method through ex-tensive experiments and ablation studies. We also provide intuitive observations for a deeper understanding of our method. We open source our implementation at https://github.com/devnkong/FLAG. (@kong2022robust)

Ariel N. Lee, Cole J. Hunter, and Nataniel Ruiz Platypus: Quick, cheap, and powerful refinement of llms *arXiv preprint arxiv:2308.07317*, 2023. **Abstract:** We present $\\}textbf{Platypus}$, a family of fine-tuned and merged Large Language Models (LLMs) that achieves the strongest performance and currently stands at first place in HuggingFace’s Open LLM Leaderboard as of the release date of this work. In this work we describe (1) our curated dataset $\\}textbf{Open-Platypus}$, that is a subset of other open datasets and which $\\}textit{we release to the public}$ (2) our process of fine-tuning and merging LoRA modules in order to conserve the strong prior of pretrained LLMs, while bringing specific domain knowledge to the surface (3) our efforts in checking for test data leaks and contamination in the training data, which can inform future research. Specifically, the Platypus family achieves strong performance in quantitative LLM metrics across model sizes, topping the global Open LLM leaderboard while using just a fraction of the fine-tuning data and overall compute that are required for other state-of-the-art fine-tuned LLMs. In particular, a 13B Platypus model can be trained on $\\}textit{a single}$ A100 GPU using 25k questions in 5 hours. This is a testament of the quality of our Open-Platypus dataset, and opens opportunities for more improvements in the field. Project page: https://platypus-llm.github.io (@platypus2023)

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis Contrastive decoding: Open-ended text generation as optimization *arXiv preprint arXiv:2210.15097*, 2022. **Abstract:** Given a language model (LM), maximum probability is a poor decoding objective for open-ended generation, because it produces short and repetitive text. On the other hand, sampling can often produce incoherent text that drifts from the original topics. We propose contrastive decoding (CD), a reliable decoding approach that optimizes a contrastive objective subject to a plausibility constraint. The contrastive objective returns the difference between the likelihood under a large LM (called the expert, e.g. OPT-13B) and a small LM (called the amateur, e.g. OPT-125M), and the constraint ensures that the outputs are plausible. CD is inspired by the fact that the failures of larger LMs (e.g., repetition, incoherence) are even more prevalent in smaller LMs, and that this difference signals which texts should be preferred. CD requires zero additional training, and produces higher quality text than decoding from the larger LM alone. It also works across model scales (OPT-13B and GPT2-1.5B) and significantly outperforms four strong decoding algorithms (e.g., nucleus, top-k) in automatic and human evaluations across wikipedia, news and story domains. (@li2022contrastive)

Chin-Yew Lin : A package for automatic evaluation of summaries In *Text Summarization Branches Out*, pp. 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL <https://www.aclweb.org/anthology/W04-1013>. **Abstract:** ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. It includes measures to automatically determine the quality of a summary by comparing it to other (ideal) summaries created by humans. The measures count the number of overlapping units such as n-gram, word sequences, and word pairs between the computer-generated summary to be evaluated and the ideal summaries created by humans. This paper introduces four different ROUGE measures: ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S included in the ROUGE summarization evaluation package and their evaluations. Three of them have been used in the Document Understanding Conference (DUC) 2004, a large-scale summarization evaluation sponsored by NIST. (@lin2004rouge)

Stephanie Lin, Jacob Hilton, and Owain Evans ruthfulQA: Measuring how models mimic human falsehoods In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 3214–3252, Dublin, Ireland, May 2022. Association for Computational Linguistics. . URL <https://aclanthology.org/2022.acl-long.229>. **Abstract:** We propose a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. We crafted questions that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts. We tested GPT-3, GPT-Neo/J, GPT-2 and a T5-based model. The best model was truthful on 58% of questions, while human performance was 94%. Models generated many false answers that mimic popular misconceptions and have the potential to deceive humans. The largest models were generally the least truthful. This contrasts with other NLP tasks, where performance improves with model size. However, this result is expected if false answers are learned from the training distribution. We suggest that scaling up models alone is less promising for improving truthfulness than fine-tuning using training objectives other than imitation of text from the web. (@lin-etal-2022-truthfulqa)

David Nukrai, Ron Mokady, and Amir Globerson Text-only training for image captioning using noise-injected clip *ArXiv*, abs/2211.00575, 2022. URL <https://api.semanticscholar.org/CorpusID:253244258>. **Abstract:** We consider the task of image-captioning using only the CLIP model and additional text data at training time, and no additional captioned images. Our approach relies on the fact that CLIP is trained to make visual and textual embeddings similar. Therefore, we only need to learn how to translate CLIP textual embeddings back into text, and we can learn how to do this by learning a decoder for the frozen CLIP text encoder using only text. We argue that this intuition is "almost correct" because of a gap between the embedding spaces, and propose to rectify this via noise injection during training. We demonstrate the effectiveness of our approach by showing SOTA zero-shot image captioning across four benchmarks, including style transfer. Code, data, and models are available on GitHub. (@Nukrai2022TextOnlyTF)

OpenAI Introducing chatgpt 2022. URL <https://openai.com/blog/chatgpt>. **Abstract:** ChatGPT has recently emerged to aid in computer programming education due to its cutting-edge functionality of generating program code, debugging, etc. This research firstly focused on what the ethical considerations and solutions are for the first-year IT students who use ChatGPT to write computer programs in an integrated assignment. And then it turned to investigate what impact ChatGPT has on the programming competencies and learning outcomes of students compared to those who do not use ChatGPT. To ensure students use ChatGPT ethically, guidance was provided together with a declaration form of ethically using ChatGPT in each phase of the assignment. Next, we collected and analyzed a survey and their declaration from students and compared student effort, time spent, and performance outcomes from those who were using and without using ChatGPT. Based on the findings, we concluded that although ChatGPT provides an opportunity to the first-year students to learn programming in the way of analysis, synthesis, and evaluation, many students still prefer the conventional way of learning programming in terms of comprehension and application. We argued that since our students in the programming course are always from different academic background levels, we would continue to use both ChatGPT and conventional eLearning resources to meet different learning requirements. (@OpenAIchatGPT)

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al Training language models to follow instructions with human feedback *Advances in Neural Information Processing Systems*, 35: 27730–27744, 2022. **Abstract:** Making language models bigger does not inherently make them better at following a user’s intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent. (@InstructGPT)

Kishore Papineni, Salim Roukos, Todd Ward, and Wei jing Zhu Bleu: a method for automatic evaluation of machine translation In *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*, pp. 311–318, 2002. **Abstract:** Human evaluations of machine translation are extensive but expensive. Human evaluations can take months to finish and involve human labor that can not be reused. We propose a method of automatic machine translation evaluation that is quick, inexpensive, and language-independent, that correlates highly with human evaluation, and that has little marginal cost per run. We present this method as an automated understudy to skilled human judges which substitutes for them when there is need for quick or frequent evaluations. (@papineni02bleu)

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, et al Multitask prompted training enables zero-shot task generalization In *International Conference on Learning Representations*, 2021. **Abstract:** Large language models have recently been shown to attain reasonable zero-shot generalization on a diverse set of tasks (Brown et al., 2020). It has been hypothesized that this is a consequence of implicit multitask learning in language models’ pretraining (Radford et al., 2019). Can zero-shot generalization instead be directly induced by explicit multitask learning? To test this question at scale, we develop a system for easily mapping any natural language tasks into a human-readable prompted form. We convert a large set of supervised datasets, each with multiple prompts with diverse wording. These prompted datasets allow for benchmarking the ability of a model to perform completely held-out tasks. We fine-tune a pretrained encoder-decoder model (Raffel et al., 2020; Lester et al., 2021) on this multitask mixture covering a wide variety of tasks. The model attains strong zero-shot performance on several standard datasets, often outperforming models up to 16x its size. Further, our approach attains strong performance on a subset of tasks from the BIG-bench benchmark, outperforming models up to 6x its size. All trained models are available at https://github.com/bigscience-workshop/t-zero and all prompts are available at https://github.com/bigscience-workshop/promptsource. (@sanh2021multitask-T0)

ShareGPT, 2023 URL <https://sharegpt.com/> **Abstract:** In this paper we study in what order a crawler should visit the URLs it has seen, in order to obtain more "important" pages first. Obtaining important pages rapidly can be very useful when a crawler cannot visit the entire Web in a reasonable amount of time. We define several importance metrics, ordering schemes, and performance evaluation measures for this problem. We also experimentally evaluate the ordering schemes on the Stanford University Web. Our results show that a crawler with a good ordering scheme can obtain important pages significantly faster than one without. (@sharegpt)

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto Stanford alpaca: An instruction-following llama model <https://github.com/tatsu-lab/stanford_alpaca>, 2023. **Abstract:** Automatic fact-checking plays a crucial role in combating the spread of misinformation. Large Language Models (LLMs) and Instruction-Following variants, such as InstructGPT and Alpaca, have shown remarkable performance in various natural language processing tasks. However, their knowledge may not always be up-to-date or sufficient, potentially leading to inaccuracies in fact-checking. To address this limitation, we propose combining the power of instruction-following language models with external evidence retrieval to enhance fact-checking performance. Our approach involves leveraging search engines to retrieve relevant evidence for a given input claim. This external evidence serves as valuable supplementary information to augment the knowledge of the pretrained language model. Then, we instruct-tune an open-sourced language model, called LLaMA, using this evidence, enabling it to predict the veracity of the input claim more accurately. To evaluate our method, we conducted experiments on two widely used fact-checking datasets: RAWFC and LIAR. The results demonstrate that our approach achieves state-of-the-art performance in fact-checking tasks. By integrating external evidence, we bridge the gap between the model’s knowledge and the most up-to-date and sufficient context available, leading to improved fact-checking outcomes. Our findings have implications for combating misinformation and promoting the dissemination of accurate information on online platforms. Our released materials are accessible at: https://thcheung.github.io/factllama. (@alpaca)

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al Llama: Open and efficient foundation language models *arXiv preprint arXiv:2302.13971*, 2023. **Abstract:** We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community. (@touvron2023llama)

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al Llama 2: Open foundation and fine-tuned chat models *arXiv preprint arXiv:2307.09288*, 2023. **Abstract:** In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs. (@touvron2023llama2)

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi Self-instruct: Aligning language model with self generated instructions *arXiv preprint arXiv:2212.10560*, 2022. **Abstract:** Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend heavily on human-written instruction data that is often limited in quantity, diversity, and creativity, therefore hindering the generality of the tuned model. We introduce Self-Instruct, a framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations. Our pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model. Applying our method to the vanilla GPT3, we demonstrate a 33% absolute improvement over the original model on Super-NaturalInstructions, on par with the performance of InstructGPT-001, which was trained with private user data and human annotations. For further evaluation, we curate a set of expert-written instructions for novel tasks, and show through human evaluation that tuning GPT3 with Self-Instruct outperforms using existing public instruction datasets by a large margin, leaving only a 5% absolute gap behind InstructGPT-001. Self-Instruct provides an almost annotation-free method for aligning pre-trained language models with instructions, and we release our large synthetic dataset to facilitate future studies on instruction tuning. Our code and data are available at https://github.com/yizhongw/self-instruct. (@wang2022selfinstruct)

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le Finetuned language models are zero-shot learners In *International Conference on Learning Representations*, 2021. **Abstract:** This paper explores a simple method for improving the zero-shot learning abilities of language models. We show that instruction tuning – finetuning language models on a collection of tasks described via instructions – substantially improves zero-shot performance on unseen tasks. We take a 137B parameter pretrained language model and instruction-tune it on over 60 NLP tasks verbalized via natural language instruction templates. We evaluate this instruction-tuned model, which we call FLAN, on unseen task types. FLAN substantially improves the performance of its unmodified counterpart and surpasses zero-shot 175B GPT-3 on 20 of 25 tasks that we evaluate. FLAN even outperforms few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze. Ablation studies reveal that number of finetuning datasets, model scale, and natural language instructions are key to the success of instruction tuning. (@wei2021finetuned-FLAN)

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang Wizardlm: Empowering large language models to follow complex instructions *arXiv preprint arXiv:2304.12244*, 2023. **Abstract:** Training large language models (LLMs) with open-domain instruction following data brings colossal success. However, manually creating such instruction data is very time-consuming and labor-intensive. Moreover, humans may struggle to produce high-complexity instructions. In this paper, we show an avenue for creating large amounts of instruction data with varying levels of complexity using LLM instead of humans. Starting with an initial set of instructions, we use our proposed Evol-Instruct to rewrite them step by step into more complex instructions. Then, we mix all generated instruction data to fine-tune LLaMA. We call the resulting model WizardLM. Human evaluations on a complexity-balanced test bed and Vicuna’s testset show that instructions from Evol-Instruct are superior to human-created ones. By analyzing the human evaluation results of the high complexity part, we demonstrate that outputs from our WizardLM are preferred to outputs from OpenAI ChatGPT. In GPT-4 automatic evaluation, WizardLM achieves more than 90\\}% capacity of ChatGPT on 17 out of 29 skills. Even though WizardLM still lags behind ChatGPT in some aspects, our findings suggest that fine-tuning with AI-evolved instructions is a promising direction for enhancing LLMs. Our code and data are public at https://github.com/nlpxucan/WizardLM (@xu2023wizardlm)

Hanwei Xu, Yujun Chen, Yulun Du, Nan Shao, Wang Yanggang, Haiyu Li, and Zhilin Yang Zeroprompt: Scaling prompt-based pretraining to 1,000 tasks improves zero-shot generalization In *Findings of the Association for Computational Linguistics: EMNLP 2022*, pp. 4235–4252, 2022. (@xu2022zeroprompt)

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi ellaSwag: Can a machine really finish your sentence? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. . URL <https://aclanthology.org/P19-1472>. **Abstract:** Recent work by Zellers et al. (2018) introduced a new task of commonsense natural language inference: given an event description such as “A woman sits at a piano,” a machine must select the most likely followup: “She sets her fingers on the keys.” With the introduction of BERT, near human-level performance was reached. Does this mean that machines can perform human level commonsense inference? In this paper, we show that commonsense inference still proves difficult for even state-of-the-art models, by presenting HellaSwag, a new challenge dataset. Though its questions are trivial for humans (\>95% accuracy), state-of-the-art models struggle (\<48%). We achieve this via Adversarial Filtering (AF), a data collection paradigm wherein a series of discriminators iteratively select an adversarial set of machine-generated wrong answers. AF proves to be surprisingly robust. The key insight is to scale up the length and complexity of the dataset examples towards a critical ‘Goldilocks’ zone wherein generated text is ridiculous to humans, yet often misclassified by state-of-the-art models. Our construction of HellaSwag, and its resulting difficulty, sheds light on the inner workings of deep pretrained models. More broadly, it suggests a new path forward for NLP research, in which benchmarks co-evolve with the evolving state-of-the-art in an adversarial way, so as to present ever-harder challenges. (@zellers-etal-2019-hellaswag)

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al Opt: Open pre-trained transformer language models *arXiv preprint arXiv:2205.01068*, 2022. **Abstract:** Large language models, which are often trained for hundreds of thousands of compute days, have shown remarkable capabilities for zero- and few-shot learning. Given their computational cost, these models are difficult to replicate without significant capital. For the few that are available through APIs, no access is granted to the full model weights, making them difficult to study. We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We show that OPT-175B is comparable to GPT-3, while requiring only 1/7th the carbon footprint to develop. We are also releasing our logbook detailing the infrastructure challenges we faced, along with code for experimenting with all of the released models. (@zhang2022opt)

Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Tom Goldstein, and Jingjing Liu Freelb: Enhanced adversarial training for natural language understanding In *International Conference on Learning Representations*, 2019. **Abstract:** Adversarial training, which minimizes the maximal risk for label-preserving input perturbations, has proved to be effective for improving the generalization of language models. In this work, we propose a novel adversarial training algorithm, FreeLB, that promotes higher invariance in the embedding space, by adding adversarial perturbations to word embeddings and minimizing the resultant adversarial risk inside different regions around input samples. To validate the effectiveness of the proposed approach, we apply it to Transformer-based models for natural language understanding and commonsense reasoning tasks. Experiments on the GLUE benchmark show that when applied only to the finetuning stage, it is able to improve the overall test scores of BERT-base model from 78.3 to 79.4, and RoBERTa-large model from 88.5 to 88.8. In addition, the proposed approach achieves state-of-the-art single-model test accuracies of 85.44\\}% and 67.75\\}% on ARC-Easy and ARC-Challenge. Experiments on CommonsenseQA benchmark further demonstrate that FreeLB can be generalized and boost the performance of RoBERTa-large model on other tasks as well. Code is available at \\}url{https://github.com/zhuchen03/FreeLB . (@zhu2019freelb)

</div>

# Appendix

## Hyperparameters

We finetune the $`7`$B parameter models on four A5000s and $`13`$B parameters on eight A5000s using bfloat16 precision. After doing an initial learning rate sweep on LLaMA-1 and Alpaca, we use learning rate of $`5e`$-$`5`$ and the Adam optimizer for all $`7`$B models after seeing $`4\%`$ improvement over baseline numbers. We train all models for $`3`$ epochs on all datasets setting the same seed for each run with an effective batch size of $`128`$ ($`4`$ cards, batch size $`4`$, $`8`$ gradient accumulation steps). When finetuning with noise we train on three levels of noise, an L$`2`$ ball of $`5`$, $`10`$, and $`15`$ over the sequence lengths and report the best one on `AlpacaEval` using ChatGPT as the evaluator. We train with sequence lengths of $`512`$ tokens (mainly for memory and speed) like the original Alpaca setting, finding that this does not effect the outputted response length or quality as corroborated by the Alpaca Leaderboard numbers. In Table <a href="#tab:ShareGPT_longseq" data-reference-type="ref" data-reference="tab:ShareGPT_longseq">13</a> we see that training with longer sequences does not change performance significantly. For ShareGPT, we split the multi-turn conversations into sequences of $`512`$ tokens using the process described by . When training $`70`$ billion parameter models, we use the finetuning hyperparameters found in except we use a sequence length of $`2048`$; i.e., we use weight decay of $`0.1`$, a batch size of $`64`$, and a learning rate of $`2\mathrm{e}{-5}`$. We finetune for a total of three epochs on Evol-Instruct 70k . When using `NEFTune` on the $`70`$B parameter model, we use $`\alpha=15`$ and did not explore other (potentially better) settings due to computational constraints. Additinonally, we saw an increase in average output character length from $`852`$ to $`1241`$ (+$`389`$).

<div id="tab:LLaMA-2_GPT-4_with_alpha">

|  | Alpaca | Evol-Instruct | ShareGPT | OpenPlatypus | Average |
|:---|:--:|:--:|:--:|:--:|:--:|
| LLaMA-2 7B | 29.79 | 70.34 | 68.74 | 62.00 | 57.71 |
| \+`NEFT` | 64.69 | 79.60 | 76.28 | 70.61 | 72.80 |
|  | ($`\alpha=5`$) | ($`\alpha=5`$) | ($`\alpha=10`$) | ($`\alpha=15`$) |  |

`AlpacaEval` Win Rate versus Text-Davinci-003 for LLaMA-2 trained on different datasets, using GPT-4 as the evaluator, showing an average improvement of $`15`$% across all datasets.

</div>

<div id="tab:OpenLLM_leaderboard_performance_alpha">

|            | Alpaca | Evol-Instruct | OpenPlatypus | ShareGPT |
|:-----------|:------:|:-------------:|:------------:|:--------:|
| OPT 6.7B   |   15   |      15       |      5       |    15    |
| LLaMA-1 7B |   10   |      10       |      15      |    10    |
| LLaMA-2 7B |   5    |       5       |      15      |    15    |

$`\alpha`$ used for Fig <a href="#fig:OpenLLM_leaderboard_performance" data-reference-type="ref" data-reference="fig:OpenLLM_leaderboard_performance">4</a>.

</div>

<div id="tab:llama-2-chat_SFT_with_alpha">

|  | LLaMA-2 (7B) | LLaMA-2-Chat (7B) | LLaMA-2 (13B) | LLaMA-2 (70B) |
|:---|:--:|:--:|:--:|:--:|
| Base | \- |  | \- |  |
| Evol-Instruct | 70.34 | 74.44 | 72.61 | 75.03 |
| +NEFT | 79.60 | 81.74 | 82.04 | 88.81 |
|  | ($`\alpha=5`$) | ($`\alpha=5`$) | ($`\alpha=5`$) | ($`\alpha=15`$) |

LLaMA-2-Chat (7B) and LLaMA-2 (13B) can be finetuned further to improve performance.

</div>

## Additional Ablation Studies

We ablated uniform and gaussian noise, finding that uniform noise performs slightly better. We also ablate the decoding hyperparameters in Figure <a href="#fig:decoding_stragies" data-reference-type="ref" data-reference="fig:decoding_stragies">10</a> finding minimal changes in performance. Thus, we use the simplest sampling strategy, greedy decoding, with a repetition penalty of $`1.2`$. We also check to see if `NEFT` continues to yield improvements as you increase the number of training epochs. From Table <a href="#tab:Alpaca_Epochs" data-reference-type="ref" data-reference="tab:Alpaca_Epochs">14</a>, we see that there is a plateau in performance that is reached at higher epoch counts. In Table <a href="#tab:training_parameters" data-reference-type="ref" data-reference="tab:training_parameters">11</a>, we freeze different parts of the model to understand if certain parts of the model are critical for `NEFT`.

<div id="fig:decoding_stragies">

<table>
<caption><code>AlpacaEval</code> Win Rate with ChatGPT (GPT-4 in parentheses) evaluator under different decoding strategies from this we can see that there seems to be little variation in performance. The WizardLM and LLaMA-Chat hyperparameters were obtained from generation config files from Hugging Face. All sampling techniques had a repetition penalty of <span class="math inline">1.2</span>.</caption>
<thead>
<tr>
<th style="text-align: left;">Hyper. Source</th>
<th style="text-align: center;">top_p</th>
<th style="text-align: center;">temp.</th>
<th style="text-align: center;">LLaMA2-7B (Evolve)</th>
<th style="text-align: center;">LLaMA2-7B-<code>NEFT</code> (Evolve)</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Base</td>
<td colspan="2" style="text-align: center;">greedy</td>
<td style="text-align: center;">62.55 (70.34)</td>
<td style="text-align: center;">67.58 (79.60)</td>
</tr>
<tr>
<td style="text-align: left;">HP 0</td>
<td style="text-align: center;">0.1</td>
<td style="text-align: center;">0.1</td>
<td style="text-align: center;">63.11</td>
<td style="text-align: center;">66.83</td>
</tr>
<tr>
<td style="text-align: left;">HP 1</td>
<td style="text-align: center;">0.35</td>
<td style="text-align: center;">0.5</td>
<td style="text-align: center;">62.48</td>
<td style="text-align: center;">66.71</td>
</tr>
<tr>
<td style="text-align: left;">WizardLM</td>
<td style="text-align: center;">0.6</td>
<td style="text-align: center;">0.9</td>
<td style="text-align: center;">62.05</td>
<td style="text-align: center;">66.96</td>
</tr>
<tr>
<td style="text-align: left;">LLaMA-2</td>
<td style="text-align: center;">0.9</td>
<td style="text-align: center;">0.6</td>
<td style="text-align: center;">63.73 (70.49)</td>
<td style="text-align: center;">65.47</td>
</tr>
</tbody>
</table>

</div>

<div id="tab:training_parameters">

| Setting                        | `AlpacaEval` (ChatGPT Eval) |
|:-------------------------------|:---------------------------:|
| standard finetuning            |            48.26            |
| `NEFT`                         |            62.55            |
| `NEFT`+Embed frozen            |            61.06            |
| `NEFT`+LM-head frozen          |            61.12            |
| `NEFT`+Attention blocks frozen |            22.17            |
| LLaMA-2 (no finetuning)        |            22.17            |

`AlpacaEval` Win Rate according to ChatGPT while varying the set of trainable parameters when finetuning LLaMA-2-7B on the Alpaca dataset. The top two rows have all parameters set as trainable.

</div>

<div id="tab:FreeLB_ablation">

|                               | ChatGPT Win Rate |
|:------------------------------|:----------------:|
| LLaMA-1-7B (Evolve)           |      62.30       |
| \+`NEFT`                      |      67.45       |
| +FreeLB (after hparam tuning) |      63.48       |

`NEFT` performs better than FreeLB.

</div>

<div id="tab:ShareGPT_longseq">

| LLaMA-2 (7B) | Split 512 | Split 1024 |
|:-------------|:---------:|:----------:|
| ShareGPT     |   63.48   |   61.68    |
| \+`NEFT`     |   64.22   |   64.35    |

Using ChatGPT as the evaluator, we observe a slight performance increase when training with longer sequences on the ShareGPT data compared to standard finetuning at the same sequence length.

</div>

<div id="tab:Alpaca_Epochs">

| Epochs | LLaMA-2 (7B) Alpaca | \+`NEFT` |
|:-------|:--------------------|:---------|
| 1      | 40.50               | 55.09    |
| 3      | 48.26               | 62.55    |
| 5      | 48.94               | 62.24    |
| 7      | 48.63               | 60.50    |
| 9      | 47.45               | 58.14    |

`AlpacaEval` ChatGPT Win Rate as a function of the number of finetuning epochs.

</div>

## Additional Analysis

#### How does the noise impact the tokens?

Since our modeling involves adding random noise to embeddings during the training stage, we examined whether the added noise changes the semantics of the token sequences in the training data. For this analysis, we sample a random $`5200`$ samples from the Alpaca dataset, embed each training point using the embedding layer of different models, and then add different levels of noise by varying the scaling factor $`\alpha`$. We then project the noised embeddings back to their closest neighbor in the embedding matrix. We compute the $`\%`$ of token flips in each sentence and average the flip rate across all the samples. We present the flip scores for $`7`$ models in Fig. <a href="#appendix_fig:noise_impact" data-reference-type="ref" data-reference="appendix_fig:noise_impact">7</a>. While none of the sentences had any flips up to $`\alpha=15`$, we see some flips when $`\alpha \geq 25`$. Note that all the results presented in the paper use $`\alpha \leq 15`$. Interestingly, a LLaMA-1 model finetuned on Alpaca does not show any flips even at higher levels of $`\alpha`$.

<figure id="appendix_fig:noise_impact">
<img src="./figures/flip_percent_alpacadata.png"" style="width:55.0%" />
<figcaption>Percentage of tokens per sentence flipped at different levels of noise added to embeddings.</figcaption>
</figure>

<figure id="fig:singular_values">
<img src="./figures/singularvals.png"" />
<figcaption>Top-100 eigenvalues of embedding similarity matrices across models.</figcaption>
</figure>

<figure id="appendix_fig:embedding_noise_flips">
<p><img src="./figures/embed_flips_LLaMA_2.png"" style="width:45.0%" /> <img src="./figures/embed_flips_LLaMA_2_Alpaca.png"" style="width:45.0%" /> <img src="./figures/embed_flips_LLaMA_2_Evol_Instruct.png"" style="width:45.0%" /></p>
<figcaption> Number of tokens flipped at different levels of noise and sequence lengths (out of <span class="math inline">32000</span>). Model names corresponding to each plot are in the title. </figcaption>
</figure>

#### Impact of noise on embedding similarity.

We also analyzed how the similarity of embeddings changes when we perform `NEFT` training. We looked at the top 100 singular values of the embedding similarity matrix for all the models. We present the results in Fig.<a href="#appendix_fig:noise_impact" data-reference-type="ref" data-reference="appendix_fig:noise_impact">7</a> (Right). For a given base model (LLaMA-1 or LLaMA-2), the singular value distribution did not change across the variants, with or without `NEFTune`. This shows that the addition of noise during training does not impact the embedding similarity distribution very much.

#### How does noise impact embeddings?

In the previous analysis, we evaluated whether any tokens flipped when noising sequence of real training data. We also examine the embeddings of 3 models, LLaMA-2, LLaMA-2 Alpaca, and LLaMA-2 Evol-Instruct in isolation. In this experiment, we sweep over all of the tokens in the vocabulary adding noise to each one, and count the number of noised embeddings whose nearest neighbor is different from their un-noised starting point. We present the results in Fig. <a href="#appendix_fig:embedding_noise_flips" data-reference-type="ref" data-reference="appendix_fig:embedding_noise_flips">9</a>. We vary 2 factors: base noise scale $`\alpha`$, and the sequence length scaling factor $`L`$ which is used in calculating the final noise coefficient. Even at high levels of noise, only a very small number of tokens actually flip ($`\leq 0.4 \%`$). This shows that `NEFT` training does not change the semantic relationships between the tokens.

Interestingly, this experiment suggests that, if considering `NEFTune` as a type of data augmentation applied to the embeddings of training tokens, the actual effects of `NEFT` are mostly a result of perturbations in intra-token space since the noise scales used rarely amount to displacements that transform any individual token into a meaningfully different one. Nevertheless, in Figure <a href="#appendix_fig:flipped_tokens_alpaca2" data-reference-type="ref" data-reference="appendix_fig:flipped_tokens_alpaca2">10</a> we show the tokens corresponding to embeddings of LLaMA-2 Alpaca model that do actually flip even with the lowest amount of noise utilized in our study.

<figure id="appendix_fig:flipped_tokens_alpaca2">
<img src="./figures/alpaca_llama2_flipped_tokens.png"" style="width:100.0%" />
<figcaption>Tokens flipped at <span class="math inline"><em>α</em> = 1</span> and sequence length = 512 in LLaMA-2 </figcaption>
</figure>

## Qualitative Examples

For these qualitative examples for LLaMA-2 finetuned models, we chose one instruction that requires knowledge explanation and one that involves “creativity.” We pull these examples from `AlpacaEval`, particularly the Vicuna Eval subset. Note we did not actively cherry-pick these examples.

<div class="tcolorbox">

**`AlpacaEval` (Vicuna Subset) Instruction**: Can you explain the basics of quantum computing?  
**LLaMA-2-Alpaca-7B Response**:  
Quantum computing is based on the principles of quantum mechanics, which allow for phenomena such as superposition and entanglement between particles. This means that information can be processed in parallel instead of sequentially like traditional computers. Quantum bits (qubits) are used to store data instead of classical bits, allowing more complex computations to take place with fewer resources than would otherwise be necessary.  
**LLaMA-2-Alpaca-7B-`NEFT` Response**:  
Quantum computing is based on the principles of quantum mechanics, which describe how particles such as electrons and photons behave at the subatomic level. Unlike traditional computers, which process information in binary digits (bits) that can be either 0 or 1, quantum bits (qubits) can exist in multiple states between 0 and 1 simultaneously. This phenomenon is known as superposition and allows for much more complex calculations to be performed than would otherwise be possible with classical computers. Quantum algorithms are designed to take advantage of this property by encoding data into qubit states so that it can be processed exponentially faster than conventional methods. Additionally, quantum entanglement–the ability for two particles to remain connected even over great distances–allows for large amounts of data to be sent across networks quickly and securely. The potential applications of quantum computing range from cryptography and artificial intelligence to drug discovery and material science. With continued research and development, quantum technology has the potential to revolutionize many industries. **LLaMA-2-ShareGPT-7B Response**:  
Quantum computers are a type of computer that use quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. They have the potential to solve certain problems much faster than classical computers, which are limited by the laws of classical physics.  
  
One way to think about how a quantum computer works is in terms of its basic building blocks or “qubits.” Qubits can exist in multiple states at once (a property known as superposition), and they can also be linked together so that their state depends on the state of other qubits (a phenomenon called entanglement). These properties allow quantum computers to perform many calculations simultaneously, taking advantage of the principles of quantum mechanics to achieve exponential speedup for certain types of computations.  
**LLaMA-2-ShareGPT-7B-`NEFT` Response**:  
Quantum computers are a type of computer that use quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. They have the potential to solve certain problems much faster than classical computers, which are limited by the laws of classical physics.  
  
Here is a brief overview of some key concepts in quantum computing:  
  
1. Superposition: In classical computing, bits can only be either 0 or 1 at any given time. However, in quantum computing, qubits (quantum bits) can exist in multiple states simultaneously, known as superpositions. This allows for many calculations to be performed simultaneously, increasing the speed of computation.  
2. Entanglement: When two particles become “entangled,” their properties become correlated even when they are separated by large distances. For example, if two photons are created with opposite polarizations, measuring the state of one photon will instantly determine the state of the other, regardless of how far apart they are. This property can be used to create a network of entangled qubits, allowing them to work together to solve complex problems.  
3. Interference: Just like waves can interfere with each other to produce new patterns, qubits can also interfere with each other through a process called quantum interference. By manipulating these interferences, quantum algorithms can be designed to take advantage of the unique properties of quantum systems.  
4. Measurement: One of the biggest challenges in working with quantum systems is that measurements collapse the wave function, effectively freezing the system into its final state. This means that once a measurement is taken, the information stored in the quantum system cannot be changed anymore. As a result, quantum computations must be designed carefully to ensure that the desired outcome is achieved before the measurement is taken.  
5. Error correction: Because of the inherent instability of quantum systems, errors can occur during computation. To overcome this challenge, error correction codes must be implemented to detect and correct mistakes made by the quantum system. These error correction techniques are critical to making practical quantum computers a reality.  
6. Topological Quantum Computing: Another approach to building a quantum computer uses topological phases of matter, where the order parameter takes values in a non-commutative space. It has been shown that certain classes of topologically ordered states can host fault-tolerant quantum computation.  
7. Degrees of freedom: In classical computing, there is usually just one degree of freedom per bit - whether it is set to 0 or 1. But in quantum computing, there can be multiple degrees of freedom per qubit, leading to more possibilities for encoding information and performing calculations.  
8. Classical vs Quantum gates: Gates are basic building blocks of quantum circuits, similar to logic gates in classical computing. While both classical and quantum gates perform operations on data, the rules governing them differ significantly. For instance, while classical gates operate on binary digits (bits), quantum gates operate on qubits, and the principles underlying their behavior are fundamentally different.  
**LLaMA-2-Evolve70k**:  
Sure, I’d be happy to help!  
Quantum computing is a type of computation that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. Unlike classical computers, which use bits (which can only have one of two values, either 0 or 1) to represent information, quantum computers use qubits (which can exist in multiple states simultaneously). This allows for much faster processing times when it comes to certain types of calculations.  
One way that quantum computers are different from classical computers is through their ability to perform parallel computations. In a classical computer, each bit must be processed sequentially, but in a quantum computer, many qubits can be manipulated at once using something called a quantum gate. These gates allow for complex algorithms to be executed quickly by taking advantage of the properties of quantum mechanics.  
Another key aspect of quantum computing is entanglement, which occurs when two or more particles become connected in such a way that the state of one particle affects the state of another, regardless of how far apart they may be. Entangled qubits can be used to perform certain types of calculations much faster than non-entangled ones because the results of measurements made on them are predetermined based on their initial state.  
Overall, while there is still much research being done into the potential applications of quantum computing, this new field has great promise for solving some of society’s most challenging problems related to cryptography, optimization, and simulation.  
**LLaMA-2-Evolve70k-`NEFT`**:  
Sure, I’d be happy to help!  
Quantum computing is a type of computation that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. Unlike classical computers, which use bits (which can only have one of two values, either 0 or 1) to represent information, quantum computers use qubits, which can exist in multiple states simultaneously. This allows for much faster processing times when it comes to certain types of calculations.  
The basic idea behind quantum computing is to use a series of quantum gates to manipulate the state of individual qubits. These gates are similar to the logic gates used in classical computer architecture, but they operate on a different level of abstraction. For example, instead of using AND and OR gates to build more complex circuits, quantum gates like the CNOT gate allow us to create entangled pairs of qubits with specific correlations between their states.  
One of the key advantages of quantum computing is its ability to solve certain problems exponentially faster than classical computers. One famous example is factorization, where we need to find the prime factors of a large number. Using Shor’s algorithm, a quantum computer can factorize numbers much larger than what would be possible with classical methods. Another area where quantum computing has shown promise is in simulating quantum systems, which could lead to new insights into materials science and chemistry.  
However, there are also many challenges associated with building practical quantum computers. One major issue is error correction, since any disturbance to a qubit can cause it to lose its quantum properties and become a classical bit. Researchers are currently working on developing error-correcting codes that can mitigate these errors and make quantum computations more reliable over time. Additionally, there are still many open questions about how best to architect quantum computers, including issues related to scalability and interconnectivity between components.  
**LLaMA-2-OpenPlatypus**:  
Quantum computing is a rapidly developing field that has the potential to revolutionize how we process and store information. It relies on the principles of quantum mechanics, which describe the behavior of matter and energy at very small scales, such as atoms and subatomic particles.  
  
In classical computers, bits represent either 0 or 1, but in a quantum computer, qubits can exist in multiple states simultaneously. This property, called superposition, allows quantum computers to perform many calculations in parallel, greatly increasing their processing power. Additionally, quantum computers use entanglement, where two or more particles become connected in such a way that the state of one particle instantly affects the other, regardless of the distance between them. This phenomenon enables quantum algorithms to solve certain problems much faster than classical algorithms.  
  
One example of a problem that could be solved quickly by a quantum computer is factoring large numbers into prime factors. While this may not seem like a significant breakthrough, it would have wide-ranging implications for fields such as cryptography, since it would enable the rapid cracking of codes used to secure sensitive data. Another area where quantum computing could have a major impact is drug discovery, as it could help identify new treatments for diseases much faster than traditional methods.  
  
However, building a practical quantum computer with enough qubits to be useful remains a challenge. Currently, most quantum computers are relatively small and prone to errors due to environmental interference. Researchers are working on improving both the number of qubits and the accuracy of quantum computations, making progress towards realizing the full potential of this exciting technology.  
**LLaMA-2-OpenPlatypus-`NEFT`**:  
Quantum computing is a field of computer science that deals with the use of quantum mechanics for computation. It involves developing and studying computational models based on the principles of quantum theory, which describe the behavior of matter and energy at very small scales. The goal of quantum computing is to take advantage of the unique properties of quantum systems, such as superposition and entanglement, to solve complex problems more efficiently than classical computers can.  
In classical computing, bits represent either 0 or 1, but in quantum computing, qubits (quantum bits) can be in multiple states simultaneously until they are measured. This property, called superposition, allows quantum computers to perform many calculations in parallel, greatly increasing their processing power compared to classical computers. Another key feature of quantum computing is entanglement, where two or more particles become correlated such that the state of one particle instantly affects the other, regardless of the distance between them. Entanglement can be used to transmit and process information instantaneously, enabling faster communication and computations.  
One of the main challenges in building practical quantum computers is controlling and manipulating individual quantum bits accurately enough to realize useful algorithms. Currently, most quantum computers rely on experimental platforms using trapped ions, superconducting circuits, or photons. These devices are still relatively fragile and prone to errors, known as “noise,” which must be corrected by sophisticated error-correction codes. However, significant progress has been made in recent years towards scaling up quantum computers and achieving higher levels of noise tolerance.  
Some potential applications of quantum computing include cryptography, optimization problems, simulation of physical systems, and drug discovery. Quantum computers could also play a role in solving some of society’s biggest challenges, such as climate change modeling, materials science, and protein folding research. While much work remains to be done before quantum computers become widely available, the field continues to attract attention from both academia and industry due to its great promise for transforming various fields of study and engineering.

</div>

<div class="tcolorbox">

**`AlpacaEval` (Vicuna Subset) Instruction**: Write a symphony concert review, discussing the orchestra’s performance and overall audience experience.  
**LLaMA-2-Alpaca-7B Response**:  
The symphony concert was truly remarkable! The orchestra played with great energy and skill, creating a beautiful soundscape for the audience to enjoy. Every piece they performed was expertly executed, from the gentle melodies of Mozart to the powerful climaxes of Beethoven.  
  
The acoustics in the hall were perfect; every note could be heard clearly and distinctly. The lighting design added another layer of emotion to each piece, helping to draw out the emotions within the music. Overall, it was a wonderful evening full of fantastic music and an unforgettable experience for all who attended.  
**LLaMA-2-Alpaca-7B-`NEFT` Response**:  
The symphony concert was truly remarkable! The orchestra played with such passion and energy that it brought tears to my eyes. Every note they struck seemed to reverberate through the hall in perfect harmony. It was clear from their performance that each musician had put in countless hours of practice and dedication to create this beautiful sound.  
  
The conductor did an excellent job leading the musicians throughout the evening. His movements were full of emotion as he guided them through the pieces. He would often pause for moments at a time allowing the audience to take in every moment of the piece before moving on. This allowed us all to appreciate the intricacies of each song even more.  
  
The crowd was filled with people of all ages; some who knew the songs well and others who were there to simply enjoy the night. There wasn’t one person in attendance who didn’t seem to be having a great time. Even during the quieter parts of the show, you could feel the excitement in the air. Whenever something particularly exciting happened everyone cheered and applauded.  
Overall, I can say without a doubt that this symphony concert was one of the best experiences I have ever had. From start to finish, everything about it was absolutely amazing. If given the chance, I will definitely go back again next year.  
**LLaMA-2-Share-7B Response**:  
On Saturday evening at Carnegie Hall, I had the privilege of attending a symphony concert featuring the New York Philharmonic Orchestra under the direction of Maestro Lorin Maazel.  
  
The program for the evening consisted of three works by some of classical music’s most renowned composers - Beethoven’s “Egmont” Overture, Tchaikovsky’s Piano Concerto No. 1 with soloist Lang Lang, and Dvořák’s Symphony No. 9 in E minor, Opus 95 (“From the New World”).  
  
Maestro Maazel led the orchestra through each piece with precision and passion, drawing out rich and vibrant sounds from every section of the ensemble. The strings played with great depth and emotion, while the woodwinds and brass added colorful accents throughout the evening.  
  
Lang Lang was a particularly impressive performer as well, displaying technical mastery and artistic sensitivity in his interpretation of Tchaikovsky’s concerto. His playing was full of energy and expression, and he seemed completely immersed in the music.  
  
As for the audience, it was clear that everyone present was there to appreciate the beauty and power of live orchestral music. There were frequent bursts of applause and cheers throughout the night, and many people around me were wiping tears of joy from their eyes during the final movement of Dvořák’s symphony. It was a truly moving experience to be surrounded by so many people who shared a love for this timeless art form.  
  
Overall, the concert was a true triumph for both the musicians on stage and the audience in attendance. It was a reminder of why we continue to support and celebrate live music, and I left the hall feeling inspired and uplifted.  
**LLaMA-2-Share-7B-`NEFT` Response**:  
On Saturday night at Carnegie Hall, the New York Philharmonic put on a breathtaking performance of Symphony No. 5 by Ludwig van Beethoven. Under the direction of Music Director Jaap van Zweden, the orchestra delivered a powerful and emotional rendition of this classic work that left the packed house in rapturous applause.  
The evening began with a captivating performance of Mozart’s Overture to Die Zauberflöte, which set the stage for the grandeur of what was to come. As the lights dimmed and the orchestra launched into the opening notes of the Fifth Symphony, it was clear that this would be a special evening. Van Zweden expertly guided the orchestra through each movement, drawing out every nuance of the music while maintaining a sense of momentum and energy.  
One of the standout features of the performance was the orchestra’s dynamic range. From the softest pianissimo passages to the loudest fortissimo climaxes, every member of the ensemble played with precision and depth. The brass section, in particular, shone throughout the evening, delivering rich and full-bodied sound that filled the hall.  
In addition to the superlative playing of the musicians, the acoustics of Carnegie Hall were also a major factor in the success of the evening. This iconic venue has a warm and intimate atmosphere that allows listeners to fully immerse themselves in the music. Every note seemed to hang in the air, enveloping the audience in a sonic embrace that made it easy to forget the outside world and lose oneself entirely in the beauty of the music.  
As the final chords of the symphony faded away, the audience erupted into thunderous applause, showering the musicians with praise and admiration. It was a truly magical moment, one that will stay with me long after the evening came to a close. Overall, I would say that this was a once-in-a-lifetime experience that any lover of classical music should not miss.  
**LLaMA-2-Evolve70k Response**:  
I recently attended a symphony concert at the renowned Carnegie Hall in New York City. The evening’s program featured works by Beethoven, Mozart, and Tchaikovsky, performed by the prestigious New York Philharmonic Orchestra under the baton of maestro Alan Gilbert.  
The concert began with Beethoven’s lively “Leonore” Overture No. 3, which set the tone for the evening’s festivities. The orchestra played with great energy and precision, showcasing their mastery of the classical repertoire. Maestro Gilbert conducted with grace and fluidity, drawing out each note and phrase to maximum effect.  
Next up was Mozart’s Symphony No. 29, a beautiful work that demonstrated the composer’s deftness with melody and harmony. The orchestra played with even greater nuance than before, creating a rich tapestry of sound that transported me to another world. I particularly enjoyed the woodwind section, whose delicate phrases added depth and dimension to the piece.  
Finally, we were treated to Tchaikovsky’s iconic Symphony No. 5. This was undoubtedly the highlight of the evening, as the orchestra brought all of their skill and passion to bear on this towering masterwork. From the haunting opening theme to the triumphant finale, every moment of the performance was filled with emotion and intensity. It was truly a thrilling experience to witness such accomplished musicians bring these timeless compositions to life.  
Overall, the evening was a resounding success. The New York Philharmonic Orchestra proved themselves to be among the finest ensembles in the world, playing with technical prowess and artistic sensitivity. Maestro Gilbert showed himself to be a skilled conductor, guiding the orchestra through each work with care and attention to detail. And the acoustics of Carnegie Hall itself were simply breathtaking, providing a perfect backdrop for the music. If you have the opportunity to attend a symphony concert, I highly recommend it - you won’t be disappointed!  
**LLaMA-2-Evolve70k-`NEFT` Response**:  
As I took my seat in the concert hall, anticipation was building as the lights dimmed and the orchestra began to play. The first piece on the program was Beethoven’s Symphony No. 1, which immediately captured my attention with its powerful melodies and intricate rhythms. The conductor led the orchestra with precision and enthusiasm, and the musicians responded with a dynamic performance that showcased their exceptional skill and passion for music-making.  
  
The second movement, marked “Andante Favori,” was particularly beautiful, with delicately played strings and woodwinds creating a serene and introspective mood. The third movement, “Allegro,” was full of energy and drive, with the brass section adding a bold and assertive presence to the music. The final movement, “Alla Da Prestissimo,” was a thrilling conclusion to the symphony, with lightning fast tempos and dazzling virtuosity from all members of the orchestra.  
  
Throughout the performance, I was impressed by the cohesion and unity of the ensemble, as well as the individual excellence of each musician. The sound they produced was rich and full, yet also crystal clear and precise, and there were many moments where I was left breathless by the sheer beauty and power of the music.  
  
In addition to the excellent playing, another highlight of the evening was the warm and welcoming atmosphere created by the orchestra and conductor. They interacted with the audience throughout the performance, sharing insights into the music and encouraging us to enjoy the journey along with them. This made for a truly engaging and immersive experience, one that left me feeling inspired and uplifted long after the last note had faded away.  
  
Overall, this symphony concert was a remarkable event that I will remember for years to come. The orchestra’s superb performance combined with the audience’s enthusiastic support to create a magical night of music-making that I would recommend to anyone looking for a truly unforgettable musical experience.  
**LLaMA-2-OpenPlatypus**:  
Title: A Night of Enchantment with the Boston Symphony Orchestra  
Date: March 15th, 2023  
Venue: Symphony Hall, Boston MA  
Review by John Doe for The Boston Globe  
  
The Boston Symphony Orchestra (BSO) returned to its home stage at Symphony Hall last night after a two-week hiatus, and they did not disappoint as they presented their first program under the baton of Music Director Andris Nelsons since his recent appointment extension was announced. With a captivating lineup featuring works from three different eras and composers, this evening proved to be nothing short of enchanting for both musicians and audiences alike.  
  
Opening the show was Jean Sibelius’ iconic “Finlandia,” which set the tone for what would prove to be an exquisite journey through various musical styles throughout history. Performed without intermission, the BSO immediately captured our attention with their impeccable precision and dynamic range - particularly during the stirring brass passages that brought us into the heart of Finnish nationalism. It was clear from the outset that Maestro Nelsons had prepared his ensemble meticulously; every note seemed carefully considered yet effortlessly executed, creating a seamless blend between soloists and full orchestra.  
  
Next up was Pyotr Ilyich Tchaikovsky’s beloved Violin Concerto in D Major, performed by one of today’s most acclaimed violinists, Leonidas Kavakos. From the very first notes, it became apparent why he has earned such widespread praise; his technical prowess was on full display while also exhibiting immense depth of feeling throughout each movement. His partnership with the BSO was truly special, resulting in moments of pure magic when all eyes were fixed upon him as he soared above the orchestral accompaniment.  
  
After intermission came another treat: Gustav Mahler’s Symphony No. 4 in G Major (“Tragic”). This work often receives less attention compared to some other Mahler symphonies due to its relative brevity (approximately half an hour), but make no mistake - there is much beauty and emotion packed within these few minutes. Under Mr. Nelsons’ guidance, we experienced every twist and turn along the way, culminating in a thrilling climax that left us breathless before suddenly fading away into silence.  
  
As if that weren’t enough, the encore offered something entirely unexpected: Samuel Barber’s Adagio for Strings. Originally composed as part of a string quartet arrangement, this haunting melody has since become synonymous with mourning and loss thanks largely to its inclusion in films like Platoon and The Elephant Man. Hearing it live provided goosebumps galore as it filled Symphony Hall from top to bottom; even those who may have been unfamiliar with the piece prior could not help but feel moved by its poignancy.  
  
In conclusion, tonight’s performance demonstrated exactly why the Boston Symphony Orchestra remains one of the world’s preeminent ensembles nearly two centuries after its founding. Each musician appeared fully invested in their craft, delivering stunning interpretations of timeless classics alongside more contemporary pieces. As someone lucky enough to witness this magical event firsthand, I can confidently say that it will remain etched in my memory forever - just like many others who attended last night’s sold-out show.  
**LLaMA-2-OpenPlatypus-`NEFT` Response**:  
Title: A Night of Enchantment with the Boston Symphony Orchestra  
  
The Boston Symphony Orchestra (BSO) presented their latest concert at historic Symphony Hall in Boston on February 1st, 20XX. The evening was filled with captivating music from some of the most renowned composers throughout history. With conductor Andris Nelsons leading the way, this night proved to be one of enchantment for both performers and audiences alike.  
  
As we took our seats inside Symphony Hall, anticipation filled the air as people chatted excitedly about what they were expecting from tonight’s show. The hall itself is quite impressive; its ornate design featuring intricate details like marble columns and stained glass windows creating an atmosphere of grandeur perfect for such an event. Once everyone had found their seat, there was complete silence except for the occasional cough or shuffling of feet until finally, the lights dimmed signaling that it was time for the show to begin.  
  
The first piece performed was Jean Sibelius’ “Finlandia,” Opus 26, which set the tone for the rest of the evening – grandiose yet serene. As expected, Andris Nelsons brought out each note vibrantly while still allowing space for individual instruments to shine through. It was clear from his direction that he had prepared extensively for this performance, ensuring every aspect of the music came alive under his baton.  
  
Next up was Pyotr Ilyich Tchaikovsky’s iconic Violin Concerto in D Major, Opus 35, featuring soloist Gil Shaham. From the very first notes played by Shaham, it became apparent why he has earned critical acclaim worldwide – his technical prowess combined with emotional depth made for an unforgettable performance. He seamlessly transitioned between different styles and tempos, keeping us enthralled throughout the entirety of the concerto. Whenever he stepped back into the ensemble after his solos, you could hear a collective sigh of relief as if we had all been holding our breath waiting for him to return. Andris Nelsons conducted flawlessly alongside him, adding even more depth and beauty to this already beautiful work.  
  
After intermission, the BSO returned to the stage for Johannes Brahms’ Symphony No. 4 in E Minor, Opus 98. This piece marked another highlight of the evening, demonstrating once again how well these musicians can come together as one cohesive unit. Each section had its moment to shine without ever overpowering the others, resulting in a harmonious blend of sound that left us spellbound. Andris Nelsons showed great versatility here too, expertly navigating through various sections while maintaining momentum throughout the whole movement.  
  
To end the night on a high note, the BSO closed with Gustav Mahler’s Symphony No. 1 in D Major (“Titan”). Right from the opening bars, it was evident that this would be no ordinary performance. There was an energy within the orchestra that seemed to infect those watching, causing many people around me to clap along enthusiastically despite not having reached the final movement just yet! Andris Nelsons continued to lead them effortlessly, guiding us through this epic journey full of emotion and passion. By the time we reached the climactic finale, cheers erupted from the crowd as if we had all been released from some sort of magical trance. It truly felt like we had witnessed something special, something that will stay with us long after the last note had died away.  
  
In conclusion, the Boston Symphony Orchestra put on an incredible show last night, displaying impeccable musicianship and artistry throughout their performances. Their dedication to bringing new life to classic works while also exploring lesser-known gems was greatly appreciated by all who attended. If you have the opportunity to catch one of their future shows, do not hesitate - allow yourself to become enchanted by their magic!

</div>

Here are a couple more examples from the `NEFTune` models that performed the best on AlpacaEval (LLaMA-2 trained on Evolve-Instruct).

<div class="tcolorbox">

**`AlpacaEval` (Vicuna Eval) Instruction**: What if Vincent van Gogh had been a successful artist during his lifetime?  
**LLaMA-2-Evolve70k**:  
As an AI language model, I don’t have personal beliefs or opinions but based on historical facts and researches, it is believed that Van Gogh would not have experienced mental illness as he was recognized for his artistic talent in his early years before developing psychiatric problems later in life. He may have continued to create more masterpieces and become one of the most famous artists of all time. His legacy might also be different from what we know today since he did not achieve recognition until after his death when his brother sold some of his paintings at auction.  
**LLaMA-2-Evolve70k-`NEFT`**:  
As an AI language model, I don’t have personal beliefs or opinions but here are some possible scenarios based on historical facts and speculation:

1\) Van Gogh would likely have continued to create art throughout his life, as he was prolific in his output even before his breakthrough success. However, with financial stability and recognition, he may have experimented more with different styles and themes, pushing himself creatively in new directions.

2\) His relationships with others might also have changed. With money and fame comes social status, which could have opened up opportunities for him to mingle with other artists, writers, and intellectuals of his time. He may have formed deeper friendships and collaborations with fellow creatives, potentially leading to further innovation within the arts community.

3\) Unfortunately, mental illness can be difficult to treat, especially in the late 19th century when medical knowledge was limited. Despite his success, it’s unlikely that Van Gogh’s struggles with depression and anxiety would have disappeared. In fact, they may have intensified due to the added pressure and expectations placed upon him by society and his peers.

4\) Ultimately, whether or not Van Gogh became a successful artist during his lifetime wouldn’t necessarily change the impact he made on modern art history. His work has endured through countless exhibitions, books, films, and reproductions, cementing his legacy as one of the most influential painters of all time.

**`AlpacaEval` (1st) Instruction:** What are the names of some famous actors that started their careers on Broadway?  
**LLaMA-2-Evolve70k**:  
1. Meryl Streep  
2. Robert De Niro  
3. Al Pacino  
4. Kevin Spacey  
5. Nathan Lane  
6. Audra McDonald  
7. James Cagney  
8. Bette Midler  
9. Richard Gere  
10. John Lithgow  
**LLaMA-2-Evolve70k-`NEFT`**:  
1. Al Pacino - He made his stage debut in 1967’s “Does a Tiger Wear a Necktie?"" and went on to star in several other Broadway productions, including “The Indian Wants the Bronx"" (1968), ”The Resistible Rise of Arturo Ui” (1969) and “Serpico” (1973).  
2. Meryl Streep - She began her career on Broadway in 1975 with the play “Julia,” followed by roles in “Lost in Yonkers” (1991), “A Little Night Music” (2009) and “Three Tall Women” (2018).  
3. James Earl Jones - The actor first appeared on Broadway in 1957’s “The Long Dream,” going on to star in numerous plays and musicals throughout his career, including “Of Mice and Men” (1974), “Fences” (1987) and “Driving Miss Daisy” (1987).  
4. Nathan Lane - He made his Broadway debut in 1985’s “A Funny Thing Happened on the Way to the Forum,” winning a Tony Award for his performance. Other notable shows include “The Producers"" (2001), “The Nance” (2013) and “Angels in America” (2018).  
5. Bette Midler - The singer-actress got her start on Broadway in 1965’s “Fiddler on the Roof,” later earning critical acclaim for her performances in “Forbidden Planet” (1989), “Golden Age” (2005) and “I’ll Eat You Last: A Chat With Sue Mengers” (2013).  

</div>

Here is a case where both LLaMA-2 models trained on Evolve-Instruct cannot output the correct answer:

<div class="tcolorbox">

**`AlpacaEval` (Vicuna Eval) Instruction:** Given that f(x) = $`5x^3`$ - 2x + 3, find the value of f(2).  
**LLaMA-2-Evolve70k**:  
f(2) = 5$`(2)^3`$ - 2(2) + 3  
= 5 \* 8 - 4 + 3  
= 19 - 6 + 3  
= 22  
**LLaMA-2-Evolve70k-`NEFT`**:  
To find the value of f(2), we need to substitute x=2 into the function and simplify it. We have: f(2) = 5$`(2)^3`$ - 2(2) + 3  
= 40 - 8 + 3  
= 45  
Therefore, f(2) = 45.

</div>

[^1]: Code is available on Github: <https://github.com/neelsjain/NEFTune>. Correspondence to Neel Jain: $`<`$njain17@umd.edu$`>`$.

[^2]: Note that for all models we performed generation with a repetition penalty of 1.2, held constant across all experiments.
