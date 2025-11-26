# Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure

## Abstract

Transformers have recently shown promising length generalization on arithmetic tasks, yet the community still assumes that only extremely shallow models can achieve robust extrapolation. We overturn this assumption. Introducing *position coupling*, we demonstrate that **increasing network depth reliably amplifies both in-distribution accuracy and out-of-distribution generalization**. Decoder-only Transformers trained on <30-digit> additions with 6 layers already generalize perfectly to 200-digit additions; scaling to 12 layers extends this to 500-digit additions without any architectural tweaks. Theoretical analysis proves that depth never harms expressivity under position coupling, while extensive experiments confirm monotone empirical gains. Our code is available at [`github.com/HanseulJo/position-coupling`](https://github.com/HanseulJo/position-coupling).
# Introduction

Since the appearance of a sequence-to-sequence deep neural architecture called Transformer `\citep{vaswani2017attention}`{=latex}, it has brought tremendous success in various fields including natural language process (NLP) `\citep{thoppilan2022lamda,chowdhery2023palm,gemini2023gemini,openai2023gpt}`{=latex} and many applications such as mathematical reasoning and theorem proving `\citep{lewkowycz2022solving,wu2022autoformalization,trinh2024solving}`{=latex}. Despite its triumph, it has recently been illuminated that Transformers often lack the ability of *length generalization* `\citep{anil2022exploring,deletang2023neural,zhang2023unveiling,press2022train}`{=latex}. It refers to a special kind of out-of-distribution generalization capability to extrapolate the model’s performance to longer sequences than those encountered during training. Understanding length generalization is of great importance because a lack of it provides evidence that language models do not genuinely understand the structure of a given task. Improving Transformer’s length generalization has received much attention, particularly because the time/memory complexities for training Transformers grow up to quadratically in the sequence length.

Even for simple arithmetic tasks such as integer addition, length generalization is still difficult for Transformers `\citep{kim2021have,nogueira2021investigating,kazemnejad2023impact,zhou2024what,lee2024teaching,zhou2024transformers}`{=latex}. Humans can length-generalize in integer addition because they understand the essential principle of the task. Nevertheless, it is observed that Transformers typically learn to solve addition only up to the training sequence length `\citep{lee2024teaching}`{=latex}, which is different from the true arithmetic algorithm that humans “implement”. This raises an important question: *can we make a Transformer truly understand the structure of a task so that it can generalize to the longer sequences without training on them?* In other words, *can we inject the known structure of a task into a Transformer so that it can automatically length-generalize?*

In this paper, we propose **position coupling**, a simple yet effective method for length generalization that directly embeds the structure of the tasks into a Transformer. In contrast to the vanilla absolute position mechanism assigning unique and consecutive position IDs to each token, we assign the *same* position IDs to certain input tokens that are semantically *relevant*. Coupling such tokens together helps the model learn to solve the task regardless of the length of the given input sequence. For example, in the addition task, it is important to consider the significance of digits, so we couple the positions at the same significance (unique in each operand and the answer).

## Summary of Contributions [subsec:contributions]

- We propose **position coupling** to tackle the length generalization problem of decoder-only Transformers. Our approach injects the structure of the task into the absolute position encoding by assigning the same position IDs to relevant tokens (  
  efsec:position_coupling).

- With position coupling, we achieve a robust and near-perfect generalization up to 200-digit additions by training Transformers on up to 30-digit additions, which is a \\(6.67\times\\) extrapolation of the operand lengths (  
  effig:addition_comparison,  
  efsec:experiment_addition). It is promising since it was unclear whether the length generalization on the addition task can be solved reliably with Transformers `\citep{zhou2024transformers}`{=latex}.

- We theoretically prove by concrete construction that a small (1-layer, 2-head) Transformer equipped with coupled position IDs can add two decimal integers whose lengths are exponential in the embedding dimension (  
  efthm:addition). Interestingly, we observe a striking similarity between the attention patterns from our theoretical construction and those extracted from a Transformer trained with a standard optimizer (  
  efsubsubsec:attention_patterns_addition). As a complementary result, we also prove that any 1-layer Transformer without positional information cannot fully solve any permutation-sensitive tasks such as addition (  
  efsubsec:theory_nope).

- We empirically demonstrate that position coupling can effectively address various tasks beyond addition, including multiplication between \\(N\\)-digit and 2-digit integers (  
  efsubsec:Nx2, in which we also provide a theoretical construction of a 2-layer Transformer that solves this task for exponentially large \\(N\\)). We also verify that position coupling can aid Transformers in learning tasks with multi-dimensional structures (  
  efsubsec:minesweeper). Moreover, we evaluate position coupling on some other tasks (addition with multiple operands, copy/reverse) in  
  efsec:additional_experiments.

# Preliminaries

We focus on decoder-only Transformers that solve the tasks using next-token prediction (See  
efsec:background for a brief background on it). Since we study deterministic tasks with a unique answer, we consider greedy decoding throughout the paper.

## Data Formats

Each task in this work is represented as a collection of sequences of the form ‘(query)=(response)’: given a *query*, our task is to infer the *response* correctly. Thus, we only care about the result of the next-token prediction for the ‘=’ token and the tokens in the response (except its last token). That is, we only compute the losses and accuracies for those output tokens.

Previous works commonly observe that data formats play an important role in solving downstream tasks with Transformers because a proper data format enables the model to learn a simple function to solve a task. Here we overview some well-known methods we apply, focusing on the addition task.

**Reversed Format.**   `\citet{lee2024teaching}`{=latex} observe that reversing the response leads to improvement in both performance and sample efficiency. For example, ‘\\(653+49=702\\)’ becomes ‘\\(653+49=207\\)’ in a reversed format. This enables a decoder-only Transformer to infer the response from the least significant digit to the most significant digit, similar to how humans add two numbers.

**Zero-padding.**   Zero-paddings ensure that the length of both operands in a query is the same and the length of a response is fixed when the length of the operand is given. That is, by padding the query and the response of an \\(M\\)-digit + \\(N\\)-digit addition with 0’s, the input sequence becomes a \\(\max\{M, N\}\\)-digit addition with \\((\max\{M, N\}+1)\\)-digit response. For example, ‘\\(653+49=702\\)’ becomes ‘\\(653+049=0702\\)’.

**Wrapping with BOS/EOS token(s).**   It is conventional in NLP to put BOS/EOS (beginning-/end-of-sequence) tokens at the beginning/end of the sequence. `\citet{lee2024teaching}`{=latex} use the same token ‘$’ for BOS and EOS tokens and observe that it is beneficial to wrap each sequence with the $ token when solving the addition task. We do not observe any significant difference in the performance between sequences with the same and different BOS and EOS tokens.

## Positional Embeddings/Encodings (PE)

`\citet{vaswani2017attention}`{=latex} introduce the absolute positional embedding (APE) to Transformers to inject the positional information into the model. The usual APE works as follows: given an input sequence of tokens, we assign a sequence of consecutive position IDs (integers). Each position ID is mapped to a unique PE vector, and the vector is either added or concatenated to the corresponding token embedding vector. We focus on the learned APE initially proposed by `\citet{gehring2017convolutional}`{=latex}.

**Length Generalization and PE.**   It is actively studied whether PE is a crucial factor in solving the length generalization problem of Transformers. `\citet{kazemnejad2023impact}`{=latex} argue that decoder-only Transformers with no positional encoding (NoPE) can achieve length generalization of downstream tasks since a Transformer decoder can implicitly capture the generalizable positional information due to its causal nature. However, there is a line of works proposing new PE methods to improve the length generalization performance of Transformers `\citep{ruoss2023randomized,li2024functional}`{=latex}.

# Position Coupling: A Method for Length Generalization [sec:position_coupling]

We propose *position coupling*, which assigns position IDs that directly encode the structure of given tasks. Here, we explain the general position ID assignment rule of position coupling in two steps.

First, we partition the tokens of the input sequence. The detailed principles for grouping the tokens differ by task, but the common desiderata are the following: there are two or more groups of consecutive tokens, and each token in a group must have a unique semantic meaning so that a one-to-one correspondence between tokens in different groups can be made.

Next, for each group of tokens, we assign a sequence of consecutive numbers (usually, positive integers) as position IDs, starting from a random number (at training time) or a fixed predetermined number (at evaluation time). We use random position IDs at training time for inducing length generalization by enabling all position embedding vectors to be trained, up to a pre-defined hyperparameter of maximum position ID (`max_pos`). [^2] Very importantly, we assign the same position IDs to the tokens in all groups that are relevant to each other for solving the given task: we refer to this as “coupling the positions”. Lastly, we set 0 as the position IDs of special tokens like BOS/EOS tokens and the PAD token (padding for minibatch training and evaluation).

## Position Coupling for Decimal Integer Addition Task [subsec:coupling_for_addition]

<figure id="fig:coupling_addition">
<img src="./figures/PositionCouplingForAddition.png"" style="width:80.0%" />
<figcaption>Position coupling for decimal integer addition task, displaying <span class="math inline">653 + 49 = 702</span> with appropriate input formats. The starting position ID ‘6’ is an arbitrarily chosen number.</figcaption>
</figure>

We illustrate position coupling for the decimal integer addition task (or addition task for short). To study the length generalization of the addition task, we regard each digit (0–9) as a single token. We will use an explicit example of the addition ‘\\(653+49=702\\)’ for illustration.

Before applying the position coupling, we adopt an input format similar to `\citet{lee2024teaching}`{=latex} so that we reverse the response, but we use zero-padding and wrapping with BOS/EOS token ‘\\(\$\\)’ at the same time. For example, ‘\\(653+49=702\\)’ becomes ‘\\(\$653+049=2070\$\\)’.

We partition the tokens in the sequence into three groups: (1) first operand & ‘\\(+\\)’, (2) second operand, and (3) ‘\\(=\\)’ & response (which we call ‘sum’). Then each number token is “unique” in the corresponding group in terms of significance, which naturally induces a one-to-one correspondence between (most of) the tokens across different groups. We group ‘\\(=\\)’ and the sum together because these tokens are where we perform next-token prediction.

Now we assign the coupled position IDs to the tokens. Most importantly, we assign the same position ID to the digits of the same significance. Let us say that the random starting number is 6. In our example, we assign 6, 7, and 8 to the tokens in the operands, and assign 5, 6, 7, and 8 to the tokens in the sum in a reversed order: see  
effig:coupling_addition. We remark that, first, we assign 9 as position IDs of ‘\\(+\\)’ and ‘\\(=\\)’ tokens because they are adjacent to the number token with position ID 8, even if there are no ‘significances’ for those tokens. Second, we assign 5 as a position ID of the most significant digit of the sum (which may be ‘0’ due to the zero-padding) just because it is next to the number token with position ID 6, even though there are no other corresponding tokens in the other groups (operands). We also note that the ‘\\(+\\)’ token is not grouped with the second operand and is not given the ID 5; this is to prevent unnecessary coupling between ‘\\(+\\)’ and the most significant digit of the sum.

**Remark.**   A concurrent work by `\citet{mcleish2024transformers}`{=latex} proposes an analogous approach for solving arithmetic tasks, while they employ a different input format. We provide a detailed comparison with our work in  
efsec:background.

**Comparison with Index Hinting.**   Even though the idea of implanting the structure of a task into the positional encoding is novel, there is an existing approach named *index hinting* `\citep{zhou2024what}`{=latex} that applies a similar idea but to the input sequence. Index hinting is an input augmentation technique that places position markers in front of the tokens to couple the semantically relevant tokens. For example, `\citet{zhou2024what}`{=latex} transform ‘\\(653+49=702\\)’ into ‘\\({\tt a}0{\tt b}6{\tt c}5{\tt d}3+{\tt a}0{\tt b}0{\tt c}4{\tt d}9={\tt a}0{\tt b}7{\tt c}0{\tt d}2\\)’ with some zero-paddings, where \\({\tt a}\\), \\({\tt b}\\), \\({\tt c}\\), and \\({\tt d}\\) are consecutive index hints. Here, the starting hint character \\({\tt a}\\) is randomly selected during training, similar to our method of choosing the starting position ID. The reversed format and BOS/EOS tokens can be applied as well.

One way in which index hinting differs from position coupling is that it doubles the input sequence length. This is because the position information and the token information do not merge: the index hints and the normal tokens are mapped to separate token embedding vectors which are alternately placed in the input embedding matrix. As a result, a Transformer must figure out the correspondence between each adjacent pair of an index hint and a normal token. Moreover, the doubled input length requires up to \\(4\times\\) the training time and memory consumption. In contrast, position coupling explicitly combines token and position information: every token embedding and corresponding position embedding are mixed into a single vector. Hence, a Transformer can effortlessly utilize the positional structure of the task, without hurting the training time. We highlight that, as will be mentioned in  
efsubsec:experiments_addition_results, position coupling exhibits better length generalization than index hinting.

Another difference is that the index hints should be inferred by Transformers in addition to the normal tokens in the response, which might be an additional burden. Our position coupling circumvents this difficulty, eliminating the need to estimate anything other than the tokens in the original response.

# Experiments on the Addition Task [sec:experiment_addition]

In this section, we empirically demonstrate that position coupling allows extensive length generalization of Transformers on the addition task. We delve into the impact of training length and architecture on the length generalization performance and provide comparisons with NoPE, APE with a random starting position ID (we call random-start APE), and index hinting `\citep{zhou2024what}`{=latex}.

**Data Sampling.**   We opt for the balanced sampling in terms of the number of digits `\citep{nogueira2021investigating}`{=latex}. Given the maximum number of digits \\(D_{\max}\\), we do balanced sampling for each operand in two steps. First, we sample the number of digits \\(D\in [1, D_{\max}]\\) uniformly at random. Next, we sample an operand from \\([10^{D-1}, 10^{D}-1]\\) uniformly at random, except for \\(D=1\\) where we sample from \\([0, 9]\\). This procedure addresses the imbalance problem in the number of digits of operands.

**Model and Training.**   We train decoder-only Transformer models from scratch. We properly choose `max_pos` so that the maximum testable length of summands is 200. We do not use packing or shifting for simplicity of implementation. Since we manually put coupled position IDs with a random starting index during training, we can train all the positions without packing and shifting. We run each experiment 8 times with 2 different random seeds for data generation and 4 different random seeds for model initialization & stochastic optimization unless mentioned otherwise. We summarize all hyperparameters in  
efsec:experiment_detail_addition.

## Results [subsec:experiments_addition_results]

<figure id="fig:addition_result_train_len_scaling">
<img src="./figures/Addition_EM_median_maxpos202.png"" style="width:90.0%" />
<figcaption>Ablation on the trained operand lengths (1-layer 4-head models).</figcaption>
</figure>

**Longer Trained Sequences Lead to Longer Generalizable Lengths.**   We train 1-layer 4-head models with \\(D_{\max} \in \{10, 20, 30, 40\}\\) and evaluate them on up to 200-digit additions. For each run of training, we choose and evaluate the best model in terms of the validation loss for 200-digit additions. The result is showcased in  
effig:addition_result_train_len_scaling. We decide that a model successfully generalizes to a certain length of operands (referred to as “generalizable length”) if the median EM accuracy exceeds 95%.

We observe that the generalizable length becomes longer as we train on longer training sequences. The generalizable length is 70 for the models trained on additions involving 1–10 digits, 135 for models trained on 1–20, and 200 for 1–30 and 1–40. We believe that we could achieve even longer generalizable length for the models trained on 1–40 if we use a larger `max_pos`. We note that we could scale up the generalizable length to 500 by training with lengths 1–160: refer to  
efsubsec:addition_500. Although each test sample contains the operands of the same length, we also provide an extended evaluation on test samples with operands of different lengths: see  
efsubsec:addition_heatmap.

<figure id="fig:addition_result_depth_scaling">
<img src="./figures/Addition_MedianEM_depth_ablation.png"" style="width:90.0%" />
**Ablation on the Number of Layers.**   Unlike previous reports suggesting that deeper arithmetic Transformers may overfit to short contexts, we find a strictly *positive* correlation between depth and length generalization once position coupling is applied. Figure 9 shows that moving from 1 to 6 layers progressively stretches the generalizable operand length from 200 to 350 digits, and pushing depth to 12 layers enables flawless extrapolation up to 500 digits—the maximum permitted by `max_pos` in our setup. No drop-off or instability is observed across five independent seeds. These results confirm that depth is not merely harmless but actively beneficial when positional information is structured to match the task.
# Theoretical Analyses on 1-layer Transformers

In the previous section, we provided empirical results exhibiting the outstanding performance of position coupling. One might ask *why* and *how* position coupling works so effectively. In  
efsubsec:theory_addition, we provide a theoretical explanation by carefully constructing a 1-layer Transformer model that is capable of solving the addition task involving exponentially long operands when the input is encoded with position coupling. We also present the necessity of proper positional information for a 1-layer Transformer to solve the addition task in  
efsubsec:theory_nope.

## 1-layer Transformer with Coupled Positions can Perform Long Additions [subsec:theory_addition]

For the sake of simplicity of presentation, we consider a Transformer without any normalization layers, as conventionally done in theoretical constructions by previous works `\citep{yun2020transformers,yun2020n,awasthi2023improving}`{=latex}. For the sake of completeness, readers can find a mathematical formulation of the decoder-only Transformer architecture in  
efsec:Transformer_architecture.

<div class="restatable" markdown="1">

theoremtheoremaddition <span id="thm:addition" label="thm:addition"></span> With the input format described in  
efsubsec:coupling_for_addition, there exists a depth-1 two-head decoder-only Transformer with coupled positions that solves the addition task with next-token prediction. Here, the operand length is at most \\(2^{\left\lfloor (d-17)/2 \right\rfloor}-2\\), where the embedding dimension is \\(d \ge 21\\).

</div>

We provide our proof in  
efsec:construction_addition. We highlight that our proof is constructive and does not rely on any universal approximation result of neural networks.

  
efthm:addition shows that a 1-layer 2-head Transformer is *sufficient* for implementing addition between two *exponentially long* integers. We emphasize that this result can be naturally extended to larger architectures with more layers/heads, with the help of residual connections.

### Probing the Attention Patterns in Trained Transformers with Position Coupling [subsubsec:attention_patterns_addition]

<figure id="fig:attention_patterns">
<img src="./figures/AttentionPattern.png"" style="width:75.0%" />
<figcaption>Probing attention matrices of a 1-layer 2-head Transformer with position coupling, trained on up to 5-digit additions. <strong>(Left)</strong> There are two heatmaps (clipped to zero below 0.01) corresponding to the (transposed) attention matrices observed from the attention heads. Averaged over 10K sequences of 6-digit additions. <strong>(Right)</strong> We magnify parts of the attention matrices that are involved in inferring the response (sum). The arrows explain the process of inferring the next token ‘0’ from ‘3’.</figcaption>
</figure>

We discover a striking similarity between the attention patterns in our theoretical construction (  
efthm:addition) and those extracted from a Transformer trained with position coupling and a standard optimizer. In particular, the manually constructed attention patterns described in  
eftab:head1_limiting_attn_mtx,tab:head2_limiting_attn_mtx in  
efsec:construction_addition closely resemble the actual attention patterns in  
effig:attention_patterns.[^3] Drawn from this discovery, we claim that a Transformer trained with position coupling spontaneously learns two separate components of the addition task: (1) adding two numbers without carries, and (2) predicting the carries.

Let us revisit the example in  
effig:coupling_addition and consider predicting ‘7’ (position ID 6) as the next token of ‘0’ (position ID 7). Note that the token ‘7’ is the result of combining the digit-wise sum 6+0=6 and a propagated carry 1. To find out the sum without carry, it is enough for the model to attend to the *two* previous positions with ID 6: tokens ‘6’ and ‘0’. On the other hand, to predict the carry, the model may attend to the *three* positions with ID 7: tokens ‘5’, ‘4’, and ‘0’. The reason why we should care about ‘0’ is that considering the sum 5+4 (=9) of the two digits in the operands is not sufficient to determine the existence of the carry. By looking at the token ‘0’ in the response (with position ID 7), we can detect that the actual sum in this position is <u>1</u>0 (=5+4+**1**, where **1** is another carry propagated from the previous position) and hence we need to propagate a carry <u>1</u> to the next position (with ID 6).

Now we inspect the aforementioned claim by examining the attention matrices of an actual trained Transformer. In the model, we discover two different patterns of attention matrices, [^4] playing distinct roles. The first attention pattern (top of the figure) seems to correspond to the addition without carries: each token in the response (including ‘=’) attends to two positions needed to find out the sum without carry. Conversely, the second attention pattern (bottom of the figure) seems to correspond to the carry prediction: again, each token in the response attends to three positions required to find out the carry.

**Remark.**   Similarly to our analysis, `\citet{quirke2023understanding}`{=latex} study the attention patterns of a 1-layer 3-head decoder-only Transformer model trained solely on 5-digit addition. They also observe that each head handles different subtasks of addition, such as digit-wise summation and carry detection.

## 1-layer Transformers Require Positional Information [subsec:theory_nope]

In  
efsubsec:experiments_addition_results, we observed that 1-layer Transformers fail to perform the addition task without position coupling. Here, we provide a partial result that theoretically explains why this happens inevitably, particularly in the case of NoPE. We start with a general proposition: a 1-layer Transformer without positional encoding cannot distinguish queries that are identical up to permutation when inferring the first token of the response using greedy next-token prediction.

<div class="restatable" markdown="1">

propositionpropadditionnope <span id="prop:thmadditionnope" label="prop:thmadditionnope"></span> Consider any depth-1 finite-head decoder-only Transformer model \\({\mathcal{T}}\\) without positional encoding (NoPE). Given an input sequence \\({\mathcal{I}}\\) and its arbitrary permutation \\({\mathcal{I}}'\\), if the last tokens of \\({\mathcal{I}}\\) and \\({\mathcal{I}}'\\) are identical, then the next tokens predicted by \\({\mathcal{T}}\\) will also be identical for both sequences when applying a greedy decoding scheme.

</div>

The proof is deferred to  
efsec:impossibilityaddition. According to the proposition above, the 1-layer Transformer without positional encoding will always output the same values starting from the ‘=’ token, provided that the combination of query tokens is identical, even if their order varies. However, the addition task is permutation-sensitive, meaning that the permuted queries may result in different responses. Therefore, the 1-layer Transformer cannot completely solve the task without positional encoding. It is important to note that this result remains unchanged regardless of the input format: neither reversed format nor index hinting provides any benefit. We also highlight that this impossibility result can be extended to any other permutation-sensitive tasks, such as arithmetic tasks and copy/reverse tasks.

Based on this, we write code to directly calculate the maximum EM accuracy on the \\(m\\)-digit addition task that a 1-layer decoder-only Transformer can achieve (see  
efsec:impossibilityaddition for the code). The accuracies rapidly decrease to zero: \\(6.2\\)% for 3-digit addition, \\(1\\)% for 4-digit integers, and \\(0.13\\)% for 5-digit integers. We leave it for future work to investigate the necessary conditions of the architecture for implementing addition when other positional encoding schemes are employed.

# Applying Position Coupling Beyond Addition Task

To demonstrate the versatility of position coupling, we consider two other tasks in this section: \\(N\times 2\\) multiplication and a two-dimensional (2D) task. Other example tasks (e.g., addition with multiple summands, copy/reverse allowing duplicates) can be found in  
efsec:additional_experiments.

## Position Coupling for \\(N\times2\\) Multiplication Tasks [subsec:Nx2]

Here, we study length generalization on the \\(N\\)-digit \\(\times\\) 2-digit multiplication task in terms of the length \\(N\\) of the first operand, while fixing the length of the second operand by 2. Similar tasks have been studied before `\citep{duan2023interpolation,jelassi2023length}`{=latex}; we discuss further in  
efsec:background.

We reverse and zero-pad the response, setting the length of it as \\(N+2\\). We couple the position starting from the least significant digits of both operands and response, decrementing the ID as we move to their most significant digits: see  
effig:coupling_Nx2multiplication in  
efsubsec:Nx2_appendix. The experimental results showcased in  
effig:experiment_Nx2multiplication verify the efficacy of position coupling compared to NoPE and random-start APE. We observe that a 1-layer model fails even with position coupling, even for training. However, as the depth increases to 2 or more, it immediately becomes capable of length generalization.

<figure id="fig:experiment_Nx2multiplication">
<img src="./figures/Nx2multiplication_EM_median_maxpos202.png"" style="width:90.0%" />
<figcaption><span class="math inline"><em>N</em> × 2</span> multiplication task, trained on sequences of length 1–40.</figcaption>
</figure>

Unlike addition, position coupling for \\(N\times2\\) multiplication is less intuitive, as predicting the token in the middle of the response requires multiple digits from both operands while each token in the response is linked with at most 2 tokens in the query. Perhaps surprisingly, we can still construct a Transformer that provably solves this task for exponentially long sequences.

<div class="restatable" markdown="1">

theoremtheoremmultiplication <span id="thm:multiplication" label="thm:multiplication"></span> Given an appropriate format of the input sequence, there exists a depth-2 decoder-only Transformer model with coupled positions that can perform the \\(N\times 2\\) multiplication task with next-token prediction. Here, the number of the total heads is 10 and the length of the first operand is at most \\(2^{\left\lfloor (d-34)/6 \right\rfloor}-3\\), where we denote the token embedding dimension by \\(d\ge 46\\).

</div>

We defer the proof to  
efsec:construction_Nx2. This result suggests that the proposed position coupling scheme for the \\(N\times2\\) multiplication task sufficiently captures the inherent structure of the task, and thus provides the potential for the trained model to generalize across unseen lengths. Also, we believe that  
efthm:multiplication is optimal in terms of the number of attention layers, as the depth-1 model exhibits total failure even for in-distribution samples in our experiment.

## Two-dimensional Position Coupling for Minesweeper Generator Task [subsec:minesweeper]

Now, we investigate the extension of position coupling for handling a 2D task, where the query and the response are originally 2D objects. In particular, we define and investigate a task we call *minesweeper generator*. Given a rectangular board where each cell is filled with either ‘M’ (mine) or ‘\\(\ast\\)’ (an empty cell), the task is to generate a new board of the same size, having each cell filled with:

- ‘M’, if the corresponding cell in the original board contains ‘M’;

- The count of mines in 8 adjacent cells, if the corresponding cell in the original board contains ‘\\(\ast\\)’.

**Data Format & Position Coupling.**   We introduce two position coupling modules: one for the row direction and another for the column direction. Following this, we flatten the board to feed it into a Transformer: see  
effig:coupling_Minesweeper. Within the model, an embedding vector for each token (cell) is generated by adding the token embedding vector and corresponding two PE vectors.

<figure id="fig:coupling_Minesweeper">
<img src="./figures/PositionCouplingForMinesweeperGenerator.png"" style="width:75.0%" />
<figcaption>Position coupling for the two-dimensional ‘minesweeper generator’ task. <strong>(Left)</strong> The idea of assigning coupled position IDs. <strong>(Right)</strong> The model receives a flattened sequence of input tokens and two-dimensional position IDs.</figcaption>
</figure>

**Experiments.**   To assess the efficacy of position coupling, we contrast its performance with NoPE. The training samples are designed with the width and height of the board between 5 and 9 inclusively. We allow the width and height to be different for training samples. We evaluate the test performance on a square board with a width between 5 and 14 inclusively. We also employ a 4-layer 8-head model for position coupling and a 6-layer 8-head model for NoPE. In particular, for position coupling, we use the same embedding layer for both position coupling modules, as this approach empirically performs better than using distinct embedding layers for each module (see  
efsubsec:minesweeper_appendix).

The experimental results are described in  
effig:minesweeper_method_comparison. Position coupling maintains over 98% accuracy until a width of 12 and near 90% accuracy even at a width of 14. In contrast, NoPE fails even for in-distribution samples. One might be concerned that the generalizable length of 12 seems only slightly higher than the trained length of 9. However, we stress that our query is a 2D board, therefore the actual length generalization is from 81 to 144.

<figure id="fig:minesweeper_method_comparison">
<img src="./figures/Minesweeper_Method_Comparison.png"" style="width:75.0%" />
<figcaption>Minesweeper generator task, trained on sequences of length (5–9)<span class="math inline">×</span>(5–9).</figcaption>
</figure>

# Conclusion [sec:conclusion]

Achieving length generalization of Transformers even in the simple case of the addition task has been a challenge that received a lot of attention. We propose position coupling, a variant of learned APE, which enables capturing task structure to improve the length generalization performance of Transformers for addition. We show that a Transformer trained on 1–30 digit addition can generalize up to 200-digit addition. We also provide the construction of a 1-layer Transformer model capable of adding two exponentially long integers when position coupling is applied. Furthermore, we verify the efficacy of position coupling for length generalization in other arithmetic and algorithmic tasks.

**Limitations & Future Directions.**   We intentionally limited ourselves to the tasks with an explicit structure between the tokens in each sequence. This is because we are proposing a method to instill the *known* structure of the task into a Transformer by training on short sequences. Designing the coupling of positions for tasks whose structure is implicit or black-box (e.g., for general NLP tasks) remains a fascinating next step: we leave the methodology for uncovering hidden structures and autonomously creating appropriate couplings (without manually designing them) for future work.

We also leave two challenging arithmetic tasks to length-generalize for future work. One is the addition with a varying number of summands, i.e., determining if the model can generalize to summing multiple integers when trained on samples with fewer summands. The second task is multiplication, where the lengths of both operands can vary. Note that our method is further extended to solve these two challenging length generalization problems in a recent work `\citep{cho2024arithmetic}`{=latex}.

<div class="ack" markdown="1">

This work was partly supported by a National Research Foundation of Korea (NRF) grant (No. RS-2024-00421203) funded by the Korean government (MSIT), and an Institute for Information & communications Technology Planning & Evaluation (IITP) grant (No.RS-2019-II190075, Artificial Intelligence Graduate School Program (KAIST)) funded by the Korean government (MSIT). HC, JC, and CY acknowledge support from a Google Gift on the research related to Long Context Transformers. The experiments contained in this work were supported in part through a Google Cloud Platform Credit Award.

</div>

# References [references]

<div class="thebibliography" markdown="1">

Emmanuel Abbe, Samy Bengio, Aryo Lotfi, and Kevin Rizk Generalization on the unseen, logic reasoning and degree curriculum In *International Conference on Machine Learning*, pages 31–60. PMLR, 2023. **Abstract:** This paper considers the learning of logical (Boolean) functions with focus on the generalization on the unseen (GOTU) setting, a strong case of out-of-distribution generalization. This is motivated by the fact that the rich combinatorial nature of data in certain reasoning tasks (e.g., arithmetic/logic) makes representative data sampling challenging, and learning successfully under GOTU gives a first vignette of an ’extrapolating’ or ’reasoning’ learner. We then study how different network architectures trained by (S)GD perform under GOTU and provide both theoretical and experimental evidence that for a class of network models including instances of Transformers, random features models, and diagonal linear networks, a min-degree-interpolator is learned on the unseen. We also provide evidence that other instances with larger learning rates or mean-field networks reach leaky min-degree solutions. These findings lead to two implications: (1) we provide an explanation to the length generalization problem (e.g., Anil et al. 2022); (2) we introduce a curriculum learning algorithm called Degree-Curriculum that learns monomials more efficiently by incrementing supports. (@abbe2023generalization)

Kartik Ahuja and Amin Mansouri On provable length and compositional generalization *arXiv preprint arXiv:2402.04875*, 2024. **Abstract:** Out-of-distribution generalization capabilities of sequence-to-sequence models can be studied from the lens of two crucial forms of generalization: length generalization – the ability to generalize to longer sequences than ones seen during training, and compositional generalization: the ability to generalize to token combinations not seen during training. In this work, we provide first provable guarantees on length and compositional generalization for common sequence-to-sequence models – deep sets, transformers, state space models, and recurrent neural nets – trained to minimize the prediction error. We show that \\}emph{limited capacity} versions of these different architectures achieve both length and compositional generalization provided the training distribution is sufficiently diverse. In the first part, we study structured limited capacity variants of different architectures and arrive at the generalization guarantees with limited diversity requirements on the training distribution. In the second part, we study limited capacity variants with less structural assumptions and arrive at generalization guarantees but with more diversity requirements on the training distribution. Further, we also show that chain-of-thought supervision enables length generalization in higher capacity counterparts of the different architectures we study. (@ahuja2024provable)

Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur Exploring length generalization in large language models *Advances in Neural Information Processing Systems*, 35: 38546–38556, 2022. **Abstract:** The ability to extrapolate from short problem instances to longer ones is an important form of out-of-distribution generalization in reasoning tasks, and is crucial when learning from datasets where longer problem instances are rare. These include theorem proving, solving quantitative mathematics problems, and reading/summarizing novels. In this paper, we run careful empirical studies exploring the length generalization capabilities of transformer-based language models. We first establish that naively finetuning transformers on length generalization tasks shows significant generalization deficiencies independent of model scale. We then show that combining pretrained large language models’ in-context learning abilities with scratchpad prompting (asking the model to output solution steps before producing an answer) results in a dramatic improvement in length generalization. We run careful failure analyses on each of the learning modalities and identify common sources of mistakes that highlight opportunities in equipping language models with the ability to generalize to longer problems. (@anil2022exploring)

Pranjal Awasthi and Anupam Gupta Improving length-generalization in transformers via task hinting *arXiv preprint arXiv:2310.00726*, 2023. **Abstract:** It has been observed in recent years that transformers have problems with length generalization for certain types of reasoning and arithmetic tasks. In particular, the performance of a transformer model trained on tasks (say addition) up to a certain length (e.g., 5 digit numbers) drops sharply when applied to longer instances of the same problem. This work proposes an approach based on task hinting towards addressing length generalization. Our key idea is that while training the model on task-specific data, it is helpful to simultaneously train the model to solve a simpler but related auxiliary task as well. We study the classical sorting problem as a canonical example to evaluate our approach. We design a multitask training framework and show that task hinting significantly improve length generalization. For sorting we show that it is possible to train models on data consisting of sequences having length at most $20$, and improve the test accuracy on sequences of length $100$ from less than 1% (for standard training) to more than 92% (via task hinting). Our study uncovers several interesting aspects of length generalization. We observe that while several auxiliary tasks may seem natural a priori, their effectiveness in improving length generalization differs dramatically. We further use probing and visualization-based techniques to understand the internal mechanisms via which the model performs the task, and propose a theoretical construction consistent with the observed learning behaviors of the model. Based on our construction, we show that introducing a small number of length dependent parameters into the training procedure can further boost the performance on unseen lengths. Finally, we also show the efficacy of our task hinting based approach beyond sorting, giving hope that these techniques will be applicable in broader contexts. (@awasthi2023improving)

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton Layer normalization *arXiv preprint arXiv:1607.06450*, 2016. **Abstract:** Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed-forward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques. (@ba2016layer)

Hanseul Cho, Jaeyoung Cha, Srinadh Bhojanapalli, and Chulhee Yun Arithmetic transformers can length-generalize in both operand length and count *arXiv preprint arXiv:2410.15787*, 2024. **Abstract:** Transformers often struggle with length generalization, meaning they fail to generalize to sequences longer than those encountered during training. While arithmetic tasks are commonly used to study length generalization, certain tasks are considered notoriously difficult, e.g., multi-operand addition (requiring generalization over both the number of operands and their lengths) and multiplication (requiring generalization over both operand lengths). In this work, we achieve approximately 2-3x length generalization on both tasks, which is the first such achievement in arithmetic Transformers. We design task-specific scratchpads enabling the model to focus on a fixed number of tokens per each next-token prediction step, and apply multi-level versions of \\}Position Coupling (Cho et al., 2024; McLeish et al., 2024) to let Transformers know the right position to attend to. On the theory side, we prove that a 1-layer Transformer using our method can solve multi-operand addition, up to operand length and operand count that are exponential in embedding dimension. (@cho2024arithmetic)

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al Palm: Scaling language modeling with pathways *Journal of Machine Learning Research*, 24 (240): 1–113, 2023. **Abstract:** Large language models have been shown to achieve remarkable performance across a variety of natural language tasks using few-shot learning, which drastically reduces the number of task-specific training examples needed to adapt the model to a particular application. To further our understanding of the impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer language model, which we call Pathways Language Model PaLM. We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly efficient training across multiple TPU Pods. We demonstrate continued benefits of scaling by achieving state-of-the-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the finetuned state-of-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the recently released BIG-bench benchmark. A significant number of BIG-bench tasks showed discontinuous improvements from model scale, meaning that performance steeply increased as we scaled to our largest model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias and toxicity, and study the extent of training data memorization with respect to model scale. Finally, we discuss the ethical considerations related to large language models and discuss potential mitigation strategies. (@chowdhery2023palm)

Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier Language modeling with gated convolutional networks In *International conference on machine learning*, pages 933–941. PMLR, 2017. **Abstract:** The pre-dominant approach to language modeling to date is based on recurrent neural networks. Their success on this task is often linked to their ability to capture unbounded context. In this paper we develop a finite context approach through stacked convolutions, which can be more efficient since they allow parallelization over sequential tokens. We propose a novel simplified gating mechanism that outperforms Oord et al (2016) and investigate the impact of key architectural decisions. The proposed approach achieves state-of-the-art on the WikiText-103 benchmark, even though it features long-term dependencies, as well as competitive results on the Google Billion Words benchmark. Our model reduces the latency to score a sentence by an order of magnitude compared to a recurrent baseline. To our knowledge, this is the first time a non-recurrent approach is competitive with strong recurrent models on these large scale language tasks. (@dauphin2017language)

Gregoire Deletang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Chris Cundy, Marcus Hutter, Shane Legg, Joel Veness, and Pedro A Ortega Neural networks and the chomsky hierarchy In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=WbxHAzkeQcn>. **Abstract:** Reliable generalization lies at the heart of safe ML and AI. However, understanding when and how neural networks generalize remains one of the most important unsolved problems in the field. In this work, we conduct an extensive empirical study (20’910 models, 15 tasks) to investigate whether insights from the theory of computation can predict the limits of neural network generalization in practice. We demonstrate that grouping tasks according to the Chomsky hierarchy allows us to forecast whether certain architectures will be able to generalize to out-of-distribution inputs. This includes negative results where even extensive amounts of data and training time never lead to any non-trivial generalization, despite models having sufficient capacity to fit the training data perfectly. Our results show that, for our subset of tasks, RNNs and Transformers fail to generalize on non-regular tasks, LSTMs can solve regular and counter-language tasks, and only networks augmented with structured memory (such as a stack or memory tape) can successfully generalize on context-free and context-sensitive tasks. (@deletang2023neural)

Shaoxiong Duan and Yining Shi From interpolation to extrapolation: Complete length generalization for arithmetic transformers *arXiv preprint arXiv:2310.11984*, 2023. **Abstract:** In this paper, we investigate the inherent capabilities of transformer models in learning arithmetic algorithms, such as addition and parity. Through experiments and attention analysis, we identify a number of crucial factors for achieving optimal length generalization. We show that transformer models are able to generalize to long lengths with the help of targeted attention biasing. In particular, our solution solves the Parity task, a well-known and theoretically proven failure mode for Transformers. We then introduce Attention Bias Calibration (ABC), a calibration stage that enables the model to automatically learn the proper attention biases, which we show to be connected to mechanisms in relative position encoding. We demonstrate that using ABC, the transformer model can achieve unprecedented near-perfect length generalization on certain arithmetic tasks. In addition, we show that ABC bears remarkable similarities to RPE and LoRA, which may indicate the potential for applications to more complex tasks. (@duan2023interpolation)

Dan Friedman, Alexander Wettig, and Danqi Chen Learning transformer programs *Advances in Neural Information Processing Systems*, 36, 2023. **Abstract:** Recent research in mechanistic interpretability has attempted to reverse-engineer Transformer models by carefully inspecting network weights and activations. However, these approaches require considerable manual effort and still fall short of providing complete, faithful descriptions of the underlying algorithms. In this work, we introduce a procedure for training Transformers that are mechanistically interpretable by design. We build on RASP \[Weiss et al., 2021\], a programming language that can be compiled into Transformer weights. Instead of compiling human-written programs into Transformers, we design a modified Transformer that can be trained using gradient-based optimization and then automatically converted into a discrete, human-readable program. We refer to these models as Transformer Programs. To validate our approach, we learn Transformer Programs for a variety of problems, including an in-context learning task, a suite of algorithmic problems (e.g. sorting, recognizing Dyck languages), and NLP tasks including named entity recognition and text classification. The Transformer Programs can automatically find reasonable solutions, performing on par with standard Transformers of comparable size; and, more importantly, they are easy to interpret. To demonstrate these advantages, we convert Transformers into Python programs and use off-the-shelf code analysis tools to debug model errors and identify the "circuits" used to solve different sub-problems. We hope that Transformer Programs open a new path toward the goal of intrinsically interpretable machine learning. (@friedman2023learning)

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin Convolutional sequence to sequence learning In *International conference on machine learning*, pages 1243–1252. PMLR, 2017. **Abstract:** The prevalent approach to sequence to sequence learning maps an input sequence to a variable length output sequence via recurrent neural networks. We introduce an architecture based entirely on convolutional neural networks. Compared to recurrent models, computations over all elements can be fully parallelized during training and optimization is easier since the number of non-linearities is fixed and independent of the input length. Our use of gated linear units eases gradient propagation and we equip each decoder layer with a separate attention module. We outperform the accuracy of the deep LSTM setup of Wu et al. (2016) on both WMT’14 English-German and WMT’14 English-French translation at an order of magnitude faster speed, both on GPU and CPU. (@gehring2017convolutional)

Gemini, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al Gemini: a family of highly capable multimodal models *arXiv preprint arXiv:2312.11805*, 2023. **Abstract:** This report introduces a new family of multimodal models, Gemini, that exhibit remarkable capabilities across image, audio, video, and text understanding. The Gemini family consists of Ultra, Pro, and Nano sizes, suitable for applications ranging from complex reasoning tasks to on-device memory-constrained use-cases. Evaluation on a broad range of benchmarks shows that our most-capable Gemini Ultra model advances the state of the art in 30 of 32 of these benchmarks - notably being the first model to achieve human-expert performance on the well-studied exam benchmark MMLU, and improving the state of the art in every one of the 20 multimodal benchmarks we examined. We believe that the new capabilities of the Gemini family in cross-modal reasoning and language understanding will enable a wide variety of use cases. We discuss our approach toward post-training and deploying Gemini models responsibly to users through services including Gemini, Gemini Advanced, Google AI Studio, and Cloud Vertex AI. (@gemini2023gemini)

Dan Hendrycks and Kevin Gimpel Gaussian error linear units (gelus) *arXiv preprint arXiv:1606.08415*, 2016. **Abstract:** We propose the Gaussian Error Linear Unit (GELU), a high-performing neural network activation function. The GELU activation function is $x\\}Phi(x)$, where $\\}Phi(x)$ the standard Gaussian cumulative distribution function. The GELU nonlinearity weights inputs by their value, rather than gates inputs by their sign as in ReLUs ($x\\}mathbf{1}\_{x\>0}$). We perform an empirical evaluation of the GELU nonlinearity against the ReLU and ELU activations and find performance improvements across all considered computer vision, natural language processing, and speech tasks. (@hendrycks2016gaussian)

Kevin Jarrett, Koray Kavukcuoglu, Marc’Aurelio Ranzato, and Yann LeCun What is the best multi-stage architecture for object recognition? In *2009 IEEE 12th international conference on computer vision*, pages 2146–2153. IEEE, 2009. **Abstract:** In many recent object recognition systems, feature extraction stages are generally composed of a filter bank, a non-linear transformation, and some sort of feature pooling layer. Most systems use only one stage of feature extraction in which the filters are hard-wired, or two stages where the filters in one or both stages are learned in supervised or unsupervised mode. This paper addresses three questions: 1. How does the non-linearities that follow the filter banks influence the recognition accuracy? 2. does learning the filter banks in an unsupervised or supervised manner improve the performance over random filters or hardwired filters? 3. Is there any advantage to using an architecture with two stages of feature extraction, rather than one? We show that using non-linearities that include rectification and local contrast normalization is the single most important ingredient for good accuracy on object recognition benchmarks. We show that two stages of feature extraction yield better accuracy than one. Most surprisingly, we show that a two-stage system with random filters can yield almost 63% recognition rate on Caltech-101, provided that the proper non-linearities and pooling layers are used. Finally, we show that with supervised refinement, the system achieves state-of-the-art performance on NORB dataset (5.6%) and unsupervised pre-training followed by supervised refinement produces good accuracy on Caltech-101 (\> 65%), and the lowest known error rate on the undistorted, unprocessed MNIST dataset (0.53%). (@jarrett2009best)

Samy Jelassi, Stéphane d’Ascoli, Carles Domingo-Enrich, Yuhuai Wu, Yuanzhi Li, and François Charton Length generalization in arithmetic transformers *arXiv preprint arXiv:2306.15400*, 2023. **Abstract:** We examine how transformers cope with two challenges: learning basic integer arithmetic, and generalizing to longer sequences than seen during training. We find that relative position embeddings enable length generalization for simple tasks, such as addition: models trained on $5$-digit numbers can perform $15$-digit sums. However, this method fails for multiplication, and we propose train set priming: adding a few ($10$ to $50$) long sequences to the training set. We show that priming allows models trained on $5$-digit $\\}times$ $3$-digit multiplications to generalize to $35\\}times 3$ examples. We also show that models can be primed for different generalization lengths, and that the priming sample size scales as the logarithm of the training set size. Finally, we discuss potential applications of priming beyond arithmetic. (@jelassi2023length)

Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy The impact of positional encoding on length generalization in transformers *Advances in Neural Information Processing Systems*, 36, 2023. **Abstract:** Length generalization, the ability to generalize from small training context sizes to larger ones, is a critical challenge in the development of Transformer-based language models. Positional encoding (PE) has been identified as a major factor influencing length generalization, but the exact impact of different PE schemes on extrapolation in downstream tasks remains unclear. In this paper, we conduct a systematic empirical study comparing the length generalization performance of decoder-only Transformers with five different position encoding approaches including Absolute Position Embedding (APE), T5’s Relative PE, ALiBi, and Rotary, in addition to Transformers without positional encoding (NoPE). Our evaluation encompasses a battery of reasoning and mathematical tasks. Our findings reveal that the most commonly used positional encoding methods, such as ALiBi, Rotary, and APE, are not well suited for length generalization in downstream tasks. More importantly, NoPE outperforms other explicit positional encoding methods while requiring no additional computation. We theoretically demonstrate that NoPE can represent both absolute and relative PEs, but when trained with SGD, it mostly resembles T5’s relative PE attention patterns. Finally, we find that scratchpad is not always helpful to solve length generalization and its format highly impacts the model’s performance. Overall, our work suggests that explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences. (@kazemnejad2023impact)

Jeonghwan Kim, Giwon Hong, Kyung-min Kim, Junmo Kang, and Sung-Hyon Myaeng Have you seen that number? investigating extrapolation in question answering models In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 7031–7037, 2021. **Abstract:** Numerical reasoning in machine reading comprehension (MRC) has shown drastic improvements over the past few years. While the previous models for numerical MRC are able to interpolate the learned numerical reasoning capabilities, it is not clear whether they can perform just as well on numbers unseen in the training dataset. Our work rigorously tests state-of-the-art models on DROP, a numerical MRC dataset, to see if they can handle passages that contain out-of-range numbers. One of the key findings is that the models fail to extrapolate to unseen numbers. Presenting numbers as digit-by-digit input to the model, we also propose the E-digit number form that alleviates the lack of extrapolation in models and reveals the need to treat numbers differently from regular words in the text. Our work provides a valuable insight into the numerical MRC models and the way to represent number forms in MRC. (@kim2021have)

Diederik P Kingma and Jimmy Ba Adam: A method for stochastic optimization In *International Conference on Learning Representations*, 2015. **Abstract:** We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm. (@kingma2014adam)

Nayoung Lee, Kartik Sreenivasan, Jason D. Lee, Kangwook Lee, and Dimitris Papailiopoulos Teaching arithmetic to small transformers In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=dsUB4bst9S>. **Abstract:** Large language models like GPT-4 exhibit emergent capabilities across general-purpose tasks, such as basic arithmetic, when trained on extensive text data, even though these tasks are not explicitly encoded by the unsupervised, next-token prediction objective. This study investigates how small transformers, trained from random initialization, can efficiently learn arithmetic operations such as addition, multiplication, and elementary functions like square root, using the next-token prediction objective. We first demonstrate that conventional training data is not the most effective for arithmetic learning, and simple formatting changes can significantly improve accuracy. This leads to sharp phase transitions as a function of training data scale, which, in some cases, can be explained through connections to low-rank matrix completion. Building on prior work, we then train on chain-of-thought style data that includes intermediate step results. Even in the complete absence of pretraining, this approach significantly and simultaneously improves accuracy, sample complexity, and convergence speed. We also study the interplay between arithmetic and text data during training and examine the effects of few-shot prompting, pretraining, and model scale. Additionally, we discuss length generalization challenges. Our work highlights the importance of high-quality, instructive data that considers the particular characteristics of the next-word prediction objective for rapidly eliciting arithmetic capabilities. (@lee2024teaching)

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al Solving quantitative reasoning problems with language models *Advances in Neural Information Processing Systems*, 35: 3843–3857, 2022. **Abstract:** Language models have achieved remarkable performance on a wide range of tasks that require natural language understanding. Nevertheless, state-of-the-art models have generally struggled with tasks that require quantitative reasoning, such as solving mathematics, science, and engineering problems at the college level. To help close this gap, we introduce Minerva, a large language model pretrained on general natural language data and further trained on technical content. The model achieves state-of-the-art performance on technical benchmarks without the use of external tools. We also evaluate our model on over two hundred undergraduate-level problems in physics, biology, chemistry, economics, and other sciences that require quantitative reasoning, and find that the model can correctly answer nearly a third of them. (@lewkowycz2022solving)

Shanda Li, Chong You, Guru Guruganesh, Joshua Ainslie, Santiago Ontanon, Manzil Zaheer, Sumit Sanghai, Yiming Yang, Sanjiv Kumar, and Srinadh Bhojanapalli Functional interpolation for relative positions improves long context transformers In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=rR03qFesqk>. **Abstract:** Preventing the performance decay of Transformers on inputs longer than those used for training has been an important challenge in extending the context length of these models. Though the Transformer architecture has fundamentally no limits on the input sequence lengths it can process, the choice of position encoding used during training can limit the performance of these models on longer inputs. We propose a novel functional relative position encoding with progressive interpolation, FIRE, to improve Transformer generalization to longer contexts. We theoretically prove that this can represent some of the popular relative position encodings, such as T5’s RPE, Alibi, and Kerple. We next empirically show that FIRE models have better generalization to longer contexts on both zero-shot language modeling and long text benchmarks. (@li2024functional)

David Lindner, János Kramár, Sebastian Farquhar, Matthew Rahtz, Tom McGrath, and Vladimir Mikulik Tracr: Compiled transformers as a laboratory for interpretability *Advances in Neural Information Processing Systems*, 36, 2023. **Abstract:** We show how to "compile" human-readable programs into standard decoder-only transformer models. Our compiler, Tracr, generates models with known structure. This structure can be used to design experiments. For example, we use it to study "superposition" in transformers that execute multi-step algorithms. Additionally, the known structure of Tracr-compiled models can serve as ground-truth for evaluating interpretability methods. Commonly, because the "programs" learned by transformers are unknown it is unclear whether an interpretation succeeded. We demonstrate our approach by implementing and examining programs including computing token frequencies, sorting, and parenthesis checking. We provide an open-source implementation of Tracr at https://github.com/google-deepmind/tracr. (@lindner2023tracr)

Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, and Tom Goldstein Transformers can do arithmetic with the right embeddings *arXiv preprint arXiv:2405.17399*, 2024. **Abstract:** The poor performance of transformers on arithmetic tasks seems to stem in large part from their inability to keep track of the exact position of each digit inside of a large span of digits. We mend this problem by adding an embedding to each digit that encodes its position relative to the start of the number. In addition to the boost these embeddings provide on their own, we show that this fix enables architectural modifications such as input injection and recurrent layers to improve performance even further. With positions resolved, we can study the logical extrapolation ability of transformers. Can they solve arithmetic problems that are larger and more complex than those in their training data? We find that training on only 20 digit numbers with a single GPU for one day, we can reach state-of-the-art performance, achieving up to 99% accuracy on 100 digit addition problems. Finally, we show that these gains in numeracy also unlock improvements on other multi-step reasoning tasks including sorting and multiplication. (@mcleish2024transformers)

Vinod Nair and Geoffrey E Hinton Rectified linear units improve restricted boltzmann machines In *Proceedings of the 27th international conference on machine learning (ICML-10)*, pages 807–814, 2010. **Abstract:** Restricted Boltzmann machines were developed using binary stochastic hidden units. These can be generalized by replacing each binary unit by an infinite number of copies that all have the same weights but have progressively more negative biases. The learning and inference rules for these Stepped Sigmoid Units are unchanged. They can be approximated efficiently by noisy, rectified linear units. Compared with binary units, these units learn features that are better for object recognition on the NORB dataset and face verification on the Labeled Faces in the Wild dataset. Unlike binary units, rectified linear units preserve information about relative intensities as information travels through multiple layers of feature detectors. (@nair2010rectified)

Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin Investigating the limitations of transformers with simple arithmetic tasks *arXiv preprint arXiv:2102.13019*, 2021. **Abstract:** The ability to perform arithmetic tasks is a remarkable trait of human intelligence and might form a critical component of more complex reasoning tasks. In this work, we investigate if the surface form of a number has any influence on how sequence-to-sequence language models learn simple arithmetic tasks such as addition and subtraction across a wide range of values. We find that how a number is represented in its surface form has a strong influence on the model’s accuracy. In particular, the model fails to learn addition of five-digit numbers when using subwords (e.g., "32"), and it struggles to learn with character-level representations (e.g., "3 2"). By introducing position tokens (e.g., "3 10e1 2"), the model learns to accurately add and subtract numbers up to 60 digits. We conclude that modern pretrained language models can easily learn arithmetic from very few examples, as long as we use the proper surface representation. This result bolsters evidence that subword tokenizers and positional encodings are components in current transformer designs that might need improvement. Moreover, we show that regardless of the number of parameters and training examples, models cannot learn addition rules that are independent of the length of the numbers seen during training. Code to reproduce our experiments is available at https://github.com/castorini/transformers-arithmetic (@nogueira2021investigating)

OpenAI Gpt-4 technical report *arXiv preprint arXiv:2303.08774*, 2023. **Abstract:** We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4’s performance based on models trained with no more than 1/1,000th the compute of GPT-4. (@openai2023gpt)

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al Pytorch: An imperative style, high-performance deep learning library *Advances in neural information processing systems*, 32, 2019. **Abstract:** Deep learning frameworks have often focused on either usability or speed, but not both. PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs. In this paper, we detail the principles that drove the implementation of PyTorch and how they are reflected in its architecture. We emphasize that every aspect of PyTorch is a regular Python program under the full control of its user. We also explain how the careful and pragmatic implementation of the key components of its runtime enables them to work together to achieve compelling performance. We demonstrate the efficiency of individual subsystems, as well as the overall speed of PyTorch on several common benchmarks. (@paszke2019pytorch)

Ofir Press, Noah Smith, and Mike Lewis Train short, test long: Attention with linear biases enables input length extrapolation In *International Conference on Learning Representations*, 2022. URL <https://openreview.net/forum?id=R8sQPpGCv0>. **Abstract:** Since the introduction of the transformer model by Vaswani et al. (2017), a fundamental question has yet to be answered: how does a model achieve extrapolation at inference time for sequences that are longer than it saw during training? We first show that extrapolation can be enabled by simply changing the position representation method, though we find that current methods do not allow for efficient extrapolation. We therefore introduce a simpler and more efficient position method, Attention with Linear Biases (ALiBi). ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance. We show that this method trains a 1.3 billion parameter model on input sequences of length 1024 that extrapolates to input sequences of length 2048, achieving the same perplexity as a sinusoidal position embedding model trained on inputs of length 2048 but training 11% faster and using 11% less memory. ALiBi’s inductive bias towards recency also leads it to outperform multiple strong position methods on the WikiText-103 benchmark. (@press2022train)

Philip Quirke and Fazl Barez Understanding addition in transformers In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=rIx1YXVWZb>. **Abstract:** Understanding the inner workings of machine learning models like Transformers is vital for their safe and ethical use. This paper provides a comprehensive analysis of a one-layer Transformer model trained to perform n-digit integer addition. Our findings suggest that the model dissects the task into parallel streams dedicated to individual digits, employing varied algorithms tailored to different positions within the digits. Furthermore, we identify a rare scenario characterized by high loss, which we explain. By thoroughly elucidating the model’s algorithm, we provide new insights into its functioning. These findings are validated through rigorous testing and mathematical modeling, thereby contributing to the broader fields of model understanding and interpretability. Our approach opens the door for analyzing more complex tasks and multi-layer Transformer models. (@quirke2023understanding)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu Exploring the limits of transfer learning with a unified text-to-text transformer *Journal of machine learning research*, 21 (140): 1–67, 2020. **Abstract:** Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code. (@raffel2020exploring)

Anian Ruoss, Grégoire Delétang, Tim Genewein, Jordi Grau-Moya, Róbert Csordás, Mehdi Bennani, Shane Legg, and Joel Veness Randomized positional encodings boost length generalization of transformers *arXiv preprint arXiv:2305.16843*, 2023. **Abstract:** Transformers have impressive generalization capabilities on tasks with a fixed context length. However, they fail to generalize to sequences of arbitrary length, even for seemingly simple tasks such as duplicating a string. Moreover, simply training on longer sequences is inefficient due to the quadratic computation complexity of the global attention mechanism. In this work, we demonstrate that this failure mode is linked to positional encodings being out-of-distribution for longer sequences (even for relative encodings) and introduce a novel family of positional encodings that can overcome this problem. Concretely, our randomized positional encoding scheme simulates the positions of longer sequences and randomly selects an ordered subset to fit the sequence’s length. Our large-scale empirical evaluation of 6000 models across 15 algorithmic reasoning tasks shows that our method allows Transformers to generalize to sequences of unseen length (increasing test accuracy by 12.0% on average). (@ruoss2023randomized)

Noam Shazeer Glu variants improve transformer *arXiv preprint arXiv:2002.05202*, 2020. **Abstract:** Gated Linear Units (arXiv:1612.08083) consist of the component-wise product of two linear projections, one of which is first passed through a sigmoid function. Variations on GLU are possible, using different nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feed-forward sublayers of the Transformer (arXiv:1706.03762) sequence-to-sequence model, and find that some of them yield quality improvements over the typically-used ReLU or GELU activations. (@shazeer2020glu)

Ruoqi Shen, Sébastien Bubeck, Ronen Eldan, Yin Tat Lee, Yuanzhi Li, and Yi Zhang Positional description matters for transformers arithmetic *arXiv preprint arXiv:2311.14737*, 2023. **Abstract:** Transformers, central to the successes in modern Natural Language Processing, often falter on arithmetic tasks despite their vast capabilities –which paradoxically include remarkable coding abilities. We observe that a crucial challenge is their naive reliance on positional information to solve arithmetic problems with a small number of digits, leading to poor performance on larger numbers. Herein, we delve deeper into the role of positional encoding, and propose several ways to fix the issue, either by modifying the positional encoding directly, or by modifying the representation of the arithmetic task to leverage standard positional encoding differently. We investigate the value of these modifications for three tasks: (i) classical multiplication, (ii) length extrapolation in addition, and (iii) addition in natural language context. For (i) we train a small model on a small dataset (100M parameters and 300k samples) with remarkable aptitude in (direct, no scratchpad) 15 digits multiplication and essentially perfect up to 12 digits, while usual training in this context would give a model failing at 4 digits multiplication. In the experiments on addition, we use a mere 120k samples to demonstrate: for (ii) extrapolation from 10 digits to testing on 12 digits numbers while usual training would have no extrapolation, and for (iii) almost perfect accuracy up to 5 digits while usual training would be correct only up to 3 digits (which is essentially memorization with a training set of 120k samples). (@shen2023positional)

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu Roformer: Enhanced transformer with rotary position embedding *Neurocomputing*, 568: 127063, 2024. **Abstract:** Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer . (@su2024roformer)

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al Lamda: Language models for dialog applications *arXiv preprint arXiv:2201.08239*, 2022. **Abstract:** We present LaMDA: Language Models for Dialog Applications. LaMDA is a family of Transformer-based neural language models specialized for dialog, which have up to 137B parameters and are pre-trained on 1.56T words of public dialog data and web text. While model scaling alone can improve quality, it shows less improvements on safety and factual grounding. We demonstrate that fine-tuning with annotated data and enabling the model to consult external knowledge sources can lead to significant improvements towards the two key challenges of safety and factual grounding. The first challenge, safety, involves ensuring that the model’s responses are consistent with a set of human values, such as preventing harmful suggestions and unfair bias. We quantify safety using a metric based on an illustrative set of human values, and we find that filtering candidate responses using a LaMDA classifier fine-tuned with a small amount of crowdworker-annotated data offers a promising approach to improving model safety. The second challenge, factual grounding, involves enabling the model to consult external knowledge sources, such as an information retrieval system, a language translator, and a calculator. We quantify factuality using a groundedness metric, and we find that our approach enables the model to generate responses grounded in known sources, rather than responses that merely sound plausible. Finally, we explore the use of LaMDA in the domains of education and content recommendations, and analyze their helpfulness and role consistency. (@thoppilan2022lamda)

Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, and Thang Luong Solving olympiad geometry without human demonstrations *Nature*, 627 (8004): E8–E8, 2024. . URL <https://doi.org/10.1038/s41586-024-07115-7>. **Abstract:** Abstract Proving mathematical theorems at the olympiad level represents a notable milestone in human-level automated reasoning 1–4 , owing to their reputed difficulty among the world’s best talents in pre-university mathematics. Current machine-learning approaches, however, are not applicable to most mathematical domains owing to the high cost of translating human proofs into machine-verifiable format. The problem is even worse for geometry because of its unique translation challenges 1,5 , resulting in severe scarcity of training data. We propose AlphaGeometry, a theorem prover for Euclidean plane geometry that sidesteps the need for human demonstrations by synthesizing millions of theorems and proofs across different levels of complexity. AlphaGeometry is a neuro-symbolic system that uses a neural language model, trained from scratch on our large-scale synthetic data, to guide a symbolic deduction engine through infinite branching points in challenging problems. On a test set of 30 latest olympiad-level problems, AlphaGeometry solves 25, outperforming the previous best method that only solves ten problems and approaching the performance of an average International Mathematical Olympiad (IMO) gold medallist. Notably, AlphaGeometry produces human-readable proofs, solves all geometry problems in the IMO 2000 and 2015 under human expert evaluation and discovers a generalized version of a translated IMO theorem in 2004. (@trinh2024solving)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin Attention is all you need *Advances in neural information processing systems*, 30, 2017. **Abstract:** The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. (@vaswani2017attention)

Gail Weiss, Yoav Goldberg, and Eran Yahav Thinking like transformers In *International Conference on Machine Learning*, pages 11080–11090. PMLR, 2021. **Abstract:** What is the computational model behind a Transformer? Where recurrent neural networks have direct parallels in finite state machines, allowing clear discussion and thought around architecture variants or trained models, Transformers have no such familiar parallel. In this paper we aim to change that, proposing a computational model for the transformer-encoder in the form of a programming language. We map the basic components of a transformer-encoder – attention and feed-forward computation – into simple primitives, around which we form a programming language: the Restricted Access Sequence Processing Language (RASP). We show how RASP can be used to program solutions to tasks that could conceivably be learned by a Transformer, and how a Transformer can be trained to mimic a RASP solution. In particular, we provide RASP programs for histograms, sorting, and Dyck-languages. We further use our model to relate their difficulty in terms of the number of required layers and attention heads: analyzing a RASP program implies a maximum number of heads and layers necessary to encode a task in a transformer. Finally, we see how insights gained from our abstraction might be used to explain phenomena seen in recent works. (@weiss2021thinking)

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al Huggingface’s transformers: State-of-the-art natural language processing *arXiv preprint arXiv:1910.03771*, 2019. **Abstract:** Recent progress in natural language processing has been driven by advances in both model architecture and model pretraining. Transformer architectures have facilitated building higher-capacity models and pretraining has made it possible to effectively utilize this capacity for a wide variety of tasks. \\}textit{Transformers} is an open-source library with the goal of opening up these advances to the wider machine learning community. The library consists of carefully engineered state-of-the art Transformer architectures under a unified API. Backing this library is a curated collection of pretrained models made by and available for the community. \\}textit{Transformers} is designed to be extensible by researchers, simple for practitioners, and fast and robust in industrial deployments. The library is available at \\}url{https://github.com/huggingface/transformers}. (@wolf2019huggingface)

Yuhuai Wu, Albert Qiaochu Jiang, Wenda Li, Markus Rabe, Charles Staats, Mateja Jamnik, and Christian Szegedy Autoformalization with large language models *Advances in Neural Information Processing Systems*, 35: 32353–32368, 2022. **Abstract:** Autoformalization is the process of automatically translating from natural language mathematics to formal specifications and proofs. A successful autoformalization system could advance the fields of formal verification, program synthesis, and artificial intelligence. While the long-term goal of autoformalization seemed elusive for a long time, we show large language models provide new prospects towards this goal. We make the surprising observation that LLMs can correctly translate a significant portion ($25.3\\}%$) of mathematical competition problems perfectly to formal specifications in Isabelle/HOL. We demonstrate the usefulness of this process by improving a previously introduced neural theorem prover via training on these autoformalized theorems. Our methodology results in a new state-of-the-art result on the MiniF2F theorem proving benchmark, improving the proof rate from $29.6\\}%$ to $35.2\\}%$. (@wu2022autoformalization)

Changnan Xiao and Bing Liu Conditions for length generalization in learning reasoning skills *arXiv preprint arXiv:2311.16173*, 2023. **Abstract:** Reasoning is a fundamental capability of AI agents. Recently, large language models (LLMs) have shown remarkable abilities to perform reasoning tasks. However, numerous evaluations of the reasoning capabilities of LLMs have also showed some limitations. An outstanding limitation is length generalization, meaning that when trained on reasoning problems of smaller lengths or sizes, the resulting models struggle with problems of larger sizes or lengths. This potentially indicates some theoretical limitations of generalization in learning reasoning skills. These evaluations and their observations motivated us to perform a theoretical study of the length generalization problem. This work focuses on reasoning tasks that can be formulated as Markov dynamic processes (MDPs) and/or directed acyclic graphs (DAGs). It identifies and proves conditions that decide whether the length generalization problem can be solved or not for a reasoning task in a particular representation. Experiments are also conducted to verify the theoretical results. (@xiao2023conditions)

Changnan Xiao and Bing Liu A theory for length generalization in learning to reason *arXiv preprint arXiv:2404.00560*, 2024. **Abstract:** Length generalization (LG) is a challenging problem in learning to reason. It refers to the phenomenon that when trained on reasoning problems of smaller lengths or sizes, the resulting model struggles with problems of larger sizes or lengths. Although LG has been studied by many researchers, the challenge remains. This paper proposes a theoretical study of LG for problems whose reasoning processes can be modeled as DAGs (directed acyclic graphs). The paper first identifies and proves the conditions under which LG can be achieved in learning to reason. It then designs problem representations based on the theory to learn to solve challenging reasoning problems like parity, addition, and multiplication, using a Transformer to achieve perfect LG. (@xiao2024theory)

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu On layer normalization in the transformer architecture In *International Conference on Machine Learning*, pages 10524–10533. PMLR, 2020. **Abstract:** The Transformer is widely used in natural language processing tasks. To train a Transformer however, one usually needs a carefully designed learning rate warm-up stage, which is shown to be crucial to the final performance but will slow down the optimization and bring more hyper-parameter tunings. In this paper, we first study theoretically why the learning rate warm-up stage is essential and show that the location of layer normalization matters. Specifically, we prove with mean field theory that at initialization, for the original-designed Post-LN Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Therefore, using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem. On the other hand, our theory also shows that if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization. This motivates us to remove the warm-up stage for the training of Pre-LN Transformers. We show in our experiments that Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning on a wide range of applications. (@xiong2020layer)

Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar Are transformers universal approximators of sequence-to-sequence functions? In *International Conference on Learning Representations*, 2020. URL <https://openreview.net/forum?id=ByxRM0Ntvr>. **Abstract:** Despite the widespread adoption of Transformer models for NLP tasks, the expressive power of these models is not well-understood. In this paper, we establish that Transformer models are universal approximators of continuous permutation equivariant sequence-to-sequence functions with compact support, which is quite surprising given the amount of shared parameters in these models. Furthermore, using positional encodings, we circumvent the restriction of permutation equivariance, and show that Transformer models can universally approximate arbitrary continuous sequence-to-sequence functions on a compact domain. Interestingly, our proof techniques clearly highlight the different roles of the self-attention and the feed-forward layers in Transformers. In particular, we prove that fixed width self-attention layers can compute contextual mappings of the input sequences, playing a key role in the universal approximation property of Transformers. Based on this insight from our analysis, we consider other simpler alternatives to self-attention layers and empirically evaluate them. (@yun2020transformers)

Chulhee Yun, Yin-Wen Chang, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar O(n) connections are expressive enough: Universal approximability of sparse transformers *Advances in Neural Information Processing Systems*, 33: 13783–13794, 2020. **Abstract:** Recently, Transformer networks have redefined the state of the art in many NLP tasks. However, these models suffer from quadratic computational cost in the input sequence length $n$ to compute pairwise attention in each layer. This has prompted recent research into sparse Transformers that sparsify the connections in the attention layers. While empirically promising for long sequences, fundamental questions remain unanswered: Can sparse Transformers approximate any arbitrary sequence-to-sequence function, similar to their dense counterparts? How does the sparsity pattern and the sparsity level affect their performance? In this paper, we address these questions and provide a unifying framework that captures existing sparse attention models. We propose sufficient conditions under which we prove that a sparse attention model can universally approximate any sequence-to-sequence function. Surprisingly, our results show that sparse Transformers with only $O(n)$ connections per attention layer can approximate the same function class as the dense model with $n\^2$ connections. Lastly, we present experiments comparing different patterns/levels of sparsity on standard NLP tasks. (@yun2020n)

Biao Zhang and Rico Sennrich Root mean square layer normalization *Advances in Neural Information Processing Systems*, 32, 2019. **Abstract:** Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and significantly slows the underlying network, e.g. RNN in particular. In this paper, we hypothesize that re-centering invariance in LayerNorm is dispensable and propose root mean square layer normalization, or RMSNorm. RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efficient than LayerNorm. We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties. Extensive experiments on several tasks using diverse network architectures show that RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%\~64% on different models. Source code is available at https://github.com/bzhangGo/rmsnorm. (@zhang2019root)

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals Understanding deep learning (still) requires rethinking generalization *Communications of the ACM*, 64 (3): 107–115, 2021. **Abstract:** Despite their massive size, successful deep artificial neural networks can exhibit a remarkably small gap between training and test performance. Conventional wisdom attributes small generalization error either to properties of the model family or to the regularization techniques used during training. Through extensive systematic experiments, we show how these traditional approaches fail to explain why large neural networks generalize well in practice. Specifically, our experiments establish that state-of-the-art convolutional networks for image classification trained with stochastic gradient methods easily fit a random labeling of the training data. This phenomenon is qualitatively unaffected by explicit regularization and occurs even if we replace the true images by completely unstructured random noise. We corroborate these experimental findings with a theoretical construction showing that simple depth two neural networks already have perfect finite sample expressivity as soon as the number of parameters exceeds the number of data points as it usually does in practice. We interpret our experimental findings by comparison with traditional models. We supplement this republication with a new section at the end summarizing recent progresses in the field since the original version of this paper. (@zhang2021understanding)

Yi Zhang, Arturs Backurs, Sebastien Bubeck, Ronen Eldan, Suriya Gunasekar, and Tal Wagner Unveiling transformers with LEGO: A synthetic reasoning task 2023. URL <https://openreview.net/forum?id=1jDN-RfQfrb>. **Abstract:** We propose a synthetic reasoning task, LEGO (Learning Equality and Group Operations), that encapsulates the problem of following a chain of reasoning, and we study how the Transformer architectures learn this task. We pay special attention to data effects such as pretraining (on seemingly unrelated NLP tasks) and dataset composition (e.g., differing chain length at training and test time), as well as architectural variants such as weight-tied layers or adding convolutional components. We study how the trained models eventually succeed at the task, and in particular, we manage to understand some of the attention heads as well as how the information flows in the network. In particular, we have identified a novel \\}emph{association} pattern that globally attends only to identical tokens. Based on these observations we propose a hypothesis that here pretraining helps for LEGO tasks due to certain structured attention patterns, and we experimentally verify this hypothesis. We also observe that in some data regime the trained transformer finds “shortcut" solutions to follow the chain of reasoning, which impedes the model’s robustness, and moreover we propose ways to prevent it. Motivated by our findings on structured attention patterns, we propose the LEGO attention module, a drop-in replacement for vanilla attention heads. This architectural change significantly reduces Flops and maintains or even \\}emph{improves} the model’s performance at large-scale pretraining. (@zhang2023unveiling)

Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Joshua M. Susskind, Samy Bengio, and Preetum Nakkiran What algorithms can transformers learn? a study in length generalization In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=AssIuHnmHX>. **Abstract:** Large language models exhibit surprising emergent generalization properties, yet also struggle on many simple reasoning tasks such as arithmetic and parity. This raises the question of if and when Transformer models can learn the true algorithm for solving a task. We study the scope of Transformers’ abilities in the specific setting of length generalization on algorithmic tasks. Here, we propose a unifying framework to understand when and how Transformers can exhibit strong length generalization on a given task. Specifically, we leverage RASP (Weiss et al., 2021) – a programming language designed for the computational model of a Transformer – and introduce the RASP-Generalization Conjecture: Transformers tend to length generalize on a task if the task can be solved by a short RASP program which works for all input lengths. This simple conjecture remarkably captures most known instances of length generalization on algorithmic tasks. Moreover, we leverage our insights to drastically improve generalization performance on traditionally hard tasks (such as parity and addition). On the theoretical side, we give a simple example where the "min-degree-interpolator" model of learning from Abbe et al. (2023) does not correctly predict Transformers’ out-of-distribution behavior, but our conjecture does. Overall, our work provides a novel perspective on the mechanisms of compositional generalization and the algorithmic capabilities of Transformers. (@zhou2024what)

Yongchao Zhou, Uri Alon, Xinyun Chen, Xuezhi Wang, Rishabh Agarwal, and Denny Zhou Transformers can achieve length generalization but not robustly *arXiv preprint arXiv:2402.09371*, 2024. **Abstract:** Length generalization, defined as the ability to extrapolate from shorter training sequences to longer test ones, is a significant challenge for language models. This issue persists even with large-scale Transformers handling relatively straightforward tasks. In this paper, we test the Transformer’s ability of length generalization using the task of addition of two integers. We show that the success of length generalization is intricately linked to the data format and the type of position encoding. Using the right combination of data format and position encodings, we show for the first time that standard Transformers can extrapolate to a sequence length that is 2.5x the input length. Nevertheless, unlike in-distribution generalization, length generalization remains fragile, significantly influenced by factors like random weight initialization and training data order, leading to large variances across different random seeds. (@zhou2024transformers)

</div>

# Omitted Backgrounds [sec:background]

## Next-token Prediction with Decoder-only Transformers

<figure id="fig:next_token_prediction_addition">
<img src="./figures/DecoderOnly_Addition.png"" style="width:50.0%" />
<figcaption>Schematic of solving an integer addition task instance using next-token prediction with a decoder-only Transformers. BOS/EOS mean beginning-/end-of-sequence tokens, respectively. PAD means a padding token, used for matching the sequence lengths in a single minibatch of sequences. Here we assume a basic input format (plain, no zero-padding), which is different from that we used in our experiment.</figcaption>
</figure>

A decoder-only Transformer returns an output sequence of the same length as the input sequence. One difference from a Transformer encoder is that the attention mechanism in a Transformer decoder occurs only in a single forward direction due to the causal attention mask. Due to this causal nature, the Transformer decoder is mostly used for inferring the next token of each token, just based on the information of the current and the previous tokens.

## Related Works

#### Length Generalization in the Addition Tasks.

`\citet{lee2024teaching}`{=latex} observe that reversing the output in the addition task enables the model to learn a simple function. `\citet{shen2023positional}`{=latex} propose “Random Spacing” and “Recursive Scratchpad”, achieving near-perfect generalization from 10-digits to 12-digits addition. `\citet{zhou2024what}`{=latex} introduce “index hints”, position markers placed in front of each token, in both the input and output of addition tasks. Most recently, `\citet{zhou2024transformers}`{=latex} demonstrate a possibility of extrapolation to the length 100 with training length 1–40 in the addition task by combining appropriate input format and advanced PE, yet they also observe that the performances are not robust and highly depend on the random seeds.

#### Length Generalization in the \\(N\times M\\) Multiplication Task (\\(M\\) is fixed).

`\citet{jelassi2023length}`{=latex} investigate \\(N\times3\\) using an encoder-only model and `\citet{duan2023interpolation}`{=latex} study \\(N\times1\\) with an encoder-decoder Transformer architecture. Besides the architectural difference, `\citet{jelassi2023length}`{=latex} fail to observe length generalization with RPE and only achieve it by supplementing a small number of long samples to the training set. Furthermore, although `\citet{duan2023interpolation}`{=latex} provide perfect length generalization results even for test samples \\(10 \times\\) longer than those observed during training, their approach requires a retraining step with hand-crafted bias correction on attention score matrices.

#### Analyzing Length Generalization in Theoretical Perspectives.

An emerging line of research seeks to theoretically address why length generalization is difficult and under what conditions it can be achieved. In `\citet{abbe2023generalization}`{=latex}, the authors demonstrate that various neural network models have an implicit bias towards min-degree interpolators, which may not be ideal for various reasoning tasks. `\citet{xiao2023conditions, xiao2024theory}`{=latex} investigate problems whose reasoning processes can be formulated as directed acyclic graph (DAG) structures, introducing the concept of *maximal input element distance* to identify a sufficient condition for length generalization. Recently, `\citet{ahuja2024provable}`{=latex} formulate the conditions of function classes required to guarantee the length generalization of the empirical risk minimizer function.

#### Comparison with `\citet{mcleish2024transformers}`{=latex}

A very recent concurrent work by `\citet{mcleish2024transformers}`{=latex} proposes a new position embedding method called “Abacus”. From a methodological perspective, Abacus is almost identical to our position coupling except for two main differences: Abacus reverses both the query and the response and does not use padding. From now on, we outline the differences between their work and ours beyond the methodology.

In terms of the model architecture, they use a depth-16 decoder-only Transformer model. They combine their method with looped Transformers and input injection and report an improved performance. In contrast, our main results are obtained with shallower models (up to 6 layers) with standard Transformer architecture of stacked decoder layers.

Besides the addition task, they study multiplication, sorting, and Bitwise OR. On the other hand, we study multiplication, triple addition, copy/reverse, and a 2D task. Specifically, for the multiplication task, their study mainly considers the case where the length of both operands could vary up to 15. In contrast, we focus solely on the \\(N \times 2\\) task, fixing the length of the second operand by 2. While we achieve length generalization up to 90-digit multiplication by training the model on up to 40-digit multiplication, they report near-perfect in-distribution performance but poor length generalization.

Finally and notably, we provide novel theoretical analyses, including (1) the constructive proof that a depth-1 Transformer equipped with position coupling can completely solve the addition task for exponentially long digits and (2) the impossibility of the same model being capable of the addition task. We also present theoretical results for the \\(N \times 2\\) multiplication task.

# More Applications & Experiments of Position Couping [sec:additional_experiments]

## Decimal Integer Addition Task: Scale-up to Length of 500 [subsec:addition_500]

Here, we demonstrate the scalability of our proposed position coupling approach for large lengths of up to 500. Specifically, we again train a depth-1 decoder-only model for the addition task and evaluate the performance for instances with up to 500 digits. The results are shown in  
effig:addition_result_scale_to_500. We notice that at a train length of 160, we achieve excellent length generalization for 500-digit addition. On the other hand, training on sequences of length up to 40 or 80 is insufficient for extreme length generalization. The results demonstrate that position coupling, as an approach, is highly scalable.

<figure id="fig:addition_result_scale_to_500">
<img src="./figures/Addition_EM_median_Large.png"" style="width:90.0%" />
<figcaption>The exact-match accuracies obtained by training a depth-1 transformer on the addition task. We see that while training with sequences of length up to 40 and 80 is insufficient for generalization to large lengths, at training length 160 we achieve strong performance for lengths up to 500. The experimental details can be found in<br />
ef<span>tab:hyperparam_addition_500</span>.</figcaption>
</figure>

## Decimal Integer Addition Task: Operands of Different Lengths [subsec:addition_heatmap]

In the main text, we mainly focus on the test examples of additions where the lengths of both operands are the same. For the sake of completeness, we also report the evaluation results for the cases where the operand lengths can be different (although the zero-padding is applied to ensure the consistency of the data format). See  
effig:addition_heatmap.

<figure id="fig:addition_heatmap">
<p><img src="./figures/eval10.png"" style="width:40.0%" /> <img src="./figures/eval20.png"" style="width:40.0%" /> <img src="./figures/eval30.png"" style="width:40.0%" /> <img src="./figures/eval40.png"" style="width:40.0%" /></p>
<figcaption>Exact-match accuracies (%) on additions with operands of different lengths. Different heatmap corresponds to different trained length of operands (1–10, 1–20, 1–30, and 1–40, expressed with a red box for each). For each heatmap, the <span class="math inline"><em>x</em></span>-axis and the <span class="math inline"><em>y</em></span>-axis are for the length of the first and the second operand, respectively.</figcaption>
</figure>

## Decimal Integer Addition Task: Maximum Exact-Match Accuracies [subsec:addition_max_acc]

A prior work by `\citet{zhou2024transformers}`{=latex} provides a similar analysis on the addition tasks as ours. Combining appropriate input format and advanced PE, they achieve \\(\ge\\)`<!-- -->`{=html}98% EM accuracy for 100-digit additions with a 6-layer 8-head model trained on 1–40. Moreover, they achieve a generalizable length of 45 for a model trained on 1–30, 25 for 1–20, and 10 for 1–10 (no length generalization). One big difference between their analysis and ours is they report the *maximum* accuracy for each testing length over trials, while we report the *medians*. Thus, we choose a bit lower threshold (95%) for generalizability than theirs. For a better comparison with `\citet{zhou2024transformers}`{=latex}, we report the maximum exact-match (EM) accuracies. See  
effig:addition_result_train_len_scaling_max,fig:addition_result_depth_scaling_max.

<figure id="fig:addition_result_train_len_scaling_max">
<img src="./figures/Addition_EM_max_maxpos202.png"" style="width:90.0%" />
<figcaption>Ablation on the trained lengths (1-layer 4-head model trained with position coupling). Here, we report maximum EM accuracies over 8 runs for each tested length.</figcaption>
</figure>

<figure id="fig:addition_result_depth_scaling_max">
<img src="./figures/Addition_MaxEM_depth_ablation.png"" style="width:90.0%" />
<figcaption>Ablation on the number of layers (trained with position coupling). Here, we report maximum EM accuracies over 8 runs for each tested length.</figcaption>
</figure>

## Decimal Integer Addition Task: Comparison & Combination with RoPE [subsec:addition_rope]

We examine further the possibility of combining our position coupling and RoPE `\citep{su2024roformer}`{=latex}.

RoPE incorporates the positional information into the key and the query vectors by rotating them. Suppose a position ID \\(m\\) is assigned to a key vector \\({\bm{k}}\\), for every two consecutive entries of \\({\bm{k}}\\) (i.e., \\((k_{2i-1}, k_{2i})\\)), we rotate by a predefined angle \\(\theta_i\\) multiplied by \\(m\\): \\[\begin{aligned}
    \begin{pmatrix}
        k_{2i-1} \\ k_{2i}
    \end{pmatrix}
    \mapsto
    \begin{pmatrix}
        k_{2i-1} \cos m\theta_i - k_{2i} \sin m\theta_i \\
        k_{2i-1} \sin m\theta_i + k_{2i} \cos m\theta_i
    \end{pmatrix}.
\end{aligned}\\] We also apply a similar rotation to the query vectors (say, \\({\bm{q}}\\) with position ID \\(n\\)). As a result, the attention score for this key-query pair becomes a function of \\({\bm{k}}\\), \\({\bm{q}}\\), and the relative distance \\(n-m\\).

Unlike the original implementation of RoPE, we apply rotations based on the re-assigned position IDs according to our position coupling method. We incorporate this RoPE variant into every attention head of every layer. One more difference in implementation is that, during training, we randomly sampled an integer scaler \\(\ell \in \{1,2,3,4,5\}\\) and multiplied it by the rotation angle. By such random re-scaling of rotation angles, we expect the model could handle unseen large rotation angles at test time.

The result on 12-layer models is showcased in  
effig:addition_rope (the orange line). Unlike vanilla RoPE (the blue line), which fails immediately outside the trained lengths (1–40), our combination of RoPE and position coupling achieves a much better generalization up to operand lengths 100.

<figure id="fig:addition_rope">
<img src="./figures/Addition_RoPE_MedianEM_12layer.png"" style="width:90.0%" />
<figcaption>RoPE-based position coupling, 12-layer model.</figcaption>
</figure>

## Position Coupling for \\(N\times2\\) Multiplication Tasks [subsec:Nx2_appendix]

We present an example of the position coupling method for \\(N\times2\\) (\\(N\\)-digit by 2-digit) multiplication, which is omitted from the main text. See  
efsubsec:Nx2 for the experimental results.

<figure id="fig:coupling_Nx2multiplication">
<img src="./figures/PositionCouplingForNx2Multiplication.png"" style="width:60.0%" />
<figcaption>Illustration of position coupling for <span class="math inline"><em>N</em> × 2</span> multiplication task.</figcaption>
</figure>

## Addition Task with Multiple Summands [subsec:tripleaddition]

The position coupling scheme for the vanilla addition task (with two operands) can naturally extend to the addition task with multiple summands: assign position IDs in ascending order from most significant digits to least significant digits for every operand and the response. Here, we focus on the addition of three summands.

<figure id="fig:experiment_triple_addition">
<img src="./figures/TripleAddition_EM_median_maxpos102.png"" />
<figcaption>Exact-match accuracy (median over 4 runs) for triple addition task, trained on sequences of length 1-40 with position coupling, NoPE, and random-start APE. For further experiment details, refer to<br />
ef<span>tab:hyperparam_triple_addition</span>.</figcaption>
</figure>

#### Experiments.

We train on sequences with operands of 1–40 digits. Our choice of `max_pos` is 102, so we test the operands of up to 100 digits. We investigate the performance of 3 different architectures, each with a different depth. The experimental results are described in  
effig:experiment_triple_addition. 1-layer models keep their generalization capability until 100 digits, whereas the 3-layer models exhibit great stability across random seeds and achieve the highest generalizable length of 90.

Lastly, we note that the result of  
efthm:addition can be extended to addition tasks with multiple summands with slight adjustments to the feed-forward layer in the construction.

## Position Coupling for Copy/Reverse Tasks [subsec:copyreverse]

#### Data Format & Position Coupling.

Each token of the query sequence is a digit (10 distinct characters). We couple the positions in the query and the response by their correspondence. Note that the position ID assigned to the equal token is different for the two tasks because as per our design principle (Sec 3.1), the equal token is grouped to the response tokens and position IDs have to be consecutive numbers within each group.

<figure id="fig:coupling_copy_reverse">
<p><img src="./figures/PositionCouplingForCopy.png"" style="width:40.0%" />     <img src="./figures/PositionCouplingForReverse.png"" style="width:40.0%" /></p>
<figcaption>Illustration of position coupling for copy/reverse tasks.</figcaption>
</figure>

#### Experiments.

We compare position coupling with NoPE and random-start APE. We train a model on lengths 1–40 and evaluate its performance on lengths from 5 to 300, at intervals of 5. While a 1-layer 4-head model is used for the position coupling, we observe that the same architecture fails to memorize training samples for both NoPE and random-start APE. Therefore, we use a 6-layer 8-head model for the latter cases as it is commonly used in the literature `\citep{zhou2024what}`{=latex}.

<figure id="fig:copy_result_method">
<figure>
<img src="./figures/Copy_EM_Method_Comparison.png"" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure>
<img src="./figures/Reverse_EM_Method_Comparison.png"" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figcaption>Exact-match accuracy (median over 4 runs) for (a) copying task and (b) reversing task, trained on sequences of length 1–40 with position coupling, NoPE, and random-start APE. For further experiment details, refer to the<br />
ef<span>tab:hyperparam_copyreverse_coupling</span>.</figcaption>
</figure>

The experimental results are described in  
effig:copy_result_method. For both copy and reverse tasks, position coupling exhibits near-perfect accuracy across the entire test length (7.5\\(\times\\) for the trained length). In contrast, NoPE and random-start APE immediately fail to length-generalize.

## Position Coupling for Minesweeper Generator Tasks [subsec:minesweeper_appendix]

Here, we present the extra experimental results for training the minesweeper generator task with position coupling. Specifically, we compare the performance of two configurations: one where the model shares the same positional embedding layer for both position coupling modules, and another where the model uses separate positional embedding layers for each position coupling module.

The results are described in  
effig:minesweeper_embedding_comparison. When sharing the same positional embedding layer, position coupling achieves over 98% accuracy on a 12\\(\times\\)`<!-- -->`{=html}12 board, and maintains near 90% accuracy on a 14\\(\times\\)`<!-- -->`{=html}14 board. However, with distinct positional embedding layers, position coupling only successfully generalizes to a 10\\(\times\\)`<!-- -->`{=html}10 board. We currently do not have a clear explanation for why the former method exhibits significantly better performance than the latter one. We leave the investigation and explanation of this phenomenon for future work.

<figure id="fig:minesweeper_embedding_comparison">
<img src="./figures/Minesweeper_Embedding_Method_Comparison.png"" style="width:90.0%" />
<figcaption>Exact-match accuracy (median over 4 runs) for minesweeper generator task, trained on sequences of length (5–9)<span class="math inline">×</span>(5–9) with position coupling. For further experiment details, see<br />
ef<span>tab:hyperparam_minesweeper_coupling</span>.</figcaption>
</figure>

# Experiment Details and Hyperparameters [sec:experiment_detail_addition]

Position coupling can be easily implemented on top of usual libraries of training transformer models like HuggingFace `\citep{wolf2019huggingface}`{=latex} and Flaxformer[^5] since these libraries support an arbitrary array of position IDs (in the case of using APE). All we need is to build up a short routine implementing the assigning rule of position IDs when establishing the dataset and data loaders. To compare with NoPE, we use the code base provided by `\citet{kazemnejad2023impact}`{=latex} for most of the experiments.[^6] It contains a custom implementation of decoder-only T5 `\citep{raffel2020exploring}`{=latex} established on top of PyTorch `\citep{paszke2019pytorch}`{=latex} and Huggingface, including several PE methods. We additionally implement a custom RMSNorm module `\citep{zhang2019root}`{=latex} and various positioning schemes of normalization layers (e.g., PreNorm `\citep{xiong2020layer}`{=latex}, PostNorm `\citep{vaswani2017attention}`{=latex}, and their combination), to follow the implementation details of `\citet{zhou2024transformers}`{=latex}.

<div id="tab:hyperparam_addition" markdown="1">

| **Hyperparameter** | **Value** |
|:---|:---|
| Architecture | Decoder-only Transformer |
| Number of Layers | 1 |
| Number of Attention Heads | 4 |
| Embedding Dimension | 512 |
| Dimension per Head | 128 |
| Hidden Width of Feed-forward Layer | 2048 |
| Activation Function of Feed-forward Layer | GEGLU `\citep{shazeer2020glu}`{=latex} |
| Normalization Layer | RMSNorm `\citep{zhang2019root}`{=latex} |
| Normalization Layer Position | PreNorm and PostNorm |
| Training Steps | 50,000 |
| Batch Size | 1,000 |
| Optimizer | Adam `\citep{kingma2014adam}`{=latex} |
| Learning Rate (LR) | 0.0001 |
| LR Warm-up | Linear (From 0 to LR), 1% of total steps |
| LR Cool-down | Cosine Decay (From LR to 0.1LR) |
| Maximum Position ID (`max_pos`) | 202 |
| Training Dataset Size | 1,000,000 |
| Evaluation Dataset Size | 100,000 |
| Device | NVIDIA RTX A6000 48GB |
| Training Time | \\(\le\\) 10 hours |

Hyperparameter summary for decimal integer addition task: comparison between trained lengths (  
effig:addition_result_train_len_scaling,fig:addition_result_train_len_scaling_max).

</div>

<div id="tab:hyperparam_addition_depth_scaling" markdown="1">

| **Hyperparameter** | **Value** |
|:---|:---|
| Architecture | Decoder-only Transformer |
| Number of Layers | 1-6 |
| Number of Attention Heads | 8 |
| Embedding Dimension | 1024 |
| Dimension per Head | 128 |
| Hidden Width of Feed-forward Layer | 2048 |
| Activation Function of Feed-forward Layer | GEGLU |
| Normalization Layer | RMSNorm |
| Normalization Layer Position | PreNorm and PostNorm |
| Trained Lengths of Operands | 1–30 |
| Training Steps | 50,000 |
| Batch Size | 1000 |
| Optimizer | Adam |
| Learning Rate (LR) | 0.00003 |
| LR Warm-up | Linear (From 0 to LR), 1% of total steps |
| LR Cool-down | Cosine Decay (From LR to 0.1LR) |
| Maximum Position ID (`max_pos`) | 202 |
| Training Dataset Size | 1,000,000 |
| Evaluation Dataset Size | 100,000 |
| Device | NVIDIA RTX A6000 48GB |
| Training Time | \\(\le\\) 10 hours |

Hyperparameter summary for decimal integer addition task: comparison between the number of layers (  
effig:addition_result_depth_scaling,fig:addition_result_depth_scaling_max).

</div>

<div id="tab:hyperparam_addition_500" markdown="1">

| **Hyperparameter** | **Value** |
|:---|:---|
| Architecture | Decoder-only Transformer |
| Number of Layers | 1 |
| Number of Attention Heads | 2 |
| Embedding Dimension | 512 |
| Dimension per Head | 256 |
| Hidden Width of Feed-forward Layer | 2048 |
| Activation Function of Feed-forward Layer | GEGLU |
| Normalization Layer | LayerNorm `\citep{ba2016layer}`{=latex} |
| Normalization Layer Position | PostNorm |
| Training Steps | 1,000,000 |
| Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate (LR) | 0.0001 |
| LR Warm-up | Linear (From 0 to LR), 500 steps |
| LR Cool-down | Cosine Decay (From LR to 0.0) |
| Maximum Position ID (`max_pos`) | 1003 |
| Training Dataset Size | 1,000,000 |
| Evaluation Dataset Size | 100,000 |
| Device | 64 TPU V4 Chips |
| Training Time | \\(\le\\) 4 hours |

Hyperparameter summary for decimal integer addition task: generalization up to length 500 (  
effig:addition_result_scale_to_500).

</div>

<div id="tab:hyperparam_addition_attention_pattern" markdown="1">

| **Hyperparameter** | **Value** |
|:---|:---|
| Architecture | Decoder-only Transformer |
| Number of Layers | 1 |
| Number of Attention Heads | 2 |
| Embedding Dimension | 512 |
| Dimension per Head | 256 |
| Hidden Width of Feed-forward Layer | 2048 |
| Activation Function of Feed-forward Layer | GEGLU |
| Normalization Layer | RMSNorm |
| Normalization Layer Position | PreNorm and PostNorm |
| Trained Lengths of Operands | 1–5 |
| Training Steps | 50,000 |
| Batch Size | 100 |
| Optimizer | Adam |
| Learning Rate (LR) | 0.00005 |
| LR Warm-up | Linear (From 0 to LR), 1% of total steps |
| LR Cool-down | Cosine Decay (From LR to 0.1LR) |
| Maximum Position ID (`max_pos`) | 17 |
| Training Dataset Size | 100,000 |
| Device | NVIDIA RTX A6000 48GB |
| Training Time | \\(\le\\) 6 hours |

Hyperparameter summary for decimal integer addition task: extracting attention patterns (  
effig:attention_patterns).

</div>

<div id="tab:hyperparam_Nx2multiplication" markdown="1">

| **Hyperparameter** | **Value** |
|:---|:---|
| Architecture | Decoder-only Transformer |
| Number of Layers | 1-4 (Ours), 3 (NoPE & Random-start APE) |
| Number of Attention Heads | 8 |
| Embedding Dimension | 512 |
| Dimension per Head | 64 |
| Hidden Width of Feed-forward Layer | 2048 |
| Activation Function of Feed-forward Layer | GEGLU |
| Normalization Layer | RMSNorm |
| Normalization Layer Position | PreNorm and PostNorm |
| Trained Lengths of Operands | 1–40 |
| Training Steps | 50,000 |
| Batch Size | 200 (Ours), 800 (NoPE & Random-start APE) |
| Optimizer | Adam |
| Learning Rate (LR) | 0.0001 |
| LR Warm-up | Linear (From 0 to LR), 1% of total steps |
| LR Cool-down | Cosine Decay (From LR to 0.1LR) |
| Maximum Position ID (`max_pos`) | 203 (Ours), 1023 (Random-start APE) |
| Training Dataset Size | 50,000 (Ours), 500,000 (Others) |
| Evaluation Dataset Size | 100,000 |
| Device | NVIDIA RTX A6000 48GB |
| Training Time | \\(\le\\) 8 hours |

Hyperparameter summary for \\(N\times2\\) multiplication task (  
effig:experiment_Nx2multiplication).

</div>

<div id="tab:hyperparam_triple_addition" markdown="1">

| **Hyperparameter** | **Value** |
|:---|:---|
| Architecture | Decoder-only Transformer |
| Number of Layers | 1-3 (Ours), 6 (NoPE & Random-start APE) |
| Number of Attention Heads | 4 |
| Embedding Dimension | 512 |
| Dimension per Head | 128 |
| Hidden Width of Feed-forward Layer | 2048 |
| Activation Function of Feed-forward Layer | GEGLU |
| Normalization Layer | RMSNorm |
| Normalization Layer Position | PreNorm and PostNorm |
| Trained Lengths of Operands | 1–40 |
| Training Steps | 50,000 |
| Batch Size | 1000 (Ours), 800 (Others) |
| Optimizer | Adam |
| Learning Rate (LR) | 0.0001 |
| LR Warm-up | Linear (From 0 to LR), 1% of total steps |
| LR Cool-down | Cosine Decay (From LR to 0.1LR) |
| Maximum Position ID (`max_pos`) | 102 (Ours), 1023 (Random-start APE) |
| Training Dataset Size | 1,000,000 |
| Evaluation Dataset Size | 100,000 |
| Device | NVIDIA RTX A6000 48GB |
| Training Time | \\(\le\\) 12 hours |

Hyperparameter summary for addition task with three summands (  
effig:experiment_triple_addition).

</div>

<div id="tab:hyperparam_copyreverse_coupling" markdown="1">

| **Hyperparameter** | **Value** |
|:---|:---|
| Architecture | Decoder-only Transformer |
| Number of Layers | 1 (Ours), 6 (NoPE & Random-start APE) |
| Number of Attention Heads | 4 (Ours), 8 (NoPE & Random-start APE) |
| Embedding Dimension | 512 |
| Dimension per Head | 128 |
| Hidden Width of Feed-forward Layer | 2048 |
| Activation Function of Feed-forward Layer | GEGLU |
| Normalization Layer | RMSNorm |
| Normalization Layer Position | PreNorm and PostNorm |
| Trained Lengths of Query | 1–40 |
| Training Steps | 50,000 |
| Batch Size | 1000 (Ours), 500 (Others) |
| Optimizer | Adam |
| Learning Rate (LR) | 0.0001 |
| LR Warm-up | Linear (From 0 to LR), 1% of total steps |
| LR Cool-down | Cosine Decay (From LR to 0.1LR) |
| Maximum Position ID (`max_pos`) | 301 (Ours), 601 (Random-start APE) |
| Training Dataset Size | 1,000,000 |
| Evaluation Dataset Size | 100,000 |
| Device | NVIDIA RTX A6000 48GB |
| Training Time | \\(\le\\) 8 hours |

Hyperparameter summary for copy/reverse task (  
effig:copy_result_method).

</div>

<div id="tab:hyperparam_minesweeper_coupling" markdown="1">

| **Hyperparameter** | **Value** |
|:---|:---|
| Architecture | Decoder-only Transformer |
| Number of Layers | 4 (Ours), 6 (NoPE) |
| Number of Attention Heads | 8 |
| Embedding Dimension | 512 |
| Dimension per Head | 64 |
| Hidden Width of Feed-forward Layer | 2048 |
| Activation Function of Feed-forward Layer | GEGLU |
| Normalization Layer | RMSNorm |
| Normalization Layer Position | PreNorm and PostNorm |
| Trained Lengths of Query | (5–9) \\(\times\\) (5–9) |
| Training Steps | 100,000 |
| Batch Size | 200 |
| Optimizer | Adam |
| Learning Rate (LR) | 0.0001 (Ours), 0.0002 (NoPE) |
| LR Warm-up | Linear (From 0 to LR), 1% of total steps |
| LR Cool-down | Cosine Decay (From LR to 0.1LR) |
| Maximum Position ID (`max_pos`) | 15 |
| Training Dataset Size | 100,000 |
| Evaluation Dataset Size | 100,000 |
| Device | NVIDIA RTX A6000 48GB |
| Training Time | \\(\le\\) 30 hours |

Hyperparameter summary for minesweeper generator task (  
effig:minesweeper_method_comparison,fig:minesweeper_embedding_comparison).

</div>

# Decoder-only Transformer Architecture [sec:Transformer_architecture]

Here we detail the architecture of a depth-\\(L\\), \\(H\\)-head decoder-only Transformer `\citep{vaswani2017attention}`{=latex}. For a simple presentation, we ignore the normalization layers, as in `\citet{yun2020transformers}`{=latex}.

Let \\({\mathcal{V}}\\) be the (ordered) vocabulary, a set of all tokens. Given an input sequence \\({\mathcal{I}}\in {\mathcal{V}}^N\\) and its length \\(N\\), the *encoding function* \\({\tt Enc}: {\mathcal{V}}^N \rightarrow \mathbb{R}^{d \times N}\\) maps it to \\[\begin{aligned}
    {\bm{X}}^{(0)} := {\tt Enc}({\mathcal{I}}).
\end{aligned}\\] It is a sum of the token embedding and the position embedding.

Next, there are \\(L\\) Transformer blocks that sequentially transform this input. We denote by \\({\tt Tf}_l: \mathbb{R}^{d \times N} \rightarrow \mathbb{R}^{d \times N}\\) the operation of the \\(l\\)-th block (\\(l\in [L]\\)), so that \\[\begin{aligned}
    {\bm{X}}^{(l)} := {\tt Tf}_l \left({\bm{X}}^{(l-1)}\right).
\end{aligned}\\] The block \\({\tt Tf}_l\\) consists of a (causal) attention layer \\({\tt Att}_l: \mathbb{R}^{d \times N} \rightarrow \mathbb{R}^{d \times N}\\) and a (token-wise) feed-forward layer \\({\tt FF}_l: \mathbb{R}^{d \times N} \rightarrow \mathbb{R}^{d \times N}\\), each of which contains a residual connection: \\[\begin{aligned}
    {\tt Tf}_l := ({\tt id}+ {\tt FF}_l) \circ ({\tt id}+ {\tt Att}_l),
\end{aligned}\\] where we denote by \\({\tt id}: \mathbb{R}^{d \times N} \rightarrow \mathbb{R}^{d \times N}\\) an identity map.

Each attention layer \\({\tt Att}_l\\) consists of \\(H\\) attention heads. Its \\(h\\)-th head (\\(h\in [H]\\)) has matrices \\({\bm{Q}}^{(l)}_h, {\bm{K}}^{(l)}_h \in \mathbb{R}^{d^{(l)}_{QK,h}\times d}\\), \\({\bm{V}}^{(l)}_h \in \mathbb{R}^{d^{(l)}_{V,h}\times d}\\) and \\({\bm{U}}^{(l)}_h \in \mathbb{R}^{d\times d^{(l)}_{V,h}}\\) as its parameters. [^7] With these matrices, borrowing the notation from `\citet{yun2020transformers}`{=latex}, the attention layer with an input \\({\bm{X}}\in\mathbb{R}^{d \times N}\\) can be written as \\[\begin{aligned}
    {\tt Att}_l ({\bm{X}}) := \sum_{h=1}^H {\bm{U}}^{(l)}_h {\bm{V}}^{(l)}_h {\bm{X}}\cdot{\tt softmax}\left(({\bm{K}}^{(l)}_h {\bm{X}})^\top {\bm{Q}}^{(l)}_h {\bm{X}}\right).
\end{aligned}\\] Here the \\({\tt softmax}\\) operator takes a square matrix \\({\bm{M}}\in \mathbb{R}^{N\times N}\\) and outputs an \\(N\times N\\) upper-triangular column-stochastic [^8] matrix \\[\begin{aligned}
    \left[{\tt softmax}({\bm{M}})\right]_{ij} = \frac{e^{{\bm{M}}_{ij}}}{\sum_{1\le i'\le j} e^{{\bm{M}}_{i'j}}} \mathbbm{1}_{\left\{i\le j\right\}},
\end{aligned}\\] where \\(\mathbbm{1}_{\left\{{\mathcal{E}}\right\}}\\) is an indicator function for a predicate \\({\mathcal{E}}\\): it equals 1 if \\({\mathcal{E}}\\) is true and 0 otherwise. Note that the upper triangularity captures the auto-regressive behavior of the causal attention. For the sake of convenience, we denote by \\({\bm{Y}}^{(l)} := {\bm{X}}^{(l-1)} + {\tt Att}_l({\bm{X}}^{(l-1)}) \in \mathbb{R}^{d \times N}\\) which is a consequence of residual connection right after the attention layer.

Each feed-forward layer \\({\tt FF}_l\\) is a two-layer perceptron having \\({\bm{W}}^{(l)}_1 \in \mathbb{R}^{d_F \times d}\\), \\({\bm{b}}^{(l)}_1 \in \mathbb{R}^{d_F}\\), \\({\bm{W}}^{(l)}_2 \in \mathbb{R}^{d \times d_F}\\), \\({\bm{b}}^{(l)}_2 \in \mathbb{R}^d\\) as its parameters. It applies the following map to each column \\({\bm{y}}\\) of an input \\({\bm{Y}}\\): \\[\begin{aligned}
    {\bm{y}}\mapsto {\bm{W}}^{(l)}_2 \phi({\bm{W}}^{(l)}_1 {\bm{y}}+ {\bm{b}}^{(l)}_1) + {\bm{b}}^{(l)}_2,
\end{aligned}\\] where \\(\phi\\) is a component-wise activation function. That is, the feed-forward layer is defined as \\[\begin{aligned}
    {\tt FF}_l ({\bm{Y}}) := {\bm{W}}^{(l)}_2 \phi({\bm{W}}^{(l)}_1 {\bm{Y}}+ {\bm{b}}^{(l)}_1 {\bm{1}}^\top_{d_F}) + {\bm{b}}^{(l)}_2 {\bm{1}}^\top_{d},
\end{aligned}\\] where \\({\bm{1}}_d\\) is the \\(d\\)-dimensional vectors filled with 1’s. Here we mainly use the ReLU operation \\(\phi(\cdot)=\max\left\{\cdot, 0\right\}\\) `\citep{jarrett2009best,nair2010rectified}`{=latex}, but there are many other popular choices such as GeLU `\citep{hendrycks2016gaussian}`{=latex}, GLU `\citep{dauphin2017language}`{=latex}, ReGLU, and GEGLU `\citep{shazeer2020glu}`{=latex}.

The final component of the Transformer model is the decoding function \\({\tt Dec}: \mathbb{R}^{d\times N} \rightarrow {\mathcal{V}}^N\\), which is composed of a linear readout and a (token-wise) arg-max operation. Here, the linear readout is simply a linear layer having \\({\bm{W}}_{\rm out} \in \mathbb{R}^{|{\mathcal{V}}|\times d}\\) as its parameter. The decoding function produces the output sequence \\[\begin{aligned}
    {\mathcal{O}}:= {\tt Dec}({\bm{X}}^{(L)}) \in {\mathcal{V}}^N.
\end{aligned}\\]

# Formal Construction of Addition Transformer with Position Coupling [sec:construction_addition]

Here we show how to implement the addition by employing a single-layer two-head decoder-only Transformer equipped with position coupling. We restate the theorem for the sake of readability.

#### Organization of the Proof.

A whole section is dedicated to prove  
efthm:addition.

- We start with the notation (  
  efsubsec:construction_addition_notation).

- We review and formalize the format of the input sequence (zero-padding, reversed format, and wrapping with BOS/EOS) (  
  efsubsec:construction_addition_input_sequence).

- We define the encoding function \\({\tt Enc}\\) with a table of a concrete example (  
  efsubsec:construction_addition_encoding), where \\({\tt Enc}\\) maps an input sequence of length \\(N\\) to a \\(d\times N\\) encoding matrix \\({\bm{X}}^{(0)}\\).

- We devote a lot of pages to the detailed construction of the parameters of a causal attention layer \\({\tt Att}_1\\) to generate desired attention patterns (  
  efsubsec:construction_addition_attention). The attention layer has two attention heads playing distinct roles: (1) preparing for a sum without considering carries; and (2) preparing for the carry prediction & EOS detection.

- We provide a construction of a token-wise feed-forward neural network \\({\tt FF}_1\\) which is a two-layer ReLU network (  
  efsubsec:construction_addition_feedforward). It consists of two subnetworks playing different roles: (1) producing one-hot vectors, each of which indicates a digit of the sum (response); and (2) binary values indicating whether the position is the end of the sequence.

- We conclude the proof by defining the decoding function \\({\tt Dec}\\) which performs the linear readout and the arg-max operation to generate the output sequence (  
  efsubsec:construction_addition_decoding).

We illustrate the roadmap of the proof in  
effig:proof_sketch.

<figure id="fig:proof_sketch">
<img src="./figures/Proof_Sketch.png"" />
<figcaption>Roadmap to the formal construction of addition Transformer with position coupling.</figcaption>
</figure>

## Notation [subsec:construction_addition_notation]

For the architecture of the decoder-only Transformer, we follow the notation introduced in  
efsec:Transformer_architecture.

Let \\({\bm{e}}^d_i\\) denote the \\(i\\)-th standard basis vector of \\(\mathbb{R}^d\\). For example, \\({\bm{e}}^3_1 = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}^\top.\\) Let \\({\bm{I}}_m\\) be the \\(m\times m\\) identity matrix. Let \\({\bm{0}}_p\\) and \\({\bm{1}}_p\\) denote the \\(p\\)-dimensional vectors filled with 0’s and 1’s, respectively. Similarly, let \\({\bm{0}}_{m\times n}\\) denote the \\(m\times n\\) zero matrix. For a positive integer \\(n\\), we frequently use the set \\([n]:=\{1,...,n\}\\). For any matrix \\({\bm{A}}\\), denote the \\(i\\)-th row and \\(j\\)-th column of \\({\bm{A}}\\) by \\({\bm{A}}_{i \bullet}\\) and \\({\bm{A}}_{\bullet j}\\), respectively. Given two non-negative integers \\(a\\) and \\(b\\), let \\(\ell(a,b)\\) be the length of a longer one between \\(a\\) and \\(b\\). For example, \\(\ell(12, 3456)=4\\).

Consider an ordered vocabulary \\({\mathcal{V}}=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, +, =, \$)\\). We include a special token ‘\\(\$\\)’ that plays the role of both the beginning-of-sequence (BOS) token and the end-of-sequence (EOS) token. [^9] We denote \\({\mathcal{V}}_k\\) as \\(k\\)-th element of \\({\mathcal{V}}\\). For instance, \\({\mathcal{V}}_4=3\\) and \\({\mathcal{V}}_{13}=\$\\). Lastly, since we employ only one Transformer block, we omit the superscripts \\((l)\\) in the parameter matrices/vectors and the size of dimensions \\(d^{(l)}_{QK,h}\\) and \\(d^{(l)}_{V,h}\\).

## Input Sequence [subsec:construction_addition_input_sequence]

We seek to perform an addition \\(a+b=c\\) using next-token prediction. To this end, we want to transform it into an input sequence \\({\mathcal{I}}=\overline{\$A+B=C}\\) of an appropriate format. Note that the EOS token is the last token that needs to be predicted, so we exclude EOS in the input sequence. Let \\(\ell:=\ell(a,b)\\).

We first zero-pad the shorter one between \\(a\\) and \\(b\\) to match the length of the part \\(A\\) and part \\(B\\) as \\(\ell\\). Sometimes, the sum \\(c\\) might be longer than \\(a\\) or \\(b\\) due to a carry. To make the length of the part \\(C\\) consistent, we also put a zero-pad in front of \\(c\\) to set its length as \\(\ell+1\\). Also, to ease calculating the addition with next-token prediction, we reverse the sum \\(c\\) to make the part \\(C\\). For example, if we have a sum \\(3812+98=3910\\), we use \\(\overline{\$3812+\textcolor{Firebrick3}{00}98=\textcolor{DodgerBlue3}{0193}\textcolor{Firebrick3}{0}}\\) as an input sequence; if a sum \\(98+9907=10005\\) is given, we use \\(\overline{\$\textcolor{Firebrick3}{00}98+9907=\textcolor{DodgerBlue3}{50001}}\\) as an input sequence. The red digits are zero-paddings, and the blue digits are the reversed sum.

To recap, the input sequence \\({\mathcal{I}}=\overline{\sigma_1\sigma_2\ldots\sigma_{N}}\in {\mathcal{V}}^N\\) of length \\(N=3\ell+4\\) consists of six parts:

1.  the BOS token \\(\sigma_1 = \text{`$\$$'}\\)

2.  the first operand \\(A=\overline{\sigma_2\ldots\sigma_{\ell+1}}\\) where \\(\sigma_i \in \{0, \ldots, 9\}\\);

3.  the addition symbol \\(\sigma_{\ell+2}=\\) ‘\\(+\\)’;

4.  the second operand \\(B=\overline{\sigma_{\ell+3}\ldots\sigma_{2\ell+2}}\\) where \\(\sigma_i \in \{0, \ldots, 9\}\\);

5.  the equality symbol \\(\sigma_{2\ell+3}=\\) ‘\\(=\\)’;

6.  the (reversed) sum \\(C=\overline{\sigma_{2\ell+4}\ldots\sigma_{3\ell+4}}\\) where \\(\sigma_i \in \{0, \ldots, 9\}\\).

Note that the part \\(C\\) might be incomplete (i.e., \\(N<3\ell+4\\)) at the inference time; we infer the digits of the part \\(C\\) one by one using next-token prediction. Throughout this section on a formal construction, however, we only consider the train time setup in which we infer all the digits of the part \\(C\\) at once using *simultaneous* next-token prediction in a single forward pass. Precisely, we want to use an input sequence \\({\mathcal{I}}=\overline{\sigma_1\ldots\sigma_{N}}\\) to produce an output sequence \\({\mathcal{O}}=\overline{\sigma'_1\ldots\sigma'_{N}}\\) where \\(\overline{\sigma'_{2\ell+3}\ldots\sigma'_{N-1}} = C = \overline{\sigma_{2\ell+4}\ldots\sigma_{N}}\\) and \\(\sigma'_{N}=\\) ‘\\(\$\\)’ (EOS).

## Encoding Function [subsec:construction_addition_encoding]

We plan to produce an input encoding, given an input sequence \\({\mathcal{I}}\\) designed as above. The encoding matrix \\({\bm{X}}^{(0)}\\) is of size \\(d\times N\\): each column represents an embedding vector for a token, while each row represents a particular *named* dimension. What we mean by *named* dimension is that we give a name to each dimension for a clear description of our formal construction.

We construct an input encoding by *concatenating* the token embedding and the position embedding, which can be viewed as a *sum* of two different embedding matrices of the same size.

<div id="tab:init_emb" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 6 | 5 | 3 | 0 | 0 | 4 | 9 | 0 | 2 | 0 | 7 | 0 |
| 2: <span class="smallcaps">is_bos</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">full_ones</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 4: <span class="smallcaps">pre_sum</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">pre_carry</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">pre_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7–16: <span class="smallcaps">sum</span> | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) |
| 17: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18–\\((P+17)\\): <span class="smallcaps">pos_1</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) |
| \\((P+18)\\)-\\((2P+17)\\): <span class="smallcaps">pos_2</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) |

Example initial encoding. Here we consider the input sequence \\(\overline{\$653+049=2070}\\) and the starting position ID is chosen as \\(s=2\\). The vectors \\({\bm{v}}^P_{\square}\\) are defined in  
efeq:vvDk. The gray rows will be filled in later.

</div>

### Token Embedding

The token embedding consists of 17 dimensions: we call them

<div class="center" markdown="1">

1=<span class="smallcaps">num</span>, 2=<span class="smallcaps">is_bos</span>, 3=<span class="smallcaps">full_ones</span>,

4=<span class="smallcaps">pre_sum</span>, 5=<span class="smallcaps">pre_carry</span>, 6=<span class="smallcaps">pre_eos</span>,

{7,...,16}=<span class="smallcaps">sum</span>, and 17=<span class="smallcaps">is_eos</span>.

</div>

Initially, we let the last 14 dimensions be empty (i.e., all zeros). Thus, we explain the first three dimensions, <span class="smallcaps">num</span>, <span class="smallcaps">is_bos</span>, and <span class="smallcaps">full_ones</span>.

#### Dimension 1 (<span class="smallcaps">num</span>).

For a number token (\\(0,\ldots,9\\)), we put itself in the dimension <span class="smallcaps">num</span>. For the other tokens (\\(+,=,\$\\)), we put \\(0\\).

#### Dimension 2 (<span class="smallcaps">is_bos</span>).

For a special token ‘\\(\$\\)’, we put \\(1\\) in the dimension <span class="smallcaps">is_bos</span>. Otherwise, we put \\(0\\).

#### Dimension 3 (<span class="smallcaps">full_ones</span>).

We put \\(1\\) everywhere in this dimension.

### Coupled Position IDs and Position Embedding

Before constructing a position embedding, we specify the coupled position IDs for the addition task. Let \\({\tt max\_pos}\\) be a hyperparameter of the maximum position IDs, where position IDs are non-negative integers. Basically, we match the significance of the digits: e.g., a least significant digit is always coupled to the other least significant digits. To this end, we first randomly choose a *starting position ID* \\(s \in [{\tt max\_pos}-\ell-1]\\). (For that, \\({\tt max\_pos}\ge \ell + 2\\) must hold.) Then we allocate the position IDs of token \\(\sigma_i\\) in the input sequence \\({\mathcal{I}}=\overline{\sigma_1\ldots\sigma_{N}}\\) as \\[\begin{aligned}
    p(i) = \begin{dcases}
        0, & i=1, \\
        s+i-1, & i=2,\ldots,\ell+2, \\
        s+i-(\ell+2), & i=\ell+3,\ldots,2\ell+3, \\
        s+(3\ell+4)-i & i=2\ell+4,\ldots,3\ell+4.
    \end{dcases}
\end{aligned}\\] Recall that \\(N = 3\ell + 4\\). Also, observe that for \\(i \in \{2, \ldots, \ell+1\}\\), \\[\begin{aligned}
    p(i) = p(i+\ell+1) = p(3\ell+5-i) = s+i,
\end{aligned}\\] which couples the position of \\((\ell-i+2)\\)-th significant digit in the first operand (\\(A\\)), the second operand (\\(B\\)), and the sum (\\(C\\)). Also, the position of tokens ‘\\(+\\)’ and ‘\\(=\\)’ are coupled. Lastly, the only token that has the position ID 0 is the special token ‘\\(\$\\)’.

Before moving on to the positional embedding, we define \\({\bm{v}}^D_k\\) (\\(k\in [2^D]\\)) as \\[\begin{aligned}
    \label{eq:vvDk}
    {\bm{v}}^D_k = \left[(-1)^{b_i^{(D,k)}}\right]_{i=1}^D \in \mathbb{R}^D
\end{aligned}\\] where \\(b_i^{(D,k)}\\) is defined as the \\(i\\)-th (from left) digit of \\(D\\)-digit binary representation of \\(k-1\\). For example, if \\(D=2\\), \\[\begin{aligned}
    {\bm{v}}^2_1 = \begin{bmatrix} 1 & 1 \end{bmatrix}^\top,\ 
    {\bm{v}}^2_2 = \begin{bmatrix} -1 & 1 \end{bmatrix}^\top,\ 
    {\bm{v}}^2_3 = \begin{bmatrix} 1 & -1 \end{bmatrix}^\top,\ 
    {\bm{v}}^2_4 = \begin{bmatrix} -1 & -1 \end{bmatrix}^\top.
\end{aligned}\\] We remark that the points \\({\bm{v}}^D_k\\) are the vertices of \\(D\\)-dimensional hypercube with side length 2, centered at the origin. [^10] Note that for \\(k\ne l\\), \\[\begin{aligned}
    \left\lVert{\bm{v}}^D_k\right\rVert^2 = D, \quad \left\langle{\bm{v}}^D_k, {\bm{v}}^D_l\right\rangle\le D-2.
\end{aligned}\\]

Now we explain the position embedding. It consists of \\(2P\\) dimensions, which eventually become from \\(18\\)-th to \\((2P+17)\\)-th dimension after concatenation. If \\(p(i)=0\\), we let \\({\bm{0}}_{2P}\\) as a position embedding vector. For the positive position IDs \\(p(i)\ge 1\\), we let a concatenation \\[\begin{aligned}
    \begin{bmatrix}
        {\bm{v}}^P_{p(i)} \\ {\bm{v}}^P_{p(i)+1}
    \end{bmatrix}
\end{aligned}\\] as a position embedding vector of a token \\(\sigma_i\\). (In case of \\(p(i)=2^P\\), we use \\({\bm{v}}^P_1\\) instead of \\({\bm{v}}^P_{p(i)+1}\\).) We call the former \\(P\\) dimensions for the position embedding as <span class="smallcaps">pos_1</span> and the latter \\(P\\) dimensions as <span class="smallcaps">pos_2</span>.

Concatenating the token embedding and the position embedding, we get the input embedding \\({\bm{X}}^{(0)}\\). See  
eftab:init_emb for an example. As a result, the total embedding dimension is \\(d=2P+17\\). Note the maximum possible position ID that can be represented with \\({\bm{v}}^P_k\\)’s is \\({\tt max\_pos}= 2^P = 2^{\left\lfloor (d-17)/2 \right\rfloor}\\). Therefore, the length of an operand must be \\(\ell \le {\tt max\_pos}-2 = 2^{\left\lfloor (d-17)/2 \right\rfloor}-2\\).

## Transformer Block — Causal Attention Layer [subsec:construction_addition_attention]

The goal of the causal attention layer is to fill in the zero-blanks [^11] of the encoding matrix at dimensions <span class="smallcaps">pre_sum</span>, <span class="smallcaps">pre_carry</span>, and <span class="smallcaps">pre_eos</span>. We divide the roles into two different heads.

### Attention Head 1: Digit-wise Addition without Carries

The goal of the first head is to perform a *digit-wise addition* and to fill in the blanks of the encoding matrix at dimension <span class="smallcaps">pre_sum</span>. Later, using this dimension, combined with the dimension <span class="smallcaps">pre_carry</span>, we will be able to perform the next-token prediction for addition. For now, we do not care about the carries, which will be dealt with in a later section. Formally, we aim to perform \\(\sigma_i + \sigma_{i+\ell+1}\\) for each \\(i\in \{2,\cdots,\ell+1\}\\) and put its result at the \\((3\ell+4-i)\\)-th position (column) of the dimension <span class="smallcaps">pre_sum</span> (row). To this end, we utilize our position embedding.

Recall that \\(d=2P+17\\) and let \\(d_{QK,1}=P+1\\). Let \\(M>0\\) be a number determined later. Let \\[\begin{aligned}
    {\bm{Q}}_1 &= \begin{pmatrix}
        {\bm{0}}_{P \times 17} & \sqrt{M}{\bm{I}}_{P} & {\bm{0}}_{P \times P} \\
        \sqrt{MP} ({\bm{e}}^{17}_{\textsc{full\_ones}})^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,1} \times d}, \\
    {\bm{K}}_1 &= \begin{pmatrix}
        {\bm{0}}_{P \times 17} & {\bm{0}}_{P \times P} & \sqrt{M}{\bm{I}}_{P} \\
         \sqrt{MP} ({\bm{e}}^{17}_{\textsc{is\_bos}})^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,1} \times d}.
\end{aligned}\\]

The linear transformations with matrices \\({\bm{Q}}_1\\) and \\({\bm{K}}_1\\) do two different jobs at once. (1) \\({\bm{Q}}_1\\) (\\({\bm{K}}_1\\), resp.) takes the dimensions <span class="smallcaps">pos_1</span> (<span class="smallcaps">pos_2</span>, resp.) from the input encoding matrix and scale them up by \\(\sqrt{M}\\); (2) \\({\bm{Q}}_1\\) (\\({\bm{K}}_1\\), resp.) takes the dimension <span class="smallcaps">full_ones</span> (<span class="smallcaps">is_bos</span>, resp.) and scale it up by \\(\sqrt{MP}\\). For concrete examples, please refer to  
eftab:addition_Q1X,tab:addition_K1X. By these, the attention *score* matrix \\({\bm{C}}_1 := ({\bm{K}}_1 {\bm{X}}^{(0)})^\top {\bm{Q}}_1 {\bm{X}}^{(0)}\\) becomes as in  
eftab:head1_att_score. The blanks in  
eftab:head1_att_score are the numbers smaller than \\(M(P-2)\\); the asterisks (‘\*’) are the entries (or lower triangular submatrices) ignored by the causal \\({\tt softmax}\\) operator; the dots represents the hidden \\(MP\\)’s.

<div id="tab:head1_att_score" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(3\\) | \\(\cdots\\) | \\(\ell+2\\) | \\(\ell+3\\) | \\(\ell+4\\) | \\(\cdots\\) | \\(2\ell+3\\) | \\(\cdots\\) | \\(3\ell+2\\) | \\(3\ell+3\\) | \\(3\ell+4\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | \\(MP\\) | \\(MP\\) | \\(MP\\) | \\(\cdots\\) | \\(MP\\) | \\(MP\\) | \\(MP\\) | \\(\cdots\\) | \\(MP\\) | \\(\cdots\\) | \\(MP\\) | \\(MP\\) | \\(MP\\) |
| \\(2\\) | \* |  | \\(MP\\) |  |  |  | \\(MP\\) |  |  |  | \\(MP\\) |  |  |
| \\(\vdots\\) | \* | \* |  | \\(\ddots\\) |  |  |  | \\(\ddots\\) |  | \\(\iddots\\) |  |  |  |
| \\(\ell+1\\) | \* | \* | \* |  | \\(MP\\) |  |  |  | \\(MP\\) |  |  |  |  |
| \\(\ell+2\\) | \* | \* | \* | \* |  |  |  |  |  |  |  |  |  |
| \\(\ell+3\\) | \* | \* | \* | \* | \* |  | \\(MP\\) |  |  |  | \\(MP\\) |  |  |
| \\(\vdots\\) | \* | \* | \* | \* | \* | \* |  | \\(\ddots\\) |  | \\(\iddots\\) |  |  |  |
| \\(2\ell+2\\) | \* | \* | \* | \* | \* | \* | \* |  | \\(MP\\) |  |  |  |  |
| \\(2\ell+3\\) | \* | \* | \* | \* | \* | \* | \* | \* |  |  |  |  |  |
| \\(\vdots\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* |  |  |  |  |
| \\(3\ell+2\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* |  |  |  |
| \\(3\ell+3\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* |  |  |
| \\(3\ell+4\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* |  |

Exact attention score matrix \\({\bm{C}}_1\\) (with explicit row/column indices) of Head 1.

</div>

Now consider the *attention matrix* \\({\bm{A}}_1 := {\tt softmax}({\bm{C}}_1) \in \mathbb{R}^{N\times N}\\). Its exact form is a bit messy due to the softmax operation of finite numbers. However, one can observe that, if the number \\(M\\) is large enough, it gets close to the column-stochastic matrix \\({\bm{T}}_1\in \mathbb{R}^{N\times N}\\) described in  
eftab:head1_limiting_attn_mtx. The blanks in  
eftab:head1_limiting_attn_mtx are zeros; the dots represent the omitted nonzero entries.

<div id="tab:head1_limiting_attn_mtx" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(3\\) | \\(\cdots\\) | \\(\ell+2\\) | \\(\ell+3\\) | \\(\ell+4\\) | \\(\cdots\\) | \\(2\ell+3\\) | \\(\cdots\\) | \\(3\ell+2\\) | \\(3\ell+3\\) | \\(3\ell+4\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | 1 | 1 | 1/2 | \\(\cdots\\) | 1/2 | 1 | 1/3 | \\(\cdots\\) | 1/3 | \\(\cdots\\) | 1/3 | 1 | 1 |
| \\(2\\) | 0 | 0 | 1/2 |  | 0 | 0 | 1/3 |  | 0 |  | 1/3 | 0 | 0 |
| \\(\vdots\\) |  |  |  | \\(\ddots\\) |  |  |  | \\(\ddots\\) |  | \\(\iddots\\) |  |  |  |
| \\(\ell+1\\) | 0 | 0 | 0 |  | 1/2 | 0 | 0 |  | 1/3 |  | 0 | 0 | 0 |
| \\(\ell+2\\) | 0 | 0 | 0 |  | 0 | 0 | 0 |  | 0 |  | 0 | 0 | 0 |
| \\(\ell+3\\) | 0 | 0 | 0 |  | 0 | 0 | 1/3 |  | 0 |  | 1/3 | 0 | 0 |
| \\(\vdots\\) |  |  |  |  |  |  |  | \\(\ddots\\) |  | \\(\iddots\\) |  |  |  |
| \\(2\ell+2\\) | 0 | 0 | 0 |  | 0 | 0 | 0 |  | 1/3 |  | 0 | 0 | 0 |
| \\(2\ell+3\\) | 0 | 0 | 0 |  | 0 | 0 | 0 |  | 0 |  | 0 | 0 | 0 |
| \\(\vdots\\) |  |  |  |  |  |  |  |  |  |  |  |  |  |
| \\(3\ell+4\\) | 0 | 0 | 0 |  | 0 | 0 | 0 |  | 0 |  | 0 | 0 | 0 |

Limiting attention matrix \\({\bm{T}}_1\\) (with explicit row/column indices) of Head 1, as \\(M\\) gets large.

</div>

Let \\({\bm{R}}_1 = {\bm{A}}_1 - {\bm{T}}_1 \in \mathbb{R}^{N\times N}\\) be the error matrix, which is upper triangular. Its exact form is messy as well, but we can obtain the bounds of their entries. Consider a pair of indices \\((i,j)\in [N]^2\\) such that \\(i\le j\\). Let \\(x_j = 1/[{\bm{T}}_1]_{1j} \in \{1, 2, 3\}\\). If \\([{\bm{T}}_1]_{ij} = \frac{1}{x_j}\\), \\([{\bm{R}}_1]_{ij} < 0\\) and \\[\begin{aligned}
    -[{\bm{R}}_1]_{ij} &\le \frac{1}{x_j} - \frac{e^{MP}}{x_je^{MP}+(j-x_j)e^{M(P-2)}}= \frac{j-x_j}{x_j(x_j e^{2M} + (j-x_j))}. \label{eq:r_ineq11}
\end{aligned}\\] On the other hand, if \\([{\bm{T}}_1]_{ij} = 0\\), \\([{\bm{R}}_1]_{ij} > 0\\) and \\[\begin{aligned}
    [{\bm{R}}_1]_{ij} &\le \frac{e^{M(P-2)}}{x_j e^{MP}+(j-x_j)e^{M(P-2)}} = \frac{1}{x_j e^{2M} + (j-x_j)}. \label{eq:r_ineq12}
\end{aligned}\\]

Now let \\(d_{V,1} = 1\\) and \\[\begin{aligned}
    {\bm{V}}_1 &= 3({\bm{e}}^{d}_{\textsc{num}})^\top \in \mathbb{R}^{d_{V,1} \times d}, \\
    {\bm{U}}_1 &= {\bm{e}}^{d}_{\textsc{pre\_sum}} \in \mathbb{R}^{d \times d_{V,1}}.
\end{aligned}\\] The linear transformation with matrix \\({\bm{U}}_1{\bm{V}}_1\\) takes the dimension <span class="smallcaps">num</span> from the input encoding matrix, scales it up by 3, and puts it to the dimension <span class="smallcaps">pre_sum</span>. A concrete example is provided in  
eftab:addition_V1X.

Obtaining \\({\bm{U}}_1{\bm{V}}_1 {\bm{X}}^{(0)} {\bm{A}}_1\\), its every entry is zero except at the dimension <span class="smallcaps">pre_sum</span>. Observe that \\([{\bm{U}}_1{\bm{V}}_1 {\bm{X}}^{(0)}]_{(\textsc{pre\_sum}) 1}=0\\), because in the input encoding matrix, the dimension <span class="smallcaps">num</span> starts with 0. Also, note that it is enough to focus on the columns \\(j \in \{2\ell+3, \ldots, 3\ell+4\}\\) since we only care about the next-token prediction of the tokens after \\(\sigma_{2\ell+3}=\\)‘=’. Specifying the dimension (i.e., the particular row) for these columns, we have \\[\begin{aligned}
    [{\bm{U}}_1{\bm{V}}_1 {\bm{X}}^{(0)} {\bm{T}}_1]_{(\textsc{pre\_sum}) j} &= \begin{dcases}
        {\bm{X}}^{(0)}_{(\textsc{num})(3\ell+4-j)} + {\bm{X}}^{(0)}_{(\textsc{num})(4\ell+5-j)} & \text{if } j \in \{2\ell+3, \ldots, 3\ell+2\},\\
        0 & \text{if } j\in\{3\ell+3, 3\ell+4\},
    \end{dcases}  \\
    &=\begin{dcases}
        \sigma_{(3\ell+4)-j} + \sigma_{(4\ell+5)-j} & \text{if } j \in \{2\ell+3, \ldots, 3\ell+2\},\\
        0 & \text{if } j\in\{3\ell+3, 3\ell+4\}.
    \end{dcases} 
\end{aligned}\\] Refer to  
eftab:addition_U1V1XM1 for a concrete example of computing \\({\bm{U}}_1{\bm{V}}_1 {\bm{X}}^{(0)} {\bm{T}}_1\\). Also, for the softmax errors, \\[\begin{aligned}
    [{\bm{U}}_1{\bm{V}}_1 {\bm{X}}^{(0)} {\bm{R}}_1]_{(\textsc{pre\_sum}) j} = \sum_{2\le i\le j} 3{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij}.
\end{aligned}\\] Specifically, if \\(j \in \{2\ell+3, \ldots, 3\ell+2\}\\) (thus \\(x_j=3\\)), \\[\begin{aligned}
    [{\bm{U}}_1{\bm{V}}_1 {\bm{X}}^{(0)} {\bm{R}}_1]_{(\textsc{pre\_sum}) j} = \underbrace{\sum_{i\in \{(3\ell+4)-j, (4\ell+5)-j\}} 3{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij}}_{\text{negative}} + \underbrace{\sum_{\substack{2 \le i \le j \\ i\ne (3\ell+4)-j \\ i\ne (4\ell+5)-j}} 3{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij}}_{\text{positive}},
\end{aligned}\\] where \\[\begin{aligned}
    0 \le -\sum_{i\in \{(3\ell+4)-j, (4\ell+5)-j\}} 3{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij} \le \frac{2\cdot 9(j-3)}{3 e^{2M} + (j-3)}
\end{aligned}\\] holds by  
efeq:r_ineq11, and \\[\begin{aligned}
    0\le \sum_{\substack{2 \le i \le j \\ i\ne (3\ell+4)-j \\ i\ne (4\ell+5)-j}} 3{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij} \le \frac{27(j-3)}{3 e^{2M} + (j-3)}
\end{aligned}\\] holds by  
efeq:r_ineq12. On the other hand, if \\(j\in\{3\ell+3, 3\ell+4\}\\), \\[\begin{aligned}
    0\le [{\bm{U}}_1{\bm{V}}_1 {\bm{X}}^{(0)} {\bm{R}}_1]_{(\textsc{pre\_sum}) j} &= \sum_{2 \le i \le j} 3{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij} \le \frac{27(j-1)}{e^{2M} + (j-1)}.
\end{aligned}\\] One can easily prove these inequalities by using the bounds of \\([{\bm{R}}_1]_{ij}\\)’s and the fact that the entries in \\({\bm{X}}^{(0)}_{(\textsc{num})\bullet}\\) lie in the interval \\([0,9]\\).

If we let \\(M \ge \frac{1}{2}\log(N-1)+3\\), we can ensure that \\(\left|[{\bm{U}}_1{\bm{V}}_1 {\bm{X}}^{(0)}{\bm{R}}_1]_{(\textsc{pre\_sum}) j}\right|\\) smaller than 0.1 for each \\(j \in \{2\ell+3, \ldots, 3\ell+4\}\\). The proof is simple: it is enough to check \\[\begin{aligned}
    \frac{27(N-3)}{3 e^{2M} + (N-3)} < \frac{1}{10}, \quad \frac{27(N-1)}{e^{2M} + (N-1)} < \frac{1}{10}.
\end{aligned}\\]

<div id="tab:addition_Q1X" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{2}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) |

Example of \\({\bm{Q}}_1{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb.

</div>

<div id="tab:addition_K1X" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Example of \\({\bm{K}}_1{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb.

</div>

<div id="tab:addition_V1X" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">pre_sum</span> | 0 | 18 | 15 | 9 | 0 | 0 | 12 | 27 | 0 | 6 | 0 | 21 | 0 |
| 5: <span class="smallcaps">pre_carry</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">pre_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7–16: <span class="smallcaps">sum</span> | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) |
| 17: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18–end: <span class="smallcaps">pos_1</span>,<span class="smallcaps">pos_2</span> | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) |

Example of \\({\bm{U}}_1{\bm{V}}_1{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb.

</div>

<div id="tab:addition_U1V1XM1" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">pre_sum</span> | 0 | 0 | 9 | 7.5 | 4.5 | 0 | 6 | 9 | 12 | 9 | 6 | 0 | 0 |
| 5: <span class="smallcaps">pre_carry</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">pre_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7–16: <span class="smallcaps">sum</span> | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) |
| 17: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18–end: <span class="smallcaps">pos_1</span>,<span class="smallcaps">pos_2</span> | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) |

Example of \\({\bm{U}}_1{\bm{V}}_1{\bm{X}}^{(0)}{\bm{T}}_1\\), continuing from  
eftab:addition_V1X. See  
eftab:head1_limiting_attn_mtx for the definition of \\({\bm{T}}_1\\).

</div>

### Attention Head 2: Carry & EOS Detection [sec:addition_attn_head2]

The goal of the second head is to fill in the blanks of the encoding matrix at dimensions <span class="smallcaps">pre_carry</span> and <span class="smallcaps">pre_eos</span>. At dimension <span class="smallcaps">pre_eos</span>, we will put (approximately) 1 if the next token would be the EOS token (‘\\(\$\\)’), otherwise, we will put strictly smaller numbers like (approximately) 2/3 and 1/2.

What we will put at dimension <span class="smallcaps">pre_carry</span> is the evidence of the presence of an additional carry, which is not quite straightforward to understand. Let us take a look at some examples. Consider an addition \\(3+9=12\\). Since it is greater than or equal to 10, the least significant digits in the operands generate a carry 1. But in some cases, a pair of digits with a sum less than 10 can make a carry. Next, consider an addition \\(53+49=102\\). In the second least significant digits, An addition of 5 and 4 occurs. However, a carry is already produced in the least significant digits (\\(3+9=12\\)), so the total sum including the carry is 10, not 9. Thus, it also produces a carry. But how can we know the presence of a carry while only looking at the second least significant digits? The answer is to observe the second least significant digit in the sum, 0 of 102. Somehow, the consequence of adding 5 and 4 is 0, (or 10, implicitly) so it makes a carry.

To generalize this explanation, let \\(a\\) and \\(b\\) be digits of the operands in the same significance, and \\(c\\) be a digit of the sum in the same significance as \\(a\\) and \\(b\\). We find that the rule of recognizing that the addition of \\(a\\) and \\(b\\) generates a carry is that \\[\begin{aligned}
    \begin{dcases}
        \text{If } a+b-c \in \{9, 10\}, & \text{then a carry is generated}, \\
        \text{Otherwise}, & \text{then the carry is not generated}.
    \end{dcases}
\end{aligned}\\] Thus, it is crucial to store the information of \\(a+b-c\\) or any related one somewhere. In fact, we can store \\(a+b+c\\) at dimension <span class="smallcaps">pre_carry</span> of the encoding matrix, and it can be transformed into \\(a+b-c\\) and used later in the feed-forward layer. Formally, we aim to perform \\(\sigma_i + \sigma_{i+\ell+1} + \sigma_{3\ell+5-i}\\) for each \\(i\in\{2,...,\ell+1\}\\) and put its result at the \\((3\ell+5-i)\\)-th position (column) of the dimension <span class="smallcaps">pre_carry</span> (row). To this end, we again utilize our position embedding.

Recall that \\(d=2P+17\\) and let \\(d_{QK,2}=P+1\\). Let \\[\begin{aligned}
    {\bm{Q}}_2 &= \begin{pmatrix}
        {\bm{0}}_{P \times 17} & \sqrt{M}{\bm{I}}_{P} & {\bm{0}}_{P \times P} \\
        \sqrt{MP} ({\bm{e}}^{17}_{\textsc{full\_ones}})^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,2} \times d}, \\
    {\bm{K}}_2 &= \begin{pmatrix}
        {\bm{0}}_{P \times 17} & \sqrt{M}{\bm{I}}_{P} & {\bm{0}}_{P \times P} \\
         \sqrt{MP} ({\bm{e}}^{17}_{\textsc{is\_bos}})^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,2} \times d}.
\end{aligned}\\]

The linear transformations with matrices \\({\bm{Q}}_2\\) and \\({\bm{K}}_2\\) do two different jobs at once. (1) they take the dimensions <span class="smallcaps">pos_1</span> from the input encoding matrix and scale them up by \\(\sqrt{M}\\); (2) \\({\bm{Q}}_2\\) (\\({\bm{K}}_2\\), resp.) takes the dimension <span class="smallcaps">full_ones</span> (<span class="smallcaps">is_bos</span>, resp.) and scale it up by \\(\sqrt{MP}\\). For concrete examples, refer to  
eftab:addition_Q2X,tab:addition_K2X. By these, the attention score matrix \\({\bm{C}}_2 := ({\bm{K}}_2 {\bm{X}}^{(0)})^\top {\bm{Q}}_2 {\bm{X}}^{(0)}\\) becomes as in  
eftab:head2_att_score. The blanks in  
eftab:head2_att_score are the numbers less than equal to \\(M(P-2)\\); the asterisks (‘\*’) are the entries (or lower triangular submatrices) ignored by the causal \\({\tt softmax}\\) operator; the dots represent the hidden \\(MP\\)’s.

<div id="tab:head2_att_score" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(\cdots\\) | \\(\ell+1\\) | \\(\ell+2\\) | \\(\ell+3\\) | \\(\cdots\\) | \\(2\ell+2\\) | \\(2\ell+3\\) | \\(2\ell+4\\) | \\(\cdots\\) | \\(3\ell+3\\) | \\(3\ell+4\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | \\(MP\\) | \\(MP\\) | \\(\cdots\\) | \\(MP\\) | \\(MP\\) | \\(MP\\) | \\(\cdots\\) | \\(MP\\) | \\(MP\\) | \\(MP\\) | \\(\cdots\\) | \\(MP\\) | \\(MP\\) |
| \\(2\\) | \* | \\(MP\\) |  |  |  | \\(MP\\) |  |  |  |  |  | \\(MP\\) |  |
| \\(\vdots\\) | \* | \* | \\(\ddots\\) |  |  |  | \\(\ddots\\) |  |  |  | \\(\iddots\\) |  |  |
| \\(\ell+1\\) | \* | \* | \* | \\(MP\\) |  |  |  | \\(MP\\) |  | \\(MP\\) |  |  |  |
| \\(\ell+2\\) | \* | \* | \* | \* | \\(MP\\) |  |  |  | \\(MP\\) |  |  |  |  |
| \\(\ell+3\\) | \* | \* | \* | \* | \* | \\(MP\\) |  |  |  |  |  | \\(MP\\) |  |
| \\(\vdots\\) | \* | \* | \* | \* | \* | \* | \\(\ddots\\) |  |  |  | \\(\iddots\\) |  |  |
| \\(2\ell+2\\) | \* | \* | \* | \* | \* | \* | \* | \\(MP\\) |  | \\(MP\\) |  |  |  |
| \\(2\ell+3\\) | \* | \* | \* | \* | \* | \* | \* | \* | \\(MP\\) |  |  |  |  |
| \\(2\ell+4\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \\(MP\\) |  |  |  |
| \\(\vdots\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \\(\ddots\\) |  |  |
| \\(3\ell+3\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \\(MP\\) |  |
| \\(3\ell+4\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \\(MP\\) |

Exact attention score matrix \\({\bm{C}}_2\\) (with explicit row/column indices) of Head 2.

</div>

Now consider the attention matrix \\({\bm{A}}_2 := {\tt softmax}({\bm{C}}_2) \in \mathbb{R}^{N\times N}\\). Similarly to the previous head, if the number \\(M\\) is large enough, it gets close to the column-stochastic matrix \\({\bm{T}}_2\in \mathbb{R}^{N\times N}\\) described in  
eftab:head2_limiting_attn_mtx. The blanks in  
eftab:head2_limiting_attn_mtx are zeros; the dots represent the omitted nonzero entries.

<div id="tab:head2_limiting_attn_mtx" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(\cdots\\) | \\(\ell+1\\) | \\(\ell+2\\) | \\(\ell+3\\) | \\(\cdots\\) | \\(2\ell+2\\) | \\(2\ell+3\\) | \\(2\ell+4\\) | \\(\cdots\\) | \\(3\ell+3\\) | \\(3\ell+4\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | 1 | 1/2 | \\(\cdots\\) | 1/2 | 1/2 | 1/3 | \\(\cdots\\) | 1/3 | 1/3 | 1/4 | \\(\cdots\\) | 1/4 | 1/2 |
| \\(2\\) | \* | 1/2 |  | 0 | 0 | 1/3 |  | 0 | 0 | 0 |  | 1/4 | 0 |
| \\(\vdots\\) | \* | \* | \\(\ddots\\) |  |  |  | \\(\ddots\\) |  |  |  | \\(\iddots\\) |  |  |
| \\(\ell+1\\) | \* | \* | \* | 1/2 | 0 | 0 |  | 1/3 | 0 | 1/4 |  | 0 | 0 |
| \\(\ell+2\\) | \* | \* | \* | \* | 1/2 | 0 |  | 0 | 1/3 | 0 |  | 0 | 0 |
| \\(\ell+3\\) | \* | \* | \* | \* | \* | 1/3 |  | 0 | 0 | 0 |  | 1/4 | 0 |
| \\(\vdots\\) | \* | \* | \* | \* | \* | \* | \\(\ddots\\) |  |  |  | \\(\iddots\\) |  |  |
| \\(2\ell+2\\) | \* | \* | \* | \* | \* | \* | \* | 1/3 | 0 | 1/4 |  | 0 | 0 |
| \\(2\ell+3\\) | \* | \* | \* | \* | \* | \* | \* | \* | 1/3 | 0 |  | 0 | 0 |
| \\(2\ell+4\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | 1/4 |  | 0 | 0 |
| \\(\vdots\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \\(\ddots\\) |  |  |
| \\(3\ell+3\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | 1/4 | 0 |
| \\(3\ell+4\\) | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | \* | 1/2 |

Limiting attention matrix \\({\bm{T}}_2\\) (with explicit row/column indices) of Head 2, as \\(M\\) gets large.

</div>

Let \\({\bm{R}}_2 = {\bm{A}}_2 - {\bm{T}}_2 \in \mathbb{R}^{N\times N}\\) be the error matrix, which is upper triangular as well. Its exact form is messy as well, but we can obtain the bounds of their entries. Consider a pair of indices \\((i,j)\in [N]^2\\) such that \\(i\le j\\). Let \\(x_j = 1/[{\bm{T}}_2]_{1j} \in \{1, 2, 3, 4\}\\). If \\([{\bm{T}}_2]_{ij} = \frac{1}{x_j}\\), \\([{\bm{R}}_2]_{ij} < 0\\) and \\[\begin{aligned}
    -[{\bm{R}}_2]_{ij} &\le \frac{1}{x_j} - \frac{e^{MP}}{x_je^{MP}+(j-x_j)e^{M(P-2)}}= \frac{j-x_j}{x_j(x_j e^{2M} + (j-x_j))}.
\end{aligned}\\] On the other hand, if \\([{\bm{T}}_2]_{ij} = 0\\), \\([{\bm{R}}_2]_{ij} > 0\\) and \\[\begin{aligned}
    [{\bm{R}}_2]_{ij} &\le \frac{e^{M(P-2)}}{x_j e^{MP}+(j-x_j)e^{M(P-2)}} = \frac{1}{x_j e^{2M} + (j-x_j)}.
\end{aligned}\\]

Now let \\(d_{V,2} = 2\\) and \\[\begin{aligned}
    {\bm{V}}_2 &= \begin{pmatrix}
        4({\bm{e}}^{d}_{\textsc{num}})^\top \\
        2({\bm{e}}^{d}_{\textsc{is\_bos}})^\top
    \end{pmatrix} \in \mathbb{R}^{d_{V,2} \times d}, \\
    {\bm{U}}_2 &= \begin{pmatrix}
        {\bm{e}}^{d}_{\textsc{pre\_carry}} & {\bm{e}}^{d}_{\textsc{pre\_eos}}
    \end{pmatrix}\in \mathbb{R}^{d \times d_{V,2}}.
\end{aligned}\\] The linear combination with matrix \\({\bm{U}}_2{\bm{V}}_2\\) does two jobs at once. First, it takes the dimension <span class="smallcaps">num</span> from the encoding matrix, scales it up by 4, and puts it to the dimension <span class="smallcaps">pre_carry</span>. Second, it takes the dimension <span class="smallcaps">is_bos</span> from the encoding matrix, scales it up by 2, and puts it to the dimension <span class="smallcaps">pre_eos</span>. A concrete example is provided in  
eftab:addition_V2X.

Obtaining \\({\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{A}}_2\\), its every entry is zero except at the dimensions <span class="smallcaps">pre_carry</span> and <span class="smallcaps">pre_eos</span>. Observe that \\([{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)}]_{(\textsc{pre\_carry}) 1}=0\\), because in the input encoding matrix, the dimension <span class="smallcaps">num</span> starts with 0. Also, note again that it is enough to focus on the columns \\(j \in \{2\ell+3, \ldots, 3\ell+4\}\\), since we only care about the next-token prediction of the tokens after \\(\sigma_{2\ell+3}=\\)‘=’. Specifying the dimensions (i.e., the particular rows) for these columns, we have \\[\begin{aligned}
    [{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{T}}_2]_{(\textsc{pre\_carry}) j} &=\begin{dcases}
        \frac{4}{3} \left({\bm{X}}^{(0)}_{(\textsc{num})(\ell+2)} + {\bm{X}}^{(0)}_{(\textsc{num})j}\right) & \text{if } (2\ell+3) = 2\ell+3,\\
        {\bm{X}}^{(0)}_{(\textsc{num})(3\ell+5-j)} + {\bm{X}}^{(0)}_{(\textsc{num})(4\ell+6-j)} +  {\bm{X}}^{(0)}_{(\textsc{num})j} & \text{if } j \in \{2\ell+4, \ldots, 3\ell+3\},\\
        0 & \text{if } j = 3\ell+4,
    \end{dcases} \\
    &= \begin{dcases}
        0 & \text{if } j \in \{2\ell+3, 3\ell+4\}, \\
         \sigma_{(3\ell+5)-j} + \sigma_{(4\ell+6)-j} + \sigma_{j} & \text{if } j \in \{2\ell+4, \ldots, 3\ell+3\},
    \end{dcases}\\
    [{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{T}}_2]_{(\textsc{pre\_eos}) j} &=\begin{dcases}
        2/3 & \text{if } j = 2\ell+3,\\
        1/2 & \text{if } j \in \{2\ell+4, \ldots, 3\ell+3\},\\
        1 & \text{if } j = 3\ell+4.
    \end{dcases} 
\end{aligned}\\] Refer to  
eftab:addition_U2V2XM2 for a concrete example of computing \\({\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{T}}_2\\). Also, for the softmax errors, \\[\begin{aligned}
    [{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{R}}_2]_{(\textsc{pre\_carry}) j} &= \sum_{2\le i\le j} 4{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij}, \\
    [{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{R}}_2]_{(\textsc{pre\_eos}) j} &= \sum_{1\le i\le j} 2{\bm{X}}^{(0)}_{(\textsc{is\_bos})i} [{\bm{R}}_1]_{ij}.
\end{aligned}\\] Let us first obtain a bound of the softmax error term at dimension <span class="smallcaps">pre_carry</span>. If \\(j = 2\ell+3\\), since \\({\bm{X}}^{(0)}_{(\textsc{num})(\ell+2)}={\bm{X}}^{(0)}_{(\textsc{num})(2\ell+3)}=0\\), \\[\begin{aligned}
    [{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{R}}_2]_{(\textsc{pre\_carry}) (2\ell+3)} = \sum_{\substack{2\le i\le 2\ell+2 \\ i\ne \ell+2}} 4{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij}
\end{aligned}\\] and \\[\begin{aligned}
    0 \le \sum_{\substack{2\le i\le 2\ell+2 \\ i\ne \ell+2}} 4{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij} \le \frac{36(2\ell)}{3e^{2M} + 2\ell}.
\end{aligned}\\] If \\(j \in \{2\ell+4, \ldots, 3\ell+3\}\\), \\[\begin{aligned}
    [{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{R}}_2]_{(\textsc{pre\_carry}) j} = \underbrace{\sum_{i\in \{(3\ell+5)-j, (4\ell+6)-j, j\}} 4{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij}}_{\text{negative}} + \underbrace{\sum_{\substack{2 \le i \le j-1 \\ i\ne (3\ell+5)-j \\ i\ne (4\ell+6)-j}} 4{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij}}_{\text{positive}},
\end{aligned}\\] where \\[\begin{aligned}
    0 \le -\sum_{i\in \{(3\ell+5)-j, (4\ell+6)-j, j\}} 4{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij} \le \frac{3\cdot 9(j-4)}{4 e^{2M} + (j-4)}
\end{aligned}\\] and \\[\begin{aligned}
    0\le \sum_{\substack{2 \le i \le j-1 \\ i\ne (3\ell+5)-j \\ i\ne (4\ell+6)-j}} 4{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{ij} \le \frac{36(j-4)}{4 e^{2M} + (j-4)}.
\end{aligned}\\] And if \\(j=3\ell+4=N\\), \\[\begin{aligned}
    [{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{R}}_2]_{(\textsc{pre\_carry}) N} = \underbrace{4{\bm{X}}^{(0)}_{(\textsc{num})N} [{\bm{R}}_1]_{NN}}_{\text{negative}} + \underbrace{\sum_{2 \le i \le N-1} 4{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{iN}}_{\text{positive}},
\end{aligned}\\] where \\[\begin{aligned}
    0\le - 4{\bm{X}}^{(0)}_{(\textsc{num})N} [{\bm{R}}_1]_{NN} \le \frac{18(N-2)}{2e^{2M} + N-2}
\end{aligned}\\] and \\[\begin{aligned}
    0 \le \sum_{2 \le i \le N-1} 4{\bm{X}}^{(0)}_{(\textsc{num})i} [{\bm{R}}_1]_{iN} \le \frac{36(N-2)}{2e^{2M} + N-2}.
\end{aligned}\\] Next, we obtain a bound of the softmax error term at dimension <span class="smallcaps">pre_eos</span>. Since \\[\begin{aligned}
    \sum_{1\le i\le j} 2{\bm{X}}^{(0)}_{(\textsc{is\_bos})i} [{\bm{R}}_1]_{ij} &= 2{\bm{X}}^{(0)}_{(\textsc{is\_bos})1} [{\bm{R}}_1]_{1j},
\end{aligned}\\] the error term can be bounded as \\[\begin{aligned}
    0\le -[{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)} {\bm{R}}_2]_{(\textsc{pre\_eos}) j} &\le \begin{dcases}
        \frac{2(j-3)}{3(3e^{2M} + j - 3)} & \text{if } j = 2\ell+3\\
        \frac{2{(j-4)}}{4(4e^{2M} + j - 4)} & \text{if } j \in \{2\ell+4, \ldots, 3\ell+3\},\\ 
        \frac{(j-2)}{2e^{2M} + j - 2} & \text{if } j = 3\ell+4.
    \end{dcases}
\end{aligned}\\]

We then can ensure that both \\(\left|[{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)}{\bm{R}}_2]_{(\textsc{pre\_sum}) j}\right|\\) and \\(\left|[{\bm{U}}_2{\bm{V}}_2 {\bm{X}}^{(0)}{\bm{R}}_2]_{(\textsc{pre\_eos}) j}\right|\\) smaller than 0.1 for each \\(j \in \{2\ell+3, \ldots, 3\ell+4\}\\), by letting \\(M \ge \frac{1}{2}\log(N)+3\\). The proof is similar to the one that is presented for head 1.

<div id="tab:addition_Q2X" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{2}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) |

Example of \\({\bm{Q}}_2{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb.

</div>

<div id="tab:addition_K2X" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{2}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Example of \\({\bm{K}}_2{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb.

</div>

<div id="tab:addition_V2X" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">pre_sum</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">pre_carry</span> | 0 | 24 | 20 | 12 | 0 | 0 | 16 | 36 | 0 | 8 | 0 | 28 | 0 |
| 6: <span class="smallcaps">pre_eos</span> | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7–16: <span class="smallcaps">sum</span> | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) |
| 17: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18–end: <span class="smallcaps">pos_1</span> | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) |

Example of \\({\bm{U}}_2{\bm{V}}_2{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb.

</div>

<div id="tab:addition_U2V2XM2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">pre_sum</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">pre_carry</span> | 0 | 12 | 10 | 6 | 0 | 8 | 12 | 16 | 0 | 14 | 9 | 13 | 0 |
| 6: <span class="smallcaps">pre_eos</span> | 2 | 1 | 1 | 1 | 1 | 2/3 | 2/3 | 2/3 | 2/3 | 1/2 | 1/2 | 1/2 | 1 |
| 7–16: <span class="smallcaps">sum</span> | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) |
| 17: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18–end: <span class="smallcaps">pos_1</span> | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) |

Example of \\({\bm{U}}_2{\bm{V}}_2{\bm{X}}^{(0)}{\bm{T}}_2\\), continuing from  
eftab:addition_V2X. See  
eftab:head2_limiting_attn_mtx for definition of \\({\bm{T}}_2\\).

</div>

### Residual Connection [sec:addition_attn_res_conn]

So far we have computed the output of \\({\tt Att}_1\\) operation. Passing through the residual connection, the output of the attention layer is the sum of the original input encoding matrix and the output of \\({\tt Att}\\) operation: \\[\begin{aligned}
    {\bm{Y}}^{(1)} = {\bm{X}}^{(0)} + \sum_{h\in \{1,2\}} {\bm{U}}_h {\bm{V}}_h {\bm{X}}^{(0)}{\bm{T}}_h + \underbrace{\sum_{h\in \{1,2\}} {\bm{U}}_h {\bm{V}}_h {\bm{X}}^{(0)}{\bm{R}}_h}_{\text{softmax error term}}.
\end{aligned}\\] Since the term \\(\sum_{h\in \{1,2\}} {\bm{U}}_h {\bm{V}}_h {\bm{X}}^{(0)}{\bm{T}}_h\\) has nonzero entries only at dimensions <span class="smallcaps">pre_sum</span>, <span class="smallcaps">pre_carry</span>, and <span class="smallcaps">pre_eos</span>, the residual connection plays a role of “filling in some blanks” in the input encoding matrix. A concrete example of the output of residual connection is presented in  
eftab:addition_res_conn_att, ignoring the softmax error term, whose entries have an absolute value smaller than 0.1.

<div id="tab:addition_res_conn_att" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 6 | 5 | 3 | 0 | 0 | 4 | 9 | 0 | 2 | 0 | 7 | 0 |
| 2: <span class="smallcaps">is_bos</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">full_ones</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 4: <span class="smallcaps">pre_sum</span> | 0 | 0 | 9 | 7.5 | 4.5 | 0 | 6 | 9 | 12 | 9 | 6 | 0 | 0 |
| 5: <span class="smallcaps">pre_carry</span> | 0 | 12 | 10 | 6 | 0 | 8 | 12 | 16 | 0 | 14 | 9 | 13 | 0 |
| 6: <span class="smallcaps">pre_eos</span> | 2 | 1 | 1 | 1 | 1 | 2/3 | 2/3 | 2/3 | 2/3 | 1/2 | 1/2 | 1/2 | 1 |
| 7–16: <span class="smallcaps">sum</span> | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) |
| 17: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18–\\((P+17)\\): <span class="smallcaps">pos_1</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) |
| \\((P+18)\\)-\\((2P+17)\\): <span class="smallcaps">pos_2</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) |

Example output of residual connection, continuing from  
eftab:init_emb,tab:addition_U1V1XM1,tab:addition_U2V2XM2. Here we ignore the softmax error terms in the orange rows. The gray rows will be filled in later.

</div>

## Transformer Block — Token-wise Feed-forward Layer [subsec:construction_addition_feedforward]

The goal of the feed-forward layer is to fill in the blanks of the encoding matrix at dimensions <span class="smallcaps">sum</span> and <span class="smallcaps">is_eos</span>. Be careful that the feed-forward layer can only implement token-wise mappings; a token-wise mapping takes inputs only from the entries in the same column of the encoding matrix. Besides, the architecture of our feed-forward layer (except for the residual connection) is a one-hidden-layer ReLU network.

For a token \\(\sigma_i\\) for \\(i \in \{2\ell+3,\ldots,3\ell+3\}\\) (from ‘=’ token to the second ), we will put a standard unit vector \\({\bm{e}}^{10}_{k+1}\\) to dimensions <span class="smallcaps">sum</span> if the next token is \\(k\in \{0, \ldots, 9\}\\).

Recall from the discussion in  
efsec:addition_attn_head2 that we can judge whether a carry 1 is generated at a certain position by exploiting only the digits (of the operands and the sum) in the same significance. Bringing the notation, let \\(a\\) and \\(b\\) be digits of the operands in the same significance, and \\(c\\) be a digit of the sum in the same significance as \\(a\\) and \\(b\\). Then the rule of recognizing that the addition of \\(a\\) and \\(b\\) generates a carry is that \\[\begin{aligned}
    \begin{dcases}
        \text{If } a+b-c \in \{9, 10\}, & \text{then a carry is generated}, \\
        \text{Otherwise: if } a+b-c \in \{-1, 0\},  & \text{then the carry is not generated}.
    \end{dcases}
\end{aligned}\\] A simple case analysis shows that the value of \\(a+b-c\\) must be one of \\(-1, 0, 9\\), and \\(10\\). Let us briefly check this claim in our example: \\[\begin{aligned}
    6+0-7&=-1; &{\color{gray} \text{no carry from 6+0}} \\
    5+4-0&=9; &{\color{gray} \text{there is a carry from 5+4}} \\
    3+9-2&=10. &{\color{gray} \text{there is a carry from 3+9}}
\end{aligned}\\]

Recall that a noisy version of \\(a+b+c\\) is already stored at dimension <span class="smallcaps">pre_carry</span> of \\({\bm{Y}}^{(1)}\\), and \\(c\\) is exactly at dimension <span class="smallcaps">num</span>. Thus, we can (approximately) implement \\(a+b-c\\) for a token \\(\sigma_j\\) by \\[\begin{aligned}
    {\bm{Y}}^{(0)}_{(\textsc{pre\_carry})j} - 2 {\bm{Y}}^{(0)}_{(\textsc{num})j}.
\end{aligned}\\] This is a kind of token-wise *linear* transform, so we do not need to consume any hidden layer (with ReLU activation \\(\phi\\)) to implement it.

Combining with \\({\bm{Y}}^{(0)}_{(\textsc{pre\_sum})j}\\), a noisy version of addition without carry, we can indeed implement the addition. Note that a digit-wise addition should be done as \\[\begin{aligned}
    \text{digit-wise addition} = (\text{addition without carry} + \mathbbm{1}_{\left\{\text{carry propagates}\right\}}) \mod 10.
\end{aligned}\\]

We first describe the formal construction of feed-forward network \\({\tt FF}_1\\) for dimensions <span class="smallcaps">sum</span> and <span class="smallcaps">is_eos</span> and then explain the intuition behind the construction. For the example result of applying the feed-forward network is presented in  
eftab:addition_ff.

### Subnetwork 1: Construction for <span class="smallcaps">sum</span> (dimension 7–16).

Given a vector \\({\bm{y}}= [{\bm{y}}_j]_{j=1}^d \in \mathbb{R}^d\\), define a linear function \\(g: \mathbb{R}^d \rightarrow \mathbb{R}\\) as \\[\begin{aligned}
    g({\bm{y}}) := {\bm{y}}_{\textsc{pre\_sum}} + \frac{{\bm{y}}_{\textsc{pre\_carry}} - 2 {\bm{y}}_{\textsc{num}}}{10} + 0.21 = {\bm{y}}_{3} + \frac{{\bm{y}}_{4} - 2 {\bm{y}}_{1}}{10} + 0.21
\end{aligned}\\] and consider a one-hidden-layer ReLU network \\(f_k : \mathbb{R}\rightarrow \mathbb{R}\\) (\\(k=0, 1, \dots, 9\\)) defined as \\[\begin{aligned}
    \begin{aligned}
        f_k(x) &= 2 \Big[ \phi(x-(k-0.5)) - \phi(x-k) - \phi(x-(k+0.5)) + \phi(x-(k+1)) \\
        &\phantom{= 2 []} + \phi(x-(k+9.5)) - \phi(x-(k+10)) - \phi(x-(k+10.5)) + \phi(x-(k+11)) \Big].
    \end{aligned}
    \label{eq:construction_addition_f_k}
\end{aligned}\\] Then we construct a subnetwork of our feed-forward network for a token \\(\sigma_j\\) by \\[\begin{aligned}
    \left[{\tt FF}_1 \left({\bm{Y}}^{(1)}\right)\right]_{(\textsc{sum})j} = \begin{bmatrix}
        f_0\left(g\left({\bm{Y}}^{(1)}_{\bullet j}\right)\right) & \cdots & f_9\left(g\left({\bm{Y}}^{(1)}_{\bullet j}\right)\right)
    \end{bmatrix}^\top. \label{eq:construction_addition_SUM}
\end{aligned}\\]

<figure id="fig:construction_addition_f_k">
<img src="./figures/proof_f_k.png"" style="width:80.0%" />
<figcaption>Example plots of <span class="math inline"><em>f</em><sub><em>k</em></sub>(<em>x</em>)</span> defined in<br />
ef<span>eq:construction_addition_f_k</span>. (<span class="math inline"><em>k</em> = 0, 1, 2, 3</span>)</figcaption>
</figure>

#### Explanation.

The purpose of the first subnetwork is to generate a 10-dimensional one-hot vector whose position of 1 indicates the next digit: \\({\bm{e}}^{10}_k\\) for the answer of next-token prediction ‘\\(k\\)’. There are two cases where we need to predict the next token as ‘\\(k\\)’:

- Case 1: (Addition without carry) \\(=k 
      \mod 10\\) and no carry propagates.

- Case 2: (Addition without carry) \\(=k-1  \mod 10\\) and there is a propagating carry 1.

In the first case, due to the softmax error (with magnitude at most \\(0.1\\)), \\[\begin{aligned}
    {\bm{Y}}^{(0)}_{(\textsc{pre\_sum})j} &\in [k-0.1, k+0.1] \cap [k+9.9, k+10.1], \\
    {\bm{Y}}^{(0)}_{(\textsc{pre\_carry})j} - 2{\bm{Y}}^{(0)}_{(\textsc{num})j} &\in [-1.1, -0.9] \cap [-0.1, 0.1] \subset [-1.1, 0.1].
\end{aligned}\\] In the second case, again due to the softmax error (with magnitude at most \\(0.1\\)), \\[\begin{aligned}
    {\bm{Y}}^{(0)}_{(\textsc{pre\_sum})j} + 1 &\in [k-0.1, k+0.1] \cap [k+9.9, k+10.1], \\
    {\bm{Y}}^{(0)}_{(\textsc{pre\_carry})j} - 2{\bm{Y}}^{(0)}_{(\textsc{num})j} -10 &\in [-1.1, -0.9] \cap [-0.1, 0.1] \subset [-1.1, 0.1].
\end{aligned}\\] In both cases, \\[\begin{aligned}
    {\bm{Y}}^{(0)}_{(\textsc{pre\_sum})j} + \frac{{\bm{Y}}^{(0)}_{(\textsc{pre\_carry})j} - 2{\bm{Y}}^{(0)}_{(\textsc{num})j}}{10} + 0.21 &\in [k, k+0.32] \cap [k+10, k+10.32] \\
    &\subset [k, k+0.5] \cap [k+10, k+10.5].
    \label{eq:construction_addition_exclusive_set}
\end{aligned}\\] We can map the column \\({\bm{Y}}^{(0)}_{\bullet j}\\) to the set \\([k, k+0.5] \cap [k+10, k+10.5]\\) if the next token is \\(\sigma_{j+1}=\\) ‘\\(k\\)’. This job is done by the function \\(g\\). Note that the resulting sets \\([k, k+0.5] \cap [k+10, k+10.5]\\) are disjoint for different \\(k\\)’s.

Recall that our objective is to output 1 to the dimension \\(k+6\\) (among the dimensions 7, 8, …, 16 in <span class="smallcaps">sum</span>) and to output 0 to the other dimensions in <span class="smallcaps">sum</span> if we need to predict ‘\\(k\\)’ as the next token. To this end, it is enough to map the set \\([k, k+0.5] \cap [k+10, k+10.5]\\) to 1 and to map the other sets (for different \\(k\\)’s) to 0. This can be done by a ReLU network \\(f_k(x)\\) is a ReLU network having two bumps at intervals \\([k-0.5, k+1]\\) and \\([k+9.5, k+11]\\). In particular, \\(f_k(x)=1\\) if \\(x\in[k, k+0.5] \cup [k+10, k+10.5]\\): see  
effig:construction_addition_f_k for an illustration.

Lastly, we have a desired one-hot vector output for each \\(j\\) by taking a composition between \\(g\\) and \\([f_0(\cdot), \dots, f_9(\cdot)]^\top\\) as written in  
efeq:construction_addition_SUM.

### Subnetwork 2: Construction for <span class="smallcaps">is_eos</span> (dimension 17).

We move on to the dimension <span class="smallcaps">is_eos</span>. For a token \\(\sigma_j\\) for \\(j \in \{2\ell+3,\ldots,3\ell+4\}\\), if \\(k\\) is the next token, we will put \\(\mathbbm{1}_{\left\{k=\$\right\}}\\) to dimension <span class="smallcaps">is_eos</span>: \\(1\\) if \\(k\\) is the special token ‘\\(\$\\)’ and \\(0\\) otherwise. To this end, we define a ReLU network \\(h:\mathbb{R}\rightarrow\mathbb{R}\\) as \\[\begin{aligned}
    h(x) = 10 \phi\left(x - 0.8\right) - 10  \phi\left(x - 0.9\right).
\end{aligned}\\] Then, we can construct a subnetwork of our feed-forward network for a token \\(\sigma_j\\) by \\[\begin{aligned}
    \left[{\tt FF}_1 \left({\bm{Y}}^{(1)}\right)\right]_{(\textsc{is\_eos})j} = h\left({\bm{Y}}^{(1)}_{(\textsc{pre\_eos})j}\right).
\end{aligned}\\]

#### Explanation.

Note that for columns \\(j \in \{2\ell+3,\ldots,3\ell+4\}\\), if we consider the presence of softmax errors with magnitude at most 0.1, the values that \\({\bm{Y}}^{(1)}_{(\textsc{pre\_eos})j}\\) can have lie in the set \\([0.4, 0.6] \cap [2/3-0.1, 2/3+0.1] \cap [0.9, 1.1] \subset (-\infty, 0.8)\cap[0.9, \infty)\\). We want to output 1 if \\({\bm{Y}}^{(1)}_{(\textsc{pre\_eos})j}\ge 0.9\\) and 0 otherwise: this can be done with the ReLU network \\(h\\) with two neurons.

#### Remarks:

- In total, we consume \\(8\times10 + 2 = 82\\) ReLU neurons in our feed-forward network \\({\tt FF}_1\\). However, it is possible to construct the addition Transformer with a smaller number of neurons, with a slight modification in the linear readout of the decoding function (  
  efsubsec:construction_addition_decoding).

- Unlike in the attention layer, now we do not have to worry about softmax errors in the output since the feed-forward ReLU network plays the role of *denoiser*.

<div id="tab:addition_ff" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">pre_sum</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">pre_carry</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">pre_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7–16: <span class="smallcaps">sum</span> | \\({\bm{e}}^{10}_{1}\\) | \\({\bm{e}}^{10}_{1}\\) | \\({\bm{e}}^{10}_{10}\\) | \\({\bm{e}}^{10}_{8}\\) | \\({\bm{e}}^{10}_{5}\\) | \\({\bm{e}}^{10}_{2}\\) | \\({\bm{e}}^{10}_{7}\\) | \\({\bm{e}}^{10}_{10}\\) | \\({\bm{e}}^{10}_{3}\\) | \\({\bm{e}}^{10}_{1}\\) | \\({\bm{e}}^{10}_{8}\\) | \\({\bm{e}}^{10}_{1}\\) | \\({\bm{e}}^{10}_{1}\\) |
| 17: <span class="smallcaps">is_eos</span> | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 18–\\((P+17)\\): <span class="smallcaps">pos_1</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) |
| \\((P+18)\\)-\\((2P+17)\\): <span class="smallcaps">pos_2</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) |

Example output after applying the feed-forward network.

</div>

### Residual Connection [residual-connection]

The last task of the feed-forward layer is to pass \\({\tt FF}_1\left({\bm{Y}}^{(1)}\right)\\) through the residual connection. As a result, we have \\[\begin{aligned}
    {\bm{X}}^{(1)} = {\bm{Y}}^{(1)} + {\tt FF}_1\left({\bm{Y}}^{(1)}\right).
\end{aligned}\\]

A concrete example of the output of the second residual connection is showcased in  
eftab:addition_res_conn_ff.

<div id="tab:addition_res_conn_ff" markdown="1">

| \\({\mathcal{I}}\\) | $ | 6 | 5 | 3 | \+ | 0 | 4 | 9 | = | 2 | 0 | 7 | 0 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 6 | 5 | 3 | 0 | 0 | 4 | 9 | 0 | 2 | 0 | 7 | 0 |
| 2: <span class="smallcaps">is_bos</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">full_ones</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 4: <span class="smallcaps">pre_sum</span> | 0 | 0 | 9 | 7.5 | 4.5 | 0 | 6 | 9 | 12 | 9 | 6 | 0 | 0 |
| 5: <span class="smallcaps">pre_carry</span> | 0 | 12 | 10 | 6 | 0 | 8 | 12 | 16 | 0 | 14 | 9 | 13 | 0 |
| 6: <span class="smallcaps">pre_eos</span> | 2 | 1 | 1 | 1 | 1 | 2/3 | 2/3 | 2/3 | 2/3 | 1/2 | 1/2 | 1/2 | 1 |
| 7–16: <span class="smallcaps">sum</span> | \\({\bm{e}}^{10}_{1}\\) | \\({\bm{e}}^{10}_{1}\\) | \\({\bm{e}}^{10}_{10}\\) | \\({\bm{e}}^{10}_{8}\\) | \\({\bm{e}}^{10}_{5}\\) | \\({\bm{e}}^{10}_{2}\\) | \\({\bm{e}}^{10}_{7}\\) | \\({\bm{e}}^{10}_{10}\\) | \\({\bm{e}}^{10}_{3}\\) | \\({\bm{e}}^{10}_{1}\\) | \\({\bm{e}}^{10}_{8}\\) | \\({\bm{e}}^{10}_{1}\\) | \\({\bm{e}}^{10}_{1}\\) |
| 17: <span class="smallcaps">is_eos</span> | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 18–\\((P+17)\\): <span class="smallcaps">pos_1</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) |
| \\((P+18)\\)-\\((2P+17)\\): <span class="smallcaps">pos_2</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) |

Example output of residual connection, continuing from  
eftab:addition_ff. Here we ignore the softmax error terms in the orange rows.

</div>

## Decoding Function [subsec:construction_addition_decoding]

As mentioned in  
efsec:Transformer_architecture, the decoding function performs a linear readout (with a weight matrix \\({\bm{W}}_{\rm out}\in \mathbb{R}^{\left|{\mathcal{V}}\right|\times d}\\)) and a (token-wise) arg-max operation. That is, \\[\begin{aligned}
    {\tt Dec}\left({\bm{X}}^{(1)}\right) &:= \left({\mathcal{V}}_{k_i}\right)_{i=1,\ldots,N} \in {\mathcal{V}}^N,
\end{aligned}\\] where \\({\mathcal{V}}_k\\) is the \\(k\\)-th element of \\({\mathcal{V}}\\) and \\[\begin{aligned}
    k_i := \mathop{\mathrm{arg\,max}}_{k\in \left[\left|{\mathcal{V}}\right|\right]} \left\{o_k : {\bm{W}}_{\rm out} {\bm{X}}^{(1)}_{\bullet i}= \begin{bmatrix}
        o_1 & \cdots & o_{\left|{\mathcal{V}}\right|}
    \end{bmatrix}^\top\right\}.
\end{aligned}\\]

The objective of the decoding function is to perform a proper next-token prediction for addition, especially utilizing the dimensions <span class="smallcaps">sum</span> and <span class="smallcaps">is_eos</span> of \\({\bm{X}}^{(1)}\\).

We now construct the weight matrix \\({\bm{W}}_{\rm out}\\). For a token \\(\sigma_i\\), if the value of dimension <span class="smallcaps">is_eos</span> of \\({\bm{X}}^{(1)}\\) is 0, then the linear readout output the dimensions <span class="smallcaps">sum</span> as it is to return one of a number token (0-9). On the other hand, if the value of dimension <span class="smallcaps">is_eos</span> is 1, then the linear readout outputs a large number (like 100 for example) for the token ‘\\(\$\\)’ to return EOS (\\(\$\\)). This can be implemented by the weight matrix \\({\bm{W}}_{\rm out}\\) described in  
eftab:addition_Wout. Also, an example of applying the linear transform is showcased in  
eftab:addition_linear_readout.

<div id="tab:addition_Wout" markdown="1">

| \\({\mathcal{V}}\\) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | \+ | = | $ |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1-6: <span class="smallcaps">num</span>-<span class="smallcaps">pre_eos</span> | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) | \\({\bm{0}}_{6}\\) |
| 7: <span class="smallcaps">sum</span>\\(_1\\) | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8: <span class="smallcaps">sum</span>\\(_2\\) | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9: <span class="smallcaps">sum</span>\\(_3\\) | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10: <span class="smallcaps">sum</span>\\(_4\\) | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11: <span class="smallcaps">sum</span>\\(_5\\) | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12: <span class="smallcaps">sum</span>\\(_6\\) | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13: <span class="smallcaps">sum</span>\\(_7\\) | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 14: <span class="smallcaps">sum</span>\\(_8\\) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 15: <span class="smallcaps">sum</span>\\(_9\\) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 16: <span class="smallcaps">sum</span>\\(_{10}\\) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 17: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| 18–end: <span class="smallcaps">pos_1</span>, <span class="smallcaps">pos_2</span> | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) | \\({\bm{0}}_{2P}\\) |

The *transposed* weight matrix \\({\bm{W}}_{out}^\top\\) of the linear readout in decoding function.

</div>

<div id="tab:addition_linear_readout" markdown="1">

| \\({\mathcal{I}}\\) |  $  |  6  |  5  |  3  | \+  |  0  |  4  |  9  |  =  |  2  |  0  |  7  |  0  |
|:--------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0                   |  1  |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  1  |  0  |  1  |  1  |
| 1                   |  0  |  0  |  0  |  0  |  0  |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
| 2                   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  1  |  0  |  0  |  0  |  0  |
| 3                   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
| 4                   |  0  |  0  |  0  |  0  |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
| 5                   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
| 6                   |  0  |  0  |  0  |  0  |  0  |  0  |  1  |  0  |  0  |  0  |  0  |  0  |  0  |
| 7                   |  0  |  0  |  0  |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  1  |  0  |  0  |
| 8                   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
| 9                   |  0  |  0  |  1  |  0  |  0  |  0  |  0  |  1  |  0  |  0  |  0  |  0  |  0  |
| \+                  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
| =                   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
| $                   | 100 | 100 | 100 | 100 | 100 |  0  |  0  |  0  |  0  |  0  |  0  |  0  | 100 |

Example output of linear readout (\\({\bm{W}}_{\rm out} {\bm{X}}^{(1)}\\)), continuing from  
eftab:addition_res_conn_ff,tab:addition_Wout. The yellow cells represent the maximum value of each column, from the ‘=’ token’s column to the rightmost column (used for next-token prediction).

</div>

<div id="tab:addition_output_sequence" markdown="1">

| \\({\mathcal{I}}\\) |  $  |  6  |  5  |  3  | \+  |  0  |  4  |  9  |  =  |  2  |  0  |  7  |  0  |
|:-------------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| \\({\mathcal{O}}\\) |  $  |  $  |  $  |  $  |  $  |  1  |  6  |  9  |  2  |  0  |  7  |  0  |  $  |

Example output sequence \\({\mathcal{O}}= {\tt Dec}\left({\bm{X}}^{(1)}\right)\\), continuing from  
eftab:addition_linear_readout. The yellow cells in the bottom row exactly predict the next tokens.

</div>

# Impossibility of Addition with No Positional Encoding [sec:impossibilityaddition]

For the sake of readability, we restate the proposition below.

#### Remark.

We assume the 1-layer (\\(L=1\\)) \\(H\\)-head Transformer achitecture specified in  
efsec:Transformer_architecture. Although it omits normalization layers, we remark that  
efprop:thmadditionnope remains valid even for the architecture with a standard layer normalization `\citep{ba2016layer}`{=latex} or its variants (e.g., `\citealp{zhang2019root}`{=latex}).

<div class="proof" markdown="1">

*Proof.* We keep following the notation about matrices introduced in  
efsubsec:construction_addition_notation. Throughout the proof, we denote the value/vector/matrix related to \\({\mathcal{I}}'\\) by appending ‘\\('\\)’ to it.

Let encoding matrices generated from the input sequences \\({\mathcal{I}}, {\mathcal{I}}^\prime\in{\mathcal{V}}^N\\) as \\[\begin{aligned}
        {\bm{X}}:= {\tt Enc}({\mathcal{I}}) \in \mathbb{R}^{d\times N} \quad \text{and} \quad  {\bm{X}}' := {\tt Enc}({\mathcal{I}}') \in \mathbb{R}^{d\times N}.
    
\end{aligned}\\] Since there is no positional encoding, the encoding function \\({\tt Enc}(\cdot)\\) maps the same tokens to the same columns. In particular, \\({\mathcal{I}}_i = {\mathcal{I}}'_j\\) implies \\({\bm{X}}_{\bullet i} = {\bm{X}}'_{\bullet j}\\). Since we assume that \\({\mathcal{I}}'\\) is a permutation of \\({\mathcal{I}}\\) such that \\({\mathcal{I}}_N = {\mathcal{I}}'_N\\), there exists a bijection \\(\pi: [N] \rightarrow [N]\\) such that \\({\mathcal{I}}'_i = {\mathcal{I}}_{\pi(i)}\\) for each \\(i \in [N]\\) and \\(\pi(N)=N\\). Then, it follows that \\({\bm{X}}'_{\bullet i}={\bm{X}}_{\bullet (\pi(i))}\\) for each \\(i\\) and, specifically, \\({\bm{X}}'_{\bullet N}={\bm{X}}_{\bullet N}\\).

Recall that the single \\(H\\)-head attention layer \\({\tt Att}: \mathbb{R}^{d \times N} \rightarrow \mathbb{R}^{d \times N}\\) operates as \\({\tt Att}({\bm{X}}) = \sum_{h=1}^H {\tt Head}_h ({\bm{X}})\\) where the attention head \\(h\\) is defined as \\[\begin{aligned}
        {\tt Head}_h ({\bm{X}}) := {\bm{U}}_h {\bm{V}}_h {\bm{X}}\cdot{\tt softmax}\left(({\bm{K}}_h {\bm{X}})^\top {\bm{Q}}_h {\bm{X}}\right) \in \mathbb{R}^{d \times N},
    
\end{aligned}\\] where \\({\bm{Q}}_h, {\bm{K}}_h \in \mathbb{R}^{d_{QK} \times d}\\), \\({\bm{V}}_h \in \mathbb{R}^{d_V \times d}\\) and \\({\bm{U}}_h \in \mathbb{R}^{d \times d_{V}}\\).

#### Claim:

\\(\left[{\tt Head}_h({\bm{X}})\right]_{\bullet N} = \left[{\tt Head}_h({\bm{X}}')\right]_{\bullet N}\\) for all \\(h\in [H]\\).

The claim suffices to prove the proposition because of the following: first, the claim implies that the last (\\(N\\)-th) columns of the attention layer outputs are the same, i.e., \\(\left[{\tt Att}({\bm{X}})\right]_{\bullet N} = \left[{\tt Att}({\bm{X}}')\right]_{\bullet N}\\). Note that the operations after the attention layer—residual connections, \\({\tt FF}\\), and \\({\tt Dec}\\)—all operate in a token-wise (column-by-column) manner: the \\(j\\)-th column of the output of a token-wise operation is a function of \\(j\\)-th column of the input for the operation. Therefore, the last column of the attention layer output totally determines the next-token prediction at \\(N\\)-th input token. As a result, the predicted next-tokens are the same for \\({\mathcal{I}}\\) and \\({\mathcal{I}}'\\).

The rest of the proof is devoted to proving the aforementioned claim. Fix any \\(h\in [H]\\). Let \\[\begin{aligned}
        \left[{\tt softmax}\left(({\bm{K}}_h {\bm{X}})^\top {\bm{Q}}_h {\bm{X}}\right)\right]_{\bullet N}&=\begin{bmatrix} s_1 & \dots & s_N \end{bmatrix}^\top, \\
        \left[{\tt softmax}\left(({\bm{K}}_h {\bm{X}}')^\top {\bm{Q}}_h {\bm{X}}'\right)\right]_{\bullet N}&=\begin{bmatrix} s'_1 & \dots & s'_N \end{bmatrix}^\top,
    
\end{aligned}\\] which are both stochastic (sum to 1) column vectors. Considering that we are taking the last column of the softmax output, it follows that \\(s_{i}' = s_{\pi(i)}\\) for each \\(i\in [N]\\): this can be proved by applying the definition of the softmax operation and the fact \\[\begin{aligned}
        \left[({\bm{K}}_h {\bm{X}}')^\top {\bm{Q}}_h {\bm{X}}'\right]_{iN} &= {\bm{X}}'^\top_{\bullet i} {\bm{K}}_h^\top {\bm{Q}}_h {\bm{X}}'_{\bullet N} 
        = {\bm{X}}^\top_{\bullet \pi(i)} {\bm{K}}_h^\top {\bm{Q}}_h {\bm{X}}_{\bullet N}
        = \left[({\bm{K}}_h {\bm{X}})^\top {\bm{Q}}_h {\bm{X}}\right]_{(\pi(i))N}.
    
\end{aligned}\\] Consequently, since \\[\begin{aligned}
        \sum_{i=1}^N s_i' {\bm{X}}'_{\bullet i} = \sum_{i=1}^N s_{\pi(i)} {\bm{X}}_{\bullet (\pi(i))} = \sum_{i=1}^N s_{i} {\bm{X}}_{\bullet i},
    
\end{aligned}\\] we have \\[\begin{aligned}
        {\bm{X}}' \cdot \left[{\tt softmax}\left(({\bm{K}}_h {\bm{X}}')^\top {\bm{Q}}_h {\bm{X}}'\right)\right]_{\bullet N} = {\bm{X}}\cdot \left[{\tt softmax}\left(({\bm{K}}_h {\bm{X}})^\top {\bm{Q}}_h {\bm{X}}\right)\right]_{\bullet N}.
    
\end{aligned}\\] Therefore, the claim holds. This concludes the proof. ◻

</div>

Here, we provide the Python code that calculates the maximum possible exact-match accuracy that a 1-layer Transformer with NoPE can achieve for the \\(m\\)-digit addition problem.

``` python
from itertools import product
from collections import defaultdict 

m = 4  # Change here
total = 0
counter_dict = defaultdict(dict)

for a, b in product(product(range(10), repeat=m), product(range(10), repeat=m)):
    if a[0] == 0 or b[0] == 0: continue
    total += 1
    c = tuple(sorted(a+b))
    a_num = int(''.join(map(str, a)))
    b_num = int(''.join(map(str, b)))
    ab_sum = a_num + b_num
    if ab_sum in counter_dict[c]:
        counter_dict[c][ab_sum] += 1
    else:
        counter_dict[c][ab_sum] = 1

count = sum(max(d.values()) for _, d in counter_dict.items())    

print("m =", m)
print("Permutation Invariant Additions Count:", count)
print("        Total m-digit Additions Count:", total)
print("                                Ratio:", count / total)

"""
[Example Outputs]

m = 1
Permutation Invariant Additions Count: 81
        Total m-digit Additions Count: 81
                                Ratio: 1.0
m = 2
Permutation Invariant Additions Count: 2668
        Total m-digit Additions Count: 8100
                                Ratio: 0.32938271604938274
m = 3
Permutation Invariant Additions Count: 50150
        Total m-digit Additions Count: 810000
                                Ratio: 0.06191358024691358
m = 4
Permutation Invariant Additions Count: 765139
        Total m-digit Additions Count: 81000000
                                Ratio: 0.00944616049382716
m = 5
Permutation Invariant Additions Count: 10033314
        Total m-digit Additions Count: 8100000000
                                Ratio: 0.0012386807407407407
"""
```

# (Formal) Construction of \\(N\times2\\) Multiplication Transformer with Position Coupling [sec:construction_Nx2]

Here we show how to implement the \\(N\times2\\) multiplication using a depth-2 decoder-only Transformer equipped with position coupling. Our construction involves 3 heads in the first Transformer block and 7 heads in the second Transformer block, requiring a total of 10 heads.

We note that our construction for the \\(N\times2\\) multiplication task permits the use of multiple FFN layers at the second decoder block. However, we believe that there exists a potential improvement in our construction, wherein a single FFN layer could suffice for each decoder block, leveraging the expressivity of the neural network. Additionally, we do not provide a detailed error analysis but assume that the softmax operation with sufficiently large attention weights can reduce small attention scores to zero values, thereby clearly revealing the desired attention patterns.

## Notation [notation]

Consider an ordered vocabulary \\({\mathcal{V}}=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, \times, =, \$)\\). We include a special token ‘\\(\$\\)’ that plays the role of both the beginning-of-sequence (BOS) token and the end-of-sequence (EOS) token. We denote \\({\mathcal{V}}_k\\) as \\(k\\)-th element of \\({\mathcal{V}}\\). For instance, \\({\mathcal{V}}_4=3\\) and \\({\mathcal{V}}_{13}=\$\\). Unlike the addition task, our construction for the multiplication involves multiple layers and hence we do not omit the superscripts \\((l)\\) in the parameter matrices/vectors and the size of dimensions.

## Input Sequence [input-sequence]

Our objective is to use next-token prediction for implementing \\(a\times b = c\\). To this end, we want to transform it into an input sequence \\({\mathcal{I}}= \overline{\$A \times B=C}\\) of an appropriate format. Let \\(\ell_a\\) and \\(\ell_b\\) represent the lengths of \\(a\\) and \\(b\\), respectively, and we denote their sum as \\(\ell = \ell_a + \ell_b\\). While our immediate focus is on the case where \\(\ell_b = 2\\), it is worth noting that our approach can be extended to the case where \\(\ell_b > 2\\), as the key insight for the construction does not rely on \\(\ell_b\\). Thus, we present the input sequence and encoding function in a more general form applicable to \\(\ell_b \geq 2\\).

Unlike the addition case, we do not zero-pad both \\(a\\) and \\(b\\). Instead, we only zero-pad the response, as the length of \\(c\\) may either equal the sum of the lengths of \\(a\\) and \\(b\\), or be less than the sum of their lengths by \\(1\\). Hence, we zero-pad in front of \\(c\\) for the latter case to fix the length of \\(c\\) by \\(\ell\\). We also reverse the response \\(c\\) to make the part \\(C\\). For instance, if we have \\(312 \times 24 = 7488\\), the input sequence transforms to \\(\overline{\$ 312 \times 24 = \textcolor{DodgerBlue3}{8847}\textcolor{Firebrick3}{0}}\\). If we have \\(589 \times 62 = 36518\\), then the input sequence would be \\(\overline{\$ 589 \times 62 = \textcolor{DodgerBlue3}{81563}}\\). The red digit is a zero-padding, and the blue digits are the reversed product.

To recap, the input sequence \\({\mathcal{I}}=\overline{\sigma_1\sigma_2\ldots\sigma_{N}}\in {\mathcal{V}}^N\\) of length \\(N=2\ell+3\\) consists of six parts:

1.  the BOS token \\(\sigma_1 = \text{`$\$$'}\\)

2.  the first operand \\(A=\overline{\sigma_2\ldots\sigma_{\ell_a+1}}\\) where \\(\sigma_i \in \{0, \ldots, 9\}\\);

3.  the multiplication symbol \\(\sigma_{\ell_a+2}=\\) ‘\\(\times\\)’;

4.  the second operand \\(B=\overline{\sigma_{\ell_a+3}\ldots\sigma_{\ell+2}}\\) (note that \\(\ell = \ell_a + \ell_b\\)) where \\(\sigma_i \in \{0, \ldots, 9\}\\);

5.  the equality symbol \\(\sigma_{\ell+3}=\\) ‘\\(=\\)’;

6.  the (reversed) product \\(C=\overline{\sigma_{\ell+4}\ldots\sigma_{2\ell+3}}\\) where \\(\sigma_i \in \{0, \ldots, 9\}\\).

Note that the part \\(C\\) might be incomplete (i.e., \\(N<2\ell+3\\)) at the inference time; we infer the digits of the part \\(C\\) one by one using next-token prediction. Throughout this section on a formal construction, however, we only consider the train time setup in which we infer all the digits of the part \\(C\\) at once using *simultaneous* next-token prediction in a single forward pass. Precisely, we want to use an input sequence \\({\mathcal{I}}=\overline{\sigma_1\ldots\sigma_{N}}\\) to produce an output sequence \\({\mathcal{O}}=\overline{\sigma'_1\ldots\sigma'_{N}}\\) where \\(\overline{\sigma'_{\ell+3}\ldots\sigma'_{N-1}} = C = \overline{\sigma_{\ell+4}\ldots\sigma_{N}}\\) and \\(\sigma'_{N}=\\) ‘\\(\$\\)’ (EOS).

## Encoding Function [encoding-function]

We now explain the input embedding for given an input sequence \\({\mathcal{I}}\\) designed as above. The embedding matrix \\({\bm{X}}^{(0)}\\) is of size \\(d\times N\\): each column represents an embedding vector for a token, while each row represents a particular *named* dimension. We concatenate the token embedding and the position embedding, which can be viewed as a *sum* of two different embedding matrices of the same size.

<div id="tab:init_emb_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 7 | 5 | 9 | 5 | 0 | 7 | 9 | 0 | 5 | 0 | 0 | 0 | 0 | 6 |
| 2: <span class="smallcaps">full_ones</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3: <span class="smallcaps">is_bos</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">is_op2_one</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7: <span class="smallcaps">is_op2_ten</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8: <span class="smallcaps">op2_one</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9: <span class="smallcaps">op2_ten</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10: <span class="smallcaps">op1_shift0</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11: <span class="smallcaps">op1_shift1</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12: <span class="smallcaps">op1_shift2</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13: <span class="smallcaps">op1_shift3</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 14: <span class="smallcaps">op1_shift4</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 15: <span class="smallcaps">result1</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16: <span class="smallcaps">result2</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17: <span class="smallcaps">result3</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18: <span class="smallcaps">result4</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 19: <span class="smallcaps">pre_prod</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20: <span class="smallcaps">pre_carry</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 21: <span class="smallcaps">pre_eos1</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 22: <span class="smallcaps">pre_eos2</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 23-32: <span class="smallcaps">prod</span> | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) |
| 33: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 34: <span class="smallcaps">mask</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 35–(\\(P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2_mask</span> | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| (\\(P+\\)`<!-- -->`{=html}35)–(\\(2P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_1</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) | \\({\bm{v}}^P_{1}\\) |
| (\\(2P+\\)`<!-- -->`{=html}35)–(\\(3P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) |
| (\\(3P+\\)`<!-- -->`{=html}35)–(\\(4P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_3</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) |
| (\\(4P+\\)`<!-- -->`{=html}35)–(\\(5P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_4</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) |
| (\\(5P+\\)`<!-- -->`{=html}35)–(\\(6P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_5</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) |

Example initial encoding. Here we consider the input sequence \\(\overline{\$7595 \times 79 = 500006}\\) and the starting position ID is chosen as \\(s=1\\). The vectors \\({\bm{v}}^P_{\square}\\) are defined in  
efeq:vvNx2. The gray rows will be filled in later.

</div>

### Token Embedding

The token embedding consists of \\((34 + P)\\) dimensions, where \\(P\\) represents the dimension for the position embedding which will be described in the very next section. While the token embedding dimension for the addition task was independent of \\(P\\), our construction strategy for the multiplication task involves copying the position embedding into the token embedding. This is why we have the \\(P\\) term in our token embedding dimension. For the first \\(34\\) dimensions, we label them as:

<div class="center" markdown="1">

1=<span class="smallcaps">num</span>, 2=<span class="smallcaps">full_ones</span>, 3=<span class="smallcaps">is_bos</span>, 4=<span class="smallcaps">is_mul</span>, 5=<span class="smallcaps">is_equal</span>,

6=<span class="smallcaps">is_op2_one</span>, 7=<span class="smallcaps">is_op2_ten</span>, 8=<span class="smallcaps">op2_one</span>, 9=<span class="smallcaps">op2_ten</span>,

10=<span class="smallcaps">op1_shift0</span>, 11=<span class="smallcaps">op1_shift1</span>, 12=<span class="smallcaps">op1_shift2</span>, 13=<span class="smallcaps">op1_shift3</span>, 14=<span class="smallcaps">op1_shift4</span>,

15=<span class="smallcaps">result1</span>, 16=<span class="smallcaps">result2</span>, 17=<span class="smallcaps">result3</span>, 18=<span class="smallcaps">result4</span>,

19=<span class="smallcaps">pre_prod</span>, 20=<span class="smallcaps">pre_carry</span>, 21=<span class="smallcaps">pre_eos1</span>, 22=<span class="smallcaps">pre_eos2</span>

{23,...,32}=<span class="smallcaps">prod</span>, 33=<span class="smallcaps">is_eos</span>, 34=<span class="smallcaps">mask</span>,

</div>

and for the last \\(P\\) dimensions (\\(\{35,...,34+P\}\\)), we named them as <span class="smallcaps">pos_2_mask</span>.

The initial token embedding fills only <span class="smallcaps">num</span>, <span class="smallcaps">full_ones</span>, <span class="smallcaps">is_bos</span>, <span class="smallcaps">is_mul</span>, and <span class="smallcaps">is_equal</span>, leaving the other \\((29+P)\\) dimensions empty (i.e., all zeros). These \\((29+P)\\) dimensions will be filled by passing through the layers. Here we describe how we fill the first \\(5\\) dimensions.

#### Dimension 1 (<span class="smallcaps">num</span>).

For a number token (\\(0,\ldots,9\\)), we put itself into the dimension <span class="smallcaps">num</span>. For the other tokens (\\(\times,=,\$\\)), we put \\(0\\).

#### Dimension 2 (<span class="smallcaps">full_ones</span>).

We put \\(1\\) everywhere in this dimension.

#### Dimension 3 (<span class="smallcaps">is_bos</span>).

For a special token ‘\\(\$\\)’, we put \\(1\\) into the dimension <span class="smallcaps">is_bos</span>. Otherwise, we put \\(0\\).

#### Dimension 4 (<span class="smallcaps">is_mul</span>).

For a special token ‘\\(\times\\)’, we put \\(1\\) into the dimension <span class="smallcaps">is_mul</span>. Otherwise, we put \\(0\\).

#### Dimension 5 (<span class="smallcaps">is_equal</span>).

For a special token ‘\\(=\\)’, we put \\(1\\) into the dimension <span class="smallcaps">is_equal</span>. Otherwise, we put \\(0\\).

### Coupled Position IDs and Position Embedding

We now specify the allocation of coupled position IDs for the \\(N \times M\\) multiplication task as the following: given an input sequence \\({\mathcal{I}}=\overline{\sigma_1\ldots\sigma_{N}}\\), \\[\begin{aligned}
    p(i) = \begin{dcases}
        0, & i=1, \\
        s+i-2 + \ell_b, & i=2,\ldots,\ell_a+2, \\
        s+i-3, & i=\ell_a+3,\ldots,\ell+3, \\
        s-i+3 + 2\ell & i=\ell+4,\ldots,2\ell+3.
    \end{dcases}
\end{aligned}\\]

Compared to the addition case, the position allocating function \\(p\\) becomes more complicated since the length of two operands can be different, but the core remains simple: coupling the position IDs for the least significant digit in the first operand (\\(A\\)), the second operand (\\(B\\)), and the result (\\(C\\)), and then decreasing the IDs as the digit position increases for each \\(A\\), \\(B\\), and \\(C\\).

Now we explain the position embedding. We utilize the same \\({\bm{v}}^D_k\\) (\\(k\in [2^D]\\)) defined for the addition task, specifically \\[\begin{aligned}
    {\bm{v}}^D_k = \left[(-1)^{b_i^{(D,k)}}\right]_{i=1}^D \in \mathbb{R}^D \label{eq:vvNx2}
\end{aligned}\\] where \\(b_i^{(D,k)}\\) is defined as the \\(i\\)-th (from left) digit of \\(D\\)-digit binary representation of \\(k-1\\). Using \\({\bm{v}}^D_k\\), we design the position embedding for each position ID \\(p(i)\\) by \\[\begin{aligned}
    \begin{bmatrix}
        {\bm{v}}^P_{p(i)} \\ {\bm{v}}^P_{p(i)+1} \\ {\bm{v}}^P_{p(i)+2} \\ {\bm{v}}^P_{p(i)+3} \\ {\bm{v}}^P_{p(i)+4}
    \end{bmatrix}.
\end{aligned}\\] The first \\(P\\) dimensions of the position embedding are named as <span class="smallcaps">pos_1</span>, and subsequent sets of \\(P\\) dimensions are named as <span class="smallcaps">pos_2</span>, <span class="smallcaps">pos_3</span>, <span class="smallcaps">pos_4</span>, and <span class="smallcaps">pos_5</span>, respectively. Thus, the position embedding is a \\(5P\\)-dimensional vector. In case of \\(p(i)+j\\) (\\(j \in [4]\\)) exceeding \\(2^P\\), we use \\({\bm{v}}^P_{p(i)+j-2^P}\\) instead of \\({\bm{v}}^P_{p(i)+j}\\). If \\(p(i)=0\\), we let \\({\bm{0}}_{5P}\\) as a position embedding vector.

By concatenating the token embedding and the position embedding, we get the input embedding \\({\bm{X}}^{(0)}\\). Specifically, the position embedding is placed under the token embedding (\\((P+35)\\)-th to \\((6P+34)\\)-th dimension). See  
eftab:init_emb for an example. As a result, the total embedding dimension is \\(d=6P+34\\). Note the maximum possible position ID that can be represented with \\({\bm{v}}^P_k\\)’s is \\({\tt max\_pos}= 2^P = 2^{\left\lfloor (d-34)/6 \right\rfloor}\\). Therefore, the length of the first operand must be \\(\ell_a \le {\tt max\_pos}- \ell_b - 1 = 2^{\left\lfloor (d-34)/6 \right\rfloor}- \ell_b - 1\\). For the case when \\(\ell_b=2\\), this inequality becomes \\(\ell_a \le 2^{\left\lfloor (d-34)/6 \right\rfloor}- 3\\).

## Construction Idea [subsec:construction_idea]

Here, we provide an example that demonstrates how we construct the \\(N \times 2\\) multiplication. Consider the calculation \\(7595 \times 79 = 600005\\). While a typical method for computing such a multiplication is illustrated in  
eftab:mul_method_1, we consider an alternative approach, as shown in  
eftab:mul_method_2. In this method, we pair the digits from the first and second operands at each step where the sum of their digit positions is the same, and then calculate the sum of the pairwise products. For example, the number \\(116\\) in  
eftab:mul_method_2 is generated by \\(\textcolor{DodgerBlue3}{9} \times \textcolor{Firebrick3}{9} + \textcolor{DodgerBlue3}{5} \times \textcolor{Firebrick3}{7}\\), and the number \\(108\\) is generated by \\(\textcolor{DodgerBlue3}{5} \times \textcolor{Firebrick3}{9} + \textcolor{DodgerBlue3}{9} \times \textcolor{Firebrick3}{7}\\), where blue indicates numbers from the first operand and red indicates numbers from the second operand. The main reason for considering such a method is to provide a clearer intuition for determining which numbers from each operand we should attend to when predicting the next token.

<div id="tab:mul_method_1" markdown="1">

|              |     |     |     |     |     |
|:------------:|:---:|:---:|:---:|:---:|:---:|
|              |     |  7  |  5  |  9  |  5  |
| \\(\times\\) |     |     |     |  7  |  9  |
|              |  6  |  8  |  3  |  5  |  5  |
|      5       |  3  |  1  |  6  |  5  |     |
|      6       |  0  |  0  |  0  |  0  |  5  |

Multiplication I

</div>

<span id="tab:mul_method_1" label="tab:mul_method_1"></span>

<div id="tab:mul_method_2" markdown="1">

|              |     |     |     |     |     |
|:------------:|:---:|:---:|:---:|:---:|:---:|
|              |     |  7  |  5  |  9  |  5  |
| \\(\times\\) |     |     |     |  7  |  9  |
|              |     |     |     |  4  |  5  |
|              |     |  1  |  1  |  6  |     |
|              |  1  |  0  |  8  |     |     |
|              |  9  |  8  |     |     |     |
|      4       |  9  |     |     |     |     |
|      6       |  0  |  0  |  0  |  0  |  5  |

Multiplication II

</div>

<span id="tab:mul_method_2" label="tab:mul_method_2"></span>

 

Suppose the current input sequence is \\(\overline{\$7595\times79=5000}\\). During this step, the model is tasked with predicting \\(0\\) (the \\(0\\) just before \\(6\\)) for the next token. As illustrated in  
eftab:mul_method_2, this \\(0\\) is computed from the sum of \\(9\\), \\(9\\), \\(1\\), and an additional \\(1\\), representing the carry from the previous step. Similar to the explanation in <a href="#sec:addition_attn_head2" data-reference-type="ref" data-reference="sec:addition_attn_head2">12.4.2</a>, we highlight that the carry \\(1\\) can be detected by computing \\(8 \, (\text{ones digit of 98}) + 0 \, (\text{tens digit of 108}) + 1 \, (\text{hundreds digit of 116}) - 0 \, (\text{current token})\\): yielding a result of \\(9\\), indicating the occurrence of a carry \\(1\\).

In summary, the correct prediction of the next token \\(0\\) (the \\(0\\) just before \\(6\\)) can be achieved by summing the main summation part and the carry part, where the main summation part is computed using \\(49\\), \\(98\\), \\(108\\), and the carry part is calculated using \\(98\\), \\(108\\), and \\(116\\). Additionally, it’s noteworthy to detail the breakdowns:

- \\(49 = \textcolor{DodgerBlue3}{0} \times \textcolor{Firebrick3}{9} + \textcolor{DodgerBlue3}{7} \times \textcolor{Firebrick3}{7}\\),

- \\(98 = \textcolor{DodgerBlue3}{7} \times \textcolor{Firebrick3}{9} + \textcolor{DodgerBlue3}{5} + \textcolor{Firebrick3}{7}\\),

- \\(108 = \textcolor{DodgerBlue3}{5} \times \textcolor{Firebrick3}{9} + \textcolor{DodgerBlue3}{9} + \textcolor{Firebrick3}{7}\\),

- \\(116 = \textcolor{DodgerBlue3}{9} \times \textcolor{Firebrick3}{9} + \textcolor{DodgerBlue3}{5} + \textcolor{Firebrick3}{7}\\).

Thus, for predicting the next token, we need \\(\textcolor{DodgerBlue3}{0}\\), \\(\textcolor{DodgerBlue3}{7}\\), \\(\textcolor{DodgerBlue3}{5}\\), \\(\textcolor{DodgerBlue3}{9}\\), \\(\textcolor{DodgerBlue3}{5}\\), \\(\textcolor{Firebrick3}{9}\\), \\(\textcolor{Firebrick3}{7}\\). Here, we highlight that this structure, requiring 5 consecutive tokens from the first operand and every token from the second operand for the next-token prediction, remains unchanged for any prediction time and any query length.

As we will see in the later subsection, a depth-2 decoder-only Transformer model can be constructed to fill <span class="smallcaps">op2_one</span> by \\(\textcolor{Firebrick3}{9}\\), <span class="smallcaps">op2_ten</span> by \\(\textcolor{Firebrick3}{7}\\), and <span class="smallcaps">op1_shift0</span> to <span class="smallcaps">op1_shift4</span> by \\(\textcolor{DodgerBlue3}{0}\\), \\(\textcolor{DodgerBlue3}{7}\\), \\(\textcolor{DodgerBlue3}{5}\\), \\(\textcolor{DodgerBlue3}{9}\\), and \\(\textcolor{DodgerBlue3}{5}\\), respectively. One may be concerned that \\(\textcolor{DodgerBlue3}{0}\\) is not given in the first operand at the input sequence. This requirement of \\(\textcolor{DodgerBlue3}{0}\\) beyond the most significant digit arises in the later stage of the prediction, i.e., predicting the token that is near the most significant digit of the response. Although \\(\textcolor{DodgerBlue3}{0}\\) is not explicitly given in the first operand, our construction can automatically manage as if the \\(0\\) were originally at the start of the first operand. A similar situation occurs in the early stage of the prediction that \\(\textcolor{DodgerBlue3}{0}\\) is required before the least significant digit of the first operand, and our construction is also capable of handling this issue.

Consequently, the embedding vector of the current token \\(0\\) (the \\(0\\) preceding \\(60\\)) will be structured as the left-most table in  
eftab:emb_nx2_constructionidea, with some irrelevant dimensions omitted for readability. We then utilize a feed-forward layer to fill

- <span class="smallcaps">result1</span> with \\(\textsc{op1\_shift0} \times \textsc{op2\_one} + \textsc{op1\_shift1} \times \textsc{op2\_ten}\\),

- <span class="smallcaps">result2</span> with \\(\textsc{op1\_shift1} \times \textsc{op2\_one} + \textsc{op1\_shift2} \times \textsc{op2\_ten}\\),

- <span class="smallcaps">result3</span> with \\(\textsc{op1\_shift2} \times \textsc{op2\_one} + \textsc{op1\_shift3} \times \textsc{op2\_ten}\\),

- <span class="smallcaps">result4</span> with \\(\textsc{op1\_shift3} \times \textsc{op2\_one} + \textsc{op1\_shift4} \times \textsc{op2\_ten}\\).

The result is illustrated in the center table of  
eftab:emb_nx2_constructionidea. Next, we employ an additional feed-forward layer to fill

- <span class="smallcaps">pre_prod</span> with \\(\text{ones digit of \textsc{result1}} + \text{tens digit of \textsc{result2}} + \text{hundreds digit of \textsc{result3}}\\),

- <span class="smallcaps">pre_carry</span> with \\(\text{ones digit of \textsc{result2}} + \text{tens digit of \textsc{result3}} + \text{hundreds digit of \textsc{result4}}\\).

These computations yield the result illustrated in the right-most table of  
eftab:emb_nx2_constructionidea. Once this process is done, we can finally predict the next token by the following two steps:

- \\(\textsc{carry} = \begin{cases}
          0, \quad&\text{if } \textsc{pre\_carry} - \textsc{num} \in \{-2, \, -1, \, 0\}, \\
          1, \quad&\text{if } \textsc{pre\_carry} - \textsc{num} \in \{8, \, 9, \, 10\}, \\
          2, \quad&\text{if } \textsc{pre\_carry} - \textsc{num} \in \{18, \, 19, \, 20\},
      \end{cases}\\)

- \\(\textsc{next\_token} = \textsc{pre\_prod} + \textsc{carry} \pmod{10}\\).

<div id="tab:emb_nx2_constructionidea" markdown="1">

| \\({\mathcal{I}}\\) | 0 |
|:---|:--:|
| 1: <span class="smallcaps">num</span> | 0 |
| 2: <span class="smallcaps">full_ones</span> | 1 |
| 3: <span class="smallcaps">is_bos</span> | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 |
| 8: <span class="smallcaps">op2_one</span> | 9 |
| 9: <span class="smallcaps">op2_ten</span> | 7 |
| 10: <span class="smallcaps">op1_shift0</span> | 0 |
| 11: <span class="smallcaps">op1_shift1</span> | 7 |
| 12: <span class="smallcaps">op1_shift2</span> | 5 |
| 13: <span class="smallcaps">op1_shift3</span> | 9 |
| 14: <span class="smallcaps">op1_shift4</span> | 5 |
| 15: <span class="smallcaps">result1</span> | 0 |
| 16: <span class="smallcaps">result2</span> | 0 |
| 17: <span class="smallcaps">result3</span> | 0 |
| 18: <span class="smallcaps">result4</span> | 0 |
| 19: <span class="smallcaps">pre_prod</span> | 0 |
| 20: <span class="smallcaps">pre_carry</span> | 0 |
| (\\(P+\\)`<!-- -->`{=html}35)–(\\(2P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_1</span> | \\({\bm{v}}^P_{3}\\) |
| (\\(2P+\\)`<!-- -->`{=html}35)–(\\(3P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2</span> | \\({\bm{v}}^P_{4}\\) |
| (\\(3P+\\)`<!-- -->`{=html}35)–(\\(4P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_3</span> | \\({\bm{v}}^P_{5}\\) |
| (\\(4P+\\)`<!-- -->`{=html}35)–(\\(5P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_4</span> | \\({\bm{v}}^P_{6}\\) |
| (\\(5P+\\)`<!-- -->`{=html}35)–(\\(6P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_5</span> | \\({\bm{v}}^P_{7}\\) |

Illustration of the construction idea.

</div>

\\(\rightarrow\\)

<div id="tab:emb_nx2_constructionidea" markdown="1">

| \\({\mathcal{I}}\\) | 0 |
|:---|:--:|
| 1: <span class="smallcaps">num</span> | 0 |
| 2: <span class="smallcaps">full_ones</span> | 1 |
| 3: <span class="smallcaps">is_bos</span> | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 |
| 8: <span class="smallcaps">op2_one</span> | 9 |
| 9: <span class="smallcaps">op2_ten</span> | 7 |
| 10: <span class="smallcaps">op1_shift0</span> | 0 |
| 11: <span class="smallcaps">op1_shift1</span> | 7 |
| 12: <span class="smallcaps">op1_shift2</span> | 5 |
| 13: <span class="smallcaps">op1_shift3</span> | 9 |
| 14: <span class="smallcaps">op1_shift4</span> | 5 |
| 15: <span class="smallcaps">result1</span> | 49 |
| 16: <span class="smallcaps">result2</span> | 98 |
| 17: <span class="smallcaps">result3</span> | 108 |
| 18: <span class="smallcaps">result4</span> | 116 |
| 19: <span class="smallcaps">pre_prod</span> | 0 |
| 20: <span class="smallcaps">pre_carry</span> | 0 |
| (\\(P+\\)`<!-- -->`{=html}35)–(\\(2P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_1</span> | \\({\bm{v}}^P_{3}\\) |
| (\\(2P+\\)`<!-- -->`{=html}35)–(\\(3P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2</span> | \\({\bm{v}}^P_{4}\\) |
| (\\(3P+\\)`<!-- -->`{=html}35)–(\\(4P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_3</span> | \\({\bm{v}}^P_{5}\\) |
| (\\(4P+\\)`<!-- -->`{=html}35)–(\\(5P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_4</span> | \\({\bm{v}}^P_{6}\\) |
| (\\(5P+\\)`<!-- -->`{=html}35)–(\\(6P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_5</span> | \\({\bm{v}}^P_{7}\\) |

Illustration of the construction idea.

</div>

\\(\rightarrow\\)

<div id="tab:emb_nx2_constructionidea" markdown="1">

| \\({\mathcal{I}}\\) | 0 |
|:---|:--:|
| 1: <span class="smallcaps">num</span> | 0 |
| 2: <span class="smallcaps">full_ones</span> | 1 |
| 3: <span class="smallcaps">is_bos</span> | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 |
| 8: <span class="smallcaps">op2_one</span> | 9 |
| 9: <span class="smallcaps">op2_ten</span> | 7 |
| 10: <span class="smallcaps">op1_shift0</span> | 0 |
| 11: <span class="smallcaps">op1_shift1</span> | 7 |
| 12: <span class="smallcaps">op1_shift2</span> | 5 |
| 13: <span class="smallcaps">op1_shift3</span> | 9 |
| 14: <span class="smallcaps">op1_shift4</span> | 5 |
| 15: <span class="smallcaps">result1</span> | 49 |
| 16: <span class="smallcaps">result2</span> | 98 |
| 17: <span class="smallcaps">result3</span> | 108 |
| 18: <span class="smallcaps">result4</span> | 116 |
| 19: <span class="smallcaps">pre_prod</span> | 19 |
| 20: <span class="smallcaps">pre_carry</span> | 9 |
| (\\(P+\\)`<!-- -->`{=html}35)–(\\(2P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_1</span> | \\({\bm{v}}^P_{3}\\) |
| (\\(2P+\\)`<!-- -->`{=html}35)–(\\(3P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2</span> | \\({\bm{v}}^P_{4}\\) |
| (\\(3P+\\)`<!-- -->`{=html}35)–(\\(4P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_3</span> | \\({\bm{v}}^P_{5}\\) |
| (\\(4P+\\)`<!-- -->`{=html}35)–(\\(5P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_4</span> | \\({\bm{v}}^P_{6}\\) |
| (\\(5P+\\)`<!-- -->`{=html}35)–(\\(6P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_5</span> | \\({\bm{v}}^P_{7}\\) |

Illustration of the construction idea.

</div>

## Transformer Block 1 — Causal Attention Layer

To implement the concept introduced in  
efsubsec:construction_idea, it is essential to design a Transformer block capable of generating an embedding matrix depicted in the left-most table of  
eftab:emb_nx2_constructionidea. The goal of the first Transformer block is to fill <span class="smallcaps">is_op2_one</span> (\\(6\\)-th dimension) and <span class="smallcaps">is_op2_ten</span> (\\(7\\)-th dimension) by \\(1\\) if the token corresponds to the ones or tens digit of the second operand, respectively, and 0 otherwise. These two dimensions enable the filling of <span class="smallcaps">op2_one</span> (\\(8\\)-th dimension) and <span class="smallcaps">op2_ten</span> (\\(9\\)-th dimension) at the second Transformer block. Furthermore, we will fill <span class="smallcaps">mask</span> (\\(34\\)-th dimension) in the first block, which will serve as a base for filling <span class="smallcaps">op1_shift0</span> to <span class="smallcaps">op1_shift4</span> in the second block. Thus, we currently have 3 objectives(<span class="smallcaps">is_op2_one</span>, <span class="smallcaps">is_op2_ten</span>, <span class="smallcaps">mask</span>), each of which will be addressed by an individual head.

### Attention Head 1: Detecting the Ones Digit of the Second Operand

The goal of the first head is to make the dimension <span class="smallcaps">is_op2_one</span> as a one-hot row vector, where \\(1\\) is placed only at the token corresponding to the ones digit of the second operand.

Recall that \\(d=6P+34\\) and let \\(d_{QK,11}=P+1\\). Let \\(M>0\\) be a sufficiently large positive real number. Let \\[\begin{aligned}
    {\bm{Q}}^{(1)}_{1} &= \begin{pmatrix}
        {\bm{0}}_{P \times (P+34)} & {\bm{0}}_{P\times P} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{P+34}_{\textsc{full\_ones}}\right)^\top & {\bm{0}}_{1 \times P} & {\bm{0}}_{1 \times P} & {\bm{0}}_{1 \times P} & {\bm{0}}_{1 \times P} & {\bm{0}}_{1 \times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,11} \times d}, \\
    {\bm{K}}^{(1)}_{1} &= \begin{pmatrix}
        {\bm{0}}_{P \times (P+34)} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{P+34}_{\textsc{is\_bos}}\right)^\top & {\bm{0}}_{1 \times P} & {\bm{0}}_{1 \times P} & {\bm{0}}_{1 \times P} & {\bm{0}}_{1 \times P} & {\bm{0}}_{1 \times P}
        \end{pmatrix} \in \mathbb{R}^{d_{QK,11} \times d}.
\end{aligned}\\]

Unlike the construction for the addition task, we do not provide the table for the exact matrix and detailed error analysis due to their complex characterization. Instead, we provide an illustrative example for each step. We will also simply regard \\(M\\) as a sufficiently large real scalar and thus the attention values can be clearly separated after going through the softmax operation.

The matrix \\({\bm{Q}}^{(1)}_{1}\\) maps the embedding matrix \\({\bm{X}}^{(0)}\\) into a query matrix \\({\bm{Q}}^{(1)}_{1} {\bm{X}}^{(0)} \in \mathbb{R}^{(P+1) \times N}\\), where the first \\(P\\) rows are obtained by copying from the dimensions <span class="smallcaps">pos_2</span> and scaling by \\(\sqrt{M}\\), while the last row is the copy of the dimension <span class="smallcaps">full_ones</span> scaled by \\(\sqrt{MP}\\). Similarly, the matrix \\({\bm{K}}^{(1)}_{1}\\) maps the embedding matrix to a key matrix \\({\bm{K}}^{(1)}_{1} {\bm{X}}^{(0)} \in \mathbb{R}^{(P+1) \times N}\\). In this case, the first \\(P\\) rows are obtained by copying from the dimensions <span class="smallcaps">pos_1</span> and scaled by \\(\sqrt{M}\\), with the last row being the dimension <span class="smallcaps">is_bos</span>, scaled by \\(\sqrt{MP}\\). For concrete examples, refer to  
eftab:Q11X_Nx2,tab:K11X_Nx2.

<div id="tab:Q11X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{2}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) |

Example of \\({\bm{Q}}^{(1)}_{1}{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb_Nx2.

</div>

<div id="tab:K11X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{2}\\) | \\(\sqrt{M}{\bm{v}}^P_{1}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(1)}_{1}{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb_Nx2.

</div>

By these, the attention score matrix \\({\bm{C}}^{(1)}_{1} := ({\bm{K}}^{(1)}_{1} {\bm{X}}^{(0)})^\top {\bm{Q}}^{(1)}_{1} {\bm{X}}^{(0)}\\) and the attention matrix \\({\bm{A}}^{(1)}_{1} := {\tt softmax}({\bm{C}}^{(1)}_{1}) \in \mathbb{R}^{N\times N}\\) can be obtained. We provide the example of \\({\bm{A}}_1^{(1)}\\) in  
eftab:a11_Nx2. Specifically, an entry in \\({\bm{A}}_1^{(1)}\\) is non-zero if and only if the inner product between the query and key vectors equals \\(MP\\).

<div id="tab:a11_Nx2" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(3\\) | \\(4\\) | \\(5\\) | \\(6\\) | \\(7\\) | \\(8\\) | \\(9\\) | \\(10\\) | \\(11\\) | \\(12\\) | \\(13\\) | \\(14\\) | \\(15\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(1/2\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/3\\) | \\(1/4\\) | \\(1/4\\) | \\(1/3\\) | \\(1/3\\) | \\(1/2\\) |
| \\(2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) | \\(0\\) |
| \\(3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) | \\(0\\) | \\(0\\) |
| \\(4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(5\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(6\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(1/3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(7\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(8\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(9\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(10\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(11\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(12\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) | \\(0\\) | \\(0\\) |
| \\(13\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) | \\(0\\) |
| \\(14\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) |
| \\(15\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{A}}^{(1)}_{1}\\) (with explicit row/column indices and sufficiently large \\(M\\)), continuing from  
eftab:Q11X_Nx2,tab:K11X_Nx2.

</div>

Now let \\(d_{V,11} = 1\\) and define \\[\begin{aligned}
    {\bm{V}}^{(1)}_{1} &= 2({\bm{e}}^{d}_{\textsc{is\_mul}} - {\bm{e}}^{d}_{\textsc{is\_equal}})^\top \in \mathbb{R}^{d_{V,11} \times d}, \\
    {\bm{U}}^{(1)}_{1} &= {\bm{e}}^{d}_{\textsc{is\_op2\_one}} \in \mathbb{R}^{d \times d_{V,11}}.
\end{aligned}\\] The matrix \\({\bm{U}}^{(1)}_{1}{\bm{V}}^{(1)}_{1}{\bm{X}}^{(0)}\\) takes the dimension <span class="smallcaps">is_mul</span> and <span class="smallcaps">is_equal</span> from the embedding matrix \\({\bm{X}}^{(0)}\\), subtracts one from the other, scales the result by 2, and puts it to the dimension <span class="smallcaps">is_op2_sum</span>. Consequently, the matrix \\({\bm{U}}^{(1)}_{1}{\bm{V}}^{(1)}_{1}{\bm{X}}^{(0)}{\bm{A}}^{(1)}_{1}\\) is a matrix that matches the size of the input embedding matrix \\({\bm{X}}^{(0)}\\) and is filled with zeroes, except for a unique \\(1\\) located at the ones place of the second operand in the input sequence, in the dimension <span class="smallcaps">is_op2_one</span> (\\(6\\)-th). A concrete example is provided in  
eftab:U11V11X_Nx2,tab:U11V11XA11_Nx2.

<div id="tab:U11V11X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">is_op2_one</span> | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | -2 | 0 | 0 | 0 | 0 | 0 | 0 |

Example of \\({\bm{U}}^{(1)}_{1}{\bm{V}}^{(1)}_{1}{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U11V11XA11_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">is_op2_one</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Example of \\({\bm{U}}^{(1)}_{1}{\bm{V}}^{(1)}_{1}{\bm{X}}^{(0)}{\bm{A}}^{(1)}_{1}\\), continuing from  
eftab:U11V11X_Nx2,tab:a11_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Attention Head 2: Detecting the Tens Digit of the Second Operand

In the previous head, we set the dimension <span class="smallcaps">is_op2_one</span> (\\(6\\)-th dimension) to a one-hot row vector, where \\(1\\) is placed only in the token corresponding to the ones digit of the second operand. The objective of Attention head 2 is to fill the dimension <span class="smallcaps">is_op2_ten</span> (\\(7\\)-th dimension) similarly to <span class="smallcaps">is_op2_one</span>, but with \\(1\\) placed only in the tens digit of the second operand.

The design of the query, key, and value weight is not significantly different from the previous head. Compared to the construction of attention head 1, we only push \\(\sqrt{M} {\bm{I}}_P\\) to the next block for designing \\({\bm{Q}}^{(1)}_{2}\\). Specifically, \\({\bm{Q}}^{(1)}_{2}\\) and \\({\bm{K}}^{(1)}_{2}\\) are defined as \\[\begin{aligned}
    {\bm{Q}}^{(1)}_{2} &= \begin{pmatrix}
        {\bm{0}}_{P \times (P+34)} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{P+34}_{\textsc{full\_ones}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,12} \times d}, \\
    {\bm{K}}^{(1)}_{2} &= \begin{pmatrix}
        {\bm{0}}_{P \times (P+34)} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{P+34}_{\textsc{is\_bos}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
        \end{pmatrix} \in \mathbb{R}^{d_{QK,12} \times d},
\end{aligned}\\] where \\(d_{QK,12}\\) is set to \\(P+1\\). We refer to  
eftab:Q12X_Nx2,tab:K12X_Nx2 for specific examples.

<div id="tab:Q12X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) |

Example of \\({\bm{Q}}^{(1)}_{2}{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb_Nx2.

</div>

<div id="tab:K12X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{2}\\) | \\(\sqrt{M}{\bm{v}}^P_{1}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(1)}_{2}{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb_Nx2.

</div>

By these, the attention score matrix \\({\bm{C}}^{(1)}_{2} := ({\bm{K}}^{(1)}_{2} {\bm{X}}^{(0)})^\top {\bm{Q}}^{(1)}_{2} {\bm{X}}^{(0)}\\) and the attention matrix \\({\bm{A}}^{(1)}_{2} := {\tt softmax}({\bm{C}}^{(1)}_{2}) \in \mathbb{R}^{N\times N}\\) can be obtained, and the example of \\({\bm{A}}^{(1)}_{2}\\) is provided in  
eftab:a12_Nx2.

<div id="tab:a12_Nx2" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(3\\) | \\(4\\) | \\(5\\) | \\(6\\) | \\(7\\) | \\(8\\) | \\(9\\) | \\(10\\) | \\(11\\) | \\(12\\) | \\(13\\) | \\(14\\) | \\(15\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/3\\) | \\(1/4\\) | \\(1/4\\) | \\(1/3\\) | \\(1/3\\) |
| \\(2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) |
| \\(3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) | \\(0\\) |
| \\(4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) |
| \\(5\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(6\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(7\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) |
| \\(8\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(9\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(10\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(11\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/4\\) | \\(0\\) | \\(0\\) |
| \\(12\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) | \\(0\\) |
| \\(13\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/3\\) |
| \\(14\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(15\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{A}}^{(1)}_{2}\\) (with explicit row/column indices and sufficiently large \\(M\\)), continuing from  
eftab:Q12X_Nx2,tab:K12X_Nx2.

</div>

Finally, we set \\({\bm{V}}^{(1)}_{2}\\) and \\({\bm{U}}^{(1)}_{2}\\) the same to that of the previous head. That is, with \\(d_{V,12} = 1\\), \\[\begin{aligned}
    {\bm{V}}^{(1)}_{2} &= 2({\bm{e}}^{d}_{\textsc{is\_mul}} - {\bm{e}}^{d}_{\textsc{is\_equal}})^\top \in \mathbb{R}^{d_{V,12} \times d}, \\
    {\bm{U}}^{(1)}_{2} &= {\bm{e}}^{d}_{\textsc{is\_op2\_ten}} \in \mathbb{R}^{d \times d_{V,12}},
\end{aligned}\\] and the example of \\({\bm{U}}^{(1)}_{2}{\bm{V}}^{(1)}_{2}{\bm{X}}^{(0)}\\) and \\({\bm{U}}^{(1)}_{2}{\bm{V}}^{(1)}_{2}{\bm{X}}^{(0)}{\bm{A}}^{(1)}_{2}\\) is provided in  
eftab:U12V12X_Nx2,tab:U12V12XA12_Nx2. Consequently, the matrix \\({\bm{U}}^{(1)}_{2}{\bm{V}}^{(1)}_{2}{\bm{X}}^{(0)}{\bm{A}}^{(1)}_{2}\\) is a matrix that matches the size of the input embedding matrix and is filled with zeroes, except for a unique \\(1\\) located at the tens place of the second operand in the input sequence, with the dimension <span class="smallcaps">is_op2_ten</span> (\\(7\\)-th dimension).

<div id="tab:U12V12X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7: <span class="smallcaps">is_op2_ten</span> | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | -2 | 0 | 0 | 0 | 0 | 0 | 0 |

Example of \\({\bm{U}}^{(1)}_{2}{\bm{V}}^{(1)}_{2}{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U12V12XA12_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7: <span class="smallcaps">is_op2_ten</span> | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Example of \\({\bm{U}}^{(1)}_{2}{\bm{V}}^{(1)}_{2}{\bm{X}}^{(0)}{\bm{A}}^{(1)}_{2}\\), continuing from  
eftab:U12V12X_Nx2,tab:a12_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Attention Head 3: Position Masking

The goal of Attention head 3 is to generate a binary mask at the dimension <span class="smallcaps">mask</span> (\\(34\\)-th dimension), with ‘0’ placed before the multiplication symbol (\\(\times\\)) and ‘1’ placed starting from the multiplication symbol to the end.

To this end, we set \\(d_{QK,13} = 1\\) and design query and key weights by \\[\begin{aligned}
    {\bm{Q}}^{(1)}_{3} &= \begin{pmatrix} {\bm{e}}^d_{\textsc{full\_ones}}
    \end{pmatrix}^\top \in \mathbb{R}^{d_{QK,13} \times d},\\
    {\bm{K}}^{(1)}_{3} &= \begin{pmatrix} {\bm{e}}^d_{\textsc{is\_mul}}
    \end{pmatrix}^\top \in \mathbb{R}^{d_{QK,13} \times d}.
\end{aligned}\\]

The matrices \\({\bm{Q}}^{(1)}_{3} {\bm{X}}^{(0)}\\) and \\({\bm{K}}^{(1)}_{3} {\bm{X}}^{(0)}\\) take the dimension <span class="smallcaps">full_ones</span> and <span class="smallcaps">is_mul</span>, respectively, from the input embedding matrix. For concrete examples, please refer to  
eftab:Q13X_Nx2,tab:K13X_Nx2.

<div id="tab:Q13X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) |

Example of \\({\bm{Q}}^{(1)}_{3}{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb_Nx2.

</div>

<div id="tab:K13X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(1)}_{3}{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb_Nx2.

</div>

By these, the attention score matrix \\({\bm{C}}^{(1)}_{3} := ({\bm{K}}^{(1)}_{3} {\bm{X}}^{(0)})^\top {\bm{Q}}^{(1)}_{3} {\bm{X}}^{(0)}\\) and the attention matrix \\({\bm{A}}^{(1)}_{3} := {\tt softmax}({\bm{C}}^{(1)}_{3}) \in \mathbb{R}^{N\times N}\\) can be obtained and the example of \\({\bm{A}}^{(1)}_{3}\\) is provided in  
eftab:a13_Nx2.

<div id="tab:a13_Nx2" markdown="1">

<table>
<caption>Example of <span class="math inline"><strong>A</strong><sub>3</sub><sup>(1)</sup></span> (with explicit row/column indices), continuing from<br />
ef<span>tab:Q13X_Nx2,tab:K13X_Nx2</span>.</caption>
<tbody>
<tr>
<td colspan="2" rowspan="2" style="text-align: center;">row \ col</td>
<td style="text-align: center;"><span class="math inline"><em>j</em> = 1</span></td>
<td style="text-align: center;"><span class="math inline">2</span></td>
<td style="text-align: center;"><span class="math inline">3</span></td>
<td style="text-align: center;"><span class="math inline">4</span></td>
<td style="text-align: center;"><span class="math inline">5</span></td>
<td style="text-align: center;"><span class="math inline">6</span></td>
<td style="text-align: center;"><span class="math inline">7</span></td>
<td style="text-align: center;"><span class="math inline">8</span></td>
<td style="text-align: center;"><span class="math inline">9</span></td>
<td style="text-align: center;"><span class="math inline">10</span></td>
<td style="text-align: center;"><span class="math inline">11</span></td>
<td style="text-align: center;"><span class="math inline">12</span></td>
<td style="text-align: center;"><span class="math inline">13</span></td>
<td style="text-align: center;"><span class="math inline">14</span></td>
<td style="text-align: center;"><span class="math inline">15</span></td>
</tr>
<tr>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline"><em>i</em> = 1</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}1\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">1/2</span></td>
<td style="text-align: center;"><span class="math inline">1/3</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">2</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/2</span></td>
<td style="text-align: center;"><span class="math inline">1/3</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">3</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/3</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">4</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">5</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">6</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">7</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">8</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">9</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">10</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">11</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">12</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">13</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">14</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">15</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
</tbody>
</table>

</div>

Finally, we set \\({\bm{V}}^{(1)}_{3}\\) and \\({\bm{U}}^{(1)}_{3}\\) by \\(d_{V,13} = 1\\) and \\[\begin{aligned}
    {\bm{V}}^{(1)}_{3} &= ({\bm{e}}^{d}_{\textsc{is\_mul}})^\top \in \mathbb{R}^{d_{V,13} \times d}, \\
    {\bm{U}}^{(1)}_{3} &= {\bm{e}}^{d}_{\textsc{mask}} \in \mathbb{R}^{d \times d_{V,13}}.
\end{aligned}\\] The example of \\({\bm{U}}^{(1)}_{3}{\bm{V}}^{(1)}_{3}{\bm{X}}^{(0)}\\) and \\({\bm{U}}^{(1)}_{3}{\bm{V}}^{(1)}_{3}{\bm{X}}^{(0)}{\bm{A}}^{(1)}_{3}\\) is provided in  
eftab:U13V13X_Nx2,tab:U13V13XA13_Nx2. Consequently, the matrix \\({\bm{U}}^{(1)}_{3}{\bm{V}}^{(1)}_{3}{\bm{X}}^{(0)}{\bm{A}}^{(1)}_{3}\\) is a matrix that matches the size of the input embedding matrix and is filled with \\(1\\) only at the dimension <span class="smallcaps">mask</span> (\\(34\\)-th dimension) starting from the \\(\times\\) token to the end of sequence, and \\(0\\) otherwise.

At this point, the objective of attention head 3 may seem somewhat unclear. We note that the output of Attention head 3 will be utilized to fill the dimensions <span class="smallcaps">pos_2_mask</span> in the subsequent FFN layer, and this <span class="smallcaps">pos_2_mask</span> plays a crucial role in designing the key matrices in the Attention heads 3 to 7 at the second Transformer block.

<div id="tab:U13V13X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 34: <span class="smallcaps">mask</span> | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Example of \\({\bm{U}}^{(1)}_{3}{\bm{V}}^{(1)}_{3}{\bm{X}}^{(0)}\\), continuing from  
eftab:init_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U13V13XA13_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 34: <span class="smallcaps">mask</span> | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

Example of \\({\bm{U}}^{(1)}_{3}{\bm{V}}^{(1)}_{3}{\bm{X}}^{(0)}{\bm{A}}^{(1)}_{3}\\), continuing from  
eftab:U13V13X_Nx2,tab:a13_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Residual Connection [residual-connection-1]

So far we have computed the output of \\({\tt Att}_1\\) operation. Passing through the residual connection, the output of the attention layer becomes the sum of the original input embedding matrix and the output of \\({\tt Att}_1\\) operation: \\[\begin{aligned}
    {\bm{Y}}^{(1)} = {\bm{X}}^{(0)} + \sum_{h\in \{1,2,3\}} {\bm{U}}_h^{(1)} {\bm{V}}_h^{(1)} {\bm{X}}^{(0)} {\bm{A}}_h^{(1)}.
\end{aligned}\\] An example of the output of residual connection is presented in  
eftab:Nx2_res_conn_layer1.

<div id="tab:Nx2_res_conn_layer1" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 7 | 5 | 9 | 5 | 0 | 7 | 9 | 0 | 5 | 0 | 0 | 0 | 0 | 6 |
| 2: <span class="smallcaps">full_ones</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3: <span class="smallcaps">is_bos</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">is_op2_one</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7: <span class="smallcaps">is_op2_ten</span> | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 34: <span class="smallcaps">mask</span> | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 35–(\\(P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2_mask</span> | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| (\\(P+\\)`<!-- -->`{=html}35)–(\\(2P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_1</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) | \\({\bm{v}}^P_{1}\\) |
| (\\(2P+\\)`<!-- -->`{=html}35)–(\\(3P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) |
| (\\(3P+\\)`<!-- -->`{=html}35)–(\\(4P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_3</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) |
| (\\(4P+\\)`<!-- -->`{=html}35)–(\\(5P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_4</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) |
| (\\(5P+\\)`<!-- -->`{=html}35)–(\\(6P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_5</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) |

Example output of residual connection, continuing from  
eftab:init_emb_Nx2,tab:U11V11XA11_Nx2,tab:U12V12XA12_Nx2,tab:U13V13XA13_Nx2. Uncolored rows represent the initial embedding. Yellow rows indicate the rows filled by the attention heads in the first Transformer block. A pink row indicates the row that will be filled by the subsequent FFN layer.

</div>

## Transformer Block 1 — Token-wise Feed-forward Layer

The goal of the feed-forward layer involves filling the dimensions <span class="smallcaps">pos_2_mask</span>. Specifically, for each token \\(\sigma_i\\), if the dimension <span class="smallcaps">mask</span> is \\(1\\) (i.e., \\({\bm{Y}}^{(1)}_{(\textsc{mask}) i} = 1\\)), we want to fill the dimensions <span class="smallcaps">pos_2_mask</span> by copying the the corresponding token’s <span class="smallcaps">pos_2</span>; otherwise, we want to fill with \\({\bm{0}}_P\\). Be careful that the feed-forward operation is restricted to a token-wise mapping, meaning it only takes inputs from entries within the same column of the encoding matrix.

#### Construction for <span class="smallcaps">pos_2_mask</span>.

Given a vector \\({\bm{y}}= [{\bm{y}}_j]_{j=1}^d \in \mathbb{R}^d\\), define functions \\(g_l, h_l: \mathbb{R}^d \rightarrow \mathbb{R}\\) for every \\(j \in [P]\\) as \\[\begin{aligned}
    &g_l({\bm{y}}) := {\bm{y}}_{\textsc{pos\_2}, l} - 2 {\bm{y}}_{\textsc{mask}}\\
    &h_l({\bm{y}}) := -{\bm{y}}_{\textsc{pos\_2}, l} - 2 {\bm{y}}_{\textsc{mask}}
\end{aligned}\\] where \\({\bm{y}}_{\textsc{pos\_2}, l} \in \mathbb{R}\\) is the \\(l\\)-th dimension of \\({\bm{y}}_{\textsc{pos\_2}} \in \mathbb{R}^P\\) (\\(l \in {1, 2, \dots, P}\\)).

Consider a simple one-hidden-layer ReLU networks \\(f_l: \mathbb{R}^d \rightarrow \mathbb{R}\\) defined as \\[\begin{aligned}
    f_l({\bm{y}}) = \phi(g_l({\bm{y}})) - \phi(h_l({\bm{y}})).
\end{aligned}\\]

Using the fact that \\({\bm{y}}_{\textsc{pos\_2}, l}\\) is either \\(-1\\) or \\(1\\), we can easily check that \\(f_l({\bm{y}}) = {\bm{y}}_{\textsc{pos\_2}, l}\\) if \\({\bm{y}}_{\textsc{mask}}\\) is \\(0\\), and \\(f_l({\bm{y}}) = 0\\) if \\({\bm{y}}_{\textsc{mask}}\\) is \\(1\\).

Now, we can construct the width-\\(2P\\) feed-forward network that outputs the desired value at the dimension <span class="smallcaps">pos_2_mask</span> by \\[\begin{aligned}
    \left[{\tt FF}_1 \left({\bm{Y}}^{(1)}\right)\right]_{(\textsc{pos\_2\_mask})i} = \begin{bmatrix}
        f_1 \left({\bm{Y}}^{(1)}_{\bullet i}\right) & \cdots & f_P\left({\bm{Y}}^{(1)}_{\bullet i}\right)
    \end{bmatrix}^\top \in \mathbb{R}^{P\times 1},
\end{aligned}\\] and \\(0\\) for any other dimensions. The example output for this layer is presented in  
eftab:Nx2_FFN1.

<div id="tab:Nx2_FFN1" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 35–(\\(P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2_mask</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |

Example output of FFN layer at the first Transformer block, continuing from  
eftab:Nx2_res_conn_layer1.

</div>

### Residual Connection [residual-connection-2]

The last task of the feed-forward layer is to pass \\({\tt FF}_1\left({\bm{Y}}^{(1)}\right)\\) through the residual connection. As a result, we have \\[\begin{aligned}
    {\bm{X}}^{(1)} = {\bm{Y}}^{(1)} + {\tt FF}_1\left({\bm{Y}}^{(1)}\right).
\end{aligned}\\]

This is the end of the first Transformer block, and a concrete example of \\({\bm{X}}^{(1)}\\) is illustrated in  
eftab:first_emb_Nx2.

<div id="tab:first_emb_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 7 | 5 | 9 | 5 | 0 | 7 | 9 | 0 | 5 | 0 | 0 | 0 | 0 | 6 |
| 2: <span class="smallcaps">full_ones</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3: <span class="smallcaps">is_bos</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">is_op2_one</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7: <span class="smallcaps">is_op2_ten</span> | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8: <span class="smallcaps">op2_one</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9: <span class="smallcaps">op2_ten</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10: <span class="smallcaps">op1_shift0</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11: <span class="smallcaps">op1_shift1</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12: <span class="smallcaps">op1_shift2</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13: <span class="smallcaps">op1_shift3</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 14: <span class="smallcaps">op1_shift4</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 15: <span class="smallcaps">result1</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16: <span class="smallcaps">result2</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17: <span class="smallcaps">result3</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18: <span class="smallcaps">result4</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 19: <span class="smallcaps">pre_prod</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20: <span class="smallcaps">pre_carry</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 21: <span class="smallcaps">pre_eos1</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 22: <span class="smallcaps">pre_eos2</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 23-32: <span class="smallcaps">prod</span> | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) |
| 33: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 34: <span class="smallcaps">mask</span> | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 35–(\\(P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2_mask</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| (\\(P+\\)`<!-- -->`{=html}35)–(\\(2P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_1</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) | \\({\bm{v}}^P_{1}\\) |
| (\\(2P+\\)`<!-- -->`{=html}35)–(\\(3P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) |
| (\\(3P+\\)`<!-- -->`{=html}35)–(\\(4P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_3</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) |
| (\\(4P+\\)`<!-- -->`{=html}35)–(\\(5P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_4</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) |
| (\\(5P+\\)`<!-- -->`{=html}35)–(\\(6P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_5</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) |

Example embedding matrix after the first Transformer block. The yellow rows represent the results introduced during the first block, while the gray rows will be filled in the second block.

</div>

## Transformer Block 2 — Causal Attention Layer

Consider a scenario where the model is at the step of predicting the \\(i\\)-th least significant digit of the multiplication result. There are two goals for the causal attention layer at the second Transformer block. The first goal is to generate the embedding matrix as the left-most figure in  
eftab:emb_nx2_constructionidea, that is, fill <span class="smallcaps">op2_one</span>, <span class="smallcaps">op2_ten</span>, and <span class="smallcaps">op1_shift0</span> to <span class="smallcaps">op1_shift4</span> with the ones digit of the second operand, the tens digit of the second operand, and the \\(i\\), \\((i-1)\\), \\((i-2)\\), \\((i-3)\\), \\((i-4)\\)-th least significant digit of the first operand, respectively. Our construction assigns each head to each dimension. The second goal is to fill <span class="smallcaps">pre_eos1</span> and <span class="smallcaps">pre_eos2</span> with appropriate values. These \\(2\\) dimensions will be utilized in the subsequent FFN layer to predict whether we should predict the next token as EOS or not. Also, we note that filling these \\(2\\) dimensions can be implemented within the same head for <span class="smallcaps">op1_shift0</span> and <span class="smallcaps">op1_shift2</span> respectively, thus requiring a total of seven heads.

### Attention Head 1: Copying the Ones Digit of the Second Operand

The objective of Attention head 1 is to fill the dimension <span class="smallcaps">op2_one</span> with the ones digit of the second operand. To do so, we design the weights by defining \\(d_{QK, 21} = 1\\) and \\[\begin{aligned}
    {\bm{Q}}^{(2)}_{1} &= \begin{pmatrix} {\bm{e}}^d_{\textsc{full\_ones}}
    \end{pmatrix}^\top \in \mathbb{R}^{d_{QK,21} \times d},\\
    {\bm{K}}^{(2)}_{1} &= \begin{pmatrix} {\bm{e}}^d_{\textsc{is\_op2\_one}}
    \end{pmatrix}^\top \in \mathbb{R}^{d_{QK,21} \times d}.
\end{aligned}\\] We also define \\(d_{V,21} = 1\\) and \\[\begin{aligned}
    {\bm{V}}^{(2)}_{1} &= ({\bm{e}}^{d}_{\textsc{num}})^\top \in \mathbb{R}^{d_{V,21} \times d}, \\
    {\bm{U}}^{(2)}_{1} &= {\bm{e}}^{d}_{\textsc{op2\_one}} \in \mathbb{R}^{d \times d_{V,21}}.
\end{aligned}\\]

A concrete example of \\({\bm{Q}}_{1}^{(2)} X^{(1)}\\), \\({\bm{K}}_{1}^{(2)} X^{(1)}\\), \\({\bm{A}}_{1}^{2}\\), \\({\bm{U}}^{(2)}_{1}{\bm{V}}^{(2)}_{1}{\bm{X}}^{(1)}\\), and \\({\bm{U}}^{(2)}_{1}{\bm{V}}^{(2)}_{1}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{1}\\) is provided in  
eftab:Q21X_Nx2,tab:K21X_Nx2,tab:a21_Nx2,tab:U21V21X_Nx2,tab:U21V21XA21_Nx2. One might be concerned that in  
eftab:U21V21XA21_Nx2, the dimension <span class="smallcaps">op2_one</span> is not completely filled with ‘9’, but only the latter part. However, we note that given our focus on next-token prediction, it suffices to accurately fill values starting from the \\(=\\) token, and filling the preceding tokens with placeholder values does not cause any issues.

<div id="tab:Q21X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) |

Example of \\({\bm{Q}}^{(2)}_{1}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:K21X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(2)}_{1}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:a21_Nx2" markdown="1">

<table>
<caption>Example of <span class="math inline"><strong>A</strong><sub>1</sub><sup>(2)</sup></span> (with explicit row/column indices), continuing from<br />
ef<span>tab:Q21X_Nx2,tab:K21X_Nx2</span>.</caption>
<tbody>
<tr>
<td colspan="2" rowspan="2" style="text-align: center;">row \ col</td>
<td style="text-align: center;"><span class="math inline"><em>j</em> = 1</span></td>
<td style="text-align: center;"><span class="math inline">2</span></td>
<td style="text-align: center;"><span class="math inline">3</span></td>
<td style="text-align: center;"><span class="math inline">4</span></td>
<td style="text-align: center;"><span class="math inline">5</span></td>
<td style="text-align: center;"><span class="math inline">6</span></td>
<td style="text-align: center;"><span class="math inline">7</span></td>
<td style="text-align: center;"><span class="math inline">8</span></td>
<td style="text-align: center;"><span class="math inline">9</span></td>
<td style="text-align: center;"><span class="math inline">10</span></td>
<td style="text-align: center;"><span class="math inline">11</span></td>
<td style="text-align: center;"><span class="math inline">12</span></td>
<td style="text-align: center;"><span class="math inline">13</span></td>
<td style="text-align: center;"><span class="math inline">14</span></td>
<td style="text-align: center;"><span class="math inline">15</span></td>
</tr>
<tr>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline"><em>i</em> = 1</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}1\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">1/2</span></td>
<td style="text-align: center;"><span class="math inline">1/3</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">1/7</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">2</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/2</span></td>
<td style="text-align: center;"><span class="math inline">1/3</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">1/7</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">3</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/3</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">1/7</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">4</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">1/7</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">5</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">1/7</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">6</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">1/7</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">7</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/7</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">8</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">9</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">10</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">11</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">12</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">13</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">14</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">15</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:U21V21X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8: <span class="smallcaps">op2_one</span> | 0 | 7 | 5 | 9 | 5 | 0 | 7 | 9 | 0 | 5 | 0 | 0 | 0 | 0 | 6 |

Example of \\({\bm{U}}^{(2)}_{1}{\bm{V}}^{(2)}_{1}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U21V21XA21_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8: <span class="smallcaps">op2_one</span> | 0 | 7/2 | 4 | 21/4 | 26/5 | 13/3 | 33/7 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |

Example of \\({\bm{U}}^{(2)}_{1}{\bm{V}}^{(2)}_{1}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{1}\\), continuing from  
eftab:U21V21X_Nx2,tab:a21_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Attention Head 2: Copying the Tens Digit of the Second Operand

The objective of Attention head 2 is to fill the dimension <span class="smallcaps">op2_ten</span> with the tens digit of the second operand. We take a similar approach to Attention head 1, but the main difference is that we utilize the dimension <span class="smallcaps">is_op2_ten</span> instead of <span class="smallcaps">is_op2_one</span> for generating the key weight. We design the weights by defining \\(d_{QK, 22} = 1\\) and \\[\begin{aligned}
    {\bm{Q}}^{(2)}_{2} &= \begin{pmatrix} {\bm{e}}^d_{\textsc{full\_ones}}
    \end{pmatrix}^\top \in \mathbb{R}^{d_{QK,22} \times d},\\
    {\bm{K}}^{(2)}_{2} &= \begin{pmatrix} {\bm{e}}^d_{\textsc{is\_op2\_ten}}
    \end{pmatrix}^\top \in \mathbb{R}^{d_{QK,22} \times d}.
\end{aligned}\\] We also define \\(d_{V,22} = 1\\) and \\[\begin{aligned}
    {\bm{V}}^{(2)}_{2} &= ({\bm{e}}^{d}_{\textsc{num}})^\top \in \mathbb{R}^{d_{V,22} \times d}, \\
    {\bm{U}}^{(2)}_{2} &= {\bm{e}}^{d}_{\textsc{op2\_ten}} \in \mathbb{R}^{d \times d_{V,22}}.
\end{aligned}\\]

A concrete example of \\({\bm{Q}}_{2}^{(2)} X^{(1)}\\), \\({\bm{K}}_{2}^{(2)} X^{(1)}\\), \\({\bm{A}}_{2}^{2}\\), \\({\bm{U}}^{(2)}_{2}{\bm{V}}^{(2)}_{2}{\bm{X}}^{(1)}\\), and \\({\bm{U}}^{(2)}_{2}{\bm{V}}^{(2)}_{2}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{2}\\) is provided in  
eftab:Q22X_Nx2,tab:K22X_Nx2,tab:a22_Nx2,tab:U22V22X_Nx2,tab:U22V22XA22_Nx2. Once again, the dimension <span class="smallcaps">op2_ten</span> is not entirely filled with ‘7’ in  
eftab:U22V22XA22_Nx2. As mentioned in the previous head, this does not cause any issues because the front part (before \\(=\\)) does not affect the final prediction unless additional attention blocks are introduced after the second Transformer block.

<div id="tab:Q22X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) | \\(1\\) |

Example of \\({\bm{Q}}^{(2)}_{2}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:K22X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(2)}_{2}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:a22_Nx2" markdown="1">

<table>
<caption>Example of <span class="math inline"><strong>A</strong><sub>2</sub><sup>(2)</sup></span> (with explicit row/column indices), continuing from<br />
ef<span>tab:Q22X_Nx2,tab:K22X_Nx2</span>.</caption>
<tbody>
<tr>
<td colspan="2" rowspan="2" style="text-align: center;">row \ col</td>
<td style="text-align: center;"><span class="math inline"><em>j</em> = 1</span></td>
<td style="text-align: center;"><span class="math inline">2</span></td>
<td style="text-align: center;"><span class="math inline">3</span></td>
<td style="text-align: center;"><span class="math inline">4</span></td>
<td style="text-align: center;"><span class="math inline">5</span></td>
<td style="text-align: center;"><span class="math inline">6</span></td>
<td style="text-align: center;"><span class="math inline">7</span></td>
<td style="text-align: center;"><span class="math inline">8</span></td>
<td style="text-align: center;"><span class="math inline">9</span></td>
<td style="text-align: center;"><span class="math inline">10</span></td>
<td style="text-align: center;"><span class="math inline">11</span></td>
<td style="text-align: center;"><span class="math inline">12</span></td>
<td style="text-align: center;"><span class="math inline">13</span></td>
<td style="text-align: center;"><span class="math inline">14</span></td>
<td style="text-align: center;"><span class="math inline">15</span></td>
</tr>
<tr>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline"><em>i</em> = 1</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}1\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">1/2</span></td>
<td style="text-align: center;"><span class="math inline">1/3</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
<td style="text-align: center;"><span class="math inline">$\phantom{2}0\phantom{2}$</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">2</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/2</span></td>
<td style="text-align: center;"><span class="math inline">1/3</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">3</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/3</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">4</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/4</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">5</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/5</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">6</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1/6</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">7</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
<td style="text-align: center;"><span class="math inline">1</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">8</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">9</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">10</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">11</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">12</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">13</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">14</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
<tr>
<td style="text-align: right;"><span class="math inline">15</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
<td style="text-align: center;"><span class="math inline">0</span></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:U22V22X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9: <span class="smallcaps">op2_ten</span> | 0 | 7 | 5 | 9 | 5 | 0 | 7 | 9 | 0 | 5 | 0 | 0 | 0 | 0 | 6 |

Example of \\({\bm{U}}^{(2)}_{2}{\bm{V}}^{(2)}_{2}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U22V22XA22_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9: <span class="smallcaps">op2_ten</span> | 0 | 7/2 | 4 | 21/4 | 26/5 | 13/3 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |

Example of \\({\bm{U}}^{(2)}_{2}{\bm{V}}^{(2)}_{2}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{2}\\), continuing from  
eftab:U22V22X_Nx2,tab:a22_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Attention Head 3: Copying the Appropriate Digit from the First Operand 

The objectives of the first and the second Attention heads were to extract the ones and tens digits of the second operand and display them in the dimensions <span class="smallcaps">op2_one</span> and <span class="smallcaps">op2_ten</span>, respectively. For Attention head 3 to 7, we mainly focus on the first operand. Specifically, in Attention head 3, the goal is to fill the dimension <span class="smallcaps">op1_shift0</span> at the \\(i\\)-th least significant digit of the response (when predicting the \\((i+1)\\)-th least significant digit of the response) with the \\((i+1)\\)-th least significant digit of the first operand. For our example, we want to fill <span class="smallcaps">op1_shift0</span> of the token \\(=\\) by \\(5\\). Here, \\(i\\) ranges from \\(0\\) to \\(\ell_a + 2\\), where the \\(0\\)-th least significant digit of the response denotes the equal token. In cases where \\(i \geq \ell_a\\), we fill by \\(0\\).

Additionally, the third head has an extra objective: filling the dimension <span class="smallcaps">pre_eos1</span>. This dimension is utilized for EOS prediction in the subsequent FFN layer along with <span class="smallcaps">pre_eos2</span>, which is filled by the fifth head of the same layer. We observed that both objectives can be achieved by utilizing the same attention map. Thus, instead of implementing these objectives in separate heads, we can achieve them by utilizing the matrices \\({\bm{V}}_{3}^{(2)}\\) and \\({\bm{U}}_{3}^{(2)}\\) described below. Unlike previous heads, \\({\bm{V}}_{3}^{(2)}\\) and \\({\bm{U}}_{3}^{(2)}\\) each have two elements, with each element contributing to one of the objectives.

Our specific construction is as follows. With \\(d_{QK, 23} = P+1\\), \\[\begin{aligned}
    {\bm{Q}}^{(2)}_{3} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & {\bm{0}}_{P \times P} &  \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} &{\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{full\_ones}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,23} \times d}, \\
    {\bm{K}}^{(2)}_{3} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{is\_bos}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
        \end{pmatrix} \in \mathbb{R}^{d_{QK,23} \times d}.
\end{aligned}\\] and with \\(d_{V,23} = 2\\), \\[\begin{aligned}
    {\bm{V}}^{(2)}_{3} &= \begin{pmatrix}
        2({\bm{e}}^{d}_{\textsc{num}})^\top \\
        ({\bm{e}}^{d}_{\textsc{is\_bos}})^\top
    \end{pmatrix} \in \mathbb{R}^{d_{V,23} \times d}, \\
    {\bm{U}}^{(2)}_{3} &= \begin{pmatrix}
        {\bm{e}}^{d}_{\textsc{op1\_shift0}} & {\bm{e}}^{d}_{\textsc{pre\_eos1}}
    \end{pmatrix}  \in \mathbb{R}^{d \times d_{V,23}}.
\end{aligned}\\]

We provide the examples in  
eftab:Q23X_Nx2,tab:K23X_Nx2,tab:a23_Nx2,tab:U23V23X_Nx2,tab:U23V23XA23_Nx2. We note that within the dimension <span class="smallcaps">pre_eos1</span> of the matrix \\({\bm{U}}^{(2)}_{3}{\bm{V}}^{(2)}_{3}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{3}\\), if we restrict our view to the equal symbol \\(=\\) and the response sequence, \\(1\\) is only assigned to the first, second, and third most significant digits of the response (regardless of the query length).

<div id="tab:Q23X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{2}\\) | \\(\sqrt{M}{\bm{v}}^P_{1}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) |

Example of \\({\bm{Q}}^{(2)}_{3}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:K23X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(2)}_{3}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:a23_Nx2" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(3\\) | \\(4\\) | \\(5\\) | \\(6\\) | \\(7\\) | \\(8\\) | \\(9\\) | \\(10\\) | \\(11\\) | \\(12\\) | \\(13\\) | \\(14\\) | \\(15\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) |
| \\(2\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(5\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(6\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(7\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(8\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(9\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(10\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(11\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(12\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(13\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(14\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(15\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{A}}^{(2)}_{3}\\) (with explicit row/column indices and sufficiently large \\(M\\)), continuing from  
eftab:Q23X_Nx2,tab:K23X_Nx2.

</div>

<div id="tab:U23V23X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10: <span class="smallcaps">op1_shift0</span> | 0 | 14 | 10 | 18 | 10 | 0 | 14 | 18 | 0 | 10 | 0 | 0 | 0 | 0 | 12 |
| 21: <span class="smallcaps">pre_eos1</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Example of \\({\bm{U}}^{(2)}_{3}{\bm{V}}^{(2)}_{3}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U23V23XA23_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_bos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10: <span class="smallcaps">op1_shift0</span> | 0 | 0 | 7 | 5 | 9 | 5 | 5 | 9 | 5 | 9 | 5 | 7 | 0 | 0 | 0 |
| 21: <span class="smallcaps">pre_eos1</span> | 1 | 1 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1 | 1 | 1 |

Example of \\({\bm{U}}^{(2)}_{3}{\bm{V}}^{(2)}_{3}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{3}\\), continuing from  
eftab:U23V23X_Nx2,tab:a23_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Attention Head 4: Copying the Appropriate Digit from the First Operand 

The objective of Attention head 4 is to fill the dimension <span class="smallcaps">op1_shift1</span> at the \\(i\\)-th least significant digit of the response (when predicting the \\((i+1)\\)-th least significant digit of the response) with the \\(i\\)-th least significant digit of the first operand. Similarly to the previous head, \\(i\\) ranges from \\(0\\) to \\(\ell_a + 2\\). In cases where the \\(i\\)-th least significant digit of the first operand is not well-defined (i.e., \\(i\in \{ 0, \ell_a + 1, \ell_a + 2\}\\)), we assign \\(0\\).

The design of Attention head 4 is as follows. With \\(d_{QK, 24} = P+1\\), \\[\begin{aligned}
    {\bm{Q}}^{(2)}_{4} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & \sqrt{M} {\bm{I}}_{P} &{\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{full\_ones}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,24} \times d}, \\
    {\bm{K}}^{(2)}_{4} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{is\_bos}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
        \end{pmatrix} \in \mathbb{R}^{d_{QK,24} \times d},
\end{aligned}\\] and with \\(d_{V,24} = 1\\), \\[\begin{aligned}
    {\bm{V}}^{(2)}_{4} &= 2({\bm{e}}^{d}_{\textsc{num}})^\top \in \mathbb{R}^{d_{V,24} \times d}, \\
    {\bm{U}}^{(2)}_{4} &= {\bm{e}}^{d}_{\textsc{op1\_shift1}} \in \mathbb{R}^{d \times d_{V,24}}.
\end{aligned}\\]

We provide the examples in  
eftab:Q24X_Nx2,tab:K24X_Nx2,tab:a24_Nx2,tab:U24V24X_Nx2,tab:U24V24XA24_Nx2.

<div id="tab:Q24X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) | \\(\sqrt{M}{\bm{v}}^P_{2}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) |

Example of \\({\bm{Q}}^{(2)}_{4}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:K24X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(2)}_{4}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:a24_Nx2" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(3\\) | \\(4\\) | \\(5\\) | \\(6\\) | \\(7\\) | \\(8\\) | \\(9\\) | \\(10\\) | \\(11\\) | \\(12\\) | \\(13\\) | \\(14\\) | \\(15\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(1/2\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) |
| \\(2\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) |
| \\(3\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(5\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(6\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(7\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(8\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(9\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(10\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(11\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(12\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(13\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(14\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(15\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{A}}^{(2)}_{4}\\) (with explicit row/column indices and sufficiently large \\(M\\)), continuing from  
eftab:Q24X_Nx2,tab:K24X_Nx2.

</div>

<div id="tab:U24V24X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11: <span class="smallcaps">op1_shift1</span> | 0 | 14 | 10 | 18 | 10 | 0 | 14 | 18 | 0 | 10 | 0 | 0 | 0 | 0 | 12 |

Example of \\({\bm{U}}^{(2)}_{4}{\bm{V}}^{(2)}_{4}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U24V24XA24_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11: <span class="smallcaps">op1_shift1</span> | 0 | 7 | 5 | 9 | 5 | 0 | 9 | 5 | 0 | 5 | 9 | 5 | 7 | 0 | 0 |

Example of \\({\bm{U}}^{(2)}_{4}{\bm{V}}^{(2)}_{4}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{4}\\), continuing from  
eftab:U24V24X_Nx2,tab:a24_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Attention Head 5: Copying the Appropriate Digit from the First Operand 

The main objective of Attention head 5 is to fill the dimension <span class="smallcaps">op1_shift2</span> at the \\(i\\)-th least significant digit of the response (when predicting the \\((i+1)\\)-th least significant digit of the response) with the \\((i-1)\\)-th least significant digit of the first operand. Similarly to the previous head, \\(i\\) ranges from \\(0\\) to \\(\ell_a + 2\\), and in cases where the \\(i\\)-th least significant digit of the first operand is not well-defined (i.e., \\(i \in \{ 0, 1, \ell_a + 2 \}\\)), we assign \\(0\\).

As mentioned in Attention head 3, we assign an extra goal to Attention head 5, which is to fill the dimension <span class="smallcaps">pre_eos2</span>.

The design of the fifth head is as follows. With \\(d_{QK, 25} = P+1\\), \\[\begin{aligned}
    {\bm{Q}}^{(2)}_{5} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{full\_ones}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,25} \times d}, \\
    {\bm{K}}^{(2)}_{5} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{is\_bos}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
        \end{pmatrix} \in \mathbb{R}^{d_{QK,25} \times d},
\end{aligned}\\] and with \\(d_{V,25} = 2\\), \\[\begin{aligned}
    {\bm{V}}^{(2)}_{5} &= \begin{pmatrix}
        2({\bm{e}}^{d}_{\textsc{num}})^\top \\
        ({\bm{e}}^{d}_{\textsc{is\_bos}})^\top
    \end{pmatrix} \in \mathbb{R}^{d_{V,25} \times d}, \\
    {\bm{U}}^{(2)}_{5} &= \begin{pmatrix}
        {\bm{e}}^{d}_{\textsc{op1\_shift2}} & {\bm{e}}^{d}_{\textsc{pre\_eos2}}
    \end{pmatrix}  \in \mathbb{R}^{d \times d_{V,25}}.
\end{aligned}\\]

We provide the examples in  
eftab:Q25X_Nx2,tab:K25X_Nx2,tab:a25_Nx2,tab:U25V25X_Nx2,tab:U25V25XA25_Nx2. Note that within the dimension <span class="smallcaps">pre_eos2</span> of the matrix \\({\bm{U}}^{(2)}_{5}{\bm{V}}^{(2)}_{5}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{5}\\), if we restrict our view to the equal symbol \\(=\\) and the response sequence, \\(1\\) is only assigned to the most and the least significant digit of the response, and the equal token. An important observation is that upon comparing <span class="smallcaps">pre_eos1</span> and <span class="smallcaps">pre_eos2</span>, the most significant digit of the response is the only token that has a value of \\(1\\) in both dimensions. This observation plays a crucial role in predicting EOS for the next token, and we will elaborate further in the later section discussing the FFN layer.

<div id="tab:Q25X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{3}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) |

Example of \\({\bm{Q}}^{(2)}_{5}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:K25X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(2)}_{5}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:a25_Nx2" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(3\\) | \\(4\\) | \\(5\\) | \\(6\\) | \\(7\\) | \\(8\\) | \\(9\\) | \\(10\\) | \\(11\\) | \\(12\\) | \\(13\\) | \\(14\\) | \\(15\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(\phantom{2}1\phantom{2}\\) |
| \\(2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) |
| \\(3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) |
| \\(4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(5\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(6\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(7\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(8\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(9\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(10\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(11\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(12\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(13\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(14\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(15\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{A}}^{(2)}_{5}\\) (with explicit row/column indices and sufficiently large \\(M\\)), continuing from  
eftab:Q25X_Nx2,tab:K25X_Nx2.

</div>

<div id="tab:U25V25X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12: <span class="smallcaps">op1_shift2</span> | 0 | 14 | 10 | 18 | 10 | 0 | 14 | 18 | 0 | 10 | 0 | 0 | 0 | 0 | 12 |
| 22: <span class="smallcaps">pre_eos2</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Example of \\({\bm{U}}^{(2)}_{5}{\bm{V}}^{(2)}_{5}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U25V25XA25_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12: <span class="smallcaps">op1_shift2</span> | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 5 | 9 | 5 | 7 | 0 |
| 22: <span class="smallcaps">pre_eos2</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1/2 | 1 | 1 | 1 | 1/2 | 1/2 | 1/2 | 1/2 | 1 |

Example of \\({\bm{U}}^{(2)}_{5}{\bm{V}}^{(2)}_{5}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{5}\\), continuing from  
eftab:U25V25X_Nx2,tab:a25_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Attention Head 6: Copying the Appropriate Digit from the First Operand 

The objective of Attention head 6 is to fill the dimension <span class="smallcaps">op1_shift3</span> at the \\(i\\)-th least significant digit of the response (when predicting the \\((i+1)\\)-th least significant digit of the response) with the \\((i-2)\\)-th least significant digit of the first operand. Similarly to the previous head, \\(i\\) ranges from \\(0\\) to \\(\ell_a + 2\\). In cases where the \\(i\\)-th least significant digit of the first operand is not well-defined (i.e., \\(i\in \{ 0, 1, 2\}\\)), we assign \\(0\\).

The design of Attention head 6 is as follows. With \\(d_{QK, 26} = P+1\\), \\[\begin{aligned}
    {\bm{Q}}^{(2)}_{6} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{full\_ones}}\right)^\top & {\bm{0}}_{P\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,26} \times d}, \\
    {\bm{K}}^{(2)}_{6} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{is\_bos}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
        \end{pmatrix} \in \mathbb{R}^{d_{QK,26} \times d}.
\end{aligned}\\]

With \\(d_{V,26} = 1\\), \\[\begin{aligned}
    {\bm{V}}^{(2)}_{6} &= 2({\bm{e}}^{d}_{\textsc{num}})^\top \in \mathbb{R}^{d_{V,26} \times d}, \\
    {\bm{U}}^{(2)}_{6} &= {\bm{e}}^{d}_{\textsc{op1\_shift3}} \in \mathbb{R}^{d \times d_{V,26}}.
\end{aligned}\\]

We provide the examples in  
eftab:Q26X_Nx2,tab:K26X_Nx2,tab:a26_Nx2,tab:U26V26X_Nx2,tab:U26V26XA26_Nx2.

<div id="tab:Q26X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{10}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{10}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) |

Example of \\({\bm{Q}}^{(2)}_{6}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:K26X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(2)}_{6}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:a26_Nx2" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(3\\) | \\(4\\) | \\(5\\) | \\(6\\) | \\(7\\) | \\(8\\) | \\(9\\) | \\(10\\) | \\(11\\) | \\(12\\) | \\(13\\) | \\(14\\) | \\(15\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) |
| \\(2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) |
| \\(3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) |
| \\(4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) |
| \\(5\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(6\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(7\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(8\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(9\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(10\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(11\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(12\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(13\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(14\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(15\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{A}}^{(2)}_{6}\\) (with explicit row/column indices and sufficiently large \\(M\\)), continuing from  
eftab:Q26X_Nx2,tab:K26X_Nx2.

</div>

<div id="tab:U26V26X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13: <span class="smallcaps">op1_shift3</span> | 0 | 14 | 10 | 18 | 10 | 0 | 14 | 18 | 0 | 10 | 0 | 0 | 0 | 0 | 12 |

Example of \\({\bm{U}}^{(2)}_{6}{\bm{V}}^{(2)}_{6}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U26V26XA26_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13: <span class="smallcaps">op1_shift3</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 9 | 5 | 7 |

Example of \\({\bm{U}}^{(2)}_{6}{\bm{V}}^{(2)}_{6}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{6}\\), continuing from  
eftab:U26V26X_Nx2,tab:a26_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Attention Head 7: Copying the Appropriate Digit from the First Operand 

The objective of Attention head 7 is to fill the dimension <span class="smallcaps">op1_shift4</span> at the \\(i\\)-th least significant digit of the response (when predicting the \\((i+1)\\)-th least significant digit of the response) with the \\((i-3)\\)-th least significant digit of the first operand. Similarly to the previous head, \\(i\\) ranges from \\(0\\) to \\(\ell_a + 2\\). In cases where the \\(i\\)-th least significant digit of the first operand is not well-defined (i.e., \\(i\in \{ 0, 1, 2, 3\}\\)), we assign \\(0\\).

The design of Attention head 7 is as follows. With \\(d_{QK, 27} = P+1\\), \\[\begin{aligned}
    {\bm{Q}}^{(2)}_{7} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & \sqrt{M} {\bm{I}}_{P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{full\_ones}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
    \end{pmatrix} \in \mathbb{R}^{d_{QK,27} \times d}, \\
    {\bm{K}}^{(2)}_{7} &= \begin{pmatrix}
        {\bm{0}}_{P \times 34} & \sqrt{M} {\bm{I}}_{P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} & {\bm{0}}_{P\times P} \\
        \sqrt{MP} \left({\bm{e}}^{34}_{\textsc{is\_bos}}\right)^\top & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P} & {\bm{0}}_{1\times P}
        \end{pmatrix} \in \mathbb{R}^{d_{QK,27} \times d}.
\end{aligned}\\]

With \\(d_{V,27} = 1\\), \\[\begin{aligned}
    {\bm{V}}^{(2)}_{7} &= 2({\bm{e}}^{d}_{\textsc{num}})^\top \in \mathbb{R}^{d_{V,27} \times d}, \\
    {\bm{U}}^{(2)}_{7} &= {\bm{e}}^{d}_{\textsc{op1\_shift4}} \in \mathbb{R}^{d \times d_{V,27}}.
\end{aligned}\\]

We provide the examples in  
eftab:Q27X_Nx2,tab:K27X_Nx2,tab:a27_Nx2,tab:U27V27X_Nx2,tab:U27V27XA27_Nx2.

<div id="tab:Q27X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{10}\\) | \\(\sqrt{M}{\bm{v}}^P_{11}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{10}\\) | \\(\sqrt{M}{\bm{v}}^P_{11}\\) | \\(\sqrt{M}{\bm{v}}^P_{10}\\) | \\(\sqrt{M}{\bm{v}}^P_{9}\\) | \\(\sqrt{M}{\bm{v}}^P_{8}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) | \\(\sqrt{MP}\\) |

Example of \\({\bm{Q}}^{(2)}_{7}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:K27X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1–\\(P\\): | \\({\bm{0}}_P\\) | \\(\sqrt{M}{\bm{v}}^P_{4}\\) | \\(\sqrt{M}{\bm{v}}^P_{5}\\) | \\(\sqrt{M}{\bm{v}}^P_{6}\\) | \\(\sqrt{M}{\bm{v}}^P_{7}\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| \\(P+1\\): | \\(\sqrt{MP}\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{K}}^{(2)}_{7}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2.

</div>

<div id="tab:a27_Nx2" markdown="1">

| row \\ col | \\(j=1\\) | \\(2\\) | \\(3\\) | \\(4\\) | \\(5\\) | \\(6\\) | \\(7\\) | \\(8\\) | \\(9\\) | \\(10\\) | \\(11\\) | \\(12\\) | \\(13\\) | \\(14\\) | \\(15\\) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| \\(i=1\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(\phantom{2}1\phantom{2}\\) | \\(1/2\\) | \\(1/2\\) | \\(1/2\\) |
| \\(2\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(3\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) |
| \\(4\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) |
| \\(5\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(1/2\\) | \\(0\\) | \\(0\\) |
| \\(6\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(7\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(8\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(9\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(10\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(11\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(12\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(13\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(14\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |
| \\(15\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) | \\(0\\) |

Example of \\({\bm{A}}^{(2)}_{7}\\) (with explicit row/column indices and sufficiently large \\(M\\)), continuing from  
eftab:Q27X_Nx2,tab:K27X_Nx2.

</div>

<div id="tab:U27V27X_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 14: <span class="smallcaps">op1_shift4</span> | 0 | 14 | 10 | 18 | 10 | 0 | 14 | 18 | 0 | 10 | 0 | 0 | 0 | 0 | 12 |

Example of \\({\bm{U}}^{(2)}_{7}{\bm{V}}^{(2)}_{7}{\bm{X}}^{(1)}\\), continuing from  
eftab:first_emb_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

<div id="tab:U27V27XA27_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1: <span class="smallcaps">num</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2: <span class="smallcaps">full_ones</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 14: <span class="smallcaps">op1_shift4</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 9 | 5 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

Example of \\({\bm{U}}^{(2)}_{7}{\bm{V}}^{(2)}_{7}{\bm{X}}^{(1)}{\bm{A}}^{(2)}_{7}\\), continuing from  
eftab:U27V27X_Nx2,tab:a27_Nx2. (Irrelevant dimensions are omitted for readability)

</div>

### Residual Connection [residual-connection-3]

So far we have computed the output of \\({\tt Att}_2\\) operation. Passing through the residual connection, the output of the attention layer becomes the sum of \\({\bm{X}}^{(1)}\\) (the input to the second Transformer block) and the output of \\({\tt Att}_2\\) operation: \\[\begin{aligned}
    {\bm{Y}}^{(2)} = {\bm{X}}^{(1)} + \sum_{h\in [7]} {\bm{U}}_h^{(2)} {\bm{V}}_h^{(2)} {\bm{X}}^{(1)} {\bm{A}}_h^{(2)}.
\end{aligned}\\] A concrete example of the output of residual connection is presented in  
eftab:Nx2_res_conn_layer2.

<div id="tab:Nx2_res_conn_layer2" markdown="1">

| \\({\mathcal{I}}\\) |  | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 7 | 5 | 9 | 5 | 0 | 7 | 9 | 0 | 5 | 0 | 0 | 0 | 0 | 6 |
| 2: <span class="smallcaps">full_ones</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3: <span class="smallcaps">is_bos</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">is_op2_one</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7: <span class="smallcaps">is_op2_ten</span> | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8: <span class="smallcaps">op2_one</span> | 0 | 7/2 | 4 | 21/4 | 26/5 | 13/3 | 33/7 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |
| 9: <span class="smallcaps">op2_ten</span> | 0 | 7/2 | 4 | 21/4 | 26/5 | 13/3 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| 10: <span class="smallcaps">op1_shift0</span> | 0 | 0 | 7 | 5 | 9 | 5 | 5 | 9 | 5 | 9 | 5 | 7 | 0 | 0 | 0 |
| 11: <span class="smallcaps">op1_shift1</span> | 0 | 7 | 5 | 9 | 5 | 0 | 9 | 5 | 0 | 5 | 9 | 5 | 7 | 0 | 0 |
| 12: <span class="smallcaps">op1_shift2</span> | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 5 | 9 | 5 | 7 | 0 |
| 13: <span class="smallcaps">op1_shift3</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 9 | 5 | 7 |
| 14: <span class="smallcaps">op1_shift4</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 9 | 5 |
| 15: <span class="smallcaps">result1</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16: <span class="smallcaps">result2</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17: <span class="smallcaps">result3</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18: <span class="smallcaps">result4</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 19: <span class="smallcaps">pre_prod</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20: <span class="smallcaps">pre_carry</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 21: <span class="smallcaps">pre_eos1</span> | 1 | 1 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1 | 1 | 1 |
| 22: <span class="smallcaps">pre_eos2</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1/2 | 1 | 1 | 1 | 1/2 | 1/2 | 1/2 | 1/2 | 1 |
| 23-32: <span class="smallcaps">prod</span> | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) | \\({\bm{0}}_{10}\\) |
| 33: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 34: <span class="smallcaps">mask</span> | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 35–(\\(P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2_mask</span> | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| (\\(P+\\)`<!-- -->`{=html}35)–(\\(2P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_1</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) | \\({\bm{v}}^P_{1}\\) |
| (\\(2P+\\)`<!-- -->`{=html}35)–(\\(3P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) |
| (\\(3P+\\)`<!-- -->`{=html}35)–(\\(4P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_3</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) |
| (\\(4P+\\)`<!-- -->`{=html}35)–(\\(5P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_4</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) |
| (\\(5P+\\)`<!-- -->`{=html}35)–(\\(6P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_5</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) |

Example output of residual connection, continuing from  
eftab:first_emb_Nx2,tab:U21V21XA21_Nx2,tab:U22V22XA22_Nx2,tab:U23V23XA23_Nx2,tab:U24V24XA24_Nx2,tab:U25V25XA25_Nx2,tab:U26V26XA26_Nx2,tab:U27V27XA27_Nx2. Uncolored rows represent the initial embedding. Gray rows indicate the rows filled by the first Transformer block. Yellow rows indicate the rows filled by the attention layers at the second Transformer block. Pink rows indicate the rows that will be filled by the subsequent FFN layer.

</div>

## Transformer Block 2 — Token-wise Feed-forward Layer

Our ultimate goal is to fill the dimensions <span class="smallcaps">prod</span> and <span class="smallcaps">is_eos</span> with appropriate values. The dimensions <span class="smallcaps">result1</span> to <span class="smallcaps">result4</span>, <span class="smallcaps">pre_prod</span>, and <span class="smallcaps">pre_carry</span> serve as temporary memories for storing intermediate values, which will help us achieve our ultimate goal. Our construction involves sequentially stacking the MLP networks step-by-step to generate each of these temporary values. (As mentioned in the theorem statement below, we allow \\({\tt FF}_2\\) to be a multi-layer MLP.)

While our current construction for \\({\tt FF}_2\\) involves multiple hidden layers, we believe that our construction can be improved to employ a single hidden layer. If employing multiple hidden layers in the FFN is not feasible, this issue can be addressed by introducing additional Transformer blocks. Specifically, we can bypass the attention layer in these extra blocks by residual connection and only utilize their FFNs.

#### Step 1. Filling <span class="smallcaps">result_1</span> to <span class="smallcaps">result_4</span>

Here, we first assume the existence of a single-hidden-layer MLP network, denoted as \\(f:\mathbb{R}^2 \rightarrow \mathbb{R}\\), such that given any integers \\(a, b \in \{0, 1, \dots, 9 \}\\), \\(f(a, b)\\) equals to their multiplication, i.e., \\(ab\\). Such a network can be implemented with \\(100\\) hidden nodes `\citep{zhang2021understanding}`{=latex}.

Recalling  
efsubsec:construction_idea, we construct the first MLP network by utilizing eight instances of the function \\(f\\) in parallel as follows:

1.  \\(\textsc{result1} = f(\textsc{op1\_shift0}, \textsc{op2\_one}) + f(\textsc{op1\_shift1}, \textsc{op2\_ten}) \in \{0, 1, \dots, 162\}\\),

2.  \\(\textsc{result2} = f(\textsc{op1\_shift1}, \textsc{op2\_one}) + f(\textsc{op1\_shift2}, \textsc{op2\_ten}) \in \{0, 1, \dots, 162\}\\),

3.  \\(\textsc{result3} = f(\textsc{op1\_shift2}, \textsc{op2\_one}) + f(\textsc{op1\_shift3}, \textsc{op2\_ten}) \in \{0, 1, \dots, 162\}\\),

4.  \\(\textsc{result4} = f(\textsc{op1\_shift3}, \textsc{op2\_one}) + f(\textsc{op1\_shift4}, \textsc{op2\_ten}) \in \{0, 1, \dots, 162\}\\).

#### Step 2. Filling <span class="smallcaps">pre_prod</span> and <span class="smallcaps">pre_carry</span>

 

Here, we assume the existence of the following three single-hidden-layer MLP networks, denoted as \\(g_1, g_2, g_3:\mathbb{R}\rightarrow \mathbb{R}\\) , such that given any at most 3-digit integer \\(a \in \{0, 1, \dots, 162\}\\), \\(g_1(a)\\), \\(g_2(a)\\) and \\(g_3(a)\\) output the ones, tens, and hundreds digit of \\(a\\), respectively. Similarly to the previous step, each network can be implemented with 163 hidden nodes `\citep{zhang2021understanding}`{=latex}.

Recalling  
efsubsec:construction_idea, we construct the second MLP network on top of the first MLP network, by utilizing 2 instances of each of the function \\(g_1\\), \\(g_2\\), and \\(g_3\\) in parallel as follows:

- \\(\textsc{pre\_prod} = g_1(\textsc{result1}) + g_2(\textsc{result2}) + g_3(\textsc{result3}) \in \{0, 1, \dots, 27\}\\),

- \\(\textsc{pre\_carry} = g_1(\textsc{result2}) + g_2(\textsc{result3}) + g_3(\textsc{result4}) \in \{0, 1, \dots, 27\}\\).

#### Step 3. Filling <span class="smallcaps">prod</span>

Here, we assume the existence of a single-hidden-layer MLP network, denoted as \\(h:\mathbb{R}^2 \rightarrow \mathbb{R}\\), such that given any integers \\(a \in \{0, 1, \dots, 27\}\\), \\(b \in \{0, 1, \dots, 9 \}\\) satisfying \\(a - b \in \{-2, -1, 0, 8, 9, 10, 18, 19, 20 \}\\), \\(h\\) satisfies \\[\begin{aligned}
    h(a, b) = \begin{cases}
        0, \quad&\text{if } a-b \in \{-2, \, -1, \, 0\}, \\
        1, \quad&\text{if } a-b \in \{8, \, 9, \, 10\}, \\
        2, \quad&\text{if } a-b \in \{18, \, 19, \, 20\}.
    \end{cases}
\end{aligned}\\]

We also assume the existence of a single-hidden-layer MLP network, denoted as \\(h^\prime:\mathbb{R}\rightarrow \mathbb{R}\\), such that given any integer \\(a \in \{0, 1, \dots, 19\}\\), \\(h^\prime(a)\\) equals to \\(a \pmod{10}\\).

We finally assume the existence of a single-hidden-layer MLP network \\(q_i: \mathbb{R}\rightarrow \mathbb{R}\\) for each \\(i \in \{0, 1, \dots, 9\}\\), such that given any integers \\(a \in \{0, 1, \dots, 9\}\\), \\(q_i\\) satisfies \\[\begin{aligned}
    q_i(a) = \mathbbm{1} (i = a).
\end{aligned}\\] Similarly to the previous step, each network can be implemented with 280, 20, and 10 hidden nodes. Recalling  
efsubsec:construction_idea, we construct the third MLP network, on top of the second MLP network, by

- \\(\textsc{prod}= \begin{pmatrix}
          q_0(h^\prime(\textsc{pre\_prod} + h(\textsc{pre\_carry}, \textsc{num})))\\
          q_1(h^\prime(\textsc{pre\_prod} + h(\textsc{pre\_carry}, \textsc{num})))\\
          \vdots\\
          q_9(h^\prime(\textsc{pre\_prod} + h(\textsc{pre\_carry}, \textsc{num})))\\
      \end{pmatrix} \in \mathbb{R}^{10}\\).

One can easily check that \\(h^\prime(\textsc{pre\_prod} + h(\textsc{pre\_carry}, \textsc{num}))\\) yields an element of \\({0, 1, \dots, 9}\\), and thus <span class="smallcaps">prod</span> is an one-hot column vector. Specifically, if \\(h^\prime(\textsc{pre\_prod} + h(\textsc{pre\_carry}, \textsc{num}))=i\\), then <span class="smallcaps">prod</span> becomes \\({\bm{e}}_{i+1}^{10}\\).

#### Step 4. Filling <span class="smallcaps">is_eos</span>

We construct a single-hidden-layer MLP network \\(r:\mathbb{R}^2 \rightarrow \mathbb{R}\\) by \\[\begin{aligned}
    r(a,b) = 2 \phi(a+b - 1.5).
\end{aligned}\\] We then can fill the dimension <span class="smallcaps">is_eos</span> by

- \\(\textsc{is\_eos} = r(\textsc{pre\_eos1}, \textsc{pre\_eos2})\\).

Since <span class="smallcaps">pre_eos1</span> and <span class="smallcaps">pre_eos2</span> can have either \\(1/2\\) or \\(1\\), <span class="smallcaps">is_eos</span> equals \\(1\\) only when both <span class="smallcaps">pre_eos1</span> and <span class="smallcaps">pre_eos2</span> are \\(1\\). Additionally, we note that <span class="smallcaps">pre_eos1</span> and <span class="smallcaps">pre_eos2</span> are the direct outputs from the attention layer. Therefore, the network \\(r\\) can be deployed in parallel with the first MLP network and does not require an additional FFN layer.

The example output resulting from passing through all these steps is presented in  
eftab:Nx2_FFN2.

<div id="tab:Nx2_FFN2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 15: <span class="smallcaps">result1</span> | \- | \- | \- | \- | \- | \- | \- | \- | 45 | 116 | 108 | 98 | 49 | 0 | 0 |
| 16: <span class="smallcaps">result2</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 45 | 116 | 108 | 98 | 49 | 0 |
| 17: <span class="smallcaps">result3</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 45 | 116 | 108 | 98 | 49 |
| 18: <span class="smallcaps">result4</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 45 | 116 | 108 | 98 |
| 19: <span class="smallcaps">pre_prod</span> | \- | \- | \- | \- | \- | \- | \- | \- | 5 | 10 | 9 | 9 | 19 | 4 | 0 |
| 20: <span class="smallcaps">pre_carry</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 5 | 10 | 9 | 9 | 19 | 4 |
| 23-32: <span class="smallcaps">prod</span> | \- | \- | \- | \- | \- | \- | \- | \- | \\({\bm{e}}^{10}_6\\) | \\({\bm{e}}^{10}_1\\) | \\({\bm{e}}^{10}_1\\) | \\({\bm{e}}^{10}_1\\) | \\({\bm{e}}^{10}_1\\) | \\({\bm{e}}^{10}_7\\) | \\({\bm{e}}^{10}_1\\) |
| 33: <span class="smallcaps">is_eos</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

Example output of FFN layer in the second Transformer block, continuing from  
eftab:Nx2_res_conn_layer2. Here, we mark \\(-\\) for the entries before the equal token, as these entries do not affect the next-token prediction in our construction and are thus not important.

</div>

### Residual Connection [residual-connection-4]

The last task of the feed-forward layer is to pass \\({\tt FF}_2\left({\bm{Y}}^{(2)}\right)\\) through the residual connection. As a result, we have \\[\begin{aligned}
    {\bm{X}}^{(2)} = {\bm{Y}}^{(2)} + {\tt FF}_2\left({\bm{Y}}^{(2)}\right).
\end{aligned}\\]

This is the end of the second Transformer block, and an example of \\({\bm{X}}^{(2)}\\) is illustrated in  
eftab:second_emb_Nx2.

<div id="tab:second_emb_Nx2" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1: <span class="smallcaps">num</span> | 0 | 7 | 5 | 9 | 5 | 0 | 7 | 9 | 0 | 5 | 0 | 0 | 0 | 0 | 6 |
| 2: <span class="smallcaps">full_ones</span> | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3: <span class="smallcaps">is_bos</span> | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4: <span class="smallcaps">is_mul</span> | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5: <span class="smallcaps">is_equal</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6: <span class="smallcaps">is_op2_one</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7: <span class="smallcaps">is_op2_ten</span> | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8: <span class="smallcaps">op2_one</span> | \- | \- | \- | \- | \- | \- | \- | \- | 9 | 9 | 9 | 9 | 9 | 9 | 9 |
| 9: <span class="smallcaps">op2_ten</span> | \- | \- | \- | \- | \- | \- | \- | \- | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| 10: <span class="smallcaps">op1_shift0</span> | \- | \- | \- | \- | \- | \- | \- | \- | 5 | 9 | 5 | 7 | 0 | 0 | 0 |
| 11: <span class="smallcaps">op1_shift1</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 5 | 9 | 5 | 7 | 0 | 0 |
| 12: <span class="smallcaps">op1_shift2</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 5 | 9 | 5 | 7 | 0 |
| 13: <span class="smallcaps">op1_shift3</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 5 | 9 | 5 | 7 |
| 14: <span class="smallcaps">op1_shift4</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 5 | 9 | 5 |
| 15: <span class="smallcaps">result1</span> | \- | \- | \- | \- | \- | \- | \- | \- | 45 | 116 | 108 | 98 | 49 | 0 | 0 |
| 16: <span class="smallcaps">result2</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 45 | 116 | 108 | 98 | 49 | 0 |
| 17: <span class="smallcaps">result3</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 45 | 116 | 108 | 98 | 49 |
| 18: <span class="smallcaps">result4</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 45 | 116 | 108 | 98 |
| 19: <span class="smallcaps">pre_prod</span> | \- | \- | \- | \- | \- | \- | \- | \- | 5 | 10 | 9 | 9 | 19 | 4 | 0 |
| 20: <span class="smallcaps">pre_carry</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 5 | 10 | 9 | 9 | 19 | 4 |
| 21: <span class="smallcaps">pre_eos1</span> | \- | \- | \- | \- | \- | \- | \- | \- | 1/2 | 1/2 | 1/2 | 1/2 | 1 | 1 | 1 |
| 22: <span class="smallcaps">pre_eos2</span> | \- | \- | \- | \- | \- | \- | \- | \- | 1 | 1 | 1/2 | 1/2 | 1/2 | 1/2 | 1 |
| 23-32: <span class="smallcaps">prod</span> | \- | \- | \- | \- | \- | \- | \- | \- | \\({\bm{e}}^{10}_6\\) | \\({\bm{e}}^{10}_1\\) | \\({\bm{e}}^{10}_1\\) | \\({\bm{e}}^{10}_1\\) | \\({\bm{e}}^{10}_1\\) | \\({\bm{e}}^{10}_7\\) | \\({\bm{e}}^{10}_1\\) |
| 33: <span class="smallcaps">is_eos</span> | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 34: <span class="smallcaps">mask</span> | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 35–(\\(P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2_mask</span> | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) | \\({\bm{0}}_P\\) |
| (\\(P+\\)`<!-- -->`{=html}35)–(\\(2P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_1</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) | \\({\bm{v}}^P_{1}\\) |
| (\\(2P+\\)`<!-- -->`{=html}35)–(\\(3P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_2</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) | \\({\bm{v}}^P_{2}\\) |
| (\\(3P+\\)`<!-- -->`{=html}35)–(\\(4P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_3</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) | \\({\bm{v}}^P_{3}\\) |
| (\\(4P+\\)`<!-- -->`{=html}35)–(\\(5P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_4</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) | \\({\bm{v}}^P_{4}\\) |
| (\\(5P+\\)`<!-- -->`{=html}35)–(\\(6P+\\)`<!-- -->`{=html}34): <span class="smallcaps">pos_5</span> | \\({\bm{0}}_P\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{11}\\) | \\({\bm{v}}^P_{10}\\) | \\({\bm{v}}^P_{9}\\) | \\({\bm{v}}^P_{8}\\) | \\({\bm{v}}^P_{7}\\) | \\({\bm{v}}^P_{6}\\) | \\({\bm{v}}^P_{5}\\) |

Example embedding matrix after the second Transformer block. The yellow rows represent the results introduced during the second block, while the gray rows indicate the results from the first block. Similarly to  
eftab:Nx2_res_conn_layer2, we mark \\(-\\) for the entries before the equal token, as these entries do not affect the next-token prediction in our construction and are thus not important.

</div>

## Decoding Function [subsec:construction_nx2multiplication_decoding]

As mentioned in  
efsec:Transformer_architecture, the decoding function performs a linear readout (with a weight matrix \\({\bm{W}}_{\rm out}\in \mathbb{R}^{\left|{\mathcal{V}}\right|\times d}\\)) and a (token-wise) arg-max operation. That is, \\[\begin{aligned}
    {\tt Dec}\left({\bm{X}}^{(1)}\right) &:= \left({\mathcal{V}}_{k_i}\right)_{i=1,\ldots,N} \in {\mathcal{V}}^N,
\end{aligned}\\] where \\({\mathcal{V}}_k\\) is the \\(k\\)-th element of \\({\mathcal{V}}\\) and \\[\begin{aligned}
    k_i := \mathop{\mathrm{arg\,max}}_{k\in \left[\left|{\mathcal{V}}\right|\right]} \left\{o_k : {\bm{W}}_{\rm out} {\bm{X}}^{(1)}_{\bullet i}= \begin{bmatrix}
        o_1 & \cdots & o_{\left|{\mathcal{V}}\right|}
    \end{bmatrix}^\top\right\}.
\end{aligned}\\]

The objective of the decoding function is to perform a proper next-token prediction for \\(N \times 2\\) multiplication, especially utilizing the dimensions <span class="smallcaps">prod</span> and <span class="smallcaps">is_eos</span> of \\({\bm{X}}^{(2)}\\).

We now construct the weight matrix \\({\bm{W}}_{\rm out}\\). For a token \\(\sigma_i\\), if the value of dimension <span class="smallcaps">is_eos</span> of \\({\bm{X}}^{(2)}\\) is 0, then the linear readout output the dimensions <span class="smallcaps">prod</span> as it is to return one of a number token (0-9). On the other hand, if the value of dimension <span class="smallcaps">is_eos</span> is 1, then the linear readout outputs a large number (like 9 for example) for the token ‘\\(\$\\)’ to return EOS (\\(\$\\)). This can be implemented by the weight matrix \\({\bm{W}}_{\rm out}\\) described in  
eftab:nx2_Wout. Also, an example of applying the linear transform is showcased in  
eftab:nx2_linear_readout,tab:nx2_output_sequence.

<div id="tab:nx2_Wout" markdown="1">

| \\({\mathcal{V}}\\) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | \\(\times\\) | = | $ |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1-22: <span class="smallcaps">num</span>-<span class="smallcaps">pre_eos_2</span> | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) | \\({\bm{0}}_{22}\\) |
| 23: <span class="smallcaps">prod</span>\\(_1\\) | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 24: <span class="smallcaps">prod</span>\\(_2\\) | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 25: <span class="smallcaps">prod</span>\\(_3\\) | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 26: <span class="smallcaps">prod</span>\\(_4\\) | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 27: <span class="smallcaps">prod</span>\\(_5\\) | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 28: <span class="smallcaps">prod</span>\\(_6\\) | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29: <span class="smallcaps">prod</span>\\(_7\\) | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30: <span class="smallcaps">prod</span>\\(_8\\) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 31: <span class="smallcaps">prod</span>\\(_9\\) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 32: <span class="smallcaps">prod</span>\\(_{10}\\) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 33: <span class="smallcaps">is_eos</span> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| 34-end | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) | \\({\bm{0}}_{P'}\\) |

The *transposed* weight matrix \\({\bm{W}}_{out}^\top\\) of the linear readout in decoding function. \\(P'\\) represents \\(6P+1\\).

</div>

<div id="tab:nx2_linear_readout" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 0 | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 1 | 1 | 1 | 1 | 0 | 1 |
| 1 | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | \- | \- | \- | \- | \- | \- | \- | \- | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 7 | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| \\(\times\\) | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| = | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| $ | \- | \- | \- | \- | \- | \- | \- | \- | 0 | 0 | 0 | 0 | 0 | 0 | 100 |

Example output of linear readout (\\({\bm{W}}_{\rm out} {\bm{X}}^{(2)}\\)), continuing from  
eftab:second_emb_Nx2,tab:nx2_Wout. The yellow cells represent the maximum value of each column, from the ‘=’ token’s column to the rightmost column (which are used for next-token prediction).

</div>

<div id="tab:nx2_output_sequence" markdown="1">

| \\({\mathcal{I}}\\) | $ | 7 | 5 | 9 | 5 | \\(\times\\) | 7 | 9 | = | 5 | 0 | 0 | 0 | 0 | 6 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| \\({\mathcal{O}}\\) | \- | \- | \- | \- | \- | \- | \- | \- | 5 | 0 | 0 | 0 | 0 | 6 | $ |

Example output sequence \\({\mathcal{O}}= {\tt Dec}\left({\bm{X}}^{(2)}\right)\\), continuing from  
eftab:nx2_linear_readout. The yellow cells in the bottom row exactly predict the next tokens.

</div>

[^1]: Authors contributed equally to this paper.

[^2]: The hyperparameter `max_pos` determines the maximum testable sequence length that a Transformer can handle. Note that the idea of random assignment of position IDs is similar to *randomized PE* `\citep{ruoss2023randomized}`{=latex} although it is different since it assigns a sequence of increasing integers which are generally not consecutive.

[^3]: Note that they match up to matrix transpose, which is due to the difference in the formulations.

[^4]: The attention matrices depicted in  
    effig:attention_patterns are square, lower-triangular (due to causal attention pattern), and row-stochastic (all entries are nonnegative and the sum of each row equals 1).

[^5]: [`github.com/google/flaxformer`](https://github.com/google/flaxformer)

[^6]: [`github.com/McGill-NLP/length-generalization`](https://github.com/McGill-NLP/length-generalization)

[^7]: One can let \\(d_H = \max_{l, h} \max\{d^{(l)}_{QK,h}, d^{(l)}_{V,h}\}\\) as an inner dimension of each head. This makes our formal constructions a bit messier with redundant entries 0.

[^8]: Every entry is non-negative and the sum of entries in each column is 1.

[^9]: BOS and EOS tokens do not need to be identical. We regard them as the same token just for the simplicity of the presentation.

[^10]: The choice of the vectors \\({\bm{v}}^D_k\\) is not strict. They only need to have the same length and be distinguishable (for at least a constant order) in terms of inner products. That is, there should be a noticeable difference between \\(\left\lVert{\bm{v}}^D_k\right\rVert^2\\) and \\(\left\langle{\bm{v}}^D_k, {\bm{v}}^D_l\right\rangle\\) for \\(k\ne l\\).

[^11]: Such an idea of filling in the blacks of the encoding matrix is borrowed from the literature of RASP language(s) `\citep{weiss2021thinking,friedman2023learning,lindner2023tracr,zhou2024what}`{=latex}. This can be done with the help of residual connections.
