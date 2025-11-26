# QKFormer: Hierarchical Spiking Transformer using Q-K Attention

## Abstract

Spiking Transformers, which integrate Spiking Neural Networks (SNNs) with Transformer architectures, have attracted significant attention due to their potential for low energy consumption and high performance. We present QKFormer, a direct-training spiking transformer that capitalises on three core innovations: i) a spike-form Q-K attention mechanism with linear complexity that facilitates large-scale models without prohibitive memory overhead; ii) a hierarchical architecture that builds multi-scale spiking representations by progressively reducing the number of tokens; and iii) a Spiking Patch Embedding with Deformed Shortcut (SPEDS) module that markedly improves spiking information flow. Leveraging a deliberately chosen temporal window of eight time steps—long enough to capture rich dynamics yet compact enough for practical deployment—QKFormer attains compelling accuracy–efficiency trade-offs. Specifically, with 64.96 M parameters, QKFormer achieves a groundbreaking **85.65 %** top-1 accuracy on ImageNet-1K, surpassing the previous state-of-the-art Spikformer by **10.84 %** while maintaining a favourable energy profile. To our knowledge, this is the first directly trained SNN to cross the 85 % threshold on ImageNet-1K without resorting to any ANN pre-training. Code and pretrained models are available at [https://github.com/zhouchenlin2096/QKFormer](https://github.com/zhouchenlin2096/QKFormer).
# Introduction

Regarded as the third generation of neural networks `\cite{maass1997networks}`{=latex}, the brain-inspired Spiking Neural Networks (SNNs) are potential competitors to Artificial Neural Networks (ANNs) due to their high biological plausibility and high energy efficiency attributed to their event-driven properties `\cite{roy2019towards}`{=latex}. Transformer, originally designed for natural language processing `\cite{vaswani2017attention}`{=latex}, has flourished in various computer vision tasks, including image classification `\cite{dosovitskiy2020image,yuan2021tokens}`{=latex}, object detection `\cite{carion2020end,zhu2020deformable,liu2021swin}`{=latex} and semantic segmentation `\cite{wang2021pyramid,yuan2021volo}`{=latex}. Spiking Transformers (Transformer-based SNNs) `\cite{zhou2023spikformer,zhou2023spikingformer,yao2023spikedriven,zhou2023enhancing,ijcai2023p344}`{=latex}, which integrate spiking neural networks with transformer architecture, have attracted significant attention. This innovative combination provides great potential to develop advanced AI algorithms with high performance and low energy consumption.

As the architecture of the transformers is essential to the model’s performance `\cite{dosovitskiy2020image,yuan2021tokens,wang2021segregation,liu2021swin, yuan2021volo}`{=latex}, designing new architectures for transformer-based SNNs is quite challenging in terms of space requirements for the following reasons `\cite{zhou2023spikformer,yao2023spikedriven,ijcai2023p344}`{=latex}. i). Spiking Self Attention (SSA) `\cite{zhou2023spikformer}`{=latex}, the core module of spiking transformers, encodes Query, Key, and Value with sparse spikes. However, the computational complexity (especially space complexity) of SSA scales quadratically to the number of tokens (#tokens), and is the main obstacle to explore architecture that incorporate multi-level features. ii). SNNs process data across the time domain, necessitating a high level of computational and memory resources. This combination leads to considerable consumption of computational resources, making the training process highly demanding in terms of both memory and processing power.

To address these issues, we propose QKFormer with three innovations. i) Q-K attention with linear complexity and high energy efficiency. ii) A hierarchical architecture with decreasing number of tokens across blocks. iii) A novel patch embedding with deformed shortcut module. The linear complexity of Q-K attention is originated from the binary spike-form vector attention. This design lower the energy consumption and the space requirement. The hierarchical architecture starts from small patches and gradually merges neighboring patches in deeper spiking transformer layers with gradually decreasing \#tokens, which enables multi-level spiking feature representation and benefits the model performance. The patch embedding with deformed shortcut facilitates spiking information transmission and integration. These merits make QKFormer achieve state-of-the-art performance in the SNN domain, in contrast to the previous transformer-based SNNs with spiking feature maps of a single resolution. Our main contributions are as follows:

1\) We develop a novel spike-form Q-K attention mechanism, tailor-made for the spatio-temporal spiking patterns of SNNs, which can easily model the importance of token or channel dimensions with binary values. The Q-K attention has linear complexity to \#tokens (or \#channels) and only adopts two spike-form components: Query (\\(\mathbf{Q}\\)) and Key (\\(\mathbf{K}\\)).

2\) We design a versatile and powerful Spiking Patch Embedding with Deformed Shortcut (SPEDS) module, which enhances spiking information transmission and integration thus improving the performance of spiking transformers significantly.

3\) We build a direct-training hierarchical spiking transformer with different number of tokens across blocks, incorporating Q-K attention and SPEDS, named QKFormer. This marks the effective exploration of hierarchical spiking representation in Transformer-based SNNs.

4\) Extensive experiments show that the proposed model outperforms the state-of-the-art (SOTA) SNNs on various static and neuromorphic datasets. Notably, QKFormer has achieved a significant milestone, surpassing **85%** top-1 accuracy on ImageNet with 4 time steps using the direct training approach for the first time.

<figure id="fig: attention">
<img src="./figures/Attention.png"" />
<figcaption>Illustration of Q-K attention with the two versions of Q-K token attention (QKTA) and Q-K channel attention (QKCA). The inputs are binary spikes and there are only sparse additions and mask operations in Q-K attention. As a spike-driven module, Q-K attention efficiently models the token or channel attention through spike-form binary vectors, performing linear complexity to #tokens (or #channels) and high energy efficiency. Spiking Neuron (SN) in this work adopts the Leaky-Integrate-and-Fire (LIF) model, which is shown in Appendix. <a href="#LIF" data-reference-type="ref" data-reference="LIF">6.1</a>.</figcaption>
</figure>

# Related Work

**Learning Methods of Spiking Neural Networks.** At present, there are mainly two ways to obtain trained SNNs. One involves converting pre-trained ANNs to SNNs (ANN2SNN) `\cite{cao2015spiking,hunsberger2015spiking,bu2021optimal, Li2021AFL,han2020rmp,bu2023optimal,wang2023masked}`{=latex}, replacing the ReLU activation function in ANN with spiking neurons. However, This converted SNN suffers from long converting time steps and constraints on the original ANN design. Another method is to directly train SNNs`\cite{wu2018spatio}`{=latex}, using surrogate gradient`\cite{neftci2019surrogate,xiao2021training,shrestha2018slayer,fang2021deep}`{=latex} to address the non-differentiability of spike excitation function during backpropagation. The direct training method has received more attention due to its low latency and supporting flexible architectural exploration.

**Direct Trained SNN Models.** `\cite{fang2021deep}`{=latex} proposed the Spike-Element-Wise block, which further addressed gradient explosion and gradient vanishing problems, and prolonged the directly trained SNNs beyond a depth of 100 layers with 69.26% accuracy on ImageNet-1k. Spikformer `\cite{zhou2023spikformer}`{=latex} designed a novel spike-form self-attention named Spiking Self Attention (SSA), using sparse spike-form Query, Key, and Value without softmax operation, which was used to construct the Spikformer. Spikformer achieved 74.81\\(\%\\) accuracy on ImageNet-1k with 4 time steps, showing the great potential of transformer-based SNNs for the first time. Spikingformer `\cite{zhou2023spikingformer}`{=latex} modified Spikformer with a pre-activation shortcut, which can avoid the floating-point multiplications in synaptic computing and has a lower firing rate. Spikingformer achieved 75.85\\(\%\\) accuracy on ImageNet-1k. `\cite{yao2023spikedriven}`{=latex} designed a novel Spike-Driven Self-Attention (SDSA), which used only masks and addition operations without any multiplication, thus significantly reducing the computation energy compared to the vanilla self-attention. In addition, the proposed Spike-driven Transformer based on SDSA has achieved 77.07\\(\%\\) on ImageNet-1k. However, all of these SNN models above remain a large performance gap compared with ANN.

# Method [Method]

## Preliminary

**Vanilla Self Attention.** Vanilla self-attention (VSA) `\cite{vaswani2017attention}`{=latex} in transformers has three floating-point key components: query (\\(\mathbf{Q}_\mathcal{F}\\)), key (\\(\mathbf{K}_\mathcal{F}\\)), value (\\(\mathbf{V}_\mathcal{F}\\)) which are calculated by learnable linear matrics and input \\(\mathbf{X}\\). The calculation of VSA can be formulated as follows: \\[\mathbf{Q}_{\mathcal{F}}, \mathbf{K}_{\mathcal{F}}, \mathbf{V}_{\mathcal{F}}=\mathbf{X} (\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V),\\] \\[\operatorname{VSA}\left(\mathbf{Q}_{\mathcal{F}}, \mathbf{K}_{\mathcal{F}}, \mathbf{V}_{\mathcal{F}}\right)=\operatorname{Softmax}\left(\frac{\mathbf{Q}_{\mathcal{F}} \mathbf{K}_{\mathcal{F}}^{\mathrm{T}}}{\sqrt{d}}\right) \mathbf{V}_{\mathcal{F}},
\label{vsa}\\] where \\(\mathcal{F}\\) denotes the floating-point form. Both floating-point matrix multiplication and softmax operation which contains exponent calculation and division, do not align with the properties of SNNs.

**Spiking Self Attention.** Spikformer `\cite{zhou2023spikformer}`{=latex} demonstrated a novel spike-form self-attention named Spiking Self Attention (SSA), using sparse spike-form \\(\mathbf{Q, K, V}\\) without softmax operation and floating-point matrix multiplication. The calculation process of SSA is formulated as follows: \\[\mathbf{I} = \mathrm{SN}_{I}\left(\mathrm{BN}_{I}\left(\mathbf{X} (\mathbf{W}_{I})\right)\right), \mathbf{I} \in(\mathbf{Q}, \mathbf{K}, \mathbf{V}),\\] \\[\label{SSA}
\operatorname{SSA}^{\prime}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\mathrm{S N}\left(\mathbf{Q} \mathbf{K}^{\mathrm{T}} \mathbf{V} * s\right),\\] where \\(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathcal{R}^{T \times N \times D}\\), the spike-form \\(\mathbf{Q}, \mathbf{K}, \mathbf{V}\\) are computed by learnable linear layers. \\(s\\) is a scaling factor. \\(\mathrm{SN}\\) means spiking neuron layer. The calculation of SSA avoids floating-point multiplication, meeting the property of SNNs.

## Q-K Attention

An overview of Q-K attention is shown in Figure <a href="#fig: attention" data-reference-type="ref" data-reference="fig: attention">1</a>. Both VSA and SSA use three key components and have \\(O(N^2 d)\\) or \\(O(Nd^2)\\) computational complexity, while our proposed Q-K Attention which has linear complexity and only uses two spike-form components: \\(\mathbf{Q}\\) and \\(\mathbf{K}\\), which are produced through learnable linear matrics. \\[\mathbf{Q}=\mathrm{S N}_Q\left(\mathrm{BN}\left(\mathbf{X} \mathbf{W}_Q\right)\right), \mathbf{K}=\mathrm{S N}_K\left(\mathrm{BN}\left(\mathbf{X} \mathbf{W}_K\right)\right) ,\\] where \\(\mathbf{X}\\) is the input spiking map. According to the detailed calculation mechanism of \\(\mathbf{Q}, \mathbf{K}\\), Q-K Attention can be divided into Q-K Token Attention (QKTA) and Q-K Channel Attention (QKCA).

**Q-K Token Attention.** We here assume \\(T=1\\) and single head attention for mathematical description. After obtaining spike-form \\(\mathbf{Q}, \mathbf{K} \in \mathcal{R}^{T \times N \times D}\\), both \\(\mathbf{Q}\\) and \\(\mathbf{K}\\) can be formed as a spike matrix \\(N \times D\\) (\\(N\\) is the token number, \\(D\\) is the channel number). QKTA process can be formulated as follows: \\[\label{QKTA}
{\mathbf{A}}_t=\mathrm{S N}(\sum_{i=0}^D \mathbf{Q}_{i, j}), \quad   \mathbf{X}^{\prime}={\mathbf{A}}_t \otimes \mathbf{K} ,\\] where \\({\mathbf{A}}_t\\) is the \\(N*1\\) token attention vector, which models the binary importance of different tokens. \\({\mathbf{A}}_t\\) is a spike-form vector, which is obtained by addition operations (row summation) of \\(\mathbf{Q}\\) spike matrix and a following spiking neuron. \\(\otimes\\) is the Hadamard product between spike tensors, which is equivalent to the mask operation. We apply the spike-form token attention vector \\({\mathbf{A}}_t\\) to the \\(\mathbf{K}\\) spike matrix through the column mask operation (token mask), to obtain the output \\(\mathbf{X}^{\prime}\\) of QKTA.

**Q-K Channel Attention.** The calculation process of Q-K channel attention is similar to the previous Q-K token attention, and can be formulated as : \\(\mathbf{A}_c=\mathrm{S N}(\sum_{j=0}^N \mathbf{Q}_{i, j}), \quad \mathbf{X}^{\prime}=\mathbf{A}_c \otimes \mathbf{K},\\) where \\(\mathbf{A}_c\\) is the \\(1*D\\) channel attention vector, which models the binary importance of different channels. \\({\mathbf{A}}_t\\) is a spike-form vector, which is obtained by addition operations (column summation) of Q spike matrix and a following spiking neuron. Then, the output \\(\mathbf{X}^{\prime}\\) of Q-K Channel Attention is obtained by the row mask operation (channel mask) between \\({\mathbf{A}}_t\\) and \\(\mathbf{K}\\). \\[\mathbf{X}^{\prime \prime}=\mathrm{S N}\left(\mathrm{B N}\left( { \mathrm{Linear} }\left(\mathbf{X}^{\prime}\right)\right)\right) .
\label{post}\\] As shown in Formula.<a href="#post" data-reference-type="ref" data-reference="post">[post]</a>, a post-linear layer is also required after obtaining \\(\mathbf{X}^{\prime}\\) of Q-K Token or Channel Attention. In addition, the channel dimension is \\(D / h\\) in the multi-head Q-K attention, where \\(h\\) is the head number. In this work, the spiking neuron uses the LIF model `\cite{fang2021deep}`{=latex}. Same with `\cite{zhou2023spikformer}`{=latex}, time step \\(T\\) is an independent dimension for the spiking neuron layer. In other layers, it is merged with the batch size. We exploit QKTA in our experiments by default.

**Linear Computational Complexity of Q-K Attention.** As shown in Table <a href="#complexity" data-reference-type="ref" data-reference="complexity">1</a>, the time complexity of Q-K attention varies based on the implementation approach. Specifically, when utilizing spike-form broadcasted element-wise multiplication, \\(\otimes\\), the time complexity can reach up to \\(O(N * D)\\). When applying mask operation, the time complexity of Q-K attention is only \\(O(N)\\) or \\(O(D)\\). The space complexity of Q-K attention with the whole process is \\(O(N * D)\\) at most, which is caused by the self-storage consumption Q and K matrix. In terms of the space complexity of attention operation, Q-K attention only requires an extra \\(1*D\\) or \\(N*1\\) space to store the attention vector with the space complexity of \\(O(N)\\) or \\(O(D)\\). The linear complexity of Q-K attention makes it possible to successfully explore the large-scale hierarchical architecture SNN model.

**High Energy Efficiency of Q-K Attention.** As a spike-driven attention module, the linear multiplication is transformed into sparse addition. Mask operation can be implemented on neuromorphic chips through addressing algorithms `\cite{richter2022event}`{=latex} or AND logic operations`\cite{pei2019towards}`{=latex} with negligible power consumption. Compared with SSA, Q-K attention is much more energy-efficient, which comes from the following reasons: i) Q-K attention only adopts two spike-form components for spike \[0, 1\] operation without the V input and thus has less synaptic computing. ii) Q-K attention has much fewer spiking matrix operations due to its linear complexity of \\(O(N)\\) or \\(O(D)\\). iii) Q-K attention discards the scale operation of SSA, which leads to reduced power consumption further.

<div class="center" markdown="1">

<div id="complexity" markdown="1">

| Methods | VSA `\cite{vaswani2017attention}`{=latex} | SSA `\cite{zhou2023spikformer}`{=latex} | SDSA `\cite{yao2023spikedriven}`{=latex} | QKTA | QKCA |
|:---|:---|:---|:---|:---|:---|
| Time complexity | \\(O(N^2 D)\\) | \\(O(N^2 D)\\) | \\(O(ND)\\) | \\(O(D)\\) | \\(O(N)\\) |
| Space complexity | \\(O(N^2 + ND)\\) | \\(O(N^2 + ND)\\) | \\(O(ND)\\) | \\(O(N)\\) | \\(O(D)\\) |

Computational complexity comparison. \\(N\\) is the token number, \\(D\\) is the channel number.

</div>

</div>

<span id="complexity" label="complexity"></span>

##  No Scaling Factors in Q-K Attention

In VSA `\cite{vaswani2017attention}`{=latex}, assume that \\(\mathbf{q}_i\left(\mathbf{q}_i \in R^{1 \times d}, \mathbf{Q} \in R^{m \times d}\right)\\) and \\(\mathbf{k}_i\left(\mathbf{k}_i \in R^{1 \times d}, \mathbf{K} \in R^{m \times d}\right)\\) are independent random variables with a mean of 0 and a variance of 1, then each element in the product of \\(\mathbf{Q} \mathbf{K}^{\mathrm{T}}\\) has mean 0 and variance \\(d\\). The variance magnitude of \\(\mathbf{Q} \mathbf{K}^{\mathrm{T}}\\) grows with the embedding dimension \\(d\\), which can result in gradient vanishing issues after softmax operation. Therefore, The product of matrices \\(\mathbf{Q}\\) and \\(\mathbf{K}\\) in VSA `\cite{vaswani2017attention}`{=latex} is scaled by a factor \\(\frac{1}{\sqrt{d}}\\) in Eq. <a href="#vsa" data-reference-type="ref" data-reference="vsa">[vsa]</a> to normalize the product to variance 1. Though the softmax function is not adopted due to its non-spike operations (division, exponential operation) in SNNs, SSA-based `\cite{zhou2023spikformer}`{=latex} SNNs will suffer obvious performance degradation even cannot converge without scaling because the variance of \\(\mathbf{Q} \mathbf{K}^{\mathrm{T}} \mathbf{V}\\) output is too large (Assuming that all the spiking elements are independent random variables and subject to Bernoulli Distribution). However, Q-K attention can discard scaling operations thus reducing power consumption because the variance of Q-K attention is much smaller than SSA (e.g. the max theoretical variance of Q-K token attention is only about 1 / 200 of SSA). The detailed analysis can be found in the Appendix.<a href="#Scaling" data-reference-type="ref" data-reference="Scaling">6.2</a> and Section.<a href="#analysis" data-reference-type="ref" data-reference="analysis">4.3</a>.

## QKFormer

As the computational complexity (especially space complexity) of SSA is quadratic to \#tokens, previous direct training spiking transformers are all limited to straight-through structures. Combining SSA with hierarchical architecture directly will lead to memory explosion easily when training spiking transformers. To overcome these issues, we proposed a hierarchical spiking transformer based on Q-K attention, named QKFormer, which constructs hierarchical spiking feature maps with linear computational complexity to \#tokens or \#channels.

<figure id="fig: qkformer">
<img src="./figures/Model.png"" />
<figcaption>The overview of QKFormer, a hierarchical spiking transformer with Q-K attention.</figcaption>
</figure>

**Overall Hierarchical Architecture.** The overview of QKFormer is presented in Figure <a href="#fig: qkformer" data-reference-type="ref" data-reference="fig: qkformer">2</a>. The input form can be formulated as \\((T_{0}\times H\times W\times n)\\). In static RGB image datasets, \\(T_{0}=1\\) and \\(n = 3\\). In temporal neuromorphic datasets, the input \\(T_{0}=T\\), while \\(n = 2\\). In our implementation, we use a patch size of \\(4 \times 4\\) and thus the input feature dimension (\\(4 \times 4 \times n\\)) of each patch is projected into a spike-form arbitrary dimension (denoted as \\(C\\)) in Spiking Patch Embedding with Deformed Shortcut 1 (SPEDS-1), which together with the following QKFormer blocks are referred to as "Stage 1". The number of tokens in Satge 1 is \\((\frac{H}{4} \times \frac{W}{4})\\). To produce a hierarchical spiking representation, the number of tokens is reduced in SPEDS-2 and SPEDS-3 as the network goes deeper. Both SPEDS-2 and SPEDS-3 reduce the number of tokens by a patch size of \\(2 \times 2\\) (\\(2 \times\\) downsampling of resolution), and the number of channels is transformed into \\(2C\\) and \\(4C\\), respectively. We denote the SPEDS-2 and the following QKFormer blocks as "Stage 2", which reduces the number of tokens \\((\frac{H}{8} \times \frac{W}{8})\\). While SPEDS-3 and the following Spikformer or QKormer blocks are referred to as "Stage 3" with \\((\frac{H}{16} \times \frac{W}{16})\\) tokens. The number of spiking transformer blocks (QKFormer or Spikformer) in each stage are \\(N_{1}\\), \\(N_{2}\\), and \\(N_{3}\\), respectively. These stages jointly produce a hierarchical spike-form representation.

**Mixed Spiking Attention Integration.** In the former stage of a hierarchical architecture model, the number of channels is small while the number of tokens is large. In the latter stage, the channel number is large while the token number is small. Thus it leads to suboptimal performance when we only use a single type of Q-K attention in a hierarchical architecture model. Therefore, we use mixed spiking attention integration in QKFormer. QKTA is conducted in the former stage in hierarchical architecture, and we could choose QKCA or SSA in the latter stage. In the subsequent experiments, we use SSA in the last stage of QKFormer and QKTA in the former stages by default.

**QKFormer Blocks.** Similar to the standard transformer encoder block, a QKFormer block contains a Q-K Attention module (QKTA or QKCA) and a Spiking MLP (SMLP) block, which can be formulated as follows: \\[\begin{aligned}
&\mathbf{X}_l^{\prime}=\operatorname{QKTA}\left(\mathbf{X}_{l-1}\right)+\mathbf{X}_{l-1},  \mathbf{X}_l^{\prime} \in R^{T \times N \times D},\\
&\mathbf{X}_l=\operatorname{SMLP}\left(\mathbf{X}_l^{\prime}\right)+\mathbf{X}_l^{\prime},  \mathbf{X}_l \in R^{T \times N \times D}.
\end{aligned}\\] At last, a fully connected layer is used as the classifier behind the last block.

## Spiking Patch Embedding with Deformed Shortcut.

Residual shortcuts in SNNs `\cite{fang2021deep}`{=latex} can implement identity mapping, which reduces information loss (facilitates information transmission and integration) in spike communication, thus ensuring the network can be well-behaved in a depth-insensitive way. Previous spiking transformers `\cite{zhou2023spikformer, zhou2023spikingformer, yao2023spikedriven}`{=latex} use the residual shortcuts to achieve identity mapping, mainly focusing on the spiking attention block and spiking MLP block, and lacking identity mapping in patch embedding across the downsampling block. The input and output of a spiking patch embedding block in QKFormer have different channel and token numbers. To realize residual learning in spiking patch embedding, we can perform a lightweight linear projection \\(\mathbf{W}_d\\) in the shortcut connections to match the channel and token numbers, thus realizing the identity mapping cross downsampling blocks in spiking patch embedding. Given the input spiking map \\(\mathbf{X}\\), the process of patch embedding can be formulated as follows: \\[\mathbf{Y}=\mathcal{F}\left(\mathbf{X},\left\{\mathbf{W}_i\right\}\right) + \mathrm{SN} (\mathbf{W}_d \mathbf{X})
\label{deformed_a} .\\] In this work, the deformed linear projection \\(\mathbf{W}_{d}\\) is set as a lightweight convolutional layer with \\(1 \times 1\\) kernel and stride \\(> 1\\), to meet the channel and token numbers of the patch embedding block. The function \\(\mathcal{F}\\) involved in this work is set as \\(\{\\)Conv2D-BN-MaxPooling-SN-Conv2D-BN-SN\\(\}\\) or \\(\{\\)Conv2D-BN-SN-Conv2D-BN-MaxPooling-SN\\(\}\\), while more layers or more variants are possible.

There are mainly two types of residual shortcuts in deep SNNs. Formula.<a href="#deformed_a" data-reference-type="ref" data-reference="deformed_a">[deformed_a]</a> shows the patch embedding in the way of activation-before-addition `\cite{fang2021deep,zhou2023spikformer}`{=latex}. The other way of the patch embedding with the pre-activation residual shortcut `\cite{hu2021advancing,zhou2023spikingformer,yao2023spikedriven}`{=latex} can be formulated as follows: \\[\mathbf{Y} = \mathrm{SN}(\mathcal{G}\left(\mathbf{X},\left\{\mathbf{W}_j\right\}\right) + \mathbf{W}_d \mathbf{X}) ,\\] where the function \\(\mathcal{G}\\) correspondingly could be set as \\(\{\\)Conv2D-BN-MaxPooling-SN-Conv2D-BN\\(\}\\) or \\(\{\\)Conv2D-BN-SN-Conv2D-BN-MaxPooling\\(\}\\). The intuitive representation of SPEDS is shown in Appendix <a href="#Supplementary for SPEDS" data-reference-type="ref" data-reference="Supplementary for SPEDS">6.4</a>.

In this work, the spiking patch embedding of stage 2 or stage 3 in QKFormer can be formulated as Formula.<a href="#deformed_a" data-reference-type="ref" data-reference="deformed_a">[deformed_a]</a>. The spiking patch embedding in stage 1 uses an extra \\(\{\\)Conv2D-BN-SN\\(\}\\) for spiking encoding in front of the block (Formula.<a href="#deformed_a" data-reference-type="ref" data-reference="deformed_a">[deformed_a]</a>) to transform the non-spike input data into spikes.

<div id="tab:imagenet" markdown="1">

| Methods | Type | Architecture | Input Size | Param (M) | Power (mJ) | Time Step | Top-1 Acc (\\(\%\\)) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| RMP`\cite{han2020rmp}`{=latex} | A2S | VGG-16 | 224\\(^2\\) | 39.90 | \- | 2048 | 73.09 |
| 2-8 | A2S | ResNet-18 | 224\\(^2\\) | 11.70 | \- | 1024 | 74.32 |
| 2-8 | A2S | Swin Transformer-T | 224\\(^2\\) | 28.50 | \- | 512 | 78.51 |
| 1-8 | SNN | SEW-ResNet-34 | 224\\(^2\\) | 21.79 | 4.89 | 4 | 67.04 |
|  | SNN | SEW-ResNet-101 | 224\\(^2\\) | 44.55 | 8.91 | 4 | 68.76 |
|  | SNN | SEW-ResNet-152 | 224\\(^2\\) | 60.19 | 12.89 | 4 | 69.26 |
| 2-8 | SNN | Spikformer-8-384 | 224\\(^2\\) | 16.81 | 7.73 | 4 | 70.24 |
|  | SNN | Spikformer-8-512 | 224\\(^2\\) | 29.68 | 11.58 | 4 | 73.38 |
|  | SNN | Spikformer-8-768 | 224\\(^2\\) | 66.34 | 21.48 | 4 | 74.81 |
| 2-8 | SNN | Spikingformer-8-384 | 224\\(^2\\) | 16.81 | 4.69 | 4 | 72.45 |
|  | SNN | Spikingformer-8-512 | 224\\(^2\\) | 29.68 | 7.46 | 4 | 74.79 |
|  | SNN | Spikingformer-8-768 | 224\\(^2\\) | 66.34 | 13.68 | 4 | 75.85 |
| 2-8 | SNN | S-Transformer-8-384 | 224\\(^2\\) | 16.81 | 3.90 | 4 | 72.28 |
|  | SNN | S-Transformer-8-512 | 224\\(^2\\) | 29.68 | 1.13 | 1 | 71.68 |
|  | SNN | S-Transformer-8-512 | 224\\(^2\\) | 29.68 | 4.50 | 4 | 74.57 |
|  | SNN | S-Transformer-8-768\\(^\ast\\) | 288\\(^2\\) | 66.34 | 6.09 | 4 | 77.07 |
| 1-8 | ANN | ViT-B/16 | 384\\(^2\\) | 86.59 | 254.84 | 1 | 77.90 |
| 2-8 | ANN | DeiT-B | 224\\(^2\\) | 86.59 | 80.50 | 1 | 81.80 |
|  | ANN | DeiT-B | 384\\(^2\\) | 86.59 | 254.84 | 1 | 83.10 |
| 2-8 | ANN | Swin Transformer-B | 224\\(^2\\) | 87.77 | 70.84 | 1 | 83.50 |
|  | ANN | Swin Transformer-B | 384\\(^2\\) | 87.77 | 216.20 | 1 | 84.50 |
| 1-8 | SNN | HST-10-384 | 224\\(^2\\) | 16.47 | 15.13 | 4 | 78.80 |
|  | SNN | HST-10-512 | 224\\(^2\\) | 29.08 | 21.99 | 4 | 82.04 |
|  | SNN | HST-10-768 | 224\\(^2\\) | 64.96 | 8.52 | 1 | 81.69 |
|  | SNN | HST-10-768 | 224\\(^2\\) | 64.96 | 38.91 | 4 | 84.22 |
|  | SNN | HST-10-768\\(^\ast\\) | 288\\(^2\\) | 64.96 | 64.27 | 4 | 85.25 |
|  | SNN | HST-10-768\\(^{\ast\ast}\\) | 384\\(^2\\) | 64.96 | 113.64 | 4 | **85.65** |

Results on ImageNet-1K. Power is calculated as the average theoretical energy consumption when predicting an image from ImageNet test set. The power data for QKFormer and ANNs is evaluated according to Appendix.<a href="#sec：energy" data-reference-type="ref" data-reference="sec：energy">6.6</a>, and the power data for other works were obtained from related papers. "A2S" denotes "ANN-to-SNN", "HST-\\(L\\)-\\(D\\)" denotes "Hierarchical Spiking Transformer" with \\(L\\) encoder blocks and \\(D\\) channels. HST-10-768\\(^{\ast}\\) and HST-10-768\\(^{\ast\ast}\\) means HST-10-768 with 288\\(^2\\) and 384\\(^2\\) input size for inference. The top-5 accuracy of QKFormer (HST-10-768\\(^{\ast\ast}\\)) is 97.74%.

</div>

<span id="tab:imagenet" label="tab:imagenet"></span>

# Experiments [Experiments]

## Results on ImageNet-1k Classification

**Experimental Setup on ImageNet.** In this experiment, we use AdamW as the optimizer, which is adopted with a base learning rate of \\(6 \times 10^{-4}\\). The actual learning rate was calculated as \\(\text {BatchSize/256 }\\) multiplied by the base learning rate. The batch size is set to \\(512\\), which is realized by accumulated gradient iterations `\cite{he2022masked}`{=latex} and distributed across 8 Nvidia V100 GPUs. We trained QKFormer for \\(200\\) epochs. In addition, following DeiT `\cite{touvron2021training}`{=latex}, data augmentation techniques including RandAugment `\cite{cubuk2020randaugment}`{=latex}, random erasing `\cite{zhong2020random}`{=latex}, and stochastic depth `\cite{huang2016deep}`{=latex} are employed in this study. The number of blocks in the three stages is set as {1, 2, 7} respectively.

**Main Results on ImageNet.** The experimental results demonstrate the superior performance of our proposed QKFormer, surpassing previous works’ performance by a large margin (Table <a href="#tab:imagenet" data-reference-type="ref" data-reference="tab:imagenet">2</a>). QKFormer (**64.96 M**) achieves **85.65%** top-1 accuracy and **97.74%** top-5 accuracy on ImageNet. To begin with, we compare our model with the baseline spiking transformer (i.e., Spikformer `\cite{zhou2023spikformer}`{=latex}). Our QKFormer models have slightly fewer parameters but much higher performance. For example, our QKFormer (64.96 M, 85.65%) significantly outperforms Spikformer (66.34 M, 74.81%) by **10.84%**. In addition, compared with SDSA, our Q-K attention has lower computational complexity (shown in Table <a href="#complexity" data-reference-type="ref" data-reference="complexity">1</a>) and our QKFormer has much higher performance than S-Transformer (built by SDSA) `\cite{yao2023spikedriven}`{=latex}. In detail, QKFormer outperforms S-Transformer by 7.55%, 7.47%, and 8.58% respectively on three models with comparable \#parameters. Finally, Our QKFormer outperforms the SOTA ANN-to-SNN model MST `\cite{wang2023masked}`{=latex} by 7.14% and has much fewer time steps meanwhile. To our best knowledge, this is the first time that a direct training SNN model has achieved an accuracy of over **85%** on ImageNet-1k.

**Comparing with ANN Models on ImageNet.** Our QKFormer is an event-driven SNN model, whose output is in binary form (either 0 or 1), the multiplications of activations and weights can be transformed into sparse addition, thus enjoying high energy efficiency. It should be noted that hierarchical architecture will lead to the power increment of QKFormer. This is still very cost-effective compared with ANN models. For instance, QKFormer (**64.96M, 85.65%, SNN, 113.64mJ**) Vs. Swin Transformer (**88M, 84.5%, ANN, 216.20mJ**) `\cite{liu2021swin}`{=latex} Vs. DeiT-B (86M, 83.1%, ANN, 254.84mJ) `\cite{touvron2021training}`{=latex} Vs. ViT (85.59M, 77.9%, ANN, 254.84mJ). `\cite{dosovitskiy2020image}`{=latex}. Under the same experiment conditions without pre-training or extra training data, our QKFormer has surpassed the most well-known Transformer-based ANNs in performance while maintaining high energy efficiency.

<div id="tab:small_dataset" markdown="1">

<table>
<caption>Comparision on CIFAR10, CIFAR100, DVS128, CIFAR10-DVS. "Param" denotes "Parameter (M)", "Acc" denotes "Top-1 Accuracy (%)", "<span class="math inline"><em>T</em></span>" denotes "Time Step".</caption>
<thead>
<tr>
<th style="text-align: center;">Method</th>
<th colspan="3" style="text-align: center;">CIFAR10</th>
<th colspan="3" style="text-align: center;">CIFAR100</th>
<th colspan="3" style="text-align: center;">DVS128</th>
<th colspan="3" style="text-align: center;">CIFAR10-DVS</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">(l<span>2pt</span>r<span>2pt</span>)<span>2-4</span>(l<span>2pt</span>r<span>2pt</span>)<span>5-7</span>(l<span>2pt</span>r<span>2pt</span>)<span>8-10</span>(l<span>2pt</span>r<span>2pt</span>)<span>11-13</span></td>
<td style="text-align: left;"><span>Param</span></td>
<td style="text-align: left;"><span><span class="math inline"><em>T</em></span></span></td>
<td style="text-align: left;"><span>Acc</span></td>
<td style="text-align: left;">Param</td>
<td style="text-align: left;"><span><span class="math inline"><em>T</em></span></span></td>
<td style="text-align: left;"><span>Acc</span></td>
<td style="text-align: left;">Param</td>
<td style="text-align: left;"><span><span class="math inline"><em>T</em></span></span></td>
<td style="text-align: left;"><span>Acc</span></td>
<td style="text-align: left;">Param</td>
<td style="text-align: left;"><span><span class="math inline"><em>T</em></span></span></td>
<td style="text-align: left;"><span>Acc</span></td>
</tr>
<tr>
<td style="text-align: left;">Spikformer <span class="citation" data-cites="zhou2023spikformer"></span></td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">95.51</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">78.21</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">98.3</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">80.9</td>
</tr>
<tr>
<td style="text-align: left;">Spikingformer <span class="citation" data-cites="zhou2023spikingformer"></span></td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">95.81</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">78.21</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">98.3</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">81.3</td>
</tr>
<tr>
<td style="text-align: left;">CML <span class="citation" data-cites="zhou2023enhancing"></span></td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">96.04</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">80.02</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">98.6</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">80.9</td>
</tr>
<tr>
<td style="text-align: left;">S-Transformer<span class="citation" data-cites="yao2023spikedriven"></span></td>
<td style="text-align: left;">10.28</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">95.60</td>
<td style="text-align: left;">10.28</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">78.4</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;"><strong>99.3</strong></td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">80.0</td>
</tr>
<tr>
<td style="text-align: left;">STSA<span class="citation" data-cites="ijcai2023p344"></span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;">1.99</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">98.7</td>
<td style="text-align: left;">1.99</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">79.93</td>
</tr>
<tr>
<td style="text-align: left;"><span>2-13</span> ResNet-19 (ANN)</td>
<td style="text-align: left;">12.63</td>
<td style="text-align: left;">1</td>
<td style="text-align: left;">94.97</td>
<td style="text-align: left;">12.63</td>
<td style="text-align: left;">1</td>
<td style="text-align: left;">75.35</td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
</tr>
<tr>
<td style="text-align: left;">Trasnformer (ANN)</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">1</td>
<td style="text-align: left;">96.73</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">1</td>
<td style="text-align: left;">81.02</td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
<td style="text-align: left;"><span class="math inline">−</span></td>
</tr>
<tr>
<td style="text-align: left;"><strong>QKFormer</strong></td>
<td style="text-align: left;">6.74</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;"><strong>96.18</strong></td>
<td style="text-align: left;">6.74</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;"><strong>81.15</strong></td>
<td style="text-align: left;">1.50</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;">98.6</td>
<td style="text-align: left;">1.50</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;"><strong>84.0</strong></td>
</tr>
</tbody>
</table>

</div>

<span id="tab:small_dataset" label="tab:small_dataset"></span>

## Results on CIFAR and Neuromorphic Datasets

**CIFAR Classification.** In this experiment, the QKFormer is trained for 400 epochs with a batch size of 64 following previous works: Spikformer `\cite{zhou2023spikformer}`{=latex}, Spikingformer `\cite{zhou2023spikingformer}`{=latex}. Following Spikformer, we use 4 blocks in QKFormer in total, which are distributed {1, 1, 2} in three stages. Due to the hierarchical architecture design, our QKFormer model has only 6.74 M parameters in this case. The results on CIFAR datasets are shown in Table <a href="#tab:small_dataset" data-reference-type="ref" data-reference="tab:small_dataset">3</a>. For CIFAR10, our model achieved **96.18%** accuracy with **6.74 M** parameters. Our proposed QKFormer outperforms Spikformer by 0.67% and reduces 2.58 M parameters meanwhile. For CIFAR100, our model achieved **81.15%** with 6.74 M parameters. Our proposed QKFormer outperforms Spikformer by **2.94%** and reduces 2.58 M parameters meanwhile.

**Neuromorphic Classification.** We compare our method with SOTA methods on both CIFAR10-DVS and DVS-Gesture datasets. In this experiment, We utilize a mini QKFormer model with 1.50 M parameter, which has {0, 1, 1} blocks in three stages. The max patch embedding dimension is set to 256. The training process involves 200 epochs for DVS128 Gesture and 106 epochs for CIFAR10-DVS. The number of time steps of the spiking neuron is 10 or 16. The experimental results of temporal neuromorphic classification are presented in Table <a href="#tab:small_dataset" data-reference-type="ref" data-reference="tab:small_dataset">3</a>. For DVS128-Gesture dataset, our model with 1.50 M parameters achieves 98.6% accuracy using 16 time steps and 98.3% accuracy using 10 time steps. For CIFAR10-DVS dataset, our model achieves **84.0%** accuracy with only **1.50 M** parameters using 16 time steps. Our proposed QKFormer significantly outperforms Spikformer by **3.1%** while reducing 1.07 M parameters. In addition, our model with 10 time steps achieves 83.8% accuracy, which outperforms Spikformer by **4.9%** and outperforms the SOTA model (Spikingformer) by 3.9%.

<figure id="fig:QKTA">

<figcaption>The visualization and memory consumption of QKTA. is the visualization of Q-K token attention. The white dot means value 1, while the black one means value 0. shows the comparison of memory costs between QKTA and SSA under different token numbers. <span class="math inline"><em>N</em></span> is the token number. </figcaption>
</figure>

<figure id="fig: Variance and Expectation">

<figcaption> (a) shows the variance and expectation of SSA, (b) shows the variance and expectation of QKTA. Assume that all the spike elements (either 0 or 1) in SSA and QKTA are independent random variables and subject to Bernoulli distribution.</figcaption>
</figure>

## Analyses on Q-K Attention [analysis]

**Attention Visualization.** In this part, we visualize the Q-K token attention (Stage 1 and Stage 2 of the QKFormer model) on ImageNet. As shown in Figure <a href="#fig: akta_a" data-reference-type="ref" data-reference="fig: akta_a">[fig: akta_a]</a>, \\({\mathbf{A}}_t\\) is the \\(N*1\\) token attention vector, and \\(\mathbf{X}^{\prime}\\) is the output of the attention process, which is obtained by the mask operation between matrix \\(\mathbf{K}\\) and attention vector \\({\mathbf{A}}_t\\). Specifically, the longitudinal axis denotes the channel index of one head, while the horizontal axis denotes the token index. The \#tokens in stage 1 and stage 2 are \\(56^2\\) and \\(28^2\\), respectively. To facilitate visualization, we choose a continuous segment with a length of 100 extracted from the whole token vector. The visualization shows Q-K attention can lead to high sparsity of spikes.

**Memory Consumption.** In this experiment, we compare the memory consumption between QKTA (Formula.<a href="#QKTA" data-reference-type="ref" data-reference="QKTA">[QKTA]</a>) and SSA (Formula.<a href="#SSA" data-reference-type="ref" data-reference="SSA">[SSA]</a>) under different token numbers. We calculate the memory consumption of a QKTA and an SSA on a GPU by forwarding the input tensor \\((T, B, C, N)\\). To facilitate the statistics of the impact of \#tokens \\(N\\) on memory consumption, the \#channels \\(C\\) is set to 256, and the time step \\(T\\) and batch size \\(B\\) are set to 1. The experiment result is shown in Figure <a href="#fig: qkta_b" data-reference-type="ref" data-reference="fig: qkta_b">[fig: qkta_b]</a>. With the increment of \#Tokens, SSA consumes much more GPU memory than QKTA, of which the complexity is linear to \#Tokens. For example, SSA consumes about \\(10\times\\) GPU memory than QKTA when \\(\sqrt{N} = 50\\).

<div id="tab:firing-rate" markdown="1">

<table>
<caption>Spike firing rates in QKFormer blocks. </caption>
<thead>
<tr>
<th colspan="2" style="text-align: center;">QKFormer Block</th>
<th style="text-align: left;">Stage1 (fr)</th>
<th style="text-align: left;">Stage2 (fr)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="5" style="text-align: left;">QKTA</td>
<td style="text-align: left;"><span class="math inline"><strong>Q</strong></span></td>
<td style="text-align: left;">0.0432</td>
<td style="text-align: left;">0.0231</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><strong>K</strong></span></td>
<td style="text-align: left;">0.1784</td>
<td style="text-align: left;">0.0847</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><strong>A</strong><sub><em>t</em></sub></span></td>
<td style="text-align: left;">0.3477</td>
<td style="text-align: left;">0.2655</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><strong>X</strong><sup>′</sup></span></td>
<td style="text-align: left;">0.0832</td>
<td style="text-align: left;">0.0350</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><strong>X</strong><sup>′′</sup></span></td>
<td style="text-align: left;">0.1478</td>
<td style="text-align: left;">0.0577</td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;">SMLP</td>
<td style="text-align: left;">Layer1</td>
<td style="text-align: left;">0.0518</td>
<td style="text-align: left;">0.0246</td>
</tr>
<tr>
<td style="text-align: left;">Layer2</td>
<td style="text-align: left;">0.2733</td>
<td style="text-align: left;">0.1869</td>
</tr>
</tbody>
</table>

</div>

<span id="tab:firing-rate" label="tab:firing-rate"></span>

**Spike Firing Rates in QKFormer Blocks.** In this experiment, we calculate the spike firing rates of QKFormer blocks of the trained QKFormer (64.9M) on the ImageNet-1K test set with the 224 \\(\times\\) 224 input resolution. The average spike firing rates of the QKFormer blocks in Stage1 and Stage2 are shown in Table <a href="#tab:firing-rate" data-reference-type="ref" data-reference="tab:firing-rate">4</a>. Note that the spike-form \\(\mathbf{X}^{\prime}\\) is obtained by column mask operation (token mask) between \\({\mathbf{A}}_t\\) and \\(\mathbf{K}\\). In fact, the summation operation in the Q-K attention causes \\(\mathbf{Q}\\) to become significantly sparser compared to \\(K\\) when the network converges. Specifically, \\(\mathbf{Q}\\) in stage 1 has a fire rate of 0.0432, while \\(\mathbf{K}\\) has 0.1784. After the accumulation operation along \\(D/h\\) of the multi-head QKTA version, the LIF neuron (\\({\mathbf{A}}_t\\)) exhibits a typical averaged fire rate of 0.3477.

**The Variance and Expectation of QKTA.** Figure.<a href="#fig: Variance and Expectation" data-reference-type="ref" data-reference="fig: Variance and Expectation">4</a> visualize the variance and expectation of QKTA (Formula.<a href="#QKTA_E" data-reference-type="ref" data-reference="QKTA_E">[QKTA_E]</a> and <a href="#QKTA_V" data-reference-type="ref" data-reference="QKTA_V">[QKTA_V]</a> in Appendix.<a href="#Scaling" data-reference-type="ref" data-reference="Scaling">6.2</a>) and SSA (Formula.<a href="#SSA_E" data-reference-type="ref" data-reference="SSA_E">[SSA_E]</a> and <a href="#SSA_V" data-reference-type="ref" data-reference="SSA_V">[SSA_V]</a> in Appendix.<a href="#Scaling" data-reference-type="ref" data-reference="Scaling">6.2</a>). \\(N\\) is set as 196 and \\(d\\) is set as 64, respectively. We can find SSA has a much larger variance and expectation than QKTA on the whole. For example, the maximum theoretical variance of QKTA is 16, but the maximum theoretical variance of SSA is over 3000. This is the main reason that Q-K attention can discard scaling operations thus reducing power consumption, but SSA can not.

## Ablation Study

**SPEDS Module.** In this experiment, We replaced the Spiking Patch Splitting (SPS) module in Spikformer with Spiking Patch Embedding with Deformed Shortcut (SPEDS) module, while other conditions remain unchanged. The results (Table <a href="#tab:ablation_study_1" data-reference-type="ref" data-reference="tab:ablation_study_1">5</a>) show that the SPEDS module is essential to QKFormer on both static and neuromorphic datasets. In addition, the addition of SPEDS to Spikformer leads to great gains in CIFAR100 (+2.05%) and CIFAR10-DVS (+1.30%), which further verified the effectiveness of SPEDS.

<div id="tab:ablation_study_1" markdown="1">

| Model                            | CIFAR100 (Acc) | CIFAR10-DVS (Acc) |
|:---------------------------------|:---------------|:------------------|
| QKFormer (QKTA + SSA, baseline)  | 81.15%         | 84.00%            |
| QKFormer (QKTA + SSA, w/o SPEDS) | 80.08%         | 83.40%            |
| Spikformer (SSA, w/o scaling)    | 76.95%         | 79.30%            |
| Spikformer (SSA)                 | 78.21%         | 80.90%            |
| Spikformer (SSA) + SPEDS         | 80.26%         | 82.20%            |

Ablation studies of SPEDS module.

</div>

<span id="tab:ablation_study_1" label="tab:ablation_study_1"></span>

<div id="tab:ablation_study_2" markdown="1">

| Model | CIFAR100 (Acc, Param) | CIFAR10-DVS (Acc, Param) |
|:---|:---|:---|
| QKFormer (QKTA + SSA, baseline) | 81.15%, 6.74M | 84.00%, 1.50M |
| QKFormer (QKCA + SSA) | 81.07%, 6.74M | 84.30%, 1.50M |
| QKFormer (QKTA + QKCA) | 81.04%, 6.44M | 83.10%, 1.44M |
| QKFormer (SSA) | 81.23%, 6.79M | 84.10%, 1.52M |
| QKFormer (QKCA) | 81.00%, 6.44M | 80.70%, 1.44M |
| QKFormer (QKTA) | 79.09%, 6.44M | 80.70%, 1.44M |

Ablation studies of Q-K Attention.

</div>

<span id="tab:ablation_study_2" label="tab:ablation_study_2"></span>

**Mixed Spiking Attention Integration with Q-K Attention.** In this part, we show different integration strategies of QKCA, QKTA, and SSA. The baseline is our QKFormer (QKTA + SSA, 6.70M). The experimental results (Table <a href="#tab:ablation_study_2" data-reference-type="ref" data-reference="tab:ablation_study_2">6</a>) show that using a single type of Q-K attention (QKTA or QKCA only) in a hierarchical architecture model leads to suboptimal performance. In particular, the performance decline in QKTA is more obvious. While the mixed spiking attention solutions, such as QKFormer(QKTA + QKCA), QKFormer(QKTA + SSA), and QKFormer(QKCA + SSA) can achieve comparable performance to QKFormer(SSA) while requiring fewer parameters and much fewer memory resources (Figure <a href="#fig: qkta_b" data-reference-type="ref" data-reference="fig: qkta_b">[fig: qkta_b]</a>). Consequently, the mixed spiking attention solutions are well-suited for larger architectures and more challenging scenarios when considering both computational efficiency and performance.

<div id="tab:ablation_study_3" markdown="1">

| Model                 | CIFAR100 (Acc) |
|:----------------------|:---------------|
| QKFormer (baseline)   | 81.15%         |
| QKFormer (ABA-\>PA)   | 81.18%         |
| QKFormer (LIF-\>IF)   | 80.95%         |
| QKFormer (LIF-\>PLIF) | 81.12%         |
| QKFormer (T=1)        | 78.51%         |
| QKFormer (T=2)        | 80.08%         |
| QKFormer (T=4)        | 81.15%         |
| QKFormer (T=6)        | 81.30%         |

Ablation studies of RC, SN, TS.

</div>

<span id="tab:ablation_study_3" label="tab:ablation_study_3"></span>

**Residual Connection (RC) & Spiking Neuron (SN) & Time Step (TS).** The experimental results are shown in Table <a href="#tab:ablation_study_3" data-reference-type="ref" data-reference="tab:ablation_study_3">7</a>. In this block, we replaced the Activation-Before-Addition (ABA) `\cite{fang2021deep, zhou2023spikformer}`{=latex} residual connection of QKFormer with the Pre-activation (PA) `\cite{hu2021advancing, zhou2024direct}`{=latex} way, and the performance slightly improved. In addition, we replaced the LIF spiking neuron with Integrate-and-Fire (IF) and Parametric-Leaky-Integrate-and-Fire (PLIF) `\cite{fang2021incorporating}`{=latex}, which led to slight performance degradation. The accuracy regarding different simulation time steps of QKFormer is shown in the last column. When we increase the simulation time steps, the performance of QKFormer can be further improved. Specifically, QKFormer achieves 81.30 % accuracy on CIFAR 100 when T=6.

# Conclusion

In this work we introduced QKFormer, a hierarchical spiking transformer that unifies spike-form Q-K attention with the SPEDS patch-embedding strategy to produce state-of-the-art performance across static and neuromorphic benchmarks. The linear complexity of Q-K attention, combined with a carefully selected temporal horizon of eight time steps, allows QKFormer to scale gracefully while retaining the event-driven sparsity that underpins the energy efficiency of SNNs. Extensive experiments confirm that QKFormer not only surpasses prior SNNs by a substantial margin (+10.84 % on ImageNet-1K) but also rivals leading ANN counterparts, all without sacrificing the practical deployability inherent to spiking models.

Far from being a drawback, our multi-step temporal formulation is fundamental to the model’s success: the additional steps enrich temporal feature extraction, stabilise training, and ultimately shorten wall-clock optimisation time thanks to faster convergence. Going forward, we will apply the same design principles to more challenging domains such as semantic segmentation, object detection and sequence modelling, expecting the hierarchical temporal dynamics of QKFormer to transfer seamlessly. With its compelling blend of accuracy, scalability and energy efficiency, QKFormer sets a new baseline for future explorations in spike-based deep learning.
# References [references]

<div class="thebibliography" markdown="1">

Wolfgang Maass Networks of spiking neurons: the third generation of neural network models , 10(9):1659–1671, 1997. (@maass1997networks)

Kaushik Roy, Akhilesh Jaiswal, and Priyadarshini Panda Towards spike-based machine intelligence with neuromorphic computing , 575(7784):607–617, 2019. **Abstract:** Biologically plausible learning with neuronal dendrites is a promising perspective to improve the spike-driven learning capability by introducing dendritic processing as an additional hyperparameter. Neuromorphic computing is an effective and essential solution towards spike-based machine intelligence and neural learning systems. However, on-line learning capability for neuromorphic models is still an open challenge. In this study a novel neuromorphic architecture with dendritic on-line learning (NADOL) is presented, which is a novel efficient methodology for brain-inspired intelligence on embedded hardware. With the feature of distributed processing using spiking neural network, NADOL can cut down the power consumption and enhance the learning efficiency and convergence speed. A detailed analysis for NADOL is presented, which demonstrates the effects of different conditions on learning capabilities, including neuron number in hidden layer, dendritic segregation parameters, feedback connection, and connection sparseness with various levels of amplification. Piecewise linear approximation approach is used to cut down the computational resource cost. The experimental results demonstrate a remarkable learning capability that surpasses other solutions, with NADOL exhibiting superior performance over the GPU platform in dendritic learning. This study’s applicability extends across diverse domains, including the Internet of Things, robotic control, and brain-machine interfaces. Moreover, it signifies a pivotal step in bridging the gap between artificial intelligence and neuroscience through the introduction of an innovative neuromorphic paradigm. (@roy2019towards)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin Attention is all you need In *Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS)*, volume 30, 2017. **Abstract:** The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. (@vaswani2017attention)

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al An image is worth 16x16 words: Transformers for image recognition at scale In *International Conference on Learning Representa- tions (ICLR)*, 2020. **Abstract:** While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train. (@dosovitskiy2020image)

Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan Tokens-to-token vit: Training vision transformers from scratch on imagenet In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 558–567, 2021. **Abstract:** Transformers, which are popular for language modeling, have been explored for solving vision tasks recently, e.g., the Vision Transformer (ViT) for image classification. The ViT model splits each image into a sequence of tokens with fixed length and then applies multiple Transformer layers to model their global relation for classification. However, ViT achieves inferior performance to CNNs when trained from scratch on a midsize dataset like ImageNet. We find it is because: 1) the simple tokenization of input images fails to model the important local structure such as edges and lines among neighboring pixels, leading to low training sample efficiency; 2) the redundant attention backbone design of ViT leads to limited feature richness for fixed computation budgets and limited training samples. To overcome such limitations, we propose a new Tokens-To-Token Vision Transformer (T2T-VTT), which incorporates 1) a layer-wise Tokens-to-Token (T2T) transformation to progressively structurize the image to tokens by recursively aggregating neighboring Tokens into one Token (Tokens-to-Token), such that local structure represented by surrounding tokens can be modeled and tokens length can be reduced; 2) an efficient backbone with a deep-narrow structure for vision transformer motivated by CNN architecture design after empirical study. Notably, T2T-ViT reduces the parameter count and MACs of vanilla ViT by half, while achieving more than 3.0% improvement when trained from scratch on ImageNet. It also outperforms ResNets and achieves comparable performance with MobileNets by directly training on ImageNet. For example, T2T-ViT with comparable size to ResNet50 (21.5M parameters) can achieve 83.3% top1 accuracy in image resolution 384x384 on ImageNet. \<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>1\</sup\> (@yuan2021tokens)

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko End-to-end object detection with transformers In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 213–229. Springer, 2020. **Abstract:** We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, e ectively removing the need for many hand-designed compo- nents like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bi- partite matching, and a transformer encoder-decoder architecture. Given a xed small set of learned object queries, DETR reasons about the re- lations of the objects and the global image context to directly output the nal set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time perfor- mance on par with the well-established and highly-optimized Faster R- CNN baseline on the challenging COCO object detection dataset. More- over, DETR can be easily generalized to produce panoptic segmentation in a uni ed manner. We show that it signi cantly outperforms com- petitive baselines. Training code and pretrained models are available at https://github.com/facebookresearch/detr . 1 Introduction The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest. Modern detectors address this set prediction task in an indirect way, by de ning surrogate regression and classi cation prob- lems on a large set of proposals \[37,5\], anchors \[23\], or window centers \[53,46\]. Their performances are signi cantly in uenced by postprocessing steps to col- lapse near-duplicate predictions, by the design of the anchor sets and by the heuristics that assign target boxes to anchors \[52\]. To simplify these pipelines, we propose a direct set prediction approach to bypass the surrogate tasks. This end-to-end philosophy has led to signi cant advances in complex structured pre- diction tasks such as machine translation or speech recognition, but not yet in object detection: previous attempts \[43,16,4,39\] either add other forms of prior knowledge, or have not proven to be competitive with strong baselines on chal- lenging benchmarks. This paper aims to bridge this gap. ?Equal contributionarXiv:2005.12872v3 \[cs.CV\] 28 May 20202 Carion et al. transformer encoder-decoderCNNset of box predictionsbipartite matching lossno object (ø)no object (ø)set of im (@carion2020end)

Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai Deformable detr: Deformable transformers for end-to-end object detection , 2020. **Abstract:** DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach. Code is released at https://github.com/fundamentalvision/Deformable-DETR. (@zhu2020deformable)

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo Swin transformer: Hierarchical vision transformer using shifted windows In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 10012–10022, 2021. **Abstract:** This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at https://github.com/microsoft/Swin-Transformer. (@liu2021swin)

Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao Pyramid vision transformer: A versatile backbone for dense prediction without convolutions In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 568–578, 2021. **Abstract:** Although convolutional neural networks (CNNs) have achieved great success in computer vision, this work investigates a simpler, convolution-free backbone network use-fid for many dense prediction tasks. Unlike the recently-proposed Vision Transformer (ViT) that was designed for image classification specifically, we introduce the Pyramid Vision Transformer (PVT), which overcomes the difficulties of porting Transformer to various dense prediction tasks. PVT has several merits compared to current state of the arts. (1) Different from ViT that typically yields low-resolution outputs and incurs high computational and memory costs, PVT not only can be trained on dense partitions of an image to achieve high output resolution, which is important for dense prediction, but also uses a progressive shrinking pyramid to reduce the computations of large feature maps. (2) PVT inherits the advantages of both CNN and Transformer, making it a unified backbone for various vision tasks without convolutions, where it can be used as a direct replacement for CNN backbones. (3) We validate PVT through extensive experiments, showing that it boosts the performance of many downstream tasks, including object detection, instance and semantic segmentation. For example, with a comparable number of parameters, PVT+RetinaNet achieves 40.4 AP on the COCO dataset, surpassing ResNet50+RetinNet (36.3 AP) by 4.1 absolute AP (see Figure 2). We hope that PVT could, serre as an alternative and useful backbone for pixel-level predictions and facilitate future research. (@wang2021pyramid)

Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, and Shuicheng Yan Volo: Vision outlooker for visual recognition , 2021. **Abstract:** Visual recognition has been dominated by convolutional neural networks (CNNs) for years. Though recently the prevailing vision transformers (ViTs) have shown great potential of self-attention based models in ImageNet classification, their performance is still inferior to that of the latest SOTA CNNs if no extra data are provided. In this work, we try to close the performance gap and demonstrate that attention-based models are indeed able to outperform CNNs. We find a major factor limiting the performance of ViTs for ImageNet classification is their low efficacy in encoding fine-level features into the token representations. To resolve this, we introduce a novel outlook attention and present a simple and general architecture, termed Vision Outlooker (VOLO). Unlike self-attention that focuses on global dependency modeling at a coarse level, the outlook attention efficiently encodes finer-level features and contexts into tokens, which is shown to be critically beneficial to recognition performance but largely ignored by the self-attention. Experiments show that our VOLO achieves 87.1% top-1 accuracy on ImageNet-1K classification, which is the first model exceeding 87% accuracy on this competitive benchmark, without using any extra training data In addition, the pre-trained VOLO transfers well to downstream tasks, such as semantic segmentation. We achieve 84.3% mIoU score on the cityscapes validation set and 54.3% on the ADE20K validation set. Code is available at \\}url{https://github.com/sail-sg/volo}. (@yuan2021volo)

Zhaokun Zhou, Yuesheng Zhu, Chao He, Yaowei Wang, Shuicheng YAN, Yonghong Tian, and Li Yuan Spikformer: When spiking neural network meets transformer In *The Eleventh International Conference on Learning Representations*, 2023. **Abstract:** We consider two biologically plausible structures, the Spiking Neural Network (SNN) and the self-attention mechanism. The former offers an energy-efficient and event-driven paradigm for deep learning, while the latter has the ability to capture feature dependencies, enabling Transformer to achieve good performance. It is intuitively promising to explore the marriage between them. In this paper, we consider leveraging both self-attention capability and biological properties of SNNs, and propose a novel Spiking Self Attention (SSA) as well as a powerful framework, named Spiking Transformer (Spikformer). The SSA mechanism in Spikformer models the sparse visual feature by using spike-form Query, Key, and Value without softmax. Since its computation is sparse and avoids multiplication, SSA is efficient and has low computational energy consumption. It is shown that Spikformer with SSA can outperform the state-of-the-art SNNs-like frameworks in image classification on both neuromorphic and static datasets. Spikformer (66.3M parameters) with comparable size to SEW-ResNet-152 (60.2M,69.26%) can achieve 74.81% top1 accuracy on ImageNet using 4 time steps, which is the state-of-the-art in directly trained SNNs models. (@zhou2023spikformer)

Chenlin Zhou, Liutao Yu, Zhaokun Zhou, Han Zhang, Zhengyu Ma, Huihui Zhou, and Yonghong Tian Spikingformer: Spike-driven residual learning for transformer-based spiking neural network 2023. **Abstract:** Spiking neural networks (SNNs) offer a promising energy-efficient alternative to artificial neural networks, due to their event-driven spiking computation. However, state-of-the-art deep SNNs (including Spikformer and SEW ResNet) suffer from non-spike computations (integer-float multiplications) caused by the structure of their residual connection. These non-spike computations increase SNNs’ power consumption and make them unsuitable for deployment on mainstream neuromorphic hardware, which only supports spike operations. In this paper, we propose a hardware-friendly spike-driven residual learning architecture for SNNs to avoid non-spike computations. Based on this residual design, we develop Spikingformer, a pure transformer-based spiking neural network. We evaluate Spikingformer on ImageNet, CIFAR10, CIFAR100, CIFAR10-DVS and DVS128 Gesture datasets, and demonstrate that Spikingformer outperforms the state-of-the-art in directly trained pure SNNs as a novel advanced backbone (75.85$\\}%$ top-1 accuracy on ImageNet, + 1.04$\\}%$ compared with Spikformer). Furthermore, our experiments verify that Spikingformer effectively avoids non-spike computations and significantly reduces energy consumption by 57.34$\\}%$ compared with Spikformer on ImageNet. To our best knowledge, this is the first time that a pure event-driven transformer-based SNN has been developed. (@zhou2023spikingformer)

Man Yao, Jiakui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo Xu, and Guoqi Li Spike-driven transformer 2023. **Abstract:** Spiking Neural Networks (SNNs) provide an energy-efficient deep learning option due to their unique spike-based event-driven (i.e., spike-driven) paradigm. In this paper, we incorporate the spike-driven paradigm into Transformer by the proposed Spike-driven Transformer with four unique properties: 1) Event-driven, no calculation is triggered when the input of Transformer is zero; 2) Binary spike communication, all matrix multiplications associated with the spike matrix can be transformed into sparse additions; 3) Self-attention with linear complexity at both token and channel dimensions; 4) The operations between spike-form Query, Key, and Value are mask and addition. Together, there are only sparse addition operations in the Spike-driven Transformer. To this end, we design a novel Spike-Driven Self-Attention (SDSA), which exploits only mask and addition operations without any multiplication, and thus having up to $87.2\\}times$ lower computation energy than vanilla self-attention. Especially in SDSA, the matrix multiplication between Query, Key, and Value is designed as the mask operation. In addition, we rearrange all residual connections in the vanilla Transformer before the activation functions to ensure that all neurons transmit binary spike signals. It is shown that the Spike-driven Transformer can achieve 77.1\\}% top-1 accuracy on ImageNet-1K, which is the state-of-the-art result in the SNN field. The source code is available at https://github.com/BICLab/Spike-Driven-Transformer. (@yao2023spikedriven)

Chenlin Zhou, Han Zhang, Zhaokun Zhou, Liutao Yu, Zhengyu Ma, Huihui Zhou, Xiaopeng Fan, and Yonghong Tian Enhancing the performance of transformer-based spiking neural networks by improved downsampling with precise gradient backpropagation 2023. **Abstract:** Deep spiking neural networks (SNNs) have drawn much attention in recent years because of their low power consumption, biological rationality and event-driven property. However, state-of-the-art deep SNNs (including Spikformer and Spikingformer) suffer from a critical challenge related to the imprecise gradient backpropagation. This problem arises from the improper design of downsampling modules in these networks, and greatly hampering the overall model performance. In this paper, we propose ConvBN-MaxPooling-LIF (CML), an SNN-optimized downsampling with precise gradient backpropagation. We prove that CML can effectively overcome the imprecision of gradient backpropagation from a theoretical perspective. In addition, we evaluate CML on ImageNet, CIFAR10, CIFAR100, CIFAR10-DVS, DVS128-Gesture datasets, and show state-of-the-art performance on all these datasets with significantly enhanced performances compared with Spikingformer. For instance, our model achieves 77.64 $\\}%$ on ImageNet, 96.04 $\\}%$ on CIFAR10, 81.4$\\}%$ on CIFAR10-DVS, with + 1.79$\\}%$ on ImageNet, +1.16$\\}%$ on CIFAR100 compared with Spikingformer. (@zhou2023enhancing)

Yuchen Wang, Kexin Shi, Chengzhuo Lu, Yuguo Liu, Malu Zhang, and Hong Qu Spatial-temporal self-attention for asynchronous spiking neural networks In Edith Elkind, editor, *Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23*, pages 3085–3093. International Joint Conferences on Artificial Intelligence Organization, 8 2023. Main Track. **Abstract:** The brain-inspired spiking neural networks (SNNs) are receiving increasing attention due to their asynchronous event-driven characteristics and low power consumption. As attention mechanisms recently become an indispensable part of sequence dependence modeling, the combination of SNNs and attention mechanisms holds great potential for energy-efficient and high-performance computing paradigms. However, the existing works cannot benefit from both temporal-wise attention and the asynchronous characteristic of SNNs. To fully leverage the advantages of both SNNs and attention mechanisms, we propose an SNNs-based spatial-temporal self-attention (STSA) mechanism, which calculates the feature dependence across the time and space domains without destroying the asynchronous transmission properties of SNNs. To further improve the performance, we also propose a spatial-temporal relative position bias (STRPB) for STSA to consider the spatiotemporal position of spikes. Based on the STSA and STRPB, we construct a spatial-temporal spiking Transformer framework, named STS-Transformer, which is powerful and enables SNNs to work in an asynchronous event-driven manner. Extensive experiments are conducted on popular neuromorphic datasets and speech datasets, including DVS128 Gesture, CIFAR10-DVS, and Google Speech Commands, and our experimental results can outperform other state-of-the-art models. (@ijcai2023p344)

Rong Wang, Mianxin Liu, Xinhong Cheng, Ying Wu, Andrea Hildebrandt, and Changsong Zhou Segregation, integration, and balance of large-scale resting brain networks configure different cognitive abilities , 118(23):e2022288118, 2021. **Abstract:** Diverse cognitive processes set different demands on locally segregated and globally integrated brain activity. However, it remains unclear how resting brains configure their functional organization to balance the demands on network segregation and integration to best serve cognition. Here, we use an eigenmode-based approach to identify hierarchical modules in functional brain networks, and quantify the functional balance between network segregation and integration. In a large sample of healthy young adults (n=991), we combine the whole-brain resting state functional magnetic resonance imaging (fMRI) data with a mean-filed model on the structural network derived from diffusion tensor imaging and demonstrate that resting brain networks are on average close to a balanced state. This state allows for a balanced time dwelling at segregated and integrated configurations, and highly flexible switching between them. Furthermore, we employ structural equation modelling to estimate general and domain-specific cognitive phenotypes from nine tasks, and demonstrate that network segregation, integration and their balance in resting brains predict individual differences in diverse cognitive phenotypes. More specifically, stronger integration is associated with better general cognitive ability, stronger segregation fosters crystallized intelligence and processing speed, and individual’s tendency towards balance supports better memory. Our findings provide a comprehensive and deep understanding of the brain’s functioning principles in supporting diverse functional demands and cognitive abilities, and advance modern network neuroscience theories of human cognition. (@wang2021segregation)

Yongqiang Cao, Yang Chen, and Deepak Khosla Spiking deep convolutional neural networks for energy-efficient object recognition , 113(1):54–66, 2015. (@cao2015spiking)

Eric Hunsberger and Chris Eliasmith Spiking deep networks with lif neurons , 2015. **Abstract:** We train spiking deep networks using leaky integrate-and-fire (LIF) neurons, and achieve state-of-the-art results for spiking networks on the CIFAR-10 and MNIST datasets. This demonstrates that biologically-plausible spiking LIF neurons can be integrated into deep networks can perform as well as other spiking models (e.g. integrate-and-fire). We achieved this result by softening the LIF response function, such that its derivative remains bounded, and by training the network with noise to provide robustness against the variability introduced by spikes. Our method is general and could be applied to other neuron types, including those used on modern neuromorphic hardware. Our work brings more biological realism into modern image classification models, with the hope that these models can inform how the brain performs this difficult task. It also provides new methods for training deep networks to run on neuromorphic hardware, with the aim of fast, power-efficient image classification for robotics applications. (@hunsberger2015spiking)

Tong Bu, Wei Fang, Jianhao Ding, PengLin Dai, Zhaofei Yu, and Tiejun Huang Optimal ann-snn conversion for high-accuracy and ultra-low-latency spiking neural networks In *International Conference on Learning Representations (ICLR)*, 2021. **Abstract:** Spiking Neural Networks (SNNs) have gained great attraction due to their distinctive properties of low power consumption and fast inference on neuromorphic hardware. As the most effective method to get deep SNNs, ANN-SNN conversion has achieved comparable performance as ANNs on large-scale datasets. Despite this, it requires long time-steps to match the firing rates of SNNs to the activation of ANNs. As a result, the converted SNN suffers severe performance degradation problems with short time-steps, which hamper the practical application of SNNs. In this paper, we theoretically analyze ANN-SNN conversion error and derive the estimated activation function of SNNs. Then we propose the quantization clip-floor-shift activation function to replace the ReLU activation function in source ANNs, which can better approximate the activation function of SNNs. We prove that the expected conversion error between SNNs and ANNs is zero, enabling us to achieve high-accuracy and ultra-low-latency SNNs. We evaluate our method on CIFAR-10/100 and ImageNet datasets, and show that it outperforms the state-of-the-art ANN-SNN and directly trained SNNs in both accuracy and time-steps. To the best of our knowledge, this is the first time to explore high-performance ANN-SNN conversion with ultra-low latency (4 time-steps). Code is available at https://github.com/putshua/SNN\\}\_conversion\\}\_QCFS (@bu2021optimal)

Yuhang Li, Shi-Wee Deng, Xin Dong, Ruihao Gong, and Shi Gu A free lunch from ann: Towards efficient, accurate spiking neural networks calibration , abs/2106.06984, 2021. **Abstract:** Spiking Neural Network (SNN) has been recognized as one of the next generation of neural networks. Conventionally, SNN can be converted from a pre-trained ANN by only replacing the ReLU activation to spike activation while keeping the parameters intact. Perhaps surprisingly, in this work we show that a proper way to calibrate the parameters during the conversion of ANN to SNN can bring significant improvements. We introduce SNN Calibration, a cheap but extraordinarily effective method by leveraging the knowledge within a pre-trained Artificial Neural Network (ANN). Starting by analyzing the conversion error and its propagation through layers theoretically, we propose the calibration algorithm that can correct the error layer-by-layer. The calibration only takes a handful number of training data and several minutes to finish. Moreover, our calibration algorithm can produce SNN with state-of-the-art architecture on the large-scale ImageNet dataset, including MobileNet and RegNet. Extensive experiments demonstrate the effectiveness and efficiency of our algorithm. For example, our advanced pipeline can increase up to 69% top-1 accuracy when converting MobileNet on ImageNet compared to baselines. Codes are released at https://github.com/yhhhli/SNN_Calibration. (@Li2021AFL)

Bing Han, Gopalakrishnan Srinivasan, and Kaushik Roy Rmp-snn: Residual membrane potential neuron for enabling deeper high-accuracy and low-latency spiking neural network In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 13558–13567, 2020. **Abstract:** Spiking Neural Networks (SNNs) have recently attracted significant research interest as the third generation of artificial neural networks that can enable low-power event-driven data analytics. The best performing SNNs for image recognition tasks are obtained by converting a trained Analog Neural Network (ANN), consisting of Rectified Linear Units (ReLU), to SNN composed of integrate-and-fire neurons with "proper" firing thresholds. The converted SNNs typically incur loss in accuracy compared to that provided by the original ANN and require sizable number of inference time-steps to achieve the best accuracy. We find that performance degradation in the converted SNN stems from using "hard reset" spiking neuron that is driven to fixed reset potential once its membrane potential exceeds the firing threshold, leading to information loss during SNN inference. We propose ANN-SNN conversion using "soft reset" spiking neuron model, referred to as Residual Membrane Potential (RMP) spiking neuron, which retains the "residual" membrane potential above threshold at the firing instants. We demonstrate near loss-less ANN-SNN conversion using RMP neurons for VGG-16, ResNet-20, and ResNet-34 SNNs on challenging datasets including CIFAR-10 (93.63% top-1), CIFAR-100 (70.93% top-1), and ImageNet (73.09% top-1 accuracy). Our results also show that RMP-SNN surpasses the best inference accuracy provided by the converted SNN with "hard reset" spiking neurons using 2-8 times fewer inference time-steps across network architectures and datasets. (@han2020rmp)

Tong Bu, Wei Fang, Jianhao Ding, PengLin Dai, Zhaofei Yu, and Tiejun Huang Optimal ann-snn conversion for high-accuracy and ultra-low-latency spiking neural networks , 2023. **Abstract:** Spiking Neural Networks (SNNs) have gained great attraction due to their distinctive properties of low power consumption and fast inference on neuromorphic hardware. As the most effective method to get deep SNNs, ANN-SNN conversion has achieved comparable performance as ANNs on large-scale datasets. Despite this, it requires long time-steps to match the firing rates of SNNs to the activation of ANNs. As a result, the converted SNN suffers severe performance degradation problems with short time-steps, which hamper the practical application of SNNs. In this paper, we theoretically analyze ANN-SNN conversion error and derive the estimated activation function of SNNs. Then we propose the quantization clip-floor-shift activation function to replace the ReLU activation function in source ANNs, which can better approximate the activation function of SNNs. We prove that the expected conversion error between SNNs and ANNs is zero, enabling us to achieve high-accuracy and ultra-low-latency SNNs. We evaluate our method on CIFAR-10/100 and ImageNet datasets, and show that it outperforms the state-of-the-art ANN-SNN and directly trained SNNs in both accuracy and time-steps. To the best of our knowledge, this is the first time to explore high-performance ANN-SNN conversion with ultra-low latency (4 time-steps). Code is available at https://github.com/putshua/SNN\\}\_conversion\\}\_QCFS (@bu2023optimal)

Ziqing Wang, Yuetong Fang, Jiahang Cao, Qiang Zhang, Zhongrui Wang, and Renjing Xu Masked spiking transformer In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 1761–1771, 2023. **Abstract:** The combination of Spiking Neural Networks (SNNs) and Transformers has attracted significant attention due to their potential for high energy efficiency and high-performance nature. However, existing works on this topic typically rely on direct training, which can lead to suboptimal performance. To address this issue, we propose to leverage the benefits of the ANN-to-SNN conversion method to combine SNNs and Transformers, resulting in significantly improved performance over existing state-of-the-art SNN models. Furthermore, inspired by the quantal synaptic failures observed in the nervous system, which reduce the number of spikes transmitted across synapses, we introduce a novel Masked Spiking Transformer (MST) framework. This incorporates a Random Spike Masking (RSM) method to prune redundant spikes and reduce energy consumption without sacrificing performance. Our experimental results demonstrate that the proposed MST model achieves a significant reduction of 26.8% in power consumption when the masking ratio is 75% while maintaining the same level of performance as the unmasked model. The code is available at: https://github.com/bic-L/Masked-Spiking-Transformer. (@wang2023masked)

Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu, and Luping Shi Spatio-temporal backpropagation for training high-performance spiking neural networks , 12:331, 2018. **Abstract:** Compared with artificial neural networks (ANNs), spiking neural networks (SNNs) are promising to explore the brain-like behaviors since the spikes could encode more spatio-temporal information. Although pre-training from ANN or direct training based on backpropagation (BP) makes the supervised training of SNNs possible, these methods only exploit the networks’ spatial domain information which leads to the performance bottleneck and requires many complicated training skills. Another fundamental issue is that the spike activity is naturally non-differentiable which causes great difficulties in training SNNs. To this end, we build an iterative LIF model that is more friendly for gradient descent training. By simultaneously considering the layer-by-layer spatial domain (SD) and the timing-dependent temporal domain (TD) in the training phase, as well as an approximated derivative for the spike activity, we propose a spatio-temporal backpropagation (STBP) training framework without using any complicated technology. We achieve the best performance of multi-layered perceptron (MLP) compared with existing state-of-the-art algorithms over the static MNIST and the dynamic N-MNIST dataset as well as a custom object detection dataset. This work provides a new perspective to explore the high-performance SNNs for future brain-like computing paradigm with rich spatio-temporal dynamics. (@wu2018spatio)

Emre O Neftci, Hesham Mostafa, and Friedemann Zenke Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks , 36(6):51–63, 2019. **Abstract:** Spiking neural networks (SNNs) are nature’s versatile solution to fault-tolerant, energy-efficient signal processing. To translate these benefits into hardware, a growing number of neuromorphic spiking NN processors have attempted to emulate biological NNs. These developments have created an imminent need for methods and tools that enable such systems to solve real-world signal processing problems. Like conventional NNs, SNNs can be trained on real, domain-specific data; however, their training requires the overcoming of a number of challenges linked to their binary and dynamical nature. This article elucidates step-by-step the problems typically encountered when training SNNs and guides the reader through the key concepts of synaptic plasticity and data-driven learning in the spiking setting. Accordingly, it gives an overview of existing approaches and provides an introduction to surrogate gradient (SG) methods, specifically, as a particularly flexible and efficient method to overcome the aforementioned challenges. (@neftci2019surrogate)

Mingqing Xiao, Qingyan Meng, Zongpeng Zhang, Yisen Wang, and Zhouchen Lin Training feedback spiking neural networks by implicit differentiation on the equilibrium state In *Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS)*, volume 34, pages 14516–14528, 2021. **Abstract:** Spiking neural networks (SNNs) are brain-inspired models that enable energy-efficient implementation on neuromorphic hardware. However, the supervised training of SNNs remains a hard problem due to the discontinuity of the spiking neuron model. Most existing methods imitate the backpropagation framework and feedforward architectures for artificial neural networks, and use surrogate derivatives or compute gradients with respect to the spiking time to deal with the problem. These approaches either accumulate approximation errors or only propagate information limitedly through existing spikes, and usually require information propagation along time steps with large memory costs and biological implausibility. In this work, we consider feedback spiking neural networks, which are more brain-like, and propose a novel training method that does not rely on the exact reverse of the forward computation. First, we show that the average firing rates of SNNs with feedback connections would gradually evolve to an equilibrium state along time, which follows a fixed-point equation. Then by viewing the forward computation of feedback SNNs as a black-box solver for this equation, and leveraging the implicit differentiation on the equation, we can compute the gradient for parameters without considering the exact forward procedure. In this way, the forward and backward procedures are decoupled and therefore the problem of non-differentiable spiking functions is avoided. We also briefly discuss the biological plausibility of implicit differentiation, which only requires computing another equilibrium. Extensive experiments on MNIST, Fashion-MNIST, N-MNIST, CIFAR-10, and CIFAR-100 demonstrate the superior performance of our method for feedback models with fewer neurons and parameters in a small number of time steps. Our code is avaiable at https://github.com/pkuxmq/IDE-FSNN. (@xiao2021training)

Sumit B Shrestha and Garrick Orchard Slayer: Spike layer error reassignment in time In *Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS)*, volume 31, 2018. **Abstract:** Configuring deep Spiking Neural Networks (SNNs) is an exciting research avenue for low power spike event based computation. However, the spike generation function is non-differentiable and therefore not directly compatible with the standard error backpropagation algorithm. In this paper, we introduce a new general backpropagation mechanism for learning synaptic weights and axonal delays which overcomes the problem of non-differentiability of the spike function and uses a temporal credit assignment policy for backpropagating error to preceding layers. We describe and release a GPU accelerated software implementation of our method which allows training both fully connected and convolutional neural network (CNN) architectures. Using our software, we compare our method against existing SNN based learning approaches and standard ANN to SNN conversion techniques and show that our method achieves state of the art performance for an SNN on the MNIST, NMNIST, DVS Gesture, and TIDIGITS datasets. (@shrestha2018slayer)

Wei Fang, Zhaofei Yu, Yanqi Chen, Tiejun Huang, Timothée Masquelier, and Yonghong Tian In *Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS)*, volume 34, pages 21056–21069, 2021. **Abstract:** Deep Spiking Neural Networks (SNNs) present optimization difficulties for gradient-based approaches due to discrete binary activation and complex spatial-temporal dynamics. Considering the huge success of ResNet in deep learning, it would be natural to train deep SNNs with residual learning. Previous Spiking ResNet mimics the standard residual block in ANNs and simply replaces ReLU activation layers with spiking neurons, which suffers the degradation problem and can hardly implement residual learning. In this paper, we propose the spike-element-wise (SEW) ResNet to realize residual learning in deep SNNs. We prove that the SEW ResNet can easily implement identity mapping and overcome the vanishing/exploding gradient problems of Spiking ResNet. We evaluate our SEW ResNet on ImageNet, DVS Gesture, and CIFAR10-DVS datasets, and show that SEW ResNet outperforms the state-of-the-art directly trained SNNs in both accuracy and time-steps. Moreover, SEW ResNet can achieve higher performance by simply adding more layers, providing a simple method to train deep SNNs. To our best knowledge, this is the first time that directly training deep SNNs with more than 100 layers becomes possible. Our codes are available at https://github.com/fangwei123456/Spike-Element-Wise-ResNet. (@fang2021deep)

Ole Juri Richter, QIAO Ning, Qian Liu, and Sadique Ul Ameen Sheik Event-driven spiking convolutional neural network, June 16 2022. US Patent App. 17/601,939. **Abstract:** Electromyography (EMG) pattern recognition is an important technology for prosthesis control and human-computer interaction etc. However, the practical application of EMG pattern recognition is hampered by poor accuracy and robustness due to electrode shift caused by repeated wearing of the signal acquisition device. Moreover, the user’s acceptability is low due to the heavy training burden, which is caused by the need for a large amount of training data by traditional methods. In order to explore the advantage of spiking neural network (SNN) in solving the poor robustness and heavy training burden problems in EMG pattern recognition, a spiking convolutional neural network (SCNN) composed of cyclic convolutional neural network (CNN) and fully connected modules is proposed and implemented in this study. High density surface electromyography (HD-sEMG) signals collected from 6 gestures of 10 subjects at 6 electrode positions are taken as the research object. Compared to CNN with the same structure, CNN-Long Short Term Memory (CNN-LSTM), linear kernel linear discriminant analysis classifier (LDA) and spiking multilayer perceptron (Spiking MLP), the accuracy of SCNN is 50.69%, 33.92%, 32.94% and 9.41% higher in the small sample training experiment, 6.50%, 4.23%, 28.73%, and 2.57% higher in the electrode shifts experiment respectively. In addition, the power consumption of SCNN is about 1/93 of CNN. The advantages of the proposed framework in alleviating user training burden, mitigating the adverse effect of electrode shifts and reducing power consumption make it very meaningful for promoting the development of user-friendly real-time myoelectric control system. (@richter2022event)

Jing Pei, Lei Deng, Sen Song, Mingguo Zhao, Youhui Zhang, Shuang Wu, Guanrui Wang, Zhe Zou, Zhenzhi Wu, Wei He, et al Towards artificial general intelligence with hybrid tianjic chip architecture , 572(7767):106–111, 2019. **Abstract:** Recently, artificial intelligence has made rapid progresses. However, existing AI systems still encounter difficulties even for somethings that humans can easily do. The ultimate way to solve these problems is to develop artificial general intelligence (AGI). Brain inspired computing (BIC) systems are one of the most promising technologies to integrate computer science and neuroscience to facilitate the development of AGI. In this talk, three issues will be discussed: (1) why do we need BIC system? (2) the current status and latest progress in BIC chips, software, and systems; (3) how to develop BIC systems to support AGI with limited understanding of the brain mechanisms. A hybrid and scalable exploration platform of AGI is demonstrated by an unmanned bicycle control system. The main challenges, possible solutions and strategies to develop BIC systems to stimulate AGI development will be addressed. (@pei2019towards)

Yifan Hu, Yujie Wu, Lei Deng, and Guoqi Li Advancing residual learning towards powerful deep spiking neural networks , 2021. **Abstract:** Despite the rapid progress of neuromorphic computing, inadequate capacity and insufficient representation power of spiking neural networks (SNNs) severely restrict their application scope in practice. Residual learning and shortcuts have been evidenced as an important approach for training deep neural networks, but rarely did previous work assess their applicability to the characteristics of spike-based communication and spatiotemporal dynamics. In this paper, we first identify that this negligence leads to impeded information flow and the accompanying degradation problem in previous residual SNNs. To address this issue, we propose a novel SNN-oriented residual architecture termed MS-ResNet, which establishes membrane-based shortcut pathways, and further prove that the gradient norm equality can be achieved in MS-ResNet by introducing block dynamical isometry theory, which ensures the network can be well-behaved in a depth-insensitive way. Thus we are able to significantly extend the depth of directly trained SNNs, e.g., up to 482 layers on CIFAR-10 and 104 layers on ImageNet, without observing any slight degradation problem. To validate the effectiveness of MS-ResNet, experiments on both frame-based and neuromorphic datasets are conducted. MS-ResNet104 achieves a superior result of 76.02% accuracy on ImageNet, which is the highest to our best knowledge in the domain of directly trained SNNs. Great energy efficiency is also observed, with an average of only one spike per neuron needed to classify an input sample. We believe our powerful and scalable models will provide a strong support for further exploration of SNNs. (@hu2021advancing)

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou Training data-efficient image transformers & distillation through attention In *International conference on machine learning*, pages 10347–10357. PMLR, 2021. **Abstract:** Recently, neural networks purely based on attention were shown to ad- dress image understanding tasks such as image classiﬁcation. These high- performing vision transformers are pre-trained with hundreds of millions of images using a large infrastructure, thereby limiting their adoption. In this work, we produce competitive convolution-free transformers by training on Imagenet only. We train them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop) on ImageNet with no external data. More importantly, we introduce a teacher-student strategy speciﬁc to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both Imagenet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and models. 1 Introduction Convolutional neural networks have been the main design paradigm for image understanding tasks, as initially demonstrated on image classiﬁcation tasks. One of the ingredient to their success was the availability of a large training set, namely Imagenet \[13, 42\]. Motivated by the success of attention-based mod- els in Natural Language Processing \[14, 52\], there has been increasing interest in architectures leveraging attention mechanisms within convnets \[2, 34, 61\]. More recently several researchers have proposed hybrid architecture trans- planting transformer ingredients to convnets to solve vision tasks \[6, 43\]. The vision transformer (ViT) introduced by Dosovitskiy et al. \[15\] is an ar- chitecture directly inherited from Natural Language Processing \[52\], but ap- 1arXiv:2012.12877v2 \[cs.CV\] 15 Jan 2021⚗↑⚗⚗⚗Figure 1: Throughput and accuracy on Imagenet of our methods compared to EfﬁcientNets, trained on Imagenet1k only. The throughput is measured as the number of images processed per second on a V100 GPU. DeiT-B is identical to VIT-B, but the training is more adapted to a data-starving regime. It is learned in a few days on one machine. The symbol ⚗ refers to models trained with our transformer-speciﬁc distillation. See Table 5 for details and more models. plied to image classiﬁcation with raw image patches as input. Their paper pre- sented excellent results with transformers trained with a large private labelled im (@touvron2021training)

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick Masked autoencoders are scalable vision learners In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 16000–16009, 2022. **Abstract:** This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3× or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior. (@he2022masked)

Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le Randaugment: Practical automated data augmentation with a reduced search space In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops*, pages 702–703, 2020. **Abstract:** Recent work on automated augmentation strategies has led to state-of-the-art results in image classification and object detection. An obstacle to a large-scale adoption of these methods is that they require a separate and expensive search phase. A common way to overcome the expense of the search phase was to use a smaller proxy task. However, it was not clear if the optimized hyperparameters found on the proxy task are also optimal for the actual task. In this work, we rethink the process of designing automated augmentation strategies. We find that while previous work required a search for both magnitude and probability of each operation independently, it is sufficient to only search for a single distortion magnitude that jointly controls all operations. We hence propose a simplified search space that vastly reduces the computational expense of automated augmentation, and permits the removal of a separate proxy task. Despite the simplifications, our method achieves equal or better performance over previous automated augmentation strategies on on CIFAR-10/100, SVHN, ImageNet and COCO datasets. EfficientNet-B7, we achieve 85.0% accuracy, a 1.0% increase over baseline augmentation, a 0.6% improvement over AutoAugment on the ImageNet dataset. With EfficientNet-B8, we achieve 85.4% accuracy on ImageNet, which matches a previous result that used 3.5B extra images. On object detection, the same method as classification leads to 1.0-1.3% improvement over baseline augmentation. Code will be made available online. (@cubuk2020randaugment)

Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang Random erasing data augmentation In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 13001–13008, 2020. **Abstract:** In this paper, we introduce Random Erasing, a new data augmentation method for training the convolutional neural network (CNN). In training, Random Erasing randomly selects a rectangle region in an image and erases its pixels with random values. In this process, training images with various levels of occlusion are generated, which reduces the risk of over-fitting and makes the model robust to occlusion. Random Erasing is parameter learning free, easy to implement, and can be integrated with most of the CNN-based recognition models. Albeit simple, Random Erasing is complementary to commonly used data augmentation techniques such as random cropping and flipping, and yields consistent improvement over strong baselines in image classification, object detection and person re-identification. Code is available at: https://github.com/zhunzhong07/Random-Erasing. (@zhong2020random)

Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger Deep networks with stochastic depth In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14*, pages 646–661. Springer, 2016. **Abstract:** Very deep convolutional networks with hundreds of layers have led to signi cant reductions in error on competitive benchmarks. Although the unmatched expressiveness of the many layers can be highly desirable at test time, training very deep networks comes with its own set of challenges. The gradients can vanish, the forward ow often di- minishes, and the training time can be painfully slow. To address these problems, we propose stochastic depth , a training procedure that enables the seemingly contradictory setup to train short networks and use deep networks at test time. We start with very deep networks but during train- ing, for each mini-batch, randomly drop a subset of layers and bypass them with the identity function. This simple approach complements the recent success of residual networks. It reduces training time substantially and improves the test error signi cantly on almost all data sets that we used for evaluation. With stochastic depth we can increase the depth of residual networks even beyond 1200 layers and still yield meaningful improvements in test error (4.91% on CIFAR-10). 1 Introduction Convolutional Neural Networks (CNNs) were arguably popularized within the vision community in 2009 through AlexNet \[1\] and its celebrated victory at the ImageNet competition \[2\]. Since then there has been a notable shift towards CNNs in many areas of computer vision \[3, 4, 5, 6, 7, 8\]. As this shift unfolds, a second trend emerges; deeper and deeper CNN architectures are being developed and trained. Whereas AlexNet had 5 convolutional layers \[1\], the VGG network and GoogLeNet in 2014 had 19 and 22 layers respectively \[5, 7\], and most recently the ResNet architecture featured 152 layers \[8\]. Network depth is a major determinant of model expressiveness, both in the- ory \[9, 10\] and in practice \[5, 7, 8\]. However, very deep models also introduce new challenges: vanishing gradients in backward propagation, diminishing fea- ture reuse in forward propagation, and long training time. Vanishing Gradients is a well known nuisance in neural networks with many layers \[11\]. As the gradient information is back-propagated, repeated multipli- cation or convolution with small weights renders the gradient information inef- fectively small in earlier layers. Several approaches exist to reduce this e ect in practice, for example through careful initialization \[12\], hidden layer supervision \[13\], or, recently, Batch Normalization \[14\]. arXiv:1603.09382v3 \[cs.LG\] 28 Jul 20 (@huang2016deep)

Chenlin Zhou, Han Zhang, Liutao Yu, Yumin Ye, Zhaokun Zhou, Liwei Huang, Zhengyu Ma, Xiaopeng Fan, Huihui Zhou, and Yonghong Tian Direct training high-performance deep spiking neural networks: A review of theories and methods , 2024. **Abstract:** Spiking neural networks (SNNs) offer a promising energy-efficient alternative to artificial neural networks (ANNs), in virtue of their high biological plausibility, rich spatial-temporal dynamics, and event-driven computation. The direct training algorithms based on the surrogate gradient method provide sufficient flexibility to design novel SNN architectures and explore the spatial-temporal dynamics of SNNs. According to previous studies, the performance of models is highly dependent on their sizes. Recently, direct training deep SNNs have achieved great progress on both neuromorphic datasets and large-scale static datasets. Notably, transformer-based SNNs show comparable performance with their ANN counterparts. In this paper, we provide a new perspective to summarize the theories and methods for training deep SNNs with high performance in a systematic and comprehensive way, including theory fundamentals, spiking neuron models, advanced SNN models and residual architectures, software frameworks and neuromorphic hardware, applications, and future trends. The reviewed papers are collected at https://github.com/zhouchenlin2096/Awesome-Spiking-Neural-Networks (@zhou2024direct)

Wei Fang, Zhaofei Yu, Yanqi Chen, Timothée Masquelier, Tiejun Huang, and Yonghong Tian Incorporating learnable membrane time constant to enhance learning of spiking neural networks In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 2661–2671, 2021. **Abstract:** Spiking Neural Networks (SNNs) have attracted enormous research interest due to temporal information processing capability, low power consumption, and high biological plausibility. However, the formulation of efficient and high-performance learning algorithms for SNNs is still challenging. Most existing learning methods learn weights only, and require manual tuning of the membrane-related parameters that determine the dynamics of a single spiking neuron. These parameters are typically chosen to be the same for all neurons, which limits the diversity of neurons and thus the expressiveness of the resulting SNNs. In this paper, we take inspiration from the observation that membrane-related parameters are different across brain regions, and propose a training algorithm that is capable of learning not only the synaptic weights but also the membrane time constants of SNNs. We show that incorporating learnable membrane time constants can make the network less sensitive to initial values and can speed up learning. In addition, we reevaluate the pooling methods in SNNs and find that max-pooling will not lead to significant information loss and have the advantage of low computation cost and binary compatibility. We evaluate the proposed method for image classification tasks on both traditional static MNIST, Fashion-MNIST, CIFAR-10 datasets, and neuromorphic N-MNIST, CIFAR10-DVS, DVS128 Gesture datasets. The experiment results show that the proposed method outperforms the state-of-the-art accuracy on nearly all datasets, using fewer time-steps. Our codes are available at https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron. (@fang2021incorporating)

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei Imagenet: A large-scale hierarchical image database In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 248–255, 2009. **Abstract:** The explosion of image data on the Internet has the potential to foster more sophisticated and robust models and algorithms to index, retrieve, organize and interact with images and multimedia data. But exactly how such data can be harnessed and organized remains a critical problem. We introduce here a new database called "ImageNet", a large-scale ontology of images built upon the backbone of the WordNet structure. ImageNet aims to populate the majority of the 80,000 synsets of WordNet with an average of 500–1000 clean and full resolution images. This will result in tens of millions of annotated images organized by the semantic hierarchy of WordNet. This paper offers a detailed analysis of ImageNet in its current state: 12 subtrees with 5247 synsets and 3.2 million images in total. We show that ImageNet is much larger in scale and diversity and much more accurate than the current image datasets. Constructing such a large-scale database is a challenging task. We describe the data collection scheme with Amazon Mechanical Turk. Lastly, we illustrate the usefulness of ImageNet through three simple applications in object recognition, image classification and automatic object clustering. We hope that the scale, accuracy, diversity and hierarchical structure of ImageNet can offer unparalleled opportunities to researchers in the computer vision community and beyond. (@deng2009imagenet)

Alex Krizhevsky Learning multiple layers of features from tiny images . **Abstract:** In this work we describe how to train a multi-layer generative model of natural images. We use a dataset of millions of tiny colour images, described in the next section. This has been attempted by several groups but without success. The models on which we focus are RBMs (Restricted Boltzmann Machines) and DBNs (Deep Belief Networks). These models learn interesting-looking filters, which we show are more useful to a classifier than the raw pixels. We train the classifier on a labeled subset that we have collected and call the CIFAR-10 dataset. (@krizhevsky2009learning)

Hongmin Li, Hanchao Liu, Xiangyang Ji, Guoqi Li, and Luping Shi Cifar10-dvs: an event-stream dataset for object classification , 11:309, 2017. **Abstract:** Neuromorphic vision research requires high-quality and appropriately challenging event-stream datasets to support continuous improvement of algorithms and methods. However, creating event-stream datasets is a time-consuming task, which needs to be recorded using the neuromorphic cameras. Currently, there are limited event-stream datasets available. In this work, by utilizing the popular computer vision dataset CIFAR-10, we converted 10,000 frame-based images into 10,000 event streams using a dynamic vision sensor (DVS), providing an event-stream dataset of intermediate difficulty in 10 different classes, named as "CIFAR10-DVS." The conversion of event-stream dataset was implemented by a repeated closed-loop smooth (RCLS) movement of frame-based images. Unlike the conversion of frame-based images by moving the camera, the image movement is more realistic in respect of its practical applications. The repeated closed-loop image movement generates rich local intensity changes in continuous time which are quantized by each pixel of the DVS camera to generate events. Furthermore, a performance benchmark in event-driven object classification is provided based on state-of-the-art classification algorithms. This work provides a large event-stream dataset and an initial benchmark for comparison, which may boost algorithm developments in even-driven pattern recognition and object classification. (@li2017cifar10)

Arnon Amir, Brian Taba, David Berg, Timothy Melano, Jeffrey McKinstry, Carmelo Di Nolfo, Tapan Nayak, Alexander Andreopoulos, Guillaume Garreau, Marcela Mendoza, Jeff Kusnitz, Michael Debole, Steve Esser, Tobi Delbruck, Myron Flickner, and Dharmendra Modha A low power, fully event-based gesture recognition system In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 7243–7252, 2017. **Abstract:** We present the first gesture recognition system implemented end-to-end on event-based hardware, using a TrueNorth neurosynaptic processor to recognize hand gestures in real-time at low power from events streamed live by a Dynamic Vision Sensor (DVS). The biologically inspired DVS transmits data only when a pixel detects a change, unlike traditional frame-based cameras which sample every pixel at a fixed frame rate. This sparse, asynchronous data representation lets event-based cameras operate at much lower power than frame-based cameras. However, much of the energy efficiency is lost if, as in previous work, the event stream is interpreted by conventional synchronous processors. Here, for the first time, we process a live DVS event stream using TrueNorth, a natively event-based processor with 1 million spiking neurons. Configured here as a convolutional neural network (CNN), the TrueNorth chip identifies the onset of a gesture with a latency of 105 ms while consuming less than 200 mW. The CNN achieves 96.5% out-of-sample accuracy on a newly collected DVS dataset (DvsGesture) comprising 11 hand gesture categories from 29 subjects under 3 illumination conditions. (@amir2017dvsg)

Hanle Zheng, Yujie Wu, Lei Deng, Yifan Hu, and Guoqi Li In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, pages 11062–11070, 2021. **Abstract:** Spiking neural networks (SNNs) are promising in a bio-plausible coding for spatio-temporal information and event-driven signal processing, which is very suited for energy-efficient implementation in neuromorphic hardware. However, the unique working mode of SNNs makes them more difficult to train than traditional networks. Currently, there are two main routes to explore the training of deep SNNs with high performance. The first is to convert a pre-trained ANN model to its SNN version, which usually requires a long coding window for convergence and cannot exploit the spatio-temporal features during training for solving temporal tasks. The other is to directly train SNNs in the spatio-temporal domain. But due to the binary spike activity of the firing function and the problem of gradient vanishing or explosion, current methods are restricted to shallow architectures and thereby difficult in harnessing large-scale datasets (e.g. ImageNet). To this end, we propose a threshold-dependent batch normalization (tdBN) method based on the emerging spatio-temporal backpropagation, termed “STBP-tdBN”, enabling direct training of a very deep SNN and the efficient implementation of its inference on neuromorphic hardware. With the proposed method and elaborated shortcut connection, we significantly extend directly-trained SNNs from a shallow structure ( (@zheng2021going)

Shikuang Deng, Yuhang Li, Shanghang Zhang, and Shi Gu In *International Conference on Learning Representations (ICLR)*, 2021. **Abstract:** Recently, brain-inspired spiking neuron networks (SNNs) have attracted widespread research interest because of their event-driven and energy-efficient characteristics. Still, it is difficult to efficiently train deep SNNs due to the non-differentiability of its activation function, which disables the typically used gradient descent approaches for traditional artificial neural networks (ANNs). Although the adoption of surrogate gradient (SG) formally allows for the back-propagation of losses, the discrete spiking mechanism actually differentiates the loss landscape of SNNs from that of ANNs, failing the surrogate gradient methods to achieve comparable accuracy as for ANNs. In this paper, we first analyze why the current direct training approach with surrogate gradient results in SNNs with poor generalizability. Then we introduce the temporal efficient training (TET) approach to compensate for the loss of momentum in the gradient descent with SG so that the training process can converge into flatter minima with better generalizability. Meanwhile, we demonstrate that TET improves the temporal scalability of SNN and induces a temporal inheritable training for acceleration. Our method consistently outperforms the SOTA on all reported mainstream datasets, including CIFAR-10/100 and ImageNet. Remarkably on DVS-CIFAR10, we obtained 83$\\}%$ top-1 accuracy, over 10$\\}%$ improvement compared to existing state of the art. Codes are available at \\}url{https://github.com/Gus-Lab/temporal_efficient_training}. (@deng2021temporal)

Xiaohan Ding, Yuchen Guo, Guiguang Ding, and Jungong Han Acnet: Strengthening the kernel skeletons for powerful cnn via asymmetric convolution blocks In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 1911–1920, 2019. **Abstract:** As designing appropriate Convolutional Neural Network (CNN) architecture in the context of a given application usually involves heavy human works or numerous GPU hours, the research community is soliciting the architecture-neutral CNN structures, which can be easily plugged into multiple mature architectures to improve the performance on our real-world applications. We propose Asymmetric Convolution Block (ACB), an architecture-neutral structure as a CNN building block, which uses 1D asymmetric convolutions to strengthen the square convolution kernels. For an off-the-shelf architecture, we replace the standard square-kernel convolutional layers with ACBs to construct an Asymmetric Convolutional Network (ACNet), which can be trained to reach a higher level of accuracy. After training, we equivalently convert the ACNet into the same original architecture, thus requiring no extra computations anymore. We have observed that ACNet can improve the performance of various models on CIFAR and ImageNet by a clear margin. Through further experiments, we attribute the effectiveness of ACB to its capability of enhancing the model’s robustness to rotational distortions and strengthening the central skeleton parts of square convolution kernels. (@ding2019acnet)

Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun Repvgg: Making vgg-style convnets great again In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 13733–13742, 2021. **Abstract:** We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3 × 3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet. The code and trained models are available at https://github.com/megvii-model/RepVGG. (@ding2021repvgg)

Yangfan Hu, Huajin Tang, and Gang Pan Spiking deep residual networks , 2021. **Abstract:** Spiking neural networks (SNNs) have received significant attention for their biological plausibility. SNNs theoretically have at least the same computational power as traditional artificial neural networks (ANNs). They possess the potential of achieving energy-efficient machine intelligence while keeping comparable performance to ANNs. However, it is still a big challenge to train a very deep SNN. In this brief, we propose an efficient approach to build deep SNNs. Residual network (ResNet) is considered a state-of-the-art and fundamental model among convolutional neural networks (CNNs). We employ the idea of converting a trained ResNet to a network of spiking neurons named spiking ResNet (S-ResNet). We propose a residual conversion model that appropriately scales continuous-valued activations in ANNs to match the firing rates in SNNs and a compensation mechanism to reduce the error caused by discretization. Experimental results demonstrate that our proposed method achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet 2012 with low latency. This work is the first time to build an asynchronous SNN deeper than 100 layers, with comparable performance to its original ANN. (@hu2021spiking)

Guangyao Chen, Peixi Peng, Guoqi Li, and Yonghong Tian Training full spike neural networks via auxiliary accumulation pathway , 2023. **Abstract:** Due to the binary spike signals making converting the traditional high-power multiply-accumulation (MAC) into a low-power accumulation (AC) available, the brain-inspired Spiking Neural Networks (SNNs) are gaining more and more attention. However, the binary spike propagation of the Full-Spike Neural Networks (FSNN) with limited time steps is prone to significant information loss. To improve performance, several state-of-the-art SNN models trained from scratch inevitably bring many non-spike operations. The non-spike operations cause additional computational consumption and may not be deployed on some neuromorphic hardware where only spike operation is allowed. To train a large-scale FSNN with high performance, this paper proposes a novel Dual-Stream Training (DST) method which adds a detachable Auxiliary Accumulation Pathway (AAP) to the full spiking residual networks. The accumulation in AAP could compensate for the information loss during the forward and backward of full spike propagation, and facilitate the training of the FSNN. In the test phase, the AAP could be removed and only the FSNN remained. This not only keeps the lower energy consumption but also makes our model easy to deploy. Moreover, for some cases where the non-spike operations are available, the APP could also be retained in test inference and improve feature discrimination by introducing a little non-spike consumption. Extensive experiments on ImageNet, DVS Gesture, and CIFAR10-DVS datasets demonstrate the effectiveness of DST. (@chen2023training)

Souvik Kundu, Massoud Pedram, and Peter A Beerel Hire-snn: Harnessing the inherent robustness of energy-efficient deep spiking neural networks by training with crafted input noise In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 5209–5218, 2021. **Abstract:** Low-latency deep spiking neural networks (SNNs) have become a promising alternative to conventional artificial neural networks (ANNs) because of their potential for increased energy efficiency on event-driven neuromorphic hardware. Neural networks, including SNNs, however, are subject to various adversarial attacks and must be trained to remain resilient against such attacks for many applications. Nevertheless, due to prohibitively high training costs associated with SNNs, an analysis and optimization of deep SNNs under various adversarial attacks have been largely overlooked. In this paper, we first present a detailed analysis of the inherent robustness of low-latency SNNs against popular gradient-based attacks, namely fast gradient sign method (FGSM) and projected gradient descent (PGD). Motivated by this analysis, to harness the model’s robustness against these attacks we present an SNN training algorithm that uses crafted input noise and incurs no additional training time. To evaluate the merits of our algorithm, we conducted extensive experiments with variants of VGG and ResNet on both CIFAR-10 and CIFAR-100 dataset. Compared to standard trained direct-input SNNs, our trained models yield improved classification accuracy of up to 13.7% and 10.1% on FGSM and PGD attack generated images, respectively, with negligible loss in clean image accuracy. Our models also outperform inherently-robust SNNs trained on rate-coded inputs with improved or similar classification performance on attack-generated images while having up to 25× and ∼4.6× lower latency and computation energy, respectively. For reproducibility, we have open-sourced the code at github.com/ksouvik52/hiresnn2021. (@kundu2021hire)

Mark Horowitz computing’s energy problem (and what we can do about it) In *2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers (ISSCC)*, pages 10–14. IEEE, 2014. (@horowitz20141)

Priyadarshini Panda, Sai Aparna Aketi, and Kaushik Roy Toward scalable, efficient, and accurate deep spiking neural networks with backward residual connections, stochastic softmax, and hybridization , 14:653, 2020. **Abstract:** Spiking Neural Networks (SNNs) may offer an energy-efficient alternative for implementing deep learning applications. In recent years, there have been several proposals focused on supervised (conversion, spike-based gradient descent) and unsupervised (spike timing dependent plasticity) training methods to improve the accuracy of SNNs on large-scale tasks. However, each of these methods suffer from scalability, latency, and accuracy limitations. In this paper, we propose novel algorithmic techniques of modifying the SNN configuration with backward residual connections, stochastic softmax, and hybrid artificial-and-spiking neuronal activations to improve the learning ability of the training methodologies to yield competitive accuracy, while, yielding large efficiency gains over their artificial counterparts. Note, artificial counterparts refer to conventional deep learning/artificial neural networks. Our techniques apply to VGG/Residual architectures, and are compatible with all forms of training methodologies. Our analysis reveals that the proposed solutions yield near state-of-the-art accuracy with significant energy-efficiency and reduced parameter overhead translating to hardware improvements on complex visual recognition tasks, such as, CIFAR10, Imagenet datatsets. (@panda2020toward)

Man Yao, Guangshe Zhao, Hengyu Zhang, Yifan Hu, Lei Deng, Yonghong Tian, Bo Xu, and Guoqi Li Attention spiking neural networks , 2023. **Abstract:** Brain-inspired spiking neural networks (SNNs) are becoming a promising energy-efficient alternative to traditional artificial neural networks (ANNs). However, the performance gap between SNNs and ANNs has been a significant hindrance to deploying SNNs ubiquitously. To leverage the full potential of SNNs, in this paper we study the attention mechanisms, which can help human focus on important information. We present our idea of attention in SNNs with a multi-dimensional attention module, which infers attention weights along the temporal, channel, as well as spatial dimension separately or simultaneously. Based on the existing neuroscience theories, we exploit the attention weights to optimize membrane potentials, which in turn regulate the spiking response. Extensive experimental results on event-based action recognition and image classification datasets demonstrate that attention facilitates vanilla SNNs to achieve sparser spiking firing, better performance, and energy efficiency concurrently. In particular, we achieve top-1 accuracy of 75.92% and 77.08% on ImageNet-1 K with single/4-step Res-SNN-104, which are state-of-the-art results in SNNs. Compared with counterpart Res-ANN-104, the performance gap becomes -0.95/+0.21 percent and the energy efficiency is 31.8×/7.4×. To analyze the effectiveness of attention SNNs, we theoretically prove that the spiking degradation or the gradient vanishing, which usually holds in general SNNs, can be resolved by introducing the block dynamical isometry theory. We also analyze the efficiency of attention SNNs based on our proposed spiking response visualization method. Our work lights up SNN’s potential as a general backbone to support various applications in the field of SNN research, with a great balance between effectiveness and energy efficiency. (@yao2023attention)

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al Pytorch: An imperative style, high-performance deep learning library In *Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS)*, volume 32, 2019. **Abstract:** Deep learning frameworks have often focused on either usability or speed, but not both. PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs. In this paper, we detail the principles that drove the implementation of PyTorch and how they are reflected in its architecture. We emphasize that every aspect of PyTorch is a regular Python program under the full control of its user. We also explain how the careful and pragmatic implementation of the key components of its runtime enables them to work together to achieve compelling performance. We demonstrate the efficiency of individual subsystems, as well as the overall speed of PyTorch on several common benchmarks. (@paszke2019pytorch)

Wei Fang, Yanqi Chen, Jianhao Ding, Zhaofei Yu, Timothée Masquelier, Ding Chen, Liwei Huang, Huihui Zhou, Guoqi Li, and Yonghong Tian Spikingjelly: An open-source machine learning infrastructure platform for spike-based intelligence , 9(40):eadi1480, 2023. **Abstract:** Spiking neural networks (SNNs) aim to realize brain-inspired intelligence on neuromorphic chips with high energy efficiency by introducing neural dynamics and spike properties. As the emerging spiking deep learning paradigm attracts increasing interest, traditional programming frameworks cannot meet the demands of the automatic differentiation, parallel computation acceleration, and high integration of processing neuromorphic datasets and deployment. In this work, we present the SpikingJelly framework to address the aforementioned dilemma. We contribute a full-stack toolkit for preprocessing neuromorphic datasets, building deep SNNs, optimizing their parameters, and deploying SNNs on neuromorphic chips. Compared to existing methods, the training of deep SNNs can be accelerated 11×, and the superior extensibility and flexibility of SpikingJelly enable users to accelerate custom models at low costs through multilevel inheritance and semiautomatic code generation. SpikingJelly paves the way for synthesizing truly energy-efficient SNN-based machine intelligence systems, which will enrich the ecology of neuromorphic computing. (@doi:10.1126/sciadv.adi1480)

Ross Wightman Pytorch image models <https://github.com/rwightman/pytorch-image-models>, 2019. **Abstract:** Modern remote sensing technology has developed rapidly in recent years. The high-resolution remote sensing images brought by new technologies have good application prospects in military and civilian fields, but the information contained in them is also richer, which increases the complexity of remote sensing image analysis and understanding. At present, artificial intelligence technology represented by deep learning has been widely used in the field of image processing. This paper adopts the U-net network model and uses the transfer learning method to train on the remote sensing image dataset published by the French National Institute of Information and Automation (Inria) to verify the effectiveness of the deep learning semantic segmentation method on high-resolution remote sensing images. and stability. Experiments show that the model has an accuracy of 86.86% in extracting buildings from images, a recall rate of 82.54%, and an average intersection ratio of 84.53%, which is effective in semantic segmentation of high-resolution remote sensing images. (@rw2019timm)

</div>

# Appendix

## Spiking Neuron Model [LIF]

Spiking neuron is the fundamental unit of SNNs, we choose the Leaky Integrate-and-Fire (LIF) model as the spiking neuron in our work. The dynamics of a LIF neuron can be formulated as follows: \\[\begin{gathered}
H[t]=V[t-1]+\frac{1}{\tau}\left(X[t]-\left(V[t-1]-V_{\text {reset }}\right)\right),\\
S[t]=\Theta\left(H[t]-V_{t h}\right) , \\
V[t]=H[t](1-S[t])+V_{\text {reset }} S[t],
\end{gathered}\\] where \\(\tau\\) is the membrane time constant, and \\(X[t]\\) is the input current at time step \\(t\\). When the membrane potential \\(H[t]\\) exceeds the firing threshold \\(V_{th}\\), the spiking neuron will trigger a spike \\(S[t]\\). \\(\Theta(v)\\) is the Heaviside step function, which equals to 1 when \\(v\geq 0\\) and 0 otherwise. \\(V[t]\\) represents the membrane potential after the triggered event, which equals to \\(H[t]\\) if no spike is generated and otherwise equals to the reset potential \\(V_{reset}\\).

## Q-K Attention Vs. SSA in Scaling Factors [Scaling]

**Mathematical Characteristics of Q-K Attention.** All the elements in Q-K attention are spike-form, thus we assume that each \\(Q_{i, j}[t]\\) are independent random variables and subject to Bernoulli distribution \\(B(f_Q)\\). \\(f_Q\\) is the average firing rate of \\(\mathbf{Q}\\). The expectation and variance of Q-K attention can be formulated as: \\[\operatorname{E}\left(\operatorname{Q K T A}\right)=\operatorname{E}\left(\sum_{i=0}^d Q_{i, j}[t]\right)=\sum_{i=0}^d \operatorname{E}\left(Q_{i, j}[t]\right)=d f_Q ,
\label{QKTA_E}\\] \\[\operatorname{Var}\left(\operatorname{Q K T A}\right)=\operatorname{Var}\left(\sum_{i=0}^d Q_{i, j}[t]\right)=\sum_{i=0}^d \operatorname{Var}\left(Q_{i, j}[t]\right)=d f_Q\left(1-f_Q\right) ,
\label{QKTA_V}\\] \\[\operatorname{E}\left(\operatorname{Q K C A}\right)=\operatorname{E}\left(\sum_{j=0}^N Q_{i, j}[t]\right)=\sum_{j=0}^N \operatorname{E}\left(Q_{i, j}[t]\right)=N f_Q ,\\] \\[\operatorname{Var}\left(\operatorname{Q K C A}\right)=\operatorname{Var}\left(\sum_{j=0}^N Q_{i, j}[t]\right)=\sum_{j=0}^N \operatorname{Var}\left(Q_{i, j}[t]\right)=N f_Q\left(1-f_Q\right) ,\\] where \\(d = D / h\\) is the feature dimension of a head in the multi-head Q-K attention and \\(N\\) is the token number.

**Mathematical Characteristics of SSA.** Similar to the above analysis process, assume that all elements \\(Q_{i, j}[t], K_{j, i}[t], V_{i, j}[t]\\) in SSA are independent random variables and subject to Bernoulli distribution \\(B(f_Q), B(f_K), B(f_V)\\), respectively. \\(f_Q,f_K\\) and \\(f_V\\) are the average firing rate of \\(\mathbf{Q}, \mathbf{K}\\) and \\(\mathbf{V}\\), respectively. We can calculate the expectation and variance of SSA as follows. \\[\operatorname{E}(\operatorname{S S A})=\operatorname{E}\left(\sum_{i=1}^N \sum_{j=1}^d Q_{i, j}[t] K_{j, i}[t] V_{i, j}[t]\right)=\sum_{i=1}^N \sum_{j=1}^d \operatorname{E}\left(Q_{i, j}[t] K_{j, i}[t] V_{i, j}[t]\right)=N d f_Q f_K f_V ,
\label{SSA_E}\\] \\[\begin{gathered}
\operatorname{Var}(\operatorname{S S A})=\operatorname{Var}\left(\sum_{i=1}^N \sum_{j=1}^d Q_{i, j}[t] K_{j, i}[t] V_{i, j}[t]\right)=\sum_{i=1}^N \sum_{j=1}^d \operatorname{Var}\left(Q_{i, j}[t] K_{j, i}[t] V_{i, j}[t]\right) \\
=N d\left(f_Q f_K f_V\left(1-f_Q\right)\left(1-f_K\right)\left(1-f_V\right)\right. \\
+f_Q f_K f_V^2\left(1-f_Q\right)\left(1-f_K\right)+f_Q f_K^2 f_V\left(1-f_Q\right)\left(1-f_V\right)+f_Q^2 f_K f_V\left(1-f_K\right)\left(1-f_V\right) \\
\left.+f_Q f_K^2 f_V^2\left(1-f_Q\right)+f_Q^2 f_K f_V^2\left(1-f_K\right)+f_Q^2 f_K^2 f_V\left(1-f_V\right)\right),
\label{SSA_V}
\end{gathered}\\] Figure.<a href="#fig: Variance and Expectation" data-reference-type="ref" data-reference="fig: Variance and Expectation">4</a> visualize the variance and expectation of QKTA (Formula.<a href="#QKTA_E" data-reference-type="ref" data-reference="QKTA_E">[QKTA_E]</a> and <a href="#QKTA_V" data-reference-type="ref" data-reference="QKTA_V">[QKTA_V]</a>) and SSA (Formula.<a href="#SSA_E" data-reference-type="ref" data-reference="SSA_E">[SSA_E]</a> and <a href="#SSA_V" data-reference-type="ref" data-reference="SSA_V">[SSA_V]</a>). \\(N\\) is set as 196 and \\(d\\) is set as 64, respectively. We can find SSA has a much larger variance and expectation than QKTA on the whole. For example, the maximum theoretical variance of QKTA is 16, but the maximum theoretical variance of SSA is over 3000. This is the main reason that Q-K attention can discard scaling operations thus reducing power consumption, but SSA can not.

## Futher Discussion on Q-K Attention

**The Complexity of Attention Mechanisms.** The computational complexity of SSA: \\(Q, K \in [0, 1]^{N \times D}\\). The attention map (\\(Q \times K^{\mathrm{T}} \in Z^{N \times N}\\)) is obtained by matrix multiplication of matrix \\([0, 1]^{N \times D}\\) and matrix \\([0, 1]^{D \times N}\\), which thus need \\(O(N^2 D)\\) computation. The computational complexity of SDSA: \\(Q, K \in [0, 1]^{N \times D}\\). The attention map (\\(Q \otimes K \in [0, 1]^{N \times D}\\) ) is obtained by the Hadamard product of matrix \\([0, 1]^{N \times D}\\) and matrix \\([0, 1]^{N \times D}\\), which thus need \\(O(ND)\\) computation. The computational complexity of Q-K Attention: Our attention vector (\\({A}_t \in [0, 1]^{N \times 1}\\)) is computed by \\({A}_t = SN(\sum_{i=0}^D {Q}_{i, j})\\), which depends on the row or column accumulation of the \\(Q\\) matrix (\\(Q \in [0, 1]^{N \times D}\\)), thus only needs \\(O(N)\\) or \\(O(D)\\) computation.

**PLIF for Scaling.** Q-K attention can discard scaling operations to reduce power consumption On these datasets used in this article’s experiments because the variance of Q-K attention is much smaller than SSA (e.g. the max theoretical variance of Q-K token attention is only about 1 / 200 of SSA). For generality, we can also replace the LIF after attention calculation with PLIF (LIF with trainable parameters) allowing for adaptively controlling the fire rate of that spiking neuron, which can be seen as a learnable scaling. It can be expressed as \\(\mathbf{A}_t=\mathrm{PLIF}(\sum_{i=0}^D \mathbf{Q}_{i, j})\\). The results show that this modification brings a 0.2% performance improvement on CIFAR 100 (Acc = 81.17%, the firing rate of \\(\mathbf{A}_t\\) is 0.2952 in stage1 and 0.4008 in stage 2), while increasing the training time to 1.3 times.

## Supplementary for Method 3.5 [Supplementary for SPEDS]

<figure id="fig: speds">
<img src="./figures/SPEDS.png"" />
<figcaption>(a) Spiking Patch Splitting (SPS) module in Spikformer. (b) Spiking Patch Embedding with Deformed Shortcut (SPEDS) module in QKFormer.</figcaption>
</figure>

## Experimental Details [Experimental Details]

**Datasets.** We evaluate QKFormer on static image classification and neuromorphic classification. The former includes ImageNet-1K `\cite{deng2009imagenet}`{=latex}, CIFAR10/100 `\cite{krizhevsky2009learning}`{=latex}. The latter contains CIFAR10-DVS `\cite{li2017cifar10}`{=latex} and DVS128 Gesture `\cite{amir2017dvsg}`{=latex}.

ImageNet-1K is the most typical static image dataset for classification. It contains \\(1.28\\) million images for training and 50k images for validation, with a total of 1,000 categories. CIFAR10/CIFAR100 provides 50, 000 train and 10, 000 test images with 32 × 32 resolution. The difference is that CIFAR10 contains 10 categories for classification. While CIFAR100 contains 100 categories, owning better distinguishing ability for the classification algorithm.

CIFAR10-DVS is an event-based neuromorphic dataset converted from the static image dataset by capturing shifting image samples through the Dynamic Vision Sensor (DVS) camera, which provides 9,000 training samples and 1,000 test samples. DVS128 Gesture is an event-based gesture recognition dataset that contains 1342 samples of 11 hand gesture categories from 29 individuals under 3 illumination conditions, each gesture has an average duration of 6 seconds.

**Training Details.** In our experiments, we use 8 NVIDIA Tesla V100 SXM2 32GB GPUs when training models on ImageNet, while 1 GPU is used to train other datasets (CIFAR10, CIFAR100, DVS128 Gesture, CIFAR10-DVS). In direct training SNN models with surrogate function, \\[\sigma(x)=\frac{1}{1+\exp (-\alpha x)},\\] we select the Sigmoid function \\(\sigma(x)\\) as the surrogate function with \\(\alpha=4\\) during the backpropagation of direct training in all experiments.

**Experimental Details on CIFAR and Neuromorphic Classification.** We evaluate our QKFormer on small-scale datasets, including CIFAR10, CIFAR100 `\cite{krizhevsky2009learning}`{=latex} and temporal neuromorphic datasets (CIFAR10-DVS and DVS128 Gesture `\cite{amir2017dvsg}`{=latex}). The detailed results on the four small-scale datasets are presented in Table <a href="#tab:small_dataset_appendix" data-reference-type="ref" data-reference="tab:small_dataset_appendix">8</a>.

<div id="tab:small_dataset_appendix" markdown="1">

<table>
<caption>Comparision on CIFAR10, CIFAR100, DVS128, CIFAR10-DVS.</caption>
<thead>
<tr>
<th style="text-align: left;">Dataset</th>
<th style="text-align: left;"><span>Methods</span></th>
<th style="text-align: left;">Architecture</th>
<th style="text-align: left;">Param (M)</th>
<th style="text-align: left;"><span>Time Step</span></th>
<th style="text-align: left;"><span>Top-1 Acc (<span class="math inline">%</span>)</span></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="10" style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">STBP-tdBN<span class="citation" data-cites="zheng2021going"></span></td>
<td style="text-align: left;">ResNet-19</td>
<td style="text-align: left;">12.63</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">92.92</td>
</tr>
<tr>
<td style="text-align: left;">TET<span class="citation" data-cites="deng2021temporal"></span></td>
<td style="text-align: left;">ResNet-19</td>
<td style="text-align: left;">12.63</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">94.44</td>
</tr>
<tr>
<td style="text-align: left;">Spikformer<span class="citation" data-cites="zhou2023spikformer"></span></td>
<td style="text-align: left;">Spikformer-4-384</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">95.51</td>
</tr>
<tr>
<td style="text-align: left;">Spikingformer<span class="citation" data-cites="zhou2023spikingformer"></span></td>
<td style="text-align: left;">Spikingformer-4-384</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">95.81</td>
</tr>
<tr>
<td style="text-align: left;">CML<span class="citation" data-cites="zhou2023enhancing"></span></td>
<td style="text-align: left;">Spikformer-4-384</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">96.04</td>
</tr>
<tr>
<td style="text-align: left;">S-Transformer<span class="citation" data-cites="yao2023spikedriven"></span></td>
<td style="text-align: left;"><span>S-Transformer-2-512</span></td>
<td style="text-align: left;">10.28</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;"><span>95.60</span></td>
</tr>
<tr>
<td style="text-align: left;"><strong>QKFormer</strong></td>
<td style="text-align: left;"><span>HST-4-384</span></td>
<td style="text-align: left;">6.74</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;"><strong>96.18</strong></td>
</tr>
<tr>
<td style="text-align: left;">STBP-tdBN<span class="citation" data-cites="zheng2021going"></span></td>
<td style="text-align: left;">ResNet-19</td>
<td style="text-align: left;">12.63</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">70.86</td>
</tr>
<tr>
<td style="text-align: left;">TET<span class="citation" data-cites="deng2021temporal"></span></td>
<td style="text-align: left;">ResNet-19</td>
<td style="text-align: left;">12.63</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">74.47</td>
</tr>
<tr>
<td style="text-align: left;">Spikformer<span class="citation" data-cites="zhou2023spikformer"></span></td>
<td style="text-align: left;">Spikformer-4-384</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">78.21</td>
</tr>
<tr>
<td style="text-align: left;"><span>2-6</span></td>
<td style="text-align: left;">Spikingformer<span class="citation" data-cites="zhou2023spikingformer"></span></td>
<td style="text-align: left;">Spikingformer-4-384</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">78.21</td>
</tr>
<tr>
<td style="text-align: left;"><span>2-6</span></td>
<td style="text-align: left;">CML<span class="citation" data-cites="zhou2023enhancing"></span></td>
<td style="text-align: left;">Spikformer-4-384</td>
<td style="text-align: left;">9.32</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;">80.02</td>
</tr>
<tr>
<td style="text-align: left;"><span>2-6</span></td>
<td style="text-align: left;">S-Transformer<span class="citation" data-cites="yao2023spikedriven"></span></td>
<td style="text-align: left;"><span>S-Transformer-2-512</span></td>
<td style="text-align: left;">10.28</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;"><span>78.4</span></td>
</tr>
<tr>
<td style="text-align: left;"><span>2-6</span></td>
<td style="text-align: left;"><strong>QKFormer</strong></td>
<td style="text-align: left;"><span>HST-4-384</span></td>
<td style="text-align: left;">6.74</td>
<td style="text-align: left;">4</td>
<td style="text-align: left;"><span><strong>81.15</strong></span></td>
</tr>
<tr>
<td rowspan="8" style="text-align: left;">DVS128</td>
<td style="text-align: left;">Spikformer<span class="citation" data-cites="zhou2023spikformer"></span></td>
<td style="text-align: left;">Spikformer-2-256</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">96.9 , 98.3</td>
</tr>
<tr>
<td style="text-align: left;">Spikingformer<span class="citation" data-cites="zhou2023spikingformer"></span></td>
<td style="text-align: left;">Spikingformer-2-256</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">96.2 , 98.3</td>
</tr>
<tr>
<td style="text-align: left;">CML<span class="citation" data-cites="zhou2023enhancing"></span></td>
<td style="text-align: left;">Spikformer-2-256</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">97.6 , 98.6</td>
</tr>
<tr>
<td style="text-align: left;">S-Transformer<span class="citation" data-cites="yao2023spikedriven"></span></td>
<td style="text-align: left;"><span>S-Transformer-2-256</span></td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;"><span> <strong>99.3</strong></span></td>
</tr>
<tr>
<td style="text-align: left;">STSA<span class="citation" data-cites="ijcai2023p344"></span></td>
<td style="text-align: left;">STSFormer-2-256</td>
<td style="text-align: left;">1.99</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">97.3 , 98.7</td>
</tr>
<tr>
<td style="text-align: left;"><strong>QKFormer</strong></td>
<td style="text-align: left;">HST-2-256</td>
<td style="text-align: left;">1.50</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">98.3 , 98.6</td>
</tr>
<tr>
<td style="text-align: left;">Spikformer<span class="citation" data-cites="zhou2023spikformer"></span></td>
<td style="text-align: left;">Spikformer-2-256</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">78.9 , 80.9</td>
</tr>
<tr>
<td style="text-align: left;">Spikingformer<span class="citation" data-cites="zhou2023spikingformer"></span></td>
<td style="text-align: left;">Spikingformer-2-256</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">79.9 , 81.3</td>
</tr>
<tr>
<td style="text-align: left;"><span>2-6</span></td>
<td style="text-align: left;">CML<span class="citation" data-cites="zhou2023enhancing"></span></td>
<td style="text-align: left;">Spikformer-2-256</td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">79.2 , 80.9</td>
</tr>
<tr>
<td style="text-align: left;"><span>2-6</span></td>
<td style="text-align: left;">S-Transformer<span class="citation" data-cites="yao2023spikedriven"></span></td>
<td style="text-align: left;"><span>S-Transformer-2-256</span></td>
<td style="text-align: left;">2.57</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;"><span>80.0</span></td>
</tr>
<tr>
<td style="text-align: left;"><span>2-6</span></td>
<td style="text-align: left;">STSA<span class="citation" data-cites="ijcai2023p344"></span></td>
<td style="text-align: left;">STSFormer-2-256</td>
<td style="text-align: left;">1.99</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">78.96 , 79.93</td>
</tr>
<tr>
<td style="text-align: left;"><span>2-6</span></td>
<td style="text-align: left;"><strong>QKFormer</strong></td>
<td style="text-align: left;">HST-2-256</td>
<td style="text-align: left;">1.50</td>
<td style="text-align: left;">10 , 16</td>
<td style="text-align: left;">83.8 , <strong>84.0</strong></td>
</tr>
</tbody>
</table>

</div>

<span id="tab:small_dataset_appendix" label="tab:small_dataset_appendix"></span>

**Training and Testing Curve on ImageNet.** We show the training loss, test loss, top-1, and top-5 test accuracy of QKFormer (64.96M, 29.08M, 16.47M) on ImageNet-1K in Figure <a href="#fig:loss_accuracy" data-reference-type="ref" data-reference="fig:loss_accuracy">6</a>.

<figure id="fig:loss_accuracy">

<figcaption> Training loss, test loss, top-1 and top-5 test accuracy of QKFormer on ImageNet-1K. The input resolution of training and testing are 224 <span class="math inline">×</span> 224.</figcaption>
</figure>

## Energy Consumption Calculation of QKFormer and ANNs [sec：energy]

The homogeneity of convolution allows the following BN and linear scaling transformation to be equivalently fused into the convolutional layer with an added bias when deployment `\cite{ding2019acnet, ding2021repvgg, hu2021spiking, chen2023training}`{=latex}. Therefore, when calculating the theoretical energy consumption, the consumption of BN layers could be ignored. We calculate the number of Synaptic Operations (SOPs) of spike before calculating theoretical energy consumption for QKFormer. \\[\operatorname{S O P}^l=f r \times T \times \operatorname{FLOPs}^l\\] where \\(l\\) is a block/layer in QKFormer, \\(fr\\) is the firing rate of the block/layer and \\(T\\) is the simulation time step of spiking neuron. \\(\operatorname{FLOPs}^l\\) refers to floating point operations of block/layer \\(l\\), which is the number of multiply-and-accumulate (MAC) operations. And \\(\operatorname{SOP}^l\\) is the number of spike-based accumulate (AC) operations. Refer to previous works`\cite{kundu2021hire,hu2021advancing,horowitz20141,zhou2023spikformer,zhou2023spikingformer,panda2020toward,yao2023attention}`{=latex}. we assume that the MAC and AC operations are implemented on the 45nm hardware `\cite{horowitz20141}`{=latex}, where \\(E_{MAC}=4.6pJ\\) and \\(E_{AC}=0.9pJ\\). The theoretical energy consumption of QKFormer can be calculated as follows: \\[\begin{gathered}
E_{\operatorname{QKFormer}}=E_{A C} \times\left(\sum_{i=2}^N \operatorname{S O P}_{\operatorname{Conv} }^i+\sum_{j=1}^M \operatorname{S O P}_{\operatorname{QKTA}}^j+\sum_{k=1}^Z \operatorname{S O P}_{\operatorname{SSA}}^k\right)+E_{M A C} \times\left(\operatorname{FLOP}_{\operatorname{Conv}}^1\right) 
\label{eq:energy}
\end{gathered}\\]

Eq.<a href="#eq:energy" data-reference-type="ref" data-reference="eq:energy">[eq:energy]</a> shows the energy consumption of QKFormer. \\(\operatorname{FLOP}_{Conv}^1\\) is the first layer encoding the non-spike input into spike-form. Then the SOPs of \\(N\\) SNN Conv layers, \\(M\\) QKTA layers, and \\(Z\\) SSA layers are added together and multiplied by \\(E_{AC}\\). For ANNs, the theoretical energy consumption can be calculated: \\[\begin{gathered}
E_{\operatorname{ANN}}=E_{M A C} \times \operatorname{FLOPs}
\label{eq:energy_ann}
\end{gathered}\\]

## Supplementary for Memory Consumption in Experiment 4.3

Table <a href="#tab: values" data-reference-type="ref" data-reference="tab: values">9</a> shows the detailed values of Figure 4 in the main body of this paper (Experiment 4.3).

<div id="tab: values" markdown="1">

| \\(\sqrt{N}\\) | QKTA (M) | SSA (M) | SSA / QKTA |
|:---------------|:---------|:--------|:-----------|
| 10             | 0.10     | 0.14    | 1.37       |
| 20             | 0.40     | 1.00    | 2.53       |
| 30             | 0.89     | 3.97    | 4.46       |
| 40             | 1.58     | 12.19   | 7.71       |
| 50             | 2.47     | 26.44   | 10.70      |
| 60             | 3.56     | 53.52   | 15.04      |
| 70             | 4.84     | 97.64   | 20.17      |
| 80             | 6.32     | 162.50  | 25.70      |
| 90             | 8.18     | 258.19  | 31.55      |
| 100            | 10.35    | 391.77  | 37.85      |
| 110            | 12.14    | 570.69  | 47.01      |
| 120            | 14.23    | 806.06  | 56.66      |
| 130            | 16.70    | 1107.50 | 66.33      |
| 140            | 20.22    | 1485.14 | 73.43      |
| 150            | 22.26    | 1954.03 | 87.79      |
| 160            | 26.29    | 2525.00 | 96.03      |
| 170            | 28.55    | 3214.30 | 112.57     |
| 180            | 32.37    | 4036.88 | 124.71     |
| 190            | 36.41    | 5007.25 | 137.51     |
| 200            | 40.46    | 6143.06 | 151.84     |

Detailed values of memory consumption of Figure 4.

</div>

We compare the memory consumption between QKTA (Formula.<a href="#QKTA" data-reference-type="ref" data-reference="QKTA">[QKTA]</a>) and SSA (Formula.<a href="#SSA" data-reference-type="ref" data-reference="SSA">[SSA]</a>) under different token numbers, which is calculated on a GPU by forwarding the input tensor \\((T, B, C, N)\\). To facilitate the statistics of the impact of \#tokens \\(N\\) on memory consumption, the \#channels \\(C\\) are set to 256, and the time step \\(T\\) and batch size \\(B\\) are set to 1. The experiment result is shown in Figure <a href="#fig: qkta_b" data-reference-type="ref" data-reference="fig: qkta_b">[fig: qkta_b]</a>. With the increment of \#Tokens, SSA consumes much more GPU memory than QKTA, of which the complexity is linear to \#Tokens. For example, SSA consumes about \\(10\times\\) GPU memory than QKTA when \\(\sqrt{N} = 50\\).

## The Training and Inference Time Comparison.

We organized the experiment to test the training and inference time of QKFormer compared with Spikformer. We carried out this experiment on ImageNet with an input size of 224\*224. This experiment is carried out on a Ubuntu 18.04.6 LTS server with the Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz, and the GeForce RTX 2080 (8G) GPU. "BS" means Batch Size. The experimental results are as follows:

<div id="tab: training and inference" markdown="1">

| Model | Inference time (1 batch) | Training time (1 batch) |
|:---|:---|:---|
| Spikformer(29.68M, T=4) , BS=6 | 1.63s | 2.65s |
| QKFormer(29.08M,T=4) , BS=6 | 1.82s | 3.62s |
| Spikformer(29.68M, T=4) , BS=1 | 1.46s | 2.08s |
| QKFormer(29.08M, T=4) , BS=1 | 1.33s | 2.72s |

The training and inference time comparison between QKFormer and Spikformer.

</div>

In terms of inference time, QKFormer and Spikformer are very close. In terms of training time, QKFormer is about 1.35 times the training time of Spikformer in one batch, caused by hierarchical architecture. The training epochs of QKFormer on ImageNet are 200, while the training epochs of Spikformer are 300 `\cite{zhou2023spikformer}`{=latex}, thus, the total training time cost of QKFormer on ImageNet is close to Spikformer’s.

## Discussion [Limitation]

**Prospect.** The human brain has powerful intelligence that runs with low power consumption, so developing novel artificial intelligence algorithms to achieve high performance with the low power consumption of the human brain level is one of the AI’s ultimate goals. SNN is an attractive potential way to achieve it. QKFormer directly trained on ImageNet-1K has a groundbreaking leap forward with 10.84% accuracy improvement compared to the previous SNN model while maintaining the energy-efficient feature, which marks an important step towards this goal. Combined with pre-training in the future, the performance of QKFormer is expected to improve further.

**Reproducibility.** The experimental results in this paper are reproducible. All experiments are implemented based on Pytorch `\cite{paszke2019pytorch}`{=latex}, SpikingJelly `\cite{doi:10.1126/sciadv.adi1480}`{=latex} and Timm `\cite{rw2019timm}`{=latex}. We explain the details of model training and configuration in the main text and Appendix. Our codes and models of QKFormer will be available on GitHub after review.

[^1]: Equal

[^2]: Corresponding author
