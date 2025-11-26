# LP-3DGS: Learning to Prune 3D Gaussian Splatting

## Abstract

Recently, 3D Gaussian Splatting (3DGS) has become one of the mainstream methodologies for novel view synthesis (NVS) due to its high quality and fast rendering speed. However, as a point-based scene representation, 3DGS potentially generates a large number of Gaussians to fit the scene, leading to high memory usage. Improvements that have been proposed require either an empirical and preset pruning ratio or importance score threshold to prune the point cloud. Such hyperparamter requires multiple rounds of training to optimize and achieve the maximum pruning ratio, while maintaining the rendering quality for each scene. In this work, we propose learning-to-prune 3DGS (LP-3DGS), where a trainable binary mask is applied to the importance score that can find optimal pruning ratio automatically. Instead of using the traditional straight-through estimator (STE) method to approximate the binary mask gradient, we redesign the masking function to leverage the Gumbel-Sigmoid method, making it differentiable and compatible with the existing training process of 3DGS. Extensive experiments have shown that LP-3DGS consistently produces a good balance that is both efficient and high quality.

# Introduction

Novel view synthesis (NVS) takes images and their corresponding camera poses as input and seek to render new images from different camera poses after 3D scene reconstruction. Neural Radiance Fields (NeRF) (`\citet{mildenhall2021nerf}`{=latex}) uses multi-layer perceptron (MLP) to implicitly represent the scene, fetching the transparency and color of a point from the MLPs. NeRF has gained considerable attention in the NVS community due to its simple implementation and excellent performance. However, in order to get a point in the space, NeRF needs to perform an MLP inference. Each pixel rendered requires processing a ray and there are many sample points on each ray. Consequently, rendering an image requires a large amount of MLP inference operations. Thus, rendering speed becomes a major drawback of NeRF method.

Besides NeRF, explicit 3D representations are also widely used. Compared with NeRF, the advantage of point-based scene representation is that modern GPU rendering is well supported, enabling fast render speed. 3D Gaussian Splatting (3DGS) (`\citet{kerbl20233d}`{=latex}) achieves good quality and high rendering speed, making it a hot topic in the community. 3DGS uses 3D Gaussian models with color parameters to fit the scene and develops a training framework to optimize the model parameters. However, the number of points required to reconstruct the scene is huge, usually in the millions. In practice, each point has dozens of floating point parameters, which makes 3DGS a memory-intensive method.

Some recent works have tried to mitigate this problem by pruning the points, such as LightGaussian (`\citet{fan2023lightgaussian}`{=latex}), RadSplat (`\citet{niemeyer2024radsplat}`{=latex}), and Mini-Splatting (`\citet{fang2024mini}`{=latex}). These methods follow a similar pruning approach through defining an *importance score* for each Gaussian point and prune the points with such importance score below than a preset empirical threshold. However, a major drawback of these methods is that such preset threshold needs to be manually tuned through multiple rounds of training process to identify the optimal pruning ratio to minimize the number of Gaussian points while keeping the rendering quality. To make it even worse, such optimal number of points may vary depending on different scenes, which requires manual pruning ratio searching for each scene. For example, the blue and red lines in Figure <a href="#fig:performance_with_ratio" data-reference-type="ref" data-reference="fig:performance_with_ratio">1</a> show the rendering quality of *Kitchen* and *Room* scenes, respectively, in MipNeRF360 scenes (`\citet{barron2022mip}`{=latex}), with sweeping 12 different pruning ratios (i.e., 12 rounds of training) following the prior RadSplat (`\citet{niemeyer2024radsplat}`{=latex}) and Mini-Splatting (`\citet{fang2024mini}`{=latex}) method. It could be clearly seen that a smaller pruning ratio will not hamper the rendering quality, and the rendering quality will start to decrease with much more aggressive pruning ratios. While, both scenes exist an optimal pruning ratio region that could maximize the pruning ratio and maintain the rendering quality. It could also be seen that such optimal pruning ratio region is different for these two scenes.

<figure id="fig:performance_with_ratio">
<figure>
<img src="./figures/PSNR_KitchenRoom_radsplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_KitchenRoom_radsplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_KitchenRoom_radsplat.png"" />
</figure>
<figcaption>The performance changes with the pruning ratio of RadSplat on the MipNeRF360 dataset <em>Kitchen</em> and <em>Room</em> scenes are shown in <span style="color: blue">blue</span> and <span style="color: green">green</span> lines, respectively. <span style="color: red">Red</span> triangles and squares represent the results of LP-3DGS on the importance score of RadSplat. <strong>LP-3DGS is able to find the optimal pruning ratio in one training session instead of requiring dozens of attempts to find the best hyperparameter</strong>.</figcaption>
</figure>

In this paper, we propose a learning-to-prune 3DGS (LP-3DGS) methodology where a trainable mask is applied to the importance score. Note that, it is compatible with different types of importance scores defined in prior works. Instead of a preset threshold to determine the 3DGS model pruning ratio, as shown by the red triangle symbol in the Figure <a href="#fig:performance_with_ratio" data-reference-type="ref" data-reference="fig:performance_with_ratio">1</a>, our method aims to integrate with existing 3DGS model training process to learn the optimal pruning ratio for minimizing the model size while maintaining the rendering quality. Since the traditional hard threshold based binary masking function is not differentiable, a recent prior work, Compact3D (`\citet{lee2023compact}`{=latex}), leverages the popular straight through estimator (STE) (`\citet{bengio2013estimating}`{=latex}) to bypass the mask gradient for adaption to the backpropagation process. Such method inevitably leads to non-optimal learned pruning ratio. In contrast, in our work, we propose to redesign the masking function leveraging the Gumbel-Sigmoid activation function to make the whole masking function differentiable and integrate with the existing training process of 3DGS. As a result, our LP-3DGS could minimize the number of Gaussian points automatically for each scene with only one-time training.

In summary, the technical contributions of our work are:

- To address the effortful 3DGS optimal pruning ratio tuning, we propose a learning-to-prune 3DGS (LP-3DGS) methodology that leverages the differentiable Gumbel-Sigmoid activation function to embed a trainable mask with different types of existing importance scores designed for pruning redundant Gaussian points. As a result, instead of fixed model size, LP-3DGS could learn an optimal Gaussian point size for individual scene with only one-time training.

- We conducted comprehensive experiments on state-of-the-art (SoTA) 3D scene datasets, including MipNeRF360 (`\citet{barron2022mip}`{=latex}), NeRF-Synthetic (`\citet{mildenhall2021nerf}`{=latex}), and Tanks & Temples (`\citet{knapitsch2017tanks}`{=latex}). We compared our method with SoTA pruning methods such as RadSplat (`\citet{niemeyer2024radsplat}`{=latex}), Mini-Splatting (`\citet{fang2024mini}`{=latex}), and Compact3D (`\citet{lee2023compact}`{=latex}). The experimental results show that our method can enable the model to learn the optimal pruning ratio and that our trainable mask method performs better than the STE mask.

# Related Work

#### Neural radiance fields (NeRFs)

NeRFs (`\citet{mildenhall2021nerf}`{=latex}) targets to represent the scene in multilayer perceptrons (MLPs) based on multi-view image inputs, enabling high-quality novel view synthesis. Due to its advancement, numerous follow-up works improved it in either rendering quality (`\citet{barron2021mip,barron2022mip}`{=latex}) or efficiency(`\citet{muller2022instant,chen2022tensorf,fridovich2022plenoxels}`{=latex}).

Although NeRF models demonstrate impressive rendering capabilities across numerous benchmarks, and considerable efforts have been made to enhance training and inference efficiency, they typically still face challenges in achieving fast training and real-time rendering.

#### Radiance Field Based On Points.

In addition to implicit representations, several works have focused on volumetric point-based methods for 3D presentation (`\citet{gross2011point}`{=latex}). Inspired by neural network concepts, (`\citet{aliev2020neural}`{=latex}) introduced a neural point-based approach to streamline the construction process. Point-NeRF (`\citet{ding2024point}`{=latex}) further applied points for volumetric representation, enhancing the effectiveness of point-based methods in radiance field modeling.

#### Gaussian Splatting

3D Gaussian Splatting (3DGS) (`\citet{kerbl20233d}`{=latex}) represents a significant advancement in novel view synthesis, utilizing 3D Gaussians as primitives to explicitly represent scenes. This approach achieves state-of-the-art rendering quality and speed while maintaining relatively short training time. A series of methods have been introduced to improve the rendering quality through using regularization for better optimization, including depth map (`\citet{chung2023depth,li2024dngaussian}`{=latex}), surface alignment (`\citet{guedon2023sugar, li2024geogaussian}`{=latex}) and rendered image frequency (`\citet{zhang2024fregs}`{=latex}). However, the extensive number of Gaussians required for scene representation often results in a model that is too large for efficient storage. Recent research has focused on compression methods to enhance the efficiency of this representation. Notably, several studies (`\citet{fan2023lightgaussian,fang2024mini,niemeyer2024radsplat}`{=latex}) have proposed using predefined scores as pruning criteria to keep Gaussians that significantly contribute to rendering quality. Compact3D (`\citet{lee2023compact}`{=latex}) introduces a method that applies a trainable mask on scale and opacity to each Gaussian and utilizes a straight-through estimator (`\citet{bengio2013estimating}`{=latex}) for gradient updates. LightGaussian (`\citet{fan2023lightgaussian}`{=latex}) employs knowledge distillation to reduce the dimension of spherical harmonics. Additionally, (`\citet{fan2023lightgaussian,lee2023compact}`{=latex}) also explored quantization techniques to further compress model storage. The previously proposed pruning methods primarily rely on predefined scores to determine the importance of each Gaussian. These approaches present two main challenges: first, whether the criteria accurately reflect the importance of the Gaussians, and second, the need for a manually selected pruning threshold to decide the level of pruning. In this work, we address these issues by introducing a trainable mask activated by a Gumbel-sigmoid function, applied to the scores derived from prior methods or directly to the scale and opacity of each Gaussian for more flexibility. Our approach automatically identifies an optimal balance between the pruning ratio and rendering quality, eliminating the need to test on various pruning ratios.

# Methodology

The conventional pruning methods leveraging predefined importance score require pruning ratio as a manually tuned parameter to reduce the size of Gaussian points in 3DGS. To seeking for the optimal pruning ratio, these methods may need to perform multiple rounds of training for each individual scene, which is inefficient. Thus motivated, we propose a **learning-to-prune 3DGS (LP-3DGS)** algorithm which learns a binary mask to determine the optimal pruning ratio for each scene automatically. Importantly, the proposed LP-3DGS is compatible with different types of pruning importance score. In this section, we will: 1) introduce the preliminary of the original 3DGS and recap different importance metrics for pruning that are proposed by prior works, and 2) present the proposed learning-to-prune 3DGS algorithm.

## 3DGS Background

#### 3DGS Parameters

3DGS is an explicit point-based 3D representation that uses Gaussian points to model the scene. Each point has the following attributes: position \\(\bf{p} \in \mathbb{R}^3\\), opacity \\(\sigma \in [0, 1]\\), scale in 3D \\(\bf{s} \in \mathbb{R}^3\\), rotation presented by 4D quaternions \\(\bf{q} \in \mathbb{R}^4\\) and forth-degree spherical harmonics (SH) coefficients \\(\bf{k} \in \mathbb{R}^{48}\\). In summary, one gaussian point has 59 parameters. The center point \\(X\\) of a Gaussian model is denoted by \\(\bf{p}\\) and covariance matrix \\(\Sigma\\) is denoted by \\(\bf{s}\\) and \\(\bf{q}\\). The SH coefficients model the color as viewed from different directions. The parameters of the Gaussians are optimized through gradient backpropagation of the loss between the rendered images and the ground truth.

#### Rendering on 3DGS

In order to render an image, the first step is projecting the Gaussians to 2D camera plane by world to camera transform matrix \\(W\\) and Jacobian \\(J\\) of affine approximation of the projective transform. The covariance matrix of projected 2D Gaussian is \\[\Sigma^{'} = JW\Sigma W^TJ^T\\] The projected Gaussians would be rendered as splat (`\citet{botsch2005high}`{=latex}), the color of one pixel could be rendered as \\[\bf{c}_i = \sum_{j=1}^{N} \cdot \bf{c}_j \cdot \alpha_j \cdot T_j \cdot \bf{G}^{2D}_j\\] Where \\(i\\) is the pixel index, \\(j\\) is the Gaussian index and \\(N\\) is the number of the Gaussians in the ray. \\(c_j\\) is the color of the Gaussian calculated by SH coefficients, \\(\alpha_j = (1 - exp^{-\sigma_j \delta_j})\\), \\(\sigma_j\\) is the opacity of the point and \\(\delta_j\\) is the interval between points. \\(T_j = \prod_{k=1}^{j-1}(1 - \alpha_k)\\) is the transmittance from the start of rendering to this point. \\(\bf{G}^{2D}_j\\) is the 2D Gaussian distribution.

#### Adaptive Density Control of 3DGS

At the start of training, the Gaussians are initialized using Structure-from-Motion (SfM) sparse points. To make the Gaussians fit the scene better, 3DGS applies an adaptive density control strategy to adjust the number of Gaussians. Periodically, 3DGS will grow Gaussians in areas that are not well reconstructed, a process called "densification." Simultaneously, Gaussians with low opacity will be pruned.

## Importance Metrics for Pruning

A straightforward way to prune the Gaussians is by sorting them based on a defined importance score and then removing the less important ones. As a result, one of the main objective of prior 3DGS pruning works is to define the importance metric.

RadSplat (`\citet{niemeyer2024radsplat}`{=latex}) defines the importance score as the maximum contribution along all rays of the training images, written as \\[\bf{S}_i = \max_{I_f\in \mathcal{I}_f, r\in I_f} \alpha_i^r \cdot T_i^r\\] Where \\(\alpha_i^r \cdot T_i^r\\) is the contribution of Gaussian \\(G_i\\) along ray \\(r\\). RadSplat performs pruning by applying a binary mask according to the importance score, where the mask value for Gaussian \\(G_i\\) is \\[m_i = m(\bf{S}_i) = \mathds{1}(\bf{S}_i < t_{prune})\\] Where \\(t_{prune} \in [0,1]\\) is the threshold of score magnitude for pruning, \\(\mathds{1} [\cdot]\\) is the indicator function.

Another recent work, Mini-Splatting (`\citet{fang2024mini}`{=latex}), uses the cumulative weight of the Gaussian as the importance score, which can be formulated as: \\[\bf{S}_i = \sum_{j=1}^{K} \omega_{ij}\\] Where K is the total number of rays intersected with Gaussian \\(G_i\\), \\(\omega_{ij}\\) is the color weight of Gaussian \\(G_i\\) on the \\(j\\)-th ray.

<figure id="fig:overview">
<img src="./figures/overview.png"" style="width:100.0%" />
<figcaption>Overall learning process of the proposed LP-3DGS.</figcaption>
</figure>

## Learning-to-Prune 3DGS

The overall LP-3DGS learning process is shown in the Figure <a href="#fig:overview" data-reference-type="ref" data-reference="fig:overview">2</a>. In general, it mainly can be divided into two stages:1) densification stage, and 2) learning-to-prune stage. Following the original 3DGS, densification stage applies an adaptive density control strategy to gradually increase the number of Gaussians. As revealed by prior pruning works (`\citet{lee2023compact,niemeyer2024radsplat}`{=latex}), 3DGS exists redundant Gaussians significantly. Subsequently, in the learning-to-prune stage, the proposed LP-3DGS learns a trainable mask upon prior defined importance metric to compress the number of Gaussians with an optimal pruning ratio automatically. Specifically, to learn a binary mask, we first initialize a real-value mask \\(m_i\\) for each point \\(i\\), and then adopt the Gumbel-sigmoid technique to binarize the mask value differentially.

#### Gumbel-Sigmoid based Trainable Mask

The binarization operation for real-value mask in pruning usually involves a hard threshold function, determining the binary mask should be 0 or 1. However, such hard threshold function is not differential during backpropagation. To solve this issue, popular straight through estimator (STE) method (`\citet{bengio2013estimating}`{=latex}) is widely used which skips the gradient of the threshold function during backpropagation. Such process may lead to a gap between trainable real-value mask and binary mask. As shown in Figure <a href="#fig:activations" data-reference-type="ref" data-reference="fig:activations">5</a>(a), the trainable mask values exist certain ratios cross the whole range from 0 to 1 after Sigmoid function, which could be inaccurate when further converting to binary mask via a hard threshold function. To better optimize the trainable mask towards binary values, we propose to apply Gumbel-Sigmoid function to learn the binary mask.

The Gumbel distribution is used to model the extreme value distribution and generate samples from the categorical distribution (`\citet{gumbel1954statistical}`{=latex}). This property is then utilized to create the Gumbel-Softmax (`\citet{jang2016categorical}`{=latex}), a differentiable categorical distribution sampling function. The sample of one category is given by:

\\[y_i = \frac{\exp((\log(\pi_i) + g_i)/\tau)}{\sum_{j=1}^{k} \exp((\log(\pi_j) + g_j)/\tau)}\\] Where \\(\tau\\) is the input adjustment parameter, \\(g_i\\) is sample from Gumbel distribution. Inspired by the Gumbel-Softmax, we treat learning the binary mask of each point as a two-class category problem. Thus, we replace the Softmax function to Sigmoid function, referring to Gumbel-Sigmoid: \\[\label{equ:gumbel_sigmoid}
    gs(m) = \frac{exp((\log(m) + g_0)/\tau)}{exp((\log(m)+g_0)/T)+exp(g_1/\tau)} = \frac{1}{1+exp(-(\log(m) + g_0 - g_1)/\tau)}\\]

<figure id="fig:activations">
<figure id="fig:sigmoid">
<img src="./figures/mask_sigmoid.png"" />
<figcaption>Mask values after Sigmoid activation</figcaption>
</figure>
<figure id="fig:gumbel_sigmoid">
<img src="./figures/mask_gumbel.png"" />
<figcaption>Mask values after Gumbel-Sigmoid activation</figcaption>
</figure>
<figcaption>Comparison between Sigmoid and Gumbel-Sigmoid. The Gumbel-Sigmoid function pushes the values closer to 0 or 1 and is a good approximation of a binarized mask.</figcaption>
</figure>

By using such Gumbel-Sigmoid function, the output value is either close to 0 or 1, as shown in Figure <a href="#fig:activations" data-reference-type="ref" data-reference="fig:activations">5</a>, and thus can be utilized as an approximation of a binary masking function. More importantly, this function remains differentiable, thus can be integrated during backpropagation.

Moreover, to prune the selected Guassians practically according to the learned binary mask, we further apply the mask value on opacity, which can be mathematically formulated as \\[\label{equ:apply_mask}
    o_{im} = o_i * gs(m_i*S_i)\\] where \\(S_i\\) is the defined importance score of each Gaussian point. The closer the mask value is to 0, the less corresponding Gaussian point contributes to the rendering. In practice, after learning the trainable mask, a one-time pruning is applied to the corresponding Gaussian points with mask value of 0.

#### Sparsity regularization

In order to compress the model as much as possible, we apply a L1 regularization term (`\citet{lee2023compact}`{=latex}) to encourage the trainable mask to be sparse, which can be formulated as: \\[R_{mask} = \frac{1}{N} \sum_{i=1}^{N} \left| m_i \right|\\] Upon that, the final loss function is defined as: \\[L = (1-\lambda_{ssim})*L_{L1} + \lambda_{ssim}*L_{ssim} + \lambda_m*R_{mask}\\] \\(L_{L1}\\) is the L1 loss between rendered image and ground truth. \\(L_{ssim}\\) is the ssim loss. \\(\lambda_{ssim}\\) and \\(\lambda_{m}\\) are two coefficients.

Moreover, we find that the trainable mask can be effectively learned in just a few hundred iterations, compared to the thousands needed for the overall training process. In practice, the mask learning function is activated for only 500 iterations. Once the mask values are learned, we follow the 3DGS training setup to further fine-tune the pruned model, maintaining the same total number of training iterations. The detailed hyper parameters are described in the later experiment section.

# Experiments [sec:experiments]

## Experimental Settings [sec:exp_settings]

#### Dataset and Baseline

We test our method on two most popular real-world datasets: the MipNeRF360 dataset (`\citet{barron2022mip}`{=latex}), which contains 9 scenes, and the *Train* and *Truck* scenes from the Tanks & Temples dataset (`\citet{knapitsch2017tanks}`{=latex}). We also test on the NeRF-Synthetic dataset (`\citet{mildenhall2021nerf}`{=latex}), which includes 8 synthetic scenes. In this section, we only list the results on MipNeRF360 dataset, rest of them are listed in appendix <a href="#appendix" data-reference-type="ref" data-reference="appendix">6</a>. In this paper, we use the SoTA RadSplat (`\citet{niemeyer2024radsplat}`{=latex}) and Mini-Splatting (`\citet{fang2024mini}`{=latex}) as the baselines, which propose two different pruning importance scores. First, we test the performance of these two methods under different pruning ratios. Since neither method is open-sourced at the time of writing, we reproduced them based on the provided equations. For each pruning ratio, we calculate the corresponding threshold based on the magnitude of the importance scores and prune the Gaussians with scores below the threshold. Note that, each pruning ratio requires one round of training. We use peak signal-to-noise ratio (PSNR), structural similarity index measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) (`\citet{zhang2018unreasonable}`{=latex}) as rendering evaluation metrics.

#### Implement Details

The machine running the experiments has an AMD 5955WX processor and two Nvidia A6000 GPUs. It should be noted that our method does not support multi-GPU training. We ran different experiments simultaneously on two GPUs. We train each scene under every setting for 30,000 iterations and train the mask during iterations 19,500 to 20,000, updating the importance score every 20 iterations. The \\(\tau\\) in Equation <a href="#equ:gumbel_sigmoid" data-reference-type="ref" data-reference="equ:gumbel_sigmoid">[equ:gumbel_sigmoid]</a> is 0.5 and the coefficient \\(\lambda_m\\) of mask loss is 5e-4.

## Experimental Results [sec:exp_results]

#### Quantitative Results

The blue lines in Figure <a href="#fig:garden" data-reference-type="ref" data-reference="fig:garden">6</a> show the results of sweeping pruning ratios using RadSplat and Mini-Splatting for the *Kitchen*, and *Room* scenes. Figures on other scenes of MipNeRF 360 are listed in Appendix <a href="#appendix" data-reference-type="ref" data-reference="appendix">6</a>. The result of the learned LP-3DGS model size and rendering quality is indicated by red triangles.

<figure id="fig:garden">
<figure>
<img src="./figures/PSNR_Kitchen_radsplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Kitchen_radsplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Kitchen_radsplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Kitchen_minisplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Kitchen_minisplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Kitchen_minisplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Room_radsplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Room_radsplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Room_radsplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Room_minisplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Room_minisplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Room_minisplat.png"" />
</figure>
<figcaption>The performance changes with the pruning ratio in different scenes</figcaption>
</figure>

The quantitative results fluctuate at lower pruning ratios but generally maintain around a certain value up to a specific point. After passing that point, the rendering quality decreases significantly. It’s worth noting that such decreasing point varies for different scenes. Instead of manually searching such optimal pruning ratio, it clearly shows our LP-3DGS method could learn the optimal model size embedding with the scene learning process, with only one-time training. The Table <a href="#tab:prune_ratio_list" data-reference-type="ref" data-reference="tab:prune_ratio_list">1</a> lists the quantitative results of all scenes in MipNeRF360 dataset. It clearly shows that each scene converge into different model size leveraging our LP-3DGS method with maintaining almost the same rendering quality. The pruning ratio varies based on what importance score is used, but LP-3DGS could find the optimal pruning ratio on the corresponding score.

<div class="adjustbox" markdown="1">

max width=

<div id="tab:prune_ratio_list" markdown="1">

| Scene | Bicycle | Bonsai | Counter | Kitchen | Room | Stump | Garden | Flowers | Treehill | AVG |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Baseline PSNR \\(\uparrow\\) | 25.087 | 32.262 | 29.079 | 31.581 | 31.500 | 26.655 | 27.254 | 21.348 | 22.561 | 27.48 |
| LP-3DGS (RadSplat Score) | 25.099 | 32.094 | 28.936 | 31.515 | 31.490 | 26.687 | 27.290 | 21.383 | 22.706 | 27.47 |
| LP-3DGS (Mini-Splatting Score) | 24.906 | 31.370 | 28.4098 | 30.785 | 31.132 | 26.679 | 27.095 | 21.150 | 22.522 | 27.12 |
| Baseline SSIM \\(\uparrow\\) | 0.7464 | 0.9460 | 0.9138 | 0.9320 | 0.9249 | 0.7700 | 0.8557 | 0.5876 | 0.6358 | 0.8125 |
| LP-3DGS (RadSplat Score) | 0.7458 | 0.9441 | 0.9120 | 0.9311 | 0.9243 | 0.7714 | 0.8548 | 0.5865 | 0.6381 | 0.8120 |
| LP-3DGS (Mini-Splatting Score) | 0.7373 | 0.9358 | 0.9017 | 0.9249 | 0.9167 | 0.7677 | 0.8493 | 0.5756 | 0.6336 | 0.8047 |
| Baseline LPIPS \\(\downarrow\\) | 0.2441 | 0.1799 | 0.1839 | 0.1164 | 0.1978 | 0.2423 | 0.1224 | 0.3601 | 0.3469 | 0.2215 |
| LP-3DGS (RadSplat Score) | 0.2516 | 0.1865 | 0.1896 | 0.1194 | 0.2032 | 0.2466 | 0.1270 | 0.3656 | 0.3527 | 0.2269 |
| LP-3DGS (Mini-Splatting Score) | 0.2642 | 0.2036 | 0.2068 | 0.1292 | 0.2208 | 0.2553 | 0.1353 | 0.3753 | 0.3618 | 0.2391 |
| RadSplat Score pruning ratio | 0.64 | 0.65 | 0.66 | 0.58 | 0.74 | 0.65 | 0.59 | 0.59 | 0.59 | 0.63 |
| Mini-Splatting Score pruning ratio | 0.57 | 0.67 | 0.64 | 0.56 | 0.71 | 0.61 | 0.60 | 0.54 | 0.54 | 0.60 |

The results comparison on the MipNeRF360 dataset shows that LP-3DGS has similar performance after pruning and achieves different pruning ratios for different scenes. This demonstrates LP-3DGS’s ability to adaptively find the optimal pruning ratio, maintaining performance while effectively compressing the model.

</div>

</div>

#### Training Cost

Table <a href="#tab:training cost" data-reference-type="ref" data-reference="tab:training cost">2</a> shows the training cost of LP-3DGS on MipNeRF360 dataset. In our setup, since after 20000th iteration, the model will be pruned based on the learned mask values. The number of Gaussian points will be significantly reduced, which makes the later stages of training take much less time than the non-pruned version. Even with the embedding of the mask learning function, the overall training cost is similar with that of the vanilla 3DGS. In most cases, the peak training memory usage is slightly larger because training the mask requires more GPU memory. However, after pruning, the 3DGS model size becomes much smaller, leading to a significant improvement in rendering speed, measured in terms of FPS.

<div class="adjustbox" markdown="1">

max width=

<div id="tab:training cost" markdown="1">

| Scene | Bicycle | Bonsai | Counter | Kitchen | Room | Stump | Garden | Flowers | Treehill |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 3DGS Training time (Minute) | 49 | 34 | 26 | 33 | 27 | 37 | 47 | 33 | 32 |
| LP-3DGS (RadSplat Score) | 43 | 27 | 28 | 34 | 30 | 35 | 46 | 34 | 33 |
| LP-3DGS (Mini-Splatting Score) | 44 | 27 | 28 | 35 | 29 | 34 | 46 | 34 | 33 |
| 3DGS Peak Memory (GB) | 14.7 | 8.6 | 9.4 | 9.3 | 10.6 | 12.2 | 15.7 | 10.3 | 9.4 |
| LP-3DGS (RadSplat Score) | 16.1 | 8.5 | 11.3 | 12.3 | 11.8 | 12.1 | 15.8 | 10.1 | 10.3 |
| LP-3DGS (Mini-Splatting Score) | 15.5 | 8.4 | 12.7 | 13.0 | 13.0 | 12.1 | 15.2 | 9.7 | 9.7 |
| 3DGS FPS | 132 | 417 | 421 | 315 | 380 | 164 | 129 | 200 | 205 |
| LP-3DGS (RadSplat Score) | 324 | 662 | 670 | 542 | 692 | 371 | 296 | 412 | 411 |
| LP-3DGS (Mini-Splatting Score) | 290 | 634 | 650 | 507 | 662 | 341 | 252 | 368 | 384 |

Training cost and on MipNeRF360 Dataset. Training time of LP-3DGS is similar with baseline but since the model is compressed, the FPS is larger.

</div>

</div>

## Ablation Study [sec:ablation]

A recent prior work Compact3D (`\citet{lee2023compact}`{=latex}) proposes to leverage STE to train a binary mask on opacity and scale of Gaussian parameter. To conduct a fair comparison between STE based mask and our LP-3DGS, we make two ablation studies, one is replacing the STE mask in Compact3D with our method, the other is applying STE mask on importance score of RadSpalt. The formula of STE mask is \\[\label{equ:ste}
    M(m) = \mathrlap{\nabla}\hspace{0.5pt}/\, ( \mathds{1}[f(m) > \epsilon] - f(m) ) + f(m)\\] \\(\mathrlap{\nabla}\hspace{0.5pt}/\\) means stop gradients, \\(\mathds{1} [\cdot]\\) is the indicator function and \\(f(\cdot)\\) is sigmoid function, \\(\epsilon\\) is masking threshold.

#### Comparison with Compact3D

We firstly apply Gumbel-sidmoid activated mask, instead of STE mask, on the opacity and scale of gaussians in the same way as proposed in Compact3D. The threshold \\(\epsilon\\) in Equation <a href="#equ:ste" data-reference-type="ref" data-reference="equ:ste">[equ:ste]</a> and mask loss coefficient follows the default settings in Compact3D. Table <a href="#tab:ste_vs_gumbel" data-reference-type="ref" data-reference="tab:ste_vs_gumbel">3</a> shows the comparison between two methods.

<div class="adjustbox" markdown="1">

max width=

<div id="tab:ste_vs_gumbel" markdown="1">

<table>
<caption>Results with/without trainable mask on Gaussian opacity and scale</caption>
<thead>
<tr>
<th style="text-align: center;"></th>
<th style="text-align: center;">Scene</th>
<th style="text-align: center;">Bicycle</th>
<th style="text-align: center;">Bonsai</th>
<th style="text-align: center;">Counter</th>
<th style="text-align: center;">Kitchen</th>
<th style="text-align: center;">Room</th>
<th style="text-align: center;">Stump</th>
<th style="text-align: center;">Garden</th>
<th style="text-align: center;">Flowers</th>
<th style="text-align: center;">Treehill</th>
<th style="text-align: center;">AVG</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;">PSNR</td>
<td style="text-align: center;">Compact3D</td>
<td style="text-align: center;">24.846</td>
<td style="text-align: center;">32.19</td>
<td style="text-align: center;">29.066</td>
<td style="text-align: center;">30.867</td>
<td style="text-align: center;">31.489</td>
<td style="text-align: center;">26.408</td>
<td style="text-align: center;">27.026</td>
<td style="text-align: center;">21.187</td>
<td style="text-align: center;">22.479</td>
<td style="text-align: center;">27.284</td>
</tr>
<tr>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">25.087</td>
<td style="text-align: center;">32.2</td>
<td style="text-align: center;">29.033</td>
<td style="text-align: center;">31.213</td>
<td style="text-align: center;">31.678</td>
<td style="text-align: center;">26.658</td>
<td style="text-align: center;">27.223</td>
<td style="text-align: center;">21.32</td>
<td style="text-align: center;">22.569</td>
<td style="text-align: center;">27.442</td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">SSIM</td>
<td style="text-align: center;">Compact3D</td>
<td style="text-align: center;">0.7292</td>
<td style="text-align: center;">0.9462</td>
<td style="text-align: center;">0.9137</td>
<td style="text-align: center;">0.925</td>
<td style="text-align: center;">0.9251</td>
<td style="text-align: center;">0.7563</td>
<td style="text-align: center;">0.8446</td>
<td style="text-align: center;">0.5773</td>
<td style="text-align: center;">0.6305</td>
<td style="text-align: center;">0.8053</td>
</tr>
<tr>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">0.7438</td>
<td style="text-align: center;">0.9461</td>
<td style="text-align: center;">0.9141</td>
<td style="text-align: center;">0.9305</td>
<td style="text-align: center;">0.9263</td>
<td style="text-align: center;">0.7687</td>
<td style="text-align: center;">0.8547</td>
<td style="text-align: center;">0.5843</td>
<td style="text-align: center;">0.6358</td>
<td style="text-align: center;">0.8116</td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">LPIPS</td>
<td style="text-align: center;">Compact3D</td>
<td style="text-align: center;">0.266</td>
<td style="text-align: center;">0.1815</td>
<td style="text-align: center;">0.1866</td>
<td style="text-align: center;">0.124</td>
<td style="text-align: center;">0.2012</td>
<td style="text-align: center;">0.2615</td>
<td style="text-align: center;">0.1401</td>
<td style="text-align: center;">0.3722</td>
<td style="text-align: center;">0.3555</td>
<td style="text-align: center;">0.2320</td>
</tr>
<tr>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">0.2526</td>
<td style="text-align: center;">0.1833</td>
<td style="text-align: center;">0.1867</td>
<td style="text-align: center;">0.1201</td>
<td style="text-align: center;">0.2013</td>
<td style="text-align: center;">0.2472</td>
<td style="text-align: center;">0.1275</td>
<td style="text-align: center;">0.3668</td>
<td style="text-align: center;">0.3513</td>
<td style="text-align: center;">0.2263</td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">#Gaussians</td>
<td style="text-align: center;">Compact3D</td>
<td style="text-align: center;">2620663</td>
<td style="text-align: center;">666558</td>
<td style="text-align: center;">570126</td>
<td style="text-align: center;">1050079</td>
<td style="text-align: center;">566332</td>
<td style="text-align: center;">1902711</td>
<td style="text-align: center;">2412796</td>
<td style="text-align: center;">1685224</td>
<td style="text-align: center;">2089515</td>
<td style="text-align: center;">1507109</td>
</tr>
<tr>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">2510992</td>
<td style="text-align: center;">542235</td>
<td style="text-align: center;">506391</td>
<td style="text-align: center;">887161</td>
<td style="text-align: center;">479681</td>
<td style="text-align: center;">2014270</td>
<td style="text-align: center;">2836989</td>
<td style="text-align: center;">1747766</td>
<td style="text-align: center;">1804155</td>
<td style="text-align: center;">1481071</td>
</tr>
</tbody>
</table>

</div>

</div>

For most cases, our LP-3DGS learns a higher pruning ratio, except Stump, Garden and Flowers scene. In terms of rendering quality, our LP-3DGS outperforms compact3D using STE based mask with even smaller model size in most scenes.

#### STE Mask on Importance Score

We also apply STE mask on the pruning importance score to compare with our method. The Equation <a href="#equ:apply_mask" data-reference-type="ref" data-reference="equ:apply_mask">[equ:apply_mask]</a> would be rewriten as \\[o_{im} = o_i * M(m_i*is)\\] where \\(M\\) is shown in Equation <a href="#equ:ste" data-reference-type="ref" data-reference="equ:ste">[equ:ste]</a>. The same as mentioned before, the parameters for STE mask are default values in Compact3D.

<div class="adjustbox" markdown="1">

max width=

<div id="tab:ste_vs_gumbel_radsplat" markdown="1">

<table>
<caption>Results using LP-3DGS and STE mask on importance score of RadSplat</caption>
<thead>
<tr>
<th style="text-align: center;"></th>
<th style="text-align: center;">Scene</th>
<th style="text-align: center;">Bicycle</th>
<th style="text-align: center;">Bonsai</th>
<th style="text-align: center;">Counter</th>
<th style="text-align: center;">Kitchen</th>
<th style="text-align: center;">Room</th>
<th style="text-align: center;">Stump</th>
<th style="text-align: center;">Garden</th>
<th style="text-align: center;">Flowers</th>
<th style="text-align: center;">Treehill</th>
<th style="text-align: center;">AVG</th>
<th style="text-align: center;"></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;">PSNR</td>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">25.099</td>
<td style="text-align: center;">32.094</td>
<td style="text-align: center;">28.936</td>
<td style="text-align: center;">31.515</td>
<td style="text-align: center;">31.490</td>
<td style="text-align: center;">26.687</td>
<td style="text-align: center;">27.290</td>
<td style="text-align: center;">21.383</td>
<td style="text-align: center;">22.706</td>
<td style="text-align: center;">27.470</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">STE mask</td>
<td style="text-align: center;">24.833</td>
<td style="text-align: center;">30.947</td>
<td style="text-align: center;">28.371</td>
<td style="text-align: center;">30.705</td>
<td style="text-align: center;">30.950</td>
<td style="text-align: center;">26.396</td>
<td style="text-align: center;">26.793</td>
<td style="text-align: center;">21.056</td>
<td style="text-align: center;">22.552</td>
<td style="text-align: center;">26.955</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">SSIM</td>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">0.7458</td>
<td style="text-align: center;">0.9441</td>
<td style="text-align: center;">0.9120</td>
<td style="text-align: center;">0.9311</td>
<td style="text-align: center;">0.9243</td>
<td style="text-align: center;">0.7714</td>
<td style="text-align: center;">0.8548</td>
<td style="text-align: center;">0.5865</td>
<td style="text-align: center;">0.6381</td>
<td style="text-align: center;">0.8120</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">STE mask</td>
<td style="text-align: center;">0.7231</td>
<td style="text-align: center;">0.9268</td>
<td style="text-align: center;">0.8925</td>
<td style="text-align: center;">0.9162</td>
<td style="text-align: center;">0.9120</td>
<td style="text-align: center;">0.7514</td>
<td style="text-align: center;">0.8289</td>
<td style="text-align: center;">0.5624</td>
<td style="text-align: center;">0.6196</td>
<td style="text-align: center;">0.7922</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">LPIPS</td>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">0.2441</td>
<td style="text-align: center;">0.1799</td>
<td style="text-align: center;">0.1839</td>
<td style="text-align: center;">0.1164</td>
<td style="text-align: center;">0.1978</td>
<td style="text-align: center;">0.2423</td>
<td style="text-align: center;">0.1224</td>
<td style="text-align: center;">0.3601</td>
<td style="text-align: center;">0.3469</td>
<td style="text-align: center;">0.2215</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">STE mask</td>
<td style="text-align: center;">0.2937</td>
<td style="text-align: center;">0.2194</td>
<td style="text-align: center;">0.2274</td>
<td style="text-align: center;">0.1480</td>
<td style="text-align: center;">0.2334</td>
<td style="text-align: center;">0.2899</td>
<td style="text-align: center;">0.1771</td>
<td style="text-align: center;">0.3988</td>
<td style="text-align: center;">0.3983</td>
<td style="text-align: center;">0.2651</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">Pruning Ratio</td>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">0.64</td>
<td style="text-align: center;">0.65</td>
<td style="text-align: center;">0.66</td>
<td style="text-align: center;">0.58</td>
<td style="text-align: center;">0.74</td>
<td style="text-align: center;">0.65</td>
<td style="text-align: center;">0.59</td>
<td style="text-align: center;">0.59</td>
<td style="text-align: center;">0.59</td>
<td style="text-align: center;">0.63</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">STE mask</td>
<td style="text-align: center;">0.84</td>
<td style="text-align: center;">0.88</td>
<td style="text-align: center;">0.88</td>
<td style="text-align: center;">0.87</td>
<td style="text-align: center;">0.89</td>
<td style="text-align: center;">0.86</td>
<td style="text-align: center;">0.85</td>
<td style="text-align: center;">0.83</td>
<td style="text-align: center;">0.83</td>
<td style="text-align: center;">0.86</td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

</div>

Table <a href="#tab:ste_vs_gumbel_radsplat" data-reference-type="ref" data-reference="tab:ste_vs_gumbel_radsplat">4</a> shows that under the same settings, after applying the mask to the importance score, the STE mask compresses the mode too much and the performance drops a lot. Trainable mask keeps the gradient of the mask and the comressed model has a more reasonable size.

<div class="adjustbox" markdown="1">

max width=

<div id="tab:ste_vs_gumbel_minisplat" markdown="1">

<table>
<caption>Results using LP-3DGS and STE mask on importance score of Mini-Splatting</caption>
<thead>
<tr>
<th style="text-align: center;"></th>
<th style="text-align: center;">Scene</th>
<th style="text-align: center;">Bicycle</th>
<th style="text-align: center;">Bonsai</th>
<th style="text-align: center;">Counter</th>
<th style="text-align: center;">Kitchen</th>
<th style="text-align: center;">Room</th>
<th style="text-align: center;">Stump</th>
<th style="text-align: center;">Garden</th>
<th style="text-align: center;">Flowers</th>
<th style="text-align: center;">Treehill</th>
<th style="text-align: center;">AVG</th>
<th style="text-align: center;"></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;">PSNR</td>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">24.906</td>
<td style="text-align: center;">31.370</td>
<td style="text-align: center;">28.4098</td>
<td style="text-align: center;">30.785</td>
<td style="text-align: center;">31.132</td>
<td style="text-align: center;">26.679</td>
<td style="text-align: center;">27.095</td>
<td style="text-align: center;">21.150</td>
<td style="text-align: center;">22.522</td>
<td style="text-align: center;">27.12</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">STE mask</td>
<td style="text-align: center;">24.894</td>
<td style="text-align: center;">30.925</td>
<td style="text-align: center;">28.334</td>
<td style="text-align: center;">30.731</td>
<td style="text-align: center;">31.032</td>
<td style="text-align: center;">26.470</td>
<td style="text-align: center;">26.863</td>
<td style="text-align: center;">20.997</td>
<td style="text-align: center;">22.559</td>
<td style="text-align: center;">26.98</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">SSIM</td>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">0.7373</td>
<td style="text-align: center;">0.9358</td>
<td style="text-align: center;">0.9017</td>
<td style="text-align: center;">0.9249</td>
<td style="text-align: center;">0.9167</td>
<td style="text-align: center;">0.7677</td>
<td style="text-align: center;">0.8493</td>
<td style="text-align: center;">0.5756</td>
<td style="text-align: center;">0.6336</td>
<td style="text-align: center;">0.8047</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">STE mask</td>
<td style="text-align: center;">0.7287</td>
<td style="text-align: center;">0.9292</td>
<td style="text-align: center;">0.8978</td>
<td style="text-align: center;">0.9232</td>
<td style="text-align: center;">0.9152</td>
<td style="text-align: center;">0.7562</td>
<td style="text-align: center;">0.8381</td>
<td style="text-align: center;">0.5629</td>
<td style="text-align: center;">0.6247</td>
<td style="text-align: center;">0.7973</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">LPIPS</td>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">0.2642</td>
<td style="text-align: center;">0.2036</td>
<td style="text-align: center;">0.2068</td>
<td style="text-align: center;">0.1292</td>
<td style="text-align: center;">0.2208</td>
<td style="text-align: center;">0.2553</td>
<td style="text-align: center;">0.1353</td>
<td style="text-align: center;">0.3753</td>
<td style="text-align: center;">0.3618</td>
<td style="text-align: center;">0.2391</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">STE mask</td>
<td style="text-align: center;">0.2821</td>
<td style="text-align: center;">0.2154</td>
<td style="text-align: center;">0.2145</td>
<td style="text-align: center;">0.1330</td>
<td style="text-align: center;">0.2241</td>
<td style="text-align: center;">0.2797</td>
<td style="text-align: center;">0.1564</td>
<td style="text-align: center;">0.3939</td>
<td style="text-align: center;">0.3852</td>
<td style="text-align: center;">0.2538</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">Pruning Ratio</td>
<td style="text-align: center;">LP-3DGS</td>
<td style="text-align: center;">0.57</td>
<td style="text-align: center;">0.67</td>
<td style="text-align: center;">0.64</td>
<td style="text-align: center;">0.56</td>
<td style="text-align: center;">0.71</td>
<td style="text-align: center;">0.61</td>
<td style="text-align: center;">0.60</td>
<td style="text-align: center;">0.54</td>
<td style="text-align: center;">0.54</td>
<td style="text-align: center;">0.60</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">STE mask</td>
<td style="text-align: center;">0.75</td>
<td style="text-align: center;">0.77</td>
<td style="text-align: center;">0.75</td>
<td style="text-align: center;">0.66</td>
<td style="text-align: center;">0.79</td>
<td style="text-align: center;">0.80</td>
<td style="text-align: center;">0.75</td>
<td style="text-align: center;">0.75</td>
<td style="text-align: center;">0.75</td>
<td style="text-align: center;">0.75</td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

</div>

# Discussion and Conclusion [sec:conclusion]

#### Broader Impact and Limitation

LP-3DGS compresses the 3DGS model to an ideal size in a single run, saving storage and computational resources by eliminating the need for parameter sweeping to find the optimal pruning ratio. However, the limitation of this work is that the rendering quality after pruning varies depending on the definition of importance scores.

#### Conclusion

In this paper, we present a novel framework, LP-3DGS, which guide the 3DGS model learn the best model size. The framework applies a trainable mask on the importance score of the gaussian points. The mask only would be trained for a certain period and prune the model once. Our method compressed the model as much as possible without significantly sacrificing performance and is able to achieve the optimal compression rate for different test scenes. Compared with STE mask method, ours achieves better performance.

<div class="ack" markdown="1">

This research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via IARPA R&D Contract No. 140D0423C0076. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon.

</div>

# References [references]

<div class="thebibliography" markdown="1">

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng Nerf: Representing scenes as neural radiance fields for view synthesis *Communications of the ACM*, 65 (1): 99–106, 2021. **Abstract:** We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an under- lying continuous volumetric scene function using a sparse set of input views. Our algorithm represents a scene using a fully-connected (non- convolutional) deep network, whose input is a single continuous 5D coor- dinate (spatial location ( x;y;z ) and viewing direction ( ; )) and whose output is the volume density and view-dependent emitted radiance at that spatial location. We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally di erentiable, the only input required to optimize our repre- sentation is a set of images with known camera poses. We describe how to e ectively optimize neural radiance elds to render photorealistic novel views of scenes with complicated geometry and appearance, and demon- strate results that outperform prior work on neural rendering and view synthesis. View synthesis results are best viewed as videos, so we urge readers to view our supplementary video for convincing comparisons. (@mildenhall2021nerf)

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis 3d gaussian splatting for real-time radiance field rendering *ACM Transactions on Graphics*, 42 (4): 1–14, 2023. **Abstract:** Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (≥ 30 fps) novel-view synthesis at 1080p resolution. First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space; Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene; Third, we develop a fast visibility-aware rendering algorithm that supports anisotropic splatting and both accelerates training and allows realtime rendering. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets. (@kerbl20233d)

Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps *arXiv preprint arXiv:2311.17245*, 2023. **Abstract:** Recent advances in real-time neural rendering using point-based techniques have enabled broader adoption of 3D representations. However, foundational approaches like 3D Gaussian Splatting impose substantial storage overhead, as Structure-from-Motion (SfM) points can grow to millions, often requiring gigabyte-level disk space for a single unbounded scene. This growth presents scalability challenges and hinders splatting efficiency. To address this, we introduce LightGaussian, a method for transforming 3D Gaussians into a more compact format. Inspired by Network Pruning, LightGaussian identifies Gaussians with minimal global significance on scene reconstruction, and applies a pruning and recovery process to reduce redundancy while preserving visual quality. Knowledge distillation and pseudo-view augmentation then transfer spherical harmonic coefficients to a lower degree, yielding compact representations. Gaussian Vector Quantization, based on each Gaussian’s global significance, further lowers bitwidth with minimal accuracy loss. LightGaussian achieves an average 15x compression rate while boosting FPS from 144 to 237 within the 3D-GS framework, enabling efficient complex scene representation on the Mip-NeRF 360 and Tank & Temple datasets. The proposed Gaussian pruning approach is also adaptable to other 3D representations (e.g., Scaffold-GS), demonstrating strong generalization capabilities. (@fan2023lightgaussian)

Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari Radsplat: Radiance field-informed gaussian splatting for robust real-time rendering with 900+ fps *arXiv preprint arXiv:2403.13806*, 2024. **Abstract:** Recent advances in view synthesis and real-time rendering have achieved photorealistic quality at impressive rendering speeds. While Radiance Field-based methods achieve state-of-the-art quality in challenging scenarios such as in-the-wild captures and large-scale scenes, they often suffer from excessively high compute requirements linked to volumetric rendering. Gaussian Splatting-based methods, on the other hand, rely on rasterization and naturally achieve real-time rendering but suffer from brittle optimization heuristics that underperform on more challenging scenes. In this work, we present RadSplat, a lightweight method for robust real-time rendering of complex scenes. Our main contributions are threefold. First, we use radiance fields as a prior and supervision signal for optimizing point-based scene representations, leading to improved quality and more robust optimization. Next, we develop a novel pruning technique reducing the overall point count while maintaining high quality, leading to smaller and more compact scene representations with faster inference speeds. Finally, we propose a novel test-time filtering approach that further accelerates rendering and allows to scale to larger, house-sized scenes. We find that our method enables state-of-the-art synthesis of complex captures at 900+ FPS. (@niemeyer2024radsplat)

Guangchi Fang and Bing Wang Mini-splatting: Representing scenes with a constrained number of gaussians *arXiv preprint arXiv:2403.14166*, 2024. **Abstract:** In this study, we explore the challenge of efficiently representing scenes with a constrained number of Gaussians. Our analysis shifts from traditional graphics and 2D computer vision to the perspective of point clouds, highlighting the inefficient spatial distribution of Gaussian representation as a key limitation in model performance. To address this, we introduce strategies for densification including blur split and depth reinitialization, and simplification through intersection preserving and sampling. These techniques reorganize the spatial positions of the Gaussians, resulting in significant improvements across various datasets and benchmarks in terms of rendering quality, resource consumption, and storage compression. Our Mini-Splatting integrates seamlessly with the original rasterization pipeline, providing a strong baseline for future research in Gaussian-Splatting-based works. \\}href{https://github.com/fatPeter/mini-splatting}{Code is available}. (@fang2024mini)

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman Mip-nerf 360: Unbounded anti-aliased neural radiance fields In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5470–5479, 2022. **Abstract:** Though neural radiance fields (NeRF) have demon-strated impressive view synthesis results on objects and small bounded regions of space, they struggle on “un-bounded” scenes, where the camera may point in any di-rection and content may exist at any distance. In this set-ting, existing NeRF-like models often produce blurry or low-resolution renderings (due to the unbalanced detail and scale of nearby and distant objects), are slow to train, and may exhibit artifacts due to the inherent ambiguity of the task of reconstructing a large scene from a small set of images. We present an extension of mip-NeRF (a NeRF variant that addresses sampling and aliasing) that uses a non-linear scene parameterization, online distillation, and a novel distortion-based regularizer to overcome the chal-lenges presented by unbounded scenes. Our model, which we dub “mip-NeRF 360” as we target scenes in which the camera rotates 360 degrees around a point, reduces mean-squared error by 57% compared to mip-NeRF, and is able to produce realistic synthesized views and detailed depth maps for highly intricate, unbounded real-world scenes. (@barron2022mip)

Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park Compact 3d gaussian representation for radiance field *arXiv preprint arXiv:2311.13681*, 2023. **Abstract:** Neural Radiance Fields (NeRFs) have demonstrated remarkable potential in capturing complex 3D scenes with high fidelity. However, one persistent challenge that hinders the widespread adoption of NeRFs is the computational bottleneck due to the volumetric rendering. On the other hand, 3D Gaussian splatting (3DGS) has recently emerged as an alternative representation that leverages a 3D Gaussisan-based representation and adopts the rasterization pipeline to render the images rather than volumetric rendering, achieving very fast rendering speed and promising image quality. However, a significant drawback arises as 3DGS entails a substantial number of 3D Gaussians to maintain the high fidelity of the rendered images, which requires a large amount of memory and storage. To address this critical issue, we place a specific emphasis on two key objectives: reducing the number of Gaussian points without sacrificing performance and compressing the Gaussian attributes, such as view-dependent color and covariance. To this end, we propose a learnable mask strategy that significantly reduces the number of Gaussians while preserving high performance. In addition, we propose a compact but effective representation of view-dependent color by employing a grid-based neural field rather than relying on spherical harmonics. Finally, we learn codebooks to compactly represent the geometric attributes of Gaussian by vector quantization. With model compression techniques such as quantization and entropy coding, we consistently show over 25$\\}times$ reduced storage and enhanced rendering speed, while maintaining the quality of the scene representation, compared to 3DGS. Our work provides a comprehensive framework for 3D scene representation, achieving high performance, fast training, compactness, and real-time rendering. Our project page is available at https://maincold2.github.io/c3dgs/. (@lee2023compact)

Yoshua Bengio, Nicholas Léonard, and Aaron Courville Estimating or propagating gradients through stochastic neurons for conditional computation *arXiv preprint arXiv:1308.3432*, 2013. **Abstract:** Stochastic neurons and hard non-linearities can be useful for a number of reasons in deep learning models, but in many cases they pose a challenging problem: how to estimate the gradient of a loss function with respect to the input of such stochastic or non-smooth neurons? I.e., can we "back-propagate" through these stochastic neurons? We examine this question, existing approaches, and compare four families of solutions, applicable in different settings. One of them is the minimum variance unbiased gradient estimator for stochatic binary neurons (a special case of the REINFORCE algorithm). A second approach, introduced here, decomposes the operation of a binary stochastic neuron into a stochastic binary part and a smooth differentiable part, which approximates the expected effect of the pure stochatic binary neuron to first order. A third approach involves the injection of additive or multiplicative noise in a computational graph that is otherwise differentiable. A fourth approach heuristically copies the gradient with respect to the stochastic output directly as an estimator of the gradient with respect to the sigmoid argument (we call this the straight-through estimator). To explore a context where these estimators are useful, we consider a small-scale version of {\\}em conditional computation}, where sparse stochastic units form a distributed representation of gaters that can turn off in combinatorially many ways large chunks of the computation performed in the rest of the neural network. In this case, it is important that the gating units produce an actual 0 most of the time. The resulting sparsity can be potentially be exploited to greatly reduce the computational cost of large deep networks for which conditional computation would be useful. (@bengio2013estimating)

Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun Tanks and temples: Benchmarking large-scale scene reconstruction *ACM Transactions on Graphics (ToG)*, 36 (4): 1–13, 2017. (@knapitsch2017tanks)

Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5855–5864, 2021. **Abstract:** The rendering procedure used by neural radiance fields (NeRF) samples a scene with a single ray per pixel and may therefore produce renderings that are excessively blurred or aliased when training or testing images observe scene content at different resolutions. The straightforward solution of supersampling by rendering with multiple rays per pixel is impractical for NeRF, because rendering each ray requires querying a multilayer perceptron hundreds of times. Our solution, which we call "mip-NeRF" (à la "mipmap"), extends NeRF to represent the scene at a continuously-valued scale. By efficiently rendering anti-aliased conical frustums instead of rays, mip-NeRF reduces objectionable aliasing artifacts and significantly improves NeRF’s ability to represent fine details, while also being 7% faster than NeRF and half the size. Compared to NeRF, mip-NeRF reduces average error rates by 17% on the dataset presented with NeRF and by 60% on a challenging multiscale variant of that dataset that we present. Mip-NeRF is also able to match the accuracy of a brute-force supersampled NeRF on our multiscale dataset while being 22× faster. (@barron2021mip)

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller Instant neural graphics primitives with a multiresolution hash encoding *ACM transactions on graphics (TOG)*, 41 (4): 1–15, 2022. **Abstract:** Neural graphics primitives, parameterized by fully connected neural networks, can be costly to train and evaluate. We reduce this cost with a versatile new input encoding that permits the use of a smaller network without sacrificing quality, thus significantly reducing the number of floating point and memory access operations: a small neural network is augmented by a multiresolution hash table of trainable feature vectors whose values are optimized through stochastic gradient descent. The multiresolution structure allows the network to disambiguate hash collisions, making for a simple architecture that is trivial to parallelize on modern GPUs. We leverage this parallelism by implementing the whole system using fully-fused CUDA kernels with a focus on minimizing wasted bandwidth and compute operations. We achieve a combined speedup of several orders of magnitude, enabling training of high-quality neural graphics primitives in a matter of seconds, and rendering in tens of milliseconds at a resolution of ${1920\\}!\\}times\\}!1080}$. (@muller2022instant)

Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su Tensorf: Tensorial radiance fields In *European Conference on Computer Vision*, pages 333–350. Springer, 2022. **Abstract:** We present TensoRF, a novel approach to model and recon- struct radiance fields. Unlike NeRF that purely uses MLPs, we model the radiance field of a scene as a 4D tensor, which represents a 3D voxel grid with per-voxel multi-channel features. Our central idea is to factorize the 4D scene tensor into multiple compact low-rank tensor components. We demonstrate that applying traditional CANDECOMP/PARAFAC (CP) decomposition – that factorizes tensors into rank-one components with compact vectors – in our framework leads to improvements over vanilla NeRF. To further boost performance, we introduce a novel vector-matrix (VM) decomposition that relaxes the low-rank constraints for two modes of a tensor and factorizes tensors into compact vector and matrix factors. Beyond superior rendering quality, our models with CP and VM decom- positions lead to a significantly lower memory footprint in comparison to previous and concurrent works that directly optimize per-voxel features. Experimentally, we demonstrate that TensoRF with CP decomposition achieves fast reconstruction ( \<30 min) with better rendering quality and even a smaller model size ( \<4 MB) compared to NeRF. Moreover, TensoRF with VM decomposition further boosts rendering quality and outperforms previous state-of-the-art methods, while reducing the recon- struction time ( \<10 min) and retaining a compact model size ( \<75 MB). 1 Introduction Modeling and reconstructing 3D scenes as representations that support high- quality image synthesis is crucial for computer vision and graphics with various applications in visual effects, e-commerce, virtual and augmented reality, and robotics. Recently, NeRF \[ 37\] and its many follow-up works \[ 70,31\] have shown success on modeling a scene as a radiance field and enabled photo-realistic rendering of scenes with highly complex geometry and view-dependent appearance effects. Despite the fact that (purely MLP-based) NeRF models require small memory, they take a long time (hours or days) to train. In this work, we pursue ⋆Equal contribution. Research done when Anpei Chen was in a remote internship with UCSD.arXiv:2203.09517v2 \[cs.CV\] 29 Nov 20222 A. Chen, Z. Xu et al. TensoRF-VM Scene NeRFPlenOctrees Quantitative Results on the Synthetic NeRF Dataset Plenoxels OursNeRF 31.01PSNR Method (Point sizes correspond to PNSRs)30.38 31.71 31.71 31.95 31.56 32.52 33.14PlenOctrees Plenoxels DVGO Ours-CP-384-30k Ours-VM-192-15k Ours-VM-192-30kSNeRG DVGOSNeRG Fig. 1: Left: We model (@chen2022tensorf)

Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa Plenoxels: Radiance fields without neural networks In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5501–5510, 2022. **Abstract:** We introduce Plenoxels (plenoptic voxels), a systemfor photorealistic view synthesis. Plenoxels represent a scene as a sparse 3D grid with spherical harmonics. This representation can be optimized from calibrated images via gradient methods and regularization without any neural components. On standard, benchmark tasks, Plenoxels are optimized two orders of magnitude faster than Neural Radiance Fields with no loss in visual quality. For video and code, please see https://alexyu.net/plenoxels. (@fridovich2022plenoxels)

Markus Gross and Hanspeter Pfister *Point-based graphics* Elsevier, 2011. **Abstract:** We present a new point-based approach for modeling the appearance of real scenes. The approach uses a raw point cloud as the geometric representation of a scene, and augments each point with a learnable neural descriptor that encodes local geometry and appearance. A deep rendering network is learned in parallel with the descriptors, so that new views of the scene can be obtained by passing the rasteriza- tions of a point cloud from new viewpoints through this network. The input rasterizations use the learned descriptors as point pseudo-colors. We show that the proposed approach can be used for modeling complex scenes and obtaining their photorealistic views, while avoiding explicit surface estimation and meshing. In particular, compelling results are ob- tained for scene scanned using hand-held commodity RGB-D sensors as well as standard RGB cameras even in the presence of objects that are challenging for standard mesh-based modeling. (@gross2011point)

Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lempitsky Neural point-based graphics In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXII 16*, pages 696–712. Springer, 2020. **Abstract:** We present a new point-based approach for modeling the appearance of real scenes. The approach uses a raw point cloud as the geometric representation of a scene, and augments each point with a learnable neural descriptor that encodes local geometry and appearance. A deep rendering network is learned in parallel with the descriptors, so that new views of the scene can be obtained by passing the rasteriza- tions of a point cloud from new viewpoints through this network. The input rasterizations use the learned descriptors as point pseudo-colors. We show that the proposed approach can be used for modeling complex scenes and obtaining their photorealistic views, while avoiding explicit surface estimation and meshing. In particular, compelling results are ob- tained for scene scanned using hand-held commodity RGB-D sensors as well as standard RGB cameras even in the presence of objects that are challenging for standard mesh-based modeling. (@aliev2020neural)

Yuhan Ding, Fukun Yin, Jiayuan Fan, Hui Li, Xin Chen, Wen Liu, Chongshan Lu, Gang Yu, and Tao Chen Point diffusion implicit function for large-scale scene neural representation *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Recent advances in implicit neural representations have achieved impressive results by sampling and fusing individual points along sampling rays in the sampling space. However, due to the explosively growing sampling space, finely representing and synthesizing detailed textures remains a challenge for unbounded large-scale outdoor scenes. To alleviate the dilemma of using individual points to perceive the entire colossal space, we explore learning the surface distribution of the scene to provide structural priors and reduce the samplable space and propose a Point Diffusion implicit Function, PDF, for large-scale scene neural representation. The core of our method is a large-scale point cloud super-resolution diffusion module that enhances the sparse point cloud reconstructed from several training images into a dense point cloud as an explicit prior. Then in the rendering stage, only sampling points with prior points within the sampling radius are retained. That is, the sampling space is reduced from the unbounded space to the scene surface. Meanwhile, to fill in the background of the scene that cannot be provided by point clouds, the region sampling based on Mip-NeRF 360 is employed to model the background representation. Expensive experiments have demonstrated the effectiveness of our method for large-scale scene novel view synthesis, which outperforms relevant state-of-the-art baselines. (@ding2024point)

Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee Depth-regularized optimization for 3d gaussian splatting in few-shot images *arXiv preprint arXiv:2311.13398*, 2023. **Abstract:** In this paper, we present a method to optimize Gaussian splatting with a limited number of images while avoiding overfitting. Representing a 3D scene by combining numerous Gaussian splats has yielded outstanding visual quality. However, it tends to overfit the training views when only a small number of images are available. To address this issue, we introduce a dense depth map as a geometry guide to mitigate overfitting. We obtained the depth map using a pre-trained monocular depth estimation model and aligning the scale and offset using sparse COLMAP feature points. The adjusted depth aids in the color-based optimization of 3D Gaussian splatting, mitigating floating artifacts, and ensuring adherence to geometric constraints. We verify the proposed method on the NeRF-LLFF dataset with varying numbers of few images. Our approach demonstrates robust geometry compared to the original method that relies solely on images. Project page: robot0321.github.io/DepthRegGS (@chung2023depth)

Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization *arXiv preprint arXiv:2403.06912*, 2024. **Abstract:** Radiance fields have demonstrated impressive performance in synthesizing novel views from sparse input views, yet prevailing methods suffer from high training costs and slow inference speed. This paper introduces DNGaussian, a depth-regularized framework based on 3D Gaussian radiance fields, offering real-time and high-quality few-shot novel view synthesis at low costs. Our motivation stems from the highly efficient representation and surprising quality of the recent 3D Gaussian Splatting, despite it will encounter a geometry degradation when input views decrease. In the Gaussian radiance fields, we find this degradation in scene geometry primarily lined to the positioning of Gaussian primitives and can be mitigated by depth constraint. Consequently, we propose a Hard and Soft Depth Regularization to restore accurate scene geometry under coarse monocular depth supervision while maintaining a fine-grained color appearance. To further refine detailed geometry reshaping, we introduce Global-Local Depth Normalization, enhancing the focus on small local depth changes. Extensive experiments on LLFF, DTU, and Blender datasets demonstrate that DNGaussian outperforms state-of-the-art methods, achieving comparable or better results with significantly reduced memory cost, a $25 \\}times$ reduction in training time, and over $3000 \\}times$ faster rendering speed. (@li2024dngaussian)

Antoine Guédon and Vincent Lepetit Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering *arXiv preprint arXiv:2311.12775*, 2023. **Abstract:** We propose a method to allow precise and extremely fast mesh extraction from 3D Gaussian Splatting. Gaussian Splatting has recently become very popular as it yields realistic rendering while being significantly faster to train than NeRFs. It is however challenging to extract a mesh from the millions of tiny 3D gaussians as these gaussians tend to be unorganized after optimization and no method has been proposed so far. Our first key contribution is a regularization term that encourages the gaussians to align well with the surface of the scene. We then introduce a method that exploits this alignment to extract a mesh from the Gaussians using Poisson reconstruction, which is fast, scalable, and preserves details, in contrast to the Marching Cubes algorithm usually applied to extract meshes from Neural SDFs. Finally, we introduce an optional refinement strategy that binds gaussians to the surface of the mesh, and jointly optimizes these Gaussians and the mesh through Gaussian splatting rendering. This enables easy editing, sculpting, rigging, animating, compositing and relighting of the Gaussians using traditional softwares by manipulating the mesh instead of the gaussians themselves. Retrieving such an editable mesh for realistic rendering is done within minutes with our method, compared to hours with the state-of-the-art methods on neural SDFs, while providing a better rendering quality. Our project page is the following: https://anttwo.github.io/sugar/ (@guedon2023sugar)

Yanyan Li, Chenyu Lyu, Yan Di, Guangyao Zhai, Gim Hee Lee, and Federico Tombari Geogaussian: Geometry-aware gaussian splatting for scene rendering *arXiv preprint arXiv:2403.11324*, 2024. **Abstract:** During the Gaussian Splatting optimization process, the scene’s geometry can gradually deteriorate if its structure is not deliberately preserved, especially in non-textured regions such as walls, ceilings, and furniture surfaces. This degradation significantly affects the rendering quality of novel views that deviate significantly from the viewpoints in the training data. To mitigate this issue, we propose a novel approach called GeoGaussian. Based on the smoothly connected areas observed from point clouds, this method introduces a novel pipeline to initialize thin Gaussians aligned with the surfaces, where the characteristic can be transferred to new generations through a carefully designed densification strategy. Finally, the pipeline ensures that the scene’s geometry and texture are maintained through constrained optimization processes with explicit geometry constraints. Benefiting from the proposed architecture, the generative ability of 3D Gaussians is enhanced, especially in structured regions. Our proposed pipeline achieves state-of-the-art performance in novel view synthesis and geometric reconstruction, as evaluated qualitatively and quantitatively on public datasets. (@li2024geogaussian)

Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, and Eric Xing Fregs: 3d gaussian splatting with progressive frequency regularization *arXiv preprint arXiv:2403.06908*, 2024. **Abstract:** 3D Gaussian splatting has achieved very impressive performance in real-time novel view synthesis. However, it often suffers from over-reconstruction during Gaussian densification where high-variance image regions are covered by a few large Gaussians only, leading to blur and artifacts in the rendered images. We design a progressive frequency regularization (FreGS) technique to tackle the over-reconstruction issue within the frequency space. Specifically, FreGS performs coarse-to-fine Gaussian densification by exploiting low-to-high frequency components that can be easily extracted with low-pass and high-pass filters in the Fourier space. By minimizing the discrepancy between the frequency spectrum of the rendered image and the corresponding ground truth, it achieves high-quality Gaussian densification and alleviates the over-reconstruction of Gaussian splatting effectively. Experiments over multiple widely adopted benchmarks (e.g., Mip-NeRF360, Tanks-and-Temples and Deep Blending) show that FreGS achieves superior novel view synthesis and outperforms the state-of-the-art consistently. (@zhang2024fregs)

Mario Botsch, Alexander Hornung, Matthias Zwicker, and Leif Kobbelt High-quality surface splatting on today’s gpus In *Proceedings Eurographics/IEEE VGTC Symposium Point-Based Graphics, 2005.*, pages 17–141. IEEE, 2005. **Abstract:** Point-based geometries evolved into a valuable alternative to surface representations based on polygonal meshes, because of their conceptual simplicity and superior flexibility. Elliptical surface splats were shown to allow for high-quality anti-aliased rendering by sophisticated EWA filtering. Since the publication of the original software-based EWA splatting, several authors tried to map this technique to the GPU in order to exploit hardware acceleration. Due to the lacking support for splat primitives, these methods always have to find a trade-off between rendering quality and rendering performance. In this paper, we discuss the capabilities of today’s GPUs for hardware-accelerated surface splatting. We present an approach that achieves a quality comparable to the original EWA splatting at a rate of more than 20M elliptical splats per second. In contrast to previous GPU renderers, our method provides per-pixel Phong shading even for dynamically changing geometries and high-quality anti-aliasing by employing a screen-space pre-filter in addition to the object-space reconstruction filter. The use of deferred shading techniques effectively avoids unnecessary shader computations and additionally provides a clear separation between the rasterization and the shading of elliptical splats, which considerably simplifies the development of custom shaders. We demonstrate quality, efficiency, and flexibility of our approach by showing several shaders on a range of models. (@botsch2005high)

Emil Julius Gumbel *Statistical theory of extreme values and some practical applications: a series of lectures*, volume 33 US Government Printing Office, 1954. (@gumbel1954statistical)

Eric Jang, Shixiang Gu, and Ben Poole Categorical reparameterization with gumbel-softmax *arXiv preprint arXiv:1611.01144*, 2016. **Abstract:** Categorical variables are a natural choice for representing discrete structure in the world. However, stochastic neural networks rarely use categorical latent variables due to the inability to backpropagate through samples. In this work, we present an efficient gradient estimator that replaces the non-differentiable sample from a categorical distribution with a differentiable sample from a novel Gumbel-Softmax distribution. This distribution has the essential property that it can be smoothly annealed into a categorical distribution. We show that our Gumbel-Softmax estimator outperforms state-of-the-art gradient estimators on structured output prediction and unsupervised generative modeling tasks with categorical latent variables, and enables large speedups on semi-supervised classification. (@jang2016categorical)

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang The unreasonable effectiveness of deep features as a perceptual metric In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 586–595, 2018. **Abstract:** While it is nearly effortless for humans to quickly assess the perceptual similarity between two images, the underlying processes are thought to be quite complex. Despite this, the most widely used perceptual metrics today, such as PSNR and SSIM, are simple, shallow functions, and fail to account for many nuances of human perception. Recently, the deep learning community has found that features of the VGG network trained on ImageNet classification has been remarkably useful as a training loss for image synthesis. But how perceptual are these so-called "perceptual losses"? What elements are critical for their success? To answer these questions, we introduce a new dataset of human perceptual similarity judgments. We systematically evaluate deep features across different architectures and tasks and compare them with classic metrics. We find that deep features outperform all previous metrics by large margins on our dataset. More surprisingly, this result is not restricted to ImageNet-trained VGG features, but holds across different deep architectures and levels of supervision (supervised, self-supervised, or even unsupervised). Our results suggest that perceptual similarity is an emergent property shared across deep visual representations. (@zhang2018unreasonable)

</div>

# Appendix / supplemental material [appendix]

*Code:* <https://github.com/dexgfsdfdsg/LP-3DGS.git>

## Experiment Results on MipNeRF360 Dataset

<figure>
<figure>
<img src="./figures/bicycle_gt.png"" />
</figure>
<figure>
<img src="./figures/bicycle_baseline.png"" />
</figure>
<figure>
<img src="./figures/bicycle_radsplat.png"" />
</figure>
<figure>
<img src="./figures/bicycle_minisplat.png"" />
</figure>
<figure>
<img src="./figures/bonsai_gt.png"" />
</figure>
<figure>
<img src="./figures/bonsai_baseline.png"" />
</figure>
<figure>
<img src="./figures/bonsai_radsplat.png"" />
</figure>
<figure>
<img src="./figures/bonsai_minisplat.png"" />
</figure>
<figure>
<img src="./figures/counter_gt.png"" />
</figure>
<figure>
<img src="./figures/counter_baseline.png"" />
</figure>
<figure>
<img src="./figures/counter_radsplat.png"" />
</figure>
<figure>
<img src="./figures/counter_minisplat.png"" />
</figure>
<figure>
<img src="./figures/kitchen_gt.png"" />
</figure>
<figure>
<img src="./figures/kitchen_baseline.png"" />
</figure>
<figure>
<img src="./figures/kitchen_radsplat.png"" />
</figure>
<figure>
<img src="./figures/kitchen_minisplat.png"" />
</figure>
<figure>
<img src="./figures/room_gt.png"" />
</figure>
<figure>
<img src="./figures/room_baseline.png"" />
</figure>
<figure>
<img src="./figures/room_radsplat.png"" />
</figure>
<figure>
<img src="./figures/room_minisplat.png"" />
</figure>
<figure>
<img src="./figures/stump_gt.png"" />
</figure>
<figure>
<img src="./figures/stump_baseline.png"" />
</figure>
<figure>
<img src="./figures/stump_radsplat.png"" />
</figure>
<figure>
<img src="./figures/stump_minisplat.png"" />
</figure>
<figure>
<img src="./figures/garden_gt.png"" />
</figure>
<figure>
<img src="./figures/garden_baseline.png"" />
</figure>
<figure>
<img src="./figures/garden_radsplat.png"" />
</figure>
<figure>
<img src="./figures/garden_minisplat.png"" />
</figure>
</figure>

<figure>
<figure>
<img src="./figures/flowers_gt.png"" />
</figure>
<figure>
<img src="./figures/flowers_baseline.png"" />
</figure>
<figure>
<img src="./figures/flowers_radsplat.png"" />
</figure>
<figure>
<img src="./figures/flowers_minisplat.png"" />
</figure>
<figure>
<img src="./figures/treehill_gt.png"" />
</figure>
<figure>
<img src="./figures/treehill_baseline.png"" />
</figure>
<figure>
<img src="./figures/treehill_radsplat.png"" />
</figure>
<figure>
<img src="./figures/treehill_minisplat.png"" />
</figure>
<figcaption>Rendered images on MipNeRF360 Dataset</figcaption>
</figure>

<figure>
<figure>
<img src="./figures/PSNR_Bicycle_radsplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Bicycle_radsplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Bicycle_radsplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Bicycle_minisplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Bicycle_minisplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Bicycle_minisplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Bonsai_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Bonsai_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Bonsai_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Bonsai_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Bonsai_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Bonsai_MiniSplat.png"" />
</figure>
</figure>

<figure id="fig:mip360curve">
<figure>
<img src="./figures/PSNR_Counter_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Counter_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Counter_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Counter_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Counter_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Counter_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Stump_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Stump_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Stump_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Stump_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Stump_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Stump_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Flowers_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Flowers_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Flowers_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Flowers_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Flowers_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Flowers_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Treehill_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Treehill_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Treehill_RadSplat.png"" />
</figure>
<figure>
<img src="./figures/PSNR_Treehill_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/SSIM_Treehill_MiniSplat.png"" />
</figure>
<figure>
<img src="./figures/LPIPS_Treehill_MiniSplat.png"" />
</figure>
<figcaption>The performance changes with the pruning ratio in different scenes</figcaption>
</figure>

## Experiment Results on NeRF Synthetic Dataset

<div class="adjustbox" markdown="1">

max width=

<div id="tab:quantitative_result_nerf_synthetic" markdown="1">

| Scene | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship | AVG |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Baseline PSNR \\(\uparrow\\) | 35.546 | 26.276 | 35.480 | 38.081 | 36.012 | 30.502 | 36.795 | 31.688 | 33.798 |
| LP-3DGS (RadSplat Score) | 35.496 | 26.221 | 35.442 | 37.976 | 35.990 | 30.374 | 36.589 | 31.584 | 33.709 |
| LP-3DGS (Mini-Splatting Score) | 35.419 | 26.102 | 35.354 | 37.728 | 35.769 | 29.883 | 36.337 | 31.375 | 33.496 |
| Baseline SSIM \\(\uparrow\\) | 0.9877 | 0.9548 | 0.9870 | 0.9854 | 0.9825 | 0.9604 | 0.9926 | 0.9062 | 0.9696 |
| LP-3DGS (RadSplat Score) | 0.9878 | 0.9547 | 0.9867 | 0.9854 | 0.9825 | 0.9598 | 0.9924 | 0.9061 | 0.9694 |
| LP-3DGS (Mini-Splatting Score) | 0.9874 | 0.9358 | 0.9867 | 0.9846 | 0.9817 | 0.9566 | 0.9919 | 0.9034 | 0.966 |
| Baseline LPIPS \\(\downarrow\\) | 0.01046 | 0.03657 | 0.01775 | 0.01977 | 0.0161 | 0.03671 | 0.00635 | 0.1058 | 0.03119 |
| LP-3DGS (RadSplat Score) | 0.01091 | 0.03723 | 0.01213 | 0.02079 | 0.01675 | 0.03817 | 0.00680 | 0.1083 | 0.03139 |
| LP-3DGS (Mini-Splatting Score) | 0.0111 | 0.03876 | 0.01217 | 0.02211 | 0.018 | 0.04323 | 0.00749 | 0.1151 | 0.0335 |
| RadSplat Score pruning ratio | 0.77 | 0.76 | 0.84 | 0.68 | 0.65 | 0.61 | 0.78 | 0.60 | 0.71 |
| Mini-Splatting Score pruning ratio | 0.63 | 0.65 | 0.65 | 0.58 | 0.58 | 0.80 | 0.60 | 0.50 | 0.62 |

The quantitative results on NeRF Synthetic Dataset

</div>

</div>

## Experiment Results on Truck & Train Scenes

<div class="adjustbox" markdown="1">

max width=

<div id="tab:quantitative_result_tt" markdown="1">

|               Scene                | Truck  | Train  |  AVG   |
|:----------------------------------:|:------:|:------:|:------:|
|    Baseline PSNR \\(\uparrow\\)    | 25.263 | 22.025 | 23.644 |
|      LP-3DGS (RadSplat Score)      | 25.376 | 21.822 | 23.599 |
|   LP-3DGS (Mini-Splatting Score)   | 25.152 | 21.675 | 23.414 |
|    Baseline SSIM \\(\uparrow\\)    | 0.8778 | 0.8118 | 0.8448 |
|      LP-3DGS (RadSplat Score)      | 0.8768 | 0.8072 | 0.8420 |
|   LP-3DGS (Mini-Splatting Score)   | 0.8724 | 0.7963 | 0.8344 |
|  Baseline LPIPS \\(\downarrow\\)   | 0.1482 | 0.2083 | 0.1783 |
|      LP-3DGS (RadSplat Score)      | 0.1541 | 0.2217 | 0.1879 |
|   LP-3DGS (Mini-Splatting Score)   | 0.162  | 0.2343 | 0.1982 |
|    RadSplat Score pruning ratio    |  0.72  |  0.63  |  0.68  |
| Mini-Splatting Score pruning ratio |  0.65  |  0.57  |  0.61  |

The quantitative results on Tanks & Temples Dataset

</div>

</div>
