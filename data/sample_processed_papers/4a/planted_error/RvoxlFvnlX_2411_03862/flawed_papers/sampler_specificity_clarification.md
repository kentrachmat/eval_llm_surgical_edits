# ROBIN: <u>Rob</u>ust and <u>In</u>visible Watermarks for Diffusion Models with Adversarial Optimization

## Abstract

Watermarking generative content serves as a vital tool for authentication, ownership protection, and mitigation of potential misuse. Existing watermarking methods face the challenge of balancing robustness and concealment. They empirically inject a watermark that is both invisible and robust and *passively* achieve concealment by limiting the strength of the watermark, thus reducing the robustness. In this paper, we propose to explicitly introduce a watermark hiding process to *actively* achieve concealment, thus allowing the embedding of stronger watermarks. Specifically, we implant a robust watermark in an intermediate diffusion state and then guide the model to hide the watermark in the final generated image. We employ an adversarial optimization algorithm to produce the optimal hiding prompt guiding signal for each watermark. The prompt embedding is optimized to minimize artifacts in the generated image, while the watermark is optimized to achieve maximum strength. The watermark can be verified by reversing the deterministic DDIM sampling trajectory, which is natively invertible.

Experiments on both latent‐ and image‐space diffusion models demonstrate that the proposed scheme remains verifiable even under significant image tampering and exhibits superior invisibility compared to state-of-the-art robust watermarking methods. These consistent results across model families confirm the practical generality of our approach. Code is available at <https://github.com/Hannah1102/ROBIN>.
# Introduction [sec:intro]

Diffusion models (DMs) are revolutionizing content creation and generating stunningly realistic imagery across diverse domains `\cite{ho2020denoising,saharia2022photorealistic,zhu2024vision+}`{=latex}. The advent of text-to-image diffusion models `\cite{rombach2022high,ramesh2022hierarchical,zhu2022discrete}`{=latex}, coupled with personalized generation techniques `\cite{zhang2023adding,couairon2022diffedit,ruiz2023dreambooth,gal2022image, wangdiffusion, zhu2024boundary}`{=latex}, enables the creation of highly specific content by virtually anyone. However, it has raised concerns about authenticity and ownership, including the risk of plagiarism `\cite{shan2023glaze,liu2024iterative}`{=latex} and the potential misuse of images of public figures `\cite{van2023anti,chen2023editshield}`{=latex}. Consequently, governments and businesses are increasingly advocating for robust mechanisms to verify the origins of generative content `\cite{whitehouse,microsoft}`{=latex}.

Watermarking offers a proactive approach to authenticate the source of generated content. This technique embeds imperceptible secret messages within the generated content. These messages serve as unique identifiers, confirming the image’s origin while remaining invisible to the human eye. They also need to be robust enough to withstand potential distortions encountered during online sharing.

Existing watermarking techniques face a significant challenge in striking a balance between concealment and robustness. Traditional post-processing methods `\cite{wolfgang1996watermark,cox1996secure}`{=latex} employ an empirical approach to identify an invisible and robust watermark and embed it within the generated image. They *passively* achieve concealment by limiting the watermark strength, consequently compromising robustness. Conversely, stronger watermarks, while enhancing robustness, can introduce visible artifacts into the generated image. Recent advancements in in-processing watermarking for diffusion models expect the generative model to learn this balance and directly produce watermarked content. However, these methods often require expensive model retraining `\cite{zhao2023recipe,xiong2023flexible,ditria2023hey}`{=latex} or can lead to unintended semantic alterations within the generated images `\cite{wen2023tree}`{=latex}.

Our ROBIN scheme introduces an explicit watermark hiding process to *actively* achieve concealment. This approach reduces the invisibility limitation of the watermark itself and thus enables the embedding of more robust watermarks. Specifically, we implant a robust watermark within an intermediate diffusion state, and then directionally guide the model to gradually conceal the implanted watermark, thus achieving invisibility in the final generated image. In this way, robust watermarks can be secretly implanted in the generated content without model retraining.

We focus on the text-to-image diffusion models, which support an additional prompt signal to guide the generation process. We employ an adversarial optimization algorithm to design an optimal prompt guidance signal specifically tailored for each watermark. **The prompt embedding is optimized to minimize artifacts in the generated image, and the watermark is optimized to achieve maximum strength.** The optimized watermark and prompt signal are universally applicable to all images. During the generation process, the watermark is implanted within an intermediate state following the semantic formation stage. Subsequently, the optimized prompt guidance signal is introduced throughout the remaining diffusion steps. After image generation, following previous works `\cite{wen2023tree,yu2024cross}`{=latex}, we reverse the diffusion process to the watermark embedding point to verify the existence of the watermark. This innovative approach offers a promising way to overcome the trade-off between watermark strength and stealth by explicitly introducing an additional watermark hiding process.

In summary, our key contributions are as follows:

- We propose a novel watermarking method for diffusion models that embed a robust watermark and subsequently employ an active hiding process to achieve imperceptibility.

- We develop an adversarial optimization algorithm to generate a prompt signal for watermark hiding and a strong watermark that can be hidden and strategically select the watermarking point within the diffusion trajectory.

- Evaluations on both latent and image diffusion models demonstrate that our scheme exhibits superior robustness against various image manipulations while preserving semantic content.

# Related work

#### Diffusion generation and inversion.

Diffusion models `\cite{ho2020denoising,dhariwal2021diffusion,song2019generative,song2020score}`{=latex} operate by iteratively transforming pure noise \\(x_{T}\sim  \mathcal{N}(0,\mathbf{I} )\\) into increasingly realistic images \\(x_{0}\sim q(x)\\) through \\(T\\) steps of denoising. The learning process involves a stochastic Markov chain in two directions. The forward process diffuses the sample \\(x_0\\) by adding random noise:

\\[q(x_t|x_{t-1})=\mathcal{N}(\sqrt[]{1-\beta_t}x_{t-1},\beta _t\mathbf{I}  ),\\] where \\(\left \{  \beta _{t}\right \} _{t=1}^{T}\\) is the scheduled variance. \\(x_t\\) can also be generated from \\(x_0\\) as:

\\[x_t=\sqrt[]{\bar{\alpha }_t }x_0+\sqrt[]{1-\bar{\alpha }_t }\epsilon,\\] where \\(\bar{\alpha }_t= {\textstyle \prod_{t=1}^{T}(1-\beta_t)}\\) and \\(\epsilon \sim \mathcal{N}(0,\mathbf{I} )\\). Then a network \\(\epsilon_{\theta}\\) is learned to predict the noise in each step, following the objective:

\\[\underset{\theta }{\min}E_{x_0,t\sim \texttt{Uniform}(1,T),\epsilon \sim \mathcal{N} (0,\mathbf{I} )}\left \| \epsilon -\epsilon _{\theta }(x_{t},t,\psi(p)) \right \|_{2}^{2},\\] where \\(x_{t}\\) is the noise latent at timesteps \\(t\\) and \\(\psi(p)\\) is the embedding of the text prompt \\(p\\).

DDIM (Denoising Diffusion Implicit Model) `\cite{song2020denoising}`{=latex} introduces the ODE solver for deterministic sampling by constructing the original one as a non-Markov process. It computes the \\(x_{t-1}\\) from \\(x_{t}\\) by predicting the estimation of \\(x_{0}\\) and the direction pointing to \\(x_{t}\\):

\\[x_{0}'=\frac{x_t-\sqrt{1-\bar{\alpha}_t } \epsilon _\theta (x_t,t,\psi(p))}{\sqrt{\bar{\alpha} _t }  } ,
    \label{eq:prex0}\\]

\\[x_{t-1}=\sqrt{\bar{\alpha} _{t-1} }x_0'+\sqrt{1-\bar{\alpha }_{t-1} }\epsilon _{\theta }(x_t,t,\psi(p)).\\] The deterministic generation properties of DDIM allow it to reconstruct the noise latent \\(\hat{x}_{t}\\) from the final image \\(x_0\\) as :

\\[\hat{x}_t=\sqrt[]{\bar{\alpha }_t}x_0+\sqrt[]{1-\bar{\alpha }_t }\epsilon _{\theta }(x_{t-1},t-1).\\] This unique characteristic allows us to selectively mark and recover an inner noise representation within the diffusion process, which serves as a powerful tool for our watermarking approach.

#### Watermarking generative models. 

The content watermark of generative models can be introduced either after the generation (post-processing) or during the sampling process (in-processing). Post-processing methods can adopt traditional digital image watermarking technology. Popular methods include frequency domain watermarking, which modifies the image representation in domains like Discrete Wavelet Transform (DWT) `\cite{xia1998wavelet}`{=latex} or Discrete Cosine Transform (DCT) `\cite{cox2007digital}`{=latex}. DwtDct watermarking `\cite{al2007combined}`{=latex} is applied in open sourced model Stable Diffusion. Frequency domain watermarks can be designed to be robust against common image manipulations like cropping, scaling, and even compression `\cite{urvoy2014perceptual}`{=latex}. HiDDeN `\cite{zhu2018hidden}`{=latex} pioneered the end-to-end approach, utilizing an encoder-decoder architecture to directly generate watermarked images. RivaGAN `\cite{zhang2019robust}`{=latex} leverages adversarial training to incorporate perturbations and image processing during model training for increased robustness.

In-processing methods make the watermark become part of the generated image by interfering with the generation process. Early approaches explored adding watermarks to training data `\cite{yu2021artificial,zhao2023recipe,cui2023diffusionshield,ditria2023hey,xiong2023flexible}`{=latex}, essentially building a watermark encoder into the model. Stable Signature `\cite{fernandez2023stable}`{=latex} simplified this process by fine-tuning only the external decoder of latent diffusion models. However, these methods all treated watermarking as a separate goal from the generation task, limiting their flexibility. The recent Tree-Ring watermarking `\cite{wen2023tree}`{=latex} shares similarities with our approach, modifying the initial noise to encode information semantically within the image. However, the semantic modifications induced by Tree-Ring watermarks are random and may compromise the faithfulness of the original model. Therefore, we aim to preserve the original semantics exactly to guarantee a similar level of text alignment compared to the original generation. Our work shows that embedding the watermark within the intermediate diffusion state and guiding the model to hide it can achieve the secret embedding of strong watermarks without model retraining.

# Methodology

## Overview of ROBIN

#### Task definition.

Diffusion model watermarking aims to embed an invisible and verifiable watermark \\(w_i\\) within the generated image \\(x_0\\), using a watermark implantation function \\(I\\). During Internet transmission, the generated content may be subjected to various image transformation operations \\(\mathcal{T}\\). The model owner aims to leverage a watermark extraction algorithm \\(E\\) to verify the presence of \\(w_i\\) within the distorted sample \\(\mathcal{T}(x_0)\\), thereby establishing image ownership.

#### Pipeline of ROBIN.

*Watermark generation.* We first generate a hiding prompt guidance signal \\(w_p\\) for each watermark \\(w_i\\) using the adversarial optimization algorithm, which is detailed in    
efalgorithm.

*Watermark implantation.* ROBIN implants \\(w_i\\) into an intermediate generation state \\(x_t\\) after the semantics have been formed as

\\[x_t^*=I(x_t, w_i, \mathbb{M}),\\] where \\(I\\) injects \\(w_i\\) into the frequency domain of \\(x_t\\) and \\(\mathbb{M}\\) is the coverage area of the watermark. During the remaining DDIM generation, ROBIN incorporates the optimized prompt guidance signal \\(w_p\\) to direct the model towards hiding the watermark \\(w_i\\) to maintain the similarity between the generated image \\(x_0^*\\) and its unwatermarked counterpart \\(x_0\\). Let \\(t_{\text{injection}}\\) be the watermark injection point, the generation of the watermarked image is as follows:

\\[p_\theta^{(t)}(x_{t-1}|x_{t})=
\begin{cases}
 \sqrt{\bar{\alpha} _{t-1} }x_0'+\sqrt{1-\bar{\alpha }_{t-1} }\epsilon _{\theta }(x_t,t,\psi(p))  & \text{ if } T\ge t > t_{\text{injection}}  \\
\sqrt{\bar{\alpha} _{t-1} }x_0'^*+\sqrt{1-\bar{\alpha }_{t-1} }\epsilon _{\theta }(x_t^*,t,\psi(p),w_p) & \text{ if } t_{\text{injection}}\ge t
\end{cases}\\]

After embedding the watermark, the model is guided by both the original input text prompt \\(p\\) and the optimized prompt embedding \\(w_p\\) to achieve reliable generation with the watermark hidden. The predicted noise then becomes

\\[\begin{aligned}
    \epsilon_\theta (x_t^*,t,\psi(p),w_p)&=\eta_1 \cdot \epsilon _\theta (x_t^*,t,\psi (p))+\eta_2 \cdot \epsilon _\theta (x_t^*,t,w_p) \\
    &\quad + (1-\eta_1-\eta_2 )\cdot \epsilon _\theta (x_t^*,t,\psi(\emptyset)),
\end{aligned}\\] where \\(\eta_1, \eta_2\\) are the guidance scale parameters to weight the guidance of the original text prompt and the optimized prompt signal.

*Watermark verification.* To verify the watermark, we reverse the transformed watermarked image \\(\mathcal{T}(x_0^*)\\) to step \\(t_{\text{injection}}\\) and retrieve the intermediate state \\(\hat{x}_t^*\\). The watermark information \\(w'=E(\hat{x}_t^*,\mathbb{M})\\) is extracted from the frequency space of \\(\hat{x}_t^*\\). L1 distance \\(D\\) is used to measure the similarity between \\(w\\) and \\(w'\\). When the distance falls below a threshold as

\\[D(w,w')=\frac{1}{\left | \mathbb{M} \right | } \sum_{m\in \mathbb{M}}\left | w_m-w_m' \right |  \le \tau,\\] the presence of the watermark within the image is confirmed.    
effig:pipeline presents the watermark generation and implantation process of ROBIN.

<figure id="fig:pipeline">
<img src="./figures/overview5.png"" />
<figcaption>The watermark optimization and implantation of ROBIN. A robust watermark is added at an intermediate state of generation, and an additional prompt guiding signal is optimized to direct the model towards hiding the embedded watermark in the generated image. The image watermark and guiding signal are optimized adversarially to improve robustness and invisibility. </figcaption>
</figure>

## Adversarial optimization algorithm [algorithm]

We employ an adversarial optimization algorithm to generate the watermark and the corresponding hiding prompt guidance signal. The prompt signal is optimized in the embedding space and guides the model to conceal the embedded image watermark, while the watermark tries to be as strong as possible while allowing for its targeted hiding by the prompt signal.

The objective of the prompt guiding signal is to minimize the impact of the watermark on the final generated image. We define the image retaining loss \\(l_{ret}\\), which penalizes excessive deviations from the original images:

\\[\begin{gathered}
    \ell_{ret}=\text{MSE}(x_0'^*-x_0),\\
     x_{0}'^*=\frac{x_t^*-\sqrt{1-\bar{\alpha}_t } \epsilon _\theta (x_t^*,t,\psi(p), w_p)}{\sqrt{\bar{\alpha} _t }  } .
\end{gathered}\\]

\\(x_0'^*\\) is the final image predicted from the watermarked noisy latent \\(x_t^*\\) through  
efeq:prex0 with an additional guidance \\(w_p\\). MSE denotes the mean squared error.

Furthermore, as the loss incurred during DDIM inversion increases proportionally with the guidance strength `\cite{mokady2023null}`{=latex}, we introduce a constraint term \\(l_{cons}\\) to prevent excessive prompt guidance:

\\[\ell _{cons} = \text{MSE}(\epsilon _\theta(x_t^*,t, w_p)-\epsilon _\theta(x_t^*,t, \psi(\emptyset))).\\] To achieve robustness, we embed the watermark in the frequency domain of the image `\cite{wen2023tree}`{=latex}. Frequency domain signals are more resistant to spatial operations compared to spatial domain signals `\cite{wei2023generative}`{=latex}. Similar to  `\cite{wen2023tree}`{=latex}, we set the watermark as multiple concentric rings, but we further optimize its value to the maximum within the aforementioned constraints for greater strength and better robustness. The optimization losses of \\(w_p\\) and \\(w_i\\) become

\\[\begin{aligned}
    \mathcal{L}_{w_p} &= \alpha \ell _{ret}+\beta \ell _{cons}, \\  
    \mathcal{L}_{w_i} &= \alpha \ell _{ret}+\beta \ell _{cons}-\lambda \left \| w_i \right \|.  
\end{aligned}\\]

Since the watermark and the prompt guiding signal are interdependent, we employ an alternating optimization method, in which we iteratively optimize one while fixing the other. More details about the watermark design and optimization algorithm are presented in    
efapp:design.

<figure id="fig:sensitivity">

<figcaption> The impact of introducing frequency domain disturbances at different diffusion steps on the predicted noise. Timestep 1000 signifies the Gaussian noise state and step 0 represents the final generated image. The Uncondition curve (orange) and the Condition curve (gray) nearly overlap in both figures. Guidance is the amplified difference of Uncondition and Condition. Full is the addition of Uncondition and Guidance. </figcaption>
</figure>

## Finding keypoints for implantation

The selection of the optimal stage for watermark embedding within the diffusion process is crucial for achieving both high image fidelity and semantic consistency with the input text prompt. We delve into the sensitivity of the predicted noise to frequency domain disturbances in different diffusion steps. According to classifier-free guidance method `\cite{nichol2021glide}`{=latex}, the predicted noise in each step can be depicted as \\(Full=Uncondition + s\cdot (Condition - Uncondition)\\). Condition and Uncondition are predicted noise with and without text conditions. Parameter \\(s\\) is the scaling factor and the second term of the addition is called Guidance. Full noise is the final noise to be removed in the current step.

   
effig:sen-a shows the evolution of mean values of various predicted noise terms throughout the generation process. We can find that after step 300, the slowdown in guidance rise indicates the completion of basic semantic formation and diminishing guidance influence. Additionally,    
effig:sen-b presents how the predicted noise changes when perturbations are added at different timesteps. When the timestep is greater than 200, the frequency domain noise interferes with the generation process mainly by disrupting the guidance term. After 200 steps, the intrinsic unconditional term is more affected. We can conclude that early generation stages establish the foundation for image semantics and excessive intervention at this point can disrupt the intended image content. Conversely, manipulating the final stages, dedicated to refining image details, may impede the model’s capacity to recover from watermark-induced noise, ultimately compromising the final image quality.

Therefore, we strategically choose the watermark insertion point between steps 300 and 200. This stage offers the sweet spot: frequency perturbations have minimal impact on the mean of the predicted noise, allowing for watermark integration without sacrificing image quality and disruption to the core semantics.

## Watermark validation

In the watermark verification phase, we leverage the exact invertibility of the DDIM sampler to retrieve the diffusion state \(\hat{x}_t^*\) at the watermark injection step. Starting from the (possibly transformed) watermarked image \(\mathcal{T}(x_0^*)\), we deterministically integrate the DDIM equations backward to timestep \(t_{\text{injection}}\) and recover the intermediate latent. We then extract the \(\mathbb{M}\) region in Fourier space and compute its L1 distance to the original watermark \(w_i\). The watermark is accepted when

\[D(w,w') = \tfrac{1}{|\mathbb{M}|}\sum_{m\in\mathbb{M}} |w_m - w'_m| \le \tau.\]

Because DDIM is the standard deterministic sampler adopted in contemporary diffusion pipelines, this single-sampler design streamlines deployment and eliminates unnecessary engineering overhead. Empirically, we find that introducing the optimized prompt during inversion is unnecessary and may hinder recovery; therefore, inversion is performed with a null-text condition and a guidance scale of 1.0. This streamlined protocol proved sufficient across all tested architectures.
# Experiments

## Experimental setting [sec:setting]

#### Model and dataset.

We conducted experiments on two distinct diffusion models operating in latent and image domains. For the latent diffusion model, we utilize the widely available Stable Diffusion-v2 `\cite{rombach2022high}`{=latex} and the stable-diffusion-prompts dataset from Gustavosta `\cite{GustavostaStableDiffusionPromptsDatasets2023}`{=latex}. We also test on a guided diffusion model `\cite{OpenaiGuideddiffusion2024}`{=latex} trained on the ImageNet `\cite{dhariwal2021diffusion}`{=latex}, which operates directly on the pixel domain and can generate images of size \\(256\times256\\) based on the category provided.

#### Evaluation metrics.

To assess the effectiveness of ROBIN, we compute the Area Under the ROC Curve (AUC-ROC) based on the L1 distance to measure the effectiveness of watermark verification. Specifically, we compute AUC using 1,000 watermarked and 1,000 clean images. For the quality of watermarked images, we employ a suite of diverse metrics. We utilize classic measures like PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and MSSIM (Multiscale SSIM) `\cite{wang2004image}`{=latex} to quantify the pixel-level differences between watermarked and original images. We employ the Fréchet Inception Distance (FID) `\cite{heusel2017gans}`{=latex} to evaluate the fidelity of the watermarked image distribution. We also leverage the CLIP score `\cite{radford2021learning}`{=latex} to measure the alignment between generated images and their corresponding text prompts. More details are provided in    
efapp:metric.

#### Implementation details.

We utilize 50 deterministic DDIM steps for both Stable Diffusion and the ImageNet diffusion model. The guidance scale is set to 7.5 for Stable Diffusion and 4.0 for the ImageNet model. Watermark and hiding prompt are optimized on 50 generated images using learning rates of 0.8 and 5e-4, respectively, over 1,000 alternating optimization rounds. The default image watermark covers 70 % of the image frequency domain. All experiments are conducted on a single NVIDIA GeForce RTX 3090 GPU.
## Effectiveness and robustness

We compare our method with five baselines: DwtDct `\cite{chen_2009}`{=latex}, DwtDctSvd `\cite{navas2008dwt}`{=latex}, RivaGAN `\cite{zhang2019robust}`{=latex}, Stable Signature `\cite{fernandez2023stable}`{=latex}, and Tree-Ring watermarks `\cite{wen2023tree}`{=latex}. To ensure the watermark’s resilience in real-world scenarios, we delve into its robustness under various image transformations. These include Gaussian blur with radius 4, Gaussian noise with intensity 10%, jpeg compression with quality 25, color jitter with brightness 6, random rotation of 75 degrees, and random cropping of 75% and rescaling. These settings are strict for watermark verification because the image has been significantly altered. ROBIN is also evaluated under a combination of attacks where we randomly selected various combination of the six transformations. The processed samples are shown in  
effig:attack_img in the Appendix.

&DwtDct `\cite{chen_2009}`{=latex} & 0.974 & 0.503 & 0.293 & 0.492 & 0.519 & 0.596 & 0.640 & 0.574 & **0.056s**  
&DwtDctSvd `\cite{navas2008dwt}`{=latex} & **1.000** & 0.979 & 0.706 & 0.753 & 0.517 & 0.431 & 0.511 & 0.702 & 0.233s  
&RivaGAN `\cite{zhang2019robust}`{=latex} & 0.999 & 0.974 & 0.888 & 0.981 & 0.963 & 0.173 & 0.999 & 0.854 & 0.437s  
&StableSig `\cite{fernandez2023stable}`{=latex} & **1.000** & 0.565 & 0.731 & 0.989 & 0.976 & 0.658 & **1.000** & 0.845 &0.112s  
&Tree-Ring `\cite{wen2023tree}`{=latex} & **1.000** & **0.999** & 0.944 & **0.999** & **0.983** & 0.935 & 0.961 & 0.975 & 2.599s  
&ROBIN & **1.000** & **0.999** & **0.954** & **0.999** & 0.975 & **0.957** & 0.994 & **0.983** &0.531s  
<span id="tab:comp_stable" label="tab:comp_stable"></span>

& DwtDct `\cite{chen_2009}`{=latex} & 0.899 & 0.512 & 0.365 & 0.522 & 0.538 & 0.478 & 0.433 & 0.536 & **0.012s**  
&DwtDctSvd `\cite{navas2008dwt}`{=latex} & **1.000** & 0.947 & 0.656 & 0.568 & 0.535 & 0.669 & 0.614 & 0.713 & 0.058s  
&RivaGAN `\cite{zhang2019robust}`{=latex} & **1.000** & 0.988 & 0.962 & **0.978** & 0.924 & 0.321 & 0.999 & 0.882 & 0.109s  
&Tree-Ring `\cite{wen2023tree}`{=latex} & 0.999 & 0.975 & 0.979 & 0.940 & 0.861 & 0.975 & 0.994 & 0.966 & 3.963s  
&ROBIN & **1.000** & **0.999** & **0.994** & 0.969 & **0.959** & **0.998** & **1.000** & **0.988** & 0.986s  
<span id="tab:comp_imagenet" label="tab:comp_imagenet"></span>

<div id="tab:comb-att" markdown="1">

|  Method   |     1     |     2     |     3     |     4     |     5     |     6     |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Tree-Ring |   0.969   |   0.809   |   0.699   |   0.520   |   0.546   |   0.509   |
|   ROBIN   | **0.973** | **0.814** | **0.759** | **0.579** | **0.558** | **0.556** |

AUC on different number of random attacks applied at the same time.

</div>

#### Robustness.

The comprehensive results of AUC comparison with baselines are presented in  
eftab:comp_stable and  
eftab:comp_imagenet. While most methods (except DwtDct) perform well for watermark verification in the absence of attacks, their accuracy degrades with strong image manipulations. Traditional frequency-domain methods show significant vulnerability. RivaGAN falters with image rotations, and Stable Signature exhibits sensitivity to blur, noise, and rotation. The Tree-Ring watermark displays better robustness due to its pattern design but remains less resilient than ROBIN.

The performance of watermark verification for Stable Diffusion under different numbers of simultaneous attacks is shown in  
eftab:comb-att. Note that due to the inherent potency of the individual attacks, their combination leads to significant image quality deterioration. The resulting images are presented in  
effig:comb-att. But ROBIN still demonstrates superior robustness compared to the state-of-the-art method Tree-Ring in such challenging scenarios.

The robustness of ROBIN on the one hand comes from the introduction of an explicit hiding process, we can implant a stronger watermark. Furthermore, fewer inversion steps during verification compared to Tree-Ring watermarks also mitigate the accumulation of DDIM inversion errors, further enhancing accuracy. The evaluation of ROBIN under more attacks is presented in   
efapp:more_attack.

#### Time cost.

The time cost of watermark verification associated with different watermarking schemes is presented in the last column of    
eftab:comp_stable and    
eftab:comp_imagenet. The simple DwtDct method demonstrates the fastest performance, achieving a validation time of less than 0.1s. DwtDctSvd exhibits a 4\\(\times\\) slowdown compared to DwtDct, while RivaGAN is 10\\(\times\\) slower. StableSig decodes the watermark directly from the image, but it requires fine-tuning the model. The verification of Tree-Ring watermarks necessitates reversing the entire generation process, resulting in significant time costs. ROBIN requires reversing only a limited number of generation steps, resulting in consumption times of 0.531s and 0.986s for the two models, which are considerably lower compared to the Tree-Ring watermark. More experimental results are presented in    
efapp:time.

## Quality of watermarked image

Traditional post-hoc watermarking methods introduce subtle visual distortions into the generated images. In contrast, the objective of ROBIN aligns with the Tree-Ring in constructing a “content watermark”: seamlessly embedding the watermark within the image content without altering its semantics. Due to this fundamental shift in watermarking philosophy, we only compare the image quality with Tree-Ring watermarks.

<div id="tab:quality" markdown="1">

<table>
<caption>Quality of generated images. PSNR, SSIM and MSSIM measure the similarity between the watermarked and unwatermarked images. CLIP evaluates how well the watermarked image aligns with the user-provided textual description. FID measures the distribution similarity between the watermarked dataset and a random dataset of real images. The subscripts indicate the standard deviation of five independent experimental runs, each initialized with a different random seed.</caption>
<thead>
<tr>
<th style="text-align: center;">Model</th>
<th style="text-align: center;">Method</th>
<th style="text-align: center;">PSNR <span class="math inline">↑</span></th>
<th style="text-align: center;">SSIM <span class="math inline">↑</span></th>
<th style="text-align: center;">MSSIM <span class="math inline">↑</span></th>
<th style="text-align: center;">CLIP <span class="math inline">↑</span></th>
<th style="text-align: center;">FID <span class="math inline">↓</span></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align: center;">Stable Diffusion <span class="citation" data-cites="rombach2022high"></span></td>
<td style="text-align: center;"><span style="color: gray">W/o watermark</span></td>
<td style="text-align: center;"><span style="color: gray"><span class="math inline">∞</span></span></td>
<td style="text-align: center;"><span style="color: gray">1.000</span></td>
<td style="text-align: center;"><span style="color: gray">1.000</span></td>
<td style="text-align: center;"><span style="color: gray">0.403</span></td>
<td style="text-align: center;"><span style="color: gray">25.53</span></td>
</tr>
<tr>
<td style="text-align: center;">Tree-Ring <span class="citation" data-cites="wen2023tree"></span></td>
<td style="text-align: center;"><span class="math inline">15.37<sub>.07</sub></span></td>
<td style="text-align: center;"><span class="math inline">0.568<sub>.003</sub></span></td>
<td style="text-align: center;"><span class="math inline">0.626<sub>.005</sub></span></td>
<td style="text-align: center;"><span class="math inline">0.364<sub>.00</sub></span></td>
<td style="text-align: center;"><span class="math inline"><strong>25.93</strong><sub><strong>.13</strong></sub></span></td>
</tr>
<tr>
<td style="text-align: center;">ROBIN</td>
<td style="text-align: center;"><span class="math inline"><strong>24.03</strong><sub><strong>.04</strong></sub></span></td>
<td style="text-align: center;"><span class="math inline"><strong>0.768</strong><sub><strong>.000</strong></sub></span></td>
<td style="text-align: center;"><span class="math inline"><strong>0.881</strong><sub><strong>.001</strong></sub></span></td>
<td style="text-align: center;"><span class="math inline"><strong>0.396</strong><sub><strong>.00</strong></sub></span></td>
<td style="text-align: center;"><span class="math inline">26.86<sub>.09</sub></span></td>
</tr>
<tr>
<td rowspan="3" style="text-align: center;">ImageNet Diffusion <span class="citation" data-cites="OpenaiGuideddiffusion2024"></span></td>
<td style="text-align: center;"><span style="color: gray">W/o watermark</span></td>
<td style="text-align: center;"><span style="color: gray"><span class="math inline">∞</span></span></td>
<td style="text-align: center;"><span style="color: gray">1.000</span></td>
<td style="text-align: center;"><span style="color: gray">1.000</span></td>
<td style="text-align: center;"><span style="color: gray">0.271</span></td>
<td style="text-align: center;"><span style="color: gray">16.25</span></td>
</tr>
<tr>
<td style="text-align: center;">Tree-Ring <span class="citation" data-cites="wen2023tree"></span></td>
<td style="text-align: center;"><span class="math inline">15.68<sub>.03</sub></span></td>
<td style="text-align: center;"><span class="math inline">0.663<sub>.002</sub></span></td>
<td style="text-align: center;"><span class="math inline">0.607<sub>.001</sub></span></td>
<td style="text-align: center;"><span class="math inline">0.267<sub>.00</sub></span></td>
<td style="text-align: center;"><span class="math inline"><strong>17.68</strong><sub><strong>.16</strong></sub></span></td>
</tr>
<tr>
<td style="text-align: center;">ROBIN</td>
<td style="text-align: center;"><span class="math inline"><strong>24.98</strong><sub><strong>.02</strong></sub></span></td>
<td style="text-align: center;"><span class="math inline"><strong>0.875</strong><sub><strong>.000</strong></sub></span></td>
<td style="text-align: center;"><span class="math inline"><strong>0.872</strong><sub><strong>.000</strong></sub></span></td>
<td style="text-align: center;"><span class="math inline"><strong>0.275</strong><sub><strong>.00</strong></sub></span></td>
<td style="text-align: center;"><span class="math inline">18.26<sub>.13</sub></span></td>
</tr>
</tbody>
</table>

</div>

<span id="tab:quality" label="tab:quality"></span>

<figure id="fig:img_comp">
<img src="./figures/img_compare2.png"" />
<figcaption>The generated images with Tree-Ring and ROBIN watermarks. </figcaption>
</figure>

The Tree-Ring approach aims to find another watermarked image that aligns with the text prompt, even if it differs from the original image. However, it is more akin to random semantic modifications and does not guarantee the same level of text alignment as the original generation.  
effig:img_comp shows that the Tree-Ring approach significantly alters the generated image’s semantics, sometimes even failing to fulfill the text prompt’s intent. This occurs because it disrupts the essential Gaussian characteristics of the initial noise, hindering the generation process. In contrast, ROBIN excels at preserving the overall image content and semantic structure, providing a better lower bound for faithfulness by preserving the original semantics.  
eftab:quality provides the quantitative results. ROBIN demonstrates significant improvements in PSNR, SSIM, MSSSIM, and CLIP score, while a slight increase in FID is observed. This is because the position of the watermark implanted in our scheme is at a later stage of generation, resulting in a slightly greater influence on the overall generation distribution. This implies a negligible trade-off for achieving a strong watermark with minimal degradation of the overall quality of the generated image.

## Ablation study

To gain further insights into the effectiveness of ROBIN, we conduct an ablation study, exploring the influence of different design choices. We additionally introduce the Mean Squared Error of Watermark (MSE) to represent the verification accuracy in some settings where the AUC is always equal to 1. It is calculated as the mean of L1 distance between the extracted and original watermark.

#### Setting variations.

To explore the individual contributions of various components in our scheme, we conduct a series of experiments presented in  
eftab:setting. Experiments in Settings 1 and 2 demonstrate that the introduction of prompt-based watermark hiding signals improves image quality, as evidenced by a 1.6 increase in PSNR and a 1.44 decrease in FID score compared to Setting 1. Setting 3 emphasizes the importance of the \\(\ell_{ret}\\) in controlling watermark strength. Without \\(\ell_{ret}\\), ROBIN prioritizes creating a highly robust watermark, leading to significant image distortion (PSNR: 18.95, SSIM: 0.48). Setting 4 presents that removing \\(\ell_{cons}\\) allows for stronger prompt guidance, but this results in increased DDIM inversion loss and a decrease of 0.13 in adversarial AUC. Setting 5 prioritizes minimal impact on the generated image by weakening the watermark. This approach leads to poorer watermark robustness and a decrease of 0.017 in adversarial AUC. Experiments under Settings 2 and 6 demonstrate that in the presence of the hiding prompt signal, the image watermark can be optimized to achieve stronger robustness while maintaining invisibility.

<div id="tab:setting" markdown="1">

<table>
<caption>Watermark accuracy and image quality under different settings. (1) random watermarks <span class="math inline"><em>w</em><sub><em>i</em></sub></span>, (2) random watermarks with prompt signal <span class="math inline"><em>w</em><sub><em>p</em></sub></span> for hiding, (3)-(5) different loss functions for optimizing <span class="math inline"><em>w</em><sub><em>i</em></sub></span> and <span class="math inline"><em>w</em><sub><em>p</em></sub></span>, (6) full loss function for optimizing both <span class="math inline"><em>w</em><sub><em>i</em></sub></span> and <span class="math inline"><em>w</em><sub><em>p</em></sub></span>.</caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;">ID</td>
<td colspan="2" style="text-align: center;">Watermark</td>
<td colspan="3" style="text-align: center;">Loss function</td>
<td colspan="2" style="text-align: center;">AUC<span class="math inline">↑</span></td>
<td colspan="4" style="text-align: center;">Image quality</td>
</tr>
<tr>
<td style="text-align: center;">Image <span class="math inline"><em>w</em><sub><em>i</em></sub></span></td>
<td style="text-align: center;">Prompt <span class="math inline"><em>w</em><sub><em>p</em></sub></span></td>
<td style="text-align: center;"><span class="math inline"><em>ℓ</em><sub><em>r</em><em>e</em><em>t</em></sub></span></td>
<td style="text-align: center;"><span class="math inline"><em>ℓ</em><sub><em>c</em><em>o</em><em>n</em><em>s</em></sub></span></td>
<td style="text-align: center;"><span class="math inline">∥<em>w</em><sub><em>i</em></sub>∥</span></td>
<td style="text-align: center;">Clean</td>
<td style="text-align: center;">Adversarial</td>
<td style="text-align: center;">PSNR <span class="math inline">↑</span></td>
<td style="text-align: center;">SSIM<span class="math inline">↑</span></td>
<td style="text-align: center;">CLIP<span class="math inline">↑</span></td>
<td style="text-align: center;">FID<span class="math inline">↓</span></td>
</tr>
<tr>
<td style="text-align: center;">(1)</td>
<td style="text-align: center;">Random</td>
<td style="text-align: center;">None</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">1.00</td>
<td style="text-align: center;">0.903</td>
<td style="text-align: center;">20.11</td>
<td style="text-align: center;">0.68</td>
<td style="text-align: center;">0.39</td>
<td style="text-align: center;">29.21</td>
</tr>
<tr>
<td style="text-align: center;">(2)</td>
<td style="text-align: center;">Random</td>
<td style="text-align: center;">Optimized</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">1.00</td>
<td style="text-align: center;">0.901</td>
<td style="text-align: center;">21.70</td>
<td style="text-align: center;">0.70</td>
<td style="text-align: center;">0.39</td>
<td style="text-align: center;">27.77</td>
</tr>
<tr>
<td style="text-align: center;">(3)</td>
<td style="text-align: center;">Optimized</td>
<td style="text-align: center;">Optimized</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">1.00</td>
<td style="text-align: center;">0.988</td>
<td style="text-align: center;">18.95</td>
<td style="text-align: center;">0.48</td>
<td style="text-align: center;">0.30</td>
<td style="text-align: center;">32.18</td>
</tr>
<tr>
<td style="text-align: center;">(4)</td>
<td style="text-align: center;">Optimized</td>
<td style="text-align: center;">Optimized</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">1.00</td>
<td style="text-align: center;">0.970</td>
<td style="text-align: center;">23.91</td>
<td style="text-align: center;">0.76</td>
<td style="text-align: center;">0.40</td>
<td style="text-align: center;">26.68</td>
</tr>
<tr>
<td style="text-align: center;">(5)</td>
<td style="text-align: center;">Optimized</td>
<td style="text-align: center;">Optimized</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">1.00</td>
<td style="text-align: center;">0.966</td>
<td style="text-align: center;">24.19</td>
<td style="text-align: center;">0.77</td>
<td style="text-align: center;">0.40</td>
<td style="text-align: center;">26.93</td>
</tr>
<tr>
<td style="text-align: center;">(6)</td>
<td style="text-align: center;">Optimized</td>
<td style="text-align: center;">Optimized</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">1.00</td>
<td style="text-align: center;">0.983</td>
<td style="text-align: center;">24.03</td>
<td style="text-align: center;">0.77</td>
<td style="text-align: center;">0.40</td>
<td style="text-align: center;">26.86</td>
</tr>
</tbody>
</table>

</div>

#### Point of implantation.

We evaluate the impact of implanting the watermark at different stages in the diffusion process. The results are presented in  
effig:combined. Watermark verification accuracy improves with later implantation due to fewer DDIM inversion steps and reduced information loss. Early implantation, while initially maintaining image quality (low FID), can significantly change the image content (low SSIM/PSNR) by disrupting semantic formation. Conversely, late implantation may leave the watermark visible due to insufficient space for hiding, leading to high FID and deviation from the original image (low SSIM). This empowers us to pinpoint the optimal embedding stage (steps 15-10) for balancing visual quality and semantic preservation.

<figure id="fig:combined">

<figcaption>Ablation experiments on embedding point and watermark strength.</figcaption>
</figure>

<figure id="fig:img_strength">
<img src="./figures/strength_sample.png"" />
<figcaption>Generated images under different watermark strengths. The top row is the result of the Tree-Ring scheme and the bottom row is the result of ROBIN. </figcaption>
</figure>

#### Watermark strength.

We also verify the influence of different watermarking strengths and the results are shown in  
effig:combined. Higher watermark strength (proportional coverage in the frequency domain) generally benefits verification accuracy, as the watermark becomes more prominent. The CLIP score and FID remain stable due to strategic embedding and guided hiding. Traditional metrics (SSIM, PSNR) decrease with stronger watermarks due to increased content modification. The watermarked images under different strengths are shown in  
effig:img_strength. Compared to Tree-Ring, the quality of generated images with ROBIN watermarks is less sensitive to watermark strength. More qualitative results are presented in    
efapp:qual_result.

# Conclusion & Discussion

This paper proposes a novel watermarking method for the diffusion model, which embeds a watermark in the intermediate diffusion state and guides the model to conceal the watermark. By explicitly introducing the active hiding process, we can implant stronger watermarks without compromising image quality. We believe this method holds promise for expanding the possibilities of reliable watermarking in diffusion models.

#### Limitations. [sec:limitation]

The verification of ROBIN watermarks relies on the reversible generation process, future advancements enabling the reversibility of other sampling algorithms would broaden the application of our method. Additionally, the inherent information loss during DDIM inversion can be reduced by exploring generative trajectories that can be reversed exactly `\cite{pan2023effective,hong2023exact,wallace2023edict,zhang2023exact}`{=latex}.

#### Social impact.  [sec:social]

Our ROBIN scheme, as a watermarking method, can help creators establish ownership and discourage unauthorized use. Furthermore, ROBIN watermarks can be implanted in a one-shot manner without retraining the whole model, making it applicable to different diffusion-based text-to-image models.

# Acknowledgment [acknowledgment]

This work was partially supported by the National Natural Science Foundation of China under grants 62372341, U20B2049, and U21B2018.

# References [references]

<div class="thebibliography" markdown="1">

Gustavosta/Stable-Diffusion-Prompts \\(\cdot\\) Datasets at Hugging Face https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts, March 2023. **Abstract:** Ensuring the availability of child facial datasets is essential for advancing AI applications, yet legal, ethical, and data scarcity concerns pose significant challenges. Current generative models such as StyleGAN excel at producing synthetic facial data but struggle with temporal consistency, control over output attributes, and diversity in rendered features. These limitations underscore the need for a more robust and adaptable framework. In this research, we propose the ChildDiffusion framework, designed to generate photorealistic child facial data using diffusion models. The framework integrates intelligent augmentations via short text prompts, employs various image samplers, and leverages ControlNet for enhanced model conditioning. Additionally, we have used large language models (LLMs) to provide complex textual guidance to enable precise image-to-image transformations, facilitating the curation of diverse, high-quality datasets. The model was validated by generating child faces with varied ethnicities, facial expressions, poses, lighting conditions, eye-blinking effects, accessories, hair colors, and multi-subject compositions. To exemplify its potential, we open-sourced a dataset of 2.5k child facial samples across five ethnic classes, which underwent rigorous qualitative and quantitative evaluations. Further, we fine-tuned a Vision Transformer model to classify child ethnicity as a downstream task, demonstrating the framework’s utility. This research advances generative AI by addressing data scarcity and ethical challenges, showcasing how diffusion models can produce realistic child facial data while ensuring compliance with privacy standards. The versatile ChildDiffusion framework offers broad potential for machine learning applications, serving as a valuable tool for AI innovation. The project website, along with the complete ChildRace dataset and the fine-tuned model, is available at (https://mali-farooq.github.io/childdiffusion/). (@GustavostaStableDiffusionPromptsDatasets2023)

Openai/guided-diffusion https://github.com/openai/guided-diffusion, May 2024. **Abstract:** Diffusion models have recently been shown to generate high-quality synthetic images, especially when paired with a guidance technique to trade off diversity for fidelity. We explore diffusion models for the problem of text-conditional image synthesis and compare two different guidance strategies: CLIP guidance and classifier-free guidance. We find that the latter is preferred by human evaluators for both photorealism and caption similarity, and often produces photorealistic samples. Samples from a 3.5 billion parameter text-conditional diffusion model using classifier-free guidance are favored by human evaluators to those from DALL-E, even when the latter uses expensive CLIP reranking. Additionally, we find that our models can be fine-tuned to perform image inpainting, enabling powerful text-driven image editing. We train a smaller model on a filtered dataset and release the code and weights at https://github.com/openai/glide-text2im. (@OpenaiGuideddiffusion2024)

Ali Al-Haj Combined dwt-dct digital image watermarking *Journal of computer science*, 3 (9): 740–746, 2007. **Abstract:** The proliferation of digitized media due to the rapid growth of networked multimedia systems, has created an urgent need for copyright enforcement technologies that can protect copyright ownership of multimedia objects. Digital image watermarking is one such technology that has been developed to protect digital images from illegal manipulations. In particular, digital image watermarking algorithms which are based on the discrete wavelet transform have been widely recognized to be more prevalent than others. This is due to the wavelets’ excellent spatial localization, frequency spread, and multi-resolution characteristics, which are similar to the theoretical models of the human visual system. In this paper, we describe an imperceptible and a robust combined DWT-DCT digital image watermarking algorithm. The algorithm watermarks a given digital image using a combination of the Discrete Wavelet Transform (DWT) and the Discrete Cosine Transform (DCT). Performance evaluation results show that combining the two transforms improved the performance of the watermarking algorithms that are based solely on the DWT transform. (@al2007combined)

Kuanchin Chen *Digital Watermarking and Steganography*, page 402–409 Jan 2009. . **Abstract:** Multimedia data in various forms is now readily available because of the widespread usage of Internet technology. Unauthorized individuals abuse multimedia material, for which they should not have access to, by disseminating it over several web pages, to defraud the original copyright owners. Numerous patient records have been compromised during the surge in COVID-19 incidents. Adding a watermark to any medical or defense documents is recommended since it protects the integrity of the information. This proposed work is recognized as a new unique method since an innovative technique is being implemented. The resilience of the watermarked picture is quite crucial in the context of steganography. As a result, the emphasis of this research study is on the resilience of watermarked picture methods. Moreover, the two-stage authentication for watermarking is built with key generation in the section on robust improvement. The Fast Fourier transform (FFT) is used in the entire execution process of the suggested framework in order to make computing more straightforward. With the Singular Value Decomposition (SVD) accumulation of processes, the overall suggested architecture becomes more resilient and efficient. A numerous quality metrics are utilized to find out how well the created technique is performing in terms of evaluation. In addition, several signal processing attacks are used to assess the effectiveness of the watermarking strategy. (@chen_2009)

Ruoxi Chen, Haibo Jin, Jinyin Chen, and Lichao Sun Editshield: Protecting unauthorized image editing by instruction-guided diffusion models *arXiv preprint arXiv:2311.12066*, 2023. **Abstract:** Text-to-image diffusion models have emerged as an evolutionary for producing creative content in image synthesis. Based on the impressive generation abilities of these models, instruction-guided diffusion models can edit images with simple instructions and input images. While they empower users to obtain their desired edited images with ease, they have raised concerns about unauthorized image manipulation. Prior research has delved into the unauthorized use of personalized diffusion models; however, this problem of instruction-guided diffusion models remains largely unexplored. In this paper, we first propose a protection method EditShield against unauthorized modifications from such models. Specifically, EditShield works by adding imperceptible perturbations that can shift the latent representation used in the diffusion process, tricking models into generating unrealistic images with mismatched subjects. Our extensive experiments demonstrate EditShield’s effectiveness among synthetic and real-world datasets. Besides, we found that EditShield performs robustly against various manipulation settings across editing types and synonymous instruction phrases. (@chen2023editshield)

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev Reproducible scaling laws for contrastive language-image learning In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2818–2829, 2023. **Abstract:** Scaling up neural networks has led to remarkable performance across a wide range of tasks. Moreover, performance often follows reliable scaling laws as a function of training set size, model size, and compute, which offers valuable guidance as large-scale experiments are becoming increasingly expensive. However, previous work on scaling laws has primarily used private data & models or focused on uni-modal language or vision learning. To address these limitations, we investigate scaling laws for contrastive language-image pre-training (CLIP) with the public LAION dataset and the open-source OpenCLIP repository. Our large-scale experiments involve models trained on up to two billion image-text pairs and identify power law scaling for multiple downstream tasks including zero-shot classification, retrieval, linear probing, and end-to-end fine-tuning. We find that the training distribution plays a key role in scaling laws as the OpenAI and OpenCLIP models exhibit different scaling behavior despite identical model architectures and similar training recipes. We open-source our evaluation workflow and all models, including the largest public CLIP models, to ensure reproducibility and make scaling laws research more accessible. Source code and instructions to reproduce this study is available at https://github.eom/LAION-AI/sealing-laws-openelip. (@cherti2023reproducible)

Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord Diffedit: Diffusion-based semantic image editing with mask guidance *arXiv preprint arXiv:2210.11427*, 2022. **Abstract:** Image generation has recently seen tremendous advances, with diffusion models allowing to synthesize convincing images for a large variety of text prompts. In this article, we propose DiffEdit, a method to take advantage of text-conditioned diffusion models for the task of semantic image editing, where the goal is to edit an image based on a text query. Semantic image editing is an extension of image generation, with the additional constraint that the generated image should be as similar as possible to a given input image. Current editing methods based on diffusion models usually require to provide a mask, making the task much easier by treating it as a conditional inpainting task. In contrast, our main contribution is able to automatically generate a mask highlighting regions of the input image that need to be edited, by contrasting predictions of a diffusion model conditioned on different text prompts. Moreover, we rely on latent inference to preserve content in those regions of interest and show excellent synergies with mask-based diffusion. DiffEdit achieves state-of-the-art editing performance on ImageNet. In addition, we evaluate semantic image editing in more challenging settings, using images from the COCO dataset as well as text-based generated images. (@couairon2022diffedit)

IJ Cox Digital watermarking and steganography *Morgan Kaufmann google schola*, 2: 893–914, 2007. **Abstract:** Sharing, disseminating, and presenting data in digital format is not just a fad, but it is becoming part of our life. Without careful planning, digitized resources could easily be misused, especially those that are shared across the Internet. Examples of such misuse include use without the owner’s permission, and modification of a digitized resource to fake ownership. One way to prevent such behaviors is to employ some form of copyright protection technique, such as digital watermarks. Digital watermarks refer to the data embedded into a digital source (e.g., images, text, audio, or video recording). They are similar to watermarks in printed materials as a message inserted into the host media typically becomes an integral part of the media. Apart from traditional watermarks in printed forms, digital watermarks may also be invisible, may be in the forms other than graphics, and may be digitally removed. (@cox2007digital)

Ingemar J Cox, Joe Kilian, Tom Leighton, and Talal Shamoon Secure spread spectrum watermarking for images, audio and video In *Proceedings of IEEE International Conference on Image Processing*, volume 3, pages 243–246. IEEE, 1996. **Abstract:** We describe a digital watermarking method for use in audio, image, video and multimedia data. We argue that a watermark must be placed in perceptually significant components of a signal if it is to be robust to common signal distortions and malicious attack. However, it is well known that modification of these components can lead to perceptual degradation of the signal. To avoid this, we propose to insert a watermark into the spectral components of the data using techniques analogous to spread spectrum communications, hiding a narrow band signal in a wideband channel that is the data. The watermark is difficult for an attacker to remove, even when several individuals conspire together with independently watermarked copies of the data. It is also robust to common signal and geometric distortions such as digital-to-analog and analog-to-digital conversion, resampling, quantization, dithering, compression, rotation, translation, cropping and scaling. The same digital watermarking algorithm can be applied to all three media under consideration with only minor modifications, making it especially appropriate for multimedia products. Retrieval of the watermark unambiguously identifies the owner, and the watermark can be constructed to make counterfeiting almost impossible. We present experimental results to support these claims. (@cox1996secure)

Yingqian Cui, Jie Ren, Han Xu, Pengfei He, Hui Liu, Lichao Sun, and Jiliang Tang Diffusionshield: A watermark for copyright protection against generative diffusion models *arXiv preprint arXiv:2306.04642*, 2023. **Abstract:** Recently, Generative Diffusion Models (GDMs) have showcased their remarkable capabilities in learning and generating images. A large community of GDMs has naturally emerged, further promoting the diversified applications of GDMs in various fields. However, this unrestricted proliferation has raised serious concerns about copyright protection. For example, artists including painters and photographers are becoming increasingly concerned that GDMs could effortlessly replicate their unique creative works without authorization. In response to these challenges, we introduce a novel watermarking scheme, DiffusionShield, tailored for GDMs. DiffusionShield protects images from copyright infringement by GDMs through encoding the ownership information into an imperceptible watermark and injecting it into the images. Its watermark can be easily learned by GDMs and will be reproduced in their generated images. By detecting the watermark from generated images, copyright infringement can be exposed with evidence. Benefiting from the uniformity of the watermarks and the joint optimization method, DiffusionShield ensures low distortion of the original image, high watermark detection performance, and the ability to embed lengthy messages. We conduct rigorous and comprehensive experiments to show the effectiveness of DiffusionShield in defending against infringement by GDMs and its superiority over traditional watermarking methods. The code for DiffusionShield is accessible in https://github.com/Yingqiancui/DiffusionShield. (@cui2023diffusionshield)

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei Imagenet: A large-scale hierarchical image database In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 248–255, 2009. **Abstract:** The explosion of image data on the Internet has the potential to foster more sophisticated and robust models and algorithms to index, retrieve, organize and interact with images and multimedia data. But exactly how such data can be harnessed and organized remains a critical problem. We introduce here a new database called "ImageNet", a large-scale ontology of images built upon the backbone of the WordNet structure. ImageNet aims to populate the majority of the 80,000 synsets of WordNet with an average of 500–1000 clean and full resolution images. This will result in tens of millions of annotated images organized by the semantic hierarchy of WordNet. This paper offers a detailed analysis of ImageNet in its current state: 12 subtrees with 5247 synsets and 3.2 million images in total. We show that ImageNet is much larger in scale and diversity and much more accurate than the current image datasets. Constructing such a large-scale database is a challenging task. We describe the data collection scheme with Amazon Mechanical Turk. Lastly, we illustrate the usefulness of ImageNet through three simple applications in object recognition, image classification and automatic object clustering. We hope that the scale, accuracy, diversity and hierarchical structure of ImageNet can offer unparalleled opportunities to researchers in the computer vision community and beyond. (@deng2009imagenet)

Prafulla Dhariwal and Alexander Nichol Diffusion models beat gans on image synthesis *Advances in Neural Information Processing Systems*, 34: 8780–8794, 2021. **Abstract:** We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models. We achieve this on unconditional image synthesis by finding a better architecture through a series of ablations. For conditional image synthesis, we further improve sample quality with classifier guidance: a simple, compute-efficient method for trading off diversity for fidelity using gradients from a classifier. We achieve an FID of 2.97 on ImageNet 128$\\}times$128, 4.59 on ImageNet 256$\\}times$256, and 7.72 on ImageNet 512$\\}times$512, and we match BigGAN-deep even with as few as 25 forward passes per sample, all while maintaining better coverage of the distribution. Finally, we find that classifier guidance combines well with upsampling diffusion models, further improving FID to 3.94 on ImageNet 256$\\}times$256 and 3.85 on ImageNet 512$\\}times$512. We release our code at https://github.com/openai/guided-diffusion (@dhariwal2021diffusion)

Luke Ditria and Tom Drummond Hey that’s mine imperceptible watermarks are preserved in diffusion generated outputs *arXiv preprint arXiv:2308.11123*, 2023. **Abstract:** Generative models have seen an explosion in popularity with the release of huge generative Diffusion models like Midjourney and Stable Diffusion to the public. Because of this new ease of access, questions surrounding the automated collection of data and issues regarding content ownership have started to build. In this paper we present new work which aims to provide ways of protecting content when shared to the public. We show that a generative Diffusion model trained on data that has been imperceptibly watermarked will generate new images with these watermarks present. We further show that if a given watermark is correlated with a certain feature of the training data, the generated images will also have this correlation. Using statistical tests we show that we are able to determine whether a model has been trained on marked data, and what data was marked. As a result our system offers a solution to protect intellectual property when sharing content online. (@ditria2023hey)

Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon The stable signature: Rooting watermarks in latent diffusion models In *IEEE/CVF International Conference on Computer Vision*, pages 22409–22420. IEEE, 2023. **Abstract:** Generative image modeling enables a wide range of applications but raises ethical concerns about responsible deployment. We introduce an active content tracing method combining image watermarking and Latent Diffusion Models. The goal is for all generated images to conceal an invisible watermark allowing for future detection and/or identification. The method quickly fine-tunes the latent decoder of the image generator, conditioned on a binary signature. A pre-trained watermark extractor recovers the hidden signature from any generated image and a statistical test then determines whether it comes from the generative model. We evaluate the invisibility and robustness of the watermarks on a variety of generation tasks, showing that the Stable Signature is robust to image modifications. For instance, it detects the origin of an image generated from a text prompt, then cropped to keep 10% of the content, with 90+% accuracy at a false positive rate below 10 \<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>−6\</sup\> . (@fernandez2023stable)

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or An image is worth one word: Personalizing text-to-image generation using textual inversion In *International Conference on Learning Representations*. OpenReview.net, 2023. **Abstract:** Text-to-image models offer unprecedented freedom to guide creation through natural language. Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes. In other words, we ask: how can we use language-guided models to turn our cat into a painting, or imagine a new product based on our favorite toy? Here we present a simple approach that allows such creative freedom. Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model. These "words" can be composed into natural language sentences, guiding personalized creation in an intuitive way. Notably, we find evidence that a single word embedding is sufficient for capturing unique and varied concepts. We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks. Our code, data and new words will be available at: https://textual-inversion.github.io (@gal2022image)

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter Gans trained by a two time-scale update rule converge to a local nash equilibrium *Advances in Neural Information Processing Systems*, 30, 2017. **Abstract:** Generative Adversarial Networks (GANs) excel at creating realistic images with complex models for which maximum likelihood is infeasible. However, the convergence of GAN training has still not been proved. We propose a two time-scale update rule (TTUR) for training GANs with stochastic gradient descent on arbitrary GAN loss functions. TTUR has an individual learning rate for both the discriminator and the generator. Using the theory of stochastic approximation, we prove that the TTUR converges under mild assumptions to a stationary local Nash equilibrium. The convergence carries over to the popular Adam optimization, for which we prove that it follows the dynamics of a heavy ball with friction and thus prefers flat minima in the objective landscape. For the evaluation of the performance of GANs at image generation, we introduce the "Fr\\}’echet Inception Distance" (FID) which captures the similarity of generated images to real ones better than the Inception Score. In experiments, TTUR improves learning for DCGANs and Improved Wasserstein GANs (WGAN-GP) outperforming conventional GAN training on CelebA, CIFAR-10, SVHN, LSUN Bedrooms, and the One Billion Word Benchmark. (@heusel2017gans)

Jonathan Ho, Ajay Jain, and Pieter Abbeel Denoising diffusion probabilistic models *Advances in Neural Information Processing Systems*, 33: 6840–6851, 2020. **Abstract:** We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at https://github.com/hojonathanho/diffusion (@ho2020denoising)

Seongmin Hong, Kyeonghyun Lee, Suh Yoon Jeon, Hyewon Bae, and Se Young Chun On exact inversion of dpm-solvers *arXiv preprint arXiv:2311.18387*, 2023. **Abstract:** Diffusion probabilistic models (DPMs) are a key component in modern generative models. DPM-solvers have achieved reduced latency and enhanced quality significantly, but have posed challenges to find the exact inverse (i.e., finding the initial noise from the given image). Here we investigate the exact inversions for DPM-solvers and propose algorithms to perform them when samples are generated by the first-order as well as higher-order DPM-solvers. For each explicit denoising step in DPM-solvers, we formulated the inversions using implicit methods such as gradient descent or forward step method to ensure the robustness to large classifier-free guidance unlike the prior approach using fixed-point iteration. Experimental results demonstrated that our proposed exact inversion methods significantly reduced the error of both image and noise reconstructions, greatly enhanced the ability to distinguish invisible watermarks and well prevented unintended background changes consistently during image editing. Project page: \\}url{https://smhongok.github.io/inv-dpm.html}. (@hong2023exact)

Makena Kelly White house rolls out plan to promote ethical ai 2023. URL <https://www.theverge.com/2023/5/4/23710533/google-microsoft-openai-white-house-ethical-ai-artificial-intelligence>. **Abstract:** The Biden administration is introducing new funding and policy guidance for developing artificial intelligence. The Biden administration is introducing new funding and policy guidance for developing artificial intelligence. The White House announced more funding and policy guidance for developing responsible artificial intelligence ahead of a Biden administration meeting with top industry executives. The actions include a $140 million investment from the National Science Foundation to launch seven new National AI Research (NAIR) Institutes, increasing the total number of AI-dedicated facilities to 25 nationwide. Google, Microsoft, Nvidia, OpenAI and other companies have also agreed to allow their language models to be publicly evaluated during this year’s Def Con. The Office of Management and Budget (OMB) also said that it would be publishing draft rules this summer for how the federal government should use AI technology. “These steps build on the Administration’s strong record of leadership to ensure technology improves the lives of the American people, and break new ground in the federal government’s ongoing effort to advance a cohesive and comprehensive approach to AI-related risks and opportunities,” the administration’s press release said. It does not specify the details of what the Def Con evaluation will include, beyond saying that it will “allow these models to be evaluated thoroughly by thousands of community partners and AI experts.” (@whitehouse)

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick Microsoft coco: Common objects in context In *European Conference on Computer Vision*, pages 740–755. springer, 2014. **Abstract:** We present a new dataset with the goal of advancing the state-of-the-art in object recognition by placing the question of object recognition in the context of the broader question of scene understanding. This is achieved by gathering images of complex everyday scenes containing common objects in their natural context. Objects are labeled using per-instance segmentations to aid in precise object localization. Our dataset contains photos of 91 objects types that would be easily recognizable by a 4 year old. With a total of 2.5 million labeled instances in 328k images, the creation of our dataset drew upon extensive crowd worker involvement via novel user interfaces for category detection, instance spotting and instance segmentation. We present a detailed statistical analysis of the dataset in comparison to PASCAL, ImageNet, and SUN. Finally, we provide baseline performance analysis for bounding box and segmentation detection results using a Deformable Parts Model. (@lin2014microsoft)

Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao Pseudo numerical methods for diffusion models on manifolds In *International Conference on Learning Representations*, 2022. **Abstract:** Denoising Diffusion Probabilistic Models (DDPMs) can generate high-quality samples such as image and audio samples. However, DDPMs require hundreds to thousands of iterations to produce final samples. Several prior works have successfully accelerated DDPMs through adjusting the variance schedule (e.g., Improved Denoising Diffusion Probabilistic Models) or the denoising equation (e.g., Denoising Diffusion Implicit Models (DDIMs)). However, these acceleration methods cannot maintain the quality of samples and even introduce new noise at a high speedup rate, which limit their practicability. To accelerate the inference process while keeping the sample quality, we provide a fresh perspective that DDPMs should be treated as solving differential equations on manifolds. Under such a perspective, we propose pseudo numerical methods for diffusion models (PNDMs). Specifically, we figure out how to solve differential equations on manifolds and show that DDIMs are simple cases of pseudo numerical methods. We change several classical numerical methods to corresponding pseudo numerical methods and find that the pseudo linear multi-step method is the best in most situations. According to our experiments, by directly using pre-trained models on Cifar10, CelebA and LSUN, PNDMs can generate higher quality synthetic images with only 50 steps compared with 1000-step DDIMs (20x speedup), significantly outperform DDIMs with 250 steps (by around 0.4 in FID) and have good generalization on different variance schedules. Our implementation is available at https://github.com/luping-liu/PNDM. (@liupseudo)

Xiao Liu, Xiaoliu Guan, Yu Wu, and Jiaxu Miao Iterative ensemble training with anti-gradient control for mitigating memorization in diffusion models In *European Conference on Computer Vision*, 2024. **Abstract:** Diffusion models, known for their tremendous ability to generate novel and high-quality samples, have recently raised concerns due to their data memorization behavior, which poses privacy risks. Recent approaches for memory mitigation either only focused on the text modality problem in cross-modal generation tasks or utilized data augmentation strategies. In this paper, we propose a novel training framework for diffusion models from the perspective of visual modality, which is more generic and fundamental for mitigating memorization. To facilitate forgetting of stored information in diffusion model parameters, we propose an iterative ensemble training strategy by splitting the data into multiple shards for training multiple models and intermittently aggregating these model parameters. Moreover, practical analysis of losses illustrates that the training loss for easily memorable images tends to be obviously lower. Thus, we propose an anti-gradient control method to exclude the sample with a lower loss value from the current mini-batch to avoid memorizing. Extensive experiments and analysis on four datasets are conducted to illustrate the effectiveness of our method, and results show that our method successfully reduces memory capacity while even improving the performance slightly. Moreover, to save the computing cost, we successfully apply our method to fine-tune the well-trained diffusion models by limited epochs, demonstrating the applicability of our method. Code is available in https://github.com/liuxiao-guan/IET_AGC. (@liu2024iterative)

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps *Advances in Neural Information Processing Systems*, 35: 5775–5787, 2022. **Abstract:** Diffusion probabilistic models (DPMs) are emerging powerful generative models. Despite their high-quality generation performance, DPMs still suffer from their slow sampling as they generally need hundreds or thousands of sequential function evaluations (steps) of large neural networks to draw a sample. Sampling from DPMs can be viewed alternatively as solving the corresponding diffusion ordinary differential equations (ODEs). In this work, we propose an exact formulation of the solution of diffusion ODEs. The formulation analytically computes the linear part of the solution, rather than leaving all terms to black-box ODE solvers as adopted in previous works. By applying change-of-variable, the solution can be equivalently simplified to an exponentially weighted integral of the neural network. Based on our formulation, we propose DPM-Solver, a fast dedicated high-order solver for diffusion ODEs with the convergence order guarantee. DPM-Solver is suitable for both discrete-time and continuous-time DPMs without any further training. Experimental results show that DPM-Solver can generate high-quality samples in only 10 to 20 function evaluations on various datasets. We achieve 4.70 FID in 10 function evaluations and 2.87 FID in 20 function evaluations on the CIFAR10 dataset, and a $4\\}sim 16\\}times$ speedup compared with previous state-of-the-art training-free samplers on various datasets. (@lu2022dpm)

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models *arXiv preprint arXiv:2211.01095*, 2022. **Abstract:** Diffusion probabilistic models (DPMs) have achieved impressive success in high-resolution image synthesis, especially in recent large-scale text-to-image generation applications. An essential technique for improving the sample quality of DPMs is guided sampling, which usually needs a large guidance scale to obtain the best sample quality. The commonly-used fast sampler for guided sampling is DDIM, a first-order diffusion ODE solver that generally needs 100 to 250 steps for high-quality samples. Although recent works propose dedicated high-order solvers and achieve a further speedup for sampling without guidance, their effectiveness for guided sampling has not been well-tested before. In this work, we demonstrate that previous high-order fast samplers suffer from instability issues, and they even become slower than DDIM when the guidance scale grows large. To further speed up guided sampling, we propose DPM-Solver++, a high-order solver for the guided sampling of DPMs. DPM-Solver++ solves the diffusion ODE with the data prediction model and adopts thresholding methods to keep the solution matches training data distribution. We further propose a multistep variant of DPM-Solver++ to address the instability issue by reducing the effective step size. Experiments show that DPM-Solver++ can generate high-quality samples within only 15 to 20 steps for guided sampling by pixel-space and latent-space DPMs. (@lu2022dpm++)

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or Null-text inversion for editing real images using guided diffusion models In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6038–6047, 2023. **Abstract:** Recent large-scale text-guided diffusion models provide powerful image generation capabilities. Currently, a massive effort is given to enable the modification of these images using text only as means to offer intuitive and versatile editing tools. To edit a real image using these state-of-the-art tools, one must first invert the image with a meaningful text prompt into the pretrained model’s domain. In this paper, we introduce an accurate inversion technique and thus facilitate an intuitive text-based modification of the image. Our proposed inversion consists of two key novel components: (i) Pivotal inversion for diffusion models. While current methods aim at mapping random noise samples to a single input image, we use a single pivotal noise vector for each timestamp and optimize around it. We demonstrate that a direct DDIM inversion is inadequate on its own, but does provide a rather good anchor for our optimization. (ii) Null-text optimization, where we only modify the unconditional textual embedding that is used for classifier-free guidance, rather than the input text embedding. This allows for keeping both the model weights and the conditional embedding intact and hence enables applying prompt-based editing while avoiding the cumbersome tuning of the model’s weights. Our null-text inversion, based on the publicly available Stable Diffusion model, is extensively evaluated on a variety of images and various prompt editing, showing high-fidelity editing of real images. (@mokady2023null)

KA Navas, Mathews Cheriyan Ajay, M Lekshmi, Tampy S Archana, and M Sasikumar Dwt-dct-svd based watermarking In *International Conference on Communication Systems Software and Middleware and Workshops*, pages 271–274. IEEE, 2008. **Abstract:** Some works are reported in the frequency domain watermarking using Single Value Decomposition (SVD). The two most commonly used methods are based on DCT-SVD and DWT-SVD. The commonly present disadvantages in traditional watermarking techniques such as inability to withstand attacks are absent in SVD based algorithms. They offer a robust method of watermarking with minimum or no distortion. DCT based watermarking techniques offer compression while DWT based compression offer scalability. Thus all the three desirable properties can be utilized to create a new robust watermarking technique. In this paper, we propose a method of non-blind transform domain watermarking based on DWT-DCT-SVD. The DCT coefficients of the DWT coefficients are used to embed the watermarking information. This method of watermarking is found to be robust and the visual watermark is recoverable without only reasonable amount of distortion even in the case of attacks. Thus the method can be used to embed copyright information in the form of a visual watermark or simple text. (@navas2008dwt)

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen Glide: Towards photorealistic image generation and editing with text-guided diffusion models In *International Conference on Machine Learning*, volume 162, pages 16784–16804. PMLR, 2022. **Abstract:** Diffusion models have recently been shown to generate high-quality synthetic images, especially when paired with a guidance technique to trade off diversity for fidelity. We explore diffusion models for the problem of text-conditional image synthesis and compare two different guidance strategies: CLIP guidance and classifier-free guidance. We find that the latter is preferred by human evaluators for both photorealism and caption similarity, and often produces photorealistic samples. Samples from a 3.5 billion parameter text-conditional diffusion model using classifier-free guidance are favored by human evaluators to those from DALL-E, even when the latter uses expensive CLIP reranking. Additionally, we find that our models can be fine-tuned to perform image inpainting, enabling powerful text-driven image editing. We train a smaller model on a filtered dataset and release the code and weights at https://github.com/openai/glide-text2im. (@nichol2021glide)

Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, and Stephen Huang Effective real image editing with accelerated iterative diffusion inversion In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 15912–15921, 2023. **Abstract:** Despite all recent progress, it is still challenging to edit and manipulate natural images with modern generative models. When using Generative Adversarial Network (GAN), one major hurdle is in the inversion process mapping a real image to its corresponding noise vector in the latent space, since it is necessary to be able to reconstruct an image to edit its contents. Likewise for Denoising Diffusion Implicit Models (DDIM), the linearization assumption in each inversion step makes the whole deterministic inversion process unreliable. Existing approaches that have tackled the problem of inversion stability often incur in significant trade-offs in computational efficiency. In this work we propose an Accelerated Iterative Diffusion Inversion method, dubbed AIDI, that significantly improves reconstruction accuracy with minimal additional overhead in space and time complexity. By using a novel blended guidance technique, we show that effective results can be obtained on a large range of image editing tasks without large classifier-free guidance in inversion. Furthermore, when compared with other diffusion inversion based works, our proposed process is shown to be more robust for fast image editing in the 10 and 20 diffusion steps’ regimes. (@pan2023effective)

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al Learning transferable visual models from natural language supervision In *International Conference on Machine Learning*, pages 8748–8763. PMLR, 2021. **Abstract:** State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at https://github.com/OpenAI/CLIP. (@radford2021learning)

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen Hierarchical text-conditional image generation with clip latents *arXiv e-prints*, pages arXiv–2204, 2022. **Abstract:** Contrastive models like CLIP have been shown to learn robust representations of images that capture both semantics and style. To leverage these representations for image generation, we propose a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding. We show that explicitly generating image representations improves image diversity with minimal loss in photorealism and caption similarity. Our decoders conditioned on image representations can also produce variations of an image that preserve both its semantics and style, while varying the non-essential details absent from the image representation. Moreover, the joint embedding space of CLIP enables language-guided image manipulations in a zero-shot fashion. We use diffusion models for the decoder and experiment with both autoregressive and diffusion models for the prior, finding that the latter are computationally more efficient and produce higher-quality samples. (@ramesh2022hierarchical)

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer High-resolution image synthesis with latent diffusion models In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10684–10695, 2022. **Abstract:** By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve new state of the art scores for image inpainting and class-conditional image synthesis and highly competitive performance on various tasks, including unconditional image generation, text-to-image synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs. (@rombach2022high)

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 22500–22510, 2023. **Abstract:** Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for "personalization" of text-to-image diffusion models. Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can be used to synthesize novel photorealistic images of the subject contextualized in different scenes. By leveraging the semantic prior embedded in the model with a new autogenous class-specific prior preservation loss, our technique enables synthesizing the subject in diverse scenes, poses, views and lighting conditions that do not appear in the reference images. We apply our technique to several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, and artistic rendering, all while preserving the subject’s key features. We also provide a new dataset and evaluation protocol for this new task of subject-driven generation. Project page: https://dreambooth.github.io/ (@ruiz2023dreambooth)

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al Photorealistic text-to-image diffusion models with deep language understanding *Advances in Neural Information Processing Systems*, 35: 36479–36494, 2022. **Abstract:** We present Imagen, a text-to-image diffusion model with an unprecedented degree of photorealism and a deep level of language understanding. Imagen builds on the power of large transformer language models in understanding text and hinges on the strength of diffusion models in high-fidelity image generation. Our key discovery is that generic large language models (e.g. T5), pretrained on text-only corpora, are surprisingly effective at encoding text for image synthesis: increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model. Imagen achieves a new state-of-the-art FID score of 7.27 on the COCO dataset, without ever training on COCO, and human raters find Imagen samples to be on par with the COCO data itself in image-text alignment. To assess text-to-image models in greater depth, we introduce DrawBench, a comprehensive and challenging benchmark for text-to-image models. With DrawBench, we compare Imagen with recent methods including VQ-GAN+CLIP, Latent Diffusion Models, and DALL-E 2, and find that human raters prefer Imagen over other models in side-by-side comparisons, both in terms of sample quality and image-text alignment. See https://imagen.research.google/ for an overview of the results. (@saharia2022photorealistic)

Shawn Shan, Jenna Cryan, Emily Wenger, Haitao Zheng, Rana Hanocka, and Ben Y Zhao Glaze: Protecting artists from style mimicry by text-to-image models In *USENIX Security Symposium*, pages 2187–2204. USENIX Association, 2023. **Abstract:** Recent text-to-image diffusion models such as MidJourney and Stable Diffusion threaten to displace many in the professional artist community. In particular, models can learn to mimic the artistic style of specific artists after "fine-tuning" on samples of their art. In this paper, we describe the design, implementation and evaluation of Glaze, a tool that enables artists to apply "style cloaks" to their art before sharing online. These cloaks apply barely perceptible perturbations to images, and when used as training data, mislead generative models that try to mimic a specific artist. In coordination with the professional artist community, we deploy user studies to more than 1000 artists, assessing their views of AI art, as well as the efficacy of our tool, its usability and tolerability of perturbations, and robustness across different scenarios and against adaptive countermeasures. Both surveyed artists and empirical CLIP-based scores show that even at low perturbation levels (p=0.05), Glaze is highly successful at disrupting mimicry under normal conditions (\>92%) and against adaptive countermeasures (\>85%). (@shan2023glaze)

Jiaming Song, Chenlin Meng, and Stefano Ermon Denoising diffusion implicit models In *International Conference on Learning Representations*. OpenReview.net, 2021. **Abstract:** Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, yet they require simulating a Markov chain for many steps to produce a sample. To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a Markovian diffusion process. We construct a class of non-Markovian diffusion processes that lead to the same training objective, but whose reverse process can be much faster to sample from. We empirically demonstrate that DDIMs can produce high quality samples $10 \\}times$ to $50 \\}times$ faster in terms of wall-clock time compared to DDPMs, allow us to trade off computation for sample quality, and can perform semantically meaningful image interpolation directly in the latent space. (@song2020denoising)

Yang Song and Stefano Ermon Generative modeling by estimating gradients of the data distribution *Advances in Neural Information Processing Systems*, 32, 2019. **Abstract:** We introduce a new generative model where samples are produced via Langevin dynamics using gradients of the data distribution estimated with score matching. Because gradients can be ill-defined and hard to estimate when the data resides on low-dimensional manifolds, we perturb the data with different levels of Gaussian noise, and jointly estimate the corresponding scores, i.e., the vector fields of gradients of the perturbed data distribution for all noise levels. For sampling, we propose an annealed Langevin dynamics where we use gradients corresponding to gradually decreasing noise levels as the sampling process gets closer to the data manifold. Our framework allows flexible model architectures, requires no sampling during training or the use of adversarial methods, and provides a learning objective that can be used for principled model comparisons. Our models produce samples comparable to GANs on MNIST, CelebA and CIFAR-10 datasets, achieving a new state-of-the-art inception score of 8.87 on CIFAR-10. Additionally, we demonstrate that our models learn effective representations via image inpainting experiments. (@song2019generative)

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole Score-based generative modeling through stochastic differential equations In *International Conference on Learning Representations*. OpenReview.net, 2021. **Abstract:** Creating noise from data is easy; creating data from noise is generative modeling. We present a stochastic differential equation (SDE) that smoothly transforms a complex data distribution to a known prior distribution by slowly injecting noise, and a corresponding reverse-time SDE that transforms the prior distribution back into the data distribution by slowly removing the noise. Crucially, the reverse-time SDE depends only on the time-dependent gradient field (\\}aka, score) of the perturbed data distribution. By leveraging advances in score-based generative modeling, we can accurately estimate these scores with neural networks, and use numerical SDE solvers to generate samples. We show that this framework encapsulates previous approaches in score-based generative modeling and diffusion probabilistic modeling, allowing for new sampling procedures and new modeling capabilities. In particular, we introduce a predictor-corrector framework to correct errors in the evolution of the discretized reverse-time SDE. We also derive an equivalent neural ODE that samples from the same distribution as the SDE, but additionally enables exact likelihood computation, and improved sampling efficiency. In addition, we provide a new way to solve inverse problems with score-based models, as demonstrated with experiments on class-conditional generation, image inpainting, and colorization. Combined with multiple architectural improvements, we achieve record-breaking performance for unconditional image generation on CIFAR-10 with an Inception score of 9.89 and FID of 2.20, a competitive likelihood of 2.99 bits/dim, and demonstrate high fidelity generation of 1024 x 1024 images for the first time from a score-based generative model. (@song2020score)

Matthieu Urvoy, Dalila Goudia, and Florent Autrusseau Perceptual dft watermarking with improved detection and robustness to geometrical distortions *IEEE Transactions on Information Forensics and Security*, 9 (7): 1108–1119, 2014. **Abstract:** More than ever, the growing amount of exchanged digital contents calls for efficient and practical techniques to protect intellectual property rights.During the past two decades, watermarking techniques have been proposed to embed and detect information within these contents, with four key requirements at hand: robustness, security, capacity and invisibility.So far, researchers mostly focused on the first three, but seldom addressed the invisibility from a perceptual perspective and instead mostly relied on objective quality metrics.In this paper, a novel DFT watermarking scheme featuring perceptually-optimal visibility versus robustness is proposed.The watermark, a noise-like square patch of coefficients, is embedded by substitution within the Fourier domain; the amplitude component adjusts the watermark strength, and the phase component holds the information.A perceptual model of the Human Visual System (HVS) based on the Contrast Sensitivity Function (CSF) and a local contrast pooling is used to determine the optimal strength at which the mark reaches the visibility threshold.A novel blind detection method is proposed to assess the presence of the watermark.The proposed approach exhibits high robustness to various kind of attacks, including geometrical distortions.Experimental results show that the robustness of the proposed method is globally slightly better than state-of-the-art.A comparative study was conducted at the visibility threshold (from subjective data) and showed that the obtained performances are more stable across various kinds of contents. (@urvoy2014perceptual)

Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc N Tran, and Anh Tran Anti-dreambooth: Protecting users from personalized text-to-image synthesis In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 2116–2127, 2023. **Abstract:** Text-to-image diffusion models are nothing but a revolution, allowing anyone, even without design skills, to create realistic images from simple text inputs. With powerful personalization tools like DreamBooth, they can generate images of a specific person just by learning from his/her few reference images. However, when misused, such a powerful and convenient tool can produce fake news or disturbing content targeting any individual victim, posing a severe negative social impact. In this paper, we explore a defense system called Anti-DreamBooth against such malicious use of DreamBooth. The system aims to add subtle noise perturbation to each user’s image before publishing in order to disrupt the generation quality of any DreamBooth model trained on these perturbed images. We investigate a wide range of algorithms for perturbation optimization and extensively evaluate them on two facial datasets over various text-to-image model versions. Despite the complicated formulation of Dream-Booth and Diffusion-based text-to-image models, our methods effectively defend users from the malicious use of those models. Their effectiveness withstands even adverse conditions, such as model or prompt/term mismatching between training and testing. Our code will be available at https://github.com/VinAIResearch/Anti-DreamBooth.git. (@van2023anti)

Bram Wallace, Akash Gokul, and Nikhil Naik Edict: Exact diffusion inversion via coupled transformations In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 22532–22541, 2023. **Abstract:** Finding an initial noise vector that produces an input image when fed into the diffusion process (known as inversion) is an important problem in denoising diffusion models (DDMs), with applications for real image editing. The standard approach for real image editing with inversion uses denoising diffusion implicit models (DDIMs \[29\]) to deterministically noise the image to the intermediate state along the path that the denoising would follow given the original conditioning. However, DDIM inversion for real images is unstable as it relies on local linearization assumptions, which result in the propagation of errors, leading to incorrect image reconstruction and loss of content. To alleviate these problems, we propose Exact Diffusion Inversion via Coupled Transformations (EDICT), an inversion method that draws inspiration from affine coupling layers. EDICT enables mathematically exact inversion of real and model-generated images by maintaining two coupled noise vectors which are used to invert each other in an alternating fashion. Using Stable Diffusion \[25\], a state-of-the-art latent diffusion model, we demonstrate that EDICT successfully reconstructs real images with high fidelity. On complex image datasets like MS-COCO, EDICT reconstruction significantly outperforms DDIM, improving the mean square error of reconstruction by a factor of two. Using noise vectors inverted from real images, EDICT enables a wide range of image edits—from local and global semantic edits to image stylization—while maintaining fidelity to the original image structure. EDICT requires no model training/finetuning, prompt tuning, or extra data and can be combined with any pretrained DDM. (@wallace2023edict)

Ruoyu Wang, Yongqi Yang, Zhihao Qian, Ye Zhu, and Yu Wu Diffusion in diffusion: Cyclic one-way diffusion for text-vision-conditioned generation In *International Conference on Learning Representations*, 2024. **Abstract:** Originating from the diffusion phenomenon in physics that describes particle movement, the diffusion generative models inherit the characteristics of stochastic random walk in the data space along the denoising trajectory. However, the intrinsic mutual interference among image regions contradicts the need for practical downstream application scenarios where the preservation of low-level pixel information from given conditioning is desired (e.g., customization tasks like personalized generation and inpainting based on a user-provided single image). In this work, we investigate the diffusion (physics) in diffusion (machine learning) properties and propose our Cyclic One-Way Diffusion (COW) method to control the direction of diffusion phenomenon given a pre-trained frozen diffusion model for versatile customization application scenarios, where the low-level pixel information from the conditioning needs to be preserved. Notably, unlike most current methods that incorporate additional conditions by fine-tuning the base text-to-image diffusion model or learning auxiliary networks, our method provides a novel perspective to understand the task needs and is applicable to a wider range of customization scenarios in a learning-free manner. Extensive experiment results show that our proposed COW can achieve more flexible customization based on strict visual conditions in different application settings. Project page: https://wangruoyu02.github.io/cow.github.io/. (@wangdiffusion)

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli Image quality assessment: from error visibility to structural similarity *IEEE Transactions on Image Processing*, 13 (4): 600–612, 2004. **Abstract:** Objective methods for assessing perceptual image quality traditionally attempted to quantify the visibility of errors (differences) between a distorted image and a reference image using a variety of known properties of the human visual system. Under the assumption that human visual perception is highly adapted for extracting structural information from a scene, we introduce an alternative complementary framework for quality assessment based on the degradation of structural information. As a specific example of this concept, we develop a structural similarity index and demonstrate its promise through a set of intuitive examples, as well as comparison to both subjective ratings and state-of-the-art objective methods on a database of images compressed with JPEG and JPEG2000. A MATLAB implementation of the proposed algorithm is available online at http://www.cns.nyu.edu//spl sim/lcv/ssim/. (@wang2004image)

Ping Wei, Qing Zhou, Zichi Wang, Zhenxing Qian, Xinpeng Zhang, and Sheng Li Generative steganography diffusion *arXiv preprint arXiv:2305.03472*, 2023. **Abstract:** Generative steganography (GS) is an emerging technique that generates stego images directly from secret data. Various GS methods based on GANs or Flow have been developed recently. However, existing GAN-based GS methods cannot completely recover the hidden secret data due to the lack of network invertibility, while Flow-based methods produce poor image quality due to the stringent reversibility restriction in each module. To address this issue, we propose a novel GS scheme called "Generative Steganography Diffusion" (GSD) by devising an invertible diffusion model named "StegoDiffusion". It not only generates realistic stego images but also allows for 100\\}% recovery of the hidden secret data. The proposed StegoDiffusion model leverages a non-Markov chain with a fast sampling technique to achieve efficient stego image generation. By constructing an ordinary differential equation (ODE) based on the transition probability of the generation process in StegoDiffusion, secret data and stego images can be converted to each other through the approximate solver of ODE – Euler iteration formula, enabling the use of irreversible but more expressive network structures to achieve model invertibility. Our proposed GSD has the advantages of both reversibility and high performance, significantly outperforming existing GS methods in all metrics. (@wei2023generative)

Yuxin Wen, John Kirchenbauer, Jonas Geiping, and Tom Goldstein Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust In *Advances in Neural Information Processing Systems 36*, 2023. **Abstract:** Watermarking the outputs of generative models is a crucial technique for tracing copyright and preventing potential harm from AI-generated content. In this paper, we introduce a novel technique called Tree-Ring Watermarking that robustly fingerprints diffusion model outputs. Unlike existing methods that perform post-hoc modifications to images after sampling, Tree-Ring Watermarking subtly influences the entire sampling process, resulting in a model fingerprint that is invisible to humans. The watermark embeds a pattern into the initial noise vector used for sampling. These patterns are structured in Fourier space so that they are invariant to convolutions, crops, dilations, flips, and rotations. After image generation, the watermark signal is detected by inverting the diffusion process to retrieve the noise vector, which is then checked for the embedded signal. We demonstrate that this technique can be easily applied to arbitrary diffusion models, including text-conditioned Stable Diffusion, as a plug-in with negligible loss in FID. Our watermark is semantically hidden in the image space and is far more robust than watermarking alternatives that are currently deployed. Code is available at https://github.com/YuxinWenRick/tree-ring-watermark. (@wen2023tree)

Kyle Wiggers Microsoft pledges to watermark ai-generated images and videos 2023. URL <https://techcrunch.com/2023/05/23/microsoft-pledges-to-watermark-ai-generated-images-and-videos/>. **Abstract:** Microsoft says that it will launch new media provenance capabilities for Designer and Bing Image creator to indicate whether images are AI-generated. (@microsoft)

Raymond B Wolfgang and Edward J Delp A watermark for digital images In *Proceedings of IEEE International Conference on Image Processing*, volume 3, pages 219–222. IEEE, 1996. **Abstract:** The growth of new imaging technologies has created a need for techniques that can be used for copyright protection of digital images and video. One approach for copyright protection is to introduce an invisible signal, known as a digital watermark, into an image or video sequence. In this paper, we describe digital watermarking techniques, known as perceptually based watermarks, that are designed to exploit aspects of the the human visual system in order to provide a transparent (invisible), yet robust watermark. In the most general sense, any watermarking technique that attempts to incorporate an invisible mark into an image is perceptually based. However, in order to provide transparency and robustness to attack, two conflicting requirements from a signal processing perspective, more sophisticated use of perceptual information in the watermarking process is required. We describe watermarking techniques ranging from simple schemes which incorporate common-sense rules in using perceptual information in the watermarking process, to more elaborate schemes which adapt to local image characteristics based on more formal perceptual models. This review is not meant to be exhaustive; its aim is to provide the reader with an understanding of how the techniques have been evolving as the requirements and applications become better defined. (@wolfgang1996watermark)

Xiang-Gen Xia, Charles G Boncelet, and Gonzalo R Arce Wavelet transform based watermark for digital images *Optics Express*, 3 (12): 497–511, 1998. **Abstract:** In this paper, we introduce a new multiresolution watermarking method for digital images. The method is based on the discrete wavelet transform (DWT). Pseudo-random codes are added to the large coefficients at the high and middle frequency bands of the DWT of an image. It is shown that this method is more robust to proposed methods to some common image distortions, such as the wavelet transform based image compression, image rescaling/stretching and image halftoning. Moreover, the method is hierarchical. (@xia1998wavelet)

Cheng Xiong, Chuan Qin, Guorui Feng, and Xinpeng Zhang Flexible and secure watermarking for latent diffusion model In *Proceedings of ACM International Conference on Multimedia*, pages 1668–1676, 2023. **Abstract:** Since the significant advancements and open-source support of latent diffusion models (LDMs) in the field of image generation, numerous researchers and enterprises start fine-tuning the pre-trained models to generate specialized images for different objectives. However, the criminals may turn their attention to generate images by LDMs and then carry out illegal activities. The watermarking technique is a typical solution to deal with this problem. But, the post-hoc watermarking methods can be easily escaped to obtain the non-watermarked images, and the existing watermarking methods designed for LDMs can only embed a fixed message, i.e., the to-be-embedded message cannot be changed unless retraining the model. Therefore, in this work, we propose an end-to-end watermarking method based on the encoder-decoder (ENDE) and message-matrix. The message can be embedded into generated images through fusing the message-matrix and intermediate outputs in the forward propagation of image generation based on LDM. Thus, the message can be flexibly changed by utilizing the message-encoder to generate message-matrix, without training the LDM again. On the other hand, the security mechanism in our watermarking method can defeat the attack that the users may escape the message-matrix usage during image generation. A series of experiments demonstrate the effectiveness and the superiority of our watermarking method compared with SOTA methods. (@xiong2023flexible)

Jiwen Yu, Xuanyu Zhang, Youmin Xu, and Jian Zhang Cross: Diffusion model makes controllable, robust and secure image steganography *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Current image steganography techniques are mainly focused on cover-based methods, which commonly have the risk of leaking secret images and poor robustness against degraded container images. Inspired by recent developments in diffusion models, we discovered that two properties of diffusion models, the ability to achieve translation between two images without training, and robustness to noisy data, can be used to improve security and natural robustness in image steganography tasks. For the choice of diffusion model, we selected Stable Diffusion, a type of conditional diffusion model, and fully utilized the latest tools from open-source communities, such as LoRAs and ControlNets, to improve the controllability and diversity of container images. In summary, we propose a novel image steganography framework, named Controllable, Robust and Secure Image Steganography (CRoSS), which has significant advantages in controllability, robustness, and security compared to cover-based image steganography methods. These benefits are obtained without additional training. To our knowledge, this is the first work to introduce diffusion models to the field of image steganography. In the experimental section, we conducted detailed experiments to demonstrate the advantages of our proposed CRoSS framework in controllability, robustness, and security. (@yu2024cross)

Ning Yu, Vladislav Skripniuk, Sahar Abdelnabi, and Mario Fritz Artificial fingerprinting for generative models: Rooting deepfake attribution in training data In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 14448–14457, 2021. **Abstract:** Photorealistic image generation has reached a new level of quality due to the breakthroughs of generative adversarial networks (GANs). Yet, the dark side of such deepfakes, the malicious use of generated media, raises concerns about visual misinformation. While existing research work on deepfake detection demonstrates high accuracy, it is subject to advances in generation techniques and adversarial iterations on detection countermeasure techniques. Thus, we seek a proactive and sustainable solution on deepfake detection, that is agnostic to the evolution of generative models, by introducing artificial fingerprints into the models.Our approach is simple and effective. We first embed artificial fingerprints into training data, then validate a surprising discovery on the transferability of such fingerprints from training data to generative models, which in turn appears in the generated deepfakes. Experiments show that our fingerprinting solution (1) holds for a variety of cutting-edge generative models, (2) leads to a negligible side effect on generation quality, (3) stays robust against image-level and model-level perturbations, (4) stays hard to be detected by adversaries, and (5) converts deepfake detection and attribution into trivial tasks and outperforms the recent state-of-the-art baselines. Our solution closes the responsibility loop between publishing pre-trained generative model inventions and their possible misuses, which makes it independent of the current arms race. (@yu2021artificial)

Guoqiang Zhang, Jonathan P Lewis, and W Bastiaan Kleijn Exact diffusion inversion via bi-directional integration approximation *arXiv preprint arXiv:2307.10829*, 2023. **Abstract:** Recently, various methods have been proposed to address the inconsistency issue of DDIM inversion to enable image editing, such as EDICT \[36\] and Null-text inversion \[22\]. However, the above methods introduce considerable computational overhead. In this paper, we propose a new technique, named \\}emph{bi-directional integration approximation} (BDIA), to perform exact diffusion inversion with neglible computational overhead. Suppose we would like to estimate the next diffusion state $\\}boldsymbol{z}\_{i-1}$ at timestep $t_i$ with the historical information $(i,\\}boldsymbol{z}\_i)$ and $(i+1,\\}boldsymbol{z}\_{i+1})$. We first obtain the estimated Gaussian noise $\\}hat{\\}boldsymbol{\\}epsilon}}(\\}boldsymbol{z}\_i,i)$, and then apply the DDIM update procedure twice for approximating the ODE integration over the next time-slot $\[t_i, t\_{i-1}\]$ in the forward manner and the previous time-slot $\[t_i, t\_{t+1}\]$ in the backward manner. The DDIM step for the previous time-slot is used to refine the integration approximation made earlier when computing $\\}boldsymbol{z}\_i$. A nice property of BDIA-DDIM is that the update expression for $\\}boldsymbol{z}\_{i-1}$ is a linear combination of $(\\}boldsymbol{z}\_{i+1}, \\}boldsymbol{z}\_i, \\}hat{\\}boldsymbol{\\}epsilon}}(\\}boldsymbol{z}\_i,i))$. This allows for exact backward computation of $\\}boldsymbol{z}\_{i+1}$ given $(\\}boldsymbol{z}\_i, \\}boldsymbol{z}\_{i-1})$, thus leading to exact diffusion inversion. It is demonstrated with experiments that (round-trip) BDIA-DDIM is particularly effective for image editing. Our experiments further show that BDIA-DDIM produces markedly better image sampling qualities than DDIM for text-to-image generation. BDIA can also be applied to improve the performance of other ODE solvers in addition to DDIM. In our work, it is found that applying BDIA to the EDM sampling procedure produces consistently better performance over four pre-trained models. (@zhang2023exact)

Kevin Alex Zhang, Lei Xu, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni Robust invisible video watermarking with attention *arXiv preprint arXiv:1909.01285*, 2019. **Abstract:** The goal of video watermarking is to embed a message within a video file in a way such that it minimally impacts the viewing experience but can be recovered even if the video is redistributed and modified, allowing media producers to assert ownership over their content. This paper presents RivaGAN, a novel architecture for robust video watermarking which features a custom attention-based mechanism for embedding arbitrary data as well as two independent adversarial networks which critique the video quality and optimize for robustness. Using this technique, we are able to achieve state-of-the-art results in deep learning-based video watermarking and produce watermarked videos which have minimal visual distortion and are robust against common video processing operations. (@zhang2019robust)

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala Adding conditional control to text-to-image diffusion models In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 3836–3847, 2023. **Abstract:** We present ControlNet, a neural network architecture to add spatial conditioning controls to large, pretrained text-to-image diffusion models. ControlNet locks the production-ready large diffusion models, and reuses their deep and robust encoding layers pretrained with billions of images as a strong backbone to learn a diverse set of conditional controls. The neural architecture is connected with "zero convolutions" (zero-initialized convolution layers) that progressively grow the parameters from zero and ensure that no harmful noise could affect the finetuning. We test various conditioning controls, e.g., edges, depth, segmentation, human pose, etc., with Stable Diffusion, using single or multiple conditions, with or without prompts. We show that the training of ControlNets is robust with small (\<50k) and large (\>1m) datasets. Extensive results show that ControlNet may facilitate wider applications to control image diffusion models. (@zhang2023adding)

Xuandong Zhao, Kexun Zhang, Zihao Su, Saastha Vasan, Ilya Grishchenko, Christopher Kruegel, Giovanni Vigna, Yu-Xiang Wang, and Lei Li Invisible image watermarks are provably removable using generative ai Aug 2023. **Abstract:** Invisible watermarks safeguard images’ copyright by embedding hidden messages only detectable by owners. They also prevent people from misusing images, especially those generated by AI models. We propose a family of regeneration attacks to remove these invisible watermarks. The proposed attack method first adds random noise to an image to destroy the watermark and then reconstructs the image. This approach is flexible and can be instantiated with many existing image-denoising algorithms and pre-trained generative models such as diffusion models. Through formal proofs and empirical results, we show that all invisible watermarks are vulnerable to the proposed attack. For a particularly resilient watermark, RivaGAN, regeneration attacks remove 93-99% of the invisible watermarks while the baseline attacks remove no more than 3%. However, if we do not require the watermarked image to look the same as the original one, watermarks that keep the image semantically similar can be an alternative defense against our attack. Our finding underscores the need for a shift in research/industry emphasis from invisible watermarks to semantically similar ones. Code is available at https://github.com/XuandongZhao/WatermarkAttacker. (@zhao_zhang_su_vasan_grishchenko_kruegel_vigna_wang_li_2023)

Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Ngai-Man Cheung, and Min Lin A recipe for watermarking diffusion models *arXiv preprint arXiv:2303.10137*, 2023. **Abstract:** Diffusion models (DMs) have demonstrated advantageous potential on generative tasks. Widespread interest exists in incorporating DMs into downstream applications, such as producing or editing photorealistic images. However, practical deployment and unprecedented power of DMs raise legal issues, including copyright protection and monitoring of generated content. In this regard, watermarking has been a proven solution for copyright protection and content monitoring, but it is underexplored in the DMs literature. Specifically, DMs generate samples from longer tracks and may have newly designed multimodal structures, necessitating the modification of conventional watermarking pipelines. To this end, we conduct comprehensive analyses and derive a recipe for efficiently watermarking state-of-the-art DMs (e.g., Stable Diffusion), via training from scratch or finetuning. Our recipe is straightforward but involves empirically ablated implementation details, providing a foundation for future research on watermarking DMs. The code is available at https://github.com/yunqing-me/WatermarkDM. (@zhao2023recipe)

Zhenyu Zhou, Defang Chen, Can Wang, and Chun Chen Fast ode-based sampling for diffusion models in around 5 steps In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7777–7786, 2024. **Abstract:** Sampling from diffusion models can be treated as solving the corresponding ordinary differential equations (ODEs), with the aim of obtaining an accurate solution with as few number of function evaluations (NFE) as possible. Recently, various fast samplers utilizing higher-order ODE solvers have emerged and achieved better performance than the initial first-order one. However, these numerical methods inherently result in certain approximation errors, which significantly degrades sample quality with extremely small NFE (e.g., around 5). In contrast, based on the geometric observation that each sampling trajectory almost lies in a two-dimensional subspace embedded in the ambient space, we propose Approximate MEan-Direction Solver (AMED-Solver) that eliminates truncation errors by directly learning the mean direction for fast diffusion sampling. Besides, our method can be easily used as a plugin to further improve existing ODE-based samplers. Extensive experiments on image synthesis with the resolution ranging from 32 to 512 demonstrate the effectiveness of our method. With only 5 NFE, we achieve 6.61 FID on CIFAR-10, 10.74 FID on ImageNet 64$\\}times$64, and 13.20 FID on LSUN Bedroom. Our code is available at https://github.com/zju-pi/diff-sampler. (@zhou2024fast)

Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei Hidden: Hiding data with deep networks In *Proceedings of the European conference on computer vision*, pages 657–672, 2018. **Abstract:** Recent work has shown that deep neural networks are highly sensitive to tiny perturbations of input images, giving rise to adversar- ial examples . Though this property is usually considered a weakness of learned models, we explore whether it can be bene cial. We nd that neural networks can learn to use invisible perturbations to encode a rich amount of useful information. In fact, one can exploit this capability for the task of data hiding. We jointly train encoder and decoder networks, where given an input message and cover image, the encoder produces a visually indistinguishable encoded image, from which the decoder can recover the original message. We show that these encodings are compet- itive with existing data hiding algorithms, and further that they can be made robust to noise: our models learn to reconstruct hidden information in an encoded image despite the presence of Gaussian blurring, pixel- wise dropout, cropping, and JPEG compression. Even though JPEG is non-di erentiable, we show that a robust model can be trained using di erentiable approximations. Finally, we demonstrate that adversarial training improves the visual quality of encoded images. (@zhu2018hidden)

Ye Zhu, Yu Wu, Kyle Olszewski, Jian Ren, Sergey Tulyakov, and Yan Yan Discrete contrastive diffusion for cross-modal music and image generation In *International Conference on Learning Representations (ICLR)*, 2023. **Abstract:** Diffusion probabilistic models (DPMs) have become a popular approach to conditional generation, due to their promising results and support for cross-modal synthesis. A key desideratum in conditional synthesis is to achieve high correspondence between the conditioning input and generated output. Most existing methods learn such relationships implicitly, by incorporating the prior into the variational lower bound. In this work, we take a different route – we explicitly enhance input-output connections by maximizing their mutual information. To this end, we introduce a Conditional Discrete Contrastive Diffusion (CDCD) loss and design two contrastive diffusion mechanisms to effectively incorporate it into the denoising process, combining the diffusion training and contrastive learning for the first time by connecting it with the conventional variational objectives. We demonstrate the efficacy of our approach in evaluations with diverse multimodal conditional synthesis tasks: dance-to-music generation, text-to-image synthesis, as well as class-conditioned image synthesis. On each, we enhance the input-output correspondence and achieve higher or competitive general synthesis quality. Furthermore, the proposed approach improves the convergence of diffusion models, reducing the number of required diffusion steps by more than 35% on two benchmarks, significantly increasing the inference speed. (@zhu2022discrete)

Ye Zhu, Yu Wu, Zhiwei Deng, Olga Russakovsky, and Yan Yan Boundary guided learning-free semantic control with diffusion models *Advances in Neural Information Processing Systems*, 36, 2024. **Abstract:** Applying pre-trained generative denoising diffusion models (DDMs) for downstream tasks such as image semantic editing usually requires either fine-tuning DDMs or learning auxiliary editing networks in the existing literature. In this work, we present our BoundaryDiffusion method for efficient, effective and light-weight semantic control with frozen pre-trained DDMs, without learning any extra networks. As one of the first learning-free diffusion editing works, we start by seeking a comprehensive understanding of the intermediate high-dimensional latent spaces by theoretically and empirically analyzing their probabilistic and geometric behaviors in the Markov chain. We then propose to further explore the critical step for editing in the denoising trajectory that characterizes the convergence of a pre-trained DDM and introduce an automatic search method. Last but not least, in contrast to the conventional understanding that DDMs have relatively poor semantic behaviors, we prove that the critical latent space we found already exhibits semantic subspace boundaries at the generic level in unconditional DDMs, which allows us to do controllable manipulation by guiding the denoising trajectory towards the targeted boundary via a single-step operation. We conduct extensive experiments on multiple DPMs architectures (DDPM, iDDPM) and datasets (CelebA, CelebA-HQ, LSUN-church, LSUN-bedroom, AFHQ-dog) with different resolutions (64, 256), achieving superior or state-of-the-art performance in various task scenarios (image semantic editing, text-based editing, unconditional semantic control) to demonstrate the effectiveness. (@zhu2024boundary)

Ye Zhu, Yu Wu, Nicu Sebe, and Yan Yan Vision+ x: A survey on multimodal learning in the light of data *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024. **Abstract:** We are perceiving and communicating with the world in a multisensory manner, where different information sources are sophisticatedly processed and interpreted by separate parts of the human brain to constitute a complex, yet harmonious and unified sensing system. To endow the machines with true intelligence, multimodal machine learning that incorporates data from various sources has become an increasingly popular research area with emerging technical advances in recent years. In this paper, we present a survey on multimodal machine learning from a novel perspective considering not only the purely technical aspects but also the intrinsic nature of different data modalities. We analyze the commonness and uniqueness of each data format mainly ranging from vision, audio, text, and motions, and then present the methodological advancements categorized by the combination of data modalities, such as Vision+Text, with slightly inclined emphasis on the visual data. We investigate the existing literature on multimodal learning from both the representation learning and downstream application levels, and provide an additional comparison in the light of their technical connections with the data nature, e.g., the semantic consistency between image objects and textual descriptions, and the rhythm correspondence between video dance moves and musical beats. We hope that the exploitation of the alignment as well as the existing gap between the intrinsic nature of data modality and the technical designs, will benefit future research studies to better address a specific challenge related to the concrete multimodal task, prompting a unified multimodal machine learning framework closer to a real human intelligence system. (@zhu2024vision+)

</div>

# Appendix [app:appendix]

# Scheme details [app:design]

## Image watermark design [app:wi_design]

To make the watermark less visible and more resistant to alterations, we embed the watermark into the frequency domain of the selected diffusion latent. Frequency domain watermarks are proven to be robust against common manipulations like cropping and compression and resilient against geometric distortions, such as scaling and rotation. Draw inspiration from Tree-Ring watermarks, we use a radiating watermark pattern, where the watermark information within each frequency band holds equal values. This design choice enhances the watermark’s robustness against image rotations. Specifically, after each optimization round of the image watermark, the values within a specific frequency band are averaged. This averaged pattern is then used for further prompt signal optimization and another round of adversarial optimization.

## Prompt signal design [app:wp_design]

Our scheme is based on the classifier-free guidance technique, where the generation relies on both unconditional and conditional predictions. The predicted noise of \\(x_t\\) at step \\(t\\) is defined as:

\\[\tilde{\epsilon }_\theta (x_t,t,\psi(p))=\eta \cdot \epsilon _\theta (x_t,t,\psi (p))+ (1-\eta )\cdot \epsilon _\theta (x_t,t,\psi(\emptyset)),\\] where \\(\eta\\) is the guidance scale parameter and \\(p\\) is the input text condition. In this way, the model can maintain its original ability to remove noise and the new function of generating specific content.

## Optimization algorithm design [app:alg_design]

The details of the adversarial optimization algorithm are presented in  
efalg:opt. Initially, the image watermark \\(w_i\\) is randomly sampled, and guidance \\(w_p\\) is set to `NULL` (representing no text prompt). In each round, we randomly select a generated sample \\(x_0\\) and obtain the noise representation \\(x_t\\) at the watermark embedding point. Then both \\(w_i\\) and \\(w_p\\) are optimized alternatively in an adversarial manner. We experimentally set the hyperparameters \\(\alpha\\) as 1.0, \\(\beta\\) as 1.0, and \\(\lambda\\) as 0.005.

<figure id="alg:opt">
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p>Dataset <span class="math inline">𝒳, 𝒫</span>; max epoch <span class="math inline"><em>N</em></span>; hyper-parameters <span class="math inline"><em>α</em>, <em>β</em>, <em>λ</em></span>; watermark mask <span class="math inline">𝕄</span> Optimized watermark pair <span class="math inline"><em>w</em><sub><em>i</em></sub>, <em>w</em><sub><em>p</em></sub></span>; Initialization<br />
<span class="math inline"><em>w</em><sub><em>i</em></sub><sup>0</sup> ← <code>rand_init</code>(<em>w</em><sub><em>i</em></sub>)</span> ;<br />
<span class="math inline"><em>w</em><sub><em>p</em></sub><sup>0</sup> ← <em>ψ</em>(<code>NULL</code>)</span> ;<br />
<span class="math inline"><em>k</em> ← 0</span> ;<br />
// get sample<br />
<span class="math inline"><em>x</em><sub>0</sub>, <em>p</em> ∼ 𝒳, 𝒫</span><br />
//get <span class="math inline"><em>x</em><sub><em>t</em></sub></span> from <span class="math inline"><em>x</em><sub>0</sub></span><br />
<span class="math inline">$x_t \leftarrow \sqrt[]{\bar{\alpha }_t }x_0+\sqrt[]{1-\bar{\alpha }_t }\epsilon ,\epsilon \sim \mathcal{N}(0,\mathbf{I} ), t \sim \texttt{Uniform}(200,300)$</span><br />
// image watermark optimization<br />
<span class="math inline">$w_i^{k+  1}=\mathop{\arg\min}\limits_{w_i}(\alpha \ell _{ret}+\beta \ell _{cons}-\lambda \left \| w_i^k \right \|)$</span> // prompt guidance optimization<br />
<span class="math inline">$w_p^{k+  1}=\mathop{\arg\min}\limits_{w_p}(\alpha \ell _{ret}+\beta \ell _{cons})$</span><br />
<span class="math inline"><em>k</em> ← <em>k</em> + 1</span> <span class="math inline">(<em>w</em><sub><em>i</em></sub><sup><em>k</em></sup>, <em>w</em><sub><em>p</em></sub><sup><em>k</em></sup>)</span></p>
</div>
<figcaption>Adversarial Optimization Algorithm</figcaption>
</figure>

# Implementation details [app:imp_details]

## Details about evaluation metric [app:metric]

#### Setting for AUC computing.

The AUC-ROC (Area Under the ROC Cure) metric is a statistical measure used to evaluate the performance of a binary classification problem, which is watermarked or not here. The ROC Curve is created by plotting the fraction of true positive results against the fraction of false positive results at various threshold settings. The AUC summarizes the overall performance across all possible thresholds. A higher AUC value means the test is more accurate in making this distinction. And an AUC of 1.0 represents perfect discrimination. For both our method and Tree-Ring watermarking, we compare the extracted image watermarks using the L1 distance. For the other three steganography-based methods, we utilize the Hamming distance between the implanted and decoded binary sequences, as these methods typically operate on binary data representations.

#### Setting for FID computing.

For Stable Diffusion, we generate 5,000 watermarked images and calculate FID against the MS-COCO-2017 dataset `\cite{lin2014microsoft}`{=latex}. For the ImageNet Diffusion model, we calculate FID using 10,000 watermarked images against the ImageNet-1K training dataset `\cite{deng2009imagenet}`{=latex}.

#### Setting for CLIP computing.

For both models, we test 1,000 images using the OpenCLIP-ViT model `\cite{cherti2023reproducible}`{=latex}. For Stable Diffusion, we work with the ground-truth text prompts, while for the ImageNet model, we construct prompts like "a photo of x", where "x" is the category of the generated image.

#### About pixel-level metrics.

The content watermarking scheme of ROBIN doesn’t aim for exact replication of the original image. Instead, it strives for a visually similar "alternative generation" that maintains both image quality and semantic integrity. While traditional watermarking schemes utilized metrics like PSNR/SSIM to assess image distortion introduced by the watermark (treated as an additional signal), we utilize them in this research as supplementary indicators to reflect the degree of semantic preservation within the watermarked image. Essentially, the higher the similarity between the watermarked and original image, the less semantic impact the watermark has introduced.

# More experimental results

& DwtDct `\cite{chen_2009}`{=latex} & 2.681 & 0.056 & 3.492 & 0.012  
& DwtDctSvd `\cite{navas2008dwt}`{=latex} & 2.749 & 0.233 & 3.511 & 0.058  
& RivaGAN `\cite{zhang2019robust}`{=latex} & 3.342 & 0.437 & 3.661 & 0.109  
& StableSig `\cite{fernandez2023stable}`{=latex} & 2.614 & 0.112 & - & -  
& Tree-Ring `\cite{wen2023tree}`{=latex} & 2.617 & 2.599 & 3.482 & 3.963  
& ROBIN & 2.682 & 0.531 & 3.592 & 0.986  

## Time overhead [app:time]

We evaluate the time cost associated with different watermarking schemes. The results are presented in    
eftab:time_comp. Traditional post-processing methods exhibit similar time requirements for watermark addition and verification. The simple DwtDct method demonstrates the fastest performance, achieving both addition and validation times of less than 0.1s. DwtDctSvd exhibits a 3\\(\times\\) slowdown compared to DwtDct, while RivaGAN is 10\\(\times\\) slower. Notably, the runtime of these methods is heavily influenced by the input image size. For in-processing watermarking, StableSig directly fine-tunes the model, incurring no additional time overhead during the generation process. The Tree-Ring method introduces minimal impact (0.003s) on generation time by solely modifying the initial random vector. However, verification necessitates reversing the entire generation process, resulting in significant time consumption (2.6s for Stable Diffusion and 3.9s for Imagenet Diffusion). ROBIN employs a one-shot approach for watermark embedding during the intermediate diffusion stage. The impact on generation time arises from the introduction of additional guidance calculations, resulting in a minimal overhead of 0.068s and 0.113s, which is negligible compared to the generation time of 2.614s and 3.479s for the two models. Verification of ROBIN watermarks requires reversing only a limited number of generation steps, resulting in consumption times of 0.531s and 0.986s for the two models, which are considerably lower compared to the Tree-Ring watermark.

## Reconstruction attacks [app:more_attack]

<div id="tab:recons-att" markdown="1">

|  Method   | VAE-Bmshj2018 | VAE-Cheng2020 | Diffusion model |
|:---------:|:-------------:|:-------------:|:---------------:|
| Tree-Ring |     0.992     |     0.993     |      0.996      |
|   ROBIN   |   **0.998**   |   **0.999**   |    **0.997**    |

Watermark verification (AUC) under reconstruction attack.

</div>

We evaluate the performance of ROBIN under different variants of reconstruction attacks `\cite{zhao_zhang_su_vasan_grishchenko_kruegel_vigna_wang_li_2023}`{=latex}. As shown in  
eftab:recons-att, ROBIN consistently exhibits stronger robustness under these adversarial conditions.

## Application to noise-to-image models

ROBIN can also be applied to noise-to-image generation models, as it does not rely on the original text prompt input. Given that large-scale pretrained diffusion models are typically conditional generative models, we chose to use the unconditional capability of Stable Diffusion to simulate the noise-to-image generation process for this evaluation.

We evaluate ROBIN on the unconditional generation of Stable Diffusion, where the original text is set to NULL (representing no text prompt). In this setup, the image is generated unconditionally before the watermark injection point. After that, we still utilize our watermarking hiding prompt embedding to guide the generation process and actively erase the watermark. The results in   
eftab:noise2img indicate that ROBIN can still function well in noise-to-image generation.

<div id="tab:noise2img" markdown="1">

| Diffusion Type | Clean | Blur  | Noise | JPEG | Bright | Rotation | Crop |  Avg  |
|:--------------:|:-----:|:-----:|:-----:|:----:|:------:|:--------:|:----:|:-----:|
| Noise-to-Image |  1.0  | 0.996 | 0.997 | 1.0  | 0.963  |  0.999   | 1.0  | 0.993 |

Watermark verification (AUC) on noise-to-image generation.

</div>

## Pixel-level optimization

In our scheme, the watermark is embedded in the latent space while the loss function is calculated at the pixel level. We believe that this approach, which combines pixel-level alignment with latent space optimization, is beneficial for improving robustness.

This is because different latent representations can map to similar pixel-level expressions, allowing us to find a latent code that maps to visually the same image but also contains robust watermarks. This provides more opportunities to embed strong and robust watermark signals without introducing noticeable visual artifacts. The benefits of this optimization method are evident when we actively aim for concealment, a feature not supported by other watermarking methods.

To further validate our approach, we also test a variant of the ROBIN scheme where the loss function is computed at the latent level rather than the pixel level. The results presented in  
eftab:diff-opt demonstrate that latent-level alignment slightly decreases the robustness of the watermark, thereby underscoring the effectiveness of our pixel-level alignment strategy.

<div id="tab:diff-opt" markdown="1">

|      Alignment      | Clean | Blur  | Noise | JPEG  | Bright | Rotation | Crop  |  Avg  |
|:-------------------:|:-----:|:-----:|:-----:|:-----:|:------:|:--------:|:-----:|:-----:|
|    Latent-level     | 1.000 | 0.999 | 0.940 | 0.999 | 0.974  |  0.927   | 0.994 | 0.972 |
| Pixel-level (ROBIN) | 1.000 | 0.999 | 0.954 | 0.999 | 0.975  |  0.957   | 0.994 | 0.983 |

Watermark verification (AUC) on different optimization settings.

</div>

## More qualitative results [app:qual_result]

<figure id="fig:attack_img">
<img src="./figures/attack_samples.png"" />
<figcaption>Samples under different attacks.</figcaption>
</figure>

<figure id="fig:comb-att">
<img src="./figures/comb-att.png"" />
<figcaption>Samples under different number of attacks applied at the same time. The sequence of attacks performed on the above images is Gaussian blur with radius 4, JEPG compression with quality 25, color jitter with brightness 6, random cropping of 75%, and random rotation of 75 degrees.</figcaption>
</figure>

<figure id="fig:compare_stable">
<img src="./figures/more_sample_stable1.png"" />
<figcaption>More qualitative comparison results with Tree-Ring watermarks for Stable Diffusion. </figcaption>
</figure>

<figure id="fig:compare_image">
<img src="./figures/more_sample_imagenet1.png"" />
<figcaption>More qualitative comparison results with Tree-Ring watermarks for the ImageNet Diffusion model. </figcaption>
</figure>

<figure id="fig:img_point">
<img src="./figures/sample_embed_point.png"" />
<figcaption>Generated images with the watermark embedded at different diffusion stages. Clean represents the images that are generated without watermarking. <span class="math inline"><em>X</em><sub><em>T</em></sub></span> means the watermark is embedded into the initial noise, and <span class="math inline"><em>X</em><sub>0</sub></span> means the watermark is implanted in the final generated image. </figcaption>
</figure>

# NeurIPS Paper Checklist [neurips-paper-checklist]

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification: The main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope.

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: Please see    
    efsec:limitation.

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

14. Justification: The paper does not include theoretical results.

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

19. Justification: The codes will be available at <https://github.com/Hannah1102/ROBIN>.

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

24. Justification: The codes will be available at <https://github.com/Hannah1102/ROBIN> and we use open source model and data, which are cited correctly in the main paper.

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

29. Justification: Please see    
    efsec:setting,    
    efapp:design and    
    efapp:imp_details.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: Please see    
    eftab:quality.

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

39. Justification: Please see    
    efsec:setting.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification: The research conducted in the paper conform with the NeurIPS Code of Ethics.

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: Please see    
    efsec:social.

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

54. Justification: The paper poses no such risks.

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification: All datasets we used are public and cited properly.

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

64. Justification: The codes and data will be available at <https://github.com/Hannah1102/ROBIN>.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification: The paper does not involve crowdsourcing nor research with human subjects.

70. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

73. Answer:

74. Justification: The paper does not involve crowdsourcing nor research with human subjects.

75. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

[^1]: Corresponding author
