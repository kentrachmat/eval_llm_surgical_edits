# COVE: Unleashing the Diffusion Feature Correspondence for Consistent Video Editing

## Abstract

Video editing is an emerging task, in which most current methods adopt the pre-trained text-to-image (T2I) diffusion model to edit the source video in a zero-shot manner. Despite extensive efforts, maintaining the temporal consistency of edited videos remains challenging due to the lack of temporal constraints in the regular T2I diffusion model. To address this issue, we propose **CO**rrespondence-guided **V**ideo **E**diting (COVE), leveraging the inherent diffusion feature correspondence to achieve high-quality and consistent video editing. Specifically, we propose an efficient sliding-window-based strategy to calculate the similarity among tokens in the diffusion features of source videos, identifying the tokens with high correspondence across frames. During the inversion and denoising process, we sample the tokens in noisy latent based on the correspondence and then perform self-attention within them. To save GPU memory usage and accelerate the editing process, we further introduce the temporal-dimensional token merging strategy, which can effectively reduce redundancy. COVE can be seamlessly integrated into the pre-trained T2I diffusion model without the need for extra training or optimization. Extensive experiment results demonstrate that COVE achieves the start-of-the-art performance in various video editing scenarios, outperforming existing methods both quantitatively and qualitatively. The code will be release at <https://github.com/wangjiangshan0725/COVE>

<figure>
<embed src="fig/fig_teaser.pdf" />
<figcaption>We propose <strong>CO</strong>rrespondence-guided <strong>V</strong>ideo <strong>E</strong>diting <span>(COVE)</span>, which leverages the correspondence information of the diffusion feature to achieve consistent and high-quality video editing. Our method is capable of generating high-quality edited videos with various kinds of prompts (style, category, background, etc.) while effectively preserving temporal consistency in generated videos.</figcaption>
</figure>

# Introduction [sec:intro]

Diffusion models `\cite{ho2020denoising, sohl2015deep,song2020score}`{=latex} have shown exceptional performance in image generation `\cite{rombach2022high}`{=latex}, thereby inspiring their application in the field of image editing `\cite{brooks2023instructpix2pix,hertz2022prompt,cao2023masactrl,parmar2023zero,tumanyan2023plug,guo2023focus}`{=latex}. These approaches typically leverage a pre-trained Text-to-Image (T2I) stable diffusion model `\cite{rombach2022high}`{=latex}, using DDIM `\cite{song2020denoising}`{=latex} inversion to transform source images into noise, which is then progressively denoised under the guidance of a prompt to generate the edited image.

Despite satisfactory performance in image editing, achieving high-quality video editing remains challenging. Specifically, unlike the well-established open-source T2I stable diffusion models `\cite{rombach2022high}`{=latex}, comparable T2V diffusion models are not as mature due to the difficulty of modeling complicated temporal motions, and training a T2V model from scratch demands substantial computational resources `\cite{ho2022imagen,ho2022video,singer2022make}`{=latex}. Consequently, there is a growing focus on adapting the pre-trained T2I diffusion for video editing `\cite{geyer2023tokenflow,kara2023rave,cong2023flatten,yang2023rerender,yang2024fresco,qi2023fatezero}`{=latex}. In this case, maintaining temporal consistency in edited videos is one of the biggest challenges, which requires the generated frames to be stylistically coherent and exhibit smooth temporal transitions, rather than appearing as a series of independent images. Numerous methods have been working on this topic while still facing various limitations, such as the inability to ensure fine-grained temporal consistency (leading to flickering `\cite{kara2023rave,qi2023fatezero}`{=latex} or blurring `\cite{geyer2023tokenflow}`{=latex} in generated videos), requiring additional components `\cite{hu2023videocontrolnet,yang2023rerender,cong2023flatten,yang2024fresco}`{=latex} or needing extra training or optimization `\cite{yang2024fresco,wu2023tune,liew2023magicedit}`{=latex}, etc.

<figure id="fig:compare_intro">
<embed src="fig/fig_compare_intro_zong.pdf" style="width:50.0%" />
<figcaption><strong>Comparison between COVE (our method) and previous methods</strong><span class="citation" data-cites="cong2023flatten yang2024fresco"></span>.</figcaption>
</figure>

In this work, our goal is to achieve highly consistent video editing by leveraging the intra-frame correspondence relationship among tokens, which is intuitively closely related to the temporal consistency of videos: If corresponding tokens across frames exhibit high similarity, the resulting video will thus demonstrate high temporal consistency. Taking a video of a man as an example, if the token representing his nose has high similarity across frames, his nose will be unlikely to deform or flicker throughout the video. However, how to obtain accurate correspondence information among tokens is still largely under-explored in existing works, although the intrinsic characteristic of the video editing task (i.e., the source video and edited video are expected to share similar motion and semantic layout) determines that it naturally exists in the source video. Some previous methods `\cite{cong2023flatten, yang2024fresco}`{=latex} leverage a pre-trained optical-flow model to obtain the flowing trajectory of each token across frames, which can be seen as a kind of coarse correspondence information. Despite the self-attention among tokens in the same trajectory can enhance the temporal consistency of the edited video, it still encounters two primary limitations: Firstly, these methods heavily rely on a highly accurate pre-trained optical-flow model to obtain the correspondence relationship of tokens, which is not available in many scenarios `\cite{jonschkowski2020matters}`{=latex}. Secondly, supposing we have access to an extremely accurate optical-flow model, it is still only able to obtain the coarse one-to-one correspondence among tokens in different frames (<a href="#fig:compare_intro" data-reference-type="ref+Label" data-reference="fig:compare_intro">1</a>a), which would lead to the loss of information because one token is highly likely to correspond to multiple tokens in other frames in most cases (<a href="#fig:compare_intro" data-reference-type="ref+Label" data-reference="fig:compare_intro">1</a>b).

Addressing these problems, we notice that the inherent diffusion features naturally contain precise correspondence information. For instance, it is easy to find the corresponding points between two images by extracting their diffusion features and calculating the cosine similarity between tokens `\cite{tang2023emergent}`{=latex}. However, until now none of the existing works have successfully utilized this characteristic in more complicated and challenging tasks such as video editing. In this paper, we propose COVE, which is the first work unleashing the potential of inherent diffusion feature correspondence to significantly enhance the quality and temporal consistency in video editing. Given a source video, we first extract the diffusion feature of each frame. Then for each token in the diffusion feature, we obtain its corresponding tokens in other frames based on their similarity. Within this process, we propose a sliding-window-based approach to ensure computational efficiency. In our sliding-window-based method, for each token, it is only required to calculate the similarity between it and the tokens in the next frame located within a small window, identifying the tokens with the top \\(K\\) (\\(K>1\\)) highest similarity. After the correspondence calculation process, for each token, the coordinates of its \\(K\\) corresponding tokens in each other frame can be obtained. During the inversion and denoising process, we sample the tokens in noisy latents based on the obtained coordinates. To reduce the redundancy and accelerate the editing process, token merging is applied in the temporal dimension, which is followed by self-attention. Our method can be seamlessly integrated into the off-the-shelf T2I diffusion model without extra training or optimization. Extensive experiments demonstrate that COVE significantly improves both the quality and the temporal consistency of generated videos, outperforming a wide range of existing methods and achieving state-of-the-art results.

# Related Works

## Diffusion-based Image and Video Generation.

Diffusion Models `\cite{ho2020denoising, sohl2015deep,song2020score}`{=latex} have recently showcased impressive results in image generation, which generates the image through gradual denoising from the standard Gaussian noise`\cite{croitoru2023diffusion,dhariwal2021diffusion,nichol2021improved,guo2023zero, rombach2022high,song2020denoising,guo2023smooth}`{=latex}. A large number of efforts on diffusion models `\cite{ho2022classifier,karras2022elucidating,salimans2022progressive}`{=latex} has enabled it to be applied to numerous scenarios `\cite{avrahami2022blended,gal2022image,kawar2022denoising,li2022srdiff,lugmayr2022repaint,meng2021sdedit,mokady2023null,fang2024real,ruiz2023dreambooth,chen2024follow,guo2024everything, ma2024followyourclick,lin2023consistent123, guo2024refir,he2023reti,he2024diffusion}`{=latex}. With the aid of large-scale pretraining `\cite{radford2021learning,schuhmann2022laion}`{=latex}, text-to-image diffusion models exhibit remarkable progress in generating diverse and high-quality images `\cite{nichol2021glide, xue2024followyourposev2, ramesh2022hierarchical,rombach2022high,saharia2022photorealistic,guo2022assessing, ma2024followyouremoji, guo2024refir}`{=latex}. ControlNet `\cite{zhang2023adding}`{=latex} enables users to provide structure or layout information for precise generation. Naturally, diffusion models have found application in video synthesis, often by integrating temporal layers into image-based DMs `\cite{blattmann2023align,ho2022imagen,ho2022video,xiang2023versvideo,chen2023seine}`{=latex}. Despite successes in unconditional video generation `\cite{ho2022video,yu2023video, ma2023magicstick}`{=latex}, text-to-video diffusion models lag behind their image counterparts.

## Text-to-Video Editing.

There are increasing works adopting the pre-trained text-to-image diffusion model to the video editing task `\cite{liu2023video,wang2023zero,wu2023tune,ma2024follow,guo2023faceclip}`{=latex}, where keeping the temporal consistency in the generated video is the most challenging. Recently, a large number of works focusing on zero-shot video editing has been proposed. FateZero `\cite{qi2023fatezero}`{=latex} proposes to use attention blending to achieve high-quality edited videos while struggling to edit long videos. TokenFlow `\cite{geyer2023tokenflow}`{=latex} reduces the effects of flickering through the linear combinations between diffusion features, while the smoothing strategy can cause blurring in the generated video. RAVE `\cite{kara2023rave}`{=latex} proposes the randomized noise shuffling method, suffering the problem of fine details flickering. There are also a large number of methods that enhance the temporal consistency with the aid of pre-trained optical-flow models `\cite{yang2024fresco,yang2023rerender,cong2023flatten,hu2023videocontrolnet}`{=latex}. Although the effectiveness of them, all of them severely rely on a pre-trained optical-flow model. Recent works `\cite{tang2023emergent}`{=latex} illustrate that the diffusion feature contains rich correspondence information. Although VideoSwap `\cite{gu2023videoswap}`{=latex} adopts this characteristic by tracking the key points across frames, it still needs users to provide the key points as the extra addition manually.

# Method

In this section, we will introduce COVE in detail, which can be seamlessly integrated into the pre-trained T2I diffusion model for high-quality and consistent video editing without the need for training or optimization (<a href="#fig:pipe" data-reference-type="ref+Label" data-reference="fig:pipe">2</a>). Specifically, given a source video, we first extract the diffusion feature of each frame using the pre-trained T2I diffusion model. Then, we calculate the one-to-many correspondence of each token across frames based on cosine similarity (<a href="#fig:pipe" data-reference-type="ref+Label" data-reference="fig:pipe">2</a>a). To reduce resource consumption during correspondence calculation, we further introduce an efficient sliding-window-based strategy (<a href="#fig:slding" data-reference-type="ref+Label" data-reference="fig:slding">3</a>). During each timestep of inversion and denoising in video editing, the tokens in the noisy latent are sampled based on the correspondence and then merged. Through the self-attention among merged tokens (<a href="#fig:pipe" data-reference-type="ref+Label" data-reference="fig:pipe">2</a>b), the quality and temporal consistency of edited videos are significantly enhanced.

<figure id="fig:pipe">
<embed src="fig/fig_pipeline.pdf" />
<figcaption><strong>The overview of COVE.</strong> (a). Given a source video, we extract the diffusion feature of each frame using the pre-trained T2I model and calculate the correspondence among tokens (detailed in <a href="#fig:slding" data-reference-type="ref+Label" data-reference="fig:slding">3</a>). (b). During the video editing process, we sample the tokens in noisy latent based on correspondence and apply self-attention among them. (c). The correspondence-guided attention can be seamlessly integrated into the T2I diffusion model for consistent and high-quality video editing.</figcaption>
</figure>

## Preliminary

**Diffusion Models.** DDPM `\cite{ho2020denoising}`{=latex} is the latent generative model trained to reconstruct a fixed forward Markov chain \\(x_1, \ldots, x_T\\). Given the data distribution \\(x_0 \sim q(x_0)\\), the Markov transition \\(q(x_t|x_{t-1})\\) is defined as a Gaussian distribution with a variance schedule \\(\beta_t \in (0, 1)\\). \\[q(\bm{x}_t | \bm{x}_{t-1}) = \mathcal{N}(\bm{x}_t; \sqrt{1-\beta_t}\bm{x}_{t-1}, \beta_t \bm{\text{I}}).
\label{eq:ddpm_forward}\\] To generate the Markov chain \\(x_0, \cdots, x_T\\), DDPM leverages the reverse process with a prior distribution \\(p(x_T) = \mathcal{N}(x_T; 0, \mathbb{I})\\) and Gaussian transitions. A neural network \\(\epsilon_{\theta}\\) is trained to predict noises, ensuring that the reverse process is close to the forward process. \\[p_{\theta}(\bm{x}_{t-1}|\bm{x}_t) = \mathcal{N}(\bm{x}_{t-1}; \mu_{\theta}(\bm{x}_t, \bm{\tau}, t),  \Sigma_{\theta}(\bm{x}_t, \bm{\tau}, t) ),
\label{eq:ddpm_backward}\\] where \\(\bm{\tau}\\) indicates the textual prompt. \\(\mu_{\theta}\\) and \\(\Sigma_{\theta}\\) are predicted by the denoising model \\(\epsilon_{\theta}\\). Since the diffusion and denoising process in the pixel space is computationally extensive, latent diffusion `\cite{rombach2022high}`{=latex} is proposed to address this issue by performing these processes in the latent space of a VAE `\cite{kingma2013auto}`{=latex}.

**DDIM Inversion.** DDIM can convert random noise to a deterministic \\(\bm{x}_0\\) during sampling `\citep{song2020denoising, dhariwal2021diffusion}`{=latex}. The inversion process in deterministic DDIM can be formulated as follows: \\[\bm{x}_{t+1} = \sqrt{\frac{\alpha_{t+1}}{\alpha_{t}}} \bm{x}_{t} + \sqrt{\alpha_{t+1}} \left( \sqrt{\frac{1}{\alpha_{t+1}-1}}-\sqrt{\frac{1}{\alpha_{t}}-1}  \right) \epsilon_{\theta}(\bm{x}_{t}),
\label{eq:ddim_inverse}\\] where \\(\alpha_{t}\\) denotes \\(\prod^t_{i=1}(1-\beta_i)\\). The inversion process of DDIM is utilized to transform the input \\(\bm{x}_{0}\\) into \\(\bm{x}_{T}\\), facilitating subsequent tasks such as reconstruction and editing.

## Correspondence Acquisition

As discussed in <a href="#sec:intro" data-reference-type="ref+Label" data-reference="sec:intro">1</a>, intra-frame correspondence is crucial for the quality and temporal consistency of edited videos while remaining largely under-explored in existing works. In this section, we introduce our method for obtaining correspondence relationships among tokens across frames.

**Diffusion Feature Extraction.** Given a source video \\(\boldsymbol{V}\\) with \\(N\\) frames, a VAE `\cite{kingma2013auto}`{=latex} is employed on each frame to extract the latent features \\(\boldsymbol{Z} = \{\boldsymbol{z}_1, \cdots, \boldsymbol{z}_N\}\\), where \\(\boldsymbol{Z} \in \mathbb{R} ^{N \times H\times W\times d}\\). Here, \\(H\\) and \\(W\\) denote the height and width of the latent feature and \\(d\\) denotes the dimension of each token. For each frame of \\(\boldsymbol{Z}\\), we add noise of a specific timestep \\(t\\) and feed the noisy frame \\(\boldsymbol{Z}^t = \{\boldsymbol{z}_1^t, \cdots, \boldsymbol{z}_N^t\}\\) into the pre-trained T2I model \\(f_\theta\\) respectively. The diffusion feature (i.e., the intermediate feature from the U-Net decoder) is extracted through a single step of denoising `\cite{tang2023emergent}`{=latex}: \\[\boldsymbol{F} = \{\boldsymbol{F}_i\} = \{f_\theta(\boldsymbol{z}_i^t)\}, i\in \{1, \cdots, N\},\\] where \\(\boldsymbol{F} \in \mathbb{R} ^{N \times H\times W\times d}\\), denoting the normalized diffusion feature of each frame.

**One-to-many Correspondence Calculation.** For each token within the diffusion feature \\(\boldsymbol{F}\\), its corresponding tokens in other frames are identified based on the cosine similarity. Without loss of generality, we could consider a specific token \\(\boldsymbol{p}_{\{i,h_i,w_i\}}\\) in the \\(i\\)th frame \\(\boldsymbol{F}_i\\) with the coordinate \\([h_i, w_i]\\). Unlike previous methods where only one corresponding token of \\(\boldsymbol{p}_{\{i,h_i,w_i\}}\\) can be identified in each frame (<a href="#fig:compare_intro" data-reference-type="ref+Label" data-reference="fig:compare_intro">1</a>a), our method can obtain the one-to-many correspondences simply by selecting tokens with the top \\(K\\) highest similarity in each frame. We record their coordinates, which are used for sampling the tokens for self-attention in the subsequent inversion and denoising process. To implement this process, the most straightforward method is through a direct matrix multiplication of the normalized diffusion feature \\({\boldsymbol{F}}\\). \\[\boldsymbol{S} = \boldsymbol{F} \cdot \boldsymbol{F}^{T},\\] where \\(\boldsymbol{S} \in \mathbb{R} ^{(N\times H\times W)\times (N\times H\times W)}\\) represents the cosine similarity between each token and all tokens in the diffusion feature of the video.

The similarity between \\(\boldsymbol{p}_{\{i,h_i,w_i\}}\\) and all \\(N\times H\times W\\) tokens in the feature is given by \\(\boldsymbol{S}[i,h_i,w_i,:,:,:]\\). The coordinates of the corresponding tokens in the \\(j\\)th frame (\\(j \in \{1, \cdots, N\}\\)) are then obtained by selecting the tokens with the top \\(K\\) similarities in the \\(j\\)th frame. \\[{h}_j^k,{w}_j^k = \text{top-$k$-argmax}_{({x}^k,{y}^k)}(\boldsymbol{S}[i,h_i,w_i, j,{x^k},{y^k}]),\\] Here the top-\\(k\\)-argmax(\\(\cdot\\)) denotes the operation to find coordinates of the top \\(K\\) biggest values in a matrix, where \\(k \in \{1,\cdots, K\}\\). \\([{h}_j^k, {w}_j^k]\\) represents the coordinates of the token in \\(j\\)th frame which has highest similarity with \\(\boldsymbol{p}_{\{i,h_i,w_i\}}\\). A similar process can be conducted for each token of \\(\boldsymbol{F}\\), thereby obtaining their correspondences among frames.

<figure id="fig:slding">
<embed src="fig/fig_correspondence.pdf" />
<figcaption><strong>Sliding-window-based strategy for correspondence calculation.</strong> <span style="background-color: 198,236,185"><span style="color: 198,236,185">t </span></span> represents the token <span class="math inline"><strong>p</strong><sub>{<em>i</em>, <em>h</em><sub><em>i</em></sub>, <em>w</em><sub><em>i</em></sub>}</sub></span>. <span style="background-color: 136,174,212"><span style="color: 136,174,212">t </span></span> and <span style="background-color: 206,239,252"><span style="color: 206,239,252">t </span></span> represents the obtained corresponded tokens in other frames. </figcaption>
</figure>

<span id="sec:corres" label="sec:corres"></span>

**Sliding-window Strategy.** Although the one-to-many correspondence among tokens can be effectively obtained through the above process, it requires excessive computational resources because \\((N\times H\times W)\\) is always a huge number, especially in long videos. As a result, the computational complexity of this process is extremely high, which can be represented as \\(\mathcal{O}(N^2\times H^2 \times W^2 \times d)\\). At the same time, multiplication between these two huge matrices consumes a substantial amount of GPU memory in practice. These limitations severely limit its applicability in many real-world scenarios, such as on mobile devices.

To address the above problem, we further propose the sliding-window-based strategy as an alternative, which not only effectively obtains the one-to-many correspondences but also significantly reduces the computational overhead (<a href="#fig:slding" data-reference-type="ref+Label" data-reference="fig:slding">3</a>). Firstly, for the token \\(\boldsymbol{p}_{\{i,h_i,w_i\}}\\), it is only necessary to calculate its similarity with the tokens in the next frame \\(\boldsymbol{F}_{i+1}\\) instead of in all frames, i.e., \\[\boldsymbol{S}_i=\boldsymbol{F}_i \cdot \boldsymbol{F}_{i+1}^T.
    \label{eqa:sim}\\] \\(\boldsymbol{S}_i \in \mathbb{R} ^{H \times W \times H \times W}\\) denotes the similarity between the tokens in \\(i\\)th frame and those in \\((i+1)\\)th frame. The overall similarity matrix is \\(\boldsymbol{S} = \{\boldsymbol{S}_i\}, i\in \{1,2,\cdots,N-1\}\\), where \\(\boldsymbol{S} \in \mathbb{R} ^{(N-1) \times H \times W \times H \times W}\\). Then, we obtain the \\(K\\) corresponded tokens of \\(\boldsymbol{p}_{\{i,h_i,w_i\}}\\) in \\(\boldsymbol{F}_{i+1}\\) through \\(\boldsymbol{S}_i\\), \\[{h}_{i+1}^k,{w}_{i+1}^k = \text{top-$k$-argmax}_{({x}^k,{y}^k)}(\boldsymbol{S}_i[h_i,w_i, {x}^k,{y}^k]),\\] For tokens in \\((i+2)\\)th frame, instead of considering \\(\boldsymbol{p}_{\{i,h_i,w_i\}}\\), we identify the tokens in \\((i+2)\\)th frame which have the top \\(K\\) largest similarity with the token \\(\boldsymbol{p}_{\{i+1,h_{i+1}^1,w_{i+1}^1\}}\\) through the \\(\boldsymbol{S}_{i+1}\\). Similarly, we can obtain the corresponding token in other future or previous frames. \\[{h}_{i+2}^k,{w}_{i+2}^k = \text{top-$k$-argmax}_{({x}^k,{y}^k)}(\boldsymbol{S}_{i+1}[{h}_{i+1}^1,{w}_{i+1}^1, {x}^k,{y}^k]),\\] Through the above process, the overall complexity is reduced to \\(\mathcal{O}((N-1) \times H^2 \times W^2 \times d)\\). Furthermore, it is noteworthy that frames in a video exhibit temporal continuity, implying that the spatial positions of corresponding tokens are unlikely to change significantly between consecutive frames. Consequently, for the token \\(\boldsymbol{p}_{\{i,h_i,w_i\}}\\), it is enough to only calculate the similarity within a small window of length \\(l\\) in the adjacent frame, where \\(l\\) is much smaller than \\(H\\) and \\(W\\), \\[\boldsymbol{F}^{w}_{i+1}=\boldsymbol{F}_{i+1}[h_i-l/2:h_i+l/2,w_i-l/2:w_i+l/2,:].\\] \\(\boldsymbol{F}^{w}_{i+1} \in \mathbb{R} ^{l\times l\times d}\\) represents the tokens in \\(\boldsymbol{F}_{i+1}\\) within the sliding window. We calculate the cosine similarity between \\(\boldsymbol{p}_{\{i,h_i,w_i\}}\\) and the tokens in \\(\boldsymbol{F}^{w}_{i+1}\\), selecting tokens with top \\(K\\) highest similarity within \\(\boldsymbol{F}^{w}_{i+1}\\). This approach further reduces the computational complexity to \\(\mathcal{O}((N-1) \times H \times W \times l^2 \times d)\\) and the GPU memory consumption is also significantly reduced in practice. Additionally, it is worth noting that calculating correspondence information from the source video is only conducted once before the inversion and denoising process of video editing. Compared with the subsequent editing process, this process only takes negligible time.

## Correspondence-guided Video Editing.

In this section, we explain how to apply the correspondence information to the video editing process (<a href="#fig:pipe" data-reference-type="ref+Label" data-reference="fig:pipe">2</a>c). In the inversion and denoising process of video editing, we sample the corresponding tokens from the noisy latent for each token based on the coordinates obtained in <a href="#sec:corres" data-reference-type="ref+Label" data-reference="sec:corres">[sec:corres]</a>. For the token \\(\boldsymbol{z}_{{i,h_i,w_i}}^t\\), the set of corresponding tokens in other frames at a timestep \\(t\\) is: \\[\boldsymbol{Corr}=\{\boldsymbol{z}^t_{\{j,h_{j}^k,w_{j}^k\}}\}, j\in \{1,\cdots,i-1, i+1, \cdots, N\}, k\in \{1,\cdots, K\}.\\] We merge these tokens following `\cite{bolya2022token}`{=latex}, which can accelerate the editing process and reduce GPU memory usage without compromising the quality of editing results: \\[\widetilde{\boldsymbol{{Corr}}} = \text{Merge}(\boldsymbol{Corr}).\\] Then, the self-attention is conducted on the merged tokens, \\[\begin{gathered}
\boldsymbol{Q} = \boldsymbol{z}_{\{i,h_i,w_i\}}^t, \boldsymbol{K} = \boldsymbol{V} = \widetilde{\boldsymbol{{Corr}}}, \\
\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{SoftMax}\left(\frac{\boldsymbol{Q} \cdot \boldsymbol{K}^T}{\sqrt{d_k} }\right)\cdot \boldsymbol{V},
\end{gathered}\\] where \\(\sqrt{d_k}\\) is the scale factor. The above process of correspondence-guided attention is illustrated in <a href="#fig:pipe" data-reference-type="ref+Label" data-reference="fig:pipe">2</a>b. Following the previous methods `\cite{yang2024fresco,cong2023flatten}`{=latex}, we also retain the spatial-temporal attention `\cite{wu2023tune}`{=latex} in the U-Net. In spatial-temporal attention, considering a query token, all tokens in the video serve as keys and values, regardless of their relevance to the query. This correspondence-agnostic self-attention is not enough to maintain temporal consistency, introducing irrelevant information into each token, and thus causing serious flickering effects `\cite{cong2023flatten,geyer2023tokenflow}`{=latex}. Our correspondence-guided attention can significantly alleviate the problems of spatial-temporal attention, increasing the similarity of corresponding tokens and thus enhancing the temporal consistency of the edited video.

# Experiment

<figure id="fig:ours_main">
<embed src="fig/fig_main_result.pdf" />
<figcaption><strong>Qualitative results of COVE.</strong> COVE can effectively handle various types of prompts, generating high-quality videos. For both global editing (e.g., style transferring and background editing) and local editing (e.g., modifying the appearance of the subject), COVE demonstrates outstanding performance. Results are best-viewed zoomed-in.</figcaption>
</figure>

## Experimental Setup

In the experiment, we adopt Stable Diffusion (SD) 2.1 from the official Huggingface repository for COVE, employing 100 steps of DDIM inversion and 50 steps of denoising. To extract the diffusion feature, the noise of the specific timestep \\(t=261\\) is added to each frame of the source video following `\cite{tang2023emergent}`{=latex}. The feature is then extracted from the intermediate layer of the 2D Unet decoder during a single step of denoising. The window size \\(l\\) is set to 9 for correspondence calculation, and \\(k\\) is set to 3 for correspondence-guided attention. The merge ratio for token merging is 50%. For both qualitative and quantitative evaluation, we select 23 videos from social media platforms such as TikTok and other publicly available sources `\cite{pexels, pixabay}`{=latex}. Among these 23 videos, 3 videos have a length of 10 frames, 15 videos have a length of 20 frames, and 5 videos have a length of 32 frames. The experiments are conducted on a single RTX 3090 GPU for our method unless otherwise specified. We compare COVE with 5 baseline methods: FateZero `\cite{qi2023fatezero}`{=latex}, TokenFlow `\cite{geyer2023tokenflow}`{=latex}, FLATTEN `\cite{cong2023flatten}`{=latex}, FRESCO `\cite{yang2024fresco}`{=latex} and RAVE `\cite{kara2023rave}`{=latex}. For all of these baseline methods, we follow the default settings from their official Github repositories. The more detailed experimental settings of our method are provided in <a href="#sec:app_set" data-reference-type="ref+Label" data-reference="sec:app_set">6</a>.

## Qualitative Results

We evaluate COVE on various videos under different types of prompts including both global and local editing (<a href="#fig:ours_main" data-reference-type="ref+Label" data-reference="fig:ours_main">4</a>). Global editing mainly involves background editing and style transferring. For background editing, COVE can modify the background while keeping the subject of the video unchanged (e.g. Third row, first column. “`a car driving in milky way`”). For style transfer, COVE can effectively modify the global style of the source video according to the prompt (e.g. Third row, second column. “`Van Gogh style`”). Our prompts for local editing include changing the subject of the video to another one (e.g. Third row, third column. “`A cute raccoon`”) and making local edits to the subject (e.g. fifth row, third column. “`A sorrow woman`”). For all of these editing tasks, COVE demonstrates outstanding performance, generating frames with high visual quality while successfully preserving temporal consistency. We also compare COVE with a wide range of state-of-the-art video editing methods (<a href="#fig:compare" data-reference-type="ref+Label" data-reference="fig:compare">5</a>). The experimental results illustrate that COVE effectively edits the video with high quality, significantly outperforming the previous methods.

<figure id="fig:compare">
<embed src="fig/fig_baseline_3.pdf" />
<figcaption><strong>Qualitative comparison of COVE and various state-of-the-art methods.</strong> Our method outperforms previous methods across a wide range of source videos and editing prompts, demonstrating superior visual quality and temporal consistency. Results are best-viewed zoomed-in.</figcaption>
</figure>

## Quantitative Results

For quantitative comparison, we follow the metrics proposed in VBench `\cite{huang2023vbench}`{=latex}, including Subject Consistency, Motion Smoothness, Aesthetic Quality, and Imaging Quality. Among them, Subject Consistency assesses whether the subject (e.g., a person) remains consistent throughout the whole video by calculating the similarity of DINO `\cite{caron2021emerging}`{=latex} feature across frames. Motion Smoothness utilizes the motion priors of the video frame interpolation model `\cite{li2023amt}`{=latex} to evaluate the smoothness of the motion in the generated video. Aesthetic Quality uses the LAION aesthetic predictor `\cite{LAIONaes}`{=latex} to assess the artistic and beauty value perceived by humans on each frame. Imaging Quality evaluates the degree of distortion in the generated frames (e.g., blurring, flickering) through the MUSIQ `\cite{ke2021musiq}`{=latex} image quality predictor. Each video undergoes editing with 3 global prompts (such as style transferring, background editing, etc.) and 2 local prompts (such as editing the appearance of the subject in the video), generating a total of 115 text-video pairs. For each metric, we report the average score of these 115 videos. We further conducted a user study with 45 participants following `\cite{yang2024fresco}`{=latex}. Participants are required to choose the most preferable results among these methods. The result is shown in <a href="#tab:compare" data-reference-type="ref+Label" data-reference="tab:compare">[tab:compare]</a>. Among various methods, COVE achieves outstanding performance in both qualitative metrics and user studies, further demonstrating its superiority.

|  |  |  |  |  |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|  | Subject | Motion | Aesthetic | Imaging | User |  |
|  | Consistency | Smoothness | Quality | Quality | Study |  |
| FateZero `\cite{qi2023fatezero}`{=latex} | 0.9622 | 0.9547 | 0.6258 | 0.6951 | 7.4% |  |
| TokenFlow `\cite{geyer2023tokenflow}`{=latex} | 0.9513 | 0.9803 | 0.6904 | 0.7354 | 13.0% |  |
| FLATTEN `\cite{cong2023flatten}`{=latex} | 0.9617 | 0.9622 | 0.6544 | 0.7155 | 14.8% |  |
| FRESCO `\cite{yang2024fresco}`{=latex} | 0.9358 | 0.9737 | 0.6582 | 0.6331 | 9.2% |  |
| RAVE `\cite{kara2023rave}`{=latex} | 0.9518 | 0.9732 | 0.6369 | 0.7355 | 11.1% |  |
| **COVE (Ours)** | **0.9731** | **0.9892** | **0.7122** | **0.7441** | **44.5%** |  |

**Quantitative comparison** among COVE and a wide range of state-of-the-art video editing methods. The evaluation metrics`\cite{huang2023vbench}`{=latex} can effectively reflect the temporal consistency and frame quality of generated videos. COVE illustrates superior performance in both keeping the temporal consistency and generating frames with high quality in edited videos. <span id="tab:compare" label="tab:compare"></span>

|           |             |            |            |            |     |
|:---------:|:-----------:|:----------:|:----------:|:----------:|:---:|
|           |   Subject   |   Motion   | Aesthetic  |  Imaging   |     |
|           | Consistency | Smoothness |  Quality   |  Quality   |     |
|    w/o    |   0.9431    |   0.9049   |   0.6913   |   0.7132   |     |
| \\(K=1\\) |   0.9637    |   0.9817   |   0.6979   |   0.7148   |     |
| \\(K=3\\) |   0.9731    | **0.9892** |   0.7122   | **0.7441** |     |
| \\(K=5\\) | **0.9745**  |   0.9886   | **0.7167** |   0.7429   |     |

**Ablation study on the value of \\(K\\) in correspondence-guided attention.** w / o means without correspondence-guided attention in Unet. When \\(K=3\\) the quality of the video is the best. <span id="tab:ablate" label="tab:ablate"></span>

<embed src="fig/fig_abla.pdf" />

<span id="fig:ablate" label="fig:ablate"></span>

## Ablation Study

We conduct an ablation study to illustrate the effectiveness of the **Correspondence-guided attention** and the number of tokens selected in each frame (i.e., the value of \\(K\\)). The experimental results (<a href="#tab:ablate" data-reference-type="ref+Label" data-reference="tab:ablate">[tab:ablate]</a> and <a href="#fig:ablate" data-reference-type="ref+Label" data-reference="fig:ablate">[fig:ablate]</a>) illustrate that without correspondence-guided attention, the edited video exhibits obvious temporal inconsistency and flickering effects (which is marked in <span style="color: 200,200,0">**yellow**</span> and <span style="color: orange">**orange**</span> boxes in <a href="#fig:ablate" data-reference-type="ref+Label" data-reference="fig:ablate">[fig:ablate]</a>), thus severely impairing the visual quality. As \\(K\\) increases from 1 to 3, the generated video contains more fine-grained details, exhibiting better visual quality. However, further increasing \\(K\\) to 5 does not significantly improve the video quality. We also illustrate the effectiveness of **temporal dimensional token merging**. By merging the tokens with high correspondence across frames, the editing process becomes more efficient (<a href="#tab:abla_merge" data-reference-type="ref+Label" data-reference="tab:abla_merge">[tab:abla_merge]</a>) while there is no significant decrease in the quality of the edited video (<a href="#fig:abla_merge" data-reference-type="ref+Label" data-reference="fig:abla_merge">[fig:abla_merge]</a>). The ablation of the **sliding-window size** \\(l\\) is shown in <a href="#app:abl_ws" data-reference-type="ref+Label" data-reference="app:abl_ws">[app:abl_ws]</a>. If the window size is too small, the actual corresponding token may not be included within the window, resulting in suboptimal correspondence and poor editing results. On the other hand, a too-large window size is not necessary for identifying the corresponding tokens, which would lead to high computational complexity and excessive memory usage. The experiment results illustrate that \\(l=9\\) is suitable to strike a balance. Additionally, we also **visualize the correspondence** obtained by COVE, which is shown in <a href="#app:vis_cor" data-reference-type="ref+Label" data-reference="app:vis_cor">8</a>.

<table>
<tbody>
<tr>
<td style="text-align: center;">Correspondence</td>
<td style="text-align: center;">Token</td>
<td rowspan="2" style="text-align: center;">Speed</td>
<td style="text-align: center;">Memory</td>
</tr>
<tr>
<td style="text-align: center;">Guided Attention</td>
<td style="text-align: center;">Merging</td>
<td style="text-align: center;">Usage</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">2.2 min</td>
<td style="text-align: center;">9 GB</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">2.7 min</td>
<td style="text-align: center;">14 GB</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">2.4 min</td>
<td style="text-align: center;">11 GB</td>
</tr>
</tbody>
</table>

**Ablation Study on the effect of temporal dimensional token merging.** Temporal dimensional token merging can speed up the editing process and save GPU memory usage while hardly impairing the quality of the generated video. The experiment is conducted on a single RTX3090 GPU with a 10-frame source video. \\(k\\) is set to 3. <span id="tab:abla_merge" label="tab:abla_merge"></span>

<embed src="fig/fig_abla_2.pdf" />

<span id="fig:abla_merge" label="fig:abla_merge"></span>

# Conclusion

In this paper, we propose COVE, which is the first to explore how to employ inherent diffusion feature correspondence in video editing to enhance editing quality and temporal consistency. Through the proposed efficient sliding-window-based strategy, the one-to-many correspondence relationship among tokens across frames is obtained. During the inversion and denoising process, self-attention is performed within the corresponding tokens to enhance temporal consistency. Additionally, we also apply token merging in the temporal dimension to improve the efficiency of the editing process. Both quantitative and qualitative experimental results demonstrate the effectiveness of our method, which outperforms a wide range of previous methods, achieving state-of-the-art editing quality.

**Limitaions.** The limitation of our method is discussed in <a href="#apdx:lim" data-reference-type="ref+Label" data-reference="apdx:lim">12</a>.

<div class="ack" markdown="1">

This work was supported by the STI 2030-Major Projects under Grant 2021ZD0201404.

</div>

# References [references]

<div class="thebibliography" markdown="1">

Pexels. <https://www.pexels.com/>, accessed: 2023-11-16 (@pexels)

Pixabay. <https://pixabay.com/>, accessed: 2023-11-16 (@pixabay)

Avrahami, O., Lischinski, D., Fried, O.: Blended diffusion for text-driven editing of natural images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18208–18218 (2022) (@avrahami2022blended)

Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22563–22575 (2023) (@blattmann2023align)

Bolya, D., Fu, C.Y., Dai, X., Zhang, P., Feichtenhofer, C., Hoffman, J.: Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461 (2022) **Abstract:** We introduce Token Merging (ToMe), a simple method to increase the throughput of existing ViT models without needing to train. ToMe gradually combines similar tokens in a transformer using a general and light-weight matching algorithm that is as fast as pruning while being more accurate. Off-the-shelf, ToMe can 2x the throughput of state-of-the-art ViT-L @ 512 and ViT-H @ 518 models on images and 2.2x the throughput of ViT-L on video with only a 0.2-0.3% accuracy drop in each case. ToMe can also easily be applied during training, improving in practice training speed up to 2x for MAE fine-tuning on video. Training with ToMe further minimizes accuracy drop, leading to 2x the throughput of ViT-B on audio for only a 0.4% mAP drop. Qualitatively, we find that ToMe merges object parts into one token, even over multiple frames of video. Overall, ToMe’s accuracy and speed are competitive with state-of-the-art on images, video, and audio. (@bolya2022token)

Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023) (@brooks2023instructpix2pix)

Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., Zheng, Y.: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 22560–22570 (2023) (@cao2023masactrl)

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021) (@caron2021emerging)

Chen, Q., Ma, Y., Wang, H., Yuan, J., Zhao, W., Tian, Q., Wang, H., Min, S., Chen, Q., Liu, W.: Follow-your-canvas: Higher-resolution video outpainting with extensive content generation. arXiv preprint arXiv:2409.01055 (2024) **Abstract:** This paper explores higher-resolution video outpainting with extensive content generation. We point out common issues faced by existing methods when attempting to largely outpaint videos: the generation of low-quality content and limitations imposed by GPU memory. To address these challenges, we propose a diffusion-based method called \\}textit{Follow-Your-Canvas}. It builds upon two core designs. First, instead of employing the common practice of "single-shot" outpainting, we distribute the task across spatial windows and seamlessly merge them. It allows us to outpaint videos of any size and resolution without being constrained by GPU memory. Second, the source video and its relative positional relation are injected into the generation process of each window. It makes the generated spatial layout within each window harmonize with the source video. Coupling with these two designs enables us to generate higher-resolution outpainting videos with rich content while keeping spatial and temporal consistency. Follow-Your-Canvas excels in large-scale video outpainting, e.g., from 512X512 to 1152X2048 (9X), while producing high-quality and aesthetically pleasing results. It achieves the best quantitative results across various resolution and scale setups. The code is released on https://github.com/mayuelala/FollowYourCanvas (@chen2024follow)

Chen, X., Wang, Y., Zhang, L., Zhuang, S., Ma, X., Yu, J., Wang, Y., Lin, D., Qiao, Y., Liu, Z.: Seine: Short-to-long video diffusion model for generative transition and prediction. In: The Twelfth International Conference on Learning Representations (2023) (@chen2023seine)

Cong, Y., Xu, M., Simon, C., Chen, S., Ren, J., Xie, Y., Perez-Rua, J.M., Rosenhahn, B., Xiang, T., He, S.: Flatten: optical flow-guided attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922 (2023) **Abstract:** Text-to-video editing aims to edit the visual appearance of a source video conditional on textual prompts. A major challenge in this task is to ensure that all frames in the edited video are visually consistent. Most recent works apply advanced text-to-image diffusion models to this task by inflating 2D spatial attention in the U-Net into spatio-temporal attention. Although temporal context can be added through spatio-temporal attention, it may introduce some irrelevant information for each patch and therefore cause inconsistency in the edited video. In this paper, for the first time, we introduce optical flow into the attention module in the diffusion model’s U-Net to address the inconsistency issue for text-to-video editing. Our method, FLATTEN, enforces the patches on the same flow path across different frames to attend to each other in the attention module, thus improving the visual consistency in the edited videos. Additionally, our method is training-free and can be seamlessly integrated into any diffusion-based text-to-video editing methods and improve their visual consistency. Experiment results on existing text-to-video editing benchmarks show that our proposed method achieves the new state-of-the-art performance. In particular, our method excels in maintaining the visual consistency in the edited videos. (@cong2023flatten)

Croitoru, F.A., Hondru, V., Ionescu, R.T., Shah, M.: Diffusion models in vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2023) (@croitoru2023diffusion)

Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. Advances in neural information processing systems **34**, 8780–8794 (2021) (@dhariwal2021diffusion)

Fang, C., He, C., Xiao, F., Zhang, Y., Tang, L., Zhang, Y., Li, K., Li, X.: Real-world image dehazing with coherence-based label generator and cooperative unfolding network. arXiv preprint arXiv:2406.07966 (2024) **Abstract:** Real-world Image Dehazing (RID) aims to alleviate haze-induced degradation in real-world settings. This task remains challenging due to the complexities in accurately modeling real haze distributions and the scarcity of paired real-world data. To address these challenges, we first introduce a cooperative unfolding network that jointly models atmospheric scattering and image scenes, effectively integrating physical knowledge into deep networks to restore haze-contaminated details. Additionally, we propose the first RID-oriented iterative mean-teacher framework, termed the Coherence-based Label Generator, to generate high-quality pseudo labels for network training. Specifically, we provide an optimal label pool to store the best pseudo-labels during network training, leveraging both global and local coherence to select high-quality candidates and assign weights to prioritize haze-free regions. We verify the effectiveness of our method, with experiments demonstrating that it achieves state-of-the-art performance on RID tasks. Code will be available at https://github.com/cnyvfang/CORUN-Colabator. (@fang2024real)

Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022) **Abstract:** Text-to-image models offer unprecedented freedom to guide creation through natural language. Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes. In other words, we ask: how can we use language-guided models to turn our cat into a painting, or imagine a new product based on our favorite toy? Here we present a simple approach that allows such creative freedom. Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model. These "words" can be composed into natural language sentences, guiding personalized creation in an intuitive way. Notably, we find evidence that a single word embedding is sufficient for capturing unique and varied concepts. We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks. Our code, data and new words will be available at: https://textual-inversion.github.io (@gal2022image)

Geyer, M., Bar-Tal, O., Bagon, S., Dekel, T.: Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373 (2023) **Abstract:** The generative AI revolution has recently expanded to videos. Nevertheless, current state-of-the-art video models are still lagging behind image models in terms of visual quality and user control over the generated content. In this work, we present a framework that harnesses the power of a text-to-image diffusion model for the task of text-driven video editing. Specifically, given a source video and a target text-prompt, our method generates a high-quality video that adheres to the target text, while preserving the spatial layout and motion of the input video. Our method is based on a key observation that consistency in the edited video can be obtained by enforcing consistency in the diffusion feature space. We achieve this by explicitly propagating diffusion features based on inter-frame correspondences, readily available in the model. Thus, our framework does not require any training or fine-tuning, and can work in conjunction with any off-the-shelf text-to-image editing method. We demonstrate state-of-the-art editing results on a variety of real-world videos. Webpage: https://diffusion-tokenflow.github.io/ (@geyer2023tokenflow)

Gu, Y., Zhou, Y., Wu, B., Yu, L., Liu, J.W., Zhao, R., Wu, J.Z., Zhang, D.J., Shou, M.Z., Tang, K.: Videoswap: Customized video subject swapping with interactive semantic point correspondence. arXiv preprint arXiv:2312.02087 (2023) **Abstract:** Current diffusion-based video editing primarily focuses on structure-preserved editing by utilizing various dense correspondences to ensure temporal consistency and motion alignment. However, these approaches are often ineffective when the target edit involves a shape change. To embark on video editing with shape change, we explore customized video subject swapping in this work, where we aim to replace the main subject in a source video with a target subject having a distinct identity and potentially different shape. In contrast to previous methods that rely on dense correspondences, we introduce the VideoSwap framework that exploits semantic point correspondences, inspired by our observation that only a small number of semantic points are necessary to align the subject’s motion trajectory and modify its shape. We also introduce various user-point interactions (\\}eg, removing points and dragging points) to address various semantic point correspondence. Extensive experiments demonstrate state-of-the-art video subject swapping results across a variety of real-world videos. (@gu2023videoswap)

Guo, H., Dai, T., Ouyang, Z., Zhang, T., Zha, Y., Chen, B., Xia, S.t.: Refir: Grounding large restoration models with retrieval augmentation. arXiv preprint arXiv:2410.05601 (2024) **Abstract:** Recent advances in diffusion-based Large Restoration Models (LRMs) have significantly improved photo-realistic image restoration by leveraging the internal knowledge embedded within model weights. However, existing LRMs often suffer from the hallucination dilemma, i.e., producing incorrect contents or textures when dealing with severe degradations, due to their heavy reliance on limited internal knowledge. In this paper, we propose an orthogonal solution called the Retrieval-augmented Framework for Image Restoration (ReFIR), which incorporates retrieved images as external knowledge to extend the knowledge boundary of existing LRMs in generating details faithful to the original scene. Specifically, we first introduce the nearest neighbor lookup to retrieve content-relevant high-quality images as reference, after which we propose the cross-image injection to modify existing LRMs to utilize high-quality textures from retrieved images. Thanks to the additional external knowledge, our ReFIR can well handle the hallucination challenge and facilitate faithfully results. Extensive experiments demonstrate that ReFIR can achieve not only high-fidelity but also realistic restoration results. Importantly, our ReFIR requires no training and is adaptable to various LRMs. (@guo2024refir)

Guo, J., Du, C., Wang, J., Huang, H., Wan, P., Huang, G.: Assessing a single image in reference-guided image synthesis. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 753–761 (2022) (@guo2022assessing)

Guo, J., Manukyan, H., Yang, C., Wang, C., Khachatryan, L., Navasardyan, S., Song, S., Shi, H., Huang, G.: Faceclip: Facial image-to-video translation via a brief text description. IEEE Transactions on Circuits and Systems for Video Technology (2023) (@guo2023faceclip)

Guo, J., Wang, C., Wu, Y., Zhang, E., Wang, K., Xu, X., Song, S., Shi, H., Huang, G.: Zero-shot generative model adaptation via image-specific prompt learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11494–11503 (2023) (@guo2023zero)

Guo, J., Xu, X., Pu, Y., Ni, Z., Wang, C., Vasu, M., Song, S., Huang, G., Shi, H.: Smooth diffusion: Crafting smooth latent spaces in diffusion models. arXiv preprint arXiv:2312.04410 (2023) **Abstract:** Recently, diffusion models have made remarkable progress in text-to-image (T2I) generation, synthesizing images with high fidelity and diverse contents. Despite this advancement, latent space smoothness within diffusion models remains largely unexplored. Smooth latent spaces ensure that a perturbation on an input latent corresponds to a steady change in the output image. This property proves beneficial in downstream tasks, including image interpolation, inversion, and editing. In this work, we expose the non-smoothness of diffusion latent spaces by observing noticeable visual fluctuations resulting from minor latent variations. To tackle this issue, we propose Smooth Diffusion, a new category of diffusion models that can be simultaneously high-performing and smooth. Specifically, we introduce Step-wise Variation Regularization to enforce the proportion between the variations of an arbitrary input latent and that of the output image is a constant at any diffusion training step. In addition, we devise an interpolation standard deviation (ISTD) metric to effectively assess the latent space smoothness of a diffusion model. Extensive quantitative and qualitative experiments demonstrate that Smooth Diffusion stands out as a more desirable solution not only in T2I generation but also across various downstream tasks. Smooth Diffusion is implemented as a plug-and-play Smooth-LoRA to work with various community models. Code is available at https://github.com/SHI-Labs/Smooth-Diffusion. (@guo2023smooth)

Guo, J., Zhao, J., Ge, C., Du, C., Ni, Z., Song, S., Shi, H., Huang, G.: Everything to the synthetic: Diffusion-driven test-time adaptation via synthetic-domain alignment. arXiv preprint arXiv:2406.04295 (2024) **Abstract:** Test-time adaptation (TTA) aims to improve the performance of source-domain pre-trained models on previously unseen, shifted target domains. Traditional TTA methods primarily adapt model weights based on target data streams, making model performance sensitive to the amount and order of target data. The recently proposed diffusion-driven TTA methods mitigate this by adapting model inputs instead of weights, where an unconditional diffusion model, trained on the source domain, transforms target-domain data into a synthetic domain that is expected to approximate the source domain. However, in this paper, we reveal that although the synthetic data in diffusion-driven TTA seems indistinguishable from the source data, it is unaligned with, or even markedly different from the latter for deep networks. To address this issue, we propose a \\}textbf{S}ynthetic-\\}textbf{D}omain \\}textbf{A}lignment (SDA) framework. Our key insight is to fine-tune the source model with synthetic data to ensure better alignment. Specifically, we first employ a conditional diffusion model to generate labeled samples, creating a synthetic dataset. Subsequently, we use the aforementioned unconditional diffusion model to add noise to and denoise each sample before fine-tuning. This Mix of Diffusion (MoD) process mitigates the potential domain misalignment between the conditional and unconditional models. Extensive experiments across classifiers, segmenters, and multimodal large language models (MLLMs, \\}eg, LLaVA) demonstrate that SDA achieves superior domain alignment and consistently outperforms existing diffusion-driven TTA methods. Our code is available at https://github.com/SHI-Labs/Diffusion-Driven-Test-Time-Adaptation-via-Synthetic-Domain-Alignment. (@guo2024everything)

Guo, Q., Lin, T.: Focus on your instruction: Fine-grained and multi-instruction image editing by attention modulation. arXiv preprint arXiv:2312.10113 (2023) **Abstract:** Recently, diffusion-based methods, like InstructPix2Pix (IP2P), have achieved effective instruction-based image editing, requiring only natural language instructions from the user. However, these methods often inadvertently alter unintended areas and struggle with multi-instruction editing, resulting in compromised outcomes. To address these issues, we introduce the Focus on Your Instruction (FoI), a method designed to ensure precise and harmonious editing across multiple instructions without extra training or test-time optimization. In the FoI, we primarily emphasize two aspects: (1) precisely extracting regions of interest for each instruction and (2) guiding the denoising process to concentrate within these regions of interest. For the first objective, we identify the implicit grounding capability of IP2P from the cross-attention between instruction and image, then develop an effective mask extraction method. For the second objective, we introduce a cross attention modulation module for rough isolation of target editing regions and unrelated regions. Additionally, we introduce a mask-guided disentangle sampling strategy to further ensure clear region isolation. Experimental results demonstrate that FoI surpasses existing methods in both quantitative and qualitative evaluations, especially excelling in multi-instruction editing task. (@guo2023focus)

He, C., Fang, C., Zhang, Y., Ye, T., Li, K., Tang, L., Guo, Z., Li, X., Farsiu, S.: Reti-diff: Illumination degradation image restoration with retinex-based latent diffusion model. arXiv preprint arXiv:2311.11638 (2023) **Abstract:** Illumination degradation image restoration (IDIR) techniques aim to improve the visibility of degraded images and mitigate the adverse effects of deteriorated illumination. Among these algorithms, diffusion model (DM)-based methods have shown promising performance but are often burdened by heavy computational demands and pixel misalignment issues when predicting the image-level distribution. To tackle these problems, we propose to leverage DM within a compact latent space to generate concise guidance priors and introduce a novel solution called Reti-Diff for the IDIR task. Reti-Diff comprises two key components: the Retinex-based latent DM (RLDM) and the Retinex-guided transformer (RGformer). To ensure detailed reconstruction and illumination correction, RLDM is empowered to acquire Retinex knowledge and extract reflectance and illumination priors. These priors are subsequently utilized by RGformer to guide the decomposition of image features into their respective reflectance and illumination components. Following this, RGformer further enhances and consolidates the decomposed features, resulting in the production of refined images with consistent content and robustness to handle complex degradation scenarios. Extensive experiments show that Reti-Diff outperforms existing methods on three IDIR tasks, as well as downstream applications. Code will be available at \\}url{https://github.com/ChunmingHe/Reti-Diff}. (@he2023reti)

He, C., Shen, Y., Fang, C., Xiao, F., Tang, L., Zhang, Y., Zuo, W., Guo, Z., Li, X.: Diffusion models in low-level vision: A survey. arXiv preprint arXiv:2406.11138 (2024) **Abstract:** Deep generative models have garnered significant attention in low-level vision tasks due to their generative capabilities. Among them, diffusion model-based solutions, characterized by a forward diffusion process and a reverse denoising process, have emerged as widely acclaimed for their ability to produce samples of superior quality and diversity. This ensures the generation of visually compelling results with intricate texture information. Despite their remarkable success, a noticeable gap exists in a comprehensive survey that amalgamates these pioneering diffusion model-based works and organizes the corresponding threads. This paper proposes the comprehensive review of diffusion model-based techniques. We present three generic diffusion modeling frameworks and explore their correlations with other deep generative models, establishing the theoretical foundation. Following this, we introduce a multi-perspective categorization of diffusion models, considering both the underlying framework and the target task. Additionally, we summarize extended diffusion models applied in other tasks, including medical, remote sensing, and video scenarios. Moreover, we provide an overview of commonly used benchmarks and evaluation metrics. We conduct a thorough evaluation, encompassing both performance and efficiency, of diffusion model-based techniques in three prominent tasks. Finally, we elucidate the limitations of current diffusion models and propose seven intriguing directions for future research. This comprehensive examination aims to facilitate a profound understanding of the landscape surrounding denoising diffusion models in the context of low-level vision tasks. A curated list of diffusion model-based techniques in over 20 low-level vision tasks can be found at https://github.com/ChunmingHe/awesome-diffusion-models-in-low-level-vision. (@he2024diffusion)

Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022) **Abstract:** Recent large-scale text-driven synthesis models have attracted much attention thanks to their remarkable capabilities of generating highly diverse images that follow given text prompts. Such text-based synthesis methods are particularly appealing to humans who are used to verbally describe their intent. Therefore, it is only natural to extend the text-driven image synthesis to text-driven image editing. Editing is challenging for these generative models, since an innate property of an editing technique is to preserve most of the original image, while in the text-based models, even a small modification of the text prompt often leads to a completely different outcome. State-of-the-art methods mitigate this by requiring the users to provide a spatial mask to localize the edit, hence, ignoring the original structure and content within the masked region. In this paper, we pursue an intuitive prompt-to-prompt editing framework, where the edits are controlled by text only. To this end, we analyze a text-conditioned model in depth and observe that the cross-attention layers are the key to controlling the relation between the spatial layout of the image to each word in the prompt. With this observation, we present several applications which monitor the image synthesis by editing the textual prompt only. This includes localized editing by replacing a word, global editing by adding a specification, and even delicately controlling the extent to which a word is reflected in the image. We present our results over diverse images and prompts, demonstrating high-quality synthesis and fidelity to the edited prompts. (@hertz2022prompt)

Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., et al.: Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022) **Abstract:** We present Imagen Video, a text-conditional video generation system based on a cascade of video diffusion models. Given a text prompt, Imagen Video generates high definition videos using a base video generation model and a sequence of interleaved spatial and temporal video super-resolution models. We describe how we scale up the system as a high definition text-to-video model including design decisions such as the choice of fully-convolutional temporal and spatial super-resolution models at certain resolutions, and the choice of the v-parameterization of diffusion models. In addition, we confirm and transfer findings from previous work on diffusion-based image generation to the video generation setting. Finally, we apply progressive distillation to our video models with classifier-free guidance for fast, high quality sampling. We find Imagen Video not only capable of generating videos of high fidelity, but also having a high degree of controllability and world knowledge, including the ability to generate diverse videos and text animations in various artistic styles and with 3D object understanding. See https://imagen.research.google/video/ for samples. (@ho2022imagen)

Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems **33**, 6840–6851 (2020) (@ho2020denoising)

Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022) **Abstract:** Classifier guidance is a recently introduced method to trade off mode coverage and sample fidelity in conditional diffusion models post training, in the same spirit as low temperature sampling or truncation in other types of generative models. Classifier guidance combines the score estimate of a diffusion model with the gradient of an image classifier and thereby requires training an image classifier separate from the diffusion model. It also raises the question of whether guidance can be performed without a classifier. We show that guidance can be indeed performed by a pure generative model without such a classifier: in what we call classifier-free guidance, we jointly train a conditional and an unconditional diffusion model, and we combine the resulting conditional and unconditional score estimates to attain a trade-off between sample quality and diversity similar to that obtained using classifier guidance. (@ho2022classifier)

Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video diffusion models. Advances in Neural Information Processing Systems **35**, 8633–8646 (2022) (@ho2022video)

Hu, Z., Xu, D.: Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073 (2023) **Abstract:** Recently, diffusion models like StableDiffusion have achieved impressive image generation results. However, the generation process of such diffusion models is uncontrollable, which makes it hard to generate videos with continuous and consistent content. In this work, by using the diffusion model with ControlNet, we proposed a new motion-guided video-to-video translation framework called VideoControlNet to generate various videos based on the given prompts and the condition from the input video. Inspired by the video codecs that use motion information for reducing temporal redundancy, our framework uses motion information to prevent the regeneration of the redundant areas for content consistency. Specifically, we generate the first frame (i.e., the I-frame) by using the diffusion model with ControlNet. Then we generate other key frames (i.e., the P-frame) based on the previous I/P-frame by using our newly proposed motion-guided P-frame generation (MgPG) method, in which the P-frames are generated based on the motion information and the occlusion areas are inpainted by using the diffusion model. Finally, the rest frames (i.e., the B-frame) are generated by using our motion-guided B-frame interpolation (MgBI) module. Our experiments demonstrate that our proposed VideoControlNet inherits the generation capability of the pre-trained large diffusion model and extends the image diffusion model to the video diffusion model by using motion information. More results are provided at our project page. (@hu2023videocontrolnet)

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:2311.17982 (2023) **Abstract:** Video generation has witnessed significant advancements, yet evaluating these models remains a challenge. A comprehensive evaluation benchmark for video generation is indispensable for two reasons: 1) Existing metrics do not fully align with human perceptions; 2) An ideal evaluation system should provide insights to inform future developments of video generation. To this end, we present VBench, a comprehensive benchmark suite that dissects "video generation quality" into specific, hierarchical, and disentangled dimensions, each with tailored prompts and evaluation methods. VBench has three appealing properties: 1) Comprehensive Dimensions: VBench comprises 16 dimensions in video generation (e.g., subject identity inconsistency, motion smoothness, temporal flickering, and spatial relationship, etc). The evaluation metrics with fine-grained levels reveal individual models’ strengths and weaknesses. 2) Human Alignment: We also provide a dataset of human preference annotations to validate our benchmarks’ alignment with human perception, for each evaluation dimension respectively. 3) Valuable Insights: We look into current models’ ability across various evaluation dimensions, and various content types. We also investigate the gaps between video and image generation models. We will open-source VBench, including all prompts, evaluation methods, generated videos, and human preference annotations, and also include more video generation models in VBench to drive forward the field of video generation. (@huang2023vbench)

Jonschkowski, R., Stone, A., Barron, J.T., Gordon, A., Konolige, K., Angelova, A.: What matters in unsupervised optical flow. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 557–572. Springer (2020) (@jonschkowski2020matters)

Kara, O., Kurtkaya, B., Yesiltepe, H., Rehg, J.M., Yanardag, P.: Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models. arXiv preprint arXiv:2312.04524 (2023) **Abstract:** Recent advancements in diffusion-based models have demonstrated significant success in generating images from text. However, video editing models have not yet reached the same level of visual quality and user control. To address this, we introduce RAVE, a zero-shot video editing method that leverages pre-trained text-to-image diffusion models without additional training. RAVE takes an input video and a text prompt to produce high-quality videos while preserving the original motion and semantic structure. It employs a novel noise shuffling strategy, leveraging spatio-temporal interactions between frames, to produce temporally consistent videos faster than existing methods. It is also efficient in terms of memory requirements, allowing it to handle longer videos. RAVE is capable of a wide range of edits, from local attribute modifications to shape transformations. In order to demonstrate the versatility of RAVE, we create a comprehensive video evaluation dataset ranging from object-focused scenes to complex human activities like dancing and typing, and dynamic scenes featuring swimming fish and boats. Our qualitative and quantitative experiments highlight the effectiveness of RAVE in diverse video editing scenarios compared to existing methods. Our code, dataset and videos can be found in https://rave-video.github.io. (@kara2023rave)

Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems **35**, 26565–26577 (2022) (@karras2022elucidating)

Kawar, B., Elad, M., Ermon, S., Song, J.: Denoising diffusion restoration models. Advances in Neural Information Processing Systems **35**, 23593–23606 (2022) (@kawar2022denoising)

Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5148–5157 (2021) (@ke2021musiq)

Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013) **Abstract:** How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions are two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results. (@kingma2013auto)

LAION-AI: aesthetic-predictor. <https://github.com/LAION-AI/aesthetic-predictor> (2022) **Abstract:** A linear estimator on top of clip to predict the aesthetic quality of pictures - LAION-AI/aesthetic-predictor (@LAIONaes)

Li, H., Yang, Y., Chang, M., Chen, S., Feng, H., Xu, Z., Li, Q., Chen, Y.: Srdiff: Single image super-resolution with diffusion probabilistic models. Neurocomputing **479**, 47–59 (2022) (@li2022srdiff)

Li, Z., Zhu, Z.L., Han, L.H., Hou, Q., Guo, C.L., Cheng, M.M.: Amt: All-pairs multi-field transforms for efficient frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9801–9810 (2023) (@li2023amt)

Liew, J.H., Yan, H., Zhang, J., Xu, Z., Feng, J.: Magicedit: High-fidelity and temporally coherent video editing. arXiv preprint arXiv:2308.14749 (2023) **Abstract:** In this report, we present MagicEdit, a surprisingly simple yet effective solution to the text-guided video editing task. We found that high-fidelity and temporally coherent video-to-video translation can be achieved by explicitly disentangling the learning of content, structure and motion signals during training. This is in contradict to most existing methods which attempt to jointly model both the appearance and temporal representation within a single framework, which we argue, would lead to degradation in per-frame quality. Despite its simplicity, we show that MagicEdit supports various downstream video editing tasks, including video stylization, local editing, video-MagicMix and video outpainting. (@liew2023magicedit)

Lin, Y., Han, H., Gong, C., Xu, Z., Zhang, Y., Li, X.: Consistent123: One image to highly consistent 3d asset using case-aware diffusion priors. arXiv preprint arXiv:2309.17261 (2023) **Abstract:** Reconstructing 3D objects from a single image guided by pretrained diffusion models has demonstrated promising outcomes. However, due to utilizing the case-agnostic rigid strategy, their generalization ability to arbitrary cases and the 3D consistency of reconstruction are still poor. In this work, we propose Consistent123, a case-aware two-stage method for highly consistent 3D asset reconstruction from one image with both 2D and 3D diffusion priors. In the first stage, Consistent123 utilizes only 3D structural priors for sufficient geometry exploitation, with a CLIP-based case-aware adaptive detection mechanism embedded within this process. In the second stage, 2D texture priors are introduced and progressively take on a dominant guiding role, delicately sculpting the details of the 3D model. Consistent123 aligns more closely with the evolving trends in guidance requirements, adaptively providing adequate 3D geometric initialization and suitable 2D texture refinement for different objects. Consistent123 can obtain highly 3D-consistent reconstruction and exhibits strong generalization ability across various objects. Qualitative and quantitative experiments show that our method significantly outperforms state-of-the-art image-to-3D methods. See https://Consistent123.github.io for a more comprehensive exploration of our generated 3D assets. (@lin2023consistent123)

Liu, S., Zhang, Y., Li, W., Lin, Z., Jia, J.: Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761 (2023) **Abstract:** This paper presents Video-P2P, a novel framework for real-world video editing with cross-attention control. While attention control has proven effective for image editing with pre-trained image generation models, there are currently no large-scale video generation models publicly available. Video-P2P addresses this limitation by adapting an image generation diffusion model to complete various video editing tasks. Specifically, we propose to first tune a Text-to-Set (T2S) model to complete an approximate inversion and then optimize a shared unconditional embedding to achieve accurate video inversion with a small memory cost. For attention control, we introduce a novel decoupled-guidance strategy, which uses different guidance strategies for the source and target prompts. The optimized unconditional embedding for the source prompt improves reconstruction ability, while an initialized unconditional embedding for the target prompt enhances editability. Incorporating the attention maps of these two branches enables detailed editing. These technical designs enable various text-driven editing applications, including word swap, prompt refinement, and attention re-weighting. Video-P2P works well on real-world videos for generating new characters while optimally preserving their original poses and scenes. It significantly outperforms previous approaches. (@liu2023video)

Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L.: Repaint: Inpainting using denoising diffusion probabilistic models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11461–11471 (2022) (@lugmayr2022repaint)

Ma, Y., Cun, X., He, Y., Qi, C., Wang, X., Shan, Y., Li, X., Chen, Q.: Magicstick: Controllable video editing via control handle transformations. arXiv preprint arXiv:2312.03047 (2023) **Abstract:** Text-based video editing has recently attracted considerable interest in changing the style or replacing the objects with a similar structure. Beyond this, we demonstrate that properties such as shape, size, location, motion, etc., can also be edited in videos. Our key insight is that the keyframe transformations of the specific internal feature (e.g., edge maps of objects or human pose), can easily propagate to other frames to provide generation guidance. We thus propose MagicStick, a controllable video editing method that edits the video properties by utilizing the transformation on the extracted internal control signals. In detail, to keep the appearance, we inflate both the pretrained image diffusion model and ControlNet to the temporal dimension and train low-rank adaptions (LORA) layers to fit the specific scenes. Then, in editing, we perform an inversion and editing framework. Differently, finetuned ControlNet is introduced in both inversion and generation for attention guidance with the proposed attention remix between the spatial attention maps of inversion and editing. Yet succinct, our method is the first method to show the ability of video property editing from the pre-trained text-to-image model. We present experiments on numerous examples within our unified framework. We also compare with shape-aware text-based editing and handcrafted motion video generation, demonstrating our superior temporal consistency and editing capability than previous works. The code and models are available on https://github.com/mayuelala/MagicStick. (@ma2023magicstick)

Ma, Y., He, Y., Cun, X., Wang, X., Chen, S., Li, X., Chen, Q.: Follow your pose: Pose-guided text-to-video generation using pose-free videos. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 4117–4125 (2024) (@ma2024follow)

Ma, Y., He, Y., Wang, H., Wang, A., Qi, C., Cai, C., Li, X., Li, Z., Shum, H.Y., Liu, W., et al.: Follow-your-click: Open-domain regional image animation via short prompts. arXiv preprint arXiv:2403.08268 (2024) **Abstract:** Despite recent advances in image-to-video generation, better controllability and local animation are less explored. Most existing image-to-video methods are not locally aware and tend to move the entire scene. However, human artists may need to control the movement of different objects or regions. Additionally, current I2V methods require users not only to describe the target motion but also to provide redundant detailed descriptions of frame contents. These two issues hinder the practical utilization of current I2V tools. In this paper, we propose a practical framework, named Follow-Your-Click, to achieve image animation with a simple user click (for specifying what to move) and a short motion prompt (for specifying how to move). Technically, we propose the first-frame masking strategy, which significantly improves the video generation quality, and a motion-augmented module equipped with a short motion prompt dataset to improve the short prompt following abilities of our model. To further control the motion speed, we propose flow-based motion magnitude control to control the speed of target movement more precisely. Our framework has simpler yet precise user control and better generation performance than previous methods. Extensive experiments compared with 7 baselines, including both commercial tools and research methods on 8 metrics, suggest the superiority of our approach. Project Page: https://follow-your-click.github.io/ (@ma2024followyourclick)

Ma, Y., Liu, H., Wang, H., Pan, H., He, Y., Yuan, J., Zeng, A., Cai, C., Shum, H.Y., Liu, W., et al.: Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. arXiv preprint arXiv:2406.01900 (2024) **Abstract:** We present Follow-Your-Emoji, a diffusion-based framework for portrait animation, which animates a reference portrait with target landmark sequences. The main challenge of portrait animation is to preserve the identity of the reference portrait and transfer the target expression to this portrait while maintaining temporal consistency and fidelity. To address these challenges, Follow-Your-Emoji equipped the powerful Stable Diffusion model with two well-designed technologies. Specifically, we first adopt a new explicit motion signal, namely expression-aware landmark, to guide the animation process. We discover this landmark can not only ensure the accurate motion alignment between the reference portrait and target motion during inference but also increase the ability to portray exaggerated expressions (i.e., large pupil movements) and avoid identity leakage. Then, we propose a facial fine-grained loss to improve the model’s ability of subtle expression perception and reference portrait appearance reconstruction by using both expression and facial masks. Accordingly, our method demonstrates significant performance in controlling the expression of freestyle portraits, including real humans, cartoons, sculptures, and even animals. By leveraging a simple and effective progressive generation strategy, we extend our model to stable long-term animation, thus increasing its potential application value. To address the lack of a benchmark for this field, we introduce EmojiBench, a comprehensive benchmark comprising diverse portrait images, driving videos, and landmarks. We show extensive evaluations on EmojiBench to verify the superiority of Follow-Your-Emoji. (@ma2024followyouremoji)

Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 (2021) **Abstract:** Guided image synthesis enables everyday users to create and edit photo-realistic images with minimum effort. The key challenge is balancing faithfulness to the user input (e.g., hand-drawn colored strokes) and realism of the synthesized image. Existing GAN-based methods attempt to achieve such balance using either conditional GANs or GAN inversions, which are challenging and often require additional training data or loss functions for individual applications. To address these issues, we introduce a new image synthesis and editing method, Stochastic Differential Editing (SDEdit), based on a diffusion model generative prior, which synthesizes realistic images by iteratively denoising through a stochastic differential equation (SDE). Given an input image with user guide of any type, SDEdit first adds noise to the input, then subsequently denoises the resulting image through the SDE prior to increase its realism. SDEdit does not require task-specific training or inversions and can naturally achieve the balance between realism and faithfulness. SDEdit significantly outperforms state-of-the-art GAN-based methods by up to 98.09% on realism and 91.72% on overall satisfaction scores, according to a human perception study, on multiple tasks, including stroke-based image synthesis and editing as well as image compositing. (@meng2021sdedit)

Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inversion for editing real images using guided diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6038–6047 (2023) (@mokady2023null)

Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 (2021) **Abstract:** Diffusion models have recently been shown to generate high-quality synthetic images, especially when paired with a guidance technique to trade off diversity for fidelity. We explore diffusion models for the problem of text-conditional image synthesis and compare two different guidance strategies: CLIP guidance and classifier-free guidance. We find that the latter is preferred by human evaluators for both photorealism and caption similarity, and often produces photorealistic samples. Samples from a 3.5 billion parameter text-conditional diffusion model using classifier-free guidance are favored by human evaluators to those from DALL-E, even when the latter uses expensive CLIP reranking. Additionally, we find that our models can be fine-tuned to perform image inpainting, enabling powerful text-driven image editing. We train a smaller model on a filtered dataset and release the code and weights at https://github.com/openai/glide-text2im. (@nichol2021glide)

Nichol, A.Q., Dhariwal, P.: Improved denoising diffusion probabilistic models. In: International conference on machine learning. pp. 8162–8171. PMLR (2021) (@nichol2021improved)

Parmar, G., Kumar Singh, K., Zhang, R., Li, Y., Lu, J., Zhu, J.Y.: Zero-shot image-to-image translation. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1–11 (2023) (@parmar2023zero)

Qi, C., Cun, X., Zhang, Y., Lei, C., Wang, X., Shan, Y., Chen, Q.: Fatezero: Fusing attentions for zero-shot text-based video editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15932–15942 (2023) (@qi2023fatezero)

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021) (@radford2021learning)

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 **1**(2),  3 (2022) **Abstract:** Contrastive models like CLIP have been shown to learn robust representations of images that capture both semantics and style. To leverage these representations for image generation, we propose a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding. We show that explicitly generating image representations improves image diversity with minimal loss in photorealism and caption similarity. Our decoders conditioned on image representations can also produce variations of an image that preserve both its semantics and style, while varying the non-essential details absent from the image representation. Moreover, the joint embedding space of CLIP enables language-guided image manipulations in a zero-shot fashion. We use diffusion models for the decoder and experiment with both autoregressive and diffusion models for the prior, finding that the latter are computationally more efficient and produce higher-quality samples. (@ramesh2022hierarchical)

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022) (@rombach2022high)

Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22500–22510 (2023) (@ruiz2023dreambooth)

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems **35**, 36479–36494 (2022) (@saharia2022photorealistic)

Salimans, T., Ho, J.: Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512 (2022) **Abstract:** Diffusion models have recently shown great promise for generative modeling, outperforming GANs on perceptual quality and autoregressive models at density estimation. A remaining downside is their slow sampling time: generating high quality samples takes many hundreds or thousands of model evaluations. Here we make two contributions to help eliminate this downside: First, we present new parameterizations of diffusion models that provide increased stability when using few sampling steps. Second, we present a method to distill a trained deterministic diffusion sampler, using many steps, into a new diffusion model that takes half as many sampling steps. We then keep progressively applying this distillation procedure to our model, halving the number of required sampling steps each time. On standard image generation benchmarks like CIFAR-10, ImageNet, and LSUN, we start out with state-of-the-art samplers taking as many as 8192 steps, and are able to distill down to models taking as few as 4 steps without losing much perceptual quality; achieving, for example, a FID of 3.0 on CIFAR-10 in 4 steps. Finally, we show that the full progressive distillation procedure does not take more time than it takes to train the original model, thus representing an efficient solution for generative modeling using diffusion at both train and test time. (@salimans2022progressive)

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems **35**, 25278–25294 (2022) (@schuhmann2022laion)

Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al.: Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792 (2022) **Abstract:** We propose Make-A-Video – an approach for directly translating the tremendous recent progress in Text-to-Image (T2I) generation to Text-to-Video (T2V). Our intuition is simple: learn what the world looks like and how it is described from paired text-image data, and learn how the world moves from unsupervised video footage. Make-A-Video has three advantages: (1) it accelerates training of the T2V model (it does not need to learn visual and multimodal representations from scratch), (2) it does not require paired text-video data, and (3) the generated videos inherit the vastness (diversity in aesthetic, fantastical depictions, etc.) of today’s image generation models. We design a simple yet effective way to build on T2I models with novel and effective spatial-temporal modules. First, we decompose the full temporal U-Net and attention tensors and approximate them in space and time. Second, we design a spatial temporal pipeline to generate high resolution and frame rate videos with a video decoder, interpolation model and two super resolution models that can enable various applications besides T2V. In all aspects, spatial and temporal resolution, faithfulness to text, and quality, Make-A-Video sets the new state-of-the-art in text-to-video generation, as determined by both qualitative and quantitative measures. (@singer2022make)

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: International conference on machine learning. pp. 2256–2265. PMLR (2015) (@sohl2015deep)

Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020) **Abstract:** Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, yet they require simulating a Markov chain for many steps to produce a sample. To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a Markovian diffusion process. We construct a class of non-Markovian diffusion processes that lead to the same training objective, but whose reverse process can be much faster to sample from. We empirically demonstrate that DDIMs can produce high quality samples $10 \\}times$ to $50 \\}times$ faster in terms of wall-clock time compared to DDPMs, allow us to trade off computation for sample quality, and can perform semantically meaningful image interpolation directly in the latent space. (@song2020denoising)

Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020) **Abstract:** Creating noise from data is easy; creating data from noise is generative modeling. We present a stochastic differential equation (SDE) that smoothly transforms a complex data distribution to a known prior distribution by slowly injecting noise, and a corresponding reverse-time SDE that transforms the prior distribution back into the data distribution by slowly removing the noise. Crucially, the reverse-time SDE depends only on the time-dependent gradient field (\\}aka, score) of the perturbed data distribution. By leveraging advances in score-based generative modeling, we can accurately estimate these scores with neural networks, and use numerical SDE solvers to generate samples. We show that this framework encapsulates previous approaches in score-based generative modeling and diffusion probabilistic modeling, allowing for new sampling procedures and new modeling capabilities. In particular, we introduce a predictor-corrector framework to correct errors in the evolution of the discretized reverse-time SDE. We also derive an equivalent neural ODE that samples from the same distribution as the SDE, but additionally enables exact likelihood computation, and improved sampling efficiency. In addition, we provide a new way to solve inverse problems with score-based models, as demonstrated with experiments on class-conditional generation, image inpainting, and colorization. Combined with multiple architectural improvements, we achieve record-breaking performance for unconditional image generation on CIFAR-10 with an Inception score of 9.89 and FID of 2.20, a competitive likelihood of 2.99 bits/dim, and demonstrate high fidelity generation of 1024 x 1024 images for the first time from a score-based generative model. (@song2020score)

Tang, L., Jia, M., Wang, Q., Phoo, C.P., Hariharan, B.: Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems **36**, 1363–1389 (2023) (@tang2023emergent)

Tumanyan, N., Geyer, M., Bagon, S., Dekel, T.: Plug-and-play diffusion features for text-driven image-to-image translation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1921–1930 (2023) (@tumanyan2023plug)

Wang, W., Jiang, Y., Xie, K., Liu, Z., Chen, H., Cao, Y., Wang, X., Shen, C.: Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599 (2023) **Abstract:** Large-scale text-to-image diffusion models achieve unprecedented success in image generation and editing. However, how to extend such success to video editing is unclear. Recent initial attempts at video editing require significant text-to-video data and computation resources for training, which is often not accessible. In this work, we propose vid2vid-zero, a simple yet effective method for zero-shot video editing. Our vid2vid-zero leverages off-the-shelf image diffusion models, and doesn’t require training on any video. At the core of our method is a null-text inversion module for text-to-video alignment, a cross-frame modeling module for temporal consistency, and a spatial regularization module for fidelity to the original video. Without any training, we leverage the dynamic nature of the attention mechanism to enable bi-directional temporal modeling at test time. Experiments and analyses show promising results in editing attributes, subjects, places, etc., in real-world videos. Code is made available at \\}url{https://github.com/baaivision/vid2vid-zero}. (@wang2023zero)

Wu, J.Z., Ge, Y., Wang, X., Lei, S.W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., Shou, M.Z.: Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7623–7633 (2023) (@wu2023tune)

Xiang, J., Huang, R., Zhang, J., Li, G., Han, X., Wei, Y.: Versvideo: Leveraging enhanced temporal diffusion models for versatile video generation. In: The Twelfth International Conference on Learning Representations (2023) (@xiang2023versvideo)

Xue, J., Wang, H., Tian, Q., Ma, Y., Wang, A., Zhao, Z., Min, S., Zhao, W., Zhang, K., Shum, H.Y., et al.: Follow-your-pose v2: Multiple-condition guided character image animation for stable pose control. arXiv e-prints pp. arXiv–2406 (2024) (@xue2024followyourposev2)

Yang, S., Zhou, Y., Liu, Z., Loy, C.C.: Rerender a video: Zero-shot text-guided video-to-video translation. In: SIGGRAPH Asia 2023 Conference Papers. pp. 1–11 (2023) (@yang2023rerender)

Yang, S., Zhou, Y., Liu, Z., Loy, C.C.: Fresco: Spatial-temporal correspondence for zero-shot video translation. arXiv preprint arXiv:2403.12962 (2024) **Abstract:** The remarkable efficacy of text-to-image diffusion models has motivated extensive exploration of their potential application in video domains. Zero-shot methods seek to extend image diffusion models to videos without necessitating model training. Recent methods mainly focus on incorporating inter-frame correspondence into attention mechanisms. However, the soft constraint imposed on determining where to attend to valid features can sometimes be insufficient, resulting in temporal inconsistency. In this paper, we introduce FRESCO, intra-frame correspondence alongside inter-frame correspondence to establish a more robust spatial-temporal constraint. This enhancement ensures a more consistent transformation of semantically similar content across frames. Beyond mere attention guidance, our approach involves an explicit update of features to achieve high spatial-temporal consistency with the input video, significantly improving the visual coherence of the resulting translated videos. Extensive experiments demonstrate the effectiveness of our proposed framework in producing high-quality, coherent videos, marking a notable improvement over existing zero-shot methods. (@yang2024fresco)

Yu, S., Sohn, K., Kim, S., Shin, J.: Video probabilistic diffusion models in projected latent space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18456–18466 (2023) (@yu2023video)

Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3836–3847 (2023) (@zhang2023adding)

</div>

# Appendix [appendix]

# Detailed Experimental Settings [sec:app_set]

In the experiment, the size of all source videos is \\(512 \times 512\\). We adopt Stable Diffusion (SD) 2.1 from the official Huggingface repository for our method. To extract the diffusion feature, following `\cite{tang2023emergent}`{=latex}, the noise of the timestep \\(t=261\\) is added to each frame of the source video. The noisy frames of video are fed into the U-net, the feature is extracted from the intermediate layer of the 2D Unet decoder. The height and weight of the diffusion feature is 64. Following previous works, at the first 40 timesteps, the diffusion features are saved during DDIM inversion and are further injected during denoising. For Spatial-temporal attention, we use the xFormers to reduce memory consumption, while it is not used in correspondence-guided attention.

# Ablation Study on the Window Size

To illustrate the influence of window size \\(l\\), we conduct the experiment on a video with 20 frames on a single A100 GPU. During the correspondence calculation process, we calculate the theoretical computational complexity, which is the total number of multiplications and additions required. We also record actual GPU memory consumed under different window sizes, the result is shown in  
eftab:abla_ws. With our sliding window strategy, the computational complexity and the GPU memory in the correspondence calculation process are significantly reduced. The visualization result is shown in  
effig:abla_ws. If the window size is too small, the motion in the video cannot be tracked, causing unsatisfying results. We choose \\(l=9\\) for the experiments in other sections, which can achieve a balance between the memory consumed and the quality of the edited video. <span id="app:abl_ws" label="app:abl_ws"></span>

<span id="tab:abla_ws" label="tab:abla_ws"></span>

<div id="tab:abla_ws" markdown="1">

|            Window Size (\\(l\\))             |   3   |  9   |  15  | w/o |
|:--------------------------------------------:|:-----:|:----:|:----:|:---:|
| Computational Complexity (\\(\times 10^9\\)) | 0.448 | 4.03 | 11.2 | 241 |
|               GPU Memory (GB)                |  11   |  14  |  18  | 32  |

**Ablation Study on the window size \\(l\\)**. w/o means that the sliding-window strategy is not applied. The sliding window strategy can significantly reduce the use of computational complexity and GPU memory.

</div>

<figure id="fig:abla_ws">
<embed src="fig/fig_appdix_abla.pdf" />
<figcaption><strong>Ablation Study on the window size <span class="math inline"><em>l</em></span>.</strong></figcaption>
</figure>

# Visualisation of the Correspondence [app:vis_cor]

We visualize the correspondence calculated by our sliding-window-based method to illustrate its effectiveness (  
effig:traj). To be specific, we calculate the correspondence based on the \\(64 \times 64\\) diffusion feature, which is extracted at the final layer of the U-net decoder. The result illustrates that our method can effectively identify the corresponding tokens.

<figure id="fig:traj">
<embed src="fig/reb_fig_coor.pdf" />
<figcaption><strong>Visualization of the correspondence in long videos.</strong> Given a long video, we first obtain the correspondence information (<span class="math inline"><em>K</em> = 3</span>) through the sliding-window strategy. Then, considering a point in the first frame (the red point in the first image of the second row), we visualize the correspondence (respectively marked in yellow, green, and blue) in each frame.</figcaption>
</figure>

# Accuracy of Correspondance

The correspondence acquired through the diffusion feature is accurate and robust. As there is no existing video dataset with the annotated keypoints on each frame, to further evaluate its accuracy quantitatively, we collect 5 videos with 30 frames and 5 videos with 60 frames and manually label some keypoints on each frame. Then we report the percentage of correct keypoints (PCK).

Specifically, for each video, given the first frame with the keypoints, we obtain the predicted corresponding keypoints on other frames through the diffusion feature. Then we evaluate the distance between the predicted points and the ground truth. The predicted point is considered to be correct if it lies in a small neighborhood of the ground truth. Finally, the total number of correctly predicted points divided by the total number of predicted points is the value of PCK. The result in  
eftab:abla_acc illustrates that the diffusion feature can accurately find the correct position in most cases for video editing.

<span id="tab:abla_acc" label="tab:abla_acc"></span>

<div id="tab:abla_acc" markdown="1">

|              Method              |   PCK    |
|:--------------------------------:|:--------:|
|   Optical-flow Correspondence    |   0.87   |
| Diffusion feature Correspondence | **0.92** |

**Accuracy of Correspondance.**

</div>

# Effectiveness of correspondence guided attention during inversion

The quality of noise obtained by inversion can significantly affect the final quality of editing. The Correspondence-Guided Attention (CGA) during inversion can increase the quality and temporal consistency of the obtained noise, which can further help to enhance the quality and consistency of edited videos. The ablation of it is shown in  
effig:corr_abla

<figure id="fig:corr_abla">
<embed src="fig/reb_fig_ablain.pdf" />
<figcaption><strong>Ablation Study about correspondence in inversion.</strong> Here <em>Without Corr</em> means not applying the correspondence-guided attention during inversion, which suffers blurring and flickering. <em>With Corr</em> means the correspondence-guided attention is applied in both inversion and denoising stages, illustrating satisfying performance.</figcaption>
</figure>

# Broader Impacts

Our work enables high-quality video editing, which is in high demand across various social media platforms, especially short video websites like TikTok. Using our method, people can easily create high-quality and creative videos, significantly reducing production costs. However, there is a potential for misuse, such as replacing the characters in videos with celebrities, which may infringe upon the celebrities’ image rights. Therefore, it is also necessary to improve relevant laws and regulations to ensure the legal use of our method.

# Limitations [apdx:lim]

Despite achieving outstanding results, our methods still encounter several limitations. First, although the correspondence calculation process is efficient through the proposed sliding window strategy, the implementation of correspondence-guided attention is still not efficient enough, leading to the extra usage of GPU memory and time (  
eftab:abla_merge). This problem is expected to be alleviated largely through the use of xFormers. We will work on it in the future.

Second, further exploration is required to optimize the application of the obtained correspondence information. In this study, we utilize the correspondence information to sample tokens during the inversion and denoising processes and do the self-attention. However, we believe that there may be more effective alternatives to self-attention that could further unleash the potential of the correspondence information.

# More Qualitative Results

We provide more qualitative results of our method to illustrate its effectiveness, which is shown in  
effig:appdix2 and  
effig:appdix1.

<figure id="fig:appdix1">
<embed src="fig/fig_appdix.pdf" />
<figcaption><strong>Qualitative results of our methods.</strong> Our method can effectively handle various kinds of prompts, generating high-quality videos. Results are best viewed in zoomed-in.</figcaption>
</figure>

<figure id="fig:appdix2">
<embed src="fig/fig_appdix_2.pdf" />
<figcaption><strong>Qualitative results of our methods.</strong> Our method can effectively handle various kinds of prompts, generating high-quality videos. Results are best viewed in zoomed-in.</figcaption>
</figure>

# NeurIPS Paper Checklist [neurips-paper-checklist]

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification: The claims made in the abstract and introduction accurately reflect the paper’s contributions and scope.

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: We have discussed the limitations of our works.

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

14. Justification: We prove that the self-attention can enhance the similarity among tokens, which is included in the appendix.

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

19. Justification: We describe the experimental details and submit the code in the supplementary materials.

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

24. Justification: We submit the code in the supplementary results while not creating a public GitHub repo.

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

29. Justification: The experiment details are specified.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: The qualitative results are important in the field of generation. The error bar is relatively not necessary.

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

39. Justification: We have described the resources required to perform our experiments.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification: Our work conform with the NeurIPS Code of Ethics in every respect.

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: We discuss this in the appendix.

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

54. Justification: Our paper poses no such risks.

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification: The creators or original owners of assets (e.g., code, data, models), used in the paper, are properly credited and the license and terms of use explicitly are mentioned and properly respected.

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

64. Justification: No new assets introduced.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification: No such experiments.

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

[^1]: Equal contribution. \\(\dagger\\) Corresponding author.
