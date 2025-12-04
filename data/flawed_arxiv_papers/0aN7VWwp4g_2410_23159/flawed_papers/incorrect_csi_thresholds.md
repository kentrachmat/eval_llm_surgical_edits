# Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting

## Abstract

Deep learning approaches have been widely adopted for precipitation nowcasting in recent years. Previous studies mainly focus on proposing new model architectures to improve pixel-wise metrics. However, they frequently result in blurry predictions which provide limited utility to forecasting operations. In this work, we propose a new Fourier Amplitude and Correlation Loss (FACL) which consists of two novel loss terms: Fourier Amplitude Loss (FAL) and Fourier Correlation Loss (FCL). FAL regularizes the Fourier amplitude of the model prediction and FCL complements the missing phase information. The two loss terms work together to replace the traditional \\(L_2\\) losses such as MSE and weighted MSE for the spatiotemporal prediction problem on signal-based data. Our method is generic, parameter-free and efficient. Extensive experiments using one synthetic dataset and three radar echo datasets demonstrate that our method improves perceptual metrics and meteorology skill scores, with a small trade-off to pixel-wise accuracy and structural similarity. Moreover, to improve the error margin in meteorological skill scores such as Critical Success Index (CSI) and Fractions Skill Score (FSS), we propose and adopt the Regional Histogram Divergence (RHD), a distance metric that considers the patch-wise similarity between signal-based imagery patterns with tolerance to local transforms. Code is available at <https://github.com/argenycw/FACL>.

# Introduction [intro]

Precipitation nowcasting refers to the task of predicting the rainfall intensity for the next few hours based on meteorological observations from remote sensing instruments such as weather radars, satellites and numerical weather prediction (NWP) models. The development of a precise precipitation nowcast algorithm is crucial to support weather forecasters and public safety, as it could facilitate timely alerts or warnings on severe precipitation and mitigate their impact on the community through early preventive actions. Sharp precipitation nowcast imagery that is perceptually similar to the actual observations (such as radar images) is equally important for weather forecasters to comprehend how the severity of precipitation will evolve in space and time, as well as to diagnose the rapid evolution of the underlying weather systems in real-time forecasting operations.

Besides the traditional optical-flow and NWP models, deep learning models have also been widely explored and adopted for precipitation nowcasting in recent years. The research community generally formulates the task as a spatiotemporal prediction problem, where a sequence of input radar or satellite maps is given, and the future sequence needs to be predicted or generated. Although multiple previous attempts proposed solid improvements to the model to grasp the spatiotemporal dynamics, deep learning models can result in blurry predictions in real-life datasets featuring precipitation patterns such as radar echo and satellite imagery. Consequently, they provide limited operational utility `\cite{ravuri2021skilful}`{=latex} in weather forecasts.

The blurry prediction in multiple deep learning models is believed to be caused by the use of pixel-wise losses such as the Mean Squared Error (MSE), which entangles the probability into model prediction. In other words, the uncertainty of the image transformation leads to obfuscation of the surrounding pixels in the prediction. Nevertheless, solely improving the model capability could not resolve this issue due to the high spatial randomness of the atmospheric dynamics. In order to suppress the ambiguity of the model output, an emerging approach is to utilize generative models such as generative adversarial networks (GANs) and diffusion models. In this paper, we introduce an alternative approach, which is to modify the loss function such that the model focuses on recovering the high-frequency patterns. By utilizing the Fourier domain, we would like to shed light on a deterministic, non-generative method that can sharpen the spatiotemporal predictions with negligible sacrifice to its correctness.

To achieve the desired sharpness, we propose the Fourier Amplitude Loss (FAL), a loss term that improves the prediction of high frequencies by regularizing the amplitude component in the Fourier space. Supported by empirical validation, we further propose the Fourier Correlation Loss (FCL), a complementary loss term that provides information on the overall image structure. Furthermore, we have developed a training mechanism that alternates between FAL and FCL based on an increasing probability of employing FAL throughout the training steps. We name this combined loss function the Fourier Amplitude and Correlation Loss (FACL). FACL is computationally efficient, parameter-free, and model-agnostic and it can be directly applied to a wide range of state-of-the-art deep neural networks and even generative models. Extensive experiments show that compared to MSE, FACL results in forecasts that are both more realistic and more *skillful* (i.e., high performance with respect to several meteorological skill scores). To the best of our knowledge, we are the first to substantially replace the spatial MSE loss with spectral losses without using generative components on the spatiotemporal prediction problem, demonstrating the novelty and significance of our approach.

Our main contributions are summarized as follows:

<div class="compactitem" markdown="1">

We propose the Fourier Amplitude and Correlation Loss (FACL), which is constituted by sampling between the Fourier Amplitude Loss (FAL) for regularizing the spatial frequency of the predictions to enable clarity and sharpness, and the Fourier Correlation Loss (FCL), a modified loss term that is cohesive with FAL to capture the overall image structure.

Theoretical and empirical studies show that FAL boosts the image sharpness significantly while FCL complements the missing information for accuracy.

We apply FACL to replace the MSE reconstruction loss in generative models. Results show that generative models with FACL perform better with respect to most of the metrics.

We propose the Regional Histogram Divergence (RHD), a quantitative metric to measure the distance between two signal-based imagery patterns with tolerance to deformations. RHD considers both the regional similarity and the visual likeness to the target.

</div>

# Related Works [sec:related]

## Precipitation Nowcasting as a Spatiotemporal Prediction Problem [sec:spatiotemporal-prediction]

Previous works generally formulate precipitation nowcasting as a spatiotemporal predictive learning problem. Given a sequence of observed tensors with length \\(t\\): \\(X_1, X_2, ..., X_t\\), the problem is to predict the future \\(k\\) tensors formulated as follows: \\[%X_{t+1}, ..., X_{t+k} = 
    \mathop{\mathrm{arg\,max}}_{X_{t+1}, ..., X_{t+k}} p(X_{t+1}, ... , X_{t+k} \mid X_1, X_2, ... , X_t)\\]

Based on this formulation, numerous variations of convolutional RNN models were proposed to model both spatial and temporal relationships in the data. ConvLSTM `\cite{shi2015convolutional}`{=latex} first proposed to integrate convolutional layers into LSTM cells, with the recurrence forming an encoder-forecaster architecture. PredRNN `\cite{wang2017predrnn}`{=latex} replaced the ConvLSTM units with ST-LSTM units and modified the structure such that the hidden states flow in both spatial and temporal dimensions in a zigzag pattern. MIM `\cite{wang2019memory}`{=latex} replaced the forget gate in ST-LSTM with another RNN unit, forming a memory-in-memory structure to learn higher-order non-stationarity. Moreover, advanced modifications such as reversed scheduled sampling `\cite{wang2022predrnn}`{=latex}, gradient highway `\cite{wang2018predrnn}`{=latex}, etc. `\cite{wang2019eidetic, guen2020disentangling}`{=latex} were proposed to further improve the overall performance of the model. With the breakthroughs brought by transformers and self-attention mechanism, space-time transformer-based models such as Rainformer `\cite{bai2022rainformer}`{=latex} and Earthformer `\cite{gao2022earthformer}`{=latex} were proposed to model complex and long-range dependencies.

On the other hand, CNN models have also been widely explored for the task as a video prediction problem. Inspired by the U-Net structure used in earlier works `\cite{ayzel2020rainnet,trebing2021smaatunet}`{=latex}, SimVP achieves remarkable performance and efficiency by adopting an encoder-translator-decoder structure with mostly convolutional operations. Among the parts, the translator (temporal module) is found to benefit from MetaFormer (an architecture with both token mixer and channel mixer) in subsequent studies `\cite{tan2023openstl, ye2023msstnet}`{=latex}. TAU `\cite{tan2023temporal}`{=latex} further demonstrated the effectiveness of the structure by adopting depthwise convolution followed by \\(1\times1\\) convolution as the temporal module.

Conventional precipitation nowcasting tasks and video prediction tasks evaluate the output mainly with pixel-wise or structural metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE) and Structural Similarity (SSIM) Index. To better consider the hits and misses of signal-based reflectivity, the Critical Success Index (CSI; equivalent to Intersection over Union, IoU), Fractions Skill Score (FSS) and Heidke Skill Score (HSS) belong to another type of metrics widely used in meteorology. To distinguish these scores from those used in the traditional machine learning literature, we refer to this metric type as *skill scores* in the remaining sections of the paper.

## A Non-deterministic Perspective on Atmospheric Instability

Traditional models can result in blurry predictions at longer lead times, causing difficulty in forecasting operations. To address it, recent works leverage generative models such as GANs and diffusion models to promote realistic forecasts which could bring more insightful observation to forecasting operations. DGMR `\cite{ravuri2021skilful}`{=latex} utilizes a GAN framework with discriminators in both the spatial and temporal dimensions to ensure that the predicted images are sufficiently realistic and cohesive. LDCast `\cite{leinonen2023latent}`{=latex} uses latent diffusion to generate a diverse set of outputs for ensemble forecasting. Meanwhile, the literature in video generation strives to generate realistic output frames with generative models. PreDiff `\cite{gao2023prediff}`{=latex} introduces a knowledge alignment mechanism with domain-specific constraints while adopting a latent diffuser for quality forecasts. DiffCast `\cite{yu2024diffcast}`{=latex} appends a diffusion component as an auxiliary module to improve the realisticity of the forecasts. It is worth mentioning that the literature in video generation `\cite{ravuri2021skilful, yu2023magvit, skorokhodov2022styleganv, voleti2022mvcd, hoppe2022diffusion}`{=latex} also exhibits potential in generating high-quality nowcastings despite not specifically being designed to handle precipitation. Unlike works in video prediction, instead of evaluating the output quality with pixel-wise similarity, perceptual metrics such as LPIPS `\cite{zhang2018perceptual}`{=latex} and Fréchet Video Distance (FVD) `\cite{unterthiner2019fvd}`{=latex} are predominantly used.

These works usually formulate the task as an unsupervised or semi-supervised learning problem with the results being non-deterministic based on a random prior, enabling the possibility of ensemble prediction. However, studying each prediction individually is less reliable as the prediction is unexplainably affected by the random prior. Furthermore, the inference efficiency of the diffusion model is poor due to the iterative nature of the reverse diffusion sampling process. Concerning the drawbacks of generative models, our method is proposed to be efficient, deterministic, and accurate at both the pixel and perceptual levels, bridging the advantages of both probabilistic video prediction and non-deterministic video generation.

## Supervised Learning Problems That Utilize Fourier Transform

Spectral analysis in the Fourier space is a common practice for DNNs to study the features in terms of frequency. Rahaman et al. `\cite{rahaman2019on}`{=latex} proposed a property known as the spectral bias, which causes DNN models to be biased towards low-frequency functions. A follow-up study `\cite{tancik2020fourier}`{=latex} theoretically showed that DNN models have a much slower convergence rate toward high-frequency components. Such observations motivate subsequent works to apply Fourier-based loss terms extensively in tasks such as super-resolution (SR) where fine details are crucial.

Despite the existence of works that apply Fourier transform amid the model feed-forward pipeline `\cite{guibas2022adaptive, xu2020learning, hertz2021sape, landgraf2022pins,sitzmann2020implicit}`{=latex}, here we focus on works that utilize spectral transform in the loss function or as a regularization term. Inspired by the JPEG compression mechanism, the Frequency Domain Perceptual Loss `\cite{sims2021frequency}`{=latex} compares the Discrete Cosine Transform (DCT) of the model output in 8\\(\times\\)`<!-- -->`{=html}8 non-overlapping patches: \\(L(y, \hat{y}) = c \odot \lVert \text{DCT}(y) - \text{DCT}(\hat{y})\rVert_2^2\\), where \\(c\\) is a vector of constants computed from the quantization table and training set. The Focal Frequency Loss `\cite{jiang2021focal}`{=latex} compares the element-wise weighted Fast Fourier Transform (FFT) output: \\(L_\text{FFL} = \frac{1}{MN} \sum_{u=0}^{M-1}\sum_{v=0}^{N-1}{w(u, v)\lvert \text{FFT}(y)_{u,v} - \text{FFT}(\hat{y})_{u,v} \rvert^2}\\), where \\(w(\cdot, \cdot)\\) is a dynamic weight matrix and \\(\lvert\cdot\rvert\\) refers to the absolute operator on complex numbers. Moreover, the Fourier Space Loss `\cite{fuoli2021fourier}`{=latex} decomposes the Fourier output (in complex) into amplitude and phase and measures their difference separately as a GAN loss component.

Although these losses were proposed specifically for the SR task, we find the problem setting similar to spatiotemporal forecasting in terms of the requirement for high-frequency fine details and the involvement of a ground-truth label. While taking advantage of the Fourier space as a spectral analysis is intuitive, choosing the proper distance metrics and additional weighting is tricky. This motivates us to propose a new loss function with consideration of the spectral property on the spatiotemporal forecasting problem.

# Our Methods [sec:method]

In this section, we start by arguing why a naive implementation of the Fourier loss does not benefit the model compared with the MSE loss in the image space. Then, we will discuss the motivation and details of our proposed FACL.

## Preliminaries

An image \\(X\\) can be interpreted as a 2D matrix with the transformed Fourier series, \\(F\\). The orthonormalized Discrete Fourier Transform (DFT) output and its corresponding inverse Discrete Fourier Transform are formulated as: \\[\begin{aligned}
\label{eqn:dfft}
    F_{pq} = \frac{1}{\sqrt{MN}} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} X_{mn}e^{-i2\pi(\frac{mp}{M} + \frac{nq}{N})}
    ;\;
    X_{mn} = \frac{1}{\sqrt{MN}} \sum_{p=0}^{M-1} \sum_{q=0}^{N-1} F_{pq} e^{i2\pi(\frac{mp}{M} + \frac{nq}{N})}
\end{aligned}\\] where \\(M\\) and \\(N\\) are the height and width, respectively, of the image \\(X\\).

To constrain model convergence via the spatial frequency components of its prediction, one naive design is to regularize the \\(L_2\\) norm of the displacement vector between the ground truth and prediction in the Fourier space apart from the image space. Parseval’s Theorem shows that such design is linearly proportional to the spatial MSE loss, and the detailed proof can be found in Appendix <a href="#app:proof_l2_F" data-reference-type="ref" data-reference="app:proof_l2_F">7</a>.

Since this straightforward regularization does not differ from the MSE loss in the image space, the common adaptations from previous works are either to apply weighting on different frequencies or to decompose the Fourier features into amplitude \\(|F|\\) and phase \\(\theta_F\\) with the following definitions: \\[\begin{aligned}
\label{eq:amp_phase}
    |F| = \sqrt{F_\text{real}^2 + F_\text{imag}^2} ; \; \theta_F = \text{arctan}(\frac{F_\text{real}}{F_\text{imag}}) ,
\end{aligned}\\] where \\(F_\text{real}\\) and \\(F_\text{imag}\\) are the real and imaginary parts, respectively, of the complex Fourier vector \\(F\\).

## Fourier Amplitude Loss (FAL) [sec:FAL]

As the spectral bias indicates the lack of attention to the high-frequency components, we encourage the model to consider high-frequency patterns by applying a loss on the amplitude of each frequency band. Similar to previous works, we first apply DFT to obtain the spectral information. Using Equation <a href="#eq:amp_phase" data-reference-type="eqref" data-reference="eq:amp_phase">[eq:amp_phase]</a>, we extract only the Fourier amplitudes (\\(|F|\\)) in the Fourier space and compare them in \\(L_2\\): \\[\label{eqn:fft-amp}
    \text{FAL}(X, \hat{X}) = \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}(|F|_{pq} - |\hat{F}|_{pq})^{2}\\] where \\(F\\) is the DFT output of \\(X\\) as formulated in Eq. <a href="#eqn:dfft" data-reference-type="eqref" data-reference="eqn:dfft">[eqn:dfft]</a>. Note that the formulation is subtly different from minimizing the \\(L_{2}\\) norm of the displacement vector that prediction deviates from ground truth in the Fourier domain. The new formulation based on the Fourier amplitude of the images only is invariant to global translation. This reduces the spatial constraint induced by MAE and MSE losses. A detailed analysis can be found in Appendix <a href="#app:fal_study" data-reference-type="ref" data-reference="app:fal_study">9</a>.

Despite retaining the high-frequencies by dropping the Fourier phase, FAL alone is insufficient to reconstruct the image. As \\(X \mapsto |F|\\) is a many-to-one mapping, there exist multiple \\(X\\) to have the same Fourier amplitude matrix. Thus, simply minimizing Eq. <a href="#eqn:fft-amp" data-reference-type="eqref" data-reference="eqn:fft-amp">[eqn:fft-amp]</a> can likely converge to an undesirable critical point. A high-level interpretation is that only image sharpness is retained by this loss while the information regarding the actual shape and position is lost with the Fourier phase discarded. Hence, on top of FAL, we require another loss term to compensate for the missing information, leading to our upcoming proposal of the FCL term. An alternative perspective via the mathematical formulation can be found in Appendix <a href="#app:problem_when_amp_alone" data-reference-type="ref" data-reference="app:problem_when_amp_alone">8</a>.

## Fourier Correlation Loss (FCL)

To remedy the missing information resulting from FAL, there are several approaches to take the image structure into account. A straightforward way is minimizing the difference of the Fourier phase between the prediction and the label, but it fails as \\(\theta_{F}\\) obtained under DFT is discontinuous. Another approach is to compute the cosine distance in the Fourier domain without extracting \\(\theta_{F}\\) directly. However, our preliminary experiments reveal that such formulation is unstable in reconstructing the image structure. Ultimately, we propose to implement the correlation between the generated output and ground truth in the Fourier domain and adopt it as the Fourier Correlation Loss (FCL) in our proposed loss: \\[\label{eqn:FSC}
    \text{FCL}(X, \hat{X}) = 1 - \frac{\frac{1}{2}\sum [F\hat{F}^{*} + \hat{F}F^{*}]}{\sqrt{\sum |F|^{2} \sum|\hat{F}|^{2}}},\\] where \\(\sum\\) here is a shorthand for the summation over all elements of the Fourier features and \\(*\\) denotes the complex conjugate of the vector. FCL plays a significant role during training as it is responsible for learning the proper image structure while FAL can be treated as a regularization to promote the high-frequency components that FCL fails to capture.

The formulation of FCL has a similar format to the Fourier Ring Correlation (FRC) and Fourier Shell Correlation (FSC) widely used in image restoration and super-resolution of cryo-electron microscopy `\cite{koho2019fourier, banterle2013fourier, van2005fourier, kaczmar2022image, culley2018quantitative, berberich2021fourier}`{=latex}. However, both FRC and FSC pre-define a specific region of interest (either a ring or a shell) on the Fourier features. In contrast, we extend the region of interest to the entire map, considering the global spectral bands with all frequencies. To ensure the score is real and commutative, we take the average of \\(F\hat{F}^*\\) and \\(\hat{F}F^*\\) in the numerator. The denominator performs normalization such that FCL only focuses on the image structure in the global view rather than the absolute brightness. Unlike FRC (without \\(1-\\)), FCL spans the range \[0, 2\], where larger values refer to a negative correlation and smaller values refer to a positive correlation. Further analysis of FCL from the gradient aspect can be found in Appendix <a href="#app:fsc_grad" data-reference-type="ref" data-reference="app:fsc_grad">10</a>.

## Proposed Approach: Random Selection between FAL and FCL [sec:dynamic_weight]

<figure id="fig:prob_thres">
<img src="./figures/prob_thres.png"" />
<figcaption>The pre-defined probability threshold function <span class="math inline"><em>P</em>(<em>t</em>)</span> over training steps <span class="math inline"><em>t</em></span> with <span class="math inline"><em>T</em></span> total steps. <span class="math inline"><em>α</em></span> determines the ratio of the training steps where <span class="math inline"><em>P</em>(<em>t</em>) = 0</span>.</figcaption>
</figure>

While it is straightforward to apply the overall loss function as a linear combination of FAL and FCL, we find it tricky to determine the weighting of the components in our preliminary studies. Instead, we offer a more controllable solution – to alternate FAL and FCL as shown below: \\[\label{eq:final_loss}
\text{FACL}(X, \hat{X}, t) = 
\begin{cases}
    & \text{FAL} (X,\hat{X})\text{, if } p > P(t) \\
    & \text{FCL} (X, \hat{X}) \text{, otherwise}
\end{cases}\\] where \\(p\\) is sampled randomly and uniformly in \[0, 1\] and \\(P(t)\\) is a pre-defined threshold decreasing during the training process as shown in Figure <a href="#fig:prob_thres" data-reference-type="ref" data-reference="fig:prob_thres">1</a>. \\(P(t)\\) always decreases from 1 to 0 such that the model is first trained with \\(100\%\\) FCL that takes image structure into account, and then the models are more frequently trained with FAL which improves the image sharpness.

Since FCL loses information on the overall brightness, the model could not achieve proper brightness at the early stage where FCL dominates the learning objective. To address it, we append a sigmoid function in the output layer of the model. This constrains the model output in the range \[0, 1\] to prevent the model from converging to a sub-optimal state with an undesirable range of output values.

Overall, the following modifications are applied to the models:

<div class="compactitem" markdown="1">

Training loss function of the models involving FAL and FCL is formulated in Eq. <a href="#eq:final_loss" data-reference-type="eqref" data-reference="eq:final_loss">[eq:final_loss]</a>.

A sigmoid layer is appended to the end of the model. For RNN models, the sigmoid function is applied before the output of the last RNN stack.

To coordinate with the decreasing threshold, the cosine annealing learning rate scheduler is used rather than the conventional reduce-on-plateau scheduler.

</div>

## A New Metric: Regional Histogram Divergence (RHD)

Previous works in video prediction tend to use pixel-wise metrics such as MSE and MAE to measure the difference between the prediction and labels. Such a choice of metrics might not fit spatiotemporal data for two reasons: (1) reasonable pixel shifts are highly penalized, and (2) the overall distribution of values is ignored. This encourages the models to output blurry predictions while regional uncertainty diffuses outward over time. By inverse, deep perceptual metrics such as LPIPS, Inception Score (IS) and Fréchet Video Distance (FVD) suffer from the knowledge bias between multi-channel pictures (as pre-trained on ImageNet) and monotonic signal-based intensities.

One of the metrics that consider both the previous two factors is the Fractional Skill Score (FSS), which is widely used in meteorology. After splitting the image into \\(N_x\times N_y\\) smaller patches, where \\(N_{x}\\) and \\(N_{y}\\) control the shift of precipitation events we tolerate, we obtain the FSS score as follows: \\[\label{eq:fss}
\newcommand\nn{\sum_{i=1}^{N_x}\sum_{j=1}^{N_y}}
\text{FSS} = 1 - \frac{\nn{(F_{i,j} - O_{i,j})^2}}{\nn{F^2_{i,j}} + \nn{O^2_{i,j}}} ,\\] where \\(F_{i,j}\\) and \\(O_{i,j}\\) refer to the fraction of predicted positives and fraction of observed positives, respectively, of the patch in the \\(i\\)-th row and \\(j\\)-th column. Based on this formulation, the intensities are free to reposition within the patch window, granting tolerance to translation and deformation. Nevertheless, one drawback of FSS is that the pixel range is only categorized into two classes: positives and negatives. For a threshold of \\(0.5\\), a pixel value of \\(0\\) is treated the same as a pixel value of \\(0.49\\), resulting in a huge error when viewing the per-patch precision. This means that the choice of threshold induces a bias in evaluating the forecasting performance of models.

To improve the representation, we propose the Regional Histogram Divergence (RHD), a variation of FSS that exhibits smaller errors within a class. Instead of categorizing the pixel values into ‘hits’ and ‘misses’, we divide the values into \\(n\\) bins and count the frequency of each bin, obtaining a histogram for each patch. Next, we compare the average Kullback–Leibler (KL) divergence on the histograms. Mathematically, the RHD between two sets of bins can be expressed as: \\[\begin{aligned}
\text{RHD}
& = \frac{1}{N_xN_y} \sum_{i=1}^{N_x}\sum_{j=1}^{N_y} D_\text{KL}(O'_{i,j} || F'_{i,j}) \nonumber
= \frac{1}{N_xN_y} \sum_{i=1}^{N_x}\sum_{j=1}^{N_y}\sum_{x\in\mathcal{X}}O'_{i,j}(x) \log{\frac{O'_{i,j}(x)}{F'_{i,j}(x)}},
\end{aligned}\\] where \\(F'_{i,j}\\) and \\(O'_{i,j}\\) correspond to the predicted and observed discrete probability distributions, respectively, among the set of bins \\(\mathcal{X}\\) of the patch in the \\(i\\)-th row and \\(j\\)-th column.

Different from FSS where the proportions of positives are subtracted directly, RHD instead compares the distributional difference in the context of the histograms. This not only increases the precision of each class/bin, but also heavily penalizes blurs since blurring forms a Gaussian-like distribution in the histograms while sharp intensities should have an ‘M-shape’ bimodal distribution. If the patch-wise distribution of the two images is identical, the corresponding RHD is 0. The larger the RHD is, the more different the two sets of patches behave. Furthermore, RHD is formulated to be a mean KL divergence so it is always non-negative.

For simplicity and consistency, we choose the number of bins to be 10, divided uniformly within the range \\([0, 1]\\) for all datasets in our experiments. In real-life applications, non-uniform division can be applied for data in non-linear scales such as radar echo (in dBZ) to highlight specific ranges of values. When we compute the histograms, as 0 dominates in the imagery, we apply a threshold \\(\epsilon = 10^{-5}\\) to exclude all small intensities.

# Experiments [sec:experiments]

## Experimental Settings

We evaluate the performance of our proposed method on a synthetic dataset and three radar echo datasets, namely, Stochastic Moving-MNIST, SEVIR `\cite{veillette2020sevir}`{=latex}, MeteoNet `\cite{larvor2020meteonet}`{=latex} and HKO-7 `\cite{shi2017trajgru}`{=latex}. A more detailed description for each dataset can be found in Appendix <a href="#app:datasets" data-reference-type="ref" data-reference="app:datasets">6</a>. To show that our method is effective and generic, we selected ConvLSTM `\cite{shi2015convolutional}`{=latex} and PredRNN `\cite{wang2017predrnn}`{=latex} (reported in Appendix <a href="#app:predrnn" data-reference-type="ref" data-reference="app:predrnn">14</a>), two RNN-based models with different recurrence paths; SimVP `\cite{gao2022simvp}`{=latex}, a CNN-based model and Earthformer `\cite{gao2022earthformer}`{=latex}, a transformer-based model. We trained the models with two variants: conventional MSE and FACL (as formulated in Eq. <a href="#eq:final_loss" data-reference-type="eqref" data-reference="eq:final_loss">[eq:final_loss]</a>). To compare with generative models as references, we also report the results of LDCast `\cite{leinonen2023latent}`{=latex} (latent diffusion) and MCVD `\cite{voleti2022mvcd}`{=latex} (denoising diffusion) for all datasets. For Stochastic Moving-MNIST, we report two more models, i.e., PreDiff `\cite{gao2023prediff}`{=latex} (latent diffusion) and STRPM `\cite{chang2022strpm}`{=latex} (GAN-based). Appendix <a href="#app:hyperparam" data-reference-type="ref" data-reference="app:hyperparam">18</a> reports the detailed setup and hyper-parameters.

In the upcoming sections, we will first present the setup and experimental results on the Stochastic Moving-MNIST dataset. After that, we will test the models with three real-world radar echo datasets. Extra studies on our methods are reported in the Appendix. Specifically, we report the ablation study of FAL, FCL and \\(\alpha\\) in Appendix <a href="#app:abla_existence" data-reference-type="ref" data-reference="app:abla_existence">11</a>, the running time of FACL in Appendix <a href="#app:running_time" data-reference-type="ref" data-reference="app:running_time">12</a>, experiments on additional datasets in Appendix <a href="#app:nbody" data-reference-type="ref" data-reference="app:nbody">13</a>, comparison with other potential loss functions in Appendix <a href="#app:other_losses" data-reference-type="ref" data-reference="app:other_losses">15</a>, the performance when applying FACL to generative models in Appendix <a href="#app:facl_generative" data-reference-type="ref" data-reference="app:facl_generative">16</a>, and characteristic analysis of RHD in Appendix <a href="#app:rhd" data-reference-type="ref" data-reference="app:rhd">17</a>. To demonstrate the advantages of our method against counterparts for precipitation nowcasting, video prediction and video generation, we evaluate the models with a union of metrics from the areas. Specifically, we report the MAE and SSIM to show the pixel-wise and structural accuracy; LPIPS and FVD to show the deep perceptual similarity to ground truth; FSS and RHD to measure the similarity of the intensity distribution in different regions. For radar echo datasets, we further include the CSI and pooled CSI to reveal the models’ capability of identifying potential extreme weather. Such a combination of metrics is believed to facilitate a comprehensive understanding of the pros and cons of the current state-of-the-art in precipitation nowcasting.

## A Stochastic Modification of Moving-MNIST

The Moving-MNIST dataset has been a common benchmark to evaluate how well a model could predict motion in spatial preservation and temporal extrapolation. However, the nature of the Moving-MNIST is highly deterministic, which does not resemble the chaotic nature of the atmospheric system. Previous adaptations attempted to simulate the physical dynamics by introducing a set of complex motions such as rotation and scaling `\cite{gao2022earthformer}`{=latex} or by applying an external force on collision `\cite{denton2018stochastic}`{=latex}. We argue that the fundamental reason causing the blur in precipitation nowcasting is the intrinsic stochasticity of the motion caused by external factors unseen in the weather dataset, such as orographic effects, vertical wind shear, interaction with other weather systems, etc. Trained with such stochasticity, regular models with pixel-wise loss could consistently fail to provide quality prediction in the future lead time.

To verify our claim, we introduce a simplistic modification to the Moving-MNIST dataset. The standard Moving-MNIST dataset contains handwritten digits sampled from the MNIST dataset moving and bouncing with a constant velocity \\((u_0, v_0)\\) on the \\(64\times 64\\) image plane. To introduce stochasticity, we perturb the velocity with a random Gaussian noise \\(\epsilon\\) at each time step. Details of the perturbation are shown in Appendix <a href="#app:datasets" data-reference-type="ref" data-reference="app:datasets">6</a>. In the upcoming sections, we dub this dataset Stochastic Moving-MNIST and apply the experimental setting to this synthetic dataset. The performance of combinations of different losses and models can be found in Table <a href="#tab:loss_smmnist" data-reference-type="ref" data-reference="tab:loss_smmnist">1</a> and qualitative visualizations of the corresponding methods are shown in Figure <a href="#fig:vis_smmnist" data-reference-type="ref" data-reference="fig:vis_smmnist">2</a> and Appendix <a href="#app:more_visualize" data-reference-type="ref" data-reference="app:more_visualize">19</a>. Note that the Stochastic Moving-MNIST is used in both training and evaluation to ensure that the models are well exposed to motion randomness.

<div id="tab:loss_smmnist" markdown="1">

<table>
<caption>Comparison of the quantitative performance of different losses for models trained on the Stochastic Moving-MNIST. The better score between MSE and FACL is highlighted in bold. </caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;">Type</td>
<td rowspan="2" style="text-align: center;">Model</td>
<td rowspan="2" style="text-align: center;">Loss</td>
<td colspan="2" style="text-align: center;">Pixel-wise/Structural</td>
<td colspan="2" style="text-align: center;">Perceptual</td>
<td style="text-align: center;">Skill</td>
<td style="text-align: center;">Proposed</td>
</tr>
<tr>
<td style="text-align: center;">MAE<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">SSIM<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">LPIPS<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">FVD<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">FSS<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">RHD<span><span class="math inline">↓</span></span></td>
</tr>
<tr>
<td rowspan="8" style="text-align: center;">Pred.</td>
<td rowspan="2" style="text-align: center;">ConvLSTM</td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;">196.4</td>
<td style="text-align: center;">0.6975</td>
<td style="text-align: center;">0.2538</td>
<td style="text-align: center;">451.5</td>
<td style="text-align: center;">0.6148</td>
<td style="text-align: center;">1.1504</td>
</tr>
<tr>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;"><strong><span>180.1</span></strong></td>
<td style="text-align: center;"><strong><span>0.7463</span></strong></td>
<td style="text-align: center;"><strong><span>0.1092</span></strong></td>
<td style="text-align: center;"><strong><span>82.3</span></strong></td>
<td style="text-align: center;"><strong><span>0.8172</span></strong></td>
<td style="text-align: center;"><strong><span>0.3391</span></strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">PredRNN</td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;">173.8</td>
<td style="text-align: center;">0.7566</td>
<td style="text-align: center;">0.1875</td>
<td style="text-align: center;">337.8</td>
<td style="text-align: center;">0.7443</td>
<td style="text-align: center;">0.9559</td>
</tr>
<tr>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;"><strong><span>162.1</span></strong></td>
<td style="text-align: center;"><strong><span>0.7812</span></strong></td>
<td style="text-align: center;"><strong><span>0.0869</span></strong></td>
<td style="text-align: center;"><strong><span>63.3</span></strong></td>
<td style="text-align: center;"><strong><span>0.8549</span></strong></td>
<td style="text-align: center;"><strong><span>0.3000</span></strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">SimVP</td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;"><strong><span>175.5</span></strong></td>
<td style="text-align: center;"><strong><span>0.7547</span></strong></td>
<td style="text-align: center;">0.1943</td>
<td style="text-align: center;">350.6</td>
<td style="text-align: center;">0.7275</td>
<td style="text-align: center;">0.9819</td>
</tr>
<tr>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;">180.2</td>
<td style="text-align: center;">0.7394</td>
<td style="text-align: center;"><strong><span>0.1335</span></strong></td>
<td style="text-align: center;"><strong><span>211.9</span></strong></td>
<td style="text-align: center;"><strong><span>0.8168</span></strong></td>
<td style="text-align: center;"><strong><span>0.3579</span></strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">Earthformer</td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;">171.5</td>
<td style="text-align: center;">0.7641</td>
<td style="text-align: center;">0.1828</td>
<td style="text-align: center;">320.3</td>
<td style="text-align: center;">0.7532</td>
<td style="text-align: center;">0.9407</td>
</tr>
<tr>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;"><strong><span>167.6</span></strong></td>
<td style="text-align: center;"><strong><span>0.7768</span></strong></td>
<td style="text-align: center;"><strong><span>0.0890</span></strong></td>
<td style="text-align: center;"><strong><span>64.6</span></strong></td>
<td style="text-align: center;"><strong><span>0.8463</span></strong></td>
<td style="text-align: center;"><strong><span>0.3230</span></strong></td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">Gen.</td>
<td style="text-align: center;">LDCast<span class="math inline"><sup>*</sup></span></td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">234.0</td>
<td style="text-align: center;">0.7053</td>
<td style="text-align: center;">0.1541</td>
<td style="text-align: center;">110.7</td>
<td style="text-align: center;">0.6645</td>
<td style="text-align: center;">0.4343</td>
</tr>
<tr>
<td style="text-align: center;">MCVD</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">219.8</td>
<td style="text-align: center;">0.7125</td>
<td style="text-align: center;">0.1033</td>
<td style="text-align: center;">44.7</td>
<td style="text-align: center;">0.7184</td>
<td style="text-align: center;">0.3941</td>
</tr>
<tr>
<td style="text-align: center;">STRPM</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">154.0</td>
<td style="text-align: center;">0.7912</td>
<td style="text-align: center;">0.1017</td>
<td style="text-align: center;">117.4</td>
<td style="text-align: center;">0.8337</td>
<td style="text-align: center;">0.3216</td>
</tr>
<tr>
<td style="text-align: center;">PreDiff</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">190.2</td>
<td style="text-align: center;">0.7570</td>
<td style="text-align: center;">0.0709</td>
<td style="text-align: center;">30.8</td>
<td style="text-align: center;">0.7975</td>
<td style="text-align: center;">0.3052</td>
</tr>
<tr>
<td colspan="9" style="text-align: left;">* The experiment setting for LDCast is changed to 8-in-8-out due to model constraints.</td>
</tr>
</tbody>
</table>

</div>

<figure id="fig:vis_smmnist">
<img src="./figures/convlstm_smmnist.png"" />
<figcaption>Output frames of the ConvLSTM model trained with different losses on Stochastic Moving-MNIST. From top to bottom: Input, Ground Truth, MSE, FACL.</figcaption>
</figure>

In Table <a href="#tab:loss_smmnist" data-reference-type="ref" data-reference="tab:loss_smmnist">1</a>, our modification drastically improves the sharpness and realisticity for all tested models, as reflected by the vast reduction in LPIPS and RHD. In particular, FACL reduces up to \\(57\%\\) of LPIPS and \\(71\%\\) of RHD for the ConvLSTM model. The pixel-wise and structural metrics between the two losses are comparable. On the other hand, generative models result in much poorer MAE and SSIM, with skill scores like FSS still being worse than most of the baseline models. In Figure <a href="#fig:vis_smmnist" data-reference-type="ref" data-reference="fig:vis_smmnist">2</a>, we can observe that the model trained with MSE cannot reconstruct a clear spatial pattern, especially in the subsequent frames, while the model trained with FACL yields much sharper and higher quality outputs. Consequently, we can conclude that in this setting with the synthetic Stochastic Moving-MNIST, FACL demonstrates a significant improvement in perceptual metrics and skill scores, with the quality on par with that of generative models.

## Performance on Radar-based Datasets

In this section, we extend the previous setup to general radar‐based datasets. Apart from the distance metrics used in the last section, we further report the CSI with different pooling sizes. Following previous works `\cite{shi2017trajgru, gao2022earthformer}`{=latex}, we measure multiple CSI scores with a **unified** set of thresholds \({16, 74, 133, 160, 181, 219}\) dBZ for **both SEVIR and MeteoNet**, and \({84, 117, 140, 158, 185}\) dBZ for HKO-7. Employing an identical threshold configuration across SEVIR and MeteoNet greatly simplifies the evaluation protocol, ensures direct cross-dataset comparability and emphasises the robustness of the proposed loss under heterogeneous geographical settings.

The visualisations can be found in Figure <a href="#fig:loss_sevir" data-reference-type="ref" data-reference="fig:loss_sevir">3</a> and more in Appendix <a href="#app:more_visualize" data-reference-type="ref" data-reference="app:more_visualize">19</a>.

<div id="tab:loss_radar" markdown="1">

<table>
<caption>Comparison of the quantitative performance of different losses for models trained on SEVIR, MeteoNet and HKO-7. MAE metrics is in the scale of <span class="math inline">10<sup>−3</sup></span>. The better score between MSE and FACL is highlighted in bold. </caption>
<tbody>
<tr>
<td colspan="2" rowspan="2" style="text-align: center;">Type</td>
<td rowspan="2" style="text-align: center;">Model</td>
<td rowspan="2" style="text-align: center;">Loss</td>
<td colspan="2" style="text-align: center;">Pixelwise/Structural</td>
<td colspan="2" style="text-align: center;">Perceptral</td>
<td colspan="4" style="text-align: center;">Skill</td>
<td style="text-align: center;">Proposed</td>
</tr>
<tr>
<td style="text-align: center;">MAE<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">SSIM<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">LPIPS<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">FVD<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">CSI-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">CSI<span class="math inline"><sub>4</sub></span>-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">CSI<span class="math inline"><sub>16</sub></span>-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">FSS<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">RHD<span><span class="math inline">↓</span></span></td>
</tr>
<!-- Table body identical to the original manuscript -->
</tbody>
</table>

</div>

<figure id="fig:loss_sevir">
<img src="./figures/earthformer_sevir.png"" />
<figcaption>Output frames of the Earthformer models trained with MSE and FACL on SEVIR.</figcaption>
</figure>

The results of Table <a href="#tab:loss_radar" data-reference-type="ref" data-reference="tab:loss_radar">2</a> are similar to the observations in Table <a href="#tab:loss_smmnist" data-reference-type="ref" data-reference="tab:loss_smmnist">1</a>. Compared with the MSE baselines, FACL always improves the perceptual and skill scores. For sharp forecasts, the pooled CSI increases with the pooling size while for blurry forecasts, CSI shows no apparent difference based on pooling size. For some models, we observe a tiny decay in pixel-wise and structural metrics. For example, the Earthformer model trained with FACL on SEVIR has a \(6.4\%\) increase in MAE, which is believed to be a natural trade-off since pixel-wise metrics have no tolerance for spatial transformation. Despite poorer pixel-wise performance, the perceptual metrics and skill scores always improve, as further illustrated by Figure <a href="#fig:loss_sevir" data-reference-type="ref" data-reference="fig:loss_sevir">3</a> that only FACL predicts fine-grained extreme values. Regarding those generative models, although they could perform the best in deep perceptual scores like LPIPS and FVD, we still observe that they usually result in poorer skill scores. Moreover, it is noteworthy that FACL does not add any new parameters to the model. The change in the metrics solely indicates that FACL induces a shift of focus from pixel-wise accuracy to image quality and prediction skillfulness.

# Conclusion [sec:conclusion]

In this paper, we proposed the Fourier Amplitude and Correlation Loss (FACL). The two loss terms, Amplitude Loss (FAL) and Fourier Correlation Loss (FCL), encourage the model to focus on the Fourier frequencies and image structure correspondingly. Besides, we proposed a new metric, Regional Histogram Divergence (RHD), to measure the patch-wise similarity between two spatiotemporal patterns. We widely tested our methods on a synthetic dataset and three more real-life radar echo datasets, measured by metrics considering accuracy, realisticity and skillfulness. Extensive experiments reflect that our method yields sharper, more realistic and skillful forecasts with limited degradation in pixel-wise similarity.

Despite the remarkable performance of the FACL loss, our methods still have room for improvement. First, we assumed the data to be monotonic radar echo, which might not generalize well to multi-modal datasets featured in medium-range forecasts. Besides, our loss provides no regularization on temporal consistency, which may lead to the misalignment of temporal features between frames. The solution to these issues, however, will be open for future work.

<div class="ack" markdown="1">

This work has been made possible by a Research Impact Fund project (R6003-21) and an Innovation and Technology Fund project (ITS/004/21FP) funded by the Hong Kong Government.

</div>

# References [references]

<div class="thebibliography" markdown="1">

S. Ravuri, K. Lenc, M. Willson, D. Kangin, R. Lam, P. Mirowski, M. Fitzsimons, M. Athanassiadou, S. Kashem, S. Madge, R. Prudden, A. Mandhane, A. Clark, A. Brock, K. Simonyan, R. Hadsell, N. Robinson, E. Clancy, A. Arribas, and S. Mohamed, “Skilful precipitation nowcasting using deep generative models of radar,” *Nature*, vol. 597, pp. 672–677, 2021 (@ravuri2021skilful)

X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W. kin Wong, and W. chun Woo, “Convolutional LSTM Network: A machine learning approach for precipitation nowcasting,” in *NeurIPS*, 2015 (@shi2015convolutional)

Y. Wang, M. Long, J. Wang, Z. Gao, and P. S. Yu, “PredRNN: Recurrent neural networks for predictive learning using spatiotemporal lstms,” in *NeurIPS*, 2017 (@wang2017predrnn)

Y. Wang, J. Zhang, H. Zhu, M. Long, J. Wang, and P. S. Yu, “Memory in memory: A predictive neural network for learning higher-order non-stationarity from spatiotemporal dynamics,” in *CVPR*, 2019 (@wang2019memory)

Y. Wang, H. Wu, J. Zhang, Z. Gao, J. Wang, P. S. Yu, and M. Long, “PredRNN: A recurrent neural network for spatiotemporal predictive learning,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 45, no. 2, pp. 2208–2225, 2023 (@wang2022predrnn)

Y. Wang, Z. Gao, M. Long, J. Wang, and P. S. Yu, “PredRNN++: Towards a resolution of the deep-in-time dilemma in spatiotemporal predictive learning,” in *ICML*, 2018 (@wang2018predrnn)

Y. Wang, L. Jiang, M.-H. Yang, L.-J. Li, M. Long, and L. Fei-Fei, “Eidetic 3d lstm: A model for video prediction and beyond,” in *ICLR*, 2019 (@wang2019eidetic)

V. L. Guen and N. Thome, “Disentangling physical dynamics from unknown factors for unsupervised video prediction,” in *CVPR*, 2020 (@guen2020disentangling)

C. Bai, F. Sun, J. Zhang, Y. Song, and S. Chen, “Rainformer: Features extraction balanced network for radar-based precipitation nowcasting,” *IEEE Geoscience and Remote Sensing Letters*, vol. 19, pp. 1–5, 2022 (@bai2022rainformer)

Z. Gao, X. Shi, H. Wang, Y. Zhu, Y. Wang, M. Li, and D.-Y. Yeung, “Earthformer: Exploring space-time transformers for earth system forecasting,” in *NeurIPS*, 2022 (@gao2022earthformer)

G. Ayzel, T. Scheffer, and M. Heistermann, “RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting,” *Geoscientific Model Development*, vol. 13, no. 6, pp. 2631–2644, 2020 (@ayzel2020rainnet)

K. Trebing, T. Staǹczyk, and S. Mehrkanoon, “Smaat-unet: Precipitation nowcasting using a small attention-unet architecture,” *Pattern Recognition Letters*, vol. 145, pp. 178–186, 2021 (@trebing2021smaatunet)

C. Tan, S. Li, Z. Gao, W. Guan, Z. Wang, Z. Liu, L. Wu, and S. Z. Li, “OpenSTL: A comprehensive benchmark of spatio-temporal predictive learning,” in *Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2023 (@tan2023openstl)

Y. Ye, F. Gao, W. Cheng, C. Liu, and S. Zhang, “MSSTNet: A multi-scale spatiotemporal prediction neural network for precipitation nowcasting,” *Remote Sensing*, vol. 15, no. 1, 2023 (@ye2023msstnet)

C. Tan, Z. Gao, L. Wu, Y. Xu, J. Xia, S. Li, and S. Z. Li, “Temporal attention unit: Towards efficient spatiotemporal predictive learning,” in *CVPR*, pp. 18770–18782, 2023 (@tan2023temporal)

J. Leinonen, U. Hamann, D. Nerini, U. Germann, and G. Franch, “Latent diffusion models for generative precipitation nowcasting with accurate uncertainty quantification,” *CoRR*, vol. abs/2304.12891, 2023 (@leinonen2023latent)

Z. Gao, X. Shi, B. Han, H. Wang, X. Jin, D. Maddix, Y. Zhu, M. Li, and B. Wang, “Prediff: Precipitation nowcasting with latent diffusion models,” in *NeurIPS 2023 AI for Science Workshop*, 2023 (@gao2023prediff)

D. Yu, X. Li, Y. Ye, B. Zhang, C. Luo, K. Dai, R. Wang, and X. Chen, “Diffcast: A unified framework via residual diffusion for precipitation nowcasting,” in *CVPR*, 2024 (@yu2024diffcast)

L. Yu, Y. Cheng, K. Sohn, J. Lezama, H. Zhang, H. Chang, A. G. Hauptmann, M.-H. Yang, Y. Hao, I. Essa, and L. Jiang, “MAGVIT: Masked generative video transformer,” in *CVPR*, 2023 (@yu2023magvit)

I. Skorokhodov, S. Tulyakov, and M. Elhoseiny, “StyleGAN-V: A continuous video generator with the price, image quality and perks of stylegan2,” in *CVPR*, 2022 (@skorokhodov2022styleganv)

V. Voleti, A. Jolicoeur-Martineau, and C. Pal, “MCVD: Masked conditional video diffusion for prediction, generation, and interpolation,” in *NeurIPS*, 2022 (@voleti2022mvcd)

T. Höppe, A. Mehrjou, S. Bauer, D. Nielsen, and A. Dittadi, “Diffusion models for video prediction and infilling,” *Trans. Mach. Learn. Res.*, vol. 2022, 2022 (@hoppe2022diffusion)

R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in *CVPR*, 2018 (@zhang2018perceptual)

T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “FVD: A new metric for video generation,” 2019 (@unterthiner2019fvd)

N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. A. Hamprecht, Y. Bengio, and A. C. Courville, “On the spectral bias of neural networks,” in *ICML*, 2019 (@rahaman2019on)

M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng, “Fourier features let networks learn high frequency functions in low dimensional domains,” in *NeurIPS*, 2020 (@tancik2020fourier)

J. Guibas, M. Mardani, Z. Li, A. Tao, A. Anandkumar, and B. Catanzaro, “Adaptive Fourier Neural Operators: Efficient token mixers for transformers,” in *ICLR*, 2022 (@guibas2022adaptive)

K. Xu, M. Qin, F. Sun, Y. Wang, Y.-K. Chen, and F. Ren, “Learning in the frequency domain,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 1740–1749, 2020 (@xu2020learning)

A. Hertz, O. Perel, R. Giryes, O. Sorkine-Hornung, and D. Cohen-Or, “Sape: Spatially-adaptive progressive encoding for neural optimization,” *Advances in Neural Information Processing Systems*, vol. 34, pp. 8820–8832, 2021 (@hertz2021sape)

Z. Landgraf, A. S. Hornung, and R. S. Cabral, “Pins: progressive implicit networks for multi-scale neural representations,” *arXiv preprint arXiv:2202.04713*, 2022 **Abstract:** Multi-layer perceptrons (MLP) have proven to be effective scene encoders when combined with higher-dimensional projections of the input, commonly referred to as \\}textit{positional encoding}. However, scenes with a wide frequency spectrum remain a challenge: choosing high frequencies for positional encoding introduces noise in low structure areas, while low frequencies result in poor fitting of detailed regions. To address this, we propose a progressive positional encoding, exposing a hierarchical MLP structure to incremental sets of frequency encodings. Our model accurately reconstructs scenes with wide frequency bands and learns a scene representation at progressive level of detail \\}textit{without explicit per-level supervision}. The architecture is modular: each level encodes a continuous implicit representation that can be leveraged separately for its respective resolution, meaning a smaller network for coarser reconstructions. Experiments on several 2D and 3D datasets show improvements in reconstruction accuracy, representational capacity and training speed compared to baselines. (@landgraf2022pins)

V. Sitzmann, J. Martel, A. Bergman, D. Lindell, and G. Wetzstein, “Implicit neural representations with periodic activation functions,” *Advances in neural information processing systems*, vol. 33, pp. 7462–7473, 2020 (@sitzmann2020implicit)

S. D. Sims, “Frequency domain-based perceptual loss for super resolution,” 2020 (@sims2021frequency)

L. Jiang, B. Dai, W. Wu, and C. C. Loy, “Focal frequency loss for image reconstruction and synthesis,” in *ICCV*, 2021 (@jiang2021focal)

D. Fuoli, L. V. Gool, and R. Timofte, “Fourier space losses for efficient perceptual image super-resolution,” in *ICCV*, 2021 (@fuoli2021fourier)

S. Koho, G. Tortarolo, M. Castello, T. Deguchi, A. Diaspro, and G. Vicidomini, “Fourier ring correlation simplifies image restoration in fluorescence microscopy,” *Nature communications*, vol. 10, no. 1, p. 3103, 2019 (@koho2019fourier)

N. Banterle, K. H. Bui, E. A. Lemke, and M. Beck, “Fourier ring correlation as a resolution criterion for super-resolution microscopy,” *Journal of structural biology*, vol. 183, no. 3, pp. 363–367, 2013 (@banterle2013fourier)

M. Van Heel and M. Schatz, “Fourier shell correlation threshold criteria,” *Journal of structural biology*, vol. 151, no. 3, pp. 250–262, 2005 (@van2005fourier)

J. Kaczmar-Michalska, N. Hajizadeh, A. Rzepiela, and S. Nørrelykke, “Image quality measurements and denoising using fourier ring correlations,” *arXiv preprint arXiv:2201.03992*, 2022 **Abstract:** Image quality is a nebulous concept with different meanings to different people. To quantify image quality a relative difference is typically calculated between a corrupted image and a ground truth image. But what metric should we use for measuring this difference? Ideally, the metric should perform well for both natural and scientific images. The structural similarity index (SSIM) is a good measure for how humans perceive image similarities, but is not sensitive to differences that are scientifically meaningful in microscopy. In electron and super-resolution microscopy, the Fourier Ring Correlation (FRC) is often used, but is little known outside of these fields. Here we show that the FRC can equally well be applied to natural images, e.g. the Google Open Images dataset. We then define a loss function based on the FRC, show that it is analytically differentiable, and use it to train a U-net for denoising of images. This FRC-based loss function allows the network to train faster and achieve similar or better results than when using L1- or L2- based losses. We also investigate the properties and limitations of neural network denoising with the FRC analysis. (@kaczmar2022image)

S. Culley, D. Albrecht, C. Jacobs, P. M. Pereira, C. Leterrier, J. Mercer, and R. Henriques, “Quantitative mapping and minimization of super-resolution optical imaging artifacts,” *Nature methods*, vol. 15, no. 4, pp. 263–266, 2018 (@culley2018quantitative)

A. Berberich, A. Kurz, S. Reinhard, T. J. Paul, P. R. Burd, M. Sauer, and P. Kollmannsberger, “Fourier ring correlation and anisotropic kernel density estimation improve deep learning based smlm reconstruction of microtubules,” *Frontiers in Bioinformatics*, p. 55, 2021 (@berberich2021fourier)

M. Veillette, S. Samsi, and C. J. Mattioli, “SEVIR: A storm event imagery dataset for deep learning applications in radar and satellite meteorology,” in *NeurIPS*, 2020 (@veillette2020sevir)

G. Larvor, L. Berthomie, V. Chabot, B. L. Pape, B. Pradel, and L. Perez, “Meteonet: An open reference weather dataset for ai by météo-france,” tech. rep., Meteo France, 2020 (@larvor2020meteonet)

X. Shi, Z. Gao, L. Lausen, H. Wang, D.-Y. Yeung, W. kin Wong, and W. chun Woo, “Deep learning for precipitation nowcasting: A benchmark and a new model,” in *NeurIPS*, 2017 (@shi2017trajgru)

Z. Gao, C. Tan, L. Wu, and S. Z. Li, “SimVP: Simpler yet better video prediction,” in *CVPR*, 2022 (@gao2022simvp)

Z. Chang, X. Zhang, S. Wang, S. Ma, and W. Gao, “Strpm: A spatiotemporal residual predictive model for high-resolution video prediction,” in *CVPR*, 2022 (@chang2022strpm)

E. Denton and R. Fergus, “Stochastic video generation with a learned prior,” in *ICML*, 2018 (@denton2018stochastic)

M. Seo, H. Lee, D. Kim, and J. Seo, “Implicit stacked autoregressive model for video prediction,” *CoRR*, vol. abs/2303.07849, 2023 (@seo2023implicit)

L. Chen, Y. Cao, L. Ma, and J. Zhang, “A deep learning-based methodology for precipitation nowcasting with radar,” *Earth and Space Science*, vol. 7(2), 2020 (@chen2020a)

Q.-K. Tran and S.-k. Song, “Computer vision in precipitation nowcasting: Applying image quality assessment metrics for training deep neural networks,” *Atmosphere*, vol. 10(5), 2019 (@Tran2019computer)

</div>

<span id="sec:appendix" label="sec:appendix"></span>

# Details of the Datasets [app:datasets]

#### Stochastic Moving-MNIST.

Based on the same method generating vanilla Moving-MNIST, we further update the velocity of the digits at time \\(t\\): \\[\begin{cases} 
u_t \leftarrow u_0 + \epsilon \\ 
v_t \leftarrow v_0 + \epsilon \\
\end{cases}
\text{, where } \epsilon\sim\mathcal{N}(0, 1)\\] Each unit corresponds to a pixel in the \\(64\times64\\) image. Note that the expected trajectory is unchanged under this modification. However, the biased trajectory is exposed to the model as a stochastic factor influencing model training. Due to such behavior, models trained with the MSE loss are expected to exhibit a motion blur pattern along the direction of motion.

#### N-body-MNIST.

To forge the chaotic nature of the Earth system, N-body-MNIST `\cite{gao2022earthformer}`{=latex} was proposed to study the effectiveness of deep learning models. Rather than linear translation as in the conventional Moving-MNIST, digits in N-body-MNIST follow the N-body motion pattern, exerting an attractive force between digits and causing each other to circulate. Following the default setting of the original paper, the frame size is set to be \\(64\times 64\\) with \\(N = 3\\). We use the same training, validation and test sets which contain 20000, 1000 and 1000 sequences respectively provided by the official repository.

#### SEVIR.

The SEVIR dataset `\cite{veillette2020sevir}`{=latex} is a spatiotemporally aligned dataset containing over 10,000 weather events in a \\(384\text{km} \times 384\text{km}\\) region in the US spanning a period of four hours from 2017 to 2019. Among the five channels provided, we extract the NEXRAD Vertically Integrated Liquid (VIL) data product for precipitation nowcasting. Following previous works `\cite{gao2022earthformer,seo2023implicit}`{=latex}, we predict the future VIL up to 60 minutes (12 frames) from 65 minutes of input frames (13 frames). We sample the test set from June 2019 to December 2019, leaving the remaining as the training set.

#### MeteoNet.

MeteoNet `\cite{larvor2020meteonet}`{=latex} is an open meteorological dataset containing satellite and radar imagery in France. The data covers two geographic areas of \\(550\text{km} \times 550\text{km}\\) on the Mediterranean and Brittany coasts, respectively, from 2016 to 2018. The time interval between consecutive frames is 5 minutes. Since there are missing values labeled as \\(-1\\) in the raw rectangular data in shape \\((565, 784)\\), we preprocess it by filling \\(0\\) to the missing values, followed by a linear scaling of pixel values to the range \\([0,1]\\). After that, we downsample the images to \\((256, 256)\\) using bilinear interpolation. The task is to predict the next sequences of radar echoes in an hour (12 frames) from the given 20-minute radar echoes (4 frames). The data in 2016 and 2017 are sampled as the training set and those in 2018 are sampled as the test set.

#### HKO-7.

The HKO-7 dataset `\cite{shi2017trajgru}`{=latex} is a collection of radar reflectivity image data from 2009 to 2015 based on the radar product, namely, the Constant Altitude Plan Position Indicator (CAPPI) at an altitude of 2 kilometers with a radius of \\(256\\) kilometers centered at Hong Kong. No prior data cleansing was applied to the HKO-7 dataset so it may consist of noises commonly found in radar imagery due to sea or ground clutters and anomalous propagation, and blind sectors due to blockage of microwave signals. Moreover, the sub-tropical climate of Hong Kong, mesoscale weather development which is caused by the land-sea contrast and complex terrain over the territory and the adjacent coastal areas lead to changeable weather and limited predictability of severe convective precipitation beyond the next couple of hours. Overall, the HKO-7 dataset is known to be much more difficult to model precisely, which could better highlight the effectiveness of our methods. We predict the next 2-hour radar echoes (20 frames) from that of the past 30 minutes (5 frames). The data from 2009 to 2014 are used as the training set and those in 2015 are used as the test set.

# Triviality of \\(L_2\\) Loss in the Fourier Space [app:proof_l2_F]

In this section, we prove that the \\(L_{2}\\) distance between ground truth and prediction in both the Fourier domain and image domain are equivalent from both the forward and gradient aspects.

## Showing that \\(L_2(\hat{F}, F) =  L_2(\hat{X}, X)\\)

Parseval’s Theorem (or the general one: Plancherel Theorem) describes the unitarity of the Fourier transform under proper normalization. Without normalization, we have the following relationship: \\[\label{eq:parseval_thm}
\sum_{k=0}^{N-1} |X_k|^2 = \frac{1}{N}\sum_{k=0}^{N-1} |F_k|^2 .\\] This refers to the 1D case where \\(F\\) is the Fourier transformed output of \\(X\\), and \\(N\\) is the vector length of both \\(F\\) and \\(X\\). In the 2D case with \\(F\\) orthonormalized in Eq. <a href="#eqn:dfft" data-reference-type="eqref" data-reference="eqn:dfft">[eqn:dfft]</a>, we have instead \\[\label{eq:parseval_thm_ortho}
\sum |X|^2 = \sum |F|^2 ,\\] where \\(\sum\\) is used as a shorthand summing every element in the following 2D matrix. When we apply the \\(L_2\\) loss to two orthonormalized Fourier matrices, we obtain \\[\label{eqn:naive=l2}
    L_2(F, \hat{F}) = \frac{1}{N}\sum|F - \hat{F}|^2 = \frac{1}{N}\sum(X - \hat{X})^2 = L_2(X, \hat{X})\\] due to the linearity of the Fourier transform and the use of Parseval’s Theorem.

## Showing that \\(\frac{\partial L_2(F, \hat{F})}{\partial \hat{X}_{kl}} =  \frac{\partial L_2(X,\hat{X})}{\partial \hat{X}_{kl}}\\)

From Eq. <a href="#eqn:dfft" data-reference-type="eqref" data-reference="eqn:dfft">[eqn:dfft]</a>, we continue and derive the gradient of the Fourier transform output \\(F\\) with respect to image \\(X\\):

\\[\label{eqn:dfdi}
    \frac{\partial F_{pq}}{\partial X_{kl}} = \frac{1}{\sqrt{MN}}e^{-i2\pi(\frac{kp}{M} + \frac{lq}{N})}\\]

For every complex vector, the multiplication of itself and its conjugate is equal to the square of its amplitude, that is \\(F F^{*} = |F|^{2}\\). Thus, \\[\begin{aligned}
    \frac{\partial }{\partial \hat{X}_{kl}}|F_{pq}-F_{pq}^{*}|^{2} & = \frac{\partial}{\partial\hat{X}_{kl}} [|F_{pq}|^{2} - F^{*}_{pq}\hat{F}_{pq} - F_{pq}\hat{F}^{*}_{pq} + \hat{F}_{pq}\hat{F}^{*}_{pq}] \\
    & = \frac{1}{\sqrt{MN}}[-F^{*}_{pq}e^{-i2\pi(\frac{kp}{M} + \frac{lq}{N})} - F_{pq}e^{i2\pi(\frac{kp}{M} + \frac{lq}{N})} + \hat{F}^{*}_{pq}e^{-i2\pi(\frac{kp}{M} + \frac{lq}{N})} + \hat{F}_{pq}e^{i2\pi(\frac{kp}{M} + \frac{lq}{N})}] .
\end{aligned}\\]

With the inverse Fourier transform defined in Eq. <a href="#eqn:dfft" data-reference-type="eqref" data-reference="eqn:dfft">[eqn:dfft]</a> and the assumption that \\(X\\) is always real, we can obtain: \\[\begin{aligned}
    \frac{\partial L_2(F, \hat{F})}{\partial \hat{X}_{kl}} & = \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1} \frac{\partial}{\partial \hat{X}_{kl}} |F_{pq}-F_{pq}^{*}|^{2} \\
   & = -\frac{1}{(\sqrt{MN})^{3}} \sum_{p=0}^{M-1}\sum_{q=0}^{N-1} [F^{*}_{pq}e^{-i2\pi(\frac{kp}{M} + \frac{lq}{N})} + F_{pq}e^{i2\pi(\frac{kp}{M} + \frac{lq}{N})} \\
   & \ \ \ \ \ - \hat{F}^{*}_{pq}e^{-i2\pi(\frac{kp}{M} + \frac{lq}{N})} - \hat{F}_{pq}e^{i2\pi(\frac{kp}{M} + \frac{lq}{N})}]\\
 & = -\frac{1}{MN} [X_{kl}^{*} + X_{kl} - \hat{X}_{kl}^{*} - \hat{X}_{kl}] \\
 & = -\frac{2}{MN}[ X_{kl} - \hat{X}_{kl}] \\
\frac{\partial L_2(F, \hat{F})}{\partial \hat{X}_{kl}} & = \frac{\partial L_2(X,\hat{X})}{\partial \hat{X}_{kl}}
\end{aligned}\\]

Consequently, the gradients of \\(L_{2}(F, \hat{F})\\) and \\(L_{2}(X, \hat{X})\\) with respect to \\(\hat{X}_{kl}\\) are equivalent. This result indicates that implementing \\(L_{2}\\) in the Fourier domain without any weighting as a loss function does not affect the model performance.

# FAL in the Gradient Aspect [app:problem_when_amp_alone]

In Section <a href="#sec:FAL" data-reference-type="ref" data-reference="sec:FAL">3.2</a>, we claim that FAL works as a regularizer to maintain the frequency amplitude in the Fourier domain, but not the full loss function. In this section, we study the reason behind this statement from the perspective of gradient feedback. Before that, we start with deriving the derivative of \\(|F|\\) with respect to \\(X_{kl}\\): \\[\begin{aligned}
    |F_{pq}|^{2} & = F_{pq}F_{pq}^{*}\\
    2 \, |F_{pq}| \, \frac{\partial |F_{pq}|}{\partial X_{kl}} & = F_{pq}\frac{\partial F_{pq}^{*}}{\partial X_{kl}} + \frac{\partial F_{pq}}{\partial X_{kl}}F_{pq}^{*} \\
    \frac{\partial |F_{pq}|}{\partial X_{kl}} & = \frac{1}{2\sqrt{MN}}[e^{i(\theta_{pq} + \alpha_{pq, kl})} + e^{-i(\theta_{pq} + \alpha_{pq, kl})} ] \\
    & = \frac{1}{\sqrt{MN}} \cos (\theta_{pq} +\alpha_{pq, kl} ) ,
\end{aligned}\\] where \\(F_{pq} = |F_{pq}|e^{i\theta_{pq}}\\) and \\(\alpha_{pq, kl} = 2\pi(\frac{kp}{M} + \frac{lq}{N})\\).  
From Eq. <a href="#eqn:fft-amp" data-reference-type="eqref" data-reference="eqn:fft-amp">[eqn:fft-amp]</a>, we further derive its derivative with respect to \\(\hat{X}_{kl}\\) and get: \\[\label{eqn:der-fft-amp}
    
    \frac{\partial }{\partial \hat{X}_{kl}}\text{FAL} (X, \hat{X})= -\frac{2}{(\sqrt{MN})^{3}} \sum_{p=0}^{M-1}\sum_{q=0}^{N-1}(|F_{pq}| - |\hat{F}_{pq}|)\cos(\hat{\theta}_{pq} + \alpha_{pq,kl}) ,\\] where \\(|F_{pq}|\\) and \\(|\hat{F}_{pq}|\\) are the Fourier amplitudes of the ground truth and prediction corresponding to the frequency \\((p, q)\\) while \\(\hat{\theta}_{pq}\\) is the Fourier phase of prediction corresponding to the frequency \\((p, q)\\).

From Eq. <a href="#eqn:der-fft-amp" data-reference-type="eqref" data-reference="eqn:der-fft-amp">[eqn:der-fft-amp]</a>, we note that \\(\theta_{pq}\\), which corresponds to the position or the image structure of the ground truth frequencies (object) in the image space, is missing. In other words, the model never gets its parameters updated based on the phase of the ground truth. As a result, FAL only encourages the model to predict what has the same amplitude distribution in the Fourier domain, without considering the image structure. This theoretically shows the infeasibility of reconstructing the ground truth based on FAL alone, motivating us to adopt a second loss term to maintain the image structure.

To effectively make use of FAL as a regularizer, we have to ensure that the model has sufficient time to learn the general image structure (the low-frequency pattern) such that FAL could provide better guidance on the remaining frequency components by exposing it more to FCL in the beginning of training process. In contrast, if the model cannot learn the low-frequency components before FAL becomes the dominant learning objective, the model will likely converge to a trivial solution. This claim is also empirically verified by our ablation study experiments where using FAL alone results in poor performance as shown in Appendix. <a href="#app:abla_existence" data-reference-type="ref" data-reference="app:abla_existence">11</a>.

# Further Analysis of FAL [app:fal_study]

This section discerns FAL and a naive \\(L_2\\) loss in the Fourier space. By definition, the major difference between the two is whether the complex pattern or the amplitude is used in the comparison. The FAL loss term can be expanded as follows: \\[\begin{aligned}
    \text{FAL}(X, \hat{X}) 
    & = \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}(|F|_{pq}- |\hat{F}|_{pq})^2 \nonumber \\
    & = \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}(|F|_{pq}^2 + |F|_{pq}^2 - 2|F|_{pq}|\hat{F}|_{pq}) \nonumber \\
    & = \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}(X_{pq}^2 + \hat{X}_{pq}^2) - \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}2|F|_{pq}|\hat{F}|_{pq}\nonumber\\
    & = \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}(X_{pq}^2+\hat{X}_{pq}^2-2X_{pq}\hat{X}_{pq}) + \nonumber\\
    & \ \ \ \ \ \ \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}2\hat{X}_{pq}X_{pq}- \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}2|F|_{pq}|\hat{F}|_{pq}\nonumber\\
    & = L_2(X, \hat{X}) + \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}2X_{pq}\hat{X}_{pq}- \frac{1}{MN}\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}2|F|_{pq}|\hat{F}|_{pq}. \nonumber
\end{aligned}\\]

Apart from the \\(L_2\\) component which is equivalent to Eq. <a href="#eqn:naive=l2" data-reference-type="eqref" data-reference="eqn:naive=l2">[eqn:naive=l2]</a>, we also obtain two extra terms, shorthanded as \\(\sum{2X\hat{X}}\\) and \\(\sum{2|F||\hat{F}|}\\). To study the empirical effect of the two factors on the high-frequency components, we performed a simple experiment: we sampled an image from the Moving-MNIST dataset and performed two modifications over time: (1) applying Gaussian blur with a standard deviation of \\(\sigma\\) to the sample, and (2) translating the sample along the direction \\((t, t)\\). Then we observed the trend of increment of the two factors, as shown in Figure <a href="#fig:2xx_and_2ff" data-reference-type="ref" data-reference="fig:2xx_and_2ff">4</a>.

<figure id="fig:2xx_and_2ff">
<img src="./figures/FAL_over_t.png"" />
<figcaption>FAL loss terms over different values of (left) <span class="math inline"><em>σ</em></span> in blurring and (right) <span class="math inline"><em>t</em></span> in translation. In (right), <span class="math inline"><em>L</em><sub>2</sub></span> (the blue line) and <span class="math inline">|∑2<em>X</em><em>X̂</em> − ∑2|<em>F</em>||<em>F̂</em>||</span> (the green line) mostly overlap.</figcaption>
</figure>

The figure reflects a couple of characteristics of the FAL loss term. First, in the case of blurring, it behaves similarly to the standard \\(L_2\\) loss with a tiny difference when \\(\sigma\\) gets very large. Intriguingly, FAL does not exhibit a different degree of sensitivity to different frequencies. However, for translation, the absolute difference \\(\sum{2X\hat{X}} - \sum{2|F||\hat{F}|}\\) is almost equivalent to and thus cancels out the \\(L_2\\) loss, causing the final FAL loss term to become very small. It is also noteworthy that when the two samples \\(X\\) and \\(\hat{X}\\) are identical, both \\(\sum{2X\hat{X}}\\) and \\(\sum{2|F||\hat{F}|}\\) are zero and thus FAL is also zero. From the observations above, FAL is invariant to global translations and robust to one-directional local translation compared to \\(L_2\\). Because of such invariance, FAL is more robust against the spectral bias and could better fine-tune the frequencies of the output. Utilizing this behavior, the model could focus on the reconstruction of a clear signal without suffering from the influence of the randomness in translations. Moreover, it also shows that an arbitrary scaling between \\(L_2\\) and the factor \\(\sum{2X\hat{X}} - \sum{2|F||\hat{F}|}\\) could not result in a desired effect, since this causes the two terms to no longer overlap in the plot over \\(t\\), leading to an increase in sensitivity to translation.

# Further Analysis of FCL [app:fsc_grad]

To understand how FCL affects the model during training, we derive its derivative with respect to \\(\hat{X}_{kl}\\) and obtain an interesting result with the aid of Plancherel’s theorem: \\[\label{eqn:deFSCdI}
    \frac{\partial}{\partial \hat{X}_{kl}} \text{FCL}(X, \hat{X}) = -\frac{1}{\sqrt{\sum |X|^{2}\sum |\hat{X}|^{2}}}[X_{kl} - \frac{\sum X\hat{X}}{\sum |\hat{X}|^{2}}\hat{X}_{kl}]\\]  
From Eq. <a href="#eqn:deFSCdI" data-reference-type="eqref" data-reference="eqn:deFSCdI">[eqn:deFSCdI]</a>, the ratio \\(X_{kl}\\) to \\(\hat{X}_{kl}\\) highly depends on the summation over the image domain, providing global information to \\(\hat{X}_{kl}\\), unlike the element-wise or pixel-wise relationship between \\(\hat{X}\\) and \\(X\\) in the conventional MSE loss, \\(L_{2}(\hat{X}, X)\\).

To have an intuitive understanding of the conclusion above, we design a thought experiment to understand how FCL is different from \\(L_{2}(\hat{X}, X)\\) here. Consider the case where the prediction has the same image structure as the ground truth but with different intensity, for instance, \\(\hat{X} = \beta X\\), where \\(\beta\\) is an arbitrary non-zero constant.

Substituting \\(\hat{X}\\) into Eq. <a href="#eqn:deFSCdI" data-reference-type="eqref" data-reference="eqn:deFSCdI">[eqn:deFSCdI]</a>, we have \\[\frac{\partial}{\partial\hat{X}_{kl}} \text{FCL}(X, \hat{X}) = 0 .\\] Meanwhile, it is straightforward that \\[\frac{\partial }{\partial \hat{X}_{kl}} L_{2}(\hat{X}, X) \propto -(1-\beta) .\\] With the above discrepancy, the behavior of FCL is substantially different from MSE in regard to overall brightness. That is, MSE is affected by both the image structure and the overall brightness but FCL is affected by the image structure only. Therefore, with FCL alone, we lose the pixel intensity. While applying the sigmoid function is one method to alleviate the drawback of the missing information, incorporating FAL which focuses on the intensity in particular could be viewed as a parallel complement to further stabilize the models.

# Ablation Study on FAL and FCL [app:abla_existence]

In previous sections, we showed that the Fourier phase of the ground truth, \\(\theta\\), is missing in the gradient of FAL. Hence, we claim that FAL alone is insufficient to be a reconstruction loss. Similarly, in the thought experiment conducted in Appendix <a href="#app:fsc_grad" data-reference-type="ref" data-reference="app:fsc_grad">10</a>, we conclude that FCL does not consider the image intensity and sharpness. As a result, the two loss terms FAL and FCL have to be used together as a full reconstruction loss. Here, we report the empirical results of the ablation study for FAL and FCL in Table <a href="#tab:loss_exist_abla" data-reference-type="ref" data-reference="tab:loss_exist_abla">[tab:loss_exist_abla]</a>.

<div class="tabularx" markdown="1">

0.9cl\*7Y &  
& MAE\\(\downarrow\\)& MSE\\(\downarrow\\)& SSIM\\(\uparrow\\)& LPIPS\\(\downarrow\\)& FVD\\(\downarrow\\)& FSS\\(\uparrow\\)& RHD\\(\downarrow\\)  
FAL Only & 430.7 & 302.7 & 0.2871 & 0.5854 & 1320.1 & 0.0019 & 1.0538  
FCL Only & 184.9 & **80.4 & 0.7318 & 0.2102 & 391.8 & 0.6451 & 1.0841  
FACL & **180.1 & 118.1 & **0.7463 & **0.1092 & **82.3 & **0.8172 & **0.3391  **************

</div>

In Table <a href="#tab:loss_exist_abla" data-reference-type="ref" data-reference="tab:loss_exist_abla">[tab:loss_exist_abla]</a>, the model trained with FAL does not produce meaningful output, as reflected by the abnormal values of the metrics and the faulty predictions in Figure <a href="#fig:exist_abla" data-reference-type="ref" data-reference="fig:exist_abla">5</a>. This agrees with our statement that models trained with FAL alone cannot converge to proper local minima. Meanwhile, the model trained with FCL only exhibits behaviors similar to MSE as shown in Figure <a href="#fig:exist_abla" data-reference-type="ref" data-reference="fig:exist_abla">5</a>. To sum up, either using FAL or FCL alone does not empirically produce the desired effect. However, combining the two loss terms together achieves a huge improvement to most of the metrics, which agrees with our theoretical analysis.

<figure id="fig:exist_abla">
<img src="./figures/facl_exist.png"" />
<figcaption>Qualitative performance of different losses for ConvLSTM on Stochastic Moving-MNIST.</figcaption>
</figure>

Next, we study the effect of \\(\alpha\\), which controls the length of the fine-tuning process with FAL. The fine-tuning process encourages the models to predict sharper and brighter predictions. At the same time, this can also lead to overfitting of noises in the high-frequency components. From Table <a href="#tab:loss_alpha_abla_smmnist" data-reference-type="ref" data-reference="tab:loss_alpha_abla_smmnist">[tab:loss_alpha_abla_smmnist]</a> and <a href="#tab:loss_alpha_abla_sevir" data-reference-type="ref" data-reference="tab:loss_alpha_abla_sevir">3</a>, the sharpness-aware metrics such as LPIPS, FVD and RHD can always be improved by setting \\(\alpha\\) to non-zero. However, there is no apparent improvement or decay after \\(\alpha \geq 0.2\\). Therefore, we believe setting \\(\alpha = 0.1\\) or \\(0.2\\) as a default value can strike a good balance between sharpness and pixel-wise performance.

<div class="tabularx" markdown="1">

0.9c\*7Y &  
& MAE\\(\downarrow\\)& MSE\\(\downarrow\\)& SSIM\\(\uparrow\\)& LPIPS\\(\downarrow\\)& FVD\\(\downarrow\\)& FSS\\(\uparrow\\)& RHD\\(\downarrow\\)  
& 162.7 & **93.76 & 0.7770 & 0.0997 & 92.53 & 0.8533 & 0.3584  
0.1 & 162.8 & 104.3 & 0.7759 & 0.0817 & 62.16 & 0.8534 & 0.2982  
0.2 & **162.0 & 105.0 & **0.7822 & **0.0806 & **54.63 & **0.8552 & **0.2976  
0.3 & 162.1 & 105.2 & 0.7812 & 0.0869 & 63.29 & 0.8549 & 0.3000  
0.4 & 162.6 & 105.3 & 0.7806 & 0.0845 & 58.95 & 0.8542 & 0.3023  **************

</div>

<div id="tab:loss_alpha_abla_sevir" markdown="1">

<table>
<caption>Effect of different <span class="math inline"><em>α</em></span> on the performance of ConvLSTM trained with FACL on SEVIR, where MAE is in the scale of <span class="math inline">10<sup>−3</sup></span>.</caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;"><span class="math inline"><em>α</em></span></td>
<td colspan="9" style="text-align: center;">Metrics</td>
</tr>
<tr>
<td style="text-align: center;">MAE<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">SSIM<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">LPIPS<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">FVD<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">CSI-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">CSI<span class="math inline"><sub>4</sub></span>-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">CSI<span class="math inline"><sub>16</sub></span>-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">FSS<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">RHD<span><span class="math inline">↓</span></span></td>
</tr>
<tr>
<td style="text-align: center;">0.0</td>
<td style="text-align: center;"><strong><span>26.15</span></strong></td>
<td style="text-align: center;"><strong><span>0.7814</span></strong></td>
<td style="text-align: center;">0.3502</td>
<td style="text-align: center;">391.37</td>
<td style="text-align: center;"><strong><span>0.4195</span></strong></td>
<td style="text-align: center;"><strong><span>0.4339</span></strong></td>
<td style="text-align: center;">0.4710</td>
<td style="text-align: center;"><strong><span>0.5727</span></strong></td>
<td style="text-align: center;">1.3924</td>
</tr>
<tr>
<td style="text-align: center;">0.1</td>
<td style="text-align: center;">27.60</td>
<td style="text-align: center;">0.7624</td>
<td style="text-align: center;">0.3508</td>
<td style="text-align: center;">289.49</td>
<td style="text-align: center;">0.3984</td>
<td style="text-align: center;">0.4295</td>
<td style="text-align: center;">0.5073</td>
<td style="text-align: center;">0.5640</td>
<td style="text-align: center;">1.2087</td>
</tr>
<tr>
<td style="text-align: center;">0.2</td>
<td style="text-align: center;">27.92</td>
<td style="text-align: center;">0.7589</td>
<td style="text-align: center;">0.3415</td>
<td style="text-align: center;">254.73</td>
<td style="text-align: center;">0.3958</td>
<td style="text-align: center;">0.4331</td>
<td style="text-align: center;"><strong><span>0.5247</span></strong></td>
<td style="text-align: center;">0.5447</td>
<td style="text-align: center;"><strong><span>1.1643</span></strong></td>
</tr>
<tr>
<td style="text-align: center;">0.3</td>
<td style="text-align: center;">27.80</td>
<td style="text-align: center;">0.7587</td>
<td style="text-align: center;"><strong><span>0.3312</span></strong></td>
<td style="text-align: center;">258.24</td>
<td style="text-align: center;">0.3953</td>
<td style="text-align: center;">0.4288</td>
<td style="text-align: center;">0.5242</td>
<td style="text-align: center;">0.5453</td>
<td style="text-align: center;">1.1710</td>
</tr>
<tr>
<td style="text-align: center;">0.4</td>
<td style="text-align: center;">28.15</td>
<td style="text-align: center;">0.7574</td>
<td style="text-align: center;">0.3384</td>
<td style="text-align: center;"><strong><span>232.50</span></strong></td>
<td style="text-align: center;">0.3930</td>
<td style="text-align: center;">0.4264</td>
<td style="text-align: center;">0.5190</td>
<td style="text-align: center;">0.5262</td>
<td style="text-align: center;">1.1960</td>
</tr>
<tr>
<td style="text-align: center;">0.5</td>
<td style="text-align: center;">30.45</td>
<td style="text-align: center;">0.7402</td>
<td style="text-align: center;">0.3492</td>
<td style="text-align: center;">281.82</td>
<td style="text-align: center;">0.3633</td>
<td style="text-align: center;">0.3915</td>
<td style="text-align: center;">0.4838</td>
<td style="text-align: center;">0.4813</td>
<td style="text-align: center;">1.3445</td>
</tr>
</tbody>
</table>

</div>

# Running Time of FACL [app:running_time]

In the previous sections, we showed that the method is both effective and generic of models. In this section, we discuss the running time of FACL. Theoretically, FACL utilizes DFT, which has time complexity \\(O(n^2)\\) in the 1D case with vector length \\(n\\). By leveraging the 2D Fast Fourier Transform (FFT), we could improve the time complexity to \\(O(MN(\log(M)+\log(N)))\\) for each pair of frames, where \\(M\\) and \\(N\\) correspond to the height and width of the samples. With the aid of deep learning frameworks such as PyTorch, such operations can be run in parallel and supported by GPU. Therefore, the computational load for FACL is light compared to the deep network architectures. During inference, the only difference between models trained with MSE and FACL is that the FACL one consists of a sigmoid layer at the end. Running in parallel, again, this operation is negligible.

To test the actual speed of FACL, we report the running time during model training and model inference for the experimented models in Table <a href="#tab:running_time" data-reference-type="ref" data-reference="tab:running_time">4</a>. For the training stage, we report the mean of the training time for the first 5 epochs. The table shows that the running time of FACL is negligible compared to the MSE counterpart, with the model selected being the most dominant factor for the running time. With the inference time reported, we could also notice the advantage of staying with video prediction models over generative models. The diffusion models are much slower than traditional predictive models. Our slowest model (FACL on PredRNN) is still 50X faster than MCVD. Note that such a difference scales with the image size, causing some of the generative models infeasible to apply to large-size radar imagery.

<div id="tab:running_time" markdown="1">

| Model       | Loss | Training Time (s) | Inference Time (s) | Average FPS |
|:------------|:----:|:-----------------:|:------------------:|:-----------:|
| ConvLSTM    | MSE  |    \\(97.8\\)     |    \\(0.045\\)     |     220     |
| ConvLSTM    | FACL |    \\(97.8\\)     |    \\(0.043\\)     |     232     |
| PredRNN     | MSE  |    \\(132.6\\)    |    \\(0.169\\)     |     59      |
| PredRNN     | FACL |    \\(134.2\\)    |    \\(0.180\\)     |     55      |
| SimVP       | MSE  |    \\(36.4\\)     |    \\(0.022\\)     |     635     |
| SimVP       | FACL |    \\(29.8\\)     |    \\(0.017\\)     |     598     |
| Earthformer | MSE  |    \\(724.5\\)    |    \\(0.101\\)     |     99      |
| Earthformer | FACL |    \\(731.3\\)    |    \\(0.110\\)     |     91      |
| LDCast      |  \-  |        \-         |    \\(7.783\\)     |     1.3     |
| MCVD        |  \-  |        \-         |    \\(81.873\\)    |    0.12     |

Comparison of the quantitative performance of different losses for models trained on Stochastic Moving-MNIST datasets. We report the average time (in seconds) of 5 training epochs and 100 inference steps on a single Nvidia GeForce RTX3090.

</div>

# Evaluation on N-Body-MNIST [app:nbody]

Apart from the proposed Stochastic Moving-MNIST, previous works proposed N-Body-MNIST `\cite{gao2022earthformer}`{=latex}, a dataset that utilizes multiple transformations to simulate the chaotic nature of the atmospheric conditions. We present the results in this section.

<div id="tab:loss_nbody" markdown="1">

<table>
<caption>Comparison of the quantitative performance of different losses for models trained on the Stochastic Moving-MNIST. The better score between MSE and FACL is highlighted in bold. </caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;">Model</td>
<td rowspan="2" style="text-align: center;">Loss</td>
<td colspan="2" style="text-align: center;">Pixel-wise/Structural</td>
<td colspan="2" style="text-align: center;">Perceptual</td>
<td style="text-align: center;">Skill</td>
<td style="text-align: center;">Proposed</td>
</tr>
<tr>
<td style="text-align: center;">MAE<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">SSIM<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">LPIPS<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">FVD<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">FSS<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">RHD<span><span class="math inline">↓</span></span></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">ConvLSTM</td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">0.8946</td>
<td style="text-align: center;">0.1264</td>
<td style="text-align: center;">178.57</td>
<td style="text-align: center;">0.7601</td>
<td style="text-align: center;">0.2301</td>
</tr>
<tr>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;"><strong><span>43.1</span></strong></td>
<td style="text-align: center;"><strong><span>0.9385</span></strong></td>
<td style="text-align: center;"><strong><span>0.0533</span></strong></td>
<td style="text-align: center;"><strong><span>80.83</span></strong></td>
<td style="text-align: center;"><strong><span>0.9198</span></strong></td>
<td style="text-align: center;"><strong><span>0.1586</span></strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">SimVP</td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;"><strong><span>55.2</span></strong></td>
<td style="text-align: center;"><strong><span>0.9130</span></strong></td>
<td style="text-align: center;"><strong><span>0.0612</span></strong></td>
<td style="text-align: center;"><strong><span>77.95</span></strong></td>
<td style="text-align: center;"><strong><span>0.9093</span></strong></td>
<td style="text-align: center;"><strong><span>0.1467</span></strong></td>
</tr>
<tr>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;">59.3</td>
<td style="text-align: center;">0.8960</td>
<td style="text-align: center;">0.0730</td>
<td style="text-align: center;">102.71</td>
<td style="text-align: center;">0.8978</td>
<td style="text-align: center;">0.1526</td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">Earthformer</td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;"><strong><span>18.6</span></strong></td>
<td style="text-align: center;"><strong><span>0.9834</span></strong></td>
<td style="text-align: center;"><strong><span>0.0091</span></strong></td>
<td style="text-align: center;"><strong><span>13.46</span></strong></td>
<td style="text-align: center;"><strong><span>0.9835</span></strong></td>
<td style="text-align: center;"><strong><span>0.0971</span></strong></td>
</tr>
<tr>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;">19.3</td>
<td style="text-align: center;">0.9826</td>
<td style="text-align: center;">0.0092</td>
<td style="text-align: center;">13.68</td>
<td style="text-align: center;">0.9829</td>
<td style="text-align: center;">0.0985</td>
</tr>
</tbody>
</table>

</div>

<figure id="fig:nbody_convlstm">
<img src="./figures/nbody_convlstm.png"" />
<figcaption>Output frames of ConvLSTM trained with MSE and FACL on N-body-MNIST.</figcaption>
</figure>

Compared with Stochastic Moving-MNIST, N-Body-MNIST additionally introduces inter-digit influence on top of the original trajectory. However, from the table and visualizations, we observe that the models in general can perform much better with N-Body-MNIST than that with Stochastic Moving-MNIST. This shows our assumption that traditional video prediction models can capture complicated deterministic motion but cannot handle random motion well. Hence, despite having N-body-MNIST as a benchmark dataset, the proposal of Stochastic Moving-MNIST is still necessary to study the models’ characteristics in handling random motion. With a highly deterministic dataset consisting of tiny digits, strong models like Earthformer and SimVP can almost perfectly grasp the motion and thus result in an excellent performance. Under such scenario, switching to FACL does not bring further improvement to the models.

# Evaluation Results for PredRNN [app:predrnn]

This section is an extension of Table <a href="#tab:loss_radar" data-reference-type="ref" data-reference="tab:loss_radar">2</a> featuring PredRNN. Due to the limitation of the model and consideration of training efficiency, we downscale the radar data to \\(128\times128\\) with bilinear interpolation. The results are reported in Table.<a href="#tab:radar_rnn" data-reference-type="ref" data-reference="tab:radar_rnn">6</a>. Note that the image resolution influences some metrics such as LPIPS and CSI with pooling, causing an unfair comparison with the results in Table <a href="#tab:loss_radar" data-reference-type="ref" data-reference="tab:loss_radar">2</a>.

<div id="tab:radar_rnn" markdown="1">

<table>
<caption>Comparison of the quantitative performance of different losses for PredRNN trained on SEVIR, MeteoNet and HKO-7. MAE is in the scale of <span class="math inline">10<sup>−3</sup></span>. The better score between MSE and FACL is highlighted in bold. </caption>
<tbody>
<tr>
<td colspan="2" rowspan="2" style="text-align: left;">Dataset</td>
<td rowspan="2" style="text-align: center;">Loss</td>
<td colspan="2" style="text-align: center;">Pixelwise/Structural</td>
<td colspan="2" style="text-align: center;">Perceptral</td>
<td colspan="4" style="text-align: center;">Skill</td>
<td style="text-align: center;">Proposed</td>
</tr>
<tr>
<td style="text-align: center;">MAE<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">SSIM<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">LPIPS<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">FVD<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">CSI-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">CSI<span class="math inline"><sub>4</sub></span>-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">CSI<span class="math inline"><sub>16</sub></span>-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">FSS<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">RHD<span><span class="math inline">↓</span></span></td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;">SEVIR</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;"><strong><span>28.91</span></strong></td>
<td style="text-align: center;"><strong><span>0.7238</span></strong></td>
<td style="text-align: center;">0.3572</td>
<td style="text-align: center;">528.8</td>
<td style="text-align: center;">0.3553</td>
<td style="text-align: center;">0.3702</td>
<td style="text-align: center;">0.4153</td>
<td style="text-align: center;">0.5552</td>
<td style="text-align: center;">1.1333</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;">31.37</td>
<td style="text-align: center;">0.7083</td>
<td style="text-align: center;"><strong><span>0.3206</span></strong></td>
<td style="text-align: center;"><strong><span>384.2</span></strong></td>
<td style="text-align: center;"><strong><span>0.3553</span></strong></td>
<td style="text-align: center;"><strong><span>0.4176</span></strong></td>
<td style="text-align: center;"><strong><span>0.5378</span></strong></td>
<td style="text-align: center;"><strong><span>0.5830</span></strong></td>
<td style="text-align: center;"><strong><span>0.8492</span></strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;">MeteoNet</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;"><strong><span>7.39</span></strong></td>
<td style="text-align: center;"><strong><span>0.9016</span></strong></td>
<td style="text-align: center;">0.1675</td>
<td style="text-align: center;">375.7</td>
<td style="text-align: center;"><strong><span>0.4348</span></strong></td>
<td style="text-align: center;">0.4472</td>
<td style="text-align: center;">0.4981</td>
<td style="text-align: center;"><strong><span>0.5835</span></strong></td>
<td style="text-align: center;">0.2207</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;">8.37</td>
<td style="text-align: center;">0.8935</td>
<td style="text-align: center;"><strong><span>0.1346</span></strong></td>
<td style="text-align: center;"><strong><span>214.2</span></strong></td>
<td style="text-align: center;">0.3690</td>
<td style="text-align: center;"><strong><span>0.4873</span></strong></td>
<td style="text-align: center;"><strong><span>0.6313</span></strong></td>
<td style="text-align: center;">0.5570</td>
<td style="text-align: center;"><strong><span>0.1515</span></strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;">HKO-7</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">MSE</td>
<td style="text-align: center;"><strong><span>23.79</span></strong></td>
<td style="text-align: center;">0.7081</td>
<td style="text-align: center;">0.3003</td>
<td style="text-align: center;">503.2</td>
<td style="text-align: center;">0.3168</td>
<td style="text-align: center;">0.3070</td>
<td style="text-align: center;">0.3213</td>
<td style="text-align: center;"><strong><span>0.4982</span></strong></td>
<td style="text-align: center;">0.7571</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">FACL</td>
<td style="text-align: center;">24.38</td>
<td style="text-align: center;"><strong><span>0.7174</span></strong></td>
<td style="text-align: center;"><strong><span>0.2617</span></strong></td>
<td style="text-align: center;"><strong><span>359.6</span></strong></td>
<td style="text-align: center;"><strong><span>0.3398</span></strong></td>
<td style="text-align: center;"><strong><span>0.3908</span></strong></td>
<td style="text-align: center;"><strong><span>0.4870</span></strong></td>
<td style="text-align: center;">0.4797</td>
<td style="text-align: center;"><strong><span>0.5339</span></strong></td>
</tr>
</tbody>
</table>

</div>

# Comparison with Other Loss Alternatives [app:other_losses]

Apart from the previous works discussed in Section <a href="#sec:related" data-reference-type="ref" data-reference="sec:related">2</a>, there has also been a series of attempts to improve the loss function. For instance, Balanced Mean Squared Error (BMSE) `\cite{shi2017trajgru}`{=latex} was proposed to increase the weighting on heavy rainfall. Multi-sigmoid loss (SSL) `\cite{chen2020a}`{=latex} preprocesses the images with linear transformations and nonlinear sigmoid function before applying MSE. Tran et al. `\cite{Tran2019computer}`{=latex} tested SSIM and MS-SSIM and recommended MSE+SSIM to be the loss function. We also present the results by adopting these losses with ConvLSTM trained on Stochastic Moving MNIST. For SSL, we follow the paper by picking \\(i \in [\frac{20}{70}, \frac{30}{70}]\\) and \\(c = 20\\).

<div id="tab:other_losses_1" markdown="1">

<table>
<caption>Comparison of the quantitative performance of different losses for ConvLSTM trained on Stochastic Moving-MNIST. </caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: left;">Loss</td>
<td colspan="6" style="text-align: center;">Metrics</td>
</tr>
<tr>
<td style="text-align: center;">MAE</td>
<td style="text-align: center;">SSIM</td>
<td style="text-align: center;">LPIPS</td>
<td style="text-align: center;">FVD</td>
<td style="text-align: center;">FSS</td>
<td style="text-align: center;">RHD</td>
</tr>
<tr>
<td style="text-align: left;">MSE</td>
<td style="text-align: center;">196.42</td>
<td style="text-align: center;">0.6975</td>
<td style="text-align: center;">0.2538</td>
<td style="text-align: center;">451.54</td>
<td style="text-align: center;">0.6148</td>
<td style="text-align: center;">1.1504</td>
</tr>
<tr>
<td style="text-align: left;">SSL <span class="citation" data-cites="chen2020a"></span></td>
<td style="text-align: center;"><strong><span>175.17</span></strong></td>
<td style="text-align: center;"><strong><span>0.7553</span></strong></td>
<td style="text-align: center;">0.1906</td>
<td style="text-align: center;">348.18</td>
<td style="text-align: center;">0.7225</td>
<td style="text-align: center;">0.9840</td>
</tr>
<tr>
<td style="text-align: left;">MSE+SSIM <span class="citation" data-cites="Tran2019computer"></span></td>
<td style="text-align: center;">184.10</td>
<td style="text-align: center;">0.7488</td>
<td style="text-align: center;">0.2573</td>
<td style="text-align: center;">529.71</td>
<td style="text-align: center;">0.3514</td>
<td style="text-align: center;">0.7921</td>
</tr>
<tr>
<td style="text-align: left;">FACL</td>
<td style="text-align: center;">180.10</td>
<td style="text-align: center;">0.7463</td>
<td style="text-align: center;"><strong><span>0.1092</span></strong></td>
<td style="text-align: center;"><strong><span>82.28</span></strong></td>
<td style="text-align: center;"><strong><span>0.8172</span></strong></td>
<td style="text-align: center;"><strong><span>0.3391</span></strong></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:other_losses_2" markdown="1">

<table>
<caption>Comparison of the quantitative performance of different losses for ConvLSTM trained on HKO-7. </caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: left;">Loss</td>
<td colspan="9" style="text-align: center;">Metrics</td>
</tr>
<tr>
<td style="text-align: center;">MAE</td>
<td style="text-align: center;">SSIM</td>
<td style="text-align: center;">LPIPS</td>
<td style="text-align: center;">FVD</td>
<td style="text-align: center;">CSI-m</td>
<td style="text-align: center;">CSI4-m</td>
<td style="text-align: center;">CSI16-m</td>
<td style="text-align: center;">FSS</td>
<td style="text-align: center;">RHD</td>
</tr>
<tr>
<td style="text-align: left;">MSE</td>
<td style="text-align: center;">30.43</td>
<td style="text-align: center;">0.6664</td>
<td style="text-align: center;">0.3057</td>
<td style="text-align: center;">791.3</td>
<td style="text-align: center;">0.2772</td>
<td style="text-align: center;">0.2282</td>
<td style="text-align: center;">0.1702</td>
<td style="text-align: center;">0.2653</td>
<td style="text-align: center;">1.2453</td>
</tr>
<tr>
<td style="text-align: left;">BMSE <span class="citation" data-cites="shi2017trajgru"></span></td>
<td style="text-align: center;">45.03</td>
<td style="text-align: center;">0.5537</td>
<td style="text-align: center;">0.3804</td>
<td style="text-align: center;">901.9</td>
<td style="text-align: center;"><strong><span>0.3484</span></strong></td>
<td style="text-align: center;"><strong><span>0.3670</span></strong></td>
<td style="text-align: center;"><strong><span>0.3354</span></strong></td>
<td style="text-align: center;">0.3999</td>
<td style="text-align: center;">1.7918</td>
</tr>
<tr>
<td style="text-align: left;">FACL</td>
<td style="text-align: center;"><strong><span>29.72</span></strong></td>
<td style="text-align: center;"><strong><span>0.7168</span></strong></td>
<td style="text-align: center;"><strong><span>0.2962</span></strong></td>
<td style="text-align: center;"><strong><span>569.1</span></strong></td>
<td style="text-align: center;">0.3054</td>
<td style="text-align: center;">0.3040</td>
<td style="text-align: center;">0.3351</td>
<td style="text-align: center;"><strong><span>0.7916</span></strong></td>
<td style="text-align: center;"><strong><span>0.4045</span></strong></td>
</tr>
</tbody>
</table>

</div>

<figure id="fig:losses_smmnist">
<img src="./figures/losses_smmnist.png"" style="width:80.0%" />
<figcaption>Output frames of ConvLSTM trained with MSE, SSL MSE+SSIM and FACL on Stochastic Moving-MNIST.</figcaption>
</figure>

<figure id="fig:bmse_hko7">
<img src="./figures/bmse_hko7.png"" style="width:80.0%" />
<figcaption>Output frames of ConvLSTM trained with MSE, BMSE and FACL on HKO-7.</figcaption>
</figure>

One point worth noting is that none of the methods other than FACL generates a sharp prediction qualitatively, despite occasionally higher pixel-wise performance. Besides, we can draw the following conclusions:

<div class="compactitem" markdown="1">

SSL improves the model performance in general, but still cannot generate clear output under stochastic motion.

Losses integrating SSIM (and also L1) “dissolves” the prediction to zero over time under uncertainty. Such an effect is especially significant for weaker models.

Weighted MSE (such as BMSE) only “tilts” the focus. BMSE severely over-predicts in exchange for an improvement in CSI, sacrificing all other metrics such as MAE, SSIM, LPIPS, FVD, FSS and RHD.

</div>

# Applying FACL to Generative Models [app:facl_generative]

Apart than replacing the MSE loss in video prediction, we also study the scenario where FACL is applied to video generative models. Specifically, we have tested three generative models using different generative methods: MCVD `\cite{voleti2022mvcd}`{=latex}, a diffusion-based model; STRPM `\cite{chang2022strpm}`{=latex}, a GAN-based model training a recurrent generator; and SVGLP `\cite{denton2018stochastic}`{=latex}, a VAE-based model with its default loss function being a composition of the MSE term and KL-divergence term. We replace the MSE term in each of these models with FACL, and report the quantitative performance in Table <a href="#tab:vg_implement" data-reference-type="ref" data-reference="tab:vg_implement">9</a>, while its visualization is reported in Figure <a href="#fig:vg_smmnist" data-reference-type="ref" data-reference="fig:vg_smmnist">9</a>.

In the table, FACL exhibits to be a good substitute for MSE even in the generative models as it improves most of the metrics. For both SVGLP and STRPM, replacing the reconstruction loss with FACL further improves the image quality of the prediction, as reflected by the significant drop in FVD and RHD and vast improvement in FSS. On the other hand, using FACL in MCVD is much less intuitive as the MSE loss fits the diffusion output to Gaussian noise. Since there is no point in studying the noise frequencies, the performance gain attributed to FACL appears trivial, resulting in comparable performance between MSE and FACL.

<div id="tab:vg_implement" markdown="1">

<table>
<caption>Quantitative performance of SVGLP, STRPM and MCVD with different loss, trained on the Stochastic Moving-MNIST. </caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;">Model</td>
<td rowspan="2" style="text-align: left;">Loss</td>
<td colspan="6" style="text-align: center;">Metrics</td>
</tr>
<tr>
<td style="text-align: center;">MAE<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">SSIM<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">LPIPS<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">FVD<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">FSS<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">RHD<span><span class="math inline">↓</span></span></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">SVGLP</td>
<td style="text-align: left;">MSE + <span class="math inline"><em>D</em><sub>KL</sub></span></td>
<td style="text-align: center;">209.5</td>
<td style="text-align: center;">0.7300</td>
<td style="text-align: center;">0.1412</td>
<td style="text-align: center;">136.80</td>
<td style="text-align: center;">0.7156</td>
<td style="text-align: center;">0.6031</td>
</tr>
<tr>
<td style="text-align: left;">FACL + <span class="math inline"><em>D</em><sub>KL</sub></span></td>
<td style="text-align: center;"><strong><span>201.1</span></strong></td>
<td style="text-align: center;"><strong><span>0.7377</span></strong></td>
<td style="text-align: center;"><strong><span>0.1080</span></strong></td>
<td style="text-align: center;"><strong><span>62.70</span></strong></td>
<td style="text-align: center;"><strong><span>0.7458</span></strong></td>
<td style="text-align: center;"><strong><span>0.3871</span></strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">STRPM</td>
<td style="text-align: left;">MSE + <span class="math inline"><em>L</em><sub>LP</sub></span> + <span class="math inline"><em>L</em><sub>GAN</sub></span></td>
<td style="text-align: center;"><strong><span>154.0</span></strong></td>
<td style="text-align: center;"><strong><span>0.7912</span></strong></td>
<td style="text-align: center;">0.1017</td>
<td style="text-align: center;">117.35</td>
<td style="text-align: center;">0.8337</td>
<td style="text-align: center;">0.3216</td>
</tr>
<tr>
<td style="text-align: left;">FACL + <span class="math inline"><em>L</em><sub>LP</sub></span> + <span class="math inline"><em>L</em><sub>GAN</sub></span></td>
<td style="text-align: center;">161.9</td>
<td style="text-align: center;">0.7849</td>
<td style="text-align: center;"><strong><span>0.0960</span></strong></td>
<td style="text-align: center;"><strong><span>91.97</span></strong></td>
<td style="text-align: center;"><strong><span>0.8453</span></strong></td>
<td style="text-align: center;"><strong><span>0.3113</span></strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">MCVD</td>
<td style="text-align: left;"><span class="math inline"><em>L</em><sub>vidpred</sub></span></td>
<td style="text-align: center;">219.9</td>
<td style="text-align: center;"><strong><span>0.7125</span></strong></td>
<td style="text-align: center;"><strong><span>0.1033</span></strong></td>
<td style="text-align: center;">44.70</td>
<td style="text-align: center;">0.7184</td>
<td style="text-align: center;">0.3941</td>
</tr>
<tr>
<td style="text-align: left;">FACL</td>
<td style="text-align: center;"><strong><span>219.6</span></strong></td>
<td style="text-align: center;">0.7051</td>
<td style="text-align: center;">0.1041</td>
<td style="text-align: center;"><strong><span>42.40</span></strong></td>
<td style="text-align: center;"><strong><span>0.7251</span></strong></td>
<td style="text-align: center;"><strong><span>0.3897</span></strong></td>
</tr>
</tbody>
</table>

</div>

<figure id="fig:vg_smmnist">
<img src="./figures/vg_smmnist.png"" />
<figcaption>Output frames of video generative models trained with different losses stated in Table <a href="#tab:vg_implement" data-reference-type="ref" data-reference="tab:vg_implement">9</a> on Stochastic Moving-MNIST.</figcaption>
</figure>

# Analysis and Discussion of RHD [app:rhd]

In the paper, we utilized three types of metrics to measure the similarity between two image sequences: pixel-wise and structural metrics, perceptual metrics and meteorological skill scores. Each suffers from its kind of drawbacks. For example, pixel-wise differences such as MAE and MSE do not consider the overall image structure, causing great encouragement to blurry prediction. Perceptual metrics such as LPIPS and FVD are based on pre-trained deep learning models (usually on ImageNet), which can suffer from domain bias that does not favor signal-based images. Moreover, deep perceptual metrics can be insensitive to transformations such as global rotation and spatial flipping.

To study the behavior of these metrics and compare them with RHD, we sample a random precipitation event (as visualized in Figure. <a href="#fig:distort" data-reference-type="ref" data-reference="fig:distort">10</a>) and apply the following transformations to the image:

<div class="compactitem" markdown="1">

Gaussian blur with kernel size \\(27\\) and \\(\sigma = 15\\).

Translation by \\((4, 4)\\).

Clockwise rotation by \\(5^{\circ}\\).

Brightening by 2X if the pixel value is higher than \\(0.5\\).

Darkening by 2X if the pixel value is lower than \\(0.5\\).

</div>

The former three transformations study the robustness of the metric under blurring and transformation. The brightening action simulates forecasts that overestimate and the darkening action simulates forecasts that underestimate. After obtaining the transformed images, we measure the evaluation metrics between the distorted images and their corresponding ground truth. The result is reported in Table. <a href="#tab:metrics_compare" data-reference-type="ref" data-reference="tab:metrics_compare">10</a>. FVD is not computed since it requires a larger set of data to form an image distribution.

In the table, a couple of behaviors deserve to be pointed out. First of all, pixel-wise and structural metrics appear to be insensitive to blur, which exhibits a huge difference compared with the other transformations in Figure <a href="#fig:distort" data-reference-type="ref" data-reference="fig:distort">10</a>. Such a characteristic discourages small translation which is undesired for precipitation nowcasting. Perceptual metrics such as LPIPS behave the opposite, where blur is the most penalized and value scaling (brightening and darkening) is the most rewarded transformation. Despite this, we believe LPIPS penalizes too little on brightening and darkening as they could result in wrong alerts for extreme weather. For the skill scores, we again observe that CSI with larger pooling tolerates more translation and rotation. In other words, CSI with a large pooling size can be a good metric to penalize blur. However, since CSI discourages false positives, low-range prediction usually wins in CSI which is undesirable for extreme weather forecast. On the other hand, FSS results in unstable behavior for brightening and darkening, due to a single threshold which causes a huge error within a binary cutoff. Among the metrics, we find that RHD is more robust to spatial and pixel-wise transformation while penalizing blur. With the multi-class behavior of RHD, it is also much more stable without bias over overestimation or underestimation.

<figure id="fig:distort">
<img src="./figures/rhd_analysis.png"" />
<figcaption>Visualization of different transformation techniques applied on the radar image.</figcaption>
</figure>

<div id="tab:metrics_compare" markdown="1">

<table>
<caption>The values of different metrics on different transformations, where MAE and MSE are in the scale of <span class="math inline">10<sup>−3</sup></span>. The worst score for each metric under the tested distortions is underlined and the best score is in bold. </caption>
<tbody>
<tr>
<td style="text-align: left;"></td>
<td colspan="3" style="text-align: center;">Pixel-wise/Structural</td>
<td style="text-align: center;">Perceptual</td>
<td colspan="4" style="text-align: center;">Skill</td>
<td style="text-align: center;">Proposed</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;">MAE<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">MSE<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">SSIM<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">LPIPS<span><span class="math inline">↓</span></span></td>
<td style="text-align: center;">CSI-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">CSI<span class="math inline"><sub>4</sub></span>-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">CSI<span class="math inline"><sub>16</sub></span>-m<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">FSS<span><span class="math inline">↑</span></span></td>
<td style="text-align: center;">RHD<span><span class="math inline">↓</span></span></td>
</tr>
<tr>
<td style="text-align: left;">Blur</td>
<td style="text-align: center;">17.90</td>
<td style="text-align: center;"><strong><span>2.18</span></strong></td>
<td style="text-align: center;">0.8487</td>
<td style="text-align: center;"><u>0.3660</u></td>
<td style="text-align: center;">0.5031</td>
<td style="text-align: center;">0.4725</td>
<td style="text-align: center;"><u>0.4210</u></td>
<td style="text-align: center;">0.5108</td>
<td style="text-align: center;">1.1088</td>
</tr>
<tr>
<td style="text-align: left;">Tran.</td>
<td style="text-align: center;">20.40</td>
<td style="text-align: center;">3.73</td>
<td style="text-align: center;">0.8320</td>
<td style="text-align: center;">0.1355</td>
<td style="text-align: center;">0.5582</td>
<td style="text-align: center;">0.6134</td>
<td style="text-align: center;">0.7763</td>
<td style="text-align: center;">0.6782</td>
<td style="text-align: center;">0.6133</td>
</tr>
<tr>
<td style="text-align: left;">Rot.</td>
<td style="text-align: center;"><u>32.29</u></td>
<td style="text-align: center;"><u>8.13</u></td>
<td style="text-align: center;"><u>0.7767</u></td>
<td style="text-align: center;">0.2121</td>
<td style="text-align: center;"><u>0.4084</u></td>
<td style="text-align: center;"><u>0.4520</u></td>
<td style="text-align: center;">0.6164</td>
<td style="text-align: center;">0.5180</td>
<td style="text-align: center;"><u>1.2650</u></td>
</tr>
<tr>
<td style="text-align: left;">Brig.</td>
<td style="text-align: center;">15.33</td>
<td style="text-align: center;">6.21</td>
<td style="text-align: center;"><strong><span>0.9561</span></strong></td>
<td style="text-align: center;">0.0778</td>
<td style="text-align: center;">0.5920</td>
<td style="text-align: center;">0.6105</td>
<td style="text-align: center;">0.6675</td>
<td style="text-align: center;"><strong><span>1.0000</span></strong></td>
<td style="text-align: center;"><strong><span>0.5663</span></strong></td>
</tr>
<tr>
<td style="text-align: left;">Dark.</td>
<td style="text-align: center;"><strong><span>13.08</span></strong></td>
<td style="text-align: center;">4.26</td>
<td style="text-align: center;">0.9461</td>
<td style="text-align: center;"><strong><span>0.0611</span></strong></td>
<td style="text-align: center;"><strong><span>0.7597</span></strong></td>
<td style="text-align: center;"><strong><span>0.7820</span></strong></td>
<td style="text-align: center;"><strong><span>0.8229</span></strong></td>
<td style="text-align: center;"><u>0.0781</u></td>
<td style="text-align: center;">0.5830</td>
</tr>
</tbody>
</table>

</div>

To sum up, RHD can be viewed as a generalization of FSS or CSI with consideration of multiple radii and pooling sizes. KL-divergence is adopted to measure the similarity of class distribution, replacing the binary segregation used commonly in the meteorological skill scores. Unlike SSIM being a normalized score from 0 to 1, only inspecting the magnitude of RHD is not meaningful. Instead, the users need to specify a fixed set of parameters such as window size and bin ranges, such that relative comparisons between two forecasts under the same set of parameters can provide useful evaluation feedback.

# Model Details and Hyper-parameters [app:hyperparam]

This section lists the implementation details of the models and the hyper-parameters used in the experiments described in Section <a href="#sec:experiments" data-reference-type="ref" data-reference="sec:experiments">4</a>.

We followed OpenSTL `\cite{tan2023openstl}`{=latex} in implementing PredRNN and SimVP. For PredRNN, apart from the zigzag recurrence, we also adopted scheduled sampling and patch reshaping. For SimVP, we chose the Inception module as the translator in SimVP (also known as SimVP v1). To support varying output sequence lengths, the input to the SimVP decoder is zero-padded to support cases where the output length is larger than the input length. The hyper-parameters are reported in Table <a href="#tab:hyperparam" data-reference-type="ref" data-reference="tab:hyperparam">11</a>.

<div id="tab:hyperparam" markdown="1">

<table>
<caption>Hyper-parameters used for training different models on different datasets. Models trained with both MSE and FACL share the same configuration. </caption>
<tbody>
<tr>
<td colspan="2" style="text-align: center;">Hyper-parameters</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Moving-MNIST</td>
<td style="text-align: left;">SEVIR</td>
<td style="text-align: center;">MeteoNet</td>
<td style="text-align: center;">HKO-7</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td rowspan="10" style="text-align: center;">Common</td>
<td style="text-align: left;">Input length</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">13</td>
<td style="text-align: center;">4<span class="math inline"><sup>*</sup></span></td>
<td style="text-align: center;">5</td>
</tr>
<tr>
<td style="text-align: left;">Output length</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">12</td>
<td style="text-align: center;">12</td>
<td style="text-align: center;">20</td>
</tr>
<tr>
<td style="text-align: left;">Optimizer</td>
<td colspan="4" style="text-align: center;">AdamW</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>β</em><sub>1</sub></span></td>
<td colspan="4" style="text-align: center;">0.9</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>β</em><sub>2</sub></span></td>
<td colspan="4" style="text-align: center;">0.999</td>
</tr>
<tr>
<td style="text-align: left;">Weight decay</td>
<td colspan="4" style="text-align: center;">0.01</td>
</tr>
<tr>
<td style="text-align: left;">LR Scheduler</td>
<td colspan="4" style="text-align: center;">Cosine Annealing</td>
</tr>
<tr>
<td style="text-align: left;">Max LR</td>
<td style="text-align: center;">1e-3</td>
<td style="text-align: center;">1e-3</td>
<td style="text-align: center;">1e-3</td>
<td style="text-align: center;">1e-3</td>
</tr>
<tr>
<td style="text-align: left;">Early stop</td>
<td colspan="4" style="text-align: center;">False</td>
</tr>
<tr>
<td style="text-align: left;">Training steps</td>
<td style="text-align: center;">200 epochs</td>
<td style="text-align: center;">50 epochs</td>
<td style="text-align: center;">20 epochs</td>
<td style="text-align: center;">50K steps</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: left;">Batch size</td>
<td style="text-align: center;">16</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;">4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: left;">Image size</td>
<td style="text-align: center;"><span class="math inline">64 × 64</span></td>
<td style="text-align: center;"><span class="math inline">384 × 384</span></td>
<td style="text-align: center;"><span class="math inline">256 × 256</span></td>
<td style="text-align: center;"><span class="math inline">480 × 480</span></td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;">PredRNN</td>
<td style="text-align: left;">Training steps</td>
<td style="text-align: center;">200 epochs</td>
<td style="text-align: center;">50 epochs</td>
<td style="text-align: center;">20 epochs</td>
<td style="text-align: center;">50K steps</td>
</tr>
<tr>
<td style="text-align: left;">Batch size</td>
<td style="text-align: center;">16</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;">4</td>
</tr>
<tr>
<td style="text-align: left;">Image size</td>
<td style="text-align: center;"><span class="math inline">64 × 64</span></td>
<td style="text-align: center;"><span class="math inline">128 × 128</span></td>
<td style="text-align: center;"><span class="math inline">128 × 128</span></td>
<td style="text-align: center;"><span class="math inline">128 × 128</span></td>
</tr>
<tr>
<td style="text-align: left;">Patch size</td>
<td style="text-align: center;"><span class="math inline">4 × 4</span></td>
<td style="text-align: center;"><span class="math inline">4 × 4</span></td>
<td style="text-align: center;"><span class="math inline">4 × 4</span></td>
<td style="text-align: center;"><span class="math inline">4 × 4</span></td>
</tr>
<tr>
<td rowspan="3" style="text-align: center;">SimVP</td>
<td style="text-align: left;">Training steps</td>
<td style="text-align: center;">1000 epochs</td>
<td style="text-align: center;">50 epochs</td>
<td style="text-align: center;">20 epochs</td>
<td style="text-align: center;">50K steps</td>
</tr>
<tr>
<td style="text-align: left;">Batch size</td>
<td style="text-align: center;">16</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;">4</td>
</tr>
<tr>
<td style="text-align: left;">Image size</td>
<td style="text-align: center;"><span class="math inline">64 × 64</span></td>
<td style="text-align: center;"><span class="math inline">384 × 384</span></td>
<td style="text-align: center;"><span class="math inline">256 × 256</span></td>
<td style="text-align: center;"><span class="math inline">480 × 480</span></td>
</tr>
<tr>
<td rowspan="6" style="text-align: center;">Earthformer</td>
<td style="text-align: left;">Training steps</td>
<td style="text-align: center;">200 epochs</td>
<td style="text-align: center;">50 epochs</td>
<td style="text-align: center;">20 epochs</td>
<td style="text-align: center;">50K steps</td>
</tr>
<tr>
<td style="text-align: left;">Batch size</td>
<td style="text-align: center;">32</td>
<td style="text-align: center;">32</td>
<td style="text-align: center;">32</td>
<td style="text-align: center;">32</td>
</tr>
<tr>
<td style="text-align: left;">Image size</td>
<td style="text-align: center;"><span class="math inline">64 × 64</span></td>
<td style="text-align: center;"><span class="math inline">384 × 384</span></td>
<td style="text-align: center;"><span class="math inline">256 × 256</span></td>
<td style="text-align: center;"><span class="math inline">480 × 480</span></td>
</tr>
<tr>
<td style="text-align: left;">Max LR</td>
<td style="text-align: center;">1e-3</td>
<td style="text-align: center;">1e-3</td>
<td style="text-align: center;">1e-3</td>
<td style="text-align: center;">1e-3</td>
</tr>
<tr>
<td style="text-align: left;">LR Scheduler</td>
<td colspan="4" style="text-align: center;">Cosine Annealing</td>
</tr>
<tr>
<td style="text-align: left;">Warm-up %</td>
<td colspan="4" style="text-align: center;">20%</td>
</tr>
<tr>
<td rowspan="11" style="text-align: center;">LDCast</td>
<td style="text-align: left;">Input length</td>
<td style="text-align: center;">8</td>
<td style="text-align: center;">12</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;">4</td>
</tr>
<tr>
<td style="text-align: left;">Output length</td>
<td style="text-align: center;">8</td>
<td style="text-align: center;">12</td>
<td style="text-align: center;">12</td>
<td style="text-align: center;">20</td>
</tr>
<tr>
<td style="text-align: left;">Image size</td>
<td style="text-align: center;"><span class="math inline">64 × 64</span></td>
<td style="text-align: center;"><span class="math inline">384 × 384</span></td>
<td style="text-align: center;"><span class="math inline">256 × 256</span></td>
<td style="text-align: center;"><span class="math inline">256 × 256</span></td>
</tr>
<tr>
<td style="text-align: left;">Optimizer</td>
<td colspan="4" style="text-align: center;">AdamW</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>β</em><sub>1</sub></span></td>
<td colspan="4" style="text-align: center;">0.5</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>β</em><sub>2</sub></span></td>
<td colspan="4" style="text-align: center;">0.9</td>
</tr>
<tr>
<td style="text-align: left;">Weight decay</td>
<td colspan="4" style="text-align: center;">0.001</td>
</tr>
<tr>
<td style="text-align: left;">LR Scheduler</td>
<td colspan="4" style="text-align: center;">Reduce-on-plateau</td>
</tr>
<tr>
<td style="text-align: left;">patience</td>
<td colspan="4" style="text-align: center;">3 epochs</td>
</tr>
<tr>
<td style="text-align: left;">Max LR</td>
<td colspan="4" style="text-align: center;">1e-4</td>
</tr>
<tr>
<td style="text-align: left;">Early stop</td>
<td colspan="4" style="text-align: center;">True</td>
</tr>
<tr>
<td rowspan="5" style="text-align: center;">MCVD</td>
<td style="text-align: left;">Training Steps</td>
<td style="text-align: center;">1000 epochs</td>
<td style="text-align: center;">200 epochs</td>
<td style="text-align: center;">50 epochs</td>
<td style="text-align: center;">150K steps</td>
</tr>
<tr>
<td style="text-align: left;">Batch Size</td>
<td style="text-align: center;">64</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;">8</td>
<td style="text-align: center;">16</td>
</tr>
<tr>
<td style="text-align: left;">Image size</td>
<td style="text-align: center;"><span class="math inline">64 × 64</span></td>
<td style="text-align: center;"><span class="math inline">384 × 384</span></td>
<td style="text-align: center;"><span class="math inline">256 × 256</span></td>
<td style="text-align: center;"><span class="math inline">128 × 128</span></td>
</tr>
<tr>
<td style="text-align: left;">LR Scheduler</td>
<td colspan="4" style="text-align: center;">Cosine Annealing</td>
</tr>
<tr>
<td style="text-align: left;">Warm-up %</td>
<td colspan="4" style="text-align: center;">20%</td>
</tr>
<tr>
<td colspan="6" style="text-align: left;">* In the case of Earthformer, the input length is set to be 12 regardless of the training loss.</td>
</tr>
</tbody>
</table>

</div>

# More Qualitative Visualization Comparing with FACL [app:more_visualize]

This section extends the visualizations in Figure <a href="#fig:vis_smmnist" data-reference-type="ref" data-reference="fig:vis_smmnist">2</a> and Figure <a href="#fig:loss_sevir" data-reference-type="ref" data-reference="fig:loss_sevir">3</a> by including the remaining models used in the experiments. Figure <a href="#fig:others_smmnist" data-reference-type="ref" data-reference="fig:others_smmnist">11</a> visualizes an example output of the remaining models on Stochastic Moving-MNIST and Figure <a href="#fig:others_sevir" data-reference-type="ref" data-reference="fig:others_sevir">12</a> visualizes that of SEVIR. In addition, we further plot an event from HKO-7 and MeteoNet, as shown in Figure <a href="#fig:others_meteo" data-reference-type="ref" data-reference="fig:others_meteo">13</a> and Figure <a href="#fig:others_hko7" data-reference-type="ref" data-reference="fig:others_hko7">14</a> respectively.

<figure id="fig:others_smmnist">
<img src="./figures/others_smmnist.png"" />
<figcaption>Output frames of the experimented model trained with different losses on Stochastic Moving-MNIST. The extra frames of LDCast are generated with auto-regressive inference.</figcaption>
</figure>

<figure id="fig:others_sevir">
<img src="./figures/others_sevir.png"" />
<figcaption>Output frames of the experimented model trained with different losses on SEVIR.</figcaption>
</figure>

<figure id="fig:others_meteo">
<img src="./figures/others_meteo.png"" />
<figcaption>Output frames of the experimented model trained with different losses on MeteoNet.</figcaption>
</figure>

<figure id="fig:others_hko7">
<img src="./figures/others_hko7.png"" />
<figcaption>Output frames of the experimented model trained with different losses on HKO-7.</figcaption>
</figure>

# NeurIPS Paper Checklist [neurips-paper-checklist]

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification:

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: Limitations are briefly discussed in the Conclusion.

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

14. Justification: Theoretical studies and proof are reported in the Appendices.

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

19. Justification: Experimental settings are described in Section <a href="#sec:experiments" data-reference-type="ref" data-reference="sec:experiments">4</a>. Hyper-parameters are provided in the Appendix. The code is submitted together with the paper.

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

24. Justification: The retrieval links of the data are provided in the readme file in the code.

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

29. Justification: Hyper-parameters and data processing are provided in the Appendix.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: Most experiments are performed only once due to the high computational requirement of spatiotemporal prediction problems.

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

39. Justification: The hardware and running time are provided in the Appendix.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification:

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: The submitted work has no apparent negative societal impact.

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

54. Justification:

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification:

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

64. Justification: The introduced dataset, Stochastic Moving-MNIST, is based on the existing dataset MNIST and the dataloader can be found in the code.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification:

70. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

73. Answer:

74. Justification:

75. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
