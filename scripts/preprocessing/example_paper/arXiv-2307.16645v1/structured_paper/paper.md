# Scaling Sentence Embeddings with  
Large Language Models

## Abstract

[1]

Large language models (LLMs) have recently garnered significant interest. With in-context learning, LLMs achieve impressive results in various natural language tasks. However, the application of LLMs to sentence embeddings remains an area of ongoing research. In this work, we propose an in-context learning-based method aimed at improving sentence embeddings performance. Our approach involves adapting the previous prompt-based representation method for autoregressive models, constructing a demonstration set that enables LLMs to perform in-context learning, and scaling up the LLMs to different model sizes. Through extensive experiments, in-context learning enables LLMs to generate high-quality sentence embeddings without any fine-tuning. It helps LLMs achieve performance comparable to current contrastive learning methods. By scaling model size, we find scaling to more than tens of billion parameters harms the performance on semantic textual similarity (STS) tasks. However, the largest model outperforms other counterparts and achieves the new state-of-the-art result on transfer tasks. We also fine-tune LLMs with current contrastive learning approach, and the 2.7B OPT model, incorporating our prompt-based method, surpasses the performance of 4.8B ST5, achieving the new state-of-the-art results on STS tasks. Our code is available at <https://github.com/kongds/scaling_sentemb>.

# Introduction

Sentence embeddings is a fundamental problem in natural language processing, requiring language models to project sentences into a vector space based on their semantics. Current methods based on contrastive learning, such as SimCSE , have successfully leveraged pretrained language models to generate high-quality embeddings. A significant amount of research has been devoted to refining the contrastive learning framework in order to further improve sentence embeddings .

Recently, large language models (LLMs), such as GPT-3  and LLaMA , have demonstrated significant potential on various natural language processing tasks such as translation, question answering, and text classification. Current research has also explored the application of LLMs for data augmentation in sentence embeddings. By generating better sentence pairs for contrastive learning, LLMs can help alleviate the scarcity of labeled data . However, directly utilizing LLMs to generate sentence embeddings presents two primary challenges. Firstly, LLMs, as autoregressive models, produce text instead of vectors, which necessitates vectorizing the output. Secondly, it is crucial to determine an effective approach for incorporating the capabilities of in-context learning into sentence embeddings.

In this work, we aim to investigate the capabilities of current LLMs for sentence embeddings, facilitated by the availability of open-source LLMs . We address the following research questions: 1) How can LLMs be used to represent sentence embeddings, and does prompt engineering, as demonstrated by PromptBERT ? 2) Can in-context learning  enhance the quality of sentence embeddings? 3) Does the scaling up the model parameters stil work when the number of parameters exceeds billions? 4) What improvements can be achieved by incorporating the current contrastive learning framework into LLMs?

To address these questions, we conduct a systematic study by evaluating LLaMA  and OPT  on both semantic textual similarity (STS) tasks and transfer tasks. Following , we utilize a prompt such as *This sentence: “* `[text]` *” means* to enable LLMs to generate sentence embeddings, where `[text]` serves as the input slot. This method outperforms traditional representation methods, such as averaging output tokens to represent sentences. Considering the causal architecture and pretraining tasks of LLMs compared to BERT, we can refine the prompt to generate better representations by instructing LLMs to encapsulate as much semantic information of the sentences as possible within the target token.

Inspired by , which uses definition sentences from a word dictionary to learn sentence embeddings, we find that performance can be further improved by adding definition sentences and corresponding words as examples to perform in-context learning. To mitigate the gap between examples and input sentences, we also use sentences from the STS-B  training set as examples by instructing ChatGPT to generate a single word to represent the meaning of sentences. By evaluating the demonstration examples based on the STS-B development set, LLMs can outperform previous contrastive learning-based sentence models, which were fine-tuned on unsupervised data.

By scaling up the parameters of LLMs, we find that transitioning from millions to billions of parameters results in improvements on STS tasks. However, continue scaling up may not yield further improvements. Even with in-context learning, 66B OPT still underperforms 6.7B OPT on STS tasks. Nonetheless, scaling up improves performance on transfer tasks. LLMs with tens of billions parameters exhibit strong performances, achieving state-of-the-art performance even without any fine-tuning.

With the advancement of parameter-efficient fine-tuning techniques and post-training quantization methods, we can also fine-tune LLMs with large batch sizes to conduct contrastive learning, even with limited computational resources. For instance, fine-tuning 7B parameter LLMs can be accomplished using the same hardware employed for previous BERT-based models like SimCSE . Even without fine-tuning the full parameters and using the 4-bit quantized method , 2.7B OPT with our sentence embeddings method outperforms a 4.8B ST5  and achieves the state-of-the-art results on STS tasks.

Our main contributions are as follows:

1.  We propose a sentence embeddings method that leverages LLMs to enhance the representation of sentences. Additionally, we incorporate in-context learning to further improve the quality of sentence embeddings. Our approach demonstrates that LLMs can generate high-quality sentence embeddings without the need for fine-tuning.

2.  We conduct an analysis of scaling up the parameters of LLMs from millions to tens of billions in sentence embeddings. We observe scaling to more than tens of billion parameters may harm the performance on STS tasks. However, the largest model can outperform other counterparts on transfer tasks.

3.  Based on our method, we discover that performance can be further enhanced by employing contrastive learning. By adopting efficient fine-tuning techniques, LLMs achieve state-of-the-art performance on STS tasks, even with limited computational resources.

# Related Work

#### Sentence Embeddings

Sentence embeddings is to convert a sentence into a fixed-size vector, which captures the semantic meaning and context of the sentence. It allows for the efficient retrieval of similar sentences through the similarity between vectors. Recently, SimCSE  demonstrated that contrastive learning is an effective approach for learning sentence embeddings using BERT in both unsupervised and supervised settings. In the unsupervised setting, SimCSE predicts the input sentence itself from in-batch negatives, with different dropout  masks applied. In the supervised setting, Natural Language Inference (NLI) datasets  are used to provide positive and negative pairs. Following the success of SimCSE, there has been a surge of work exploring contrastive learning-based methods. DiffCSE  incorporates a replaced token detection loss into the contrastive learning framework. PromptBERT  reveals that prompts can enhance BERT’s ability to represent sentences. Additionally, several studies  have investigated data augmentation for sentence embeddings using LLMs. SentenceT5 (ST5)  leverages the encoder-decoder structure of models, such as T5 , for generating sentence embeddings and demonstrates improvements by scaling T5 from millions to billions of parameters. However, directly using large language models (LLMs) to generate sentence embeddings remains an area of ongoing research.

#### Large Language Models

LLMs  recently show impressive performance on various natural language process, benefiting from their large parameter sizes compared to previous pretrained language models. LLMs can efficiently learn a new task with in-context learning by using training data as demonstrations . Without any gradient updates, LLMs with in-context learning can solve challenging tasks like multitask language understanding , commonsense reasoning , and math problems . This performance can be further improved by scaling up language models .

# Methodology

In this section, we first discuss current sentence embeddings methods with LLMs, and then introduce a new Prompt-based method with Explicit One word Limitation (PromptEOL) for LLMs in Section <a href="#sec:represent_llm" data-reference-type="ref" data-reference="sec:represent_llm">3.1</a>. Based on this method, we describe two settings: without and with fine-tuning. For the setting without fine-tuning, we utilize the in-context learning ability of LLMs to enhance sentence embeddings. To address the issue of lacking textual outputs, we propose two methods to automatically generate demonstrations for in-context learning in Section <a href="#sec:method_icl" data-reference-type="ref" data-reference="sec:method_icl">3.2</a>. For the setting with fine-tuning, we employ contrastive learning framework, and combine it with the efficient fine-tuning method to alleviate substantial memory requirement in Section <a href="#sec:contrastive_method" data-reference-type="ref" data-reference="sec:contrastive_method">3.3</a>.

## Represent Sentence with LLMs

Previous works  have extensively studied on improving sentence embeddings from encoder-based pretrained models, like BERT without fine-tuning. Recently, PromptBERT  leverage a prompt-based method to represent sentence. It uses manual templates like *This sentence: “* `[text]` *” means* `[MASK].`, where `[text]` is the placeholder for a sentence. The output vector of `[MASK]` token is used as sentence embeddings. It demonstrates superior results compared to previous sentence representation methods like averaging output hidden vectors or the output vector of `[CLS]` token.

<div class="wrapfigure">

r7cm

<img src="./figures/LLM_represent.png" alt="image" />

</div>

Considering to LLMs as autoregression models, which do not have special tokens like `[CLS]` or `[MASK]`, we modify the prompt-based method in  to make it compatible with LLMs. We use *This sentence: “* `[text]` *” means* to prompt LLMs generate next token and extract the hidden vectors of the final token as sentence embeddings. To validate the prompt-based method with LLMs, we compare it with two other methods, such as averaging or using the last token as sentence embeddings. For LLMs, we use OPT  from 125 million parameters to 66 billions and evaluate it on STS-B development set in Figure <a href="#fig:LLM_rep" data-reference-type="ref" data-reference="fig:LLM_rep">[fig:LLM_rep]</a>. Following the results in , we observe that prompt-based method can enhance sentence representation across all OPTs, ranging from millions to billions parameters. Despite that the previous prompt-based method also improved LLMs like OPT on sentence representations, OPT, even with significantly more parameters, still fails to outperform BERT.

Consider to bidirectional attention in BERT, we hypothesize that BERT can implicitly condense the entire semantic information corresponding to a sentence into a single `[MASK]` token when using templates like “*This sentence: “* `[text]` *” means* `[MASK].`”. Since the `[MASK]` token follows a period, this implicitly restricts BERT to explain meaning into one word. However, this template fails to add the similar “one word limitation” when it is used in autoregression models like OPT with unidirectional attention. To validate this, we simply remove the period in template to transfer it into “*This sentence: “* `[text]` *” means* `[MASK]`”. Despite only one word difference, and no modification to meaning of the template, the performance of BERT on STS-B development set plummeted from 73.44 to 33.89 Spearman correlation, which means BERT without this implicit “one word limitation” fails to represent sentence.

Inspired by this, our objective is to enhance prompt-based method for LLMs by introducing a “one word limitation”. We propose a new Prompt-based method with Explicit One word Limitation (PromptEOL) for LLMs. PromptEOL is simple and straightforward by directly adding some tokens in the template to instruct LLMs in predicting the meaning of sentence in one word. The template we used after modification is following:

*This sentence: “* `[text]` *” means in one word: “*

Compared to the template in , we introduce two simple modifications for LLMs. First, we append *in one word* to the prompt to constrain LLMs in predicting semantic information in next token. Secondly, we incorporate *: “* at the end of template to prevent model form generating punctuations in next token, as *This sentence: “* is used to indicate the input of a sentence. We find this template improve all OPT models and allow them to match or even outperform BERT with prompt-based method in Figure <a href="#fig:PromptEOL_rep" data-reference-type="ref" data-reference="fig:PromptEOL_rep">3</a>.

<figure><img src="./figures/framework.png" id="fig:framework" alt=" An illustration of in-context learning based sentence embeddings. The green sentences denote the demonstration sentence, and the blue words denote the demonstration words. The corresponding color blocks refer to their slots in the template. " /><figcaption aria-hidden="true"> An illustration of in-context learning based sentence embeddings. The <span style="color: greensentence">green</span> sentences denote the demonstration sentence, and the <span style="color: blueword">blue</span> words denote the demonstration words. The corresponding color blocks refer to their slots in the template. </figcaption></figure>

## Improve Sentence Embeddings with In-context Learning

In-context learning is widely utilized as an effective method to help LLMs understand problems. It improves their comprehension of inputs and outputs by directly adding a few examples in the prompts. However, when considering the problem of sentence embeddings, we need to project sentences into vectors based on their semantic information, separately. In other word, sentence embeddings lack textual outputs that could be used as examples to perform in-context learning, such as answers for question answer problems or labels for text classification problems. Moreover, there are also no predetermined gold vectors for a given sentence.

To leverage in-context learning in sentence embeddings, we propose an framework to automatically build demonstration sets and search demonstration to improve LLMs sentence embeddings in Figure <a href="#fig:framework" data-reference-type="ref" data-reference="fig:framework">1</a>. For the demonstration set, the goal is to create sentence and word pairs, where the word can represents the semantic information of the sentence. We propose two methods to generate pairs.

The first method involves using ChatGPT to generate corresponding words according to the semantic information of given sentences from STS-B training set. By asking ChatGPT with same template in Figure <a href="#fig:framework" data-reference-type="ref" data-reference="fig:framework">1</a>, ChatGPT outputs one word summary for the given sentence. We also find “one word limitation” in Section <a href="#sec:represent_llm" data-reference-type="ref" data-reference="sec:represent_llm">3.1</a> is important for ChatGPT. Consider to our prompt-based representation method, we employ the hidden state of the next token as the sentence embeddings. By removing *in one word* from the template, it tends to explain the meaning of a sentence in a lengthy way, and the first word often becomes an article such as “The”, which lacks clear meaning. For example, given the sentence “A jockey riding a horse.”, the hidden state achieves the highest dot product similarity for “Equestrain” among its word embeddings. However, without “one word limitation”, it will achieve the highest dot product similarity for word without specific meaning such as “The” among its word embeddings, which can not represent sentence properly. Inspired by DefSent , which leverages definition sentences with their words as labels to train unsupervised sentence embedding, our second method is also based on a word dictionary. We directly use words and their definition sentences in the Oxford dictionary as word-sentence pairs.

Based on these methods, we construct a demonstration set consisting of 300 pairs of sentences and words. 100 pairs are from STS-B training set, with words labeled by ChatGPT, while the remaining are from the Oxford dictionary. To find demonstration that help model to represent sentences, we directly evaluate each demonstration on the STS-B development set and use the demonstration with the best Spearman correlation as the demonstration for corresponding models. We also visualize the distribution of Spearman correlations for OPT from 125M to 66B parameters in Figure <a href="#fig:LLM_icl_hist" data-reference-type="ref" data-reference="fig:LLM_icl_hist">2</a>. Following the previous study , we notice that in-context learning achieves better performance, when increasing model parameter from 125M to 2.7B. For example, there are only one demonstration that helps the 125M OPT achieve better performance compared to without demonstration. However, around 98% of demonstrations improve the performance of the 2.7B OPT. In-context learning significantly enhance the sentence embeddings, especially for OPT with more than 1B parameters. With only in-context learning, OPT with more than 1.3B parameters even achieve better results on STS tasks compared to contrastive learning based method like SimCSE  in Table <a href="#tab:sts_wo_ft" data-reference-type="ref" data-reference="tab:sts_wo_ft">[tab:sts_wo_ft]</a>.

<figure><img src="./figures/LLM_icl_hist.png" id="fig:LLM_icl_hist" alt=" Distribution of Spearman correlations on the STS-B development set with different in-context learning demonstrations. The red dash line represents the Spearman correlation of the corresponding model without any demonstration. The blue area represents demonstrations that negatively impact the performance, and the percentage refers to the proportion of these demonstrations to the total number of demonstrations. " /><figcaption aria-hidden="true"> Distribution of Spearman correlations on the STS-B development set with different in-context learning demonstrations. The red dash line represents the Spearman correlation of the corresponding model without any demonstration. The blue area represents demonstrations that negatively impact the performance, and the percentage refers to the proportion of these demonstrations to the total number of demonstrations. </figcaption></figure>

## Contrastive Learning with Efficient Fine-tuning

Since in-context learning boosts sentence embeddings performances without any gradient update, we also exploit contrastive learning on LLMs, which has been demonstrated as an efficient way to learn sentence embeddings . It can be divided into unsupervised and supervised settings, according to the datasets. For unsupervised setting, the sentences in dataset lack corresponding positive and negative sentences to perform contrastive learning. For supervised setting, natural language inference (NLI) datasets are used as the datasets, and each sentence has corresponding positive and negative sentences. In this section, we focus on the supervised setting to fully leverage LLMs for sentence embeddings.

However, contrastive learning requires a large batch size to increase the number of negative samples, which demands a high amount of GPU memory, especially in the supervised setting. For example, SimCSE uses a batch size of 512 to fine-tune 110M BERT in the supervised setting. Each batch includes 1536 sentences, containing both their positive and hard negative sentences. It requires 58GB of GPU memory on 4 GPUs. As a result, fine-tuning LLMs with contrastive learning becomes challenging due to the memory requirements, particularly for models with significantly larger parameter sizes than BERT.

To solve this problem, we leverage current efficient fine-tuning method QLoRA . QLoRA combines two techniques to significantly reduces memory usage: 4-bit quantization and parameter efficient fine-tuning. Quantization reduces the memory usage of LLMs by quantizing their weight from 16-bit to 4-bit. Parameter efficient fine-tuning with LoRA  significantly reduces the memory usage of optimizer compared to full fine-tuning by only fine-tuning small proportion of weight.

Following , we use SNLI and MNLI datasets where each sentence *x*<sub>*i*</sub> has corresponding a positive sentence *x*<sub>*i*</sub><sup>+</sup> and a hard negative sentence *x*<sub>*i*</sub><sup>−</sup>. To represent sentence, we use our prompt-based method in Section <a href="#sec:represent_llm" data-reference-type="ref" data-reference="sec:represent_llm">3.1</a>. Formally, given sentence *x*<sub>*i*</sub>, we first add *x*<sub>*i*</sub> to the template and get hidden states:

$$\\begin{split}
  \\mathbf{h}\_{i1}, \\ldots, \\mathbf{h}\_{il} = {\\rm LLM}(\\textit{This  sen}&\\textit{tence: \`\`} x\_i \\textit{'' means in one word: \`\`})\\\\
  %h &= h\_L
\\end{split}$$

where *l* is the number of hidden states. We then use last token hidden state as its sentence embedding **h**<sub>*i*</sub> = **h**<sub>*i**l*</sub>. Since we can represent the sentence pair (*x*<sub>*i*</sub>, *x*<sub>*i*</sub><sup>+</sup>, *x*<sub>*i*</sub><sup>−</sup>) to their embeddings (**h**<sub>*i*</sub>, **h**<sub>*i*</sub><sup>+</sup>, **h**<sub>*i*</sub><sup>−</sup>). Our training objective is following:

$$\\ell\_{i}=-\\log \\frac{e^{\\operatorname{cos}\\left(\\mathbf{h}\_i, \\mathbf{h}\_i^{+}\\right) / \\tau}}{\\sum\_{j=1}^N\\left(e^{\\operatorname{cos}\\left(\\mathbf{h}\_i, \\mathbf{h}\_j^{+}\\right) / \\tau}+e^{\\operatorname{cos}\\left(\\mathbf{h}\_i, \\mathbf{h}\_j^{-}\\right) / \\tau}\\right)}$$

where *N* is the batch size and *τ* is the temperature hyperparameter in contrastive learning.

# Experiment

## Implementation Details

For the setting without fine-tuning, we use OPT from 125M to 66B parameters, and LLaMA from 7B to 65B parameters. All models use the same template in Section <a href="#sec:represent_llm" data-reference-type="ref" data-reference="sec:represent_llm">3.1</a>. We use 300 pairs of sentences and words as demonstration set for in-context learning. Among these, 100 pairs are from the STS-B training set, and we use `=gpt-3.5-turbo` to label their words. The remaining 200 pairs are from the Oxford dictionary. We provide all demonstrations in Appendix <a href="#apx:demo" data-reference-type="ref" data-reference="apx:demo">7</a>. For each model, we choose only one demonstration that has the highest Spearman correlation on the STS-B development set as their demonstration for evaluation. All results from models with 16-bit weights. We also present results using quantization methods in Appendix <a href="#apx:quant" data-reference-type="ref" data-reference="apx:quant">8</a>.

For the setting with fine-tuning, we use QLoRA  to fine-tune OPT and LLaMA with contrastive learning. Following QLoRA, we use LoRA *r* = 64, *α* = 16, dropout  = 0.05, and add LoRA modules on all linear layers of the 4-bit quantized model. We fine-tune models on the NLI datasets  with one epoch, temperature *τ* = 0.5 and learning rate 5e-4. Due to hardware limitations, we only conduct our experiments with model parameters less than or equal to 13B with 8 RTX-3090 GPUs. For models with fewer than 7B parameters, we fine-tune them on 2 GPUs with a batch size of 256. For 7B models, we use 4 GPUs with a batch size of 256. For 13B models, we use 8 GPUs with a batch size of 200.

## Dataset

Following previous works , We use the SentEval toolkit  to conduct our experiments on seven STS datasets and seven transfer learning datasets. The STS datasets include STS tasks 2012-2016  STS-B , SICK-R . Sentence pairs in each STS dataset are scored from 0 to 5 to indicate semantic similarity. Spearman correlation is used as a metric to evaluate the correlation between the cosine similarity of sentence embeddings and the golden similarity scores. The transfer learning datasets include MR , CR , SUBJ , MPQA , SST-2 , TREC  and MRPC . Sentence embeddings are used as input feature to train corresponding logistic regression classification.

## Results

We compare our method with BERT-based methods such as SBERT , SimCSE , and PromptBERT . In addition, we include other sentence methods based on LLMs as baselines, such as ST5  and SGPT . Among these baselines, ST5 achieves state-of-the-art results on both STS and transfer learning tasks by further fine-tuning 4.8B parameters T5 encoder with contrastive learning.

**STS tasks without fine-tuning** Table <a href="#tab:sts_wo_ft" data-reference-type="ref" data-reference="tab:sts_wo_ft">[tab:sts_wo_ft]</a> shows the results of PromptEOL with and without in-context learning on STS tasks. Even without corresponding textual outputs for sentence embeddings, in-context learning still helps model to generate better embeddings. As the model size grows, improvements from in-context learning also increase. Moreover, in-context learning shows significantly improvements on STS tasks for model with more than billions parameters. For instances, it raises the Spearman correlation from 68.84 to 78.19 on 66B OPT. Our method with in-context learning also outperforms among methods without fine-tuning. Even if we do not use any method to avoid anisotropy , which is widely regarded as the main reason for poor performance on STS tasks , our method still outperforms unsupervised methods such as SimCSE and PromptBERT, which use contrastive learning to avoid anistoropy. Additionally, we find the performance is not sensitive to the model size while scaling model beyond a billion parameters. Smaller models, such as 1.3B OPT, even outperforms SimCSE without fine-tuning.

**STS tasks with fine-tuning** Table <a href="#tab:sts_w_ft" data-reference-type="ref" data-reference="tab:sts_w_ft">[tab:sts_w_ft]</a> shows the results by fine-tuning with PromptEOL on the supervised dataset. Compared to ST5-Enc, which fine-tuned all 4.8B parameters on Community QA and NLI datasets, our method with 2.7B OPT achieves superior results through parameter-efficient fine tuning on the 4-bit model with only NLI datasets. Keep scaling up the parameters size, 13B OPT and LLaMA achieve the best performance on STS tasks. However, the improvement in scaling model parameters from 2.7B to 13B is not significant.

**Transfer tasks** We also report the results of our method on the transfer learning tasks in Table <a href="#tab:tran_w_ft" data-reference-type="ref" data-reference="tab:tran_w_ft">[tab:tran_w_ft]</a>. Unlike STS tasks, we observe that LLMs achieve better performance as the model size increases. Specifically, the 66B OPT and 65B LLaMA models outperform their smaller counterparts with our representation method. Based on our representation method, LLMs show good performance without in-context learning and contrastive learning. Following ST5 , we find that applying contrastive learning solely on NLI datasets can even harm performance on transfer tasks. To solve this problem, ST5 utilizes additional datasets, such as the Community QA dataset, to enhance its performance in transfer tasks. For in-context learning, as it is widely used in text classification, we find that using examples not relevant to tasks, such as STS-B or the dictionary, does not enhance transfer task performance. We present these results in Appendix <a href="#apx:transfer_task" data-reference-type="ref" data-reference="apx:transfer_task">9</a>.

# Analysis

## Sentence Representation Methods

We present the results obtained using three sentence representation methods, across models ranging in size from 125M to 66B parameters, as shown in Figure <a href="#fig:PromptEOL_rep" data-reference-type="ref" data-reference="fig:PromptEOL_rep">3</a>. Different representation methods can yield significantly different results. Prompt-based methods outperform direct averaging in three settings. Among these methods, PromptEOL exhibits the best performance, as it introduces an explicit “one-word limitation”. More detail results can be find in Appendix <a href="#apx:sentence_rep" data-reference-type="ref" data-reference="apx:sentence_rep">10</a>.

<figure><img src="./figures/PromptEOL_compare.png" id="fig:PromptEOL_rep" alt=" Influence of different sentence representation methods on three settings. “avg.” refers to use averaging output tokens as sentence embeddings. “prompt” refers to extract sentence embeddings using the template from  . Dash lines represent the results from the base-size BERT. " /><figcaption aria-hidden="true"> Influence of different sentence representation methods on three settings. “avg.” refers to use averaging output tokens as sentence embeddings. “prompt” refers to extract sentence embeddings using the template from <span class="citation" data-cites="jiang2022promptbert"></span> . Dash lines represent the results from the base-size BERT. </figcaption></figure>

## In-context Learning

<div class="tabular">

lp2.8cmcc & Sentence & Word & Improve  
125M & A man is smoking. & Smoking & 0.46  
350M & A man is playing on a guitar and singing. & Music & 3.99  
1.3B & relating to switzerland or its people. & Swiss & 4.34  
2.7B & A jockey riding a horse. & Equestrian & 8.88  
6.7B & The man is riding a horse. & Horseback-riding & 6.98  
13B & meat from a deer. & Venison & 7.18  
30B & The man is riding a motorcycle down the road. & Motorcycling & 6.51  
66B & of or relating to tutors or tutoring. & Tutorial & 9.35  

</div>

We demonstrate in-context learning examples that were obtained from each model on the STS-B development set, along with corresponding improvements on Spearman correlation for STS tasks. As the size of the model increases to 2.7B, the improvements in in-context learning become more and more pronounced, and related examples are usually more implicit. For instance, the 125M OPT uses examples where words are incorporated within the sentence.

# Conclusion

In this paper, we focus on exploiting Large Language Models (LLMs) to improve sentence embeddings. To achieve this, we propose a new sentence embeddings method called PromptEOL, which adapts previous prompt-based methods to autoregression models. Furthermore, we leverage in-context learning to generate superior sentence embeddings by utilizing ChatGPT and the Oxford dictionary to create sentence embeddings demonstrations. It demonstrates in-context learning allows LLMs to achieve performance comparable to current contrastive learning methods. With our promtp-based method, we also discover that further fine-tuning of LLMs can achieve the state-of-the-art performance using only efficient fine-tuning methods.

# References

<div class="thebibliography">

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Rada Mihalcea, German Rigau, and Janyce Wiebe Semeval-2014 task 10: Multilingual semantic textual similarity In *Proceedings of the 8th international workshop on semantic evaluation (SemEval 2014)*, pages 81–91, 2014. **Abstract:** Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Rada Mihalcea, German Rigau, Janyce Wiebe. Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014). 2014. (@agirre2014semeval)

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Inigo Lopez-Gazpio, Montse Maritxalar, Rada Mihalcea, et al Semeval-2015 task 2: Semantic textual similarity, english, spanish and pilot on interpretability In *Proceedings of the 9th international workshop on semantic evaluation (SemEval 2015)*, pages 252–263, 2015. **Abstract:** Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Iñigo Lopez-Gazpio, Montse Maritxalar, Rada Mihalcea, German Rigau, Larraitz Uria, Janyce Wiebe. Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval 2015). 2015. (@agirre2015semeval)

Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, Aitor Gonzalez Agirre, Rada Mihalcea, German Rigau Claramunt, and Janyce Wiebe Semeval-2016 task 1: Semantic textual similarity, monolingual and cross-lingual evaluation In *SemEval-2016. 10th International Workshop on Semantic Evaluation; 2016 Jun 16-17; San Diego, CA. Stroudsburg (PA): ACL; 2016. p. 497-511.* ACL (Association for Computational Linguistics), 2016. **Abstract:** Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Rada Mihalcea, German Rigau, Janyce Wiebe. Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016). 2016. (@agirre2016semeval)

Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, and Weiwei Guo sem 2013 shared task: Semantic textual similarity In *Second joint conference on lexical and computational semantics (\* SEM), volume 1: proceedings of the Main conference and the shared task: semantic textual similarity*, pages 32–43, 2013. **Abstract:** In Semantic Textual Similarity (STS), systems rate the degree of semantic equivalence, on a graded scale from 0 to 5, with 5 being the most similar. This year we set up two tasks: (i) a core task (CORE), and (ii) a typed-similarity task (TYPED). CORE is similar in set up to SemEval STS 2012 task with pairs of sentences from sources related to those of 2012, yet different in genre from the 2012 set, namely, this year we included newswire headlines, machine translation evaluation datasets and multiple lexical resource glossed sets. TYPED, on the other hand, is novel and tries to characterize why two items are deemed similar, using cultural heritage items which are described with metadata such as title, author or description. Several types of similarity have been defined, including similar author, similar time period or similar location. The annotation for both tasks leverages crowdsourcing, with relative high interannotator correlation, ranging from 62% to 87%. The CORE task attracted 34 participants with 89 runs, and the TYPED task attracted 6 teams with 14 runs. (@agirre2013sem)

Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre Semeval-2012 task 6: A pilot on semantic textual similarity In *SEM 2012: The First Joint Conference on Lexical and Computational Semantics–Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012)*, pages 385–393, 2012. **Abstract:** Semantic Textual Similarity (STS) measures the degree of semantic equivalence between two texts. This paper presents the results of the STS pilot task in Semeval. The training data contained 2000 sentence pairs from previously existing paraphrase datasets and machine translation evaluation resources. The test data also comprised 2000 sentences pairs for those datasets, plus two surprise datasets with 400 pairs from a different machine translation evaluation corpus and 750 pairs from a lexical resource mapping exercise. The similarity of pairs of sentences was rated on a 0-5 scale (low to high similarity) by human judges using Amazon Mechanical Turk, with high Pearson correlation scores, around 90%. 35 teams participated in the task, submitting 88 runs. The best results scored a Pearson correlation &gt;80%, well above a simple lexical baseline that only scored a 31% correlation. This pilot task opens an exciting way ahead, although there are still open issues, specially the evaluation metric. (@agirre2012semeval)

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei Language models are few-shot learners In *Advances in Neural Information Processing Systems*, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. **Abstract:** Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3’s few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general. (@gpt3)

Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation , 2017. **Abstract:** Semantic Textual Similarity (STS) measures the meaning similarity of sentences. Applications include machine translation (MT), summarization, generation, question answering (QA), short answer grading, semantic search, dialog and conversational systems. The STS shared task is a venue for assessing the current state-of-the-art. The 2017 task focuses on multilingual and cross-lingual pairs with one sub-track exploring MT quality estimation (MTQE) data. The task obtained strong participation from 31 teams, with 17 participating in all language tracks. We summarize performance and review a selection of well performing methods. Analysis highlights common errors, providing insight into the limitations of existing models. To support ongoing work on semantic representations, the STS Benchmark is introduced as a new shared training and evaluation set carefully selected from the corpus of English STS shared task data (2012-2017). (@cer2017semeval)

Yung-Sung Chuang, Rumen Dangovski, Hongyin Luo, Yang Zhang, Shiyu Chang, Marin Soljačić, Shang-Wen Li, Wen-tau Yih, Yoon Kim, and James Glass Diffcse: Difference-based contrastive learning for sentence embeddings , 2022. **Abstract:** We propose DiffCSE, an unsupervised contrastive learning framework for learning sentence embeddings. DiffCSE learns sentence embeddings that are sensitive to the difference between the original sentence and an edited sentence, where the edited sentence is obtained by stochastically masking out the original sentence and then sampling from a masked language model. We show that DiffSCE is an instance of equivariant contrastive learning (Dangovski et al., 2021), which generalizes contrastive learning and learns representations that are insensitive to certain types of augmentations and sensitive to other "harmful" types of augmentations. Our experiments show that DiffCSE achieves state-of-the-art results among unsupervised sentence representation learning methods, outperforming unsupervised SimCSE by 2.3 absolute points on semantic textual similarity tasks. (@chuang2022diffcse)

Alexis Conneau and Douwe Kiela Senteval: An evaluation toolkit for universal sentence representations , 2018. **Abstract:** We introduce SentEval, a toolkit for evaluating the quality of universal sentence representations. SentEval encompasses a variety of tasks, including binary and multi-class classification, natural language inference and sentence similarity. The set of tasks was selected based on what appears to be the community consensus regarding the appropriate evaluations for universal sentence representations. The toolkit comes with scripts to download and preprocess datasets, and an easy interface to evaluate sentence encoders. The aim is to provide a fairer, less cumbersome and more centralized way for evaluating sentence representations. (@conneau2018senteval)

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al Training verifiers to solve math word problems , 2021. **Abstract:** State-of-the-art language models can match human performance on many tasks, but they still struggle to robustly perform multi-step mathematical reasoning. To diagnose the failures of current models and support research, we introduce GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems. We find that even the largest transformer models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution. To increase performance, we propose training verifiers to judge the correctness of model completions. At test time, we generate many candidate solutions and select the one ranked highest by the verifier. We demonstrate that verification significantly improves performance on GSM8K, and we provide strong empirical evidence that verification scales more effectively with increased data than a finetuning baseline. (@cobbe2021training)

Alexis Conneau, Douwe Kiela, Holger Schwenk, Loı̈c Barrault, and Antoine Bordes Supervised learning of universal sentence representations from natural language inference data In *emnlp*, pages 670–680, 2017. **Abstract:** Many modern NLP systems rely on word embeddings, previously trained in an unsupervised manner on large corpora, as base features. Efforts to obtain embeddings for larger chunks of text, such as sentences, have however not been so successful. Several attempts at learning unsupervised representations of sentences have not reached satisfactory enough performance to be widely adopted. In this paper, we show how universal sentence representations trained using the supervised data of the Stanford Natural Language Inference datasets can consistently outperform unsupervised methods like SkipThought vectors on a wide range of transfer tasks. Much like how computer vision uses ImageNet to obtain features, which can then be transferred to other tasks, our work tends to indicate the suitability of natural language inference for transfer learning to other NLP tasks. Our encoder is publicly available. (@conneau-etal-2017-supervised-infersent)

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al Palm: Scaling language modeling with pathways , 2022. **Abstract:** Large language models have been shown to achieve remarkable performance across a variety of natural language tasks using few-shot learning, which drastically reduces the number of task-specific training examples needed to adapt the model to a particular application. To further our understanding of the impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer language model, which we call Pathways Language Model PaLM. We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly efficient training across multiple TPU Pods. We demonstrate continued benefits of scaling by achieving state-of-the-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the finetuned state-of-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the recently released BIG-bench benchmark. A significant number of BIG-bench tasks showed discontinuous improvements from model scale, meaning that performance steeply increased as we scaled to our largest model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias and toxicity, and study the extent of training data memorization with respect to model scale. Finally, we discuss the ethical considerations related to large language models and discuss potential mitigation strategies. (@chowdhery2022palm)

Qinyuan Cheng, Xiaogui Yang, Tianxiang Sun, Linyang Li, and Xipeng Qiu Improving contrastive learning of sentence embeddings from ai feedback , 2023. **Abstract:** Contrastive learning has become a popular approach in natural language processing, particularly for the learning of sentence embeddings. However, the discrete nature of natural language makes it difficult to ensure the quality of positive and negative sample pairs generated through data augmentation methods. Although supervised contrastive learning can produce more accurate sample pairs with human feedback labels, it still lacks fine-grained training signals. In this paper, we propose to improve {}textbf{C}ontrastive {}textbf{L}earning of sentence embeddings from {}textbf{AI} {}textbf{F}eedback {}textbf{(CLAIF)}. Our method utilizes AI feedback from large pre-trained language models (LLMs) to construct sample pairs with fine-grained sample similarity scores to improve contrastive learning. Besides, we combine human feedback and AI feedback to provide better supervision signals for supervised contrastive learning of sentence embeddings. Experimental results show that our method achieves state-of-the-art performance on several semantic textual similarity (STS) and transfer learning tasks compared to other unsupervised and supervised contrastive learning methods. (@cheng2023improving)

William B Dolan and Chris Brockett Automatically constructing a corpus of sentential paraphrases In *Proceedings of the Third International Workshop on Paraphrasing (IWP2005)*, 2005. **Abstract:** An obstacle to research in automatic paraphrase identification and generation is the lack of large-scale, publiclyavailable labeled corpora of sentential paraphrases. This paper describes the creation of the recently-released Microsoft Research Paraphrase Corpus, which contains 5801 sentence pairs, each hand-labeled with a binary judgment as to whether the pair constitutes a paraphrase. The corpus was created using heuristic extraction techniques in conjunction with an SVM-based classifier to select likely sentence-level paraphrases from a large corpus of topicclustered news data. These pairs were then submitted to human judges, who confirmed that 67% were in fact semantically equivalent. In addition to describing the corpus itself, we explore a number of issues that arose in defining guidelines for the human raters. (@mrpc2005)

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer Qlora: Efficient finetuning of quantized llms , 2023. **Abstract:** We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) double quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) paged optimziers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. A lemon-picked analysis demonstrates where Guanaco fails compared to ChatGPT. We release all of our models and code, including CUDA kernels for 4-bit training. (@dettmers2023qlora)

Kawin Ethayarajh How contextual are contextualized word representations? comparing the geometry of bert, elmo, and gpt-2 embeddings In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 55–65, 2019. **Abstract:** Replacing static word embeddings with contextualized word representations has yielded significant improvements on many NLP tasks. However, just how contextual are the contextualized representations produced by models such as ELMo and BERT? Are there infinitely many context-specific representations for each word, or are words essentially assigned one of a finite number of word-sense representations? For one, we find that the contextualized representations of all words are not isotropic in any layer of the contextualizing model. While representations of the same word in different contexts still have a greater cosine similarity than those of two different words, this self-similarity is much lower in upper layers. This suggests that upper layers of contextualizing models produce more context-specific representations, much like how upper layers of LSTMs produce more task-specific representations. In all layers of ELMo, BERT, and GPT-2, on average, less than 5% of the variance in a word’s contextualized representations can be explained by a static embedding for that word, providing some justification for the success of contextualized representations. (@ethayarajh2019contextual)

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh Gptq: Accurate post-training quantization for generative pre-trained transformers , 2022. **Abstract:** Generative Pre-trained Transformer models, known as GPT or OPT, set themselves apart through breakthrough performance across complex language modelling tasks, but also by their extremely high computational and storage costs. Specifically, due to their massive size, even inference for large, highly-accurate GPT models may require multiple performant GPUs, which limits the usability of such models. While there is emerging work on relieving this pressure via model compression, the applicability and performance of existing compression techniques is limited by the scale and complexity of GPT models. In this paper, we address this challenge, and propose GPTQ, a new one-shot weight quantization method based on approximate second-order information, that is both highly-accurate and highly-efficient. Specifically, GPTQ can quantize GPT models with 175 billion parameters in approximately four GPU hours, reducing the bitwidth down to 3 or 4 bits per weight, with negligible accuracy degradation relative to the uncompressed baseline. Our method more than doubles the compression gains relative to previously-proposed one-shot quantization methods, preserving accuracy, allowing us for the first time to execute an 175 billion-parameter model inside a single GPU for generative inference. Moreover, we also show that our method can still provide reasonable accuracy in the extreme quantization regime, in which weights are quantized to 2-bit or even ternary quantization levels. We show experimentally that these improvements can be leveraged for end-to-end inference speedups over FP16, of around 3.25x when using high-end GPUs (NVIDIA A100) and 4.5x when using more cost-effective ones (NVIDIA A6000). The implementation is available at https://github.com/IST-DASLab/gptq. (@frantar2022gptq)

Tianyu Gao, Xingcheng Yao, and Danqi Chen Simcse: Simple contrastive learning of sentence embeddings , 2021. **Abstract:** This paper presents SimCSE, a simple contrastive learning framework that greatly advances state-of-the-art sentence embeddings. We first describe an unsupervised approach, which takes an input sentence and predicts itself in a contrastive objective, with only standard dropout used as noise. This simple method works surprisingly well, performing on par with previous supervised counterparts. We find that dropout acts as minimal data augmentation, and removing it leads to a representation collapse. Then, we propose a supervised approach, which incorporates annotated pairs from natural language inference datasets into our contrastive learning framework by using "entailment" pairs as positives and "contradiction" pairs as hard negatives. We evaluate SimCSE on standard semantic textual similarity (STS) tasks, and our unsupervised and supervised models using BERT base achieve an average of 76.3% and 81.6% Spearman’s correlation respectively, a 4.2% and 2.2% improvement compared to the previous best results. We also show – both theoretically and empirically – that the contrastive learning objective regularizes pre-trained embeddings’ anisotropic space to be more uniform, and it better aligns positive pairs when supervised signals are available. (@gao2021simcse)

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt Measuring massive multitask language understanding , 2020. **Abstract:** We propose a new test to measure a text model’s multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model’s academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings. (@hendrycks2020measuring)

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al Training compute-optimal large language models , 2022. **Abstract:** We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4${}times$ more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher. (@hoffmann2022training)

Minqing Hu and Bing Liu Mining and summarizing customer reviews In *ACM SIGKDD international conference on Knowledge discovery and data mining*, 2004. **Abstract:** Merchants selling products on the Web often ask their customers to review the products that they have purchased and the associated services. As e-commerce is becoming more and more popular, the number of customer reviews that a product receives grows rapidly. For a popular product, the number of reviews can be in hundreds or even thousands. This makes it difficult for a potential customer to read them to make an informed decision on whether to purchase the product. It also makes it difficult for the manufacturer of the product to keep track and to manage customer opinions. For the manufacturer, there are additional difficulties because many merchant sites may sell the same product and the manufacturer normally produces many kinds of products. In this research, we aim to mine and to summarize all the customer reviews of a product. This summarization task is different from traditional text summarization because we only mine the features of the product on which the customers have expressed their opinions and whether the opinions are positive or negative. We do not summarize the reviews by selecting a subset or rewrite some of the original sentences from the reviews to capture the main points as in the classic text summarization. Our task is performed in three steps: (1) mining product features that have been commented on by customers; (2) identifying opinion sentences in each review and deciding whether each opinion sentence is positive or negative; (3) summarizing the results. This paper proposes several novel techniques to perform these tasks. Our experimental results using reviews of a number of products sold online demonstrate the effectiveness of the techniques. (@hu2004mining\_cr)

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen Lora: Low-rank adaptation of large language models , 2021. **Abstract:** An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at https://github.com/microsoft/LoRA. (@hu2021lora)

Ting Jiang, Jian Jiao, Shaohan Huang, Zihan Zhang, Deqing Wang, Fuzhen Zhuang, Furu Wei, Haizhen Huang, Denvy Deng, and Qi Zhang Promptbert: Improving bert sentence embeddings with prompts , 2022. **Abstract:** We propose PromptBERT, a novel contrastive learning method for learning better sentence representation. We firstly analyze the drawback of current sentence embedding from original BERT and find that it is mainly due to the static token embedding bias and ineffective BERT layers. Then we propose the first prompt-based sentence embeddings method and discuss two prompt representing methods and three prompt searching methods to make BERT achieve better sentence embeddings. Moreover, we propose a novel unsupervised training objective by the technology of template denoising, which substantially shortens the performance gap between the supervised and unsupervised settings. Extensive experiments show the effectiveness of our method. Compared to SimCSE, PromptBert achieves 2.29 and 2.58 points of improvement based on BERT and RoBERTa in the unsupervised setting. (@jiang2022promptbert)

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei Scaling laws for neural language models , 2020. **Abstract:** We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence. (@kaplan2020scaling)

Stephanie Lin, Jacob Hilton, and Owain Evans Truthfulqa: Measuring how models mimic human falsehoods , 2021. **Abstract:** We propose a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. We crafted questions that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts. We tested GPT-3, GPT-Neo/J, GPT-2 and a T5-based model. The best model was truthful on 58% of questions, while human performance was 94%. Models generated many false answers that mimic popular misconceptions and have the potential to deceive humans. The largest models were generally the least truthful. This contrasts with other NLP tasks, where performance improves with model size. However, this result is expected if false answers are learned from the training distribution. We suggest that scaling up models alone is less promising for improving truthfulness than fine-tuning using training objectives other than imitation of text from the web. (@lin2021truthfulqa)

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing , 55(9):1–35, 2023. **Abstract:** This article surveys and organizes research works in a new paradigm in natural language processing, which we dub “prompt-based learning.” Unlike traditional supervised learning, which trains a model to take in an input x and predict an output y as P ( y\|x ), prompt-based learning is based on language models that model the probability of text directly. To use these models to perform prediction tasks, the original input x is modified using a template into a textual string prompt x′ that has some unfilled slots, and then the language model is used to probabilistically fill the unfilled information to obtain a final string x̂ , from which the final output y can be derived. This framework is powerful and attractive for a number of reasons: It allows the language model to be pre-trained on massive amounts of raw text, and by defining a new prompting function the model is able to perform few-shot or even zero-shot learning, adapting to new scenarios with few or no labeled data. In this article, we introduce the basics of this promising paradigm, describe a unified set of mathematical notations that can cover a wide variety of existing work, and organize existing work along several dimensions, e.g., the choice of pre-trained language models, prompts, and tuning strategies. To make the field more accessible to interested beginners, we not only make a systematic review of existing works and a highly structured typology of prompt-based concepts but also release other resources, e.g., a website NLPedia–Pretrain including constantly updated survey and paperlist. (@liu2023pre)

Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li On the sentence embeddings from pre-trained language models , 2020. **Abstract:** Pre-trained contextual representations like BERT have achieved great success in natural language processing. However, the sentence embeddings from the pre-trained language models without fine-tuning have been found to poorly capture semantic meaning of sentences. In this paper, we argue that the semantic information in the BERT embeddings is not fully exploited. We first reveal the theoretical connection between the masked language model pre-training objective and the semantic similarity task theoretically, and then analyze the BERT sentence embeddings empirically. We find that BERT always induces a non-smooth anisotropic semantic space of sentences, which harms its performance of semantic similarity. To address this issue, we propose to transform the anisotropic sentence embedding distribution to a smooth and isotropic Gaussian distribution through normalizing flows that are learned with an unsupervised objective. Experimental results show that our proposed BERT-flow method obtains significant performance gains over the state-of-the-art sentence embeddings on a variety of semantic textual similarity tasks. The code is available at https://github.com/bohanli/BERT-flow. (@li2020sentence)

Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, Roberto Zamparelli, et al A sick cure for the evaluation of compositional distributional semantic models In *Lrec*, pages 216–223. Reykjavik, 2014. (@marelli2014sick)

Niklas Muennighoff Sgpt: Gpt sentence embeddings for semantic search , 2022. **Abstract:** Decoder transformers have continued increasing in scale reaching hundreds of billions of parameters. Due to their scale the same decoder sets state-of-the-art results on various language tasks via prompting or fine-tuning. Yet, these large foundation models remain unusable for the related fields of semantic search and sentence embeddings. This prevents possibly new state-of-the-art results and forces organizations to train and maintain separate models. To this end, we propose SGPT to use decoders for sentence embeddings and semantic search via prompting or fine-tuning. At 5.8 billion parameters SGPT improves on the previously best sentence embeddings by a margin of 7% and outperforms a concurrent method with 175 billion parameters as measured on the BEIR search benchmark. Code, models and result files are freely available at https://github.com/Muennighoff/sgpt. (@muennighoff2022sgpt)

Jianmo Ni, Gustavo Hernández Ábrego, Noah Constant, Ji Ma, Keith B Hall, Daniel Cer, and Yinfei Yang Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models , 2021. **Abstract:** We provide the first exploration of sentence embeddings from text-to-text transformers (T5). Sentence embeddings are broadly useful for language processing tasks. While T5 achieves impressive performance on language tasks cast as sequence-to-sequence mapping problems, it is unclear how to produce sentence embeddings from encoder-decoder models. We investigate three methods for extracting T5 sentence embeddings: two utilize only the T5 encoder and one uses the full T5 encoder-decoder model. To support our investigation, we establish a new sentence representation transfer benchmark, SentGLUE, which extends the SentEval toolkit to nine tasks from the GLUE benchmark. Our encoder-only models outperforms Sentence-BERT and SimCSE sentence embeddings on both SentEval and SentGLUE transfer tasks, including semantic textual similarity (STS). Scaling up T5 from millions to billions of parameters is found to produce consistent further improvements. Finally, our encoder-decoder method achieves a new state-of-the-art on STS when using sentence embeddings. Our models are released at https://tfhub.dev/google/collections/sentence-t5/1. (@sentencet5)

Bo Pang and Lillian Lee A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts In *acl*, pages 271–278, 2004. **Abstract:** Sentiment analysis seeks to identify the viewpoint(s) underlying a text span; an example application is classifying a movie review as "thumbs up" or "thumbs down". To determine this sentiment polarity, we propose a novel machine-learning method that applies text-categorization techniques to just the subjective portions of the document. Extracting these portions can be implemented using efficient techniques for finding minimum cuts in graphs; this greatly facilitates incorporation of cross-sentence contextual constraints. (@pang2004sentimental\_subj)

Bo Pang and Lillian Lee Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales In *acl*, pages 115–124, 2005. **Abstract:** We address the rating-inference problem, wherein rather than simply decide whether a review is "thumbs up" or "thumbs down", as in previous sentiment analysis work, one must determine an author’s evaluation with respect to a multi-point scale (e.g., one to five "stars"). This task represents an interesting twist on standard multi-class text categorization because there are several different degrees of similarity between class labels; for example, "three stars" is intuitively closer to "four stars" than to "one star". We first evaluate human performance at the task. Then, we apply a meta-algorithm, based on a metric labeling formulation of the problem, that alters a given n-ary classifier’s output in an explicit attempt to ensure that similar items receive similar labels. We show that the meta-algorithm can provide significant improvements over both multi-class and regression versions of SVMs when we employ a novel similarity measure appropriate to the problem. (@pang2005seeing\_mr)

Nils Reimers and Iryna Gurevych Sentence-bert: Sentence embeddings using siamese bert-networks , 2019. **Abstract:** BERT (Devlin et al., 2018) and RoBERTa (Liu et al., 2019) has set a new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity (STS). However, it requires that both sentences are fed into the network, which causes a massive computational overhead: Finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations ( 65 hours) with BERT. The construction of BERT makes it unsuitable for semantic similarity search as well as for unsupervised tasks like clustering. In this publication, we present Sentence-BERT (SBERT), a modification of the pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65 hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT. We evaluate SBERT and SRoBERTa on common STS tasks and transfer learning tasks, where it outperforms other state-of-the-art sentence embeddings methods. (@reimers2019sentence)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu Exploring the limits of transfer learning with a unified text-to-text transformer , 21(1):5485–5551, 2020. **Abstract:** Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code. (@raffel2020exploring)

Teven Le Scao, 388 Authors, and Thomas Wolf : A 176B-parameter open-access multilingual language model , abs/2211.05100, 2022. **Abstract:** Large language models (LLMs) have been shown to be able to perform new tasks based on a few demonstrations or natural language instructions. While these capabilities have led to widespread adoption, most LLMs are developed by resource-rich organizations and are frequently kept from the public. As a step towards democratizing this powerful technology, we present BLOOM, a 176B-parameter open-access language model designed and built thanks to a collaboration of hundreds of researchers. BLOOM is a decoder-only Transformer language model that was trained on the ROOTS corpus, a dataset comprising hundreds of sources in 46 natural and 13 programming languages (59 in total). We find that BLOOM achieves competitive performance on a wide variety of benchmarks, with stronger results after undergoing multitask prompted finetuning. To facilitate future research and applications using LLMs, we publicly release our models and code under the Responsible AI License. (@bloom)

Jianlin Su, Jiarun Cao, Weijie Liu, and Yangyiwen Ou Whitening sentence representations for better semantics and faster retrieval , 2021. **Abstract:** Pre-training models such as BERT have achieved great success in many natural language processing tasks. However, how to obtain better sentence representation through these pre-training models is still worthy to exploit. Previous work has shown that the anisotropy problem is an critical bottleneck for BERT-based sentence representation which hinders the model to fully utilize the underlying semantic features. Therefore, some attempts of boosting the isotropy of sentence distribution, such as flow-based model, have been applied to sentence representations and achieved some improvement. In this paper, we find that the whitening operation in traditional machine learning can similarly enhance the isotropy of sentence representations and achieve competitive results. Furthermore, the whitening technique is also capable of reducing the dimensionality of the sentence representation. Our experimental results show that it can not only achieve promising performance but also significantly reduce the storage cost and accelerate the model retrieval speed. (@su2021whitening)

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov Dropout: a simple way to prevent neural networks from overfitting , 15(1):1929–1958, 2014. **Abstract:** Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. During training, dropout samples from an exponential number of different networks. At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. This significantly reduces overfitting and gives major improvements over other regularization methods. We show that dropout improves the performance of neural networks on supervised learning tasks in vision, speech recognition, document classification and computational biology, obtaining state-of-the-art results on many benchmark data sets. (@srivastava2014dropout)

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts Recursive deep models for semantic compositionality over a sentiment treebank In *emnlp*, pages 1631–1642, 2013. **Abstract:** Semantic word spaces have been very useful but cannot express the meaning of longer phrases in a principled way. Further progress towards understanding compositionality in tasks such as sentiment detection requires richer supervised training and evaluation resources and more powerful models of composition. To remedy this, we introduce a Sentiment Treebank. It includes fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences and presents new challenges for sentiment compositionality. To address them, we introduce the Recursive Neural Tensor Network. When trained on the new treebank, this model outperforms all previous methods on several metrics. It pushes the state of the art in single sentence positive/negative classification from 80% up to 85.4%. The accuracy of predicting fine-grained sentiment labels for all phrases reaches 80.7%, an improvement of 9.7% over bag of features baselines. Lastly, it is the only model that can accurately capture the effects of negation and its scope at various tree levels for both positive and negative phrases. (@socher2013recursive\_sst-2)

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al Llama: Open and efficient foundation language models , 2023. **Abstract:** We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community. (@touvron2023llama)

Hayato Tsukagoshi, Ryohei Sasano, and Koichi Takeda efSent: Sentence embeddings using definition sentences In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*, pages 411–418, Online, August 2021. Association for Computational Linguistics. **Abstract:** Hayato Tsukagoshi, Ryohei Sasano, Koichi Takeda. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 2021. (@tsukagoshi-etal-2021-defsent)

Ellen M Voorhees and Dawn M Tice Building a question answering test collection In *the 23rd annual international ACM SIGIR conference on Research and development in information retrieval*, pages 200–207, 2000. **Abstract:** The TREC-8 Question Answering (QA) Track was the first large-scale evaluation of domain-independent question answering systems. In addition to fostering research on the QA task, the track was used to investigate whether the evaluation methodology used for document retrieval is appropriate for a different natural language processing task. As with document relevance judging, assessors had legitimate differences of opinions as to whether a response actually answers a question, but comparative evaluation of QA systems was stable despite these differences. Creating a reusable QA test collection is fundamentally more difficult than creating a document retrieval test collection since the QA task has no equivalent to document identifiers. (@voorhees2000building\_trec)

Xing Wu, Chaochen Gao, Zijia Lin, Jizhong Han, Zhongyuan Wang, and Songlin Hu nfoCSE: Information-aggregated contrastive learning of sentence embeddings In *Findings of the Association for Computational Linguistics: EMNLP 2022*, pages 3060–3070, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. **Abstract:** Contrastive learning has been extensively studied in sentence embedding learning, which assumes that the embeddings of different views of the same sentence are closer. The constraint brought by this assumption is weak, and a good sentence representation should also be able to reconstruct the original sentence fragments. Therefore, this paper proposes an information-aggregated contrastive learning framework for learning unsupervised sentence embeddings, termed InfoCSE.InfoCSE forces the representation of \[CLS\] positions to aggregate denser sentence information by introducing an additional Masked language model task and a well-designed network. We evaluate the proposed InfoCSE on several benchmark datasets w.r.t the semantic text similarity (STS) task. Experimental results show that InfoCSE outperforms SimCSE by an average Spearman correlation of 2.60% on BERT-base, and 1.77% on BERT-large, achieving state-of-the-art results among unsupervised sentence representation learning methods. (@wu-etal-2022-infocse)

Qiyu Wu, Chongyang Tao, Tao Shen, Can Xu, Xiubo Geng, and Daxin Jiang Pcl: Peer-contrastive learning with diverse augmentations for unsupervised sentence embeddings , 2022. **Abstract:** Learning sentence embeddings in an unsupervised manner is fundamental in natural language processing. Recent common practice is to couple pre-trained language models with unsupervised contrastive learning, whose success relies on augmenting a sentence with a semantically-close positive instance to construct contrastive pairs. Nonetheless, existing approaches usually depend on a mono-augmenting strategy, which causes learning shortcuts towards the augmenting biases and thus corrupts the quality of sentence embeddings. A straightforward solution is resorting to more diverse positives from a multi-augmenting strategy, while an open question remains about how to unsupervisedly learn from the diverse positives but with uneven augmenting qualities in the text field. As one answer, we propose a novel Peer-Contrastive Learning (PCL) with diverse augmentations. PCL constructs diverse contrastive positives and negatives at the group level for unsupervised sentence embeddings. PCL performs peer-positive contrast as well as peer-network cooperation, which offers an inherent anti-bias ability and an effective way to learn from diverse augmentations. Experiments on STS benchmarks verify the effectiveness of PCL against its competitors in unsupervised sentence embeddings. (@wu2022pcl)

Janyce Wiebe, Theresa Wilson, and Claire Cardie Annotating expressions of opinions and emotions in language , 39(2-3):165–210, 2005. **Abstract:** This paper describes a corpus annotation project to study issues in the manual annotation of opinions, emotions, sentiments, speculations, evaluations and other private states in language. The resulting corpus annotation scheme is described, as well as examples of its use. In addition, the manual annotation process and the results of an inter-annotator agreement study on a 10,000-sentence corpus of articles drawn from the world press are presented. (@wiebe2005annotating\_mpqa)

Junlei Zhang, Zhenzhong Lan, and Junxian He Contrastive learning of sentence embeddings from scratch , 2023. **Abstract:** Contrastive learning has been the dominant approach to train state-of-the-art sentence embeddings. Previous studies have typically learned sentence embeddings either through the use of human-annotated natural language inference (NLI) data or via large-scale unlabeled sentences in an unsupervised manner. However, even in the case of unlabeled data, their acquisition presents challenges in certain domains due to various reasons. To address these issues, we present SynCSE, a contrastive learning framework that trains sentence embeddings with synthesized data. Specifically, we explore utilizing large language models to synthesize the required data samples for contrastive learning, including (1) producing positive and negative annotations given unlabeled sentences (SynCSE-partial), and (2) generating sentences along with their corresponding annotations from scratch (SynCSE-scratch). Experimental results on sentence similarity and reranking tasks indicate that both SynCSE-partial and SynCSE-scratch greatly outperform unsupervised baselines, and SynCSE-partial even achieves comparable performance to the supervised models in most settings. (@zhang2023contrastive)

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al Opt: Open pre-trained transformer language models , 2022. **Abstract:** Large language models, which are often trained for hundreds of thousands of compute days, have shown remarkable capabilities for zero- and few-shot learning. Given their computational cost, these models are difficult to replicate without significant capital. For the few that are available through APIs, no access is granted to the full model weights, making them difficult to study. We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We show that OPT-175B is comparable to GPT-3, while requiring only 1/7th the carbon footprint to develop. We are also releasing our logbook detailing the infrastructure challenges we faced, along with code for experimenting with all of the released models. (@zhang2022opt)

</div>

# Demonstrations

<div class="longtable">

p12cmc Over 100 dead as typhoon slams central Philippines. & Disaster  
Woman in red overalls standing on the sidewalk. & Observation  
India starts voting in world’s largest election. & Democracy  
Three dogs pulling a man on a bicycle through the snow. & Adventure  
Spain approves new restrictive abortion law. & Legislation  
A man dives into a pool. & Activity  
Saudi to give Lebanese army $3 billion & Aid  
Updated - Two explosions near finish line of Boston Marathon & Terrorism  
A gray cat with green eyes looks at the camera. & Portrayal  
Egypt interior minister survives bomb & Survival  
A man is playing a large flute. & Music  
A man is spreading shreded cheese on a pizza. & Cooking  
Three men are playing chess. & Strategy  
A man is playing the cello. & Music  
Some men are fighting. & Conflict  
A man is smoking. & Smoking  
The man is playing the piano. & Music  
A man is playing on a guitar and singing. & Music  
A person is throwing a cat on to the ceiling. & Cruelty  
The man hit the other man with a stick. & Violence  
A woman picks up and holds a baby kangaroo. & Caring  
A man is playing a flute. & Music  
A person is folding a piece of paper. & Origami  
A man is running on the road. & Exercise  
A dog is trying to get bacon off his back. & Humorous  
The polar bear is sliding on the snow. & Playful  
A woman is writing. & Writing  
A cat is rubbing against baby’s face. & Affection  
The man is riding a horse. & Horseback-riding  
A man pours oil into a pot. & Cooking  
A man is playing a guitar. & Music  
A panda is sliding down a slide. & Playful  
A woman is eating something. & Eating  
A woman peels a potato. & Cooking  
The boy fell off his bike. & Accident  
The woman is playing the flute. & Music  
A rabbit is running from an eagle. & Escape  
The woman is frying a breaded pork chop. & Cooking  
A girl is flying a kite. & Recreation  
A man is riding a mechanical bull. & Entertainment  
The man is playing the guitar. & Music  
A woman is dancing and singing with other women. & Celebration  
A man is slicing a bun. & Cooking  
A man is pouring oil into a pan. & Cooking  
A lion is playing with people. & Dangerous  
A dog rides a skateboard. & Unusual  
Someone is carving a statue. & Art  
A woman is slicing an onion. & Cooking  
A woman is dancing. & Dancing  
Two green and white trains sitting on the tracks. & Arrangement  
A small white cat with glowing eyes standing underneath a chair. & Mysterious  
A large boat in the water at the marina. & Yacht  
a bus driving in a street. & Movement  
A passenger train waiting in a station. & Stationary  
a woman at a dinner table writing on her notebook. & Observation  
An Apple computer sitting on the floor. & Description  
A close-up of a brown horse’s head. & Detail  
A group of people eat at a table outside. & Alfresco  
A jockey riding a horse. & Equestrian  
The man is riding a motorcycle down the road. & Motorcycling  
A woman riding a brown horse. & Equestrian  
A kid jumping a ledge with a bike. & Stunt  
A black dog standing in front of yellow flowers. & Contrast  
Close up of a bottle of water. & Zoom  
A close up of a brown faced cat. & Intense  
sheep standing in afield. & Pastoral  
A longed-haired cat with it’s eyes closed. & Sleeping  
A woman in a gray shirt smiles for the camera while the woman behind her makes a face. & Contrast  
A silver and blue Amtrak train on the tracks near a small train station. & Railway  
A person in a blue shirt reclines near a coffee table and television. & Relaxation  
A black and white photo of a woman showing a horse. & Monochrome  
A dark brown horse standing in a field. & Equine  
A pitched tent with a horse in the background. & Camping  
A group of people sitting around a table with food on it. & Gathering  
A brown horse stands in a lush green field. & Pastoral  
a black and white cow in hay. & Cow  
An elderly woman stands in a kitchen with two cats at her feet. & Domesticity  
A school bus is driving uphill on a rural road. & Ascend  
Camouflage airplane sitting on grassy field. & Concealment  
Three young women standing in a room together. & Group  
Red double decker bus driving through the streets. & Transportation  
A white sheep on a hillside looking at the camera. & Observation  
A group of sheep in a field. & Flock  
A close-up, distorted photo of an empty glass Coke bottle. & Abstract  
Very crowded office desk with computer monitor on. & Cluttered  
A man sitting in a cluttered room. & Disorderly  
Two white cows in a green pasture. & Scene  
Black cow walking under trees in pasture. & Nature  
Two people sitting at a table at a restaurant. & Dining  
A smiling woman with a beer sitting outside with another smiling woman. & Companionship  
A bird holding on to a metal gate. & Perching  
The skinny cows are standing on the grass. & Cattle  
A women laying across two men sitting on a sofa. & Entanglement  
a woman with a big necklace. & Opulent  
Brown cow with horns standing in a field. & Cattle  
A cruise liner docked at the shoreline. & Berthed  
Black and white cat lying under bush. & Camouflage  
Brown and white cow standing in grass at side of road. & Cow  
A small dog looking up at the camera while standing on grass. & Adorable  
the process or result of becoming smaller or pressed together. & Contraction  
done, produced, or occurring once a week. & Weekly  
the chief bishop of an eparchy. & Eparch  
a native or inhabitant of guatemala, or a person of guatemalan descent. & Guatemalan  
the energy transmitted by radiation. & Radiation  
a necktie tied in a loose knot with two hanging ends, popular in the late 19th and early 20th centuries. & Four-in-hand  
relating to germany, its people, or their language. & German  
not yet used or soiled. & Fresh  
the chemical composition and properties of a substance or body. & Chemistry  
insects of the order Hemiptera; true bugs. & Hemiptera  
an act of counting something again, especially votes in an election. & Recount  
a very helpful or valuable event, person, or article. & Godsend  
the part of a theatre where the orchestra plays, typically in front of the stage and on a lower level. & Orchestra  
the eighth star in a constellation. & Theta  
abnormally low blood pressure. & Hypotension  
high-flown style; excessive use of verbal ornamentation. & Rhetoric  
impetuous or flamboyant vigour and confidence; panache. & Dash  
a large and densely populated urban area; may include several independent administrative districts. & Metropolis  
the side of an object that is opposite its front. & Backside  
an outward semblance that misrepresents the true nature of something. & Disguise  
the action of reasserting or confirming something. & Reaffirmation  
an idea or conclusion having general application. & Generalization  
the choicest or most essential or most vital part of some idea or experience. & Nub  
the way in which something is done or operated. & Mechanics  
relating to switzerland or its people. & Swiss  
an inhabitant of a particular town or city. & Citizen  
a compound present in some kinds of ergot. an alkaloid, it causes constriction of blood vessels and is used in the treatment of migraine. & Ergotamine  
the descendants of one individual. & Parentage  
things done to express interest in or please someone. & Attention  
the branch of technology that deals with dimensions and tolerances of less than 100 nanometres, especially the manipulation of individual atoms and molecules. & Nanotechnology  
a printed heading on stationery, stating a person or organization’s name and address. & Letterhead  
people who are destined to die soon. & Doomed  
the cross on which christ was crucified. & Cross  
a member of a sect. & Sectary  
an inanimate object worshipped for its supposed magical powers or because it is considered to be inhabited by a spirit. & Fetish  
denoting the offspring of a cross. & Filial  
create or prepare methodically. & Formulate  
a small old world songbird of the thrush family, with black, white, and brown coloration and a harsh call. & Chat  
make oneself thinner by dieting and sometimes exercising. & Slim  
head into a specified direction. & Make  
a white new zealander as opposed to a maori. & Pakeha  
a place of inviolable privacy. & Sanctum  
a person who has matriculated. & Matriculate  
agriculture developed along industrial lines. & Agro-industry  
a naval officer of the second most senior rank, above vice admiral and below admiral of the fleet or fleet admiral. & Admiral  
ease the grief or distress of. & Comfort  
come under, be classified or included. & Fall  
be a sign or indication of. & Denote  
the starting point for a new state or experience. & Threshold  
an instance of sleeping in rough accommodation or on an improvised bed. & Doss  
a writer of any of the hagiographa. & Hagiographer  
relating to or denoting a paraprofessional. & Paraprofessional  
intense and eager enjoyment, interest, or approval. & Enthusiasm  
kill and prepare for market or consumption. & Dress  
an unexpected and surprising event, especially an unpleasant one. & Bombshell  
obtain or seek to obtain by cadging or wheedling. & Scrounge  
a mechanical device consisting of a cylindrical tube around which the hair is wound to curl it. & Crimper  
an established ceremony prescribed by a religion. & Rite  
a continuous period of being seated, especially when engaged in a particular activity. & Sitting  
the cultivation of flowers. & Floriculture  
settle or establish firmly. & Cement  
meat from a deer. & Venison  
a deep red colour like that of burgundy wine. & Burgundy  
a temporary board fence erected round a building site. & Hoarding  
haunt like a ghost; pursue. & Obsess  
the quality of transparency or purity. & Clarity  
a push or blow, especially one given with the head. & Butt  
a standard or typical example. & Paradigm  
praise enthusiastically and publicly. & Acclaim  
pass through a hole or opening. & Reeve  
relating to or characteristic of java, a large island in the malay archipelago. & Javan  
a substance obtained by mining. & Mineral  
the solid part of a comet’s head. & Nucleus  
confine or restrain with or as if with manacles or handcuffs. & Manacle  
cause extensive destruction or ruin utterly. & Devastate  
a person being dealt with by social or medical services. & Client  
make or become very warm, especially through exposure to the heat of the sun or a fire. & Roast  
say something with difficulty, repeating the initial consonants of words. & Stutter  
a body of students who are taught together. & Class  
euphemistic expressions for death. & Release  
of or relating to or resembling fish. & Fishy  
the part of a sphere cut off by any plane not passing through the centre. & Segment  
a crossbar in front of a wagon with a swingletree at each end, enabling two horses to be harnessed. & Doubletree  
a strong blow with a knife or other sharp pointed instrument. & Thrust  
a shiny silicate mineral with a layered structure, found as minute scales in granite and other rocks, or as crystals. it is used as a thermal or electrical insulator. & Mica  
coins or other articles made of gold. & Gold  
living quarters provided for public convenience. & Accommodation  
unwillingness to do something contrary to your custom. & Loath  
move or cause to move gradually or with difficulty into another position. & Work  
move or sway in a rising and falling or wavelike pattern. & Fluctuate  
a flexible covering for the base of a gear lever or other mechanical part. & Gaiter  
done or existing alone. & Solitary  
of or relating to tutors or tutoring. & Tutorial  
come or be in close contact with; stick or hold together and resist separation. & Cling  
swell or cause to swell. & Belly  
relating to mongolia, its people, or their language. & Mongolian  
a longing or yearning. & Yen  
the sound made by the vibration of vocal folds modified by the resonance of the vocal tract. & Vocalisation  
the neurophysiological processes, including memory, by which an organism becomes aware of and interprets external stimuli. & Perception  
the process or action by which something is reabsorbed. & Resorption  
a public statement containing information about an event that has happened or is going to happen. & Promulgation  
in an advanced stage of pregnancy. & Heavy  
a smoky outdoor fire that is lit to keep off insects or protect plants against frost. & Smudge  
direct in spatial dimensions; proceeding without deviation or interruption; straight and short. & Direct  
a dead body, especially of a human being rather than an animal. & Corpse  
distinctive and stylish elegance. & Style  
a very typical example of a certain person or thing. & Archetype  
a person who replies to something, especially one supplying information for a questionnaire or responding to an advertisement. & Respondent  
the action of entering something. & Entry  
on the italian or roman side of the alps. & Ultramontane  
a projecting piece of wood made for insertion into a mortise in another piece. & Tenon  
a display of pretended or exaggerated suffering to obtain sympathy. & Martyrdom  
a malevolent spirit or person. & Cacodemon  
something or someone that causes anxiety; a source of unhappiness. & Vexation  
impose or inflict forcefully. & Clamp  
a long essay on a particular subject, especially one written for a university degree or diploma. & Dissertation  
be close or similar. & Approximate  
of uncertain outcome; especially fraught with risk. & Chancy  
the brotherhood of freemasons. & Craft  
a supporter of the american side during the war of american independence. & Whig  
a formal document giving notice of your intention to resign. & Resignation  
a device used in taxis that automatically records the distance travelled and the fare payable. & Taximeter  
any long object resembling a thin line. & Thread  
a set of reasons or a logical basis for a course of action or belief. & Rationale  
a person appointed to select a representative team in a sport. & Selector  
the manner in which someone behaves towards or deals with someone or something. & Treatment  
refuse to acknowledge someone or something as having authority. & Revolt  
a branch of an army assigned to a particular kind of work. & Corps  
an event resulting in great loss and misfortune. & Cataclysm  
occupy or take on. & Strike  
move with sweeping, effortless, gliding motions. & Sweep  
a high point, level, or figure. & High  
a large luxurious passenger ship of a type formerly used on a regular line. & Liner  
more distant than another object of the same kind. & Far  
the underground lair of a badger or fox. & Earth  
the central principle or part of a policy, system, etc., on which all else depends. & Keystone  
chequer with contrasting colours. & Counterchange  
the condition of being fenestrate. & Fenestration  
observe with care or pay close attention to. & Observe  
a dark greenish-blue colour. & Teal  
a mystic syllable, considered the most sacred mantra in hinduism and tibetan buddhism. it appears at the beginning and end of most sanskrit recitations, prayers, and texts. & Om  
set the level or character of. & Gear  
be sexually unfaithful to one’s partner in marriage. & Betray  
a round button for adjusting or controlling a machine. & Knob  
an army unit consisting of soldiers who fight on foot. & Foot  
people who are fearful and cautious. & Timid  
the trait of being excessively fastidious and easily shocked. & Squeamishness  
demand something forcefully, not accepting refusal. & Insist  
a secret word or phrase known only to a restricted group. & Word  
to compress with violence, out of natural shape or condition. & Squelch  
a salt containing the anion hco<sub>3</sub><sup>−</sup>. & Bicarbonate  
the length of time that a person has lived or a thing has existed. & Age  
used to indicate that one is waiting for an answer or explanation from someone. & Well  
a quantity or supply of something kept for use as needed. & Store  
a person or group that oppresses people. & Oppressor  
eject the contents of the stomach through the mouth. & Spue  
make a loud, high-pitched sound. & Scream  
objective or physical; not subjective. & Outer  
full of nervous energy, especially through taking amphetamines or similar drugs. & Amp  
an adhesive solution; gum or glue. & Mucilage  
a fastener consisting of two buttons joined with a bar, used in formal wear to fasten a shirt front or to fasten a collar to a shirt. & Stud  
the air passage from the throat to the lungs; the trachea. & Windpipe  
a curtain or piece of fabric fastened so as to hang in a drooping curve. & Swag  
rope that is used for fastening something to something else. & Lashing  
to say, state, or perform again. & Restate  
being complete of its kind and without defect or blemish. & Perfect  
creating a picture with paints. & Painting  
make amorous advances towards. & Solicit  
very beautiful or attractive. & Lovely  
filled with soft feathers. & Downy  
a high explosive consisting chiefly of a gel of nitroglycerine with added cellulose nitrate. & Gelatin  
the capacity to experience the sense of touch. & Feeling  
furnish with new or different furniture. & Refurnish  
remove from the centre of activity or attention; place in a less influential position. & Sideline  
rise up as in fear. & Uprise  
the celebration of something in a joyful and exuberant way. & Festivity  
stay or cause to stay at a certain value or level. & Hold  
to arouse hope, desire, or curiosity without satisfying them. & Tease  
liquid preparation having a soothing or antiseptic or medicinal action when applied to the skin. & Application  
change or be different within limits. & Run  
everything that exists anywhere. & Cosmos  
uncomfortably humid or airless. & Close  
a type of four-wheel-drive all-terrain military vehicle, or a similar vehicle intended for civilian use. & Hummer  
covered with or containing or consisting of ice. & Icy  
a caustic surface or curve. & Caustic  
the antibody which is involved in allergic reactions, causing the release of histamine when it combines with antigen in tissue, and capable of producing sensitivity to the antigen when introduced into the skin of a normal individual. & Reagin  
to prepare verbally, either for written or spoken delivery. & Prepare  
a building or community occupied by or consisting of friars. & Friary  
a preliminary round in a sporting competition. & Preliminary  
load or cover with stacks. & Stack  
a cavity in a plant, animal body, or organ. & Chamber  
a periodic variation of an electromagnetic field in the propagation of light or other radiation through a medium or vacuum. & Wave  
ornamentation by means of figures or designs. & Figuration  
make or place parallel to something. & Collimate  
be in accord; be in agreement. & Hold  
brush or drive away with a waving movement. & Fan  
vigorously energetic or forceful. & High-power  
an australian acacia tree with delicate fern-like leaves and yellow flowers. & Mimosa  
make hard or harder. & Harden  
a tropical old world plant of the daisy family, with large brightly coloured flowers, cultivated under glass in cooler regions. & Gerbera  
the round fruit of a tree of the rose family, which typically has thin green or red skin and crisp flesh. & Apple  

</div>

# Influence of Quantization

We analyze the influence of quantization in Table <a href="#tab:quantization" data-reference-type="ref" data-reference="tab:quantization">[tab:quantization]</a> between the 16bit models and 4bit models, which are quantized by bitsandbytes [2] with 4-bit normalfloat and double quantization. We find large models tend to show better results on STS tasks after 4-bit quantization. For example, PromptEOL+ICL with 6.7B OPT improve Spearman correlation from 79.08 to 79.38.

# Transfer Tasks

The results of PromptEOL with in-context learning (ICL) and contrastive learning (CSE) are shown in Table <a href="#fig:transfer_icl_cse" data-reference-type="ref" data-reference="fig:transfer_icl_cse">[fig:transfer_icl_cse]</a>. Compared to PromptEOL, both PromptEOL+ICL and PromptEOL+CSE appeared to hinder performance on transfer tasks. We anticipate that the incorporation of additional datasets, such as the Community QA dataset, in accordance with ST5 , or the implementation of full-model fine-tuning, might enhance the performance of PromptEOL+CSE in transfer tasks, which we leave in future. For PromptEOL+ICL, using STS-B or a dictionary as the example did not improve the performance on transfer tasks. We discover that using examples from a task with the label as the word in the example can improve the original performance. For instance, if we use one positive example and one negative example from training set of MR tasks, it increases the accuracy on MR in 6.7B OPT by approximately one point. We find these examples also beneficial to other transfer tasks, improving the average accuracy from 91.34 to 91.78, which can exceed 66B OPT performance.

# Sentence Representation Methods

We supplemented detail results in Table 1 and 2 for different sentence representation methods.

[1] † Corresponding Author.

[2] https://github.com/TimDettmers/bitsandbytes
