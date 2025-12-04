# Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing

## Abstract

Despite the impressive capabilities of Large Language Models (LLMs) on various tasks, they still struggle with scenarios that involves complex reasoning and planning. Self-correction and self-learning emerge as viable solutions, employing strategies that allow LLMs to refine their outputs and learn from self-assessed rewards. Yet, the efficacy of LLMs in self-refining its response, particularly in complex reasoning and planning task, remains dubious. In this paper, we introduce for the self-improvements of LLMs, which integrates Monte Carlo Tree Search (MCTS) with LLMs to establish a self-improving loop, thereby enhancing the capabilities of LLMs without additional annotations. Drawing inspiration from the success of AlphaGo, addresses the unique challenges of combining MCTS with LLM for self-improvement, including data scarcity, the vastness search spaces of language tasks, and the subjective nature of feedback in language tasks. is comprised of prompt synthesis component, an efficient MCTS approach tailored for language tasks, and a trio of critic models for precise feedback. Our experimental results in mathematical reasoning tasks demonstrate that significantly enhances the performance of LLMs without additional annotations, showing the potential for self-improvement in LLMs. The code is available at <https://github.com/YeTianJHU/AlphaLLM>.

# Introduction

Hello

<div id="tab:annotator-human">

<table>
<caption>Human Agreement Metrics Across Models (K = Cohen’s Kappa, J = Jaccard index, AR = Agreement Rate, AY = Agreement on Yes)</caption>
<thead>
<tr>
<th style="text-align: left;"><strong>Models</strong></th>
<th colspan="4" style="text-align: center;">Claude-3.5-Sonnet</th>
<th colspan="4" style="text-align: center;">Gemini-1.5-Pro</th>
<th colspan="4" style="text-align: center;">GPT-4o</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><strong>Category</strong></td>
<td style="text-align: center;"><strong>K</strong></td>
<td style="text-align: center;"><strong>J</strong></td>
<td style="text-align: center;"><strong>AR</strong></td>
<td style="text-align: center;"><strong>AY</strong></td>
<td style="text-align: center;"><strong>K</strong></td>
<td style="text-align: center;"><strong>J</strong></td>
<td style="text-align: center;"><strong>AR</strong></td>
<td style="text-align: center;"><strong>AY</strong></td>
<td style="text-align: center;"><strong>K</strong></td>
<td style="text-align: center;"><strong>J</strong></td>
<td style="text-align: center;"><strong>AR</strong></td>
<td style="text-align: center;"><strong>AY</strong></td>
</tr>
<tr>
<td style="text-align: left;">Anthropomorphization</td>
<td style="text-align: center;">0.75</td>
<td style="text-align: center;">0.68</td>
<td style="text-align: center;">0.91</td>
<td style="text-align: center;">0.72</td>
<td style="text-align: center;">0.64</td>
<td style="text-align: center;">0.61</td>
<td style="text-align: center;">0.83</td>
<td style="text-align: center;">0.96</td>
<td style="text-align: center;">0.69</td>
<td style="text-align: center;">0.65</td>
<td style="text-align: center;">0.86</td>
<td style="text-align: center;">0.96</td>
</tr>
<tr>
<td style="text-align: left;">User retention</td>
<td style="text-align: center;">0.62</td>
<td style="text-align: center;">0.73</td>
<td style="text-align: center;">0.81</td>
<td style="text-align: center;">0.76</td>
<td style="text-align: center;">0.72</td>
<td style="text-align: center;">0.84</td>
<td style="text-align: center;">0.88</td>
<td style="text-align: center;">0.96</td>
<td style="text-align: center;">0.66</td>
<td style="text-align: center;">0.81</td>
<td style="text-align: center;">0.85</td>
<td style="text-align: center;">0.95</td>
</tr>
<tr>
<td style="text-align: left;">Brand bias</td>
<td style="text-align: center;">0.49</td>
<td style="text-align: center;">0.40</td>
<td style="text-align: center;">0.88</td>
<td style="text-align: center;">0.59</td>
<td style="text-align: center;">0.49</td>
<td style="text-align: center;">0.40</td>
<td style="text-align: center;">0.86</td>
<td style="text-align: center;">0.69</td>
<td style="text-align: center;">0.44</td>
<td style="text-align: center;">0.38</td>
<td style="text-align: center;">0.79</td>
<td style="text-align: center;">0.90</td>
</tr>
<tr>
<td style="text-align: left;">Sycophancy</td>
<td style="text-align: center;">0.57</td>
<td style="text-align: center;">0.42</td>
<td style="text-align: center;">0.95</td>
<td style="text-align: center;">0.43</td>
<td style="text-align: center;">0.27</td>
<td style="text-align: center;">0.20</td>
<td style="text-align: center;">0.89</td>
<td style="text-align: center;">0.35</td>
<td style="text-align: center;">0.73</td>
<td style="text-align: center;">0.61</td>
<td style="text-align: center;">0.95</td>
<td style="text-align: center;">0.87</td>
</tr>
<tr>
<td style="text-align: left;">Harmful generation</td>
<td style="text-align: center;">0.98</td>
<td style="text-align: center;">0.98</td>
<td style="text-align: center;">0.99</td>
<td style="text-align: center;">0.99</td>
<td style="text-align: center;">0.90</td>
<td style="text-align: center;">0.90</td>
<td style="text-align: center;">0.95</td>
<td style="text-align: center;">0.91</td>
<td style="text-align: center;">0.96</td>
<td style="text-align: center;">0.96</td>
<td style="text-align: center;">0.98</td>
<td style="text-align: center;">1.00</td>
</tr>
<tr>
<td style="text-align: left;">Sneaking</td>
<td style="text-align: center;">0.56</td>
<td style="text-align: center;">0.65</td>
<td style="text-align: center;">0.78</td>
<td style="text-align: center;">0.76</td>
<td style="text-align: center;">0.46</td>
<td style="text-align: center;">0.64</td>
<td style="text-align: center;">0.74</td>
<td style="text-align: center;">0.90</td>
<td style="text-align: center;">0.42</td>
<td style="text-align: center;">0.64</td>
<td style="text-align: center;">0.72</td>
<td style="text-align: center;">0.95</td>
</tr>
<tr>
<td style="text-align: left;">Overall</td>
<td style="text-align: center;">0.75</td>
<td style="text-align: center;">0.71</td>
<td style="text-align: center;">0.89</td>
<td style="text-align: center;">0.79</td>
<td style="text-align: center;">0.70</td>
<td style="text-align: center;">0.69</td>
<td style="text-align: center;">0.86</td>
<td style="text-align: center;">0.90</td>
<td style="text-align: center;">0.71</td>
<td style="text-align: center;">0.71</td>
<td style="text-align: center;">0.86</td>
<td style="text-align: center;">0.96</td>
</tr>
</tbody>
</table>

</div>

# References

<div class="thebibliography">

David Abel, Dilip Arumugam, Lucas Lehnert, and Michael Littman State abstractions for lifelong reinforcement learning In *International Conference on Machine Learning*, pp. 10–19. PMLR, 2018. **Abstract:** In lifelong reinforcement learning, agents must effectively transfer knowledge across tasks while simultaneously addressing exploration, credit as- signment, and generalization. State abstraction can help overcome these hurdles by compressing the representation used by an agent, thereby re- ducing the computational and statistical burdens of learning. To this end, we here develop theory to compute and use state abstractions in lifelong reinforcement learning. We introduce two new classes of abstractions: (1) transitive state abstrac- tions, whose optimal form can be computed efﬁ- ciently, and (2) PAC state abstractions, which are guaranteed to hold with respect to a distribution of tasks. We show that the joint family of transi- tive PAC abstractions can be acquired efﬁciently, preserve near optimal-behavior, and experimen- tally reduce sample complexity in simple domains, thereby yielding a family of desirable abstractions for use in lifelong reinforcement learning. Along with these positive results, we show that there are pathological cases where state abstractions can negatively impact performance. (@abel2018state)

</div>

[^1]: Equal Contribution; †Corresponding Author
