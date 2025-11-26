# Algebraic Positional Encodings

## Abstract

We introduce a novel positional encoding strategy for Transformer-style models, addressing the shortcomings of existing, often *ad hoc*, approaches. Our framework provides a flexible mapping from the algebraic specification of a domain to an interpretation as orthogonal operators. This design preserves the algebraic characteristics of the source domain, ensuring that the model upholds its desired structural properties. Our scheme can accommodate various structures, including sequences, grids and trees, as well as their compositions. We conduct a series of experiments to demonstrate the practical applicability of our approach. Results suggest performance on par with or surpassing the current state-of-the-art, without hyper-parameter optimizations or “task search” of any kind. Code is available through <https://aalto-quml.github.io/ape/>.

# Introduction

Attention-based models inheriting from the Transformer `\citep{vaswani2017attention}`{=latex} have become ubiquitous in neural computation, supplanting the go-to models of the last decade and driving a continuous stream of breakthroughs across diverse domains. Their success is perhaps at odds with the Transformer’s structural lenience – its key building block, dot-product attention, is by default unable to perceive and utilize the structure and arrangement of the input/output tokens being processed. To address this limitation, a plethora of works have sought to endow Transformers with appropriate inductive biases. The most common strategy is to adjust token representations via so-called *positional encodings*; vector operations that hint at the structure being modeled. Nonetheless, most positional encoding schemes to date are either empirically motivated, or tailored to specific tasks. This renders their theoretical evaluation challenging, and hinders any prospects of a unifying framework.

In this study, we seek to fill this gap with a theory-first approach. Through the lens of group theory, we scrutinize some of the most commonly targeted data structures, and express them by means of inductive definitions that reveal and explicate their structural properties. Leveraging this analysis, our modeling strategy invokes a homomorphic interpretation that maps each domain into **algebraic positional encodings** (<span class="smallcaps">ape</span>): attention-compatible vector operations parameterizing (subgroups of) the orthogonal group. In the sequential context, algebraic positional encodings streamline the widely adopted rotary encodings of `\citet{su2023roformer}`{=latex}, while also offering clear theoretical insights on their success. More importantly, algebraic positional encodings naturally extend to non-sequential domains, such as \\(\kappa\\)-ary trees and multidimensional regular grids, paving the way for a simple and elegant methodology for interpretable and domain-general structurally-refined Transformers. We carry out an experimental evaluation in settings that allow for reproducible and statistically sound conclusions. Across the tasks considered, algebraic positional encodings consistently and significantly outperform strong baselines at an aggregate level, providing initial but compelling evidence that they constitute not just a *sensible meta-theory* for positional encodings, but also an *actionable alternative* to the current state of the art.

er

<div class="tabular" markdown="1">

l@ p0.95@  
  
  
  
  
§<a href="#sec:sequences" data-reference-type="ref" data-reference="sec:sequences">3.1</a> & *Sequences* are an isomorphism of the free group \\(\langle \ensuremath{\mathbb{1}}\rangle\\) (*i.e.*, the integers, \\(\mathbb{Z}\\)), and can be interpreted as a single generator subgroup of the orthogonal group \\(O(d)\\).  
§<a href="#sec:rope" data-reference-type="ref" data-reference="sec:rope">3.2</a> & *Rotary Positional Encodings* correspond to a (quite literally) special case of this interpretation: \\(SO(d)\\).  
§<a href="#sec:trees" data-reference-type="ref" data-reference="sec:trees">3.3</a> & *k-ary Trees* are an isomorphism of the finitely generated group \\(\langle \ensuremath{\mathbb{1}}, \ensuremath{\mathbb{2}}\dots \kappa \rangle\\), and can be interpreted as a finitely generated subgroup of \\(O(d)\\).  
§<a href="#sec:grids" data-reference-type="ref" data-reference="sec:grids">3.4</a> & *Regular Grids* are the group direct sum of multiple sequences. They can be interpreted as the matrix direct sum of their components’ interpretations.  
§<a href="#sec:variants" data-reference-type="ref" data-reference="sec:variants">3.5</a> & *Extensions* can be obtained in multiple directions.  
  
  
  
  
  
  

</div>

# Background [sec:background]

## The Problem with Dot-Product Attention [sec:attention]

All transformer variants employ some variation of the multi-head scaled dot-product attention mechanism of `\citet{vaswani2017attention}`{=latex}. For each attention head, the dot-product attention between queries \\({\bm{X}} \in \mathbb{R}^{m \times d}\\) and keys \\({\bm{Y}} \in \mathbb{R}^{n\times d}\\) is defined as: \\[\mathrm{atn}({\bm{X}}, {\bm{Y}}) :=
        \mathrm{softmax}_{(n)}\!
        \left(
            \frac{
                ({\bm{X}}{\bm{\Phi}}^{(q)})
                ({\bm{Y}}{\bm{\Phi}}^{(k)})^\top
            }{
                \sqrt{d}
            }
        \right)
        {\bm{Y}}{\bm{\Phi}}^{(v)}
       \label{eq:atn}\\] In equation (<a href="#eq:atn" data-reference-type="ref" data-reference="eq:atn">[eq:atn]</a>), matrices \\({\bm{\Phi}}^{(q)}, {\bm{\Phi}}^{(k)}, {\bm{\Phi}}^{(v)} : \mathbb{R}^{d\times d}\\) enact linear functions, applied point-wise (broadcasted) across all \\(m\\) and \\(n\\) entries of \\({\bm{X}}\\) and \\({\bm{Y}}\\). The dot-product term \\(({\bm{X}}{\bm{\Phi}}^{(q)})({\bm{Y}}{\bm{\Phi}}^{(k)})^\top\\) contains unnormalized attention scores in the Cartesian product of queries and keys. Unmodified, dot-product attention is permutation *invariant* with respect to its second argument; that is, for any arbitrary permutation \\(p_n \in \ensuremath{\mathcal{S}_{n}}\\): \\[\mathrm{atn}({\bm{X}}, {\bm{Y}}) \equiv \mathrm{atn}({\bm{X}}, p_n({\bm{Y}}))\\] Unless one is dealing with orderless structures like multisets or fully connected graphs, this property is generally undesirable. The lack of structural biases is typically counteracted by the component-wise addition of unidimensional periodic signals of varying frequencies. These, however, often prove inadequate in data-scarce domains, where extensive pretraining is impossible, and structure-rich domains, where a sequence-of-tokens projection is too radical of a simplification.

## Recap on Group Theory [sec:group]

To address this issue, we propose an algebraic treatment of positional encodings, based on principles lent from group theory. For the sake of convenience and accessibility, we provide a brief recap of the notions of interest here. A *group* \\(G\\) consists of a set of *elements* and a *binary operation* (\_\\(\cdot\\)\_) satisfying four fundamental laws:

- The group is *closed* under the the group operation. For all \\(a\\), \\(b\\) in \\(G\\), \\(a\cdot b\\) is also in \\(G\\).

- The group operation is *associative*. For all \\(a\\), \\(b\\), \\(c\\) in \\(G\\), \\((a\cdot b)\cdot c = a\cdot (b\cdot c)\\).

- The group operation has an *identity* element \\(e\\), such that for all \\(a\\) in \\(G\\), \\(a\cdot e = e\cdot a = a\\).

- Each group member has an *inverse*. For all \\(a\\) in \\(G\\), there exists some element \\(\overline{a}\\) such that \\(a\overline{a} = \overline{a}a = e\\), where \\(e\\) is the identity element.

A group is characterized as *finite* or *infinite* depending on the number of elements it has. If all elements of a group \\(G\\) can be expressed as a combination of a subset \\(S\\) of the group elements (combined by means of the group operation, applied either on the elements themselves or on their inverses), we write \\(G = \langle S \rangle\\). We say that \\(G\\) is *generated* by \\(S\\), and we call the elements of \\(S\\) the *generators* of \\(G\\). A group with a single generator is called *cyclic*.

# The Algebra(s) of Positions [sec:algebra]

Our objective is to establish a framework that offers general and extensible *semantics* for positions across various structures – what we commonly encounter in the literature as *positional encodings*. Most existing proposals adopt a rather parochial stance, relying on maneuvers or heuristics tailored to specific applications and driven, predominantly, by extensive empirical investigations. As such, they fall short with respect to accommodating or reflecting the properties of the underlying structure. In this work, we follow a different approach. We adopt Montague’s perspective, succinctly paraphrased as:

> “*syntax is an algebra, semantics is an algebra, and meaning is a homomorphism between them*” `\citep{janssen2014foundations}`{=latex}.

We begin by noting that “positions” do not exist in isolation, but only in the context of some underlying ambient structure. We contend that reasonable positional encodings (*semantics*) may only be reliably obtained by taking into account exactly this structure, its formation rules and properties (*syntax*), and then applying an appropriate interpretation (*meaning*). This is *not* just an academic exercise: a careful syntactic specification is a prerequisite if we aim for semantics that adhere to certain properties, which is arguably preferable to searching for these properties in the wild.

## Sequences [sec:sequences]

#### Syntax

We start from the simplest structure, and incidentally also the most standard one: the sequence. The full range of positions a token can occupy within a sequence coincides exactly with the naturals, \\(\mathbb{N}\\). Relative paths \\(\mathbb{P}\\) between any two positions can then be seen as the integers, \\(\mathbb{Z}\\), with positive (resp. negative) numbers denoting forward (resp. backward) offsets. Using this insight, it is handy to inspect how the standard inductive definition of the integers provides the building blocks for path formation. We start with two constants: the empty path (\\(\ensuremath{\mathbb{0}}\\)), which relates any given point to itself, and the unit path (\\(\ensuremath{\mathbb{1}}\\)), which relates any point to its immediate next. We may compose simple paths into complex ones with the aid of a binary operation \\(\mathbin +_{\ensuremath{\mathbb{P}}}\\). This already suffices to specify all forward offsets. In order to construct backward offsets, we need a unary operation \\((\mathop{-})_{\ensuremath{\mathbb{P}}}\\), such that \\(\mathop{-}\rho\\) denotes the inverse of \\(\rho\\). We can summarize the above by the grammar: \\[\begin{aligned}
    \ensuremath{\mathbb{P}}:= \ensuremath{\mathbb{0}}~ | ~ \ensuremath{\mathbb{1}}~ | ~ \ensuremath{\mathbb{P}}\mathbin +_{\ensuremath{\mathbb{P}}} \ensuremath{\mathbb{P}}~ | ~ \mathop{-}\ensuremath{\mathbb{P}}
    \label{def:spath}
\end{aligned}\\] For this to make sense, the operations must be *coherent*; that is, all ways to start from point \\(\rho_1\\) and end up in point \\(\rho_2\\) should be equivalent, even if apparently distinct. The needed equivalences exactly correspond to the group laws, with closure internalized by the inductive definition of (<a href="#def:spath" data-reference-type="ref" data-reference="def:spath">[def:spath]</a>): \\[\begin{aligned}
    (\rho_1 \mathbin +_{\ensuremath{\mathbb{P}}} \rho_2) \mathbin +_{\ensuremath{\mathbb{P}}} \rho_3 &= \rho_1 \mathbin +_{\ensuremath{\mathbb{P}}} (\rho_2 \mathbin +_{\ensuremath{\mathbb{P}}} \rho_3)
    \tag{L1}
    \label{prop:assoc_of_plus}\\
    \rho \mathbin +_{\ensuremath{\mathbb{P}}} \ensuremath{\mathbb{0}}&= \rho = \ensuremath{\mathbb{0}}\mathbin +\rho
    \tag{L2}
    \label{prop:id_of_plus}\\
    \rho \mathbin +_{\ensuremath{\mathbb{P}}} (\mathop{-}\rho )&= \ensuremath{\mathbb{0}}
    \tag{L3}
    \label{prop:def_of_inv}
\end{aligned}\\] The (unsurprising) insight here is that paths in a sequence form a free group, generated by a single generator (\\(\ensuremath{\mathbb{1}}\\)) – the uniqueness of the generator exceptionally also makes the group abelian (*i.e.*, commutative). For convenience, we adopt the notational shorthand \\(\ensuremath{\mathbb{1}}^p\\), where: \\[\ensuremath{\mathbb{1}}^{p} :=         \begin{cases}
        \underbrace{\ensuremath{\mathbb{1}}\mathbin +_{\ensuremath{\mathbb{P}}} \cdots \mathbin +_{\ensuremath{\mathbb{P}}} \ensuremath{\mathbb{1}}}_{p} & p\geq 0\\
        \underbrace{(\mathop{-}\ensuremath{\mathbb{1}})\mathbin +_{\ensuremath{\mathbb{P}}} \cdots \mathbin +_{\ensuremath{\mathbb{P}}} (\mathop{-}\ensuremath{\mathbb{1}})}_{-p}) & p<0 
    \end{cases}\\]

#### Semantics [sec:seq_semantics]

The syntactic specifications of the previous paragraph impose constraints on the candidate semantic targets. Among these candidates, we isolate and focus on \\(\langle {\bm{W}} \rangle\\), the subgroup of the orthogonal group \\(O(d)\\) that is generated by a single orthogonal matrix \\({\bm{W}}\\). This semantics is not only sound [^1] with respect to the structure under scrutiny, but also a familiar object in machine learning literature `\cite[\textit{inter alia}]{arjovsky2016unitary,bernardy2022unitary}`{=latex}. Note that for \\(\langle {\bm{W}} \rangle\\), the group axioms are obtained for free from the orthogonal group, and the additional requirement of commutativity is again satisfied by the uniqueness of the generator.

To illustrate the correspondence between the two structures (and at risk of being pedantic), we spell out the homomorphism \\(\ensuremath\lceil{.}\rceil\\), which maps paths \\(\ensuremath{\mathbb{P}}\\) to elements of \\(\langle {\bm{W}} \rangle\\), and path operations to operations on orthogonal matrices of size \\(d\\). For the primitives, we have \\(\ensuremath\lceil{\ensuremath{\mathbb{0}}}\rceil := {\bm{I}}_{d}\\) and \\(\ensuremath\lceil{\ensuremath{\mathbb{1}}}\rceil := {\bm{W}}\\). Path composition amounts to matrix multiplication, *i.e.*, \\(\ensuremath\lceil{\rho_1 \mathbin +_{\ensuremath{\mathbb{P}}} \rho_2}\rceil := \ensuremath\lceil{\rho_1}\rceil\ensuremath\lceil{\rho_2}\rceil\\), while path inversion corresponds to matrix transposition, *i.e.*, \\(\ensuremath\lceil{\mathop{-}\rho}\rceil := \ensuremath\lceil{\rho}\rceil^{-1} \equiv \ensuremath\lceil{\rho}\rceil^{\top}\\). The fact that orthogonal matrices form a group under multiplication is folklore; one can easily verify that the group laws hold also for the semantics. [^2]

#### Implementation [sec:seq-implementation]

In practice, we have \\(\ensuremath\lceil{\ensuremath{\mathbb{1}}^{p}}\rceil \mapsto {\bm{W}}^p\\); a norm-preserving bilinear form \\(\mathbb{R}^d \times \mathbb{R}^{d} \to \mathbb{R}\\) which can be used to mediate the dot-product between a query \\(q\\) and a key \\(k\\) offset by a relative distance of \\(p\\). The representation of all paths up to length \\(p\\) can thus be implemented as a matrix collection \\([{\bm{W}}^{0},\dots ,{\bm{W}}^{p}]\\), which can asymptotically be obtained using \\(\mathcal{O}(\lceil \mathrm{log}_2(p) \rceil)\\) matrix products (of exponentially larger matrices), and taking up the storage space equivalent of \\((pd^2)\\) floats. Transposed, the same matrices also serve to represent backwards paths \\([{\bm{W}}^{-p},\dots,{\bm{W}}^{0}]\\). Storing the representations of all relative paths between queries and keys in a tensor \\({\bm{T}} : \mathbb{R}^{m\times n\times d\times d}\\), we may then substitute the dot-product term of equation (<a href="#eq:atn" data-reference-type="ref" data-reference="eq:atn">[eq:atn]</a>) for the tensor contraction: \\[\sum_{\alpha,\beta} {\bm{X}}^{}_{m\alpha} {\bm{\Phi}}^{(q)}_{\alpha\beta} {\bm{T}}^{}_{mn\beta\gamma} {\bm{Y}}^{}_{n\delta}{\bm{\Phi}}^{(k)}_{\delta\gamma}
    \label{eq:bad}\\] Albeit transparent, this reduction strategy is computationally unappealing due to the doubly quadratic nature of \\({\bm{T}}\\). We can do better by noting that \\({\bm{T}}_{mn}\\) is (definitionally) equal to: \\[{\bm{T}}_{mn\alpha\beta} = \sum_{\gamma} {\bm{A}}^{(X)}_{m\gamma\alpha} {\bm{A}}^{(Y)}_{n\gamma\beta}\\] where \\({\bm{A}}^{(X)}\\) and \\({\bm{A}}^{(Y)}\\) are the matrices containing representations for the *absolute* positions of the entries in \\({\bm{X}}\\) and \\({\bm{Y}}\\), respectively. Concretely, a single relative representation is built by composing the *inverted* representation of the source with the representation of the target. Intuitively, each query follows the path that takes it *back* to the origin, which then allows it to directly combine with each forward-offset key; see Figure <a href="#fig:sequential" data-reference-type="ref" data-reference="fig:sequential">1</a> for a visual example. This insight allows us to keep the memory footprint of equation (<a href="#eq:atn" data-reference-type="ref" data-reference="eq:atn">[eq:atn]</a>) unchanged, replacing expression (<a href="#eq:bad" data-reference-type="ref" data-reference="eq:bad">[eq:bad]</a>) with: \\[\sum_{\alpha,\beta,\gamma,\delta,\epsilon}
        {\bm{X}}^{}_{m\alpha} {\bm{\Phi}}^{(q)}_{\alpha\beta} 
        {\bm{A}}^{(X)}_{m\gamma\beta}
        {\bm{A}}^{(Y)}_{n\gamma\delta}
        {\bm{Y}}^{}_{n\epsilon}
        {\bm{\Phi}}^{(k)}_{\epsilon\delta}
    \label{eq:good}\\] This version decomposes the tensor contraction into two matrix multiplications, essentially transforming (rotating or reflecting) the entries of \\({\bm{X}}\\) and \\({\bm{Y}}\\) independently according to their positions.

## Intermezzo: Equivalence with <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> [sec:rope]

The story so far should be reminiscent of the rotary positional encoding scheme of `\citet[\textsc{r}\textnormal{o}\textsc{pe}{}]{su2023roformer}`{=latex}. Not unlike our approach, <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> substitutes the vanilla dot-product for a position-dependent bilinear form. Underlying the form is a \\(d\times d\\)-dimensional matrix \\({\bm{R}}\\) with a block-diagonal structure, where each \\(2\times 2\\)-sized block corresponds to a rotation matrix that acts on a \\(2\\)-dimensional subspace of \\(\mathbb{R}^d\\). These independent rotations are parameterized by a (fixed) set of base angles \\(\Theta := [\theta_1, \dots,\theta_{d/2}]\\). To incorporate position-dependence, *i.e.*, for a query/key pair at a relative distance of \\(p\\), the base angles are multiplied by \\(p\\), effectively altering the rotations applied.

At first glance, rotary encodings appear to be under-parameterized, and thus strictly weaker than orthogonal ones. However, any orthogonal matrix \\({\bm{W}} \in O(d)\\) admits a canonical form \\({\bm{W}} = {\bm{P}}{\bm{Q}}{\bm{P}}^\top\\), where \\({\bm{P}}\\) is an orthogonal change of basis, and \\({\bm{Q}}\\) is block-diagonal, with the \\(2\times 2\\)-sized blocks being, once again, \\(2-\\)dimensional rotation matrices `\citep{murnaghan1931canonical}`{=latex} [^3]. Owing to the orthogonality of \\({\bm{P}}\\), raising \\({\bm{W}}\\) to its \\(p\\)th power is equal to \\({\bm{P}}{\bm{Q}}^p{\bm{P}}^\top\\) (*i.e.*, it leaves the change of basis unaffected). In turn, raising \\({\bm{Q}}\\) to its \\(p\\)th power is equivalent to simply multiplying the rotation angles of its blocks by \\(p\\). Finally, given the linearity of the transformations \\({\bm{\Phi}}^{(q)}\\) and \\({\bm{\Phi}}^{(k)}\\), their compositions with \\({\bm{P}}\\) are also linear. By identifying \\({\bm{Q}}\\) with <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>’s \\({\bm{R}}\\), we can then see that, for any given collection of angles \\(\Theta\\), <span class="smallcaps">ape</span> and <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> coincide under the substitutions: \\[\begin{array}{ccc}
         {\bm{\Phi}}^{(q)}_{\textnormal{\textsc{r}\textnormal{o}\textsc{pe}}} = {\bm{\Phi}}^{(q)}_{\vphantom{\textsc{r}\textnormal{o}\textsc{pe}}}  {\bm{Q}} & 
         \text{and} &
         {\bm{\Phi}}^{(k)}_{\textnormal{\textsc{r}\textnormal{o}\textsc{pe}}} = {\bm{\Phi}}^{(k)}_{\vphantom{\textsc{r}\textnormal{o}\textsc{pe}}}  {\bm{Q}}
    \end{array}\\] In other words, *<span class="smallcaps">ape</span> is practically equivalent to a *trainable* version of <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>, where the rotation angles \\(\mathit{\Theta}\\) may vary and be optimized during training* [^4].

Which of the two parameterizations is preferable is up to debate. On the one hand, <span class="smallcaps">ape</span>’s formulation is FLOP-optimized (being just matrix multiplications), and obviates the need for backpropagating through trigonometric functions (which are periodic, non-monotonic, and prone to gradient instabilities). On the other hand, <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>’s diagonalized form gives access to a memory-efficient contraction that does away with the matrix multiplications of expression (<a href="#eq:good" data-reference-type="ref" data-reference="eq:good">[eq:good]</a>) altogether; we direct the interested reader to `\citet[Section 3.4.2]{su2023roformer}`{=latex} for a reference implementation [^5].

In either case, the equivalence between the two is confined to the *sequential* setup; we will now move on to generalize our strategy to other, *previously inaccessible*, structures.

## Trees [sec:trees]

#### Syntax

In the previous section, we characterized the structure of relative paths on a sequence as the free group with one generator, and uncovered a (practically) isomorphic interpretation in the subgroup of orthogonal matrices with a single generator. Upon closer inspection, we note that a sequence can be viewed as a special case of the more general structure of \\(\kappa\\)-ary branching trees, where the branching factor \\(\kappa\\) just so happens to be 1. Denoting the more general case as \\(\ensuremath{\mathbb{P}}_{\kappa}\\), we must first extend the set of primitives to include all branching options, \\(\ensuremath{\mathbb{1}}, \ensuremath{\mathbb{2}}, \dots \mathbb{\kappa} : \ensuremath{\mathbb{P}}_{\kappa}\\). Each primitive now denotes a choice of branch (except for \\(\ensuremath{\mathbb{0}}\\), which is again the empty path). Paths now form a free group with \\(\kappa\\) distinct generators. The presence of multiple generators means that commutativity no longer holds; \\(\ensuremath{\mathbb{1}}\mathbin +_{\ensuremath{\mathbb{P}}_{\kappa}} \ensuremath{\mathbb{2}}\\) is distinct from \\(\ensuremath{\mathbb{2}}\mathbin +_{\ensuremath{\mathbb{P}}_\kappa} \ensuremath{\mathbb{1}}\\) (the former prescribes a descent down branch \\(\ensuremath{\mathbb{1}}\\) then branch \\(\ensuremath{\mathbb{2}}\\), whereas the latter prescribes a descent down branch \\(\ensuremath{\mathbb{2}}\\) then branch \\(\ensuremath{\mathbb{1}}\\)). Inversion is as before: for every path from each local root to some descendant down the line, there is also an inverse path from that descendant up to its ancestor. Perhaps more interestingly, upwards and downwards paths can be joined, allowing the precise specification of relative paths between any two nodes, even when the two do not share a single line of descent (think nephews, aunts and all other sorts of distant relatives, see Figure <a href="#fig:tree" data-reference-type="ref" data-reference="fig:tree">2</a> for an example). Adjusting grammar (<a href="#def:spath" data-reference-type="ref" data-reference="def:spath">[def:spath]</a>) accordingly, we have: \\[\begin{aligned}
    \ensuremath{\mathbb{P}}_{\kappa} := \ensuremath{\mathbb{0}}~ | ~ \ensuremath{\mathbb{1}}~ | ~ \ensuremath{\mathbb{2}}~ | \dots ~ | ~ \kappa ~ | ~ \ensuremath{\mathbb{P}}_{\kappa} \mathbin +_{\ensuremath{\mathbb{P}}_{\kappa}} \ensuremath{\mathbb{P}}_{\kappa} ~ | ~ \mathop{-}\ensuremath{\mathbb{P}}_{\kappa}
    \label{def:tpath}
\end{aligned}\\] with laws <a href="#prop:assoc_of_plus" data-reference-type="ref" data-reference="prop:assoc_of_plus">[prop:assoc_of_plus]</a>, <a href="#prop:id_of_plus" data-reference-type="ref" data-reference="prop:id_of_plus">[prop:id_of_plus]</a> and <a href="#prop:def_of_inv" data-reference-type="ref" data-reference="prop:def_of_inv">[prop:def_of_inv]</a> still in effect.

#### Semantics [semantics]

The interpretation follows along the same lines as before. This time around, however, we cannot make do with a single orthogonal matrix \\({\bm{W}}\\) – we need a collection of \\(\kappa\\) matrices, one for each branch option. As a consequence, the semantic target is now \\(\langle {\bm{W}}_1, {\bm{W}}_2, \dots {\bm{W}}_\kappa \rangle\\). Note that the target is no longer commutative (in alignment with the source).

#### Implementation [implementation]

For a tree structure of depth \\(\delta\\) and branching factor \\(\kappa\\), let \\(\nu\\) denote the number of *unique* absolute positions occupied (upper bound by \\(\kappa^\delta\\) in the case of a complete tree). Their representations can be computed in \\(\delta \kappa\\) steps of parallel matrix-matrix multiplications and a memory cost of \\(\nu{}d^2\\), as follows. First, we can build up a collection of all unique absolute paths, each represented as a (right-padded) word of length \\(\delta\\) from the vocabulary of primitives. Their corresponding representations constitute a tensor of size \\(\nu{}\times d \times d\\), initialized as \\(\nu\\) identity matrices. We can then iterate across these words in parallel, one primitive per step (*i.e.*, depth) \\(t\\), selecting all words that take the same branching direction at the current depth, and right-multiplying their representations by the corresponding orthogonal generator. Finally, absolute paths can be composed into relative ones using the modified dot-product attention of expression (<a href="#eq:good" data-reference-type="ref" data-reference="eq:good">[eq:good]</a>), just like before.

## Grids [sec:grids]

The generalization from sequences to trees rests on the observation that a sequence is a tree with a deficit of choices. An altogether different axis of generalization can be obtained by recalling that composite groups can be constructed by joining together two or more elementary groups. Moreover, if it just so happens that the original groups were abelian, then so is their composition; in that case, we call the composite a *group direct sum*. This construction provides access to an extension from sequences to multidimensional regular grids.

For the sake of simplicity and without loss of generality, we consider a standard instance of a two-dimensional grid: an image. An image is a collection of pixels (or pixel patches) that inhabit a coordinate system \\((h, w)\\). Each of \\(h\\) and \\(w\\) is the product of grammar (<a href="#def:spath" data-reference-type="ref" data-reference="def:spath">[def:spath]</a>), inheriting all path-related notions discussed earlier. Since \\(\ensuremath{\mathbb{P}}\\) is an abelian group, the coordinate system also constitutes an abelian group \\(\ensuremath{\mathbb{P}}^2 := \ensuremath{\mathbb{P}}\oplus \ensuremath{\mathbb{P}}\\). The new group and inversion operations are \\(\mathbin +_{\ensuremath{\mathbb{P}}^{2}}\\) and \\((\mathop{-})_{\ensuremath{\mathbb{P}}^2}\\), and denote the act of joining and inverting two-dimensional paths, respectively. Both are canonically defined component-wise, on the basis of their one-dimensional counterparts: \\[\begin{aligned}
    (x, y) \mathbin +_{\ensuremath{\mathbb{P}}^2} (z,w)    &:= (x \mathbin +_{\ensuremath{\mathbb{P}}} y, z\mathbin +_{\ensuremath{\mathbb{P}}} w)\\
    \mathop{-}(x,y)                  &:= (\mathop{-}x, \mathop{-}y)
\end{aligned}\\] with \\(\ensuremath{\mathbb{0}}^2 := (\ensuremath{\mathbb{0}}, \ensuremath{\mathbb{0}})\\) as the new neutral element. Intuitively, \\(\mathbin +_{\ensuremath{\mathbb{P}}^2}\\) corresponds to vector addition, and \\((\mathop{-})_{\ensuremath{\mathbb{P}}^2}\\) to a reflection about the origin with respect to both axes.

#### Semantics [semantics-1]

The specifications above allow us to reuse the notions from Section <a href="#sec:seq_semantics" data-reference-type="ref" data-reference="sec:seq_semantics">3.1.0.2</a> in order to interpret the components and operations of \\(\ensuremath{\mathbb{P}}^2\\). What is left unspecified is the interpretation of the group elements themselves; that is, we have yet to explicate what an object of \\(\ensuremath\lceil{\ensuremath{\mathbb{P}}\oplus \ensuremath{\mathbb{P}}}\rceil\\) looks like. The quest is a short one; the notion of a direct sum carries over to matrices, where it is defined as: \\[\begin{aligned}
 {\bm{A}} \oplus {\bm{B}} &:= 
    \begin{bmatrix}
        {\bm{A}} & {\bm{0}}\\
        {\bm{0}} & {\bm{B}} 
    \end{bmatrix}
\end{aligned}\\] From this, we get the (rather straightforward) interpretation \\(\ensuremath\lceil{(\rho_1, \rho_2)}\rceil \mapsto \ensuremath\lceil{\rho_1}\rceil \oplus \ensuremath\lceil{\rho_2}\rceil\\).

#### Implementation [implementation-1]

In practice, we now split the vector space in two independent parts. The first part is modulated by orthogonal matrices from \\(\langle {\bm{H}} \rangle\\), and the second part by orthogonal matrices from \\(\langle {\bm{W}} \rangle\\). For a query \\(q\\) and a key \\(k\\) that reside at a relative distance of \\((h, w)\\), their attention score is computed as \\(q({\bm{H}}^h \oplus {\bm{W}}^w)k\\) – see Figure <a href="#fig:grid" data-reference-type="ref" data-reference="fig:grid">3</a> for an illustration. Each axis contributes an additive but separable factor to the attention score, forcing the model to learn contextual alignments between token pairs on the basis of their coordinate-wise distances. Not much else is different: we can still compute all matrices in parallel, temporally bound by a logarithmic complexity of \\(\mathrm{log}_2(\mathrm{max}(h, w))\\) and \\(\mathrm{max}(h,w)(\frac{d}{2})^2\\) storage space, given a grid of size \\((h,w)\\). Subquadratic memory complexity can once more be achieved by virtue of diagonalization, just as in the sequential case.

<figure id="fig:images">
<figure id="fig:sequential">

<figcaption>The half-axis of absolute positions on a sequence, with a visualization of the two directions of relative paths between points 1 and 4. In either case, the interpretation is the matrix multiplication of the inverted source against the target.</figcaption>
</figure>
<figure id="fig:tree">

<figcaption>The space of paths on binary branching trees, with an illustration of the relative path from <span class="math inline">𝟚𝟙</span> to <span class="math inline">𝟙𝟚</span>. Same as before, the interpretation is the matrix multiplication of the inverted source against the target</figcaption>
</figure>
<figure id="fig:grid">

<figcaption>The quarter-plane of absolute positions on a 2-dimensional grid, with a visualization of the two directions of relative paths between points <span class="math inline">(3, 0)</span> and <span class="math inline">(1, 3)</span>. The interpretation is now a block-diagonal matrix consisting of the blocks interpreting the path over each coordinate.</figcaption>
</figure>
<figcaption>Example paths and their interpretations across the structures examined.</figcaption>
</figure>

## Variants & Extensions [sec:variants]

The structures that we have seen so far are not the only ones that our methodology can tackle – in fact, many other group-like structures are amenable to similar interpretations. We sketch out some enticing examples below.

#### Absolute Positions [sec:absolute]

Our analysis has so far focused on paths *relative* to positions. Fixing the point of origin allows a straightforward simplification to *absolute* positions. The new structure is that of a *monoid*: there’s no longer an inversion, and laws <a href="#prop:assoc_of_plus" data-reference-type="ref" data-reference="prop:assoc_of_plus">[prop:assoc_of_plus]</a> and <a href="#prop:id_of_plus" data-reference-type="ref" data-reference="prop:id_of_plus">[prop:id_of_plus]</a> only are now in effect. The framework remains largely unchanged: one can still use subgroups of matrices to represent positions, except this time applying them on either the queries or the keys (rather than both).

#### Periodic Domains [sec:periodic]

Under addition, the integers form an *infinite* cyclic group. An interesting twist would be to consider the positional encodings of *finite* cyclic groups instead. Such structures are not uncommon; in chemistry, for instance, a benzene molecule comprises six carbon atoms arranged in a ring. The semantics of such a structure would need to be of a matching period; that is, we would need a generator \\({\bm{W}}\\) such that \\({\bm{W}}^6 = {\bm{I}}\\). Such a parameterization is straightforward; we simply need to fix the orthogonal matrix so as to have it implement rotations at angle-multiples of \\(\pi / 3\\).

#### Time Series & Subsampling

Our sequential case analysis assumed a dense sequence with a uniform sampling rate. However, our strategy also applies to any series, even if sparsely sampled, as long as the sampling rate is quantized (*i.e.*, a multiple of some constant step). That is, positional indices (and their representations) do not need to match the placement of tokens in the sequence.

#### Composite Groups

The direct sum interpretation of Section <a href="#sec:grids" data-reference-type="ref" data-reference="sec:grids">3.4</a> is applicable for arbitrary groups that can be described as products, commutative or otherwise. This allows the representation of positional encodings for several other kinds of composite structures that can be concocted using the same principles, such as sequences of trees, trees of grids, etc.

#### Beyond Dot-Product Attention

Throughout the previous sections, we have adopted a dot-product formulation for the attention weight function. Nonetheless, <span class="smallcaps">ape</span> can be readily integrated into any other attention mechanism, such as linear `\citep{katharopoulos2020transformers}`{=latex}, cluster `\citep{vyas2020fast}`{=latex} and “softmax-free” `\citep{lu2021soft}`{=latex} variants, *inter alia*.

# Experiments [sec:experiments]

To assess the viability of our approach, we conduct a series of experiments across a range of tasks, in setups that allow for replicable and reliable comparisons with alternatives. When using <span class="smallcaps">ape</span>, we follow `\citet{wu2021transformer}`{=latex} in scaling the dot-product score between two tokens at a distance of \\(p\\) (*i.e.*, \\(p\\) steps away) by \\(p^c\\); here, we set \\(c := 0.98\\). This serves to stabilize training by introducing a locality bias (or long-distance decay) factor. For the sake of parameter compression, we share the orthogonal matrices between the different encoder/decoder layers, but use a distinct matrix (or collection of matrices) per head. To isolate and quantify the effect of initialization, we report results on two different initialization strategies: one where the orthogonal operators are set to mimic <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> rotations (default), and one where they are set to be close to the identity (no init). Similarly, to isolate and quantify the effect of trainability when comparing to <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>, we report results over both fixed (frozen) and trainable (tuned) rotation angles.

We provide an extensive account of our experimental setups in Appendix <a href="#appendix:esetup" data-reference-type="ref" data-reference="appendix:esetup">9</a>.

## Sequence Transduction [sec:strans]

#### Machine Translation

First, we follow `\citet{vaswani2017attention}`{=latex} in training a Transformer~<span class="smallcaps">base</span>~ model on machine translation over <span class="smallcaps">wmt14 en\\(\to\\)de</span> `\citep{bojar2014findings}`{=latex}.

To provide a comprehensive comparison, we pit our proposed methodology against standard positional encoding schemes from the literature: the vanilla *Sinusoidal* encodings of  `\citet{vaswani2017attention}`{=latex}, the *Absolute* encodings of `\citet{gehring2017convolutional}`{=latex}, the *Relative* encodings of `\citet{shaw2018self}`{=latex} and the *Rotary* encodings of `\citet{su2023roformer}`{=latex}. To ensure a fair comparison, we allow all models the exact same budgets (both memory and time).

#### Synthetic Tasks

We further examine three standard sequence transduction tasks: sequence copying, sequence reversal, and sequence repetition. These are meant to directly assess each model’s capacity for algorithmic induction, in setups where explicit position-based addressing, both absolute and relative, is required.

<div class="subtable" markdown="1">

1

er

<div class="tabularx" markdown="1">

@ l@ C C C C C C C @   &  
& *Sinusoidal* & *Absolute* & *Relative* & &  
& & & & *(frozen)* & *(tuned)* & *(/w init)* & *(w/o init)*  
<span class="smallcaps">wmt14 en\\(\to\\)de</span> & \\(14.57\\)<span style="color: gray!70">\\(\pm\\)\\(0.12\\)</span> & \\(22.09\\)<span style="color: gray!70">\\(\pm\\)\\(0.11\\)</span> & \\(23.15\\)<span style="color: gray!70">\\(\pm\\)\\(0.03\\)</span> & \\(\underline{\textcolor{red}{24.03}}\\)<span style="color: gray!70">\\(\pm\\)\\(0.06\\)</span> & \\(\underline{23.92}\\)<span style="color: gray!70">\\(\pm\\)\\(0.20\\)</span> & \\(\underline{23.93}\\)<span style="color: gray!70">\\(\pm\\)\\(0.10\\)</span> & \\({23.84}\\)<span style="color: gray!70">\\(\pm\\)\\(0.10\\)</span>  
  
<span class="smallcaps">Copy</span> & \\(1.01\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(1.11\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span>  
<span class="smallcaps">Repeat</span> & \\(1.85\\)<span style="color: gray!70">\\(\pm{}0.15\\)</span> & \\(3.66\\)<span style="color: gray!70">\\(\pm{}0.06\\)</span> & \\(1.44\\)<span style="color: gray!70">\\(\pm{}0.16\\)</span> & \\(\underline{1.08}\\)<span style="color: gray!70">\\(\pm{}0.12\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(1.02\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span>  
<span class="smallcaps">Reverse</span> & \\(3.92\\)<span style="color: gray!70">\\(\pm{}0.99\\)</span> & \\(4.62\\)<span style="color: gray!70">\\(\pm{}0.67\\)</span> & \\(4.08\\)<span style="color: gray!70">\\(\pm{}1.12\\)</span> & \\(1.09\\)<span style="color: gray!70">\\(\pm{}0.02\\)</span> & \\(\underline{\textcolor{red}{1.01}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.01}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{1.03}\\)<span style="color: gray!70">\\(\pm{}0.02\\)</span>  
  

</div>

</div>

<div class="subtable" markdown="1">

1.005

er

<div class="tabularx" markdown="1">

@ l@  c@ 0.5ptc@ 5pt c@ 0.5ptc@ 5pt c@ 0.5ptc@ 5pt c@ 2.5ptc@  &  
& & & &  
(l3ptr3pt)2-3 (l3ptr3pt)4-5 (l3ptr3pt)6-7 (l3ptr3pt)8-9 & breadth & depth & breadth & depth & breadth & depth & breadth & depth  
*Sinusoidal* & \\(1.06\\)<span style="color: gray!70">\\(\pm{}0.01\\)</span> & \\(5.68\\)<span style="color: gray!70">\\(\pm{}0.63\\)</span> & \\(6.93\\)<span style="color: gray!70">\\(\pm{}0.38\\)</span> & \\(7.13\\)<span style="color: gray!70">\\(\pm{}0.35\\)</span> & \\(2.66\\)<span style="color: gray!70">\\(\pm{}0.10\\)</span> & \\(2.78\\)<span style="color: gray!70">\\(\pm{}0.08\\)</span> & \\(20.53\\)<span style="color: gray!70">\\(\pm\\)\\(7.11\\)</span> & \\(64.86\\)<span style="color: gray!70">\\(\pm\\)\\(6.41\\)</span>  
*Tree-SQ* & \\(1.29\\)<span style="color: gray!70">\\(\pm{}0.01\\)</span> & \\(1.07\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(2.60\\)<span style="color: gray!70">\\(\pm{}0.16\\)</span> & \\(1.87\\)<span style="color: gray!70">\\(\pm{}0.24\\)</span> & \\(2.27\\)<span style="color: gray!70">\\(\pm{}0.59\\)</span> & \\(2.29\\)<span style="color: gray!70">\\(\pm{}0.24\\)</span> & \\(19.18\\)<span style="color: gray!70">\\(\pm\\)\\(3.23\\)</span> & \\(16.41\\)<span style="color: gray!70">\\(\pm\\)\\(6.14\\)</span>  
*Absolute* & \\(6.64\\)<span style="color: gray!70">\\(\pm{}0.12\\)</span> & \\(7.02\\)<span style="color: gray!70">\\(\pm{}0.17\\)</span> & \\(7.77\\)<span style="color: gray!70">\\(\pm{}0.15\\)</span> & \\(7.24\\)<span style="color: gray!70">\\(\pm{}0.20\\)</span> & \\(2.77\\)<span style="color: gray!70">\\(\pm{}0.21\\)</span> & \\(2.79\\)<span style="color: gray!70">\\(\pm{}0.22\\)</span> & \\(37.78\\)<span style="color: gray!70">\\(\pm\\)\\(0.72\\)</span> & \\(48.91\\)<span style="color: gray!70">\\(\pm\\)\\(5.83\\)</span>  
*Relative* & \\(1.01\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(6.12\\)<span style="color: gray!70">\\(\pm{}0.06\\)</span> & \\(6.00\\)<span style="color: gray!70">\\(\pm{}0.25\\)</span> & \\(7.72\\)<span style="color: gray!70">\\(\pm{}0.28\\)</span> & \\(1.70\\)<span style="color: gray!70">\\(\pm{}0.07\\)</span> & \\(2.43\\)<span style="color: gray!70">\\(\pm{}0.04\\)</span> & \\(2.36\\)<span style="color: gray!70">\\(\pm{}0.02\\)</span> & \\(16.86\\)<span style="color: gray!70">\\(\pm\\)\\(1.27\\)</span>  
*Rotary (frozen)* & \\(\underline{1.42}\\)<span style="color: gray!70">\\(\pm{}0.58\\)</span> & \\(2.46\\)<span style="color: gray!70">\\(\pm{}0.59\\)</span> & \\(4.58\\)<span style="color: gray!70">\\(\pm{}0.30\\)</span> & \\(4.97\\)<span style="color: gray!70">\\(\pm{}1.79\\)</span> & \\(1.55\\)<span style="color: gray!70">\\(\pm{}0.34\\)</span> & \\(2.15\\)<span style="color: gray!70">\\(\pm{}0.22\\)</span> & \\(2.53\\)<span style="color: gray!70">\\(\pm{}0.08\\)</span> & \\(33.54\\)<span style="color: gray!70">\\(\pm\\)\\(9.04\\)</span>  
*Rotary (tuned)* & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(1.70\\)<span style="color: gray!70">\\(\pm{}0.05\\)</span> & \\(4.07\\)<span style="color: gray!70">\\(\pm{}0.34\\)</span> & \\(2.60\\)<span style="color: gray!70">\\(\pm{}0.11\\)</span> & \\(1.08\\)<span style="color: gray!70">\\(\pm{}0.02\\)</span> & \\(1.90\\)<span style="color: gray!70">\\(\pm{}0.22\\)</span> & \\(2.55\\)<span style="color: gray!70">\\(\pm{}0.05\\)</span> & \\(20.87\\)<span style="color: gray!70">\\(\pm\\)\\(0.33\\)</span>  
*Algebraic (**seq**)* & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(1.63\\)<span style="color: gray!70">\\(\pm{}0.06\\)</span> & \\(2.95\\)<span style="color: gray!70">\\(\pm{}0.08\\)</span> & \\(2.48\\)<span style="color: gray!70">\\(\pm{}0.27\\)</span> & \\(1.07\\)<span style="color: gray!70">\\(\pm{}0.01\\)</span> & \\(1.83\\)<span style="color: gray!70">\\(\pm{}0.02\\)</span> & \\(2.30\\)<span style="color: gray!70">\\(\pm{}0.03\\)</span> & \\(20.05\\)<span style="color: gray!70">\\(\pm\\)\\(0.36\\)</span>  
*w/o init* & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(2.36\\)<span style="color: gray!70">\\(\pm{}0.63\\)</span> & \\(5.18\\)<span style="color: gray!70">\\(\pm{}0.10\\)</span> & \\(5.72\\)<span style="color: gray!70">\\(\pm{}1.23\\)</span> & \\(1.45\\)<span style="color: gray!70">\\(\pm{}0.08\\)</span> & \\(2.29\\)<span style="color: gray!70">\\(\pm{}0.06\\)</span> & \\(\underline{\textcolor{red}{1.75}}\\)<span style="color: gray!70">\\(\pm{}0.74\\)</span> & \\(29.26\\)<span style="color: gray!70">\\(\pm\\)\\(9.15\\)</span>  
*Algebraic (**tree**)* & \\(1.01\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.05}}\\)<span style="color: gray!70">\\(\pm{}0.01\\)</span> & \\(\underline{\textcolor{red}{1.01}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(2.24\\)<span style="color: gray!70">\\(\pm{}0.06\\)</span> & \\(\underline{\textcolor{red}{1.83}}\\)<span style="color: gray!70">\\(\pm{}0.02\\)</span>  
*w/o init* & \\(1.07\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(\underline{1.04}\\)<span style="color: gray!70">\\(\pm{}0.08\\)</span> & \\(1.44\\)<span style="color: gray!70">\\(\pm{}0.15\\)</span> & \\(1.27\\)<span style="color: gray!70">\\(\pm{}0.15\\)</span> & \\(\underline{1.05}\\)<span style="color: gray!70">\\(\pm{}0.10\\)</span> & \\(\underline{\textcolor{red}{1.00}}\\)<span style="color: gray!70">\\(\pm{}0.00\\)</span> & \\(2.42\\)<span style="color: gray!70">\\(\pm{}0.01\\)</span> & \\(1.86\\)<span style="color: gray!70">\\(\pm{}0.01\\)</span>  
  

</div>

</div>

<div class="subtable" markdown="1">

1 er

<div class="tabular" markdown="1">

@ lcc@  &  
Scheme & &  
 *Sinusoidal 2D* & \\(91.57\\)<span style="color: gray!70">\\(\pm{}0.01\\)</span> & \\(92.79\\)<span style="color: gray!70">\\(\pm{}0.20\\)</span>  
 *Absolute* & \\(90.86\\)<span style="color: gray!70">\\(\pm{}0.19\\)</span> & \\(92.68\\)<span style="color: gray!70">\\(\pm{}0.39\\)</span>  
 *Algebraic (**seq**)* & \\(92.68\\)<span style="color: gray!70">\\(\pm{}0.24\\)</span> & \\(\underline{94.59}\\)<span style="color: gray!70">\\(\pm{}0.15\\)</span>  
  *w/o init* & \\(88.93\\)<span style="color: gray!70">\\(\pm{}0.19\\)</span> & \\(91.09\\)<span style="color: gray!70">\\(\pm{}0.20\\)</span>  
 *Algebraic (**grid**)* & \\(\underline{\textcolor{red}{93.13}}\\)<span style="color: gray!70">\\(\pm{}0.33\\)</span> & \\(\underline{\textcolor{red}{94.67}}\\)<span style="color: gray!70">\\(\pm{}0.06\\)</span>  
  *w/o init* & \\(92.95\\)<span style="color: gray!70">\\(\pm{}0.07\\)</span> & \\(94.48\\)<span style="color: gray!70">\\(\pm{}0.18\\)</span>  
  

</div>

</div>

## Tree Transduction [sec:treetrans]

Next, we consider four algorithmic transduction tasks on binary branching trees: tree copying, recursive tree rotation up to a fixpoint, algebraic reduction of C~3~ expressions, and self-referential tree manipulation; see Appendix <a href="#appendix:esetup" data-reference-type="ref" data-reference="appendix:esetup">9</a> for details.

In addition to previous sequential baselines, we compare our model to the encodings of `\citet[\textit{Tree-SQ}]{shiv2019novel}`{=latex}. For all four tasks, we experiment with both breadth-first and depth-first decoding.

## Image Recognition [sec:imagerec]

Finaly, we train a Compact Convolutional Transformer `\citep{hassani2021escaping}`{=latex} on <span class="smallcaps">cifar</span>-10 `\citep{krizhevsky2009learning}`{=latex}.

Typically, attention-based architectures for vision rely on additive positional encoding schemes, applied on the image prior to it being sequentialized (row-by-row flattened). Here, we compare fixed `\cite[\textit{Sinusoidal 2D}]{wang2019translating}`{=latex} and parametric `\cite[\textit{Absolute}]{gehring2017convolutional}`{=latex} variants of the above against both the sequential and the grid-structured versions of our scheme.

## Results [sec:results]

We repeat each experiment three times, varying the seeds used for weight initialization and optimization, but fixing the data across repetitions. We report means and 95% CIs in Table <a href="#table:results" data-reference-type="ref" data-reference="table:results">[table:results]</a>. We highlight each category’s best (in red), and underline scores where the CI spans the mean of the respective best.

At the macro level and consistently across modalities, domain-appropriate algebraic interpretations match or surpass strong and specialized baselines – without *any* hyper-parameter tuning or search. Specifically, across the 13 setups considered, <span class="smallcaps">ape</span> is the uncontested top performer in 8, ranks among the best in 3, and falls within the confidence margin of the top performer in one. Exceptionally, in the breadth-first version of the tree-copy task, tree algebraic encodings are surpassed by a handful of sequential alternatives; this is no surprise, since in this case the tree structure is practically a task-irrelevant syntactic confound. Perhaps more surprisingly, in the breadth-first version of the tree-manipulation task, tree algebraic encodings are surpassed only by their non-initialized, sequential version; this is likely a statistical anomaly, since one of the three repetitions resulted in an unusually low perplexity score.

We also note three general trends. First, initializing <span class="smallcaps">ape</span> to match <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> frequency bands at the start of training consistently and significantly improves performance, possibly because <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> rotary primitives have undergone empirical tuning for stability and performance. Second, given identical initialization, a sequential <span class="smallcaps">ape</span> generally outperforms a trainable <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>, despite their theoretical equivalence. This might be due to the difficulty of optimizing periodic signals (*i.e.*, <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>’s trigonometric functions) compared to <span class="smallcaps">ape</span>’s (orthogonal) matrix multiplications. Third, a frozen <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> performs comparably to a randomly initialized <span class="smallcaps">ape</span> in most tasks considered, suggesting that adjusting rotoreflection angles during training is not necessarily better than adjusting rotation planes while keeping the angles fixed. Contrary to all the above, a frozen <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> weakly outperforms both a tunable <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> and an initialized <span class="smallcaps">ape</span> in the neural machine translation task; likely an artifact of attention overfitting to specific positional patterns.

# Related Work [sec:relwork]

Dense attention is by now a foundational component of various problem- and domain-general architectures. Combined with its structural indifference, this underscores the pressing need for learning strategies capable of injecting structural biases directly at the representation level. As such, positional encodings have garnered significant community attention in recent years – too much, in fact, to permit an exhaustive enumeration here. An extensive survey and meta-review is provided by `\citet{dufter-etal-2022-position}`{=latex} who group and rank these works on the basis of several criteria. Our work presents a universal, intuitive and formally grounded recipe that meets *all* these criteria: it is *trainable*, amenable to problem-specific and data-driven tuning; *reference-adjustable*, allowing both absolute and relative positional specifications; *unbounded*, capable of representing enumerably infinite positions irrespective of model instantiation and/or the targeted data size; *contextual*, implementing a dynamic effect that varies depending on token content; *effective*, consistently matching or surpassing baselines in the tasks considered; and, finally, *efficient*, exhibiting generally favorable asymptotic complexities.

We must point out that the concept of positional encodings as sequence homomorphisms has already been hinted at, first by `\citet{wang2020encoding}`{=latex} and later by `\citet{su2023roformer}`{=latex}, even if not explicitly formulated as such. Despite approaching the problem from different angles, both approaches interpret positions as multiplicative, norm-preserving (rotation-like) operations. Our proposal expands upon these two, first in providing a proper algebraic framing of the problem, and second in extending the interpretation from rotations around the axes to rotations and reflections about arbitrary planes. In the case of a single generator matrix (*i.e.*, sequences), this difference turns to be non-essential, being practically neutralized by the Transformer’s trainable weights. This no longer holds, however, in the case of multiple generator matrices (*i.e.*, grids or trees), where each generator should be able to rotate and reflect different sets of planes. In that sense, algebraic positional encodings offer an appealing unifying perspective of a multidimensional generalization to the aforementioned rotation-based frameworks. This sentiment is shared by `\citet{lim2023positional}`{=latex} who, in parallel to our work, similarly advocate for positional encodings as group homomorphisms, there framed as irreducible group representations. Modulo presentation, the two approaches are variations on a common theme; theirs is technically concerned with post-hoc representation of symmetries and equivariances at a per-datum scale, whereas ours focuses on the interpretation of domain signatures at the dataset scale.

More generally, algebraic manipulations are not uncommon in modern machine learning literature. The recognition of abstract algebra as a practical tool for imposing structural well-behavedness has led to its increased adoption as a reliable recipe for structure-informed neural architectures, largely obsoleting the inefficient and ad-hoc augmentation routines of the past. This line of work can be traced back to the group equivariant convolutions of `\citet{cohen2016group}`{=latex}, which have by now bloomed into a field of their own; see `\citet{weiler2023equivariant}`{=latex} for an up-to-date overview.

# Limitations [sec:limitations]

We recognize weaknesses and limitations across three fronts. On the *theoretical* front, we have limited our scope to simple inductive groups, consciously ignoring potential interpretations of more complex constructions. We defer this to future work. On the *empirical* front, having to recompute positional encodings once per batch increases a model’s temporal complexity during training. While this is barely noticeable in sequential and grid constructions, which scale logarithmically, it becomes evident when dealing with complete trees, which scale linearly and require explicit for-loops. On the *epistemic* front, we conducted a limited set of experiments, focusing primarily on replicability and fairness. We leave more exhaustive empirical comparisons on practical downstream tasks to future work or interested parties.

# Conclusion

We have presented a theoretically motivated approach towards constructing positional encodings for a variety of structures. Without any significant modification or overhead, our methodology can capture sequences and their (multi-dimensional as well as multi-branching) generalizations. In doing so, it reconciles powerful but structurally oblivious models with their missing inductive biases, permitting structure-aware architectural refinements across a range of tasks and setups (see also `\citet{kogkalidis_learning_2024}`{=latex} for parallel work employing the methodology in a neurosymbolic representation learning setup). Beyond that, our approach grants full control over how these biases are to be implemented, while also being amenable to adjustments and extensions. Our work indicates that generality and extensibility are not *in spite of*, but rather *due to* structural discipline and abstraction. We perceive it as an important step towards data-efficient, general and transparent models of neural computation.

KK and VG were supported by Saab-WASP via the project “Neurodynamic Programming and Reinforcement Learning” (grant 411025). VG also acknowledges the support from Academy of Finland (grant 342077) for “Human-steered next-generation machine learning for reviving drug design”, and the Jane and Aatos Erkko Foundation (grant 7001703) for “Biodesign: Use of artificial intelligence in enzyme design for synthetic biology”.

# References [references]

<div class="thebibliography" markdown="1">

M. Arjovsky, A. Shah, and Y. Bengio Unitary evolution recurrent neural networks In *International conference on machine learning*, pages 1120–1128. PMLR, 2016. **Abstract:** Recurrent neural networks (RNNs) are notoriously difficult to train. When the eigenvalues of the hidden to hidden weight matrix deviate from absolute value 1, optimization becomes difficult due to the well studied issue of vanishing and exploding gradients, especially when trying to learn long-term dependencies. To circumvent this problem, we propose a new architecture that learns a unitary weight matrix, with eigenvalues of absolute value exactly 1. The challenge we address is that of parametrizing unitary matrices in a way that does not require expensive computations (such as eigendecomposition) after each weight update. We construct an expressive unitary weight matrix by composing several structured matrices that act as building blocks with parameters to be learned. Optimization with this parameterization becomes feasible only when considering hidden states in the complex domain. We demonstrate the potential of this architecture by achieving state of the art results in several hard tasks involving very long-term dependencies. (@arjovsky2016unitary)

J.-P. Bernardy and S. Lappin Unitary recurrent networks: Algebraic and linear structures for syntax In *Algebraic Structures in Natural Language*, pages 243–278. CRC Press, 2022. **Abstract:** The emergence of powerful deep learning systems has largely displaced classical symbolic algebraic models of linguistic representation in computational linguistics. While deep neural networks have achieved impressive results across a wide variety of AI and NLP tasks, they have become increasingly opaque and inaccessible to a clear understanding of how they acquire the generalisations that they extract from the data to which they apply. This is particularly true of BERT, and similar non-directional transformers. We study an alternative deep learning system, Unitary-Evolution Recurrent Neural Networks (URNs) (Arjovsky et al., 2016), which are strictly compositional in their combination of state matrices. As a result they are fully transparent. They can be understood entirely in terms of the linear algebraic operations that they apply to each input state matrix to obtain its output successor. We review these operations in some detail, clarifying the compositional nature of URNs. We then present experimental evidence from three NLP tasks to show that these models achieve an encouraging level of precision in handling long distance dependencies. The learning required to solve these tasks involves acquiring and representing complex hierarchical syntactic structures. (@bernardy2022unitary)

O. Bojar, C. Buck, C. Federmann, B. Haddow, P. Koehn, J. Leveling, C. Monz, P. Pecina, M. Post, H. Saint-Amand, et al Findings of the 2014 workshop on statistical machine translation. In *Proceedings of the ninth workshop on statistical machine translation*, pages 12–58, 2014. **Abstract:** We propose and study a set of algorithms for discovering community structure in networks-natural divisions of network nodes into densely connected subgroups. Our algorithms all share two definitive features: first, they involve iterative removal of edges from the network to split it into communities, the edges removed being identified using any one of a number of possible "betweenness" measures, and second, these measures are, crucially, recalculated after each removal. We also propose a measure for the strength of the community structure found by our algorithms, which gives us an objective metric for choosing the number of communities into which a network should be divided. We demonstrate that our algorithms are highly effective at discovering community structure in both computer-generated and real-world network data, and show how they can be used to shed light on the sometimes dauntingly complex structure of networked systems. (@bojar2014findings)

T. Cohen and M. Welling Group equivariant convolutional networks In *International conference on machine learning*, pages 2990–2999. PMLR, 2016. **Abstract:** We introduce Group equivariant Convolutional Neural Networks (G-CNNs), a natural generalization of convolutional neural networks that reduces sample complexity by exploiting symmetries. G-CNNs use G-convolutions, a new type of layer that enjoys a substantially higher degree of weight sharing than regular convolution layers. G-convolutions increase the expressive capacity of the network without increasing the number of parameters. Group convolution layers are easy to use and can be implemented with negligible computational overhead for discrete groups generated by translations, reflections and rotations. G-CNNs achieve state of the art results on CI- FAR10 and rotated MNIST. (@cohen2016group)

P. Dufter, M. Schmitt, and H. Schütze Position information in transformers: An overview *Computational Linguistics*, 48 (3): 733–763, Sept. 2022. . URL <https://aclanthology.org/2022.cl-3.7>. **Abstract:** Abstract Transformers are arguably the main workhorse in recent natural language processing research. By definition, a Transformer is invariant with respect to reordering of the input. However, language is inherently sequential and word order is essential to the semantics and syntax of an utterance. In this article, we provide an overview and theoretical comparison of existing methods to incorporate position information into Transformer models. The objectives of this survey are to (1) showcase that position information in Transformer is a vibrant and extensive research area; (2) enable the reader to compare existing methods by providing a unified notation and systematization of different approaches along important model dimensions; (3) indicate what characteristics of an application should be taken into account when selecting a position encoding; and (4) provide stimuli for future research. (@dufter-etal-2022-position)

P. Gage A new algorithm for data compression *The C Users Journal*, 12 (2): 23–38, 1994. **Abstract:** People tend to store a lot of files inside theirs storage. When the storage nears it limit, they then try to reduce those files size to minimum by using data compression software. In this paper we propose a new algorithm for data compression, called j-bit encoding (JBE). This algorithm will manipulates each bit of data inside file to minimize the size without losing any data after decoding which is classified to lossless compression. This basic algorithm is intended to be combining with other data compression algorithms to optimize the compression ratio. The performance of this algorithm is measured by comparing combination of different data compression algorithms. (@gage1994new)

J. Gehring, M. Auli, D. Grangier, D. Yarats, and Y. N. Dauphin Convolutional sequence to sequence learning In *International conference on machine learning*, pages 1243–1252. PMLR, 2017. **Abstract:** The prevalent approach to sequence to sequence learning maps an input sequence to a variable length output sequence via recurrent neural networks. We introduce an architecture based entirely on convolutional neural networks. Compared to recurrent models, computations over all elements can be fully parallelized during training and optimization is easier since the number of non-linearities is fixed and independent of the input length. Our use of gated linear units eases gradient propagation and we equip each decoder layer with a separate attention module. We outperform the accuracy of the deep LSTM setup of Wu et al. (2016) on both WMT’14 English-German and WMT’14 English-French translation at an order of magnitude faster speed, both on GPU and CPU. (@gehring2017convolutional)

A. Hassani, S. Walton, N. Shah, A. Abuduweili, J. Li, and H. Shi Escaping the big data paradigm with compact transformers *arXiv preprint arXiv:2104.05704*, 2021. **Abstract:** With the rise of Transformers as the standard for language processing, and their advancements in computer vision, there has been a corresponding growth in parameter size and amounts of training data. Many have come to believe that because of this, transformers are not suitable for small sets of data. This trend leads to concerns such as: limited availability of data in certain scientific domains and the exclusion of those with limited resource from research in the field. In this paper, we aim to present an approach for small-scale learning by introducing Compact Transformers. We show for the first time that with the right size, convolutional tokenization, transformers can avoid overfitting and outperform state-of-the-art CNNs on small datasets. Our models are flexible in terms of model size, and can have as little as 0.28M parameters while achieving competitive results. Our best model can reach 98% accuracy when training from scratch on CIFAR-10 with only 3.7M parameters, which is a significant improvement in data-efficiency over previous Transformer based models being over 10x smaller than other transformers and is 15% the size of ResNet50 while achieving similar performance. CCT also outperforms many modern CNN based approaches, and even some recent NAS-based approaches. Additionally, we obtain a new SOTA result on Flowers-102 with 99.76% top-1 accuracy, and improve upon the existing baseline on ImageNet (82.71% accuracy with 29% as many parameters as ViT), as well as NLP tasks. Our simple and compact design for transformers makes them more feasible to study for those with limited computing resources and/or dealing with small datasets, while extending existing research efforts in data efficient transformers. Our code and pre-trained models are publicly available at https://github.com/SHI-Labs/Compact-Transformers. (@hassani2021escaping)

T. Janssen *Foundations and applications of Montague grammar* PhD thesis, University of Amsterdam, 2014. Originally published: April 1983 (UvA). (@janssen2014foundations)

A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret Transformers are rnns: Fast autoregressive transformers with linear attention In *International conference on machine learning*, pages 5156–5165. PMLR, 2020. **Abstract:** Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input’s length, they are prohibitively slow for very long sequences. To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from $\\}mathcal{O}\\}left(N\^2\\}right)$ to $\\}mathcal{O}\\}left(N\\}right)$, where $N$ is the sequence length. We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. Our linear transformers achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences. (@katharopoulos2020transformers)

K. Kogkalidis, O. Melkonian, and J.-P. Bernardy Learning structure-aware representations of dependent types In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. **Abstract:** Agda is a dependently-typed programming language and a proof assistant, pivotal in proof formalization and programming language theory. This paper extends the Agda ecosystem into machine learning territory, and, vice versa, makes Agda-related resources available to machine learning practitioners. We introduce and release a novel dataset of Agda program-proofs that is elaborate and extensive enough to support various machine learning applications – the first of its kind. Leveraging the dataset’s ultra-high resolution, detailing proof states at the sub-type level, we propose a novel neural architecture targeted at faithfully representing dependently-typed programs on the basis of structural rather than nominal principles. We instantiate and evaluate our architecture in a premise selection setup, where it achieves strong initial results. (@kogkalidis_learning_2024)

A. Krizhevsky, G. Hinton, et al Learning multiple layers of features from tiny images . **Abstract:** In this work we describe how to train a multi-layer generative model of natural images. We use a dataset of millions of tiny colour images, described in the next section. This has been attempted by several groups but without success. The models on which we focus are RBMs (Restricted Boltzmann Machines) and DBNs (Deep Belief Networks). These models learn interesting-looking filters, which we show are more useful to a classifier than the raw pixels. We train the classifier on a labeled subset that we have collected and call the CIFAR-10 dataset. (@krizhevsky2009learning)

Y. Li, R. Zemel, M. Brockschmidt, and D. Tarlow Gated graph sequence neural networks In *Proceedings of ICLR’16*, 2016. **Abstract:** Graph-structured data appears frequently in domains including chemistry, natural language semantics, social networks, and knowledge bases. In this work, we study feature learning techniques for graph-structured inputs. Our starting point is previous work on Graph Neural Networks (Scarselli et al., 2009), which we modify to use gated recurrent units and modern optimization techniques and then extend to output sequences. The result is a flexible and broadly useful class of neural network models that has favorable inductive biases relative to purely sequence-based models (e.g., LSTMs) when the problem is graph-structured. We demonstrate the capabilities on some simple AI (bAbI) and graph algorithm learning tasks. We then show it achieves state-of-the-art performance on a problem from program verification, in which subgraphs need to be matched to abstract data structures. (@li2016gated)

D. Lim, H. Lawrence, N. T. Huang, and E. H. Thiede Positional encodings as group representations: A unified framework . (@lim2023positional)

I. Loshchilov and F. Hutter Decoupled weight decay regularization *arXiv preprint arXiv:1711.05101*, 2017. **Abstract:** L$\_2$ regularization and weight decay regularization are equivalent for standard stochastic gradient descent (when rescaled by the learning rate), but as we demonstrate this is \\}emph{not} the case for adaptive gradient algorithms, such as Adam. While common implementations of these algorithms employ L$\_2$ regularization (often calling it "weight decay" in what may be misleading due to the inequivalence we expose), we propose a simple modification to recover the original formulation of weight decay regularization by \\}emph{decoupling} the weight decay from the optimization steps taken w.r.t. the loss function. We provide empirical evidence that our proposed modification (i) decouples the optimal choice of weight decay factor from the setting of the learning rate for both standard SGD and Adam and (ii) substantially improves Adam’s generalization performance, allowing it to compete with SGD with momentum on image classification datasets (on which it was previously typically outperformed by the latter). Our proposed decoupled weight decay has already been adopted by many researchers, and the community has implemented it in TensorFlow and PyTorch; the complete source code for our experiments is available at https://github.com/loshchil/AdamW-and-SGDW (@loshchilov2017decoupled)

J. Lu, J. Yao, J. Zhang, X. Zhu, H. Xu, W. Gao, C. Xu, T. Xiang, and L. Zhang Soft: Softmax-free transformer with linear complexity *Advances in Neural Information Processing Systems*, 34: 21297–21309, 2021. **Abstract:** Vision transformers (ViTs) have pushed the state-of-the-art for various visual recognition tasks by patch-wise image tokenization followed by self-attention. However, the employment of self-attention modules results in a quadratic complexity in both computation and memory usage. Various attempts on approximating the self-attention computation with linear complexity have been made in Natural Language Processing. However, an in-depth analysis in this work shows that they are either theoretically flawed or empirically ineffective for visual recognition. We further identify that their limitations are rooted in keeping the softmax self-attention during approximations. Specifically, conventional self-attention is computed by normalizing the scaled dot-product between token feature vectors. Keeping this softmax operation challenges any subsequent linearization efforts. Based on this insight, for the first time, a softmax-free transformer or SOFT is proposed. To remove softmax in self-attention, Gaussian kernel function is used to replace the dot-product similarity without further normalization. This enables a full self-attention matrix to be approximated via a low-rank matrix decomposition. The robustness of the approximation is achieved by calculating its Moore-Penrose inverse using a Newton-Raphson method. Extensive experiments on ImageNet show that our SOFT significantly improves the computational efficiency of existing ViT variants. Crucially, with a linear complexity, much longer token sequences are permitted in SOFT, resulting in superior trade-off between accuracy and complexity. (@lu2021soft)

F. Murnaghan and A. Wintner A canonical form for real matrices under orthogonal transformations *Proceedings of the National Academy of Sciences*, 17 (7): 417–420, 1931. **Abstract:** If A is a square matrix of order n with real or complex elements it is well known that it may be reduced by means of a unitary transformation U to a matrix of the same order all of whose elements below the leading diagonal are zero.’ Even when the elements of A are real the elements of the transforming matrix U are complex if the characteristic numbers of A are not all real and it is desirable to give a canonical form which may be reached by the use of real unitary (i.e., orthogonal) matrices. The derivation of this canonical form differs only in detail from that given by Schur. The characteristic numbers X of the matrix A are determined by the equation det(A XE) = 0, where E is the unit matrix, and they may be real or complex. If, as we suppose, the elements of A are real the complex roots will occur in conjugate imaginary pairs. If all the characteristic numbers are real the unitary transformations occurring in Schur’s derivation will be real and the canonical form sought for is that given by Schur. On the other hand, let Xi = ,u + iv and X2 = iv be a pair of conjugate complex characteristic numbers of the matrix A (j,, v real, v $ 0); on denoting by xi = a + ib, (a, b, real) a characteristic vector of A associated with the characteristic number Xi we have Ax, = Xix, which implies the two equations (@murnaghan1931canonical)

M. Post A call for clarity in reporting BLEU scores In *Proceedings of the Third Conference on Machine Translation: Research Papers*, pages 186–191, Belgium, Brussels, Oct. 2018. Association for Computational Linguistics. URL <https://www.aclweb.org/anthology/W18-6319>. **Abstract:** The field of machine translation faces an under-recognized problem because of inconsistency in the reporting of scores from its dominant metric. Although people refer to “the” BLEU score, BLEU is in fact a parameterized metric whose values can vary wildly with changes to these parameters. These parameters are often not reported or are hard to find, and consequently, BLEU scores between papers cannot be directly compared. I quantify this variation, finding differences as high as 1.8 between commonly used configurations. The main culprit is different tokenization and normalization schemes applied to the reference. Pointing to the success of the parsing community, I suggest machine translation researchers settle upon the BLEU scheme used by the annual Conference on Machine Translation (WMT), which does not allow for user-supplied reference processing, and provide a new tool, SACREBLEU, to facilitate this. (@post-2018-call)

R. Sennrich, B. Haddow, and A. Birch Neural machine translation of rare words with subword units In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1715–1725, 2016. **Abstract:** Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem.Previous work addresses the translation of out-of-vocabulary words by backing off to a dictionary.In this paper, we introduce a simpler and more effective approach, making the NMT model capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units.This is based on the intuition that various word classes are translatable via smaller units than words, for instance names (via character copying or transliteration), compounds (via compositional translation), and cognates and loanwords (via phonological and morphological transformations).We discuss the suitability of different word segmentation techniques, including simple character ngram models and a segmentation based on the byte pair encoding compression algorithm, and empirically show that subword models improve over a back-off dictionary baseline for the WMT 15 translation tasks English→German and English→Russian by up to 1.1 and 1.3 BLEU, respectively. (@sennrich2016neural)

P. Shaw, J. Uszkoreit, and A. Vaswani Self-attention with relative position representations In *Proceedings of NAACL-HLT*, pages 464–468, 2018. **Abstract:** Peter Shaw, Jakob Uszkoreit, Ashish Vaswani. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers). 2018. (@shaw2018self)

V. Shiv and C. Quirk Novel positional encodings to enable tree-based transformers *Advances in neural information processing systems*, 32, 2019. **Abstract:** Neural models optimized for tree-based problems are of great value in tasks like SQL query extraction and program synthesis. On sequence-structured data, transformers have been shown to learn relationships across arbitrary pairs of positions more reliably than recurrent models. Motivated by this property, we propose a method to extend transformers to tree-structured data, enabling sequence-to-tree, tree-to-sequence, and tree-to-tree mappings. Our approach abstracts the transformer’s sinusoidal positional encodings, allowing us to instead use a novel positional encoding scheme to represent node positions within trees. We evaluated our model in tree-to-tree program translation and sequence-to-tree semantic parsing settings, achieving superior performance over both sequence-to-sequence transformers and state-of-the-art tree-based LSTMs on several datasets. In particular, our results include a 22% absolute increase in accuracy on a JavaScript to CoffeeScript translation dataset. (@shiv2019novel)

J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu Roformer: Enhanced transformer with rotary position embedding *Neurocomputing*, page 127063, 2023. **Abstract:** Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer . (@su2023roformer)

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin Attention is all you need *Advances in neural information processing systems*, 30, 2017. **Abstract:** The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. (@vaswani2017attention)

A. Vyas, A. Katharopoulos, and F. Fleuret Fast transformers with clustered attention *Advances in Neural Information Processing Systems*, 33: 21665–21674, 2020. **Abstract:** Transformers have been proven a successful model for a variety of tasks in sequence modeling. However, computing the attention matrix, which is their key component, has quadratic complexity with respect to the sequence length, thus making them prohibitively expensive for large sequences. To address this, we propose clustered attention, which instead of computing the attention for every query, groups queries into clusters and computes attention just for the centroids. To further improve this approximation, we use the computed clusters to identify the keys with the highest attention per query and compute the exact key/query dot products. This results in a model with linear complexity with respect to the sequence length for a fixed number of clusters. We evaluate our approach on two automatic speech recognition datasets and show that our model consistently outperforms vanilla transformers for a given computational budget. Finally, we demonstrate that our model can approximate arbitrarily complex attention distributions with a minimal number of clusters by approximating a pretrained BERT model on GLUE and SQuAD benchmarks with only 25 clusters and no loss in performance. (@vyas2020fast)

B. Wang, Z. Donghao, L. Christina, Q. Li, Z. Peng, J. G. Simonsen, et al Encoding word order in complex embeddings In *ICLR 2020-Proceedings of Eighth International Conference on Learning Representations*, 2020. **Abstract:** Sequential word order is important when processing text. Currently, neural networks (NNs) address this by modeling word position using position embeddings. The problem is that position embeddings capture the position of individual words, but not the ordered relationship (e.g., adjacency or precedence) between individual word positions. We present a novel and principled solution for modeling both the global absolute positions of words and their order relationships. Our solution generalizes word embeddings, previously defined as independent vectors, to continuous word functions over a variable (position). The benefit of continuous functions over variable positions is that word representations shift smoothly with increasing positions. Hence, word representations in different positions can correlate with each other in a continuous function. The general solution of these functions is extended to complex-valued domain due to richer representations. We extend CNN, RNN and Transformer NNs to complex-valued versions to incorporate our complex embedding (we make all code available). Experiments on text classification, machine translation and language modeling show gains over both classical word embeddings and position-enriched word embeddings. To our knowledge, this is the first work in NLP to link imaginary numbers in complex-valued representations to concrete meanings (i.e., word order). (@wang2020encoding)

Z. Wang and J.-C. Liu Translating math formula images to latex sequences using deep neural networks with sequence-level training 2019. **Abstract:** —In this paper we propose a deep neural network mode l with an encoder-decoder architecture that translate s images of math formulas into their LaTeX markup sequences. The enc oder is a convolutional neural network (CNN) that transforms images into a group of feature maps. To better capture the spatia l relationships of math symbols, the feature maps are augmented with 2 D positional encoding before being unfolded into a vector. The d ecoder is a stacked bidirectional long short-term memory (LSTM) model integrated with the soft attention mechanism, which works as a language model to translate the encoder output into a sequence of LaTeX tokens. The neural network is trained in two steps. The first step is token-level training using the Maximum-Like lihood Estimation (MLE) as the objective function. At comp letion of the token-level training, the sequence-level training o bjective function is employed to optimize the overall model based on the policy gradient algorithm from reinforcement learning. Our design a lso overcomes the exposure bias problem by closing the feedback l oop in the decoder during sequence-level training, i.e., feedi ng in the predicted token instead of the ground truth token at every ti me step. The model is trained and evaluated on the IM2LATEX-100K datas et and shows state-of-the-art performance on both sequence-based and image- based evaluation metrics. (@wang2019translating)

M. Weiler, P. Forré, E. Verlinde, and M. Welling Equivariant and coordinate independent convolutional networks *A Gauge Field Theory of Neural Networks*, 2023. **Abstract:** Motivated by the vast success of deep convolutional networks, there is a great interest in generalizing convolutions to non-Euclidean manifolds. A major complication in comparison to flat spaces is that it is unclear in which alignment a convolution kernel should be applied on a manifold. The underlying reason for this ambiguity is that general manifolds do not come with a canonical choice of reference frames (gauge). Kernels and features therefore have to be expressed relative to arbitrary coordinates. We argue that the particular choice of coordinatization should not affect a network’s inference – it should be coordinate independent. A simultaneous demand for coordinate independence and weight sharing is shown to result in a requirement on the network to be equivariant under local gauge transformations (changes of local reference frames). The ambiguity of reference frames depends thereby on the G-structure of the manifold, such that the necessary level of gauge equivariance is prescribed by the corresponding structure group G. Coordinate independent convolutions are proven to be equivariant w.r.t. those isometries that are symmetries of the G-structure. The resulting theory is formulated in a coordinate free fashion in terms of fiber bundles. To exemplify the design of coordinate independent convolutions, we implement a convolutional network on the M\\}"obius strip. The generality of our differential geometric formulation of convolutional networks is demonstrated by an extensive literature review which explains a large number of Euclidean CNNs, spherical CNNs and CNNs on general surfaces as specific instances of coordinate independent convolutions. (@weiler2023equivariant)

C. Wu, F. Wu, and Y. Huang -Transformer: Distance-aware transformer In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2059–2068, 2021. **Abstract:** Transformer has achieved great success in the NLP field by composing various advanced models like BERT and GPT. However, Transformer and its existing variants may not be optimal in capturing token distances because the position or distance embeddings used by these methods usually cannot keep the precise information of real distances, which may not be beneficial for modeling the orders and relations of contexts. In this paper, we propose DA-Transformer, which is a distance-aware Transformer that can exploit the real distance. We propose to incorporate the real distances between tokens to re-scale the raw self-attention weights, which are computed by the relevance between attention query and key. Concretely, in different self-attention heads the relative distance between each pair of tokens is weighted by different learnable parameters, which control the different preferences on long- or short-term information of these heads. Since the raw weighted real distances may not be optimal for adjusting self-attention weights, we propose a learnable sigmoid function to map them into re-scaled coefficients that have proper ranges. We first clip the raw self-attention weights via the ReLU function to keep non-negativity and introduce sparsity, and then multiply them with the re-scaled coefficients to encode real distance information into self-attention. Extensive experiments on five benchmark datasets show that DA-Transformer can effectively improve the performance of many tasks and outperform the vanilla Transformer and its several variants. (@wu2021transformer)

Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, et al Google’s neural machine translation system: Bridging the gap between human and machine translation *arXiv preprint arXiv:1609.08144*, 2016. **Abstract:** Neural Machine Translation (NMT) is an end-to-end learning approach for automated translation, with the potential to overcome many of the weaknesses of conventional phrase-based translation systems. Unfortunately, NMT systems are known to be computationally expensive both in training and in translation inference. Also, most NMT systems have difficulty with rare words. These issues have hindered NMT’s use in practical deployments and services, where both accuracy and speed are essential. In this work, we present GNMT, Google’s Neural Machine Translation system, which attempts to address many of these issues. Our model consists of a deep LSTM network with 8 encoder and 8 decoder layers using attention and residual connections. To improve parallelism and therefore decrease training time, our attention mechanism connects the bottom layer of the decoder to the top layer of the encoder. To accelerate the final translation speed, we employ low-precision arithmetic during inference computations. To improve handling of rare words, we divide words into a limited set of common sub-word units ("wordpieces") for both input and output. This method provides a good balance between the flexibility of "character"-delimited models and the efficiency of "word"-delimited models, naturally handles translation of rare words, and ultimately improves the overall accuracy of the system. Our beam search technique employs a length-normalization procedure and uses a coverage penalty, which encourages generation of an output sentence that is most likely to cover all the words in the source sentence. On the WMT’14 English-to-French and English-to-German benchmarks, GNMT achieves competitive results to state-of-the-art. Using a human side-by-side evaluation on a set of isolated simple sentences, it reduces translation errors by an average of 60% compared to Google’s phrase-based production system. (@wu2016google)

</div>

# Parameterizing <span class="smallcaps">ape</span> [appendix:init]

## Orthogonalization [appendix:orthogonalization]

The orthogonal primitives underlying <span class="smallcaps">ape</span> can be procured by matrix-exponentiating skew-symmetric bases. Concretely, for some cyclic group \\(\langle {\bm{C}} \rangle\\):

1.  Start with an *upper triangular* matrix \\({\bm{A}}\\); this matrix parameterizes the entire group.

2.  Obtain the *skew symmetric* \\({\bm{B}} := {\bm{A}} - {\bm{A}}^\top\\)

3.  Obtain the *matrix exponent* \\({\bm{C}} := \mathrm{exp}({\bm{B}})\\); the resulting matrix is *orthogonal*, and acts as the group’s generator.

## Switching between <span class="smallcaps">ape</span> and <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> [appendix:switching]

In the commutative (direct sum of finitely many cyclic groups) case, it is possible to switch freely between <span class="smallcaps">ape</span> and <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>. Doing so might be useful, *e.g.*, for initializing <span class="smallcaps">ape</span>, for inspecting the learned rotoreflections post-training, or for making use of <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>’s memory-optimized vector-multiplication formula in a system originally trained with <span class="smallcaps">ape</span>. Note that here we consider the purely real-valued version of <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> (and <span class="smallcaps">ape</span>).

#### <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> \\(\to\\) <span class="smallcaps">ape</span>

To convert <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> to <span class="smallcaps">ape</span> for some collection of angles \\(\Theta := [\theta_1, \dots \theta_n]\\):

1.  Expand \\(\Theta\\) into a rotation matrix \\({\bm{C}}\\), \\[{\bm{C}} := \begin{bmatrix}
                cos\theta_1 & -sin\theta_1 & 0 & 0 & \dots \\
                sin\theta_1 & cos\theta_1 & 0 & 0 & \dots \\
                0 & 0 & cos\theta_2 & -sin\theta_2 & \dots \\
                0 & 0 & sin\theta_2 & cos\theta_2 & \dots\\
                \vdots & \vdots & \vdots & \vdots & \ddots\\
            \end{bmatrix}\\] **Note**: Stop here if not interested in parameterizing \\({\bm{C}}\\).

2.  Use a solver to approximate the *matrix logarithm* of \\({\bm{C}}\\), \\({\bm{B}} := \mathrm{log}({\bm{C}})\\).

3.  Find a matrix \\({\bm{A}}\\) such that \\(\mathrm{mse}({\bm{B}}, {\bm{A}}-{\bm{A}}^\top) \leq \epsilon\\), *e.g.*, using a numerical optimizer. Matrix \\({\bm{A}}\\) can be used to parameterize the group, *cf.* <a href="#appendix:orthogonalization" data-reference-type="ref" data-reference="appendix:orthogonalization">8.1</a>.

#### <span class="smallcaps">ape</span> \\(\to\\) <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>

To convert <span class="smallcaps">ape</span> to <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> for some cyclic group \\(\langle {\bm{W}} \rangle\\):

1.  Find the normal form \\({\bm{W}} = {\bm{P}}{\bm{Q}}{\bm{P}}^\top\\).

2.  Extract the angles in each block of \\({\bm{Q}}\\); the resulting collection of angles is <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>’s \\(\Theta\\).

3.  For each attention head involved, right-compose the Transformer’s \\({\bm{\Phi}}^{(q)}\\) and \\({\bm{\Phi}}^{(k)}\\) with \\({\bm{P}}\\).

# Experimental Setups [appendix:esetup]

## Machine Translation

For our machine translation experiments, we use the official dataset breakdown (including the extended evaluation set). We tokenize the training and evaluation sets with <span class="smallcaps">Moses</span> [^6], using the standard pipeline: punctuation normalization \\(\to\\) unicode normalization \\(\to\\) language-specific tokenization. We apply byte-pair encoding `\citep{gage1994new,sennrich2016neural}`{=latex} using the `subword-nmt` package [^7]. We apply 32k merges across the source and target training corpora, without truncating the resulting (shared) vocabulary (of size 35 533). Our loss term is given as the cross-entropy between the teacher-forced predictions and the ground-true labels, smoothed by 10%. We train in a distributed environment consisting of 4 GPUs, with a batch size of 3 072 target tokens per GPU. We average gradients and update parameters once every 2 GPU iterations (or: 8 batches). We optimize using Adam with a learning rate dictated by the schedule prescribed by `\citet{vaswani2017attention}`{=latex}. We stop optimizing after 150 000 parameter updates or 16 hours, whichever comes first. Throughout training, we circularly store the 10 best checkpoints, ranked on the basis of dev set loss (evaluated once every 500 updates). During inference, we average the 10 checkpoints into a single model, and select hypotheses from a beam of width 4 and a length penalty of 0.6 `\citep{wu2016google}`{=latex}. We report <span class="smallcaps">bleu</span> scores over the *test set* (`newstest2014`), comparing the <span class="smallcaps">bpe</span>-merged and detokenized output against the raw references using `sacrebleu` `\citep{post-2018-call}`{=latex} [^8].

er

<div class="tabular" markdown="1">

@ lccc@  &  
& NMT & Transduction & Image  
Convolution Size & – & – & (3,3)  
Convolution Stride & – & – & 1  
Embedding Size & 512 & 512 & 256  
Feedforward Size (enc) & 2048 & 512 & 512  
Feedforward Size (dec) & 2048 & 1024 & –  
Feedforward Activation & ReLU & ReLU & GELU  
\# Layers (enc, dec) & (6, 6) & (2,2) & (7, 0)  
\# Heads & 8 & 8 & 4  
Norm & LayerNorm & LayerNorm & LayerNorm  
Norm Position & Post & Pre & Pre  

</div>

## Synthetic Transduction

#### Tree Task Descriptions

The tree copy task is morally identical to its sequential version – the tree structure (and its positional specification) is practically a confound.

In the tree rotation\\(^\star\\) task, the output tree is the result of recursively right-rotating all subtrees of the input. The task is challenging but purely structural, in the sense that its resolution requires no real interaction between content and position.

For the algebraic expression reduction task, we consider input trees that specify a complex expression from the cyclic group C~3~, and task the model with producing the result of a single reduction step (*i.e.*, reducing all subtrees of depth 1 into a leaf). This time around, the model has to identify reducible subtrees, match operators to their argument and collapse the three into a single node depending on their content.

The tree operations task, finally, combines the aspects of the other three, requiring content-based addressing, structure manipulation and dynamic semantics resolution. Concretely, we generate an input tree consisting of unique nodes, and randomly select one of its subtrees as well as one of four operators. We then construct a deeper tree, where the new root corresponds to the chosen operator, its left branch corresponds to the numerical index of the selected subtree, and the right branch is the original tree in its entirety. The model is then tasked with producing the correct output given this combination of an operator, a tree, and an index. We consider four operations: extraction (*i.e.*, return the indexed subtree), flip-extraction (*i.e.*, return the indexed subtree, rotated), truncation (*i.e.*, return the full tree with the indexed subtree removed) and a no-op (*i.e.*, return the full tree as-is, ignoring indexing).

#### Hyperparameters

For all synthetic tasks, we generate disjoint train, dev and test sets of sizes 6 000, 2 000 and 2 000. We train a small Transformer model, optimizing with AdamW `\citep{loshchilov2017decoupled}`{=latex} for 400 epochs and a batch size of 64, using a linear warmup – cosine decay schedule. For the sequential tasks, we populate the datasets with words of random lengths from \\(\mathcal{N}(100, 10)\\) and a vocabulary size of \\(20\\) (to ensure token repetition and diffuse the possibility for leaning on content-based addressing). For the tree tasks, we populate the datasets with non-uniform trees of random depths sampled from \\(\mathcal{N}(7, 1)\\). For the tree-ops task, exceptionally, we set the vocabulary size to 128 so as to have enough unique nodes to allow content-based addressing.

When using a positional encoding scheme that requires fixing the size of the structure being modeled (*i.e.*, the *Tree*, *Relative*, and *Absolute* schemes), we fix it at approximately the maximum training size, practically ensuring the most stringent comparison.

In all experiments, we share source and target embedding weights between both the encoder-decoder embedding layers, and the decoder’s classification head.

## Image Recognition [image-recognition]

For our image recognition experiments, we largely rely on the setup of `\citet{hassani2021escaping}`{=latex}. Concretely, we apply a small-step “tokenizing” convolution on the input image, downsample the result with max pooling and flatten the result into a sequence. After we pass the sequence through the encoder, we apply a global soft attention `\cite[\textit{inter alia}]{li2016gated}`{=latex} (rediscovered by `\citet{hassani2021escaping}`{=latex}, there dubbed “sequence pooling”) to aggregate into a single vector prior to applying the classifier. To attain competitive scores, we apply standard <span class="smallcaps">cifar</span>-10 data augmentations and more aggressive regularization: a 10% attention weight dropout, a stochastic depth of 10% for each consecutive layer, and a weight decay of \\(3\cdot 10^{-2}\\). The above settings and the hyperparameter setup are taken without modification from `\citet{hassani2021escaping}`{=latex}.

# NeurIPS Paper Checklist [neurips-paper-checklist]

1.  **Claims**

2.  Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

3.  Answer:

4.  Justification: We carefully summarize our contributions and refrain from making any claims that we cannot theoretically or empirically support.

5.  Guidelines:

    - The answer NA means that the abstract and introduction do not include the claims made in the paper.

    - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

    - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

    - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6.  **Limitations**

7.  Question: Does the paper discuss the limitations of the work performed by the authors?

8.  Answer:

9.  Justification: We have a dedicated limitations section (§<a href="#sec:limitations" data-reference-type="ref" data-reference="sec:limitations">6</a>), and openly and explicitly discuss algorithm complexity and experimental scope in the relevant sections.

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

14. Justification: Our algebraic connections make no assumptions and are fully explicit in their presentation. The equivalence with <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span> clarifies all assumptions it makes.

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

19. Justification: We provide the reviewers with both an extensive appendix detailing our experimental setup, and the code used to implement our methodology and conduct our experiments.

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

24. Justification: Yes – see answer above. Our training scripts are provided virtually unchanged.

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

29. Justification: Yes, see above.

30. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental material.

31. **Experiment Statistical Significance**

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

33. Answer:

34. Justification: We take extra care to conduct our experiments openly and transparently so as to deliver statistically sound results and draw solid conclusions. We repeat *all* experiments multiple times, and report means and 95% confidence intervals. For each experiment, we visually mark all models that overlap with the best performer in the category.

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

39. Justification: While we do report hardware infrastructure, we do not report memory consumption or clock times. With the exception of machine translation, our experiments are moderately cheap to run, requiring no specialized hardware other than GPU accelaration.

40. Guidelines:

    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. **Code Of Ethics**

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

43. Answer:

44. Justification: Checked and done.

45. Guidelines:

    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. **Broader Impacts**

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

48. Answer:

49. Justification: We do, albeit briefly. We do not see possible negative implications.

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

54. Justification: We perceive no risks that would require safeguards of any kind.

55. Guidelines:

    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. **Licenses for existing assets**

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

58. Answer:

59. Justification: We cite all software libraries and datasets we use, and comply with their licenses.

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

64. Justification: While we do provide reference implementations, we do not see them as assets per se, neither do we hand them out as ready-to-use integrations.

65. Guidelines:

    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. **Crowdsourcing and Research with Human Subjects**

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

68. Answer:

69. Justification: No human subjects were involved in this study.

70. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

73. Answer:

74. Justification: See above.

75. Guidelines:

    - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

[^1]: It is also complete except for the odd case where \\({\bm{W}}^p={\bm{I}}\\) for some \\(p\\). In practice, this kind of periodic behaviour does not arise randomly, and we can think of \\(\langle {\bm{W}} \rangle\\) as being *isomorphic* to \\(\ensuremath{\mathbb{P}}\\).

[^2]: The story is no different for \\({\bm{W}}\\) unitary, with the group structure provided by the unitary group \\(U(d)\\), and path inversion interpreted as the matrix conjugate transpose.

[^3]: We alert the reader that a *constructive* proof of this decomposition has proven surprisingly difficult to find.

[^4]: An alternative reading is that even though orthogonal matrices are generally more expressive than rotation matrices (allowing not just rotations but also reflections), the Transformer’s architecture makes up for <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>’s reduced expressivity by supplying a free change of basis through its trainable weights \\({\bm{\Phi}}\\).

[^5]: For more practical insights on initializing and parameterizing <span class="smallcaps">ape</span> and translating between <span class="smallcaps">ape</span> and <span class="smallcaps">r</span><span class="nodecor">o</span><span class="smallcaps">pe</span>, please refer to Appendix <a href="#appendix:init" data-reference-type="ref" data-reference="appendix:init">8</a>.

[^6]: See <https://github.com/moses-smt/mosesdecoder>

[^7]: See <https://github.com/rsennrich/subword-nmt>.

[^8]: Signature: `nrefs:1` \| `case:lc` \| `eff:no` \| `tok:13a` \| `smooth:exp` \| `version:2.4.2`.
