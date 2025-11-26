# Data Structure

This directory contains sample processed papers organized by flaw category. The data follows a hierarchical structure where papers are categorized according to thirteen specific flaw types, grouped into five main categories.

## Directory Structure

```
data/
└── sample_processed_papers/
    ├── 1a/  # Insufficient Baselines
    ├── 1b/  # Weak Scope
    ├── 1c/  # No Ablation
    ├── 1d/  # Flawed Metrics
    ├── 2a/  # Fundamental Design Limitation
    ├── 2b/  # Missing Theory
    ├── 2c/  # Technical/Math Error
    ├── 3a/  # Insufficient Novelty
    ├── 3b/  # Overclaims
    ├── 4a/  # Lack of Clarity
    ├── 4b/  # Missing Reproducibility Details
    ├── 5a/  # Unacknowledged Limitations
    └── 5b/  # Unaddressed Ethical Issues
```

Each category directory contains:

- `latest/`: Camera-ready versions (true good revisions) - papers that genuinely address the identified flaws
- `planted_error/`: Papers with surgically injected flaws - the baseline flawed versions
- `de-planted_error/`: LLM-generated fixes - superficial revisions that look good but lack substance

## Flaw Category Taxonomy

To evaluate the sensitivity of LLMs to specific scientific deficits, papers were categorized according to the following thirteen flaw types:

### Category 1: Empirical Evaluation Flaws

#### 1a: **Baselines** *(Insufficient Baselines/Comparisons)*

The evaluation is missing comparisons to relevant, state-of-the-art, or obvious alternative methods.

#### 1b: **Scope** *(Weak or Limited Scope of Experiments)*

The experiments are too narrow to support the paper's general claims (e.g., "toy" problems, insufficient data, no real-world testing).

#### 1c: **Ablation** *(Lack of Necessary Ablation or Analysis)*

The paper fails to analyze why its method works, missing ablation studies, cost/scalability analysis, or parameter sensitivity checks.

#### 1d: **Metrics** *(Flawed Evaluation Metrics or Setup)*

The metrics used are inappropriate or misleading, or the experimental setup is unreliable.

### Category 2: Methodological Flaws

#### 2a: **Design** *(Fundamental Technical Design Limitation)*

The proposed method has an inherent design flaw that severely restricts its applicability or performance (e.g., requires unrealistic inputs, cannot scale by design).

#### 2b: **Lacks Theory** *(Missing or Incomplete Theoretical Foundation)*

The paper requires but lacks a formal theoretical justification for its method (e.g., no convergence guarantees, no formal proof).

#### 2c: **Math Error** *(Technical or Mathematical Error)*

The paper contains a demonstrable error in its mathematical derivations, proofs, or algorithm description.

### Category 3: Positioning

#### 3a: **Novelty** *(Insufficient Novelty / Unacknowledged Prior Work)*

The core contribution is highly similar to or an uncredited rediscovery of existing work.

#### 3b: **Overclaims** *(Overstated Claims or Mismatch Between Claim and Evidence)*

The paper's claims in the abstract, introduction, or conclusion are stronger than what the experimental results actually support.

### Category 4: Presentation

#### 4a: **Clarity** *(Lack of Clarity / Ambiguity)*

The paper is written in a way that is ambiguous or difficult to understand, preventing an expert from properly interpreting the work.

#### 4b: **Reproducibility** *(Missing Implementation or Methodological Details)*

The paper omits crucial details needed for reproduction (e.g., key hyperparameters, data processing steps, source code).

### Category 5: Limitations

#### 5a: **Limitations** *(Unacknowledged Technical Limitations)*

The paper fails to discuss or downplays obvious or crucial limitations of its method, evaluation, or theoretical assumptions.

#### 5b: **Ethical** *(Unaddressed Ethical or Societal Impact)*

The paper fails to address potential negative societal impacts, risks of misuse, fairness, or other ethical considerations raised by the research.

## Paper Organization Within Each Category

Within each category directory (e.g., `1a/`), papers are organized as follows:

```
{category_id}/
├── latest/
│   └── {paper_id}/
│       └── structured_paper_output/
│           └── paper.md                    # Camera-ready version
├── planted_error/
│   └── {paper_id}/
│       ├── flawed_papers/
│       │   └── {flaw_id}.md                # Paper with surgically injected flaw
│       └── {paper_id}_modifications_summary.csv  # Summary of modifications
└── de-planted_error/                        # (Some categories only)
    └── {paper_id}/
        ├── {paper_id}_fix_summary.csv
        └── (fixed paper files)
```

Where:

- `{paper_id}`: Unique identifier for each paper (e.g., `fOQunr2E0T_2412_14076`)
- `{flaw_id}`: Identifier for the specific flaw instance

## Usage in Evaluation

The evaluation framework uses triplets of papers:

1. **Baseline** (`planted_error/`): Paper with a surgically injected flaw
2. **Treatment** (`latest/`): Camera-ready version that genuinely addresses the flaw
3. **Placebo** (`de-planted_error/`): LLM-generated fix that superficially addresses the flaw

This structure enables controlled intervention experiments to assess whether LLMs can distinguish between genuine improvements and superficial edits.
