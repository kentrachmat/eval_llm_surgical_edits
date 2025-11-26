# Evaluating LLM Judgment via Surgical Edits

<div align="center">
  <a href="https://github.com/ktgiahieu/eval_llm_surgical_edits/raw/master/PAPER.pdf">
    <img src="https://img.shields.io/badge/📄_Download_Paper-PDF-764ba2?style=for-the-badge" height="600">
  </a>
</div>

## Abstract

The growing volume of scientific paper submissions raises interest in LLMs as review assistants, but imitation of historical reviews risks reproducing spurious correlations and biases. We propose a controlled-intervention protocol to test whether an agent is causally sensitive to real quality improvements. Each instance uses a triplet of papers (original flawed, substantive human revision, and superficial LLM revision) to assess whether the agent detects genuine improvements while ignoring superficial edits. The strongest models successfully detect gains in flaw categories requiring internal logical consistency, while exhibiting near-zero or negative sensitivity to those requiring external empirical verification.

### Micro-averaged Overall Effect Comparison

![Micro-averaged Overall Effect Comparison](https://github.com/ktgiahieu/eval_llm_surgical_edits/blob/master/figs/overall/micro_avg_overall_effect_comparison_all.png)

### Effect Size by Flaw Category (Gemini 2.5 Pro)

![Effect Size by Flaw Category](https://github.com/ktgiahieu/eval_llm_surgical_edits/blob/master/figs/planted_no_location/effect_size_gemini_2-5_pro.png)

---

This repository contains the code and results for the paper "Evaluating LLM Judgment via Surgical Edits". The project implements a comprehensive evaluation framework that includes error planting, placebo generation, and revision verification across multiple models and ablation scenarios.

## Overview

The evaluation framework consists of three main components:

1. **Error Planting**: Systematically introducing methodological flaws into research papers based on reviewer-identified issues
2. **Placebo Generation**: Creating "sham surgery" versions that maintain writing style but don't introduce flaws
3. **Verification**: Testing whether LLMs can distinguish between:
   - Real revisions (camera-ready papers that genuinely address flaws)
   - LLM-generated fixes (superficial changes without substantive improvements)

The framework uses a controlled intervention protocol with triplets of papers (original flawed, substantive human revision, and superficial LLM revision) to assess whether LLMs detect genuine improvements while ignoring superficial edits.

## Project Structure

```
eval_llm_surgical_edits/
├── scripts/
│   ├── paper_manipulation/
│   │   ├── plant_quality_errors/      # Plant errors and generate placebos
│   │   ├── deplant_planted_error/     # Generate LLM-based fixes for planted errors
│   │   └── prune_bibliography/        # Prune references for ablation studies
│   ├── verify_change/                  # Verify LLM capability to detect changes
│   └── utils/                          # Utility scripts for data processing
├── data/
│   └── sample_processed_papers/        # Sample processed papers organized by category
├── PAPER.pdf                           # Research paper
└── README.md                           # This file
```

## Usage and How to Run

For detailed usage instructions, installation guide, and examples, please see the [scripts README](https://github.com/ktgiahieu/eval_llm_surgical_edits/blob/master/scripts/README.md).

## Visualization

The project evaluates multiple LLM models across various ablation scenarios. Additional visualizations can be found in the `figs/` directory.

## Citation

If you use this code or results in your research, please cite the associated paper. The full paper, including appendices, can be found in [PAPER.pdf](https://github.com/ktgiahieu/eval_llm_surgical_edits/blob/master/PAPER.pdf) in this repository.

## Contributing

This is a research codebase. For questions or issues, please open an issue on the repository.

## Acknowledgments

This work was performed using HPC resources from GENCI-IDRIS (Grant 2025-AD011016658). This work was supported by the Microsoft Accelerate Foundation Models Research (AFMR) grant program and a grant of Paris Région Ile-de-France.
