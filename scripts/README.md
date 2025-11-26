# Usage and How to Run

This directory contains all the scripts for the evaluation framework. This guide explains how to use each component.

## Installation

### Requirements

```bash
pip install pandas google-generativeai python-dotenv pydantic tqdm openreview-py matplotlib seaborn numpy
```

### Environment Setup

Create a `.env` file in the project root with your Gemini API keys (you can obtain free API keys by going to [Gemini API keys](https://aistudio.google.com/api-keys) --> "Create API keys" --> "Select a Cloud Project" --> "Create project").

```env
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
GEMINI_API_KEY_3=your_key_3
# ... add more keys as needed
```

## Key Features

### 1. Error Planting (`paper_manipulation/plant_quality_errors/`)

The error planting system introduces methodological flaws into research papers based on flaw descriptions from peer review data. Key capabilities:

- **Multi-threaded processing** with multiple Gemini API keys
- **Intelligent rate limiting** (RPM and TPM tracking with sliding windows)
- **Surgical modifications**: Precisely modifies specific sections while maintaining paper coherence
- **Placebo generation**: Creates "sham surgery" versions by learning writing style and rewriting original sections without introducing flaws

**Usage:**

```bash
python scripts/paper_manipulation/plant_quality_errors/plant_errors_and_placebo.py \
    --csv_file data/ICLR2024/filtered_pairs_with_human_scores.csv \
    --base_dir data/ICLR2024/latest \
    --output_dir data/ICLR2024 \
    --max_workers 3
```

### 2. Error De-planting (`paper_manipulation/deplant_planted_error/`)

Generates LLM-based fixes for planted errors, creating "fake good" revisions that superficially address flaws without substantive improvements.

**Usage:**

```bash
python scripts/paper_manipulation/deplant_planted_error/deplant_planted_error.py \
    --data_dir data/with_appendix \
    --conference NeurIPS2024 \
    --model_name gemini-2.0-flash \
    --max_workers 4
```

### 3. Verification (`verify_change/`)

Evaluates whether LLMs can distinguish between:

- **True Good**: Camera-ready papers with real revisions
- **Fake Good**: LLM-generated fixes that look good but lack substance

The verification system uses dual scoring:

- **Quality Score (1-9)**: Does the revision theoretically solve the problem?
- **Verifiability Score (1-9)**: Can you trust the evidence provided?

**Usage:**

```bash
python scripts/verify_change/verify_revision_quality.py \
    --data_dir data/sampled_data_verify_change/no_appendix \
    --model_name gemini-2.0-flash-lite \
    --comparison_type true_good_vs_fake_good \
    --dual_scores
```

### 4. Ablation Analysis (`verify_change/compare_ablation_scenarios.py`)

Compares model performance across different ablation scenarios. The verification script supports various ablation flags:

**Available Ablation Flags:**

- `--ablation_name no_location`: Excludes location information about where changes were made
- `--ablation_name with_location`: Includes location information about where changes were made
- `--remove_tables`: Removes all markdown tables from papers before evaluation
- `--use_v1_as_flawed`: Uses v1 submitted papers as flawed papers instead of planted_error versions
- `--include_rebuttals`: Includes author rebuttals from OpenReview in the evaluation prompt
- `--snippets_only`: Uses only changed sections instead of full papers
- `--dual_scores`: Uses dual scoring (Quality + Verifiability) instead of single score

**Special Case: No Bibliography Ablation**

For the `no_bibliography` ablation, you must first prune references from the papers using the `prune_references.py` script before running verification:

```bash
# Step 1: Prune references from papers
python scripts/paper_manipulation/prune_bibliography/prune_references.py \
    source_dir dest_dir

# Step 2: Run verification on the pruned papers
python scripts/verify_change/verify_revision_quality.py \
    --data_dir dest_dir \
    --ablation_name no_bibliography \
    ...
```

**Usage:**

```bash
# Example: Run with location information
python scripts/verify_change/verify_revision_quality.py \
    --data_dir data/sampled_data_verify_change/no_appendix \
    --model_name gemini-2.0-flash-lite \
    --ablation_name with_location \
    --dual_scores

# Example: Compare all ablations
python scripts/verify_change/compare_ablation_scenarios.py
```

## Complete Evaluation Pipeline

1. **Prepare Data**: Organize papers with flaw descriptions in CSV format

   ```bash
   # Use utility scripts to filter and prepare data
   python scripts/utils/filter_by_category.py ...
   python scripts/utils/get_paper_pairs.py ...
   ```
2. **Plant Errors**: Introduce flaws into papers

   ```bash
   python scripts/paper_manipulation/plant_quality_errors/plant_errors_and_placebo.py ...
   ```
3. **Generate Fixes**: Create LLM-based fixes for planted errors

   ```bash
   python scripts/paper_manipulation/deplant_planted_error/deplant_planted_error.py ...
   ```
4. **Verify Changes**: Evaluate LLM capability to detect differences

   ```bash
   python scripts/verify_change/verify_revision_quality.py ...
   ```
5. **Compare Ablations**: Analyze results across different scenarios

   ```bash
   python scripts/verify_change/compare_ablation_scenarios.py
   ```

## Key Metrics

The evaluation uses several metrics to assess LLM performance:

- **Overall Effect**: Sum of all score differences (positive - negative)
- **Local Effect**: Average difference per paper/revision pair
- **Micro-averaged Overall Effect**: Fraction of positive differences, averaged across papers
- **Quality Score**: Assessment of whether revisions theoretically solve the problem (1-9)
- **Verifiability Score**: Assessment of trustworthiness and reproducibility of evidence (1-9)

## Model Support

The framework supports multiple LLM providers:

- **Google Gemini**: `gemini-2.0-flash-lite`, `gemini-2.5-pro`, `gemini-3-pro`, etc.
- **OpenAI GPT**: `gpt-5.1` (via `verify_revision_quality_openai.py`)
- **Qwen (via API)**: `qwen3-235B` (via `verify_revision_quality_qwen.py`)
- **Open-source models**: `llama4-maverick-17b`, `qwen3-235B`, ... (by hosting your intended model with [vLLM](https://github.com/vllm-project/vllm), then run `verify_revision_quality_vllm.py`)

## Rate Limiting

The system implements sophisticated rate limiting:

- **RPM (Requests Per Minute)**: Sliding window tracking per API key
- **TPM (Tokens Per Minute)**: Token usage tracking with conservative thresholds
- **Dynamic Backoff**: Automatic delay adjustment on 429 errors
- **Multi-key Support**: Round-robin distribution across multiple API keys

## Data Structure

### Generated Papers Structure

The evaluation framework expects papers organized in the following structure:

```
{venue_name}/
├── {category_id}/
│   ├── latest/
│   │   └── {paper_folder}/
│   │       └── structured_paper_output/
│   │           └── paper.md
│   ├── planted_error/
│   │   └── {paper_folder}/
│   │       ├── flawed_papers/
│   │       │   └── {flaw_id}.md
│   │       └── {paper_folder_name}_modifications_summary.csv
│   ├── de-planted_error/
│   │   └── {paper_folder}/
│   │       ├── {paper_folder_name}_fix_summary.csv
│   │       └── (fixed paper files)
│   ├── v1/
│   │   └── {paper_folder}/
│   │       └── structured_paper_output/
│   │           └── paper.md
│   └── sham_surgery/
│       └── {paper_folder}/
│           └── (sham surgery files)
```

Where:

- `{venue_name}`: Conference name (e.g., `NeurIPS2024`)
- `{category_id}`: Flaw category identifier (e.g., `1a`, `2b`, etc.)
- `{paper_folder}`: Paper identifier folder
- `latest/`: Camera-ready versions (true good revisions)
- `planted_error/`: Papers with planted flaws
- `de-planted_error/`: LLM-generated fixes (fake good revisions)
- `v1/`: Original submitted versions
- `sham_surgery/`: Placebo versions with style changes but no flaws

### Error Planting Output

```
output_dir/
├── planted_error/
│   ├── {paper_folder}/
│   │   ├── flaw_1.md
│   │   ├── flaw_1.json
│   │   └── ...
│   └── ...
├── sham_surgery/
│   ├── {paper_folder}/
│   │   ├── flaw_1.md
│   │   ├── flaw_1.json
│   │   └── ...
│   └── ...
└── planting_results.csv
```

### Verification Output

```
output_dir/
├── llm_verification_{model_name}/
│   ├── verification_scores.csv
│   ├── score_differences.csv
│   └── plots/
│       ├── overall_effect_by_category.png
│       └── ...
└── ...
```
