#!/usr/bin/env python3
"""
Compare models across different ablation scenarios.
Analyzes score_differences.csv files and generates comparison statistics and plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Model name mapping to canonical names
MODEL_NAME_MAPPING = {
    'llm_verification_gemini_2_5_pro': 'Gemini 2.5 Pro',
    'llm_verification_gemini_3_pro': 'Gemini 3 Pro',
    'llm_verification_gemini_2_0_flash': 'Gemini 2.0 Flash',
    'llm_verification_gemini_2_5_flash': 'Gemini 2.5 Flash',
    'llm_verification_gemini_2_5_flash_lite': 'Gemini 2.5 Flash Lite',
    'llm_verification_qwen3_235B': 'Qwen3-VL-235B-A22B-Instruct',
    'llm_verification_llama4_maverick': 'Llama-4-Maverick-17B-128E-Instruct',
    'llm_verification_gpt_5_1_no_location': 'GPT-5.1',
    'llm_verification_gpt_5_1': 'GPT-5.1',
    # Handle variations
    'gemini_2_5_pro': 'Gemini 2.5 Pro',
    'gemini_3_pro': 'Gemini 3 Pro',
    'gemini_2_0_flash': 'Gemini 2.0 Flash',
    'gemini_2_5_flash': 'Gemini 2.5 Flash',
    'gemini_2_5_flash_lite': 'Gemini 2.5 Flash Lite',
    'qwen3_235B': 'Qwen3-VL-235B-A22B-Instruct',
    'llama4_maverick': 'Llama-4-Maverick-17B-128E-Instruct',
    'gpt_5_1_no_location': 'GPT-5.1',
    'gpt_5_1': 'GPT-5.1',
}

def get_canonical_model_name(model_folder_name):
    """Convert model folder name to canonical name."""
    # Remove llm_verification_ prefix if present
    model_key = model_folder_name.replace('llm_verification_', '')
    # Remove common suffixes like _no_location, _with_location
    base_key = model_key.replace('_no_location', '').replace('_with_location', '')
    
    # Try exact match first
    if model_folder_name in MODEL_NAME_MAPPING:
        return MODEL_NAME_MAPPING[model_folder_name]
    elif model_key in MODEL_NAME_MAPPING:
        return MODEL_NAME_MAPPING[model_key]
    elif base_key in MODEL_NAME_MAPPING:
        return MODEL_NAME_MAPPING[base_key]
    # If no mapping found, return cleaned name
    return base_key.replace('_', ' ').title()

def find_ablation_folders(base_dir):
    """Find all ablation folders in the data directory."""
    base_path = Path(base_dir)
    ablation_folders = []
    
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            ablation_folders.append(item.name)
    
    return sorted(ablation_folders)

def find_models_in_ablation(ablation_path):
    """Find all model folders in an ablation directory."""
    models = []
    for item in ablation_path.iterdir():
        if item.is_dir() and item.name.startswith('llm_verification_'):
            models.append(item.name)
    return sorted(models)

def normalize_model_name(model_folder_name):
    """Normalize model folder name by removing suffixes for comparison."""
    # Remove common suffixes
    normalized = model_folder_name.replace('_no_location', '').replace('_with_location', '')
    return normalized

def load_score_differences(csv_path):
    """Load and return score differences CSV."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df

def calculate_statistics(df):
    """Calculate all required statistics from score differences dataframe."""
    if df is None or len(df) == 0:
        return None
    
    stats = {}
    
    # Overall effect: sum of all differences (positive - negative)
    stats['overall_effect'] = df['difference'].sum()
    
    # Standard error for overall effect (sum): SE = std * sqrt(n)
    std_diff = df['difference'].std()
    n = len(df)
    stats['overall_effect_se'] = std_diff * np.sqrt(n) if n > 0 else 0
    
    # Local effect: average difference
    stats['local_effect'] = df['difference'].mean()
    stats['local_effect_std'] = df['difference'].std()
    
    # Count positive and negative differences
    positive_count = (df['difference'] > 0).sum()
    negative_count = (df['difference'] < 0).sum()
    zero_count = (df['difference'] == 0).sum()
    total_count = len(df)
    
    stats['positive_count'] = positive_count
    stats['negative_count'] = negative_count
    stats['zero_count'] = zero_count
    stats['total_count'] = total_count
    
    # Micro-averaged Overall effect: fraction of positive differences
    # Group by paper_folder and calculate fraction of positive differences per paper
    paper_stats = df.groupby('paper_folder').agg({
        'difference': lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
    }).reset_index()
    paper_stats.columns = ['paper_folder', 'fraction_positive']
    
    stats['micro_avg_overall_effect'] = paper_stats['fraction_positive'].mean()
    stats['micro_avg_std'] = paper_stats['fraction_positive'].std()
    stats['micro_avg_se'] = stats['micro_avg_std'] / np.sqrt(len(paper_stats)) if len(paper_stats) > 0 else 0
    stats['num_papers'] = len(paper_stats)
    
    # Additional statistics
    stats['mean_difference'] = df['difference'].mean()
    stats['median_difference'] = df['difference'].median()
    stats['std_difference'] = df['difference'].std()
    
    return stats

def analyze_all_models(base_dir):
    """Analyze all models across all ablation scenarios."""
    base_path = Path(base_dir)
    
    # Find all ablation folders
    ablation_folders = find_ablation_folders(base_path)
    
    if 'reference' not in ablation_folders:
        print("Warning: 'reference' folder not found. Using first folder as reference.")
        reference_name = ablation_folders[0] if ablation_folders else None
    else:
        reference_name = 'reference'
    
    if reference_name is None:
        print("Error: No ablation folders found.")
        return pd.DataFrame()
    
    # Collect all unique models across all ablations (normalized)
    all_models_map = {}  # normalized_name -> list of actual folder names per ablation
    for ablation_name in ablation_folders:
        ablation_path = base_path / ablation_name
        models = find_models_in_ablation(ablation_path)
        for model_folder in models:
            normalized = normalize_model_name(model_folder)
            if normalized not in all_models_map:
                all_models_map[normalized] = {}
            all_models_map[normalized][ablation_name] = model_folder
    
    results = []
    
    # For each normalized model, try to find it in each ablation
    for normalized_model, ablation_folders_dict in all_models_map.items():
        canonical_name = get_canonical_model_name(normalized_model)
        
        for ablation_name in ablation_folders:
            # Try to find the model in this ablation
            if ablation_name in ablation_folders_dict:
                model_folder = ablation_folders_dict[ablation_name]
            else:
                # Try to find any variant of this model
                model_folder = None
                for variant_folder in ablation_folders_dict.values():
                    if normalize_model_name(variant_folder) == normalized_model:
                        model_folder = variant_folder
                        break
                if model_folder is None:
                    continue
            
            model_path = base_path / ablation_name / model_folder
            csv_path = model_path / 'score_differences.csv'
            
            df = load_score_differences(csv_path)
            stats = calculate_statistics(df)
            
            if stats is not None:
                results.append({
                    'model': canonical_name,
                    'model_folder': model_folder,
                    'ablation': ablation_name,
                    **stats
                })
    
    return pd.DataFrame(results)

def create_comparison_plots(results_df, output_dir, reference_name='reference'):
    """Create comparison plots with all ablations in the same plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all ablations
    all_ablations = sorted(results_df['ablation'].unique())
    models = sorted(results_df['model'].unique())
    
    # Prepare data: pivot to have each ablation as a column
    pivot_overall = results_df.pivot_table(
        index='model', 
        columns='ablation', 
        values='overall_effect', 
        aggfunc='first'
    )
    pivot_overall_se = results_df.pivot_table(
        index='model', 
        columns='ablation', 
        values='overall_effect_se', 
        aggfunc='first'
    )
    
    pivot_micro = results_df.pivot_table(
        index='model', 
        columns='ablation', 
        values='micro_avg_overall_effect', 
        aggfunc='first'
    )
    pivot_micro_se = results_df.pivot_table(
        index='model', 
        columns='ablation', 
        values='micro_avg_se', 
        aggfunc='first'
    )
    
    # Sort by reference values (low to high)
    if reference_name in pivot_overall.columns:
        sort_order = pivot_overall[reference_name].sort_values(ascending=True)
        pivot_overall = pivot_overall.loc[sort_order.index]
        pivot_overall_se = pivot_overall_se.loc[sort_order.index]
    
    if reference_name in pivot_micro.columns:
        sort_order_micro = pivot_micro[reference_name].sort_values(ascending=True)
        pivot_micro = pivot_micro.loc[sort_order_micro.index]
        pivot_micro_se = pivot_micro_se.loc[sort_order_micro.index]
    
    model_order = pivot_overall.index.tolist()
    model_order_micro = pivot_micro.index.tolist()
    
    # Define colors and markers for each ablation
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # Plot 1: Overall Effect Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(model_order))
    
    # Plot each ablation
    for idx, ablation in enumerate(all_ablations):
        if ablation not in pivot_overall.columns:
            continue
        
        values = pivot_overall[ablation].values
        errors = pivot_overall_se[ablation].values * 1.96  # 95% CI
        
        label = ablation.replace('_', ' ').title()
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.errorbar(x_pos, values, yerr=errors,
                    marker=marker, linewidth=2, markersize=8, label=label, 
                    alpha=0.8, color=color, capsize=5, capthick=2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, fontsize=13, rotation=45, ha='right')
    ax.set_ylabel('Overall Effect', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Overall Effect Comparison Across All Ablations', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_name = 'overall_effect_comparison_all.png'
    plt.savefig(output_path / plot_name, bbox_inches='tight')
    print(f"Saved plot: {output_path / plot_name}")
    plt.close()
    
    # Plot 2: Micro-averaged Overall Effect with Error Bars
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(model_order_micro))
    
    # Plot each ablation
    for idx, ablation in enumerate(all_ablations):
        if ablation not in pivot_micro.columns:
            continue
        
        values = pivot_micro[ablation].values
        errors = pivot_micro_se[ablation].values * 1.96  # 95% CI
        
        label = ablation.replace('_', ' ').title()
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.errorbar(x_pos, values, yerr=errors,
                    marker=marker, linewidth=2, markersize=8, label=label, 
                    alpha=0.8, color=color, capsize=5, capthick=2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order_micro, fontsize=13, rotation=45, ha='right')
    ax.set_ylabel('Micro-averaged Overall Effect (Fraction of Positive Differences)', 
                  fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Micro-averaged Overall Effect with 95% Confidence Intervals', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.set_ylim(0.1, 0.55)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_name = 'micro_avg_overall_effect_comparison_all.png'
    plt.savefig(output_path / plot_name, bbox_inches='tight')
    print(f"Saved plot: {output_path / plot_name}")
    plt.close()

def create_summary_table(results_df, output_dir):
    """Create a summary table of all results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary table
    summary_cols = ['model', 'ablation', 'overall_effect', 'local_effect', 
                    'micro_avg_overall_effect', 'micro_avg_se', 'positive_count', 
                    'negative_count', 'total_count', 'num_papers']
    
    summary_df = results_df[summary_cols].copy()
    
    # Round numeric columns
    numeric_cols = ['overall_effect', 'local_effect', 'micro_avg_overall_effect', 'micro_avg_se']
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(3)
    
    # Save as CSV
    summary_df.to_csv(output_path / 'model_comparison_summary.csv', index=False)
    print(f"Saved summary table: {output_path / 'model_comparison_summary.csv'}")
    
    return summary_df

def main():
    base_dir = '/Users/ktgiahieu/Documents/paper-review-assistant/experiments/data/for_plotting/data'
    output_dir = '/Users/ktgiahieu/Documents/paper-review-assistant/experiments/data/for_plotting/model_comparison_results'
    
    print("Finding ablation folders...")
    base_path = Path(base_dir)
    ablation_folders = find_ablation_folders(base_path)
    print(f"Found {len(ablation_folders)} ablation folders: {ablation_folders}")
    
    print("\nAnalyzing all models...")
    results_df = analyze_all_models(base_dir)
    
    if len(results_df) == 0:
        print("No results found. Please check the directory structure.")
        return
    
    print(f"\nAnalyzed {len(results_df)} model/ablation combinations")
    print("\nResults summary:")
    print(results_df[['model', 'ablation', 'overall_effect', 'local_effect', 
                      'micro_avg_overall_effect']].to_string())
    
    print("\nCreating summary tables...")
    summary_df = create_summary_table(results_df, output_dir)
    
    print("\nCreating comparison plots...")
    create_comparison_plots(results_df, output_dir, reference_name='reference')
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
