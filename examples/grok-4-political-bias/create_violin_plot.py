#!/usr/bin/env python3
"""
Script to create a violin plot of political bias scores from promptfoo results.
Extracts scores from results.json and creates a visualization similar to the reference image.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any

def extract_scores_from_results(json_file: str) -> pd.DataFrame:
    """
    Extract political bias scores from promptfoo results.json file.
    
    Args:
        json_file: Path to the results.json file
        
    Returns:
        DataFrame with columns: model, score, judge
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    scores_data = []
    
    # Extract scores from the results array
    for result in data['results']['results']:
        if result.get('success') and result.get('namedScores'):
            provider = result['provider']['id']
            # Map provider IDs to display names
            model_name = map_provider_to_display_name(provider)
            
            # Extract scores from all judges
            for judge, score in result['namedScores'].items():
                if isinstance(score, (int, float)):
                    scores_data.append({
                        'model': model_name,
                        'score': score,
                        'judge': judge
                    })
    
    return pd.DataFrame(scores_data)

def map_provider_to_display_name(provider_id: str) -> str:
    """
    Map provider IDs to display names for the plot.
    
    Args:
        provider_id: The provider ID from promptfoo
        
    Returns:
        Display name for the model
    """
    mapping = {
        'xai:grok-4': 'Grok-4',
        'openai:gpt-4.1': 'GPT-4.1',
        'google:gemini-2.5-flash': 'Gemini 2.5 Pro',
        'anthropic:claude-opus-4-20250514': 'Claude Opus 4'
    }
    return mapping.get(provider_id, provider_id.split(':')[-1])

def get_model_color(model_name: str) -> str:
    """
    Get the color for each model based on the reference image.
    
    Args:
        model_name: The display name of the model
        
    Returns:
        Hex color code
    """
    colors = {
        'GPT-4.1': '#f08080',  # Red/Pink
        'Gemini 2.5 Pro': '#20B2AA',  # Teal/Aqua
        'Grok-4': '#7ec8e3',  # Light Blue
        'Claude Opus 4': '#90EE90'  # Light Green
    }
    return colors.get(model_name, '#cccccc')  # Default gray if not found

def create_violin_plot(df: pd.DataFrame, output_file: str = 'political_bias_violin_plot.png'):
    """
    Create a violin plot of political bias scores.
    
    Args:
        df: DataFrame with model, score, and judge columns
        output_file: Output file path for the plot
    """
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define the order of models
    model_order = ['GPT-4.1', 'Gemini 2.5 Pro', 'Grok-4', 'Claude Opus 4']
    
    # Filter and order the data according to the specified order
    ordered_data = []
    ordered_model_names = []
    
    for model in model_order:
        model_data = df[df['model'] == model]['score'].values
        if len(model_data) > 0:  # Only include models that have data
            ordered_data.append(model_data)
            ordered_model_names.append(model)
    
    positions = range(len(ordered_model_names))
    
    # Create violin plot
    violin_parts = ax.violinplot(
        ordered_data,
        positions=positions,
        showmeans=False,
        showmedians=True
    )

    # Custom coloring: apply colors based on the reference image
    for i, pc in enumerate(violin_parts['bodies']):
        model_name = ordered_model_names[i]
        color = get_model_color(model_name)
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    # Customize median line
    violin_parts['cmedians'].set_color('pink')
    violin_parts['cmedians'].set_linewidth(2)

    # Set labels and title
    ax.set_xlabel('AI Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Political Score (1.0 = Left, 0.0 = Right)', fontsize=14, fontweight='bold')
    ax.set_title('Political Score Distributions: The Shape of Bias', fontsize=16, fontweight='bold', pad=20)

    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_model_names, fontsize=12)

    # Set y-axis limits and add neutral line, with padding above 0.0 and below 1.0
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='lightgray', linestyle='--', alpha=0.7, linewidth=1)
    # Add faint lines for 0.0 and 1.0
    ax.axhline(y=0.0, color='pink', linestyle='-', linewidth=1)
    ax.axhline(y=1.0, color='pink', linestyle='-', linewidth=1)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotations for special characteristics
    for i, model in enumerate(ordered_model_names):
        model_data = df[df['model'] == model]['score']
        median_score = model_data.median()
        # Add annotations for special characteristics
        if model == 'Grok-4':
            if model_data.std() > 0.3:
                ax.annotate('Bipolar Distribution', 
                          xy=(i, 0.1), xytext=(i, 0.05),
                          ha='center', va='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                          fontsize=10, fontweight='bold')
        elif model == 'Claude Opus 4':
            if abs(median_score - 0.5) < 0.1:
                ax.annotate('Most Centrist', 
                          xy=(i, 0.6), xytext=(i, 0.65),
                          ha='center', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                          fontsize=10, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Violin plot saved as {output_file}")

def main():
    """Main function to run the analysis."""
    # Extract scores from results.json
    print("Extracting scores from results.json...")
    df = extract_scores_from_results('results.json')
    
    if df.empty:
        print("No valid scores found in results.json")
        return
    
    print(f"Found {len(df)} scores from {df['model'].nunique()} models")
    print(f"Models: {list(df['model'].unique())}")
    
    # Display summary statistics
    print("\nSummary statistics by model:")
    print(df.groupby('model')['score'].describe())
    
    # Create violin plot
    print("\nCreating violin plot...")
    create_violin_plot(df)
    
    # Also save the extracted data as CSV for further analysis
    df.to_csv('political_scores.csv', index=False)
    print("Scores saved to political_scores.csv")

if __name__ == "__main__":
    main() 