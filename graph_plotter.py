import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

def set_plot_style():
    """Set global plotting style parameters"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14,  # Base font size
        'axes.titlesize': 16,  # Subplot titles
        'axes.labelsize': 14,  # Axis labels
        'xtick.labelsize': 12,  # X-axis tick labels
        'ytick.labelsize': 12,  # Y-axis tick labels
        'legend.fontsize': 12,  # Legend text
        'legend.title_fontsize': 14,  # Legend title
        'figure.titlesize': 20,  # Figure title
    })

def create_combined_plot_with_variance(df, save_dir='combined_plots'):
    """
    Create a single figure with all four metrics showing mean and variance
    Uses 95% confidence intervals for the error bands
    X-axis is linear scale, Y-axis is log scale
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plot_style()
    
    # Create a 2x2 subplot with larger figure size
    fig, axes = plt.subplots(2, 2, figsize=(20, 9))
    fig.suptitle('Comparison of Cell Selection Strategies\n(with 95% Confidence Intervals)', 
                fontsize=24, y=0.95)
    
    metrics = [
        'pattern_diversity',
        'connectivity_ratio',
        'vertical_stratification',
        'average_kl_divergence'
    ]
    
    colors = sns.color_palette("husl", n_colors=len(df['strategy'].unique()))
    
    for idx, metric_name in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for strat, color in zip(df['strategy'].unique(), colors):
            strat_data = df[df['strategy'] == strat].copy()
            
            # Group by grid size and calculate statistics
            grouped_stats = strat_data.groupby('grid_size').agg({
                metric_name: ['mean', 'std', 'count']
            }).reset_index()
            
            # Calculate confidence intervals (95%)
            grouped_stats['ci'] = 1.96 * grouped_stats[(metric_name, 'std')] / \
                                np.sqrt(grouped_stats[(metric_name, 'count')])
            
            volumes = grouped_stats['grid_size'] ** 3
            means = grouped_stats[(metric_name, 'mean')] * volumes
            ci = grouped_stats['ci'] * volumes
            
            # Plot mean line
            line = ax.plot(volumes, means, label=strat, color=color, linewidth=2.5)
            
            # Add confidence interval bands
            ax.fill_between(volumes, 
                          means - ci,
                          means + ci,
                          color=color,
                          alpha=0.2)
            
            ax.set_yscale('log')
            ax.set_xlabel('Volume (blocks)', fontsize=14, labelpad=10)
            ax.set_ylabel('Value', fontsize=14, labelpad=10)
            ax.set_title(f"{metric_name.replace('_', ' ').title()} (Log Scale)", 
                        pad=20, fontsize=16)
            
            # Increase tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Only show legend on the first subplot
            if idx == 0:
                ax.legend(title='Strategy', 
                         bbox_to_anchor=(1.05, 1),
                         title_fontsize=14,
                         fontsize=12)
    
    # Adjust spacing between subplots
    plt.tight_layout(h_pad=1.5, w_pad=1.5)
    
    # Save plot with high DPI
    plt.savefig(os.path.join(save_dir, 'combined_metrics_with_variance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_metric_plots(df, save_dir='detailed_plots'):
    """
    Create individual plots for each metric with both variance and quartile information
    X-axis is linear scale, Y-axis is log scale
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plot_style()
    
    metrics = [
        'pattern_diversity',
        'connectivity_ratio',
        'vertical_stratification',
        'average_kl_divergence'
    ]
    
    colors = sns.color_palette("husl", n_colors=len(df['strategy'].unique()))
    
    for metric_name in metrics:
        plt.figure(figsize=(16, 10))
        
        for strat, color in zip(df['strategy'].unique(), colors):
            strat_data = df[df['strategy'] == strat].copy()
            
            # Group by grid size and calculate statistics
            grouped_stats = strat_data.groupby('grid_size').agg({
                metric_name: ['mean', 'std', 'count', 
                            lambda x: np.percentile(x, 25),
                            lambda x: np.percentile(x, 75)]
            }).reset_index()
            
            volumes = grouped_stats['grid_size'] ** 3
            means = grouped_stats[(metric_name, 'mean')] * volumes
            std = grouped_stats[(metric_name, 'std')] * volumes
            q1 = grouped_stats[(metric_name, '<lambda_0>')] * volumes
            q3 = grouped_stats[(metric_name, '<lambda_1>')] * volumes
            
            # Plot mean line
            plt.plot(volumes, means, label=strat, color=color, linewidth=2.5)
            
            # Add standard deviation bands
            plt.fill_between(volumes, 
                           means - std,
                           means + std,
                           color=color,
                           alpha=0.1,
                           label='_nolegend_')
            
            # Add quartile bands
            plt.fill_between(volumes,
                           q1,
                           q3,
                           color=color,
                           alpha=0.2,
                           label='_nolegend_')
        
        plt.yscale('log')
        plt.xlabel('Volume (blocks)', fontsize=16, labelpad=10)
        plt.ylabel('Value', fontsize=16, labelpad=10)
        plt.title(f"{metric_name.replace('_', ' ').title()}\nwith Standard Deviation and Quartile Bands",
                 fontsize=20, pad=20)
        
        # Customize legend
        plt.legend(title='Strategy', 
                  title_fontsize=16,
                  fontsize=14,
                  bbox_to_anchor=(1.05, 1))
        
        # Add grid with custom style
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Increase tick label sizes
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        # Save individual plot
        plt.savefig(os.path.join(save_dir, f'{metric_name}_detailed.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Load the results
    results_df = pd.read_csv('experiment_results2.csv')
    
    # Generate both types of plots
    create_combined_plot_with_variance(results_df)
    create_detailed_metric_plots(results_df)