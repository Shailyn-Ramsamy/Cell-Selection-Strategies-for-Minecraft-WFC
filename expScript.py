import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from wfcpls import EnhancedWFC3DBuilder
from glm import ivec3
from gdpc import Editor
from newMetrics import WFCMetricsAnalyzer
from scipy.signal import savgol_filter

def run_strategy_experiment(args):
    strategy, initial_size, num_runs = args
    strategy_results = []
    current_size = initial_size
    
    print(f"\nStarting strategy: {strategy}")
    
    for run in range(num_runs):
        # Update dimensions every 20 runs
        if run > 0 and run % 20 == 0:
            current_size += 1
            print(f"\nStrategy {strategy}: Increasing dimensions to {current_size}x{current_size}x{current_size}")
        
        print(f"Strategy {strategy}: Run {run+1}/{num_runs} - Grid size: {current_size}x{current_size}x{current_size}")
        
        builder = EnhancedWFC3DBuilder('adjacencies.json', strategy=strategy)
        editor = Editor(buffering=True)
        start_pos = ivec3(0, -67, 0)
        length, width, height = current_size, current_size, current_size
        result = builder.generate_grid(editor, length, width, height, start_pos)
        
        editor.flushBuffer()
        
        analyzer = WFCMetricsAnalyzer(result)
        metrics = analyzer.analyze_grid()
        
        run_results = {
            'strategy': strategy,
            'run': run,
            'grid_size': current_size,
            'grid_label': f'{current_size}×{current_size}×{current_size}',
            'pattern_diversity': metrics['pattern_metrics']['pattern_diversity'],
            'pattern_richness': metrics['pattern_metrics']['pattern_richness'],
            'pattern_evenness': metrics['pattern_metrics']['pattern_evenness'],
            'unique_patterns': metrics['transition_metrics']['unique_patterns'],
            'total_transitions': metrics['transition_metrics']['total_transitions'],
            'unique_transitions': metrics['transition_metrics']['unique_transitions'],
            'connectivity_ratio': metrics['transition_metrics']['connectivity_ratio'],
            'vertical_stratification': metrics['spatial_metrics']['vertical_stratification'],
            'layer_density_variation': metrics['spatial_metrics']['layer_density_variation'],
            'average_layer_density': metrics['spatial_metrics']['average_layer_density'],
            'average_kl_divergence': metrics['kl_metrics']['average_kl_divergence'],
            'max_kl_divergence': metrics['kl_metrics']['max_kl_divergence'],
            'uniform_kl_divergence': metrics['kl_metrics']['uniform_kl_divergence']
        }
        
        strategy_results.append(run_results)
    
    return strategy_results

def run_experiments_parallel(strategies, num_runs=120, initial_size=8):
    # Determine number of processes to use (one per strategy, up to CPU count)
    num_processes = min(len(strategies), cpu_count())
    print(f"Running experiments with {num_processes} parallel processes")
    
    # Create argument list for each strategy
    args_list = [(strategy, initial_size, num_runs) for strategy in strategies]
    
    # Create process pool and run experiments
    with Pool(processes=num_processes) as pool:
        all_results = pool.map(run_strategy_experiment, args_list)
    
    # Flatten results list
    flat_results = [result for strategy_results in all_results for result in strategy_results]
    
    # Convert to DataFrame
    df = pd.DataFrame(flat_results)
    return df
        
        
def plot_semi_normalized_metric(df, metric_name, normalization='log', save_dir='semi_normalized_plots'):
    """
    Create plots with slight normalization for better visualization.
    
    normalization options:
    - 'log': Use log scale for y-axis
    - 'volume_ratio': Divide by volume but multiply by a scaling factor
    - 'baseline': Normalize relative to smallest volume but preserve scaling
    """
    import os
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("husl", n_colors=len(df['strategy'].unique()))
    
    for strat, color in zip(df['strategy'].unique(), colors):
        strat_data = df[df['strategy'] == strat].copy()
        grouped_data = strat_data.groupby('grid_size').agg({
            metric_name: 'mean'
        }).reset_index()
        
        volumes = grouped_data['grid_size'] ** 3
        raw_values = grouped_data[metric_name] * volumes
        
        if normalization == 'log':
            # Just use log scale for y-axis
            values_to_plot = raw_values
            plt.yscale('log')
            
        elif normalization == 'volume_ratio':
            # Normalize by volume but multiply by a scaling factor to keep numbers meaningful
            scale_factor = volumes.max()  # or choose another scaling factor
            values_to_plot = (raw_values / volumes) * scale_factor
            
        elif normalization == 'baseline':
            # Normalize relative to smallest volume but preserve scaling
            baseline = raw_values.iloc[0] / volumes.iloc[0]
            values_to_plot = raw_values / baseline
            
        plt.plot(volumes, values_to_plot, label=strat, color=color, linewidth=2)
    
    title_suffix = {
        'log': '(Log Scale)',
        'volume_ratio': '(Volume Adjusted)',
        'baseline': '(Baseline Normalized)'
    }
    
    plt.title(f"{metric_name.replace('_', ' ').title()} {title_suffix[normalization]}", 
              pad=20, fontsize=12)
    plt.xlabel('Volume (blocks)', fontsize=10)
    ylabel_prefix = {
        'log': 'Raw Value',
        'volume_ratio': 'Scaled Value',
        'baseline': 'Relative Growth'
    }
    plt.ylabel(ylabel_prefix[normalization], fontsize=10)
    
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{normalization}_{metric_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_semi_normalized_metrics(df, normalization='log'):
    """Plot all metrics with the specified normalization."""
    metrics = [
        'pattern_diversity', 'pattern_richness', 'pattern_evenness',
        'unique_patterns', 'total_transitions', 'unique_transitions',
        'connectivity_ratio', 'vertical_stratification', 
        'layer_density_variation', 'average_layer_density',
        'average_kl_divergence', 'max_kl_divergence', 'uniform_kl_divergence'
    ]
    
    for metric in metrics:
        plot_semi_normalized_metric(df, metric, normalization)        

if __name__ == "__main__":
    strategies = ['collapsed_neighbors', 'height_priority', 'from_center', 
                 'from_corners', 'alternating_layers', 'random_walk']
    
    # # Run experiments in parallel and get results DataFrame
    # results_df = run_experiments_parallel(strategies, num_runs=120, initial_size=8)
    
    # # Save results to CSV
    # results_df.to_csv('experiment_results.csv', index=False)
    results_df = pd.read_csv('experiment_results.csv')
    # plot_all_raw_metrics(results_df)
    
    # # Option 1: Log scale (maintains upward trend but makes large numbers more manageable)
    plot_all_semi_normalized_metrics(results_df, normalization='log')

    # # Option 2: Volume ratio (scales relative to volume but multiplied by max volume to keep numbers meaningful)
    # plot_all_semi_normalized_metrics(results_df, normalization='volume_ratio')

    # # Option 3: Baseline (shows growth relative to smallest volume)
    # plot_all_semi_normalized_metrics(results_df, normalization='baseline')

    # # Or for a single metric:
    # plot_semi_normalized_metric(results_df, 'pattern_diversity', normalization='log')
    
    # analysis_results = analyze_all_metrics_by_volume(results_df, save_dir='volume_analysis')
    # plot_pattern_distribution(results_df)
    
    # # Create overview plot
    # fig = plot_metrics_enhanced(results_df)
    # fig.savefig('metric_plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # # Create individual metric plots
    # for metric in results_df.columns[4:]:  # Skip non-metric columns
    #     fig = plot_individual_metric_enhanced(results_df, metric)
    #     fig.savefig(f'metric_plots/metric_{metric}.png', dpi=300, bbox_inches='tight')
    #     plt.close()