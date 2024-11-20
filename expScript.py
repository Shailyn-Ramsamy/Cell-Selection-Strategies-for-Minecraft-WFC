import pandas as pd
from multiprocessing import Pool, cpu_count
from WFC import EnhancedWFC3DBuilder
from glm import ivec3
from gdpc import Editor
from get_metrics import WFCMetricsAnalyzer

def run_strategy_experiment(args):
    strategy, initial_size, num_runs = args
    strategy_results = []
    current_size = initial_size
    
    print(f"\nStarting strategy: {strategy}")
    
    for run in range(num_runs):
        # Update dimensions every 50 runs
        if run > 0 and run % 40 == 0:
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

def run_experiments_parallel(strategies, num_runs=240, initial_size=8):
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
        
    
if __name__ == "__main__":
    strategies = ['entropy', 'height_priority', 'from_center', 
                 'random_walk']
    
    results_df = run_experiments_parallel(strategies, num_runs=240, initial_size=8)
    
    results_df.to_csv('experiment_results2.csv', index=False)
