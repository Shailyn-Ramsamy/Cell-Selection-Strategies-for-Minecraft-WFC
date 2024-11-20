import numpy as np
from collections import Counter, defaultdict
from scipy.stats import entropy
from scipy.special import rel_entr
import networkx
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import pickle

@dataclass
class WFCMetricsAnalyzer:
    def __init__(self, grid):
        self.grid = grid
        self.shape = (len(grid), len(grid[0]), len(grid[0][0]))
        
    def get_cell_value(self, x: int, y: int, z: int) -> Tuple[str, int, int]:
        """
        Get the module, rotation, and variation for a cell if it's collapsed,
        filtering out unwanted patterns
        """
        cell = self.grid[z][y][x]
        if len(cell) == 1:
            pattern = next(iter(cell))
            module, rotation, variation = pattern
            
            # Filter out dirt tiles completely
            if module == 'dirt':
                return None
                
            # Filter air tiles - only keep variations 1 and 2
            if module == 'air':
                if variation == 0:  # Skip air variation 0
                    return None
                return pattern  # Keep air variations 1 and 2
                
            # Keep all other patterns
            return pattern
        return None
    
    def calculate_kl_divergence_metrics(self) -> Dict:
        """Calculate KL divergence based metrics"""
        # Get pattern distributions per layer
        layer_distributions = []
        
        for z in range(self.shape[0]):
            layer_patterns = []
            for y in range(self.shape[1]):
                for x in range(self.shape[2]):
                    cell_value = self.get_cell_value(x, y, z)
                    if cell_value:
                        layer_patterns.append(cell_value)
            
            if layer_patterns:
                # Calculate probability distribution for this layer
                counter = Counter(layer_patterns)
                total = len(layer_patterns)
                dist = {k: v/total for k, v in counter.items()}
                layer_distributions.append(dist)

        if not layer_distributions:
            return {
                'average_kl_divergence': 0,
                'max_kl_divergence': 0,
                'layer_kl_divergences': [],
                'uniform_kl_divergence': 0
            }

        # Calculate KL divergence between adjacent layers
        kl_divergences = []
        for i in range(len(layer_distributions) - 1):
            # Get all unique patterns from both layers
            patterns = set(layer_distributions[i].keys()) | set(layer_distributions[i + 1].keys())
            
            # Create probability vectors with smoothing
            epsilon = 1e-10  # Small value to avoid log(0)
            p = np.array([layer_distributions[i].get(p, epsilon) for p in patterns])
            q = np.array([layer_distributions[i + 1].get(p, epsilon) for p in patterns])
            
            # Normalize
            p = p / p.sum()
            q = q / q.sum()
            
            # Calculate KL divergence
            kl_div = sum(rel_entr(p, q))
            kl_divergences.append(kl_div)
            print(f"KL divergence between layers {i} and {i+1}: {kl_div:.3f}")

        # Calculate KL divergence from uniform distribution
        # Get all unique patterns across all layers
        all_patterns = set()
        for dist in layer_distributions:
            all_patterns.update(dist.keys())
        
        uniform_prob = 1.0 / len(all_patterns)
        uniform_dist = {p: uniform_prob for p in all_patterns}
        
        # Calculate KL divergence from uniform for each layer
        uniform_kl_divergences = []
        for layer_dist in layer_distributions:
            # Create probability vectors
            patterns = all_patterns
            p = np.array([layer_dist.get(p, epsilon) for p in patterns])
            q = np.array([uniform_dist[p] for p in patterns])
            
            # Normalize
            p = p / p.sum()
            q = q / q.sum()
            
            # Calculate KL divergence
            kl_div = sum(rel_entr(p, q))
            uniform_kl_divergences.append(kl_div)

        return {
            'average_kl_divergence': np.mean(kl_divergences) if kl_divergences else 0,
            'max_kl_divergence': np.max(kl_divergences) if kl_divergences else 0,
            'layer_kl_divergences': kl_divergences,
            'uniform_kl_divergence': np.mean(uniform_kl_divergences)
        }
        
    def calculate_pattern_metrics(self) -> Dict:
        """Calculate pattern-based metrics"""
        patterns = []
        pattern_positions = defaultdict(list)  # Track positions for debugging
        
        for z in range(self.shape[0]):
            for y in range(self.shape[1]):
                for x in range(self.shape[2]):
                    cell_value = self.get_cell_value(x, y, z)
                    if cell_value:
                        patterns.append(cell_value)
                        pattern_positions[cell_value].append((x, y, z))
        
        if not patterns:
            return {
                'pattern_diversity': 0,
                'pattern_richness': 0,
                'pattern_evenness': 0,
                'most_common_patterns': [],
                'total_patterns': 0
            }
        
        pattern_counts = Counter(patterns)
        total_patterns = len(patterns)
        
        # Debug print
        print(f"\nTotal patterns found: {total_patterns}")
        print(f"Unique patterns: {len(pattern_counts)}")
        
        pattern_probs = [count/total_patterns for count in pattern_counts.values()]
        pattern_diversity = entropy(pattern_probs)
        max_entropy = np.log(len(pattern_counts))
        pattern_evenness = pattern_diversity / max_entropy if max_entropy > 0 else 0
        
        return {
            'pattern_diversity': pattern_diversity,
            'pattern_richness': len(pattern_counts),
            'pattern_evenness': pattern_evenness,
            'most_common_patterns': pattern_counts.most_common(5),
            'total_patterns': total_patterns
        }
    
    def calculate_transition_metrics(self) -> Dict:
        """Calculate transition metrics based on adjacent tile patterns"""
        transitions = defaultdict(int)
        unique_patterns = set()
        
        # Helper function to get pattern from cell
        def get_pattern(cell):
            if len(cell) == 1:
                return next(iter(cell))  # Returns the (module, rotation, variation) tuple
            return None
        
        # Count adjacent pattern transitions
        for z in range(self.shape[0]):
            for y in range(self.shape[1]):
                for x in range(self.shape[2]):
                    current_pattern = get_pattern(self.grid[z][y][x])
                    if not current_pattern:
                        continue
                    
                    unique_patterns.add(current_pattern)
                    
                    # Check right neighbor
                    if x + 1 < self.shape[2]:
                        right_pattern = get_pattern(self.grid[z][y][x + 1])
                        if right_pattern:
                            transition = tuple(sorted([current_pattern, right_pattern]))
                            transitions[transition] += 1
                    
                    # Check forward neighbor
                    if y + 1 < self.shape[1]:
                        forward_pattern = get_pattern(self.grid[z][y + 1][x])
                        if forward_pattern:
                            transition = tuple(sorted([current_pattern, forward_pattern]))
                            transitions[transition] += 1
                    
                    # Check upward neighbor
                    if z + 1 < self.shape[0]:
                        up_pattern = get_pattern(self.grid[z + 1][y][x])
                        if up_pattern:
                            transition = tuple(sorted([current_pattern, up_pattern]))
                            transitions[transition] += 1
        
        # Calculate metrics
        total_patterns = len(unique_patterns)
        total_transitions = sum(transitions.values())
        unique_transitions = len(transitions)
        
        # Calculate connectivity ratio
        possible_transitions = (total_patterns * (total_patterns - 1)) // 2
        connectivity_ratio = unique_transitions / possible_transitions if possible_transitions > 0 else 0
        
        # Group transitions by module type
        module_transitions = defaultdict(int)
        for (p1, p2), count in transitions.items():
            module_pair = tuple(sorted([p1[0], p2[0]]))  # Just the module names
            module_transitions[module_pair] += count
        
        # Find most common transitions
        most_common_transitions = sorted(
            transitions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Calculate module-level statistics
        module_counts = Counter(p[0] for p in unique_patterns)
        most_common_modules = module_counts.most_common()
        
        print("\nTransition Analysis:")
        print(f"Total unique patterns (module+rotation+variation): {total_patterns}")
        print(f"Unique modules: {len(module_counts)}")
        print(f"Total transitions observed: {total_transitions}")
        print(f"Unique transition pairs: {unique_transitions}")
        print(f"Connectivity ratio: {connectivity_ratio:.3f}")
        
        print("\nModule Distribution:")
        for module, count in most_common_modules:
            print(f"  {module}: {count} patterns")
        
        print("\nMost Common Transitions:")
        for (p1, p2), count in most_common_transitions:
            print(f"  {p1[0]} (rot:{p1[1]}, var:{p1[2]}) <-> {p2[0]} (rot:{p2[1]}, var:{p2[2]}): {count}")
        
        return {
            'unique_patterns': total_patterns,
            'total_transitions': total_transitions,
            'unique_transitions': unique_transitions,
            'connectivity_ratio': connectivity_ratio,
            'module_counts': dict(module_counts),
            'most_common_transitions': most_common_transitions,
            'module_transitions': dict(module_transitions)
        }

    
    def calculate_spatial_metrics(self) -> Dict:
        """Calculate spatial organization metrics"""
        def calculate_layer_similarity(z1: int, z2: int) -> float:
            layer1_patterns = set()
            layer2_patterns = set()
            
            for y in range(self.shape[1]):
                for x in range(self.shape[2]):
                    v1 = self.get_cell_value(x, y, z1)
                    v2 = self.get_cell_value(x, y, z2)
                    if v1: layer1_patterns.add(str(v1))
                    if v2: layer2_patterns.add(str(v2))
            
            if not layer1_patterns and not layer2_patterns:
                return 0
                
            intersection = layer1_patterns.intersection(layer2_patterns)
            union = layer1_patterns.union(layer2_patterns)
            
            # Debug print for layers
            print(f"\nLayer {z1} patterns: {len(layer1_patterns)}")
            print(f"Layer {z2} patterns: {len(layer2_patterns)}")
            print(f"Common patterns: {len(intersection)}")
            
            return len(intersection) / len(union)
        
        # Calculate vertical stratification
        vertical_similarities = []
        for z in range(self.shape[0] - 1):
            similarity = calculate_layer_similarity(z, z + 1)
            vertical_similarities.append(similarity)
            print(f"Similarity between layers {z} and {z+1}: {similarity:.3f}")
        
        # Calculate pattern density per layer
        layer_densities = []
        total_cells_per_layer = self.shape[1] * self.shape[2]
        
        for z in range(self.shape[0]):
            collapsed_count = sum(1 for x in range(self.shape[2]) 
                                for y in range(self.shape[1]) 
                                if self.get_cell_value(x, y, z) is not None)
            density = collapsed_count / total_cells_per_layer if total_cells_per_layer > 0 else 0
            layer_densities.append(density)
            print(f"Layer {z} density: {density:.3f} ({collapsed_count}/{total_cells_per_layer} cells)")
            
        return {
            'vertical_stratification': np.mean(vertical_similarities) if vertical_similarities else 0,
            'layer_density_variation': np.std(layer_densities) if layer_densities else 0,
            'average_layer_density': np.mean(layer_densities) if layer_densities else 0,
            'layer_densities': layer_densities
        }
    
    def analyze_grid(self) -> Dict:
        print("Starting grid analysis...")
        print(f"Grid dimensions: {self.shape}")
        metrics = {
            'pattern_metrics': self.calculate_pattern_metrics(),
            'transition_metrics': self.calculate_transition_metrics(),
            'spatial_metrics': self.calculate_spatial_metrics(),
            'kl_metrics': self.calculate_kl_divergence_metrics()  # Add KL metrics
        }
        return metrics

def print_analysis_results(metrics: Dict) -> None:
    """Print formatted analysis results"""
    print("\nWFC Grid Analysis Results")
    print("=" * 30)
    
    print("\nPattern Metrics:")
    print("-" * 20)
    pm = metrics['pattern_metrics']
    print(f"Total Patterns: {pm['total_patterns']}")
    print(f"Pattern Diversity: {pm['pattern_diversity']:.3f}")
    print(f"Pattern Richness: {pm['pattern_richness']}")
    print(f"Pattern Evenness: {pm['pattern_evenness']:.3f}")
    
    print("\nMost Common Patterns:")
    for pattern, count in pm['most_common_patterns']:
        print(f"  {pattern}: {count} occurrences ({count/pm['total_patterns']*100:.1f}%)")
    
    print("\nTransition Metrics:")
    print("-" * 20)
    tm = metrics['transition_metrics']
    print(f"Unique Patterns: {tm['unique_patterns']}")
    print(f"Total Transitions: {tm['total_transitions']}")
    print(f"Unique Transitions: {tm['unique_transitions']}")
    print(f"Connectivity Ratio: {tm['connectivity_ratio']:.3f}")
    
    # print("\nMost Common Module Types:")
    # for module, count in Counter(tm['module_counts']).most_common():
    #     print(f"  {module}: {count} patterns")
    
    # print("\nMost Common Transitions:")
    # for (p1, p2), count in tm['most_common_transitions']:
    #     print(f"  {p1[0]} <-> {p2[0]}: {count} occurrences")
    
    print("\nSpatial Metrics:")
    print("-" * 20)
    sm = metrics['spatial_metrics']
    print(f"Vertical Stratification: {sm['vertical_stratification']:.3f}")
    print(f"Layer Density Variation: {sm['layer_density_variation']:.3f}")
    print(f"Average Layer Density: {sm['average_layer_density']:.3f}")
    
    print("\nKL Divergence Metrics:")
    print("-" * 20)
    kl = metrics['kl_metrics']
    print(f"Average KL Divergence between layers: {kl['average_kl_divergence']:.3f}")
    print(f"Maximum KL Divergence between layers: {kl['max_kl_divergence']:.3f}")
    print(f"Average KL Divergence from uniform: {kl['uniform_kl_divergence']:.3f}")
    
    print("\nLayer-wise KL Divergences:")
    for i, div in enumerate(kl['layer_kl_divergences']):
        print(f"  Layers {i} -> {i+1}: {div:.3f}")
        
def load_wfc_grid(filename: str):
    with open(filename, 'rb') as f:
        grid = pickle.load(f)
    return grid

if __name__ == "__main__":
    try:
        # Load the grid
        grid = load_wfc_grid('wfc_grid.pkl')
        print(grid[0][0][0])
        
        # Run analysis
        analyzer = WFCMetricsAnalyzer(grid)
        metrics = analyzer.analyze_grid()
        print_analysis_results(metrics)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
    