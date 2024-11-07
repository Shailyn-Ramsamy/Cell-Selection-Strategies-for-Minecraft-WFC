import numpy as np
from collections import Counter, defaultdict
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Dict, List, Tuple
import networkx
from sklearn.cluster import DBSCAN

@dataclass
class BuildingMetrics:
    def __init__(self, encoded_tiles: np.ndarray, pattern_map: Dict, block_array: np.ndarray = None):
        self.encoded_tiles = encoded_tiles
        self.pattern_map = pattern_map
        self.block_array = block_array
        self.shape = encoded_tiles.shape
        
    def calculate_complexity_metrics(self) -> Dict:
        """
        Calculate complexity-related metrics
        - Pattern transition complexity
        - Structural complexity
        - Interface complexity
        """
        # Calculate pattern transitions
        transitions = defaultdict(int)
        for axis in range(3):
            slices1 = [slice(None)] * 3
            slices2 = [slice(None)] * 3
            slices1[axis] = slice(None, -1)
            slices2[axis] = slice(1, None)
            
            patterns1 = self.encoded_tiles[tuple(slices1)].flatten()
            patterns2 = self.encoded_tiles[tuple(slices2)].flatten()
            
            for p1, p2 in zip(patterns1, patterns2):
                if p1 != -1 and p2 != -1:
                    transitions[(min(p1, p2), max(p1, p2))] += 1

        # Create transition graph using networkx directly
        G = networkx.Graph()
        for (p1, p2), weight in transitions.items():
            G.add_edge(p1, p2, weight=weight)

        # Calculate interface complexity (surface area to volume ratio)
        surface_area = 0
        volume = 0
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    if self.encoded_tiles[x,y,z] != -1:
                        volume += 1
                        # Check if on edge
                        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                            nx, ny, nz = x+dx, y+dy, z+dz
                            if (nx < 0 or nx >= self.shape[0] or
                                ny < 0 or ny >= self.shape[1] or
                                nz < 0 or nz >= self.shape[2] or
                                self.encoded_tiles[nx,ny,nz] == -1):
                                surface_area += 1

        try:
            graph_density = networkx.density(G)
            avg_clustering = networkx.average_clustering(G)
        except:
            graph_density = 0
            avg_clustering = 0

        return {
            'transition_complexity': len(transitions),
            'graph_density': graph_density,
            'average_clustering': avg_clustering,
            'surface_to_volume_ratio': surface_area/volume if volume > 0 else 0
        }

    def calculate_variety_metrics(self) -> Dict:
        """
        Calculate variety-related metrics
        """
        # Get pattern frequencies
        pattern_freqs = Counter(p for p in self.encoded_tiles.flatten() if p != -1)
        total_patterns = sum(pattern_freqs.values())
        
        if total_patterns == 0:
            return {
                'pattern_diversity': 0,
                'pattern_richness': 0,
                'pattern_evenness': 0,
                'block_diversity': 0,
                'unique_blocks': 0,
                'block_distribution': {}
            }
        
        # Calculate Shannon diversity for patterns
        pattern_probs = [count/total_patterns for count in pattern_freqs.values()]
        pattern_diversity = entropy(pattern_probs)
        
        # Calculate Pielou's evenness
        max_entropy = np.log(len(pattern_freqs))
        pattern_evenness = pattern_diversity / max_entropy if max_entropy > 0 else 0
        
        # Calculate block diversity
        block_counts = defaultdict(int)
        for pattern_id, count in pattern_freqs.items():
            pattern_info = next(info for info in self.pattern_map.values() 
                              if info['pattern_id'] == pattern_id)
            for block, block_count in pattern_info['block_counts'].items():
                block_counts[block] += block_count * count
        
        block_probs = [count/sum(block_counts.values()) for count in block_counts.values()]
        block_diversity = entropy(block_probs)

        return {
            'pattern_diversity': pattern_diversity,
            'pattern_richness': len(pattern_freqs),
            'pattern_evenness': pattern_evenness,
            'block_diversity': block_diversity,
            'unique_blocks': len(block_counts),
            'block_distribution': dict(block_counts)
        }

    def calculate_spatial_metrics(self) -> Dict:
        """
        Calculate spatial organization metrics
        """
        def calculate_morans_i(data: np.ndarray) -> float:
            n = len(data)
            if n < 2:
                return 0
                
            mean = np.mean(data)
            deviations = data - mean
            
            weights = np.zeros((n, n))
            for i in range(n-1):
                weights[i, i+1] = weights[i+1, i] = 1
                
            numerator = np.sum(weights * np.outer(deviations, deviations))
            denominator = np.sum(deviations ** 2)
            
            if denominator == 0:
                return 0
                
            return (n / np.sum(weights)) * (numerator / denominator)

        # Calculate spatial autocorrelation for each axis
        spatial_autocorr = {
            'x': calculate_morans_i(np.mean(self.encoded_tiles != -1, axis=(1,2))),
            'y': calculate_morans_i(np.mean(self.encoded_tiles != -1, axis=(0,2))),
            'z': calculate_morans_i(np.mean(self.encoded_tiles != -1, axis=(0,1)))
        }

        # Calculate vertical changes
        vertical_changes = []
        for y in range(self.shape[1]-1):
            layer1 = set(p for p in self.encoded_tiles[:,y,:].flatten() if p != -1)
            layer2 = set(p for p in self.encoded_tiles[:,y+1,:].flatten() if p != -1)
            if layer1 or layer2:  # If either layer has patterns
                similarity = len(layer1.intersection(layer2)) / len(layer1.union(layer2)) if layer1 or layer2 else 0
                vertical_changes.append(similarity)

        # Calculate density
        density = (self.encoded_tiles != -1).astype(float)
        local_density = np.zeros_like(density)
        
        for x in range(1, self.shape[0]-1):
            for y in range(1, self.shape[1]-1):
                for z in range(1, self.shape[2]-1):
                    local_density[x,y,z] = np.mean(density[x-1:x+2, y-1:y+2, z-1:z+2])

        return {
            'spatial_autocorrelation': spatial_autocorr,
            'vertical_stratification': np.mean(vertical_changes) if vertical_changes else 0,
            'density_variation': np.std(local_density),
            'density_gradient': np.mean(np.gradient(local_density))
        }

    def calculate_all_metrics(self) -> Dict:
        """Calculate all building metrics"""
        return {
            'variety_metrics': self.calculate_variety_metrics(),
            'spatial_metrics': self.calculate_spatial_metrics(),
            'complexity_metrics': self.calculate_complexity_metrics()
        }

def analyze_building(encoded_tiles_path: str, pattern_map_path: str):
    """Analyze building using all metrics"""
    # Load data
    encoded_tiles = np.load(encoded_tiles_path)
    pattern_map = np.load(pattern_map_path, allow_pickle=True).item()
    
    # Create analyzer
    analyzer = BuildingMetrics(encoded_tiles, pattern_map)
    
    # Calculate metrics
    try:
        metrics = analyzer.calculate_all_metrics()
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None, analyzer

    # Print results
    print("\nBuilding Analysis Results")
    print("========================")
    
    print("\nVariety Metrics:")
    vm = metrics['variety_metrics']
    print(f"Pattern Diversity: {vm['pattern_diversity']:.3f} [Shannon-Wiener Diversity Index (H')]")
    print(f"Pattern Richness: {vm['pattern_richness']} [Species Richness (S)]")
    print(f"Pattern Evenness: {vm['pattern_evenness']:.3f} [Pielou's Evenness Index (J')]")
    print(f"Block Diversity: {vm['block_diversity']:.3f} [Shannon's Entropy]")
    
    print("\nSpatial Metrics:")
    sm = metrics['spatial_metrics']
    print("Spatial Autocorrelation:")
    for axis, value in sm['spatial_autocorrelation'].items():
        print(f"  {axis}-axis: {value:.3f} [Moran's I Index]")
    print(f"Vertical Stratification: {sm['vertical_stratification']:.3f} [Jaccard Similarity Index]")
    print(f"Density Variation: {sm['density_variation']:.3f} [Standard Deviation of Local Density]")
    
    print("\nComplexity Metrics:")
    cm = metrics['complexity_metrics']
    print(f"Transition Complexity: {cm['transition_complexity']} [Edge Count in Transition Graph]")
    print(f"Graph Density: {cm['graph_density']:.3f} [Network Density δ]")
    print(f"Surface/Volume Ratio: {cm['surface_to_volume_ratio']:.3f} [Specific Surface Area (SSA)]")
    
    return metrics, analyzer

if __name__ == "__main__":
    metrics, analyzer = analyze_building('encoded_tiles.npy', 'pattern_map.npy')