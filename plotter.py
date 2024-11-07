import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

class MinecraftPatternVisualizer:
    def __init__(self, encoded_tiles: np.ndarray, pattern_map: Dict):
        self.encoded_tiles = encoded_tiles
        self.pattern_map = pattern_map
        self.shape = encoded_tiles.shape
        self.mixed_patterns = set(x for x in encoded_tiles.flatten() if x != -1)
        
    def plot_pattern_distribution(self) -> plt.Figure:
        """Plot pattern distribution with emphasis on mixed vs pure tiles"""
        # Count patterns (excluding -1)
        pattern_counts = Counter(x for x in self.encoded_tiles.flatten() if x != -1)
        total_tiles = self.encoded_tiles.size
        pure_tiles = np.sum(self.encoded_tiles == -1)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of pure vs mixed tiles
        pure_pct = (pure_tiles / total_tiles) * 100
        mixed_pct = 100 - pure_pct
        
        ax1.pie([pure_pct, mixed_pct], 
                labels=['Pure Tiles', 'Mixed Tiles'],
                autopct='%1.1f%%',
                colors=['lightgray', 'lightblue'])
        ax1.set_title('Pure vs Mixed Tiles Distribution')
        
        # Bar plot of top pattern frequencies
        patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        x, y = zip(*patterns)
        
        bars = ax2.bar(range(len(x)), y)
        ax2.set_xticks(range(len(x)))
        ax2.set_xticklabels([f'P{i}' for i in x], rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        ax2.set_title('Top 15 Pattern Frequencies')
        ax2.set_xlabel('Pattern ID')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    
    def plot_layer_analysis(self) -> plt.Figure:
        """Create layer-by-layer analysis visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        
        # Calculate statistics for each layer
        y_layers = self.shape[1]  # Number of vertical layers
        layer_stats = []
        
        for y in range(y_layers):
            layer = self.encoded_tiles[:, y, :]
            total = layer.size
            pure = np.sum(layer == -1)
            mixed = total - pure
            unique_patterns = len(set(x for x in layer.flatten() if x != -1))
            
            layer_stats.append({
                'pure': pure,
                'mixed': mixed,
                'unique_patterns': unique_patterns
            })
        
        # Plot stacked bar chart for composition
        y_pos = np.arange(y_layers)
        pure_counts = [stats['pure'] for stats in layer_stats]
        mixed_counts = [stats['mixed'] for stats in layer_stats]
        
        ax1.bar(y_pos, pure_counts, label='Pure Tiles', color='lightgray')
        ax1.bar(y_pos, mixed_counts, bottom=pure_counts, label='Mixed Tiles', color='lightblue')
        
        ax1.set_title('Layer Composition Analysis')
        ax1.set_xlabel('Vertical Layer')
        ax1.set_ylabel('Number of Tiles')
        ax1.legend()
        
        # Plot unique patterns per layer
        unique_patterns = [stats['unique_patterns'] for stats in layer_stats]
        ax2.plot(y_pos, unique_patterns, marker='o', linewidth=2, color='darkblue')
        ax2.set_title('Unique Patterns per Layer')
        ax2.set_xlabel('Vertical Layer')
        ax2.set_ylabel('Number of Unique Patterns')
        
        plt.tight_layout()
        return fig
    
    def plot_3d_pattern_distribution(self) -> go.Figure:
        """Create interactive 3D visualization showing pattern distribution"""
        # Collect positions for pure and mixed tiles
        pure_positions = []
        pattern_positions = defaultdict(list)
        
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    pattern = self.encoded_tiles[x, y, z]
                    if pattern == -1:
                        pure_positions.append((x, y, z))
                    else:
                        pattern_positions[pattern].append((x, y, z))
        
        # Create visualization
        fig = go.Figure()
        
        # Add pure tiles
        if pure_positions:
            x, y, z = zip(*pure_positions)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                name='Pure Tiles',
                marker=dict(
                    size=4,
                    color='lightgray',
                    opacity=0.5
                )
            ))
        
        # Add mixed pattern tiles
        colors = plt.cm.rainbow(np.linspace(0, 1, len(pattern_positions)))
        for (pattern, positions), color in zip(pattern_positions.items(), colors):
            if positions:
                x, y, z = zip(*positions)
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    name=f'Pattern {pattern}',
                    marker=dict(
                        size=4,
                        color=f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})',
                        opacity=0.8
                    )
                ))
        
        fig.update_layout(
            title='3D Pattern Distribution',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=800,
            height=800
        )
        
        return fig

def visualize_patterns(encoded_tiles_path: str, pattern_map_path: str):
    """Create and save all visualizations"""
    # Load data
    encoded_tiles = np.load(encoded_tiles_path)
    pattern_map = np.load(pattern_map_path, allow_pickle=True).item()
    
    # Create visualizer
    viz = MinecraftPatternVisualizer(encoded_tiles, pattern_map)
    
    # Generate plots
    dist_fig = viz.plot_pattern_distribution()
    layer_fig = viz.plot_layer_analysis()
    pattern_3d = viz.plot_3d_pattern_distribution()
    
    # Save plots
    dist_fig.savefig('pattern_distribution.png', bbox_inches='tight')
    layer_fig.savefig('layer_analysis.png', bbox_inches='tight')
    pattern_3d.write_html('pattern_3d.html')
    
    return viz

if __name__ == "__main__":
    visualizer = visualize_patterns('encoded_tiles.npy', 'pattern_map.npy')