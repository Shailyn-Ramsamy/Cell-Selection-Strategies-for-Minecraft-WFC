import random
from typing import Dict, List, Tuple

class MarkovChainBuilder:
    def __init__(self, tiles: List[str]):
        self.tiles = tiles
        self.transition_matrix: Dict[str, Dict[str, float]] = {tile: {t: 0 for t in tiles} for tile in tiles}
        
    def train(self, structures: List[List[str]]):
        for structure in structures:
            for i in range(len(structure) - 1):
                current_tile = structure[i]
                next_tile = structure[i + 1]
                self.transition_matrix[current_tile][next_tile] += 1
        
        # Normalize probabilities
        for tile in self.tiles:
            total = sum(self.transition_matrix[tile].values())
            if total > 0:
                for next_tile in self.tiles:
                    self.transition_matrix[tile][next_tile] /= total
    
    def generate(self, length: int, start_tile: str = None) -> List[str]:
        if start_tile is None:
            start_tile = random.choice(self.tiles)
        
        structure = [start_tile]
        current_tile = start_tile
        
        for _ in range(length - 1):
            next_tile = random.choices(
                self.tiles,
                weights=[self.transition_matrix[current_tile][t] for t in self.tiles]
            )[0]
            structure.append(next_tile)
            current_tile = next_tile
        
        return structure

# Example usage
tiles = ["corner_entrance", "wall", "interior", "inner_corner"]
builder = MarkovChainBuilder(tiles)

# Train with some example structures
example_structures = [
    ["corner_entrance", "wall", "wall", "inner_corner"],
    ["corner_entrance", "wall", "interior", "wall"],
    ["inner_corner", "wall", "interior", "inner_corner"],
]
builder.train(example_structures)

# Generate a new structure
new_structure = builder.generate(10)
print("Generated structure:", new_structure)