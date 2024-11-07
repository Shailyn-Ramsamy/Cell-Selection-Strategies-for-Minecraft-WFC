import json
import random
from collections import defaultdict
from gdpc import Editor, Block, Transform
from glm import ivec3
from typing import Dict, List, Tuple

from structures import (
    corner_entrance_top, corner_entrance, corner_entrance_corner, corner_entrance_front, inner_corner_front, inner_corner_top, inner_corner_bottom, 
    interior_bottom1, interior_bottom2, interior_top, wall_bottom, wall_top, wall_front, air, dirt, corner_mid, inner_corner_mid, corner_mid_corner, corner_mid_front, inner_corner_mid_front,
    interior_mid, wall_mid, wall_mid_front, interior_mid_bamb, interior_bottom_bamb, interior_top_bamb, wall_front_v2, dirt_top_1, dirt_top_2,corner_mid_corner_v2, wall_mid_front_v2, 
    inner_corner_mid_front_v2, corner_mid_front_v2
)

from structure import Structure, load_structure, build_structure


class CellSelectionStrategy:
    @staticmethod
    def select_by_collapsed_neighbors(grid, candidates):
        """
        Classic selection strategy that prefers cells with more collapsed neighbors.
        Creates more structured, continuous builds.
        """
        def count_collapsed_neighbors(x, y, z):
            count = 0
            for dx, dy, dz in [(0,1,0), (0,-1,0), (1,0,0), (-1,0,0), (0,0,1), (0,0,-1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (0 <= nx < len(grid[0][0]) and 
                    0 <= ny < len(grid[0]) and 
                    0 <= nz < len(grid)):
                    if len(grid[nz][ny][nx]) == 1:
                        count += 1
            return count
        
        return max(candidates, key=lambda pos: count_collapsed_neighbors(*pos))

    @staticmethod
    def select_by_height_priority(grid, candidates):
        """
        Prioritizes collapsing cells from bottom to top.
        Creates more stable, grounded structures.
        """
        return min(candidates, key=lambda pos: pos[1])

    @staticmethod
    def select_from_center(grid, candidates):
        """
        Prioritizes cells closer to the center of the grid.
        Creates more centralized, symmetric structures.
        """
        center_x = len(grid[0][0]) // 2
        center_y = len(grid[0]) // 2
        center_z = len(grid) // 2
        
        def distance_to_center(pos):
            x, y, z = pos
            return ((x - center_x) ** 2 + 
                   (y - center_y) ** 2 + 
                   (z - center_z) ** 2) ** 0.5
        
        return min(candidates, key=distance_to_center)

    @staticmethod
    def select_from_corners(grid, candidates):
        """
        Prioritizes corner and edge cells.
        Creates more complex, intricate structures with interesting corners.
        """
        max_x = len(grid[0][0]) - 1
        max_y = len(grid[0]) - 1
        max_z = len(grid) - 1
        
        def corner_priority(pos):
            x, y, z = pos
            # Calculate how close the position is to any corner
            corner_distances = []
            for corner_x in (0, max_x):
                for corner_y in (0, max_y):
                    for corner_z in (0, max_z):
                        dist = ((x - corner_x) ** 2 + 
                               (y - corner_y) ** 2 + 
                               (z - corner_z) ** 2) ** 0.5
                        corner_distances.append(dist)
            return min(corner_distances)
        
        return min(candidates, key=corner_priority)

    @staticmethod
    def select_alternating_layers(grid, candidates):
        """
        Alternates between collapsing layers horizontally and vertically.
        Creates interesting layered structures.
        """
        def get_layer_completion(pos):
            x, y, z = pos
            horizontal_layer = sum(1 for cell in grid[z][y] 
                                 if len(cell) == 1)
            vertical_layer = sum(1 for layer in grid 
                               if len(layer[y][x]) == 1)
            # Prefer the less complete layer
            return min(horizontal_layer, vertical_layer)
        
        return min(candidates, key=get_layer_completion)

    @staticmethod
    def select_random_walk(grid, candidates, previous_pos=None):
        """
        Selects cells based on a random walk from the previous position.
        Creates more organic, flowing structures.
        """
        import random
        
        if not previous_pos:
            return random.choice(candidates)
            
        def distance_from_previous(pos):
            x, y, z = pos
            px, py, pz = previous_pos
            return ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
        
        # Filter candidates to those within a reasonable distance
        max_step = 2
        close_candidates = [pos for pos in candidates 
                          if distance_from_previous(pos) <= max_step]
        
        if close_candidates:
            return random.choice(close_candidates)
        return random.choice(candidates)


class WFC3DBuilder:
    def __init__(self, adjacency_file):
        with open(adjacency_file, 'r') as f:
            self.adjacencies = json.load(f)
        
        self.all_modules = list(self.adjacencies.keys())
        self.directions = ['NORTH', 'EAST', 'SOUTH', 'WEST', 'UP', 'DOWN']
        self.opposite_directions = {
            'NORTH': 'SOUTH', 'SOUTH': 'NORTH',
            'EAST': 'WEST', 'WEST': 'EAST',
            'UP': 'DOWN', 'DOWN': 'UP'
        }
        self.structures = self.load_structures()
        self.max_attempts = 100
        
    def load_structures(self) -> Dict[str, List[Tuple[Structure, float]]]:
        # Now each structure is paired with its weight
        structures = {
            "corner_entrance_top": [(load_structure(corner_entrance_top), 1.0)],
            "corner_entrance": [(load_structure(corner_entrance), 1.0)],
            "corner_entrance_corner": [(load_structure(corner_entrance_corner), 1.0)],
            "corner_entrance_front": [(load_structure(corner_entrance_front), 1.0)],
            "inner_corner_top": [(load_structure(inner_corner_top), 1.0)],
            "inner_corner_bottom": [(load_structure(inner_corner_bottom), 1.0)],
            "inner_corner_front": [(load_structure(inner_corner_front), 1.0)],
            "interior_top": [(load_structure(interior_top), 1.0)],
            "interior_bottom": [
                (load_structure(interior_bottom1), 0.7),  # 70% chance
                (load_structure(interior_bottom2), 0.3),  # 30% chance
            ],
            "wall_top": [(load_structure(wall_top), 1.0)],
            "wall_bottom": [(load_structure(wall_bottom), 1.0)],
            "wall_front": [
                (load_structure(wall_front), 0.6),     # 80% chance
                (load_structure(wall_front_v2), 0.4),  # 20% chance
            ],
            "air": [
                (load_structure(air), 0.1),          # 60% chance
                (load_structure(dirt_top_1), 0.8),   # 20% chance
                (load_structure(dirt_top_2), 0.1),   # 20% chance
            ],
            "dirt": [(load_structure(dirt), 1.0)],
            "corner_mid": [(load_structure(corner_mid), 1.0)],
            "inner_corner_mid": [(load_structure(inner_corner_mid), 1.0)],
            "corner_mid_corner": [
                (load_structure(corner_mid_corner), 0.5),
                (load_structure(corner_mid_corner_v2), 0.5)
                ],
            "corner_mid_front": [
                (load_structure(corner_mid_front), 0.5),
                (load_structure(corner_mid_front_v2), 0.5),
                ],
            "inner_corner_mid_front": [
                (load_structure(inner_corner_mid_front), 0.5),
                (load_structure(inner_corner_mid_front_v2), 0.5),
                                       ],
            "interior_mid": [(load_structure(interior_mid), 1.0)],
            "wall_mid": [(load_structure(wall_mid), 1.0)],
            "wall_mid_front": [
                (load_structure(wall_mid_front), 0.5),
                (load_structure(wall_mid_front_v2), 0.5)
                               ],
            "interior_mid_bamb": [(load_structure(interior_mid_bamb), 1.0)],
            "interior_top_bamb": [(load_structure(interior_top_bamb), 1.0)],
            "interior_bottom_bamb": [(load_structure(interior_bottom_bamb), 1.0)],
        }
        
        # Validate weights sum to 1.0 for each module
        for module, variants in structures.items():
            total_weight = sum(weight for _, weight in variants)
            if not (0.99 <= total_weight <= 1.01):  # Allow for small floating point errors
                print(f"Warning: Weights for {module} sum to {total_weight}, should be 1.0")
                # Normalize weights
                normalized_variants = [(struct, weight/total_weight) for struct, weight in variants]
                structures[module] = normalized_variants
                
        return structures
        
    def select_weighted_structure(self, module: str, exclude_indices: list[int] = None) -> Structure:
        if module not in self.structures:
            raise ValueError(f"Unknown module: {module}")
            
        variants = self.structures[module]
        if exclude_indices:
            # Filter out excluded variants and their weights
            filtered_variants = [(struct, weight) for i, (struct, weight) in enumerate(variants) 
                            if i not in exclude_indices]
            if not filtered_variants:
                raise ValueError(f"No valid variants remaining after exclusion for module: {module}")
            
            # Renormalize weights
            total_weight = sum(weight for _, weight in filtered_variants)
            filtered_variants = [(struct, weight/total_weight) for struct, weight in filtered_variants]
            
            # Use filtered variants for selection
            r = random.random()
            cumulative_weight = 0.0
            for structure, weight in filtered_variants:
                cumulative_weight += weight
                if r <= cumulative_weight:
                    return structure
            return filtered_variants[-1][0]
        else:
            # Original behavior for no exclusions
            if len(variants) == 1:
                return variants[0][0]
                
            r = random.random()
            cumulative_weight = 0.0
            for structure, weight in variants:
                cumulative_weight += weight
                if r <= cumulative_weight:
                    return structure
            return variants[-1][0]
    
    def initialize_grid(self, width, height, depth):
        grid = [[[set() for _ in range(width)] for _ in range(height)] for _ in range(depth)]
        
        # Set bottom layer (z=0) to dirt
        for y in range(height):
            for x in range(width):
                grid[0][y][x] = {('dirt', 0)}
        
        # Set top layer (z=depth-1) to air
        for y in range(height):
            for x in range(width):
                grid[depth-1][y][x] = {('air', 0)}
        
        # Set middle layers to all possible modules, except on the edges
        for z in range(1, depth-1):
            for y in range(height):
                for x in range(width):
                    if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                        grid[z][y][x] = {('air', 0)}
                    else:
                        grid[z][y][x] = set((module, rotation) for module in self.all_modules for rotation in range(4))
        
        return grid
    
    def get_valid_neighbors(self, module, rotation, direction):
        valid_neighbors = set()
        if str(rotation) in self.adjacencies[module]:
            for neighbor in self.adjacencies[module][str(rotation)][direction]:
                valid_neighbors.add((neighbor['structure'], neighbor['rotation']))
        return valid_neighbors
    
    def propagate(self, grid, x, y, z):
        stack = [(x, y, z)]
        while stack:
            cx, cy, cz = stack.pop()
            current_modules = grid[cz][cy][cx]
            
            for dx, dy, dz, direction in [
                (0, -1, 0, 'NORTH'), (1, 0, 0, 'EAST'), (0, 1, 0, 'SOUTH'), (-1, 0, 0, 'WEST'),
                (0, 0, 1, 'UP'), (0, 0, -1, 'DOWN')
            ]:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if 0 <= nx < len(grid[0][0]) and 0 <= ny < len(grid[0]) and 0 <= nz < len(grid):
                    neighbor_modules = grid[nz][ny][nx]
                    valid_neighbors = set()
                    for module, rotation in current_modules:
                        valid_neighbors.update(self.get_valid_neighbors(module, rotation, direction))
                    
                    new_neighbor_modules = neighbor_modules.intersection(valid_neighbors)
                    if new_neighbor_modules != neighbor_modules:
                        grid[nz][ny][nx] = new_neighbor_modules
                        stack.append((nx, ny, nz))
    
    
    def collapse(self, grid, stack):
        min_entropy = float('inf')
        min_entropy_cells = []
        
        for z in range(1, len(grid) - 1):  # Collapse all layers except the first and last
            for y in range(len(grid[0])):
                for x in range(len(grid[0][0])):
                    entropy = len(grid[z][y][x])
                    if entropy > 1:
                        if entropy < min_entropy:
                            min_entropy = entropy
                            min_entropy_cells = [(x, y, z)]
                        elif entropy == min_entropy:
                            min_entropy_cells.append((x, y, z))
        
        if not min_entropy_cells:
            return False
        
        x, y, z = self.select_cell(grid, min_entropy_cells)
        options = grid[z][y][x].copy()
        stack.append((x, y, z, options))
        
        chosen_module = random.choice(list(grid[z][y][x]))
        grid[z][y][x] = {chosen_module}
        self.propagate(grid, x, y, z)
        return True

    
    def is_fully_collapsed(self, grid):
        return all(len(cell) == 1 for layer in grid for row in layer for cell in row)

    def get_valid_modules(self, grid, x, y, z):
        valid_modules = set(grid[z][y][x])
        for dx, dy, dz, direction in [
            (0, -1, 0, 'NORTH'), (1, 0, 0, 'EAST'), (0, 1, 0, 'SOUTH'), (-1, 0, 0, 'WEST'),
            (0, 0, 1, 'UP'), (0, 0, -1, 'DOWN')
        ]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < len(grid[0][0]) and 0 <= ny < len(grid[0]) and 0 <= nz < len(grid):
                neighbor_modules = grid[nz][ny][nx]
                if len(neighbor_modules) == 1:
                    neighbor_module, neighbor_rotation = next(iter(neighbor_modules))
                    valid_neighbors = self.get_valid_neighbors(neighbor_module, neighbor_rotation, self.opposite_directions[direction])
                    valid_modules.intersection_update(valid_neighbors)
        return valid_modules
    
    def generate(self, width, height, depth, max_iterations=100000):
        grid = self.initialize_grid(width, height, depth)
        
        # Propagate initial conditions
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if len(grid[z][y][x]) == 1:
                        self.propagate(grid, x, y, z)
        
        stack = []
        iterations = 0
        while iterations < max_iterations:
            if not self.collapse(grid, stack):
                if not stack:
                    break
                # Backtrack
                x, y, z, options = stack.pop()
                grid[z][y][x] = options
            iterations += 1
        
        if iterations == max_iterations:
            print(f"Warning: Reached maximum iterations ({max_iterations}). The result may be incomplete.")
        
        return grid
    
    def get_2d_matrix(self, grid):
        matrix = []
        for layer in grid:
            row = []
            for cell in layer:
                if len(cell) == 1:
                    module, rotation = next(iter(cell))
                    row.append(f"{module}:{rotation}")
                else:
                    row.append("?")
            matrix.append(row)
        return matrix

    def print_2d_matrix(self, matrix):
        for row in matrix:
            print(" ".join(f"{cell:20}" for cell in row))

    def build_in_minecraft(self, editor: Editor, grid, start_pos: ivec3):
        for z, layer in enumerate(grid):
            for y, row in enumerate(layer):
                for x, cell in enumerate(row):
                    if len(cell) == 1:
                        module, rotation = next(iter(cell))
                        if module in self.structures:
                            
                            structure = self.select_weighted_structure(module)
                            
                            if module == "air" and z != 1:
                                structure = self.structures[module][0][0]
                            elif module == "air" and z == 1:
                                structure = self.select_weighted_structure(module, exclude_indices=[0])
                            elif module == "corner_mid_corner" and z % 2 == 0:
                                structure = self.select_weighted_structure(module, exclude_indices=[0])
                            elif module == "corner_mid_corner" and z % 2 != 0:
                                structure = self.select_weighted_structure(module, exclude_indices=[1])
                            elif module == "wall_mid_front" and z % 2 == 0:
                                structure = self.select_weighted_structure(module, exclude_indices=[0])
                            elif module == "wall_mid_front" and z % 2 != 0:
                                structure = self.select_weighted_structure(module, exclude_indices=[1])
                            elif module == "inner_corner_mid_front" and z % 2 == 0:
                                structure = self.select_weighted_structure(module, exclude_indices=[0])
                            elif module == "inner_corner_mid_front" and z % 2 != 0:
                                structure = self.select_weighted_structure(module, exclude_indices=[1])
                            elif module == "corner_mid_front" and z % 2 == 0:
                                structure = self.select_weighted_structure(module, exclude_indices=[0])
                            elif module == "corner_mid_front" and z % 2 != 0:
                                structure = self.select_weighted_structure(module, exclude_indices=[1])
                                
                            
                                
                            position = start_pos + ivec3(x * structure.size.x, z * structure.size.y, y * structure.size.z)
                            with editor.pushTransform(Transform(translation=position)):
                                build_structure(editor, structure, rotation)
                                
    def generate_and_build(self, editor: Editor, width, height, depth, start_pos: ivec3):
        for attempt in range(self.max_attempts):
            print(f"Attempt {attempt + 1}/{self.max_attempts}")
            grid = self.generate(width, height, depth)
            if self.is_fully_collapsed(grid):
                print(f"Successfully generated a valid structure on attempt {attempt + 1}")
                self.build_in_minecraft(editor, grid, start_pos)
                return grid
        
        print(f"Failed to generate a valid structure after {self.max_attempts} attempts")
        return None
    
class EnhancedWFC3DBuilder(WFC3DBuilder):
    def __init__(self, adjacency_file, strategy='collapsed_neighbors'):
        super().__init__(adjacency_file)
        self.strategy = strategy
        self.previous_pos = None
        self.strategies = {
            'collapsed_neighbors': CellSelectionStrategy.select_by_collapsed_neighbors,
            'height_priority': CellSelectionStrategy.select_by_height_priority,
            'from_center': CellSelectionStrategy.select_from_center,
            'from_corners': CellSelectionStrategy.select_from_corners,
            'alternating_layers': CellSelectionStrategy.select_alternating_layers,
            'random_walk': CellSelectionStrategy.select_random_walk
        }

    def select_cell(self, grid, candidates):
        if self.strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        strategy_func = self.strategies[self.strategy]
        
        if self.strategy == 'random_walk':
            selected = strategy_func(grid, candidates, self.previous_pos)
            self.previous_pos = selected
            return selected
            
        return strategy_func(grid, candidates)

# # Example usage:
# builder = EnhancedWFC3DBuilder('adjacencies.json', strategy='random_walk')

# # builder = WFC3DBuilder('adjacencies.json')
# editor = Editor(buffering=True)

# start_pos = ivec3(320, -60, 369)
# result = builder.generate_and_build(editor, 15, 15, 15, start_pos) 

# if result:
#     print("2D Matrix representation of the generated structure:")
#     for z, layer in enumerate(result):
#         print(f"\nLayer {z}:")
#         matrix = builder.get_2d_matrix(layer)
#         builder.print_2d_matrix(matrix)
# else:
#     print("Failed to generate a valid structure")

# if result:
#     for z, layer in enumerate(result):
#         print(f"\nLayer {z}:")
#         print(builder.get_2d_matrix(layer))

# editor.flushBuffer()