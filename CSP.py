from gdpc import Editor, Transform
from glm import ivec3
import random
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from structures import (
    corner_entrance_top, corner_entrance, corner_entrance_corner, corner_entrance_front, 
    inner_corner_front, inner_corner_top, inner_corner_bottom, interior_bottom1, 
    interior_bottom2, interior_top, wall_bottom, wall_top, wall_front, air, dirt, 
    corner_mid, inner_corner_mid, corner_mid_corner, corner_mid_front, 
    inner_corner_mid_front, interior_mid, wall_mid, wall_mid_front, interior_mid_bamb, 
    interior_bottom_bamb, interior_top_bamb, interior_mid_bamb_v2, wall_front_v2, 
    dirt_top_1, dirt_top_2
)
from structure import Structure, load_structure, build_structure

class CSPBuilder:
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
        # Cache for domain reduction
        self.neighbor_cache = self._build_neighbor_cache()
        
    def load_structures(self) -> Dict[str, List[Tuple[Structure, float]]]:
        """Load structures with weights"""
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
                (load_structure(interior_bottom1), 0.7),
                (load_structure(interior_bottom2), 0.3),
            ],
            "wall_top": [(load_structure(wall_top), 1.0)],
            "wall_bottom": [(load_structure(wall_bottom), 1.0)],
            "wall_front": [
                (load_structure(wall_front), 0.6),
                (load_structure(wall_front_v2), 0.4),
            ],
            "air": [
                (load_structure(air), 0.3),
                (load_structure(dirt_top_1), 0.6),
                (load_structure(dirt_top_2), 0.1),
            ],
            "dirt": [(load_structure(dirt), 1.0)],
            "corner_mid": [(load_structure(corner_mid), 1.0)],
            "inner_corner_mid": [(load_structure(inner_corner_mid), 1.0)],
            "corner_mid_corner": [(load_structure(corner_mid_corner), 1.0)],
            "corner_mid_front": [(load_structure(corner_mid_front), 1.0)],
            "inner_corner_mid_front": [(load_structure(inner_corner_mid_front), 1.0)],
            "interior_mid": [(load_structure(interior_mid), 1.0)],
            "wall_mid": [(load_structure(wall_mid), 1.0)],
            "wall_mid_front": [(load_structure(wall_mid_front), 1.0)],
            "interior_mid_bamb": [(load_structure(interior_mid_bamb), 1.0)],
            "interior_top_bamb": [(load_structure(interior_top_bamb), 1.0)],
            "interior_bottom_bamb": [(load_structure(interior_bottom_bamb), 1.0)],
        }
        
        return structures

    def _build_neighbor_cache(self) -> Dict[str, Dict[int, Dict[str, Set[Tuple[str, int]]]]]:
        """Build cache of valid neighbors for each module/rotation/direction"""
        cache = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        
        for module in self.all_modules:
            for rotation in range(4):
                if str(rotation) in self.adjacencies[module]:
                    for direction in self.directions:
                        if direction in self.adjacencies[module][str(rotation)]:
                            neighbors = self.adjacencies[module][str(rotation)][direction]
                            cache[module][rotation][direction] = {
                                (n['structure'], n['rotation']) for n in neighbors
                            }
        return cache

    def get_valid_neighbors(self, module: str, rotation: int, direction: str) -> Set[Tuple[str, int]]:
        """Get valid neighbors from cache"""
        return self.neighbor_cache[module][rotation][direction]

    def select_weighted_structure(self, module: str, exclude_indices: List[int] = None) -> Structure:
        """Select a weighted structure variant"""
        if module not in self.structures:
            raise ValueError(f"Unknown module: {module}")
            
        variants = self.structures[module]
        if exclude_indices:
            variants = [(struct, weight) for i, (struct, weight) in enumerate(variants) 
                       if i not in exclude_indices]
            if not variants:
                raise ValueError(f"No valid variants for {module}")
                
            total_weight = sum(weight for _, weight in variants)
            variants = [(struct, weight/total_weight) for struct, weight in variants]
        
        r = random.random()
        cumulative = 0
        for structure, weight in variants:
            cumulative += weight
            if r <= cumulative:
                return structure
        return variants[-1][0]

    def get_domain(self, grid: List[List[List[Tuple[str, int]]]], pos: Tuple[int, int, int]) -> List[Tuple[str, int]]:
        """Get valid domain for a position based on current constraints"""
        x, y, z = pos
        
        # Handle fixed positions
        if z == 0:
            return [('dirt', 0)]
        if z == len(grid)-1:
            return [('air', 0)]
        if (x == 0 or x == len(grid[0][0])-1 or 
            y == 0 or y == len(grid[0])-1):
            return [('air', 0)]
            
        # Get constraints from neighbors
        domain = set()
        for module in self.all_modules:
            for rotation in range(4):
                valid = True
                for dx, dy, dz, direction in [
                    (0, -1, 0, 'NORTH'), (1, 0, 0, 'EAST'),
                    (0, 1, 0, 'SOUTH'), (-1, 0, 0, 'WEST'),
                    (0, 0, 1, 'UP'), (0, 0, -1, 'DOWN')
                ]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < len(grid[0][0]) and 
                        0 <= ny < len(grid[0]) and 
                        0 <= nz < len(grid)):
                        neighbor = grid[nz][ny][nx]
                        if neighbor is not None:
                            # Check if current module can connect to neighbor
                            valid_neighbors = self.get_valid_neighbors(module, rotation, direction)
                            if neighbor not in valid_neighbors:
                                valid = False
                                break
                                
                            # Check if neighbor can connect to current module
                            if neighbor[0] not in ['air', 'dirt']:
                                opposite_dir = self.opposite_directions[direction]
                                neighbor_module, neighbor_rotation = neighbor
                                valid_neighbors = self.get_valid_neighbors(
                                    neighbor_module, neighbor_rotation, opposite_dir)
                                if (module, rotation) not in valid_neighbors:
                                    valid = False
                                    break
                
                if valid:
                    domain.add((module, rotation))
                    
        return list(domain)

    def get_next_variable(self, grid: List[List[List[Tuple[str, int]]]], 
                         domains: Dict[Tuple[int, int, int], List[Tuple[str, int]]]) -> Tuple[int, int, int]:
        """Select next variable using MRV (Minimum Remaining Values) heuristic"""
        min_domain_size = float('inf')
        best_pos = None
        
        for z in range(len(grid)):
            for y in range(len(grid[0])):
                for x in range(len(grid[0][0])):
                    pos = (x, y, z)
                    if grid[z][y][x] is None:
                        domain_size = len(domains[pos])
                        if 0 < domain_size < min_domain_size:
                            min_domain_size = domain_size
                            best_pos = pos
        
        return best_pos

    def solve_csp(self, grid: List[List[List[Tuple[str, int]]]], 
                  domains: Dict[Tuple[int, int, int], List[Tuple[str, int]]]) -> bool:
        """Solve the CSP using backtracking with forward checking"""
        # Check if all variables are assigned
        if all(all(all(cell is not None for cell in row) for row in layer) for layer in grid):
            return True
            
        # Select unassigned variable
        pos = self.get_next_variable(grid, domains)
        if pos is None:
            return True
            
        x, y, z = pos
        
        # Try values from domain
        for module, rotation in domains[pos]:
            grid[z][y][x] = (module, rotation)
            
            # Store old domains
            old_domains = {p: d.copy() for p, d in domains.items()}
            
            # Forward checking
            valid = True
            for dx, dy, dz, direction in [
                (0, -1, 0, 'NORTH'), (1, 0, 0, 'EAST'),
                (0, 1, 0, 'SOUTH'), (-1, 0, 0, 'WEST'),
                (0, 0, 1, 'UP'), (0, 0, -1, 'DOWN')
            ]:
                nx, ny, nz = x + dx, y + dy, z + dz
                npos = (nx, ny, nz)
                if npos in domains:
                    # Update domain of neighbor
                    new_domain = self.get_domain(grid, npos)
                    if not new_domain:
                        valid = False
                        break
                    domains[npos] = new_domain
            
            if valid and self.solve_csp(grid, domains):
                return True
                
            # Backtrack
            grid[z][y][x] = None
            domains.update(old_domains)
        
        return False

    def _initialize_grid(self, width: int, height: int, depth: int) -> List[List[List[Tuple[str, int]]]]:
        """Initialize the grid with fixed positions"""
        return [[[None for _ in range(width)] for _ in range(height)] for _ in range(depth)]

    def generate(self, width: int, height: int, depth: int) -> List[List[List[Tuple[str, int]]]]:
        """Generate structure using CSP"""
        grid = self._initialize_grid(width, height, depth)
        
        # Initialize domains for all positions
        domains = {}
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    pos = (x, y, z)
                    domains[pos] = self.get_domain(grid, pos)
        
        # Solve the CSP
        if self.solve_csp(grid, domains):
            return grid
        else:
            raise RuntimeError("Failed to generate valid structure")

    def build_in_minecraft(self, editor: Editor, grid: List[List[List[Tuple[str, int]]]], 
                          start_pos: ivec3):
        """Build the generated structure in Minecraft"""
        for z, layer in enumerate(grid):
            for y, row in enumerate(layer):
                for x, cell in enumerate(row):
                    if cell is not None:
                        module, rotation = cell
                        if module in self.structures:
                            if module == "air" and z != 1:
                                structure = self.structures[module][0][0]
                            elif module == "air" and z == 1:
                                structure = self.select_weighted_structure(module, [0])
                            else:
                                structure = self.select_weighted_structure(module)
                            
                            position = start_pos + ivec3(
                                x * structure.size.x,
                                z * structure.size.y,
                                y * structure.size.z
                            )
                            with editor.pushTransform(Transform(translation=position)):
                                build_structure(editor, structure, rotation)

    def generate_and_build(self, editor: Editor, width: int, height: int, depth: int, 
                          start_pos: ivec3):
        """Generate and build a structure in one step"""
        grid = self.generate(width, height, depth)
        self.build_in_minecraft(editor, grid, start_pos)
        return grid

    def print_grid(self, grid: List[List[List[Tuple[str, int]]]]):
        """Print a readable representation of the grid"""
        for z, layer in enumerate(grid):
            print(f"\nLayer {z}:")
            for row in layer:
                print(" ".join(f"{str(cell):20}" if cell else " "*20 for cell in row))

# Example usage
if __name__ == "__main__":
    csp_builder = CSPBuilder('adjacencies.json')
    editor = Editor(buffering=True)
    start_pos = ivec3(473, -60, 58)  
    
    try:
        result = csp_builder.generate_and_build(editor, 12, 12, 5, start_pos)
        print("Successfully generated structure!")
        csp_builder.print_grid(result)
    except RuntimeError as e:
        print(f"Generation failed: {e}")
        
    editor.flushBuffer()