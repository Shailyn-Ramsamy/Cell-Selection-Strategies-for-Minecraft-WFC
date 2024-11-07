from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import json
from gdpc import Block, Editor
from glm import ivec3
from structure import Structure

class WFCSubdivider:
    def __init__(self, subdivision_factor: int = 2):
        """
        subdivision_factor: how many times to subdivide each axis
        e.g. subdivision_factor=2 turns a 1x1 tile into 2x2
        """
        self.subdivision_factor = subdivision_factor
        
    def subdivide_structure(self, structure: Structure) -> Dict[str, Structure]:
        """Subdivide a structure into smaller structures"""
        subdivided = {}
        
        # Calculate new size for subdivided pieces
        sub_size_x = structure.size.x // self.subdivision_factor
        sub_size_y = structure.size.y // self.subdivision_factor
        sub_size_z = structure.size.z // self.subdivision_factor
        
        # Iterate through each subdivision
        for x in range(self.subdivision_factor):
            for y in range(self.subdivision_factor):
                for z in range(self.subdivision_factor):
                    # Calculate bounds for this subdivision
                    min_x = x * sub_size_x
                    min_y = y * sub_size_y
                    min_z = z * sub_size_z
                    
                    # Extract blocks for this subdivision
                    sub_blocks = {}
                    for bx in range(sub_size_x):
                        for by in range(sub_size_y):
                            for bz in range(sub_size_z):
                                orig_pos = (min_x + bx, min_y + by, min_z + bz)
                                if orig_pos in structure.blocks:
                                    new_pos = (bx, by, bz)
                                    sub_blocks[new_pos] = structure.blocks[orig_pos]
                    
                    # Create subdivision identifier (e.g. "0_0_0" for first subdivision)
                    sub_id = f"{x}_{y}_{z}"
                    
                    # Create new structure for this subdivision
                    subdivided[sub_id] = Structure(
                        name=f"{structure.name}_sub_{sub_id}",
                        offset=ivec3(min_x, min_y, min_z),
                        size=ivec3(sub_size_x, sub_size_y, sub_size_z),
                        blocks=sub_blocks
                    )
        
        return subdivided

    def generate_adjacencies(self, structure_name: str) -> Dict:
        """Generate adjacency rules based on subdivision positions"""
        adjacencies = {}
        
        # For each possible subdivision position
        for x in range(self.subdivision_factor):
            for y in range(self.subdivision_factor):
                for z in range(self.subdivision_factor):
                    sub_id = f"{x}_{y}_{z}"
                    sub_name = f"{structure_name}_sub_{sub_id}"
                    
                    # Initialize adjacency rules for this subdivision
                    adjacencies[sub_name] = {
                        "NORTH": [], "SOUTH": [], "EAST": [], "WEST": [],
                        "UP": [], "DOWN": []
                    }
                    
                    # Check each direction for neighbors
                    # North neighbor
                    if z > 0:
                        adjacencies[sub_name]["NORTH"].append(
                            f"{structure_name}_sub_{x}_{y}_{z-1}")
                    
                    # South neighbor
                    if z < self.subdivision_factor - 1:
                        adjacencies[sub_name]["SOUTH"].append(
                            f"{structure_name}_sub_{x}_{y}_{z+1}")
                    
                    # East neighbor
                    if x < self.subdivision_factor - 1:
                        adjacencies[sub_name]["EAST"].append(
                            f"{structure_name}_sub_{x+1}_{y}_{z}")
                    
                    # West neighbor
                    if x > 0:
                        adjacencies[sub_name]["WEST"].append(
                            f"{structure_name}_sub_{x-1}_{y}_{z}")
                    
                    # Up neighbor
                    if y < self.subdivision_factor - 1:
                        adjacencies[sub_name]["UP"].append(
                            f"{structure_name}_sub_{x}_{y+1}_{z}")
                    
                    # Down neighbor
                    if y > 0:
                        adjacencies[sub_name]["DOWN"].append(
                            f"{structure_name}_sub_{x}_{y-1}_{z}")
        
        return adjacencies

def subdivide_all_structures(structures: Dict[str, Structure], subdivision_factor: int) -> Tuple[Dict[str, Structure], Dict]:
    """
    Subdivide all structures and generate their adjacencies
    Returns (subdivided_structures, adjacency_rules)
    """
    subdivider = WFCSubdivider(subdivision_factor)
    
    # Subdivide all structures
    all_subdivided = {}
    for name, structure in structures.items():
        subdivided = subdivider.subdivide_structure(structure)
        all_subdivided.update(subdivided)
    
    # Generate adjacencies for all subdivided structures
    all_adjacencies = {}
    for name in structures:
        adjacencies = subdivider.generate_adjacencies(name)
        all_adjacencies.update(adjacencies)
    
    return all_subdivided, all_adjacencies