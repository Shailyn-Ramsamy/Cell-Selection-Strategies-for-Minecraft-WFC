from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json



class Direction(Enum):
    NORTH = "NORTH"
    EAST = "EAST"
    SOUTH = "SOUTH"
    WEST = "WEST"
    UP = "UP"
    DOWN = "DOWN"

@dataclass
class TileConnection:
    structure: str
    rotation: int

class TileAdjacencyUtils:
    def __init__(self, adjacency_data: Dict):
        self.data = adjacency_data

    def get_valid_connections(self, structure: str, rotation: int, direction: Direction) -> List[TileConnection]:
        """
        Get all valid connections for a given structure, rotation, and direction.
        
        Args:
            structure: The type of structure (e.g., "corner_entrance_top")
            rotation: The rotation of the structure (0-3)
            direction: The direction to check for connections
        
        Returns:
            A list of TileConnection objects representing valid connections
        """
        try:
            connections = self.data[structure][str(rotation)][direction.value]
            return [TileConnection(c["structure"], c["rotation"]) for c in connections]
        except KeyError:
            return []

    def can_connect(self, structure1: str, rotation1: int, 
                    structure2: str, rotation2: int, direction: Direction) -> bool:
        """
        Check if two structures can connect in the given direction.
        
        Args:
            structure1: The type of first structure
            rotation1: The rotation of the first structure
            structure2: The type of second structure
            rotation2: The rotation of the second structure
            direction: The direction from structure1 to structure2
        
        Returns:
            True if the structures can connect, False otherwise
        """
        connections = self.get_valid_connections(structure1, rotation1, direction)
        return any(c.structure == structure2 and c.rotation == rotation2 for c in connections)

    def get_all_possible_neighbors(self, structure: str, rotation: int) -> Dict[Direction, List[TileConnection]]:
        """
        Get all possible neighboring tiles for all directions.
        
        Args:
            structure: The type of structure
            rotation: The rotation of the structure
        
        Returns:
            A dictionary mapping directions to lists of possible connections
        """
        return {
            direction: self.get_valid_connections(structure, rotation, Direction(direction))
            for direction in Direction.__members__
        }

    def find_compatible_rotations(self, structure1: str, structure2: str, 
                                 direction: Direction) -> List[Tuple[int, int]]:
        """
        Find all compatible rotations between two structures in a given direction.
        
        Args:
            structure1: The type of first structure
            structure2: The type of second structure
            direction: The direction from structure1 to structure2
        
        Returns:
            A list of tuples (rotation1, rotation2) representing compatible rotations
        """
        compatible_rotations = []
        for rot1 in range(4):
            connections = self.get_valid_connections(structure1, rot1, direction)
            for conn in connections:
                if conn.structure == structure2:
                    compatible_rotations.append((rot1, conn.rotation))
        return compatible_rotations

# Example usage
def example_usage(adjacency_data: Dict):
    utils = TileAdjacencyUtils(adjacency_data)
    
    # Example 1: Check if two structures can connect
    can_connect = utils.can_connect(
        "corner_entrance_top", 0, "wall_top", 1, Direction.NORTH
    )
    print(f"Can connect: {can_connect}")
    
    # Example 2: Get all possible neighbors
    neighbors = utils.get_all_possible_neighbors("corner_entrance_top", 0)
    print("\nPossible neighbors for corner_entrance_top rotation 0:")
    for direction, connections in neighbors.items():
        if connections:
            print(f"{direction.value}:")
            for conn in connections:
                print(f"  - {conn.structure} (rotation {conn.rotation})")
    
    # Example 3: Find compatible rotations
    compatible_rots = utils.find_compatible_rotations(
        "corner_entrance_top", "wall_top", Direction.NORTH
    )
    print("\nCompatible rotations between corner_entrance_top and wall_top (NORTH):")
    for rot1, rot2 in compatible_rots:
        print(f"  corner_entrance_top rotation {rot1} -> wall_top rotation {rot2}")
        
        
# Load your JSON data
with open('adjacencies.json', 'r') as f:
    adjacency_data = json.load(f)

# Create the utils instance
utils = TileAdjacencyUtils(adjacency_data)

# Now you can use the helper functions
# For example:
possible_connections = utils.get_valid_connections("inner_corner_top", 0, Direction.WEST)


print(possible_connections)