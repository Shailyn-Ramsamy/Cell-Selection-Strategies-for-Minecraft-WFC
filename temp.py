from enum import Enum
from typing import Dict, List, Tuple, Union
import json

class Direction(Enum):
    NORTH = "NORTH"
    EAST = "EAST"
    SOUTH = "SOUTH"
    WEST = "WEST"
    UP = "UP"
    DOWN = "DOWN"

# Define structure types
structure_types = [
    "corner_entrance_top",
    "corner_entrance",
    "wall_top",
    "wall_bottom",
    "inner_corner_top",
    "inner_corner_bottom",
    "interior_top",
    "interior_bottom",
    "air",
    "dirt"
]

# Define the number of rotations for each structure type
rotations = {structure: 4 for structure in structure_types}  # Default 4 rotations for each

# Initialize new adjacency dictionary
new_adjacencies = {
    structure: {
        str(rotation): {  # Changed back to str(rotation)
            **{direction.value: [] for direction in Direction}
        } for rotation in range(rotations[structure])
    } for structure in structure_types
}

# Define the opposite directions
opposite_directions = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP
}

# Define rotation transforms
direction_rotations = {
    Direction.NORTH: [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST],
    Direction.EAST: [Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH],
    Direction.SOUTH: [Direction.SOUTH, Direction.WEST, Direction.NORTH, Direction.EAST],
    Direction.WEST: [Direction.WEST, Direction.NORTH, Direction.EAST, Direction.SOUTH],
    Direction.UP: [Direction.UP] * 4,
    Direction.DOWN: [Direction.DOWN] * 4
}

def get_rotated_direction(direction: Direction, rotation: int) -> Direction:
    return direction_rotations[direction][rotation % 4]

def define_adjacency(from_structure: str, to_structure: str, direction: Direction, 
                     base_from_rotation: int, base_to_rotation: Union[int, List[int]]):
    if from_structure not in structure_types or to_structure not in structure_types:
        raise ValueError(f"Invalid structure type: {from_structure} or {to_structure}")
    
    # Convert single rotation to list for consistent handling
    to_rotations = [base_to_rotation] if isinstance(base_to_rotation, int) else base_to_rotation
    
    # For each possible rotation of the 'from' structure
    for rotation_offset in range(4):
        # Calculate actual 'from' rotation
        from_rotation = (base_from_rotation + rotation_offset) % 4
        
        # Calculate rotated direction
        rotated_direction = get_rotated_direction(direction, rotation_offset)
        
        # For each specified 'to' rotation
        for base_to_rot in to_rotations:
            # Calculate actual 'to' rotation
            to_rotation = (base_to_rot + rotation_offset) % 4
            
            # Add the forward connection - Using str(from_rotation)
            new_adjacencies[from_structure][str(from_rotation)][rotated_direction.value].append({
                "structure": to_structure,
                "rotation": to_rotation
            })

            # Add the reverse connection - Using str(to_rotation)
            opposite_direction = opposite_directions[rotated_direction]
            new_adjacencies[to_structure][str(to_rotation)][opposite_direction.value].append({
                "structure": from_structure,
                "rotation": from_rotation
            })

def save_adjacencies_to_json(filename: str):
    with open(filename, 'w') as json_file:
        json.dump(new_adjacencies, json_file, indent=4)

# Function to print connections for debugging
def print_connections(structure: str, rotation: int):
    connections = new_adjacencies[structure][str(rotation)]
    print(f"{structure} rotation {rotation}:")
    for direction in Direction:
        conns = connections[direction.value]
        if conns:
            print(f"  {direction.value}: {conns}")

# Corner_entrance_top
define_adjacency("corner_entrance_top", "wall_top", Direction.NORTH, 0, 1)
define_adjacency("corner_entrance_top", "inner_corner_top", Direction.NORTH, 0, 0)
define_adjacency("corner_entrance_top", "corner_entrance_top", Direction.NORTH, 0, 1)

define_adjacency("corner_entrance_top", "wall_top", Direction.EAST, 0, 0)
define_adjacency("corner_entrance_top", "inner_corner_top", Direction.EAST, 0, 0)
define_adjacency("corner_entrance_top", "corner_entrance_top", Direction.EAST, 0, 3)

define_adjacency("corner_entrance_top", "air", Direction.WEST, 0, [0,1,2,3])

define_adjacency("corner_entrance_top", "air", Direction.SOUTH, 0, [0,1,2,3])

define_adjacency("corner_entrance_top", "air", Direction.UP, 0, [0,1,2,3])

define_adjacency("corner_entrance_top", "corner_entrance", Direction.DOWN, 0, 0)

#corner_entrance

define_adjacency("corner_entrance", "wall_bottom", Direction.NORTH, 0, 1)
define_adjacency("corner_entrance", "inner_corner_bottom", Direction.NORTH, 0, 0)  
define_adjacency("corner_entrance", "corner_entrance", Direction.NORTH, 0, 1)  

define_adjacency("corner_entrance", "wall_bottom", Direction.EAST, 0, 0)
define_adjacency("corner_entrance", "inner_corner_bottom", Direction.EAST, 0, 0)
define_adjacency("corner_entrance", "corner_entrance", Direction.EAST, 0, 3)

define_adjacency("corner_entrance", "air", Direction.WEST, 0, [0,1,2,3])

define_adjacency("corner_entrance", "air", Direction.SOUTH, 0, [0,1,2,3])

define_adjacency("corner_entrance", "dirt", Direction.DOWN, 0, [0,1,2,3])

#wall_top

define_adjacency("wall_top", "interior_top", Direction.NORTH, 0, [0,1,2,3])

define_adjacency("wall_top", "wall_top", Direction.EAST, 0, 0)
define_adjacency("wall_top", "inner_corner_top", Direction.EAST, 0, 0)

define_adjacency("wall_top", "wall_top", Direction.WEST, 0, 0)
define_adjacency("wall_top", "inner_corner_top", Direction.WEST, 0, 3)

define_adjacency("wall_top", "air", Direction.SOUTH, 0, [0,1,2,3])

define_adjacency("wall_top", "air", Direction.UP, 0, [0,1,2,3])

define_adjacency("wall_top", "wall_bottom", Direction.DOWN, 0, 0)

#wall_bottom

define_adjacency("wall_bottom", "interior_bottom", Direction.NORTH, 0, [0,1,2,3])

define_adjacency("wall_bottom", "wall_bottom", Direction.EAST, 0, 0)
define_adjacency("wall_bottom", "inner_corner_bottom", Direction.EAST, 0, 0)

define_adjacency("wall_bottom", "wall_bottom", Direction.WEST, 0, 0)
define_adjacency("wall_bottom", "inner_corner_bottom", Direction.WEST, 0, 3)

define_adjacency("wall_bottom", "air", Direction.SOUTH, 0, [0,1,2,3])

define_adjacency("wall_bottom", "dirt", Direction.DOWN, 0, [0,1,2,3])

#inner_corner_top

define_adjacency("inner_corner_top", "interior_top", Direction.NORTH, 0, [0,1,2,3])

define_adjacency("inner_corner_top", "interior_top", Direction.EAST, 0, [0,1,2,3])

define_adjacency("inner_corner_top", "air", Direction.UP, 0, [0,1,2,3])

define_adjacency("inner_corner_top", "inner_corner_bottom", Direction.DOWN, 0, [0,1,2,3])

#inner_corner_bottom

define_adjacency("inner_corner_bottom", "interior_bottom", Direction.NORTH, 0, [0,1,2,3])

define_adjacency("inner_corner_bottom", "interior_bottom", Direction.EAST, 0, [0,1,2,3])

define_adjacency("inner_corner_bottom", "dirt", Direction.DOWN, 0, [0,1,2,3])

#interior_top
define_adjacency("interior_top", "interior_top", Direction.NORTH, 0, [0,1,2,3])
define_adjacency("interior_top", "interior_bottom", Direction.DOWN, 0, [0,1,2,3])
define_adjacency("interior_top", "air", Direction.UP, 0, [0,1,2,3])

#interior_bottom

define_adjacency("interior_bottom", "interior_bottom", Direction.NORTH, 0, [0,1,2,3])
define_adjacency("interior_bottom", "dirt", Direction.DOWN, 0, [0,1,2,3])

#dirt

define_adjacency("dirt", "dirt", Direction.NORTH, 0, 0)
define_adjacency("dirt", "dirt", Direction.EAST, 0, 0)
define_adjacency("dirt", "air", Direction.UP, 0, 0)

#air

define_adjacency("air", "air", Direction.NORTH, 0, 0)
define_adjacency("air", "air", Direction.EAST, 0, 0)
define_adjacency("air", "air", Direction.UP, 0, 0)

# Print all rotations for corner_entrance
for rotation in range(4):
    print_connections("corner_entrance", rotation)

save_adjacencies_to_json('adjacencies.json')