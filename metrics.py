import requests
from typing import Tuple, Dict, List
import time
from collections import Counter, defaultdict
import numpy as np

def get_blocks(x: int, y: int, z: int, dx: int, dy: int, dz: int, 
               max_retries: int = 5, retry_delay: float = 1.0) -> List[Dict]:
    """Fetch all blocks in a specified area"""
    BASE_URL = "http://localhost:9000"
    
    params = {
        'x': x,
        'y': y,
        'z': z,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'includeState': 'true'
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Fetching blocks from ({x}, {y}, {z}) to ({x+dx}, {y+dy}, {z+dz})")
            response = requests.get(f"{BASE_URL}/blocks", params=params)
            response.raise_for_status()
            blocks = response.json()
            print(f"Successfully fetched {len(blocks)} blocks")
            return blocks
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error fetching blocks (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to fetch blocks after {max_retries} attempts: {e}")

def create_block_array(blocks: List[Dict], start_coords: Tuple[int, int, int], 
                      dimensions: Tuple[int, int, int]) -> np.ndarray:
    """Convert list of blocks into a 3D numpy array"""
    start_x, start_y, start_z = start_coords
    dx, dy, dz = dimensions
    
    # Create 3D array filled with 'minecraft:air' as default
    block_array = np.full((dx, dy, dz), 'minecraft:air', dtype=object)
    
    # Fill array with block IDs
    for block in blocks:
        rel_x = block['x'] - start_x
        rel_y = block['y'] - start_y
        rel_z = block['z'] - start_z
        if 0 <= rel_x < dx and 0 <= rel_y < dy and 0 <= rel_z < dz:
            block_array[rel_x, rel_y, rel_z] = block['id']
    
    return block_array

def tile_to_key(tile: np.ndarray) -> str:
    """Convert a tile to a hashable string key"""
    # Convert the tile to a string representation
    return '\n'.join(','.join(row) for row in tile.reshape(-1, tile.shape[-1]))

def should_skip_tile(block_counts: Counter, threshold: float = 0.95) -> bool:
    """
    Determine if a tile should be skipped based on its block composition
    
    Args:
        block_counts: Counter of block types
        threshold: Minimum percentage for a block type to be considered dominant
    
    Returns:
        bool: True if tile should be skipped
    """
    total_blocks = sum(block_counts.values())
    
    # Skip pure tiles
    if len(block_counts) == 1:
        block_type = next(iter(block_counts.keys()))
        return block_type in {'minecraft:air', 'minecraft:dirt'}
    
    # Skip tiles that are mostly dirt and grass
    dirt_and_grass = (block_counts.get('minecraft:dirt', 0) + 
                     block_counts.get('minecraft:grass_block', 0))
    
    if dirt_and_grass / total_blocks >= threshold:
        return True
        
    return False

def should_skip_tile(block_counts: Counter, threshold: float = 0.95) -> bool:
    """
    Determine if a tile should be skipped based on its block composition
    
    Args:
        block_counts: Counter of block types
        threshold: Minimum percentage for a block type to be considered dominant
    
    Returns:
        bool: True if tile should be skipped
    """
    total_blocks = sum(block_counts.values())
    
    # Skip pure tiles
    if len(block_counts) == 1:
        block_type = next(iter(block_counts.keys()))
        return block_type in {'minecraft:air', 'minecraft:dirt'}
    
    # Skip tiles that are mostly dirt and grass
    dirt_and_grass = (block_counts.get('minecraft:dirt', 0) + 
                     block_counts.get('minecraft:grass_block', 0))
    
    if dirt_and_grass / total_blocks >= threshold:
        return True
        
    return False

def extract_tiles(block_array: np.ndarray, tile_size: Tuple[int, int, int] = (9, 7, 9), 
                 stride: Tuple[int, int, int] = (1, 1, 1)) -> Tuple[np.ndarray, Dict]:
    """Extract tiles from block array using sliding window and encode unique patterns, 
    skipping pure and dirt/grass tiles"""
    tx, ty, tz = tile_size
    sx, sy, sz = stride
    dx, dy, dz = block_array.shape
    
    # Calculate output dimensions
    out_x = max(1, (dx - tx) // sx + 1)
    out_y = max(1, (dy - ty) // sy + 1)
    out_z = max(1, (dz - tz) // sz + 1)
    
    # Initialize output array with -1 (indicating skipped tiles)
    encoded_tiles = np.full((out_x, out_y, out_z), -1, dtype=int)
    pattern_map = {}
    next_pattern_id = 1
    
    print(f"Processing tiles with dimensions {out_x}x{out_y}x{out_z}")
    
    # Track skip statistics
    skip_stats = defaultdict(int)
    total_processed = 0
    
    # Sliding window over the block array
    for i in range(0, dx - tx + 1, sx):
        for j in range(0, dy - ty + 1, sy):
            for k in range(0, dz - tz + 1, sz):
                total_processed += 1
                
                # Extract tile
                tile = block_array[i:i+tx, j:j+ty, k:k+tz]
                
                # Get block counts for this tile
                block_counts = Counter(tile.flatten())
                
                # Check if tile should be skipped
                if len(block_counts) == 1:
                    block_type = next(iter(block_counts.keys()))
                    if block_type == 'minecraft:air':
                        skip_stats['pure_air'] += 1
                        continue
                    elif block_type == 'minecraft:dirt':
                        skip_stats['pure_dirt'] += 1
                        continue
                
                # Check for dirt/grass dominated tiles
                dirt_and_grass = (block_counts.get('minecraft:dirt', 0) + 
                                block_counts.get('minecraft:grass_block', 0))
                if dirt_and_grass / sum(block_counts.values()) >= 0.95:
                    skip_stats['dirt_grass'] += 1
                    continue
                
                # Process non-skipped tiles
                tile_key = tile_to_key(tile)
                
                # Get or create pattern ID
                if tile_key not in pattern_map:
                    pattern_map[tile_key] = {
                        'pattern_id': next_pattern_id,
                        'block_counts': block_counts,
                        'example_coords': (i, j, k)
                    }
                    next_pattern_id += 1
                
                # Store encoded tile
                encoded_tiles[i//sx, j//sy, k//sz] = pattern_map[tile_key]['pattern_id']
                
                if (i//sx) % 10 == 0 and (j//sy) % 10 == 0 and (k//sz) % 10 == 0:
                    print(f"Processing position ({i//sx}, {j//sy}, {k//sz})")
    
    # Print detailed statistics
    print(f"\nTile Processing Statistics:")
    print(f"Total tiles processed: {total_processed}")
    print(f"Pure air tiles skipped: {skip_stats['pure_air']} ({skip_stats['pure_air']/total_processed*100:.1f}%)")
    print(f"Pure dirt tiles skipped: {skip_stats['pure_dirt']} ({skip_stats['pure_dirt']/total_processed*100:.1f}%)")
    print(f"Dirt/grass dominated tiles skipped: {skip_stats['dirt_grass']} ({skip_stats['dirt_grass']/total_processed*100:.1f}%)")
    print(f"Total tiles skipped: {sum(skip_stats.values())} ({sum(skip_stats.values())/total_processed*100:.1f}%)")
    print(f"Unique patterns found: {len(pattern_map)}")
    
    return encoded_tiles, pattern_map

def analyze_patterns(encoded_tiles: np.ndarray, pattern_map: Dict) -> None:
    """Analyze pattern distribution with enhanced statistics"""
    print("\nTile Analysis:")
    total_tiles = encoded_tiles.size
    unique_patterns = len(pattern_map)
    
    # Count patterns, excluding -1 (skipped tiles)
    pattern_frequencies = Counter(x for x in encoded_tiles.flatten() if x != -1)
    mixed_tiles = sum(pattern_frequencies.values())
    skipped_tiles = total_tiles - mixed_tiles
    
    print(f"Total tiles: {total_tiles}")
    print(f"Skipped tiles: {skipped_tiles}")
    print(f"Mixed tiles: {mixed_tiles}")
    print(f"Unique patterns: {unique_patterns}")
    print(f"Average occurrences per pattern: {mixed_tiles / unique_patterns:.2f}")
    print(f"Encoded tiles shape: {encoded_tiles.shape}")
    
    # Analyze remaining patterns
    print("\nRemaining Pattern Analysis:")
    print("Most common patterns (top 5):")
    for pattern_id, count in pattern_frequencies.most_common(5):
        percentage = (count / mixed_tiles) * 100  # Percentage of mixed tiles
        pattern_info = next(info for info in pattern_map.values() if info['pattern_id'] == pattern_id)
        coords = pattern_info['example_coords']
        print(f"\nPattern ID: {pattern_id}")
        print(f"Count: {count} ({percentage:.1f}% of mixed tiles)")
        print(f"Example location: {coords}")
        print("Block composition:")
        for block_id, block_count in pattern_info['block_counts'].most_common(5):
            print(f"  {block_id}: {block_count} blocks")

if __name__ == "__main__":
    # Define area
    start_x, start_y, start_z = 473, -60, 58
    end_x, end_y, end_z = 653, -25, 238
    
    # Calculate dimensions
    dx = end_x - start_x
    dy = end_y - start_y
    dz = end_z - start_z
    
    try:
        # Fetch blocks
        blocks = get_blocks(x=start_x, y=start_y, z=start_z, dx=dx, dy=dy, dz=dz)
        
        # Convert to array
        print("Converting blocks to array...")
        block_array = create_block_array(
            blocks, 
            (start_x, start_y, start_z), 
            (dx, dy, dz)
        )
        print(f"Created block array with shape {block_array.shape}")
        
        # Extract and encode tiles
        print("Extracting and encoding tiles...")
        tile_size = (9, 7, 9)
        stride = (9, 7, 9)
        
        encoded_tiles, pattern_map = extract_tiles(block_array, tile_size, stride)
        
        # Analyze patterns
        analyze_patterns(encoded_tiles, pattern_map)
        
        # Save results
        print("\nSaving results...")
        np.save('encoded_tiles.npy', encoded_tiles)
        np.save('pattern_map.npy', pattern_map)
        print("Results saved to encoded_tiles.npy and pattern_map.npy")
        
        # Print dimension analysis
        tile_dimensions = encoded_tiles.shape
        print("\nDimensional Analysis:")
        print(f"X dimension: {tile_dimensions[0]} tiles ({tile_dimensions[0] * tile_size[0]} blocks)")
        print(f"Y dimension: {tile_dimensions[1]} tiles ({tile_dimensions[1] * tile_size[1]} blocks)")
        print(f"Z dimension: {tile_dimensions[2]} tiles ({tile_dimensions[2] * tile_size[2]} blocks)")
        
    except Exception as e:
        print(f"Error: {e}")
        raise