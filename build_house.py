from WFC import EnhancedWFC3DBuilder
from glm import ivec3
from gdpc import Editor

# Define available strategies
strategies = ['entropy', 'height_priority', 'from_center', 'random_walk']

# Prompt user for strategy
print("Available strategies:")
for i, strategy in enumerate(strategies):
    print(f"{i + 1}. {strategy}")
    
while True:
    try:
        strategy_choice = int(input("Enter the number corresponding to your chosen strategy: ")) - 1
        if 0 <= strategy_choice < len(strategies):
            chosen_strategy = strategies[strategy_choice]
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Prompt user for house position
while True:
    try:
        x = int(input("Enter the x-coordinate of the starting position: "))
        y = int(input("Enter the y-coordinate of the starting position: "))
        z = int(input("Enter the z-coordinate of the starting position: "))
        start_pos = ivec3(x, y, z)
        break
    except ValueError:
        print("Invalid input. Please enter integers for the coordinates.")

# Prompt user for house dimensions
while True:
    try:
        length = int(input("Enter the length of the house: "))
        width = int(input("Enter the width of the house: "))
        height = int(input("Enter the height of the house: "))
        break
    except ValueError:
        print("Invalid input. Please enter integers for the dimensions.")

# Initialize builder and editor
builder = EnhancedWFC3DBuilder('adjacencies.json', strategy=chosen_strategy)
editor = Editor(buffering=True)

# Generate and build the house
result = builder.generate_grid(editor, length, width, height, start_pos)
builder.build_in_minecraft(editor, result, start_pos)

# Apply changes to Minecraft
editor.flushBuffer()

print(f"House generated at position {start_pos} using the '{chosen_strategy}' strategy.")