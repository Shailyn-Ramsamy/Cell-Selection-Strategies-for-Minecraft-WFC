from WFC import EnhancedWFC3DBuilder
from glm import ivec3
from gdpc import Editor


strategies = ['entropy', 'height_priority', 'from_center', 
                 'random_walk']

builder = EnhancedWFC3DBuilder('adjacencies.json', strategy=strategies[3])
editor = Editor(buffering=True)
start_pos = ivec3(0, -67, 0)
length, width, height = 18, 18, 8
result = builder.generate_grid(editor, length, width, height, start_pos)

builder.build_in_minecraft(editor, result, start_pos)

editor.flushBuffer()