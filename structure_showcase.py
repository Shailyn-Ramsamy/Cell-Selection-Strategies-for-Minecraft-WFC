from typing import List

from gdpc import Block, Editor, Transform
from gdpc import geometry as geo
from glm import ivec3

from structure import Structure, build_structure, load_structure
from structures import (
    corner_entrance_top, corner_entrance, corner_entrance_corner, corner_entrance_front, inner_corner_front, inner_corner_top, inner_corner_bottom, 
    interior_top, wall_bottom, wall_top, wall_front, air, dirt, corner_mid, inner_corner_mid, corner_mid_corner, corner_mid_front, inner_corner_mid_front,
    interior_mid, wall_mid, wall_mid_front, wall_front_v2
)


def build_strucutre_showcase(editor: Editor, structures: List[Structure], space_between_structures = 3):
    
    # same for all strucures
    strucutre_size = structures[0].size

    geo.placeCuboid(editor, 
                    ivec3(0,0,0), 
                    ivec3(4*(strucutre_size.x+2*space_between_structures), 16, len(structures)*(strucutre_size.z+space_between_structures)),
                    Block("air"))
    
    editor.flushBuffer()

    for rotation in range(4):
        with editor.pushTransform(Transform(translation=ivec3(rotation*(strucutre_size.x+2*space_between_structures), 0, 0))):
            for structure_idx, structure in enumerate(structures):
                with editor.pushTransform(Transform(translation=ivec3(0, 0, structure_idx*(strucutre_size.z+space_between_structures)))):
                    build_structure(editor, structure, rotation)
                    
    editor.flushBuffer()


def main():
    ED = Editor(buffering=True)

    try:
        ED.transform @= Transform(translation=ivec3(0, 30, 60))

        structures = [
            # load_structure(air),
            # load_structure(air),
            # load_structure(air),
            # load_structure(air),
            # load_structure(air),
            # load_structure(air),
            # load_structure(air),
            # load_structure(air),
            load_structure(corner_entrance_front),
            load_structure(corner_entrance_corner),
            load_structure(inner_corner_front),
            load_structure(wall_front),
            load_structure(wall_front_v2),
            # load_structure(),
        ]

        print("Building structure showcase")
        build_strucutre_showcase(editor=ED, structures=structures)

        print("Done!")

    except KeyboardInterrupt: # useful for aborting a run-away program
        print("Pressed Ctrl-C to kill program.")


if __name__ == '__main__':
    main()
