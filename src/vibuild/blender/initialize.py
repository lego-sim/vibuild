"""Create registered cameras in an empty, packed project. Executed by Blender."""

import json
from pathlib import Path
import os
import sys


def create(cameras, output):
    import bpy
    from mathutils import Matrix

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "NONE"
    scene.unit_settings.scale_length = 1
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.8,
        0.8,
        0.8,
        1,
    )
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    convention = Matrix.Diagonal((1, -1, -1, 1))
    for frame in cameras:
        data = bpy.data.cameras.new(f"Camera_{frame['index']}")
        obj = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(obj)
        rotation = Matrix(frame["rotation"])
        matrix = rotation.to_4x4()
        matrix.translation = frame["center"]
        obj.matrix_world = matrix @ convention
        obj["source_image"] = frame["name"]
        obj["registered"] = True
        K = frame["K"]
        width = frame["width"]
        height = frame["height"]
        data.type = "PERSP"
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = 36
        data.lens = K[0][0] * 36 / width
        data.shift_x = ((width - 1) / 2 - K[0][2]) / width
        data.shift_y = (K[1][2] - (height - 1) / 2) / width
        data.clip_start = 0.01
        data.clip_end = 10000
        if scene.camera is None:
            scene.camera = obj
            scene.render.resolution_x = width
            scene.render.resolution_y = height
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(
        filepath=str(output), check_existing=False, compress=False
    )


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    create(json.loads(Path(arguments[0]).read_text()), Path(arguments[1]))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
