"""One evaluated mesh BVH per requested camera frame, checked against Blender rays."""
import time


def apply_visibility(scene, view_layer, selected_objects=None):
    # Collection render flags are not honoured by scene.ray_cast automatically.
    # Preserve excluded/hidden layer collections and object visibility as well.
    visible_collections = set()

    def visit(layer, excluded=False):
        excluded = excluded or layer.exclude or layer.hide_viewport or layer.collection.hide_viewport or layer.collection.hide_render
        if not excluded:
            visible_collections.update(o.name for o in layer.collection.objects)
        for child in layer.children:
            visit(child, excluded)

    visit(view_layer.layer_collection)
    for obj in scene.objects:
        if obj.type != 'MESH' or obj.name not in view_layer.objects:
            continue
        excluded = (obj.hide_render or obj.hide_get() or obj.hide_viewport or obj.name not in visible_collections
                    or (selected_objects is not None and obj.name not in selected_objects))
        obj.hide_set(bool(excluded))
        if excluded:
            obj.hide_render = True
    view_layer.update()


def build_bvh(deps):
    from mathutils.bvhtree import BVHTree
    start = time.monotonic()
    vertices, triangles, owners, normals = [], [], [], []
    for inst in deps.object_instances:
        obj = inst.object
        if obj.type != 'MESH' or not inst.show_self or obj.hide_render:
            continue
        if not inst.is_instance and not obj.original.visible_get():
            continue
        if inst.is_instance and inst.parent and inst.parent.original.hide_render:
            continue
        mesh = obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            offset = len(vertices)
            vertices.extend(inst.matrix_world @ v.co for v in mesh.vertices)
            normal_matrix = inst.matrix_world.to_3x3().inverted_safe().transposed()
            for tri in mesh.loop_triangles:
                triangles.append(tuple(offset + v for v in tri.vertices))
                owners.append(obj.original)
                # World triangle winding flips under negative scale; Blender's
                # ray_cast returns the inverse-transpose object-space normal.
                normals.append((normal_matrix @ tri.normal).normalized())
        finally:
            obj.to_mesh_clear()
    tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True) if triangles else None

    def cast(origin, direction, distance):
        if tree is None:
            return None
        point, normal, face, _ = tree.ray_cast(origin, direction, distance)
        return None if face is None else (point, normals[face], owners[face])

    return cast, {'triangles': len(triangles), 'build_seconds': time.monotonic() - start}


def canonical_cast(scene, deps, origin, direction, distance):
    hit, point, normal, _, obj, _ = scene.ray_cast(deps, origin, direction, distance=distance)
    return (point, normal, obj.original) if hit else None


def validate_rays(scene, deps, cast, origin, directions, count, distance):
    import numpy as np
    from mathutils import Vector
    count = min(count, len(directions))
    indices = np.random.default_rng(42).choice(len(directions), count, replace=False)
    max_error = 0.
    for index in indices:
        direction = Vector(directions[index]).normalized()
        a = canonical_cast(scene, deps, origin, direction, distance)
        b = cast(origin, direction, distance)
        if (a is None) != (b is None):
            raise ValueError('BVH visibility differs from Blender; no output will be cached')
        if a is not None:
            error = (a[0] - b[0]).length
            max_error = max(max_error, error)
            if error > 1e-4 or a[2].name != b[2].name or a[1].dot(b[1]) < .999:
                raise ValueError(f'BVH differs from Blender: owner={a[2].name}/{b[2].name}, position_error={error}, normal_dot={a[1].dot(b[1])}; use --ray-backend scene to diagnose')
    return {'rays': count, 'max_position_error_scene_units': max_error, 'passed': True}
