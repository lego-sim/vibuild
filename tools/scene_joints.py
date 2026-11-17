"""Explicit, sampled floor/ceiling joints on evaluated world meshes (scene units, Z up).

Manifest: {"joints": [{"id": "wall-base", "kind": "floor_wall",
"floor": "Floor", "wall": "Wall", "path_xy": [[0,0],[2,0]],
"max_spacing": 0.1, "max_embed": 0.005}]}. Use kind floor_floor with
floor and other_floor for transition paths where BOTH slabs must support the path.
Use kind wall_ceiling with ceiling and wall for wall-top/ceiling-underside contact.
Paths must include relevant endpoints/corners and threshold contacts. max_embed is
an evidence-backed scene-unit bound, not a physical tolerance or an inferred scale.
"""
from __future__ import annotations

import math
import numpy as np


def stations(path, spacing):
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('path_xy needs at least two finite XY points')
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('max_spacing must be finite and positive')
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        count = max(1, math.ceil(float(np.linalg.norm(b-a)) / spacing))
        result.extend(a + (b-a)*i/count for i in range(1, count+1))
    return result


def evaluated_mesh(name, depsgraph):
    import bpy
    obj = bpy.context.scene.objects.get(name)
    if obj is None or obj.type != 'MESH':
        raise ValueError(f'joint object is not a scene mesh: {name}')
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = np.array([tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices], dtype=float)
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError(f'empty/nonfinite evaluated mesh: {name}')
        triangles = vertices[np.array([tuple(t.vertices) for t in mesh.loop_triangles], dtype=int)]
        if not len(triangles):
            raise ValueError(f'no evaluated triangles: {name}')
        normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals / np.where(lengths > 0, lengths, 1)[:,None]
        issues = []
        if obj.hide_render or obj.hide_get() or obj.hide_viewport:
            issues.append('object_hidden')
        if any(m.show_render != m.show_viewport for m in obj.modifiers):
            issues.append('render_viewport_modifier_mismatch')
        return {'triangles': triangles, 'normals': normals, 'vertices': vertices,
                'issues': issues, 'modifiers': [{'name': m.name, 'type': m.type,
                'show_viewport': m.show_viewport, 'show_render': m.show_render} for m in obj.modifiers]}
    finally:
        evaluated.to_mesh_clear()


def intersections(mesh, xy, tolerance):
    """Vertical line / triangle intersections, retaining geometric winding normals.

    Float64 arithmetic avoids adding substantial error to Blender's float32 world
    coordinates. XY inclusion tolerates only the documented numerical uncertainty.
    """
    hits = []
    for triangle, normal in zip(mesh['triangles'], mesh['normals']):
        a, b, c = triangle
        u, v = b[:2]-a[:2], c[:2]-a[:2]
        determinant = u[0]*v[1]-u[1]*v[0]
        if abs(determinant) <= np.finfo(float).eps * max(float(np.linalg.norm(u)*np.linalg.norm(v)), 1):
            continue
        d = np.asarray(xy)-a[:2]
        s = (d[0]*v[1]-d[1]*v[0])/determinant
        t = (u[0]*d[1]-u[1]*d[0])/determinant
        weights = np.array([1-s-t, s, t])
        if np.any(weights < 0) or np.any(weights > 1):
            # A barycentric tolerance divided by a tiny XY area can extrapolate
            # Z hundreds of units off a near-vertical triangle. Instead snap
            # only to its nearest projected edge, with convex weights.
            closest = None
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = triangle[j,:2]-triangle[i,:2]
                length2 = float(edge @ edge)
                fraction = float(np.clip((np.asarray(xy)-triangle[i,:2]) @ edge / length2, 0, 1)) if length2 else 0.
                point = triangle[i,:2]+fraction*edge
                distance = float(np.linalg.norm(np.asarray(xy)-point))
                if closest is None or distance < closest[0]:
                    edge_weights = np.zeros(3)
                    edge_weights[i], edge_weights[j] = 1-fraction, fraction
                    closest = distance, edge_weights
            if closest[0] > tolerance:
                continue
            weights = closest[1]
        z = float(weights @ triangle[:,2])
        hits.append({'z': z, 'normal': normal.tolist()})
    hits.sort(key=lambda h: h['z'], reverse=True)
    merged = []
    for hit in hits:
        if merged and abs(hit['z']-merged[-1]['z']) <= tolerance:
            merged[-1]['normals'].append(hit['normal'])
        else:
            merged.append({'z': hit['z'], 'normals': [hit['normal']]})
    return merged


def cap_intersections(mesh, xy, tolerance):
    """Separate Z-facing caps from predominantly vertical side surfaces.

    A cap's largest normal component is Z (either sign). This is an explicit
    geometric domain for this floor/base checker, not a physical gap tolerance.
    Steeper surfaces alone cannot establish floor/base support. Preserve sides
    for diagnostics; never turn their intersections into invented slab thickness.
    """
    caps, sides = [], []
    for hit in intersections(mesh, xy, tolerance):
        normals = [n for n in hit['normals'] if abs(n[2]) >= max(abs(n[0]), abs(n[1]))]
        if normals:
            caps.append({**hit, 'normals': normals, 'coincident_normals': hit['normals']})
        if len(normals) != len(hit['normals']):
            sides.append(hit)
    return caps, sides


def support(mesh, xy, tolerance):
    hits, sides = cap_intersections(mesh, xy, tolerance)
    if not hits:
        return {'issues': ['no_supporting_floor'], 'side_intersections': sides}
    top = hits[0]
    result = {'top_z': top['z'], 'top_normals': top['normals'],
              'top_coincident_normals': top['coincident_normals'],
              'side_intersections': sides, 'issues': []}
    if any(n[2] <= 0 for n in top['normals']):
        result['issues'].append('reversed_floor_top_normal')
    if len(hits) > 1:
        underside = hits[1]
        result.update(bottom_z=underside['z'], thickness=top['z']-underside['z'],
                      bottom_normals=underside['normals'], bottom_coincident_normals=underside['coincident_normals'])
        if any(n[2] >= 0 for n in underside['normals']):
            result['issues'].append('invalid_floor_underside_normal')
    return result


def ceiling_normal_thickness(mesh, xy, underside, tolerance):
    """Finite inward-normal ray proof at an underside edge, not vertical clearance."""
    origin = np.array([*xy, underside['z']])
    direction = -np.asarray(underside['normals'][0])
    tangent = np.cross(direction, [1., 0., 0.])
    tangent /= np.linalg.norm(tangent)
    basis = np.column_stack((tangent, np.cross(direction, tangent), direction))
    ray_mesh = {'triangles': (mesh['triangles']-origin) @ basis,
                'normals': mesh['normals'] @ basis}
    # At an extrusion edge the ray lies along the rim. Its near-zero normal
    # dot product is float32 noise, not a crossing of the solid's boundary.
    crossing = np.abs(ray_mesh['normals'][:,2]) > 32 * np.finfo(np.float32).eps
    ray_mesh = {key: value[crossing] for key, value in ray_mesh.items()}
    # Local Z is ray distance; world-space surface coordinates stay explicit.
    hits = [h for h in intersections(ray_mesh, [0., 0.], tolerance) if h['z'] > tolerance]
    for hit in reversed(hits):
        normals = [basis @ np.asarray(n) for n in hit['normals']]
        caps = [n for n in normals if abs(n[2]) >= max(abs(n[0]), abs(n[1]))]
        # A rim ray can drift by float32 coordinate error along a finite edge.
        # Reject a genuine side crossing, but not one whose entire normal
        # displacement is still within the existing positional uncertainty.
        if caps or any(abs(np.dot(n, direction))*hit['z'] > tolerance for n in normals):
            break
    else:
        return {'issues': ['ceiling_without_verified_thickness']}
    result = {'normal_thickness_distance': hit['z'],
              'normal_thickness_direction_world': direction.tolist(),
              'normal_top_hit_world': (origin+direction*hit['z']).tolist(),
              'normal_top_normals_world': [n.tolist() for n in caps], 'issues': []}
    if not caps:
        result['issues'].append('ceiling_without_verified_thickness')
    elif any(n[2] <= 0 or np.dot(n, direction) <= 0 for n in caps):
        result['issues'].append('reversed_ceiling_top_normal')
    return result


def ceiling_station(ceiling_mesh, wall_mesh, xy, tolerance, max_embed):
    hits, sides = cap_intersections(ceiling_mesh, xy, tolerance)
    ceiling = {'side_intersections': sides, 'issues': []}
    if not hits:
        ceiling['issues'].append('no_ceiling_at_station')
    else:
        underside = hits[-1]
        ceiling.update(underside_z=underside['z'], underside_normals=underside['normals'])
        if any(n[2] >= 0 for n in underside['normals']):
            ceiling['issues'].append('reversed_ceiling_underside_normal')
        if len(hits) < 2:
            proof = ceiling_normal_thickness(ceiling_mesh, xy, underside, tolerance)
            ceiling['issues'].extend(proof.pop('issues'))
            ceiling.update(proof)
        else:
            top = hits[-2]
            ceiling.update(top_z=top['z'], top_normals=top['normals'], thickness=top['z']-underside['z'])
            if any(n[2] <= 0 for n in top['normals']):
                ceiling['issues'].append('reversed_ceiling_top_normal')
    item = {'xy': xy.tolist(), 'ceiling': ceiling, 'issues': list(ceiling['issues'])}
    wall_hits, wall_sides = cap_intersections(wall_mesh, xy, tolerance)
    item['wall_side_intersections'] = wall_sides
    if not wall_hits:
        item['issues'].append('no_wall_at_station')
    else:
        top = wall_hits[0]
        item.update(wall_top_z=top['z'], wall_top_normals=top['normals'])
        if any(n[2] <= 0 for n in top['normals']):
            item['issues'].append('reversed_wall_top_normal')
        if 'underside_z' in ceiling:
            distance = ceiling['underside_z']-top['z']
            item['signed_ceiling_underside_minus_wall_top'] = distance
            if distance > tolerance:
                item['issues'].append('gap')
            elif distance < -tolerance:
                if -distance > max_embed+tolerance:
                    item['issues'].append('embed_exceeds_declared_bound')
                if 'top_z' not in ceiling:
                    item['issues'].append('embed_without_vertical_ceiling_containment')
                elif top['z'] > ceiling['top_z']+tolerance:
                    item['issues'].append('embed_beyond_ceiling_thickness')
                if any(ceiling['underside_z']+tolerance < h['z'] < top['z']-tolerance
                       for h in ceiling['side_intersections']):
                    item['issues'].append('embed_crosses_ceiling_side')
                if not item['issues']:
                    item['contact'] = 'bounded_embed'
            elif not item['issues']:
                item['contact'] = 'numerical_contact'
    item['passed'] = not item['issues']
    return item


def check_joint(joint, meshes):
    kind = joint.get('kind', 'floor_wall')
    if kind not in ('floor_wall', 'floor_floor', 'wall_ceiling'):
        raise ValueError('joint kind must be floor_wall, floor_floor or wall_ceiling')
    first_name = joint['ceiling' if kind == 'wall_ceiling' else 'floor']
    second_name = joint['other_floor' if kind == 'floor_floor' else 'wall']
    first, second = meshes[first_name], meshes[second_name]
    max_embed = float(joint.get('max_embed', 0))
    if not math.isfinite(max_embed) or max_embed < 0 or (kind == 'floor_floor' and max_embed != 0):
        raise ValueError('max_embed must be finite/nonnegative and zero for floor_floor')
    scale = max(1., float(np.abs(first['vertices']).max()), float(np.abs(second['vertices']).max()))
    tolerance = float(32 * np.finfo(np.float32).eps * scale)
    samples = []
    for xy in stations(joint['path_xy'], float(joint['max_spacing'])):
        if kind == 'wall_ceiling':
            samples.append(ceiling_station(first, second, xy, tolerance, max_embed))
            continue
        floor = support(first, xy, tolerance)
        item = {'xy': xy.tolist(), 'floor': floor, 'issues': list(floor['issues'])}
        if kind == 'floor_floor':
            other = support(second, xy, tolerance)
            item['other_floor'] = other
            item['issues'].extend(other['issues'])
            if 'top_z' in floor and 'top_z' in other:
                delta = other['top_z']-floor['top_z']
                item['signed_top_difference'] = delta
                if abs(delta) > tolerance:
                    item['issues'].append('floor_transition_height_mismatch')
        else:
            hits, sides = cap_intersections(second, xy, tolerance)
            item['wall_side_intersections'] = sides
            if not hits:
                item['issues'].append('no_wall_at_station')
            else:
                base = hits[-1]
                top = hits[0]
                item.update(wall_base_z=base['z'], wall_base_normals=base['normals'],
                            wall_top_z=top['z'], wall_top_normals=top['normals'])
                if any(n[2] >= 0 for n in base['normals']):
                    item['issues'].append('reversed_wall_base_normal')
                if any(n[2] <= 0 for n in top['normals']):
                    item['issues'].append('reversed_wall_top_normal')
                if 'top_z' in floor:
                    distance = base['z']-floor['top_z']
                    item['signed_base_minus_floor_top'] = distance
                    if distance > tolerance:
                        item['issues'].append('gap')
                    elif distance < -tolerance:
                        if -distance > max_embed+tolerance:
                            item['issues'].append('embed_exceeds_declared_bound')
                        if 'bottom_z' not in floor:
                            item['issues'].append('embed_without_verified_thickness')
                        elif base['z'] < floor['bottom_z']-tolerance:
                            item['issues'].append('embed_beyond_floor_thickness')
                        if not item['issues']:
                            item['contact'] = 'bounded_embed'
                    else:
                        item['contact'] = 'numerical_contact'
        item['passed'] = not item['issues']
        samples.append(item)
    object_issues = {name: meshes[name]['issues'] for name in (first_name, second_name) if meshes[name]['issues']}
    return {'id': joint['id'], 'kind': kind, 'objects': [first_name, second_name],
            'manifest': joint, 'numerical_tolerance': tolerance,
            'tolerance_rule': '32 * float32_epsilon * max(1, abs(evaluated world coordinates))',
            'object_issues': object_issues, 'samples': samples,
            'passed': not object_issues and all(s['passed'] for s in samples)}


def check(manifest):
    import bpy
    joints = manifest.get('joints', [])
    if not joints or len({j['id'] for j in joints}) != len(joints):
        raise ValueError('manifest needs nonempty joints with unique IDs')
    names = set()
    for joint in joints:
        kind = joint.get('kind', 'floor_wall')
        names.update((joint['ceiling' if kind == 'wall_ceiling' else 'floor'],
                      joint['other_floor' if kind == 'floor_floor' else 'wall']))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = {name: evaluated_mesh(name, depsgraph) for name in sorted(names)}
    results = [check_joint(joint, meshes) for joint in joints]
    return {'schema_version': 1, 'units': 'scene_units', 'physical_meter_scale_verified': False,
            'evaluation': 'current scene frame; evaluated viewport modifiers; world triangle winding normals; bounded vertical Z intersections; support caps have Z as largest absolute normal component',
            'blender_frame': bpy.context.scene.frame_current,
            'limitations': 'Explicit stations only; no whole-seam or light-tightness proof. Z-dominant caps only; steeper side surfaces cannot establish floor/base support. Require fixed calibrated-camera seam closeups. Floor-floor paths require both named floors at every station.',
            'objects': {name: {'evaluated_vertices': len(mesh['vertices']), 'evaluated_triangles': len(mesh['triangles']),
                       'modifiers': mesh['modifiers']} for name, mesh in meshes.items()},
            'joints': results, 'passed': all(r['passed'] for r in results)}
