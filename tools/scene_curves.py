"""Source-controlled continuous rails and walls; import from a scene iterate script.

All coordinates/lengths are in the caller's object-local coordinate system. No
registration changes, Blender operators, file writes, or automatic curve fitting.
"""
from dataclasses import dataclass
import math

import numpy as np


def _points(value, minimum=2):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum or not np.isfinite(points).all():
        raise ValueError(f"Expected at least {minimum} finite XYZ points")
    return points


def _unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-10:
        raise ValueError("Degenerate direction or repeated path point")
    return vector / length


def _positive(value, label):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def line(start, end, *, corner=False):
    """Exact straight section. corner=True permits a hard join at its start."""
    return {"points": [start, end], "corner": corner}


def bezier(start, control1, control2, end, *, corner=False):
    """Cubic section; handles explicitly control endpoint tangent directions."""
    return {"points": [start, control1, control2, end], "corner": corner}


def sample_path(sections, *, max_step=0.1, tolerance=0.002, join_angle=5.0):
    """Sample connected line/cubic sections without smoothing their controls.

    Cubics subdivide until control-polygon length <= max_step and distance
    from controls to the endpoint chord <= tolerance. Unexpected gaps/tangent
    breaks raise; deliberate corners require corner=True on the next section.
    Open paths only. Limits use scene units, not assumed metres.
    """
    _positive(max_step, "max_step")
    _positive(tolerance, "tolerance")
    if not math.isfinite(join_angle) or not 0 <= join_angle < 180:
        raise ValueError("join_angle must be in [0, 180)")
    result, previous = [], None

    def subdivide(p, depth=0):
        chord = p[-1] - p[0]
        length = np.linalg.norm(chord)
        if length < 1e-10:
            raise ValueError("Cubic endpoints must be distinct")
        flatness = max(np.linalg.norm(np.cross(q - p[0], chord)) / length for q in p[1:-1])
        if np.linalg.norm(np.diff(p, axis=0), axis=1).sum() <= max_step and flatness <= tolerance:
            result.append(p[-1])
            return
        if depth >= 20:
            raise ValueError("Curve exceeds subdivision limit; check controls/tolerance")
        a, b, c = (p[:-1] + p[1:]) / 2
        d, e = (a + b) / 2, (b + c) / 2
        middle = (d + e) / 2
        subdivide(np.array([p[0], a, d, middle]), depth + 1)
        subdivide(np.array([middle, e, c, p[3]]), depth + 1)

    for section in sections:
        p = _points(section["points"])
        if len(p) not in (2, 4):
            raise ValueError("Each section must be a line (2 points) or cubic (4 points)")
        first, last = _unit(p[1] - p[0]), _unit(p[-1] - p[-2])
        if previous is not None:
            endpoint, tangent = previous
            if np.linalg.norm(endpoint - p[0]) > 1e-8:
                raise ValueError("Sections must share an exact endpoint")
            angle = math.degrees(math.acos(np.clip(np.dot(tangent, first), -1, 1)))
            if angle > join_angle and not section.get("corner", False):
                raise ValueError("Tangent break: align handles or declare an intentional corner")
        else:
            result.append(p[0])
        if len(p) == 2:
            count = max(1, math.ceil(np.linalg.norm(p[1] - p[0]) / max_step))
            result.extend(p[0] + (p[1] - p[0]) * i / count for i in range(1, count + 1))
        else:
            subdivide(p)
        previous = p[-1], last
    path = _points(result)
    if np.linalg.norm(path[-1] - path[0]) < 1e-8:
        raise ValueError("Closed paths require an explicit closure design; this tool accepts open paths")
    return path


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: list
    side_faces: int


def _loft(rings):
    count, width, _ = rings.shape
    faces = []
    for i in range(count - 1):
        for j in range(width):
            k = (j + 1) % width
            faces.append((i * width + j, i * width + k, (i + 1) * width + k, (i + 1) * width + j))
    side_faces = len(faces)
    faces.extend([tuple(reversed(range(width))), tuple((count - 1) * width + j for j in range(width))])
    return Mesh(rings.reshape(-1, 3), faces, side_faces)


def tube_mesh(path, *, radius, sides=16):
    """One capped tube using parallel-transport rings (no independently joined rods).

    Tight bends must have clearance above radius. Source-defined sharp joints
    should use separate designed fittings; reversal/degenerate tangents reject.
    """
    p = _points(path)
    _positive(radius, "radius")
    if not isinstance(sides, int) or sides < 3:
        raise ValueError("sides must be an integer >= 3")
    directions = np.array([_unit(v) for v in np.diff(p, axis=0)])
    tangents = np.array([directions[0], *[_unit(a + b) for a, b in zip(directions[:-1], directions[1:])], directions[-1]])
    # Reject locally folded offsets rather than silently generating self intersections.
    for i in range(1, len(p) - 1):
        turn = math.acos(np.clip(np.dot(directions[i - 1], directions[i]), -1, 1))
        if radius * math.tan(turn / 2) >= min(np.linalg.norm(p[i] - p[i - 1]), np.linalg.norm(p[i + 1] - p[i])):
            raise ValueError("Tube radius is too large for this bend/sample spacing")
    axis = np.eye(3)[np.argmin(abs(tangents[0]))]
    normal = _unit(np.cross(tangents[0], axis))
    angles = np.arange(sides) * 2 * math.pi / sides
    rings = []
    previous = tangents[0]
    for point, tangent in zip(p, tangents):
        cross = np.cross(previous, tangent)
        sine, cosine = np.linalg.norm(cross), np.dot(previous, tangent)
        if sine > 1e-10:
            rotation = cross / sine
            normal = normal * cosine + np.cross(rotation, normal) * sine + rotation * np.dot(rotation, normal) * (1 - cosine)
        normal = _unit(normal - tangent * np.dot(normal, tangent))
        binormal = np.cross(tangent, normal)
        rings.append(point + radius * (np.cos(angles)[:, None] * normal + np.sin(angles)[:, None] * binormal))
        previous = tangent
    return _loft(np.asarray(rings))


def wall_mesh(path, *, thickness, bottom, top, max_miter=4.0):
    """Capped continuous wall with shared offset joins, not cuboid strips.

    Uses path XY; bottom/top are absolute object-local Z scalars or one value per
    sampled point. Supply heights explicitly for slopes; path Z is ignored.
    max_miter bounds intentional sharp corners. Side shading stays caller-owned.
    """
    p = _points(path)
    _positive(thickness, "thickness")
    _positive(max_miter, "max_miter")
    lower, upper = [np.broadcast_to(np.asarray(v, dtype=float), (len(p),)) for v in (bottom, top)]
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower):
        raise ValueError("Each finite wall top must be above its bottom")
    directions = np.array([_unit(v) for v in np.diff(p[:, :2], axis=0)])
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    offsets = [normals[0] * thickness / 2]
    for a, b in zip(normals[:-1], normals[1:]):
        bisector = _unit(a + b)
        factor = 1 / np.dot(bisector, b)
        if factor > max_miter:
            raise ValueError("Wall corner exceeds max_miter; use a source-supported rounded or chamfered section")
        offsets.append(bisector * thickness / 2 * factor)
    offsets.append(normals[-1] * thickness / 2)
    offsets = np.asarray(offsets)
    # Detect offset reversal along either side, which produces folded wall faces.
    left, right = p[:, :2] + offsets, p[:, :2] - offsets
    if any(np.any(np.sum(np.diff(side, axis=0) * directions, axis=1) <= 0) for side in (left, right)):
        raise ValueError("Wall thickness folds an offset; reduce thickness or correct the path")
    rings = np.array([[(l[0], l[1], lo), (l[0], l[1], hi), (r[0], r[1], hi), (r[0], r[1], lo)]
                      for l, r, lo, hi in zip(left, right, lower, upper)])
    return _loft(rings)


def replace_mesh(obj, mesh, *, smooth_sides=False, side_material=0, cap_material=0):
    """Replace an explicitly selected object's mesh; never save or delete objects.

    Keeps object transform, custom properties, collections, modifiers and material
    slots. Topology-dependent UVs/attributes/vertex groups must be rebuilt explicitly;
    reject those inputs instead of silently discarding them. New objects can be
    created explicitly by the caller and then passed here.
    """
    import bpy
    if obj.type != "MESH":
        raise ValueError("Target must be a mesh object")
    old = obj.data
    custom_attributes = [a for a in old.attributes if not a.is_internal and a.name not in {"position", ".edge_verts", ".corner_vert", ".corner_edge"}]
    if old.shape_keys or old.uv_layers or obj.vertex_groups or custom_attributes:
        raise ValueError("Target has topology-dependent data; explicitly rebuild/preserve it before replacement")
    slots = list(old.materials)
    if any(not isinstance(i, int) or i < 0 or i >= max(1, len(slots)) for i in (side_material, cap_material)):
        raise ValueError("Material index outside preserved slots")
    new = bpy.data.meshes.new(old.name + "_continuous")
    try:
        new.from_pydata(np.asarray(mesh.vertices).tolist(), [], mesh.faces)
        if new.validate():
            raise ValueError("Generated mesh required repair")
        for material in slots:
            new.materials.append(material)
        for polygon in new.polygons:
            side = polygon.index < mesh.side_faces
            polygon.use_smooth = smooth_sides and side
            polygon.material_index = side_material if side else cap_material
        new.update()
        obj.data = new
    except Exception:
        bpy.data.meshes.remove(new)
        raise
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj
