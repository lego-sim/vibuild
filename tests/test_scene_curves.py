"""Geometry invariants plus an optional actual Blender integration smoke test."""

import json
import os
from pathlib import Path
import subprocess
from collections import Counter

import numpy as np
import pytest

from vibuild.blender.curves import bezier, line, sample_path, tube_mesh, wall_mesh


def assert_solid(mesh):
    edges = Counter()
    volume = 0.0
    for face in mesh.faces:
        for a, b in zip(face, (*face[1:], face[0])):
            edges[tuple(sorted((a, b)))] += 1
        for i in range(1, len(face) - 1):
            a, b, c = mesh.vertices[list((face[0], face[i], face[i + 1]))]
            volume += np.dot(a, np.cross(b, c)) / 6
    assert set(edges.values()) == {2}, "Every edge must join exactly two faces"
    assert volume > 0, "Outward face winding must enclose positive volume"
    return volume


def test_straight_landing_and_tangential_curve_keep_exact_join_and_endpoints():
    p = sample_path(
        [
            line((0, 0, 0), (1, 0, 0)),
            bezier((1, 0, 0), (2, 0, 0), (2, 1, 1), (2, 2, 1)),
        ],
        max_step=0.1,
    )
    assert np.array_equal(p[0], [0, 0, 0]) and np.array_equal(p[-1], [2, 2, 1])
    assert np.count_nonzero(np.all(p == [1, 0, 0], axis=1)) == 1
    assert np.all(p[p[:, 0] <= 1, 1:] == 0)
    assert np.max(np.linalg.norm(np.diff(p, axis=0), axis=1)) <= 0.10000001
    mesh = tube_mesh(p, radius=0.03)
    assert_solid(mesh)
    rings = mesh.vertices.reshape(len(p), 16, 3)
    assert np.allclose(rings.mean(axis=1), p)
    assert np.allclose(np.linalg.norm(rings - p[:, None, :], axis=2), 0.03)


def test_gaps_and_unintended_tangent_breaks_reject():
    with pytest.raises(ValueError, match="endpoint"):
        sample_path([line((0, 0, 0), (1, 0, 0)), line((1.001, 0, 0), (2, 0, 0))])
    with pytest.raises(ValueError, match="Tangent break"):
        sample_path([line((0, 0, 0), (1, 0, 0)), line((1, 0, 0), (1, 1, 0))])
    p = sample_path(
        [line((0, 0, 0), (1, 0, 0)), line((1, 0, 0), (1, 1, 0), corner=True)],
        max_step=1,
    )
    assert np.array_equal(p, [[0, 0, 0], [1, 0, 0], [1, 1, 0]])


def test_wall_continuity_sloped_heights_and_exact_straight_volume():
    p = sample_path([line((0, 0, 0), (2, 0, 0))])
    mesh = wall_mesh(p, thickness=0.2, bottom=p[:, 0] * 0.1, top=2 + p[:, 0] * 0.1)
    assert assert_solid(mesh) == pytest.approx(0.8)
    assert len(mesh.vertices) == 4 * len(p)
    # One continuous curved wall retains shared vertices across all sampled sections.
    curved = sample_path([bezier((0, 0, 0), (1, 0, 0), (2, 1, 0), (2, 2, 0))])
    assert_solid(wall_mesh(curved, thickness=0.1, bottom=0, top=2))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: sample_path([line((0, 0, 0), (1, 0, 0))], max_step=0),
        lambda: tube_mesh([[0, 0, 0], [0, 0, 0]], radius=0.1),
        lambda: tube_mesh([[0, 0, 0], [1, 0, 0], [0, 0, 0]], radius=0.1),
        lambda: wall_mesh([[0, 0, 0], [1, 0, 0]], thickness=0.1, bottom=2, top=1),
        lambda: wall_mesh(
            [[0, 0, 0], [1, 0, 0], [1, 0.01, 0]], thickness=1, bottom=0, top=2
        ),
    ],
)
def test_invalid_or_folded_geometry_rejects(operation):
    with pytest.raises(ValueError):
        operation()


def test_blender_replaces_geometry_preserving_scene_contract(tmp_path):
    blender = os.environ.get("VIBUILD_TEST_BLENDER")
    if not blender:
        pytest.skip("Set VIBUILD_TEST_BLENDER for actual Blender smoke test")
    script = tmp_path / "smoke.py"
    tools = str(Path(__file__).parents[1] / "src")
    script.write_text(f"""import sys, os, json
import bpy, bmesh
sys.path.insert(0, {tools!r})
from vibuild.blender.curves import *
bpy.ops.wm.read_factory_settings(use_empty=True)
collection=bpy.data.collections.new("R05_owned")
bpy.context.scene.collection.children.link(collection)
obj=bpy.data.objects.new("R05_rail", bpy.data.meshes.new("rail"))
collection.objects.link(obj)
obj.location=(3,4,5)
obj["owner"]="R05"
material=bpy.data.materials.new("existing_rail")
obj.data.materials.append(material)
path=sample_path([line((0,0,0),(1,0,0)),bezier((1,0,0),(2,0,0),(2,1,1),(2,2,1))])
replace_mesh(obj,tube_mesh(path,radius=.03),smooth_sides=True)
assert tuple(obj.location)==(3,4,5) and obj["owner"]=="R05"
assert list(obj.users_collection)==[collection] and obj.data.materials[0]==material
bm=bmesh.new();bm.from_mesh(obj.data)
assert all(edge.is_manifold for edge in bm.edges)
assert bm.calc_volume(signed=True)>0
bm.free()
wall=bpy.data.objects.new("R05_wall",bpy.data.meshes.new("wall"))
collection.objects.link(wall)
replace_mesh(wall,wall_mesh(path,thickness=.1,bottom=0,top=2))
bm=bmesh.new();bm.from_mesh(wall.data)
assert all(edge.is_manifold for edge in bm.edges)
assert bm.calc_volume(signed=True)>0
bm.free()
old=obj.data
obj.data.uv_layers.new(name="protected_uv")
try: replace_mesh(obj,tube_mesh(path,radius=.04))
except ValueError: pass
else: raise AssertionError("Existing UV must not silently disappear")
assert obj.data==old
print(json.dumps({{"rail_vertices":len(obj.data.vertices),"wall_vertices":len(wall.data.vertices),"scene_saved":bool(bpy.data.filepath)}}))
sys.stdout.flush();sys.stderr.flush();os._exit(0)
""")
    result = subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = next(
        json.loads(row)
        for row in result.stdout.splitlines()
        if row.startswith('{"rail_vertices"')
    )
    assert report["rail_vertices"] > report["wall_vertices"] > 8
    assert report["scene_saved"] is False
