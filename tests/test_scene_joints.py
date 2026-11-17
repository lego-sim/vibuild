"""Joint regression fixtures are generated only under pytest temporary directories."""

import json
import os
from pathlib import Path
import subprocess

import pytest

import vibuild.blender.scene as scene


@pytest.fixture(scope="module")
def joint_scene(tmp_path_factory):
    blender = os.environ.get("VIBUILD_TEST_BLENDER")
    if not blender:
        pytest.skip("set VIBUILD_TEST_BLENDER for evaluated floor joint regression")
    root = tmp_path_factory.mktemp("joint_scene")
    script = root / "fixture.py"
    script.write_text("""import bpy,sys,os
bpy.ops.wm.read_factory_settings(use_empty=True)
def mesh(name,vertices,faces):
 m=bpy.data.meshes.new(name);m.from_pydata(vertices,[],faces);m.update()
 o=bpy.data.objects.new(name,m);bpy.context.scene.collection.objects.link(o)
 return o
def floor(name,x0,x1,y0,y1,slope=0,reverse=False,solid=True):
 o=mesh(name,[(x0,y0,slope*x0),(x1,y0,slope*x1),(x1,y1,slope*x1),(x0,y1,slope*x0)],[(3,2,1,0) if reverse else (0,1,2,3)])
 if solid:
  mod=o.modifiers.new('evaluated thickness','SOLIDIFY');mod.thickness=.2;mod.offset=-1
 return o
def wall(name,slope,offset=0):
 bpy.ops.mesh.primitive_cube_add(size=2)
 o=bpy.context.object;o.name=name
 for v in o.data.vertices:
  x=v.co.x;y=v.co.y*.1
  v.co=(x,y,2 if v.co.z>0 else slope*x+offset)
 return o
floor('slope',-2,2,-1,1,.1)
floor('slope_edge',-1,2,-1,1,.1)
edge_reversed=floor('slope_edge_reversed',-1,2,-1,1,.1)
bpy.context.view_layer.objects.active=edge_reversed
bpy.ops.object.modifier_apply(modifier=edge_reversed.modifiers[0].name)
for polygon in edge_reversed.data.polygons:polygon.flip()
wall('horizontal',0)
wall('following',.1)
inverted=wall('inverted',.1)
for polygon in inverted.data.polygons:polygon.flip()
wall('embedded',.1,-.01)
wall('too_deep',.1,-.3)
floor('reversed',-2,2,-1,1,reverse=True,solid=False)
floor('flat',-2,2,-1,1)
a=floor('hole',-2,-.3,-1,1,solid=False)
b=floor('hole_part',.3,2,-1,1,solid=False)
bpy.ops.object.select_all(action='DESELECT');a.select_set(True);b.select_set(True)
bpy.context.view_layer.objects.active=a;bpy.ops.object.join()
floor('transition_left',-2,0,-1,1)
floor('transition_right',0,2,-1,1)
floor('transition_gapped',.08,2,-1,1)
ceiling=floor('ceiling',-2,2,-1,1,.1)
ceiling.location.z=2;ceiling.modifiers[0].offset=1
for name,damage in [('ceiling_edge',None),('ceiling_edge_broken','delete'),('ceiling_edge_bad_top','flip')]:
 o=floor(name,-2,1,-1,1,.1);o.location.z=2;o.modifiers[0].offset=1
 bpy.context.view_layer.objects.active=o
 bpy.ops.object.modifier_apply(modifier=o.modifiers[0].name)
 if damage=='flip':
  for polygon in o.data.polygons:
   if polygon.normal.z>.5:polygon.flip()
 if damage=='delete':
  import bmesh
  bm=bmesh.new();bm.from_mesh(o.data);bm.normal_update()
  bmesh.ops.delete(bm,geom=[f for f in bm.faces if f.normal.z>.5],context='FACES')
  bm.to_mesh(o.data);bm.free()
reversed_ceiling=floor('reversed_ceiling',-2,2,-1,1,.1)
reversed_ceiling.location.z=2;reversed_ceiling.modifiers[0].offset=1
bpy.context.view_layer.objects.active=reversed_ceiling
bpy.ops.object.modifier_apply(modifier=reversed_ceiling.modifiers[0].name)
for polygon in reversed_ceiling.data.polygons:polygon.flip()
sheet=floor('ceiling_sheet',-2,2,-1,1,.1,reverse=True,solid=False);sheet.location.z=2
missing=floor('missing_ceiling',2,3,-1,1,.1);missing.location.z=2
def actual_ceiling_z(x,y):return -2.728775-.00729*(x-46.83)+.0073*(y+6)
o=mesh('ceiling_scene_scale_edge',[(x,y,actual_ceiling_z(x,y)) for x,y in
 [(46.83,-12),(52,-12),(52,-2),(46.83,-2)]],[(0,1,2,3)])
mod=o.modifiers.new('normal extrusion','SOLIDIFY');mod.thickness=.08;mod.offset=1
o=wall('wall_scene_scale_edge',0)
for v in o.data.vertices:
 x=46.83+(v.co.x+1)*.5;y=-5.5+v.co.y*5
 v.co=(x,y,actual_ceiling_z(x,y) if v.co.z>1 else -5)
for name,offset in [('ceiling_following',0),('ceiling_embedded',.01),('ceiling_excess',.3)]:
 o=wall(name,0)
 for v in o.data.vertices:
  if v.co.z>1:v.co.z+=.1*v.co.x+offset
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[-1])
sys.stdout.flush();sys.stderr.flush();os._exit(0)
""")
    blend = root / "fixture.blend"
    subprocess.run(
        [
            blender,
            "--background",
            "-noaudio",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            str(blend),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return blender, blend


def test_evaluated_sloped_joints_and_failure_modes(tmp_path, joint_scene):
    blender, blend = joint_scene
    pairs = [
        ("gap", "slope", "horizontal", 0),
        ("corrected", "slope", "following", 0),
        ("embed", "slope", "embedded", 0.02),
        ("beyond_slab", "slope", "too_deep", 0.4),
        ("unbounded", "slope", "embedded", 0),
        ("reversed", "reversed", "horizontal", 0),
        ("hole", "hole", "horizontal", 0),
        ("inverted", "slope", "inverted", 0),
        ("edge", "slope_edge", "following", 0),
        ("edge_reversed", "slope_edge_reversed", "following", 0),
    ]
    joints = [
        dict(
            id=ident,
            floor=floor,
            wall=wall,
            path_xy=[[-1, 0], [1, 0]],
            max_spacing=0.25,
            max_embed=embed,
        )
        for ident, floor, wall, embed in pairs
    ]
    for ident, other in [
        ("transition", "transition_right"),
        ("transition_gap", "transition_gapped"),
    ]:
        joints.append(
            dict(
                id=ident,
                kind="floor_floor",
                floor="transition_left",
                other_floor=other,
                path_xy=[[0, -0.9], [0, 0.9]],
                max_spacing=0.3,
            )
        )
    manifest = tmp_path / "joints.json"
    manifest.write_text(json.dumps({"joints": joints}))
    original = scene.digest(blend)
    assert (
        scene.main(
            [
                "joints",
                "--blend",
                str(blend),
                "--manifest",
                str(manifest),
                "--blender",
                blender,
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    report = json.loads((tmp_path / "out/joints.json").read_text())
    results = {j["id"]: j for j in report["joints"]}
    assert report["source_sha256"] == original == scene.digest(blend)
    assert report["manifest_sha256"] == scene.digest(manifest)
    assert report["registered_cameras_unchanged"] and not report["saved_scene"]
    assert report["objects"]["slope"]["evaluated_vertices"] == 8
    assert (
        results["corrected"]["passed"]
        and results["embed"]["passed"]
        and results["transition"]["passed"]
    )
    for sample in results["corrected"]["samples"]:
        assert sample["floor"]["thickness"] == pytest.approx(
            0.2 * (1.01**0.5), abs=1e-6
        )
    assert results["edge"]["passed"]
    edge = results["edge"]["samples"][0]["floor"]
    assert any(n[2] < 0 for n in edge["top_coincident_normals"])
    assert all(n[2] > 0 for n in edge["top_normals"])
    assert results["gap"]["samples"][0]["signed_base_minus_floor_top"] == pytest.approx(
        0.1
    )
    assert len(results["gap"]["samples"]) == 9
    for ident, issue in [
        ("gap", "gap"),
        ("beyond_slab", "embed_beyond_floor_thickness"),
        ("unbounded", "embed_exceeds_declared_bound"),
        ("reversed", "reversed_floor_top_normal"),
        ("hole", "no_supporting_floor"),
        ("transition_gap", "no_supporting_floor"),
        ("inverted", "reversed_wall_base_normal"),
        ("inverted", "reversed_wall_top_normal"),
        ("edge_reversed", "reversed_floor_top_normal"),
    ]:
        assert not results[ident]["passed"]
        assert any(issue in s["issues"] for s in results[ident]["samples"])
    # The corrected fixture alone returns a successful gate, with the same source hash.
    manifest.write_text(json.dumps({"joints": [joints[1]]}))
    assert (
        scene.main(
            [
                "joints",
                "--blend",
                str(blend),
                "--manifest",
                str(manifest),
                "--blender",
                blender,
                "--output",
                str(tmp_path / "passed"),
            ]
        )
        == 0
    )


def test_polyline_includes_corner_and_rejects_unbounded_spacing():
    from vibuild.blender.joints import stations

    assert [p.tolist() for p in stations([[0, 0], [1, 0], [1, 1]], 0.5)] == [
        [0, 0],
        [0.5, 0],
        [1, 0],
        [1, 0.5],
        [1, 1],
    ]
    with pytest.raises(ValueError):
        stations([[0, 0], [1, 0]], float("nan"))


def test_wall_ceiling_slope_signed_gaps_and_required_slab(tmp_path, joint_scene):
    blender, blend = joint_scene
    pairs = [
        ("horizontal", "ceiling", "horizontal", 0),
        ("following", "ceiling", "ceiling_following", 0),
        ("embedded", "ceiling", "ceiling_embedded", 0.02),
        ("excess", "ceiling", "ceiling_excess", 0.4),
        ("reversed", "reversed_ceiling", "ceiling_following", 0),
        ("sheet", "ceiling_sheet", "ceiling_following", 0),
        ("missing", "missing_ceiling", "ceiling_following", 0),
    ]
    joints = [
        dict(
            id=ident,
            kind="wall_ceiling",
            ceiling=ceiling,
            wall=wall,
            path_xy=[[-1, 0], [1, 0]],
            max_spacing=0.25,
            max_embed=embed,
        )
        for ident, ceiling, wall, embed in pairs
    ]
    manifest = tmp_path / "ceilings.json"
    manifest.write_text(json.dumps({"joints": joints}))
    original = scene.digest(blend)
    assert (
        scene.main(
            [
                "joints",
                "--blend",
                str(blend),
                "--manifest",
                str(manifest),
                "--blender",
                blender,
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    report = json.loads((tmp_path / "out/joints.json").read_text())
    assert report["source_sha256"] == original == scene.digest(blend)
    results = {j["id"]: j for j in report["joints"]}
    assert results["following"]["passed"] and results["embedded"]["passed"]
    horizontal = results["horizontal"]["samples"]
    assert horizontal[0]["signed_ceiling_underside_minus_wall_top"] == pytest.approx(
        -0.1, abs=1e-6
    )
    assert horizontal[-1]["signed_ceiling_underside_minus_wall_top"] == pytest.approx(
        0.1, abs=1e-6
    )
    for sample in results["following"]["samples"]:
        assert sample["ceiling"]["thickness"] == pytest.approx(
            0.2 * (1.01**0.5), abs=1e-6
        )
        assert "floor" not in sample
    for ident, issue in [
        ("horizontal", "gap"),
        ("horizontal", "embed_exceeds_declared_bound"),
        ("excess", "embed_beyond_ceiling_thickness"),
        ("reversed", "reversed_ceiling_underside_normal"),
        ("reversed", "reversed_ceiling_top_normal"),
        ("sheet", "ceiling_without_verified_thickness"),
        ("missing", "no_ceiling_at_station"),
    ]:
        assert not results[ident]["passed"]
        assert any(issue in s["issues"] for s in results[ident]["samples"])


def test_near_vertical_projection_never_extrapolates_height_or_proves_support():
    import numpy as np
    from vibuild.blender.joints import intersections, support

    triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-8, 2.0]]])
    normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    mesh = {"triangles": triangle, "normals": normal}
    # Previously enormous barycentric margins accepted a height near 2000.
    hits = intersections(mesh, [0.5, 1e-5], 2e-4)
    assert hits and all(0 <= hit["z"] <= 2 for hit in hits)
    assert support(mesh, [0.5, 1e-5], 2e-4)["issues"] == ["no_supporting_floor"]
    assert not intersections(mesh, [0.5, 0.0024], 2e-4)


def test_ceiling_normal_extruded_finite_edge_thickness_and_embed_containment(
    tmp_path, joint_scene
):
    blender, blend = joint_scene
    pairs = [
        ("contact", "ceiling_edge", "ceiling_following"),
        ("broken", "ceiling_edge_broken", "ceiling_following"),
        ("bad_top", "ceiling_edge_bad_top", "ceiling_following"),
        ("embed", "ceiling_edge", "ceiling_embedded"),
        ("gap", "ceiling_edge", "horizontal"),
    ]
    joints = [
        dict(
            id=ident,
            kind="wall_ceiling",
            ceiling=ceiling,
            wall=wall,
            path_xy=[[1, -0.08], [1, 0.08]],
            max_spacing=0.04,
            max_embed=0.02,
        )
        for ident, ceiling, wall in pairs
    ]
    joints.append(
        {**joints[0], "id": "outside", "path_xy": [[1.0024, -0.08], [1.0024, 0.08]]}
    )
    joints.append(
        dict(
            id="scene_scale_edge",
            kind="wall_ceiling",
            ceiling="ceiling_scene_scale_edge",
            wall="wall_scene_scale_edge",
            path_xy=[[46.83, -6], [46.83, -5]],
            max_spacing=0.1,
        )
    )
    manifest = tmp_path / "edge.json"
    manifest.write_text(json.dumps({"joints": joints}))
    assert (
        scene.main(
            [
                "joints",
                "--blend",
                str(blend),
                "--manifest",
                str(manifest),
                "--blender",
                blender,
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    report = json.loads((tmp_path / "out/joints.json").read_text())
    assert report["checker_sha256"] == scene.digest(
        Path(scene.__file__).with_name("joints.py")
    )
    assert report["wrapper_sha256"] == scene.digest(Path(scene.__file__))
    results = {j["id"]: j for j in report["joints"]}
    assert results["contact"]["passed"]
    assert results["scene_scale_edge"]["passed"]
    for sample in results["scene_scale_edge"]["samples"]:
        assert sample["ceiling"]["normal_thickness_distance"] == pytest.approx(
            0.08, abs=2e-6
        )
    for sample in results["contact"]["samples"]:
        ceiling = sample["ceiling"]
        assert "top_z" not in ceiling
        assert ceiling["normal_thickness_distance"] == pytest.approx(0.2, abs=1e-6)
        assert ceiling["normal_top_hit_world"][0] < sample["xy"][0]
        assert sample["contact"] == "numerical_contact"
    for ident, issue in [
        ("broken", "ceiling_without_verified_thickness"),
        ("bad_top", "reversed_ceiling_top_normal"),
        ("embed", "embed_without_vertical_ceiling_containment"),
        ("gap", "gap"),
        ("outside", "no_ceiling_at_station"),
    ]:
        assert not results[ident]["passed"]
        assert any(issue in s["issues"] for s in results[ident]["samples"])


def test_slab_underside_is_cap_not_intervening_tilted_side():
    import numpy as np
    from vibuild.blender.joints import support

    triangles = np.array(
        [
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]],
            [[-1, -1, -0.08], [0, 1, -0.08], [1, -1, -0.08]],
            [[-1, 0, -0.00025], [1, 0, -0.00025], [1, 0.0008, -0.08025]],
        ],
        dtype=float,
    )
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    result = support({"triangles": triangles, "normals": normals}, [0, 0], 1e-5)
    assert not result["issues"]
    assert result["thickness"] == pytest.approx(0.08)
    assert result["side_intersections"]
    # An actual opposed cap remains a normal failure, even with an upward cap.
    both = np.concatenate([triangles, triangles[:1, ::-1]])
    normals = np.cross(both[:, 1] - both[:, 0], both[:, 2] - both[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    assert (
        "reversed_floor_top_normal"
        in support({"triangles": both, "normals": normals}, [0, 0], 1e-5)["issues"]
    )
