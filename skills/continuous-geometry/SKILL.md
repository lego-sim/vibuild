---
name: continuous-geometry
description: Repair segmented rails and curved walls using source-controlled continuous paths, exact straight landings, and shared mesh joins in the evolving Blender scene.
---

Use [scene_curves.py](../../tools/scene_curves.py) inside the existing `scene iterate`
edit script. The helper performs no file saves and never changes registration.

1. Inspect original frames from multiple registered viewpoints and current object
   geometry. Identify continuous curved runs, exact straight landings, and actual
   corners/fittings. Place endpoints and Bézier handles from that evidence; do not
   fit one global smoothing spline through all old piece endpoints.
2. Express a run with `line(start, end)` and
   `bezier(start, control1, control2, end)`. Adjacent sections share exact endpoints;
   align their endpoint handles for a smooth transition. Set `corner=True` on the
   next section only for an observed hard corner. Preserve straight runs exactly.
3. `sample_path(sections, max_step=..., tolerance=...)` rejects gaps and unexpected
   tangent breaks. Choose tolerances in existing scene units, relative to rail/wall
   size and image detail; these units are not independently verified metres.
4. `tube_mesh(path, radius=..., sides=16)` produces one capped continuous rail.
   `wall_mesh(path, thickness=..., bottom=..., top=...)` produces a single capped
   wall with shared offset joins. Wall XY follows the path; bottom/top specify
   absolute local Z as scalars or one value per sampled point. Slopes require
   explicit heights. Do not replace curved walls with independent cuboid strips.
5. Pass an explicitly owned mesh object to `replace_mesh(obj, mesh, ...)`.
   Coordinates are object-local: transform source world coordinates through
   `obj.matrix_world.inverted()` first. Object transform, material slots,
   collections, modifiers and ownership remain. Existing UVs, shape keys, custom
   attributes or vertex groups cause rejection; preserve/rebuild those deliberately,
   never clear them blindly just to bypass the guard. Material indices for side/cap
   faces are explicit. Smooth tube sides, keep end caps flat. Wall shading is flat
   by default; handle curved vertical surfaces separately from straight top/bottom.
6. Remove superseded segments only within the assigned ownership scope after the
   continuous replacement is ready. Preserve source-required fittings and gaps.
   Compare fixed source/before/after views, including both ends and landing joins.
   Local offset guards do not prove distant sections cannot self-intersect: inspect
   the full curve's clearance, backing-wall penetration and endpoint alignment.
   Follow the score-first loop. Inspect contacts in the comparison images; omit
   numerical contact/support/clearance diagnostics and pre-save assertions about them.

Example inside a managed edit (the points are illustrative, not source evidence):

```python
from scene_curves import line, bezier, sample_path, tube_mesh, replace_mesh

sections = [line((0, 0, 0), (1, 0, 0)),
            bezier((1, 0, 0), (2, 0, 0), (2, 1, 1), (2, 2, 1))]
path = sample_path(sections, max_step=0.1, tolerance=0.002)
replace_mesh(owned_rail, tube_mesh(path, radius=0.03), smooth_sides=True)
```

The edit script must add the repository's `tools/` directory to `sys.path` if it
is not already available. Do not write or copy a `.blend`; `scene iterate` owns
the single root save. Closed loops and complex wall junctions need an explicit
closure/junction design; this helper intentionally accepts open paths only.
