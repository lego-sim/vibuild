# Targeted structure and material checks

Use source/render comparisons for visible structure and material defects. Numerical
contact, support, hardware-distance and clearance diagnostics are excluded from modeling
iterations and closeout, including assertions inside edit scripts. Do not recreate them
from historical scripts. Keep camera/scale, ownership and geometry-integrity protections.
PLAN.md defines the one-review visual closeout budget; skipped checks are not passes.

## Calibration and depth

- Keep camera poses, world frame and scale fixed. Score native floating-point depth
  and confidence; retain original RGB separately. Never score colorized depth or a
  resized grid without its calibrated K/pixel transform.
- `X_scene = B @ (X_combined - origin)`: stored scene XYZ already includes this transform.
  Original H5 depth may need per-image lift/video Sim3 scale; registered DA3 XYZ must
  not receive it again. Preserve recorded pixel-centre, crop and letterbox conventions.
- Camera `rotation` is OpenCV camera-to-scene: world-to-camera uses `R=rotation.T`,
  `t=-R@center`; Blender uses `rotation @ diag(1,-1,-1)`. Distinguish optical Z/ray range.
- Exclude transparent/reflected/unreliable depth from opaque-plane fits. Report coverage
  and residuals. A floor disagreement does not establish a step/ramp. Verify sills and
  cross-frame boundaries first; repeated predictions from one camera are correlated.

## Connected structure

- Check floor/ceiling interfaces for gaps, duplicates and unobserved steps. Verify
  normals, finite coordinates, thickness and bounded booleans. Do not remove an entire
  long wall to repair one view or expand a boolean outside the intended bounds.
- Treat stair flights, landings, enclosing walls, soffits and rails as connected parts.
  Confirm one-/two-sided rails and metal posts/infill. Straight-stair checks do not
  validate curved stairs; camera clearance is not a human swept-volume test.
- Inspect walls and columns from both approaches. Partial arcs can fit false radii;
  fits at parameter bounds need more evidence. Avoid jagged silhouette-only meshes or
  unsupported space completion from a weak depth plane.

### Visible wall and floor joins

Model wall bases and tops to follow the observed floor and ceiling surfaces. Inspect
visible gaps or penetrations in the fixed comparison views and repair actual findings.
Do not generate raycast/contact manifests, numerical tolerances or support assertions.

## Continuous rails and curved walls

Use [continuous geometry](skills/continuous-geometry/SKILL.md) and `tools/scene_curves.py`
inside the managed edit. Source-controlled straight and cubic sections share endpoints
and intended tangents. Sweep one tube or wall ribbon per physically continuous run;
retain real fittings and corners. Do not approximate curved walls with separate cuboids
or smooth only the shading of disconnected rail rods. Compare silhouette, constant
section, landing transition and backing-wall clearance from several source cameras.
These are targeted repair checks, not a new mandatory full-scene suite each iteration.

## Connection coverage

Jxx evidence must include crossing and both approaches, plus context, to assess the
completeness of the space between regions.
Inspect missing passage volume, floors/landings, walls/openings, ceiling/soffits, rails
and doors/glazing. Unknown coverage blocks connection acceptance; an unmodeled gap cannot
be declared out of scope merely because neither region's original primary views show it.

## Doors and glazing

- Confirm leaf count, hinge pivot, swing direction and observed state. Fit jamb/head/
  sill across frames, maintaining clearances and floor contact. Preserve CAD provenance.
  Wood accents are not extra leaves; a single open state does not justify animation.
- Check handle shape, orientation, height and front/back attachment. Update transforms
  before placing hardware and inspect world bounds.
- Separate outer frames, mullions/transoms, leaves and individual panes. Verify counts,
  alignment, adjacency and gaps across views. Whole-window masks cannot prove these
  details. Use chronological frames for temporal segmentation; do not lower confirmation
  thresholds merely to force detections. Empty masks remain missing evidence.
- Reuse one Glass Master with plausible thickness and local tint/roughness. Keep pane
  and frame geometry distinct. Through-glass depth is not an opaque-wall label; missing
  outdoor reflection evidence does not justify invented indoor walls.

## Materials and independent decals

- Match physical tile density, size/orientation, grout, joints, inlays, tread/riser edges
  and gloss. Shader bricks are not measured tile instances. Check continuity across views.
- Reuse PBR maps; edit local material instances without changing other regions' Masters.
  Separate geometry, albedo, roughness and lighting changes. Never hide errors per camera.
- Each poster/sign uses an independent rectified source image with frame, UV and size
  provenance. Do not invent text or combine unrelated posters into one atlas.
- Re-render the fixed affected/interface views after edits. Record remaining defects
  and regressions explicitly. Final evidence follows the [audit contract](docs/audit-workflow.md).
