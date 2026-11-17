# Targeted audits and final acceptance

Ordinary local repair uses visual cards and affected-interface checks in [PLAN.md](../PLAN.md).
Its one-review closeout budget applies across restarts; this diagnostic reference must
not trigger repeated local audits or full rescoring after a seam correction.
Run a numeric audit for a concrete uncertainty, milestone or final acceptance. Keep
native depth, semantic/components, appearance and plausibility separate. Missing
mandatory evidence remains unknown/blocked; partial local work remains explicitly incomplete.

## Fixed evidence and paired comparison

Freeze frame IDs, calibrated K/pose, reference NPZs, required classes, thresholds and
optional RGB before comparing. Use representative model frames for repair and held-out
audit frames for acceptance. Do not change masks, classes or frame sets after scoring.

```bash
./vibuild tool evaluate compare --manifest protocol.json \
  --baseline baseline.blend --candidate candidate.blend --output comparisons/v001
```

Add `--check-only` to preflight without rendering. Explicitly calibrated RGB can be
assessed when depth is absent, but it does not fill missing geometry/component evidence.
Depth and RGB render at their respective native grids/K/poses; scoring never silently
resizes them. Malformed calibration is an input error. Outputs include paired score
reports and `comparison.json` with gains, regressions, cache use and timings.

A successful command means reports were produced, not that gates passed. The direct
`evaluate --require-passed` option makes its exit status enforce the structure gate.
Changed geometry needs its own `--candidate-review` plausibility evidence; unsupported
checks remain unknown. Changing renderer settings or reference evidence changes the
protocol: remeasure both scenes under one fixed protocol before claiming improvement.

Candidate-selection policy permits up to **1 percentage point** of depth coverage loss
only if coverage still meets the frozen floor (default 95%) and depth error does not
worsen. Raw losses stay visible; a measured gain is still required. Missing metrics,
lost passed gates, excessive losses and matched-component regressions block selection.
Checklist fractions while plausibility is blocked, and instance counts with no semantic
matches, are advisory. These tolerances do not relax final acceptance requirements.

Review existing paired reports without rendering or modifying them:

```bash
./vibuild tool evaluate review --baseline baseline_scores.json \
  --candidate candidate_scores.json --output candidate_review.json
```

The report records source hashes, protocol and policy. It supports independent regional
review; it does not itself publish a scene or advance a region. Preserve unresolved
criteria and ensure evidence refers to the exact candidate revision.

## Geometry, rendering and cache

Build evaluation NPZs with float depth, K/pose, reference confidence/opaque mask,
semantic IDs and part-instance IDs. Score depth on the stored native output grid;
previews may be smaller but are not numeric depth evidence. Require relevant per-class
IoU, one-to-one instance matching/count/coverage and depth residuals. Plausibility claims
need evidence paths. Optional RGB scoring needs explicit matching appearance calibration.

The standard geometry renderer uses evaluated mesh BVH and fixed-seed Blender ray
cross-checks. Pure RGB previews require no depth ray check. Hidden/excluded collections
stay hidden; preserve instances and negative scales. `scene render --ray-backend scene`
is the slow diagnostic oracle. RGB uses deterministic CPU Cycles; freeze sample/thread
settings for paired comparisons. Use shared tools rather than copied render scripts.

Sessions inherit `VIBUILD_CACHE_DIR` with access to that exact tool-managed directory,
not sibling workspaces. Cache keys cover scene/dependency content, cameras/grids, settings,
code and runtime; score keys also cover references, masks and review evidence. Identical
requests coalesce; changed inputs invalidate reuse. Failed or partial outputs are never
cache hits. Outputs are independent copies. A cache hit is not evidence of acceptance.

For a bounded bpy edit followed by an audit, use `evaluate compare --build-script edit.py`
with the fixed manifest and baseline. The wrapper owns save/exit and verifies cameras
and scale. Scripts must not overwrite sources or launch background work. `scene patch`
provides a standalone build; `scene inspect` exposes geometry/materials/provenance.
Use new output paths and keep one-off edit scripts inside the task workspace.

## Model evidence

Perception adapters are optional; ordinary modeling and RGB scoring need neither DA3
nor SAM. Configure their Python, source, and checkpoint fields only when invoking
`tool perceive` (`--help` lists the adapter contract); weights stay outside records.
DA3 receives registered world-to-camera
poses, exact K and scale alignment; preserve returned float depth/confidence/XYZ and
output K/E. Its configured processing grid is not original-resolution RGB depth.
SAM3.1 preserves native masks and explicit pixel mapping; instance IDs are concept/
session scoped. Empty detections are missing evidence, not perfect reconstruction.
A whole-window mask does not establish separate panes, frames or mullions.

Do not substitute backends silently or infer readiness from a path alone. Optional
retrieval/generation models do not establish architectural truth. Where a reviewed
object-label mapping is used, require exact existing object names, supported classes
and evidence; apply it only in the disposable rendering process and include it in
protocol hashes. Follow relevant [structure checks](../STRUCTURE_CHECKS.md).
