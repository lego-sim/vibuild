# Workflow

1. **Initialize:** copy source videos and local configuration into a project.
2. **Register:** sample frames with decoded source timestamps; run the pinned VidMap
   checkout, or import a COLMAP reconstruction. Export fixed camera and RGB evidence.
3. **Review partition:** inspect the spatial preview, define finite regional boxes and
   alignment, and review topology. Approve explicitly with `--reviewed`; this freezes
   inputs and creates an empty Blender scene containing the registered cameras.
4. **Model:** each fresh Codex session works on the active region, connection, or cycle
   closure. It submits a scoped edit job; Python executes, renders, waits, and scores.
   The next decision receives current images and findings, without conversation history.
5. **Closeout:** after visual passage, inspect existing seam views. At most one
   correction and one review are allowed per stage, across restarts. Seam-only
   corrections do not restart scoring. Accept, defer, or hand off through `vibuild stage`.
6. **Audit:** after all stages pass, assess the current scene on fit and reserved views
   and recheck observed connections. Missing/deferred work prevents final acceptance.

One project has one persistent `scene.blend`. Edits save atomically; rendering or
scoring failure preserves the latest saved geometry. Region cores are box unions;
their enclosing box does not grant extra scope. Connections include intervening space.
Trajectory crossings support topology candidates, not measured doorway extents.

The fixed [scoring policy](../src/vibuild/assessment/policy.json) defines weights,
aggregation, local/global gates, and closeout limits. The program applies those gates;
assessors report visible evidence and located defects. Unknown or off-screen content
is not invented. Required missing structures take priority over material refinements.
Numeric depth/audit tools are opt-in; ordinary iterations use visual comparisons.

The writer and independent assessment locks protect the scene. `pause` sets a durable
stop marker and cancels managed work; `resume` clears it only after the service exits.
No inference is spent waiting for processes. A silent tool stage times out after 900
seconds, leaves a receipt, and requires a fresh decision; scripts are never auto-retried.

Project files:

```text
project.json, vibuild.json   project identity and resolved configuration
inputs/                     frozen record, camera calibration, RGB, source videos
partition.json, .html        reviewed scope, topology, and preview
scene.blend                 latest saved geometry and packed assets
plan.json, status.json       program-owned stages and progress
Rxx/, Lxx/, Jxx/             regional, cycle-closure, and connection evidence
  iterations/NNN/           edit and reference/before/after/difference cards
  scores/NNN/               independent assessment
.runtime/                   execution queue, sessions, locks, caches, full logs
audit.json                  final current-scene result
```
