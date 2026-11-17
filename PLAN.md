# Workflow

## Registration

Raw video -> VidMap COLMAP reconstruction -> reviewed fixed alignment/partition ->
registered Blender scene -> regional/connection iteration -> global audit.
`third_party/vidmap` is a pinned Git submodule. Configure `runtime.vidmap.source`
and `runtime.vidmap.python`, install its dependencies separately, then use
`vibuild tool vidmap run --input VIDEO --output work/registration --device cuda`.
Retain source timestamps and calibration when exporting the registered input format in
[docs/inputs.md](docs/inputs.md). Freeze reviewed inputs under records before modeling.
Existing registered records skip this stage; model inference is not repeated per iteration.

## Layout

Each task uses `work/modeling_YYYYMMDD_HHMMSS/`:

```text
scene.blend                 latest saved, packed scene
plan.json, status.json      scope, dependencies, progress, acceptance
Rxx/, Lxx/, Jxx/             region, cycle closure, connection tasks
  evidence/                 source frames and fixed calibration
  iterations/NNN/           edit script and comparison cards
  scores/NNN/               independent assessment
.runtime/                   queue, locks, logs, and reusable cache
```

The reviewed partition supplies the execution order; no building-specific order is
built into the service. `vibuild task` initializes the layout from that partition and
a supplied registered scene. Task initialization does not run registration or validate scene quality.

## Iteration

1. Read current guidance, scoped geometry, source views, and exact score feedback.
   Missing floors, walls, doors, or connecting volumes precede appearance work.
2. Author one edit script and reuse the indexed job template. Submit once:
   `./vibuild tool jobs submit --task TASK_ROOT --manifest JOB_JSON`. End the turn.
3. The program executes `scene iterate`, `score prepare`, optional `score supplement`,
   then `score run`. It waits without inference and starts a fresh decision with the
   compact receipt. Assessors get only current packet/images, with no parent history.
4. Repair failed scores using their located findings. Collect only specifically missing
   evidence. After visual passage apply the closeout below and advance ready work.

Only mutation holds the writer. Read-only planning can overlap; assessments hold a
frozen-scene lock. All edits open the same current root scene. Save succeeds atomically
before rendering; subsequent render or scoring failure preserves that saved geometry.
Failed stages stop the batch without automatic retries or rollback. Historical receipts
are evidence, not current gates. See [job format](docs/execution-jobs.md).

Use tools/materials.py for selected fields/functions, with batched reads. Reuse fixed
views, calibration, and render settings. Program-managed hashes stay out of decisions.
No model runners, model-owned tool polling, or temporary execution wrappers. No queued
job means the service exits idle. A tool silent for 900 seconds fails locally. Pause is
`status.json.execution_paused` or `.runtime/main/STOP`; the service never clears either.

## Scope and dependencies

Every task names finite volumes and owned components. A dedicated Jxx connection task
samples approach/crossing views and repairs intervening space across its assigned group;
regional keyframes alone do not establish connection coverage. Use box unions, not the
enclosing AABB. Crossing trajectories suggest interfaces, not verified doorway geometry.

```bash
./vibuild tool connections --partition PARTITION_JSON --resources RECORD_RESOURCES --id J01 \
  --regions R01 R02 --output TASK_ROOT/J01/evidence/v001
```

Add connection tasks and dependencies to plan.json before dispatch. Connection scoring
uses `score prepare --connection CONNECTION_JSON` and retains all required interface
views. It checks passage, floors, walls/openings, ceilings, stairs/rails, and glazing;
missing required components block passage despite a high average.

Construction readiness and acceptance are separate. A connection can build against
existing endpoint geometry before endpoint scores pass; preserve final acceptance
dependencies. For a critical defect owned elsewhere, set the owner's work_objective,
blocked_by and resume_region, switch status.active_region, and end the decision. Return
after that repair and closeout. Do not reopen explicit user-deferred work or schedule
a second writer for an already-owned closure. Final global audit follows local work.

## Closeout

After visual passage inspect <=3 visible seam locations in <=2 existing views. Allow
at most one correction and one targeted visual review per task, including restarts.
With no correction, the initial inspection is the review. Seam-only edits do not
restart full scoring. No numerical contact, support-ray, clearance, or hardware-distance
checks here or in edit scripts. Record views, review/correction counts, current scene
binding, and deferred findings. Skipped checks are not passes.

Fixed weights: structure 30%, completeness 25%, alignment 20%, appearance 15%,
connectivity 10%. Local gate: total >=75, each dimension >=60, >=2 useful views,
confidence >=0.7, required evidence known, no critical defects. Global gate: equal-region
mean >=80, all regions and required connection/held-out gates passing. Jxx scores are
separate gates, not extra regions. Acceptance binds current evidence and geometry;
invalidate affected dependencies only. Explicit user closure is not a fabricated pass.

## Tools

- `vibuild tool vidmap`: configured upstream video registration; outputs and logs stay in work.

- `vibuild tool scene`: inspect, render, and atomic visual iteration.
- `vibuild tool score`: prepare, supplement, run, report, cancel-assessor.
- `vibuild tool jobs`: submit, work, status; service execution uses ordinary processes.
- `vibuild tool connections`: calibrated connection evidence and finite scope.
- `vibuild tool evaluate`: explicit numeric/final audits, outside ordinary iterations.
- `vibuild tool perceive`: optional configured DA3/SAM adapters, never a core dependency.
- `tools/materials.py`: selected JSON fields, Python functions, or line ranges.
- `tools/task_monitor.py`: local status/timing; cannot grant acceptance.
- `tools/token_report.py`: local rollout usage by session/time, without model calls.
- `skills/continuous-geometry/SKILL.md`: continuous swept rails and curved walls.

The service uses one decision prompt: instructions/decision.md. Runtime tools and model
settings come from the selected config; no machine paths, proxy, or site plan is implicit.
