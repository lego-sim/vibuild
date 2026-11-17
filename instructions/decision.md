Make one scoped modeling decision. PROJECT_CONTRACT and MATERIAL_INDEX are already
supplied; do not reread them. Inspect attached current images directly, in listed order.
Use indexed paths for additional views only when needed. Older scores/objects are
historical; visibility does not grant ownership. Read PLAN.md only for transitions.

Aim for 4-6 terminal calls: one batched material read, necessary geometric calculations,
write edit.py and update job.json together, submit once. This is a target, not a reason
to skip necessary evidence. Do not echo authored scripts, manifests or successful checks.
Return only decision-relevant results and artifact paths; full logs stay on disk.

Read exact fields/functions with the task material reader:
python3 PROJECT_ROOT/tools/materials.py 'FILE.json#/field/0' 'EDIT.py#function' 'FILE#L10-L30'
No selector gives a JSON/Python index; an explicit selector returns the complete selected
value without truncation. Batch independent reads in one command. Avoid whole JSON dumps, inventories, HTML, glob scans and script histories.
Use analysis_python for calculations; print only the values needed to choose the edit.
Before acceptance, read all unresolved findings and uncertainties from the indexed score.

Reuse prepared views, render settings and job_template. Write a self-contained edit.py;
old scripts are geometry references, not code to execute/import. Do not copy old numeric
contact/clearance assertions. For curved geometry read skills/continuous-geometry/SKILL.md
and selected functions from tools/scene_curves.py. Use CLI --help only for a new option.
The program handles syntax, scene binding, execution and waiting. Do not inspect internal
binding receipts, recheck calibrated views or dump source caches/inventories. Findings
include spatial candidate objects selected by the program. These are approximate AABB
matches, not confirmed identities or write permission; inspect only the implicated geometry.
Batch iterate -> prepare -> optional supplement -> run; docs/execution-jobs.md describes
generated diagnostics. Split only when a new decision genuinely needs new evidence.

Read MATERIAL_INDEX.guidance.text. Submitting confirms receipt; the program records the
supplied version and actual session automatically. Submit once, return its receipt and end:
python3 PROJECT_ROOT/tools/jobs.py submit --task TASK_ROOT --manifest JOB_JSON
Never execute/poll Blender, scoring or agents. Apply the contract's visual gate and
one-review closeout; preserve closures and dependency ownership. Pause overrides work.

Resolve critical missing structure before appearance work. If the highest-impact blocker
belongs to another task, update plan.json with blocked_by and the owner's work_objective,
set status.active_region to that owner and end without a cosmetic job. The service starts
a fresh owner decision. Existing usable geometry permits connection construction before
endpoint acceptance; preserve acceptance dependencies separately. After the owner's repair
and bounded closeout, return to resume_region for assessment. Never waive its unresolved
score, drop difficult views or repeat material edits while the structural blocker remains.

When routing.work_mode is inspect_once, submit only the supplied prepare/run assessment
job on first visit. After its result, record findings and advance to resume_region without
an edit or rescore loop. If no next inspection is assigned, report and end without a job.
Explicit user-deferred tasks must not be reopened by owner routing.
