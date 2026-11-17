# Agent contract

Runtime prompts and generated technical prose are English. Read PLAN.md for task
transitions. Frozen record resources, cameras, world transform, and scale are read-only;
absolute metres are unverified unless independently established.

Use one self-contained task-root `scene.blend`, containing the latest saved progress.
Fresh session means fresh conversation, not an empty scene. Use `scene iterate --script`
for packed atomic saves. Never retain iteration blends, restore old scenes, or merge
regional projects. Acceptance is separate metadata; render/score failure does not roll
back saved geometry.

Each task has finite scope and explicit object ownership. Box unions do not include
their enclosing AABB. Camera ownership does not limit visible surfaces: preserve
source-visible structures beyond a core. A connection task owns its assigned regions
and intervening volumes. Shared components have one owner; unrelated geometry is
read-only. Do not fabricate topology or alter registered cameras to improve a score.

The program service owns execution, waiting, timeouts, receipts, and assessor dispatch.
A fresh decision authors a bounded script/job, submits once, and ends. Never spawn
LLM runners, poll tools, or invent execution wrappers. Assessors receive only their
packet/images. Actual mutation holds the writer; independent assessments freeze writes
until recorded or explicitly cancelled after the assessor stops. Pause/STOP blocks work.

Use the supplied MATERIAL_INDEX, attached images, exact findings, and job template.
Read selected fields/functions with tools/materials.py; batch independent reads.
Avoid full inventories, directory scans, historical conversations, and hash bookkeeping.
Consume current user guidance before mutation. Submission records receipt, not repair.
Declare generated diagnostics in the supplement stage; the program binds their inputs.
Return results and artifact paths; full logs stay on disk. Record actual IDs and timings.

Prioritize missing major structures over appearance. Approximate object matches do not
grant ownership. Transfer cross-task blockers using plan dependencies and owner routing;
never duplicate shared components. Unknown evidence requests inspection, not invented
geometry. Preserve material/collection dependencies and texture provenance. Never invent
sign text, hide defects per camera, or substitute perception backends silently.

Follow PLAN.md's score-first loop and bounded closeout. Local gate: total >=75, each
dimension >=60, >=2 useful views, confidence >=0.7, required evidence known, no critical
defects. Global gate: equal-region mean >=80, every region passing, required connection
and held-out checks. A score alone does not grant acceptance. Keep rubric weights,
calibration, masks, and render settings fixed. Select views by source visibility before
scoring; never omit a difficult view to raise its score. Off-screen content is not missing.

Do not run numerical contact/clearance checks in iterations or closeout, including
floor rays, hardware distances, or seam tolerances. Preserve ownership, registration,
input integrity, and atomic saves. After visual passage inspect <=3 seam locations in
<=2 existing views, with at most one correction and one targeted review per task across
restarts. Seam-only corrections do not restart scoring. Record deferred defects honestly;
missing required structure cannot be waived by an average or a contact-only check.

Use skills/continuous-geometry/SKILL.md for fragmented rails and curved walls. Keep
straight runs exact and preserve observed corners; replace fragments coherently.
STRUCTURE_CHECKS.md provides modeling guidance; docs/audit-workflow.md covers explicit
final audits. RGB previews require no depth arrays. Deliver numbered reference/before/
after/difference cards and inspect them before reporting the result.
