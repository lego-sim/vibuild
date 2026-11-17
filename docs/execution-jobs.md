# Program-owned modeling execution

A decision session prepares a script and job, submits once, and returns. The host
service executes ordinary processes, waits, records completion and starts the next
fresh decision. Only modeling decisions and independent visual assessment call models.
There are no LLM runners or new wrapper scripts per iteration.

```bash
./vibuild run --task work/modeling_YYYYMMDD_HHMMSS
```

Run the service in tmux after explicitly resuming a paused task. It honors both
`status.json.execution_paused` and `.runtime/main/STOP`; it never clears them itself.
An active main must be stopped before replacing its launcher. The service uses the
same main lock, single scene and independent-assessment locks as the existing tools.

The service supplies a small MATERIAL_INDEX (current hash, scope, exact evidence paths,
three priority findings, affected objects and the configured analysis Python). It writes
one job.json template in the new decision directory when a prior build/prepare recipe
exists. Edit that template and write edit.py; fixed camera/render arguments are reused.
No extra context files or schema registry are needed. Historical scores are explicitly
marked when their scene hash differs. Additional findings remain in the indexed score;
read all unresolved findings before acceptance. Read selected JSON fields, not whole
packets/inventories/HTML. Use CLI --help only for unfamiliar options, not source searches.

The decision gets current comparison images and user guidance directly. Use
`python3 tools/materials.py FILE.json#/field/0 EDIT.py#function FILE#L10-L30` for
batched selected reads (quote selectors in shells). No selector returns a structure
index; explicit selectors return complete values. There is no application output budget
or silent truncation. Print only decision-relevant results and artifact paths.

The service stores scene/guidance binding privately in the decision's .runtime directory.
The authored job contains only region, round and steps. Submission confirms receipt of
supplied guidance, records CODEX_THREAD_ID, checks syntax and binds inputs without an
extra acknowledgment or hash command. A changed scene or guidance still stops submission.
The worker's durable receipts retain integrity metadata; model summaries omit it.
Failures identify the underlying exception and log path. Status returns pending/recent
jobs; monitor summary returns current status, timing totals and the on-disk report path.

Fresh sessions omit automatic skill instructions, bundled skills, default plugins and
agent tools. The useful continuous-geometry skill remains an explicit on-demand project
reference; the scorer needs no skills or tool calls. Only the managed task and existing
cache receive write access for modeling; scorers remain read-only.

The scoring prompt contains source/render images, finite scope, view IDs, required
connection checks and any specifically requested diagnostic facts. It contains no object
catalogue, matrices, raw projections or verification hashes. Findings identify a visual
element and image box. Python ranks spatial object candidates afterward and exposes only
relevant bounds/materials to the repair decision. Candidate overlap is approximate, not
identity, visibility or ownership proof. Missing elements remain findings even with no
matched objects. Frozen images, rubric, gates and full internal evidence remain intact.

The model submits a JSON manifest through `./vibuild tool jobs submit --task TASK_ROOT
--manifest JOB_JSON`. No command execution starts inside the submitting model turn.
Paths below are placeholders to replace with exact project-relative or absolute paths.
All render settings must be the already frozen values, not these illustrative numbers.

```json
{
  "region": "R07",
  "round": 4,
  "steps": [
    {"tool": "scene", "args": ["iterate", "--baseline", "TASK_ROOT/scene.blend",
      "--script", "TASK_ROOT/R07/repair004.py", "--views", "TASK_ROOT/R07/views.json",
      "--region", "R07", "--what", "Repair observed doorway geometry",
      "--why", "Latest fixed-view assessment identifies a misplaced leaf",
      "--output", "TASK_ROOT/R07/iterations/004", "--width", "960",
      "--samples", "24", "--threads", "4"]},
    {"tool": "score", "args": ["prepare", "--record", "demo",
      "--partition", "TASK_ROOT/.runtime/partition.json", "--scene", "TASK_ROOT/scene.blend",
      "--region", "R07", "--frame", "73", "--frame", "76", "--frame", "80",
      "--output", "TASK_ROOT/R07/scores/004", "--width", "960",
      "--samples", "24", "--threads", "4"]},
    {"tool": "score", "args": ["run", "--output", "TASK_ROOT/R07/scores/004",
      "--region", "R07"]}
  ]
}
```

A score-only job can run an existing frozen packet. Numerical contact/clearance checks
are excluded from the execution queue and edit scripts; closeout uses existing images.
For generated evidence, insert this step between prepare and run:

```json
{"tool":"score","args":["supplement","--output","TASK_ROOT/R07/scores/004","--region","R07"],
 "diagnostics":{"images":[{"id":"detail","role":"Doorway depth","path":"TASK_ROOT/R07/iterations/004/detail.jpg"}],
                "facts":[{"role":"Doorway geometry","path":"TASK_ROOT/R07/iterations/004/geometry.json"}]}}
```

Declare only actual needed evidence; omit unused images/facts keys. Image items require
id/role/path; fact items require role/path. The edit or another earlier stage must create
each file inside its --output directory. Files must not exist at submission or worker
startup. After the producing stage the worker captures the current scene and file hashes;
before supplement it verifies both and the prepared packet, then writes the scene-bound
manifest locally. Missing, changed or old evidence stops the batch. A fact's own scene
hash, if present, must match too; generated facts should omit it and let the worker bind
the saved scene. Do not predict a post-save hash or start another model
decision to fill one. Existing already-bound evidence can still use --diagnostics FILE;
it is never rebound. Do not combine --diagnostics with inline diagnostics.

All script/view/manifest inputs are hash-bound at submission. No shell command strings.
Outputs remain under the assigned region. Existing shared tools enforce frozen source,
scene registration and one-root save rules; submitting a script does not authorize
unrelated geometry edits.

Queue and receipts live in `.runtime/jobs/JOB_ID/`: request.json, state.json, result.json
and one log per step. Logs stay on disk; models receive short terminal summaries. A
failed visual score is a completed execution with a failed score, never scene acceptance.
A failed command stops the batch; later steps do not run. Latest saved geometry survives
render/scoring failure. A previously running job after a crash becomes interrupted and
is not rerun. A stale scene/input hash requires a fresh decision. Do not queue multiple
scene-changing jobs against the same baseline expecting them to rebase automatically.

The 900-second silent-output timeout and pause checks happen inside Python, consuming
zero model tokens. Silence timeout is an execution failure, not proof of bad geometry.
No job submitted means idle: the service exits rather than asking the model repeatedly.
The terminal notification queue is also available at `.runtime/jobs/notifications.jsonl`.
For read-only monitoring use `./vibuild tool jobs status --task TASK_ROOT`.

The continuous-geometry skill applies inside submitted edit scripts; it does not own
execution. Final visual inspection follows PLAN.md.
