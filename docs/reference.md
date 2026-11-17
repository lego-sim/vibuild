# Reference

Run `vibuild --help` or `vibuild COMMAND --help`. `--config FILE` overrides local or
project configuration. `doctor` and `status` accept `--json` for scripting. `VIBUILD_CONFIG` selects a default for advanced tool calls.

| Command | Purpose |
| --- | --- |
| `setup --model MODEL` | Detect tools and write local configuration |
| `doctor --stage model\|register [--deep]` | Check prerequisites and print repair commands |
| `install blender\|ffmpeg\|vidmap` | Explicit installation; no implicit downloads |
| `init PROJECT [--videos PATH]` | Create a project from one video or a directory |
| `register PROJECT --device cuda [--fps 2]` | Sample and register video |
| `register PROJECT --from REC --images DIR [--sources FILE]` | Import COLMAP output |
| `partition PROJECT [--spec FILE --reviewed]` | Preview or approve spatial scope |
| `run PROJECT`, `status PROJECT`, `pause PROJECT`, `resume PROJECT` | Execution lifecycle |
| `audit PROJECT` | Final visual acceptance on the current scene |

`resume --no-run` only clears the pause. Keep an unattended `run` in tmux or another
process supervisor. Ctrl-C pauses the service; geometry remains saved.

## Partition specification

`partition.review.json` contains `transform` (proper rotation `basis`, `origin`, positive
`scale`) and `regions`. Each region has an Rxx `id`, `name`, `bounds_min`, `bounds_max`,
and optional `core_boxes`. Regions must not overlap. Coordinates after transformation
are `scale * basis @ (raw_point - origin)`. Identity is the initial template; choose an
alignment from source evidence. Scale is not independently verified metres.

`start_region` chooses the first stage. Observed camera crossings through shared finite
faces propose interfaces. Review them against the source. An explicit `interfaces` list
can supply reviewed connections. Empty/unknown edges do not become invented neighbors.
Stages are Rxx regions and Jxx connections only. The default order builds every region,
then schedules one ordinary J stage for each observed interface. Loops and branches
use the same connection workflow. No extra stage type is needed. Each J stage owns
its endpoint regions and intervening connection space. Unknown edges remain unknown.
An explicit `execution_plan` can provide ordered `chains[].steps` to group connection
work; include each region once and reference preceding stages in dependencies.
The optional capture recipe in `examples/indoor_partition.py` illustrates finite
multi-box scopes.

## Advanced tools

`vibuild tools NAME --help` lists options. These tools are installed package modules,
not scripts that depend on the checkout directory.

| Name | Operations |
| --- | --- |
| `scene` | Inspect/render and atomic `iterate --script` |
| `score` | Prepare/supplement/run/report or cancel a stopped assessment |
| `jobs` | Submit a bounded argv manifest, execute it, read compact status |
| `connections` | Prepare calibrated transition and approach evidence |
| `vidmap` | Run/check the configured upstream environment |
| `materials` | Read selected JSON fields, Python functions, or line ranges |
| `usage` | Summarize local Codex session tokens by actor/time |
| `monitor` | Read local execution state and timing |
| `audit` | Explicit native-grid depth, semantics, or appearance metrics |
| `perception` | Optional separately configured DA3/SAM adapters |

Tools return results and artifact paths on success, specific causes on failure. Large
logs and inventories remain on disk. Use `materials` selectors for additional detail.
A completed process does not imply a passing scene.

A decision submits `{ "region": "R01", "round": 1, "steps": [...] }` using its supplied
job template. Each step has `tool` and `args` (an argv array, never a shell string).
The program checks syntax, scope, input binding, and fixed render settings before
execution. Standard order: scene iterate, score prepare, optional supplement, score run.
Inline supplement diagnostics identify only newly generated files within earlier step
outputs; Python binds them to the saved scene. No model-visible hashes are needed.

Stage transitions are program-owned:

```bash
vibuild stage PROJECT review --score PROJECT/R01/scores/001 --view FRAME_ID
vibuild stage PROJECT accept
vibuild stage PROJECT defer --reason "User deferred this area"
vibuild stage PROJECT handoff --target J01 --reason "Missing shared passage"
```

For one seam correction, use `review --begin --score DIR`, submit one scene-only job,
inspect its comparisons, then `review --finish --corrected --view FRAME_ID`. Review
counts persist across restarts. Deferral advances ready work without claiming acceptance.

## Development

`src/vibuild/` is divided into registration, Blender, assessment, runtime, and agents.
Prompts and the rubric ship with the wheel. The only external source checkout is the
pinned VidMap submodule. Update its pin deliberately and verify upstream dependencies.

Run `python -m pytest`; real Blender tests require `VIBUILD_TEST_BLENDER=EXECUTABLE`.
Tests use temporary projects and fake Codex processes; they make no paid model requests.
Verify releases with a wheel installation outside the source checkout. Do not publish
local configs, credentials, videos, model weights, or project execution artifacts.
