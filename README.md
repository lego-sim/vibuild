# vibuild

Video-guided Blender modeling. Fresh Codex decisions edit one self-contained
`scene.blend`; a Python service executes jobs and independent visual scoring.

## Setup

Linux, Python 3.10+ with pip and venv support:

```bash
git clone --recurse-submodules https://github.com/klee101/vibuild.git
cd vibuild
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[tools,test]'
cp configs/example.json configs/local.json
```

Install [Blender](https://www.blender.org/download/) and
[Codex CLI](https://developers.openai.com/codex/cli/) separately. Neither is bundled
or assumed installed. Tested with Blender 5.2.2 and Codex CLI 0.169.0.

Set `runtime.analysis_python`, `runtime.blender`, and `runtime.codex` in
`configs/local.json` to installed executables. Commands on PATH or paths relative
to this checkout are supported; for example, `.venv/bin/python`.
Set `session.model` to an available image-capable Codex model and
`session.reasoning_effort` as needed. Then authenticate with your configured Codex:

```bash
codex login
codex login status
./vibuild doctor
```

Sessions reuse that login and pass model settings explicitly. User Codex config,
memories, plugins, and unrelated skills are excluded. No MCP setup is required.
Proxy settings are inherited; `run --proxy URL` optionally overrides them.

Install [VidMap](third_party/vidmap/README.md#setup) in its own environment, including
its native/model dependencies. Set `runtime.vidmap.python` to that environment's
Python; `runtime.vidmap.source` already points to `third_party/vidmap`. Git pins the
upstream revision and nested dependencies; use `git submodule update --init --recursive`
for an existing clone.

VidMap registration uses upstream models including DA3. Subsequent Blender modeling,
RGB rendering, and visual scoring need no DA3/SAM runtime. `tool perceive` remains
an optional adapter; see [audits](docs/audit-workflow.md).

## Use

Register video with the configured VidMap tool, then export/review the fixed alignment
and partition described in [inputs](docs/inputs.md). Supply the resulting frozen record,
`partition_v1` JSON, and a packed Blender scene with matching registered cameras.
Already registered inputs can skip VidMap.

```bash
./vibuild tool vidmap check
./vibuild tool vidmap run --input VIDEO.mp4 --output work/registration --device cuda
./vibuild check --record demo
./vibuild report --partition partition.json --output work/plan.html
./vibuild task --record demo --partition partition.json --scene initial.blend \
  --output work/modeling_20261116_120000
./vibuild run --task work/modeling_20261116_120000
```

Use `--config FILE` on `doctor`, `report`, `run`, or before the tool name:
`./vibuild tool --config FILE score --help`. Tool options: `./vibuild tool NAME --help`.
Pause by setting `status.json.execution_paused` or creating `.runtime/main/STOP`
under the task. Clear both only when intentionally resuming.

## Structure

- `src/vibuild/`: configuration, records, sessions, scoring, and shared state.
- `tools/`: scene/evidence tools and program-owned execution.
- `instructions/`, `skills/`: one decision prompt and continuous geometry guidance.
- `third_party/vidmap/`: pinned upstream submodule; `examples/`: capture-specific recipe.
- `records/`: frozen inputs; `work/`: scenes and run artifacts. Both are Git-ignored.

[Workflow](PLAN.md) · [Agent contract](AGENT.md) · [Scoring](docs/regional-scoring.md)

```bash
python -m pytest -q
# Optional real Blender coverage:
VIBUILD_TEST_BLENDER=blender python -m pytest -q
```
