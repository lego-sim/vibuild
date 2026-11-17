# vibuild

Rebuild indoor spaces from video in Blender. VidMap registers the cameras; Codex
iterates through reviewed regions and connections in one self-contained `scene.blend`.

## Install

Linux, Python 3.10+:

```bash
git clone --recurse-submodules https://github.com/klee101/vibuild.git
cd vibuild
python -m venv .venv
source .venv/bin/activate
pip install -e .
vibuild setup --model MODEL
```

Install and log in to [Codex CLI](https://developers.openai.com/codex/cli/).
Setup detects existing tools; it does not download them.

```bash
codex login
vibuild install blender            # or: setup --blender /path/to/blender
vibuild doctor --stage model
```

For video registration, install [VidMap's prerequisites](https://github.com/cvg/vidmap#setup)
in a separate environment, then configure it:

```bash
vibuild install vidmap --python /path/to/vidmap-env/bin/python
vibuild install ffmpeg             # only if ffmpeg is missing
vibuild doctor --stage register --deep
```

VidMap uses DA3 internally. Modeling from registered inputs requires only Blender
and Codex. See [setup](docs/setup.md) for configuration and existing reconstructions.

## Rebuild a space

```bash
vibuild init projects/demo --videos /path/to/videos
vibuild register projects/demo --device cuda
vibuild partition projects/demo    # one region: initialize and continue immediately
vibuild run projects/demo
```

A single region uses the whole-scene fast path without manual partition review.
For multiple regions, review `partition.html` and approve the specification with
`partition PROJECT --spec FILE --reviewed`. Scene scale is not a survey.

```bash
vibuild status projects/demo
vibuild pause projects/demo
vibuild resume projects/demo
vibuild audit projects/demo         # final fit, connection, and held-out assessment
```

Outputs live in the project: `scene.blend`, numbered comparison cards, scores, and
logs. Pause/resume keeps the latest saved scene. Final acceptance requires all stages
and the current-scene audit; deferred work is reported as incomplete.

[Workflow](docs/workflow.md) · [Commands and formats](docs/reference.md)

Development: `pip install -e '.[test]' && python -m pytest`.
Set `VIBUILD_TEST_BLENDER` to an installed executable to include Blender tests.
