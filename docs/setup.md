# Setup

Install the Python package with `pip install .` or `pip install -e '.[test]'`
for development. Wheels include the CLI, prompts, schemas, and HTML template;
projects can live anywhere. NumPy and Pillow are installed with the package.

`vibuild setup --model MODEL` writes `vibuild.local.json` in the current directory.
It detects installed Blender, Codex, and ffmpeg executables. It never downloads tools
or changes authentication. Rerunning setup preserves existing values unless overridden.
`--config FILE` selects another file. Relative paths resolve from that file's directory.
`vibuild init` copies the resolved configuration into the project's `vibuild.json`.

| Component | Setup | Needed for |
| --- | --- | --- |
| Blender | `vibuild install blender` or `setup --blender EXECUTABLE` | Modeling and rendering |
| Codex CLI | Official installer, `codex login`, `setup --codex EXECUTABLE` | Decisions and assessment |
| ffmpeg | `vibuild install ffmpeg` or `setup --ffmpeg EXECUTABLE` | Sampling source videos |
| VidMap | Upstream native prerequisites, then `vibuild install vidmap --python PYTHON` | New camera registration |

Blender's installer downloads a checksum-verified Linux x86_64 release into
`.vibuild/tools` beside the configuration. Other platforms require a configured binary.
ffmpeg installation uses `imageio-ffmpeg` in the active environment. VidMap uses a
pinned checkout under `third_party/vidmap`; models, PyTorch, and native COLMAP dependencies
stay in its separate environment. Install logs identify native build failures. No
installer runs during modeling. A GPU is needed for the usual VidMap CUDA configuration;
Blender renders on CPU. Cached registered inputs need no VidMap/DA3/SAM inference.

`vibuild doctor --stage model` checks Blender, Codex CLI capabilities/login, and the
selected model. `--stage register` checks ffmpeg and the configured VidMap checkout.
`--deep` also starts Blender or imports VidMap's native dependencies. These checks make
no model requests. A model still needs to be available on the authenticated account.

Codex sessions are fresh and use the existing login. Provider settings are inherited;
unrelated project instructions, memory, plugins, and automatic skill loading are excluded.
Optional `session.profile` and `session.codex_home` select a Codex profile or account.
Use an image-capable model in `session.model`; `session.reasoning_effort` defaults to high.
Credentials stay outside projects. Proxies are inherited from the environment;
`vibuild run PROJECT --proxy URL` provides an explicit override.

## Register an image sequence

```bash
vibuild init projects/demo
vibuild register projects/demo --images /path/to/frames --fps 5 --device cuda
```

This runs VidMap and imports its reconstruction, including point tracks. Image names
are naturally ordered; `--fps` supplies timing when original timestamps are absent.
Video input also runs VidMap. Optional `runtime.vidmap.args` forwards upstream options.
Only explicit `--from REC` skips it and imports existing
calibration. The registration result reports `backend: vidmap` or `backend: import`.

## Import an existing reconstruction

```bash
vibuild init projects/demo
vibuild register projects/demo --from /path/to/rec --images /path/to/images \
  --sources /path/to/sources.json
vibuild partition projects/demo
```

COLMAP binary and text models are supported. Source images must match the registered
camera resolution. Distorted images are resampled to the calibrated pinhole grid;
original pixels are retained separately. Supported models: SIMPLE_PINHOLE, PINHOLE,
SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, and OPENCV_FISHEYE. Other models require
an upstream undistorted export.

`sources.json` is a list of `{ "name": "cam0/frame.jpg", "video": "clip", "time": 1.5 }`
entries, optionally with `evaluation_split: "fit"` or `"audit"`. It is unnecessary when
filenames already express source seconds. Without explicit splits, every tenth
registered image is reserved for the final audit. Each region needs at least two useful
fit and two held-out views. A single whole-scene task selects calibrated RGB directly
without regional sparse-track coverage; the multi-region coverage rule is unchanged.
Insufficient evidence still blocks acceptance.
