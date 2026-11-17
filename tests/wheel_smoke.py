"""Installed-wheel lifecycle smoke with real Blender and fake, unpaid Codex responses."""

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image
from vibuild.paths import PACKAGE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--blender", required=True)
args = parser.parse_args()
root = Path(tempfile.mkdtemp(prefix="vibuild-wheel-smoke-"))
assert "/site-packages/" in str(PACKAGE), PACKAGE
cli = [sys.executable, "-m", "vibuild"]
env = dict(os.environ)
env.pop("PYTHONPATH", None)
fake = root / "codex"
fake.write_text(
    "#!"
    + sys.executable
    + "\n"
    + """
import json,sys,subprocess
from pathlib import Path
if sys.argv[1:]==['login','status']:sys.exit(0)
text=sys.stdin.read()
if 'PACKET=' in text:
 packet=json.JSONDecoder().raw_decode(text.split('PACKET=',1)[1])[0]
 result={'region':packet['region'],'views':[{'id':v,'scores':{k:90 for k in ['structure','completeness','alignment','appearance','connectivity']},'confidence':.9,'notes':'Deterministic fixture assessment'} for v in packet['views']], 'findings':[], 'uncertainties':[]}
 Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps(result))
else:
 index=json.JSONDecoder().raw_decode(text.split('MATERIAL_INDEX=',1)[1])[0]
 task=Path(text.split('TASK_ROOT=',1)[1].splitlines()[0])
 base=[sys.executable,'-m','vibuild']
 if 'score' in index:
  score=Path(index['score']['path']).parent.parent
  subprocess.run(base+['stage',str(task),'review','--score',str(score),'--view',str(index['evidence']['views'][0]['id'])],check=True)
  subprocess.run(base+['stage',str(task),'accept'],check=True)
 else:
  subprocess.run(base+['tools','jobs','submit','--task',str(task),'--manifest',index['job_template']],check=True)
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':10}}))
"""
)
fake.chmod(0o755)
log = root / "commands.log"


def run(args):
    with log.open("a") as stream:
        stream.write("COMMAND " + str(args) + "\n")
        stream.flush()
        result = subprocess.run(
            cli + list(map(str, args)),
            cwd=root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
    if result.returncode:
        raise RuntimeError(f"{args}: see {log}")


config = root / "local.json"
run(
    [
        "setup",
        "--config",
        config,
        "--blender",
        args.blender,
        "--codex",
        fake,
        "--model",
        "fixture-model",
    ]
)
project = root / "project"
run(["init", project, "--config", config])
rec = root / "rec"
rec.mkdir()
images = root / "images"
images.mkdir()
(rec / "cameras.txt").write_text("1 PINHOLE 64 32 45 45 32 16\n")
rows = []
for i in range(20):
    name = f"{i * 0.5:.1f}.png"
    Image.new("RGB", (64, 32), (120, 140, 150)).save(images / name)
    rows += [f"{i + 1} 1 0 0 0 {-i * 0.01} 0 0 1 {name}", "32 16 1 33 16 2"]
(rec / "images.txt").write_text("\n".join(rows) + "\n")
(rec / "points3D.txt").write_text("1 -.1 0 4 255 0 0 0\n2 .1 0 4 0 255 0 0\n")
run(["register", project, "--from", rec, "--images", images])
run(["partition", project])
run(["partition", project, "--spec", project / "partition.review.json", "--reviewed"])
run(["pause", project])
run(["resume", project, "--no-run"])
run(["run", project])
state = json.loads((project / "status.json").read_text())
assert state["active_region"] is None, state
run(["audit", project])
run(["status", project])
assert (
    json.loads((project / "status.json").read_text())["global_acceptance"] == "passed"
)
assert list(project.rglob("*.blend")) == [project / "scene.blend"]
print(
    json.dumps(
        {
            "installed_package": str(PACKAGE),
            "project": str(project),
            "state": "passed",
            "single_scene": True,
            "model_requests": "deterministic fake only",
        }
    )
)
