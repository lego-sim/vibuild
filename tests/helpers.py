"""Small fake-process fixtures shared by process and tool contract tests."""
from pathlib import Path
from vibuild import resources


def config(tmp_path, body='', **budget):
    fake = tmp_path / 'codex'
    fake.write_text('#!/usr/bin/env python3\nimport sys,json,time,subprocess,os\nfrom pathlib import Path\n'
                    'if sys.argv[1:]==["login","status"]:sys.exit(0)\n'
                    'prompt=sys.stdin.read()\n' + body)
    fake.chmod(0o755)
    path = tmp_path / 'config.json'
    resources.save(path, {'runtime': {'codex': str(fake)}, 'session': {'model': 'test-model'}, 'budget': budget})
    return path
