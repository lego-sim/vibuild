"""Portable setup and fresh task bootstrapping, without installed modeling tools."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from vibuild import resources
from vibuild.cli import main
from vibuild.config import doctor, settings, tool


def test_unconfigured_tools_fail_without_launching_processes(tmp_path, monkeypatch, capsys):
    config = tmp_path/'config.json'
    resources.save(config, {'runtime': {'analysis_python': '', 'blender': '', 'codex': ''}})
    monkeypatch.setattr('vibuild.cli.subprocess.run', lambda *a, **k: pytest.fail('Unconfigured tool launched'))
    assert main(['doctor', '--config', str(config)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result['ok'] and set(result['tools']) == {'analysis_python', 'blender', 'codex', 'vidmap'}
    assert all('Configure runtime.' in item['error'] for item in result['tools'].values())
    assert main(['run', '--config', str(config), '--task', str(tmp_path/'absent')]) == 1


def test_relative_tool_paths_resolve_from_checkout_without_da3(tmp_path, monkeypatch):
    executable = tmp_path/'bin/python'
    executable.parent.mkdir(); executable.write_text('#!/bin/sh\nexit 0\n'); executable.chmod(0o755)
    config = tmp_path/'config.json'
    resources.save(config, {'runtime': dict.fromkeys(('analysis_python', 'blender', 'codex'), 'bin/python')})
    monkeypatch.setattr(resources, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    assert tool(config, 'blender') == str(executable)
    assert 'executable' in doctor(config)['tools']['blender']
    assert 'error' in doctor(config)['tools']['vidmap']
    assert 'da3' not in settings(config)['runtime']


def test_tool_paths_preserve_virtual_environment_symlinks(tmp_path, monkeypatch):
    executable = tmp_path/'.venv/bin/python'
    executable.parent.mkdir(parents=True); executable.symlink_to(sys.executable)
    config = tmp_path/'config.json'
    resources.save(config, {'runtime': {'analysis_python': '.venv/bin/python'}})
    monkeypatch.setattr(resources, 'ROOT', tmp_path)
    assert tool(config, 'analysis_python') == str(executable)


def test_task_uses_partition_order_and_seeds_first_assessment(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools'))
    monkeypatch.setattr(resources, 'ROOT', tmp_path)
    partition = tmp_path/'partition.json'
    resources.save(partition, {'schema_version': 'partition_v1',
        'regions': [{'id': 'R42', 'name': 'Room', 'bounds_min': [0,0,0], 'bounds_max': [1,1,1]}],
        'interfaces': [], 'frames': []})
    scene = tmp_path/'initial.blend'; scene.write_bytes(b'BLENDERfixture')
    task = tmp_path/'modeling_20261116_120000'
    args = ['task', '--record', 'demo', '--partition', str(partition), '--scene', str(scene), '--output', str(task)]
    assert main(args) == 0
    assert resources.load(task/'plan.json')['execution_order'] == ['R42']
    assert resources.load(task/'status.json')['active_region'] == 'R42'
    assert resources.load(task/'R42/job_template.json')['steps'][0]['args'][0] == 'prepare'
    assert (task/'scene.blend').read_bytes() == scene.read_bytes()
    assert list(task.rglob('*.blend')) == [task/'scene.blend']
    assert main(args) == 1
