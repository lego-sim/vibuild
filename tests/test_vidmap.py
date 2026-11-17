"""Pinned source routing and frozen-input protection without native/model dependencies."""
import json
from pathlib import Path
import sys
import venv

import pytest

sys.path[:0] = [str(Path(__file__).parents[1]/'src'), str(Path(__file__).parents[1]/'tools')]
import registration as adapter
from vibuild.config import settings


def configured(tmp_path):
    environment = tmp_path/'python'
    venv.EnvBuilder(with_pip=False).create(environment)
    source = tmp_path/'third_party/vidmap'
    package = source/'vidmap'; package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'run.py').write_text('''import argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--input_data');p.add_argument('--output');p.add_argument('--device')
p.add_argument('--terminate',action='store_true')
a,extra=p.parse_known_args()
assert a.device=='cpu' and a.terminate and extra==['frontend.roma.compile=false']
assert Path(a.input_data).read_bytes()==b'video'
(Path(a.output)/'rec').mkdir()
print('registered')
''')
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'runtime':{'vidmap':{'source':str(source),'python':str(environment/'bin/python')}}}))
    return config, source


def test_vidmap_uses_configured_checkout_and_separate_python(tmp_path):
    config, source = configured(tmp_path)
    result = adapter.execute(config, 'check')
    assert Path(result['import']).is_relative_to(source)
    video = tmp_path/'input.mp4'; video.write_bytes(b'video')
    output = tmp_path/'registration'
    result = adapter.execute(config, 'run', video, output, 'cpu', ['frontend.roma.compile=false'])
    assert Path(result['reconstruction']).is_dir()
    assert Path(result['log']).read_text().strip() == 'registered'
    assert video.read_bytes() == b'video'


def test_vidmap_rejects_frozen_output_and_hidden_input_overrides(tmp_path, monkeypatch):
    config, _ = configured(tmp_path)
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    video = tmp_path/'input.mp4'; video.write_bytes(b'video')
    with pytest.raises(ValueError, match='outside frozen'):
        adapter.execute(config, 'run', video, tmp_path/'records/demo', 'cpu')
    with pytest.raises(ValueError, match='named tool options'):
        adapter.execute(config, 'run', video, tmp_path/'output', 'cpu', ['--input_data=other'])


def test_vidmap_source_is_relative_to_checkout(tmp_path, monkeypatch):
    from vibuild import resources
    config, source = configured(tmp_path)
    value=json.loads(config.read_text());value['runtime']['vidmap']['source']='third_party/vidmap'
    config.write_text(json.dumps(value))
    monkeypatch.setattr(resources, 'ROOT', tmp_path)
    assert settings(config)['runtime']['vidmap']['source'] == str(source)
