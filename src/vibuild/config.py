"""Explicit tool configuration shared by the CLI and execution service."""
import os
from pathlib import Path
import shutil

from . import resources


def default_config():
    selected = os.environ.get('VIBUILD_CONFIG')
    candidates = ([Path(selected)] if selected else []) + [
        resources.ROOT/'configs/local.json', resources.ROOT/'configs/example.json']
    return next((path for path in candidates if path.is_file()), candidates[-1])


def settings(path):
    value = resources.load(path)
    # Tool paths are relative to the checkout, never the caller's working directory.
    runtime = value.setdefault('runtime', {})
    for name in ('analysis_python', 'blender', 'codex'):
        command = runtime.get(name)
        if isinstance(command, str) and command and ('/' in command or command.startswith('~')):
            target = Path(command).expanduser()
            runtime[name] = os.path.abspath(resources.ROOT/target)
    vidmap = runtime.get('vidmap', {})
    for key in ('source', 'python'):
        command = vidmap.get(key)
        if isinstance(command, str) and command and (key == 'source' or '/' in command or command.startswith('~')):
            vidmap[key] = os.path.abspath(resources.ROOT/Path(command).expanduser())
    return value


def tool(config, name):
    command = settings(config)['runtime']
    for key in name.split('.'):
        command = command.get(key) if isinstance(command, dict) else None
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f'Configure runtime.{name} in {config}')
    if not shutil.which(command):
        raise ValueError(f'Tool not executable: runtime.{name}={command}')
    return command


def vidmap_runtime(config):
    runtime = settings(config)['runtime'].get('vidmap', {})
    source = runtime.get('source')
    if not source or not (Path(source)/'vidmap/run.py').is_file():
        raise ValueError('Configure runtime.vidmap.source; initialize the VidMap submodule')
    return {'source': source, 'python': tool(config, 'vidmap.python')}


def doctor(config):
    result = {}
    for name in ('analysis_python', 'blender', 'codex'):
        try:
            result[name] = {'executable': tool(config, name)}
        except ValueError as exc:
            result[name] = {'error': str(exc)}
    try:
        result['vidmap'] = vidmap_runtime(config)
    except ValueError as exc:
        result['vidmap'] = {'error': str(exc)}
    return {'ok': all('error' not in item for item in result.values()), 'tools': result}
