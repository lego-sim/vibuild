"""Run the configured, pinned VidMap checkout in its own Python environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vibuild.config import default_config, vidmap_runtime


def execute(config, action, source=None, output=None, device=None, overrides=()):
    if any(arg.split('=', 1)[0] in {'--input_data', '--output', '--device'} for arg in overrides):
        raise ValueError('Set input, output, and device through the named tool options')
    runtime = vidmap_runtime(config)
    env = dict(os.environ)
    env['PYTHONPATH'] = runtime['source'] + os.pathsep + env.get('PYTHONPATH', '')
    if action == 'check':
        result = subprocess.run([runtime['python'], '-c', 'import vidmap; print(vidmap.__file__)'],
            cwd=runtime['source'], env=env, capture_output=True, text=True, check=True)
        if not Path(result.stdout.strip()).is_relative_to(runtime['source']):
            raise ValueError('VidMap Python imports another checkout; install the configured source in that environment')
        return {'source': runtime['source'], 'import': result.stdout.strip()}
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.exists():raise ValueError('VidMap input must be an existing video or frame directory')
    if output == source or any(output.is_relative_to(root) for root in (ROOT/'records', ROOT/'third_party')):
        raise ValueError('VidMap output must be outside frozen records, third-party source, and its input')
    output.mkdir(parents=True, exist_ok=True)
    log = output/'vidmap.log'
    command = [runtime['python'], '-u', '-m', 'vidmap.run', '--input_data', str(source),
               '--output', str(output), '--device', device, '--terminate', *overrides]
    with log.open('a') as stream:
        result = subprocess.run(command, cwd=runtime['source'], env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'VidMap exited {result.returncode}; see {log}')
    if not (output/'rec').is_dir():raise RuntimeError(f'VidMap produced no reconstruction; see {log}')
    return {'reconstruction': str(output/'rec'), 'log': str(log)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_config())
    parser.add_argument('action', choices=['check', 'run'])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args, overrides = parser.parse_known_args(argv)
    if args.action == 'run' and not all((args.input, args.output, args.device)):
        parser.error('run requires --input, --output, and --device')
    try:
        print(json.dumps(execute(args.config, args.action, args.input, args.output, args.device, overrides)))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'vidmap: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
