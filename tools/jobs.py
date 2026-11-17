"""Durable execution queue: no LLM runner, no model calls while waiting."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vibuild.config import default_config, settings
from scene_storage import file_lock, task_root
from artifact_cache import digest
from task_monitor import read, write, guidance, record_guidance, state_lock

ALLOWED = {'scene': {'iterate', 'render', 'inspect'},
           'score': {'prepare', 'supplement', 'run', 'report'},
           'perceive': None, 'evaluate': {'compare', 'review'}}
INPUT_FLAGS = {'--script', '--views', '--protocol', '--partition', '--connection', '--diagnostics', '--manifest'}
TERMINAL = {'completed', 'failed', 'cancelled', 'idle_timeout', 'interrupted', 'stale'}


def paused(root):
    return read(root/'status.json', {}).get('execution_paused', False) or (root/'.runtime/main/STOP').exists()


def queue(root):
    return root/'.runtime/jobs'


def absolute(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT/p).resolve()


def option(args, flag):
    return args[args.index(flag)+1] if flag in args else None


def generated_evidence(request):
    """Small inline supplement contract: artifacts produced by earlier job stages."""
    outputs, evidence = [], []
    for index, step in enumerate(request['steps']):
        args = step['args']
        if 'diagnostics' in step:
            data = step['diagnostics']
            if (step['tool'], args[0]) != ('score', 'supplement') or '--diagnostics' in args:
                raise ValueError('Inline diagnostics belong only to score supplement, without --diagnostics')
            if not isinstance(data, dict) or not data or set(data)-{'images', 'facts'}:
                raise ValueError('Diagnostics contain only images and facts')
            for kind, items in data.items():
                if not isinstance(items, list) or not 1 <= len(items) <= 12:
                    raise ValueError('Provide 1-12 diagnostic items per kind')
                fields = {'id', 'role', 'path'} if kind == 'images' else {'role', 'path'}
                for item in items:
                    if not isinstance(item, dict) or set(item) != fields or not all(isinstance(v, str) and v for v in item.values()):
                        raise ValueError(f'Diagnostic {kind} require exactly {sorted(fields)}')
                    path = absolute(item['path'])
                    producer = next((n for n, out in reversed(outputs) if path.is_relative_to(out) and path != out), None)
                    if producer is None:
                        raise ValueError('Generated diagnostics must be inside an earlier stage output directory')
                    evidence.append((index, producer, kind, {**item, 'path': str(path)}))
        if option(args, '--output') and (step['tool'], args[0]) in {('scene', 'iterate'), ('scene', 'render'), ('scene', 'inspect'), ('score', 'prepare')}:
            outputs.append((index, absolute(option(args, '--output'))))
    return evidence


def validate(root, request):
    if task_root(root) != root:
        raise ValueError('Expected a managed task root')
    if not request.get('steps') or len(request['steps']) > 8:
        raise ValueError('A job needs 1-8 ordered tool steps')
    region = request.get('region')
    if region not in {s['id'] for s in read(root/'plan.json', {})['steps']}:
        raise ValueError('Unknown task region')
    inputs = {}
    for step in request['steps']:
        tool, args = step.get('tool'), step.get('args')
        if tool not in ALLOWED or not isinstance(args, list) or not args or not all(isinstance(a, str) for a in args):
            raise ValueError('Steps require a known tool and an argv array; shell strings are forbidden')
        if ALLOWED[tool] is not None and args[0] not in ALLOWED[tool]:
            raise ValueError('Unsupported job action; numerical contact checks, external assessor dispatch and raw patch/merge are outside this workflow')
        if tool == 'score' and option(args, '--region') != region:
            raise ValueError('Score region differs from job region')
        for flag in ('--output',):
            value = option(args, flag)
            if value and (not absolute(value).is_relative_to(root/region) or absolute(value) == root/region):
                raise ValueError('Job outputs must be inside its assigned regional directory')
        if tool == 'scene' and args[0] == 'iterate':
            if absolute(option(args, '--baseline') or '') != root/'scene.blend' or '--script' not in args:
                raise ValueError('Build must use the current root and a managed edit script')
            if option(args, '--region') != region:
                raise ValueError('Build region differs from job region')
        if (tool, args[0]) in [('scene', 'iterate'), ('score', 'prepare')]:
            if any(flag not in args for flag in ('--width', '--samples', '--threads')):
                raise ValueError('Explicit frozen render width/samples/threads required')
        if tool == 'score' and args[0] == 'prepare' and absolute(option(args, '--scene') or '') != root/'scene.blend':
            raise ValueError('Score the current root scene')
        for i, flag in enumerate(args[:-1]):
            if flag in INPUT_FLAGS:
                path = absolute(args[i+1])
                if not path.is_file():
                    raise ValueError(f'Missing input: {path}')
                if flag == '--script':
                    try:compile(path.read_text(), str(path), 'exec')
                    except SyntaxError as exc:
                        raise ValueError(f'{path}:{exc.lineno}: {exc.msg}') from None
                inputs[str(path)] = digest(path)
    for _, _, _, item in generated_evidence(request):
        if Path(item['path']).exists():
            raise ValueError('Generated diagnostic path already exists; use a new output path, not old evidence')
    return inputs


def submit(root, manifest):
    root = root.resolve()
    request = read(manifest, {})
    active = read(root/'status.json', {}).get('active_region')
    if active and request.get('region') != active:
        raise ValueError(f'Active task is {active}; submit through its current owner decision')
    if paused(root):
        raise ValueError('Task is paused; no work queued')
    binding = read(manifest.resolve().parent/'.runtime/binding.json', {})
    if binding.get('task') != str(root):
        raise ValueError('Submit the job from the supplied decision directory for this task')
    notice = guidance(root)
    if notice and notice['pending'] and binding.get('guidance_sha256') != notice['sha256']:
        raise ValueError('User guidance changed or was not supplied; obtain the current decision context')
    inputs = validate(root, request)
    expected = binding.get('scene_sha256')
    if not expected or expected != digest(root/'scene.blend'):
        raise ValueError('Job was planned against another scene; rebase before submission')
    if notice and notice['pending']:
        record_guidance(root, binding['guidance_sha256'], os.environ.get('CODEX_THREAD_ID', ''))
    ident = f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    directory = queue(root)/ident
    directory.mkdir(parents=True)
    write(directory/'request.json', {**request, 'expected_scene_sha256': expected, 'input_hashes': inputs})
    write(directory/'state.json', {'id': ident, 'state': 'queued', 'submitted': time.time(),
                                    'region': request['region'], 'round': request.get('round')})
    return {'state': 'queued', 'receipt': str(directory/'result.json')}


def stop_process(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def execute(root, command, log, env, idle_seconds=900):
    """Local waiting consumes zero model tokens. Silence timeout never retries a task."""
    proc = None
    try:
        with log.open('wb') as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            changed, size = time.monotonic(), 0
            while proc.poll() is None:
                current = log.stat().st_size
                if current != size:
                    changed, size = time.monotonic(), current
                if paused(root):
                    stop_process(proc)
                    return 'cancelled', proc.returncode
                if time.monotonic()-changed >= idle_seconds:
                    stop_process(proc)
                    return 'idle_timeout', proc.returncode
                time.sleep(.2)
        return ('completed' if proc.returncode == 0 else 'failed'), proc.returncode
    finally:
        if proc is not None and proc.poll() is None:
            stop_process(proc)


def publish(root, result):
    with state_lock(root):
        state = read(root/'status.json', {})
        state['execution_job'] = {k: result.get(k) for k in ('id', 'state', 'region', 'round')}
        if not state.get('execution_paused'):
            state.update(active_region=result.get('region'), round=result.get('round'),
                         state='executing_job' if result['state']=='running' else 'awaiting_decision',
                         current_task=f"Job {result['id']}: {result['state']}", updated=time.time())
        write(root/'status.json', state)


def failure_detail(log, output=None, since=0):
    """Return the underlying exception, not a wrapper traceback or a Blender log dump."""
    logs = [log]
    if output:
        logs += sorted((output/'.runtime').glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    fallback = ''
    for path in logs:
        if path.stat().st_mtime < since:continue
        lines = path.read_text(errors='replace').splitlines()
        errors = [i for i, line in enumerate(lines) if re.match(r'^\w*(?:Error|Exception|Interrupt):', line)]
        if errors:
            i = errors[-1]
            frames = [j for j in range(i) if lines[j].lstrip().startswith('File "')]
            detail = '\n'.join([lines[frames[-1]], lines[i]]) if frames else lines[i]
            if path != log:return {'error_tail': detail, 'error_log': str(path)}
            fallback = detail
        elif path == log:fallback = '\n'.join(lines[-4:])
    return {'error_tail': fallback, 'error_log': str(log)}


def brief(result):
    return {k: result[k] for k in ('id', 'state', 'region', 'round', 'score', 'comparisons',
                                  'error', 'error_tail', 'error_log') if k in result}


def run_job(root, directory, config, proxy=None):
    request = read(directory/'request.json', {})
    state = read(directory/'state.json', {})
    result = {**state, 'state': 'running', 'started': time.time(), 'steps': []}
    write(directory/'state.json', result)
    publish(root, result)
    previous = signal.getsignal(signal.SIGTERM)
    def interrupt(*_):
        (root/'.runtime/main/STOP').parent.mkdir(parents=True, exist_ok=True)
        (root/'.runtime/main/STOP').touch()
        raise KeyboardInterrupt('Execution service stopped')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        if paused(root):
            result['state'] = 'cancelled'
            return result
        notice = guidance(root)
        if notice and notice['pending']:
            raise ValueError('New user guidance requires acknowledgment before executing queued work')
        if digest(root/'scene.blend') != request['expected_scene_sha256']:
            result.update(state='stale', error='Scene changed after planning; no step executed')
            return result
        for path, expected in request['input_hashes'].items():
            if digest(Path(path)) != expected:
                raise ValueError(f'Input changed after submission: {path}')
        evidence = generated_evidence(request)
        captured = {}
        for _, _, _, item in evidence:
            if Path(item['path']).exists():
                raise ValueError('Generated diagnostic appeared before execution; no old evidence may be rebound')
        env = dict(os.environ, VIBUILD_CACHE_DIR=str(root/'.runtime/cache'))
        if proxy:
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
                env[key] = proxy
        for index, step in enumerate(request['steps']):
            if paused(root):
                result['state'] = 'cancelled'
                break
            log = directory/f'{index+1:02}_{step["tool"]}.log'
            args = list(step['args'])
            if 'diagnostics' in step:
                scene_hash = digest(root/'scene.blend')
                packet = read(absolute(option(args, '--output'))/request['region']/'packet.json', {})
                if packet.get('scene_sha256') != scene_hash:
                    raise ValueError('Prepared scoring packet does not match the saved scene')
                bound = {'scene_sha256': scene_hash, 'images': [], 'facts': []}
                for consumer, producer, kind, item in evidence:
                    if consumer != index:continue
                    proof = captured[(producer, item['path'])]
                    if proof != (scene_hash, digest(Path(item['path']))):
                        raise ValueError('Generated diagnostic or scene changed after its producing stage')
                    bound[kind].append(item)
                manifest = directory/f'{index+1:02}_diagnostics.json'
                write(manifest, bound)
                args += ['--diagnostics', str(manifest)]
            command = [str(ROOT/'vibuild'), 'tool', '--config', str(config.resolve()), step['tool'], *args]
            begin = time.time()
            outcome, code = execute(root, command, log, env)
            row = {'tool': step['tool'], 'action': step['args'][0], 'state': outcome,
                   'seconds': round(time.time()-begin, 3), 'exit_code': code, 'log': str(log)}
            result['steps'].append(row)
            result['state'] = outcome
            write(directory/'state.json', result)
            if outcome != 'completed':
                output = option(args, '--output')
                result.update(failure_detail(log, absolute(output) if output else None, begin))
                break
            produced = [item for _, producer, _, item in evidence if producer == index]
            if produced:
                scene_hash = digest(root/'scene.blend')
                for item in produced:
                    captured[(index, item['path'])] = (scene_hash, digest(Path(item['path'])))
            if step['tool'] == 'score' and step['args'][0] == 'run':
                out = absolute(option(step['args'], '--output'))
                score_path = out/request['region']/'score.json'
                score = read(score_path, {})
                result['score'] = {k: score.get(k) for k in ('score', 'status', 'confidence')}
                result['score']['path'] = str(score_path)
            if step['tool'] == 'scene' and step['args'][0] == 'iterate':
                result['comparisons'] = str(absolute(option(step['args'], '--output'))/'review.html')
    except KeyboardInterrupt:
        (root/'.runtime/main').mkdir(parents=True, exist_ok=True)
        (root/'.runtime/main/STOP').touch()
        result.update(state='interrupted', error='Execution service stopped; no automatic retry')
    except Exception as exc:
        result.update(state='failed', error=str(exc))
    finally:
        result.update(finished=time.time(), current_scene_sha256=digest(root/'scene.blend'))
        write(directory/'state.json', result)
        write(directory/'result.json', result)
        publish(root, result)
        with (queue(root)/'notifications.jsonl').open('a') as stream:
            stream.write(json.dumps({'job': directory.name, 'state': result['state'],
                                     'receipt': str(directory/'result.json')})+'\n')
        signal.signal(signal.SIGTERM, previous)
    return result


def work(root, config, proxy=None):
    queue(root).mkdir(parents=True, exist_ok=True)
    results = []
    with file_lock(queue(root)/'worker.lock'):
        for path in sorted(queue(root).glob('*/state.json')):
            state = read(path, {})
            if paused(root):
                break
            # A killed worker must never rerun a possibly committed script.
            if state.get('state') == 'running':
                state.update(state='interrupted', error='Previous worker stopped; inspect latest root before planning new work')
                write(path, state); write(path.parent/'result.json', state)
                results.append(state)
                continue
            if state.get('state') != 'queued':
                continue
            results.append(run_job(root, path.parent, config, proxy))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['submit', 'work', 'status'])
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--config', type=Path, default=default_config())
    p.add_argument('--proxy')
    a = p.parse_args(argv);root = a.task.resolve()
    try:
        if a.action == 'submit':
            if not a.manifest:p.error('--manifest is required')
            result = submit(root, a.manifest)
        elif a.action == 'work':result = [brief(r) for r in work(root, a.config, a.proxy)]
        else:
            states = [read(x, {}) for x in sorted(queue(root).glob('*/state.json'))]
            result = {'paused': paused(root), 'total': len(states),
                      'pending': [brief(r) for r in states if r.get('state') not in TERMINAL],
                      'recent': [brief(r) for r in states if r.get('state') in TERMINAL][-3:],
                      'receipts': str(queue(root))}
    except (ValueError, OSError) as exc:
        print(json.dumps({'error': str(exc)}, separators=(',', ':')))
        return 1
    print(json.dumps(result, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
