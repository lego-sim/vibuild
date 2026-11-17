"""Run short fresh decisions separated by deterministic execution; no LLM runners."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vibuild.config import default_config, settings
from vibuild import sessions
from jobs import work, queue, paused, brief
from scene_storage import file_lock
from task_monitor import read, write
from decision_context import material_index, decision_images


def decision_prompt(root, results, config=None, directory=None, index=None):
    status = read(root/'status.json', {})
    plan = read(root/'plan.json', {})
    state = {k: status.get(k) for k in ('active_region', 'current_task', 'global_acceptance')}
    steps = [{k: step.get(k) for k in ('id', 'state', 'score', 'acceptance')} for step in plan.get('steps', [])]
    instructions = (ROOT/'instructions/decision.md').read_text()
    index = index if index is not None else material_index(root, config or default_config(), directory)
    encode = lambda value: json.dumps(value, separators=(',', ':'))
    return (instructions+'\nPROJECT_CONTRACT:\n'+(ROOT/'AGENT.md').read_text()+f'\nPROJECT_ROOT={ROOT}\nTASK_ROOT={root}\n'
            +'CURRENT='+encode(state)+'\nSTEPS='+encode(steps)
            +'\nLAST_EXECUTION='+encode([brief(row) for row in results[-3:]])
            +'\nMATERIAL_INDEX='+encode(index)+'\nATTACHED_CURRENT_IMAGES='+encode(decision_images(index)))


def run(root, config, proxy=None):
    runtime = root/'.runtime/main';runtime.mkdir(parents=True, exist_ok=True)
    with file_lock(runtime/'main.lock'):
        state = {'state': 'starting', 'pid': os.getpid(), 'started': time.time(),
                 'mode': 'program_execution_between_fresh_decisions', 'proxy': proxy}
        results = []
        previous = signal.getsignal(signal.SIGTERM)
        def interrupt(*_):
            (runtime/'STOP').touch()
            raise KeyboardInterrupt('User stopped service')
        signal.signal(signal.SIGTERM, interrupt)
        parent_stops = json.loads(os.environ.get('VIBUILD_PARENT_STOPS', '[]'))
        os.environ['VIBUILD_PARENT_STOPS'] = json.dumps(parent_stops+[str(runtime/'STOP')])
        previous_env = {key: os.environ.get(key) for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy')}
        if proxy:
            for key in previous_env:os.environ[key] = proxy
        try:
            while not paused(root):
                state.update(state='executing', updated=time.time())
                write(runtime/'state.json', state)
                results = work(root, config, proxy)
                if paused(root):break
                # Always return execution failures to one fresh decision; never retry scripts automatically.
                directory = root/'.runtime/decisions'/f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
                state.update(state='deciding', decision=str(directory), updated=time.time())
                write(runtime/'state.json', state)
                index = material_index(root, config, directory)
                decision = sessions.run(directory, decision_prompt(root, results, config, directory, index), config,
                                        images=decision_images(index), writable_roots=(root,))
                state.update(last_decision_state=decision['state'], thread_id=decision.get('thread_id'))
                if decision['state'] != 'completed':
                    state.update(state='blocked', reason='Decision process did not complete; see its local receipt')
                    break
                queued = any(read(p, {}).get('state') == 'queued' for p in queue(root).glob('*/state.json'))
                if not queued and read(root/'status.json', {}).get('active_region') != index.get('region'):
                    continue  # Explicit owner handoff starts a fresh decision, never an idle polling loop.
                if not queued:
                    state.update(state='idle', reason='Decision submitted no execution job; no automatic model polling')
                    break
            if paused(root):state['state'] = 'paused'
        except KeyboardInterrupt:
            (runtime/'STOP').touch()
            state['state'] = 'paused'
        finally:
            state.update(finished=time.time())
            write(runtime/'state.json', state)
            signal.signal(signal.SIGTERM, previous)
            if parent_stops:os.environ['VIBUILD_PARENT_STOPS'] = json.dumps(parent_stops)
            else:os.environ.pop('VIBUILD_PARENT_STOPS', None)
            for key, value in previous_env.items():
                if value is None:os.environ.pop(key, None)
                else:os.environ[key] = value
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, type=Path)
    p.add_argument('--config', type=Path, default=default_config())
    p.add_argument('--proxy', help='optional proxy override; otherwise inherit the environment')
    args = p.parse_args()
    print(json.dumps(run(args.task.resolve(), args.config.resolve(), args.proxy), indent=2))


if __name__ == '__main__':
    main()
