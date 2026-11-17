"""One persistent modeling scene, atomic replacement, and assessment write exclusion."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import time
from artifact_cache import digest


def task_root(path):
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        plan = parent / 'plan.json'
        if plan.is_file() and json.loads(plan.read_text()).get('scene_storage') == 'single_working_scene':
            return parent
    return None


def active_assessments(root):
    for protocol in root.glob('*/scores/*/.runtime/protocol.json'):
        if Path(json.loads(protocol.read_text())['scene']).resolve() != root / 'scene.blend':
            continue
        for handoff in protocol.parent.parent.glob('*/.runtime/handoff.json'):
            state = json.loads(handoff.read_text()).get('state')
            if state not in ('completed', 'cancelled'):
                yield handoff


@contextlib.contextmanager
def file_lock(path, shared=False):
    """Nonblocking advisory lock; separate readers may share immutable inputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Scene or assessment is busy: {path}; finish the active operation before retrying') from None
        yield


@contextlib.contextmanager
def scene_lock(root, writing=False, shared=False):
    if writing and shared:
        raise ValueError('A scene writer requires an exclusive lock')
    if root is None:
        yield
        return
    runtime = root / '.runtime'
    with file_lock(runtime / 'scene.lock', shared=shared):
        if writing:
            state_path = root / 'status.json'
            if state_path.exists() and json.loads(state_path.read_text()).get('execution_paused'):
                raise ValueError('Task execution is paused')
            pending = list(active_assessments(root))
            if pending:
                raise ValueError(f'Scene frozen for assessment: {pending[0]}; record or explicitly cancel it first')
        yield


def commit(root, pending, expected_sha, region, round_number):
    """Save latest progress; this never grants acceptance."""
    target = root / 'scene.blend'
    if digest(target) != expected_sha:
        raise ValueError('Working scene changed during edit; refusing to overwrite newer progress')
    with pending.open('rb') as stream:
        if stream.read(7) != b'BLENDER':
            raise ValueError('Incomplete Blender save')
        os.fsync(stream.fileno())
    new_sha = digest(pending)
    os.replace(pending, target)
    state_path = root / 'status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.update(scene_storage='single_working_scene', scene=str(target),
                 current_candidate={'region': region, 'round': round_number, 'scene': str(target),
                                    'sha256': new_sha, 'assessment': 'pending'},
                 working_scene_sha256=new_sha, global_acceptance='not_assessed', updated=time.time())
    state.pop('published_scene', None)
    tmp = state_path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(tmp, state_path)
    return new_sha
