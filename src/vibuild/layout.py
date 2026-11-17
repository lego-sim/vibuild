"""Task execution details always live under .runtime."""
from pathlib import Path
import json
import re


def closeout_policy():
    """Planning limits; the orchestrator records usage in the existing step receipt."""
    return {'reference': 'PLAN.md#closeout',
            'budget_scope': 'task_across_iterations_and_restarts',
            'max_check_batches': 1, 'max_seam_locations': 3, 'max_views': 2,
            'max_corrections': 1, 'max_reviews': 1, 'full_rescore_for_seam_only': False}


def task_paths(root, region=None, round_number=None):
    """Paths for one evolving global scene and per-session evidence."""
    root = Path(root).resolve()
    result = {'root': root, 'scene': root/'scene.blend', 'runtime': root/'.runtime',
              'cache': root/'.runtime/cache', 'plan': root/'plan.json',
              'status': root/'status.json', 'progress': root/'progress.html'}
    if region is not None:
        if not re.fullmatch(r'[RLJ][0-9]{2,}', region):
            raise ValueError('Unknown regional, closure or connection session')
        child = root/region
        result.update(region=child, region_runtime=child/'.runtime',
                      iterations=child/'iterations', scores=child/'scores')
        if round_number is not None:
            if type(round_number) is not int or round_number < 1:
                raise ValueError('Round must be a positive integer')
            result.update(iteration=child/'iterations'/f'{round_number:03d}',
                          score=child/'scores'/f'{round_number:03d}')
    elif round_number is not None:
        raise ValueError('Round requires a regional or closure session')
    return result


def initialize_task(root, steps):
    """Create the dated layout and a plan, without launching or completing work."""
    paths = task_paths(root)
    if not re.fullmatch(r'modeling_\d{8}_\d{6}', paths['root'].name):
        raise ValueError('Task root must be modeling_YYYYMMDD_HHMMSS')
    if paths['plan'].exists():
        raise ValueError('Task plan already exists')
    steps = [dict(step) if isinstance(step, dict) else {'id': step} for step in steps]
    identifiers = [step['id'] for step in steps]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError('Task steps must be nonempty and unique')
    for region in identifiers:
        task_paths(root, region)
    paths['cache'].mkdir(parents=True, exist_ok=True)
    for region in identifiers:
        child = task_paths(root, region)
        for key in ('region_runtime', 'iterations', 'scores'):
            child[key].mkdir(parents=True, exist_ok=True)
    plan = {'schema_version': 1, 'scene': 'scene.blend', 'scene_storage': 'single_working_scene', 'state': 'planned',
            'closeout_policy': closeout_policy(),
            'global_acceptance': 'not_assessed', 'execution_order': identifiers,
            'steps': [{**step, 'kind': step.get('kind', 'closure' if step['id'].startswith('L') else 'connection' if step['id'].startswith('J') else 'region'),
                       'state': 'planned', 'directory': step['id']} for step in steps]}
    paths['plan'].write_text(json.dumps(plan, indent=2)+'\n')
    return paths


def runtime_file(work, name):
    return Path(work) / '.runtime' / name


def workspace_of(path):
    return Path(path).parent.parent


def session_files(root, name):
    yield from Path(root).rglob('.runtime/' + name)
