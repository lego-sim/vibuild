"""Prepare fixed regional evidence, run independent judges, and aggregate their scores."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import functools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vibuild.config import default_config, settings
from vibuild import resources, sessions
from vibuild.calibration import view as _view
from scene_storage import task_root, scene_lock, file_lock
load, save, digest = resources.load, resources.save, resources.digest


def locked_scene(function=None, *, shared=False):
    if function is None:
        return lambda wrapped: locked_scene(wrapped, shared=shared)
    @functools.wraps(function)
    def run(args):
        root = task_root(args.output)
        if root is None and getattr(args, 'scene', None):root = task_root(args.scene)
        with scene_lock(root, shared=shared):
            if root and getattr(args, 'scene', None) and args.scene.resolve() != root/'scene.blend':
                raise ValueError('Score the current task-root scene.blend, not a retired snapshot')
            return function(args)
    return run


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def result_schema(packet):
    schema = load(ROOT/'schemas/score.schema.json')
    if 'connection_requirements' in packet:
        schema['required'].append('connection_checks')
        schema['properties']['connection_checks'] = {'type':'array','items':{
            'type':'object','additionalProperties':False,'required':['id','status','frame_ids','notes'],
            'properties':{'id':{'type':'string'},'status':{'type':'string','enum':['present','missing','unknown','not_applicable']},
                          'frame_ids':{'type':'array','items':{'type':'string'}},'notes':{'type':'string'}}}}
    return schema


def regional_coverage(frame, region):
    return max((v['fraction_of_registered_observations'] for v in frame.get('visible_regions', [])
                if v['region'] == region), default=0.)


def regional_candidates(frames, region):
    """Source relevance is checked once, for automatic and explicit primary views alike."""
    return [f for f in frames if not {'audit', 'holdout'} & {f.get('evaluation_split'), f.get('split')}
            and regional_coverage(f, region) >= .35]


def select_views(frames, scope, count=3):
    """Fixed deterministic model views with substantial regional coverage and pose diversity."""
    strong = regional_candidates(frames, scope['id'])
    local = [f for f in strong if f.get('core_region') == scope['id']]
    pool = local if len(local) >= 2 else strong
    if not pool: return []
    preferred = [str(n) for n in scope.get('suggested_first_frames', [])]
    ident = lambda f: str(f.get('index', f.get('id')))
    pool.sort(key=lambda f: (preferred.index(ident(f)) if ident(f) in preferred else len(preferred), ident(f)))
    selected = [pool.pop(0)]
    def distance(a, b):
        def pose(f):
            if 'camera_to_world' in f:
                value=np.array(f['camera_to_world']);return value[:3,3],value[:3,2]
            return np.array(f.get('center',f.get('scene_center'))),np.array(f.get('rotation',f.get('camera_to_scene_rotation')))[:,2]
        ca,za=pose(a);cb,zb=pose(b)
        return min(np.linalg.norm(ca-cb)/5, 1.) + (1-float(za@zb))/2
    while pool and len(selected) < count:
        chosen = max(pool, key=lambda f: min(distance(f, prev) for prev in selected) * (.5 + .5*regional_coverage(f, scope['id'])))
        selected.append(chosen); pool.remove(chosen)
    return selected


def scaled_view(frame, source, width):
    view = _view(frame, source)
    w, h = view['width'], view['height']; width = min(width, w); height = round(h*width/w)
    K = np.array(view['K'], dtype=float)
    K[0] *= width/w; K[0, 2] += (width/w-1)/2
    K[1] *= height/h; K[1, 2] += (height/h-1)/2
    return {**view, 'width': width, 'height': height, 'K': K.tolist(), 'geometry': False, 'appearance': True}


def connection_spec(path, partition_path, partition, raw, source, width):
    """Use every frozen transition view, with original calibration and explicit coverage."""
    from connections import COMPONENTS, validate_boxes
    value = load(path)
    if value.get('schema_version') != 'connection_task_v1' or value.get('partition_sha256') != digest(partition_path):
        raise ValueError('Connection evidence belongs to another partition')
    rid = value['id']
    import re
    if not re.fullmatch(r'J[0-9]{2,}', rid):raise ValueError('Expected a Jxx connection task')
    edges = {e['id']: e for e in partition['interfaces'] if e.get('status') == 'observed_transition'}
    if not value['interfaces'] or not set(value['interfaces']) <= set(edges):
        raise ValueError('Connection requires observed interfaces')
    if not set(value['regions']) <= {r['id'] for r in partition['regions']}:
        raise ValueError('Unknown connection regions')
    boxes = value['envelope']['boxes']; validate_boxes(boxes, value['regions'], value['interfaces'])
    for ident in value['interfaces']:
        if not set(edges[ident]['regions']) <= set(value['regions']):
            raise ValueError('Connection omits an interface endpoint')
    selected = {}; originals = {}; by_interface = {ident: set() for ident in value['interfaces']}
    roles = {ident:set() for ident in value['interfaces']}
    excluded = set(partition.get('validation',{}).get('excluded_audit_holdout_frames',[]))
    required_roles = {'crossing','approach_from','approach_to','context'}
    for item in value['evidence']:
        frame = raw.get(item['source_frame'])
        if frame is None or item['source_frame'] in excluded or {'audit','holdout'} & {frame.get('split'),frame.get('evaluation_split')}:
            raise ValueError('Connection requires known non-audit source frames')
        if item['interface'] not in by_interface:raise ValueError('Unknown evidence interface')
        if item['role'] not in required_roles:raise ValueError('Unknown connection evidence role')
        roles[item['interface']].add(item['role'])
        for key in ('center','rotation','K','width','height','original_rgb'):
            if frame[key] != item['frame'][key]:raise ValueError('Connection calibration differs from frozen source')
        original = Path(frame['original_rgb'])
        if not original.is_absolute():original = source/original
        if original not in originals:originals[original] = digest(original)
        if originals[original] != item['frame'].get('original_sha256'):
            raise ValueError('Connection original image changed since evidence selection')
        view = {**scaled_view(frame, source, width), 'id': item['view_id']}
        if view['id'] in selected and selected[view['id']] != view:
            raise ValueError('Duplicate connection view has inconsistent source')
        selected[view['id']] = view; by_interface[item['interface']].add(view['id'])
    if any(len(ids) < 2 for ids in by_interface.values()):
        raise ValueError('Every connection interface needs at least two source views')
    required = [{'id': f'{ident}/{component}', 'interface': ident, 'component': component,
                 'frame_ids': sorted(by_interface[ident])}
                for ident in value['interfaces'] for component in COMPONENTS]
    coverage = value['coverage']
    if len(coverage) != len(roles) or {c['interface'] for c in coverage} != set(roles):
        raise ValueError('Connection coverage must include every interface exactly once')
    for item in coverage:
        if set(item['missing_roles']) != required_roles - roles[item['interface']]:
            raise ValueError('Connection role coverage differs from actual evidence')
    blockers = [f'{ident}: missing {role}' for ident in roles for role in sorted(required_roles-roles[ident])]
    if value.get('status') != 'prepared':blockers.append('Connection route evidence is incomplete')
    return {'region':rid,'name':'Connection '+', '.join(value['regions']),
            'scope':{'id':rid,'core_boxes':boxes,'scope':'Assigned connection boxes and whole-component ownership; not the enclosing AABB.'},
            'views':list(selected.values()),'connection_requirements':required,'evidence_blockers':blockers}


def object_context(inventory, scope, original, views):
    if scope.get('core_boxes'):
        boxes=[(np.asarray(b['bounds_min']),np.asarray(b['bounds_max'])) for b in scope['core_boxes']]
    else:
        curved = not scope.get('polygon')
        poly = np.array(scope['curved_centerline'] if curved else scope['polygon'], dtype=float)
        radius = scope['curved_radius'] if curved else 0
        z = scope.get('core_z_interval', scope.get('z'))
        boxes=[(np.r_[poly.min(axis=0)-radius,min(z)],np.r_[poly.max(axis=0)+radius,max(z)])]
    named = {o['name'] if isinstance(o, dict) else o for k in ('core_candidates', 'shared_read_only') for o in original.get(k, [])}
    result = []
    for obj in inventory['objects']:
        if obj['type'] != 'MESH' or obj['hide_render']: continue
        bounds = np.array(obj['world_bounds']); lo, hi = bounds.min(axis=0), bounds.max(axis=0)
        if obj['name'] not in named and not any((hi >= low-.3).all() and (lo <= high+.3).all() for low,high in boxes):continue
        projections = {}
        for v in views:
            pose = np.array(v['camera_to_world']); points = (bounds-pose[:3, 3])@pose[:3, :3]
            if (points[:, 2] <= .01).all(): continue
            visible = points[points[:, 2] > .01]; pixel = visible@np.array(v['K']).T
            xy = pixel[:, :2]/pixel[:, 2:]/[v['width'], v['height']]
            a, b = xy.min(axis=0), xy.max(axis=0)
            if (b < 0).any() or (a > 1).any(): continue
            projections[v['id']] = np.clip(np.r_[a, b], 0, 1).round(3).tolist()
        result.append({'name': obj['name'], 'bounds': np.r_[lo, hi].round(3).tolist(),
                       'materials': obj['materials'], 'projected_bounds': projections})
    return result


def run_tool(args, cfg, log):
    with log.open('w') as stream:
        subprocess.run([cfg['runtime']['analysis_python'], str(ROOT/'tools/scene.py'), *map(str, args),
                        '--blender', cfg['runtime']['blender']], stdout=stream, stderr=subprocess.STDOUT, check=True)


def make_card(region, view, render, dest):
    reference = Image.open(view['reference_rgb']).convert('RGB').resize((view['width'], view['height']), Image.Resampling.LANCZOS)
    prediction = Image.open(render).convert('RGB')
    if prediction.size != reference.size: raise ValueError('Preview grid differs')
    w, h = prediction.size
    board = Image.new('RGB', (w*2, h+64), '#f4f5f7'); draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    draw.text((12, 7), f'{region} | VIEW {view["id"]} | LEFT: VIDEO   RIGHT: CURRENT BLEND', font=font, fill='#163a5f')
    draw.text((12, 35), 'Same calibrated pose / normalized boxes refer to each image, excluding header', fill='#444')
    board.paste(reference, (0, 64)); board.paste(prediction, (w, 64)); board.save(dest, quality=91)


@locked_scene
def prepare(args):
    from vibuild.scoring import RUBRIC, rubric_hash
    begin = time.monotonic(); record = resources.record_path(args.record).resolve(); out = args.output.resolve()
    cfg = settings(args.config); baseline = args.scene.resolve(); partition_path = args.partition.resolve()
    root = task_root(baseline)
    if root and (out.parent.name != 'scores' or out.parent.parent.parent != root):
        raise ValueError('Managed score output must be TASK_ROOT/REGION/scores/ROUND for assessment write exclusion')
    if out.is_relative_to(record/'resources'):
        raise ValueError('Scoring output must not modify frozen resources')
    partition = load(partition_path)
    if partition.get('schema_version') != 'partition_v1':
        raise ValueError('Expected partition_v1 with registered source provenance')
    res = record/'resources'; index = load(res/'index.json'); frame_path = res/index['frames']
    coordinate_path = res/index['coordinates']
    sources = partition.get('provenance',{}).get('source_sha256',{})
    for path in (frame_path,coordinate_path):
        if sources.get(str(path.relative_to(res))) != digest(path):
            raise ValueError('Partition calibration/frames differ from this record')
    # Whitelist raw calibration. Old zone annotations are never ranking priors.
    fields = ('name','index','video','time','center','rotation','K','width','height',
              'original_rgb','image','evaluation_split','split')
    raw = {f['name']:{k:f[k] for k in fields if k in f} for f in load(frame_path)}
    specs = []; all_regions = [r['id'] for r in partition['regions']]
    connection = getattr(args, 'connection', None)
    if connection:
        if getattr(args, 'frame', None):raise ValueError('Connection frames are fixed by its manifest; do not omit views')
        specs = [connection_spec(connection, partition_path, partition, raw, res, args.width)]
    for region in ([] if connection else partition['regions']):
        if args.region and region['id'] not in args.region:continue
        rid = region['id']; lo,hi = region['bounds_min'],region['bounds_max']
        scope = {'id':rid,'polygon':[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]],
                 'core_z_interval':[lo[2],hi[2]]}
        if region.get('core_boxes'):
            scope.update(core_boxes=region['core_boxes'],scope='Own only the union of core_boxes; polygon and z are a display envelope, not additional scope.')
        frames = []
        for association in partition['frames']:
            visible = association.get('visible_regions',[])
            if association.get('core_region') != rid and not any(v['region']==rid for v in visible):continue
            frame = raw.get(association['name'])
            if frame is None:raise ValueError('Partition refers to an unknown calibrated frame')
            frames.append({**frame,'visible_regions':visible,'core_region':association.get('core_region')})
        frozen_ids = getattr(args, 'frame', None)
        if frozen_ids:
            eligible = {str(f.get('index')): f for f in regional_candidates(frames, rid)}
            if len(set(frozen_ids)) != len(frozen_ids) or not set(frozen_ids) <= set(eligible):
                raise ValueError('Frozen frames must be unique non-audit IDs with sufficient regional coverage')
            chosen = [eligible[ident] for ident in frozen_ids]
        else:
            chosen = select_views(frames,scope,args.views)
        views = [scaled_view(f,res,args.width) for f in chosen]
        specs.append({'region':rid,'name':region['name'],'scope':scope,'views':views})
    if not specs or (args.region and set(args.region) != {s['region'] for s in specs}):raise ValueError('Unknown/empty regions')
    if any(len(s['views']) < 2 for s in specs):raise ValueError('Each region needs at least two calibrated non-audit views')
    protocol = {'version':1,'scene':str(baseline),'scene_sha256':digest(baseline),'rubric_hash':rubric_hash(),
                'all_regions':all_regions,
                'settings':{'samples':args.samples,'width':args.width,'threads':args.threads,'denoise':True},
                'regions':specs,
                'source_hashes':{str(path):digest(path) for path in (partition_path,frame_path,coordinate_path)},
                'reference_hashes':{v['reference_rgb']:digest(v['reference_rgb']) for s in specs for v in s['views']}}
    if connection:
        protocol['source_hashes'][str(connection.resolve())] = digest(connection)
        protocol['assessment_kind'] = 'connection'
    rt = out/'.runtime'; rt.mkdir(parents=True, exist_ok=True)
    if (rt/'protocol.json').exists() and load(rt/'protocol.json') != protocol: raise ValueError('Output belongs to a different protocol; use a new directory')
    save(rt/'protocol.json', protocol); save(out/'rubric.json', RUBRIC)
    run_tool(['inspect','--blend',baseline,'--output',rt/'inventory'], cfg, rt/'inspect.log')
    inventory = load(rt/'inventory/inventory.json')
    if inventory['source_sha256'] != protocol['scene_sha256']: raise ValueError('Stale inventory')
    unique = {}
    for spec in specs:
        for view in spec['views']:
            if view['id'] in unique and unique[view['id']] != view: raise ValueError('Inconsistent duplicate frame calibration')
            unique[view['id']] = view
    jobs = min(args.workers, len(unique)); batches = [list(unique.values())[i::jobs] for i in range(jobs)]
    def render(i):
        manifest = rt/f'views_{i}.json'; save(manifest, {'frames': batches[i]})
        run_tool(['render','--blend',baseline,'--manifest',manifest,'--output',rt/f'render_{i}',
                  '--samples',args.samples,'--threads',args.threads,'--denoise'], cfg, rt/f'render_{i}.log')
        return {v['id']:rt/f'render_{i}'/f'{v["id"]}.png' for v in batches[i]}
    renders = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for value in pool.map(render, range(jobs)): renders.update(value)
    for spec in specs:
        directory = out/spec['region']; directory.mkdir(exist_ok=True)
        views = []
        for v in spec['views']:
            card = directory/f'view_{v["id"]}.jpg'; make_card(spec['region'], v, renders[v['id']], card)
            views.append({**v, 'card': str(card), 'card_sha256': digest(card)})
        scope = spec['scope']; original = {}
        context = object_context(inventory, scope, original, views)
        packet = {'region': spec['region'], 'name': spec['name'], 'scene_sha256': protocol['scene_sha256'], 'rubric_hash': rubric_hash(),
                  'scope': {k: scope[k] for k in ('polygon','core_z_interval','core_boxes','scope','level') if k in scope},
                  'views': views, 'objects': context,
                  'preview_settings': protocol['settings'],
                  'limitations': 'Projected boxes are AABB candidates, not visibility/semantic proofs. Scene units have unverified absolute scale. RGB lighting/exposure differs; do not treat color residual as geometric error. Basic bounds/material metadata cannot prove manifoldness or hidden connections.'}
        if 'connection_requirements' in spec:
            packet.update(connection_requirements=spec['connection_requirements'], evidence_blockers=spec['evidence_blockers'])
        if scope.get('curved_centerline'):
            points = scope['curved_centerline']
            packet['scope'].update(curved_centerline=points[::max(1,len(points)//24)]+[points[-1]], curved_radius=scope['curved_radius'])
        save(directory/'packet.json', packet)
    if digest(baseline) != protocol['scene_sha256']:
        raise ValueError('Scene changed during evidence preparation; use a new frozen candidate')
    stats = {'prepare_seconds': time.monotonic()-begin, 'regions': len(specs), 'unique_render_views': len(unique),
             'blender_processes_max': jobs, 'inspect_requests': 1, 'render_requests': jobs,
             'scene_sha256': protocol['scene_sha256'], 'rubric_hash': rubric_hash()}
    save(rt/'preparation.json', stats)
    print(json.dumps({'state': 'prepared', 'output': str(out)}), flush=True)



def verified_packet(work, protocol):
    from vibuild.scoring import rubric_hash
    if protocol.get('rubric_hash') != rubric_hash():
        raise ValueError('Historical rubric cannot validate a current assessment')
    if digest(Path(protocol['scene'])) != protocol['scene_sha256']:
        raise ValueError('Scene changed since evidence preparation')
    packet = load(work/'packet.json')
    if packet['region'] != work.name or packet['scene_sha256'] != protocol['scene_sha256'] or packet['rubric_hash'] != rubric_hash():
        raise ValueError('Packet scene/region/rubric differs from the frozen scoring protocol')
    spec = next(s for s in protocol['regions'] if s['region'] == work.name)
    if [{k: v[k] for k in expected} for v, expected in zip(packet['views'], spec['views'])] != spec['views'] or len(packet['views']) != len(spec['views']):
        raise ValueError('Packet views differ from frozen protocol')
    for key in ('connection_requirements','evidence_blockers'):
        if packet.get(key) != spec.get(key):raise ValueError('Connection requirements differ from frozen protocol')
    for view in [*packet['views'], *packet.get('diagnostic_views', [])]:
        if digest(Path(view['card'])) != view['card_sha256']:
            raise ValueError('Scoring card has changed')
    return packet


def verified_score(work, protocol):
    from vibuild.scoring import summarize_region
    packet = verified_packet(work, protocol); receipt = load(work/'.runtime/scored.json')
    if receipt['packet_sha256'] != signature(packet) or receipt['result_sha256'] != digest(work/'.runtime/final.json'):
        raise ValueError('Score receipt no longer matches evidence/result')
    return summarize_region(load(work/'.runtime/final.json'), packet)


@locked_scene(shared=True)
def judge(args):
    # Independent judges share the frozen scene but never an assessment's output.
    work = args.output.resolve()/args.region[0]
    with file_lock(work/'.runtime/judge.lock'):
        return judge_assessment(args)


def judge_assessment(args):
    from vibuild.scoring import prompt, model_packet, summarize_region, validate_result
    out = args.output.resolve(); work = out/args.region[0]
    handoff_path = work/'.runtime/handoff.json'
    handoff = load(handoff_path) if handoff_path.exists() else None
    if handoff and handoff.get('state') in ('running', 'cancelled'):
        raise ValueError('External assessor is running or cancelled; reconcile it and prepare a new packet before standalone scoring')
    packet = verified_packet(work, load(out/'.runtime/protocol.json'))
    text = prompt(packet) + '\nPACKET=' + json.dumps(model_packet(packet), ensure_ascii=False, separators=(',', ':'))
    key = signature({'packet': packet, 'prompt': text, 'model': args.model, 'effort': args.reasoning_effort,
                     'schema': signature(result_schema(packet))})
    receipt = work/'.runtime/scored.json'
    if receipt.exists() and load(receipt).get('key') == key:
        save(work/'score.json', verified_score(work, load(out/'.runtime/protocol.json')))
        print(json.dumps({'result': str(work/'score.json')}), flush=True); return
    receipt.unlink(missing_ok=True)
    schema_path = work/'.runtime/scoring.schema.json'; save(schema_path, result_schema(packet))
    state = sessions.run(work, text, args.config, output_schema=schema_path, model=args.model,
                         reasoning_effort=args.reasoning_effort, read_only=True,
                         images=[v['card'] for v in [*packet['views'], *packet.get('diagnostic_views', [])]])
    if state['state'] != 'completed': raise RuntimeError(f'{packet["region"]}: {state["state"]}')
    value = load(work/'.runtime/final.json'); validate_result(value, packet)
    verified_packet(work, load(out/'.runtime/protocol.json'))
    summary = summarize_region(value, packet); save(work/'score.json', summary)
    tool_calls = 0; usage = {}
    for line in (work/'.runtime/events.jsonl').read_text().splitlines():
        event = json.loads(line); item = event.get('item', {})
        if event.get('type') == 'item.completed' and item.get('type') in ('command_execution','mcp_tool_call','web_search'): tool_calls += 1
        if event.get('type') == 'turn.completed': usage = event.get('usage', {})
    save(receipt, {'key': key, 'packet_sha256': signature(packet), 'result_sha256': digest(work/'.runtime/final.json'),
                   'seconds': state.get('finished',time.time())-state['started'], 'tool_calls': tool_calls, 'usage': usage,
                   'model': args.model, 'reasoning_effort': args.reasoning_effort})
    if handoff is not None:
        handoff.update(state='completed', session_id=state.get('thread_id'),
                       context_mode='standalone_fresh', finished=state.get('finished', time.time()))
        save(handoff_path, handoff)
    print(json.dumps({'score': summary['score'], 'result': str(work/'score.json')}), flush=True)


def report(out):
    from vibuild.scoring import summarize, render_report
    regions = []; missing = []
    protocol = load(out/'.runtime/protocol.json')
    for spec in protocol['regions']:
        work = out/spec['region']
        if (work/'.runtime/scored.json').exists():
            result = verified_score(work, protocol); save(work/'score.json', result); regions.append(result)
        else: missing.append(spec['region'])
    connection = protocol.get('assessment_kind') == 'connection'
    expected = [s['region'] for s in protocol['regions']] if connection else protocol.get('all_regions', [s['region'] for s in protocol['regions']])
    missing = sorted(set(expected)-{r['region'] for r in regions})
    summary = summarize(regions, expected_regions=expected if not missing else None)
    summary.update(expected_regions=len(expected), missing_regions=missing)
    if missing: summary.update(passed=False, status='incomplete', partial_score=summary['score'], score=None)
    receipts = [load(out/r['region']/'.runtime/scored.json') for r in regions]
    preparation = load(out/'.runtime/preparation.json') if (out/'.runtime/preparation.json').exists() else {}
    judging = load(out/'.runtime/judging.json') if (out/'.runtime/judging.json').exists() else {}
    durations = [r['seconds'] for r in receipts if 'seconds' in r]
    summary['performance'] = {'prepare_seconds': preparation.get('prepare_seconds'),
                              'run_wall_seconds': judging.get('first_wall_seconds', judging.get('wall_seconds')),
                              'latest_run_seconds': judging.get('wall_seconds'),
                              'latest_cache_hits': judging.get('cache_hits', 0),
                              'model_sessions': len(receipts),
                              'tool_calls': sum(r['tool_calls'] for r in receipts) if all(isinstance(r.get('tool_calls'), int) for r in receipts) else None,
                              'session_seconds_median': float(np.median(durations)) if durations else None,
                              'session_seconds_sum': sum(durations),
                              'models': sorted({r['model'] for r in receipts if 'model' in r})}
    summary['dimension_means'] = {key: round(sum(r['dimensions'][key]['score'] for r in regions)/len(regions),2)
                                  for key in summary['rubric']['dimensions']} if regions else {}
    if connection:
        passed = bool(regions) and not missing and all(r['passed'] for r in regions)
        summary.update(assessment_kind='connection', passed=passed, global_acceptance='not_evaluated',
                       weighting='separate connection gates; excluded from global equal-region mean',
                       status='incomplete' if missing else 'passed' if passed else
                              'needs_evidence' if any(r['status']=='needs_evidence' for r in regions) else 'failed')
    save(out/'scores.json', summary); render_report(out, summary, regions)
    print(json.dumps({'score': summary.get('score'), 'missing': missing, 'report':str(out/'report.html')},ensure_ascii=False),flush=True)


@locked_scene
def supplement(args):
    """Attach frozen diagnostic evidence; never create an external assessor session."""
    out=args.output.resolve();work=out/args.region[0];rt=work/'.runtime';rt.mkdir(exist_ok=True)
    if (rt/'state.json').exists() or (rt/'scored.json').exists():raise ValueError('Assessment already started; prepare a new packet before changing evidence')
    protocol=load(out/'.runtime/protocol.json');packet=verified_packet(work,protocol)
    packet['diagnostic_views']=[];packet['diagnostic_facts']=[]
    if args.diagnostics:
        diagnostics=load(args.diagnostics)
        if diagnostics['scene_sha256']!=packet['scene_sha256']:raise ValueError('Diagnostics belong to another scene')
        for item in diagnostics.get('images',[]):
            path=(args.diagnostics.resolve().parent/item['path']).resolve()
            packet['diagnostic_views'].append({'id':item['id'],'role':item['role'],'card':str(path),'card_sha256':digest(path)})
        for item in diagnostics.get('facts',[]):
            path=(args.diagnostics.resolve().parent/item['path']).resolve();value=load(path)
            if value.get('scene_sha256',packet['scene_sha256'])!=packet['scene_sha256']:
                raise ValueError('Diagnostic fact belongs to another scene')
            packet['diagnostic_facts'].append({'role':item['role'],'source_sha256':digest(path),'value':value})
    save(work/'packet.json',packet);verified_packet(work,protocol)
    print(json.dumps({'state': 'evidence_ready', 'output': str(out)}))


@locked_scene
def cancel_assessor(args):
    if not args.reason:raise ValueError('Cancellation requires --reason after stopping the assessor')
    rt=args.output.resolve()/args.region[0]/'.runtime'
    handoff=load(rt/'handoff.json')
    if handoff.get('state')=='completed':raise ValueError('Assessment is already completed')
    handoff.update(state='cancelled',cancelled_at=time.time(),reason=args.reason)
    save(rt/'handoff.json',handoff)
    print(json.dumps({'state':'cancelled','region':args.region[0],'reason':args.reason}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run','report','_judge','supplement','cancel-assessor'])
    p.add_argument('--record'); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene',type=Path); p.add_argument('--partition',type=Path)
    p.add_argument('--connection',type=Path,help='prepare: fixed Jxx transition evidence; every selected view is primary')
    p.add_argument('--config', type=Path, default=default_config())
    p.add_argument('--region', action='append'); p.add_argument('--views',type=int,default=3)
    p.add_argument('--frame', action='append', help='prepare only: two or three frozen source frame IDs, in fixed order; requires one region')
    p.add_argument('--diagnostics',type=Path,help='supplement only: scene-bound supplemental images/facts manifest')
    p.add_argument('--reason',help='cancel-assessor only: why the stopped assessment is being cancelled')
    p.add_argument('--width',type=int,default=640); p.add_argument('--samples',type=int,default=8)
    p.add_argument('--threads',type=int,default=4); p.add_argument('--workers',type=int,default=3)
    p.add_argument('--model'); p.add_argument('--reasoning-effort')
    args = p.parse_args(argv)
    session = settings(args.config).get('session', {})
    args.model = args.model or session.get('model')
    args.reasoning_effort = args.reasoning_effort or session.get('reasoning_effort', 'high')
    if args.action in ('run', '_judge') and not args.model:p.error('Configure session.model or supply --model')
    if min(args.samples,args.threads,args.workers) < 1 or args.width < 16 or args.views not in (2,3): p.error('positive resource settings, width >=16 and views 2 or 3 required')
    if args.action == 'prepare':
        if not args.record or not args.scene or not args.partition:p.error('--record, --scene and --partition required')
        if args.frame and (not args.region or len(args.region)!=1 or len(args.frame) not in (2,3)):
            p.error('--frame requires exactly one region and two or three frame IDs')
        prepare(args)
    elif args.action in ('supplement','cancel-assessor'):
        if not args.region or len(args.region)!=1:p.error('Assessor handoff requires exactly one region')
        {'supplement':supplement,'cancel-assessor':cancel_assessor}[args.action](args)
    elif args.action == '_judge': judge(args)
    elif args.action == 'report': report(args.output.resolve())
    else:
        begin = time.monotonic(); out = args.output.resolve(); protocol = load(out/'.runtime/protocol.json')
        regions = [s['region'] for s in protocol['regions'] if not args.region or s['region'] in args.region]
        if not regions or (args.region and set(args.region) != set(regions)): raise ValueError('Unknown/empty requested regions')
        def execute(region):
            directory = out/region/'.runtime'; directory.mkdir(exist_ok=True)
            receipt = directory/'scored.json'
            previous = receipt.stat().st_mtime_ns if receipt.exists() else None
            with (directory/'runner.log').open('w') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_judge','--output',str(out),
                    '--config',str(args.config.resolve()),'--region',region,'--model',args.model,'--reasoning-effort',args.reasoning_effort], stdout=log, stderr=subprocess.STDOUT)
            print(f'{region}: {"completed" if result.returncode == 0 else "FAILED (see runner.log)"}',flush=True)
            cached = result.returncode == 0 and receipt.exists() and receipt.stat().st_mtime_ns == previous
            return region, result.returncode, cached
        with ThreadPoolExecutor(max_workers=args.workers) as pool: results = list(pool.map(execute,regions))
        elapsed = time.monotonic()-begin
        prior = load(out/'.runtime/judging.json') if (out/'.runtime/judging.json').exists() else {}
        prior_success = bool(prior.get('results')) and all(item[1] == 0 for item in prior['results'])
        first = prior.get('first_wall_seconds',prior.get('wall_seconds',elapsed)) if prior_success else elapsed
        save(out/'.runtime/judging.json', {'wall_seconds':elapsed,'first_wall_seconds':first,'workers':args.workers,
                                         'cache_hits':sum(cached for _,_,cached in results),'results':results})
        report(out)
        if any(code for _,code,_ in results): return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
