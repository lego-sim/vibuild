"""Prepare bounded multi-region connection evidence; never open or reserve a scene."""
import argparse
from collections import deque
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re


COMPONENTS = ('passage_and_missing_volume', 'floor_and_landings', 'walls_and_openings',
              'ceiling_and_soffits', 'stairs_and_rails', 'doors_and_glazing')


def digest(path):
    with Path(path).open('rb') as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
        return result.hexdigest()


def shortest_path(graph, start, end):
    queue = deque([[start]]); seen = {start}
    while queue:
        path = queue.popleft()
        if path[-1] == end:
            return path
        for neighbor in sorted(graph[path[-1]]):
            if neighbor not in seen:
                seen.add(neighbor); queue.append(path + [neighbor])
    return []


def resolve_scope(partition, requested):
    known = {r['id'] for r in partition['regions']}
    requested = sorted(set(requested))
    if len(requested) < 2 or not set(requested) <= known:
        raise ValueError('Select at least two known regions')
    edges = [e for e in partition['interfaces'] if e.get('status') == 'observed_transition']
    graph = {rid: set() for rid in known}
    for edge in edges:
        a, b = edge['regions']; graph[a].add(b); graph[b].add(a)
    expanded = set(requested); relations = []
    for a, b in combinations(requested, 2):
        path = shortest_path(graph, a, b); expanded.update(path)
        relations.append({'regions': [a, b], 'direct_edge': 'observed' if b in graph[a] else 'not_observed',
                          'known_route': path, 'status': 'observed_route' if path else 'needs_evidence'})
    return sorted(expanded), sorted((e for e in edges if set(e['regions']) <= expanded), key=lambda e: e['id']), relations


def eligible(frame, excluded):
    return (frame['name'] not in excluded and
            not {'audit', 'holdout'} & {frame.get('split'), frame.get('evaluation_split')} and
            all(k in frame for k in ('center', 'rotation', 'K', 'width', 'height', 'original_rgb')))


def select_evidence(edge, frames, window=8.0):
    """Four complementary views, selected without camera-core filtering."""
    selected = []; roles = set()
    events = sorted(edge.get('evidence', []), key=lambda e: (e['video'], e['time_range']))
    if not events:
        return [], ['crossing', 'approach_from', 'approach_to', 'context']
    # Prefer the event whose two approaches have the most available original frames.
    def support(event):
        lo, hi = event['time_range']
        pool = [f for f in frames if f['video'] == event['video']]
        return (sum(lo-window <= f['time'] <= lo for f in pool) > 0) + (sum(hi <= f['time'] <= hi+window for f in pool) > 0)
    event = max(events, key=support)
    lo, hi = event['time_range']; mid = (lo+hi)/2
    pool = [f for f in frames if f['video'] == event['video']]
    used = set()
    def add(role, candidates, key):
        candidates = [f for f in candidates if f['name'] not in used]
        if candidates:
            f = min(candidates, key=lambda f: (key(f), f['name']))
            used.add(f['name']); roles.add(role)
            selected.append({'frame': f, 'role': role, 'event': event,
                             'offset_from_crossing_seconds': round(f['time']-mid, 6) if f['video'] == event['video'] else None})
    add('crossing', [f for f in pool if lo-2 <= f['time'] <= hi+2], lambda f: abs(f['time']-mid))
    def approach_rank(f, target):
        supported = set(edge['regions']) <= {v['region'] for v in f.get('visible_regions', [])}
        return (not supported, abs(f['time']-target))
    add('approach_from', [f for f in pool if lo-window <= f['time'] <= lo], lambda f: approach_rank(f, lo-3))
    add('approach_to', [f for f in pool if hi <= f['time'] <= hi+window], lambda f: approach_rank(f, hi+3))
    # Prefer an observed return traversal, then another image showing both sides.
    reverse = [e for e in events if e.get('from') == event.get('to') and e.get('to') == event.get('from')]
    reverse_pool = [f for f in frames if any(f['video'] == e['video'] and e['time_range'][0]-2 <= f['time'] <= e['time_range'][1]+2 for e in reverse)]
    context = [f for f in frames if set(edge['regions']) <= {v['region'] for v in f.get('visible_regions', [])}]
    add('context', reverse_pool or context,
        lambda f: -sum(v.get('observed_sparse_points', 0) for v in f.get('visible_regions', []) if v['region'] in edge['regions']))
    return selected, sorted({'crossing', 'approach_from', 'approach_to', 'context'}-roles)


def validate_boxes(boxes, regions, interfaces):
    if not boxes:
        raise ValueError('At least one bounded connection box is required')
    for box in boxes:
        lo, hi = box['bounds_min'], box['bounds_max']
        if len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(x) for x in lo+hi) or not all(a < b for a,b in zip(lo,hi)):
            raise ValueError('Connection boxes require finite positive-volume bounds')
        if not set(box.get('regions', [])) <= set(regions) or not box.get('regions'):
            raise ValueError('Each connection box must declare affected regions in the known route')
        if not set(box.get('interfaces', [])) <= set(interfaces) or not box.get('interfaces'):
            raise ValueError('Each connection box must name observed interfaces')
    if not set(interfaces) <= {i for box in boxes for i in box['interfaces']}:
        raise ValueError('Connection envelope must cover every selected interface')


def prepare(partition, requested, task_id, envelope=None, window=8.0, padding=2.0):
    if not re.fullmatch(r'J\d{2,}', task_id):
        raise ValueError('Connection task IDs must be stable Jxx identifiers')
    if not math.isfinite(window) or window <= 0 or not math.isfinite(padding) or padding <= 0:
        raise ValueError('Window and padding must be finite positive values')
    regions, edges, relations = resolve_scope(partition, requested)
    excluded = set(partition.get('validation', {}).get('excluded_audit_holdout_frames', []))
    frames = [f for f in partition['frames'] if eligible(f, excluded)]
    evidence = []; coverage = []; boxes = []
    for edge in edges:
        chosen, missing = select_evidence(edge, frames, window)
        coverage.append({'interface': edge['id'], 'regions': edge['regions'], 'missing_roles': missing,
                         'visual_confirmation': 'pending', 'crossing_event_count': len(edge.get('evidence', []))})
        for item in chosen:
            f = item.pop('frame')
            evidence.append({**item, 'interface': edge['id'], 'source_frame': f['name'], 'frame': f, 'primary': True})
        # A connection is a volume around both approaches, not just a zero-width seam.
        points = [x['frame']['center'] for x in evidence if x['interface'] == edge['id'] and x['role'] != 'context']
        for patch in edge.get('patches', [edge]):
            points.extend([patch['bounds_min'], patch['bounds_max']])
        boxes.append({'id': f'{task_id}/{edge["id"]}', 'interfaces': [edge['id']], 'regions': edge['regions'],
                      'bounds_min': [round(min(p[i] for p in points)-padding, 6) for i in range(3)],
                      'bounds_max': [round(max(p[i] for p in points)+padding, 6) for i in range(3)]})
    # Intermediate rooms can contain the omitted connecting volume, not just seams.
    # Preserve exact unions; ownership still needs an explicit object-level review.
    for region in partition['regions']:
        if region['id'] in set(regions)-set(requested) and 'bounds_min' in region:
            incident = [e['id'] for e in edges if region['id'] in e['regions']]
            for i, core in enumerate(region.get('core_boxes') or [region]):
                boxes.append({'id': f'{task_id}/{region["id"]}/core_{i+1}', 'regions': [region['id']],
                              'interfaces': incident, 'bounds_min': core['bounds_min'], 'bounds_max': core['bounds_max'],
                              'reason': 'Intermediate connecting volume; preserve existing geometry unless explicitly assigned.'})
    if envelope is not None:
        boxes = envelope['boxes']
    if edges:
        validate_boxes(boxes, regions, [e['id'] for e in edges])
    checklist = [{'id': f'{e["id"]}/{c}', 'interface': e['id'], 'component': c, 'status': 'unreviewed', 'evidence_frames': [],
                  'finding': '', 'owner': task_id, 'objects': []} for e in edges for c in COMPONENTS]
    return {'schema_version': 'connection_task_v1', 'id': task_id, 'kind': 'connection',
            'status': 'prepared' if edges and all(r['known_route'] for r in relations) else 'needs_evidence',
            'requested_regions': sorted(set(requested)), 'regions': regions,
            'intermediate_regions': sorted(set(regions)-set(requested)), 'relations': relations,
            'interfaces': [e['id'] for e in edges], 'envelope': {'boxes': boxes, 'semantics': 'Union of boxes; never the enclosing AABB.',
                'status': 'explicit' if envelope else 'proposed_from_crossing_and_approaches',
                'padding_scene_units': padding if envelope is None else None},
            'write_scope': {'owner': task_id, 'regions': regions, 'objects': [], 'status': 'needs_visual_scope_confirmation',
                'rule': 'One owner per component. List affected existing objects and new components inside the envelope before editing. Outside geometry is read-only. Re-scope whole components explicitly if needed.'},
            'evidence': evidence, 'coverage': coverage, 'mandatory_components': checklist,
            'acceptance': {'visual_total_min': 75, 'dimension_min': 60, 'minimum_confidence': 0.7,
                'mandatory_evidence': 'Every checklist row requires source-backed present/missing/not_applicable; unknown blocks acceptance.',
                'after_visual_pass': 'Bounded contact, connectivity, and affected-region regression checks on the current scene hash.'},
            'scene': 'TASK_ROOT/scene.blend', 'writer_reserved': False,
            'source_sha256': partition.get('provenance', {}).get('source_sha256', {}),
            'evidence_policy': 'Every selected view is primary; no three-view cap for connection assessment. Cache exact scene/evidence renders and keep repair comparison cards targeted.',
            'limitations': ['Trajectory crossing is evidence of a route, not a measured physical portal.',
                           'Sparse visibility is a candidate-selection hint, not confirmed surface coverage.',
                           'Selected approaches are temporal sides of the trajectory, not guaranteed opposing sight lines.',
                           'Preparation does not confirm visibility, accept geometry, or dispatch an agent.']}


def authoritative_partition(partition, resource_root):
    """Use frozen raw calibration; association labels only come from the partition."""
    root = Path(resource_root).resolve()
    hashes = partition.get('provenance', {}).get('source_sha256', {})
    index = json.loads((root/'index.json').read_text())
    for relative in (index['frames'], index['coordinates']):
        if digest(root/relative) != hashes.get(relative):
            raise ValueError(f'Frozen source mismatch: {relative}')
    raw = {f['name']: f for f in json.loads((root/index['frames']).read_text())}
    fields = ('name', 'index', 'video', 'time', 'center', 'rotation', 'K', 'width', 'height',
              'original_rgb', 'image', 'split', 'evaluation_split')
    frames = []
    for association in partition['frames']:
        f = raw[association['name']]
        frames.append({**{k:f[k] for k in fields if k in f},
                       'core_region': association.get('core_region'), 'visible_regions': association.get('visible_regions', [])})
    return {**partition, 'frames': frames}


def write_packet(packet, output, partition_path):
    output.mkdir(parents=True, exist_ok=True)
    views = []; known = {}
    for item in packet['evidence']:
        frame = item['frame']; name = frame['name']
        if name not in known:
            path = Path(frame['original_rgb'])
            frame['original_sha256'] = digest(path)
            matrix = [list(row)+[frame['center'][i]] for i,row in enumerate(frame['rotation'])]+[[0,0,0,1]]
            view = {'id': f'{packet["id"]}_{len(views)+1:02d}', 'source_frame': name,
                    'reference_rgb': str(path), 'width': frame['width'], 'height': frame['height'],
                    'K': frame['K'], 'camera_to_world': matrix,
                    'appearance_calibration': {'reference': {'K': frame['K'], 'camera_to_world': matrix}}}
            known[name] = view['id']; views.append(view)
        item['view_id'] = known[name]
    packet['partition_sha256'] = digest(partition_path)
    (output/'views.json').write_text(json.dumps({'frames': views}, indent=2)+'\n')
    (output/'connection.json').write_text(json.dumps(packet, indent=2)+'\n')
    lines = [f'# {packet["id"]}: connection repair', '',
             'Regions: '+', '.join(packet['regions'])+'. Interfaces: '+', '.join(packet['interfaces'])+'.',
             'Inspect missing space and complete structures across the bounded connection volume, not only touching edges.',
             'Preparation only: no scene mutation or writer reservation. Confirm the proposed envelope and object ownership before editing.', '',
             '| Interface | Role | View | Original frame | Offset (s) |', '| --- | --- | --- | --- | --- |']
    lines += [f'| {e["interface"]} | {e["role"]} | {e["view_id"]} | {e["source_frame"]} | {e["offset_from_crossing_seconds"]} |' for e in packet['evidence']]
    lines += ['', 'Missing role coverage: '+json.dumps([c for c in packet['coverage'] if c['missing_roles']]),
              '', 'For each interface inspect passage/missing volume, floors/landings, walls/openings, ceilings/soffits, stairs/rails, doors/glazing.',
              'Mark every component with source evidence. Unknown or unreviewed required evidence blocks acceptance.',
              'Render the single current root scene using views.json. Score first; after visual passage check affected contacts and neighboring regressions.']
    (output/'brief.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', required=True, type=Path)
    parser.add_argument('--resources', required=True, type=Path, help='Frozen record resources')
    parser.add_argument('--id', required=True)
    parser.add_argument('--regions', required=True, nargs='+')
    parser.add_argument('--envelope', type=Path, help='JSON with boxes: bounds_min/max, regions, interfaces')
    parser.add_argument('--window', type=float, default=8)
    parser.add_argument('--padding', type=float, default=2)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    partition = json.loads(args.partition.read_text())
    root = args.resources
    if args.output.resolve().is_relative_to(root.resolve()):
        parser.error('Output must be outside frozen resources')
    if (args.output/'connection.json').exists():
        parser.error('Packet exists; use a new output directory to preserve fixed evidence')
    packet = prepare(authoritative_partition(partition, root), args.regions, args.id,
                     json.loads(args.envelope.read_text()) if args.envelope else None, args.window, args.padding)
    write_packet(packet, args.output, args.partition)
    print(json.dumps({'id': packet['id'], 'status': packet['status'], 'regions': packet['regions'],
                      'interfaces': packet['interfaces'], 'views': len({e['view_id'] for e in packet['evidence']}),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
