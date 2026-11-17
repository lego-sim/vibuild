"""Fixed, evidence-bound regional visual ratings; no scene or workflow mutation."""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
from statistics import mean


RUBRIC = {
    'version': 'regional-score-v2',
    'dimensions': {
        'structure': {'label': 'Spatial structure', 'weight': .30},
        'completeness': {'label': 'Component completeness', 'weight': .25},
        'alignment': {'label': 'Position and proportion', 'weight': .20},
        'appearance': {'label': 'Material appearance', 'weight': .15},
        'connectivity': {'label': 'Connectivity', 'weight': .10},
    },
    'anchors': {'100': 'No discernible difference within the observed scope', '90': 'Only tiny differences',
                '80': 'Broadly consistent with minor defects', '70': 'Obvious local defects',
                '50': 'Major elements clearly wrong', '25': 'Only fragments match', '0': 'Absent or wholly contradictory'},
    'thresholds': {'region': 75, 'dimension': 60, 'global': 80, 'confidence': .7, 'views': 2},
    'aggregation': '0.7 * view mean + 0.3 * view minimum; unknown contributes zero to lower bound',
    'scope': 'Rapid visual and structural-clue assessment; not measured depth accuracy, physical safety or complete-scene acceptance',
}


def rubric_hash():
    return hashlib.sha256(json.dumps(RUBRIC, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'Invalid {name} fields')


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_result(value, packet):
    keys = ['region', 'views', 'findings', 'uncertainties']
    if 'connection_requirements' in packet:keys.append('connection_checks')
    _keys(value, keys, 'score result')
    if value['region'] != packet['region']:
        raise ValueError('Score region differs from evidence packet')
    if packet.get('rubric_hash', rubric_hash()) != rubric_hash():
        raise ValueError('Evidence packet uses another rubric')
    expected = [str(view['id']) for view in packet['views']]
    if len(expected) != len(set(expected)):
        raise ValueError('Duplicate packet view IDs')
    if not isinstance(value['views'], list):raise ValueError('Views must be a list')
    seen = []
    for view in value['views']:
        _keys(view, ('id', 'scores', 'confidence', 'notes'), 'view')
        if not isinstance(view['id'], str):raise ValueError('View ID must be a string')
        seen.append(view['id'])
        _keys(view['scores'], RUBRIC['dimensions'], 'dimension')
        for score in view['scores'].values():
            if score is not None and (type(score) is not int or not 0 <= score <= 100 or score % 5):
                raise ValueError('Scores must be integer multiples of five in [0,100], or null')
        if not _number(view['confidence'], 0, 1) or not _text(view['notes']):
            raise ValueError('Each view needs confidence in [0,1] and evidence notes')
    if len(seen) != len(set(seen)) or set(seen) != set(expected):
        raise ValueError('Score must cover exactly the frozen view IDs')
    if 'connection_requirements' in packet:
        requirements = {r['id']:r for r in packet['connection_requirements']}
        checks = value['connection_checks']
        if not isinstance(checks,list) or len(checks) != len(requirements):
            raise ValueError('Every mandatory connection component needs a check')
        checked = []
        for check in checks:
            _keys(check, ('id','status','frame_ids','notes'), 'connection check')
            ident = check['id']; checked.append(ident)
            if ident not in requirements or check['status'] not in ('present','missing','unknown','not_applicable'):
                raise ValueError('Invalid connection check')
            ids = check['frame_ids']
            if (not isinstance(ids,list) or not all(isinstance(i,str) for i in ids) or
                    not ids or not set(ids) <= set(requirements[ident]['frame_ids']) or not _text(check['notes'])):
                raise ValueError('Connection check needs its own source views and evidence notes')
        if len(set(checked)) != len(requirements):raise ValueError('Duplicate connection checks')
    if not isinstance(value['uncertainties'], list) or any(not _text(s) for s in value['uncertainties']):
        raise ValueError('Uncertainties must be nonempty strings')
    if any(s is None for view in value['views'] for s in view['scores'].values()) and not value['uncertainties']:
        raise ValueError('Unknown scores require explicit uncertainties')
    objects = {obj['name'] if isinstance(obj, dict) else obj for obj in packet['objects']}
    if not isinstance(value['findings'], list):raise ValueError('Findings must be a list')
    findings = []
    for finding in value['findings']:
        if 'element' in finding:
            _keys(finding, ('dimension', 'severity', 'frame_id', 'bbox', 'element', 'observation', 'action'), 'visual finding')
            finding = {**{k: v for k, v in finding.items() if k != 'element'},
                       'objects': [], 'missing_element': finding['element']}
        findings.append(finding)
    value = {**value, 'findings': findings}
    for finding in value['findings']:
        _keys(finding, ('dimension', 'severity', 'frame_id', 'bbox', 'objects',
                        'missing_element', 'observation', 'action'), 'finding')
        if finding['dimension'] not in RUBRIC['dimensions'] or finding['severity'] not in ('critical', 'major', 'minor'):
            raise ValueError('Unknown finding dimension/severity')
        if finding['frame_id'] not in seen:raise ValueError('Finding refers to an unscored view')
        box = finding['bbox']
        if (not isinstance(box, list) or len(box) != 4 or any(not _number(n, 0, 1) for n in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('Finding bbox must be normalized x0,y0,x1,y1 with positive area')
        if (not isinstance(finding['objects'], list) or any(not isinstance(name, str) or name not in objects for name in finding['objects'])
                or len(finding['objects']) != len(set(finding['objects']))):
            raise ValueError('Finding objects must be unique exact packet object names')
        if not isinstance(finding['missing_element'], str) or (not finding['objects'] and not _text(finding['missing_element'])):
            raise ValueError('Unmapped findings must describe the missing/unmapped element')
        if not _text(finding['observation']) or not _text(finding['action']):
            raise ValueError('Finding needs an observation and a concrete action')
    low = {key for view in value['views'] for key, score in view['scores'].items()
           if score is not None and score < RUBRIC['thresholds']['dimension']}
    if not low <= {f['dimension'] for f in value['findings']}:
        raise ValueError(f"Every dimension below {RUBRIC['thresholds']['dimension']} needs a localized evidence finding")
    return value


def _aggregate(values):
    return .7 * mean(values) + .3 * min(values) if values else None


def summarize_region(value, packet):
    value = validate_result(value, packet)
    dimensions = {}
    for key, spec in RUBRIC['dimensions'].items():
        scores = [v['scores'][key] for v in value['views']]
        known = [s for s in scores if s is not None]
        dimensions[key] = {**spec, 'score': round(_aggregate([s or 0 for s in scores]) or 0, 2),
                           'observed_score': round(_aggregate(known), 2) if known else None,
                           'coverage': len(known) / len(scores) if scores else 0}
    score = round(sum(d['weight'] * d['score'] for d in dimensions.values()), 2)
    coverage = sum(d['weight'] * d['coverage'] for d in dimensions.values())
    known_weight = sum(d['weight'] for d in dimensions.values() if d['observed_score'] is not None)
    observed = (round(sum(d['weight'] * d['observed_score'] for d in dimensions.values()
                          if d['observed_score'] is not None) / known_weight, 2) if known_weight else None)
    confidence = min((v['confidence'] for v in value['views']), default=0)
    limits = RUBRIC['thresholds']
    evidence_ready = coverage >= 1 - 1e-9 and len(value['views']) >= limits['views'] and confidence >= limits['confidence']
    checks = value.get('connection_checks', [])
    evidence_ready = evidence_ready and not packet.get('evidence_blockers') and not any(c['status']=='unknown' for c in checks)
    passed = (evidence_ready and score >= limits['region'] and
              all(d['score'] >= limits['dimension'] for d in dimensions.values()) and
              not any(c['status']=='missing' for c in checks) and
              not any(f['severity'] == 'critical' for f in value['findings']))
    findings = []
    for finding in value['findings']:
        d = dimensions[finding['dimension']]
        multiplier = {'critical': 1.5, 'major': 1., 'minor': .5}[finding['severity']]
        # Evidence gaps block acceptance but must not masquerade as observed defects.
        deficit = 100 - d['observed_score'] if d['observed_score'] is not None else 0
        findings.append({**finding, 'candidate_objects': object_candidates(finding, packet),
                         'priority': round(d['weight'] * deficit * multiplier, 2)})
    findings.sort(key=lambda f: -f['priority'])
    source_views = {str(v['id']): v for v in packet['views']}
    views = []
    for view in value['views']:
        views.append({**view, 'card': source_views[view['id']].get('card', ''),
                      'width': source_views[view['id']].get('width'),
                      'height': source_views[view['id']].get('height'),
                      'score': round(sum(RUBRIC['dimensions'][k]['weight'] * (s or 0) for k, s in view['scores'].items()), 2)})
    return {'region': value['region'], 'name': packet.get('name', value['region']),
            'scene_sha256': packet.get('scene_sha256'), 'rubric_hash': rubric_hash(),
            'dimensions': dimensions, 'score': score, 'observed_score': observed,
            'coverage': round(coverage, 6), 'confidence': confidence, 'passed': passed,
            **({'connection_checks':checks,'evidence_blockers':packet.get('evidence_blockers',[])} if 'connection_requirements' in packet else {}),
            'status': 'passed' if passed else 'failed' if evidence_ready else 'needs_evidence',
            'evidence_requests': [{'frame_id': v['id'], 'unknown_dimensions': [k for k,s in v['scores'].items() if s is None],
                                   'low_confidence': v['confidence'] < limits['confidence']}
                                  for v in value['views'] if any(s is None for s in v['scores'].values()) or v['confidence'] < limits['confidence']],
            'views': views, 'findings': findings, 'uncertainties': value['uncertainties']}


def summarize(results, expected_regions=None):
    ids = [r['region'] for r in results]
    if len(ids) != len(set(ids)):raise ValueError('Duplicate regional scores')
    if expected_regions is not None and set(ids) != set(expected_regions):
        raise ValueError('Global score requires all expected regions')
    if len({r['scene_sha256'] for r in results}) > 1 or any(r['rubric_hash'] != rubric_hash() for r in results):
        raise ValueError('Global score cannot mix scenes or rubrics')
    score = round(mean(r['score'] for r in results), 2) if results else 0
    ready = bool(results) and all(r['status'] != 'needs_evidence' for r in results)
    passed = bool(results) and all(r['passed'] for r in results) and score >= RUBRIC['thresholds']['global']
    return {'rubric': RUBRIC, 'rubric_hash': rubric_hash(),
            'scene_sha256': results[0]['scene_sha256'] if results else None,
            'regions': ids, 'count': len(results), 'score': score,
            'coverage': round(mean(r['coverage'] for r in results), 6) if results else 0,
            'passed_regions': sum(r['passed'] for r in results), 'passed': passed,
            'status': 'passed' if passed else 'failed' if ready else 'needs_evidence',
            'weighting': 'equal regions; every region must pass; not area-weighted scene accuracy'}


def model_packet(packet):
    """Only evidence needed for visual judgment; object lookup is program-owned."""
    scope = packet.get('scope', {})
    if scope.get('core_boxes'):
        scope = {'core_boxes': [{k: b[k] for k in ('id', 'bounds_min', 'bounds_max', 'use') if k in b}
                                for b in scope['core_boxes']], 'scope': scope.get('scope', 'Union of assigned boxes.')}
    value = {'region': packet['region'], 'scope': scope,
             'views': [v['id'] for v in packet['views']]}
    for key in ('connection_requirements', 'evidence_blockers'):
        if key in packet:value[key] = packet[key]
    if packet.get('diagnostic_views'):
        value['diagnostic_views'] = [{'id': v['id'], 'role': v.get('role', 'Detail comparison')} for v in packet['diagnostic_views']]
    if packet.get('diagnostic_facts'):
        value['diagnostic_facts'] = [{'role': f['role'], 'value': {k: v for k, v in f['value'].items()
                                     if not k.endswith('_sha256')}} for f in packet['diagnostic_facts']]
    return value


def object_candidates(finding, packet):
    """Rank spatial candidates locally. AABB overlap is never an exact object binding."""
    a = finding['bbox']
    area = (a[2]-a[0])*(a[3]-a[1])
    ranked = []
    for obj in packet.get('objects', []):
        if not isinstance(obj, dict):continue
        b = obj.get('projected_bounds', {}).get(finding['frame_id'])
        if not b:continue
        overlap = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
        union = area + (b[2]-b[0])*(b[3]-b[1]) - overlap
        iou = overlap/union if union > 0 else 0
        if iou >= .1:ranked.append((iou, obj['name']))
    return [name for _, name in sorted(ranked, key=lambda row: (-row[0], row[1]))[:5]]


def prompt(packet):
    connection = ''
    if 'connection_requirements' in packet:
        connection = '\nFor EVERY connection_requirements ID, return connection_checks with status present, missing, unknown or not_applicable, supporting frame_ids and notes. Inspect intervening space, not only region cores. Present requires consistent modeled source-required structure; not_applicable needs source evidence. Missing or unknown blocks passage.\n'
    return f"""Independently assess {packet['region']} using only the attached source/render comparisons and supplied scope. Images follow views order, then diagnostic_views. Do not call tools or edit files. Return the required JSON in English.
Score visible assigned-region content and interfaces. Scope boxes form a union, not their enclosing box. Do not charge distant unrelated defects. Source-visible missing structures count even when absent from the model inventory. Off-screen elements are not omissions. An unusable or neighbor-dominated view is an evidence problem: record uncertainty, not an invented quality score.
Dimensions and weights: structure 30% (walls/floors/ceilings/openings/spaces), completeness 25% (visible components and omissions), alignment 20% (position/proportion), appearance 15% (materials, distinguishing lighting), connectivity 10% (visible continuity/interpenetration). Do not claim hidden topology, physical safety or measured depth accuracy; absolute scene scale is unverified.
For each frozen view, score every dimension 0-100 in steps of 5: 100 no discernible difference; 90 tiny; 80 minor; 70 obvious local; 50 major errors; 25 fragments; 0 absent/contradictory. Use null for unsupported judgments and explain uncertainties. Confidence measures evidence quality. Give concrete notes explaining low scores.
Return up to five actionable findings, covering every dimension scored below {RUBRIC['thresholds']['dimension']}. Each gives the visual element, dimension, severity, frame_id, bbox normalized within the reference panel, observation and repair action. Do not guess Blender object names; the program locates spatial candidates afterward. Critical means major spatial/passage/structural contradictions, not ordinary material defects.
The program aggregates scores and applies fixed gates: local {RUBRIC['thresholds']['region']}, every dimension {RUBRIC['thresholds']['dimension']}, confidence {RUBRIC['thresholds']['confidence']}, global {RUBRIC['thresholds']['global']}. Missing required evidence or critical defects blocks passage. Do not omit difficult views or tune scores to pass.""" + connection


def render_report(output, summary, regions):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    esc = lambda value: html.escape(str(value), quote=True)
    complete = summary.get('score') is not None
    headline = (f'Overall {summary["score"]:.1f} / 100' if complete else
                f'Assessment incomplete · Mean of scored regions {summary.get("partial_score", 0):.1f} / 100')
    labels = [spec['label'] for spec in RUBRIC['dimensions'].values()]
    rows = []
    details = []
    for region in regions:
        rows.append('<tr><td><a href="#' + esc(region['region']) + '">' + esc(region['region']) + '</a> ' + esc(region['name']) + '</td>' +
                    ''.join(f'<td>{d["score"]:.1f}</td>' for d in region['dimensions'].values()) +
                    f'<td><b>{region["score"]:.1f}</b></td><td>{esc(region["observed_score"])}</td><td>{region["coverage"]:.0%}</td><td>{esc(region["status"])}</td></tr>')
        cards = []
        for view in region['views']:
            card = view.get('card')
            if card:
                path = Path(card)
                href = os.path.relpath(path, output) if path.is_absolute() else str(path)
                width, height = view.get('width'), view.get('height')
                overlay = ''
                if width and height:
                    boxes = []
                    for number, f in enumerate(region['findings'], 1):
                        if f['frame_id'] != view['id']:continue
                        x0, y0, x1, y1 = f['bbox']
                        x, y = x0 * width, y0 * height + 64
                        boxes.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{(x1-x0)*width:.1f}" height="{(y1-y0)*height:.1f}" fill="none" stroke="#ff3c24" stroke-width="3"/><rect x="{x:.1f}" y="{y:.1f}" width="23" height="24" fill="#b51d0b"/><text x="{x+5:.1f}" y="{y+18:.1f}" fill="white" font-size="17" font-weight="bold">{number}</text>')
                    overlay = f'<svg aria-label="Numbered findings on reference image" viewBox="0 0 {width*2} {height+64}">{"".join(boxes)}</svg>'
                per_dimension = ' · '.join(f'{RUBRIC["dimensions"][key]["label"]} {score if score is not None else "Unknown"}' for key, score in view['scores'].items())
                cards.append(f'<figure><a class="card" href="{esc(href)}"><img loading="lazy" src="{esc(href)}">{overlay}</a><figcaption>Frame {esc(view["id"])} · {view["score"]:.1f} / 100<br>{esc(per_dimension)}<br>{esc(view["notes"])}</figcaption></figure>')
        findings = ''.join(f'<li><b>{esc(RUBRIC["dimensions"][f["dimension"]]["label"])} · {esc(f["severity"])} · Priority {f["priority"]:.1f}</b>: {esc(f["observation"])}<br>Frame {esc(f["frame_id"])}; box {esc(f["bbox"])}; objects {esc(", ".join(f["objects"]) or f["missing_element"])}<br>Action: {esc(f["action"])}</li>' for f in region['findings'])
        connection = ''
        if 'connection_checks' in region:
            rows_checked = ''.join(f'<tr><td>{esc(c["id"])}</td><td>{esc(c["status"])}</td><td>{esc(", ".join(c["frame_ids"]))}</td><td>{esc(c["notes"])}</td></tr>' for c in region['connection_checks'])
            connection = '<h3>Required connection components</h3><table><tr><th>Component</th><th>Status</th><th>Views</th><th>Evidence</th></tr>'+rows_checked+'</table>'
            if region.get('evidence_blockers'):
                connection += '<p>Missing evidence: '+esc('; '.join(region['evidence_blockers']))+'</p>'
        details.append(f'<section id="{esc(region["region"])}"><h2>{esc(region["region"])} · {esc(region["name"])} · {region["score"]:.1f}</h2><ol>{findings}</ol>{connection}<p>Numbered boxes appear only on the left reference. The right render is unchanged; boxes do not establish exact object correspondence.</p><p>Evidence limitations: {esc("; ".join(region["uncertainties"]) or "No additional limitations reported; only the fixed views are covered")}</p>{"".join(cards)}</section>')
    missing = summary.get('missing_regions', [])
    missing_note = '<p>Unscored regions: ' + esc(', '.join(missing)) + '. No global score yet.</p>' if missing else ''
    if summary.get('assessment_kind') == 'connection':
        headline = 'Connection gates · '+', '.join(f'{r["region"]} {r["score"]:.1f} / 100 ({r["status"]})' for r in regions)
        missing_note = '<p>Connection scores are separate gates, excluded from the global equal-region mean. Required component checks and post-score geometry checks still apply.</p>'
    perf = summary.get('performance', {})
    timing_note = ('<p>Measured overhead: evidence preparation ' + esc(perf.get('prepare_seconds')) + ' s; first assessment wall time ' +
                   esc(perf.get('run_wall_seconds')) + ' s; median assessor session ' + esc(perf.get('session_seconds_median')) +
                   ' s; assessor tool calls ' + esc(perf.get('tool_calls')) + '; latest request ' +
                   esc(perf.get('latest_run_seconds')) + ' s; cache hits ' + esc(perf.get('latest_cache_hits')) + ' regions.</p>') if perf else ''
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Regional scoring</title>
<style>body{{font:16px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#222}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%;display:block}}.card{{display:inline-block;position:relative}}.card svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}figure{{margin:20px 0}}li{{margin:12px 0}}section{{border-top:2px solid #ddd;margin-top:32px}}small{{color:#666}}</style>
<h1>Regional visual scoring · {headline}</h1><p>{summary['passed_regions']} / {summary.get('expected_regions',summary['count'])} regions meet the provisional gate; known-score coverage {summary['coverage']:.1%}; {esc(summary['status'])}.</p>{missing_note}
<p>Spatial structure 30% · Component completeness 25% · Position and proportion 20% · Material appearance 15% · Connectivity 10%. Each dimension = 70% view mean + 30% worst view; unknown contributes 0 to the conservative score and cannot pass. Region ≥{RUBRIC['thresholds']['region']}; each dimension ≥{RUBRIC['thresholds']['dimension']}; at least two views; confidence ≥0.7; no critical defects; equal-region global mean ≥{RUBRIC['thresholds']['global']} and all regions must pass.</p>
<p>Provisional visual criteria, not measured depth accuracy, physical safety or complete-scene acceptance. Confidence describes evidence quality, not calibrated probability. Equal region weights do not represent area-weighted scene accuracy.</p><small>Scene {esc(summary['scene_sha256'])}<br>Rubric {esc(summary['rubric_hash'])}</small>
<p>Observed scores exclude unknowns and diagnose visible defects; they cannot replace the conservative passing gate. Repair priority uses observed dimension scores. Obtain additional views or diagnostic renders for unknowns; missing evidence is not a geometry defect.</p>
{timing_note}
<table><thead><tr><th>Region</th>{''.join('<th>'+label+'</th>' for label in labels)}<th>Conservative total</th><th>Observed score</th><th>Known coverage</th><th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table>{''.join(details)}</html>'''
    path = output / 'report.html'
    path.write_text(body, encoding='utf-8')
    return path
