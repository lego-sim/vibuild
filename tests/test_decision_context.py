import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).parents[1]/'tools'), str(Path(__file__).parents[1]/'src')]
from decision_context import material_index
from test_execution_jobs import diagnostic_job
import jobs
import task_service


def test_index_has_exact_paths_bounded_findings_and_reusable_job(tmp_path):
    root, manifest, config, _, score_out = diagnostic_job(tmp_path)
    jobs.write(root/'status.json', {'active_region': 'R07'})
    jobs.write(config, {'runtime': {'analysis_python': '/configured/python'}})
    queued = jobs.submit(root, manifest)
    directory = Path(queued['receipt']).parent
    request = jobs.read(directory/'request.json', {})
    # Completed producer stages; the next decision must retain their settings.
    jobs.write(directory/'result.json', {'state': 'completed', 'steps': [{'state': 'completed'}]*4,
                                       'current_scene_sha256': jobs.digest(root/'scene.blend')})
    findings = [{'priority': n, 'objects': [f'wall{n}'], 'observation': 'Visible defect'} for n in range(20)]
    jobs.write(score_out/'R07/score.json', {'score': 50, 'scene_sha256': 'old-scene', 'findings': findings})
    jobs.write(score_out/'R07/packet.json', {'scene_sha256': 'old-scene', 'views': [{'id': '73', 'card': 'card.jpg',
        'K': ['large omitted calibration']*1000}], 'objects': [{'name': f'wall{n}', 'bounds': [0,0,0,1,1,1],
        'projected_bounds': ['large omitted projection']*1000} for n in range(20)]})
    jobs.write(root/'R07/region.json', {'region': {'id': 'R07', 'boxes': [[0,0,0,1,1,1]]},
                                      'frames': ['omitted source catalogue']*10000})
    (score_out/'.runtime').mkdir()
    jobs.write(score_out/'.runtime/protocol.json', {'regions': [{'region': 'R07',
        'views': [{'id': 'first', 'reference_rgb': '/fixed/first.jpg'},
                  {'id': 'second', 'reference_rgb': '/fixed/second.jpg'}]}]})
    output = root/'.runtime/decisions/new'
    index = material_index(root, config, output)
    assert index['analysis_python'] == '/configured/python'
    assert index['score']['matches_current_scene'] is False
    assert index['score']['remaining_findings'] == 17
    assert [f['priority'] for f in index['score']['findings']] == [19,18,17]
    assert len(index['evidence']['affected_objects']) == 3
    assert 'omitted' not in json.dumps(index)
    assert len(json.dumps(index)) < 6000
    template = jobs.read(Path(index['job_template']), {})
    assert template['round'] == 2
    assert 'expected_scene_sha256' not in template
    assert 'sha256' not in json.dumps(index)
    assert jobs.read(output/'.runtime/binding.json', {})['scene_sha256'] == jobs.digest(root/'scene.blend')
    assert [s['args'][0] for s in template['steps']] == ['iterate','prepare','run']
    assert jobs.option(template['steps'][0]['args'], '--samples') == '16'
    assert jobs.option(template['steps'][0]['args'], '--script') == str(output/'edit.py')
    (output/'edit.py').write_text('# model-authored edit')
    jobs.validate(root, template)
    assert [v['id'] for v in jobs.read(output/'views.json', {})['frames']] == ['first', 'second']
    args = template['steps'][1]['args']
    assert [args[i+1] for i, a in enumerate(args[:-1]) if a == '--frame'] == ['first', 'second']
    prompt = task_service.decision_prompt(root, [], config, output)
    assert prompt.count('# Agent contract') == 1
    assert 'MATERIAL_INDEX=' in prompt and 'omitted' not in prompt.split('MATERIAL_INDEX=')[1]


def test_index_does_not_treat_cancelled_recipe_as_executed(tmp_path):
    root, manifest, config, _, _ = diagnostic_job(tmp_path)
    jobs.write(root/'status.json', {'active_region': 'R07'})
    queued = jobs.submit(root, manifest)
    jobs.write(Path(queued['receipt']), {'state': 'cancelled'})
    index = material_index(root, config, root/'.runtime/decisions/new')
    assert 'edit' not in index and 'job_template' not in index


def test_attaches_only_current_images(tmp_path):
    from decision_context import decision_images
    card = tmp_path/'card.jpg';card.write_bytes(b'image')
    index = {'evidence': {'matches_current_scene': False,
             'views': [{'card': str(card)}]}, 'edit': {'matches_current_scene': False, 'comparisons': [str(card)]}}
    assert decision_images(index) == []
    index['edit']['matches_current_scene'] = True
    assert decision_images(index) == [str(card)]
    index['evidence']['matches_current_scene'] = True
    assert decision_images(index) == [str(card)]


def test_first_connection_recipe_retains_three_previews_and_all_scoring_views(tmp_path):
    root, manifest, config, _, _ = diagnostic_job(tmp_path)
    jobs.write(root/'status.json', {'active_region': 'R07'})
    recipe=jobs.read(manifest,{})
    build=recipe['steps'][0];prepare=recipe['steps'][1]
    preview=tmp_path/'preview.json';jobs.write(preview,{'frames':[{'id':'a'},{'id':'b'},{'id':'c'}]})
    build['args'] += ['--views',str(preview)]
    connection=tmp_path/'connection.json';jobs.write(connection,{'views':list('abcdefg')})
    prepare['args'] += ['--connection',str(connection)]
    jobs.write(root/'R07/job_template.json',{'steps':[build,prepare]})
    directory=root/'.runtime/decisions/first'
    index=material_index(root,config,directory)
    template=jobs.read(Path(index['job_template']),{})
    assert jobs.read(directory/'views.json',{})['frames']==[{'id':'a'},{'id':'b'},{'id':'c'}]
    assert jobs.option(template['steps'][1]['args'],'--connection')==str(connection)
    assert '--frame' not in template['steps'][1]['args']


def test_inspection_seed_creates_score_only_job(tmp_path):
    root, manifest, config, _, _ = diagnostic_job(tmp_path)
    jobs.write(root/'status.json', {'active_region': 'R07'})
    recipe=jobs.read(manifest,{})
    prepare=next(s for s in recipe['steps'] if s['tool']=='score' and s['args'][0]=='prepare')
    jobs.write(root/'R07/job_template.json',{'steps':[prepare]})
    directory=root/'.runtime/decisions/check'
    index=material_index(root,config,directory)
    template=jobs.read(Path(index['job_template']),{})
    assert [s['args'][0] for s in template['steps']]==['prepare','run']
    assert jobs.option(template['steps'][0]['args'],'--scene')==str(root/'scene.blend')
    assert not (directory/'edit.py').exists()
    jobs.validate(root,template)
