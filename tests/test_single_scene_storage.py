"""Latest scene replacement, failure recovery, and independent-assessment exclusion."""
import json
from pathlib import Path
import shutil
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
import iteration
from scene_storage import scene_lock, active_assessments
from vibuild.layout import initialize_task
from vibuild.resources import digest
from test_scene import fixture_scene


def setup_task(tmp_path):
    root=tmp_path/'modeling_20261110_170000';initialize_task(root, ['R11', 'R02', 'R05'])
    (root/'scene.blend').write_bytes(b'BLENDERoriginal')
    script=tmp_path/'edit.py';script.write_text('')
    views=tmp_path/'views.json'
    views.write_text(json.dumps({'frames':[{'id':'1','width':32,'height':16,
        'K':[[20,0,16],[0,20,8],[0,0,1]],'camera_to_world':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}))
    return root,script,views


def arguments(root,script,views,number=1):
    return ['--baseline',str(root/'scene.blend'),'--output',str(root/'R05/iterations'/f'{number:03d}'),
            '--script',str(script),'--views',str(views),'--region','R05','--what','Repair width',
            '--why','Source-supported continuity','--samples','1','--width','32']


def fake_scene(fail=None):
    def run(args):
        action=args[0];out=Path(args[args.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
        if fail==action or fail==out.name:return 1
        if action=='patch':
            (out/'candidate.blend').write_bytes(b'BLENDERnew')
            (out/'patch.json').write_text('{}')
        elif action=='render':
            Image.new('RGB',(32,16),'red' if out.name=='after' else 'black').save(out/'1.png')
            (out/'render_manifest.json').write_text(json.dumps({'camera_signature':'fixed','scene_unit_scale':1,'request_seconds':.1}))
        return 0
    return run


def test_iteration_updates_only_root_and_keeps_comparison_receipts(tmp_path,monkeypatch):
    root,script,views=setup_task(tmp_path);before=digest(root/'scene.blend')
    monkeypatch.setattr(iteration.scene,'main',fake_scene())
    assert iteration.main(arguments(root,script,views))==0
    out=root/'R05/iterations/001';receipt=json.loads((out/'.runtime/iteration.json').read_text())
    assert list(root.rglob('*.blend'))==[root/'scene.blend']
    assert receipt['baseline_sha256']==before and receipt['candidate_sha256']==digest(root/'scene.blend')
    assert receipt['candidate']==str(root/'scene.blend')
    assert '../../../scene.blend' in (out/'review.html').read_text()
    assert json.loads((root/'status.json').read_text())['current_candidate']['assessment']=='pending'


@pytest.mark.parametrize('failure,expected',[('patch',b'BLENDERoriginal'),('after',b'BLENDERnew')])
def test_failed_build_preserves_old_scene_failed_preview_preserves_latest_saved_work(tmp_path,monkeypatch,failure,expected):
    root,script,views=setup_task(tmp_path)
    monkeypatch.setattr(iteration.scene,'main',fake_scene(failure))
    with pytest.raises(RuntimeError):iteration.main(arguments(root,script,views))
    assert (root/'scene.blend').read_bytes()==expected
    assert list(root.rglob('*.blend'))==[root/'scene.blend']


def test_exported_assessment_blocks_edit_until_recorded_or_cancelled(tmp_path,monkeypatch):
    root,script,views=setup_task(tmp_path)
    out=root/'R05/scores/001';(out/'.runtime').mkdir(parents=True)
    (out/'.runtime/protocol.json').write_text(json.dumps({'scene':str(root/'scene.blend')}))
    rt=out/'R05/.runtime';rt.mkdir(parents=True)
    handoff=rt/'handoff.json';handoff.write_text(json.dumps({'state':'prepared'}))
    monkeypatch.setattr(iteration.scene,'main',lambda args:pytest.fail('Edit started while assessment was frozen'))
    with pytest.raises(ValueError,match='frozen'):iteration.main(arguments(root,script,views))
    handoff.write_text(json.dumps({'state':'cancelled'}))
    assert not list(active_assessments(root))
    with scene_lock(root):
        with pytest.raises(ValueError,match='busy'):
            with scene_lock(root,writing=True):pass


@pytest.mark.parametrize('state', ['prepared', 'running', 'unknown'])
def test_stale_score_cannot_unfreeze_pending_handoff(tmp_path,state):
    root,_,_=setup_task(tmp_path)
    out=root/'R05/scores/001';(out/'.runtime').mkdir(parents=True)
    (out/'.runtime/protocol.json').write_text(json.dumps({'scene':str(root/'scene.blend')}))
    rt=out/'R05/.runtime';rt.mkdir(parents=True)
    (rt/'handoff.json').write_text(json.dumps({'state':state}))
    (rt/'scored.json').write_text('{"packet_sha256":"old"}')
    with pytest.raises(ValueError,match='frozen'):
        with scene_lock(root,writing=True):pass
    for terminal in ('completed','cancelled'):
        (rt/'handoff.json').write_text(json.dumps({'state':terminal}))
        with scene_lock(root,writing=True):pass


def test_independent_judges_share_scene_and_preserve_cached_results(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from types import SimpleNamespace
    from test_score_tool import score, scored_fixture
    from vibuild.scoring import RUBRIC

    root,_,_=setup_task(tmp_path)
    out,_=scored_fixture(root,completed=())
    ready=Barrier(2);calls=[]
    def run(work,text,config,**kwargs):
        calls.append(work.name)
        ready.wait(timeout=5)  # Both model calls must hold the shared root concurrently.
        with pytest.raises(ValueError,match='busy'):
            with scene_lock(root,writing=True):pass
        with pytest.raises(ValueError,match='busy'):
            score.judge(SimpleNamespace(output=out,region=[work.name]))
        packet=score.load(work/'packet.json')
        result={'region':work.name,'views':[
            {'id':v['id'],'scores':{d:90 for d in RUBRIC['dimensions']},
             'confidence':.9,'notes':'Visible geometry matches.'} for v in packet['views']],
            'findings':[],'uncertainties':[]}
        score.save(work/'.runtime/final.json',result)
        (work/'.runtime/events.jsonl').write_text('')
        return {'state':'completed','started':1,'finished':2}
    monkeypatch.setattr(score.sessions,'run',run)
    args=[SimpleNamespace(output=out,region=[r],model='test',reasoning_effort='high',config=None)
          for r in ('S01','S02')]
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(score.judge,args))
    assert sorted(calls)==['S01','S02']
    for arg in args:
        (out/arg.region[0]/'score.json').unlink()
        score.judge(arg)
        assert score.load(out/arg.region[0]/'score.json')['score']==90
    assert len(calls)==2
    with scene_lock(root,writing=True):pass


def test_real_blender_single_scene_iteration(tmp_path,fixture_scene):
    blender,blend,views=fixture_scene
    root,script,_=setup_task(tmp_path);shutil.copyfile(blend,root/'scene.blend')
    script.write_text("import bpy\nbpy.data.objects['wall'].location.z += 0.1\n")
    assert iteration.main([*arguments(root,script,views),'--blender',blender])==0
    assert list(root.rglob('*.blend'))==[root/'scene.blend']
    assert not list(root.rglob('*.blend1'))
