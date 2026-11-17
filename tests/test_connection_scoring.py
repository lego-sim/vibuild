"""Connection scoring must preserve all views and gate explicit spatial coverage."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
import score
from connections import COMPONENTS
from vibuild.scoring import RUBRIC, rubric_hash, summarize_region, validate_result


@pytest.fixture
def fixture(tmp_path):
    source=tmp_path/'record/resources';source.mkdir(parents=True)
    frames=[];evidence=[]
    for n in range(8):
        rgb=source/f'{n}.png';Image.new('RGB',(32,20),'white').save(rgb)
        frame={'name':f'f{n}','index':n,'video':'v','time':float(n),
               'center':[float(n),0,0],'rotation':[[1,0,0],[0,1,0],[0,0,1]],
               'K':[[20,0,16],[0,20,10],[0,0,1]],'width':32,'height':20,
               'original_rgb':str(rgb),'original_sha256':score.digest(rgb),'evaluation_split':'model','split':'model'}
        frames.append(frame)
        evidence.append({'interface':'I01' if n<4 else 'I02','source_frame':frame['name'],
                         'view_id':f'J01_{n:02d}','frame':copy.deepcopy(frame),'primary':True,
                         'role':['approach_from','crossing','approach_to','context'][n%4]})
    score.save(source/'frames.json',frames);score.save(source/'coordinates.json',{})
    score.save(source/'index.json',{'frames':'frames.json','coordinates':'coordinates.json'})
    partition={'schema_version':'partition_v1',
               'regions':[{'id':r} for r in ('R01','R02','R03')],
               'interfaces':[{'id':i,'regions':rs,'status':'observed_transition'}
                             for i,rs in [('I01',['R01','R02']),('I02',['R02','R03'])]],
               'provenance':{'source_sha256':{n:score.digest(source/n)
                                             for n in ['frames.json','coordinates.json']}},
               'validation':{}}
    partition_path=tmp_path/'partition.json';score.save(partition_path,partition)
    manifest={'schema_version':'connection_task_v1','id':'J01','regions':['R01','R02','R03'],
              'interfaces':['I01','I02'],'status':'prepared','evidence':evidence,
              'coverage':[{'interface':i,'missing_roles':[]} for i in ['I01','I02']],
              'envelope':{'boxes':[{'bounds_min':[-1,-1,-1],'bounds_max':[9,1,1],
                                   'regions':['R01','R02','R03'],'interfaces':['I01','I02']}]},
              'partition_sha256':score.digest(partition_path)}
    path=tmp_path/'connection.json';score.save(path,manifest)
    scene=tmp_path/'scene.blend';scene.write_bytes(b'BLENDERfixture')
    config=tmp_path/'config.json';score.save(config,{})
    args=SimpleNamespace(record=str(source.parent),output=tmp_path/'score',scene=scene,
                         partition=partition_path,config=config,connection=path,region=['J01'],
                         frame=None,views=3,width=32,samples=1,threads=1,workers=2)
    return SimpleNamespace(source=source,raw={f['name']:f for f in frames},partition=partition,
                           manifest=manifest,path=path,args=args)


def spec(fixture):
    score.save(fixture.path,fixture.manifest)
    return score.connection_spec(fixture.path,fixture.args.partition,fixture.partition,
                                 fixture.raw,fixture.source,32)


def packet_and_result(fixture):
    packet={**spec(fixture),'objects':[],'rubric_hash':rubric_hash(),'scene_sha256':'scene'}
    result={'region':'J01','views':[{'id':v['id'],'scores':{k:95 for k in RUBRIC['dimensions']},
                                  'confidence':.95,'notes':'Source-required structure is visible.'}
                                 for v in packet['views']],
            'findings':[],'uncertainties':[],
            'connection_checks':[{'id':r['id'],'status':'present','frame_ids':r['frame_ids'],
                                  'notes':'Observed consistently in the assigned views.'}
                                 for r in packet['connection_requirements']]}
    return packet,result


def test_prepare_keeps_every_connection_view_primary_and_freezes_requirements(fixture,monkeypatch):
    rendered=[]
    def run_tool(args,cfg,log):
        out=Path(args[args.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
        if args[0]=='inspect':
            score.save(out/'inventory.json',{'source_sha256':score.digest(fixture.args.scene),'objects':[]})
        else:
            views=score.load(Path(args[args.index('--manifest')+1]))['frames']
            for view in views:
                rendered.append(view['id']);Image.new('RGB',(view['width'],view['height'])).save(out/f'{view["id"]}.png')
    monkeypatch.setattr(score,'run_tool',run_tool)
    monkeypatch.setattr(score.resources,'record_path',lambda name:fixture.source.parent)
    score.prepare(fixture.args)
    protocol=score.load(fixture.args.output/'.runtime/protocol.json')
    work=fixture.args.output/'J01';packet=score.verified_packet(work,protocol)
    expected={e['view_id'] for e in fixture.manifest['evidence']}
    assert len(expected)==8 and {v['id'] for v in packet['views']}==expected
    assert set(rendered)==expected
    assert protocol['assessment_kind']=='connection'
    assert len(packet['connection_requirements'])==2*len(COMPONENTS)
    assert protocol['source_hashes'][str(fixture.path)]==score.digest(fixture.path)
    assert packet['evidence_blockers']==[]
    packet['connection_requirements'].pop();score.save(work/'packet.json',packet)
    with pytest.raises(ValueError,match='requirements differ'):score.verified_packet(work,protocol)


@pytest.mark.parametrize('status,expected',[('present','passed'),('missing','failed'),('unknown','needs_evidence')])
def test_high_score_cannot_hide_missing_or_unknown_connection(fixture,status,expected):
    packet,result=packet_and_result(fixture)
    result['connection_checks'][0]['status']=status
    summary=summarize_region(result,packet)
    assert summary['score']==95 and summary['status']==expected
    assert summary['passed']==(status=='present')


def test_coverage_blocker_prevents_high_score_acceptance(fixture):
    fixture.manifest['coverage'][0]['missing_roles']=['approach_to']
    fixture.manifest['evidence']=[e for e in fixture.manifest['evidence']
                                if not (e['interface']=='I01' and e['role']=='approach_to')]
    packet,result=packet_and_result(fixture)
    assert packet['evidence_blockers']
    assert summarize_region(result,packet)['status']=='needs_evidence'


def test_report_explains_connection_failure_without_a_low_average(fixture,tmp_path):
    from vibuild.scoring import summarize, render_report
    packet,result=packet_and_result(fixture)
    result['connection_checks'][0].update(status='missing',notes='The connecting floor is absent.')
    region=summarize_region(result,packet)
    summary=summarize([region],expected_regions=['J01'])
    summary['assessment_kind']='connection'
    report=render_report(tmp_path/'report',summary,[region]).read_text()
    assert 'Connection gates' in report and 'The connecting floor is absent.' in report
    assert 'Required connection components' in report and 'excluded from the global' in report


@pytest.mark.parametrize('defect',['missing_checks','omitted_component','duplicate_component','wrong_interface_view','missing_view','wrong_view'])
def test_omitted_or_misattributed_evidence_is_rejected(fixture,defect):
    packet,result=packet_and_result(fixture)
    if defect=='missing_checks':result.pop('connection_checks')
    elif defect=='omitted_component':result['connection_checks'].pop()
    elif defect=='duplicate_component':result['connection_checks'][-1]=copy.deepcopy(result['connection_checks'][0])
    elif defect=='wrong_interface_view':result['connection_checks'][0]['frame_ids']=['J01_07']
    elif defect=='missing_view':result['views'].pop()
    else:result['views'][-1]['id']='unassigned'
    with pytest.raises(ValueError):validate_result(result,packet)


def test_connection_schema_does_not_change_standard_region_contract(fixture):
    packet,result=packet_and_result(fixture)
    schema=score.result_schema(packet)
    assert 'connection_checks' in schema['required']
    assert schema['properties']['connection_checks']['items']['additionalProperties'] is False
    assert set(schema['properties']['connection_checks']['items']['properties'])=={'id','status','frame_ids','notes'}
    validate_result(result,packet)
    without=copy.deepcopy(result);without.pop('connection_checks')
    with pytest.raises(ValueError):validate_result(without,packet)
    standard=score.result_schema({'region':'R01'})
    assert standard==score.load(score.ROOT/'schemas/score.schema.json')
    assert 'connection_checks' not in standard['properties']


@pytest.mark.parametrize('key,value',[('center',[999,0,0]),('K',[[1,0,0],[0,1,0],[0,0,1]]),('width',99),('original_rgb','other.png')])
def test_tampered_connection_calibration_is_rejected(fixture,key,value):
    fixture.manifest['evidence'][0]['frame'][key]=value
    with pytest.raises(ValueError,match='calibration'):spec(fixture)


@pytest.mark.parametrize('field,value',[('split','audit'),('evaluation_split','holdout')])
def test_source_holdout_cannot_be_reintroduced_by_manifest(fixture,field,value):
    fixture.raw['f0'][field]=value
    with pytest.raises(ValueError,match='non-audit'):spec(fixture)


def test_partition_excluded_holdout_cannot_be_reintroduced(fixture):
    fixture.partition['validation']['excluded_audit_holdout_frames']=['f0']
    with pytest.raises(ValueError,match='non-audit|holdout|excluded'):spec(fixture)


def test_missing_interface_coverage_cannot_silently_pass(fixture):
    fixture.manifest['coverage']=fixture.manifest['coverage'][:1]
    with pytest.raises(ValueError,match='coverage|Coverage'):spec(fixture)


def test_removed_crossing_view_cannot_keep_prepared_coverage_claim(fixture):
    fixture.manifest['evidence']=[e for e in fixture.manifest['evidence']
                                if not (e['interface']=='I01' and e['role']=='crossing')]
    with pytest.raises(ValueError,match='coverage'):spec(fixture)


def test_connection_report_uses_local_gate_not_global_80(fixture,monkeypatch):
    packet,result=packet_and_result(fixture)
    for view in result['views']:
        view['scores']={key:75 for key in RUBRIC['dimensions']}
    region=summarize_region(result,packet)
    assert region['passed']
    out=fixture.args.output
    score.save(out/'.runtime/protocol.json',{'assessment_kind':'connection',
        'regions':[{'region':'J01'}],'all_regions':['R01','R02','R03']})
    score.save(out/'J01/.runtime/scored.json',{})
    monkeypatch.setattr(score,'verified_score',lambda work,protocol:region)
    score.report(out)
    summary=score.load(out/'scores.json')
    assert summary['score']==75 and summary['passed'] and summary['status']=='passed'
    assert summary['missing_regions']==[] and summary['global_acceptance']=='not_evaluated'


def test_changed_original_cannot_silently_establish_new_reference(fixture):
    Path(fixture.manifest['evidence'][0]['frame']['original_rgb']).write_bytes(b'changed reference')
    with pytest.raises(ValueError,match='original image changed'):spec(fixture)
