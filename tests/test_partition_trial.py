"""Spatial ownership and serial planning must preserve real seams and closure gates."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from partition_report import graph_analysis,build,connection_references
sys.path.insert(0,str(ROOT/'examples'))
from indoor_partition import assign, apply_spatial_review, SPECS, shared_face
from vibuild.cli import parser


def graph(names,pairs):
    return ([{'id':n} for n in names],
            [{'id':f'I{i}','regions':list(pair),'status':'observed_transition'} for i,pair in enumerate(pairs)])


def test_branch_requires_explicit_order_without_absorbing_regions():
    regions,edges=graph('abcdef',[('a','b'),('b','c'),('c','a'),('a','d'),('a','e'),('b','f')])
    result=graph_analysis(regions,edges,start='d')
    assert not result['cycles_alone_form_one_chain'] and len(result['branch_blocks'])==1
    assert result['execution_plan']['status']=='needs_plan'
    assert result['execution_plan']['chains']==[]


def test_disconnected_or_candidate_edges_do_not_become_fake_neighbors():
    regions,edges=graph('abcd',[('a','b'),('c','d')])
    edges.append({'id':'candidate','regions':['b','c'],'status':'candidate'})
    result=graph_analysis(regions,edges)
    assert result['connected_components']==2 and len(result['execution_plan']['chains'])==2
    assert result['observed_edges']==2
    chains=result['execution_plan']['chains']
    assert chains[0]['steps'][0]['depends_on']==[]
    assert chains[1]['steps'][0]['depends_on']==[chains[0]['steps'][-1]['id']]
    for chain in chains:
        assert chain['steps'][0]['connection_interfaces']==[]
        assert chain['steps'][0]['check_dependencies']==[]
    assert result['execution_plan']['closure_agent_count']==0


def test_ring_stays_separate_regions_with_one_extra_closure_agent():
    regions,edges=graph('abcd',[('a','b'),('b','c'),('c','a'),('c','d')])
    result=graph_analysis(regions,edges,start='a')
    assert result['cycles_alone_form_one_chain']
    plan=result['execution_plan'];steps=plan['chains'][0]['steps']
    assert plan['region_agent_count']==4 and plan['closure_agent_count']==1
    assert [s['id'] for s in steps]==['a','b','c','L01','d']
    assert steps[-1]['connection_interfaces']==['I3']
    assert steps[-1]['scene_after']=='L01'
    assert steps[-1]['check_dependencies']==['c']
    assert steps[3]['write_interfaces']==['I2']


def test_reviewed_execution_uses_one_global_scene_with_real_connection_checks():
    regions,edges=graph(['R01','R02','R03','R04','R05','R07','R08','R09','R10','R11'],[
        ('R01','R03'),('R01','R04'),('R02','R03'),('R02','R04'),('R02','R05'),
        ('R02','R11'),('R05','R08'),('R05','R10'),('R05','R07'),('R07','R09'),('R08','R09')])
    for e,n in zip(edges,[1,2,3,4,5,6,8,9,11,13,14]):e['id']=f'I{n:02d}'
    plan=graph_analysis(regions,edges,start='R11')['execution_plan']
    assert plan['status']=='planned' and plan['controller_implemented']
    steps=plan['chains'][0]['steps'];lookup={s['id']:s for s in steps}
    assert [s['id'] for s in steps]==['R11','R02','R03','R01','R04','L01','R05','R08','R09','R07','L02','R10']
    assert plan['region_agent_count']==10 and plan['closure_agent_count']==2 and plan['execution_agent_count']==12
    assert plan['max_active_modeling_agents']==1
    assert plan['scene']['mode']=='single_global_project'
    assert plan['scene']['initial_geometry']=='empty'
    assert plan['scene']['self_contained']
    assert sorted(s['id'] for s in steps if s['kind']=='region')==sorted(r['id'] for r in regions)
    prior=set()
    for i,s in enumerate(steps):
        assert s['depends_on']==([steps[i-1]['id']] if i else [])
        assert set(s['depends_on']+s['check_dependencies'])<=prior
        assert s['scene_after']==(steps[i-1]['id'] if i else None)
        assert 'input_interfaces' not in s and 'export_interfaces' not in s
        if s['kind']=='region':
            assert len(s['connection_interfaces'])==(1 if i else 0)
            assert s['result']==('locally_passed_closure_pending' if s['cycle'] else 'accepted')
        prior.add(s['id'])
    assert lookup['L01']['write_interfaces']==['I04'] and set(lookup['L01']['regions'])=={'R02','R04'}
    assert lookup['L02']['write_interfaces']==['I11'] and set(lookup['L02']['regions'])=={'R05','R07'}
    assert set(lookup['L01']['check_interfaces'])=={'I01','I02','I03','I04','I06'}
    assert set(lookup['L02']['check_interfaces'])=={'I08','I14','I13','I11','I05'}
    for region,after,neighbor,interface in [('R05','L01','R02','I05'),('R10','L02','R05','I09')]:
        assert lookup[region]['scene_after']==after
        assert lookup[region]['connection_interfaces']==[interface]
        assert lookup[region]['check_dependencies']==[neighbor]
    # Every real seam has exactly one construction responsibility; closure gates
    # additionally recheck already built seams instead of silently dropping them.
    built=[p for s in steps if s['kind']=='region' for p in s['connection_interfaces']]
    built += [e for s in steps if s['kind']=='closure' for e in s['write_interfaces']]
    assert sorted(built)==sorted(e['id'] for e in edges)
    assert plan['global_audit']['depends_on']==['L01','L02','R10','R11']
    assert plan['global_audit']['accepted_region_stages']['R02']=='L01'
    assert plan['global_audit']['accepted_region_stages']['R05']=='L02'
    # Ordering must be independent of JSON list order.
    assert graph_analysis(list(reversed(regions)),list(reversed(edges)),start='R11')['execution_plan']==plan


def test_connection_references_preserve_time_offsets_and_exclude_audit():
    data={'frames':[
        {'name':'before','video':'v','time':1,'split':'model'},
        {'name':'after','video':'v','time':3,'split':'model'},
        {'name':'audit','video':'v','time':2,'evaluation_split':'audit'},
        {'name':'holdout','video':'v','time':2.1,'split':'holdout'},
        {'name':'other','video':'w','time':2,'split':'model'}],
        'interfaces':[{'id':'I1','evidence':[{'video':'v','time_range':[1.2,2.1]}]},
                      {'id':'I2','evidence':[{'video':'missing','time_range':[0,1]}]}]}
    connection_references(data)
    assert data['interfaces'][0]['reference_frames']==[
        {'frame':'before','crossing_time':1.2,'time_offset_seconds':-0.2},
        {'frame':'after','crossing_time':2.1,'time_offset_seconds':0.9}]
    assert data['interfaces'][1]['reference_frames']==[]


def test_non_simple_cycle_or_internal_chain_start_requires_planning():
    regions,edges=graph('abcd',[('a','b'),('b','c'),('c','a'),('a','d'),('d','c')])
    assert graph_analysis(regions,edges)['execution_plan']['status']=='needs_plan'
    regions,edges=graph('abc',[('a','b'),('b','c')])
    plan=graph_analysis(regions,edges,start='b')['execution_plan']
    assert plan['status']=='needs_plan' and plan['chains']==[]


def test_half_open_boxes_have_unique_3d_ownership():
    regions=[{'bounds_min':[0,0,-2],'bounds_max':[2,2,0]},
             {'bounds_min':[0,0,0],'bounds_max':[2,2,2]}]
    np.testing.assert_array_equal(assign(np.array([[1,1,-1],[1,1,0],[1,1,1],[2,1,1]]),regions),[0,1,1,-1])


def test_reviewed_box_union_keeps_lobbies_separate_and_preserves_real_loop():
    regions=[{'id':f'R{i+1:02d}','name':spec[0],'bounds_min':spec[1],'bounds_max':spec[2],
              'color':spec[3],'point_count':1,'robust_scope_point_count':1,'camera_count':1,
              'camera_center_frame_count':1,'evidence':[],'uncertainties':[]} for i,spec in enumerate(SPECS)]
    lookup={r['id']:r for r in regions}
    pairs=[('I07','R05','R06'),('I08','R05','R08'),('I11','R06','R07'),
           ('I12','R06','R08'),('I13','R07','R09'),('I14','R08','R09'),('I10','R05','R16')]
    interfaces=[]
    for ident,a,b in pairs:
        axis,lo,hi=shared_face(lookup[a],lookup[b])
        interfaces.append({'id':ident,'regions':[a,b],'status':'observed_transition',
            'bounds_min':lo.tolist(),'bounds_max':hi.tolist(),'plane':{'axis':int(axis),'coordinate':float(lo[axis])},
            'evidence':[{'video':'v','time_range':[1,2]}],'crossing_count':1,
            'patches':[{'id':'P01','bounds_min':lo.tolist(),'bounds_max':hi.tolist(),'evidence_indices':[0]}]})
    cameras=[{'name':r['id'],'video':'v','time':i,'core_region':r['id'],
              'center':((np.array(r['bounds_min'])+r['bounds_max'])/2).tolist()} for i,r in enumerate(regions)]
    cameras[1]['time']=20  # An active camera after an excluded exterior excursion.
    frames=[{'name':r['id'],'core_region':r['id'],'registered_sparse_observations':1,
             'visible_regions':[{'region':r['id'],'observed_sparse_points':1}]} for r in regions]
    data={'regions':regions,'interfaces':interfaces,'cameras':cameras,'frames':frames,
          'point_cloud':{'sampling':'fixture','positions':[c['center'] for c in cameras],
                         'colors':[[0,0,0]]*16,'region_ids':[r['id'] for r in regions]},
          'validation':{'camera_count':16,'reference_frame_count':16,'assigned_raw_points':16,'robust_scope_points':16}}
    result=apply_spatial_review(data)
    active=result['regions'];ids=[r['id'] for r in active]
    assert len(active)==10 and 'R06' not in ids and not set(ids)&{f'R{i}' for i in range(12,17)}
    centers=np.array([c['center'] for c in cameras if c['name'] in ('R05','R06','R07','R08')])
    assert [ids[i] for i in assign(centers,active)]==['R05','R05','R07','R08']
    edges={frozenset(e['regions']):e for e in result['interfaces']}
    assert frozenset(['R01','R02']) not in edges
    for a,b in zip(['R05','R08','R09','R07'],['R08','R09','R07','R05']):
        assert frozenset([a,b]) in edges
    assert edges[frozenset(['R07','R09'])]['id']=='I13'
    combined=edges[frozenset(['R05','R08'])]
    assert len(combined['patches'])==2 and combined['plane'] is None
    assert [p['evidence_indices'] for p in combined['patches']]==[[0],[1]]
    assert result['validation']['camera_count']==11 and result['validation']['excluded_scope_cameras']==5
    camera_map={c['name']:c for c in result['cameras']}
    assert camera_map['R01']['segment']!=camera_map['R02']['segment']
    assert result['validation']['active_core_box_count']==11
    assert set(result['point_cloud']['region_ids'])<=set(ids)


def test_report_is_self_contained_and_escapes_embedded_data(tmp_path):
    regions,edges=graph(['R01'],[])
    regions[0].update(name='</script><script>bad()</script>',bounds_min=[0,0,0],bounds_max=[1,1,1],color='#123456')
    source=tmp_path/'partition.json';source.write_text(json.dumps({'regions':regions,'interfaces':edges,'frames':[]}))
    output=tmp_path/'review.html';build(source,output)
    html=output.read_text()
    assert '</script><script>bad()' not in html and '\\u003c/script' in html
    assert 'src="https:' not in html and 'src="http:' not in html
    assert '__PARTITION_DATA__' not in html


@pytest.mark.parametrize('command',['start','workflow','workflow-task'])
def test_retired_orchestration_has_no_cli_alias(command):
    with pytest.raises(SystemExit):parser().parse_args([command,'--record','record'])


def test_partition_entry_is_explicit_and_score_remains_a_tool():
    args=parser().parse_args(['report','--partition','partition.json','--output','work/review.html'])
    assert args.cmd=='report'
    assert parser().parse_args(['tool','score','report','--output','work/scores']).name=='score'
