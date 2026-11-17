"""Candidate selection tolerances must not turn incomplete evidence into acceptance."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from comparison import compare_reports


def report(coverage=1., error=.18):
    return {'protocol_sha256':'fixed', 'structure_status':'blocked', 'structure_accepted':False,
        'frames':[{'id':'313', 'status':'failed','limits':{'depth_coverage':.95},'components':{
            'depth':{'status':'failed','coverage':coverage,'abs_rel':error},
            'pane':{'status':'failed','semantic_iou':0.,'precision':0.,'recall':0.,'count_relative_error':1.}}}],
        'plausibility':{'status':'blocked','score_0_100':25.},
        'appearance':{'status':'scored','frames':[{'id':'313','status':'scored','score_0_100':78.}]}}


def test_small_coverage_loss_with_depth_gain_selects_candidate():
    a=report();b=report(.996765,.12684)
    b['plausibility']['score_0_100']=0.
    b['frames'][0]['components']['pane']['count_relative_error']=3.
    result=compare_reports(a,b)
    assert result['candidate_recommended'] and result['measured_improvement']
    assert not result['regressions'] and len(result['raw_regressions'])==3
    assert len(result['tolerated_regressions'])==3
    assert b['structure_status']=='blocked' and not b['structure_accepted']


@pytest.mark.parametrize('coverage,error,floor',[(.989,.12,.95),(.949,.12,.95),(.996,.19,.95),(.996,.12,.999)])
def test_excessive_loss_floor_crossing_or_worse_depth_still_blocks(coverage,error,floor):
    a=report();b=report(coverage,error)
    a['frames'][0]['limits']['depth_coverage']=floor;b['frames'][0]['limits']['depth_coverage']=floor
    result=compare_reports(a,b)
    assert '313/depth/coverage' in result['regressions'] and not result['candidate_recommended']


def test_missing_metrics_and_lost_passed_gates_still_block():
    a=report();b=report(.996,.12)
    a['plausibility']['status']='passed'
    result=compare_reports(a,b)
    assert 'plausibility/gate' in result['regressions'] and not result['candidate_recommended']
    a=report();b=report(.996,.12);del b['frames'][0]['components']['pane']
    assert compare_reports(a,b)['regressions']


def test_matched_component_counts_cannot_be_hidden_as_unmatched():
    a=report();b=report(.996,.12)
    for r in (a,b):r['frames'][0]['components']['pane'].update(semantic_iou=.2,precision=.5,recall=.5)
    b['frames'][0]['components']['pane']['count_relative_error']=3.
    result=compare_reports(a,b)
    assert '313/pane/count_relative_error' in result['regressions']
    assert not result['candidate_recommended']


def test_warnings_without_measured_gain_are_not_improvement():
    a=report();b=report(.996,.18)
    b['plausibility']['score_0_100']=0.
    assert not compare_reports(a,b)['candidate_recommended']
    # A new fraction on a still-blocked manual checklist is not a gain either.
    a=report();b=copy.deepcopy(a);b['plausibility']['score_0_100']=50.
    assert not compare_reports(a,b)['measured_improvement']




def test_review_command_selects_candidate_and_preserves_original_reports(tmp_path):
    import json
    import evaluate
    before=tmp_path/'before.json';after=tmp_path/'after.json';out=tmp_path/'review.json'
    before.write_text(json.dumps(report()));after.write_text(json.dumps(report(.996,.12)))
    original=(before.read_bytes(),after.read_bytes())
    args=['review','--baseline',str(before),'--candidate',str(after),'--output',str(out)]
    assert evaluate.main(args)==0
    result=json.loads(out.read_text())
    assert result['decision']=='advance_candidate' and result['overall_acceptance']=='not_assessed'
    assert result['groups'][0]['review_policy']['version']==2
    assert (before.read_bytes(),after.read_bytes())==original
    frozen=out.read_bytes()
    assert evaluate.main(args)==2 and out.read_bytes()==frozen


def test_review_command_rejects_protocol_mismatch_before_publishing(tmp_path):
    import json
    import evaluate
    a=report();b=report(.996,.12);b['protocol_sha256']='changed'
    before=tmp_path/'before.json';after=tmp_path/'after.json';out=tmp_path/'review.json'
    before.write_text(json.dumps(a));after.write_text(json.dumps(b))
    assert evaluate.main(['review','--baseline',str(before),'--candidate',str(after),'--output',str(out)])==2
    assert not out.exists()
