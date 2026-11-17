"""Render admission survives independently of model-role scheduling."""
from pathlib import Path
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from vibuild import rendering
from vibuild.resources import save


def test_render_capacity_is_bounded_cancellable_and_released(tmp_path,monkeypatch):
    cfg=tmp_path/'config.json';save(cfg,{'rendering':{'max_workers':1}})
    with rendering.render_reservation(tmp_path,cfg):
        with pytest.raises(TimeoutError,match='deadline'):
            with rendering.render_reservation(tmp_path,cfg,deadline=time.time()+.02,poll_seconds=.005):pass
        stop=tmp_path/'stop';stop.touch()
        with pytest.raises(RuntimeError,match='cancelled'):
            with rendering.render_reservation(tmp_path,cfg,stop_paths=[stop]):pass
    with rendering.render_reservation(tmp_path,cfg):pass
    monkeypatch.setattr(rendering.os,'sched_getaffinity',lambda _: {0})
    assert rendering.policy(cfg)['effective_workers']==1


@pytest.mark.parametrize('value',[{'max_workers':0},{'max_workers':True},{'threads_per_worker':1.5},{'survey_target':1}])
def test_invalid_render_capacity_rejected(tmp_path,value):
    cfg=tmp_path/'config.json';save(cfg,{'rendering':value})
    with pytest.raises(ValueError):rendering.policy(cfg)
