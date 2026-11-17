"""Model session budgets reject invalid values and unused scheduler settings."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from vibuild import runtime
from vibuild.resources import save


@pytest.mark.parametrize('budget', [{'hours': 1}, {'max_rounds': 1.5},
    {'session_minutes': float('nan')}, {'idle_minutes': float('inf')},
    {'session_minutes': True}, {'idle_minutes': -1}, {'session_minutes': 0}])
def test_invalid_session_budget(tmp_path, budget):
    config = tmp_path/'config.json'; save(config, {'budget': budget})
    with pytest.raises(ValueError):runtime.limits(config)


def test_configured_and_default_limits(tmp_path):
    config = tmp_path/'config.json'; save(config, {'budget': {'idle_minutes': 2}})
    assert runtime.limits(config) == {'session_minutes': 60, 'idle_minutes': 2}
