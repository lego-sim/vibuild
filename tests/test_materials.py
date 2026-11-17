import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
from materials import read_materials


def test_unqualified_read_is_index_and_explicit_selection_is_complete(tmp_path):
    path = tmp_path/'source.json'
    path.write_text(json.dumps({'points': [[1,2,3]]*10000, 'geometry': {'radius': 1.25}, 'a/b': {'~key': 'exact'}}))
    reply = read_materials([str(path), f'{path}#/points', f'{path}#/geometry', f'{path}#/a~1b/~0key'])
    rows = json.loads(reply)
    assert rows[0]['index'] and rows[0]['index']['points']['count'] == 10000
    assert len(rows[1]['value']) == 10000
    assert rows[1]['value'][-1] == [1,2,3]
    assert rows[2]['value'] == {'radius': 1.25}
    assert rows[3]['value'] == 'exact'


def test_code_read_is_not_execution(tmp_path):
    path = tmp_path/'edit.py'
    path.write_text('raise RuntimeError("must not execute")\ndef rail():\n    return 3\n')
    rows = json.loads(read_materials([str(path), f'{path}#rail', f'{path}#L1-L1', f'{path}#missing']))
    assert rows[0]['index']['definitions'][0]['lines'] == [2, 3]
    assert rows[1]['value'] == 'def rail():\n    return 3'
    assert rows[2]['value'].startswith('1: raise')
    assert 'error' in rows[3]
