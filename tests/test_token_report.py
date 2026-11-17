import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('token_report', Path(__file__).parents[1]/'tools/token_report.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_deduplicates_requests_and_excludes_parent_cumulative_and_outside_window(tmp_path):
    def event(thread, response, stamp):
        return {'type': 'token_usage_record', 'timestamp': stamp, 'payload': {
            'thread_id': thread, 'turn_id': 'turn', 'response_id': response,
            'usage': dict(zip(m.FIELDS, [100, 80, 0, 10, 7, 110]))}}
    e = event('child', 'response', '2026-11-10T16:00:00Z')
    lines = [e, e, event('parent', 'parentresponse', e['timestamp']),
             event('child', 'outside', '2026-11-10T16:01:00Z'),
             {'type': 'event_msg', 'payload': {'type': 'token_count', 'total': 9999}}]
    f = tmp_path/'child.jsonl'
    f.write_text(''.join(json.dumps(v)+'\n' for v in lines)+'{"partial":')
    r = m.build_report([('score', 'child')], tmp_path,
                       m.timestamp('2026-11-10T16:00:00Z'), m.timestamp('2026-11-10T16:01:00Z'))
    assert r['totals']['requests'] == 1
    assert r['totals']['total_tokens'] == 110
    assert r['totals']['uncached_input_tokens'] == 20
    assert r['totals']['output_tokens'] == 10  # Reasoning is already included.
    assert len(r['warnings']) == 1
