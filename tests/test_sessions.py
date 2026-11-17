import json, os, sys, tempfile, unittest, time
from unittest.mock import patch
import pytest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vibuild import sessions,resources
from helpers import config

class SessionContracts(unittest.TestCase):
 def test_existing_record_directory_is_accepted_without_allowing_escape(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'records'/'test';p.mkdir(parents=True);resources.save(p/'record.json',{'id':'test'})
   with patch.object(resources,'ROOT',root):
    self.assertEqual(resources.record_path(str(p)),p)
    with self.assertRaises(ValueError):resources.record_path(str(root/'elsewhere'))
    with self.assertRaises(ValueError):resources.record_path(str(p/'..'/'test'))
 def test_fresh_command_never_resumes_or_copies_credentials(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);cfg=p/'config.json';cfg.write_text(json.dumps({'runtime':{'codex':'codex'}}));cmd=sessions.command(p,cfg)
   self.assertIn('project_doc_max_bytes=0',cmd);self.assertIn('memories.use_memories=false',cmd);self.assertIn('--ephemeral',cmd);self.assertIn('--ignore-user-config',cmd);self.assertNotIn('resume',cmd);self.assertNotIn('fork',cmd);self.assertNotIn('--dangerously-bypass-approvals-and-sandbox',cmd)
 def test_process_zero_exit_requires_completed_event_and_schema(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);fake=p/'fakecodex';fake.write_text('#!/usr/bin/env python3\nimport sys\nif sys.argv[1:]==["login","status"]:sys.exit(0)\nsys.stdin.read()\nprint("{}")\n');fake.chmod(0o755);cfg=p/'cfg.json';cfg.write_text(json.dumps({'runtime':{'codex':str(fake)}}))
   result=sessions.run(p/'run','No mutation',cfg);self.assertEqual(result['state'],'failed')
 def test_raw_record_requires_video_and_blocks_path_traversal(self):
  with self.assertRaises(ValueError):resources.record_path('../other')

if __name__=='__main__':unittest.main()


def test_strict_output_schema_rejects_optional_field_before_launch(tmp_path):
 import pytest
 schema=tmp_path/'schema.json'
 resources.save(schema,{'type':'object','properties':{'iteration':{'type':'string'}},'required':[],'additionalProperties':False})
 config=tmp_path/'config.json';resources.save(config,{'runtime':{'codex':'must-not-launch'}})
 with pytest.raises(ValueError,match='every property must be required'):
  sessions.command(tmp_path,config,schema)
 for name in ('score',):
  sessions.validate_output_schema(resources.ROOT/f'schemas/{name}.schema.json')


def test_real_runner_deadline_kills_tool_child(tmp_path):
    pidfile = tmp_path / 'child'
    cfg = config(tmp_path, f'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\nPath({str(pidfile)!r}).write_text(str(p.pid))\nprint(json.dumps({{"type":"thread.started","thread_id":"fake"}}),flush=True)\ntime.sleep(60)\n', session_minutes=.01)
    begin = time.monotonic()
    result = sessions.run(tmp_path / 'work', 'test', cfg)
    assert result['state'] == 'timed_out'
    assert time.monotonic() - begin < 5
    pid = int(pidfile.read_text()); stat = Path(f'/proc/{pid}/stat')
    assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    assert (tmp_path / 'work/.runtime/stop').exists()


def test_idle_and_parent_cancellation(tmp_path, monkeypatch):
    cfg = config(tmp_path, 'time.sleep(60)\n', idle_minutes=.005)
    assert sessions.run(tmp_path / 'idle', 'test', cfg)['state'] == 'idle_timeout'
    stop = tmp_path / 'stop'; stop.touch()
    monkeypatch.setenv('VIBUILD_PARENT_STOPS', json.dumps([str(stop)]))
    assert sessions.run(tmp_path / 'cancel', 'test', cfg)['state'] == 'cancelled'


def test_streamed_activity_and_completed_event(tmp_path, capsys):
    cfg = config(tmp_path, 'print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"measured S06"}}),flush=True)\nprint(json.dumps({"type":"turn.completed"}),flush=True)\n')
    result = sessions.run(tmp_path / 'work', 'test', cfg)
    assert result['state'] == 'completed'
    assert 'measured S06' in capsys.readouterr().err


def test_inherited_deadline_cannot_be_extended(tmp_path, monkeypatch):
    cfg = config(tmp_path, 'raise RuntimeError("should not start")\n')
    monkeypatch.setenv('VIBUILD_DEADLINE', str(time.time()-1))
    result = sessions.run(tmp_path / 'work', 'test', cfg, deadline=time.time()+100)
    assert result['state'] == 'timed_out'


def test_resume_cleanup_ignores_recycled_pid(tmp_path):
    resources.save(tmp_path / 'tasks/T1/attempts/01/.runtime/state.json', dict(state='running', pid=os.getpid(), process_birth='wrong'))
    with patch.object(os, 'killpg') as kill:
        sessions.cancel_previous(tmp_path)
        kill.assert_not_called()
    assert resources.load(tmp_path / 'tasks/T1/attempts/01/.runtime/state.json')['state'] == 'interrupted'


def test_runtime_layout_only_enumerates_runtime_files(tmp_path):
 from vibuild.layout import runtime_file, session_files, workspace_of
 work=tmp_path/'R01/session'
 current=runtime_file(work,'state.json');resources.save(current,{'state':'completed'})
 resources.save(work/'state.json',{'state':'obsolete'})
 assert list(session_files(tmp_path,'state.json'))==[current]
 assert workspace_of(current)==work


def test_readonly_scoring_images_are_attached_without_granting_write_access(tmp_path):
    cfg=tmp_path/'config.json';resources.save(cfg,{'runtime':{'codex':'codex'}})
    first=tmp_path/'view 1.jpg';second=tmp_path/'view2.jpg'
    first.write_bytes(b'image');second.write_bytes(b'image')
    cmd=sessions.command(tmp_path,cfg,read_only=True,images=[first,second])
    assert cmd.count('--image')==2
    assert [cmd[i+1] for i,value in enumerate(cmd) if value=='--image']==[str(first),str(second)]
    assert cmd[cmd.index('--sandbox')+1]=='read-only'
    assert '--approve-for-me' not in cmd and '--add-dir' not in cmd
    assert cmd[-1]=='-'
    with pytest.raises(ValueError,match='Missing session image'):
        sessions.command(tmp_path,cfg,read_only=True,images=[tmp_path/'absent.jpg'])


def test_decision_grants_task_root_and_excludes_unneeded_abilities_without_output_budget(tmp_path):
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'runtime': {'codex': 'codex'}}))
    task=tmp_path/'task';task.mkdir()
    cmd=sessions.command(tmp_path, cfg, writable_roots=(task,))
    assert cmd[cmd.index('--add-dir')+1] == str(task)
    assert not any('tool_output_token_limit' in arg for arg in cmd)
    assert 'skills.include_instructions=false' in cmd and 'skills.bundled.enabled=false' in cmd
    assert 'plugins._default.enabled=false' in cmd and 'agents.enabled=false' in cmd
    assert '--add-dir' not in sessions.command(tmp_path, cfg, read_only=True, writable_roots=(task,))
