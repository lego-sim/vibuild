"""Fresh noninteractive Codex sessions; credentials stay in Codex's existing login."""
from __future__ import annotations
import fcntl,json,os,signal,subprocess,time,sys
from pathlib import Path
from .resources import load,save
from .config import settings, tool
from .layout import runtime_file,session_files

def validate_output_schema(path):
 """Catch the strict-output object contract before any model call or budget use."""
 def visit(value,where):
  if isinstance(value,dict):
   if value.get('type')=='object' or 'properties' in value:
    if set(value.get('required',[]))!=set(value.get('properties',{})) or value.get('additionalProperties') is not False:
     raise ValueError(f'Invalid structured output schema at {where}: every property must be required and additionalProperties must be false')
   for key,item in value.items():visit(item,where+'/'+key)
  elif isinstance(value,list):
   for i,item in enumerate(value):visit(item,where+'/'+str(i))
 visit(load(path),str(path))


def command(work,config,output_schema=None,model=None,read_only=False,reasoning_effort=None,images=(),writable_roots=()):
 if output_schema:validate_output_schema(output_schema)
 cfg=settings(config)
 if not cfg.get('runtime',{}).get('codex'):raise ValueError('Configure runtime.codex')
 cmd=[cfg['runtime']['codex'],'exec','--json','--skip-git-repo-check','--ephemeral','--ignore-user-config','--cd',str(work),'--color','never','--output-last-message',str(work/'.runtime'/('final.json' if output_schema else 'final.txt'))]
 if read_only:cmd+=['--sandbox','read-only','-c','approval_policy="never"']
 else:
  cmd+=['--approve-for-me']
  for root in writable_roots:cmd+=['--add-dir',str(Path(root).resolve())]
 # No previous session, fork, memory or unrelated project instructions enter the packet.
 cmd+=['-c','memories.use_memories=false','-c','project_doc_max_bytes=0','-c','web_search="disabled"',
       '-c','skills.include_instructions=false','-c','skills.bundled.enabled=false',
       '-c','plugins._default.enabled=false','-c','agents.enabled=false']
 model=model or cfg.get('session',{}).get('model')
 reasoning_effort=reasoning_effort or cfg.get('session',{}).get('reasoning_effort')
 if model:cmd+=['--model',model]
 if reasoning_effort:cmd+=['-c','model_reasoning_effort='+json.dumps(reasoning_effort)]
 if output_schema:cmd+=['--output-schema',str(output_schema)]
 for image in images:
  if not Path(image).is_file():raise ValueError(f'Missing session image: {image}')
  cmd+=['--image',str(Path(image).resolve())]
 return cmd+['-']

def process_birth(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
 except (OSError,IndexError):return None


def cancel_previous(root):
 # Recover after SIGKILL/host interruption without signalling a recycled PID.
 for path in session_files(root,'state.json'):
  value=load(path)
  if value.get('state')!='running':continue
  (path.parent/'stop').touch()
  pid=value.get('pid');birth=value.get('process_birth')
  if pid and birth and process_birth(pid)==birth:
   try:os.killpg(pid,signal.SIGTERM)
   except ProcessLookupError:pass
   time.sleep(.2)
   if process_birth(pid)==birth:
    try:os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:pass
  value.update(state='interrupted',finished=time.time());save(path,value)


def terminate(proc):
 # Also signal the group after its leader exited: tools can outlive the CLI.
 try: os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError: return
 try: proc.wait(timeout=3)
 except subprocess.TimeoutExpired: pass
 try: os.killpg(proc.pid,signal.SIGKILL)
 except ProcessLookupError: pass
 proc.wait()


def run(work,prompt,config,output_schema=None,model=None,dry_run=False,read_only=False,reasoning_effort=None,deadline=None,images=(),writable_roots=()):
 from .runtime import limits
 work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
 (work/'.runtime').mkdir(exist_ok=True)
 budget=limits(config);now=time.time()
 ends=min(now+budget['session_minutes']*60, deadline or float('inf'),float(os.environ.get('VIBUILD_DEADLINE','inf')))
 prompt+=f'\nSESSION_DEADLINE_UTC_UNIX={ends}. Return the required structured result before this deadline; the runner persists it.\n'
 (runtime_file(work,'instruction.txt')).write_text(prompt)
 cmd=command(work,config,output_schema,model,read_only,reasoning_effort,images,writable_roots)
 save(runtime_file(work,'launch.json'),dict(command=cmd,cwd=str(work),auth='existing Codex login; no credentials copied',fresh_context=True,dry_run=dry_run,deadline=ends))
 if dry_run:return dict(state='prepared',workspace=str(work),command=cmd)
 if now>=ends:return dict(state='timed_out',workspace=str(work))
 auth=subprocess.run([tool(config,'codex'),'login','status'],capture_output=True,text=True,timeout=min(20,ends-now))
 if auth.returncode:raise RuntimeError('Codex is not logged in. Run codex login once, then use this external entrypoint.')
 lock=(runtime_file(work,'session.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 proc=None;state=dict(state='starting',started=now,workspace=str(work),last_activity=now)
 parent_stops=json.loads(os.environ.get('VIBUILD_PARENT_STOPS','[]'))
 stop=runtime_file(work,'stop');stop.unlink(missing_ok=True)
 env=dict(os.environ,VIBUILD_DEADLINE=str(ends),VIBUILD_PARENT_STOPS=json.dumps(parent_stops+[str(stop)]))
 env['VIBUILD_CONFIG']=str(Path(config).resolve())
 env['VIBUILD_WORKSPACE']=str(work)
 if runtime_file(work,'task.json').exists():env['VIBUILD_TASK']=str(runtime_file(work,'task.json'))
 previous_handler=signal.getsignal(signal.SIGTERM)
 def interrupted(signum,frame):raise KeyboardInterrupt('Termination requested')
 signal.signal(signal.SIGTERM,interrupted)
 try:
  with (runtime_file(work,'events.jsonl')).open('w') as events,(runtime_file(work,'stderr.log')).open('w') as err, (runtime_file(work,'instruction.txt')).open() as stdin:
   proc=subprocess.Popen(cmd,stdin=stdin,stdout=events,stderr=err,text=True,start_new_session=True,env=env)
   state.update(state='running',pid=proc.pid,process_birth=process_birth(proc.pid));save(runtime_file(work,'state.json'),state)
   print(f'[vibuild] started {work}; limit={max(0,ends-time.time())/60:.1f}min',file=sys.stderr,flush=True)
   terminal=None;errors=0;last_save=0;buffer='';stderr_size=0
   with (runtime_file(work,'events.jsonl')).open() as stream:
    while True:
     chunk=stream.read()
     if chunk:state['last_activity']=time.time();buffer+=chunk
     while '\n' in buffer:
      line,buffer=buffer.split('\n',1)
      try:event=json.loads(line)
      except ValueError:continue
      kind=event.get('type');item=event.get('item',{})
      if kind=='thread.started':state['thread_id']=event.get('thread_id')
      if kind in ('turn.completed','turn.failed'):terminal=kind
      if kind=='error':errors+=1
      if item.get('type') in ('agent_message','command_execution'):
       message=item.get('text') or item.get('command') or item.get('status','')
       state['activity']=str(message).replace('\n',' ')[:400]
       print(f'[{work.name}] {kind}: {state["activity"]}',file=sys.stderr,flush=True)
     now=time.time();size=(runtime_file(work,'stderr.log')).stat().st_size
     if size!=stderr_size:state['last_activity']=now;stderr_size=size
     if now-last_save>=2:
      state.update(elapsed_seconds=now-state['started'],deadline=ends)
      save(runtime_file(work,'state.json'),state);last_save=now
     if proc.poll() is not None:
      # Drain any output written between the read and process termination.
      if stream.tell()<(runtime_file(work,'events.jsonl')).stat().st_size:continue
      break
     reason=('cancelled' if any(Path(p).exists() for p in parent_stops) else
             'timed_out' if now>=ends else 'idle_timeout' if now-state['last_activity']>=budget['idle_minutes']*60 else None)
     if reason:state['state']=reason;terminate(proc);break
     time.sleep(.2)
   state.update(exit_code=proc.returncode,reported_error_events=errors,finished=time.time())
   if state['state']=='running':state['state']='completed' if proc.returncode==0 and terminal=='turn.completed' else 'failed'
   if state['state']=='completed' and output_schema:
    try:
     value=load(runtime_file(work,'final.json'))
    except (ValueError,OSError,TypeError) as exc:state.update(state='invalid_result',error=str(exc))
   save(runtime_file(work,'state.json'),state)
   print(f'[{work.name}] {state["state"]}',file=sys.stderr,flush=True)
   return state
 except BaseException:
  state.update(state='interrupted',finished=time.time());save(runtime_file(work,'state.json'),state);raise
 finally:
  stop.touch() # Tool children observe cancellation across process groups.
  if proc:terminate(proc)
  signal.signal(signal.SIGTERM,previous_handler);lock.close()
