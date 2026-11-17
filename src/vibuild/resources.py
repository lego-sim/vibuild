"""Record resource import and indexing. No destructive cleanup of the source project."""
from __future__ import annotations
import hashlib,json,re,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(p):return json.loads(Path(p).read_text())
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(p)
def record_path(name):
 if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',name):
  path=Path(name)
  if '..' not in path.parts:
   path=path.resolve()
   if path.parent==(ROOT/'records').resolve() and (path/'record.json').is_file():return path
  raise ValueError('Record must be a simple ID or an existing direct record directory under this project records/.')
 return ROOT/'records'/name
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def copy(source,dest):
 source=Path(source);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 if source.is_dir():
  dest.mkdir(exist_ok=True)
  for p in source.iterdir():
   if p.name=='__pycache__' or p.suffix in ['.pyc','.blend1']:continue
   copy(p,dest/p.name)
 else:subprocess.run(['cp','--reflink=auto','--',str(source),str(dest)],check=True)
def init_record(name,videos):
 record=record_path(name)
 if (record/'record.json').exists():raise FileExistsError(f'Record exists: {name}')
 files=sorted(p for p in Path(videos).resolve().iterdir() if p.suffix.lower() in ['.mp4','.mov','.mkv','.avi'])
 if not files:raise ValueError('No videos in inbox; add videos before starting.')
 res=record/'resources';dest=res/'input/videos';dest.mkdir(parents=True,exist_ok=True)
 for p in files:copy(p,dest/p.name)
 save(res/'index.json',dict(version=1,coordinate_status='unregistered',videos=[str(p.relative_to(res)) for p in dest.iterdir()],provenance='Original videos copied; no camera/depth/alignment claims yet.'))
 save(record/'record.json',dict(version=1,id=name,resources='resources/index.json',state='needs_reconstruction',created_at=time.time()))
 return record

def validate_record(record,hashes=False):
 record=Path(record).resolve();res=record/'resources';index=load(res/'index.json');missing=[];changed=[]
 for entry in load(record/'inventory.json').get('files',[]) if (record/'inventory.json').exists() else []:
  p=res/entry['path']
  if not p.is_file():missing.append(entry['path'])
  elif p.stat().st_size!=entry['bytes'] or (hashes and digest(p)!=entry['sha256']):changed.append(entry['path'])
 for key in ['frames','cameras','coordinates','baseline','geometry','partition','evaluation']:
  if key in index and not (res/index[key]).exists():missing.append(index[key])
 for p in index.get('videos',[]):
  if not (res/p).is_file():missing.append(p)
 return dict(ok=not missing and not changed,missing=missing,changed=changed,hashes_verified=hashes)
