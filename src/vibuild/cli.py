"""Video-guided modeling with configured local tools."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from . import resources
from .config import default_config, doctor, settings, tool


def parser():
    root=argparse.ArgumentParser(prog='vibuild',description=__doc__)
    commands=root.add_subparsers(dest='cmd',required=True)
    def config(p):p.add_argument('--config',type=Path,default=default_config())
    def record(p):p.add_argument('--record',required=True)
    p=commands.add_parser('init',help='create a record for original videos')
    record(p);p.add_argument('--videos',type=Path,required=True)
    p=commands.add_parser('doctor',help='check configured runtimes and login');config(p)
    p=commands.add_parser('check',help='verify frozen record inputs')
    record(p);p.add_argument('--hashes',action='store_true')
    p=commands.add_parser('report',help='build an offline spatial/topology report')
    config(p);p.add_argument('--partition',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('task',help='initialize a task from reviewed partition and registered scene')
    record(p);p.add_argument('--partition',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('run',help='execute fresh decisions and queued jobs')
    config(p);p.add_argument('--task',type=Path,required=True);p.add_argument('--proxy')
    p=commands.add_parser('tool',help='shared scene, perception, numerical audit and visual scoring tools')
    config(p);p.add_argument('name',choices=['vidmap','perceive','scene','evaluate','score','connections','jobs'])
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.cmd=='tool':
            extra=list(args.arguments)
            script=resources.ROOT/'tools'/('registration.py' if args.name=='vidmap' else f'{args.name}.py')
            if '--help' in extra:
                return subprocess.call([sys.executable,str(script),*extra])
            if args.name=='vidmap':
                return subprocess.call([sys.executable,str(script),
                    '--config',str(args.config.resolve()),*extra])
            if args.name in ('perceive','score','jobs') and '--config' not in extra:
                extra+=['--config',str(args.config.resolve())]
            if (args.name=='scene' or args.name=='evaluate' and extra[:1]==['compare']) and '--blender' not in extra:
                extra+=['--blender',tool(args.config,'blender')]
            env=dict(os.environ,VIBUILD_CONFIG=str(args.config.resolve()))
            return subprocess.call([tool(args.config,'analysis_python'),
                                    str(script),*extra],env=env)
        if args.cmd=='run':
            for name in ('analysis_python','blender','codex'):tool(args.config,name)
            if not settings(args.config).get('session',{}).get('model'):
                raise ValueError('Configure session.model before running decisions')
            command=[tool(args.config,'analysis_python'),str(resources.ROOT/'tools/task_service.py'),
                     '--task',str(args.task.resolve()),'--config',str(args.config.resolve())]
            if args.proxy:command+=['--proxy',args.proxy]
            return subprocess.call(command)
        if args.cmd=='task':
            from .layout import initialize_task
            partition=resources.load(args.partition)
            if partition.get('schema_version')!='partition_v1':raise ValueError('Expected partition_v1')
            execution=partition.get('execution_plan') or partition.get('topology',{}).get('execution_plan')
            if not execution:
                sys.path.insert(0,str(resources.ROOT/'tools'))
                from partition_report import graph_analysis
                execution=graph_analysis(partition['regions'],partition['interfaces'],
                    start=partition.get('topology',{}).get('start_region'))['execution_plan']
            if execution.get('status')!='planned':raise ValueError('Partition requires a reviewed execution plan')
            if not args.scene.is_file():raise ValueError('Provide a self-contained scene with registered cameras')
            with args.scene.open('rb') as stream:
                if stream.read(7)!=b'BLENDER':raise ValueError('Initial scene must be an uncompressed Blender project')
            output=args.output.resolve()
            if output.is_relative_to(resources.ROOT/'records'):
                raise ValueError('Task output must not modify frozen records')
            if output.exists() and any(output.iterdir()):raise ValueError('Task output must be empty')
            paths=initialize_task(output,[s for chain in execution['chains'] for s in chain['steps']])
            shutil.copyfile(args.scene,paths['scene'])
            plan=resources.load(paths['plan']);plan.update(record=args.record,partition=str(args.partition.resolve()))
            resources.save(paths['plan'],plan)
            regions={r['id']:r for r in partition['regions']}
            for step in plan['steps']:
                resources.save(output/step['id']/'stage.json',step)
                if step['id'] in regions:
                    resources.save(output/step['id']/'region.json',{'region':regions[step['id']]})
                    resources.save(output/step['id']/'job_template.json',{'steps':[{'tool':'score','args':[
                        'prepare','--record',args.record,'--partition',str(args.partition.resolve()),
                        '--scene',str(paths['scene']),'--region',step['id'],
                        '--output',str(output/step['id']/'scores/001')]}]})
            resources.save(paths['status'],{'state':'ready','execution_paused':False,
                'active_region':plan['execution_order'][0],'global_acceptance':'not_assessed'})
            print(json.dumps({'task':str(output),'scene':str(paths['scene'])}));return 0
        if args.cmd=='report':
            out=args.output.resolve()
            if out.is_relative_to((resources.ROOT/'records').resolve()):
                raise ValueError('Report output must not modify frozen records')
            python=tool(args.config,'analysis_python')
            subprocess.run([python,str(resources.ROOT/'tools/partition_report.py'),
                            '--input',str(args.partition.resolve()),'--output',str(out)],check=True)
            return 0
        if args.cmd=='doctor':
            result=doctor(args.config)
            if 'error' not in result['tools']['codex']:
                auth=subprocess.run([tool(args.config,'codex'),'login','status'],capture_output=True,timeout=20)
                result['codex_login']=auth.returncode==0
                result['ok']=result['ok'] and result['codex_login']
            if not settings(args.config).get('session',{}).get('model'):
                result.update(ok=False,model_error='Configure session.model')
        elif args.cmd=='check':
            result=resources.validate_record(resources.record_path(args.record),args.hashes)
        else:result=str(resources.init_record(args.record,args.videos))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return int(isinstance(result,dict) and result.get('ok') is False)
    except (ValueError,KeyError,OSError,subprocess.SubprocessError) as exc:
        print(f'vibuild: {exc}',file=sys.stderr);return 1


if __name__=='__main__':
    raise SystemExit(main())
