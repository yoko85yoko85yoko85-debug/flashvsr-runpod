"""Restart only an idle ComfyUI on the requested port, preserving other services."""
import argparse,json,os,signal,socket,subprocess,time,urllib.request,urllib.error
from pathlib import Path


def process_table(proc=Path('/proc')):
    result={}
    for p in proc.iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0');args=[a.decode(errors='replace') for a in args if a]
            cwd=os.readlink(p/'cwd');ppid=int(next(x for x in (p/'status').read_text().splitlines() if x.startswith('PPid:')).split()[1])
            result[int(p.name)]=(args,cwd,ppid)
        except (OSError,ValueError,StopIteration):pass
    return result


def is_comfy(args,cwd):
    if not args:return False
    return any(Path(a).name=='main.py' for a in args[1:]) and ('comfy' in cwd.lower() or any('comfy' in a.lower() for a in args))


def port_of(args):
    try:
        if '--port' in args:return int(args[args.index('--port')+1])
        return int(next(a.split('=',1)[1] for a in args if a.startswith('--port='))) if any(a.startswith('--port=') for a in args) else 8188
    except (ValueError,IndexError):return None


def ancestors(pid,table):
    result=set()
    while pid in table and pid not in result:
        result.add(pid);pid=table[pid][2]
    return result


def check_idle(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/queue',timeout=5) as r:d=json.load(r)
    except Exception:raise RuntimeError('Cannot confirm existing ComfyUI is idle; it was NOT stopped. Check its queue and retry.')
    if not isinstance(d.get('queue_running'),list) or not isinstance(d.get('queue_pending'),list):raise RuntimeError('Unrecognized ComfyUI queue response; no service was stopped.')
    if d.get('queue_running') or d.get('queue_pending'):
        raise RuntimeError('ComfyUI has running/queued work; it was NOT stopped. Finish it before retrying.')


def is_runpod_template():
    start=Path('/start.sh')
    return Path('/opt/comfyui-auth-proxy/auth_proxy.py').is_file() and start.is_file() and 'COMFYUI_INTERNAL_PORT=18188' in start.read_text()


def connection_config(public_port):
    if public_port==8188 and is_runpod_template():return 18188,'127.0.0.1',True
    return public_port,'0.0.0.0',False


def check_proxy(public_port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{public_port}/',timeout=5) as r:pass
    except urllib.error.HTTPError as e:
        if e.code==401:return
        raise RuntimeError('Template password proxy is not ready; no service was stopped.')
    except Exception:
        raise RuntimeError('Template password proxy is unavailable. Set COMFYUI_PASSWORD in Pod settings and let template startup complete. No service was stopped.')
    raise RuntimeError('Expected the template password proxy on public8188. Refusing to bypass it.')


def wait_for_port(listen,port,seconds=15):
    # Match aiohttp's listener semantics. Without SO_REUSEADDR, TIME_WAIT
    # connections from the old server can look like a live port owner.
    deadline=time.monotonic()+seconds
    while True:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:
                probe.bind((listen,port))
                return
            except OSError:
                if time.monotonic()>=deadline:
                    raise RuntimeError('A live listener still occupies port '+str(port)+'. No unrelated service was stopped. Run: ss -ltnp | grep '+str(port)) from None
        time.sleep(1)


def restart(root,port):
    config=json.loads((root/'target.json').read_text())
    comfy=Path(config['comfy']);python=Path(config['python'])
    public_port=port;port,listen,preserve_proxy=connection_config(public_port)
    if preserve_proxy:check_proxy(public_port)
    table=process_table();pids=[pid for pid,(args,cwd,_) in table.items() if is_comfy(args,cwd) and port_of(args)==port]
    if len(pids)>1:raise RuntimeError('Multiple ComfyUI processes detected; no process was stopped')
    if pids:
        pid=pids[0]
        if Path(table[pid][1]).resolve()!=comfy.resolve():raise RuntimeError('A different ComfyUI owns the port. No service was stopped.')
        if pid==1:raise RuntimeError('ComfyUI is PID1; refusing to stop the Pod')
        check_idle(port)
        # The selected RunPod template launches ComfyUI as a child of /start.sh,
        # without a respawn loop. Leave /start.sh, auth proxy and other services up.
        if not is_runpod_template():raise RuntimeError('This restart helper requires the selected RunPod ComfyUI template')
        os.kill(pid,signal.SIGTERM)
        for _ in range(30):
            try:os.kill(pid,0)
            except ProcessLookupError:break
            time.sleep(1)
    wait_for_port(listen,port)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(VIRTUAL_ENV=str(python.parent.parent),PATH=str(python.parent)+':'+env.get('PATH',''),PYTHONUNBUFFERED='1')
    command=[str(python),str(comfy/'main.py'),'--listen',listen,'--port',str(port),'--disable-auto-launch','--max-upload-size','512']
    log=root/'comfyui.log'
    with log.open('ab',buffering=0) as f:
        process=subprocess.Popen(command,cwd=comfy,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(180):
        if process.poll() is not None:raise RuntimeError('ComfyUI exited; see '+str(log))
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/object_info',timeout=3) as r:info=json.load(r)
            required={'LoadVideo','FlashVSRPodTurbo'}
            if not required.issubset(info):raise RuntimeError('Missing nodes: '+str(required-set(info)))
            print('READY: FlashVSR+ Turbo in ComfyUI, public port '+str(public_port)+'. GPU inference has not yet been tested on this Pod.',flush=True)
            return
        except (urllib.error.URLError,TimeoutError,ConnectionError):time.sleep(1)
    raise RuntimeError('Startup timed out; see '+str(log))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8188);a=p.parse_args()
    if not 1024<=a.port<=65535:raise SystemExit('Invalid port')
    restart(a.root,a.port)
