"""ComfyUI adapter. All FlashVSR imports and GPU work run in a separate venv."""
import json,os,shutil,signal,subprocess,tempfile,time,uuid
from pathlib import Path
import folder_paths
import comfy.model_management as mm
from comfy_api.latest import ComfyExtension,io,ui,InputImpl,Types

RUNTIME=Path(os.environ.get('FLASHVSR_POD_DIR','/opt/flashvsr-pod'))
SCALES={'1.5倍':1.5,'2倍':2,'4倍（公式推奨）':4}
MOTION={'標準（速度優先）':4,'大きな動き向け':2,'動き重視（処理時間増加）':1}


def stop_child(child):
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()


def run_engine(directory):
    env=os.environ.copy()
    env.pop('PYTHONPATH',None)
    env.update(VIRTUAL_ENV=str(RUNTIME/'venv'),PATH=str(RUNTIME/'venv/bin')+':'+env.get('PATH',''),
               IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg',CC='/usr/bin/gcc',CXX='/usr/bin/g++',PYTHONUNBUFFERED='1')
    log=directory/'engine.log'
    command=[str(RUNTIME/'venv/bin/python'),str(RUNTIME/'engine.py'),'--directory',str(directory)]
    with log.open('wb') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        started=time.monotonic()
        try:
            while child.poll() is None:
                mm.throw_exception_if_processing_interrupted()
                if time.monotonic()-started>1800:raise TimeoutError('FlashVSR exceeded30minutes')
                time.sleep(.5)
        except BaseException:
            stop_child(child);raise
    if child.returncode:
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-8000));tail=f.read().decode(errors='replace')
        raise RuntimeError('FlashVSR failed:\n'+tail)
    return json.loads((directory/'result.json').read_text())


def copy_source(video,directory):
    try:source=video.get_stream_source()
    except NotImplementedError:source=None
    target=directory/'input.mp4'
    if isinstance(source,(str,Path)):
        path=Path(source)
        if not path.is_file():raise ValueError('Input video was not found')
        if path.stat().st_size>512*1024**2:raise ValueError('Initial input limit is512MiB')
        try:os.link(path,target)
        except OSError:shutil.copyfile(path,target)
    else:
        video.save_to(str(target),format=Types.VideoContainer('mp4'),codec=Types.VideoCodec('auto'))
        if target.stat().st_size>512*1024**2:raise ValueError('Initial input limit is512MiB')


class FlashVSRPod(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='FlashVSRPodTurbo',display_name='FlashVSR+ Turbo（動画拡大・保存）',
          category='video/upscale',is_output_node=True,
          inputs=[io.Video.Input('video'),
                  io.Combo.Input('scale',options=list(SCALES),default='2倍'),
                  io.Combo.Input('motion',options=list(MOTION),default='標準（速度優先）'),
                  io.Int.Input('seed',default=1,min=0,max=2147483647,control_after_generate=True)],
          outputs=[io.Video.Output()])

    @classmethod
    def execute(cls,video,scale,motion,seed):
        if scale not in SCALES or motion not in MOTION or type(seed) is not int or not 0<=seed<=2147483647:
            raise ValueError('Invalid FlashVSR settings')
        if not (RUNTIME/'venv/bin/python').is_file():raise RuntimeError('Run the FlashVSR Pod installer first')
        jobs=RUNTIME/'jobs';jobs.mkdir(parents=True,exist_ok=True)
        # The queue owns one execution at a time; free cached GPU models before
        # the independent runtime loads its model. No global packages are changed.
        with tempfile.TemporaryDirectory(prefix='job-',dir=jobs) as tmp:
            directory=Path(tmp);copy_source(video,directory)
            settings={'scale':SCALES[scale],'temporal_compression':MOTION[motion],'seed':seed}
            (directory/'settings.json').write_text(json.dumps(settings))
            mm.unload_all_models();mm.soft_empty_cache()
            result=run_engine(directory)
            if result.get('inference_verified') is not True:raise RuntimeError('Inference was not verified')
            name='flashvsr_'+uuid.uuid4().hex[:16]+'.mp4'
            output=Path(folder_paths.get_output_directory())/'FlashVSR-Pod';output.mkdir(parents=True,exist_ok=True)
            artifact=output/name
            shutil.move(str(directory/'output.mp4'),artifact)
            (output/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print('[FlashVSR-Pod] '+json.dumps(result,ensure_ascii=False),flush=True)
        return io.NodeOutput(InputImpl.VideoFromFile(str(artifact)),
            ui=ui.PreviewVideo([ui.SavedResult(name,'FlashVSR-Pod',io.FolderType.output)]))


class Extension(ComfyExtension):
    async def get_node_list(self):return [FlashVSRPod]

async def comfy_entrypoint():return Extension()
