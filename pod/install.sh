#!/usr/bin/env bash
# Standalone FlashVSR+ Turbo. No video generation model is installed.
set -euo pipefail
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_EXTRA_INDEX_URL PIP_INDEX_URL PYTHONPATH
export PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
FLASHVSR_POD_DIR="${FLASHVSR_POD_DIR:-/opt/flashvsr-pod}"
FLASHVSR_COMFY_DIR="${FLASHVSR_COMFY_DIR:-/workspace/runpod-slim/ComfyUI}"
FLASHVSR_COMFY_PYTHON="${FLASHVSR_COMFY_PYTHON:-$FLASHVSR_COMFY_DIR/.venv-cu132/bin/python}"
FLASHVSR_PORT="${FLASHVSR_PORT:-8188}"
export FLASHVSR_POD_DIR FLASHVSR_COMFY_DIR FLASHVSR_COMFY_PYTHON FLASHVSR_PORT
if [[ "${1:-}" == --help ]]; then
 echo 'FlashVSR+ Turbo installer for runpod/comfyui:2.4.1-comfyuiv0.44.0-cuda13.2'
 echo 'Run in Jupyter Terminal as root after template startup. NVIDIA24GB+, Python3.11-3.12,35GiB free.'
 exit 0
fi
[[ $(id -u) == 0 ]] || { echo 'Run in the Pod Jupyter Terminal as root.'; exit 1; }
python3 - <<'PREFLIGHT'
import json,os,pathlib,platform,shutil,subprocess,sys
if platform.system()!='Linux' or platform.machine()!='x86_64':raise SystemExit('Linux x86_64 is required')
if not (3,11)<=sys.version_info[:2]<=(3,12):raise SystemExit('Use Python3.11 or3.12')
if not shutil.which('nvidia-smi'):raise SystemExit('Use a GPU Pod')
subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],check=True)
comfy=pathlib.Path(os.environ['FLASHVSR_COMFY_DIR']).resolve();python=pathlib.Path(os.environ['FLASHVSR_COMFY_PYTHON'])
if not (comfy/'main.py').is_file() or not python.is_file():raise SystemExit('Let the selected ComfyUI template finish its first startup before installing.')
if not pathlib.Path('/opt/comfyui-auth-proxy/auth_proxy.py').is_file():raise SystemExit('This installer expects the selected RunPod ComfyUI template')
root=pathlib.Path(os.environ['FLASHVSR_POD_DIR']).resolve()
if root in (pathlib.Path('/'),pathlib.Path('/opt'),pathlib.Path('/workspace')):raise SystemExit('Choose a dedicated install directory')
if root.exists() and any(root.iterdir()) and not (root/'.managed').exists():raise SystemExit('Unmanaged files exist in the install directory; choose another directory')
root.mkdir(parents=True,exist_ok=True)
if not (root/'.managed').exists() and shutil.disk_usage(root).free<35*1024**3:raise SystemExit('Need at least35GiB free on the container disk;80GB container disk recommended')
if not os.environ['FLASHVSR_PORT'].isdigit() or not 1024<=int(os.environ['FLASHVSR_PORT'])<=65535:raise SystemExit('Invalid port')
(root/'.managed').touch()
(root/'target.json').write_text(json.dumps({'comfy':str(comfy),'python':str(python)}))
print('Runtime and models:',root,'; existing ComfyUI:',comfy)
PREFLIGHT
mkdir -p "$FLASHVSR_POD_DIR/tmp" "$FLASHVSR_POD_DIR/cache" "$FLASHVSR_POD_DIR/jobs"
export TMPDIR="$FLASHVSR_POD_DIR/tmp" HF_HOME="$FLASHVSR_POD_DIR/cache/hf"
exec 9>"$FLASHVSR_POD_DIR/install.lock"
flock -n 9 || { echo 'Another FlashVSR install is running.'; exit 1; }
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ffmpeg python3-venv python3-dev build-essential libgomp1
if [[ ! -x "$FLASHVSR_POD_DIR/venv/bin/python" ]]; then python3 -m venv "$FLASHVSR_POD_DIR/venv"; fi
PYTHON="$FLASHVSR_POD_DIR/venv/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1
printf '%s\n' 'torch==2.7.1' > "$FLASHVSR_POD_DIR/constraints.txt"
# FlashVSR dependencies are installed only in the dedicated venv.

cat > "$FLASHVSR_POD_DIR/requirements.txt" <<'FLASH_EMBED_REQUIREMENTS_TXT'
einops==0.8.1
imageio==2.37.0
imageio-ffmpeg==0.6.0
numpy==2.2.6
opencv-python-headless==4.11.0.86
Pillow==11.2.1
safetensors==0.5.3
tqdm==4.67.1
huggingface-hub==0.36.0

FLASH_EMBED_REQUIREMENTS_TXT

cat > "$FLASHVSR_POD_DIR/engine.py" <<'FLASH_EMBED_ENGINE_PY'
"""Official FlashVSR+ Turbo pipeline. GPU imports stay in the isolated child."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import media

ROOT = Path(__file__).resolve().parent / 'FlashVSR_Plus'
UPSTREAM_REVISION = 'b515ffc6dba57fdc9ad84c66cadd87d05f27590a'


def ffmpeg(args):
    subprocess.run(['/usr/bin/ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    *args], check=True, timeout=300)


def geometry_filter(meta):
    # Preserve the entire frame: bicubic resize, edge padding to the model's
    # 128-pixel grid, then crop only that added border after inference.
    return (f"scale={meta['width']}:{meta['height']}:flags=bicubic,"
            f"pad={meta['model_width']}:{meta['model_height']}:0:0,"
            "fillborders=left=0:top=0:"
            f"right={meta['model_width']-meta['width']}:"
            f"bottom={meta['model_height']-meta['height']}:mode=smear,setsar=1")


def prepare(source, destination, meta):
    ffmpeg(['-protocol_whitelist', 'file,pipe', '-i', str(source), '-map', '0:v:0',
            '-an', '-vf', geometry_filter(meta), '-vsync', '0',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '0',
            '-threads', '2', '-pix_fmt', 'yuv420p', str(destination)])


def finalize(raw, source, destination, meta):
    ffmpeg(['-i', str(raw), '-protocol_whitelist', 'file,pipe', '-i', str(source),
            '-map', '0:v:0', '-map', '1:a:0?',
            '-vf', f"crop={meta['width']}:{meta['height']}:0:0,setsar=1",
            '-vsync', '0', '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '18', '-threads', '2', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k', '-af', 'apad',
            '-t', str(meta['duration_seconds']), '-movflags', '+faststart', str(destination)])


def verify_video(path, meta):
    actual = media.probe(path)
    v = next(s for s in actual['streams'] if s['codec_type'] == 'video')
    if (int(v['width']), int(v['height']), int(v.get('nb_frames', 0))) != (
            meta['width'], meta['height'], meta['frames']):
        raise RuntimeError('Output dimensions or frame count changed')
    from fractions import Fraction
    if abs(float(Fraction(v['avg_frame_rate']))-meta['fps']) > .001:
        raise RuntimeError('Output FPS changed')
    audio = any(s['codec_type'] == 'audio' for s in actual['streams'])
    if audio != meta['has_audio']:
        raise RuntimeError('Output audio presence changed')
    return audio


def environment():
    import torch
    import triton
    import flashvsr_attn
    from models import attention_kernel_status
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA GPU with BF16 support is required')
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 22*1024**3:
        raise RuntimeError('This configuration requires a 24GB-class GPU or larger')
    status = attention_kernel_status()
    if not status['triton_block_sparse']:
        raise RuntimeError('Official Triton sparse attention is unavailable')
    return dict(gpu=props.name, vram_gib=round(props.total_memory/1024**3, 2),
                pytorch=torch.__version__, cuda_build=torch.version.cuda,
                triton=triton.__version__, attention_kernel='triton_block_sparse',
                upstream_revision=UPSTREAM_REVISION)


def infer(folder, settings, env):
    import torch
    from inference import InferenceRequest, build_inference_pipeline
    from inference.presets import stream_config_for_mode, model_paths_for_mode
    source = folder/'input.mp4'
    meta = media.metadata(source, settings['scale'])
    started = time.monotonic()
    prepared, raw, output = folder/'prepared.mp4', folder/'raw.mp4', folder/'output.mp4'
    prepare(source, prepared, meta)
    preparation = time.monotonic()-started
    load_started = time.monotonic()
    config = stream_config_for_mode('turbo', seed=settings['seed'], decode_every=4,
        temporal_compression=settings['temporal_compression'],
        attn_kernel='triton_block_sparse', cache_offload_mode='auto')
    pipeline = build_inference_pipeline(config, model_paths_for_mode(ROOT, 'turbo'))
    torch.cuda.synchronize()
    loading = time.monotonic()-load_started
    inference_started = time.monotonic()
    with torch.inference_mode():
        result = pipeline.infer(InferenceRequest(input_path=prepared, output_path=raw,
            height=meta['model_height'], width=meta['model_width'], fps=meta['fps']),
            progress=lambda message: print('[FlashVSR] '+message, flush=True))
    torch.cuda.synchronize()
    elapsed = time.monotonic()-inference_started
    if result.frames != meta['frames'] or result.metadata['attn_kernel'] != 'triton_block_sparse':
        raise RuntimeError('Official pipeline frame count or attention backend mismatch')
    finalize(raw, source, output, meta)
    audio = verify_video(output, meta)
    return dict(ok=True, model='FlashVSR+ Turbo', inference_verified=True,
        runtime_environment=env, input_width=meta['input_width'], input_height=meta['input_height'],
        width=meta['width'], height=meta['height'], model_width=meta['model_width'],
        model_height=meta['model_height'], frames=meta['frames'], fps=meta['fps'],
        duration_seconds=meta['duration_seconds'], audio=audio,
        scale=settings['scale'], seed=settings['seed'], temporal_compression=settings['temporal_compression'],
        preparation_seconds=round(preparation, 2), model_load_seconds=round(loading, 2),
        inference_and_encode_seconds=round(elapsed, 2),
        processing_seconds=round(time.monotonic()-started, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    folder = args.directory
    # Reject invalid clips before creating the CUDA context or loading weights.
    if not args.diagnose and not args.smoke_test:
        settings = json.loads((folder/'settings.json').read_text())
        media.metadata(folder/'input.mp4', settings['scale'])
    env = environment()
    if args.diagnose:
        result = dict(ok=True, inference_verified=False, runtime_environment=env,
                      note='Imports and GPU checked; smoke_test performs actual model inference')
    else:
        if args.smoke_test:
            ffmpeg(['-f', 'lavfi', '-i', 'testsrc2=size=128x128:rate=4', '-frames:v', '5',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(folder/'input.mp4')])
            settings = dict(scale=4, seed=1, temporal_compression=4)
        result = infer(folder, settings, env)
        if args.smoke_test:
            result['smoke_test'] = True
    (folder/'result.json').write_text(json.dumps(result))

if __name__ == '__main__':
    main()

FLASH_EMBED_ENGINE_PY

cat > "$FLASHVSR_POD_DIR/media.py" <<'FLASH_EMBED_MEDIA_PY'
"""FFmpeg metadata validation shared by CPU checks and the GPU child."""
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

MAX_FRAMES = 1800


def probe(path):
    r = subprocess.run(['/usr/bin/ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-show_streams', '-show_format', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=30, check=True)
    return json.loads(r.stdout)


def metadata(path, scale):
    p = probe(path)
    videos = [s for s in p['streams'] if s.get('codec_type') == 'video' and not s.get('disposition', {}).get('attached_pic')]
    if len(videos) != 1:
        raise ValueError('Input must have exactly one video stream')
    v = videos[0]
    if v.get('pix_fmt', '').startswith(('yuv420p10', 'yuv422p10', 'yuv444p10')) or v.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('This first version accepts SDR 8-bit video; HDR is not supported')
    if v.get('sample_aspect_ratio', '1:1') not in ('1:1', '0:1', 'N/A'):
        raise ValueError('Anamorphic video is not supported; export square pixels')
    width, height = int(v['width']), int(v['height'])
    rotation = next((float(s['rotation']) for s in v.get('side_data_list', []) if 'rotation' in s), 0) % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError('Unsupported video rotation')
    if rotation in (90, 270):
        width, height = height, width
    duration = float(v.get('duration') or p.get('format', {}).get('duration') or 0)
    try:
        fps = Fraction(v.get('avg_frame_rate', '0/1'))
    except (ValueError, ZeroDivisionError):
        raise ValueError('Invalid video frame rate')
    if not math.isfinite(duration) or not 0 < duration <= 30.1 or not 1 <= fps <= 60:
        raise ValueError('Input must be at most 30 seconds and 1-60 fps')
    if min(width, height) < 64 or max(width, height) > 1920 or width * height > 1920 * 1080:
        raise ValueError('Input dimensions must be 64-1920px, with area <= 1920x1080')
    ow, oh = width * scale, height * scale
    if ow % 2 or oh % 2:
        raise ValueError('Output dimensions must be even')
    mw, mh = ((ow+127)//128)*128, ((oh+127)//128)*128
    if max(mw, mh) > 2048 or mw*mh > 2048*2048:
        raise ValueError('Initial version limits model output to 2048px per side; use 2x or a smaller source')
    # Inspect timestamps: reject VFR rather than silently changing its timing.
    r = subprocess.run(['/usr/bin/ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-select_streams', 'v:0', '-read_intervals', '%+#2000', '-show_entries', 'frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], capture_output=True, text=True, timeout=45, check=True)
    frames = json.loads(r.stdout).get('frames', [])
    if not 2 <= len(frames) <= MAX_FRAMES:
        raise ValueError('Input must contain 2-1800 video frames')
    times = [float(f['best_effort_timestamp_time']) for f in frames]
    if any(not math.isfinite(t) for t in times):
        raise ValueError('Invalid video timestamps')
    if any(abs((b-a)-float(1/fps)) > max(.002, float(1/fps)*.05) for a,b in zip(times,times[1:])):
        raise ValueError('Variable-frame-rate input is unsupported; export constant frame rate')
    return {'input_width': width, 'input_height': height, 'width': ow, 'height': oh,
            'fps_fraction': str(fps), 'fps': float(fps), 'frames': len(frames),
            'duration_seconds': float(len(frames)/fps),
            'model_width': mw, 'model_height': mh, 'rotation': rotation,
            'has_audio': any(s.get('codec_type') == 'audio' for s in p['streams'])}

FLASH_EMBED_MEDIA_PY

cat > "$FLASHVSR_POD_DIR/download_models.py" <<'FLASH_EMBED_DOWNLOAD_MODELS_PY'
"""Download immutable, SHA-256 verified official Turbo weights during build."""
from pathlib import Path
import hashlib
REVISION = '46b79520946ac67dec94d1566490690953c7debc'
FILES = {
 'turbo/dit.safetensors': (5676070392, '7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b'),
 'turbo/lq_proj_in.safetensors': (1151384272, 'e77b378ae12342eefbb58f924256f362a151b93ce73dd32808fcdaca9c051496'),
 'TCDecoder.ckpt': (189018333, 'e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161'),
}

def verify(path, size, digest):
    if path.stat().st_size != size:
        raise RuntimeError('Weight size mismatch: '+path.name)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    if h.hexdigest() != digest:
        raise RuntimeError('Weight SHA256 mismatch: '+path.name)

if __name__ == '__main__':
    from huggingface_hub import hf_hub_download
    for name, (size, digest) in FILES.items():
        path = Path(hf_hub_download('ruchangyao/FlashVSR_plus', name,
            revision=REVISION, local_dir=Path(__file__).resolve().parent/'FlashVSR_Plus/ckpts'))
        verify(path, size, digest)
        print('Verified '+name, flush=True)

FLASH_EMBED_DOWNLOAD_MODELS_PY

cat > "$FLASHVSR_POD_DIR/node.py" <<'FLASH_EMBED_NODE_PY'
"""ComfyUI adapter. All FlashVSR imports and GPU work run in a separate venv."""
import json,os,shutil,signal,subprocess,tempfile,time,uuid
from pathlib import Path
import folder_paths
import comfy.model_management as mm
from comfy_api.latest import ComfyExtension,io,ui,InputImpl,Types

RUNTIME=Path(os.environ.get('FLASHVSR_POD_DIR','/opt/flashvsr-pod'))
SCALES={'2倍':2,'4倍（公式推奨）':4}
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

FLASH_EMBED_NODE_PY

cat > "$FLASHVSR_POD_DIR/workflow.py" <<'FLASH_EMBED_WORKFLOW_PY'
import json
from pathlib import Path

def build():
    note='''## FlashVSR+ Turbo：アップスケール専用

1. Load Videoで動画をアップロード。
2. 倍率を2倍または4倍に設定。
3. Queueを実行すると、拡大後のMP4を保存・プレビューします。

初期値：2倍・標準（速度優先）・シード1。
公式モデル推奨は4倍です。2倍は実用サイズで比較する選択肢です。

30秒まで・固定fps1〜60・SDR動画。出力各辺2048pxまで。
例：800×800は2倍で1600×1600。4倍では上限を超えます。
入力512MiBまで。アプリ版の64MiB転送制限はありません。
縦横比・フレーム数・fpsを保ち、音声をAACで引き継ぎます。
細部が変わる可能性があるため、元映像と比較してください。

ComfyUIのoutput/FlashVSR-PodにMP4と処理情報JSONを保存します。
時間・解像度によりGPUメモリ不足や時間切れになる場合があります。
初回はモデル読込とTritonのコンパイルで時間が増えます。
'''
    nodes=[
      {'id':1,'type':'LoadVideo','title':'元動画を読み込む','pos':[0,0],'size':[420,360],
       'flags':{},'order':0,'mode':0,'inputs':[],
       'outputs':[{'name':'VIDEO','type':'VIDEO','links':[1],'slot_index':0}],
       'properties':{'Node name for S&R':'LoadVideo'},'widgets_values':['flashvsr-example.mp4','video']},
      {'id':2,'type':'FlashVSRPodTurbo','pos':[500,0],'size':[480,520],
       'flags':{},'order':1,'mode':0,'inputs':[{'name':'video','type':'VIDEO','link':1}],
       'outputs':[{'name':'VIDEO','type':'VIDEO','links':[],'slot_index':0}],
       'properties':{'Node name for S&R':'FlashVSRPodTurbo'},
       'widgets_values':['2倍','標準（速度優先）',1,'fixed']},
      {'id':3,'type':'MarkdownNote','title':'使い方','pos':[0,420],'size':[460,670],
       'flags':{},'order':2,'mode':0,'inputs':[],'outputs':[],'properties':{},'widgets_values':[note]}]
    return {'last_node_id':3,'last_link_id':1,'nodes':nodes,'links':[[1,1,0,2,0,'VIDEO']],
            'groups':[],'config':{},'extra':{'ds':{'scale':.8,'offset':[60,60]}},'version':.4}

def write(comfy):
    dest=Path(comfy)/'user/default/workflows';dest.mkdir(parents=True,exist_ok=True)
    (dest/'FlashVSR-Pod-Turbo.json').write_text(json.dumps(build(),ensure_ascii=False,indent=2))

if __name__=='__main__':
    import sys
    write(sys.argv[1])

FLASH_EMBED_WORKFLOW_PY

cat > "$FLASHVSR_POD_DIR/manage.py" <<'FLASH_EMBED_MANAGE_PY'
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

FLASH_EMBED_MANAGE_PY

cat > "$FLASHVSR_POD_DIR/setup.py" <<'FLASH_EMBED_SETUP_PY'
import json,os,shutil,subprocess,sys
from pathlib import Path
from workflow import write
ROOT=Path(__file__).resolve().parent
config=json.loads((ROOT/'target.json').read_text());comfy=Path(config['comfy'])
module=comfy/'custom_nodes/flashvsr_pod_turbo';module.mkdir(parents=True,exist_ok=True)
node=(ROOT/'node.py').read_text().replace("RUNTIME=Path(os.environ.get('FLASHVSR_POD_DIR','/opt/flashvsr-pod'))",'RUNTIME=Path('+repr(str(ROOT))+')')
(module/'__init__.py').write_text(node)
write(comfy)
example=comfy/'input/flashvsr-example.mp4';example.parent.mkdir(parents=True,exist_ok=True)
if not example.exists():
    subprocess.run(['/usr/bin/ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=128x128:rate=4',
                    '-frames:v','5','-c:v','libx264','-pix_fmt','yuv420p',str(example)],check=True)
print('Installed FlashVSR node and workflow. No GPU inference was submitted.')

FLASH_EMBED_SETUP_PY

"$PYTHON" -m pip install -c "$FLASHVSR_POD_DIR/constraints.txt" -r "$FLASHVSR_POD_DIR/requirements.txt"
UPSTREAM="$FLASHVSR_POD_DIR/FlashVSR_Plus"
REV=b515ffc6dba57fdc9ad84c66cadd87d05f27590a
if [[ ! -d "$UPSTREAM/.git" ]]; then git init "$UPSTREAM"; git -C "$UPSTREAM" remote add origin https://github.com/OpenImagingLab/FlashVSR_Plus.git; fi
if [[ $(git -C "$UPSTREAM" rev-parse HEAD 2>/dev/null || true) != "$REV" ]]; then
 if [[ -n $(git -C "$UPSTREAM" status --porcelain --untracked-files=no) ]]; then echo 'Upstream source has local edits; not overwriting.'; exit 1; fi
 git -C "$UPSTREAM" fetch --depth 1 origin "$REV"
 git -C "$UPSTREAM" checkout --detach "$REV"
fi
"$PYTHON" -m pip install --no-deps "$UPSTREAM" "$UPSTREAM/operators/flashvsr_attn"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
"$PYTHON" - <<'COMPILER'
import ctypes,os,pathlib,subprocess,sysconfig,tempfile,torch,triton,flashvsr_attn
if not torch.cuda.is_available():raise SystemExit('CUDA initialization failed; models not downloaded')
p=torch.cuda.get_device_properties(0)
if p.total_memory<22*1024**3:raise SystemExit('Use a24GB-class NVIDIA GPU or larger')
with tempfile.TemporaryDirectory() as d:
 src=pathlib.Path(d)/'check.c';so=pathlib.Path(d)/'check.so'
 src.write_text('#include <Python.h>\nint check(void){return PY_MAJOR_VERSION;}\n')
 subprocess.run([os.environ['CC'],'-shared','-fPIC','-I'+sysconfig.get_path('include'),str(src),'-o',str(so)],check=True)
 assert ctypes.CDLL(str(so)).check()==3
print('GPU:',p.name,'torch:',torch.__version__,'Triton:',triton.__version__,'C compiler OK; inference not yet tested')
COMPILER
"$PYTHON" "$FLASHVSR_POD_DIR/download_models.py"
"$PYTHON" "$FLASHVSR_POD_DIR/setup.py"
cat > "$FLASHVSR_POD_DIR/restart-comfyui.sh" <<'RESTART'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/venv/bin/python" "$HERE/manage.py" --root "$HERE" --port "${FLASHVSR_PORT:-8188}"
RESTART
chmod +x "$FLASHVSR_POD_DIR/restart-comfyui.sh"
if [[ "${FLASHVSR_NO_RESTART:-0}" != 1 ]]; then bash "$FLASHVSR_POD_DIR/restart-comfyui.sh"; fi
echo 'Open public8188, reload ComfyUI, and load FlashVSR-Pod-Turbo.json.'
echo 'Runtime+models are on the container disk; models total7.02GB. No generation model was installed.'
echo 'No GPU inference has been submitted. First try the synthetic sample at2x.'
