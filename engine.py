"""Official FlashVSR+ Turbo pipeline. GPU imports stay in the isolated child."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import media

ROOT = Path('/opt/FlashVSR_Plus')
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
