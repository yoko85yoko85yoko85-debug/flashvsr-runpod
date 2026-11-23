"""RunPod transport and resource limits. No CUDA work during module import."""
import base64
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import transfer

VERSION = 'flashvsr-plus-turbo-v1.3'
HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()
MAX_INPUT = 6 * 1024**2
MAX_OUTPUT = 6 * 1024**2


def safe_text(value):
    s = str(value)
    for name, secret in os.environ.items():
        if any(k in name.upper() for k in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY')) and len(secret) >= 4:
            s = s.replace(secret, '[redacted]')
    s = re.sub(r'token=[a-f0-9]{64}', 'token=[redacted]', s)
    s = re.sub(r'(?:hf_|rpa_|ghp_)[A-Za-z0-9_-]+', '[redacted]', s)
    return s[-3000:]


def validate(p):
    allowed = {'video_base64', 'scale', 'seed', 'temporal_compression', 'diagnose', 'smoke_test', 'video_url', 'output_upload_url', 'output_download_url'}
    if not isinstance(p, dict) or set(p) - allowed:
        raise ValueError('Unsupported FlashVSR input')
    for flag in ('diagnose', 'smoke_test'):
        if flag in p and type(p[flag]) is not bool:
            raise ValueError(flag + ' must be true or false')
    flags = [flag for flag in ('diagnose', 'smoke_test') if p.get(flag)]
    if flags:
        if len(flags) != 1 or set(p) != {flags[0]}:
            raise ValueError('Use one diagnostic flag without other input')
        return {flags[0]: True}
    linked = 'video_url' in p
    if linked:
        if 'video_base64' in p:raise ValueError('Choose video_url or video_base64')
        transfer.validate_links(p)
        data = None
    else:
        if 'output_upload_url' in p or 'output_download_url' in p:raise ValueError('Transfer output requires video_url')
        encoded = p.get('video_base64')
        if not isinstance(encoded,str) or not 1 <= len(encoded) <= 4*((MAX_INPUT+2)//3):
            raise ValueError('Supply video_base64, maximum 6 MiB, or app transfer links')
        data=base64.b64decode(encoded,validate=True)
        if not data or len(data)>MAX_INPUT:raise ValueError('Video exceeds 6 MiB; use app file transfer')
    scale = p.get('scale', 4)
    if type(scale) is not int or scale not in (2, 4):
        raise ValueError('scale must be 2 or 4; official recommendation is 4')
    seed = p.get('seed', 1)
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError('seed must be an integer from 0 to 2147483647')
    compression = p.get('temporal_compression', 4)
    if type(compression) is not int or compression not in (1, 2, 4):
        raise ValueError('temporal_compression must be 1, 2 or 4')
    return dict(data=data, scale=scale, seed=seed, temporal_compression=compression, **({k:p[k] for k in ('video_url','output_upload_url','output_download_url')} if linked else {}))


def worker_child(folder, args):
    command = [sys.executable, str(HERE / 'engine.py'), '--directory', str(folder), *args]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True)
    try:
        output, _ = child.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
        raise TimeoutError('VSR processing exceeded 30 minutes; worker child was stopped')
    if child.returncode:
        raise RuntimeError('VSR runtime failed: ' + safe_text(output))
    result = json.loads((folder / 'result.json').read_text())
    if not isinstance(result, dict):
        raise RuntimeError('Invalid worker result')
    return result


def execute(p):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='flashvsr-') as name:
        folder = Path(name)
        if p.get('diagnose'):
            result = worker_child(folder, ['--diagnose'])
        elif p.get('smoke_test'):
            result = worker_child(folder, ['--smoke-test'])
        else:
            if p.get('video_url'):
                transfer.download(p['video_url'],folder/'input.mp4')
            else:
                (folder / 'input.mp4').write_bytes(p['data'])
            settings = {k: p[k] for k in ('scale', 'seed', 'temporal_compression')}
            (folder / 'settings.json').write_text(json.dumps(settings))
            result = worker_child(folder, [])
            video = folder / 'output.mp4'
            if p.get('output_upload_url'):
                transfer.upload(p['output_upload_url'],video)
                result.update(video_url=p['output_download_url'],mime_type='video/mp4',transport_version=2)
            else:
                if not video.is_file() or not 1<=video.stat().st_size<=MAX_OUTPUT:
                    raise RuntimeError('Output exceeds 6 MiB JSON limit; use app file transfer')
                result.update(video_base64=base64.b64encode(video.read_bytes()).decode(),mime_type='video/mp4')
        result.update(worker_version=VERSION, worker_total_seconds=round(time.monotonic()-started, 2))
        return result
