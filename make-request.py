"""Create a local RunPod request JSON; never submits a paid GPU job."""
import argparse
import base64
import json
from pathlib import Path
import media
import runtime
p = argparse.ArgumentParser()
p.add_argument('video', type=Path)
p.add_argument('--output', type=Path, default=Path('request.json'))
p.add_argument('--scale', type=int, choices=(2,4), default=4)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--temporal-compression', type=int, choices=(1,2,4), default=4)
a = p.parse_args()
if a.video.stat().st_size > runtime.MAX_INPUT:
    raise SystemExit('Input exceeds 6 MiB')
meta = media.metadata(a.video, a.scale)
params = dict(video_base64=base64.b64encode(a.video.read_bytes()).decode(),
              scale=a.scale, seed=a.seed, temporal_compression=a.temporal_compression)
runtime.validate(params)
a.output.write_text(json.dumps({'input': params}))
print(f"Saved {a.output}: {meta['input_width']}x{meta['input_height']} -> {meta['width']}x{meta['height']}")
