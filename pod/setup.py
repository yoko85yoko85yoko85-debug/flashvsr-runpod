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
