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
