from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'installer-prefix.sh').read_text()
for name in ('requirements.txt','engine.py','media.py','download_models.py','node.py','workflow.py','manage.py','setup.py'):
    marker='FLASH_EMBED_'+name.replace('.','_').upper()
    s+='\ncat > "$FLASHVSR_POD_DIR/'+name+'" <<\''+marker+"'\n"+(root/name).read_text()+'\n'+marker+'\n'
s+='''
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
 src.write_text('#include <Python.h>\\nint check(void){return PY_MAJOR_VERSION;}\\n')
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
'''
(root/'install.sh').write_text(s)
