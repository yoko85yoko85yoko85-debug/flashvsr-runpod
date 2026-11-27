#!/usr/bin/env bash
set -euo pipefail
ROOT="${FLASHVSR_POD_DIR:-/opt/flashvsr-pod}"
[[ -x "$ROOT/venv/bin/python" && -f "$ROOT/target.json" && -f "$ROOT/restart-comfyui.sh" ]] || { echo 'FlashVSR Pod installation was not found. Run the current full installer on a new Pod.'; exit 1; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BASE='https://raw.githubusercontent.com/yoko85yoko85yoko85-debug/flashvsr-runpod/4903c424e84ba6c47de55dff698557708b9aff38/pod'
for file in node.py media.py engine.py FlashVSR-Pod-Turbo.json; do
 curl -fL --retry 3 "$BASE/$file" -o "$TMP/$file"
done
"$ROOT/venv/bin/python" -m py_compile "$TMP/node.py" "$TMP/media.py" "$TMP/engine.py"
"$ROOT/venv/bin/python" - "$ROOT" "$TMP" <<'UPDATE'
import json,shutil,sys,time
from pathlib import Path
root=Path(sys.argv[1]);tmp=Path(sys.argv[2])
comfy=Path(json.loads((root/'target.json').read_text())['comfy'])
module=comfy/'custom_nodes/flashvsr_pod_turbo'
if not (module/'__init__.py').is_file():raise SystemExit('Installed FlashVSR node was not found')
backup=root/'backups'/('scale15-'+str(time.time_ns()));backup.mkdir(parents=True)
for source,name in [(module/'__init__.py','node.py'),(root/'media.py','media.py'),(root/'engine.py','engine.py')]:
 if source.exists():shutil.copy2(source,backup/name)
text=(tmp/'node.py').read_text().replace("RUNTIME=Path(os.environ.get('FLASHVSR_POD_DIR','/opt/flashvsr-pod'))",'RUNTIME=Path('+repr(str(root))+')')
(module/'__init__.py').write_text(text)
(root/'node.py').write_text(text)
for name in ['media.py','engine.py']:shutil.copy2(tmp/name,root/name)
# A separate workflow leaves any user-edited original workflow untouched.
workflow=comfy/'user/default/workflows';workflow.mkdir(parents=True,exist_ok=True)
shutil.copy2(tmp/'FlashVSR-Pod-Turbo.json',workflow/'FlashVSR-Pod-Turbo-1_5.json')
print('Added 1.5x. Previous code backup:',backup)
print('1280x720 -> 1920x1080. No models or Python packages downloaded.')
UPDATE
bash "$ROOT/restart-comfyui.sh"
echo 'Reload ComfyUI and select 1.5倍 in scale. Your existing workflow also uses the new choices.'
