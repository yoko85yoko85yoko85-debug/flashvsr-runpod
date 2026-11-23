FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg CC=/usr/bin/gcc CXX=/usr/bin/g++
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg python3-venv libgomp1 build-essential && rm -rf /var/lib/apt/lists/*
RUN python -m venv --system-site-packages /opt/flashvsr-venv
ENV PATH="/opt/flashvsr-venv/bin:${PATH}"
COPY requirements.txt /tmp/requirements.txt
RUN python -c "import torch; print('torch=='+torch.__version__.split('+')[0])" > /tmp/torch-constraint.txt && python -m pip install -c /tmp/torch-constraint.txt -r /tmp/requirements.txt
RUN git init /opt/FlashVSR_Plus && cd /opt/FlashVSR_Plus && git remote add origin https://github.com/OpenImagingLab/FlashVSR_Plus.git && git fetch --depth 1 origin b515ffc6dba57fdc9ad84c66cadd87d05f27590a && git checkout --detach FETCH_HEAD && rm -rf .git
RUN python -m pip install --no-deps /opt/FlashVSR_Plus && python -m pip install --no-deps /opt/FlashVSR_Plus/operators/flashvsr_attn
WORKDIR /app
RUN python -c "import ctypes, os, pathlib, subprocess, sysconfig, tempfile; d=pathlib.Path(tempfile.mkdtemp()); src=d/'check.c'; so=d/'check.so'; src.write_text('#include <Python.h>\nint check(void){return PY_MAJOR_VERSION;}\n'); subprocess.run([os.environ['CC'], '-shared', '-fPIC', '-I'+sysconfig.get_path('include'), str(src), '-o', str(so)], check=True); assert ctypes.CDLL(str(so)).check()==3; print('BUILD_C_COMPILER_OK: Python-header C shared library compiled and loaded')"
COPY download_models.py ./
RUN python download_models.py
COPY handler.py runtime.py media.py engine.py ./
RUN python -m py_compile handler.py runtime.py media.py engine.py && python -c "import torch, triton, flashvsr_attn; from inference import InferenceRequest; from models import attention_kernel_status; assert torch.__version__.startswith('2.7.1'); assert attention_kernel_status()['triton_block_sparse']; print('BUILD_IMPORT_OK; GPU inference not tested during build')"
CMD ["python", "-u", "handler.py"]
