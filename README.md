# FlashVSR+ Turbo on RunPod Serverless

Dedicated video super-resolution endpoint. Uses the **official FlashVSR+ Turbo**
model and its official **Triton block-sparse streaming attention**. Does not load
H3, ComfyUI, RTX VSR, or Real-ESRGAN. FlashVSR+ is the official newer FlashVSR family
release, not the older FlashVSR v1.1 pipeline.

## RunPod deployment

Connect this repository, then select:

| Setting | Value |
|---|---|
| Branch | `main` |
| Dockerfile path | `/Dockerfile` |
| Build context | `.` |
| Endpoint type | Queue |
| GPU | L4 24GB for initial test (RTX 4090 24GB also a candidate) |
| GPU count | 1 |
| Active / minimum workers | 0 |
| Maximum workers | 1 |
| Container disk | 60GB |
| Allowed CUDA | 12.8 or newer compatible driver |
| Initialization timeout | 900 seconds |
| Execution timeout | 1200 seconds |
| Idle timeout | 5 seconds |
| Network/global volume | Not required |
| Cached model field / HF token / start command | Leave empty |

L4 is a **test target**, not a verified GPU benchmark for this wrapper.
The upstream FlashVSR+ README describes 24GB RTX 4090 operation. Original
FlashVSR's ~17FPS benchmark was measured on A100 and must not be promised on L4.
Do not compare inference-only time with full RunPod execution time.

### First: imports and device

Run a request in RunPod Requests:

```json
{"input":{"diagnose":true}}
```

Expected: `ok: true`, `inference_verified: false`, GPU and
`attention_kernel: triton_block_sparse`. This is **not model inference**.

### Second: actual inference smoke test

```json
{"input":{"smoke_test":true}}
```

Generates a synthetic 128x128, five-frame test pattern and runs the real Turbo
model at 4x to 512x512. Success requires frame count, FPS, and dimensions to match.
Expected: `inference_verified: true`, `smoke_test: true`. This consumes GPU time.
It verifies model execution, not perceptual quality on ordinary videos.

Send the result JSON after this succeeds; then connect the dedicated video app.
Do not use the RTX VSR app with this endpoint: its request settings differ.

## Video requests

```json
{
  "input": {
    "video_base64": "BASE64_OF_LOCAL_VIDEO",
    "scale": 4,
    "seed": 1,
    "temporal_compression": 4
  }
}
```

- `scale`: 2 or 4, default 4. The official family is primarily designed for 4x;
  2x is exposed for practical comparisons and may give different restoration quality.
- `seed`: fixed integer 0–2147483647, default 1.
- `temporal_compression`: 4 (default, faster), 2 or 1 (large-motion option,
  increased computation). It does not change output FPS or drop source frames.
- Max input: 6 MiB, 8 seconds, 480 frames, 1–60 FPS, constant frame rate, SDR,
  square pixels. Reject unsupported HDR and VFR rather than silently converting.
- Initial model-output limit: 2048px on each side. Examples: 512²→2048² at 4x,
  800²→1600² at 2x. This is a service limit, not a claim that all allowed clips
  will fit L4 VRAM. Source dimensions max1920 and area<=1920x1080.
- Exact source aspect and field of view are preserved. Resize bicubically, pad
  right/bottom edges to the required 128px model grid, process, then remove only
  the added border. No center crop of the original content.
- Source audio is preserved by AAC re-encoding, source FPS/frame count checked.
- Output MP4: H.264 CRF18, CPU x264 encoding, at most6MiB in this transport.
  Oversized output is an explicit error; no automatic quality reduction.
- Restoration can invent/change details. Compare with the source before use.

Generate request JSON locally (does not submit):

```bash
python make-request.py short.mp4 --scale 4 --output request.json
```

Result includes `video_base64`, dimensions, FPS, audio, seed, compression,
`model_load_seconds`, `inference_and_encode_seconds`, `processing_seconds`,
`worker_total_seconds`, exact upstream revision and attention kernel.
RunPod adds queue `delayTime` and full `executionTime` separately.
Each job uses an isolated GPU process; model loading repeats between jobs.
Weights are baked into the image, avoiding a fresh Hugging Face download per job.
Cold image pulls, first Triton JIT compilation and CUDA initialization still cost time.
The child process group is stopped after15min; tempfiles are removed.

## Reproducibility / attention

- Upstream code: OpenImagingLab/FlashVSR_Plus
  `b515ffc6dba57fdc9ad84c66cadd87d05f27590a`
- HF models: ruchangyao/FlashVSR_plus
  `46b79520946ac67dec94d1566490690953c7debc`
- Turbo DiT, LQ projection, tiny decoder: **7,016,472,997 bytes (~7.02GB)**.
  Downloaded at image build; SHA-256 and size verified in `download_models.py`.
  No VAE/base weights downloaded. Default released tiny decoder is used.
- PyTorch2.7.1 CUDA12.8; pinned upstream supports torch>=2.5,<2.8.
  An isolated venv inherits the image's torch; dependency installation constrains
  torch so it cannot silently change CUDA runtime.
- Explicit `triton_block_sparse`; an unavailable/broken sparse kernel is an error,
  not an automatic fallback to slow dense attention.
- Upstream README mentions FlashAttention installation for its Triton backend.
  At this pinned revision, the installed operator directly imports torch/triton
  and uses its own stream kernel; the selected path does **not** import flash_attn.
  We therefore omit the unrelated FlashAttention extension build. GPU smoke test
  is still required to confirm the complete CUDA execution path.
- Released Turbo hybrid attention, one diffusion step, temporal compression4,
  numeric decode interval4 (bounded streaming), cache offloadauto. No arbitrary
  sampler/model substitutions. No color-correction modification enabled.

## Verification status

CPU checks (8 passing) exercise validation, secret redaction, SHA checking, job cleanup,
real FFmpeg padding/cropping, output dimensions/FPS/frame count and source audio.
The transport test uses a fixture output and does not claim model inference.
Official upstream pipeline + Triton operator imports were also checked with CPU PyTorch2.7.1 and Triton3.3.1 (without FlashAttention). Base Docker image tag resolves.
Container image build and real GPU smoke must be run on RunPod before readiness
or speed/quality is claimed. No paid GPU test was submitted by this implementation.

```bash
python -m unittest -v test_runtime
```

## Sources / licenses

Official code and model are Apache-2.0. The upstream license remains in the image.

- https://github.com/OpenImagingLab/FlashVSR_Plus
- https://huggingface.co/ruchangyao/FlashVSR_plus
- https://github.com/OpenImagingLab/FlashVSR

## v1.1 FFmpeg compatibility fix

Use `/usr/bin/ffmpeg` and `/usr/bin/ffprobe` explicitly, matching the existing
ImageIO FFmpeg setting. Conda's older FFmpeg cannot shadow the system executable.
Both preprocessing and final encoding use `-vsync 0` (timestamp passthrough),
which also works with older system FFmpeg versions lacking `-fps_mode`.
Worker version: `flashvsr-plus-turbo-v1.1`. Model and attention settings unchanged.
