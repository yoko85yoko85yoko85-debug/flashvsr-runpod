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
