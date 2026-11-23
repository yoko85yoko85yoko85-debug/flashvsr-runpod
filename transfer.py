"""Bounded video transfers to the app's exact HTTPS origin; no redirects."""
from pathlib import Path
import re
from urllib.parse import urlsplit,parse_qs
import requests
ORIGIN='https://frame-flashvsr-studio-nov23.sableflint8.chatgpt.site'
MAX_BYTES=64*1024**2
# Sites' edge rejects Python's default user-agent on media requests.
HEADERS={'User-Agent':'Mozilla/5.0'}

def valid_url(value,kind):
    if not isinstance(value,str):return False
    u=urlsplit(value)
    return (u.scheme+'://'+u.netloc==ORIGIN and not u.fragment
        and re.fullmatch('/media/[a-f0-9]{32}/'+kind,u.path) is not None
        and re.fullmatch('token=[a-f0-9]{64}',u.query) is not None)

def validate_links(p):
    for key,kind in [('video_url','input'),('output_upload_url','output'),('output_download_url','output')]:
        if not valid_url(p.get(key),kind):raise ValueError('Invalid video transfer link: '+key)
    a,b,c=(urlsplit(p[k]).path for k in ('video_url','output_upload_url','output_download_url'))
    if a.split('/')[2]!=b.split('/')[2] or b!=c:raise ValueError('Video transfer IDs must match')

def download(url,destination):
    with requests.get(url,headers=HEADERS,stream=True,timeout=(15,120),allow_redirects=False) as r:
        if r.status_code!=200:raise RuntimeError('Input video transfer failed ('+str(r.status_code)+')')
        length=int(r.headers.get('Content-Length','0'))
        if length>MAX_BYTES:raise ValueError('Input video exceeds 64 MiB')
        size=0
        with Path(destination).open('wb') as f:
            for block in r.iter_content(1024*1024):
                size+=len(block)
                if size>MAX_BYTES:raise ValueError('Input video exceeds 64 MiB')
                f.write(block)
        if not size:raise ValueError('Input video is empty')

def upload(url,path):
    path=Path(path)
    if not 0<path.stat().st_size<=MAX_BYTES:raise ValueError('Output video exceeds 64 MiB')
    with path.open('rb') as f:
        r=requests.put(url,data=f,headers={**HEADERS,'Content-Type':'video/mp4','Content-Length':str(path.stat().st_size)},timeout=(15,180),allow_redirects=False)
    try:
        if r.status_code!=200:raise RuntimeError('Output video transfer failed ('+str(r.status_code)+')')
    finally:r.close()
