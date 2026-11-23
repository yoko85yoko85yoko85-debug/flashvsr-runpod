import base64
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import engine
import media
import runtime
import handler
from download_models import verify

class Tests(unittest.TestCase):
    def request(self, **kwargs):
        return dict(video_base64=base64.b64encode(b'test').decode(), **kwargs)

    def test_validation(self):
        self.assertEqual(runtime.validate(self.request())['scale'],4)
        for kw in ({'scale':True},{'scale':3},{'seed':-1},{'temporal_compression':3}):
            with self.assertRaises(ValueError): runtime.validate(self.request(**kw))
        with self.assertRaises(ValueError): runtime.validate({'diagnose':True,'smoke_test':True})
        with self.assertRaises(ValueError): runtime.validate({'smoke_test':True,'scale':4})
        self.assertEqual(runtime.validate({'smoke_test':True}), {'smoke_test':True})

    def test_diagnostic_and_cleanup(self):
        folders=[]
        def child(folder,args):
            folders.append(folder)
            self.assertEqual(args,['--diagnose'])
            return {'ok':True,'inference_verified':False}
        with patch.object(runtime,'worker_child',side_effect=child):
            self.assertTrue(runtime.execute({'diagnose':True})['ok'])
        self.assertFalse(folders[0].exists())

    def test_failure_cleanup(self):
        folders=[]
        def child(folder,args):
            folders.append(folder)
            raise RuntimeError('fixture failure')
        with patch.object(runtime,'worker_child',side_effect=child):
            with self.assertRaises(RuntimeError): runtime.execute({'smoke_test':True})
        self.assertFalse(folders[0].exists())

    def test_transport_output(self):
        def child(folder,args):
            self.assertEqual(json.loads((folder/'settings.json').read_text())['scale'],2)
            (folder/'output.mp4').write_bytes(b'fixture')
            return {'ok':True,'inference_verified':True}
        with patch.object(runtime,'worker_child',side_effect=child):
            r=runtime.execute(runtime.validate(self.request(scale=2)))
        self.assertEqual(base64.b64decode(r['video_base64']),b'fixture')
        self.assertEqual(r['worker_version'],'flashvsr-plus-turbo-v1.3')

    def test_digest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fixture'; p.write_bytes(b'fixture')
            verify(p,7,hashlib.sha256(b'fixture').hexdigest())
            with self.assertRaises(RuntimeError): verify(p,7,'wrong')

    def test_redaction(self):
        with patch.dict('os.environ', {'HF_TOKEN':'fixture-secret'}):
            self.assertNotIn('fixture-secret',runtime.safe_text('fixture-secret'))

    def test_real_video_geometry_audio(self):
        # Exercise actual decoding, border padding/crop and audio handling.
        # The middle file stands in for a model output; no GPU claim is made.
        for audio in (True,False):
            with tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp); source=d/'input.mp4'; prep=d/'prepared.mp4'; out=d/'out.mp4'
                cmd=['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=150x100:rate=12']
                if audio: cmd += ['-f','lavfi','-i','sine=frequency=440:sample_rate=48000']
                cmd += ['-t','1','-c:v','libx264','-pix_fmt','yuv420p']
                if audio: cmd += ['-c:a','aac']
                subprocess.run(cmd+[str(source)],check=True)
                meta=media.metadata(source,4)
                self.assertEqual((meta['width'],meta['height']),(600,400))
                self.assertEqual((meta['model_width'],meta['model_height']),(640,512))
                engine.prepare(source,prep,meta)
                p=media.probe(prep)['streams'][0]
                self.assertEqual((p['width'],p['height']),(640,512))
                engine.finalize(prep,source,out,meta)
                self.assertEqual(engine.verify_video(out,meta),audio)
                with self.assertRaises(ValueError): media.metadata(source,32)

    def test_video_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'long.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
                'testsrc2=size=128x128:rate=2','-t','31','-c:v','libx264',str(p)],check=True)
            with self.assertRaises(ValueError): media.metadata(p,4)


class LongVideoTests(unittest.TestCase):
    def links(self):
        import transfer
        base=transfer.ORIGIN+'/media/'+'a'*32
        return dict(video_url=base+'/input?token='+'b'*64,
                    output_upload_url=base+'/output?token='+'c'*64,
                    output_download_url=base+'/output?token='+'d'*64)

    def test_links_validation(self):
        p=runtime.validate(self.links())
        self.assertIsNone(p['data'])
        for key,value in [('video_url','http://127.0.0.1/secret'),
                          ('output_upload_url','https://example.com/output')]:
            with self.assertRaises(ValueError):runtime.validate({**self.links(),key:value})
        with self.assertRaises(ValueError):runtime.validate({**self.links(),'video_base64':'dGVzdA=='})
        with self.assertRaises(ValueError):runtime.validate({**self.links(),'output_download_url':self.links()['output_download_url'].replace('a'*32,'e'*32)})
        self.assertNotIn('b'*64,runtime.safe_text(self.links()['video_url']))

    def test_linked_transport_cleanup(self):
        def download(url,path):path.write_bytes(b'input fixture')
        def child(folder,args):
            self.assertEqual((folder/'input.mp4').read_bytes(),b'input fixture')
            (folder/'output.mp4').write_bytes(b'output fixture')
            return {'ok':True,'inference_verified':True}
        with patch.object(runtime.transfer,'download',side_effect=download),patch.object(runtime.transfer,'upload') as upload,patch.object(runtime,'worker_child',side_effect=child):
            result=runtime.execute(runtime.validate(self.links()))
        self.assertEqual(result['video_url'],self.links()['output_download_url'])
        self.assertNotIn('video_base64',result)
        self.assertNotIn('output_upload_url',result)
        self.assertFalse(upload.call_args.args[1].parent.exists())

    def test_30_seconds_60fps_full_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'30s.mp4'
            subprocess.run(['/usr/bin/ffmpeg','-v','error','-y','-f','lavfi','-i',
              'testsrc2=size=128x128:rate=60','-t','30','-c:v','libx264','-preset','ultrafast',str(p)],check=True)
            m=media.metadata(p,2)
            self.assertEqual(m['frames'],1800)
            self.assertEqual(m['duration_seconds'],30)

    def test_transfer_no_redirect_and_byte_limit(self):
        import transfer
        from unittest.mock import MagicMock
        response=MagicMock();response.status_code=302;response.headers={}
        response.__enter__.return_value=response
        with tempfile.TemporaryDirectory() as tmp,patch.object(transfer.requests,'get',return_value=response) as get:
            with self.assertRaises(RuntimeError):transfer.download(self.links()['video_url'],Path(tmp)/'input')
            self.assertFalse(get.call_args.kwargs['allow_redirects'])
            response.status_code=200;response.iter_content.return_value=[b'12345',b'67890']
            with patch.object(transfer,'MAX_BYTES',8):
                with self.assertRaises(ValueError):transfer.download(self.links()['video_url'],Path(tmp)/'input')

if __name__=='__main__': unittest.main()
