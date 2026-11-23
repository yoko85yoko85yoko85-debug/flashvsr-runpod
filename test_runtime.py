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
        self.assertEqual(r['worker_version'],'flashvsr-plus-turbo-v1')

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
                'testsrc2=size=128x128:rate=2','-t','9','-c:v','libx264',str(p)],check=True)
            with self.assertRaises(ValueError): media.metadata(p,4)

if __name__=='__main__': unittest.main()
