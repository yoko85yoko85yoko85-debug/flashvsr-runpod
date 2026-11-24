import importlib.util,json,sys,tempfile,types,unittest,subprocess
from pathlib import Path
from unittest.mock import patch,MagicMock
import engine,media,workflow,manage

class Tests(unittest.TestCase):
    def adapter(self):
        mm=types.ModuleType('comfy.model_management');mm.throw_exception_if_processing_interrupted=lambda:None
        mm.unload_all_models=MagicMock();mm.soft_empty_cache=MagicMock()
        comfy=types.ModuleType('comfy');comfy.model_management=mm
        latest=types.ModuleType('comfy_api.latest');latest.ComfyExtension=object
        latest.io=types.SimpleNamespace(ComfyNode=object);latest.ui=types.SimpleNamespace()
        latest.InputImpl=types.SimpleNamespace();latest.Types=types.SimpleNamespace()
        folder=types.ModuleType('folder_paths')
        patches={'folder_paths':folder,'comfy':comfy,'comfy.model_management':mm,'comfy_api':types.ModuleType('comfy_api'),'comfy_api.latest':latest}
        with patch.dict(sys.modules,patches):
            spec=importlib.util.spec_from_file_location('flash_adapter',Path(__file__).parent/'node.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module,mm

    def test_source_copy_keeps_original(self):
        m,_=self.adapter()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.webm';source.write_bytes(b'video fixture');job=root/'job';job.mkdir()
            m.copy_source(types.SimpleNamespace(get_stream_source=lambda:str(source)),job)
            self.assertEqual((job/'input.mp4').read_bytes(),source.read_bytes())
            (job/'input.mp4').unlink();self.assertTrue(source.exists())

    def test_cancel_terminates_runtime_group(self):
        m,mm=self.adapter();mm.throw_exception_if_processing_interrupted=MagicMock(side_effect=RuntimeError('interrupted'))
        child=MagicMock();child.poll.return_value=None
        with tempfile.TemporaryDirectory() as tmp,patch.object(m.subprocess,'Popen',return_value=child) as popen,patch.object(m,'stop_child') as stop:
            with self.assertRaisesRegex(RuntimeError,'interrupted'):m.run_engine(Path(tmp))
            stop.assert_called_once_with(child)
            env=popen.call_args.kwargs['env']
            self.assertTrue(env['PATH'].startswith('/opt/flashvsr-pod/venv/bin:'))
            self.assertEqual(env['CC'],'/usr/bin/gcc')

    def test_workflow(self):
        graph=workflow.build();nodes={n['id']:n for n in graph['nodes']}
        self.assertEqual(nodes[1]['type'],'LoadVideo');self.assertEqual(nodes[2]['type'],'FlashVSRPodTurbo')
        self.assertEqual(nodes[2]['widgets_values'][:2],['2倍','標準（速度優先）'])
        self.assertEqual(graph['links'],[[1,1,0,2,0,'VIDEO']])

    def test_actual_ffmpeg_geometry_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);source=d/'input.mp4';prepared=d/'prepared.mp4';out=d/'out.mp4'
            subprocess.run(['/usr/bin/ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=96x128:rate=12',
                '-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t','1','-c:v','libx264','-c:a','aac',str(source)],check=True)
            meta=media.metadata(source,2);engine.prepare(source,prepared,meta);engine.finalize(prepared,source,out,meta)
            self.assertTrue(engine.verify_video(out,meta));self.assertEqual((meta['width'],meta['height']),(192,256))

    def test_local_model_path_and_versions(self):
        self.assertEqual(engine.ROOT,Path(__file__).parent/'FlashVSR_Plus')
        self.assertEqual(engine.UPSTREAM_REVISION,'b515ffc6dba57fdc9ad84c66cadd87d05f27590a')
        text=(Path(__file__).parent/'install.sh').read_text()
        self.assertNotIn('runpod==',text)
        self.assertIn('torch==2.7.1',text);self.assertIn('--index-url https://download.pytorch.org/whl/cu128',text)
        self.assertIn('build-essential',text)

    def test_restart_auth_boundary(self):
        with patch.object(manage,'is_runpod_template',return_value=True):
            self.assertEqual(manage.connection_config(8188),(18188,'127.0.0.1',True))
        with patch.object(manage.urllib.request,'urlopen',return_value=MagicMock()):
            with self.assertRaisesRegex(RuntimeError,'bypass'):manage.check_proxy(8188)

    def test_embedded_scripts_compile(self):
        text=(Path(__file__).parent/'install.sh').read_text()
        markers=['PREFLIGHT','COMPILER','FLASH_EMBED_ENGINE_PY','FLASH_EMBED_MEDIA_PY','FLASH_EMBED_DOWNLOAD_MODELS_PY','FLASH_EMBED_NODE_PY','FLASH_EMBED_WORKFLOW_PY','FLASH_EMBED_MANAGE_PY','FLASH_EMBED_SETUP_PY']
        for marker in markers:
            code=text.split("<<'"+marker+"'\n",1)[1].split('\n'+marker+'\n',1)[0]
            compile(code,'embedded-'+marker,'exec')

if __name__=='__main__':unittest.main()
