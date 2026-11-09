"""End-to-end CLI tests use only a new synthetic five-note song."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import parselmouth
import soundfile as sf

from synthetic_fixture import create_fixture


class WorkflowTests(unittest.TestCase):
    def temporary_folder(self):
        base=os.environ.get('VOCAL_PIPELINE_TEST_TMP')
        if base:
            Path(base).mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    def render_fixture(self,mode,lead_mode='tuned',*,extra=None,expected_channels=2):
        folder=self.temporary_folder()
        fixture=create_fixture(folder/'inputs')
        before={k:hashlib.sha256(fixture[k].read_bytes()).hexdigest()
                for k in ('audio','midi','lyrics')}
        output=folder/'render'
        command=[sys.executable,str(Path(__file__).with_name('vocal_pipeline.py')),
                 'render','--recording',str(fixture['audio']),'--midi',str(fixture['midi']),
                 '--lyrics',str(fixture['lyrics']),'--mode',mode,'--lead-mode',lead_mode,
                 '--no-asr','--out',str(output)]
        command.extend(extra or [])
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        info=sf.info(output/'finished.wav')
        samples,sr=sf.read(output/'finished.wav',always_2d=True)
        self.assertEqual(report['status'],'complete')
        self.assertEqual(info.subtype,'PCM_24')
        self.assertEqual(info.channels,expected_channels)
        self.assertTrue(np.isfinite(samples).all())
        self.assertGreater(info.duration,2.)
        self.assertLess(info.duration,8.)
        self.assertLessEqual(report['render']['true_peak_4x_dbfs'],-1.09)
        self.assertTrue(any('Transcription disabled' in w for w in report['warnings']))
        self.assertEqual(json.loads((output/'status.json').read_text())['status'],'complete')
        for k,digest in before.items():
            self.assertEqual(hashlib.sha256(fixture[k].read_bytes()).hexdigest(),digest)
        return output,report

    def test_solo_moves_slightly_sharp_vowels_toward_score(self):
        output,report=self.render_fixture('solo')
        lead,sr=sf.read(output/'stages'/'lead.wav')
        automation=np.load(output/'stages'/'automation.npz')
        p=parselmouth.Sound(lead,sr).to_pitch_ac(time_step=.005,pitch_floor=150,pitch_ceiling=450)
        t=p.xs();hz=p.selected_array['frequency']
        target=np.interp(t,automation['output_time'],automation['effective_lead_midi'])
        note=np.full_like(hz,np.nan);note[hz>0]=69+12*np.log2(hz[hz>0]/440)
        stable=(hz>0)&(abs(np.gradient(target))<.02)
        error=100*abs(note[stable]-target[stable])
        self.assertGreater(len(error),150)
        self.assertLess(np.median(error),15.,'Corrected notes should improve the 35-cent source detuning')
        self.assertEqual(report['render']['harmony_count'],0)

    def test_original_lead_duet_keeps_lead_and_renders_one_harmony(self):
        output,report=self.render_fixture('duet','original')
        self.assertTrue(report['render']['lead_samples_preserved'])
        self.assertTrue(report['render']['lead_timing_preserved'])
        self.assertEqual(report['render']['harmony_count'],1)
        lead,sr=sf.read(output/'stages'/'lead.wav')
        harmony,hsr=sf.read(output/'stages'/'harmony_1.wav')
        self.assertEqual(sr,hsr)
        self.assertEqual(len(lead),len(harmony))
        self.assertGreater(np.max(abs(harmony)),.001)
        self.assertTrue(np.isfinite(harmony).all())
        # The original-lead mode must retain the deliberately sharp source pitch.
        p=parselmouth.Sound(lead,sr).to_pitch_ac(time_step=.005,pitch_floor=150,pitch_ceiling=450)
        f0=p.selected_array['frequency'];midi=69+12*np.log2(f0[f0>0]/440)
        error=(midi-np.round(midi))*100
        self.assertGreater(np.median(error),20.)
        self.assertLess(np.median(error),48.)

    def test_dry_solo_uses_source_clock_and_only_constant_export_gain(self):
        output,report=self.render_fixture('solo',extra=['--processing','dry',
                              '--pitch-style','transparent'],expected_channels=1)
        dry,sr=sf.read(output/'finished.wav')
        lead,lsr=sf.read(output/'stages'/'lead.wav')
        self.assertEqual(sr,lsr)
        self.assertEqual(dry.ndim,1)
        self.assertEqual(len(dry),len(lead))
        export=report['render']['dry_export']
        self.assertEqual(report['render']['processing'],'dry')
        for effect in ['eq','compression','reverb','timing_edits']:
            self.assertFalse(export[effect])
        self.assertEqual(export['channels'],1)
        gain=export['gain']
        # Two PCM24 quantizations are the only allowed difference between the
        # lead stem times its reported scalar gain and the exported dry sample.
        np.testing.assert_allclose(dry,lead*gain,rtol=0,atol=2e-7*(1+abs(gain)))
        automation=np.load(output/'stages'/'automation.npz')
        np.testing.assert_allclose(automation['output_time'],automation['source_time'],atol=1e-9)
        np.testing.assert_allclose(automation['duration_ratio'],1.,atol=1e-9)
        first,last=report['source_trim_seconds']
        self.assertAlmostEqual(len(dry)/sr,last-first,delta=2/sr)

    def test_file_in_place_of_output_directory_fails_without_traceback(self):
        folder=self.temporary_folder()
        fixture=create_fixture(folder/'inputs')
        output=folder/'blocked-output'
        output.write_text('Existing file must remain unchanged',encoding='utf-8')
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('vocal_pipeline.py')),
                  'render','--recording',str(fixture['audio']),'--midi',str(fixture['midi']),
                  '--no-asr','--out',str(output)],capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('Traceback',result.stdout+result.stderr)
        self.assertIn('Could not finish',result.stdout+result.stderr)
        self.assertEqual(output.read_text(encoding='utf-8'),'Existing file must remain unchanged')


if __name__=='__main__':
    unittest.main()
