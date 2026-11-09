"""Comping regressions use new synthetic words and tones only."""
import unittest
import numpy as np
from pipeline_comp import plan_comp,apply_comp


class CompingTests(unittest.TestCase):
    def fixture(self, first_note=62., second_note=57.2):
        sr=24000
        time=np.arange(sr*2)/sr
        x=np.zeros(len(time))
        pt=np.arange(0,2,.01)
        f0=np.zeros(len(pt))
        for start,end,note in [(.2,.7,first_note),(1.,1.5,second_note)]:
            active=(time>=start)&(time<end)
            hz=440*2**((note-69)/12)
            x[active]=.1*np.sin(2*np.pi*hz*time[active])
            f0[(pt>=start)&(pt<end)]=hz
        guide=np.full(len(pt),57.)
        words=[{'word':'Lanterns','start':.2,'end':.7},
               {'word':'Lanterns','start':1.,'end':1.5}]
        return x,sr,words,pt,f0,guide

    def test_bad_first_repeat_is_muted_with_fades_second_is_preserved(self):
        x,sr,words,pt,f0,guide=self.fixture()
        original=x.copy()
        edits=plan_comp(x,sr,words,pt,f0,guide,lyrics='Lanterns shimmer')
        self.assertEqual(len(edits),1)
        self.assertLess(edits[0]['end'],1.)
        cleaned=apply_comp(x,sr,edits)
        np.testing.assert_array_equal(x,original)
        self.assertEqual(len(cleaned),len(x))
        self.assertTrue(np.all(cleaned[round(.3*sr):round(.6*sr)]==0))
        np.testing.assert_array_equal(cleaned[sr:round(1.5*sr)],x[sr:round(1.5*sr)])
        edit=edits[0]
        self.assertGreater(edit['fade_seconds'],0)
        a=round(edit['start']*sr);b=round(edit['end']*sr)
        self.assertEqual(cleaned[a],x[a])
        self.assertEqual(cleaned[b-1],x[b-1])

    def test_intended_lyric_repetition_is_not_removed(self):
        x,sr,words,pt,f0,guide=self.fixture()
        self.assertEqual(plan_comp(x,sr,words,pt,f0,guide,lyrics='Lanterns lanterns shimmer'),[])

    def test_similar_quality_repetitions_are_not_deleted(self):
        x,sr,words,pt,f0,guide=self.fixture(57.3,57.2)
        self.assertEqual(plan_comp(x,sr,words,pt,f0,guide,lyrics='Lanterns shimmer'),[])


if __name__=='__main__':unittest.main()
