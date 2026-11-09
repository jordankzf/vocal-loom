import unittest
from types import SimpleNamespace
import numpy as np
from pipeline_alignment import Performance,align_score


class SyntheticScore:
    def __init__(self,pitches,durations=None):
        self.pitches=np.asarray(pitches,dtype=float)
        self.edges=np.r_[0,np.cumsum(np.ones(len(pitches)) if durations is None else durations)]
        self.duration=float(self.edges[-1])
    def melody_at(self,times,**kwargs):
        indices=np.searchsorted(self.edges,np.asarray(times),side='right')-1
        out=np.full(len(times),np.nan)
        valid=(indices>=0)&(indices<len(self.pitches))
        out[valid]=self.pitches[indices[valid]]
        return SimpleNamespace(pitches=out,confidence=1.,warnings=())


def performance(score,durations=None,prefix=None,wrong=None):
    count=len(score.pitches) if prefix is None else prefix
    durations=np.ones(count) if durations is None else np.asarray(durations[:count])
    edges=.65+np.r_[0,np.cumsum(durations)]
    t=np.arange(0,edges[-1]+.4,.01);f=np.zeros(len(t));strength=np.zeros(len(t))
    for i in range(count):
        mask=(t>=edges[i])&(t<edges[i+1]);note=score.pitches[i]-24
        if wrong and i in wrong:note+=wrong[i]
        f[mask]=440*2**((note-69)/12);strength[mask]=.99
    return Performance(t,f,strength),edges


class AlignmentTests(unittest.TestCase):
    def test_rubato_has_local_timing_anchors(self):
        score=SyntheticScore([69,71,73,68,76,74,72,75,70,67,73,69])
        p,edges=performance(score,[1.3,.75,1.45,.7,1.35,.8,1.2,.75,1.3,.7,1.25,.8])
        aligned=align_score(score,p)
        mapped=np.interp(edges[:-1]+.2,aligned['source_seconds'],aligned['midi_seconds'])
        expected=score.edges[:-1]+.2/np.diff(edges)
        self.assertLess(np.median(abs(mapped-expected)),.15)
        self.assertTrue(np.all(np.diff(aligned['midi_seconds'])>0))
        rate=np.diff(aligned['midi_seconds'])/np.diff(aligned['source_seconds'])
        self.assertGreaterEqual(rate.min(),.249)
        self.assertLessEqual(rate.max(),4.001)

    def test_wrong_note_does_not_move_to_matching_later_note(self):
        score=SyntheticScore([69,71,73,68,76,74,72,75,70,67,73,69])
        p,edges=performance(score,wrong={4:-3})
        aligned=align_score(score,p)
        actual=np.interp(edges[4]+.5,aligned['source_seconds'],aligned['midi_seconds'])
        self.assertAlmostEqual(actual,4.5,delta=.25)
        self.assertTrue(any(abs(x['error_semitones'])>2 for x in aligned['pitch_mismatches']))


    def test_large_tempo_drift_does_not_reinterpret_wrong_note(self):
        notes=[69,71,73,68,76,74,72,75,70,67,73,69,
               72,76,71,68,74,70,73,67,75,72,69,76]
        score=SyntheticScore(notes)
        durations=np.r_[np.ones(8)*1.35,np.ones(8)*.7,np.ones(8)*1.3]
        p,edges=performance(score,durations,wrong={12:3})
        aligned=align_score(score,p)
        mapped=np.interp(edges[:-1]+durations/2,
                         aligned['source_seconds'],aligned['midi_seconds'])
        self.assertLess(np.percentile(abs(mapped-np.arange(24)-.5),90),.3)
        self.assertAlmostEqual(mapped[12],12.5,delta=.25)
        self.assertEqual(aligned['alignment_method'],'bounded_monotone_note_sequence')

    def test_abandoned_opening_is_flagged_and_continuation_aligned(self):
        score=SyntheticScore([69,71,73,68,76,74,72,75,70,67,73,69])
        p,edges=performance(SyntheticScore([69,71]+list(score.pitches)))
        aligned=align_score(score,p)
        self.assertTrue(aligned['restart_candidates'])
        start=edges[2]
        self.assertAlmostEqual(np.interp(start+.5,aligned['source_seconds'],
                                        aligned['midi_seconds']),.5,delta=.2)

    def test_prefix_is_not_stretched_to_score_end(self):
        score=SyntheticScore([69,71,73,68,76,74,72,75,70,67,73,69,74,71,68,72,76,69,73,70])
        p,edges=performance(score,prefix=8)
        aligned=align_score(score,p)
        end=np.interp(edges[-1],aligned['source_seconds'],aligned['midi_seconds'])
        self.assertAlmostEqual(end,8,delta=.3)
        self.assertLess(aligned['inferred_score_end_seconds'],10)


if __name__=='__main__':
    unittest.main()
