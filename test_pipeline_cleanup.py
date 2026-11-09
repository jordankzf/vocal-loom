"""Automated de-clicking must protect voiced tone and sustained consonant noise."""
import unittest
import numpy as np
from scipy.signal import butter,sosfiltfilt
from pipeline_cleanup import cleanup_audio


class CleanupTests(unittest.TestCase):
    def test_click_repair_preserves_voicing_and_fricative(self):
        sr=24000
        t=np.arange(sr*2)/sr
        rng=np.random.default_rng(7)
        x=rng.normal(0,.00004,len(t))
        v=((t>.15)&(t<.55))|((t>1.65)&(t<1.95))
        x[v]+=.08*np.sin(2*np.pi*140*t[v])
        s=sosfiltfilt(butter(3,5000,fs=sr,btype='highpass',output='sos'),rng.normal(0,.025,len(t)))
        fric=(t>.90)&(t<1.12)
        x[fric]+=s[fric]
        click=int(1.35*sr);x[click]+=.6
        pt=np.arange(0,2,.005)
        pv=((pt>.15)&(pt<.55))|((pt>1.65)&(pt<1.95))
        original=x.copy()
        y,a=cleanup_audio(x,sr,pitch_times=pt,voiced=pv,breath_reduction=.9)
        self.assertEqual(y.shape,x.shape)
        self.assertTrue(np.isfinite(y).all())
        self.assertLess(abs(y[click]),abs(x[click])*.1)
        np.testing.assert_array_equal(x,original)
        self.assertTrue(np.allclose(y[v],x[v],atol=1e-9))
        self.assertLess(abs(10*np.log10(np.sum(y[fric]**2)/np.sum(x[fric]**2))),.2)


if __name__=='__main__':unittest.main()
