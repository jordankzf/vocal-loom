"""Signal-level integration tests for the generic resynthesis backend."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import parselmouth

from pipeline_dsp import render_vocal


class DSPTests(unittest.TestCase):
    def fixture(self, rate, frequency=220.):
        duration = 1.25
        t = np.arange(round(duration*rate))/rate
        # Harmonic-rich synthetic vowel, with soft edges to avoid click transients.
        x = .13*np.sin(2*np.pi*frequency*t)+.045*np.sin(2*np.pi*frequency*2*t)+.02*np.sin(2*np.pi*frequency*3*t)
        fade = min(1000, len(x)//5)
        x[:fade] *= np.linspace(0, 1, fade)
        x[-fade:] *= np.linspace(1, 0, fade)
        times = np.arange(.03, duration-.03, .01)
        performance = SimpleNamespace(times=times, f0=np.full(len(times), frequency))
        return x, performance

    def test_solo_pitch_and_unchanged_duration_44100(self):
        x, p = self.fixture(44100)
        result = render_vocal(x, 44100, p, p.times, np.full(len(p.times), 59.), mode="solo", pitch_style="tight")
        self.assertEqual(result.stems["lead"].shape, x.shape)
        self.assertTrue(np.isfinite(result.audio).all())
        pitch = parselmouth.Sound(result.stems["lead"], 44100).to_pitch_ac(pitch_floor=100, pitch_ceiling=500)
        f0 = pitch.selected_array["frequency"]
        target = 440*2**((59-69)/12)
        self.assertLess(abs(1200*np.log2(np.median(f0[f0>0])/target)), 15)
        self.assertLessEqual(result.report["true_peak_4x_dbfs"], -1.09)

    def test_duet_48000_retimes_without_repitching(self):
        x, p = self.fixture(48000)
        result = render_vocal(x, 48000, p, p.times*1.1, np.full(len(p.times), 57.),
                              harmony_midi=np.full((1, len(p.times)), 64.), mode="duet", pitch_style="tight")
        self.assertEqual(len(result.stems["lead"]), round(len(x)*1.1))
        self.assertEqual(set(result.stems), {"lead", "harmony_1"})
        self.assertEqual(result.audio.shape[1], 2)
        y = result.stems["harmony_1"]
        middle = y[len(y)//4:3*len(y)//4]
        spectrum = abs(np.fft.rfft(middle*np.hanning(len(middle))))
        peak = np.fft.rfftfreq(len(middle), 1/48000)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak, 329.63, delta=5)

    def test_silence_and_invalid_clock_rejected(self):
        x, p = self.fixture(44100)
        with self.assertRaisesRegex(ValueError, "silent"):
            render_vocal(np.zeros_like(x), 44100, p, p.times, np.full(len(p.times), 57.))
        with self.assertRaisesRegex(ValueError, "increase"):
            render_vocal(x, 44100, p, np.zeros_like(p.times), np.full(len(p.times), 57.))

    def test_stack_mutes_missing_harmony_without_duplicating_noise(self):
        x, p = self.fixture(48000)
        harmonies = np.vstack((np.full(len(p.times), np.nan),
                               np.full(len(p.times), 64.), np.full(len(p.times), 67.)))
        result = render_vocal(x, 48000, p, p.times, np.full(len(p.times), 57.),
                              harmony_midi=harmonies, mode="vocoder", pitch_style="tight")
        self.assertEqual(len(result.stems), 4)
        self.assertTrue(np.all(result.stems["harmony_1"] == 0))
        self.assertGreater(np.max(abs(result.stems["harmony_2"])), .01)
        self.assertTrue(np.isfinite(result.audio).all())

    def test_original_lead_is_exact_and_harmony_still_tuned(self):
        x, p = self.fixture(48000)
        # A substantial proposed timing edit and different lead note must both
        # be ignored, while the requested E4 harmony is still synthesized.
        with patch("pipeline_dsp.praat_voice", side_effect=AssertionError("Lead resynthesis must be bypassed")):
            result = render_vocal(x, 48000, p, p.times*1.2, np.full(len(p.times), 64.),
                                  harmony_midi=np.full((1, len(p.times)), 64.),
                                  mode="duet", lead_mode="original", pitch_style="tight", harmony_engine="world")
        np.testing.assert_array_equal(result.stems["lead"], x)
        self.assertEqual(len(result.stems["lead"]), len(x))
        self.assertAlmostEqual(result.report["corrected_vocal_duration_seconds"], len(x)/48000)
        np.testing.assert_allclose(result.automation["output_time"], p.times, atol=1e-12)
        self.assertTrue(result.report["lead_samples_preserved"])
        y = result.stems["harmony_1"]
        middle = y[len(y)//4:3*len(y)//4]
        spectrum = abs(np.fft.rfft(middle*np.hanning(len(middle))))
        peak = np.fft.rfftfreq(len(middle), 1/48000)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak, 329.63, delta=5)

    def test_harmonies_follow_flat_lead_including_cents(self):
        actual = 220*2**(-.4/12)
        x, p = self.fixture(48000, actual)
        intended_harmony = np.full((1, len(p.times)), 61.)
        before = intended_harmony.copy()
        result = render_vocal(x, 48000, p, p.times*1.1, np.full(len(p.times), 57.),
                              harmony_midi=intended_harmony, mode="duet",
                              lead_mode="original", harmony_follow="lead", pitch_style="tight")
        np.testing.assert_array_equal(result.stems["lead"], x)
        np.testing.assert_array_equal(intended_harmony, before)
        np.testing.assert_allclose(result.automation["effective_harmony_midi"], 60.6, atol=1e-9)
        np.testing.assert_allclose(result.automation["harmony_follow_offset"], -.4, atol=1e-9)
        self.assertEqual(result.report["harmony_follow"], "lead")
        y = result.stems["harmony_1"]
        middle = y[len(y)//4:3*len(y)//4]
        spectrum = abs(np.fft.rfft(middle*np.hanning(len(middle))))
        measured = np.fft.rfftfreq(len(middle), 1/48000)[np.argmax(spectrum)]
        expected = 440*2**((60.6-69)/12)
        self.assertAlmostEqual(measured, expected, delta=2)

    def test_harmonies_follow_large_wrong_note_and_reject_tracker_spike(self):
        actual_e3 = 440*2**((52-69)/12)
        x, p = self.fixture(44100, actual_e3)
        p.strength = np.ones(len(p.times))
        p.f0[len(p.f0)//2] *= 2  # isolated octave tracker error, not a sung note
        original_targets = np.full((1, len(p.times)), 56.)
        result = render_vocal(x, 44100, p, p.times, np.full(len(p.times), 49.),
                              harmony_midi=original_targets, mode="duet",
                              lead_mode="original", harmony_follow="lead", pitch_style="tight")
        np.testing.assert_array_equal(result.stems["lead"], x)
        np.testing.assert_allclose(result.automation["effective_harmony_midi"], 59., atol=1e-9)
        np.testing.assert_allclose(result.automation["harmony_follow_offset"], 3., atol=1e-9)
        np.testing.assert_array_equal(original_targets, np.full_like(original_targets, 56.))
        with self.assertRaisesRegex(ValueError, "original"):
            render_vocal(x, 44100, p, p.times, np.full(len(p.times), 49.),
                          harmony_midi=original_targets, mode="duet", harmony_follow="lead")

    def vibrato_fixture(self, rate=44100, detune=.4):
        duration = 2.4
        audio_time = np.arange(round(duration*rate))/rate
        midi = 57+detune+.2*np.sin(2*np.pi*5*audio_time)
        frequency = 440*2**((midi-69)/12)
        phase = 2*np.pi*np.cumsum(frequency)/rate
        x = .14*np.sin(phase)+.04*np.sin(phase*2)+.015*np.sin(phase*3)
        times = np.arange(.03, duration-.03, .005)
        f0 = np.interp(times, audio_time, frequency)
        return x, SimpleNamespace(times=times, f0=f0), np.full(len(times), 57.)

    def test_transparent_corrects_center_and_keeps_vibrato(self):
        x, p, target = self.vibrato_fixture()
        result = render_vocal(x, 44100, p, p.times, target, mode="solo", pitch_style="transparent")
        effective = result.automation["effective_lead_midi"]
        interior = (p.times > .3) & (p.times < 2.1)
        self.assertLess(abs(np.median(effective[interior])-57), .10)
        source_midi = 69+12*np.log2(p.f0/440)
        before = np.percentile(source_midi[interior], 95)-np.percentile(source_midi[interior], 5)
        after = np.percentile(effective[interior], 95)-np.percentile(effective[interior], 5)
        self.assertGreater(after/before, .8)
        pitch = parselmouth.Sound(result.stems["lead"], 44100).to_pitch_ac(
            time_step=.005, pitch_floor=100, pitch_ceiling=400)
        measured = pitch.selected_array["frequency"]
        good = (pitch.xs() > .3) & (pitch.xs() < 2.1) & (measured > 0)
        cents = 1200*np.log2(measured[good]/220)
        self.assertLess(abs(np.median(cents)), 10)
        amplitude = (np.percentile(cents, 95)-np.percentile(cents, 5))/2
        self.assertGreater(amplitude, 16)

    def test_transparent_bypasses_intune_waveform_and_corrects_wrong_note(self):
        x, p, target = self.vibrato_fixture(detune=0)
        result = render_vocal(x, 44100, p, p.times, target, mode="solo", pitch_style="transparent")
        np.testing.assert_array_equal(result.stems["lead"], x)
        self.assertTrue(result.report["lead_processing_bypassed"])
        wrong, wp = self.fixture(48000, frequency=440*2**((60-69)/12))
        fixed = render_vocal(wrong, 48000, wp, wp.times, np.full(len(wp.times), 57.),
                             mode="solo", pitch_style="transparent")
        np.testing.assert_allclose(fixed.automation["effective_lead_midi"], 57., atol=1e-8)

    def test_natural_psola_stack_preserves_corrected_expression(self):
        x, p, target = self.vibrato_fixture()
        harmony = np.vstack([target+4, target+7, target+12])
        result = render_vocal(x, 44100, p, p.times, target, harmony_midi=harmony,
                              mode="vocoder", pitch_style="transparent", harmony_engine="psola")
        self.assertEqual(len(result.stems), 4)
        residual = result.automation["effective_lead_midi"]-target
        np.testing.assert_allclose(result.automation["effective_harmony_midi"]-harmony,
                                   np.broadcast_to(residual, harmony.shape), atol=1e-8)
        self.assertEqual(result.report["harmony_engine"], "psola")
        self.assertTrue(np.isfinite(result.audio).all())


if __name__ == "__main__":
    unittest.main()
