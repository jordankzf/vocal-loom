"""Synthetic MIDI tests cover musical timing and parser edge cases."""
import json
from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np

from pipeline_midi import load_score, melody_at, harmonic_snapshots


class ScoreTests(unittest.TestCase):
    def write(self, directory, tracks, midi_type=1):
        path = Path(directory) / "fixture.mid"
        midi = mido.MidiFile(type=midi_type, ticks_per_beat=480)
        midi.tracks.extend(mido.MidiTrack(messages) for messages in tracks)
        midi.save(path)
        return path

    def test_tempo_change_and_cross_track_sustain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, [
                [mido.MetaMessage("set_tempo", tempo=500_000),
                 mido.Message("control_change", channel=0, control=64, value=127),
                 mido.MetaMessage("set_tempo", tempo=1_000_000, time=480),
                 mido.Message("control_change", channel=0, control=64, value=0, time=480)],
                [mido.MetaMessage("track_name", name="Vocal melody"),
                 mido.Message("note_on", note=60, velocity=90),
                 mido.Message("note_on", note=60, velocity=0, time=480),
                 mido.MetaMessage("end_of_track", time=480)],
            ])
            score = load_score(path)
            self.assertAlmostEqual(score.duration, 1.5)
            self.assertEqual(len(score.notes), 1)
            self.assertAlmostEqual(score.notes[0].end, 1.5)
            self.assertEqual([t.microseconds_per_beat for t in score.tempo_changes], [500_000, 1_000_000])
            result = score.melody_at([.2, .8, 1.49, 1.5])
            np.testing.assert_equal(result.pitches, [60, 60, 60, np.nan])
            self.assertEqual(result.selected_tracks, (1,))

    def test_melody_track_wins_over_bass_chords_and_drums(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, [
                [mido.MetaMessage("track_name", name="Piano chords"),
                 *[mido.Message("note_on", note=n, velocity=80) for n in (48, 52, 55)],
                 mido.Message("note_off", note=48, time=960),
                 mido.Message("note_off", note=52), mido.Message("note_off", note=55)],
                [mido.MetaMessage("track_name", name="Lead vocal"),
                 mido.Message("note_on", channel=1, note=72, velocity=90),
                 mido.Message("note_off", channel=1, note=72, time=480),
                 mido.Message("note_on", channel=1, note=74, velocity=90, time=120),
                 mido.Message("note_off", channel=1, note=74, time=360)],
                [mido.MetaMessage("track_name", name="Drums"),
                 mido.Message("note_on", channel=9, note=100, velocity=100),
                 mido.Message("note_off", channel=9, note=100, time=960)],
            ])
            score = load_score(path)
            result = melody_at(score, [.1, .55, .7])
            self.assertEqual(result.selected_tracks, (1,))
            np.testing.assert_equal(result.pitches, [72, np.nan, 74])
            held = melody_at(score, [.1, .55, .7], hold_rests=True)
            np.testing.assert_equal(held.pitches, [72, 72, 74])
            self.assertGreater(result.confidence, .8)
            self.assertEqual(score.tracks[2].drum_note_count, 1)
            self.assertFalse(any(100 in s.pitches for s in harmonic_snapshots(score)))

    def test_polyphonic_type_zero_reports_ambiguity_and_rests(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, [[
                mido.Message("note_on", note=60, velocity=80, time=480),
                mido.Message("note_on", note=64, velocity=80),
                mido.Message("note_off", note=60, time=480),
                mido.Message("note_off", note=64),
                mido.Message("note_on", note=67, velocity=80, time=480),
                mido.Message("note_off", note=67, time=480),
            ]], midi_type=0)
            score = load_score(path)
            result = melody_at(score, [.1, .6, 1.2, 1.6])
            np.testing.assert_equal(result.pitches, [np.nan, 64, np.nan, 67])
            self.assertLess(result.confidence, .6)
            self.assertIn("instrumental_melody_ambiguous", {w.code for w in result.warnings})
            json.dumps(result.to_dict(), allow_nan=False)
            override = melody_at(score, [.6], track_selection=0)
            self.assertEqual(override.selected_tracks, (0,))

    def test_controller_release_and_unreleased_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, [[
                mido.Message("note_on", note=60, velocity=80),
                mido.Message("control_change", control=64, value=127),
                mido.Message("control_change", control=123, value=0, time=480),
                mido.Message("control_change", control=121, value=0, time=480),
                mido.Message("note_on", note=64, velocity=80),
                mido.MetaMessage("end_of_track", time=480),
            ]])
            score = load_score(path)
            self.assertEqual([(n.pitch, n.start, n.end) for n in score.notes], [(60, 0, 1), (64, 1, 1.5)])
            self.assertIn("unreleased_notes", {w.code for w in score.warnings})

    def test_type_two_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, [[mido.MetaMessage("end_of_track", time=480)]], midi_type=2)
            with self.assertRaisesRegex(ValueError, "Type-2"):
                load_score(path)


if __name__ == "__main__":
    unittest.main()
