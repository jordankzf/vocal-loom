"""Musical invariants for the single upper duet voice."""
from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np

from duet_part import choose_duet


class DuetTests(unittest.TestCase):
    def score(self, folder, chords):
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        for chord in chords:
            for note in chord:
                track.append(mido.Message("note_on", note=note, velocity=90))
            for i, note in enumerate(chord):
                track.append(mido.Message("note_off", note=note, time=960 if i == 0 else 0))
        path = Path(folder)/"score.mid"
        midi.save(path)
        return path

    def test_third_then_sixth_and_actual_chord_tones(self):
        with tempfile.TemporaryDirectory() as folder:
            # The singer moves A2→E2 while the score holds an A chord, then
            # sings G#2 over F#m: C#3 is allowed even though it is a fourth.
            path = self.score(folder, [[45, 57, 61, 64, 69], [42, 54, 57, 61, 66]])
            times = np.arange(0, 2, .01)
            lead = np.where(times < .5, 45., np.where(times < 1, 40., 44.))
            targets, audit, warnings = choose_duet(times, lead, path)
            self.assertEqual(targets.shape, (1, len(times)))
            self.assertEqual(targets[0, 0], 49)  # A2 → C#3, not E3
            self.assertEqual(targets[0, 70], 49)  # E2 → C#3, a sixth
            self.assertEqual(targets[0, 150], 49)  # G#2 → C#3, a score-supported fourth
            intervals = targets[0]-lead
            self.assertTrue(np.all((intervals >= 3) & (intervals <= 16)))
            for entry in audit:
                self.assertIn(int(entry["harmony_midi"]) % 12, entry["source_chord_pitch_classes"])

    def test_no_invented_fixed_third_and_explicit_octave_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            # Only C and G exist; E, the usual third above C, is forbidden.
            path = self.score(folder, [[48, 60, 67]])
            targets, audit, _ = choose_duet(np.array([.1, .4]), np.array([48., 48.]), path)
            self.assertTrue(all(int(n) % 12 in {0, 7} for n in targets[0]))
            self.assertNotEqual(targets[0, 0], 52)
            mono = self.score(folder, [[60]])
            targets, _, warnings = choose_duet(np.array([.1, .4]), np.array([48., 48.]), mono)
            np.testing.assert_array_equal(targets, [[60, 60]])
            self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
