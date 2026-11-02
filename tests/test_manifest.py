"""Own recordings: the manifest path the recording protocol depends on."""

import json

import numpy as np
import pytest
import soundfile as sf

from simulst.data import load_manifest


@pytest.fixture
def recordings(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "setA").mkdir()
    for index in range(3):
        audio = rng.standard_normal(48000).astype(np.float32) * 0.01
        sf.write(tmp_path / "setA" / f"{index}.wav", audio, 48000)

    lines = [
        {"audio": f"setA/{i}.wav", "set": "A" if i < 2 else "B",
         "sentence": f"frase {i}", "translation": f"sentence {i}"}
        for i in range(3)
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(line) for line in lines))
    return manifest


class TestLoadManifest:
    def test_every_entry_is_loaded(self, recordings):
        assert len(load_manifest(recordings)) == 3

    def test_audio_is_resampled_to_the_target(self, recordings):
        """Recordings will not arrive at 16 kHz; Whisper needs them there."""
        example = load_manifest(recordings, target_rate=16000)[0]
        assert abs(example.duration_s - 1.0) < 0.01
        assert len(example.audio) == pytest.approx(16000, abs=2)

    def test_the_set_label_is_carried_through(self, recordings):
        assert [e.group for e in load_manifest(recordings)] == ["A", "A", "B"]

    def test_paths_resolve_against_the_manifest(self, recordings):
        """So a manifest can be moved with its audio and still work."""
        assert load_manifest(recordings)[0].audio.size > 0

    def test_a_missing_field_names_the_line(self, recordings):
        recordings.write_text(json.dumps({"audio": "setA/0.wav"}))
        with pytest.raises(ValueError, match="missing \\['translation'\\]"):
            load_manifest(recordings)

    def test_bad_json_names_the_line(self, recordings):
        recordings.write_text('{"audio": "setA/0.wav"\n')
        with pytest.raises(ValueError, match=":1:"):
            load_manifest(recordings)

    def test_a_missing_file_is_reported_with_its_path(self, recordings):
        recordings.write_text(json.dumps({"audio": "nope.wav", "translation": "x"}))
        with pytest.raises(FileNotFoundError, match="nope.wav"):
            load_manifest(recordings)

    def test_blank_lines_are_ignored(self, recordings):
        recordings.write_text(
            json.dumps({"audio": "setA/0.wav", "translation": "x"}) + "\n\n\n"
        )
        assert len(load_manifest(recordings)) == 1

    def test_an_empty_manifest_is_an_error(self, recordings):
        recordings.write_text("\n\n")
        with pytest.raises(RuntimeError, match="no usable entries"):
            load_manifest(recordings)
