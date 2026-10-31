import numpy as np

from simulst.audio import resample


def test_resample_changes_length_by_the_rate_ratio(tone):
    audio, rate = tone
    out = resample(audio, rate, 16000)
    assert abs(len(out) - len(audio) // 3) <= 1


def test_resample_preserves_duration(tone):
    audio, rate = tone
    out = resample(audio, rate, 16000)
    assert abs(len(out) / 16000 - len(audio) / rate) < 1e-3


def test_resample_preserves_frequency(tone):
    """A 440 Hz tone must still peak at 440 Hz after resampling."""
    audio, rate = tone
    out = resample(audio, rate, 16000)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(len(out), 1 / 16000)[np.argmax(spectrum)]
    assert abs(peak_hz - 440) < 5


def test_resample_is_a_noop_at_matching_rates(tone):
    audio, _ = tone
    assert np.array_equal(resample(audio, 16000, 16000), audio)


def test_resample_handles_empty_input():
    assert resample(np.array([], dtype=np.float32), 48000, 16000).size == 0
