import numpy as np
import pytest

from decision_geometry.analysis import cross_temporal_decode, decode_timecourse


def _decodable_fixture(seed=3):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], 40)
    rates = rng.normal(size=(80, 10, 8))
    rates[labels == 1, :4, 4:] += 2.0
    return rates, labels


def test_timecourse_detects_post_signal():
    rates, labels = _decodable_fixture()
    scores = decode_timecourse(rates, labels)
    assert scores[4:].mean() > 0.85
    assert scores[:4].mean() < 0.7


def test_cross_temporal_matrix_has_expected_shape_and_signal():
    rates, labels = _decodable_fixture()
    scores = cross_temporal_decode(rates, labels)
    assert scores.shape == (8, 8)
    assert scores[4:, 4:].mean() > 0.85


def test_decoder_rejects_misaligned_labels():
    rates, _ = _decodable_fixture()
    with pytest.raises(ValueError, match="match the trial count"):
        decode_timecourse(rates, np.array([0, 1]))


def test_decoder_rejects_an_unlabeled_dataset():
    rates, _ = _decodable_fixture()
    with pytest.raises(ValueError, match="non-negative label"):
        cross_temporal_decode(rates, np.full(rates.shape[0], -1))
