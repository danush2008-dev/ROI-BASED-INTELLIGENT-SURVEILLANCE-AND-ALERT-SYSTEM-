"""
test_trusted_cctv.py — Trusted person registered from the camera itself
(clicked in the video window) is recognised; other people are not.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

import config


def _person(shirt, trousers, seed, h=120, w=50):
    """Synthetic CCTV crop: shirt colour on top, trousers below, some noise."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.int16)
    img[:h // 2] = shirt
    img[h // 2:] = trousers
    img += rng.integers(-12, 13, img.shape).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


GREY_SHIRT, DARK_TROUSERS = (150, 150, 150), (40, 35, 30)


@pytest.fixture
def detector(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USE_TRUSTED_PERSONS", True)
    monkeypatch.setattr(config, "TRUSTED_FACES_DIR", str(tmp_path))
    from trusted_person import TrustedPersonDetector
    return TrustedPersonDetector()


def _check(det, crop):
    frame = np.zeros((300, 300, 3), np.uint8)
    frame[50:50 + crop.shape[0], 100:100 + crop.shape[1]] = crop
    return det.is_trusted(frame, 100, 50, 100 + crop.shape[1], 50 + crop.shape[0])


def test_clicked_person_is_saved_and_recognised(detector, tmp_path):
    assert not detector.enabled                       # nothing registered yet
    path = detector.add_cctv_reference(_person(GREY_SHIRT, DARK_TROUSERS, 1), "trusted")
    assert path and os.path.exists(path) and "_cctv_" in path
    ok, name, score = _check(detector, _person(GREY_SHIRT, DARK_TROUSERS, 2))
    assert ok and name == "trusted" and score >= config.CCTV_MATCH_THRESHOLD


def test_other_people_are_not_trusted(detector):
    detector.add_cctv_reference(_person(GREY_SHIRT, DARK_TROUSERS, 1), "trusted")
    for shirt, trousers in (((40, 40, 200), (40, 35, 30)),     # red shirt
                            ((40, 35, 30), (150, 150, 150)),   # colours swapped
                            ((200, 120, 40), (200, 200, 60))):
        ok, _, _ = _check(detector, _person(shirt, trousers, 3))
        assert not ok


def test_snapshots_are_loaded_on_next_run(detector, tmp_path):
    detector.add_cctv_reference(_person(GREY_SHIRT, DARK_TROUSERS, 1), "uncle")
    from trusted_person import TrustedPersonDetector
    again = TrustedPersonDetector()                   # a later run
    ok, name, _ = _check(again, _person(GREY_SHIRT, DARK_TROUSERS, 5))
    assert ok and name == "uncle"
