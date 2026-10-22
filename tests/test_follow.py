"""
test_follow.py — The box follows a person the detector misses (follow.py).

Synthetic scene: textured background, a textured "person" moving behind a
railing of dark bars (the hard case: the bars don't move, the person does).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import pytest

from follow import follow_box, BackgroundModel, make_anchor

_rng = np.random.default_rng(1)
_BG = cv2.GaussianBlur(_rng.integers(40, 200, (360, 640, 3), dtype=np.uint8), (5, 5), 0)
_PERSON = cv2.GaussianBlur(_rng.integers(0, 255, (60, 30, 3), dtype=np.uint8), (3, 3), 0)


def _box(i, vx, vy):
    fy, fx = 320 + i * vy, 580 + i * vx
    return (int(round(fx - 15)), int(round(fy - 60)),
            int(round(fx + 15)), int(round(fy)))


def _frame(i, vx, vy, bars=True, person=True):
    f = _BG.copy()
    if person:
        x1, y1, x2, y2 = _box(i, vx, vy)
        f[y1:y2, x1:x2] = _PERSON
    if bars:
        for x in range(500, 640, 18):
            cv2.rectangle(f, (x, 0), (x + 4, 360), (30, 30, 30), -1)
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def _follow(vx, vy, frames=45, bars=True, gone_from=None):
    """Detected at frame 0, then only followed. Returns final box or None."""
    model = BackgroundModel()
    for i in range(-10, 0):
        model.update(_frame(i, vx, vy, bars))
    prev = _frame(0, vx, vy, bars)
    prev_fg = model.update(prev)
    box = tuple(float(v) for v in _box(0, vx, vy))
    anchor = make_anchor(prev, prev_fg, box)
    vel = (vx, vy)
    for i in range(1, frames + 1):
        visible = gone_from is None or i < gone_from
        cur = _frame(i, vx, vy, bars, person=visible)
        fg = model.update(cur)
        found = follow_box(prev, cur, box, vel, prev_fg=prev_fg,
                           anchor=anchor, fg=fg)
        if found is None:
            return None
        new = found[0]
        vel = (new[0] - box[0], new[1] - box[1])
        box, prev, prev_fg = new, cur, fg
    return box


@pytest.mark.parametrize("vx, vy, bars", [
    (-0.35, -1, False),      # slow climb, open view
    (-0.35, -1, True),       # slow climb behind the railing
    (-2, -2, True),          # faster, diagonal
    (-3, 0, True),           # running sideways
    (1, -0.5, True),
])
def test_box_stays_on_moving_person(vx, vy, bars):
    box = _follow(vx, vy, bars=bars)
    assert box is not None
    tx1, ty1, _, _ = _box(45, vx, vy)
    assert abs(box[0] - tx1) <= 2 and abs(box[1] - ty1) <= 2


def test_box_does_not_stay_when_person_leaves():
    # person disappears at frame 5: following must stop, no ghost box
    assert _follow(-0.35, -1, frames=20, gone_from=5) is None
