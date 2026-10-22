"""
test_pose_sitting.py — sitting vs standing classification using synthetic
MediaPipe-style landmarks (no MediaPipe or video needed).

Coordinates are normalised to a crop of CW x CH pixels, like MediaPipe returns.
"""
import sys, os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pose_estimator import PoseEstimator

CW, CH = 140, 200          # crop size in pixels


def make_landmarks(points, vis=0.9):
    """points: {index: (x_px, y_px)}; everything else gets visibility 0."""
    lm = [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
    for idx, (x, y) in points.items():
        lm[idx] = SimpleNamespace(x=x / CW, y=y / CH, visibility=vis)
    return lm


def classify(points, bbox_w=100, bbox_h=180, feet_y=None):
    est = PoseEstimator.__new__(PoseEstimator)   # skip MediaPipe init
    return est._classify_from_landmarks(make_landmarks(points), bbox_w, bbox_h, CW, CH, feet_y)


# torso = 40 px (shoulders y=60, hips y=100)
SHOULDERS = {11: (55, 60), 12: (85, 60)}
HIPS      = {23: (60, 100), 24: (80, 100)}


def test_standing_full_body():
    pts = {**SHOULDERS, **HIPS,
           25: (60, 140), 26: (80, 140),       # knees 1 torso below hips
           27: (60, 185), 28: (80, 185)}       # ankles ~2.1 torso below
    assert classify(pts, 60, 180) == 'standing'


def test_sitting_on_chair_side_view():
    pts = {**SHOULDERS, **HIPS,
           25: (100, 104), 26: (100, 104),     # thigh horizontal
           27: (100, 145), 28: (100, 145)}     # shins vertical, ankles ~1.1 torso below
    assert classify(pts, 90, 110) == 'sitting'


def test_sitting_on_chair_front_view():
    pts = {**SHOULDERS, **HIPS,
           25: (58, 118), 26: (82, 118),       # thighs foreshortened
           27: (58, 148), 28: (82, 148)}
    assert classify(pts, 60, 120) == 'sitting'


def test_sitting_knees_only():
    pts = {**SHOULDERS, **HIPS, 25: (95, 106), 26: (95, 106)}
    assert classify(pts, 80, 100) == 'sitting'


def test_standing_knees_only_ankles_cropped():
    pts = {**SHOULDERS, **HIPS, 25: (60, 142), 26: (80, 142)}
    assert classify(pts, 60, 150) == 'standing'


def test_crouching():
    pts = {**SHOULDERS, **HIPS,
           25: (70, 96), 26: (90, 96),
           27: (60, 112), 28: (80, 112)}       # ankles ~0.3 torso below hips
    assert classify(pts, 70, 80) == 'crouching'


def test_lying():
    pts = {11: (20, 100), 12: (20, 110), 23: (70, 100), 24: (70, 110)}   # torso horizontal
    assert classify(pts, 150, 60) == 'lying'


def test_legs_hidden_behind_desk_is_sitting():
    # upper body only: bbox height ~2.3 torso lengths, wide-ish
    pts = {**SHOULDERS, **HIPS}
    assert classify(pts, 70, 92) == 'sitting'


def test_legs_hidden_but_tall_narrow_is_standing():
    pts = {**SHOULDERS, **HIPS}
    assert classify(pts, 45, 150) == 'standing'


def test_hallucinated_low_visibility_legs_are_ignored():
    """MediaPipe 'guesses' hidden legs with low visibility - must not count as standing."""
    lm = make_landmarks({**SHOULDERS, **HIPS})
    for idx, (x, y) in {25: (60, 140), 26: (80, 140), 27: (60, 185), 28: (80, 185)}.items():
        lm[idx] = SimpleNamespace(x=x / CW, y=y / CH, visibility=0.2)
    est = PoseEstimator.__new__(PoseEstimator)
    assert est._classify_from_landmarks(lm, 70, 92, CW, CH) == 'sitting'


# ── Box-based feet (F = (box bottom - hip_y) / torso), torso = 40 px, hips y = 100 ──

def test_box_feet_seated_man_from_real_video():
    # seated man: F = 1.43-1.59 in the log -> feet_y = 100 + F*40
    for f in (1.43, 1.50, 1.59):
        assert classify({**SHOULDERS, **HIPS}, 80, 160, feet_y=100 + f * 40) == 'sitting'


def test_box_feet_standing_man_from_real_video():
    # standing man: F = 1.64, 1.74, 2.24 in the log
    for f in (1.64, 1.74, 2.24):
        assert classify({**SHOULDERS, **HIPS}, 60, 160, feet_y=100 + f * 40) == 'standing'


def test_box_feet_beats_wrong_ankle_landmarks():
    """Real bug: MediaPipe put a standing man's ankles only ~1.1 torso below the hips
    (looks like sitting). The box bottom says he is standing - the box must win."""
    pts = {**SHOULDERS, **HIPS, 27: (60, 144), 28: (80, 144)}    # ankle drop = 1.1
    assert classify(pts, 60, 160, feet_y=190) == 'standing'


def test_box_feet_crouching():
    assert classify({**SHOULDERS, **HIPS}, 70, 100, feet_y=120) == 'crouching'


def test_cut_off_box_falls_back_to_landmarks():
    """feet_y=None (person cut off by frame bottom) -> use ankle landmarks as before."""
    pts = {**SHOULDERS, **HIPS, 27: (60, 185), 28: (80, 185)}
    assert classify(pts, 60, 180, feet_y=None) == 'standing'


# ── Motion cue ──────────────────────────────────────────────────────────────
from collections import deque
from pose_estimator import update_motion


def test_motion_walking_person_is_moving():
    h = deque()
    d = 0
    for i in range(0, 40):                       # 40 samples over ~2.6 s, walking 5 px / sample
        d = update_motion(h, i * 0.066, 500 + i * 5, 200)
    assert d >= 35


def test_motion_seated_person_with_jitter_is_not_moving():
    h = deque()
    d = 0
    for i in range(0, 60):                       # same spot, +-4 px box jitter
        d = update_motion(h, i * 0.066, 628 + (i % 3) * 4 - 4, 300 + (i % 2) * 3)
    assert d < 35


def test_motion_needs_enough_history():
    h = deque()
    d = update_motion(h, 0.0, 100, 100)
    d = update_motion(h, 0.1, 400, 100)          # big jump but only 0.1 s of history
    assert d == 0.0
