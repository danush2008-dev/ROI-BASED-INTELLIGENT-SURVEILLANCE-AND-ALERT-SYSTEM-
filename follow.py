"""
follow.py — Keep a person's box moving with them when YOLO misses them.

Inside an ROI people are often small or partly hidden (stair railing, gate),
and the detector drops them for a few frames.  Instead of freezing the box
where they were last detected, follow.py moves it with the person's image
from one frame to the next:

  1. Template matching of how the person looked when last DETECTED (so
     slow movement adds up instead of being lost frame to frame), in a
     search window around the box, comparing only "foreground" pixels - those that differ from a
     slowly-updated background model.  A static railing or gate in front of
     the person is background, so it can't pin the box in place.
     If that fails the person is treated as gone.
  Without such a look (box too small when detected):
  2. The same with the previous frame's box.
  3. Otherwise (person standing still, no foreground): Lucas-Kanade optical
     flow on corner points inside the box, median movement.
  4. Otherwise: plain template matching.

Only standard OpenCV functions are used (no opencv-contrib needed).
"""

import cv2
import numpy as np


_LK_PARAMS = dict(winSize=(9, 9), maxLevel=3,     # small window: fits between railing bars
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


class BackgroundModel:
    """Running-average background; what differs from it is foreground."""

    def __init__(self, learn_rate=0.02, threshold=18):
        self.learn_rate = learn_rate      # ~ 1 / (frames to adapt)
        self.threshold = threshold
        self._bg = None

    def update(self, gray):
        """Feed a frame; returns its foreground mask (uint8, 255 = moving)."""
        if self._bg is None:
            self._bg = gray.astype(np.float32)
        fg = cv2.absdiff(gray, cv2.convertScaleAbs(self._bg))
        _, fg = cv2.threshold(fg, self.threshold, 255, cv2.THRESH_BINARY)
        # close small holes without growing the mask over thin static
        # things next to the person (railing bars)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cv2.accumulateWeighted(gray, self._bg, self.learn_rate)
        return fg


def make_anchor(gray, fg, box):
    """The person's look (grey patch + foreground mask) at a detection."""
    h_img, w_img = gray.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (gray[y1:y2, x1:x2].copy(),
            None if fg is None else fg[y1:y2, x1:x2].copy())


def follow_box(prev_gray, gray, box, velocity=(0.0, 0.0), min_match=0.5,
               prev_fg=None, anchor=None, fg=None):
    """
    Where is the person who was in `box` (x1, y1, x2, y2) in prev_gray now,
    in gray?

    velocity: the person's recent movement in pixels per frame (dx, dy),
              used to prefer points that move with the person.
    prev_fg:  foreground mask of prev_gray (BackgroundModel.update)
    anchor:   make_anchor() from the last frame the person was detected in
    fg:       foreground mask of gray

    Returns (box, method): the new box (same size, moved, as floats) and
    "anchor" when it was matched to the person's detected look (reliable),
    "frame" otherwise - or None if the person can't be found with
    reasonable confidence.
    """
    h_img, w_img = gray.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None

    rect = (x1, y1, x2, y2)
    if anchor is not None:
        pos = _anchor_position(gray, anchor, rect, fg)
        if pos is not None:
            ax, ay = pos
            new = (ax, ay, ax + (box[2] - box[0]), ay + (box[3] - box[1]))
            if _has_foreground(fg, new):
                return new, "anchor"
        # Not found looking like themselves: gone (or turned away). The
        # frame-to-frame fallbacks below could lock onto the background.
        return None
    shift = None
    if prev_fg is not None:
        shift = _template_shift(prev_gray, gray, rect, min_match, prev_fg)
    if shift is None:
        shift = _flow_shift(prev_gray, gray, rect, velocity, prev_fg)
    if shift is None:
        shift = _template_shift(prev_gray, gray, rect, min_match)
    if shift is None:
        return None

    dx, dy = shift
    ox1, oy1, ox2, oy2 = box
    new = (ox1 + dx, oy1 + dy, ox2 + dx, oy2 + dy)
    if not _has_foreground(fg, new):
        return None
    return new, "frame"


def _has_foreground(fg, box, min_fraction=0.1):
    """
    A followed person must show up as something that differs from the
    background. If nothing in the box does, they have left (or the box is
    on empty background) - stop following rather than leave a ghost box.
    """
    if fg is None:
        return True
    h_img, w_img = fg.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    return np.count_nonzero(fg[y1:y2, x1:x2]) >= min_fraction * (x2 - x1) * (y2 - y1)


def _anchor_position(gray, anchor, rect, fg=None):
    """Top-left of the anchor look in a window around rect, or None."""
    template, mask = anchor
    th, tw = template.shape[:2]
    x1, y1, x2, y2 = rect
    h_img, w_img = gray.shape[:2]
    sx1, sy1 = max(0, x1 - tw // 2), max(0, y1 - th // 2)
    sx2, sy2 = min(w_img, x1 + tw + tw // 2), min(h_img, y1 + th + th // 2)
    search = gray[sy1:sy2, sx1:sx2]
    if search.shape[0] < th or search.shape[1] < tw or float(template.std()) < 2.0:
        return None
    if mask is not None and np.count_nonzero(mask) >= 0.15 * mask.size:
        if fg is not None:
            # Static background in the new frame (railing bars now covering
            # other parts of the person) is set to a neutral grey, so it
            # neither matches nor spoils the match of the person's pixels
            neutral = float(template[mask > 0].mean())
            search = np.where(fg[sy1:sy2, sx1:sx2] > 0, search, neutral)
            search = search.astype(np.float32)
            template = template.astype(np.float32)
            min_match = 0.4
        else:
            min_match = 0.25
        res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    else:
        res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        min_match = 0.5
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < min_match:
        return None
    return float(sx1 + loc[0]), float(sy1 + loc[1])


def _flow_shift(prev_gray, gray, box, velocity, prev_fg=None):
    x1, y1, x2, y2 = box
    pts = None
    if prev_fg is not None:
        # Points on the person only (a standing-still person has little
        # foreground: then use all points - the box should stay put anyway)
        pts = cv2.goodFeaturesToTrack(prev_gray[y1:y2, x1:x2], maxCorners=80,
                                      qualityLevel=0.01, minDistance=3,
                                      mask=prev_fg[y1:y2, x1:x2])
        if pts is not None and len(pts) < 6:
            pts = None
    if pts is None:
        pts = cv2.goodFeaturesToTrack(prev_gray[y1:y2, x1:x2], maxCorners=80,
                                      qualityLevel=0.01, minDistance=3)
    if pts is None or len(pts) < 6:
        return None
    pts = pts.reshape(-1, 2) + np.float32([x1, y1])
    pts = pts.reshape(-1, 1, 2)

    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **_LK_PARAMS)
    if nxt is None:
        return None
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **_LK_PARAMS)
    if back is None:
        return None
    fb_err = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.5)
    if good.sum() < 5:
        return None

    moves = (nxt - pts).reshape(-1, 2)[good]
    vx, vy = velocity
    speed = float(np.hypot(vx, vy))
    if speed > 0.5:
        # The person was moving: points that stay perfectly still belong to
        # a railing / door frame in front of them and would hold the box back
        moving = moves[np.hypot(moves[:, 0], moves[:, 1]) > 0.3]
        if len(moving) >= 4:
            moves = moving
    if speed > 1.5:
        # ...and of the moving points, prefer those moving like the person
        dist = np.hypot(moves[:, 0] - vx, moves[:, 1] - vy)
        with_person = moves[dist <= max(1.5, 0.6 * speed)]
        if len(with_person) >= 4:
            moves = with_person
    return float(np.median(moves[:, 0])), float(np.median(moves[:, 1]))


def _template_shift(prev_gray, gray, box, min_match, prev_fg=None):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    template = prev_gray[y1:y2, x1:x2]
    h_img, w_img = gray.shape[:2]
    sx1, sy1 = max(0, x1 - bw // 2), max(0, y1 - bh // 2)
    sx2, sy2 = min(w_img, x2 + bw // 2), min(h_img, y2 + bh // 2)
    search = gray[sy1:sy2, sx1:sx2]
    if search.shape[0] < bh or search.shape[1] < bw:
        return None
    if float(template.std()) < 2.0:          # flat patch: nothing to match
        return None
    if prev_fg is not None:
        mask = prev_fg[y1:y2, x1:x2]
        if np.count_nonzero(mask) < 0.15 * mask.size:
            return None                        # too little of the person moving
        # compare the person's pixels only
        res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        # bars in front of the person hide part of them in the new frame,
        # so even the right position scores lower than an open view
        min_match = min(min_match, 0.25)
    else:
        res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < min_match:
        return None
    return float(sx1 + loc[0] - x1), float(sy1 + loc[1] - y1)


_APPEARANCE_SIZE = (24, 48)   # (w, h) the person's patch is shrunk to


def appearance(gray, box):
    """Small normalised grey patch of the person, for still-the-same checks."""
    h_img, w_img = gray.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    patch = cv2.resize(gray[y1:y2, x1:x2], _APPEARANCE_SIZE).astype(np.float32)
    patch -= patch.mean()
    norm = float(np.linalg.norm(patch))
    return patch / norm if norm > 1e-6 else None


def similarity(a, b):
    """Correlation of two appearance() patches: 1 = identical, ~0 = unrelated."""
    if a is None or b is None:
        return 0.0
    return float((a * b).sum())
