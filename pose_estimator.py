"""
pose_estimator.py — MediaPipe Pose behaviour classification.

Classifies each detected person's pose into one of:
  - 'standing'   : upright, normal
  - 'sitting'    : seated on a chair / stool / ledge
  - 'crouching'  : bent low (e.g. hiding, tampering)
  - 'lying'      : horizontal on the ground (fallen / incapacitated)
  - 'unknown'    : not enough landmarks visible

Uses MediaPipe Pose (Lite model) on the cropped person bounding box for
CPU-efficient inference.  Runs every POSE_EVERY_N_FRAMES to save CPU.

Classification heuristics:
  1. Lying: torso angle from vertical > LYING_TORSO_ANGLE  OR
            bbox width/height > LYING_ASPECT_RATIO
  2. Crouching: hip-to-ankle distance / shoulder-to-hip distance < CROUCH_HIP_RATIO
  3. Sitting vs standing: how far the hips are above the bottom of the
     person's box (the feet), measured in torso lengths (config.FEET_*).
     If the box is cut off by the frame edge, use ankle / knee landmarks
     (config.STAND_ANKLE_RATIO etc.), and finally the box shape.
"""

import math
import numpy as np
import cv2

try:
    import mediapipe as mp
    # Check that the legacy solutions API is available
    _test = mp.solutions.pose
    _HAS_MEDIAPIPE = True
except (ImportError, AttributeError):
    _HAS_MEDIAPIPE = False

import config


def update_motion(history, t, x, y, window_s=None):
    """
    Track a person's foot point over time.

    history: a collections.deque kept per person.  Appends (t, x, y), drops
    entries older than the window, and returns how far the foot point has
    moved (pixels, start of window -> now).  Returns 0.0 until at least half
    a window of history exists, so a couple of jittery frames never count.
    """
    window_s = window_s or config.MOVE_WINDOW_SECONDS
    history.append((t, x, y))
    while history and t - history[0][0] > window_s:
        history.popleft()
    if t - history[0][0] < 0.5 * window_s:
        return 0.0
    return math.hypot(x - history[0][1], y - history[0][2])


class PoseEstimator:
    """
    MediaPipe Pose wrapper for behaviour classification.

    Call classify(frame, x1, y1, x2, y2) for each detected person.
    Returns (behaviour_str, landmarks_or_None).
    """

    def __init__(self):
        self._enabled = config.USE_POSE and _HAS_MEDIAPIPE
        self._pose = None
        self.last_debug = ""          # numbers behind the last decision (for tuning)

        if self._enabled:
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=True,         # each call is a different person's crop
                model_complexity=0,        # 0 = Lite (fastest, CPU-friendly)
                smooth_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_draw_styles = mp.solutions.drawing_styles
            print("[Pose] MediaPipe Pose (Lite) initialized.")
        else:
            if config.USE_POSE and not _HAS_MEDIAPIPE:
                print("[Pose] WARNING: USE_POSE=True but mediapipe not installed.")
            else:
                print("[Pose] Pose estimation disabled.")

    @property
    def enabled(self):
        return self._enabled

    def classify(self, frame, x1, y1, x2, y2):
        """
        Classify the pose of a person within the given bounding box.

        Args:
            frame: full video frame (BGR)
            x1, y1, x2, y2: bounding box of the person

        Returns:
            (behaviour: str, landmarks: list or None)
            behaviour is one of: 'standing', 'crouching', 'lying', 'unknown'
            landmarks is a list of (x, y, visibility) tuples in frame coords,
            or None if pose detection failed.
        """
        if not self._enabled:
            return 'unknown', None

        h, w = frame.shape[:2]

        # Clamp bbox to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        bw = x2 - x1
        bh = y2 - y1
        if bw < 20 or bh < 20:
            return 'unknown', None

        # Crop person region with small padding
        pad = 10
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]

        # MediaPipe expects RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb_crop)

        if not results.pose_landmarks:
            # Fallback: use aspect ratio heuristic
            aspect = bw / bh if bh > 0 else 0
            if aspect > config.LYING_ASPECT_RATIO:
                return 'lying', None
            return 'unknown', None

        # Extract landmarks in frame coordinates
        landmarks_frame = []
        ch, cw = crop.shape[:2]
        lm = results.pose_landmarks.landmark

        for l in lm:
            fx = int(l.x * cw + cx1)
            fy = int(l.y * ch + cy1)
            landmarks_frame.append((fx, fy, l.visibility))

        # ── Classification logic ──────────────────────────────────────────

        # bottom of the YOLO box (the feet), in crop pixels - only trusted when
        # the person is not cut off by the bottom edge of the frame
        feet_y = (y2 - cy1) if y2 < h - 2 else None
        behaviour = self._classify_from_landmarks(lm, bw, bh, cw, ch, feet_y)
        self.last_debug = self._format_debug()

        return behaviour, landmarks_frame

    def _classify_from_landmarks(self, landmarks, bbox_w, bbox_h,
                                 crop_w=None, crop_h=None, feet_y=None):
        self._dbg = {}
        return self._classify_core(landmarks, bbox_w, bbox_h,
                                   crop_w, crop_h, feet_y, self._dbg)

    def _format_debug(self):
        d = getattr(self, '_dbg', {}) or {}
        def f(k, fmt):
            v = d.get(k)
            return '-' if v is None else format(v, fmt)
        return (f"{d.get('branch', '?')} A:{f('A', '.2f')} K:{f('K', '.2f')} "
                f"F:{f('F', '.2f')} B:{f('B', '.1f')} vis:{f('AV', '.2f')}")

    def _classify_core(self, landmarks, bbox_w, bbox_h,
                       crop_w, crop_h, feet_y, dbg):
        """
        Determine pose behaviour from MediaPipe landmarks.

        PoseLandmark indices used:
          11/12 shoulders, 23/24 hips, 25/26 knees, 27/28 ankles

        MediaPipe landmarks are normalised to the crop, so x and y have
        different scales.  We convert to pixels before measuring anything.
        """
        lm = landmarks
        cw = crop_w or bbox_w
        ch = crop_h or bbox_h

        def get_lm(idx, min_vis=0.4):
            l = lm[idx]
            if l.visibility < min_vis:
                return None
            return (l.x * cw, l.y * ch)          # pixel coords

        def mean_pt(pts):
            return (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))

        leg_vis = config.LEG_VISIBILITY_MIN
        shoulders = [p for p in (get_lm(11), get_lm(12)) if p is not None]
        hips      = [p for p in (get_lm(23), get_lm(24)) if p is not None]
        knees     = [p for p in (get_lm(25, leg_vis), get_lm(26, leg_vis)) if p is not None]
        ankles    = [p for p in (get_lm(27, leg_vis), get_lm(28, leg_vis)) if p is not None]

        aspect = bbox_w / bbox_h if bbox_h > 0 else 0
        dbg['AV'] = max(lm[27].visibility, lm[28].visibility)

        # Need at least one shoulder and one hip
        if len(shoulders) < 1 or len(hips) < 1:
            dbg['branch'] = 'few-landmarks'
            if aspect > config.LYING_ASPECT_RATIO:
                return 'lying'
            return 'unknown'

        shoulder_mid = mean_pt(shoulders)
        hip_mid = mean_pt(hips)

        # ── Check 1: Lying (torso angle from vertical) ────────────────────
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        torso_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

        if torso_angle > config.LYING_TORSO_ANGLE:
            dbg['branch'] = 'lying-angle'
            return 'lying'
        if aspect > config.LYING_ASPECT_RATIO:
            dbg['branch'] = 'lying-aspect'
            return 'lying'

        torso_len = math.hypot(dx, dy)
        if torso_len < 3:                      # degenerate / far too small
            return 'unknown'

        if feet_y is not None:
            dbg['F'] = (feet_y - hip_mid[1]) / torso_len     # hips -> bottom of box
        dbg['B'] = bbox_h / torso_len

        # ── Check 2a: box-based feet (primary) ────────────────────────────
        #   The YOLO box bottom is a far more reliable "feet" position than
        #   MediaPipe's ankle landmarks, which drift on small / distant people.
        if config.USE_BOX_FEET and dbg.get('F') is not None:
            f = dbg['F']
            dbg['branch'] = 'box'
            if f < config.FEET_CROUCH_RATIO:
                return 'crouching'
            if f < config.FEET_STAND_RATIO:
                return 'sitting'
            return 'standing'

        # Vertical drops below the hips, in torso lengths
        ankle_drop = None
        if ankles:
            ankle_drop = (mean_pt(ankles)[1] - hip_mid[1]) / torso_len
        knee_drop = None
        if knees:
            knee_drop = (mean_pt(knees)[1] - hip_mid[1]) / torso_len
        dbg['A'] = ankle_drop
        dbg['K'] = knee_drop

        # ── Check 2: ankles visible -> best evidence ──────────────────────
        #   standing        : ankles far below hips   (~2 torso lengths)
        #   sitting (chair) : ankles ~1 torso length below hips
        #   crouching       : hips almost down at ankle level
        if ankle_drop is not None:
            dbg['branch'] = 'ankle'
            if ankle_drop < config.CROUCH_HIP_RATIO:
                return 'crouching'
            if ankle_drop < config.STAND_ANKLE_RATIO:
                return 'sitting'
            return 'standing'

        # ── Check 3: only knees visible (ankles cropped / hidden) ─────────
        if knee_drop is not None:
            dbg['branch'] = 'knee'
            if knee_drop < 0.0:
                return 'crouching'             # knees higher than hips
            if knee_drop < config.SIT_KNEE_RATIO:
                return 'sitting'               # thigh roughly horizontal
            return 'standing'

        # ── Check 4: legs not visible (behind desk/table) -> bbox shape ───
        #   A full standing body is ~3.3+ torso lengths tall and narrow.
        #   An upper body (or seated body) is shorter and wider.
        dbg['branch'] = 'bbox'
        body_ratio = bbox_h / torso_len
        if body_ratio < config.SIT_BODY_RATIO or aspect > config.SIT_ASPECT_RATIO:
            return 'sitting'
        return 'standing'

    def draw_landmarks(self, frame, landmarks, colour=(0, 255, 0)):
        """
        Draw pose skeleton on the frame.

        Args:
            frame: video frame to draw on
            landmarks: list of (x, y, visibility) tuples
            colour: BGR colour for the skeleton
        """
        if not landmarks:
            return

        # Key connections for a simplified skeleton
        connections = [
            (11, 12),  # shoulders
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (11, 23), (12, 24),  # torso sides
            (23, 24),  # hips
            (23, 25), (25, 27),  # left leg
            (24, 26), (26, 28),  # right leg
        ]

        # Draw connections
        for i, j in connections:
            if i < len(landmarks) and j < len(landmarks):
                x1, y1, v1 = landmarks[i]
                x2, y2, v2 = landmarks[j]
                if v1 > 0.3 and v2 > 0.3:
                    cv2.line(frame, (x1, y1), (x2, y2), colour, 1)

        # Draw key landmark dots
        for idx in [11, 12, 23, 24, 27, 28]:  # shoulders, hips, ankles
            if idx < len(landmarks):
                x, y, v = landmarks[idx]
                if v > 0.3:
                    cv2.circle(frame, (x, y), 3, colour, -1)

    def release(self):
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None
