"""
trusted_person.py — Trusted person recognition.

Two kinds of reference images live in `trusted_faces/`:

  * Face photos (e.g. uncle.jpeg): OpenCV Haar face detection + histogram
    comparison.  Only useful when the camera sees the face up close.
  * Camera snapshots (file names containing "_cctv", e.g.
    uncle_cctv_20261022_191500.png): the whole person as THIS camera sees
    them - clothes, colours, build.  Compared with each detected person's
    box (upper and lower half colour histograms).  This is what works for
    an overhead CCTV camera, where faces are tiny or not visible.
    They are created by clicking the person in the video window
    (phy.py, video 2) - see add_cctv_reference().

When a detected person matches a reference, all alerts are suppressed.
"""

import os
from datetime import datetime

import cv2
import numpy as np

import config


class TrustedPersonDetector:
    """
    Detects whether a person in a bounding box matches a trusted face.

    Uses a two-step pipeline:
      1. Haar cascade face detection on the person crop
      2. Histogram comparison against enrolled reference faces
    """

    def __init__(self):
        self._enabled = config.USE_TRUSTED_PERSONS
        self._references = []  # face photos: list of (name, face_hist, body_hist)
        self._cctv_refs = []   # camera snapshots: list of (name, appearance signature)
        self._face_cascade = None

        if not self._enabled:
            print("[Trusted] Trusted person recognition disabled.")
            return

        # Load Haar cascade for face detection (only needed for face photos;
        # camera snapshots work without it)
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        except (AttributeError, cv2.error):
            self._face_cascade = None
        if self._face_cascade is None or self._face_cascade.empty():
            print("[Trusted] WARNING: Could not load face cascade classifier "
                  "- face photos skipped, camera snapshots still used.")
            self._face_cascade = None

        # Load reference faces
        self._load_references()

    @property
    def enabled(self):
        return self._enabled and (len(self._references) > 0 or len(self._cctv_refs) > 0)

    @property
    def can_enroll(self):
        """Trusted-person recognition is on, so people can be added to it."""
        return self._enabled

    # ── Camera snapshots ("_cctv" references) ─────────────────────────────

    @staticmethod
    def appearance_signature(person_crop):
        """
        Colour signature of a whole person as the camera sees them: H-S
        histograms of the upper half (shirt) and lower half (trousers),
        so e.g. a grey shirt with dark trousers differs from the reverse.
        """
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        h = hsv.shape[0]
        sig = []
        for part in (hsv[:h // 2], hsv[h // 2:]):
            hist = cv2.calcHist([part], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
            sig.append(hist)
        return sig

    @staticmethod
    def signature_similarity(a, b):
        """0..1-ish correlation of two appearance signatures."""
        return float(np.mean([cv2.compareHist(x, y, cv2.HISTCMP_CORREL)
                              for x, y in zip(a, b)]))

    def add_cctv_reference(self, person_crop, name="trusted", save=True):
        """
        Enrol a person from the camera itself (their box from the video).
        Saved to trusted_faces/<name>_cctv_<time>.png so later runs
        recognise them automatically. Returns the saved path or None.
        """
        if person_crop is None or person_crop.size == 0 \
                or person_crop.shape[0] < 20 or person_crop.shape[1] < 10:
            return None
        self._cctv_refs.append((name, self.appearance_signature(person_crop)))
        if not save:
            return None
        os.makedirs(config.TRUSTED_FACES_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # PNG: lossless, JPEG would blur the colour statistics being compared
        path = os.path.join(config.TRUSTED_FACES_DIR, f"{name}_cctv_{stamp}.png")
        cv2.imwrite(path, person_crop)
        return path

    def best_cctv_match(self, person_crop):
        """(name, score) of the closest camera snapshot, or (None, 0.0)."""
        if not self._cctv_refs or person_crop.size == 0:
            return None, 0.0
        sig = self.appearance_signature(person_crop)
        best_name, best = None, 0.0
        for name, ref in self._cctv_refs:
            score = self.signature_similarity(sig, ref)
            if score > best:
                best_name, best = name, score
        return best_name, best

    def _compute_face_hist(self, face_img):
        """Compute a normalised HSV histogram for a face image."""
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        # Use H and S channels for skin-tone matching
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _load_references(self):
        """Load and process all reference face images from trusted_faces/."""
        faces_dir = config.TRUSTED_FACES_DIR
        if not os.path.isdir(faces_dir):
            os.makedirs(faces_dir, exist_ok=True)
            print(f"[Trusted] Created {faces_dir}/ — add reference face images there.")
            return

        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(faces_dir)
                 if f.lower().endswith(image_exts)]

        if not files:
            print(f"[Trusted] No reference images in {faces_dir}/.")
            return

        for fname in sorted(files):
            img_path = os.path.join(faces_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Trusted] Could not read: {fname}")
                continue

            if "_cctv" in fname.lower():
                name = fname[:fname.lower().index("_cctv")]
                self.add_cctv_reference(img, name, save=False)
                continue

            if self._face_cascade is None:
                continue                # face photo, but no face detector

            # Compute body histogram (upper 60% of image for color profile)
            upper_body = img[:int(img.shape[0] * 0.6), :]
            body_hist = self._compute_face_hist(upper_body)

            # Detect face in reference image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                # Use the whole image as face region (assume it's a face crop)
                hist = self._compute_face_hist(img)
                self._references.append((fname, hist, body_hist))
                print(f"[Trusted] Enrolled (full image): {fname}")
            else:
                # Use the largest detected face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_crop = img[y:y+h, x:x+w]
                hist = self._compute_face_hist(face_crop)
                self._references.append((fname, hist, body_hist))
                print(f"[Trusted] Enrolled (face detected): {fname}")

        print(f"[Trusted] {len(self._references)} face photo(s) and "
              f"{len(self._cctv_refs)} camera snapshot(s) loaded.")
        if not self._cctv_refs:
            print("[Trusted] Tip: click the trusted person in the video window to "
                  "register them from this camera (works far better than a face photo).")

    def is_trusted(self, frame, x1, y1, x2, y2):
        """
        Check if the person in the bounding box is a trusted person.

          1. A clearly visible face (>= 40 px) matching a face photo
          2. The whole person matching a camera snapshot (overhead CCTV)

        Args:
            frame: full video frame (BGR)
            x1, y1, x2, y2: person bounding box

        Returns:
            (is_trusted: bool, match_name: str or None, confidence: float)
        """
        if not self.enabled:
            return False, None, 0.0
        if not self._references and not self._cctv_refs:
            return False, None, 0.0

        h, w = frame.shape[:2]
        # Clamp to frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return False, None, 0.0

        # ── Tier 1: Face detection + histogram match ──────────────────────
        # Only faces big enough to tell apart: tiny "faces" on an overhead
        # camera are mostly false detections, and skin tones correlate
        min_face = 40
        faces = ()
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        if (self._references and self._face_cascade is not None
                and person_crop.shape[1] >= min_face):
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(min_face, min_face)
            )

        if len(faces) > 0:
            # Get the largest face
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_crop = person_crop[fy:fy+fh, fx:fx+fw]

            if face_crop.size > 0 and face_crop.shape[0] >= 10 and face_crop.shape[1] >= 10:
                face_hist = self._compute_face_hist(face_crop)

                best_score = 0.0
                best_name = None
                for ref_name, ref_hist, _ in self._references:
                    score = cv2.compareHist(face_hist, ref_hist, cv2.HISTCMP_CORREL)
                    if score > best_score:
                        best_score = score
                        best_name = ref_name

                if best_score >= config.FACE_MATCH_THRESHOLD:
                    return True, best_name, best_score

        # ── Tier 2: camera snapshots of the person (clothes / build) ──────
        # (The old fallback compared shirt colours with the face PHOTO,
        # which matched strangers - it is not used any more.)
        name, score = self.best_cctv_match(person_crop)
        if name is not None and score >= config.CCTV_MATCH_THRESHOLD:
            return True, name, score
        return False, None, max(score, 0.0)


