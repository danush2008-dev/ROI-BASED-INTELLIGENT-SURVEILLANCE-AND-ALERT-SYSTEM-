"""
trusted_person.py — Face-based trusted person recognition.

Uses OpenCV's Haar cascade for face detection and histogram-based
comparison for recognition.  Trusted reference images are stored in
the `trusted_faces/` directory (one or more .jpg/.png per person).

When a detected person's face matches a trusted reference (correlation
above FACE_MATCH_THRESHOLD), they are assigned track ID -1 and all
alerts are suppressed.
"""

import os
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
        self._references = []  # list of (name, hist) tuples
        self._face_cascade = None

        if not self._enabled:
            print("[Trusted] Trusted person recognition disabled.")
            return

        # Load Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

        if self._face_cascade.empty():
            print("[Trusted] WARNING: Could not load face cascade classifier.")
            self._enabled = False
            return

        # Load reference faces
        self._load_references()

    @property
    def enabled(self):
        return self._enabled and len(self._references) > 0

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

        for fname in files:
            img_path = os.path.join(faces_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Trusted] Could not read: {fname}")
                continue

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

        print(f"[Trusted] {len(self._references)} reference face(s) loaded.")

    def is_trusted(self, frame, x1, y1, x2, y2):
        """
        Check if the person in the bounding box matches a trusted face.

        Uses a two-tier approach:
          1. Try face detection + histogram match (most accurate)
          2. Fallback: upper body histogram match (works when face not visible)

        Args:
            frame: full video frame (BGR)
            x1, y1, x2, y2: person bounding box

        Returns:
            (is_trusted: bool, match_name: str or None, confidence: float)
        """
        if not self.enabled:
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
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15)
        )

        if len(faces) == 0:
            # Try with the upper half of the person box (head region)
            upper_half = person_crop[:person_crop.shape[0] // 2, :]
            if upper_half.size > 0:
                gray_upper = cv2.cvtColor(upper_half, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(
                    gray_upper, scaleFactor=1.05, minNeighbors=2, minSize=(12, 12)
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

        # ── Tier 2: Body histogram fallback ───────────────────────────────
        # Compare upper body color profile (works when face isn't visible)
        upper_body = person_crop[:int(person_crop.shape[0] * 0.6), :]
        if upper_body.size == 0:
            return False, None, 0.0

        body_hist = self._compute_face_hist(upper_body)
        body_threshold = 0.15  # Low threshold for body matching (less precise)

        best_body_score = 0.0
        best_body_name = None
        for ref_name, _, ref_body_hist in self._references:
            score = cv2.compareHist(body_hist, ref_body_hist, cv2.HISTCMP_CORREL)
            if score > best_body_score:
                best_body_score = score
                best_body_name = ref_name

        # Debug: print score every ~30 checks
        if best_body_score >= body_threshold:
            return True, best_body_name, best_body_score

        return False, None, max(best_body_score, 0.0)


