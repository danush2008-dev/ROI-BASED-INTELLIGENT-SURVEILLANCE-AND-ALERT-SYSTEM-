"""
phy.py — Main entry point for ROI-Based Intelligent Surveillance System.

Pipeline:
  Video → YOLOv5 → confidence filter → foot-point ROI check → ByteTrack IDs
  → temporal confirmation → pose estimation → dwell time + severity
  → actions (log/email/alarm)

Run:
  python phy.py                                          # default video
  python phy.py --source "input vedio for testing.mp4"   # specific video
  python phy.py --source 0                               # webcam
  python phy.py --dry-run                                # no email/alarm
  python phy.py --headless                               # no GUI windows
  python phy.py --sim-time 22:00                         # simulate night
"""
import faulthandler
faulthandler.enable()

import os
import sys
import math
import json
import time
import argparse
import warnings
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np

# Suppress noisy warnings before importing heavy modules
warnings.filterwarnings("ignore")

import config
from roi_manager import ROIManager
from detector import Detector
from alerts import AlertWorker
from evidence import EvidenceManager, blur_non_intruders
from confirmation import TemporalConfirmer
from tracker import Tracker, DwellTracker
from pose_estimator import PoseEstimator, update_motion, foot_speed
from event_db import EventDB
from trusted_person import TrustedPersonDetector
import severity as sev


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ROI-Based Intelligent Surveillance and Alert System"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video file path or camera index (0 for webcam). "
             "Default: value from config.DEFAULT_SOURCE"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI windows (for evaluation/server use)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Disable email sending and alarm playback."
    )
    parser.add_argument(
        "--sim-time", type=str, default=None,
        help="Simulate a time of day for testing, e.g. '22:00'."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Enable demo mode with clean overlay for presentation recording."
    )
    parser.add_argument(
        "--config-json", type=str, default=None,
        help="JSON string of config overrides (used by ablation runner)."
    )
    parser.add_argument(
        "--draw-roi", action="store_true",
        help="Force interactive ROI drawing (opens video frame for click-to-draw)."
    )
    return parser.parse_args()


def _draw_box_label(frame, box, label, colour, thickness):
    """Box with a filled label bar above it (black text)."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    # keep the label on screen near the top / right edges
    ty = max(y1, th + 12)
    lx = max(0, min(x1, frame.shape[1] - tw - 8))
    cv2.rectangle(frame, (lx, ty - th - 12), (lx + tw + 8, ty), colour, -1)
    cv2.putText(frame, label, (lx + 4, ty - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def _draw_status(frame, severity, reasons, dwell):
    """Top-left severity banner (+ dwell when given) and bottom alert text."""
    colour = sev.SEVERITY_COLORS.get(severity, (0, 0, 255))
    banner = f"{severity} ALERT: {', '.join(reasons)}"
    (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (5, 10), (15 + bw, 40), colour, -1)
    cv2.putText(frame, banner, (10, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2)

    if dwell is not None and dwell > 0:
        cv2.putText(frame, f"Dwell: {dwell:.1f}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    if severity == sev.MEDIUM:
        cv2.putText(frame, "MEDIUM ALERT - Person detected in ROI",
                    (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 165, 255), 2)
    elif severity == sev.HIGH:
        msg = "!! HIGH ALERT - INTRUDER DETECTED !!"
        (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        mx = (frame.shape[1] - mw) // 2
        cv2.rectangle(frame, (mx - 10, frame.shape[0] - 65),
                      (mx + mw + 10, frame.shape[0] - 30), (0, 0, 255), -1)
        cv2.putText(frame, msg, (mx, frame.shape[0] - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def _iou(a, b):
    """Intersection-over-union of two detection dicts (x1, y1, x2, y2)."""
    ix1, iy1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    ix2, iy2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    return inter / float(area_a + area_b - inter)


def main():
    args = parse_args()

    # ── Apply CLI overrides to config ──────────────────────────────────────
    if args.sim_time:
        config.SIMULATED_TIME = args.sim_time
    if args.demo:
        config.DEMO_MODE = True

    # ── Apply JSON config overrides (ablation testing) ─────────────────────
    overrides = {}
    if args.config_json:
        try:
            overrides = json.loads(args.config_json)
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"[Config] Override: {key} = {value}")
        except json.JSONDecodeError as e:
            print(f"[WARNING] Invalid --config-json: {e}")
            overrides = {}

    def set_default(key, value):
        """Per-video setting; an explicit --config-json override wins."""
        if key not in overrides:
            setattr(config, key, value)

    # ── Determine video source ─────────────────────────────────────────────
    source = args.source if args.source is not None else config.DEFAULT_SOURCE
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # Work from the project folder, wherever phy.py was started from, so the
    # evidence / photos / email log / databases always land where the
    # dashboard looks for them.
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if isinstance(source, str):
        if not os.path.exists(source) and os.path.exists(os.path.join(project_dir, source)):
            source = os.path.join(project_dir, source)
        source = os.path.abspath(source)
    os.chdir(project_dir)

    run_tag = "input_video"     # label for this source in the dashboard

    # ── Per-video path isolation ───────────────────────────────────────────
    # When running on a video file (not webcam), derive a tag from the
    # filename and give each video its own ROI file, evidence directory,
    # database, and photos folder so they don't overwrite each other.
    if isinstance(source, str):
        _base = os.path.splitext(os.path.basename(source))[0]
        _tag = _base.replace(" ", "_")

        # Only isolate if this is NOT the original default video
        if _base != os.path.splitext(config.DEFAULT_SOURCE)[0]:
            run_tag = _tag
            set_default("ROI_JSON_PATH",       f"rois_{_tag}.json")
            set_default("EVIDENCE_DIR",        f"evidence_{_tag}")
            set_default("CLIPS_DIR",           os.path.join(config.EVIDENCE_DIR, "clips"))
            set_default("EVIDENCE_LOG_PATH",   os.path.join(config.EVIDENCE_DIR, "evidence_log.jsonl"))
            set_default("DB_PATH",             f"events_{_tag}.db")
            set_default("DETECTED_PHOTOS_DIR", f"detected_photos_{_tag}")
            set_default("DEMO_OUTPUT_PATH",    f"demo_output_{_tag}.mp4")
            print(f"[INFO] Per-video paths set for '{_base}':")
            print(f"       ROI file  : {config.ROI_JSON_PATH}")
            print(f"       Evidence  : {config.EVIDENCE_DIR}/")
            print(f"       Database  : {config.DB_PATH}")
            print(f"       Photos    : {config.DETECTED_PHOTOS_DIR}/")

    # ── Per-video behaviour overrides ──────────────────────────────────────
    if isinstance(source, str):
        _src_lower = os.path.basename(source).lower()

        # Video 1: running person -> HIGH alert + alarm (default behaviour)
        #          Trusted person detection OFF
        if 'video 1' in _src_lower or 'video_1' in _src_lower:
            set_default("USE_TRUSTED_PERSONS", False)
            # Climbing stairs is slow in body heights / s, so a lower bar
            # than on flat ground catches someone rushing up the stairs.
            set_default("RUNNING_RAISES_ALARM", True)
            set_default("RUN_SPEED_BH_PER_S", 0.5)
            set_default("RUNNING_HOLD_SECONDS", 3.0)
            set_default("SEVERITY_LATCH", True)
            print("[INFO] Video 1: Trusted person detection DISABLED.")
            print(f"[INFO] Video 1: Running up the stairs (>= {config.RUN_SPEED_BH_PER_S} "
                  f"body heights/s) -> HIGH alert + audible alarm.")

        # Video 2: trusted uncle detection ON
        elif 'video 2' in _src_lower or 'video_2' in _src_lower:
            set_default("USE_TRUSTED_PERSONS", True)
            print("[INFO] Video 2: Trusted person detection ENABLED (uncle).")

        # Video 3: vehicle passing -> LOW, person walking past -> LOW (log only)
        #          Trusted person detection OFF
        elif 'video 3' in _src_lower or 'video_3' in _src_lower:
            set_default("ALERT_ON_VEHICLES", True)
            set_default("USE_TRUSTED_PERSONS", False)
            set_default("WALKING_IS_LOW", True)
            set_default("PASSERBY_IS_LOW", True)
            set_default("RUNNING_RAISES_ALARM", False)
            print("[INFO] Video 3: Vehicle alerts ENABLED (LOW).")
            print("[INFO] Video 3: Person walking past -> LOW (log only, no email/alarm).")
            print("[INFO] Video 3: Trusted person detection DISABLED.")

    # ── Open video capture ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame from video source.")
        cap.release()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Video properties ───────────────────────────────────────────────────
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    is_live = isinstance(source, int)  # True for webcam

    # ── Preprocess helper ──────────────────────────────────────────────────
    def preprocess(img):
        height, width = img.shape[:2]
        ratio = height / width
        new_w = config.INFERENCE_SIZE
        new_h = int(new_w * ratio)
        return cv2.resize(img, (new_w, new_h))

    first_frame_resized = preprocess(first_frame)

    # ── ROI setup ──────────────────────────────────────────────────────────
    roi_mgr = ROIManager()
    loaded = roi_mgr.load()

    if args.draw_roi:
        # Force interactive drawing (clear old ROIs so user starts fresh)
        roi_mgr.rois = []
        print("[INFO] Interactive ROI drawing mode. Draw your zone(s) on the video frame.")
        roi_mgr.draw_interactive(first_frame_resized)
    elif not loaded and not args.headless:
        print("[INFO] No saved ROIs found. Please draw ROI(s) on the video frame.")
        roi_mgr.draw_interactive(first_frame_resized)
    elif not loaded and args.headless:
        print("[WARNING] No ROIs found and running headless -- "
              "no intrusion detection will be performed.")

    roi_bounds = roi_mgr.get_combined_bounding_rect()

    # ── Initialize modules ─────────────────────────────────────────────────
    print("[INFO] Loading YOLOv5 model...")
    detector = Detector()
    print("[INFO] Model loaded.")

    alert_worker = AlertWorker(dry_run=args.dry_run, source_label=run_tag)
    evidence_mgr = EvidenceManager()
    confirmer = TemporalConfirmer()
    tracker = Tracker()
    dwell_tracker = DwellTracker()
    pose_estimator = PoseEstimator()
    event_db = EventDB()
    trusted_detector = TrustedPersonDetector()

    # Track IDs confirmed as trusted (skip alerts for these)
    trusted_track_ids = {}  # {incident_id: match_name}

    # Set tracker frame rate
    tracker.set_frame_rate(video_fps)

    # Initialize clip recorder with actual FPS
    evidence_mgr.init_clip_recorder(video_fps)

    # ── Per-track state ────────────────────────────────────────────────────
    last_email_time = {}        # {track_id: datetime}
    track_max_severity = {}     # {track_id: severity_level}
    confirmed_intruder_ids = set()  # track IDs confirmed as intruders this frame
    track_behaviours = {}       # {incident_id: str} last known pose
    track_pose_history = {}     # {incident_id: deque of recent pose labels}
    track_pose_debug = {}       # {incident_id: debug string}
    track_foot_hist = {}        # {incident_id: deque of (time, x, y) foot points}
    track_running_until = {}    # {incident_id: video time until which 'running' holds}
    track_last_seen = {}        # {track_id: last box/label/colour, for brief dropouts}
    track_max_reasons = {}      # {incident_id: reasons behind its highest severity}
    confirmed_until = {}        # {entity_key: video time its confirmation is held}
    last_status = None          # ((severity, reasons, dwell), video_time) for the banner
    ALARM_COOLDOWN_SECONDS = 5
    last_alarm_time = None

    # ── Frame counters ─────────────────────────────────────────────────────
    frame_count = 0
    fps_start_time = datetime.now()
    fps_frame_count = 0
    display_fps = 0.0
    last_detections = []
    last_raw_detections = []

    # ── Run statistics (saved to the dashboard at the end of the run) ──────
    run_started = time.time()
    run_started_iso = datetime.now().isoformat(timespec="seconds")
    stat_detections = 0
    stat_severity = Counter()
    stat_incidents = set()
    stat_dwell_sum = 0.0
    stat_dwell_max = 0.0
    pending_db_events = []      # SQLite inserts are batched (one commit per batch)

    # ── Display ────────────────────────────────────────────────────────────
    if not args.headless:
        cv2.namedWindow('Video')
    demo_writer = None

    # ── Print startup info ─────────────────────────────────────────────────
    print(f"\n[INFO] Starting surveillance on source: {source}")
    print(f"[INFO] Headless={args.headless}, DryRun={args.dry_run}")
    print(f"[INFO] ROIs loaded: {len(roi_mgr.rois)} from {config.ROI_JSON_PATH}")
    for _roi in roi_mgr.rois:
        print(f"[INFO]   {_roi['name']}: {_roi['points']}")
    print(f"[INFO] Tracking: {config.USE_TRACKING} (ByteTrack)")
    print(f"[INFO] Foot-point: {config.USE_FOOT_POINT}")
    print(f"[INFO] Conf threshold: {config.CONF_THRESHOLD if config.USE_CONF_THRESHOLD else 'off'}")
    print(f"[INFO] Temporal confirm: {config.USE_TEMPORAL_CONFIRMATION} "
          f"(N={config.TEMPORAL_N}, M={config.TEMPORAL_M})")
    if config.SIMULATED_TIME:
        print(f"[INFO] Simulated time: {config.SIMULATED_TIME}")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN VIDEO LOOP
    # ═══════════════════════════════════════════════════════════════════════
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video ended or frame could not be read.")
                break

            frame_count += 1
            fps_frame_count += 1
            frame = preprocess(frame)

            # ── Compute video time (seconds from start) ───────────────────
            if is_live:
                video_time = (datetime.now() - fps_start_time).total_seconds()
            else:
                video_time = frame_count / video_fps

            # ── FPS calculation ────────────────────────────────────────────
            elapsed = (datetime.now() - fps_start_time).total_seconds()
            if elapsed >= 1.0:
                display_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = datetime.now()

            # ── Detection (with frame skipping) ────────────────────────────
            use_tracker = config.USE_TRACKING and tracker.enabled
            run_detection = (frame_count % config.DETECT_EVERY_N_FRAMES == 0)

            if run_detection and len(roi_mgr.rois) > 0:
                # With tracking, low-confidence boxes are passed on as well:
                # ByteTrack only uses them to keep EXISTING tracks alive (a
                # person half hidden by the stair railing or the gate), never
                # to start new ones - so a person's box doesn't vanish as soon
                # as YOLO gets less sure of them inside the ROI.
                all_detections = detector.detect(
                    frame, roi_bounds=roi_bounds,
                    min_conf=config.TRACK_LOW_CONF if use_tracker else None)
                raw_detections = [
                    d for d in all_detections
                    if not config.USE_CONF_THRESHOLD
                    or d['confidence'] >= config.CONF_THRESHOLD]
                stat_detections += len(raw_detections)

                # ── ByteTrack tracking ─────────────────────────────────────
                if use_tracker:
                    detections = tracker.update(all_detections, frame.shape[:2])
                else:
                    detections = raw_detections

                last_detections = detections
                last_raw_detections = raw_detections
            elif not run_detection and len(roi_mgr.rois) > 0:
                detections = last_detections
                raw_detections = last_raw_detections
            else:
                detections = []
                raw_detections = []

            # ── Push frame to clip recorder circular buffer ────────────────
            clip_done = evidence_mgr.push_frame(frame)
            if clip_done:
                print(f"[Main] Clip finalized: {clip_done}")

            # ── Draw ROIs first so every bounding box is drawn on top ─────
            if len(roi_mgr.rois) > 0:
                roi_mgr.draw_rois_on_frame(frame)

            # ── Boxes the tracker has not picked up (yet) ─────────────────
            # ByteTrack drops detections it cannot match - typically a brand-new
            # track. Draw them anyway so a person is never invisible.
            if use_tracker:
                for raw in raw_detections:
                    if any(_iou(raw, d) > 0.3 for d in detections):
                        continue
                    rx1, ry1, rx2, ry2 = raw['x1'], raw['y1'], raw['x2'], raw['y2']
                    foot = ROIManager.get_foot_point(rx1, ry1, rx2, ry2)
                    inside = any(roi_mgr.box_in_roi(r, rx1, ry1, rx2, ry2, foot)
                                 for r in roi_mgr.rois)
                    colour = (0, 165, 255) if inside else (255, 255, 0)
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), colour, 3 if inside else 1)
                    cv2.putText(frame, f"{raw['name']} (tracking...)", (rx1, ry1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

            # ── Per-frame state ────────────────────────────────────────────
            active_entity_keys = set()
            confirmed_intruder_ids = set()  # reset per-frame
            seen_track_ids = set()
            motion_cache = {}       # {key: (move_px, speed_bh, is_moving, is_running)}
            frame_alerts = []       # (severity, reasons, dwell or None) for the banner
            frame_events = []       # evidence to save once the frame is fully drawn

            def get_motion(key, foot, box_h):
                """Foot-point motion for one person, updated once per frame."""
                if key in motion_cache:
                    return motion_cache[key]
                fh = track_foot_hist.setdefault(key, deque())
                move_px = update_motion(fh, video_time, foot[0], foot[1])
                speed_bh = foot_speed(fh, video_time, box_h)
                is_moving = config.USE_MOTION_CUE and move_px >= config.MOVE_MIN_PIXELS
                was_running = video_time < track_running_until.get(key, -1.0)
                if config.USE_MOTION_CUE and speed_bh >= config.RUN_SPEED_BH_PER_S:
                    track_running_until[key] = video_time + config.RUNNING_HOLD_SECONDS
                    if not was_running:
                        print(f"[Motion] id={key} RUNNING "
                              f"({speed_bh:.2f} body heights/s)")
                is_running = video_time < track_running_until.get(key, -1.0)
                motion_cache[key] = (move_px, speed_bh, is_moving, is_running)
                return motion_cache[key]

            # ── Trusted-person check (before counting people in the ROI) ──
            for det in detections:
                if det['name'] != config.INTRUSION_CLASS:
                    continue
                track_id = det.get('track_id')
                t_key = str(track_id) if track_id is not None else 'unk'
                if not trusted_detector.enabled or t_key in trusted_track_ids:
                    continue
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']

                # A trusted person whose track was briefly lost (and came back
                # with a new ID) keeps their trusted status
                if track_id is not None:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    for tid, info in track_last_seen.items():
                        if (info.get('trusted') and info['t'] < video_time
                                and video_time - info['t'] <= config.TRUST_INHERIT_SECONDS):
                            ix1, iy1, ix2, iy2 = info['box']
                            dist = math.hypot(cx - (ix1 + ix2) / 2, cy - (iy1 + iy2) / 2)
                            if dist <= 0.75 * max(y2 - y1, iy2 - iy1):
                                trusted_track_ids[t_key] = info['trusted']
                                print(f"[Trusted] ID {t_key} continues trusted "
                                      f"track {tid} ({info['trusted']})")
                                break
                    if t_key in trusted_track_ids:
                        continue

                if frame_count % 2 == 0:  # check every 2nd frame
                    is_trusted, match_name, match_conf = (
                        trusted_detector.is_trusted(frame, x1, y1, x2, y2)
                    )
                    if is_trusted:
                        # Derive friendly name from filename (e.g. "uncle.jpeg" -> "Uncle")
                        friendly = match_name or "Unknown"
                        friendly = os.path.splitext(friendly)[0]  # strip ext
                        friendly = os.path.splitext(friendly)[0]  # strip double ext
                        friendly = friendly.replace('_', ' ').replace('-', ' ').title()
                        trusted_track_ids[t_key] = friendly
                        print(f"[Trusted] ID {t_key} recognised as {friendly} "
                              f"(score {match_conf:.2f})")

            # ── First pass: count (non-trusted) persons per ROI ────────────
            persons_per_roi = {roi['name']: 0 for roi in roi_mgr.rois}
            for det in detections:
                if det['name'] != config.INTRUSION_CLASS:
                    continue
                t_id = det.get('track_id')
                if str(t_id if t_id is not None else 'unk') in trusted_track_ids:
                    continue
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                if config.USE_FOOT_POINT:
                    tp = ROIManager.get_foot_point(x1, y1, x2, y2)
                else:
                    tp = ROIManager.get_center_point(x1, y1, x2, y2)
                for roi in roi_mgr.rois:
                    if roi_mgr.box_in_roi(roi, x1, y1, x2, y2, tp):
                        persons_per_roi[roi['name']] += 1

            # ── Second pass: process each detection ────────────────────────
            for det in detections:
                name = det['name']
                x1, y1 = det['x1'], det['y1']
                x2, y2 = det['x2'], det['y2']
                track_id = det.get('track_id')  # None if tracking is off
                if track_id is not None:
                    seen_track_ids.add(track_id)

                # Determine test point
                if config.USE_FOOT_POINT and name == config.INTRUSION_CLASS:
                    test_point = ROIManager.get_foot_point(x1, y1, x2, y2)
                else:
                    test_point = ROIManager.get_center_point(x1, y1, x2, y2)

                # ── Default bounding box (cyan, thin) — will be
                #    overdrawn with severity color if inside ROI ──────────
                default_color = (255, 255, 0)  # cyan
                cv2.rectangle(frame, (x1, y1), (x2, y2), default_color, 1)
                label_parts = [name]
                if track_id is not None:
                    label_parts.append(f"ID:{track_id}")
                cv2.putText(frame, " ".join(label_parts), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, default_color, 1)
                if track_id is not None:
                    track_last_seen[track_id] = {
                        't': video_time, 'box': (x1, y1, x2, y2),
                        'colour': default_color, 'label': " ".join(label_parts),
                        'thick': 1, 'trusted': None}

                # Draw test point
                if config.USE_FOOT_POINT and name == config.INTRUSION_CLASS:
                    tx, ty = test_point
                    tri_pts = np.array([
                        [tx - 6, ty - 8], [tx + 6, ty - 8], [tx, ty + 2]
                    ], dtype=np.int32)
                    cv2.fillPoly(frame, [tri_pts], (0, 165, 255))
                else:
                    cv2.circle(frame, test_point, 4, (0, 0, 255), -1)

                # ── Skip non-intrusion classes ─────────────────────────────
                is_vehicle = name in config.VEHICLE_CLASSES
                if not is_vehicle and name != config.INTRUSION_CLASS:
                    continue
                if is_vehicle and not config.ALERT_ON_VEHICLES:
                    continue

                # ── Motion (every frame, inside the ROI or not, so a person
                #    already running when they reach the ROI is caught) ─────
                if name == config.INTRUSION_CLASS and track_id is not None:
                    get_motion(str(track_id), test_point, y2 - y1)

                # ── Trusted person: green box, LOW, no alert ───────────────
                cur_incident_id = str(track_id) if track_id is not None else 'unk'
                if name == config.INTRUSION_CLASS and cur_incident_id in trusted_track_ids:
                    person_name = trusted_track_ids[cur_incident_id]
                    trusted_roi = next(
                        (r for r in roi_mgr.rois
                         if roi_mgr.box_in_roi(r, x1, y1, x2, y2, test_point)), None)
                    trust_label = f"TRUSTED - {person_name}"
                    if trusted_roi is not None:
                        trust_label += " | LOW"
                    _draw_box_label(frame, (x1, y1, x2, y2), trust_label, (0, 200, 0), 3)
                    cv2.putText(frame, "NO ALERT",
                                (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                    if track_id is not None:
                        track_last_seen[track_id].update(
                            colour=(0, 200, 0), label=trust_label, thick=3,
                            trusted=person_name)
                    if trusted_roi is not None:
                        reasons = [f"trusted_person:{person_name}"]
                        frame_alerts.append((sev.LOW, reasons, None))
                        frame_events.append(dict(
                            incident_id=cur_incident_id, severity=sev.LOW,
                            reasons=reasons, dwell_seconds=0.0, behaviour=None,
                            roi_name=trusted_roi['name'], track_id=track_id,
                            force_photo=False, escalated=False, prev_severity=None))
                    continue

                # ── ROI check for each ROI ─────────────────────────────────
                for roi in roi_mgr.rois:
                    if name == config.INTRUSION_CLASS:
                        in_roi = roi_mgr.box_in_roi(roi, x1, y1, x2, y2, test_point)
                    else:
                        in_roi = ROIManager.point_in_polygon(test_point, roi["points"])

                    # ── Entity key for confirmation ────────────────────────
                    if config.USE_TRACKING and track_id is not None:
                        entity_key = f"t{track_id}"
                    else:
                        entity_key = confirmer.match_by_position(
                            test_point[0], test_point[1]
                        )
                    active_entity_keys.add(entity_key)

                    # ── Temporal confirmation ──────────────────────────────
                    if config.USE_TEMPORAL_CONFIRMATION:
                        confirmed = confirmer.update(entity_key, in_roi)
                        pos_count, _ = confirmer.get_count(entity_key)
                        # Once confirmed, a few missed detections (occlusion)
                        # don't send the person back to "unconfirmed"
                        if confirmed:
                            confirmed_until[entity_key] = (
                                video_time + config.CONFIRM_HOLD_SECONDS)
                        elif in_roi and video_time <= confirmed_until.get(entity_key, -1.0):
                            confirmed = True

                        if in_roi and not confirmed:
                            unconf_label = f"Unconfirmed ({pos_count}/{config.TEMPORAL_N})"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(frame, unconf_label,
                                        (x1, y1 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                            if track_id is not None:
                                track_last_seen[track_id].update(
                                    colour=(0, 255, 255), label=unconf_label, thick=2)
                            # Update dwell even when unconfirmed (for accurate timing)
                            if config.USE_TRACKING and track_id is not None:
                                dwell_tracker.update(track_id, roi['name'], in_roi, video_time)
                            continue

                        if not in_roi:
                            # Update dwell tracker with out-of-ROI state
                            if config.USE_TRACKING and track_id is not None:
                                dwell_tracker.update(track_id, roi['name'], False, video_time)
                            continue
                    else:
                        if not in_roi:
                            continue

                    # ═══════════════════════════════════════════════════════
                    # CONFIRMED INTRUSION
                    # ═══════════════════════════════════════════════════════
                    incident_id = str(track_id) if track_id is not None else entity_key

                    # ── Motion cue: is this person's foot point travelling? ─
                    move_px, speed_bh, is_moving, is_running = get_motion(
                        incident_id, test_point, y2 - y1)

                    # ── Dwell time ─────────────────────────────────────────
                    if config.USE_TRACKING and track_id is not None:
                        dwell_s = dwell_tracker.update(
                            track_id, roi['name'], True, video_time
                        )
                    else:
                        dwell_s = 0.0

                    # ── Pose estimation (Phase 4) ──────────────────────────
                    behaviour = None
                    pose_landmarks = None
                    run_pose = (
                        config.USE_POSE
                        and pose_estimator.enabled
                        and name == config.INTRUSION_CLASS
                        and frame_count % config.POSE_EVERY_N_FRAMES == 0
                    )
                    if run_pose:
                        behaviour, pose_landmarks = pose_estimator.classify(
                            frame, x1, y1, x2, y2
                        )
                        # A person walking across the room is not sitting
                        moved_override = False
                        if is_moving and behaviour == 'sitting':
                            behaviour = 'standing'
                            moved_override = True
                        # Smooth: majority vote over the last few pose results
                        # so the label doesn't flicker between sitting/standing
                        hist = track_pose_history.setdefault(
                            incident_id,
                            deque(maxlen=config.POSE_SMOOTH_FRAMES))
                        if behaviour != 'unknown':
                            hist.append(behaviour)
                        if hist:
                            behaviour = Counter(hist).most_common(1)[0][0]
                        track_behaviours[incident_id] = behaviour
                        track_pose_debug[incident_id] = (
                            f"{pose_estimator.last_debug} mv:{move_px:.0f} "
                            f"spd:{speed_bh:.2f}bh/s"
                            + (" MOVING" if moved_override else ""))
                        if config.DEBUG_POSE and frame_count % 15 == 0:
                            print(f"[PoseDebug] id={incident_id} "
                                  f"{track_pose_debug[incident_id]}")
                    else:
                        # Use cached behaviour from last pose frame
                        behaviour = track_behaviours.get(incident_id)

                    # ── Running: fast foot-point motion (any pose but lying)
                    #    Checked every frame, so it works without pose too.
                    if name == config.INTRUSION_CLASS and behaviour != 'lying':
                        if is_running:
                            behaviour = 'running'
                        elif config.PASSERBY_IS_LOW and is_moving:
                            behaviour = 'walking'

                    # ── Draw skeleton + behaviour label ────────────────────
                    if pose_landmarks:
                        pose_estimator.draw_landmarks(frame, pose_landmarks)

                    # ── Severity ───────────────────────────────────────────
                    cur_severity, reasons = sev.compute_severity(
                        dwell_seconds=dwell_s,
                        person_count_in_roi=persons_per_roi.get(roi['name'], 1),
                        behavior=behaviour,
                        roi=roi,
                        is_vehicle=is_vehicle,
                        is_moving=is_moving,
                    )
                    severity_color = sev.SEVERITY_COLORS.get(cur_severity, (0, 0, 255))

                    # ── Thick box in severity colour + ID | Severity | Dwell | Posture
                    overlay_parts = [cur_severity]
                    if track_id is not None:
                        overlay_parts.append(f"ID:{track_id}")
                    if dwell_s > 0:
                        overlay_parts.append(f"Dwell:{dwell_s:.1f}s")
                    if behaviour and behaviour != 'unknown':
                        overlay_parts.append(behaviour.upper())
                    overlay_label = " | ".join(overlay_parts)
                    _draw_box_label(frame, (x1, y1, x2, y2), overlay_label,
                                    severity_color, 3)
                    if track_id is not None:
                        track_last_seen[track_id].update(
                            colour=severity_color, label=overlay_label, thick=3)

                    # Posture label below the box
                    if behaviour and behaviour != 'unknown':
                        pose_color = (0, 0, 255) if behaviour == 'running' else severity_color
                        pose_text = f"POSTURE: {behaviour.upper()}"
                        (pw, _), _ = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.putText(frame, pose_text,
                                    (max(0, min(x1, frame.shape[1] - pw - 4)),
                                     min(y2 + 20, frame.shape[0] - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, pose_color, 2)

                    frame_alerts.append((cur_severity, reasons, dwell_s))

                    # ── Track this as a confirmed intruder ─────────────────
                    confirmed_intruder_ids.add(incident_id)

                    # ── Check for severity escalation ──────────────────────
                    prev_severity = track_max_severity.get(incident_id, None)
                    escalated = (
                        prev_severity is not None
                        and sev.severity_escalated(prev_severity, cur_severity)
                    )
                    # Only store upward — never downgrade the max
                    if prev_severity is None or escalated:
                        track_max_severity[incident_id] = cur_severity
                        track_max_reasons[incident_id] = reasons
                    elif (config.SEVERITY_LATCH and prev_severity != cur_severity
                          and sev.severity_escalated(cur_severity, prev_severity)):
                        # Video 1: someone who ran up the stairs stays HIGH
                        # while they are in the ROI
                        cur_severity = prev_severity
                        reasons = track_max_reasons.get(incident_id, reasons)
                        severity_color = sev.SEVERITY_COLORS.get(cur_severity, (0, 0, 255))
                        overlay_label = " | ".join([cur_severity] + overlay_parts[1:])
                        _draw_box_label(frame, (x1, y1, x2, y2), overlay_label,
                                        severity_color, 3)
                        if track_id is not None:
                            track_last_seen[track_id].update(
                                colour=severity_color, label=overlay_label)
                        frame_alerts[-1] = (cur_severity, reasons, dwell_s)

                    frame_events.append(dict(
                        incident_id=incident_id, severity=cur_severity,
                        reasons=reasons, dwell_seconds=dwell_s,
                        behaviour=behaviour, roi_name=roi['name'],
                        track_id=track_id,
                        # always photograph a severity escalation
                        force_photo=escalated, escalated=escalated,
                        prev_severity=prev_severity))

                    break  # one detection per ROI is enough

            # ── Keep showing people the detector briefly lost ──────────────
            # (occluded by the railing / gate, motion blur) at their last box
            for tid in list(track_last_seen):
                info = track_last_seen[tid]
                age = video_time - info['t']
                if age > max(config.BOX_HOLD_SECONDS, config.TRUST_INHERIT_SECONDS):
                    del track_last_seen[tid]
                    continue
                if tid in seen_track_ids or age > config.BOX_HOLD_SECONDS:
                    continue
                if info['thick'] > 1:      # only people that were in/near the ROI
                    _draw_box_label(frame, info['box'], info['label'],
                                    info['colour'], 2)

            # ── Top-left status: severity, reasons and dwell ───────────────
            # Held for a moment after the person was last confirmed, so it
            # doesn't flicker when the detector drops a frame.
            if frame_alerts:
                order = {sev.LOW: 0, sev.MEDIUM: 1, sev.HIGH: 2}
                top = max(frame_alerts, key=lambda a: order.get(a[0], 0))
                last_status = (top, video_time)
            if last_status and video_time - last_status[1] <= config.BANNER_HOLD_SECONDS:
                _draw_status(frame, *last_status[0])

            # ── Evidence, logging and alerts (frame is now fully drawn, so
            #    photos show the boxes, posture and severity banner) ────────
            if frame_events:
                evidence_frame = blur_non_intruders(
                    frame.copy(), detections, confirmed_intruder_ids)
            for ev in frame_events:
                incident_id = ev['incident_id']
                cur_severity = ev['severity']
                reasons = ev['reasons']
                dwell_s = ev['dwell_seconds']

                # ── Save evidence photo ────────────────────────────────────
                saved_path = evidence_mgr.save_photo(
                    evidence_frame, incident_id, force=ev['force_photo'])

                # ── Trigger clip on HIGH severity ───────────────────────────
                clip_path = None
                if sev.should_save_clip(cur_severity):
                    clip_path = evidence_mgr.trigger_clip(incident_id)

                # ── Log event to tamper-evident chain ──────────────────────
                log_record = evidence_mgr.log_event(
                    incident_id=incident_id,
                    severity=cur_severity,
                    reasons=reasons,
                    dwell_seconds=dwell_s,
                    behaviour=ev['behaviour'],
                    roi_name=ev['roi_name'],
                    photo_path=saved_path,
                    clip_path=clip_path,
                    track_id=ev['track_id'],
                )

                # ── Log event to SQLite database (batched) ──────────────────
                pending_db_events.append(log_record)
                if len(pending_db_events) >= 50:
                    event_db.insert_events(pending_db_events)
                    pending_db_events = []

                stat_severity[cur_severity] += 1
                stat_incidents.add(incident_id)
                stat_dwell_sum += dwell_s
                stat_dwell_max = max(stat_dwell_max, dwell_s)

                # ── Alarm (severity-gated + cooldown) ──────────────────────
                if sev.should_play_alarm(cur_severity):
                    now_alarm = datetime.now()
                    alarm_since = (
                        (now_alarm - last_alarm_time).total_seconds()
                        if last_alarm_time is not None
                        else ALARM_COOLDOWN_SECONDS + 1
                    )
                    if alarm_since >= ALARM_COOLDOWN_SECONDS:
                        alert_worker.enqueue(alert_type="alarm")
                        last_alarm_time = now_alarm

                # ── Email (severity-gated + per-track cooldown) ────────────
                if sev.should_send_email(cur_severity):
                    now = datetime.now()
                    last_time = last_email_time.get(incident_id)
                    seconds_since = (
                        (now - last_time).total_seconds()
                        if last_time is not None
                        else config.EMAIL_COOLDOWN_SECONDS + 1
                    )

                    # Send if cooldown expired OR severity escalated
                    if seconds_since >= config.EMAIL_COOLDOWN_SECONDS or ev['escalated']:
                        email_image = saved_path or evidence_mgr.get_latest_photo()
                        if email_image:
                            extra = (f"Severity: {cur_severity}\n"
                                     f"Reasons: {', '.join(reasons)}\n"
                                     f"Dwell: {dwell_s:.1f}s")
                            subject = f"Intruder Alert [{cur_severity}]"
                            alert_worker.enqueue(
                                alert_type="email",
                                image_path=email_image,
                                subject=subject,
                                extra_body=extra,
                            )
                            last_email_time[incident_id] = now
                            if ev['escalated']:
                                print(f"[Main] Severity ESCALATED "
                                      f"{ev['prev_severity']}→{cur_severity} "
                                      f"for {incident_id}")

            # ── Mark absent entities ───────────────────────────────────────
            if config.USE_TEMPORAL_CONFIRMATION:
                confirmer.mark_absent_entities(active_entity_keys)
                if frame_count % 100 == 0:
                    confirmer.cleanup_stale()

            # ── FPS overlay ────────────────────────────────────────────────
            cv2.putText(frame, f"FPS: {display_fps:.1f}",
                        (frame.shape[1] - 140, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ── Demo output recording ─────────────────────────────────────
            if config.DEMO_MODE:
                if demo_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    demo_writer = cv2.VideoWriter(
                        config.DEMO_OUTPUT_PATH, fourcc, video_fps, (w, h)
                    )
                demo_writer.write(frame)

            # ── Display ───────────────────────────────────────────────────
            if not args.headless:
                cv2.imshow("Video", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quit requested.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # ── Cleanup ────────────────────────────────────────────────────────────
    print("[INFO] Cleaning up...")
    cap.release()
    if demo_writer is not None:
        demo_writer.release()
    evidence_mgr.release()
    if not args.headless:
        cv2.destroyAllWindows()
    alert_worker.shutdown()

    if pending_db_events:
        event_db.insert_events(pending_db_events)

    # ── Push this run's events into the dashboard database right away ──────
    if not args.config_json:
        try:
            n_new = EventDB(db_path=config.DASHBOARD_DB_PATH).sync_from_jsonl(
                config.EVIDENCE_LOG_PATH, run_label=run_tag)
            print(f"[INFO] Dashboard updated: {n_new} new events for '{run_tag}'")
        except Exception as e:
            print(f"[WARNING] Could not update dashboard database: {e}")

    # ── Record this run for the dashboard (Ablation tab) ───────────────────
    # Ablation profiles (--config-json) are recorded by ablation.py itself.
    if not args.config_json:
        n_events = sum(stat_severity.values())
        wall = time.time() - run_started
        try:
            EventDB(db_path=config.DASHBOARD_DB_PATH).save_ablation_run({
                "run_label": f"{run_tag} @ {run_started_iso.replace('T', ' ')}",
                "timestamp": run_started_iso,
                "config": {k: getattr(config, k) for k in (
                    "USE_FOOT_POINT", "USE_CONF_THRESHOLD",
                    "USE_TEMPORAL_CONFIRMATION", "USE_TRACKING", "USE_POSE",
                    "USE_TRUSTED_PERSONS", "ALERT_ON_VEHICLES",
                    "PASSERBY_IS_LOW", "RUNNING_RAISES_ALARM",
                    "RUN_SPEED_BH_PER_S")},
                "source": str(source),
                "total_frames": frame_count,
                "wall_seconds": round(wall, 2),
                "avg_fps": round(frame_count / wall, 1) if wall > 0 else 0.0,
                "total_detections": stat_detections,
                "confirmed_intrusions": n_events,
                "low_count": stat_severity[sev.LOW],
                "medium_count": stat_severity[sev.MEDIUM],
                "high_count": stat_severity[sev.HIGH],
                "unique_tracks": len(stat_incidents),
                "avg_dwell": round(stat_dwell_sum / n_events, 2) if n_events else 0.0,
                "max_dwell": round(stat_dwell_max, 2),
                "notes": f"{run_tag} run" + (" (dry-run)" if args.dry_run else ""),
            })
            print(f"[INFO] Run summary saved to {config.DASHBOARD_DB_PATH}")
        except Exception as e:
            print(f"[WARNING] Could not save run summary: {e}")

    # ── Verify evidence chain on exit ──────────────────────────────────────
    valid, count, err = evidence_mgr.verify_chain()
    if count > 0:
        if valid:
            print(f"[Evidence] Hash chain verified: {count} records, integrity OK ✓")
        else:
            print(f"[Evidence] ⚠ Hash chain BROKEN: {err}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
