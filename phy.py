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
from pose_estimator import PoseEstimator, update_motion
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


def main():
    args = parse_args()

    # ── Apply CLI overrides to config ──────────────────────────────────────
    if args.sim_time:
        config.SIMULATED_TIME = args.sim_time
    if args.demo:
        config.DEMO_MODE = True

    # ── Apply JSON config overrides (ablation testing) ─────────────────────
    if args.config_json:
        try:
            overrides = json.loads(args.config_json)
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"[Config] Override: {key} = {value}")
        except json.JSONDecodeError as e:
            print(f"[WARNING] Invalid --config-json: {e}")

    # ── Determine video source ─────────────────────────────────────────────
    source = args.source if args.source is not None else config.DEFAULT_SOURCE
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # ── Per-video path isolation ───────────────────────────────────────────
    # When running on a video file (not webcam), derive a tag from the
    # filename and give each video its own ROI file, evidence directory,
    # database, and photos folder so they don't overwrite each other.
    if isinstance(source, str):
        _base = os.path.splitext(os.path.basename(source))[0]
        _tag = _base.replace(" ", "_")

        # Only isolate if this is NOT the original default video
        if _base != os.path.splitext(config.DEFAULT_SOURCE)[0]:
            config.ROI_JSON_PATH       = f"rois_{_tag}.json"
            config.EVIDENCE_DIR        = f"evidence_{_tag}"
            config.CLIPS_DIR           = os.path.join(config.EVIDENCE_DIR, "clips")
            config.EVIDENCE_LOG_PATH   = os.path.join(config.EVIDENCE_DIR, "evidence_log.jsonl")
            config.DB_PATH             = f"events_{_tag}.db"
            config.DETECTED_PHOTOS_DIR = f"detected_photos_{_tag}"
            config.DEMO_OUTPUT_PATH    = f"demo_output_{_tag}.mp4"
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
            config.USE_TRUSTED_PERSONS = False
            print("[INFO] Video 1: Trusted person detection DISABLED.")
            print("[INFO] Video 1: Running person -> HIGH alert + audible alarm.")

        # Video 2: trusted uncle detection ON
        elif 'video 2' in _src_lower or 'video_2' in _src_lower:
            config.USE_TRUSTED_PERSONS = True
            print("[INFO] Video 2: Trusted person detection ENABLED (uncle).")

        # Video 3: vehicle passing -> LOW, walking person -> LOW
        #          Trusted person detection OFF
        elif 'video 3' in _src_lower or 'video_3' in _src_lower:
            config.ALERT_ON_VEHICLES = True
            config.USE_TRUSTED_PERSONS = False
            config.WALKING_IS_LOW = True  # Custom flag: keep walking at LOW
            print("[INFO] Video 3: Vehicle alerts ENABLED (LOW).")
            print("[INFO] Video 3: Walking person -> LOW alert.")
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

    alert_worker = AlertWorker(dry_run=args.dry_run)
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
    ALARM_COOLDOWN_SECONDS = 5
    last_alarm_time = None

    # ── Frame counters ─────────────────────────────────────────────────────
    frame_count = 0
    fps_start_time = datetime.now()
    fps_frame_count = 0
    display_fps = 0.0
    last_detections = []

    # ── Display ────────────────────────────────────────────────────────────
    if not args.headless:
        cv2.namedWindow('Video')
    demo_writer = None

    # ── Print startup info ─────────────────────────────────────────────────
    print(f"\n[INFO] Starting surveillance on source: {source}")
    print(f"[INFO] Headless={args.headless}, DryRun={args.dry_run}")
    print(f"[INFO] ROIs loaded: {len(roi_mgr.rois)}")
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
            run_detection = (frame_count % config.DETECT_EVERY_N_FRAMES == 0)

            if run_detection and len(roi_mgr.rois) > 0:
                detections = detector.detect(frame, roi_bounds=roi_bounds)

                # ── ByteTrack tracking ─────────────────────────────────────
                if config.USE_TRACKING and tracker.enabled:
                    detections = tracker.update(detections, frame.shape[:2])

                last_detections = detections
            elif not run_detection and len(roi_mgr.rois) > 0:
                detections = last_detections
            else:
                detections = []

            # ── Push frame to clip recorder circular buffer ────────────────
            clip_done = evidence_mgr.push_frame(frame)
            if clip_done:
                print(f"[Main] Clip finalized: {clip_done}")

            # ── Per-ROI person counting ────────────────────────────────────
            # Count confirmed persons per ROI for severity calculation
            persons_per_roi = {roi['name']: 0 for roi in roi_mgr.rois}

            # ── Track active entity keys for temporal confirmation ─────────
            active_entity_keys = set()
            confirmed_intruder_ids = set()  # reset per-frame

            # ── First pass: count persons per ROI ──────────────────────────
            for det in detections:
                if det['name'] != config.INTRUSION_CLASS:
                    continue
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                if config.USE_FOOT_POINT:
                    tp = ROIManager.get_foot_point(x1, y1, x2, y2)
                else:
                    tp = ROIManager.get_center_point(x1, y1, x2, y2)
                for roi in roi_mgr.rois:
                    if ROIManager.point_in_polygon(tp, roi["points"]):
                        persons_per_roi[roi['name']] += 1

            # ── Second pass: process each detection ────────────────────────
            for det in detections:
                name = det['name']
                x1, y1 = det['x1'], det['y1']
                x2, y2 = det['x2'], det['y2']
                conf = det['confidence']
                track_id = det.get('track_id')  # None if tracking is off

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

                # ── Trusted person check ───────────────────────────────────
                if (name == config.INTRUSION_CLASS
                        and trusted_detector.enabled
                        and frame_count % 2 == 0):  # check every 2nd frame
                    is_trusted, match_name, match_conf = (
                        trusted_detector.is_trusted(frame, x1, y1, x2, y2)
                    )
                    if is_trusted:
                        t_id = track_id if track_id is not None else 'unk'
                        # Derive friendly name from filename (e.g. "uncle.jpeg" -> "Uncle")
                        friendly = match_name or "Unknown"
                        friendly = os.path.splitext(friendly)[0]  # strip ext
                        friendly = os.path.splitext(friendly)[0]  # strip double ext
                        friendly = friendly.replace('_', ' ').replace('-', ' ').title()
                        trusted_track_ids[str(t_id)] = friendly

                # Skip alert processing for trusted persons
                cur_incident_id = str(track_id) if track_id is not None else 'unk'
                if cur_incident_id in trusted_track_ids:
                    person_name = trusted_track_ids[cur_incident_id]
                    # Draw green box with TRUSTED - Name label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
                    trust_label = f"TRUSTED - {person_name}"
                    (tw, th), _ = cv2.getTextSize(trust_label,
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1),
                                  (0, 200, 0), -1)
                    cv2.putText(frame, trust_label,
                                (x1 + 4, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(frame, "NO ALERT",
                                (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                    continue

                # ── ROI check for each ROI ─────────────────────────────────
                for roi in roi_mgr.rois:
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

                        if in_roi and not confirmed:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(frame,
                                        f"Unconfirmed ({pos_count}/{config.TEMPORAL_N})",
                                        (x1, y1 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
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
                    fh = track_foot_hist.setdefault(incident_id, deque())
                    move_px = update_motion(fh, video_time,
                                            test_point[0], test_point[1])
                    is_moving = config.USE_MOTION_CUE and move_px >= config.MOVE_MIN_PIXELS
                    is_running = config.USE_MOTION_CUE and move_px >= config.RUNNING_MIN_PIXELS

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
                        # Running detection: fast motion + upright = running
                        if is_running and behaviour in ('standing', 'unknown'):
                            behaviour = 'running'
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
                            f"{pose_estimator.last_debug} mv:{move_px:.0f}"
                            + (" MOVING" if moved_override else ""))
                        if config.DEBUG_POSE and frame_count % 15 == 0:
                            print(f"[PoseDebug] id={incident_id} "
                                  f"{track_pose_debug[incident_id]}")
                    else:
                        # Use cached behaviour from last pose frame
                        behaviour = track_behaviours.get(incident_id)

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
                    )
                    severity_color = sev.SEVERITY_COLORS.get(cur_severity, (0, 0, 255))

                    # ── Draw THICK bounding box in severity color ──────────
                    #    This overdraws the thin cyan default box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), severity_color, 3)

                    # ── Rich overlay: ID | Severity | Dwell | Posture ──────
                    overlay_parts = [cur_severity]
                    if track_id is not None:
                        overlay_parts.append(f"ID:{track_id}")
                    if dwell_s > 0:
                        overlay_parts.append(f"Dwell:{dwell_s:.1f}s")
                    if behaviour and behaviour != 'unknown':
                        overlay_parts.append(behaviour.upper())
                    overlay_label = " | ".join(overlay_parts)

                    # Draw label background for readability
                    (tw, th), _ = cv2.getTextSize(overlay_label,
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1),
                                  severity_color, -1)
                    cv2.putText(frame, overlay_label,
                                (x1 + 4, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                    # Posture label below the box
                    if behaviour and behaviour != 'unknown':
                        pose_color = (0, 0, 255) if behaviour == 'running' else severity_color
                        cv2.putText(frame, f"POSTURE: {behaviour.upper()}",
                                    (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, pose_color, 2)

                    # Top-left severity status banner
                    banner = f"{cur_severity} ALERT: {', '.join(reasons)}"
                    (bw, bh), _ = cv2.getTextSize(banner,
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (5, 10), (15 + bw, 40), severity_color, -1)
                    cv2.putText(frame, banner,
                                (10, 33), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 2)

                    # Dwell time display on screen
                    if dwell_s > 0:
                        cv2.putText(frame, f"Dwell: {dwell_s:.1f}s",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, severity_color, 2)

                    # MEDIUM alert text message on screen
                    if cur_severity == sev.MEDIUM:
                        msg = "MEDIUM ALERT - Person detected in ROI"
                        cv2.putText(frame, msg,
                                    (10, frame.shape[0] - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 165, 255), 2)

                    # HIGH alert text message on screen
                    if cur_severity == sev.HIGH:
                        msg = "!! HIGH ALERT - INTRUDER DETECTED !!"
                        (mw, mh), _ = cv2.getTextSize(msg,
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        mx = (frame.shape[1] - mw) // 2
                        cv2.rectangle(frame, (mx - 10, frame.shape[0] - 65),
                                      (mx + mw + 10, frame.shape[0] - 30),
                                      (0, 0, 255), -1)
                        cv2.putText(frame, msg,
                                    (mx, frame.shape[0] - 38),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2)

                    # ── Track this as a confirmed intruder ─────────────────
                    confirmed_intruder_ids.add(incident_id)

                    # ── Privacy blurring for evidence ──────────────────────
                    evidence_frame = frame.copy()
                    evidence_frame = blur_non_intruders(
                        evidence_frame, detections, confirmed_intruder_ids
                    )

                    # ── Save evidence photo ────────────────────────────────
                    saved_path = evidence_mgr.save_photo(evidence_frame, incident_id)

                    # ── Check for severity escalation ──────────────────────
                    prev_severity = track_max_severity.get(incident_id, None)
                    escalated = (
                        prev_severity is not None
                        and sev.severity_escalated(prev_severity, cur_severity)
                    )
                    # Only store upward — never downgrade the max
                    if prev_severity is None or sev.severity_escalated(prev_severity, cur_severity):
                        track_max_severity[incident_id] = cur_severity

                    # ── Trigger clip on HIGH severity ───────────────────────
                    clip_path = None
                    if sev.should_save_clip(cur_severity):
                        clip_path = evidence_mgr.trigger_clip(incident_id)

                    # ── Log event to tamper-evident chain ──────────────────
                    log_record = evidence_mgr.log_event(
                        incident_id=incident_id,
                        severity=cur_severity,
                        reasons=reasons,
                        dwell_seconds=dwell_s,
                        behaviour=behaviour,
                        roi_name=roi['name'],
                        photo_path=saved_path,
                        clip_path=clip_path,
                        track_id=track_id,
                    )

                    # ── Log event to SQLite database ────────────────────────
                    event_db.insert_event(log_record)

                    # ── Alarm (severity-gated + cooldown) ──────────────────
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

                    # ── Email (severity-gated + per-track cooldown) ────────
                    if sev.should_send_email(cur_severity):
                        now = datetime.now()
                        last_time = last_email_time.get(incident_id)
                        seconds_since = (
                            (now - last_time).total_seconds()
                            if last_time is not None
                            else config.EMAIL_COOLDOWN_SECONDS + 1
                        )

                        # Send if cooldown expired OR severity escalated
                        if seconds_since >= config.EMAIL_COOLDOWN_SECONDS or escalated:
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
                                if escalated:
                                    print(f"[Main] Severity ESCALATED "
                                          f"{prev_severity}→{cur_severity} "
                                          f"for {incident_id}")

                    break  # one detection per ROI is enough

            # ── Mark absent entities ───────────────────────────────────────
            if config.USE_TEMPORAL_CONFIRMATION:
                confirmer.mark_absent_entities(active_entity_keys)
                if frame_count % 100 == 0:
                    confirmer.cleanup_stale()

            # ── Draw ROIs on frame ─────────────────────────────────────────
            if len(roi_mgr.rois) > 0:
                roi_mgr.draw_rois_on_frame(frame)

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
