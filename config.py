"""
config.py — Central configuration for ROI-Based Intelligent Surveillance System.

All settings, feature flags, and thresholds are defined here as module-level
constants.  Every feature can be toggled on/off for ablation testing (Phase 6).
"""

import os
from dotenv import load_dotenv

# ── Load .env for credentials ──────────────────────────────────────────────
load_dotenv()

# ── Credentials (loaded from .env) ─────────────────────────────────────────
EMAIL_SENDER       = os.getenv("EMAIL_SENDER", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
EMAIL_RECEIVER     = os.getenv("EMAIL_RECEIVER", "")

# ── Email settings ─────────────────────────────────────────────────────────
EMAIL_COOLDOWN_SECONDS = 30          # minimum seconds between emails (per-track)

# ── Paths ──────────────────────────────────────────────────────────────────
ALARM_PATH          = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "Alarm", "alarm.wav")
DETECTED_PHOTOS_DIR = "detected_photos"
ROI_JSON_PATH       = "rois.json"
EVIDENCE_DIR        = "evidence"
CLIPS_DIR           = os.path.join(EVIDENCE_DIR, "clips")
EVIDENCE_LOG_PATH   = os.path.join(EVIDENCE_DIR, "evidence_log.jsonl")
DB_PATH             = "events.db"
DASHBOARD_DB_PATH   = "dashboard_all.db"   # unified DB the dashboard reads (all videos)
EMAIL_LOG_PATH      = "email_log.json"     # every email (sent, simulated or dry-run)

# ── Video source ───────────────────────────────────────────────────────────
DEFAULT_SOURCE      = "input vedio for testing.mp4"

# ── Detection settings ─────────────────────────────────────────────────────
TARGET_CLASSES      = ['car', 'bus', 'truck', 'person', 'motorcycle']
INTRUSION_CLASS     = 'person'
VEHICLE_CLASSES     = ['car', 'bus', 'truck', 'motorcycle']

# ── Feature flags (toggle for ablation) ────────────────────────────────────
USE_FOOT_POINT             = True     # Phase 2: bottom-center instead of box center
ROI_MIN_OVERLAP            = 0.3      # a person also counts as in the ROI when this
                                      #   fraction of their box lies inside it (0 = off)
USE_CONF_THRESHOLD         = True     # Phase 2: filter low-confidence detections
USE_TEMPORAL_CONFIRMATION  = True     # Phase 2: N-of-M confirmation
USE_TRACKING               = True     # Phase 3: ByteTrack persistent IDs
USE_POSE                   = True     # Phase 4: MediaPipe pose classification
ENABLE_VLM                 = False    # Phase 4: VLM false-positive check (disabled)
ALERT_ON_VEHICLES          = False    # Phase 3: trigger alerts for vehicles too
BLUR_NON_INTRUDERS         = False    # Phase 5: blur non-intruders in evidence

# ── Trusted person recognition ─────────────────────────────────────────
USE_TRUSTED_PERSONS        = True     # Enable face-based trusted person bypass
TRUSTED_FACES_DIR          = "trusted_faces"   # Dir with reference images
FACE_MATCH_THRESHOLD       = 0.35     # Histogram correlation threshold (0-1, higher=stricter)
CCTV_MATCH_THRESHOLD       = 0.6      # camera-snapshot match (same person ~0.8-0.99,
                                      #   other people measured <= 0.49)
TRUSTED_CLICK_NAME         = "trusted"  # name used for people you click to trust
TRUSTED_SNAPSHOTS_PER_RUN  = 5        # extra snapshots saved of a clicked person
TRUST_DECIDE_SECONDS       = 1.0      # a new person in the ROI is "IDENTIFYING..." (no
                                      #   alert) this long while checked against trusted
TRUSTED_PERSON_ID          = -1       # Track ID assigned to trusted persons

# ── Speed / performance options ────────────────────────────────────────────
DETECT_EVERY_N_FRAMES = 1            # run detector every N-th frame (1=every frame)
INFERENCE_SIZE        = 640           # YOLO input size (416 or 640)
CROP_TO_ROI_BOUNDS    = False         # crop frame to ROI bounding rect before YOLO

# ── Confidence threshold ───────────────────────────────────────────────────
CONF_THRESHOLD = 0.5                  # ignore detections below this
TRACK_LOW_CONF = 0.15                 # with tracking: weaker boxes (down to this) may
                                      #   keep an EXISTING track alive - e.g. a person
                                      #   half hidden by a railing - but never start one
TRACK_ACTIVATION_CONF = 0.4           # ByteTrack: a new track needs > this + 0.1
TRACK_LOST_BUFFER = 60                # frames a lost track is remembered (re-use its ID)

# ── Display hold-over for brief detector dropouts ──────────────────────────
FOLLOW_LOST_PERSONS  = False          # experimental: when YOLO misses a person in/near
                                      #   the ROI, move their box by following their image.
                                      #   Off: on real footage it can latch onto a moving
                                      #   door/gate after the person has gone behind it
FOLLOW_MAX_SECONDS   = 5.0            # ...for at most this long without a detection
FOLLOW_MIN_SIMILARITY = 0.3           # stop following when the box no longer looks
                                      #   like the person (0-1)
BOX_HOLD_SECONDS     = 0.5            # keep a lost person's last box this long, then
                                      #   remove it (avoids flicker, never "stuck")
BANNER_HOLD_SECONDS  = 2.0            # keep the top-left severity banner this long
CONFIRM_HOLD_SECONDS = 4.0            # a confirmed person stays confirmed through
                                      #   brief detection gaps
SEVERITY_LATCH       = False          # True: an incident's severity never drops while
                                      #   the person stays in the ROI (video 1)
REID_SECONDS         = 4.0            # a new track appearing where a person was lost
                                      #   this recently is treated as the same person

# ── Zoomed second detection pass over the ROI ─────────────────────────────
ROI_ZOOM               = True         # re-run YOLO on the enlarged ROI area
ROI_ZOOM_ACTIVE_SECONDS = 3.0         # keep zooming this long after a person was near
ROI_ZOOM_PROBE_EVERY   = 5            # otherwise zoom every N-th frame to catch
                                      #   people too small for the full-frame pass

# ── Temporal confirmation (N-of-M) ────────────────────────────────────────
TEMPORAL_N = 8                        # must be detected in N of last M frames
TEMPORAL_M = 10

# ── Evidence / photos ─────────────────────────────────────────────────────
MAX_PHOTOS_PER_INCIDENT = 3           # per-incident photo limit

# ── Dwell / loitering ─────────────────────────────────────────────────────
DWELL_GRACE_SECONDS = 2.0             # grace period for brief tracking dropouts

# ── Time-based restrictions ────────────────────────────────────────────────
SIMULATED_TIME = None                 # e.g. "22:00" to simulate a time of day

# ── Severity thresholds (seconds) ─────────────────────────────────────────
LOW_TO_MEDIUM_SECONDS  = 10           # dwell threshold for LOW → MEDIUM
MEDIUM_TO_HIGH_SECONDS = 30           # dwell threshold for MEDIUM → HIGH
HIGH_PERSON_COUNT      = 2            # number of people in ROI for auto-HIGH

# ── Pre/post clip buffer ──────────────────────────────────────────────────
CLIP_PRE_SECONDS  = 5                 # seconds of frames to keep before event
CLIP_POST_SECONDS = 5                 # seconds of frames to record after event

# ── Pose behavior thresholds ──────────────────────────────────────────────
POSE_EVERY_N_FRAMES    = 3            # run pose every N-th frame (CPU saving)
CROUCH_HIP_RATIO       = 0.6         # hip-to-ankle / torso ratio for "crouching"
LYING_ASPECT_RATIO     = 1.2         # bbox width/height ratio for "lying/fallen"
LYING_TORSO_ANGLE      = 45          # torso angle from vertical for "lying"

# Sitting vs standing (all vertical distances are measured in "torso lengths",
# i.e. divided by shoulder-to-hip distance, so they work at any distance from camera)
# Box-based feet measurement (more reliable than MediaPipe's ankle landmarks on small people):
#   F = (bottom of person's box - hip_y) / torso length
#   measured on this camera: seated man 1.43-1.59 (15 readings),
#                            standing man 1.46-2.24 (mostly 1.64-1.74)
USE_BOX_FEET           = True
FEET_STAND_RATIO       = 1.62        # F >= this -> standing
FEET_CROUCH_RATIO      = 0.9         # F <  this -> crouching, in between -> sitting

# Motion cue: a person whose feet travel across the frame is standing/walking,
# whatever the pose model says (a seated person's feet stay in one place).
USE_MOTION_CUE         = True
MOVE_WINDOW_SECONDS    = 2.0         # look at the last N seconds of video
MOVE_MIN_PIXELS        = 12          # net foot-point movement that counts as "moving"
RUNNING_MIN_PIXELS     = 15          # (legacy, no longer used - see RUN_SPEED_* below)

# Running = speed of the foot point, measured in body heights per second
# (bbox heights / s), so it works at any distance from the camera.
#   walking across the frame ~0.4-0.9, running ~1.5+
#   climbing stairs normally ~0.2-0.4, running up stairs ~0.6+
RUN_SPEED_WINDOW_SECONDS = 0.6       # speed is measured over this much video
RUN_SPEED_BH_PER_S       = 1.2       # >= this -> 'running'  (overridden per video)
RUNNING_HOLD_SECONDS     = 1.5       # keep the 'running' label this long after the burst
BOX_JUMP_RATIO           = 0.25      # box height changing more than this between frames
                                     #   = partly hidden person, not movement
STAND_ANKLE_RATIO      = 1.4         # (ankle_y - hip_y) / torso >= this -> standing.
                                     #   standing ~1.8-2.2, sitting on chair ~0.8-1.2
SIT_KNEE_RATIO         = 0.55        # (knee_y - hip_y) / torso < this -> thigh ~horizontal -> sitting
SIT_BODY_RATIO         = 3.0         # fallback when legs not visible: bbox_height / torso < this -> sitting
SIT_ASPECT_RATIO       = 0.75        # fallback when legs not visible: bbox w/h above this -> sitting
LEG_VISIBILITY_MIN     = 0.5         # knees/ankles below this visibility are ignored
                                     #   (MediaPipe "guesses" occluded legs, e.g. behind a desk)
POSE_SMOOTH_FRAMES     = 5           # majority vote over last N pose results per person
DEBUG_POSE             = True        # show the pose numbers under each label (set False when done tuning)

# ── Per-video behaviour flags (set dynamically by phy.py) ──────────────
WALKING_IS_LOW         = False       # when True, walking persons stay LOW (for video 3)
PASSERBY_IS_LOW        = False       # when True, a person walking past (moving) is always
                                     #   LOW / log-only: pose and head-count escalation are
                                     #   ignored; only a long stationary dwell escalates (video 3)
PASSERBY_MEDIUM_SECONDS = None       # passer-by mode: someone walking in the ROI this
                                     #   long (s) becomes MEDIUM (5 for video 3); None = never
RUNNING_RAISES_ALARM   = False       # 'running' -> HIGH + alarm (turned on for video 1:
                                     #   running up the stairs)

# ── Demo mode ─────────────────────────────────────────────────────────────
DEMO_MODE           = False           # clean overlay for presentation recording
DEMO_OUTPUT_PATH    = "demo_output.mp4"

# ── Hash-chain genesis ────────────────────────────────────────────────────
GENESIS_HASH = "0" * 64              # 64-char zero hash for first record
