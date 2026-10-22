"""Quick test to find where phy.py crashes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("Step 1: Importing config...")
import config
print("  OK")

print("Step 2: Importing alerts (pygame)...")
try:
    from alerts import AlertWorker
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")

print("Step 3: Importing detector...")
from detector import Detector
print("  OK")

print("Step 4: Creating Detector (loads YOLO)...")
try:
    det = Detector()
    print("  OK - Model loaded!")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print("Step 5: Importing tracker...")
from tracker import Tracker
print("  OK")

print("Step 6: Creating Tracker...")
try:
    t = Tracker()
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")

print("Step 7: Importing pose_estimator...")
from pose_estimator import PoseEstimator
print("  OK")

print("Step 8: Creating PoseEstimator...")
try:
    p = PoseEstimator()
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")

print("Step 9: Opening video 1.mp4...")
import cv2
cap = cv2.VideoCapture("video 1.mp4")
ok, frame = cap.read()
print(f"  OK - frame read: {ok}, shape: {frame.shape if ok else 'N/A'}")

print("Step 10: Running detection on 1 frame...")
try:
    dets = det.detect(frame)
    print(f"  OK - {len(dets)} detections")
    for d in dets[:5]:
        print(f"    {d['name']} conf={d['confidence']:.2f}")
except Exception as e:
    print(f"  FAILED: {e}")

cap.release()
print("\nALL STEPS PASSED!")
