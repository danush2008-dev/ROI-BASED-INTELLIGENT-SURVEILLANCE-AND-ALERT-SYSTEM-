"""
test_video_rules.py — Per-video severity rules, running detection, and
dashboard data persistence/sync.
"""

import sys
import os
import json
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import config
import severity as sev
from pose_estimator import update_motion, foot_speed


@pytest.fixture
def video_flags(monkeypatch):
    """Reset the per-video flags phy.py sets, so tests don't leak state."""
    monkeypatch.setattr(config, "WALKING_IS_LOW", False)
    monkeypatch.setattr(config, "PASSERBY_IS_LOW", False)
    monkeypatch.setattr(config, "RUNNING_RAISES_ALARM", True)
    return monkeypatch


# ── Video 3: person walking past is LOW / log only ────────────────────────

class TestPasserby:

    def _video3(self, mp):
        mp.setattr(config, "WALKING_IS_LOW", True)
        mp.setattr(config, "PASSERBY_IS_LOW", True)
        mp.setattr(config, "RUNNING_RAISES_ALARM", False)

    @pytest.mark.parametrize("behaviour", [
        "walking", "running", "crouching", "standing", None])
    def test_moving_person_is_low(self, video_flags, behaviour):
        self._video3(video_flags)
        level, reasons = sev.compute_severity(
            2.0, 2, behavior=behaviour, is_moving=True)
        assert level == sev.LOW
        assert not sev.should_play_alarm(level)
        assert not sev.should_send_email(level)

    def test_misread_pose_mid_stride_is_low(self, video_flags):
        # Before the motion cue kicks in, a walker may read as "crouching"
        self._video3(video_flags)
        level, _ = sev.compute_severity(1.0, 1, behavior="crouching")
        assert level == sev.LOW

    def test_long_stationary_dwell_still_escalates(self, video_flags):
        self._video3(video_flags)
        level, _ = sev.compute_severity(35.0, 1, behavior="standing")
        assert level == sev.HIGH


# ── Video 1: running up the stairs is HIGH + alarm ────────────────────────

class TestRunning:

    def test_running_is_high_with_alarm(self, video_flags):
        level, reasons = sev.compute_severity(1.0, 1, behavior="running")
        assert level == sev.HIGH
        assert "pose:running" in reasons
        assert sev.should_play_alarm(level)

    def test_running_without_alarm_flag_is_not_high(self, video_flags):
        video_flags.setattr(config, "RUNNING_RAISES_ALARM", False)
        level, _ = sev.compute_severity(1.0, 1, behavior="running")
        assert level == sev.MEDIUM

    def _speed(self, px_per_s, body_h, seconds=1.0, fps=25):
        h = deque()
        s = 0.0
        for i in range(int(seconds * fps)):
            t = i / fps
            update_motion(h, t, 300, 400 - px_per_s * t)
            s = foot_speed(h, t, body_h)
        return s

    def test_speed_is_in_body_heights(self):
        assert self._speed(100, 100) == pytest.approx(1.0, rel=0.05)
        # same pixel speed for a person twice as close -> half the speed
        assert self._speed(100, 200) == pytest.approx(0.5, rel=0.05)

    def test_fast_climb_vs_normal_climb(self):
        thr = 0.6      # video 1 threshold set in phy.py
        assert self._speed(40, 50) >= thr     # 0.8 body heights / s
        assert self._speed(15, 50) < thr      # 0.3 body heights / s

    def test_speed_needs_history(self):
        h = deque()
        update_motion(h, 0.0, 100, 100)
        update_motion(h, 0.1, 300, 100)
        assert foot_speed(h, 0.1, 100) == 0.0

    def test_jitter_is_not_running(self):
        h = deque()
        s = 0.0
        for i in range(50):
            t = i / 25
            update_motion(h, t, 300 + (i % 3) * 3 - 3, 400 + (i % 2) * 3)
            s = foot_speed(h, t, 80)
        assert s < 0.3


# ── Dashboard DB: incremental, never deletes previous runs ────────────────

def _write(path, records, mode="a"):
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _rec(i, sev_level="LOW"):
    return {"timestamp": f"2026-01-01T00:00:{i:02d}", "incident_id": str(i),
            "severity": sev_level, "reasons": ["x"], "dwell_seconds": 1.0}


class TestSync:

    def test_incremental_sync_keeps_old_runs(self, tmp_path):
        from event_db import EventDB
        db = EventDB(db_path=str(tmp_path / "d.db"))
        log = str(tmp_path / "log.jsonl")

        _write(log, [_rec(1), _rec(2)])
        assert db.sync_from_jsonl(log, "video_1") == 2
        assert db.sync_from_jsonl(log, "video_1") == 0      # nothing new

        _write(log, [_rec(3, "HIGH")])                      # second run appends
        assert db.sync_from_jsonl(log, "video_1") == 1
        assert db.get_event_count("video_1") == 3

    def test_partial_line_waits_for_next_sync(self, tmp_path):
        from event_db import EventDB
        db = EventDB(db_path=str(tmp_path / "d.db"))
        log = str(tmp_path / "log.jsonl")
        _write(log, [_rec(1)])
        with open(log, "a", encoding="utf-8") as f:
            f.write('{"timestamp": "2026')                  # being written
        assert db.sync_from_jsonl(log, "v") == 1
        with open(log, "a", encoding="utf-8") as f:
            f.write('-01-01T00:00:09", "severity": "LOW"}\n')
        assert db.sync_from_jsonl(log, "v") == 1
        assert db.get_event_count("v") == 2

    def test_first_sync_replaces_old_full_import(self, tmp_path):
        from event_db import EventDB
        db = EventDB(db_path=str(tmp_path / "d.db"))
        log = str(tmp_path / "log.jsonl")
        _write(log, [_rec(1), _rec(2)])
        db.import_from_jsonl(log, run_label="v")            # old dashboard import
        db.sync_from_jsonl(log, "v")
        assert db.get_event_count("v") == 2                 # no duplicates

    def test_batch_insert(self, tmp_path):
        from event_db import EventDB
        db = EventDB(db_path=str(tmp_path / "d.db"))
        db.insert_events([_rec(i) for i in range(5)])
        assert db.get_event_count() == 5


# ── Emails: dry-run emails are logged, and nothing is dropped ─────────────

class TestEmailLog:

    def test_dry_run_email_is_logged_and_kept(self, tmp_path, monkeypatch):
        import alerts
        log = str(tmp_path / "email_log.json")
        monkeypatch.setattr(alerts, "EMAIL_LOG_PATH", log)
        with open(log, "w", encoding="utf-8") as f:
            json.dump([{"subject": f"old {i}"} for i in range(250)], f)

        worker = alerts.AlertWorker(dry_run=True, source_label="video_1")
        worker.enqueue(alert_type="email", image_path="x.jpg",
                       subject="Intruder Alert [HIGH]")
        worker.shutdown()

        with open(log, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 251
        assert data[-1]["source"] == "video_1"
        assert data[-1]["status"] == "dry-run (not sent)"
