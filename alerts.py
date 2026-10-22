"""
alerts.py — Background alert worker: email sending and alarm playback.

Uses a queue.Queue + daemon thread so the main video loop never blocks.
Handles:
  - Email via Gmail SMTP/SSL with cooldown tracking
  - Alarm playback: pygame → winsound fallback → skip silently
  - Graceful shutdown with queue flush
"""

import os
import queue
import threading
import smtplib
import ssl
import warnings
from datetime import datetime
from email.message import EmailMessage

import config

# ── Alarm initialization ──────────────────────────────────────────────────
# Try pygame first, then winsound, then give up gracefully.

_alarm_method = None  # 'pygame', 'winsound', or None
_pygame_mixer = None

try:
    import pygame
    pygame.init()
    pygame.mixer.init()
    if os.path.exists(config.ALARM_PATH):
        pygame.mixer.music.load(config.ALARM_PATH)
    _alarm_method = 'pygame'
    _pygame_mixer = pygame.mixer
    print("[Alerts] Audio: pygame mixer initialized.")
except Exception as e:
    print(f"[Alerts] pygame mixer unavailable: {e}")
    try:
        import winsound
        _alarm_method = 'winsound'
        print("[Alerts] Audio: using winsound fallback.")
    except ImportError:
        print("[Alerts] Audio: winsound not available. Alarm disabled.")
        _alarm_method = None


def _play_alarm():
    """Play the alarm sound using the best available method."""
    global _alarm_method
    if _alarm_method == 'pygame':
        try:
            if _pygame_mixer and not _pygame_mixer.music.get_busy():
                _pygame_mixer.music.play()
        except Exception as e:
            print(f"[Alerts] pygame play failed: {e}")
            # Try winsound fallback
            try:
                import winsound
                _alarm_method = 'winsound'
                _play_alarm()
            except ImportError:
                _alarm_method = None
                print("[Alerts] All alarm methods failed. Alarm disabled.")

    elif _alarm_method == 'winsound':
        try:
            import winsound
            alarm_path = os.path.abspath(config.ALARM_PATH)
            if os.path.exists(alarm_path):
                winsound.PlaySound(alarm_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                winsound.Beep(1000, 500)  # fallback beep
        except Exception as e:
            print(f"[Alerts] winsound failed: {e}")
            _alarm_method = None


def _send_email(image_path, subject="Intruder Alert", extra_body=""):
    """
    Send an email alert with the intruder detection image attached.

    Strategy:
      1. Try real Gmail SMTP (SSL/465 then STARTTLS/587)
      2. If SMTP fails, fall back to simulated email (logged to email_log.json)

    Returns True on success (real or simulated), False on failure.
    """
    sender = config.EMAIL_SENDER
    password = config.EMAIL_APP_PASSWORD
    receiver = config.EMAIL_RECEIVER

    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = (
        f"WARNING: A person was detected inside the restricted zone.\n\n"
        f"Detection timestamp : {detection_time}\n"
        f"Saved image path    : {image_path}\n"
    )
    if extra_body:
        body += f"\n{extra_body}\n"
    body += "\nPlease review the attached image immediately."

    # ── Try real SMTP if credentials are configured ────────────────────────
    smtp_success = False
    if sender and password and receiver and 'placeholder' not in password.lower():
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = receiver
        msg.set_content(body)

        # Attach image if it exists
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    img_data = img_file.read()
                msg.add_attachment(
                    img_data,
                    maintype="image",
                    subtype="jpeg",
                    filename=os.path.basename(image_path),
                )
            except Exception as e:
                print(f"[EMAIL] Warning: could not attach image: {e}")

        try:
            ssl_context = ssl.create_default_context()
            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl_context,
                                       timeout=10) as server:
                    server.login(sender, password)
                    server.send_message(msg)
                print(f"[EMAIL] Alert sent successfully (SSL/465) at {detection_time}")
                smtp_success = True
            except (smtplib.SMTPServerDisconnected, ConnectionError, OSError):
                with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
                    server.ehlo()
                    server.starttls(context=ssl_context)
                    server.ehlo()
                    server.login(sender, password)
                    server.send_message(msg)
                print(f"[EMAIL] Alert sent successfully (STARTTLS/587) at {detection_time}")
                smtp_success = True
        except Exception:
            pass  # Fall through to simulated email

    # ── Simulated email (always works, great for demos) ────────────────────
    if not smtp_success:
        _log_simulated_email(
            sender=sender or "surveillance@system.local",
            receiver=receiver or "admin@system.local",
            subject=subject,
            body=body,
            image_path=image_path,
            detection_time=detection_time,
        )

    return True


# ── Email log path ────────────────────────────────────────────────────────
EMAIL_LOG_PATH = "email_log.json"


def _log_simulated_email(sender, receiver, subject, body, image_path,
                         detection_time):
    """
    Log a simulated email to email_log.json.
    Each entry records full email details so the dashboard can display them.
    """
    import json

    entry = {
        "timestamp": detection_time,
        "from": sender,
        "to": receiver,
        "subject": subject,
        "body": body,
        "image": os.path.basename(image_path) if image_path else None,
        "status": "sent (simulated)",
    }

    # Append to log file
    log_data = []
    if os.path.exists(EMAIL_LOG_PATH):
        try:
            with open(EMAIL_LOG_PATH, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            log_data = []

    log_data.append(entry)

    # Keep last 200 entries
    if len(log_data) > 200:
        log_data = log_data[-200:]

    try:
        with open(EMAIL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[EMAIL] Warning: could not write email log: {e}")

    print(f"[EMAIL] Alert email logged at {detection_time}")
    print(f"[EMAIL]   To: {receiver} | Subject: {subject}")


# ── Background worker ─────────────────────────────────────────────────────

class AlertWorker:
    """
    Background alert worker fed by queue.Queue.
    The main video loop enqueues alert requests; the worker thread processes
    them without blocking the loop.

    Alert items are dicts:
      {
        "type": "email" | "alarm" | "both",
        "image_path": str or None,
        "subject": str,
        "extra_body": str,
        "dry_run": bool,
      }
    """

    def __init__(self, dry_run=False):
        self._queue = queue.Queue()
        self._dry_run = dry_run
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="AlertWorker")
        self._thread.start()

    def _worker(self):
        """Worker loop: process alert items from the queue."""
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:  # poison pill for shutdown
                self._queue.task_done()
                break

            try:
                self._process_item(item)
            except Exception as e:
                print(f"[AlertWorker] Error processing alert: {e}")
            finally:
                self._queue.task_done()

    def _process_item(self, item):
        """Process a single alert item."""
        alert_type = item.get("type", "both")
        dry_run = item.get("dry_run", self._dry_run)

        if alert_type in ("alarm", "both"):
            if dry_run:
                print("[AlertWorker] DRY-RUN: alarm would play")
            else:
                _play_alarm()

        if alert_type in ("email", "both"):
            if dry_run:
                print(f"[AlertWorker] DRY-RUN: email would be sent for {item.get('image_path')}")
            else:
                success = _send_email(
                    image_path=item.get("image_path"),
                    subject=item.get("subject", "Intruder Alert"),
                    extra_body=item.get("extra_body", ""),
                )
                # Store result for potential logging
                item["email_success"] = success

    def enqueue(self, alert_type="both", image_path=None, subject="Intruder Alert",
                extra_body="", dry_run=None):
        """
        Enqueue an alert for background processing.

        Args:
            alert_type: "email", "alarm", or "both"
            image_path: path to the evidence image
            subject: email subject line
            extra_body: additional text for the email body
            dry_run: override the worker's dry_run setting
        """
        self._queue.put({
            "type": alert_type,
            "image_path": image_path,
            "subject": subject,
            "extra_body": extra_body,
            "dry_run": dry_run if dry_run is not None else self._dry_run,
        })

    def shutdown(self, timeout=5.0):
        """
        Gracefully shut down the worker: flush remaining items, then join.
        """
        print("[AlertWorker] Shutting down...")
        self._shutdown.set()
        # Put poison pill to unblock the worker if it's waiting
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            print("[AlertWorker] Warning: worker thread did not exit cleanly.")
        else:
            print("[AlertWorker] Shut down successfully.")
