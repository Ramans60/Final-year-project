"""
NILA - DEMO REALTIME - continuous RTSP capture (no per-frame ffmpeg spawn)

Pipeline:
  RTSP -> (one long-running ffmpeg) -> JPEG frames on stdout pipe
  Python capture thread parses JPEG boundaries and keeps only the latest frame bytes
  Main loop writes latest frame to FRAME_PATH and runs RKNN YOLO detection binary
  Detections -> GuidanceBrain -> GPIO haptics + audio alerts
"""

from __future__ import annotations

import os
import random
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

# Optional debug visualization (does not affect detection pipeline).
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEMO_DIR = "/root/rknn_yolov8_demo"
DEMO_BIN = "/root/rknn_yolov8_demo/rknn_yolov8_demo"
MODEL_PATH = "/root/rknn_yolov8_demo/model/yolov8n_6out.rknn"
VOICES_DIR = "/root/voices"
GPIO_PIN = 32

RTSP_URL = "rtsp://172.32.0.93:554/live/0"
INPUT_SIZE = 640

# Confidence threshold after normalization (0..1).
CONF_THRESHOLD = 0.5

FRAME_PATH = "/tmp/frame.jpg"
FRAME_TMP_PATH = "/tmp/frame.tmp.jpg"
JPEG_QUALITY = 2
FFMPEG_LOGLEVEL = "error"

DEMO_ENV = {**os.environ, "LD_LIBRARY_PATH": "/root/rknn_yolov8_demo/lib"}

# Presentation stability knobs.
TARGET_HZ = 6.0  # cap main loop so audio/haptics and CPU don't thrash
FRAME_STALE_SEC = 2.0  # if we haven't seen a frame recently, skip inference
STALE_PRINT_EVERY_SEC = 1.0  # don't spam stall warnings

# Debug view (OpenCV) - low-res display for performance.
DEBUG_VIEW = True
DEBUG_WINDOW_NAME = "NILA Debug"
DEBUG_DISPLAY_SIZE = 320  # display as 320x320 regardless of INPUT_SIZE
DEBUG_DRAW_ALL_DETECTIONS = True

PRIORITY: Dict[str, int] = {
    "truck": 5,
    "bus": 5,
    "train": 5,
    "car": 4,
    "motorcycle": 4,
    "bicycle": 3,
    "traffic light": 3,
    "stop sign": 3,
    "person": 2,
    "bench": 1,
    "chair": 1,
}

VOICE_MAP: Dict[Tuple[str, str], List[str]] = {
    ("person", "CENTER"): ["person_center_1", "person_center_2", "person_center_3"],
    ("person", "LEFT"): ["person_left_1", "person_left_2"],
    ("person", "RIGHT"): ["person_right_1", "person_right_2"],
    ("car", "CENTER"): ["car_center_1", "car_center_2", "car_center_3"],
    ("car", "LEFT"): ["car_left_1", "car_left_2"],
    ("car", "RIGHT"): ["car_right_1", "car_right_2"],
    ("motorcycle", "CENTER"): ["bike_center_1", "bike_center_2"],
    ("motorcycle", "LEFT"): ["bike_left_1"],
    ("motorcycle", "RIGHT"): ["bike_right_1"],
    ("bicycle", "CENTER"): ["bike_center_1", "bike_center_2"],
    ("bicycle", "LEFT"): ["bike_left_1"],
    ("bicycle", "RIGHT"): ["bike_right_1"],
    ("bus", "CENTER"): ["bus_center_1", "bus_center_2"],
    ("bus", "LEFT"): ["bus_left_1"],
    ("bus", "RIGHT"): ["bus_right_1"],
    ("truck", "CENTER"): ["truck_center_1", "truck_center_2"],
    ("truck", "LEFT"): ["truck_left_1"],
    ("truck", "RIGHT"): ["truck_right_1"],
    ("traffic light", "CENTER"): ["traffic_light"],
    ("traffic light", "LEFT"): ["traffic_light"],
    ("traffic light", "RIGHT"): ["traffic_light"],
    ("stop sign", "CENTER"): ["stop_sign"],
    ("stop sign", "LEFT"): ["stop_sign"],
    ("stop sign", "RIGHT"): ["stop_sign"],
}


def resolve_leo_voice_key(name: str, voices_dir: str) -> str:
    """
    Prefer Leo-named direction prompts if present.
    Example mapping: person_center_1 -> leo_center_1
    Falls back to the original `name` to keep the demo working.
    """
    try:
        parts = name.split("_")
        for zone in ("left", "center", "right"):
            if zone in parts:
                idx = parts.index(zone)
                suffix = "_".join(parts[idx + 1 :]).strip()
                candidate = f"leo_{zone}_{suffix}" if suffix else f"leo_{zone}"
                candidate_path = os.path.join(voices_dir, f"{candidate}.wav")
                if os.path.exists(candidate_path):
                    return candidate
                return name
    except Exception:
        return name
    return name


def _zone_from_cx(cx: float) -> str:
    if cx < 0.4 * INPUT_SIZE:
        return "LEFT"
    elif cx > 0.6 * INPUT_SIZE:
        return "RIGHT"
    else:
        return "CENTER"


def _draw_debug_view(frame_jpeg: bytes, detections: List[Dict]) -> None:
    """
    Show a low-res debug view using OpenCV:
    - camera frame
    - LEFT/CENTER/RIGHT vertical guides
    - bounding boxes + label + confidence
    """
    if not DEBUG_VIEW or cv2 is None or np is None:
        return

    try:
        arr = np.frombuffer(frame_jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return

        # Resize for display performance.
        disp = cv2.resize(img, (DEBUG_DISPLAY_SIZE, DEBUG_DISPLAY_SIZE), interpolation=cv2.INTER_AREA)

        sx = DEBUG_DISPLAY_SIZE / float(INPUT_SIZE)
        sy = DEBUG_DISPLAY_SIZE / float(INPUT_SIZE)

        # Draw zone guides (thirds).
        x1 = int(DEBUG_DISPLAY_SIZE / 3)
        x2 = int(2 * DEBUG_DISPLAY_SIZE / 3)
        cv2.line(disp, (x1, 0), (x1, DEBUG_DISPLAY_SIZE), (255, 255, 255), 1)
        cv2.line(disp, (x2, 0), (x2, DEBUG_DISPLAY_SIZE), (255, 255, 255), 1)
        cv2.putText(disp, "LEFT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(disp, "CENTER", (x1 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(disp, "RIGHT", (x2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Optionally only draw relevant detections (by priority).
        to_draw = detections if DEBUG_DRAW_ALL_DETECTIONS else [d for d in detections if PRIORITY.get(d.get("label", ""), 0) > 0]

        # Draw boxes + label/confidence.
        for d in to_draw:
            try:
                x_min = int(d["xmin"] * sx)
                y_min = int(d["ymin"] * sy)
                x_max = int(d["xmax"] * sx)
                y_max = int(d["ymax"] * sy)
                label = str(d.get("label", "obj"))
                conf = float(d.get("confidence", 0.0))

                zone = _zone_from_cx(float(d.get("cx", 0.0)))
                color = (0, 255, 0) if zone == "CENTER" else (0, 165, 255) if zone == "LEFT" else (255, 0, 0)

                cv2.rectangle(disp, (x_min, y_min), (x_max, y_max), color, 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(disp, text, (x_min, max(20, y_min - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            except Exception:
                continue

        cv2.imshow(DEBUG_WINDOW_NAME, disp)
        cv2.waitKey(1)
    except Exception:
        # If running headless or OpenCV errors, disable silently.
        return


# ─────────────────────────────────────────────────────────────
# HARDWARE
# ─────────────────────────────────────────────────────────────
class GPIO:
    def __init__(self, pin: int):
        self.pin = pin
        try:
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(pin))
        except Exception:
            pass

        try:
            with open(f"/sys/class/gpio/gpio{pin}/direction", "w") as f:
                f.write("out")
            self.write(0)
        except Exception:
            pass

    def write(self, val: int) -> None:
        try:
            with open(f"/sys/class/gpio/gpio{self.pin}/value", "w") as f:
                f.write(str(val))
        except Exception:
            pass

    def pulse(self, times: int = 1, on_ms: int = 200, off_ms: int = 150) -> None:
        for i in range(times):
            self.write(1)
            time.sleep(on_ms / 1000)
            self.write(0)
            if i < times - 1:
                time.sleep(off_ms / 1000)

    def continuous(self, duration_ms: int = 600) -> None:
        self.write(1)
        time.sleep(duration_ms / 1000)
        self.write(0)


class AudioPlayer:
    def __init__(self, voices_dir: str):
        self.voices_dir = voices_dir
        self._proc: Optional[subprocess.Popen] = None

    def play(self, name: str, block: bool = False) -> None:
        name = resolve_leo_voice_key(name, self.voices_dir)
        path = os.path.join(self.voices_dir, f"{name}.wav")
        if not os.path.exists(path):
            print(f"[Audio] Missing: {name}.wav")
            return

        # Prevent overlapping audio: if something is already playing, ignore this request.
        if self._proc and self._proc.poll() is None:
            return

        if block:
            subprocess.run(
                ["aplay", "-q", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        def _run() -> None:
            self._proc = subprocess.Popen(
                ["aplay", "-q", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                self._proc.wait()
            finally:
                self._proc = None

        threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────
def detect(jpg_path: str) -> List[Dict]:
    """
    Run rknn_yolov8_demo binary on jpg and parse detections from stdout.

    Expected stdout line format (example):
      "person @ (211 241 283 506) 1.987"
    """
    try:
        result = subprocess.run(
            [DEMO_BIN, MODEL_PATH, jpg_path],
            cwd=DEMO_DIR,
            env=DEMO_ENV,
            capture_output=True,
            text=True,
            timeout=10,
        )

        output = result.stdout
        detections: List[Dict] = []
        import math
        for line in output.splitlines():
            if " @ (" not in line or ")" not in line:
                continue

            try:
                parts = line.strip().split(" @ (")
                label = parts[0].strip()
                rest = parts[1].split(")")
                coords = rest[0].split()
                conf_raw = float(rest[1].strip())

                # Normalize confidence to 1..1.
                # Some builds print probabilities already; others print logits (>1).
                if 0.0 <= conf_raw <= 1.0:
                    conf = conf_raw
                else:
                    # Sigmoid for logits; clamp extremes for stability.
                    if conf_raw >= 50:
                        conf = 1.0
                    elif conf_raw <= -50:
                        conf = 0.0
                    else:
                        conf = 1.0 / (1.0 + math.exp(-conf_raw))

                # Filter low-confidence predictions (after normalization).
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                cx = (x1 + x2) / 2.0

                detections.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "cx": cx,
                        "xmin": x1,
                        "xmax": x2,
                        "ymin": y1,
                        "ymax": y2,
                    }
                )
            except Exception:
                continue

        return detections
    except subprocess.TimeoutExpired:
        print("[Detect] Timeout!")
        return []
    except Exception as e:
        print(f"[Detect] Error: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# GUIDANCE BRAIN
# ─────────────────────────────────────────────────────────────
class GuidanceBrain:
    def __init__(self, audio: AudioPlayer, gpio: GPIO):
        self.audio = audio
        self.gpio = gpio
        self.last_label: Optional[str] = None
        self.last_zone: Optional[str] = None
        self.last_time = 0.0
        self.repeat_count = 0
        # 2-second cooldown between alert events (audio + haptics).
        self.cooldown = 2.0
        self.last_alert_time = 0.0
        # Stability: require 3 consecutive frames of same label to confirm.
        self._pending_label: Optional[str] = None
        self._pending_count = 0
        self._pending_zones: List[str] = []

    def get_zone(self, cx: float) -> str:
        if cx < INPUT_SIZE / 3:
            return "LEFT"
        if cx > 2 * INPUT_SIZE / 3:
            return "RIGHT"
        return "CENTER"

    def haptic(self, zone: str, priority: int) -> None:
        if zone == "CENTER":
            self.gpio.continuous(800 if priority >= 4 else 500)
        elif zone == "LEFT":
            self.gpio.pulse(1, on_ms=250)
        elif zone == "RIGHT":
            self.gpio.pulse(2, on_ms=200, off_ms=150)

    def process(self, detections: List[Dict]) -> None:
        now = time.time()

        relevant = [d for d in detections if PRIORITY.get(d["label"], 0) > 0]
        if not relevant:
            if self.last_label and (now - self.last_time) > 4.0:
                self.audio.play("now_clear")
                self.last_label = None
                self.last_zone = None
                self.repeat_count = 0
            return

        # Avoid alert spam: once we emitted an alert, wait at least 2s.
        if (now - self.last_alert_time) < self.cooldown:
            return

        relevant.sort(
            key=lambda d: (
                -PRIORITY.get(d["label"], 0),
                0 if self.get_zone(d["cx"]) == "CENTER" else abs(d["cx"] - INPUT_SIZE / 2),
            )
        )

        labels = [d["label"] for d in relevant]
        zones = [self.get_zone(d["cx"]) for d in relevant]

        # Many obstacles
        if len(relevant) >= 3:
            if (now - self.last_time) > self.cooldown:
                self.audio.play("multi_danger")
                self.gpio.continuous(800)
                self.last_time = now
                self.last_alert_time = now
                self.last_label = "multi"
            return

        # Car + person combo
        if "car" in labels and "person" in labels and "LEFT" in zones and "RIGHT" in zones:
            if (now - self.last_time) > self.cooldown:
                self.audio.play("multi_person_car")
                self.gpio.continuous(600)
                self.last_time = now
                self.last_alert_time = now
                self.last_label = "multi"
            return

        top = relevant[0]
        label = top["label"]
        zone = self.get_zone(top["cx"])
        priority = PRIORITY.get(label, 0)
        if priority == 0:
            return

        # Confirm only if same label appears in 3 consecutive frames.
        if label == self._pending_label:
            self._pending_count += 1
            self._pending_zones.append(zone)
            self._pending_zones = self._pending_zones[-3:]
        else:
            self._pending_label = label
            self._pending_count = 1
            self._pending_zones = [zone]

        if self._pending_count < 3:
            return

        # Simple smoothing: majority vote over last 3 zones for this label.
        counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
        for z in self._pending_zones:
            if z in counts:
                counts[z] += 1
        zone = max(counts, key=counts.get)

        same = label == self.last_label and zone == self.last_zone
        if same:
            if (now - self.last_time) < self.cooldown:
                return
            self.repeat_count += 1
            voice = "still_there_1" if self.repeat_count == 2 else "still_there_2"
            if self.repeat_count > 2:
                self.repeat_count = 2
            self.audio.play(voice)
            self.last_time = now
            self.last_alert_time = now
            threading.Thread(target=self.haptic, args=(zone, priority), daemon=True).start()
            return

        self.repeat_count = 0
        self.last_label = label
        self.last_zone = zone
        self.last_time = now

        options = VOICE_MAP.get((label, zone))
        voice = random.choice(options) if options else ("caution" if zone == "CENTER" else "slow_down")

        print(f"[Guide] {label} | {zone} | conf={top['confidence']:.2f} | -> {voice}")
        threading.Thread(target=self.haptic, args=(zone, priority), daemon=True).start()
        self.audio.play(voice)
        self.last_alert_time = now


# ─────────────────────────────────────────────────────────────
# CONTINUOUS CAPTURE (single ffmpeg process)
# ─────────────────────────────────────────────────────────────
class LatestFrame:
    def __init__(self) -> None:
        self.ts = 0.0
        self.jpeg: Optional[bytes] = None


def _start_ffmpeg_mjpeg_pipe() -> subprocess.Popen:
    """
    Start ffmpeg once and output an MJPEG stream to stdout.

    We parse individual JPEG frames by SOI/EOI markers:
      SOI: 0xFFD8
      EOI: 0xFFD9
    """
    # Important:
    # - Our Python parser extracts JPEGs by SOI/EOI markers, so ffmpeg must output
    #   a continuous concatenation of JPEG frames to stdout.
    # - This command avoids ffmpeg options that are not available in some embedded builds
    #   (e.g. "-reconnect") and uses a scale+format filter chain that is supported.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        FFMPEG_LOGLEVEL,
        "-rtsp_transport",
        "tcp",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
        "-max_delay",
        "0",
        "-reorder_queue_size",
        "0",
        "-i",
        RTSP_URL,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        # Keep filter chain simple + widely supported by embedded ffmpeg builds.
        f"scale={INPUT_SIZE}:{INPUT_SIZE}:flags=bilinear,format=yuv420p",
        "-vsync",
        "drop",
        "-c:v",
        "mjpeg",
        "-q:v",
        str(JPEG_QUALITY),
        "-f",
        "image2pipe",
        # Write to stdout so the Python process can read the JPEG bytestream.
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


def _capture_thread_fn(latest: LatestFrame, lock: threading.Lock, stop_event: threading.Event) -> None:
    """
    Continuously parse JPEG frames from ffmpeg stdout and store only the latest.
    """
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"

    while not stop_event.is_set():
        proc: Optional[subprocess.Popen] = None
        try:
            proc = _start_ffmpeg_mjpeg_pipe()
            buf = bytearray()

            stdout = proc.stdout
            if stdout is None:
                raise RuntimeError("ffmpeg stdout is None")

            while not stop_event.is_set():
                chunk = stdout.read(4096)
                if not chunk:
                    break

                buf.extend(chunk)

                # Extract all complete JPEGs currently in buffer.
                while True:
                    start = buf.find(SOI)
                    if start == -1:
                        # Keep buffer bounded.
                        if len(buf) > 1_000_000:
                            buf = buf[-200_000:]
                        break

                    end = buf.find(EOI, start + 2)
                    if end == -1:
                        # Need more data.
                        # Keep only tail after start to avoid unbounded growth.
                        if start > 0:
                            buf = buf[start:]
                        break

                    # end is index of 0xFFD9 first byte; include 2 bytes.
                    frame_bytes = bytes(buf[start : end + 2])
                    del buf[: end + 2]

                    with lock:
                        latest.jpeg = frame_bytes
                        latest.ts = time.time()
        except Exception as e:
            print(f"[Capture] ffmpeg/capture error: {e}")
            time.sleep(1.0)
        finally:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=1)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main() -> None:
    # If your system requires it, rkipc should be running for RKNN.
    try:
        subprocess.Popen(["rkipc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(1.0)

    print("=" * 50)
    print("  NILA - DEMO REALTIME - continuous capture")
    print("=" * 50)

    gpio = GPIO(GPIO_PIN)
    audio = AudioPlayer(VOICES_DIR)

    gpio.pulse(2, on_ms=150, off_ms=100)
    audio.play("startup", block=True)
    audio.play("offline_mode")

    brain = GuidanceBrain(audio, gpio)

    latest = LatestFrame()
    latest_lock = threading.Lock()
    stop_event = threading.Event()

    cap_thread = threading.Thread(
        target=_capture_thread_fn,
        args=(latest, latest_lock, stop_event),
        daemon=True,
    )
    cap_thread.start()

    fc = 0
    last_used_ts = 0.0
    last_stale_print = 0.0
    min_loop_dt = 1.0 / TARGET_HZ if TARGET_HZ > 0 else 0.0

    try:
        while True:
            loop_start = time.time()
            fc += 1

            # Grab latest frame bytes (latest-only, no backlog).
            frame_bytes: Optional[bytes] = None
            frame_ts = 0.0
            with latest_lock:
                if latest.jpeg is not None and latest.ts != last_used_ts:
                    frame_bytes = latest.jpeg
                    frame_ts = latest.ts

            if frame_bytes is None:
                # No new frame yet.
                if (time.time() - last_stale_print) > STALE_PRINT_EVERY_SEC:
                    print("[Camera] Waiting for frame...")
                    last_stale_print = time.time()
                time.sleep(0.05)
                continue

            last_used_ts = frame_ts
            if (time.time() - frame_ts) > FRAME_STALE_SEC:
                # Avoid running inference on stale frames (helps presentation stability).
                now = time.time()
                if (now - last_stale_print) > STALE_PRINT_EVERY_SEC:
                    print("[Camera] Frame stale - skipping inference")
                    last_stale_print = now
                time.sleep(0.05)
                continue

            # Write latest frame to disk for the existing RKNN demo binary.
            try:
                with open(FRAME_TMP_PATH, "wb") as f:
                    f.write(frame_bytes)
                os.replace(FRAME_TMP_PATH, FRAME_PATH)
            except Exception as e:
                print(f"[Main] Failed writing {FRAME_PATH}: {e}")
                continue

            t0 = time.time()
            detections = detect(FRAME_PATH)
            elapsed = time.time() - t0

            # Debug view uses the same captured bytes; does not affect inference pipeline.
            _draw_debug_view(frame_bytes, detections)

            relevant = [d for d in detections if PRIORITY.get(d["label"], 0) > 0]
            # Keep output stable and readable for demos: print only top relevant object.
            if relevant:
                # brain sorts by priority internally, but here we want a stable top print.
                relevant.sort(
                    key=lambda d: (
                        -PRIORITY.get(d["label"], 0),
                        0 if brain.get_zone(d["cx"]) == "CENTER" else abs(d["cx"] - INPUT_SIZE / 2),
                    )
                )
                top = relevant[0]
                zone = brain.get_zone(top["cx"])
                print(
                    f"[Frame {fc}] DETECTED: {top['label']} | {zone} | conf={top['confidence']:.2f} | infer={elapsed:.2f}s"
                )
            else:
                print(f"[Frame {fc}] Clear | infer={elapsed:.2f}s")

            try:
                brain.process(detections)
            except Exception as e:
                # Never let guidance logic crash the demo.
                print(f"[Guide] Error: {e}")

            # Throttle loop to keep audio/haptics usable during presentations.
            dt = time.time() - loop_start
            if min_loop_dt > 0 and dt < min_loop_dt:
                time.sleep(min_loop_dt - dt)
    except KeyboardInterrupt:
        print("\n[System] Stopped.")
    finally:
        stop_event.set()
        try:
            gpio.write(0)
        except Exception:
            pass
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()

