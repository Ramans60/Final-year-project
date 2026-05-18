"""
NILA - Leo's AI Navigation Companion
Uses rknn_yolov8_demo binary directly — guaranteed to work!
Camera -> jpg -> binary -> parse detections -> voice + haptic + VISUALIZATION
"""

import subprocess
import threading
import time
import os
import random
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEMO_DIR      = "/root/rknn_yolov8_demo"
DEMO_BIN      = "/root/rknn_yolov8_demo/rknn_yolov8_demo"
MODEL_PATH    = "/root/rknn_yolov8_demo/model/yolov8n_6out.rknn"
VOICES_DIR    = "/root/voices"
GPIO_PIN      = 32
RTSP_URL      = "rtsp://127.0.0.1:554/live/0"
INPUT_SIZE    = 640
CONF_THRESHOLD = 0.5
FRAME_PATH    = "/tmp/frame.jpg"

DEMO_ENV = {**os.environ, 'LD_LIBRARY_PATH': '/root/rknn_yolov8_demo/lib'}

PRIORITY = {
    "truck": 5, "bus": 5, "train": 5,
    "car": 4, "motorcycle": 4, "bicycle": 3,
    "traffic light": 3, "stop sign": 3,
    "person": 2,
    "bench": 1, "chair": 1,
}

VOICE_MAP = {
    ("person",     "CENTER"): ["person_center_1", "person_center_2", "person_center_3"],
    ("person",     "LEFT"):   ["person_left_1",   "person_left_2"],
    ("person",     "RIGHT"):  ["person_right_1",  "person_right_2"],
    ("car",        "CENTER"): ["car_center_1",    "car_center_2",   "car_center_3"],
    ("car",        "LEFT"):   ["car_left_1",      "car_left_2"],
    ("car",        "RIGHT"):  ["car_right_1",     "car_right_2"],
    ("motorcycle", "CENTER"): ["bike_center_1",   "bike_center_2"],
    ("motorcycle", "LEFT"):   ["bike_left_1"],
    ("motorcycle", "RIGHT"):  ["bike_right_1"],
    ("bicycle",    "CENTER"): ["bike_center_1",   "bike_center_2"],
    ("bicycle",    "LEFT"):   ["bike_left_1"],
    ("bicycle",    "RIGHT"):  ["bike_right_1"],
    ("bus",        "CENTER"): ["bus_center_1",    "bus_center_2"],
    ("bus",        "LEFT"):   ["bus_left_1"],
    ("bus",        "RIGHT"):  ["bus_right_1"],
    ("truck",      "CENTER"): ["truck_center_1",  "truck_center_2"],
    ("truck",      "LEFT"):   ["truck_left_1"],
    ("truck",      "RIGHT"):  ["truck_right_1"],
    ("traffic light","CENTER"):["traffic_light"],
    ("traffic light","LEFT"):  ["traffic_light"],
    ("traffic light","RIGHT"): ["traffic_light"],
    ("stop sign",  "CENTER"): ["stop_sign"],
    ("stop sign",  "LEFT"):   ["stop_sign"],
    ("stop sign",  "RIGHT"):  ["stop_sign"],
}


def resolve_leo_voice_key(name: str) -> str:
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
                candidate_path = os.path.join(VOICES_DIR, f"{candidate}.wav")
                if os.path.exists(candidate_path):
                    return candidate
                return name
    except Exception:
        return name
    return name

# ─────────────────────────────────────────────────────────────
# GPIO
# ─────────────────────────────────────────────────────────────
class GPIO:
    def __init__(self, pin):
        self.pin = pin
        try:
            with open('/sys/class/gpio/export', 'w') as f:
                f.write(str(pin))
        except:
            pass
        try:
            with open(f'/sys/class/gpio/gpio{pin}/direction', 'w') as f:
                f.write('out')
            self.write(0)
        except Exception as e:
            print(f"[GPIO] {e}")

    def write(self, val):
        try:
            with open(f'/sys/class/gpio/gpio{self.pin}/value', 'w') as f:
                f.write(str(val))
        except:
            pass

    def pulse(self, times=1, on_ms=200, off_ms=150):
        for i in range(times):
            self.write(1)
            time.sleep(on_ms / 1000)
            self.write(0)
            if i < times - 1:
                time.sleep(off_ms / 1000)

    def continuous(self, duration_ms=600):
        self.write(1)
        time.sleep(duration_ms / 1000)
        self.write(0)

# ─────────────────────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────────────────────
class AudioPlayer:
    def __init__(self, voices_dir):
        self.voices_dir = voices_dir
        self._proc = None

    def play(self, name, block=False):
        name = resolve_leo_voice_key(name)
        path = os.path.join(self.voices_dir, f"{name}.wav")
        if not os.path.exists(path):
            print(f"[Audio] Missing: {name}.wav")
            return
        # Prevent overlapping audio: if something is already playing, ignore.
        if self._proc and self._proc.poll() is None:
            return
        if block:
            subprocess.run(["aplay", "-q", path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            def _run():
                self._proc = subprocess.Popen(
                    ["aplay", "-q", path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._proc.wait()
                self._proc = None
            threading.Thread(target=_run, daemon=True).start()

# ─────────────────────────────────────────────────────────────
# DETECTION using binary
# ─────────────────────────────────────────────────────────────
def detect(jpg_path):
    """Run rknn_yolov8_demo binary on jpg, parse detections"""
    try:
        result = subprocess.run(
            [DEMO_BIN, MODEL_PATH, jpg_path],
            cwd="/root/rknn_yolov8_demo",
            env=DEMO_ENV,
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout
        detections = []
        import math
        for line in output.splitlines():
            # Format: "person @ (211 241 283 506) 1.987"
            if ' @ (' in line and ')' in line:
                try:
                    parts = line.strip().split(' @ (')
                    label = parts[0].strip()
                    rest  = parts[1].split(')')
                    coords = rest[0].split()
                    conf_raw  = float(rest[1].strip())

                    # Normalize to 0..1; use sigmoid for logits.
                    if 0.0 <= conf_raw <= 1.0:
                        conf = conf_raw
                    else:
                        if conf_raw >= 50:
                            conf = 1.0
                        elif conf_raw <= -50:
                            conf = 0.0
                        else:
                            conf = 1.0 / (1.0 + math.exp(-conf_raw))

                    if conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    cx = (x1 + x2) / 2.0

                    detections.append({
                        "label":      label,
                        "confidence": conf,
                        "cx":         cx,
                        "xmin":       x1,
                        "xmax":       x2,
                        "ymin":       y1,
                        "ymax":       y2,
                    })
                except Exception as e:
                    pass
        return detections
    except subprocess.TimeoutExpired:
        print("[Detect] Timeout!")
        return []
    except Exception as e:
        print(f"[Detect] Error: {e}")
        return []

# ─────────────────────────────────────────────────────────────
# VISUALIZATION - Draw bounding boxes on image
# ─────────────────────────────────────────────────────────────
def draw_detections(image_path, detections, output_path="/tmp/frame_with_boxes.jpg"):
    """Draw bounding boxes and labels on the image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # Color palette for different objects
        colors = {
            "person": (0, 255, 0),        # Green
            "car": (255, 0, 0),           # Blue
            "motorcycle": (255, 165, 0),  # Orange
            "bicycle": (0, 255, 255),     # Yellow
            "truck": (128, 0, 128),       # Purple
            "bus": (255, 0, 255),         # Magenta
            "traffic light": (0, 128, 255),  # Orange-Red
            "stop sign": (0, 0, 255),     # Red
            "chair": (255, 255, 0),       # Cyan
            "bench": (128, 128, 0),       # Teal
            "bottle": (0, 128, 128),      # Dark Cyan
            "laptop": (128, 0, 0),        # Dark Red
        }
        
        for det in detections:
            label = det["label"]
            x1, y1 = int(det["xmin"]), int(det["ymin"])
            x2, y2 = int(det["xmax"]), int(det["ymax"])
            conf = det["confidence"]
            
            # Get color (default to bright cyan if not in map)
            color = colors.get(label, (0, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label_text = f"{label} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size for background
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            text_x = x1
            text_y = y1 - 5
            
            # Draw background rectangle for text
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 4, text_y + 4), color, -1)
            
            # Draw text
            cv2.putText(image, label_text, (text_x + 2, text_y - 2),
                       font, font_scale, (255, 255, 255), thickness)
        
        # Save output image
        cv2.imwrite(output_path, image)
        
    except Exception as e:
        print(f"[Draw] Error: {e}")

def display_image(image_path, window_name="Detection Result", wait_ms=100):
    """Display image in OpenCV window"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # Resize for display if too large
        height, width = image.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_ms)
        
    except Exception as e:
        print(f"[Display] Error: {e}")

# ─────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────
def capture_frame():
    """Capture one jpg frame from RTSP"""
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-rtsp_transport', 'tcp',
            '-i', RTSP_URL,
            '-vframes', '1',
            '-vf', f'scale={INPUT_SIZE}:{INPUT_SIZE}',
            '-q:v', '2',
            FRAME_PATH
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)

        if os.path.exists(FRAME_PATH) and os.path.getsize(FRAME_PATH) > 1000:
            return True
    except Exception as e:
        print(f"[Camera] {e}")
    return False

# ─────────────────────────────────────────────────────────────
# GUIDANCE BRAIN
# ─────────────────────────────────────────────────────────────
class GuidanceBrain:
    def __init__(self, audio, gpio):
        self.audio = audio
        self.gpio  = gpio
        self.last_label   = None
        self.last_zone    = None
        self.last_time    = 0
        self.repeat_count = 0
        self.cooldown     = 2.0  # 2s cooldown between alerts
        self.last_alert_time = 0.0
        # Stability: require 3 consecutive frames of same label to confirm.
        self._pending_label = None
        self._pending_count = 0
        self._pending_zones = []

    def get_zone(self, cx):
        if cx < INPUT_SIZE / 3:
            return "LEFT"
        elif cx > 2 * INPUT_SIZE / 3:
            return "RIGHT"
        return "CENTER"

    def haptic(self, zone, priority):
        if zone == "CENTER":
            self.gpio.continuous(800 if priority >= 4 else 500)
        elif zone == "LEFT":
            self.gpio.pulse(1, on_ms=250)
        elif zone == "RIGHT":
            self.gpio.pulse(2, on_ms=200, off_ms=150)

    def process(self, detections):
        now = time.time()

        relevant = [d for d in detections if PRIORITY.get(d["label"], 0) > 0]

        if not relevant:
            if self.last_label and (now - self.last_time) > 4.0:
                self.audio.play("now_clear")
                self.last_label = None
                self.last_zone  = None
                self.repeat_count = 0
            return

        # Avoid alert spam (voice + haptic).
        if (now - self.last_alert_time) < self.cooldown:
            return

        relevant.sort(key=lambda d: (
            -PRIORITY.get(d["label"], 0),
            0 if self.get_zone(d["cx"]) == "CENTER" else abs(d["cx"] - INPUT_SIZE/2)
        ))

        labels = [d["label"] for d in relevant]
        zones  = [self.get_zone(d["cx"]) for d in relevant]

        # Many obstacles
        if len(relevant) >= 3:
            if (now - self.last_time) > self.cooldown:
                self.audio.play("multi_danger")
                self.gpio.continuous(800)
                self.last_time  = now
                self.last_alert_time = now
                self.last_label = "multi"
            return

        # Car + person combo
        if "car" in labels and "person" in labels and "LEFT" in zones and "RIGHT" in zones:
            if (now - self.last_time) > self.cooldown:
                self.audio.play("multi_person_car")
                self.gpio.continuous(600)
                self.last_time  = now
                self.last_alert_time = now
                self.last_label = "multi"
            return

        top      = relevant[0]
        label    = top["label"]
        zone     = self.get_zone(top["cx"])
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

        # Zone smoothing: majority vote over last 3 zones.
        counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
        for z in self._pending_zones:
            if z in counts:
                counts[z] += 1
        zone = max(counts, key=counts.get)

        same = (label == self.last_label and zone == self.last_zone)
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
        self.last_label   = label
        self.last_zone    = zone
        self.last_time    = now

        options = VOICE_MAP.get((label, zone))
        voice   = random.choice(options) if options else ("caution" if zone == "CENTER" else "slow_down")

        print(f"[Guide] {label} | {zone} | conf={top['confidence']:.2f} | -> {voice}")
        threading.Thread(target=self.haptic, args=(zone, priority), daemon=True).start()
        self.audio.play(voice)
        self.last_alert_time = now

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  NILA - Leo's Navigation Companion")
    print("=" * 50)

    gpio  = GPIO(GPIO_PIN)
    audio = AudioPlayer(VOICES_DIR)

    gpio.pulse(2, on_ms=150, off_ms=100)
    audio.play("startup", block=True)
    audio.play("offline_mode")

    brain = GuidanceBrain(audio, gpio)

    print("[System] Running! Press Ctrl+C to stop.\n")

    fc = 0
    errors = 0

    while True:
        t0 = time.time()
        fc += 1

        # Capture frame
        ok = capture_frame()
        if not ok:
            errors += 1
            print(f"[Frame {fc}] Capture failed (errors={errors})")
            time.sleep(2 if errors > 5 else 0.5)
            if errors > 5:
                errors = 0
            continue

        errors = 0

        # Detect objects
        detections = detect(FRAME_PATH)

        elapsed = time.time() - t0
        relevant = [d for d in detections if PRIORITY.get(d["label"], 0) > 0]

        # Draw bounding boxes on image
        if detections:
            draw_detections(FRAME_PATH, detections, "/tmp/frame_with_boxes.jpg")
            display_image("/tmp/frame_with_boxes.jpg", "NILA - Object Detection", wait_ms=1)

        if relevant:
            info = ", ".join([
                f"{d['label']}({'L' if d['cx']<213 else 'R' if d['cx']>427 else 'C'}) {d['confidence']:.1f}"
                for d in relevant
            ])
            print(f"[Frame {fc}] {info} | {elapsed:.2f}s")
        else:
            print(f"[Frame {fc}] Clear | {elapsed:.2f}s")

        brain.process(detections)
        time.sleep(max(0, 0.1 - (time.time() - t0)))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[System] Stopped.")
        try:
            with open(f'/sys/class/gpio/gpio{GPIO_PIN}/value', 'w') as f:
                f.write('0')
        except:
            pass
