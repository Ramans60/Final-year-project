# ================= CONFIG =================
import os
import time
import subprocess   # ✅ ADD THIS
DEMO_DIR = "/root/rknn_yolov8_demo"
DEMO_BIN = "/root/rknn_yolov8_demo/rknn_yolov8_demo"
MODEL_PATH = "/root/rknn_yolov8_demo/model/yolov8n_6out.rknn"
VOICES_DIR = "/root/voices"
GPIO_PIN = 32
DEMO_ENV = os.environ.copy()

RTSP_URL = "rtsp://172.32.0.93:554/live/0"
INPUT_SIZE = 640

# 🔥 FIX 1: LOWER CONFIDENCE
CONF_THRESHOLD = 0.2   # changed from 0.5

FRAME_PATH = "/tmp/frame.jpg"
FRAME_TMP_PATH = "/tmp/frame.tmp.jpg"

# ================= DETECT =================
def detect(jpg_path):
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
        detections = []
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

                if 0.0 <= conf_raw <= 1.0:
                    conf = conf_raw
                else:
                    conf = 1.0 / (1.0 + math.exp(-conf_raw))

                # 🔥 FIX 1 APPLIED HERE
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, coords)
                cx = (x1 + x2) / 2

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "cx": cx,
                    "xmin": x1,
                    "xmax": x2,
                    "ymin": y1,
                    "ymax": y2,
                })

            except:
                continue

        return detections

    except Exception as e:
        print("[Detect Error]", e)
        return []

# ================= MAIN LOOP =================
def main():
    while True:
        try:
            # Capture frame (already in your code)
            # ...

            detections = detect(FRAME_PATH)

            # 🔥 FIX 2: DEBUG PRINT
            print("RAW DETECTIONS:", detections)

            # 🔥 FIX 3: REMOVE FILTER (TEMP)
            relevant = detections

            if relevant:
                top = relevant[0]

                cx = top["cx"]
                if cx < INPUT_SIZE / 3:
                    zone = "LEFT"
                elif cx > 2 * INPUT_SIZE / 3:
                    zone = "RIGHT"
                else:
                    zone = "CENTER"

                print(f"DETECTED: {top['label']} | {zone} | conf={top['confidence']:.2f}")

            else:
                print("Clear")

        except KeyboardInterrupt:
            print("Stopped")
            break

# ================= RUN =================
if __name__ == "__main__":
    main()
