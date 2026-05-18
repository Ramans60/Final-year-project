import subprocess, os, time

DEMO_DIR = "/root/rknn_yolov8_demo"
DEMO_ENV = {**os.environ, 'LD_LIBRARY_PATH': '/root/rknn_yolov8_demo/lib'}
VOICES = "/root/voices"
GPIO_PIN = 32
ALERT_COOLDOWN_SEC = 2.0

_audio_proc = None
_last_alert_time = 0.0


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
                if os.path.exists(f"{VOICES}/{candidate}.wav"):
                    return candidate
                return name
    except Exception:
        return name
    return name

# ---------- GPIO AUTO INIT ----------
def init_gpio():
    try:
        with open('/sys/class/gpio/export','w') as f:
            f.write(str(GPIO_PIN))
    except:
        pass
    try:
        with open(f'/sys/class/gpio/gpio{GPIO_PIN}/direction','w') as f:
            f.write('out')
    except:
        pass

def gpio_write(val):
    try:
        with open(f'/sys/class/gpio/gpio{GPIO_PIN}/value','w') as f:
            f.write(str(val))
    except:
        pass

def buzz(ms=300):
    gpio_write(1)
    time.sleep(ms/1000)
    gpio_write(0)

# ---------- AUDIO ----------
def play(name):
    name = resolve_leo_voice_key(name)
    global _audio_proc
    # Prevent overlapping audio playback.
    if _audio_proc and _audio_proc.poll() is None:
        return

    _audio_proc = subprocess.Popen(
        ['aplay', '-q', f'{VOICES}/{name}.wav'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# ---------- DETECTION ----------
def detect(image):
    r = subprocess.run(
        ['./rknn_yolov8_demo','model/yolov8n_6out.rknn', image],
        cwd=DEMO_DIR,
        env=DEMO_ENV,
        capture_output=True,
        text=True,
        timeout=5
    )
    dets = []
    import math
    for line in r.stdout.splitlines():
        if ' @ (' in line:
            label = line.split(' @ (')[0].strip()
            coords_part = line.split('(')[1].split(')')[0]
            coords = coords_part.split()
            # Parse confidence if present after ')'
            try:
                conf_raw = float(line.split(')')[1].strip())
            except Exception:
                conf_raw = 1.0

            if 0.0 <= conf_raw <= 1.0:
                conf = conf_raw
            else:
                if conf_raw >= 50:
                    conf = 1.0
                elif conf_raw <= -50:
                    conf = 0.0
                else:
                    conf = 1.0 / (1.0 + math.exp(-conf_raw))

            if conf < 0.5:
                continue

            x1, x2 = int(coords[0]), int(coords[2])
            cx = (x1 + x2) / 2
            # Frame is scaled to 320x320 in the live capture.
            # Use 1/3 and 2/3 split like the other demos (but for 320).
            zone = "LEFT" if cx < 107 else "RIGHT" if cx > 213 else "CENTER"
            dets.append((label, zone))
    return dets

# ---------- START ----------
print("="*50)
print(" NILA - FAST DEMO MODE")
print("="*50)

init_gpio()

buzz(200)
time.sleep(0.1)
buzz(200)
play("startup")
time.sleep(2)

# ---------- LOOP ----------
fc = 0

while True:
    fc += 1

    # Skip frames (speed boost)
    if fc % 2 != 0:
        continue

    # Capture frame safely
    try:
        subprocess.run([
            'ffmpeg',
            '-rtsp_transport','tcp',
            '-i','rtsp://127.0.0.1:554/live/0',
            '-vframes','1',
            '-vf','scale=320:320',
            '-q:v','2',
            '/tmp/live.jpg'
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=2
        )
    except:
        print("[WARN] Frame skip (camera delay)")
        continue

    # Detect
    dets = detect('/tmp/live.jpg')

    relevant = [(l, z) for l, z in dets if l in ['person','car','bus','truck']]

    if relevant:
        label, zone = relevant[0]
        print(f"[Frame {fc}] DETECTED: {label} | {zone}")

        now = time.time()
        if (now - _last_alert_time) < ALERT_COOLDOWN_SEC:
            # Skip new alert (audio + haptics) to reduce repetition/overlap.
            time.sleep(0.2)
            continue

        if zone == "CENTER":
            buzz(600)
        elif zone == "LEFT":
            buzz(200)
        else:
            buzz(200)
            time.sleep(0.1)
            buzz(200)

        voice_map = {
            ("person", "CENTER"): "person_center_1",
            ("person", "LEFT"): "person_left_1",
            ("person", "RIGHT"): "person_right_1",
            ("car", "CENTER"): "car_center_1",
            ("car", "LEFT"): "car_left_1",
            ("car", "RIGHT"): "car_right_1",
            ("bus", "CENTER"): "bus_center_1",
            ("bus", "LEFT"): "bus_left_1",
            ("bus", "RIGHT"): "bus_right_1",
            ("truck", "CENTER"): "truck_center_1",
            ("truck", "LEFT"): "truck_left_1",
            ("truck", "RIGHT"): "truck_right_1",
        }
        voice = voice_map.get((label, zone), "caution")
        print(f"  AUDIO: {voice}")
        play(voice)
        _last_alert_time = now

    else:
        print(f"[Frame {fc}] Clear")

    time.sleep(0.2)
