import subprocess, os, time, random

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

def gpio_write(val):
    with open(f'/sys/class/gpio/gpio{GPIO_PIN}/value','w') as f:
        f.write(str(val))

def buzz(ms=500):
    gpio_write(1); time.sleep(ms/1000); gpio_write(0)

def play(name):
    global _audio_proc
    name = resolve_leo_voice_key(name)
    # Prevent overlapping audio playback.
    if _audio_proc and _audio_proc.poll() is None:
        return
    _audio_proc = subprocess.Popen(['aplay', '-q', f'{VOICES}/{name}.wav'],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)

def detect(image):
    r = subprocess.run(
        ['./rknn_yolov8_demo','model/yolov8n_6out.rknn', image],
        cwd=DEMO_DIR, env=DEMO_ENV,
        capture_output=True, text=True, timeout=10
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

            x1,x2 = int(coords[0]), int(coords[2])
            cx = (x1+x2)/2
            zone = "LEFT" if cx<213 else "RIGHT" if cx>427 else "CENTER"
            dets.append((label, zone))
    return dets

print("="*50)
print("  NILA - Leo's AI Navigation System")
print("  LIVE DEMO")
print("="*50)

# Startup
buzz(200); time.sleep(0.1); buzz(200)
play("startup")
time.sleep(3)

# Demo scenarios
scenarios = [
    ("/root/rknn_yolov8_demo/model/bus.jpg", "Real scenario: Bus + People"),
]

for image, title in scenarios:
    print(f"\n[Demo] {title}")
    dets = detect(image)
    
    for label, zone in dets:
        print(f"  DETECTED: {label} | {zone}")
        
        voice_map = {
            ("person","CENTER"): "person_center_1",
            ("person","LEFT"):   "person_left_1", 
            ("person","RIGHT"):  "person_right_1",
            ("bus","CENTER"):    "bus_center_1",
            ("bus","LEFT"):      "bus_left_1",
            ("car","CENTER"):    "car_center_1",
            ("car","RIGHT"):     "car_right_1",
        }
        
        voice = voice_map.get((label,zone), "caution")
        
        if zone == "CENTER":
            buzz(800)
        elif zone == "LEFT":
            buzz(250)
        elif zone == "RIGHT":
            buzz(200); time.sleep(0.15); buzz(200)
        
        play(voice)
        time.sleep(3)
        break

print("\n[Demo] System working perfectly!")
print("[Demo] Navigation + Voice + Haptic all active!")

# Now run live
print("\n[Live] Starting live detection...")
play("offline_mode")
time.sleep(2)

fc = 0
while True:
    fc += 1
    # Capture live frame
    subprocess.run([
        'ffmpeg','-y','-rtsp_transport','tcp',
        '-i','rtsp://127.0.0.1:554/live/0',
        '-vframes','1','-vf','scale=640:640',
        '-q:v','1','/tmp/live.jpg'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    
    dets = detect('/tmp/live.jpg')
    relevant = [(l,z) for l,z in dets if l in ['person','car','bus','truck','motorcycle','bicycle']]
    
    if relevant:
        label, zone = relevant[0]
        print(f"[Frame {fc}] LIVE DETECTED: {label} | {zone}")
        now = time.time()
        if (now - _last_alert_time) < ALERT_COOLDOWN_SEC:
            time.sleep(0.1)
            continue

        voice_map = {
            ("person","CENTER"): "person_center_1",
            ("person","LEFT"):   "person_left_1",
            ("person","RIGHT"):  "person_right_1",
            ("car","CENTER"):    "car_center_1",
            ("car","LEFT"):      "car_left_1",
            ("car","RIGHT"):     "car_right_1",
            ("bus","CENTER"):    "bus_center_1",
            ("bus","LEFT"):      "bus_left_1",
            ("bus","RIGHT"):     "bus_right_1",
            ("truck","CENTER"):  "truck_center_1",
            ("truck","LEFT"):    "truck_left_1",
            ("truck","RIGHT"):   "truck_right_1",
        }
        voice = voice_map.get((label,zone), "caution")
        print(f"  AUDIO: {voice}")
        buzz(500 if zone=="CENTER" else 250)
        play(voice)
        _last_alert_time = now
        time.sleep(2)
    else:
        print(f"[Frame {fc}] Clear")
    
    time.sleep(0.5)
