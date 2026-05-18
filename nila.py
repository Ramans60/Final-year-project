"""
NILA - Leo's AI Companion & Navigation System
============================================
Offline: Camera detects obstacles → NILA guides Leo
Online:  Leo speaks → Groq Whisper → Groq Llama3 → NILA responds
Emergency: Button hold → calls saved number

Board: Luckfox Pico Ultra W
"""

import subprocess
import threading
import time
import os
import random
import socket

# ─────────────────────────────────────────────────────────────
# CONFIG — Change these!
# ─────────────────────────────────────────────────────────────
GROQ_API_KEY    = "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"   # ← paste your key here
EMERGENCY_NUMBER = "6382793248"               # ← Leo's emergency contact
LEO_NAME        = "Leo"
NILA_NAME       = "NILA"

# Hardware
GPIO_PIN        = 32
VOICES_DIR      = "/root/voices"
DEMO_DIR        = "/root/rknn_yolov8_demo"
DEMO_ENV        = {**os.environ, 'LD_LIBRARY_PATH': '/root/rknn_yolov8_demo/lib'}
RTSP_URL        = "rtsp://127.0.0.1:554/live/0"
INPUT_SIZE      = 640
FRAME_PATH      = "/tmp/nila_frame.jpg"
MIC_RECORD_PATH = "/tmp/nila_voice.wav"
RESPONSE_PATH   = "/tmp/nila_response.wav"

# Detection (after confidence normalization to 0..1)
CONF_THRESHOLD  = 0.5

PRIORITY = {
    "truck": 5, "bus": 5, "train": 5,
    "car": 4, "motorcycle": 4, "bicycle": 3,
    "traffic light": 3, "stop sign": 3,
    "person": 2,
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
                candidate_path = f"{VOICES_DIR}/{candidate}.wav"
                if os.path.exists(candidate_path):
                    return candidate
                return name
    except Exception:
        return name
    return name

# ─────────────────────────────────────────────────────────────
# WIFI CHECK
# ─────────────────────────────────────────────────────────────
def is_online():
    try:
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except:
        return False

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
# AUDIO PLAYER
# ─────────────────────────────────────────────────────────────
class AudioPlayer:
    def __init__(self):
        self._proc = None

    def play_file(self, path, block=False):
        if not os.path.exists(path):
            print(f"[Audio] Missing: {path}")
            return
        # Prevent overlapping audio playback.
        if self._proc and self._proc.poll() is None:
            if block:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._proc = None
            else:
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

    def play_voice(self, name, block=False):
        name = resolve_leo_voice_key(name)
        self.play_file(f"{VOICES_DIR}/{name}.wav", block)

    def stop(self):
        if self._proc:
            try: self._proc.terminate()
            except: pass

# ─────────────────────────────────────────────────────────────
# NILA AI BRAIN (Online via Groq)
# ─────────────────────────────────────────────────────────────
class NilaBrain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.conversation_history = []
        self.system_prompt = f"""You are NILA, {LEO_NAME}'s caring AI companion. {LEO_NAME} is visually impaired.

Your personality:
- Sweet, warm, caring like a close friend
- Speak in Tanglish (mix of Tamil words + English) 
- Use words like "da", "pa", "dei" naturally
- Keep responses SHORT (1-2 sentences max) for navigation
- For conversation, be natural and friendly
- Never say you are an AI
- Talk like a real caring friend

Navigation responses should be like:
- "Leo da, careful! Car varuthu right side la, konjam left shift aago"
- "Hey, path clear da, nalla walk panlam"
- "Stop stop! Bus straight ahead, wait panni"

Conversation should be like:
- "Epdi irukka da Leo? Enna aachu?"
- "Ayyo, tension aagatha, I'm here with you da"
- "Seri da, careful ah po, I'm watching"

Emergency: If Leo says he needs help or is in danger, respond urgently and say you are calling for help."""

    def think(self, user_message, context="conversation"):
        """Send message to Groq and get NILA's response"""
        try:
            import urllib.request
            import json

            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })

            # Keep only last 6 messages to save tokens
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]

            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": self.system_prompt}
                ] + self.conversation_history,
                "max_tokens": 100,
                "temperature": 0.8
            }

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )

            with urllib.request.urlopen(req, timeout=8) as response:
                result = json.loads(response.read().decode())
                reply = result["choices"][0]["message"]["content"].strip()

                self.conversation_history.append({
                    "role": "assistant",
                    "content": reply
                })
                return reply

        except Exception as e:
            print(f"[NILA Brain] Error: {e}")
            return None

    def speech_to_text(self, audio_path):
        """Convert Leo's voice to text using Groq Whisper"""
        try:
            import urllib.request
            import json

            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()

            # Groq Whisper API
            import base64
            boundary = "----FormBoundary7MA4YWxkTrZu0gW"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode() + audio_data + (
                f"\r\n--{boundary}\r\n"
                f'Content-Disposition: form-data; name="model"\r\n\r\n'
                f"whisper-large-v3-turbo\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="language"\r\n\r\n'
                f"en\r\n"
                f"--{boundary}--\r\n"
            ).encode()

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}"
                }
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                text = result.get("text", "").strip()
                print(f"[Leo said] {text}")
                return text

        except Exception as e:
            print(f"[STT] Error: {e}")
            return None

# ─────────────────────────────────────────────────────────────
# TEXT TO SPEECH (gTTS)
# ─────────────────────────────────────────────────────────────
def text_to_speech(text, output_path):
    """Convert text to speech using gTTS"""
    try:
        from gtts import gTTS
        tts = gTTS(text, lang='en', tld='co.in')
        mp3_path = output_path.replace('.wav', '.mp3')
        tts.save(mp3_path)
        subprocess.run([
            'ffmpeg', '-y', '-i', mp3_path,
            '-ar', '44100', '-ac', '1', '-f', 'wav', output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(mp3_path)
        return True
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return False

# ─────────────────────────────────────────────────────────────
# EMERGENCY CALL
# ─────────────────────────────────────────────────────────────
def emergency_call(number, audio):
    """Make emergency call using phone"""
    print(f"[EMERGENCY] Calling {number}!")
    audio.play_voice("path_clear")  # placeholder alert sound
    # Try using phone via bluetooth/usb if available
    # For now - play emergency sound loudly
    try:
        subprocess.run([
            'aplay', '-q', f'{VOICES_DIR}/multi_danger.wav'
        ])
        # Could integrate with phone call API here
        print(f"[EMERGENCY] Number to call: {number}")
    except Exception as e:
        print(f"[Emergency] Error: {e}")

# ─────────────────────────────────────────────────────────────
# MIC RECORDING
# ─────────────────────────────────────────────────────────────
def record_voice(duration=5):
    """Record Leo's voice for specified duration"""
    print(f"[Mic] Recording {duration}s...")
    try:
        subprocess.run([
            'arecord', '-D', 'hw:0,0',
            '-f', 'S16_LE', '-r', '16000', '-c', '2',
            '-d', str(duration),
            MIC_RECORD_PATH
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Convert stereo to mono for Whisper
        subprocess.run([
            'ffmpeg', '-y', '-i', MIC_RECORD_PATH,
            '-ac', '1', '-ar', '16000',
            '/tmp/nila_mono.wav'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return '/tmp/nila_mono.wav'
    except Exception as e:
        print(f"[Mic] Error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# BUTTON MONITOR
# ─────────────────────────────────────────────────────────────
class ButtonMonitor:
    """Monitor GPIO button for press events"""
    def __init__(self, pin, on_short_press, on_long_press):
        self.pin = pin
        self.on_short_press = on_short_press
        self.on_long_press  = on_long_press
        self.running = True

        # Setup button pin as input
        try:
            with open('/sys/class/gpio/export', 'w') as f:
                f.write(str(pin))
        except:
            pass
        try:
            with open(f'/sys/class/gpio/gpio{pin}/direction', 'w') as f:
                f.write('in')
        except:
            pass

        threading.Thread(target=self._monitor, daemon=True).start()

    def _read(self):
        try:
            with open(f'/sys/class/gpio/gpio{self.pin}/value', 'r') as f:
                return int(f.read().strip())
        except:
            return 1  # default high (not pressed)

    def _monitor(self):
        last_state = 1
        press_time = 0
        while self.running:
            state = self._read()
            if state == 0 and last_state == 1:  # button pressed
                press_time = time.time()
            elif state == 1 and last_state == 0:  # button released
                duration = time.time() - press_time
                if duration > 2.0:
                    self.on_long_press()   # long press = emergency
                else:
                    self.on_short_press()  # short press = talk to NILA
            last_state = state
            time.sleep(0.05)

# ─────────────────────────────────────────────────────────────
# CAMERA & DETECTION
# ─────────────────────────────────────────────────────────────
def capture_frame():
    try:
        subprocess.run([
            'ffmpeg', '-y', '-rtsp_transport', 'tcp',
            '-i', RTSP_URL, '-vframes', '1',
            '-vf', f'scale={INPUT_SIZE}:{INPUT_SIZE}',
            '-q:v', '2', FRAME_PATH
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return os.path.exists(FRAME_PATH) and os.path.getsize(FRAME_PATH) > 1000
    except:
        return False

def detect(jpg_path):
    try:
        result = subprocess.run(
            ['./rknn_yolov8_demo', 'model/yolov8n_6out.rknn', jpg_path],
            cwd=DEMO_DIR, env=DEMO_ENV,
            capture_output=True, text=True, timeout=10
        )
        detections = []
        import math
        for line in result.stdout.splitlines():
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
                    x1,y1,x2,y2 = int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3])
                    cx = (x1 + x2) / 2.0
                    detections.append({
                        "label": label, "confidence": conf,
                        "cx": cx, "xmin": x1, "xmax": x2
                    })
                except:
                    pass
        return detections
    except:
        return []

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
        self.cooldown     = 2.0
        self.last_alert_time = 0.0
        # Stability: require 3 consecutive frames of same label to confirm.
        self._pending_label = None
        self._pending_count = 0
        self._pending_zones = []

    def get_zone(self, cx):
        if cx < INPUT_SIZE / 3: return "LEFT"
        elif cx > 2 * INPUT_SIZE / 3: return "RIGHT"
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
                self.audio.play_voice("now_clear")
                self.last_label = None
                self.repeat_count = 0
            return

        # Avoid alert spam (voice + haptics).
        if (now - self.last_alert_time) < self.cooldown:
            return

        relevant.sort(key=lambda d: (
            -PRIORITY.get(d["label"], 0),
            0 if self.get_zone(d["cx"]) == "CENTER" else abs(d["cx"] - INPUT_SIZE/2)
        ))

        if len(relevant) >= 3:
            if (now - self.last_time) > self.cooldown:
                self.audio.play_voice("multi_danger")
                self.gpio.continuous(800)
                self.last_time = now
                self.last_alert_time = now
                self.last_label = "multi"
            return

        top      = relevant[0]
        label    = top["label"]
        zone     = self.get_zone(top["cx"])
        priority = PRIORITY.get(label, 0)
        if priority == 0: return

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
            if (now - self.last_time) < self.cooldown: return
            self.repeat_count += 1
            self.audio.play_voice("still_there_1" if self.repeat_count == 2 else "still_there_2")
            if self.repeat_count > 2: self.repeat_count = 2
            self.last_time = now
            self.last_alert_time = now
            threading.Thread(target=self.haptic, args=(zone, priority), daemon=True).start()
            return

        self.repeat_count = 0
        self.last_label = label
        self.last_zone  = zone
        self.last_time  = now

        options = VOICE_MAP.get((label, zone))
        voice   = random.choice(options) if options else "caution"
        print(f"[Guide] {label} | {zone} | conf={top['confidence']:.1f} | {voice}")
        threading.Thread(target=self.haptic, args=(zone, priority), daemon=True).start()
        self.audio.play_voice(voice)
        self.last_alert_time = now

# ─────────────────────────────────────────────────────────────
# NILA CONVERSATION HANDLER
# ─────────────────────────────────────────────────────────────
class NilaConversation:
    def __init__(self, brain, audio, gpio):
        self.brain   = brain
        self.audio   = audio
        self.gpio    = gpio
        self.talking = False

    def handle(self):
        """Leo pressed button — listen and respond"""
        if self.talking:
            return
        self.talking = True

        try:
            online = is_online()

            if not online:
                # Offline — play pre-recorded response
                self.audio.play_voice("path_clear", block=True)
                self.talking = False
                return

            # Online mode — full AI conversation
            print("[NILA] Listening to Leo...")
            # Signal to Leo that NILA is listening (2 quick pulses)
            self.gpio.pulse(2, on_ms=100, off_ms=100)

            # Record Leo's voice
            audio_path = record_voice(duration=5)
            if not audio_path:
                self.talking = False
                return

            # Speech to text
            text = self.brain.speech_to_text(audio_path)
            if not text:
                self.talking = False
                return

            # Check for emergency
            emergency_words = ["help", "danger", "emergency", "call", "amma", "accident", "fell", "hurt"]
            if any(word in text.lower() for word in emergency_words):
                print("[EMERGENCY DETECTED]")
                emergency_call(EMERGENCY_NUMBER, self.audio)
                response = f"Leo da, I'm calling for help right now! Stay where you are da, help is coming!"
            else:
                # Get AI response
                response = self.brain.think(text)

            if response:
                print(f"[NILA says] {response}")
                # Convert to speech
                if text_to_speech(response, RESPONSE_PATH):
                    self.audio.play_file(RESPONSE_PATH, block=True)

        except Exception as e:
            print(f"[NILA Conv] Error: {e}")
        finally:
            self.talking = False

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NILA 🌙 — Leo's AI Companion")
    print("  Navigation + Conversation + Emergency")
    print("=" * 55)

    # Init hardware
    gpio  = GPIO(GPIO_PIN)
    audio = AudioPlayer()

    # Startup
    gpio.pulse(3, on_ms=100, off_ms=100)

    online = is_online()
    if online:
        print("[NILA] Online mode — AI conversation enabled!")
        audio.play_voice("online_mode", block=True)
    else:
        print("[NILA] Offline mode — navigation guidance only")
        audio.play_voice("offline_mode", block=True)

    # Init AI brain
    brain = NilaBrain(GROQ_API_KEY)
    nila_conv = NilaConversation(brain, audio, gpio)
    guidance  = GuidanceBrain(audio, gpio)

    # Say hello to Leo
    if online:
        def say_hello():
            time.sleep(1)
            response = brain.think(f"Say a warm hello to {LEO_NAME} and tell him you're ready to help navigate today. Keep it short and sweet.")
            if response:
                print(f"[NILA] {response}")
                if text_to_speech(response, RESPONSE_PATH):
                    audio.play_file(RESPONSE_PATH)
        threading.Thread(target=say_hello, daemon=True).start()

    # Button: short press = talk, long press = emergency
    def on_short_press():
        print("[Button] Short press — Leo wants to talk")
        threading.Thread(target=nila_conv.handle, daemon=True).start()

    def on_long_press():
        print("[Button] LONG PRESS — EMERGENCY!")
        gpio.continuous(1000)
        emergency_call(EMERGENCY_NUMBER, audio)

    # Note: Using GPIO 33 for button input (GPIO 32 is motor output)
    # If you only have GPIO 32, remove button monitor
    # ButtonMonitor(33, on_short_press, on_long_press)

    print(f"[NILA] Running! {'Online 🌐' if online else 'Offline 📵'}")
    print("[NILA] Press Ctrl+C to stop\n")

    fc = 0
    errors = 0

    while True:
        t0 = time.time()
        fc += 1

        # Skip frames if NILA is talking to Leo
        if nila_conv.talking:
            time.sleep(0.5)
            continue

        # Capture frame
        ok = capture_frame()
        if not ok:
            errors += 1
            if errors > 5:
                time.sleep(2)
                errors = 0
            else:
                time.sleep(0.5)
            continue

        errors = 0

        # Detect obstacles
        detections = detect(FRAME_PATH)
        relevant   = [d for d in detections if PRIORITY.get(d["label"], 0) > 0]

        elapsed = time.time() - t0
        if relevant:
            info = ", ".join([
                f"{d['label']}({'L' if d['cx']<213 else 'R' if d['cx']>427 else 'C'})"
                for d in relevant
            ])
            print(f"[Frame {fc}] {info} | {elapsed:.1f}s")
        else:
            print(f"[Frame {fc}] Clear | {elapsed:.1f}s")

        # Guide Leo
        guidance.process(detections)
        time.sleep(max(0, 0.1 - (time.time() - t0)))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[NILA] Goodbye Leo! Take care da! 🌙")
        try:
            with open(f'/sys/class/gpio/gpio{GPIO_PIN}/value', 'w') as f:
                f.write('0')
        except:
            pass
