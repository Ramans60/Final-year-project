import subprocess
import os
import time

GPIO_PIN = 32
VOICES = "/root/voices"
RTSP = "rtsp://127.0.0.1:554/live/0"

def gpio(val):
    with open(f'/sys/class/gpio/gpio{GPIO_PIN}/value','w') as f:
        f.write(str(val))

def buzz(ms=300):
    gpio(1); time.sleep(ms/1000); gpio(0)

def play(name):
    subprocess.Popen(['aplay','-q',f'{VOICES}/{name}.wav'])

def capture():
    subprocess.run([
        'ffmpeg','-y','-rtsp_transport','tcp',
        '-i', RTSP, '-vframes','1',
        '-vf','scale=320:240',
        '-q:v','2', '/tmp/f.jpg'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    return os.path.exists('/tmp/f.jpg')

def get_brightness():
    """Get average brightness of frame — simple motion/presence detection"""
    try:
        result = subprocess.run([
            'ffmpeg','-y','-rtsp_transport','tcp',
            '-i', RTSP, '-vframes','1',
            '-vf','scale=32:32,format=gray',
            '-f','rawvideo','-pix_fmt','gray','/tmp/tiny.raw'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        
        if os.path.exists('/tmp/tiny.raw'):
            with open('/tmp/tiny.raw','rb') as f:
                data = f.read()
            if data:
                avg = sum(data) / len(data)
                return avg
    except:
        pass
    return 128

print("="*40)
print("  NILA - Simple Detection Mode")
print("="*40)

buzz(200); time.sleep(0.1); buzz(200)
play("startup")
time.sleep(3)

prev_brightness = get_brightness()
frame = 0
alert_cooldown = 0

while True:
    frame += 1
    curr = get_brightness()
    diff = abs(curr - prev_brightness)
    
    print(f"[Frame {frame}] brightness={curr:.1f} diff={diff:.1f}")
    
    if diff > 8 and time.time() > alert_cooldown:
        print(f"  MOTION DETECTED!")
        buzz(500)
        play("person_center_1")
        alert_cooldown = time.time() + 3
    
    prev_brightness = curr
    time.sleep(0.5)
