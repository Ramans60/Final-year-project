import subprocess
import time
import os

time.sleep(10)

# Start rkipc first
subprocess.Popen(['rkipc'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(12)

# Start Third Eye
os.chdir('/root/rknn_yolov8_demo')
log = open('/tmp/thirdeye.log', 'w', buffering=1)
subprocess.Popen(
    ['python3', '/root/third_eye.py'],
    stdout=log,
    stderr=log,
    stdin=subprocess.DEVNULL,
    close_fds=True
)
