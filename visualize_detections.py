#!/usr/bin/env python3
"""
Standalone visualization script for RKNN YOLOv8 detections
Captures frame -> detects objects -> draws boxes -> displays result
"""

import subprocess
import os
import cv2
import time

# Configuration
DEMO_DIR = "/root/rknn_yolov8_demo"
DEMO_BIN = "/root/rknn_yolov8_demo/rknn_yolov8_demo"
MODEL_PATH = "/root/rknn_yolov8_demo/model/yolov8n_6out.rknn"
RTSP_URL = "rtsp://127.0.0.1:554/live/0"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.3
FRAME_PATH = "/tmp/frame.jpg"
OUTPUT_PATH = "/tmp/frame_with_boxes.jpg"

DEMO_ENV = {**os.environ, 'LD_LIBRARY_PATH': '/root/rknn_yolov8_demo/lib'}

# Color map for different object classes
COLORS = {
    "person": (0, 255, 0),          # Green
    "car": (255, 0, 0),             # Blue
    "motorcycle": (255, 165, 0),    # Orange
    "bicycle": (0, 255, 255),       # Yellow
    "truck": (128, 0, 128),         # Purple
    "bus": (255, 0, 255),           # Magenta
    "traffic light": (0, 128, 255), # Orange-Red
    "stop sign": (0, 0, 255),       # Red
    "chair": (255, 255, 0),         # Cyan
    "bench": (128, 128, 0),         # Teal
    "bottle": (0, 128, 128),        # Dark Cyan
    "laptop": (128, 0, 0),          # Dark Red
}


def capture_frame():
    """Capture frame from RTSP stream"""
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
        
        return os.path.exists(FRAME_PATH) and os.path.getsize(FRAME_PATH) > 1000
    except Exception as e:
        print(f"[Camera] {e}")
        return False


def detect_objects(jpg_path):
    """Run detection using rknn_yolov8_demo binary"""
    try:
        result = subprocess.run(
            [DEMO_BIN, MODEL_PATH, jpg_path],
            cwd=DEMO_DIR,
            env=DEMO_ENV,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stdout
        detections = []
        import math
        
        for line in output.splitlines():
            if ' @ (' in line and ')' in line:
                try:
                    parts = line.strip().split(' @ (')
                    label = parts[0].strip()
                    rest = parts[1].split(')')
                    coords = rest[0].split()
                    conf_raw = float(rest[1].strip())
                    
                    # Normalize confidence
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
                        "label": label,
                        "confidence": conf,
                        "cx": cx,
                        "xmin": x1,
                        "xmax": x2,
                        "ymin": y1,
                        "ymax": y2,
                    })
                except Exception:
                    pass
        
        return detections
    except subprocess.TimeoutExpired:
        print("[Detect] Timeout!")
        return []
    except Exception as e:
        print(f"[Detect] {e}")
        return []


def draw_boxes_on_image(image_path, detections, output_path):
    """Draw bounding boxes with labels on image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Draw] Cannot read image: {image_path}")
            return False
        
        for det in detections:
            label = det["label"]
            x1, y1 = int(det["xmin"]), int(det["ymin"])
            x2, y2 = int(det["xmax"]), int(det["ymax"])
            conf = det["confidence"]
            
            # Get color
            color = COLORS.get(label, (0, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare text
            text = f"{label} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = x1
            text_y = y1 - 8
            
            # Draw background for text
            cv2.rectangle(image, 
                         (text_x - 2, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 2, text_y + 4),
                         color, -1)
            
            # Draw text
            cv2.putText(image, text, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Draw center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
        
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"[Draw] {e}")
        return False


def display_result(image_path):
    """Display the result image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Display] Cannot read image: {image_path}")
            return
        
        # Resize if too large
        height, width = image.shape[:2]
        if width > 1024:
            scale = 1024 / width
            new_width = 1024
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        cv2.imshow("NILA - Object Detection Visualization", image)
        cv2.waitKey(100)
    except Exception as e:
        print(f"[Display] {e}")


def main():
    print("=" * 60)
    print("  NILA - Real-Time Object Detection Visualization")
    print("=" * 60)
    
    frame_count = 0
    
    try:
        while True:
            frame_count += 1
            start_time = time.time()
            
            # Capture
            print(f"\n[Frame {frame_count}] Capturing...", end=" ", flush=True)
            if not capture_frame():
                print("FAILED")
                continue
            print("OK")
            
            # Detect
            print(f"[Frame {frame_count}] Detecting...", end=" ", flush=True)
            detections = detect_objects(FRAME_PATH)
            print(f"Found {len(detections)} objects")
            
            if detections:
                # List detections
                for det in detections:
                    zone = "LEFT" if det["cx"] < INPUT_SIZE/3 else "RIGHT" if det["cx"] > 2*INPUT_SIZE/3 else "CENTER"
                    print(f"  → {det['label']:15} | Zone: {zone:6} | Conf: {det['confidence']:.2f}")
                
                # Draw boxes
                print(f"[Frame {frame_count}] Drawing boxes...", end=" ", flush=True)
                if draw_boxes_on_image(FRAME_PATH, detections, OUTPUT_PATH):
                    print("OK")
                    print(f"[Frame {frame_count}] Displaying...", end=" ", flush=True)
                    display_result(OUTPUT_PATH)
                    print("OK")
                else:
                    print("FAILED")
            else:
                print(f"[Frame {frame_count}] No detections - area is clear")
            
            elapsed = time.time() - start_time
            print(f"[Frame {frame_count}] Time: {elapsed:.2f}s")
            
    except KeyboardInterrupt:
        print("\n\n[System] Stopped by user")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
