import os
import sys
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------
# Parse command-line arguments
# ------------------------------
parser = argparse.ArgumentParser(description="YOLO Detection with /dev/video0")
parser.add_argument('--model', required=True,
                    help='Path to YOLO model file (.pt or .onnx)')
parser.add_argument('--thresh', type=float, default=0.5,
                    help='Minimum confidence threshold (default: 0.5)')
parser.add_argument('--resolution', default="640x480",
                    help='Resolution WxH for camera (default: 640x480)')
parser.add_argument('--record', action='store_true',
                    help='Record video output as demo1.avi')
args = parser.parse_args()

model_path = args.model
min_thresh = args.thresh
resW, resH = map(int, args.resolution.split('x'))
record = args.record

# ------------------------------
# Validate model file
# ------------------------------
if not os.path.isfile(model_path):
    print(f"ERROR: Model file '{model_path}' not found.")
    sys.exit(1)

# ------------------------------
# Load YOLO or ONNX model
# ------------------------------
try:
    model = YOLO(model_path)  # ultralytics handles both .pt and .onnx
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    sys.exit(1)

labels = model.names

# ------------------------------
# Set up /dev/video0 capture
# ------------------------------
cap = cv2.VideoCapture(0)  # /dev/video0 is usually index 0
if not cap.isOpened():
    print("ERROR: Cannot open /dev/video0")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# ------------------------------
# Set up recording if enabled
# ------------------------------
if record:
    recorder = cv2.VideoWriter(
        'demo1.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (resW, resH)
    )

# Bounding box colors
bbox_colors = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209),
    (178, 182, 133), (88, 159, 106), (96, 202, 231),
    (159, 124, 168), (169, 162, 241), (98, 118, 150),
    (172, 176, 184)
]

# FPS tracking
frame_rate_buffer = []
fps_avg_len = 200
avg_frame_rate = 0

# ------------------------------
# Inference loop
# ------------------------------
while True:
    t_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from /dev/video0")
        break

    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(det.cls.item())
        classname = labels[classidx]
        conf = det.conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f"{classname}: {int(conf * 100)}%"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    # Draw FPS and object count
    cv2.putText(frame, f"FPS: {avg_frame_rate:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {object_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

    cv2.imshow("YOLO detection results", frame)

    if record:
        recorder.write(frame)

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

    # FPS calculation
    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# ------------------------------
# Cleanup
# ------------------------------
cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
print(f"Average pipeline FPS: {avg_frame_rate:.2f}")
