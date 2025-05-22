import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
import random

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Supported animal classes + human-related labels
animal_classes = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'lion',
    'tiger', 'monkey', 'kangaroo', 'panda', 'rabbit',
    'human', 'squirrel', 'deer', 'fox', 'wolf', 'crocodile'
]

# COCO might label humans as "person"
human_aliases = ['person', 'human']

# Assign a distinct color to each class
def generate_colors(classes):
    color_map = {}
    for cls in classes + ['User: Human']:
        color_map[cls] = tuple(random.randint(0, 255) for _ in range(3))
    return color_map

class_colors = generate_colors(animal_classes)

# Initialize tracker
tracker = DeepSort(max_age=30)

# === Handle image file input ===
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to read image.")
        sys.exit()

    results = model(image)[0]

    max_area = 0
    user_box = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in human_aliases:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                user_box = (x1, y1, x2, y2)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in animal_classes or label in human_aliases:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if user_box and (x1, y1, x2, y2) == user_box:
                label = "User: Human"

            color = class_colors.get(label, (255, 0, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.namedWindow("Animal Detection - Image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Animal Detection - Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Animal Detection - Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()

# === Real-time webcam mode ===
cap = cv2.VideoCapture(0)

print("Press 's' to stop, 'r' to reset tracking...")

# Fullscreen display setup
cv2.namedWindow("Animal Detection & Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Animal Detection & Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    max_area = 0
    user_box = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in human_aliases:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                user_box = (x1, y1, x2, y2)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in animal_classes or label in human_aliases:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            w, h = x2 - x1, y2 - y1

            if user_box and (x1, y1, x2, y2) == user_box:
                label = "User: Human"

            detections.append(([x1, y1, w, h], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        label = track.get_det_class()
        color = class_colors.get(label, (0, 255, 0))

        cv2.rectangle(frame, (l, t), (l + w, t + h), color, 2)
        cv2.putText(frame, f'{label} ID {track_id}', (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Animal Detection & Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Stop
        print("[INFO] Stopping detection...")
        break
    elif key == ord('r'):  # Reset tracker
        print("[INFO] Tracker reset.")
        tracker = DeepSort(max_age=30)

cap.release()
cv2.destroyAllWindows()
