# 🐾 Real-Time Animal Detection and Tracking using YOLOv8 + Deep SORT

This project is a real-time object detection and tracking system focused on identifying various animals (and humans) from a webcam or image input. It uses:

- 🔍 **YOLOv8** for fast and accurate object detection  
- 🎯 **Deep SORT** for object tracking across video frames  
- 🖼️ Fullscreen view with **color-coded labels per animal type**

---

## 🚀 Features

- Detects animals and humans in **real-time** via webcam or image input
- Uses **distinct colors** for each animal/human class for visual clarity
- Tracks animals with unique IDs using **Deep SORT**
- Identifies largest visible person as **"User: Human"**
- Supports **resetting tracker** or **stopping program** via keypress

---

## 🐍 Requirements

Install the dependencies via pip:

```bash
pip install opencv-python ultralytics deep_sort_realtime
```

> Make sure you have Python 3.7+ installed.

---

## 📦 Files

- `animal_tracker.py`: Main Python script for image/video detection
- `yolov8s.pt`: Pre-trained YOLOv8 model from [Ultralytics](https://github.com/ultralytics/ultralytics)

---

## 🖼️ Image Mode

To detect animals in an image:

```bash
python animal_tracker.py path/to/image.jpg
```

---

## 📹 Real-Time Webcam Mode

To run live animal tracking:

```bash
python animal_tracker.py
```

### Controls:
- Press **`s`** → Stop detection
- Press **`r`** → Reset tracker

---

## 🧠 Supported Classes

> Includes common COCO animals + extended wildlife support

- Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe  
- Lion, Tiger, Monkey, Kangaroo, Panda, Rabbit, Squirrel, Deer, Fox  
- Wolf, Crocodile, Human (shown as **"User: Human"** if primary)

---

## 🖍️ Visual Output

- Objects are shown in fullscreen mode
- Each class gets a **unique, randomly generated color**
- Labels and track IDs are drawn above bounding boxes

---


## 📌 Notes

- You can switch the YOLO model (`yolov8s.pt`) to `yolov8m.pt` or others for more accuracy.
- Extend the `animal_classes` list to support more wildlife if using a custom-trained model.

---

## 📄 License

This project is for educational and research purposes.

---
```
