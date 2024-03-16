from ultralytics import YOLO
import cv2
# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/pose/train/weights/best.pt')  # load a custom model

# Predict with the model

# results = model('dataset/images/val/ZDSsleeping20230329_V2_sample_factory_in_100_11512.jpg')  # predict on an image

results = model('E:\\pythonFile\\ultralytics-main-state\\datasets\\images\\val\\0002.jpg',show=True,save=True)
results[0].plot()
"""
cap = cv2.VideoCapture(0)
while cap.isOpend():
    res, frame = cap.read()
    if res:
        result = model(frame)
"""