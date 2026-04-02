from ultralytics import YOLO

# load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# train model
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640
)
