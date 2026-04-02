from ultralytics import YOLO

model = YOLO(r"weights\weights\best.pt")

model.predict(
    source="Machine_small_bubble.mp4",
    conf=0.05,          # lower but not too low
    iou=0.35,           # allow nearby bubbles
    imgsz=640,
    max_det=1200,       # allow many detections
    agnostic_nms=True,  # important for dense clusters
    show=True,
    show_labels=False,
    show_conf=False,
    save=True
)
