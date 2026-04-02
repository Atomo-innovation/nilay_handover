from ultralytics import YOLO

model = YOLO(r"weights\weights\best.pt")

model.predict(
    source="blue_bubble_small_big.mp4",
    conf=0.15,        # increase slightly
    iou=0.4,          # safer NMS
    imgsz=640,
    max_det=800,      # still high but controlled
    show=True,
    show_labels=False,
    show_conf=False,
    save=False
)
