from ultralytics import YOLO

# load trained model
model = YOLO(r"with_person\best (1).pt")

# predict on test images
model.predict(
    source= r"vedio_check\43_J PTZ Ganthiyad Chandkheda_14May2025_150434_14May2025_150505.MKV",
    conf=0.25,
    show = True,
    save = True
)