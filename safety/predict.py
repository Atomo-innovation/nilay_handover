from ultralytics import YOLO

# load trained model
model = YOLO(r"runs\detect\train6\weights\best.pt")

# predict on test images
model.predict(
    source= "WhatsApp Video 2026-03-12 at 5.37.59 PM.mp4",
    conf=0.38,
    show =True,
    save = True
)