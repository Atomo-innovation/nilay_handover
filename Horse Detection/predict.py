from ultralytics import YOLO

# load trained model
model = YOLO(r"horse_training_results\content\runs\detect\train2\weights\best.pt")

# predict on test images
model.predict(
    source=(r"C:\Users\Admin\Downloads\20671-311357890_medium.mp4"),
    conf=0.15,
    show =True,
    save = True
)


