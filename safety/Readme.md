# 🦺 Construction Safety PPE Detection using YOLOv8

## 📌 Overview

This project focuses on **Construction Safety PPE (Personal Protective Equipment) Detection** using the YOLOv8 object detection model. The system is designed to identify whether workers are properly equipped with safety gear such as helmets, vests, and other protective equipment in construction environments.

The model is optimized for both **image and video-based detection**, making it suitable for real-world monitoring and surveillance applications.

---

## 🚀 Model Details

* Model Used: **YOLOv8**
* Training Epochs:

  * Minimum: **300 epochs**
  * Recommended Maximum: **400 epochs**
* Confidence Threshold:

  * General Use: **0.30 – 0.40**
  * Best Accuracy: **0.50**

---

## 📊 Dataset Information

* The dataset was originally created using **Roboflow**
* Although the original dataset project was deleted, a **backup dataset was preserved and reused from GitHub**
* Dataset Format:

  * YOLO format (images + corresponding `.txt` annotation files)
* The `data.yaml` file contains:

  * Training and validation paths
  * Class labels
  * Dataset configuration details

---

## 🎯 Key Features

* Detects safety compliance in construction environments
* Supports both **image and video inference**
* Works effectively in **real-world conditions**
* Compatible with CCTV and surveillance systems
* Trained for higher epochs to improve robustness and accuracy

---

## 📁 Project Structure

The repository includes the following files:

* `data.yaml` → Dataset configuration (YOLO format)
* `best.pt` → Best trained model weights
* `last.pt` → Final epoch model weights
* `train.py` → Script for model training
* `predict.py` → Script for detection/inference
* Sample Media:

  * Input Videos (raw footage)
  * Output Videos (with PPE detection results)

---

## ⚙️ Usage

### 🔹 Training

Train the model using:

```bash id="trn33x"
python train.py
```

### 🔹 Prediction

Run detection on images or videos:

```bash id="prd44y"
python predict.py
```

---

## 📈 Recommendations

* Train the model for at least **300 epochs** for stable performance
* Extend training up to **400 epochs** for improved accuracy
* Use **confidence threshold between 0.30–0.40** for balanced results
* Use **0.50 confidence** when higher precision is required
* Ensure dataset diversity for better real-world performance

---

## 🧠 Conclusion

This project demonstrates a reliable approach to **construction safety monitoring** using YOLOv8. Despite the loss of the original dataset source, the preserved dataset and trained weights allow continued development and deployment. The model performs effectively in detecting PPE compliance in practical scenarios.

---

## 👨‍💻 Author

Developed as part of a computer vision and deep learning project focused on safety and automation.
