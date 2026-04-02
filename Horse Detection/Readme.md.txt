# 🐎 Horse Detection using YOLOv8

## 📌 Overview

This project focuses on **Horse Detection** using the YOLOv8 object detection model. The model is designed to accurately detect horses in both **images and videos**, making it suitable for real-world applications such as surveillance, wildlife monitoring, and automated detection systems.

The system performs effectively across different environments and is optimized for both static and dynamic inputs.

---

## 🚀 Model Details

* Model Used: **YOLOv8**
* Training Epochs:

  * Minimum: **150–200 epochs**
  * Recommended Maximum: **300 epochs**
* Supports:

  * Image detection
  * Video detection

---

## 📊 Dataset Information

* Dataset is structured in **YOLO format** (images with corresponding `.txt` annotations)
* The `data.yaml` file includes:

  * Dataset paths (train/validation)
  * Class definitions (horse)
  * Configuration details for training

---

## 🎯 Key Features

* Accurate horse detection in **images and videos**
* Works well in **real-world environments**
* Supports custom training and inference pipelines
* Optimized for both **photo-based and motion-based detection**
* Lightweight and efficient using YOLOv8

---

## ⚠️ Known Challenges

While the model performs well, there are some limitations:

### ❗ False / Negative Detection

* The model may sometimes detect objects incorrectly (false positives)
* This issue occurs due to limited background diversity or insufficient negative samples

---

## 🛠️ Solutions & Improvements

To improve model accuracy and reduce false detections:

* ✅ **Add Negative Images**

  * Include images without horses to help the model distinguish better

* ✅ **Use Background Images**

  * If focusing only on horse detection, add diverse background images (without horses)

* ✅ **Increase Dataset Diversity**

  * Include different:

    * Environments
    * Lighting conditions
    * Angles and distances

* ✅ **Balanced Dataset**

  * Maintain a good balance between horse images and non-horse/background images

---

## 📁 Project Structure

The repository includes the following files:

* `data.yaml` → Dataset configuration file
* `best.pt` → Best trained model weights
* `last.pt` → Final trained model weights
* `train.py` → Script for training the model
* `predict.py` → Script for detection/inference
* Sample Media:

  * Input Videos (raw footage)
  * Output Videos (detection results)

---

## ⚙️ Usage

### 🔹 Training

Run the training script:

```bash
python train.py
```

### 🔹 Prediction

Run detection on images or videos:

```bash
python predict.py
```

---

## 📈 Recommendations

* Train for at least **150–200 epochs** for stable performance
* Increase training up to **300 epochs** for better accuracy
* Improve dataset quality to reduce false detections
* Regularly test on real-world videos for validation

---

## 🧠 Conclusion

This project demonstrates an effective approach to horse detection using YOLOv8. The model performs well on both images and videos, but accuracy can be further improved by enhancing dataset quality and adding negative/background samples. With proper tuning, it can be deployed in real-world applications.

---

## 👨‍💻 Author

Developed as part of a computer vision and deep learning project.
