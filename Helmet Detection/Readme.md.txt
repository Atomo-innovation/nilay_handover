# 🪖 Helmet Detection using YOLOv8

## 📌 Overview

This project focuses on **Helmet Detection** using the YOLOv8 object detection model. The goal is to accurately detect whether a person is wearing a helmet in real-time scenarios such as construction sites, roads, and CCTV surveillance.

We have developed and tested multiple dataset versions (**v2 to v6**), where all versions perform well, but **Version 4, 5, and 6 provide the best results**.

---

## 🚀 Model Details

* Model Used: **YOLOv8**
* Training Epochs:

  * Minimum: **200 epochs**
  * Recommended Maximum: **350 epochs**
* Confidence Threshold:

  * Default: **0.25**

---

## 📊 Dataset Versions

We experimented with multiple dataset versions:

* **Version 2 → Version 6**: All versions are functional and provide good detection.
* **Version 4 & 5**:

  * Best suited for **images and photo-based detection**
  * High accuracy in controlled environments
* **Version 6 (Augmented Dataset)**:

  * Designed for **real-world scenarios**
  * Performs well on:

    * CCTV footage
    * Low-quality images
    * Different lighting conditions
  * Most robust and practical version

---

## 🎯 Key Features

* Real-time helmet detection
* Works on both **images and videos**
* Optimized for **CCTV and real-world environments**
* Supports custom training and prediction pipelines
* High accuracy with properly trained datasets

---

## 📁 Project Structure

The repository includes the following files:

* `data.yaml` → Dataset configuration file
* `best.pt` → Best trained model weights
* `last.pt` → Last epoch model weights
* `train.py` → Script for training the model
* `predict.py` → Script for inference/prediction
* Sample Input/Output:

  * 2 Input Videos
  * 2 Output Videos (with detection results)

---

## ⚙️ Usage

### 🔹 Training

Run the training script:

```bash
python train.py
```

### 🔹 Prediction

Run detection on images/videos:

```bash
python predict.py
```

---

## 📈 Recommendations

* Use **Version 6 dataset** for real-world deployment
* Train for at least **200 epochs** for good performance
* Increase to **300–350 epochs** for better accuracy
* Keep confidence threshold at **0.25** for balanced detection

---

## 🧠 Conclusion

This project demonstrates an efficient and scalable approach to helmet detection using YOLOv8. While all dataset versions perform well, **Version 6 stands out for real-world applications**, making it ideal for deployment in surveillance systems.

---

## 👨‍💻 Author

Developed as part of a practical deep learning and computer vision project.
