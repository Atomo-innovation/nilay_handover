# 🫧 Bubble Detection using YOLOv8

## 📌 Overview

This project focuses on **Bubble Detection** using the YOLOv8 object detection model. The system is designed to detect bubbles of different sizes in video-based environments, making it suitable for industrial applications such as liquid monitoring, manufacturing processes, and quality inspection systems.

The model is specifically optimized for **video analysis**, where bubble detection accuracy is critical.

---

## 🚀 Model Details

* Model Used: **YOLOv8**
* Training Epochs:

  * Minimum: **150 epochs**
  * Recommended Maximum: **300 epochs**
* Confidence Threshold:

  * Defined within the prediction scripts

---

## 📊 Model Variants

This project includes multiple trained model weights:

* **Nilay Model**

  * `nilay_best.pt`
  * `nilay_last.pt`

* **Neha Model**

  * `neha_best.pt`
  * `neha_last.pt`

These models allow comparison and flexibility based on performance and use-case requirements.

---

## 🎯 Prediction Modules

Two separate prediction scripts are designed for different bubble detection scenarios:

* **`machinebubblepredict.py`**

  * Specialized for detecting **very small bubbles**
  * Suitable for high-precision industrial environments

* **`predicttiny.py`**

  * Detects **small, medium, and large bubbles**
  * More generalized and versatile detection

---

## 📁 Project Structure

The repository includes the following files:

* `data.yaml` → Dataset configuration (YOLO format)
* `train.py` → Script for model training
* Model Weights:

  * `nilay_best.pt`
  * `nilay_last.pt`
  * `neha_best.pt`
  * `neha_last.pt`
* Prediction Scripts:

  * `machinebubblepredict.py`
  * `predicttiny.py`
* Sample Media:

  * 3 Input Videos
  * 3 Output Videos (detection results)

---

## ⚙️ Usage

### 🔹 Training

Run the training script:

```bash
python train.py
```

### 🔹 Prediction

#### For Very Small Bubble Detection:

```bash
python machinebubblepredict.py
```

#### For Multi-Size Bubble Detection:

```bash
python predicttiny.py
```

---

## 🎥 Testing & Performance

* The model has been primarily tested on **video inputs**
* Performs effectively in detecting bubbles across:

  * Different sizes
  * Continuous motion environments
* Sample input and output videos are included for demonstration

---

## 📈 Recommendations

* Train for at least **150 epochs** for baseline performance
* Increase up to **300 epochs** for improved detection accuracy
* Choose prediction script based on use-case:

  * Use **machinebubblepredict.py** for very small bubbles
  * Use **predicttiny.py** for general detection
* Fine-tune confidence values inside scripts for optimal results

---

## 🧠 Conclusion

This project demonstrates a specialized implementation of YOLOv8 for bubble detection in video environments. With multiple trained models and dedicated prediction scripts, it provides flexibility and accuracy for detecting bubbles of varying sizes in practical scenarios.

---

## 👨‍💻 Author

Developed as part of a computer vision and deep learning project.
