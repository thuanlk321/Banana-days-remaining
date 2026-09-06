# 🍌 Banana Ripeness Prediction

**Predict the remaining days until a banana spoils using Computer Vision and Deep Learning**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Introduction

This project uses **Deep Learning** and **Computer Vision** to predict how many days a banana has left before it goes bad (remaining shelf life).  

By taking a photo or using a webcam, the model analyzes the banana’s appearance and estimates the remaining days (from 0 to 7 days).

The system is built with **Transfer Learning** (MobileNetV2) and can run in real-time through a webcam application.

---

## ✨ Features

- Predict **remaining days** until the banana spoils
- Real-time prediction using webcam
- Capture result with a single key press (`SPACE`)
- Beautiful result screen showing the captured image + prediction
- Proper data splitting by Banana ID (prevents data leakage)
- Data augmentation for better generalization
- Easy to train and easy to run

---

## 📸 Demo

| Live Webcam | Result Screen |
|:-----------:|:-------------:|
| ![Webcam](assets/webcam_demo.gif) | ![Result](assets/result.png) |

**Controls:**
- `SPACE` → Capture current prediction and show result
- `Q` → Quit the application

---

## 🗂️ Project Structure

```bash
banana-ripeness-prediction/
│
├── bananaML.py              # Training script
├── webcamApp.py             # Real-time webcam application
├── banana_model.h5          # Trained model
├── requirements.txt         # Required libraries
├── README.md
│
└── banana/                  # Dataset
    ├── Banana_ID_1/
    │   ├── Day_0.jpg
    │   ├── Day_1.jpg
    │   ├── ...
    │   └── Day_7.jpg
    ├── Banana_ID_2/
    └── ...

🧠 Model Architecture

Backbone: MobileNetV2 (pre-trained on ImageNet)
Custom Head:
GlobalAveragePooling2D
Dense(128, activation='relu')
Dropout(0.3)
Dense(1, activation='linear') → predicts remaining days


Training Strategy:

Phase 1: Freeze the backbone and train for 30 epochs
Phase 2: Unfreeze and fine-tune for 10 epochs with a very low learning rate

Loss Function: Mean Squared Error (MSE)

Metric: Mean Absolute Error (MAE)

📊 Dataset

The dataset consists of multiple bananas tracked over 8 days (Day 0 to Day 7).
Each banana has its own folder (Banana_ID_x).
Target variable: remaining_days = 7 - current_day
Data is split by Banana ID (not by individual images) to avoid data leakage.


🚀 How to Run
1. Clone the repository
Bashgit clone https://github.com/your-username/banana-ripeness-prediction.git
cd banana-ripeness-prediction

2. Install dependencies
Bashpip install -r requirements.txt

3. Train the model (optional)
Bashpython bananaML.py
After training, the file banana_model.h5 will be generated

4. Run the webcam application
Bashpython webcamApp.py


📈 Results
The model is evaluated on the test set after training.

The script prints:

Test Loss (MSE)
Test MAE
Several sample predictions (Actual vs Predicted)


🛠️ Requirements

Python 3.10 or higher
TensorFlow 2.x
OpenCV
Pandas
Scikit-learn
NumPy
Pillow

Install all at once:
pip install tensorflow opencv-python pandas scikit-learn numpy pillow



👤 Author
Thuan Le

Bachelor of Computer Science

University of New Brunswick (UNB)
