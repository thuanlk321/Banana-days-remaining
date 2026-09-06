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
