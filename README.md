# MindTrack: Real-Time Mental Health Mood Detection and Alert System

![Project Banner](https://via.placeholder.com/1200x300.png?text=MindTrack%20-%20AI%20Mood%20Detection)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/YOUR_USERNAME/MindTrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MindTrack is an innovative proof-of-concept application designed to provide real-time mood detection using multi-modal inputs (facial expressions and vocal tone). It leverages deep learning and machine learning models to analyze video and audio streams, identify emotional states, and trigger alerts for potential mental health distress. The system is served through a user-friendly web interface built with Flask.

## 📜 Table of Contents
	
- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Installation and Setup](#-installation-and-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#️-important-disclaimer)

## 🌟 Overview

Mental well-being is crucial, and early detection of negative emotional patterns can be a key step towards seeking help. MindTrack aims to be a non-invasive tool that can passively monitor a user's emotional state during computer interaction. By analyzing facial cues via a webcam and vocal characteristics through a microphone, it classifies moods like 'Happy', 'Sad', 'Neutral', 'Angry', and 'Anxious'. If a pattern of negative emotions is detected over a period, the system can generate a configurable alert, encouraging the user to take a break or reach out for support.

## ✨ Features

- **Real-Time Facial Expression Recognition (FER):** Uses OpenCV to capture video frames and a TensorFlow/Keras CNN model to classify emotions from facial landmarks.
- **Vocal Tone Emotion Analysis:** Leverages Librosa to extract audio features (like MFCCs) and a pre-trained Scikit-learn model to detect emotion in speech.
- **Multi-Modal Fusion (Optional):** Combines predictions from both video and audio for a more robust and accurate mood assessment.
- **Web-Based Interface:** A clean and simple UI built with Flask that displays the real-time camera feed, detected emotion, and confidence score.
- **Configurable Alert System:** Triggers on-screen notifications if persistent negative emotions are detected.
- **Lightweight and Modular:** Built with a clean architecture, making it easy to extend or modify.

🧠 Models
Facial Emotion Recognition
Architecture: Convolutional Neural Network (CNN) built with TensorFlow/Keras.
Training Data: The model was trained on the FER-2013 dataset, which contains 48x48 pixel grayscale images of faces.
Classes: Angry, Happy, Sad, Neutral, Surprise.
Vocal Emotion Recognition
Architecture: A Support Vector Machine (SVM) from Scikit-learn.
Features: Mel-Frequency Cepstral Coefficients (MFCCs), Chroma, and Mel-spectrogram features extracted using Librosa.
Training Data: The model was trained on the RAVDESS audio dataset.
Classes: Angry, Happy, Sad, Neutral, Calm.
🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.
Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
This project is distributed under the MIT License. See LICENSE file for more information.
⚠️ Important Disclaimer
This is not a medical device.
MindTrack is an experimental project and a proof-of-concept. It is not intended for medical, diagnostic, or therapeutic purposes. The mood detection is based on generalized models and may not be accurate for every individual or situation. This tool should not be used as a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing mental health concerns, please consult a qualified healthcare professional.

## ⚙️ How It Works

1.  **Data Capture:** The Flask web application accesses the user's webcam and microphone through the browser.
2.  **Video Stream Processing:**
    -   OpenCV captures video frames in real-time.
    -   A Haar Cascade or MTCNN model detects faces within each frame.
    -   The detected face region is pre-processed (resized, grayscaled, normalized).
    -   The processed face image is fed into a trained TensorFlow CNN model for emotion prediction.
3.  **Audio Stream Processing:**
    -   Librosa and SoundFile capture and process audio chunks.
    -   Features like MFCC, Chroma, and Mel-spectrogram are extracted.
    -   The feature vector is passed to a trained Scikit-learn classifier (e.g., SVM, RandomForest) loaded via Joblib.
4.  **Inference and Display:**
    -   The model's predictions (e.g., "Sad: 85%") are overlaid on the video feed.
    -   The results are sent back to the Flask front-end and displayed to the user.
5.  **Alert Logic:** The backend keeps a short history of detected emotions. If a high frequency of negative emotions crosses a predefined threshold, an alert is triggered.

## 🛠️ Tech Stack

- **Backend:** Flask
- **Machine Learning / Deep Learning:** TensorFlow, Scikit-learn
- **Data Manipulation:** NumPy, Pandas
- **Computer Vision:** OpenCV-Python
- **Audio Processing:** Librosa, SoundFile
- **Model Persistence:** Joblib

## 🏗️ System Architecture



## 🚀 Installation and Setup

Follow these steps to get MindTrack running on your local machine.

### Prerequisites

-   Python 3.8 or higher
-   `pip` and `venv`
-   Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MindTrack.git
cd MindTrack
