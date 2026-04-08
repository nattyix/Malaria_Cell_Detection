# Malaria_Cell_Detection
Automated malaria detection using MobileNetV2 to classify blood cell images, deployed as a real-time web app over a 5G telemedicine network for rapid and accessible diagnosis.
# Automated Malaria Detection System

## Overview

This project presents an AI-based malaria detection system using a lightweight deep learning model (MobileNetV2) to classify blood smear images as Parasitized or Uninfected.

The system is deployed as a real-time web application using a Python backend, enabling fast and reliable diagnosis suitable for resource-limited environments and telemedicine use cases.

---

## Objectives

* Automate malaria detection using deep learning
* Provide rapid and accurate diagnosis
* Enable deployment in low-resource healthcare settings
* Support real-time inference through a web interface

---

## Model Details

* Model: MobileNetV2 (Transfer Learning)
* Framework: TensorFlow / Keras
* Key advantages:

  * Lightweight (~2.4M parameters)
  * Fast inference on CPU
  * High accuracy with low computational cost

---

## Dataset

* Source: NIH Malaria Cell Image Dataset
* Total Images: 27,558

  * 13,779 Parasitized
  * 13,779 Uninfected
* Split:

  * 80% Training
  * 20% Validation

Each image represents a single red blood cell under a microscope.

---

## Preprocessing Pipeline

* Resize images to 128×128
* Normalize pixel values (0–255 → 0–1)
* Data augmentation (training only):

  * Rotation
  * Flips
  * Zoom

---

## Model Architecture

* MobileNetV2 base (pretrained on ImageNet, initially frozen)
* Global Average Pooling
* Batch Normalization
* Dense layers with Dropout
* Sigmoid output layer for binary classification

---

## Training Strategy

### Phase 1: Frozen Base

* Train only classification head
* Learning rate: 0.001
* Achieves ~90% validation accuracy

### Phase 2: Fine-Tuning

* Unfreeze top layers of MobileNetV2
* Learning rate: 1e-5
* Improves performance further
* Early stopping applied

---

## Results

* Accuracy: 94.3%
* AUC-ROC: 0.9846

The model is slightly conservative, prioritizing detection of infected cells, which is safer for screening scenarios.

---

## Web Application

### Backend

* Python (Flask via `app.py`)

### API Endpoints

* POST /predict — Upload image and get prediction
* GET /history — Retrieve prediction history
* GET /stats — System statistics

### Features

* Image upload and preview
* Real-time prediction
* Simple and user-friendly interface

---

## Workflow

1. User uploads a blood smear image
2. Image is sent to backend API
3. Model performs inference
4. Prediction is returned
5. Result is displayed

---

## Performance

* Inference time: ~225–272 ms
* End-to-end latency: ~0.5s – 1.5s

---

## Key Highlights

* Lightweight and efficient model
* Real-time deployment (not just experimental)
* Full dataset utilization
* Two-phase fine-tuning approach
* Suitable for telemedicine applications

---

## Future Work

* Grad-CAM for model explainability
* Multi-class malaria stage detection
* Mobile/edge deployment
* Broader dataset generalization

---

## Contributors

* Natalia Mathews
* [Your Friend's Name]

---

## License

This project is intended for educational and research purposes.

---

## Support

If you find this project useful, consider giving it a star on GitHub.
