🦠 MalariaDetection AI
AI-Powered Malaria Cell Detection using MobileNetV2 Transfer Learning and Streamlit
<p align="center">










</p>
🌍 Overview

MalariaVision AI is an intelligent deep learning application designed to automatically detect malaria-infected blood cells from microscopic blood smear images.

Built using MobileNetV2 Transfer Learning, the system provides fast and accurate predictions through an interactive Streamlit web application, enabling users to upload blood smear images and receive instant classification results with prediction confidence.

Designed with lightweight architecture and optimized inference, the project demonstrates how Artificial Intelligence can assist healthcare professionals in rapid malaria screening, particularly in resource-constrained environments.

🚀 Key Features

🦠 Binary Malaria Cell Classification

🧠 MobileNetV2 Transfer Learning

⚡ Real-Time Prediction

📊 Confidence Score Visualization

📈 Live Prediction Statistics

📝 Prediction History Tracking

🎨 Interactive Dark-Themed Streamlit Dashboard

🚀 Cached Model Loading for Faster Inference

📉 Training & Evaluation Visualization

☁️ Deployment Ready

🧠 AI Pipeline
Blood Smear Image
        │
        ▼
 Image Preprocessing
        │
        ▼
Resize & Normalize
        │
        ▼
 MobileNetV2 Backbone
        │
        ▼
 Feature Extraction
        │
        ▼
Dense Classification Head
        │
        ▼
 Binary Prediction
 ├── Parasitized
 └── Uninfected
        │
        ▼
Confidence Score
🏗️ Project Architecture
                  User Upload
                       │
                       ▼
          Microscopic Blood Cell Image
                       │
                       ▼
             Image Preprocessing
                       │
                       ▼
             MobileNetV2 CNN Model
                       │
             Feature Extraction
                       │
                       ▼
           Fully Connected Layers
                       │
                       ▼
        Binary Classification Output
              ┌───────────────┐
              ▼               ▼
      Parasitized      Uninfected
                       │
                       ▼
        Streamlit Dashboard Display
📂 Project Structure
MalariaVision-AI/
│
├── app.py                     # Streamlit web application
├── malaria_detection.py       # Model training pipeline
├── requirements.txt
├── runtime.txt
│
├── training_history.png       # Training accuracy & loss plots
├── evaluation_plots.png       # Confusion matrix & ROC analysis
│
└── README.md
📊 Dataset
NIH Malaria Cell Images Dataset

The model is trained using the NIH Malaria Cell Image Dataset, a benchmark dataset widely used for automated malaria diagnosis research.

Dataset Statistics
Total Images: 27,558
🦠 Parasitized Cells: 13,779
✅ Uninfected Cells: 13,779
Data Split
80% Training
20% Validation
🧬 Deep Learning Model
Component	Description
Model	MobileNetV2
Learning Method	Transfer Learning
Framework	TensorFlow / Keras
Classification	Binary
Input Size	128 × 128 RGB Images
Output	Parasitized / Uninfected
📈 Training Strategy

The model is trained using a two-stage transfer learning approach.

🔹 Phase 1 — Feature Extraction
Freeze MobileNetV2 backbone
Train custom classification layers
Adam Optimizer
Learning Rate: 0.001
🔹 Phase 2 — Fine-Tuning
Unfreeze the final MobileNetV2 layers
Lower Learning Rate (1e-5)
Early Stopping
ReduceLROnPlateau
Model Checkpointing

This strategy improves model generalization while preventing overfitting.

📈 Model Performance
Metric	Performance
Accuracy	94.3%
AUC-ROC	0.9846
Backbone	MobileNetV2
Classification	Binary
Deployment	Streamlit
⚙️ Tech Stack
Programming
Python
Deep Learning
TensorFlow
Keras
MobileNetV2
Machine Learning
Scikit-learn
Computer Vision
OpenCV
NumPy
Pillow
Visualization
Matplotlib
Deployment
Streamlit
🖥️ Installation

Clone the repository

git clone https://github.com/yourusername/MalariaVision-AI.git

Navigate to the project

cd MalariaVision-AI

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app.py
🚀 Usage
Launch the Streamlit application.
Upload a microscopic blood smear image.
The image is automatically preprocessed.
The AI model predicts whether the cell is infected.
View prediction confidence and classification results.
Monitor session statistics and prediction history.
🌟 Dashboard Highlights

✅ Blood Smear Image Upload

✅ AI-Based Malaria Detection

✅ Prediction Confidence Scores

✅ Session Statistics

✅ Prediction History

✅ Interactive Dark UI

✅ Fast Real-Time Inference

📊 Evaluation Metrics

The project evaluates model performance using:

Accuracy
Precision
Recall
ROC-AUC Score
Confusion Matrix
ROC Curve
Training & Validation Loss
Training & Validation Accuracy
🔮 Future Improvements
🔥 Grad-CAM Explainability
📱 Mobile Application
☁️ Cloud Deployment
🌍 Multi-Class Parasite Detection
📡 REST API Integration
🩺 Clinical Decision Support
⚡ Edge AI Optimization
🧬 Vision Transformer (ViT) Architecture
🤝 Contributing

Contributions are welcome!

Fork the repository
Create a feature branch
Commit your changes
Push the branch
Open a Pull Request
📜 Disclaimer

This project is intended for educational and research purposes only.

It is not a certified medical diagnostic system and should not replace professional clinical judgment.

👨‍💻 Authors

MalariaVision AI Team

Natalia Mathews
Limnisha Changkakati
Prema Malipatil
🌟 Support

If you found this project helpful,

⭐ Star the repository

🍴 Fork the project

📢 Share it with others

Together, let's build AI solutions that make healthcare more accessible and impactful. 🦠🚀
