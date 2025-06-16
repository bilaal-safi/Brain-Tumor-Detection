
# Brain Tumor Classification using CNN and Flask

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify MRI images as either "Brain Tumor" or "Healthy". It includes a web-based interface built with Flask to upload medical images and get real-time classification results.

# Data Source
Dataset is publically available on Kaggle: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset/data


# Features 
- Preprocessing: Image resizing, normalization, and augmentation during training.
- CNN Model: 4 convolutional layers + 4 Max Pooling layers + 2 fully connected layers with LeakyReLU activations.
- Visualization: Loss and accuracy tracked and plotted during training.
- Accuracy: Achieves up to 96.9% accuracy on validation set after 10 epochs.

- Flask Web Interface: Upload and classify brain MRI images through a web form.

# CNN Architecture

Input → Conv1 (16) → MaxPool → Conv2 (32) → MaxPool → Conv3 (64) → MaxPool → Conv4 (128) → MaxPool  → Flatten → Fully Connected Layer 1 (1024) → Fully Connected Layer (2) → Output (0: Tumor, 1: Healthy)

# Training Highlights
- Dataset: Brain MRI images with ~4600 examples

- Image Size: Resized to 128x128

- Data Augmentation: Flip, rotation, normalization

#  How to Run the Flask App
- Install dependencies: pip install flask torch torchvision pillow
- Run the app: python app.py
- Visit in browser: http://127.0.0.1:5000/
- Upload an MRI image and see the classification result instantly!


