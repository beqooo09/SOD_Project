# Salient Object Detection Using CNN

## Final Project Report
# 1. Introduction

Salient Object Detection (SOD) is a computer vision task that aims to automatically identify the most visually important object or region in an image. The goal of SOD is to generate a saliency map or binary mask that highlights the dominant foreground object while suppressing background information.

Salient object detection is widely used in many real-world applications such as image segmentation, autonomous systems, robotics, object tracking, image editing, medical imaging, and visual attention modeling.

In recent years, deep learning and convolutional neural networks (CNNs) have significantly improved the performance of salient object detection systems. Encoder-decoder architectures are commonly used because they can learn hierarchical image features and generate pixel-wise segmentation masks.

This project implements a deep learning-based salient object detection system using PyTorch. A custom encoder-decoder convolutional neural network was trained on the DUTS saliency detection dataset using GPU acceleration. The model predicts binary saliency masks from RGB images and evaluates performance using multiple segmentation metrics.

---

# 2. Problem Statement

The objective of this project is to design and implement a convolutional neural network capable of detecting salient objects in images. Given an RGB image as input, the model should generate a corresponding binary saliency mask that identifies the most visually important object in the scene.

The main challenges of salient object detection include:

- Distinguishing foreground objects from complex backgrounds
- Preserving object boundaries
- Handling different object scales and shapes
- Reducing noisy predictions
- Producing accurate pixel-level segmentation masks

The project focuses on building a complete deep learning pipeline including dataset preparation, model architecture, GPU training, evaluation metrics, and prediction visualization.

---

# 3. Dataset

# 4. Methodology

The project follows a deep learning-based semantic segmentation pipeline for salient object detection. The overall workflow consists of the following stages:

1. Dataset loading and preprocessing
2. CNN model construction
3. GPU-based model training
4. Validation and evaluation
5. Prediction visualization

The system was implemented using Python, PyTorch, OpenCV, NumPy, and Matplotlib.

---

## 4.1 Data Loading and Preprocessing

A custom PyTorch Dataset class was implemented to load RGB images and corresponding saliency masks.

The preprocessing pipeline performs the following operations:

- Reads RGB input images
- Reads grayscale binary masks
- Resizes images and masks to 256x256
- Normalizes pixel values to the range [0,1]
- Converts data into PyTorch tensors

The dataset was split into:
- 80% training data
- 20% validation data

PyTorch DataLoader objects were used to efficiently load mini-batches during training.

---

## 4.2 Model Architecture

The implemented model uses an encoder-decoder convolutional neural network architecture.

The encoder extracts hierarchical image features using convolutional layers and max pooling operations. The decoder reconstructs spatial information and generates pixel-level saliency masks.

The encoder contains:
- Convolutional layers
- ReLU activation functions
- Max pooling layers

The decoder contains:
- Transposed convolution layers
- Upsampling operations
- Final sigmoid activation layer

The sigmoid activation function converts the output into probability values between 0 and 1, where:
- 0 represents background
- 1 represents salient foreground regions

The final output is a single-channel saliency mask.

---

## 4.3 GPU Training

The model was trained using GPU acceleration on an NVIDIA RTX 4070 Laptop GPU with CUDA support enabled through PyTorch.

Training configuration:
- Image size: 256x256
- Batch size: 8
- Epochs: 50
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Binary Cross Entropy Loss (BCELoss)

The Adam optimizer was used because of its fast convergence and stability during deep learning optimization.

Binary Cross Entropy loss was selected because salient object detection is treated as a binary segmentation problem.

During training:
1. Images were passed through the CNN
2. Predicted masks were generated
3. Loss was computed against ground truth masks
4. Backpropagation updated network weights
5. Validation loss was calculated after each epoch

GPU acceleration significantly reduced training time and improved computational efficiency.

---

# 5. Evaluation Metrics

The trained model was evaluated using several segmentation metrics commonly used in computer vision tasks.

## 5.1 Intersection over Union (IoU)

Intersection over Union measures the overlap between the predicted mask and the ground truth mask.

Higher IoU values indicate better segmentation performance.

## 5.2 Precision

Precision measures how many predicted salient pixels are actually correct.

High precision indicates fewer false positive predictions.

## 5.3 Recall

Recall measures how many ground truth salient pixels were successfully detected by the model.

High recall indicates fewer missed object regions.

## 5.4 F1 Score

The F1 Score combines precision and recall into a single balanced metric.

It provides an overall measurement of segmentation quality.