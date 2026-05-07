# Salient Object Detection Using CNN

## Final Project Report

---

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

This project uses the DUTS saliency detection dataset. DUTS is a large-scale benchmark dataset for salient object detection containing RGB images and corresponding binary saliency masks.

The dataset contains:
- Training images
- Ground truth saliency masks
- Pixel-level annotations

The training dataset used in this project contains 10,553 image-mask pairs.

Dataset structure:

dataset/DUTS-TR/DUTS-TR-Image  
dataset/DUTS-TR/DUTS-TR-Mask

The dataset was stored locally and excluded from GitHub using a `.gitignore` configuration due to large file sizes.

Images and masks were resized to 256x256 resolution during preprocessing. Pixel values were normalized to the range [0,1] before being converted into PyTorch tensors.

---

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

The final model uses a pretrained U-Net architecture with a ResNet34 encoder.

The encoder extracts hierarchical image features while the decoder reconstructs spatial information to generate high-quality segmentation masks.

The model contains:
- Convolutional layers
- Batch Normalization
- ReLU activation functions
- Skip connections
- Upsampling layers
- Sigmoid output activation

Skip connections help preserve fine spatial information and improve object boundary reconstruction.

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
- Epochs: 30
- Optimizer: Adam
- Learning rate: 0.0001
- Loss function: BCE + Dice Loss

The Adam optimizer was used because of its fast convergence and stability during deep learning optimization.

A combined Binary Cross Entropy and Dice Loss function was used to improve segmentation overlap quality and object boundary accuracy.

During training:
1. Images were passed through the neural network
2. Predicted masks were generated
3. Loss was computed against ground truth masks
4. Backpropagation updated model weights
5. Validation loss was calculated after each epoch

The best model checkpoint was automatically saved based on validation loss performance.

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

---

# 6. Results and Discussion

## Baseline CNN Results

- Mean IoU: 0.3891
- Mean Precision: 0.6786
- Mean Recall: 0.5000
- Mean F1 Score: 0.5235

## Improved U-Net Results

- Mean IoU: 0.9186
- Mean Precision: 0.9634
- Mean Recall: 0.9515
- Mean F1 Score: 0.9535

The improved U-Net model with a pretrained ResNet34 encoder significantly outperformed the baseline CNN architecture.

The addition of:
- pretrained feature extraction
- skip connections
- Batch Normalization
- Dice Loss optimization

greatly improved segmentation quality and object boundary reconstruction.

The improved model produced cleaner saliency masks, reduced noisy predictions, and achieved substantially higher IoU and F1-score values.

Several prediction examples demonstrated:
- Accurate foreground localization
- Clean object boundaries
- High-quality segmentation masks
- Strong generalization on validation images

The final U-Net architecture achieved strong segmentation performance and demonstrated the effectiveness of pretrained encoder-decoder architectures for salient object detection tasks.

---

# 7. Challenges Encountered

Several technical challenges were encountered during the development process.

## 7.1 GPU Configuration

Initially, the model was trained on CPU instead of GPU because the incorrect Python environment was being used. The issue was resolved by activating the correct Conda environment containing the CUDA-enabled PyTorch installation.

## 7.2 Dataset Management

Large dataset files exceeded GitHub upload limits. To solve this problem, datasets and model files were excluded from version control using a `.gitignore` configuration.

## 7.3 Model Optimization

The initial baseline CNN produced noisy segmentation masks and lower IoU scores. The problem was solved by replacing the baseline architecture with a pretrained U-Net model using a ResNet34 encoder and Dice-enhanced loss optimization.

---

# 8. Future Improvements

Several additional improvements could further improve segmentation performance:

- Attention-based segmentation architectures
- Transformer-based models
- Larger saliency datasets
- Multi-scale feature fusion
- Real-time optimization
- Advanced data augmentation
- Ensemble segmentation models

Several improvements were successfully implemented in the final version of the project using a pretrained U-Net architecture with a ResNet34 encoder and Dice-enhanced loss optimization.

---

# 9. Conclusion

This project successfully implemented a complete salient object detection system using deep learning and convolutional neural networks.

The final system includes:
- Dataset preprocessing
- Custom PyTorch data loading
- GPU-based model training
- U-Net segmentation architecture
- Evaluation metrics
- Prediction visualization
- Improved segmentation optimization

The final improved model achieved strong segmentation performance with:
- Mean IoU: 0.9186
- Mean Precision: 0.9634
- Mean Recall: 0.9515
- Mean F1 Score: 0.9535

The project demonstrates practical understanding of:
- Deep learning
- Semantic segmentation
- GPU acceleration
- Computer vision pipelines
- PyTorch implementation
- Model optimization techniques

Overall, the project provides a strong implementation of salient object detection using modern deep learning segmentation techniques and establishes a foundation for future research and improvement.