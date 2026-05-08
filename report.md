# Salient Object Detection Using Deep Learning

# Final Project Report

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

* Distinguishing foreground objects from complex backgrounds
* Preserving object boundaries
* Handling different object scales and shapes
* Reducing noisy predictions
* Producing accurate pixel-level segmentation masks

The project focuses on building a complete deep learning pipeline including dataset preparation, model architecture, GPU training, evaluation metrics, and prediction visualization.

---

# 3. Dataset

This project uses the DUTS saliency detection dataset. DUTS is a large-scale benchmark dataset for salient object detection containing RGB images and corresponding binary saliency masks.

The training dataset used in this project contains 10,553 image-mask pairs.

Images and masks were resized to 256x256 resolution during preprocessing. Pixel values were normalized to the range [0,1] before being converted into PyTorch tensors.

---

# 4. Methodology

The project follows a deep learning-based semantic segmentation pipeline for salient object detection.

The system was implemented using Python, PyTorch, OpenCV, NumPy, and Matplotlib.

## 4.1 Data Loading and Preprocessing

A custom PyTorch Dataset class was implemented to load RGB images and corresponding saliency masks.

The preprocessing pipeline performs the following operations:

* Reads RGB input images
* Reads grayscale binary masks
* Resizes images and masks to 256x256
* Normalizes pixel values to the range [0,1]
* Converts data into PyTorch tensors

The dataset was split into:

* 80% training data
* 20% validation data

## 4.2 Model Architecture

The final model uses a pretrained U-Net architecture with a ResNet34 encoder.

The model contains:

* Convolutional layers
* Batch Normalization
* ReLU activation functions
* Skip connections
* Upsampling layers
* Sigmoid output activation

## 4.3 GPU Training

The model was trained using GPU acceleration on an NVIDIA RTX 4070 Laptop GPU with CUDA support enabled through PyTorch.

Training configuration:

| Parameter     | Value           |
| ------------- | --------------- |
| Image Size    | 256x256         |
| Batch Size    | 8               |
| Epochs        | 30              |
| Optimizer     | Adam            |
| Learning Rate | 0.0001          |
| Loss Function | BCE + Dice Loss |

---

# 5. Evaluation Metrics

The trained model was evaluated using:

* Intersection over Union (IoU)
* Precision
* Recall
* F1 Score

---

# 6. Results and Discussion

## Final Model Results (U-Net + ResNet34)

| Metric         | Score  |
| -------------- | ------ |
| Mean IoU       | 0.9186 |
| Mean Precision | 0.9634 |
| Mean Recall    | 0.9515 |
| Mean F1 Score  | 0.9535 |

Inference Time:

* ~0.13 seconds per image

The improved U-Net architecture with a pretrained ResNet34 encoder achieved strong segmentation performance and significantly outperformed the baseline CNN model.

## Baseline CNN Comparison

| Metric         | Score  |
| -------------- | ------ |
| Mean IoU       | 0.3891 |
| Mean Precision | 0.6786 |
| Mean Recall    | 0.5000 |
| Mean F1 Score  | 0.5235 |

---

# 7. Prediction Visualization

The following examples demonstrate the segmentation performance of the final model on validation images.

![Prediction Example](results/predictions_unet/prediction_1.png)

The visualizations include:

* Input RGB image
* Ground truth saliency mask
* Prediction heatmap
* Overlay visualization

---

# 8. Challenges Encountered

Several technical challenges were encountered during the development process.

* GPU environment configuration
* Dataset size limitations
* Baseline model optimization

The baseline CNN was replaced with a pretrained U-Net architecture using a ResNet34 encoder and Dice-enhanced loss optimization.

---

# 9. Future Improvements

Several additional improvements could further improve segmentation performance:

* Attention-based segmentation architectures
* Transformer-based models
* Larger saliency datasets
* Multi-scale feature fusion
* Real-time optimization
* Advanced data augmentation

---

# 10. Conclusion

This project successfully implemented a complete salient object detection system using deep learning and convolutional neural networks.

The final improved model achieved strong segmentation performance with:

| Metric         | Score  |
| -------------- | ------ |
| Mean IoU       | 0.9186 |
| Mean Precision | 0.9634 |
| Mean Recall    | 0.9515 |
| Mean F1 Score  | 0.9535 |

The project demonstrates practical understanding of:

* Deep learning
* Semantic segmentation
* GPU acceleration
* Computer vision pipelines
* PyT
