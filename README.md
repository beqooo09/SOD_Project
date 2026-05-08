## Dataset

This project uses the DUTS saliency detection dataset. The dataset contains RGB images and corresponding binary saliency masks.

The dataset is not uploaded to GitHub because of file size limits. It should be placed locally as:

dataset/DUTS-TR/DUTS-TR-Image
dataset/DUTS-TR/DUTS-TR-Mask

## Training

The model was trained locally using an NVIDIA RTX 4070 Laptop GPU.

Training configuration:
- Image size: 256x256
- Batch size: 8
- Epochs: 50
- Optimizer: Adam
- Loss function: Binary Cross Entropy

## Results

Baseline CNN evaluation results:

Improved U-Net + ResNet34
IoU: 0.9186
Precision: 0.9634
Recall: 0.9515
F1: 0.9535

The model successfully detects salient regions, but some predictions contain noisy boundaries and fragmented masks. This is expected for a simple baseline encoder-decoder CNN.

## Project Structure

src/data_loader.py - Loads images and masks for training  
src/model.py - Defines the CNN encoder-decoder model  
src/train.py - Trains the model on GPU  
src/evaluate.py - Evaluates the model and saves prediction visualizations  
results/predictions - Contains sample prediction outputs  
models - Stores trained model weights locally  

## Future Improvements

- Add Batch Normalization
- Add Dropout
- Use U-Net style skip connections
- Try a pretrained encoder
- Improve data augmentation
