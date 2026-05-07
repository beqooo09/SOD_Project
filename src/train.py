import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_loader import SODDataset
from model import SimpleSODNet


# =========================
# 1. CONFIGURATION
# =========================
# Here we define the main settings for training.

IMAGE_DIR = "dataset/DUTS-TR/DUTS-TR-Image"
MASK_DIR = "dataset/DUTS-TR/DUTS-TR-Mask"

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = "models/sod_model.pth"


# =========================
# 2. DEVICE SETUP
# =========================
# If GPU is available, we use it. Otherwise we use CPU.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 3. LOAD DATASET
# =========================
# The dataset loads image-mask pairs.
# Image = input
# Mask = correct answer

dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    img_size=IMG_SIZE
)

print("Total samples:", len(dataset))


# =========================
# 4. TRAIN / VALIDATION SPLIT
# =========================
# We split data into training and validation.
# Training data teaches the model.
# Validation data checks how well the model performs on unseen data.

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# =========================
# 5. MODEL, LOSS, OPTIMIZER
# =========================
# Model: CNN encoder-decoder
# Loss: Binary Cross Entropy because mask pixels are 0 or 1
# Optimizer: Adam updates model weights

model = SimpleSODNet().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =========================
# 6. TRAINING LOOP
# =========================
# For each epoch:
# - model sees images
# - predicts masks
# - compares predictions with real masks
# - updates weights to reduce error

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)


    # =========================
    # 7. VALIDATION LOOP
    # =========================
    # Validation checks performance without updating model weights.

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )


# =========================
# 8. SAVE MODEL
# =========================
# After training, we save the model weights.

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("Model saved to:", MODEL_SAVE_PATH)