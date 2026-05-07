import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_loader import SODDataset
from model import get_model


IMAGE_DIR = "dataset/DUTS-TR/DUTS-TR-Image"
MASK_DIR = "dataset/DUTS-TR/DUTS-TR-Mask"

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001

MODEL_SAVE_PATH = "models/unet_resnet34_best.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    img_size=IMG_SIZE
)

print("Total samples:", len(dataset))

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


model = get_model().to(device)

bce_loss = nn.BCELoss()


def dice_loss(pred, target, smooth=1e-7):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

    return 1 - dice


def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)


criterion = combined_loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")
os.makedirs("models", exist_ok=True)


for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

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
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model saved with Val Loss: {best_val_loss:.4f}")


print("Training finished.")
print("Best model saved to:", MODEL_SAVE_PATH)