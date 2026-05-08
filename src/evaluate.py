import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from data_loader import SODDataset
from model import get_model


# =========================
# CONFIGURATION
# =========================

IMAGE_DIR = "dataset/DUTS-TR/DUTS-TR-Image"
MASK_DIR = "dataset/DUTS-TR/DUTS-TR-Mask"

IMG_SIZE = 256
BATCH_SIZE = 8

MODEL_PATH = "models/unet_resnet34_best.pth"
RESULTS_DIR = "results/predictions_unet"

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# DEVICE SETUP
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# METRIC FUNCTION
# =========================

def calculate_metrics(pred, mask, threshold=0.5):
    pred_binary = (pred > threshold).float()
    mask_binary = (mask > threshold).float()

    intersection = (pred_binary * mask_binary).sum()
    union = pred_binary.sum() + mask_binary.sum() - intersection

    iou = (intersection + 1e-7) / (union + 1e-7)

    precision = (intersection + 1e-7) / (pred_binary.sum() + 1e-7)
    recall = (intersection + 1e-7) / (mask_binary.sum() + 1e-7)

    f1 = (2 * precision * recall + 1e-7) / (
        precision + recall + 1e-7
    )

    return iou.item(), precision.item(), recall.item(), f1.item()


# =========================
# LOAD DATASET
# =========================

dataset = SODDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    img_size=IMG_SIZE
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Validation samples:", len(val_dataset))


# =========================
# LOAD MODEL
# =========================

model = get_model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully.")


# =========================
# EVALUATION
# =========================

ious = []
precisions = []
recalls = []
f1_scores = []

saved_images = 0

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        for i in range(images.size(0)):
            pred = outputs[i]
            mask = masks[i]

            iou, precision, recall, f1 = calculate_metrics(pred, mask)

            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            # Save first 10 visual examples
            if saved_images < 10:
                image_np = images[i].cpu().permute(1, 2, 0).numpy()
                mask_np = mask.cpu().squeeze().numpy()
                pred_np = pred.cpu().squeeze().numpy()

                # Binary mask for overlay only
                pred_binary = (pred_np > 0.5).astype(np.float32)

                # Enhance heatmap contrast
                heatmap = (pred_np - pred_np.min()) / (
                    pred_np.max() - pred_np.min() + 1e-8
                )
                heatmap = np.power(heatmap, 0.6)

                # Overlay
                overlay = image_np.copy()
                overlay[:, :, 0] = np.maximum(
                    overlay[:, :, 0],
                    pred_binary
                )

                fig, axs = plt.subplots(1, 4, figsize=(16, 5))

                axs[0].imshow(image_np)
                axs[0].set_title("Input Image")
                axs[0].axis("off")

                axs[1].imshow(mask_np, cmap="gray")
                axs[1].set_title("Ground Truth")
                axs[1].axis("off")

                axs[2].imshow(heatmap, cmap="inferno")
                axs[2].set_title("Prediction Heatmap")
                axs[2].axis("off")

                axs[3].imshow(overlay)
                axs[3].set_title("Overlay")
                axs[3].axis("off")

                metrics_text = (
                    f"IoU: {iou:.4f}    "
                    f"Precision: {precision:.4f}    "
                    f"Recall: {recall:.4f}    "
                    f"F1 Score: {f1:.4f}"
                )

                plt.figtext(
                    0.5,
                    0.02,
                    metrics_text,
                    ha="center",
                    fontsize=12,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.9,
                        "pad": 8
                    }
                )

                save_path = os.path.join(
                    RESULTS_DIR,
                    f"prediction_{saved_images + 1}.png"
                )

                plt.tight_layout(rect=[0, 0.08, 1, 1])

                plt.savefig(
                    save_path,
                    bbox_inches="tight",
                    pad_inches=0.3
                )

                plt.close()

                saved_images += 1


# =========================
# FINAL RESULTS
# =========================

print("\nEvaluation Results")
print("------------------")
print(f"Mean IoU:       {np.mean(ious):.4f}")
print(f"Mean Precision: {np.mean(precisions):.4f}")
print(f"Mean Recall:    {np.mean(recalls):.4f}")
print(f"Mean F1 Score:  {np.mean(f1_scores):.4f}")

print(f"\nSaved prediction examples to: {RESULTS_DIR}")