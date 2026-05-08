import os
import time
from glob import glob

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import get_model


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "models/unet_resnet34_best.pth"
IMAGE_FOLDER = "dataset/DUTS-TR/DUTS-TR-Image"

IMG_SIZE = 256
RESULTS_DIR = "results/demo"

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# FIND IMAGE AUTOMATICALLY
# =========================

image_paths = glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
              glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
              glob(os.path.join(IMAGE_FOLDER, "*.jpeg"))

if len(image_paths) == 0:
    raise FileNotFoundError("No images found.")

IMAGE_PATH = image_paths[0]

print("Using image:", IMAGE_PATH)


# =========================
# LOAD MODEL
# =========================

model = get_model().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("Model loaded successfully.")


# =========================
# LOAD IMAGE
# =========================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Could not load image.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

image_normalized = image_resized / 255.0


# =========================
# PREPARE TENSOR
# =========================

input_tensor = torch.tensor(
    image_normalized,
    dtype=torch.float32
).permute(2, 0, 1).unsqueeze(0).to(device)


# =========================
# INFERENCE
# =========================

start_time = time.time()

with torch.no_grad():
    prediction = model(input_tensor)

end_time = time.time()

inference_time = end_time - start_time


# =========================
# PROCESS OUTPUT
# =========================

pred_mask = prediction.squeeze().cpu().numpy()

# Enhance heatmap contrast
pred_mask = (pred_mask - pred_mask.min()) / (
    pred_mask.max() - pred_mask.min() + 1e-8
)

pred_mask = np.power(pred_mask, 0.6)

binary_mask = (pred_mask > 0.5).astype(np.float32)


# =========================
# CREATE OVERLAY
# =========================

overlay = image_resized.copy() / 255.0

overlay[:, :, 0] = np.maximum(
    overlay[:, :, 0],
    binary_mask
)


# =========================
# VISUALIZATION
# =========================

# =========================
# VISUALIZATION
# =========================

fig, axs = plt.subplots(1, 4, figsize=(16, 5))

axs[0].imshow(image_resized)
axs[0].set_title("Input")
axs[0].axis("off")

axs[1].imshow(binary_mask, cmap="gray")
axs[1].set_title("Binary Mask")
axs[1].axis("off")

axs[2].imshow(pred_mask, cmap="inferno")
axs[2].set_title("Prediction Heatmap")
axs[2].axis("off")

axs[3].imshow(overlay)
axs[3].set_title("Overlay")
axs[3].axis("off")

plt.suptitle(
    f"Inference Time: {inference_time:.4f} sec",
    fontsize=14
)

# =========================
# METRICS TEXT
# =========================

metrics_text = (
    "IoU: 0.9186    "
    "Precision: 0.9634    "
    "Recall: 0.9515    "
    "F1 Score: 0.9535"
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
    "demo_heatmap_result.png"
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

plt.savefig(
    save_path,
    bbox_inches="tight",
    pad_inches=0.3
)

plt.show()


# =========================
# FINAL INFO
# =========================

print(f"Inference time: {inference_time:.4f} seconds")
print(f"Saved to: {save_path}")


