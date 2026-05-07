import os
import cv2
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset

class SODDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*")))
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        image = image / 255.0
        mask = mask / 255.0

        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask