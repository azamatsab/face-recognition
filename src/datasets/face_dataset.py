import os
from glob import glob

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def transform_image(image, transform):
    image = transform(image=image)["image"]
    return image


class TrainDataset:
    def __init__(self, root, transform):
        self.ds = ImageFolder(root, None)
        self.transform = transform
        self.classes = sorted(list(set(self.ds.targets)))

    def __len__(self):
        # return 100
        return len(self.ds)

    def __getitem__(self, idx):
        image, target = self.ds[idx]
        image = np.array(image.convert("L"))

        if self.transform is not None:
            image = transform_image(image, self.transform)
        return {"img": image, "target": target}


class TestDataset:

    TARGET_TRANSFORM = {"negative": 0, "positive": 1}
    
    def __init__(self, root, transform):
        self.transform = transform
        self.img_pairs, self.targets = self.parse_folders__(root)
    
    def parse_folders__(self, root):
        labels = glob(os.path.join(root, "*"))
        img_pairs, targets = [], []
        for label in labels:
            folders = glob(os.path.join(label, "*"))
            imgs = [glob(os.path.join(folder, "*")) for folder in folders]
            img_pairs += imgs
            target = self.TARGET_TRANSFORM[os.path.split(label)[1]]
            targets += [target] * len(imgs)

        return img_pairs, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        path1, path2 = self.img_pairs[idx]
        target = self.targets[idx]
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            img1 = transform_image(img1, self.transform)
            img2 = transform_image(img2, self.transform)
        
        return {"img1": img1, "img2": img2, "target": target}
        
