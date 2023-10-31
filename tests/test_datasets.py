import os

import pytest
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.datasets import TrainDataset, TestDataset


ROOT = "data/casia_faces"
ROOT_TEST = "data/lfw"
IMG_SIZE = 32


@pytest.fixture
def transform():
    img_transform = A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE, p=1.0),
                A.Normalize(mean=(0.5), std=(0.5), p=1),
                ToTensorV2(),
            ],
            p=1,
        )
    return img_transform


def test_train_dataset(transform):
    ds = TrainDataset(ROOT, transform)
    item = ds[0]
    img, label = item["img"], item["target"]
    classes = ds.classes
    assert len(ds) == 490623
    assert img.shape == (1, IMG_SIZE, IMG_SIZE), (f"Image shape is {img.shape}")
    assert isinstance(label, int), ("Label type must be int")
    assert len(classes) == len(next(os.walk(ROOT))[1]), ("Inconsistent number of classes")


def test_test_dataset(transform):
    ds = TestDataset(ROOT_TEST, transform)
    item = ds[0]
    targets = ds.targets
    img1, img2, label = item["img1"], item["img2"], item["target"]
    assert img1.shape == (1, IMG_SIZE, IMG_SIZE), (f"Image shape is {img1.shape}")
    assert img2.shape == (1, IMG_SIZE, IMG_SIZE), (f"Image shape is {img2.shape}")
    assert isinstance(label, int), ("Label type must be int")
    assert len(ds) == 6000
    assert targets.count(0) == 3000
    assert targets.count(1) == 3000
