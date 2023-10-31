import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


class BaseModel:
    def __init__(self, cfg):
        self.model = None
        self.device = cfg.device
        self.cfg = cfg

    def iteration(self, data):
        pass

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def binarize(self, outputs, labels, thresh):
        pass

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def train_transform(self):
        img_size = self.cfg.dataset.img_size
        img_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Resize(img_size[0], img_size[1], p=1.0),
                A.Normalize(mean=(0.5), std=(0.5), p=1.0),
                ToTensorV2(),
            ],
            p=1,
        )
        return img_transform

    def test_transform(self):
        img_size = self.cfg.dataset.img_size
        img_transform = A.Compose(
            [
                A.Resize(img_size[0], img_size[1], p=1.0),
                A.Normalize(mean=(0.5), std=(0.5), p=1.0),
                ToTensorV2(),
            ],
            p=1,
        )
        return img_transform

    def get_transform_dicts(self):
        train_tr = self.train_transform()
        test_tr = self.test_transform()
        return A.to_dict(train_tr), A.to_dict(test_tr)
