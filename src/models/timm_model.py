import logging

import numpy as np
import torch

from .base_model import BaseModel
from .fr_model_timm import FaceTimmModel


class TimmModel(BaseModel):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = FaceTimmModel(cfg)
        self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model)
        self.criterion = torch.nn.CrossEntropyLoss()

        optimizer = getattr(torch.optim, cfg.strat.optimizer)
        self.optimizer = optimizer(self.model.parameters(), **cfg.strat.opt_params)
        scheduler = getattr(torch.optim.lr_scheduler, self.cfg.strat.scheduler)
        self.scheduler = scheduler(self.optimizer, **self.cfg.strat.sch_params)

    def iteration(self, data, train=False):
        if train:
            img, labels = data["img"], data["target"]
            inputs = img.to(self.device, dtype=torch.float)
            labels = labels.to(self.device, dtype=torch.long)

            outputs = self.model(inputs, labels, train)
            loss = self.criterion(outputs, labels)
            return loss

        img1, img2, labels = data["img1"], data["img2"], data["target"]
        inp1 = img1.to(self.device, dtype=torch.float)
        inp2 = img2.to(self.device, dtype=torch.float)
        embeddings1 = self.model(inp1, None, train).detach().cpu().numpy()
        embeddings2 = self.model(inp2, None, train).detach().cpu().numpy()
        return embeddings1, embeddings2, labels.numpy()
