import os
import warnings
import logging

import hydra
import mlflow

from src.trainer import Trainer
from src.datasets import get_loaders
import src.models as models


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="confs", config_name="conf")
def main(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    mlflow.set_tracking_uri(orig_cwd + "/mlruns")
    model = getattr(models, cfg.model.name)
    model = model(cfg)
    cfg.dataset.train_path = os.path.join(orig_cwd, cfg.dataset.train_path)
    cfg.dataset.test_path = os.path.join(orig_cwd, cfg.dataset.test_path)
    train_loader, test_loader, num_classes = get_loaders(cfg, model.train_transform(), model.test_transform())
    assert num_classes == cfg.num_classes, f"Num classes in config = {cfg.num_classes}, but dataset has {num_classes} classes"
    trainer = Trainer(model, cfg)
    trainer.run(train_loader, test_loader)


if __name__ == "__main__":
    main()