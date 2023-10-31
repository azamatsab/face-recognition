import torch
from . import TrainDataset, TestDataset


def get_loaders(cfg, train_transform, test_transform):
    batch_size = cfg.strat.batch_size
    num_workers = cfg.num_workers

    train_dataset = TrainDataset(cfg.dataset.train_path, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    test_dataset = TestDataset(cfg.dataset.test_path, test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_loader, test_loader, len(train_dataset.classes)
