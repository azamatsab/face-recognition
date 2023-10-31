
from timm.models import create_model
import torch.nn as nn
from torch.nn import Module

from src.utils import l2_norm
from src.losses import Arcface


def change_first_layer(m):
  for name, child in m.named_children():
    if isinstance(child, nn.Conv2d):
      kwargs = {
          'out_channels': child.out_channels,
          'kernel_size': child.kernel_size,
          'stride': child.stride,
          'padding': child.padding,
          'bias': False if child.bias == None else True
      }
      m._modules[name] = nn.Conv2d(1, **kwargs)
      return True
    else:
      if(change_first_layer(child)):
        return True
  return False


class FaceTimmModel(Module):
    def __init__(self, cfg):
        super(FaceTimmModel, self).__init__()
        model_name = cfg.model.architecture
        self.backbone = create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=cfg.strat.embedding_size,
            drop_rate=cfg.strat.dropout_rate,
        )
        change_first_layer(self.backbone)
        self.head = Arcface(cfg)

    def forward(self, x, label, train=True):
        x = self.backbone(x)
        x = l2_norm(x)
        if train:
            x = self.head(x, label)
        return x
