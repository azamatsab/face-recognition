from collections import OrderedDict
import torch


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def freeze(model, until):
    flag = False
    for name, param in model.named_parameters():
        if name == until:
            flag = True
        param.requires_grad = flag

def remove_parallel(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict