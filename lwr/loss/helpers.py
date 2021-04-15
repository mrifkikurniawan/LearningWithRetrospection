from typing import Union, List, Dict

import torch
from torch import nn
from torch.nn import KLDivLoss

from lwr.utils import initialize_loss
from lwr import loss

def aggregate_loss(loss_fs: List[Dict], preds: dict(), targets: dict(), prefix: str, retros: bool):
    losses = dict()
    prefix = str(prefix)

    # looping through all loss functions
    for loss_f in loss_fs:
        loss_module = loss_f["module"]
        
        # skip if not retros
        if isinstance(loss_module, KLDivLoss) and retros == False:
            continue
        
        # get preds and targets
        y_hat = preds[loss_f['inputs']]
        y = targets[loss_f['targets']]

        loss_weight = loss_f['weight'].type_as(y)
        
        loss_name = loss_module.__class__.__name__

        loss_val = loss_weight * loss_module(y_hat, y)
        losses[f"{prefix}_{loss_name}"] = loss_val
    
    # sum up the all loss
    losses[f'{prefix}_total_loss'] = sum([losses[loss_name] for loss_name in losses.keys()])
    return losses

def generate_loss(loss_cfg: Union[List, Dict, str]) -> dict:
    loss_fs = list()

    if isinstance(loss_cfg, str):
        # set weight to 1 if w is not pre-defined
        loss_f = dict(module = initialize_loss(loss, loss_cfg), weight=torch.tensor([1], dtype=torch.float32, inputs='preds', targets='targets'))
        loss_fs.append(loss_f)

    elif isinstance(loss_cfg, list):
        for single_loss in loss_cfg:
            if isinstance(single_loss, str):
                # set weight to 1 if w is not pre-defined
                loss_f = dict(module = initialize_loss(loss, single_loss), weight=torch.tensor([1], dtype=torch.float32, inputs='preds', targets='targets'))
                loss_fs.append(loss_f)
            elif isinstance(single_loss, dict):
                loss_w = single_loss["weight"]
                loss_w = torch.tensor([loss_w], dtype=torch.float32)
                loss_f = dict(module = initialize_loss(loss, single_loss["module"]), weight=loss_w, 
                                                       inputs=single_loss['preds'], targets=single_loss['targets'])
                loss_fs.append(loss_f)

    elif isinstance(loss_cfg, dict):
        loss_w = loss_cfg["weight"]
        loss_w = torch.tensor([loss_w], dtype=torch.float32)
        loss_f = dict(module = initialize_loss(loss, loss_cfg["module"]), weight=loss_w,
                                               inputs=loss_cfg['preds'], targets=loss_cfg['targets'])
        loss_fs.append(loss_f)
    
    return loss_fs