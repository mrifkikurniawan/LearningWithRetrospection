# built-in
from typing import Union, Dict, Callable
from collections import namedtuple
import os.path as osp

# library
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import Tensor

from lwr.utils.activation import SoftmaxT


__all__ = ['SoftLabelsDataset']

class SoftLabelsDataset(Dataset):
    def __init__(self,
                 transform: Callable = None,
                 target_transform: Callable = None):
    
        self.softlabels = list()
        self.transform = transform

    def __len__(self):
        return len(self.softlabels)
    
    def __getitem__(self, index: int):
        # vars
        softlabels = self.softlabels[index]

        # transform image
        if self.transform:
            softlabels = self.transform(softlabels)

        return softlabels
    
    def append(self, inputs: Tensor):
        # detach from gpu, move to cpu
        inputs = inputs.detach().to('cpu')

        # append to state
        self.softlabels.append(inputs)
            
    def update(self, inputs: Tensor, batch_idx):
        # detach from gpu, move to cpu
        inputs = inputs.detach().to('cpu')

        # update
        self.softlabels[batch_idx] = inputs