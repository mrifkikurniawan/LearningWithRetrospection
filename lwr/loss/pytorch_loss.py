from torch import nn

__all__ = ['CrossEntropyLoss', 'BCELoss', 'KLDivLoss']

def CrossEntropyLoss(**args):
    return nn.CrossEntropyLoss(**args)

def BCELoss(**args):
    return nn.BCELoss(**args)

def KLDivLoss(**args):
    return nn.KLDivLoss(**args)