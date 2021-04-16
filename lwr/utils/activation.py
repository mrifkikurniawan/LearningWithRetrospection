from typing import Optional, Union

from torch import nn
from torch.nn import functional as F
from torch import Tensor

class SoftmaxT(nn.Module):

    def __init__(self, temperature: Union[float, int, Tensor]) -> None:
        super(SoftmaxT, self).__init__()

        if isinstance(temperature, float) or isinstance(temperature, int):
            temperature = Tensor([temperature])
        self.temperature = temperature

    def forward(self, inputs: Tensor, act_f: str) -> Tensor:
        temperature = self.temperature.type_as(inputs)
        out = inputs/temperature

        if act_f == 'softmax':
            out = F.softmax(out, dim=1)
        elif act_f == 'log_softmax':
            out = F.log_softmax(out, dim=1)
        return out