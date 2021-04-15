from typing import Optional, Union

from torch import nn
from torch import functional as F
from torch import Tensor

class SoftmaxT(nn.Module):
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, temperature: Union[float, int, Tensor]) -> None:
        super(SoftmaxT, self).__init__()

        if isinstance(temperature, float) or isinstance(temperature, int):
            temperature = Tensor([temperature])
        self.temperature = temperature
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self,   : Tensor, forward_softmax: bool) -> Tensor:
        temperature = self.temperature.type_as(inputs)
        out = inputs/temperature

        if forward_softmax:
            out = F.softmax(out, dim=1, _stacklevel=5)
        return out