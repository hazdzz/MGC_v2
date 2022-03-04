import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
import cvnn.init as cinit

class CLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features, out_features, bias):
        super(CLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.complex(torch.FloatTensor(in_features, out_features), \
                        torch.FloatTensor(in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.complex(torch.FloatTensor(out_features), torch.FloatTensor(out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        cinit.complex_kaiming_uniform_(self.weight)

        if self.bias is not None:
            fan_in_real, _ = init._calculate_fan_in_and_fan_out(self.weight.real)
            bound_real = 1 / math.sqrt(fan_in_real) if fan_in_real > 0 else 0
            init.uniform_(self.bias.real, -bound_real, bound_real)

            fan_in_imag, _ = init._calculate_fan_in_and_fan_out(self.weight.imag)
            bound_imag = 1 / math.sqrt(fan_in_imag) if fan_in_imag > 0 else 0
            init.uniform_(self.bias.imag, -bound_imag, bound_imag)

    def forward(self, x):
        if x.is_sparse:
            c_linear = torch.sparse.mm(x, self.weight)
        else:
            c_linear = torch.mm(x, self.weight)
        
        if self.bias is not None:
            return torch.add(input=c_linear, other=self.bias, alpha=1)
        else:
            return c_linear

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )