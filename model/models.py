import torch
import torch.nn as nn
from model import layers
from cvnn import activation as act
from cvnn import dropout

class MGC(nn.Module):
    def __init__(self, nn_type, n_feat, n_hid, n_class, enable_bias, droprate):
        super(MGC, self).__init__()
        if nn_type == 'complex':
            self.graph_aug_linear = layers.CLinear(in_features=n_feat, out_features=n_hid, bias=enable_bias)
        else:
            self.graph_aug_linear = nn.Linear(in_features=n_feat, out_features=n_hid, bias=enable_bias)
        self.linear = nn.Linear(in_features=n_hid, out_features=n_class, bias=enable_bias)
        self.r_glu = act.RGLU()
        self.r_relu_2 = act.RSquaredReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.graph_aug_linear(x)
        x = self.r_glu(x)
        #x = self.r_relu_2(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.log_softmax(x)

        return x