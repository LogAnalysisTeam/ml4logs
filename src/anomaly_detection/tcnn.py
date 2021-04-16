"""
***************************************************************************************
*    Title: TCN source code
*    Authors: Shaojie Bai, J. Zico Kolter and Vladlen Koltun
*    Date: 2018
*    Code version: 8845f88
*    Availability: https://github.com/locuslab/TCN
*
***************************************************************************************
* local changes and adjustments for the purpose of this thesis
"""
import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
import numpy as np

SEED = 160121

np.random.seed(SEED)
torch.manual_seed(SEED)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, is_last=False):
        super(TemporalBlock, self).__init__()
        self.is_last_block = is_last
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1,
                                 self.relu1,
                                 self.dropout1,
                                 self.conv2,
                                 self.relu2,
                                 self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) if not self.is_last_block else (out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, include_last_relu=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            is_last_block = i == num_levels - 1 and not include_last_relu
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=((kernel_size - 1) * dilation_size) // 2, dropout=dropout,
                                     is_last=is_last_block)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
