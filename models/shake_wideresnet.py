from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from models.shakeshake import ShakeShake
from models.shakeshake import Shortcut

class ShakeBasicUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(ShakeBasicUnit, self).__init__()
        self.equal_io = in_channels == out_channels
        self.shortcut = None if self.equal_io else Shortcut(in_channels, out_channels, stride=stride)

        self.branch1 = self._make_branch(in_channels, out_channels, stride, dropout)
        self.branch2 = self._make_branch(in_channels, out_channels, stride, dropout)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_channels, out_channels, stride, dropout):
        return nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(in_channels)),
            ("1_activation", nn.ReLU(inplace=True)),
            ("2_convolution", nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False)),
            ("3_normalization", nn.BatchNorm2d(out_channels)),
            ("4_activation", nn.ReLU(inplace=True)),
            ("5_convolution", nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)),
            ("6_normalization", nn.BatchNorm2d(out_channels)),
            #("5_dropout", nn.Dropout(dropout, inplace=True)),
        ]))

class ShakeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float):
        super(ShakeBlock, self).__init__()
        self.block = nn.Sequential(
            ShakeBasicUnit(in_channels, out_channels, stride=stride, dropout=dropout),
            *(ShakeBasicUnit(out_channels, out_channels, stride=1, dropout=dropout) for _ in range(depth - 1))
        )

    def forward(self, x):
        return self.block(x)

class ShakeWideResNet(nn.Module):
    def __init__(self, depth: int, width_factor: int, dropout: float, in_channels: int, num_classes: int):
        super(ShakeWideResNet, self).__init__()

        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)

        self.f = nn.Sequential(OrderedDict([
            ("0_convolution", nn.Conv2d(in_channels, self.filters[0], (3, 3), stride=1, padding=1, bias=False)),
            ("1_block", ShakeBlock(self.filters[0], self.filters[1], 1, self.block_depth, dropout)),
            ("2_block", ShakeBlock(self.filters[1], self.filters[2], 2, self.block_depth, dropout)),
            ("3_block", ShakeBlock(self.filters[2], self.filters[3], 2, self.block_depth, dropout)),
            ("4_normalization", nn.BatchNorm2d(self.filters[3])),
            ("5_activation", nn.ReLU(inplace=True)),
            ("6_pooling", nn.AvgPool2d(kernel_size=8)),
            ("7_flattening", nn.Flatten()),
            ("8_classification", nn.Linear(in_features=self.filters[3], out_features=num_classes)),
        ]))

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)