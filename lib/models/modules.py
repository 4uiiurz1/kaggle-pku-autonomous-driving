import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, ws=False):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.ws = ws

    def forward(self, input):
        weight = self.weight
        if self.ws:
            weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                      keepdim=True).mean(dim=3, keepdim=True)
            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
            weight = weight / std.expand_as(weight)
            return self.conv2d_forward(input, weight)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
