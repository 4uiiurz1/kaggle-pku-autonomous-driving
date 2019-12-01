import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class AddCoordinates(object):

    r"""Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).
    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.
    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.
    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input x. Default: `False`
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`
    Examples:
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_adder(input)
        >>> device = torch.device("cuda:0")
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_adder(input)
    """

    def __init__(self, with_r=False):
        self.with_r = with_r

    def __call__(self, x):
        b, c, h, w = x.size()

        y_coords = 2.0 * torch.arange(h).unsqueeze(1).expand(h, w) / (h - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(w).unsqueeze(0).expand(h, w) / (w - 1.0) - 1.0

        # coords = torch.stack((y_coords, x_coords), dim=0)
        coords = y_coords

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(b, 1, 1, 1)
        coords = coords.float()

        x = torch.cat([coords.to(x.device), x], 1)

        return x


class ResNetFPNCConv(nn.Module):
    def __init__(self, backbone, heads, num_filters=256, pretrained=True, freeze_bn=False):
        super().__init__()

        self.heads = heads

        pretrained = 'imagenet' if pretrained else None

        if backbone == 'resnet18':
            self.backbone = pretrainedmodels.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif backbone == 'resnet34':
            self.backbone = pretrainedmodels.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif backbone == 'resnet50':
            self.backbone = pretrainedmodels.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet101':
            self.backbone = pretrainedmodels.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet152':
            self.backbone = pretrainedmodels.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        else:
            raise NotImplementedError

        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        self.lateral4 = nn.Sequential(
            nn.Conv2d(num_bottleneck_filters, num_filters,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True))
        self.lateral3 = nn.Sequential(
            nn.Conv2d(num_bottleneck_filters // 2,
                      num_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True))
        self.lateral2 = nn.Sequential(
            nn.Conv2d(num_bottleneck_filters // 4,
                      num_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True))
        self.lateral1 = nn.Sequential(
            nn.Conv2d(num_bottleneck_filters // 8,
                      num_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True))

        self.decode3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True))
        self.decode2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True))
        self.decode1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True))

        self.cconv = AddCoordinates()

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(num_filters + 1, num_filters // 2,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters // 2, num_output,
                          kernel_size=1))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x1 = self.backbone.conv1(x)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        x1 = self.backbone.maxpool(x1)

        x1 = self.backbone.layer1(x1)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        lat4 = self.lateral4(x4)
        lat3 = self.lateral3(x3)
        lat2 = self.lateral2(x2)
        lat1 = self.lateral1(x1)

        map4 = lat4
        map3 = lat3 + F.interpolate(map4, scale_factor=2, mode="nearest")
        map3 = self.decode3(map3)
        map2 = lat2 + F.interpolate(map3, scale_factor=2, mode="nearest")
        map2 = self.decode2(map2)
        map1 = lat1 + F.interpolate(map2, scale_factor=2, mode="nearest")
        map1 = self.decode1(map1)

        map1 = self.cconv(map1)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(map1)
        return ret
