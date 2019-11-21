import torch.nn as nn
import torch.nn.functional as F
from . import resnet_fpn
from . import dla


def get_model(name, heads, num_filters=256, **kwargs):
    if name == 'resnet18_fpn':
        model = resnet_fpn.ResNetFPN('resnet18', heads, num_filters)
    elif name == 'resnet34_fpn':
        model = resnet_fpn.ResNetFPN('resnet34', heads, num_filters)
    elif name == 'resnet50_fpn':
        model = resnet_fpn.ResNetFPN('resnet50', heads, num_filters)
    elif name == 'resnet101_fpn':
        model = resnet_fpn.ResNetFPN('resnet101', heads, num_filters)
    elif name == 'resnet152_fpn':
        model = resnet_fpn.ResNetFPN('resnet152', heads, num_filters)
    elif name == 'dla34_ddd_3dop':
        model = dla.get_dla34(heads, pretrained='ddd_3dop')
    elif name == 'dla34_ddd_sub':
        model = dla.get_dla34(heads, pretrained='ddd_sub')
    else:
        raise NotImplementedError

    return model
