import torch.nn as nn
import torch.nn.functional as F
from . import resnet_fpn
from . import resnet_fpn_cconv
from . import resnet_fpn_dcn
from . import resnet_fpn_gn
from . import resnet_fpn_wsgn
from . import dla


def get_model(name, heads, num_filters=256, freeze_bn=False, **kwargs):
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
    elif name == 'resnet18_fpn_cconv':
        model = resnet_fpn_cconv.ResNetFPNCConv('resnet18', heads, num_filters)
    elif name == 'resnet18_fpn_dcn':
        model = resnet_fpn_dcn.ResNetFPNDCN('resnet18', heads, num_filters)
    elif name == 'resnet18_fpn_gn':
        model = resnet_fpn_gn.ResNetFPNGN('resnet18', heads, num_filters)
    elif name == 'resnet18_fpn_wsgn':
        model = resnet_fpn_wsgn.ResNetFPNWSGN('resnet18', heads, num_filters)
    else:
        raise NotImplementedError

    return model
