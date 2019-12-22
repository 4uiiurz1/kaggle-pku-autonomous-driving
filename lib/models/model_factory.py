import torch.nn as nn
import torch.nn.functional as F
from . import resnet_fpn
from . import dla


def get_model(name, heads, head_conv=128, num_filters=[256, 256, 256],
              gn=False, ws=False, freeze_bn=False, **kwargs):
    if 'res' in name and 'fpn' in name:
        backbone = '_'.join(name.split('_')[:-1])
        model = resnet_fpn.ResNetFPN(backbone, heads, head_conv, num_filters,
                                     gn=gn, ws=ws, freeze_bn=freeze_bn)
    elif 'dla' in name:
        pretrained = '_'.join(name.split('_')[1:])
        model = dla.get_dla34(heads, pretrained=pretrained)
    else:
        raise NotImplementedError

    return model
