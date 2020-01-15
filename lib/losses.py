import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask):
        loss = F.binary_cross_entropy(
            input * mask, target * mask, reduction='sum')
        loss /= mask.sum()
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask):
        loss = F.l1_loss(output * mask, target * mask, reduction='sum')
        loss /= mask.sum()
        return loss


class DepthL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask):
        output = 1. / (torch.sigmoid(output) + 1e-6) - 1.
        loss = F.l1_loss(output * mask, target * mask, reduction='sum')
        loss /= mask.sum()
        return loss


def _neg_loss(pred, gt, mask):
    pos_inds = gt.eq(1).float() * mask
    neg_inds = gt.lt(1).float() * mask

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * \
        neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.neg_loss = _neg_loss

    def forward(self, output, target, mask):
        output = torch.sigmoid(output)
        loss = self.neg_loss(output, target, mask)
        return loss


class DotProductLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask):
        output *= mask
        target *= mask

        # normalize
        # output /= torch.norm(output, p=2, dim=1, keepdim=True) + 1e-4

        dot = torch.sum(output * target, 1, keepdim=True)
        loss = (1 - dot) / 2
        loss = loss.sum() / mask.sum()
        return loss
