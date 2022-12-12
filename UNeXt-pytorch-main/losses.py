import torch
import torch.nn as nn
import torch.nn.functional as F
#copy from ACENet.lossmodule 
from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['DiceLoss_ace','BCEDiceLoss', 'LovaszHingeLoss']

class DiceLoss_ace(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target, weights=None, ignore_index=None, binary=False):
        """
        Forward pass
        :param output: NxCxHxW logits
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :param binary: bool for binarized one chaneel(C=1) input
        :return: torch.tensor
        """
        output = F.softmax(output, dim=1)
        if binary:
            return self._dice_loss_binary(output, target)
        return self._dice_loss_multichannel(output, target, weights, ignore_index)

    @staticmethod
    def _dice_loss_binary(output, target):
        """
        Dice loss for one channel binarized input
        :param output: Nx1xHxW logits
        :param target: NxHxW LongTensor
        :return:
        """
        eps = 0.0001

        intersection = output * target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + target
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        return loss_per_channel.sum() / output.size(1)

    @staticmethod
    def _dice_loss_multichannel(output, target, weights=None, ignore_index=63):
        """
        Forward pass
        :param output: NxCxHxW Variable
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :param binary: bool for binarized one chaneel(C=1) input
        :return:
        """
        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
