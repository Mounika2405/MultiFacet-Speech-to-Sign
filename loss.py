# coding: utf-8
"""
Module to implement training loss
"""

from torch import nn, Tensor
import torch

from soft_dtw_cuda import SoftDTW

class RegLoss(nn.Module):
    """
    Regression Loss
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss == "huber":
            self.criterion = nn.HuberLoss()
        elif self.loss == "softdtw":
            self.criterion = SoftDTW(True, 1, False)
        elif self.loss == "softdtw_huber":
            self.dtw = SoftDTW(True, 1, False)
            self.huberloss = nn.HuberLoss()
            self.criterion = lambda x,y : self.dtw(x,y) + self.huberloss(x,y)

        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    # pylint: disable=arguments-differ
    def forward(self, preds, targets):

        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        # Calculate loss just over the masked predictions
        loss = self.criterion(preds_masked, targets_masked).mean()

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss

class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        # standard xent loss
        self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                    reduction='sum')

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        # targets: indices with batch*seq_len
        targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)

        return loss
    

class BCELogitLoss(nn.Module):
    """
    Binary Cross-Entropy Loss
    """

    def __init__(self, target_pad: int):
        super(BCELogitLoss, self).__init__()
        self.target_pad = target_pad
        # standard BCE loss
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets, targets_mask):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        # targets: indices with batch*seq_len
        targets = targets.contiguous().view(-1, targets.size(-1))
        log_probs = log_probs.contiguous().view(-1, log_probs.size(-1))
        mask = targets_mask[:, :, 1:].contiguous().view(-1)
        loss = self.criterion(log_probs, targets).T * mask
        loss = torch.sum(loss) / (torch.sum(mask) * 41)
        return loss


class MockLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MockLoss, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, log_probs, targets, targets_mask=None):
        return torch.tensor(0, device=self.dummy_param.device, dtype=torch.float32, requires_grad=True)
