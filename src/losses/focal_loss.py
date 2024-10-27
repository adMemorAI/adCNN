import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, logits=True, reduce=True):
        """
        Initialize Focal Loss.

        Args:
            alpha (list, optional): Weights for each class.
            gamma (float): Focusing parameter.
            logits (bool): If True, apply sigmoid to inputs.
            reduce (bool): If True, return the mean loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)  # pt is the probability of being classified to the true class
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha).to(inputs.device)
                alpha = alpha[targets.long().squeeze(1)]
            else:
                alpha = self.alpha
            F_loss = alpha * F_loss

        if self.reduce:
            return F_loss.mean()
        else:
            return F_loss
