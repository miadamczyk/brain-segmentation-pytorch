import torch.nn as nn
import numpy as np
import torch

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y, lbl):
        criterion = nn.MSELoss(reduction="mean")
        criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
        veci = 5. * torch.from_numpy(lbl[:, 1:]).cuda()
        loss = criterion(y[:, :2], veci)
        loss /= 2.
        loss2 = criterion2(y[:, -1], torch.from_numpy(lbl[:, 0] > 0.5)..cuda().float())
        loss = loss + loss2
        return loss
